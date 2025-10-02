# src/train_kd_lora.py
# --------------------------------------------------------------------------------------
# Smart Learning: Teaching a small AI model from a big AI model
#   Big Teacher:  Llama-3.1-70B-Instruct (like a wise professor)
#   Small Student:  Llama-3.1-8B-Instruct (like a smart student)
#
# What this script does
# ---------------------
# 1) Reads conversation files we created earlier:
#       - data/opencodeinstruct_python_train.jsonl
#       - data/codesearchnet_python_train.jsonl
#    Each line is a conversation between people and AI assistant.
#
# 2) Turns conversations into the special format that Llama-3 understands.
#
# 3) Uses "LoRA" - a clever way to teach just parts of the student model
#    (like adding sticky notes to a book instead of rewriting the whole book).
#
# 4) The learning process has two parts:
#       - Learning from the big teacher's wisdom
#       - Learning from the correct answers in our data
#
# 5) Saves what the student learned in a special folder.
#
# Notes
# -----
# - This uses all available GPUs automatically
# - For quick testing, set MAX_SAMPLES to a small number
# - You can continue training where you left off
#
# --------------------------------------------------------------------------------------

from __future__ import annotations
import os
import json
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, TaskType

# ---- Our helper tools ----
from .utils import Config, build_chat_text


# ===========================
# Make things predictable
# ===========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Optional: slightly faster matmul on Ampere+
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Use special folders for downloads if user wants
for env_var in ("HF_HOME", "HF_DATASETS_CACHE"):
    v = os.environ.get(env_var)
    if v:
        os.makedirs(v, exist_ok=True)


# ===========================
# Reading our conversation files
# ===========================
class ChatJsonlReader(Dataset):
    """Memory-light JSONL reader with byte offsets."""

    def __init__(self, files: List[str], max_samples: Optional[int] = None):
        self.index: List[Tuple[str, int]] = []  # (file_path, byte_offset)
        total = 0
        for fp in files:
            if not os.path.isfile(fp):
                continue
            with open(fp, "rb") as f:
                offset = 0
                for line in f:
                    self.index.append((fp, offset))
                    offset += len(line)
                    total += 1
                    if max_samples and total >= max_samples:
                        break
            if max_samples and total >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        fp, offset = self.index[i]
        with open(fp, "rb") as f:
            f.seek(offset)
            raw = f.readline()
        ex = json.loads(raw.decode("utf-8"))
        return ex


@dataclass
class TokenizedBatch:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor


class CausalCollator:
    """
    Converts chats to Llama3 prompt format, tokenizes, pads, and builds labels.
    By default we supervise all non-padding tokens (simple KD+CE).
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        self.tok = tokenizer
        self.max_len = max_seq_len

    def __call__(self, features: List[Dict[str, Any]]) -> TokenizedBatch:
        texts = [build_chat_text(ex["messages"]) for ex in features]

        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Supervise all non-pad tokens (ignore padding)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return TokenizedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


# ===========================
# KD helpers
# ===========================
def shift_for_next_token(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
    B, T, V = logits.shape
    s_logits = logits[:, :-1, :]
    labels_shifted = labels[:, 1:]
    attn_shifted = attention_mask[:, 1:]
    valid = (attn_shifted > 0) & (labels_shifted != -100)
    return s_logits, labels_shifted, valid


def kd_loss(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    T: float = 1.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    s_next, gold, valid_mask = shift_for_next_token(s_logits, labels, attention_mask)
    t_next, _, _ = shift_for_next_token(t_logits, labels, attention_mask)

    idx = valid_mask.view(-1)
    if idx.sum().item() == 0:
        return torch.zeros([], device=s_logits.device, dtype=s_logits.dtype)

    s_sel = s_next.view(-1, s_next.size(-1))[idx]
    t_sel = t_next.view(-1, t_next.size(-1))[idx]
    gold_sel = gold.view(-1)[idx]

    sT = s_sel / T
    tT = t_sel / T
    log_p_s = F.log_softmax(sT, dim=-1)
    p_t = F.softmax(tT, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

    ce = F.cross_entropy(s_sel, gold_sel, ignore_index=-100)
    return alpha * kl + (1.0 - alpha) * ce


# ===========================
# KD Trainer
# ===========================
class KDTrainer(Trainer):
    """Runs student forward; gets teacher logits under no-grad; combines losses."""

    def __init__(self, *args, teacher: nn.Module, T: float, alpha: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.T = float(T)
        self.alpha = float(alpha)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        s_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        with torch.no_grad():
            t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        loss = kd_loss(
            s_out.logits,
            t_out.logits,
            labels,
            attention_mask,
            T=self.T,
            alpha=self.alpha,
        )
        return (loss, s_out) if return_outputs else loss


# ===========================
# Main Training Program
# ===========================
def main():
    cfg = Config.load().cfg
    P = cfg["paths"]
    M = cfg["models"]
    Tcfg = cfg["training"]

    os.makedirs(P["outputs_dir"], exist_ok=True)
    os.makedirs(P["lora_dir"], exist_ok=True)

    MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))  # 0 = use all

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(M["student_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # Load TEACHER & STUDENT with device maps
    # -------------------------
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("Loading the WISE TEACHER:", M["teacher_id"])
    teacher = AutoModelForCausalLM.from_pretrained(
        M["teacher_id"],
        torch_dtype=dtype,                 # bf16 on A100, else fp16
        device_map="auto",                 # let Accelerate spread across GPUs
        low_cpu_mem_usage=True,
        offload_state_dict=True,           # avoid loading full weights in RAM
    )
    print("Loading the SMART STUDENT:", M["student_id"])
    student = AutoModelForCausalLM.from_pretrained(
        M["student_id"],
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_state_dict=True,
    )

    # Enable grad checkpointing before PEFT to reduce memory
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    # -------------------------
    # LoRA
    # -------------------------
    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    student = get_peft_model(student, lcfg)

    # Parameter count (for info)
    trainable, total = 0, 0
    for p in student.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"Teaching only: {trainable:,} parts out of {total:,} total "
          f"({100*trainable/total:.2f}%) - that's efficient!")

    # -------------------------
    # Dataset + collator
    # -------------------------
    data_files = [
        os.path.join(P["data_dir"], "opencodeinstruct_python_train.jsonl"),
        os.path.join(P["data_dir"], "codesearchnet_python_train.jsonl"),
    ]
    ds = ChatJsonlReader(
        [fp for fp in data_files if os.path.isfile(fp)],
        max_samples=(MAX_SAMPLES if MAX_SAMPLES > 0 else None),
    )
    if len(ds) == 0:
        raise FileNotFoundError("No learning materials found! Please run data_prep.py first.")
    collator = CausalCollator(tokenizer, max_seq_len=Tcfg["seq_len"])

    # -------------------------
    # TrainingArguments
    #   - place_model_on_device=False prevents Trainer from calling .to(device)
    #     on meta-sharded models: fixes the meta-tensor crash.
    #   - bf16 auto if supported; otherwise fp16.
    # -------------------------
    args = TrainingArguments(
        output_dir=os.path.join(P["outputs_dir"], "llama31_8b_kd_lora"),
        num_train_epochs=Tcfg["epochs"],
        per_device_train_batch_size=Tcfg["per_device_batch_size"],
        gradient_accumulation_steps=Tcfg["grad_accum"],
        learning_rate=Tcfg["learning_rate"],
        warmup_ratio=0.03,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        bf16_full_eval=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to=[],                      # no external logging
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,       # safer for custom collators/LMs
        place_model_on_device=False,       # CRITICAL: avoid meta->device .to()
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = KDTrainer(
        model=student,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
        teacher=teacher,
        T=float(Tcfg.get("kd_temp", 1.0)),
        alpha=float(Tcfg.get("kd_alpha", 0.5)),
    )

    # Train
    trainer.train()

    # Save LoRA adapters (and tokenizer for convenience)
    print("Saving the student's knowledge to:", P["lora_dir"])
    student.save_pretrained(P["lora_dir"])
    tokenizer.save_pretrained(P["lora_dir"])
    print("✅ Great success! The student has learned well.")


if __name__ == "__main__":
    warnings.filterwarnings("once", category=UserWarning)
    main()
