# src/train_kd.py
# --------------------------------------------------------------------------------------
# Knowledge Distillation with LoRA (Llama 3.1 70B -> 8B) for Python code generation
#
# - Teacher:  meta-llama/Meta-Llama-3.1-70B-Instruct
# - Student:  meta-llama/Meta-Llama-3.1-8B-Instruct  (fine-tuned with LoRA adapters)
#
# Pipeline:
#   (1) Read chat-style JSONL from ./data (each line: {"messages": [...]})
#   (2) Format to Llama-3 chat, tokenize, build labels
#   (3) KD loss = alpha * KL( student || teacher_T ) * T^2  +  (1 - alpha) * CE(gold)
#   (4) Train only LoRA params (tiny %) and save adapters to outputs/lora
#
# Engineering choices that avoid your recent errors:
#   - Load both models with device_map="auto" and dtype=... (not torch_dtype)
#   - Override Trainer._move_model_to_device to NO-OP (prevents meta-tensor crash)
#   - Collator returns a PLAIN DICT (not a dataclass) so Trainer can inspect it
#   - KDTrainer.compute_loss accepts **kwargs / num_items_in_batch (HF>=4.45)
#
# Quick smoke run:
#   CUDA_VISIBLE_DEVICES=0 MAX_SAMPLES=2000 "$CONDA_PREFIX/bin/python" -u -m src.train_kd --bf16 True
# --------------------------------------------------------------------------------------

from __future__ import annotations
import os
import json
import random
import warnings
import argparse
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

# Project helpers (expects ./configs/project.yaml and ./src/utils.py)
from .utils import Config, build_chat_text


# ------------------------------
# Reproducibility & tiny speedups
# ------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

try:
    torch.set_float32_matmul_precision("high")  # small matmul speedup on Ampere+
except Exception:
    pass

# Respect custom cache dirs if set
for env_var in ("HF_HOME", "HF_DATASETS_CACHE"):
    v = os.environ.get(env_var)
    if v:
        os.makedirs(v, exist_ok=True)


# ------------------------------
# Lightweight JSONL reader
# ------------------------------
class ChatJsonlReader(Dataset):
    """
    Memory-light dataset:
      - Builds an index of byte offsets (file, offset) across N JSONL files
      - __getitem__ seeks and reads one line only (O(1) memory)
      - Optional MAX_SAMPLES limit via env var or constructor
    """

    def __init__(self, files: List[str], max_samples: Optional[int] = None):
        self.index: List[Tuple[str, int]] = []  # (file_path, byte_offset)
        total = 0
        for fp in files:
            if not os.path.isfile(fp):
                continue
            with open(fp, "rb") as f:
                offset = 0
                for line in f:
                    if not line.strip():
                        offset += len(line)
                        continue
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
        return json.loads(raw.decode("utf-8"))


# ------------------------------
# Collator: chats -> tokenized tensors -> labels
# ------------------------------
class CausalCollator:
    """
    Converts chat messages to Llama-3 chat format, tokenizes, pads, and builds labels.
    Returns a PLAIN DICT so `transformers.Trainer` can consume it directly.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        self.tok = tokenizer
        self.max_len = max_seq_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Convert JSONL "messages" -> Llama 3 chat-formatted string
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

        # Supervise all non-pad tokens (ignore padding with -100)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ------------------------------
# KD helpers (next-token alignment)
# ------------------------------
def shift_for_next_token(
    logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Align for next-token prediction:
      - Compare logits at time t to gold token at t+1
      - Mask with attention and ignore_index
    """
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
    """
    loss = alpha * KL( student || teacher_T ) * T^2  +  (1 - alpha) * CE(student, gold)
    """
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


# ------------------------------
# Custom Trainer with NO-OP device move
# ------------------------------
class KDTrainer(Trainer):
    """
    Trainer that:
      - Keeps model placement as loaded by device_map="auto" (NO .to(device) calls)
      - Runs teacher in no_grad
      - Computes KD loss
    """

    def __init__(self, *args, teacher: nn.Module, T: float, alpha: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.T = float(T)
        self.alpha = float(alpha)

    # CRITICAL: prevent Trainer from moving a sharded/meta-initialized model
    # (avoids: "Cannot copy out of meta tensor; no data!")
    def _move_model_to_device(self, model, device):
        return model

    # Accept **kwargs/num_items_in_batch to be compatible with new Trainer API
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
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


# ------------------------------
# CLI + main
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", type=str, default=None,
                    help="True/False to force bfloat16; default: auto if supported.")
    ap.add_argument("--fp16", type=str, default=None,
                    help="True/False to force float16; default: opposite of bf16 when CUDA.")
    ap.add_argument("--logging_steps", type=int, default=int(os.environ.get("LOGGING_STEPS", "20")))
    return ap.parse_args()

def str2bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    return x.lower() in ("1", "true", "t", "yes", "y")

def main():
    args_cli = parse_args()

    cfg = Config.load().cfg
    P = cfg["paths"]
    M = cfg["models"]
    Tcfg = cfg["training"]

    os.makedirs(P["outputs_dir"], exist_ok=True)
    os.makedirs(P["lora_dir"], exist_ok=True)

    MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))  # 0 = use all rows

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(M["student_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # Numerics (bf16/fp16 selection)
    # -------------------------
    if torch.cuda.is_available():
        auto_bf16 = torch.cuda.is_bf16_supported()
        force_bf16 = str2bool(args_cli.bf16)
        force_fp16 = str2bool(args_cli.fp16)

        if force_bf16 is True:
            use_bf16 = True
        elif force_bf16 is False:
            use_bf16 = False
        else:
            use_bf16 = auto_bf16

        if force_fp16 is True:
            use_fp16 = True
            use_bf16 = False
        elif force_fp16 is False:
            use_fp16 = False
        else:
            use_fp16 = not use_bf16
    else:
        use_bf16 = False
        use_fp16 = False

    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    # -------------------------
    # Load Teacher & Student (sharded)
    # -------------------------
    print("Loading the WISE TEACHER:", M["teacher_id"])
    teacher = AutoModelForCausalLM.from_pretrained(
        M["teacher_id"],
        dtype=dtype,                  # replaces deprecated torch_dtype
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_state_dict=True,
    )
    print("Loading the SMART STUDENT:", M["student_id"])
    student = AutoModelForCausalLM.from_pretrained(
        M["student_id"],
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_state_dict=True,
    )

    # Enable grad checkpointing before PEFT to reduce activation memory
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    # -------------------------
    # LoRA config
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

    # Info: trainable vs total params
    trainable, total = 0, 0
    for p in student.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"Teaching only: {trainable:,} parts out of {total:,} total "
          f"({100 * trainable / total:.2f}%) - that's efficient!")

    # -------------------------
    # Dataset & collator
    # -------------------------
    data_files = [
        os.path.join(P["data_dir"], "opencodeinstruct_python_train.jsonl"),
        os.path.join(P["data_dir"], "codesearchnet_python_train.jsonl"),
    ]
    # Keep only files that exist and are non-empty
    data_files = [fp for fp in data_files if os.path.isfile(fp) and os.path.getsize(fp) > 0]

    ds = ChatJsonlReader(
        data_files,
        max_samples=(MAX_SAMPLES if MAX_SAMPLES > 0 else None),
    )
    if len(ds) == 0:
        raise FileNotFoundError(
            "No learning materials found! Run data_prep.py to build JSONL data first."
        )
    collator = CausalCollator(tokenizer, max_seq_len=Tcfg["seq_len"])

    # -------------------------
    # TrainingArguments
    #   NOTE: We DO NOT pass `place_model_on_device` (not available everywhere).
    #         Our KDTrainer overrides `_move_model_to_device` to NO-OP instead.
    # -------------------------
    targs = TrainingArguments(
        output_dir=os.path.join(P["outputs_dir"], "llama31_8b_kd_lora"),
        num_train_epochs=Tcfg["epochs"],
        per_device_train_batch_size=Tcfg["per_device_batch_size"],
        gradient_accumulation_steps=Tcfg["grad_accum"],
        learning_rate=Tcfg["learning_rate"],
        warmup_ratio=0.03,
        logging_steps=int(args_cli.logging_steps),
        save_steps=500,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to=[],                # no external logging
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False, # safer for custom collators/LMs
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = KDTrainer(
        model=student,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,  # deprecation warning is harmless for now
        teacher=teacher,
        T=float(Tcfg.get("kd_temp", 1.0)),
        alpha=float(Tcfg.get("kd_alpha", 0.5)),
    )

    # -------------------------
    # Train
    # -------------------------
    trainer.train()

    # -------------------------
    # Save LoRA adapters (and tokenizer)
    # -------------------------
    print("Saving the student's knowledge to:", P["lora_dir"])
    student.save_pretrained(P["lora_dir"])
    tokenizer.save_pretrained(P["lora_dir"])
    print("✅ Great success! The student has learned well.")


if __name__ == "__main__":
    warnings.filterwarnings("once", category=UserWarning)
    main()
