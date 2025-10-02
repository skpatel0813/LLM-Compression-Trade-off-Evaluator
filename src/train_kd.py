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
# Why we do it this way
# ---------------------
# - The small model learns to act like the big, smart model
# - LoRA makes training faster and uses less computer memory
# - Everything works reliably on shared computers
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
import math
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
    DataCollatorWithPadding,
)

from peft import LoraConfig, get_peft_model, TaskType

# ---- Our helper tools ----
from .utils import Config, build_chat_text


# ===========================
# Make things predictable
# ===========================
def set_seed(seed: int = 42) -> None:
    """Set random seeds so we get the same results each time"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Use special folders for downloads if user wants
HF_HOME = os.environ.get("HF_HOME")
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE")
if HF_HOME:
    os.makedirs(HF_HOME, exist_ok=True)
if HF_DATASETS_CACHE:
    os.makedirs(HF_DATASETS_CACHE, exist_ok=True)


# ===========================
# Reading our conversation files
# ===========================
class ChatJsonlReader(Dataset):
    """
    Smart file reader that doesn't load everything into memory at once.
    Like reading a big book by remembering page numbers instead of memorizing every word.
    """

    def __init__(self, files: List[str], max_samples: Optional[int] = None):
        self.index: List[Tuple[str, int]] = []  # (file_path, position_in_file)
        total = 0
        for fp in files:
            if not os.path.isfile(fp):
                continue
            with open(fp, "rb") as f:
                offset = 0
                for line in f:
                    # Remember where this conversation is in the file
                    self.index.append((fp, offset))
                    offset += len(line)
                    total += 1
                    # Stop if we have enough examples
                    if max_samples and total >= max_samples:
                        break
            if max_samples and total >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        # Read one conversation from its saved position
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
    Prepares conversations for the AI model to understand.
    
    - Turns conversations into the special Llama-3 format
    - Converts text into numbers (tokenization)
    - Makes sure all examples in a batch are the same length
    - Marks which parts the model should learn from
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        self.tok = tokenizer
        self.max_len = max_seq_len

    def __call__(self, features: List[Dict[str, Any]]) -> TokenizedBatch:
        # 1) Convert conversations to Llama-3 format
        texts = [build_chat_text(ex["messages"]) for ex in features]

        # 2) Turn text into numbers
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # 3) Mark which tokens the model should learn from
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # -100 means "ignore this"

        return TokenizedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


# ===========================
# Smart Learning Formula
# ===========================
def shift_for_next_token(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
    """
    Prepares everything for "guess the next word" learning.
    
    Like showing a sentence and asking "what comes next?" for each word.
    """
    B, T, V = logits.shape

    # Look at all positions except the last one
    s_logits = logits[:, :-1, :]
    labels_shifted = labels[:, 1:]
    attn_shifted = attention_mask[:, 1:]

    # Find positions that matter for learning
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
    Combined learning formula:
      Total Learning = 
        (learning from teacher) + (learning from correct answers)
    
    The teacher part helps the student think like the wise teacher.
    The correct answer part helps the student get facts right.
    """

    # 1) Prepare for "next word" learning
    s_next, gold, valid_mask = shift_for_next_token(s_logits, labels, attention_mask)
    t_next, _, _ = shift_for_next_token(t_logits, labels, attention_mask)

    # Find which positions actually matter
    if valid_mask.ndim == 2:
        idx = valid_mask.view(-1)
    else:
        idx = valid_mask

    # If nothing to learn from in this batch, skip
    if idx.sum().item() == 0:
        return torch.zeros([], device=s_logits.device, dtype=s_logits.dtype)

    # Get only the important parts
    s_sel = s_next.view(-1, s_next.size(-1))[idx]
    t_sel = t_next.view(-1, t_next.size(-1))[idx]
    gold_sel = gold.view(-1)[idx]

    # 2) Learn from teacher's wisdom
    sT = s_sel / T
    tT = t_sel / T
    log_p_s = F.log_softmax(sT, dim=-1)
    p_t = F.softmax(tT, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

    # 3) Learn from correct answers
    ce = F.cross_entropy(s_sel, gold_sel, ignore_index=-100)

    # Combine both types of learning
    return alpha * kl + (1.0 - alpha) * ce


# ===========================
# Smart Teacher Class
# ===========================
class KDTrainer(Trainer):
    """
    A special teacher that:
      - Lets the student try to answer
      - Asks the wise teacher for the best answer
      - Combines both to help the student learn better
    """

    def __init__(self, *args, teacher: nn.Module, T: float, alpha: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval()  # Teacher doesn't learn, just helps
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.T = T
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # Student tries to answer
        s_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Teacher shows the wise way (no learning for teacher)
        with torch.no_grad():
            t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Calculate how much the student needs to improve
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
    # Load our recipe (settings)
    cfg = Config.load().cfg
    P = cfg["paths"]
    M = cfg["models"]
    Tcfg = cfg["training"]

    os.makedirs(P["outputs_dir"], exist_ok=True)
    os.makedirs(P["lora_dir"], exist_ok=True)

    # For quick testing, use fewer examples
    MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))  # 0 = use all

    # -------------------------
    # Text to Number Converter
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(M["student_id"], use_fast=True)
    # Make sure we have a special "padding" token for making batches even
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # Load our TEACHER & STUDENT
    # -------------------------
    print("Loading the WISE TEACHER:", M["teacher_id"])
    teacher = AutoModelForCausalLM.from_pretrained(
        M["teacher_id"],
        torch_dtype=torch.bfloat16,          # Use efficient number format
        device_map="auto",                   # Use all available GPUs
        low_cpu_mem_usage=True,
    )
    print("Loading the SMART STUDENT:", M["student_id"])
    student = AutoModelForCausalLM.from_pretrained(
        M["student_id"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # -------------------------
    # Add special learning notes (LoRA)
    # -------------------------
    # LoRA lets us teach just the important parts of the student
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
    
    # Count how many parts we're actually teaching
    trainable, total = 0, 0
    for p in student.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"Teaching only: {trainable:,} parts out of {total:,} total "
          f"({100*trainable/total:.2f}%) - that's efficient!")

    # -------------------------
    # Prepare learning materials
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
        raise FileNotFoundError(
            "No learning materials found! Please run data_prep.py first."
        )
    collator = CausalCollator(tokenizer, max_seq_len=Tcfg["seq_len"])

    # -------------------------
    # Learning schedule
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
        bf16=True,                            # Use efficient number format
        gradient_checkpointing=True,          # Use less memory
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to=[],                         # Don't send reports anywhere
        dataloader_num_workers=2,             # How many helpers to use
        ddp_find_unused_parameters=False,     # Better for speed
    )

    # -------------------------
    # Start the learning session!
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

    # Begin learning!
    trainer.train()

    # Save what the student learned
    print("Saving the student's knowledge to:", P["lora_dir"])
    student.save_pretrained(P["lora_dir"])
    tokenizer.save_pretrained(P["lora_dir"])
    print("Great success! The student has learned well.")


if __name__ == "__main__":
    # Quiet down the less important messages
    warnings.filterwarnings("once", category=UserWarning)
    main()