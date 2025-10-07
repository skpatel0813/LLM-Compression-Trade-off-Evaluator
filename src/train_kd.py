# src/train_kid.py
# =================================================================================================
# Knowledge Distillation (KD) for Llama-3.1-8B (student) taught by Llama-3.1-70B (teacher)
#
# Key features:
#   • Multi-GPU safe: device_map="auto", per-GPU memory caps, no-op device move
#   • Optional teacher quantization (4-bit / 8-bit) to avoid OOM
#   • LoRA adapters for parameter-efficient student training (toggleable)
#   • BF16/FP16 precision controls
#   • Robust collator returning dict (Trainer-compatible)
#   • KD loss = alpha * KL(student||teacher_T) * T^2 + (1-alpha) * CE(student,gold)
#   • Optional MLflow GPU/power logging via LiteMLflowCallback
#
# Example runs:
#   # single GPU, BF16 if supported, LoRA on
#   CUDA_VISIBLE_DEVICES=0 MAX_SAMPLES=2000 "$CONDA_PREFIX/bin/python" -u -m src.train_kid --bf16 True
#
#   # multi-GPU sharding, teacher in 4-bit to save VRAM, LoRA off (full-param KD)
#   CUDA_VISIBLE_DEVICES=0,1,2,3 MAX_SAMPLES=2000 "$CONDA_PREFIX/bin/python" -u -m src.train_kid \
#     --bf16 True --teacher_4bit True --lora False --seq_len 1024 --max_memory_frac 0.90
#
# Requirements:
#   transformers>=4.44, accelerate>=0.33, peft
#   bitsandbytes (only if using 4/8-bit)
#   (optional) mlflow, pynvml + src/callbacks_mlflow.py for GPU/power logging
# =================================================================================================

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

# Optional bitsandbytes for 4/8-bit quantization
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# Optional MLflow GPU/power logging callback (safe if missing)
try:
    from .callbacks_mlflow import LiteMLflowCallback
    _HAS_MLFLOW_CB = True
except Exception:
    _HAS_MLFLOW_CB = False

# Project helpers
from .utils import Config, build_chat_text


# -------------------------------------------------------------------------------------------------
# Reproducibility & CUDA allocator settings
# -------------------------------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Modest speedup on Ampere+ (no-op elsewhere)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Reduce fragmentation on long runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Respect custom HF caches if provided
for env_var in ("HF_HOME", "HF_DATASETS_CACHE"):
    v = os.environ.get(env_var)
    if v:
        os.makedirs(v, exist_ok=True)


# -------------------------------------------------------------------------------------------------
# Lightweight JSONL reader (O(1) memory with byte offsets)
# -------------------------------------------------------------------------------------------------
class ChatJsonlReader(Dataset):
    """
    Builds an index of (file_path, byte_offset). __getitem__ seeks and reads one JSON line.
    Set MAX_SAMPLES (env or arg) to cap data volume when testing.
    """
    def __init__(self, files: List[str], max_samples: Optional[int] = None):
        self.index: List[Tuple[str, int]] = []
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


# -------------------------------------------------------------------------------------------------
# Collator: format chats -> tokenize -> pad -> labels
# -------------------------------------------------------------------------------------------------
class CausalCollator:
    """
    Converts list of {messages:[...]} into Llama-3 chat text, tokenizes, pads,
    and produces labels (-100 on pad positions) for language modeling loss.
    Returns a plain dict compatible with transformers.Trainer.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        self.tok = tokenizer
        self.max_len = max_seq_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore pad on loss

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -------------------------------------------------------------------------------------------------
# KD helpers (next-token alignment)
# -------------------------------------------------------------------------------------------------
def _shift_for_next_token(
    logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Align teacher/student token predictions with the next-token targets.
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
    Knowledge Distillation loss:
      L = alpha * KL( student || teacher_T ) * T^2 + (1 - alpha) * CE(student, gold)
    """
    s_next, gold, valid_mask = _shift_for_next_token(s_logits, labels, attention_mask)
    t_next, _, _ = _shift_for_next_token(t_logits, labels, attention_mask)

    idx = valid_mask.view(-1)
    if idx.sum().item() == 0:
        # No valid tokens in this small batch; return zero to keep training loop happy
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


# -------------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------------
def make_max_memory_map(frac: float = 0.90) -> Dict[int, str]:
    """
    Build a HuggingFace-compatible per-GPU max_memory dict using FRACTION of TOTAL memory.
    Example return: {0: "70GiB", 1: "70GiB", ...}
    """
    max_mem = {}
    if not torch.cuda.is_available():
        return max_mem
    n = torch.cuda.device_count()
    for i in range(n):
        # Use total*frac (safer than "free" which can be near-zero during loading)
        total = torch.cuda.get_device_properties(i).total_memory  # bytes
        cap = int(total * frac)
        cap_gib = max(1, cap // (1024 ** 3))
        max_mem[i] = f"{cap_gib}GiB"
    return max_mem


def str2bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    return x.lower() in ("1", "true", "t", "yes", "y")


# -------------------------------------------------------------------------------------------------
# Custom Trainer (no-op device move; teacher forward under no_grad; accepts **kwargs)
# -------------------------------------------------------------------------------------------------
class KDTrainer(Trainer):
    """
    - Prevents Trainer from relocating model (important for device_map="auto" sharded models)
    - Computes KD loss by running teacher forward with no_grad
    """
    def __init__(self, *args, teacher: nn.Module, T: float, alpha: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.T = float(T)
        self.alpha = float(alpha)

    # Avoid meta-tensor crash by not moving model away from HF/Accelerate placement
    def _move_model_to_device(self, model, device):
        return model

    # Newer HF may pass extra kwargs (e.g., num_items_in_batch); accept **kwargs
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        s_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        with torch.no_grad():
            t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        loss = kd_loss(
            s_out.logits, t_out.logits, labels, attention_mask,
            T=self.T, alpha=self.alpha
        )
        return (loss, s_out) if return_outputs else loss


# -------------------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train student with KD from teacher (Llama3.1)")

    # Precision
    ap.add_argument("--bf16", type=str, default=None,
                    help="True/False to force bfloat16; default: auto if supported.")
    ap.add_argument("--fp16", type=str, default=None,
                    help="True/False to force float16; default: opposite of bf16 when CUDA.")

    # Sequence length override
    ap.add_argument("--seq_len", type=int, default=None, help="Override seq_len from config.")

    # Memory & quantization
    ap.add_argument("--max_memory_frac", type=float, default=float(os.environ.get("MAX_MEMORY_FRAC", 0.90)),
                    help="Per-GPU total memory fraction cap for model loading (default 0.90).")
    ap.add_argument("--teacher_4bit", type=str, default="False",
                    help="Quantize teacher in 4-bit (requires bitsandbytes).")
    ap.add_argument("--teacher_8bit", type=str, default="False",
                    help="Quantize teacher in 8-bit (requires bitsandbytes).")

    # LoRA toggle
    ap.add_argument("--lora", type=str, default="True",
                    help="Use LoRA adapters for student (True/False). If False = full-parameter KD.")

    # Logging & callbacks
    ap.add_argument("--logging_steps", type=int, default=int(os.environ.get("LOGGING_STEPS", "20")))
    ap.add_argument("--use_mlflow_cb", type=str, default="True",
                    help="Attach MLflow GPU/power callback if available (True/False).")

    return ap.parse_args()


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------
def main():
    args_cli = parse_args()

    # Load project config
    cfg = Config.load().cfg
    P = cfg["paths"]
    M = cfg["models"]
    Tcfg = cfg["training"]

    if args_cli.seq_len is not None:
        Tcfg["seq_len"] = int(args_cli.seq_len)

    os.makedirs(P["outputs_dir"], exist_ok=True)
    os.makedirs(P["lora_dir"], exist_ok=True)

    MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))  # 0 => all samples

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(M["student_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # Precision selection
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
        use_bf16, use_fp16 = False, False

    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    # -------------------------
    # Per-GPU max_memory caps for safe sharding
    # -------------------------
    max_memory = make_max_memory_map(frac=float(args_cli.max_memory_frac)) if torch.cuda.is_available() else None

    # -------------------------
    # Teacher loader kwargs (with optional quantization)
    # -------------------------
    teacher_kwargs = dict(
        dtype=dtype,                     # use dtype unless quantized
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )

    teacher_4bit = str2bool(args_cli.teacher_4bit)
    teacher_8bit = str2bool(args_cli.teacher_8bit)

    if (teacher_4bit or teacher_8bit) and not _HAS_BNB:
        raise RuntimeError(
            "Requested teacher 4/8-bit quantization but bitsandbytes is not installed. "
            "Install it (pip install bitsandbytes) or disable the flags."
        )

    if teacher_4bit:
        teacher_kwargs.pop("dtype", None)
        teacher_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
        )
    elif teacher_8bit:
        teacher_kwargs.pop("dtype", None)
        teacher_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    # -------------------------
    # Load TEACHER (70B) & STUDENT (8B)
    # -------------------------
    print("Loading the WISE TEACHER:", M["teacher_id"])
    teacher = AutoModelForCausalLM.from_pretrained(M["teacher_id"], **teacher_kwargs)

    print("Loading the SMART STUDENT:", M["student_id"])
    student = AutoModelForCausalLM.from_pretrained(
        M["student_id"],
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )

    # Reduce activation memory
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    # -------------------------
    # LoRA (optional)
    # -------------------------
    use_lora = str2bool(args_cli.lora) if args_cli.lora is not None else True
    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        student = get_peft_model(student, lcfg)
    else:
        print("LoRA disabled → full-parameter KD fine-tune. Expect much higher VRAM usage.")

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
    data_files = [fp for fp in data_files if os.path.isfile(fp) and os.path.getsize(fp) > 0]

    ds = ChatJsonlReader(
        data_files,
        max_samples=(MAX_SAMPLES if MAX_SAMPLES > 0 else None),
    )
    if len(ds) == 0:
        raise FileNotFoundError("No training JSONL found. Run: python -m src.data_prep")

    collator = CausalCollator(tokenizer, max_seq_len=Tcfg["seq_len"])

    # -------------------------
    # TrainingArguments
    # -------------------------
    out_dir = os.path.join(P["outputs_dir"], "llama31_8b_kd_lora" if use_lora else "llama31_8b_kd_full")
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=Tcfg["epochs"],
        per_device_train_batch_size=Tcfg["per_device_batch_size"],
        gradient_accumulation_steps=Tcfg["grad_accum"],
        learning_rate=Tcfg["learning_rate"],
        warmup_ratio=0.03,
        logging_steps=int(args_cli.logging_steps),
        save_steps=500,
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to=[],                # no external logging integrations by default
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False, # safer for custom collators/LMs
    )

    # -------------------------
    # Trainer (attach MLflow callback if available & requested)
    # -------------------------
    callbacks = []
    use_cb = str2bool(args_cli.use_mlflow_cb) if args_cli.use_mlflow_cb is not None else True
    if use_cb and _HAS_MLFLOW_CB:
        callbacks.append(LiteMLflowCallback(log_gpu_every_n_steps=int(args_cli.logging_steps)))

    trainer = KDTrainer(
        model=student,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,  # deprecation warning okay; HF keeps backwards compat
        teacher=teacher,
        T=float(Tcfg.get("kd_temp", 1.0)),
        alpha=float(Tcfg.get("kd_alpha", 0.5)),
        callbacks=callbacks,
    )

    # -------------------------
    # Train
    # -------------------------
    trainer.train()

    # -------------------------
    # Save (LoRA adapters OR full model) + tokenizer
    # -------------------------
    save_dir = P["lora_dir"] if use_lora else out_dir
    print("Saving the student's knowledge to:", save_dir)
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("✅ Distillation complete.")


if __name__ == "__main__":
    warnings.filterwarnings("once", category=UserWarning)
    main()
