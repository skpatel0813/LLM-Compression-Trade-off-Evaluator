# src/train_kd.py
"""
Knowledge Distillation (KD) for Llama-3.1-8B (student) taught by Llama-3.1-70B (teacher)
with MBPP dataset, validation loss tracking, and agreement metrics.

Features:
  • Train on MBPP dataset
  • Track training AND validation loss
  • Compute agreement metrics (token accuracy, top-k agreement, KL divergence)
  • Multi-GPU safe with device_map="auto"
  • Optional teacher quantization (4-bit / 8-bit)
  • LoRA adapters for efficient training
  • BF16/FP16 precision controls

Usage:
  # Single GPU
  CUDA_VISIBLE_DEVICES=0 python -u -m src.train_kd --bf16 True --seq_len 2048

  # Multi-GPU with quantized teacher
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m src.train_kd \
    --bf16 True --teacher_4bit True --seq_len 2048
"""

from __future__ import annotations
import os
import json
import random
import warnings
import argparse
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

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
    TrainerCallback,
)

# Optional bitsandbytes for 4/8-bit quantization
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# Optional MLflow GPU/power logging callback
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

# Modest speedup on Ampere+
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Respect custom HF caches
for env_var in ("HF_HOME", "HF_DATASETS_CACHE"):
    v = os.environ.get(env_var)
    if v:
        os.makedirs(v, exist_ok=True)


# -------------------------------------------------------------------------------------------------
# Lightweight JSONL reader with train/val split
# -------------------------------------------------------------------------------------------------
class ChatJsonlReader(Dataset):
    """
    Reads JSONL files with byte offsets for memory efficiency.
    Supports train and validation splits.
    """
    def __init__(self, files: List[str], max_samples: Optional[int] = None):
        self.index: List[Tuple[str, int]] = []
        total = 0
        for fp in files:
            if not os.path.isfile(fp):
                print(f"[warning] File not found: {fp}")
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
        
        print(f"[dataset] Loaded {len(self.index)} examples from {len(files)} files")

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
    Converts {messages:[...]} into Llama-3 chat text, tokenizes, pads,
    and produces labels for language modeling.
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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# -------------------------------------------------------------------------------------------------
# KD helpers with agreement metrics
# -------------------------------------------------------------------------------------------------
def _shift_for_next_token(
    logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align predictions with next-token targets."""
    s_logits = logits[:, :-1, :]
    labels_shifted = labels[:, 1:]
    attn_shifted = attention_mask[:, 1:]
    valid = (attn_shifted > 0) & (labels_shifted != -100)
    return s_logits, labels_shifted, valid


def compute_agreement_metrics(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute agreement metrics between student and teacher:
    - token_accuracy: % of tokens where student's top-1 matches teacher's top-1
    - top5_agreement: % of tokens where student's top-1 is in teacher's top-5
    - kl_divergence: KL divergence between distributions
    """
    s_next, gold, valid_mask = _shift_for_next_token(s_logits, labels, attention_mask)
    t_next, _, _ = _shift_for_next_token(t_logits, labels, attention_mask)
    
    idx = valid_mask.view(-1)
    if idx.sum().item() == 0:
        return {
            "token_accuracy": 0.0,
            "top5_agreement": 0.0,
            "kl_divergence": 0.0,
        }
    
    s_sel = s_next.view(-1, s_next.size(-1))[idx]
    t_sel = t_next.view(-1, t_next.size(-1))[idx]
    
    # Token accuracy: student top-1 matches teacher top-1
    s_pred = s_sel.argmax(dim=-1)
    t_pred = t_sel.argmax(dim=-1)
    token_acc = (s_pred == t_pred).float().mean().item()
    
    # Top-5 agreement: student top-1 in teacher top-5
    t_top5 = t_sel.topk(5, dim=-1).indices
    s_pred_expanded = s_pred.unsqueeze(-1).expand_as(t_top5)
    top5_agree = (s_pred_expanded == t_top5).any(dim=-1).float().mean().item()
    
    # KL divergence
    log_p_s = F.log_softmax(s_sel, dim=-1)
    p_t = F.softmax(t_sel, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean").item()
    
    return {
        "token_accuracy": token_acc,
        "top5_agreement": top5_agree,
        "kl_divergence": kl,
    }


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
      L = alpha * KL(student || teacher_T) * T^2 + (1-alpha) * CE(student, gold)
    """
    s_next, gold, valid_mask = _shift_for_next_token(s_logits, labels, attention_mask)
    t_next, _, _ = _shift_for_next_token(t_logits, labels, attention_mask)

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


# -------------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------------
def make_max_memory_map(frac: float = 0.90) -> Dict[int, str]:
    """Build per-GPU max_memory dict using fraction of total memory."""
    max_mem = {}
    if not torch.cuda.is_available():
        return max_mem
    n = torch.cuda.device_count()
    for i in range(n):
        total = torch.cuda.get_device_properties(i).total_memory
        cap = int(total * frac)
        cap_gib = max(1, cap // (1024 ** 3))
        max_mem[i] = f"{cap_gib}GiB"
    return max_mem


def str2bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    return x.lower() in ("1", "true", "t", "yes", "y")


# -------------------------------------------------------------------------------------------------
# Metrics Logging Callback
# -------------------------------------------------------------------------------------------------
class MetricsCallback(TrainerCallback):
    """
    Callback to log and save training/validation metrics including agreement metrics.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, "training_metrics.jsonl")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics file
        with open(self.metrics_file, "w") as f:
            f.write("")  # Create empty file
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs."""
        if logs:
            # Add step/epoch info
            logs["step"] = state.global_step
            logs["epoch"] = state.epoch
            
            # Write to file
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(logs) + "\n")
            
            # Print important metrics
            if "loss" in logs:
                print(f"Step {state.global_step}: train_loss={logs['loss']:.4f}")
            if "eval_loss" in logs:
                print(f"Step {state.global_step}: val_loss={logs['eval_loss']:.4f}")
            if "eval_token_accuracy" in logs:
                print(f"  → token_accuracy={logs['eval_token_accuracy']:.4f}, "
                      f"top5_agreement={logs['eval_top5_agreement']:.4f}, "
                      f"kl_div={logs['eval_kl_divergence']:.4f}")


# -------------------------------------------------------------------------------------------------
# Custom Trainer with validation and agreement metrics
# -------------------------------------------------------------------------------------------------
class KDTrainer(Trainer):
    """
    Custom trainer that:
    - Prevents device relocation (for device_map="auto")
    - Computes KD loss with teacher forward under no_grad
    - Tracks agreement metrics during evaluation
    """
    def __init__(self, *args, teacher: nn.Module, T: float, alpha: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.T = float(T)
        self.alpha = float(alpha)

    def _move_model_to_device(self, model, device):
        """Avoid moving model (important for device_map='auto')."""
        return model

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """Compute KD loss during training."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # Student forward
        s_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Teacher forward (no grad)
        with torch.no_grad():
            t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Compute KD loss
        loss = kd_loss(
            s_out.logits, t_out.logits, labels, attention_mask,
            T=self.T, alpha=self.alpha
        )
        
        return (loss, s_out) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, 
                       ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluation loop to compute agreement metrics.
        """
        model = self.model
        model.eval()
        
        all_losses = []
        all_agreement_metrics = defaultdict(list)
        
        print(f"\n[evaluation] Running {description}...")
        
        for step, inputs in enumerate(dataloader):
            # Move inputs to device
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                # Student forward
                s_out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False
                )
                
                # Teacher forward
                t_out = self.teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False
                )
                
                # Compute loss
                loss = kd_loss(
                    s_out.logits, t_out.logits,
                    inputs["labels"], inputs["attention_mask"],
                    T=self.T, alpha=self.alpha
                )
                all_losses.append(loss.item())
                
                # Compute agreement metrics
                metrics = compute_agreement_metrics(
                    s_out.logits, t_out.logits,
                    inputs["labels"], inputs["attention_mask"]
                )
                for k, v in metrics.items():
                    all_agreement_metrics[k].append(v)
            
            # Progress
            if step % 10 == 0:
                print(f"  Eval step {step}/{len(dataloader)}")
        
        # Aggregate metrics
        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
        
        result = {
            f"{metric_key_prefix}_loss": avg_loss,
        }
        
        for k, v_list in all_agreement_metrics.items():
            result[f"{metric_key_prefix}_{k}"] = sum(v_list) / len(v_list) if v_list else 0.0
        
        print(f"[evaluation] {description} complete:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Token Accuracy: {result.get(f'{metric_key_prefix}_token_accuracy', 0):.4f}")
        print(f"  Top-5 Agreement: {result.get(f'{metric_key_prefix}_top5_agreement', 0):.4f}")
        print(f"  KL Divergence: {result.get(f'{metric_key_prefix}_kl_divergence', 0):.4f}")
        
        return result


# -------------------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train student with KD on MBPP")

    # Precision
    ap.add_argument("--bf16", type=str, default=None,
                    help="True/False for bfloat16; default: auto")
    ap.add_argument("--fp16", type=str, default=None,
                    help="True/False for float16; default: opposite of bf16")

    # Sequence length
    ap.add_argument("--seq_len", type=int, default=None,
                    help="Override seq_len from config")

    # Memory & quantization
    ap.add_argument("--max_memory_frac", type=float, default=0.90,
                    help="Per-GPU memory fraction cap (default 0.90)")
    ap.add_argument("--teacher_4bit", type=str, default="False",
                    help="Quantize teacher in 4-bit")
    ap.add_argument("--teacher_8bit", type=str, default="False",
                    help="Quantize teacher in 8-bit")

    # LoRA toggle
    ap.add_argument("--lora", type=str, default="True",
                    help="Use LoRA adapters (True/False)")

    # Logging
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--eval_steps", type=int, default=100,
                    help="Run evaluation every N steps")
    ap.add_argument("--use_mlflow_cb", type=str, default="True",
                    help="Use MLflow callback if available")

    return ap.parse_args()


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------
def main():
    args_cli = parse_args()

    # Load config
    cfg = Config.load().cfg
    P = cfg["paths"]
    M = cfg["models"]
    Tcfg = cfg["training"]

    if args_cli.seq_len is not None:
        Tcfg["seq_len"] = int(args_cli.seq_len)

    os.makedirs(P["outputs_dir"], exist_ok=True)

    print("\n" + "="*70)
    print("Knowledge Distillation Training on MBPP")
    print("="*70)

    # --------------------------
    # Tokenizer
    # --------------------------
    print("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(M["student_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded: {M['student_id']}")

    # --------------------------
    # Precision selection
    # --------------------------
    print("\n[2/7] Configuring precision...")
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
    print(f"✓ Precision: {dtype}")

    # --------------------------
    # Memory caps
    # --------------------------
    max_memory = make_max_memory_map(frac=float(args_cli.max_memory_frac)) if torch.cuda.is_available() else None
    if max_memory:
        print(f"✓ Max memory per GPU: {list(max_memory.values())[0]}")

    # --------------------------
    # Teacher kwargs
    # --------------------------
    print("\n[3/7] Loading teacher model (70B)...")
    teacher_kwargs = dict(
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )

    teacher_4bit = str2bool(args_cli.teacher_4bit)
    teacher_8bit = str2bool(args_cli.teacher_8bit)

    if (teacher_4bit or teacher_8bit) and not _HAS_BNB:
        raise RuntimeError("Quantization requested but bitsandbytes not installed")

    if teacher_4bit:
        teacher_kwargs.pop("dtype", None)
        teacher_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
        )
        print("✓ Using 4-bit quantization for teacher")
    elif teacher_8bit:
        teacher_kwargs.pop("dtype", None)
        teacher_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print("✓ Using 8-bit quantization for teacher")

    # Load teacher
    teacher = AutoModelForCausalLM.from_pretrained(M["teacher_id"], **teacher_kwargs)
    print(f"✓ Teacher loaded: {M['teacher_id']}")

    # --------------------------
    # Student
    # --------------------------
    print("\n[4/7] Loading student model (8B)...")
    student = AutoModelForCausalLM.from_pretrained(
        M["student_id"],
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )

    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
    print(f"✓ Student loaded: {M['student_id']}")

    # --------------------------
    # LoRA
    # --------------------------
    use_lora = str2bool(args_cli.lora) if args_cli.lora is not None else True
    if use_lora:
        print("\n[5/7] Applying LoRA adapters...")
        from peft import LoraConfig, get_peft_model, TaskType
        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        student = get_peft_model(student, lcfg)
        print("✓ LoRA applied")
    else:
        print("\n[5/7] Full-parameter training (no LoRA)")

    # Trainable params
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"✓ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --------------------------
    # Datasets (MBPP)
    # --------------------------
    print("\n[6/7] Loading MBPP datasets...")
    train_files = [os.path.join(P["data_dir"], "mbpp_train.jsonl")]
    val_files = [os.path.join(P["data_dir"], "mbpp_val.jsonl")]
    
    # Verify files exist
    for f in train_files + val_files:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Missing {f}\n"
                f"Run: python -m src.data_prep"
            )
    
    train_ds = ChatJsonlReader(train_files)
    val_ds = ChatJsonlReader(val_files)
    
    print(f"✓ Train: {len(train_ds)} examples")
    print(f"✓ Val: {len(val_ds)} examples")
    
    collator = CausalCollator(tokenizer, max_seq_len=Tcfg["seq_len"])

    # --------------------------
    # TrainingArguments
    # --------------------------
    print("\n[7/7] Configuring training...")
    out_dir = os.path.join(P["outputs_dir"], 
                          "llama31_8b_mbpp_kd_lora" if use_lora else "llama31_8b_mbpp_kd_full")
    
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=Tcfg["epochs"],
        per_device_train_batch_size=Tcfg["per_device_batch_size"],
        per_device_eval_batch_size=Tcfg["per_device_batch_size"],
        gradient_accumulation_steps=Tcfg["grad_accum"],
        learning_rate=Tcfg["learning_rate"],
        warmup_ratio=0.03,
        logging_steps=int(args_cli.logging_steps),
        eval_strategy="steps",
        eval_steps=int(args_cli.eval_steps),
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to=[],
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    print(f"✓ Output directory: {out_dir}")
    print(f"✓ Epochs: {Tcfg['epochs']}")
    print(f"✓ Batch size: {Tcfg['per_device_batch_size']}")
    print(f"✓ Gradient accumulation: {Tcfg['grad_accum']}")
    print(f"✓ Learning rate: {Tcfg['learning_rate']}")
    print(f"✓ Eval every: {args_cli.eval_steps} steps")

    # --------------------------
    # Callbacks
    # --------------------------
    callbacks = [MetricsCallback(out_dir)]
    
    use_cb = str2bool(args_cli.use_mlflow_cb) if args_cli.use_mlflow_cb is not None else True
    if use_cb and _HAS_MLFLOW_CB:
        callbacks.append(LiteMLflowCallback(log_gpu_every_n_steps=int(args_cli.logging_steps)))

    # --------------------------
    # Trainer
    # --------------------------
    print("\n" + "="*70)
    print("Starting Knowledge Distillation Training")
    print("="*70 + "\n")
    
    trainer = KDTrainer(
        model=student,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        teacher=teacher,
        T=float(Tcfg.get("kd_temp", 1.0)),
        alpha=float(Tcfg.get("kd_alpha", 0.5)),
        callbacks=callbacks,
    )

    # --------------------------
    # Train
    # --------------------------
    trainer.train()

    # --------------------------
    # Final evaluation
    # --------------------------
    print("\n" + "="*70)
    print("Running Final Evaluation")
    print("="*70 + "\n")
    
    final_metrics = trainer.evaluate()
    
    print("\nFinal Validation Metrics:")
    print(f"  Loss: {final_metrics['eval_loss']:.4f}")
    print(f"  Token Accuracy: {final_metrics['eval_token_accuracy']:.4f}")
    print(f"  Top-5 Agreement: {final_metrics['eval_top5_agreement']:.4f}")
    print(f"  KL Divergence: {final_metrics['eval_kl_divergence']:.4f}")
    
    # Save final metrics
    final_metrics_path = os.path.join(out_dir, "final_metrics.json")
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\n✓ Final metrics saved: {final_metrics_path}")

    # --------------------------
    # Save model
    # --------------------------
    save_dir = P["lora_dir"] if use_lora else out_dir
    print(f"\n✓ Saving model to: {save_dir}")
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print("\n" + "="*70)
    print("✅ Knowledge Distillation Complete!")
    print("="*70)
    print(f"\nModel saved to: {save_dir}")
    print(f"Metrics saved to: {out_dir}/training_metrics.jsonl")
    print(f"Final metrics: {final_metrics_path}")


if __name__ == "__main__":
    warnings.filterwarnings("once", category=UserWarning)
    main()