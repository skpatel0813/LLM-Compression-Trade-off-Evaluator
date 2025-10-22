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

  # Multi-GPU full precision (8 GPUs)
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m src.train_kd \
    --bf16 True --lora True --seq_len 2048
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
from transformers.trainer_utils import EvalLoopOutput

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
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Modest speedup on Ampere+ GPUs
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Reduce memory fragmentation on long training runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Respect custom HuggingFace cache directories if provided
for env_var in ("HF_HOME", "HF_DATASETS_CACHE"):
    v = os.environ.get(env_var)
    if v:
        os.makedirs(v, exist_ok=True)


# -------------------------------------------------------------------------------------------------
# Lightweight JSONL reader with train/val split support
# -------------------------------------------------------------------------------------------------
class ChatJsonlReader(Dataset):
    """
    Memory-efficient JSONL dataset reader using byte offsets.
    Supports training and validation splits from MBPP data.
    
    Each record should have a 'messages' field containing chat-formatted conversations.
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
        """Load a single example by seeking to its byte offset."""
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
    Data collator that:
    1. Converts {messages:[...]} into Llama-3 chat format
    2. Tokenizes the text
    3. Pads sequences to the same length
    4. Creates labels for language modeling (with -100 for padding positions)
    
    Returns a dictionary compatible with HuggingFace Trainer.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        self.tok = tokenizer
        self.max_len = max_seq_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Convert chat messages to Llama-3 formatted text
        texts = [build_chat_text(ex["messages"]) for ex in features]
        
        # Tokenize with padding and truncation
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Create labels (same as input_ids but with -100 for padding)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore padding in loss computation

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# -------------------------------------------------------------------------------------------------
# Knowledge Distillation loss and agreement metrics
# -------------------------------------------------------------------------------------------------
def _shift_for_next_token(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shift logits and labels for next-token prediction alignment.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Target token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
    
    Returns:
        Tuple of (shifted_logits, shifted_labels, valid_mask)
    """
    # Shift logits to align with next token prediction
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    shifted_attention = attention_mask[:, 1:]
    
    # Valid positions: attended tokens that aren't padding (-100)
    valid_mask = (shifted_attention > 0) & (shifted_labels != -100)
    
    return shifted_logits, shifted_labels, valid_mask


def compute_agreement_metrics(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute agreement metrics between student and teacher predictions.
    
    Metrics computed:
    - token_accuracy: Percentage of tokens where student's top-1 matches teacher's top-1
    - top5_agreement: Percentage of tokens where student's top-1 is in teacher's top-5
    - kl_divergence: KL divergence between student and teacher distributions
    
    Args:
        s_logits: Student model logits
        t_logits: Teacher model logits
        labels: Ground truth labels
        attention_mask: Attention mask
    
    Returns:
        Dictionary with agreement metrics
    """
    # Align predictions with next-token targets
    s_next, gold, valid_mask = _shift_for_next_token(s_logits, labels, attention_mask)
    t_next, _, _ = _shift_for_next_token(t_logits, labels, attention_mask)
    
    # Flatten and select only valid positions
    idx = valid_mask.view(-1)
    if idx.sum().item() == 0:
        # No valid tokens in batch (rare edge case)
        return {
            "token_accuracy": 0.0,
            "top5_agreement": 0.0,
            "kl_divergence": 0.0,
        }
    
    s_sel = s_next.view(-1, s_next.size(-1))[idx]
    t_sel = t_next.view(-1, t_next.size(-1))[idx]
    
    # Token accuracy: student's top-1 prediction matches teacher's top-1
    s_pred = s_sel.argmax(dim=-1)
    t_pred = t_sel.argmax(dim=-1)
    token_acc = (s_pred == t_pred).float().mean().item()
    
    # Top-5 agreement: student's top-1 appears in teacher's top-5
    t_top5 = t_sel.topk(5, dim=-1).indices
    s_pred_expanded = s_pred.unsqueeze(-1).expand_as(t_top5)
    top5_agree = (s_pred_expanded == t_top5).any(dim=-1).float().mean().item()
    
    # KL divergence between distributions
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
    Compute Knowledge Distillation loss.
    
    Loss formula:
        L = alpha * KL(student || teacher_T) * T^2 + (1 - alpha) * CE(student, gold)
    
    Where:
        - KL term: Distillation loss from teacher's soft targets
        - CE term: Standard cross-entropy loss with ground truth
        - T: Temperature for softening distributions
        - alpha: Balance between distillation and ground truth
    
    Args:
        s_logits: Student model logits
        t_logits: Teacher model logits
        labels: Ground truth token IDs
        attention_mask: Attention mask
        T: Temperature for distillation (default: 1.0)
        alpha: Weight for distillation loss (default: 0.5)
    
    Returns:
        Combined distillation loss
    """
    # Align predictions with next-token targets
    s_next, gold, valid_mask = _shift_for_next_token(s_logits, labels, attention_mask)
    t_next, _, _ = _shift_for_next_token(t_logits, labels, attention_mask)

    # Select only valid (non-padding) positions
    idx = valid_mask.view(-1)
    if idx.sum().item() == 0:
        # No valid tokens in this batch (edge case)
        return torch.zeros([], device=s_logits.device, dtype=s_logits.dtype)

    s_sel = s_next.view(-1, s_next.size(-1))[idx]
    t_sel = t_next.view(-1, t_next.size(-1))[idx]
    gold_sel = gold.view(-1)[idx]

    # Apply temperature scaling
    sT = s_sel / T
    tT = t_sel / T
    
    # KL divergence term (distillation from teacher)
    log_p_s = F.log_softmax(sT, dim=-1)
    p_t = F.softmax(tT, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

    # Cross-entropy term (learning from ground truth)
    ce = F.cross_entropy(s_sel, gold_sel, ignore_index=-100)
    
    # Weighted combination
    return alpha * kl + (1.0 - alpha) * ce


# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------
def make_max_memory_map(frac: float = 0.90) -> Dict[int, str]:
    """
    Create a per-GPU memory limit map for HuggingFace model loading.
    
    Uses a fraction of total GPU memory (not free memory) to avoid OOM during loading.
    
    Args:
        frac: Fraction of total GPU memory to use (default: 0.90 = 90%)
    
    Returns:
        Dictionary mapping GPU index to memory limit string (e.g., {0: "70GiB", 1: "70GiB"})
    """
    max_mem = {}
    if not torch.cuda.is_available():
        return max_mem
    
    n = torch.cuda.device_count()
    for i in range(n):
        # Use fraction of total memory (safer than using "free" which can be misleading)
        total = torch.cuda.get_device_properties(i).total_memory  # bytes
        cap = int(total * frac)
        cap_gib = max(1, cap // (1024 ** 3))  # Convert to GiB
        max_mem[i] = f"{cap_gib}GiB"
    
    return max_mem


def str2bool(x: Optional[str]) -> Optional[bool]:
    """Convert string to boolean (handles CLI arguments)."""
    if x is None:
        return None
    return x.lower() in ("1", "true", "t", "yes", "y")


# -------------------------------------------------------------------------------------------------
# Metrics Logging Callback
# -------------------------------------------------------------------------------------------------
class MetricsCallback(TrainerCallback):
    """
    Callback to log and save training/validation metrics to a JSONL file.
    
    Saves all metrics including:
    - Training loss
    - Validation loss
    - Token accuracy
    - Top-5 agreement
    - KL divergence
    
    Each metric is appended to training_metrics.jsonl with step and epoch info.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, "training_metrics.jsonl")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics file
        with open(self.metrics_file, "w") as f:
            f.write("")  # Create empty file
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called whenever the trainer logs metrics."""
        if logs:
            # Add step/epoch information
            logs["step"] = state.global_step
            logs["epoch"] = state.epoch
            
            # Append to JSONL file
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(logs) + "\n")
            
            # Print important metrics to console
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
    Custom HuggingFace Trainer for Knowledge Distillation.
    
    Key features:
    - Prevents automatic model device relocation (important for device_map="auto")
    - Computes KD loss using both student and teacher outputs
    - Tracks agreement metrics during evaluation
    - Returns proper EvalLoopOutput for compatibility
    """
    def __init__(self, *args, teacher: nn.Module, T: float, alpha: float, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Freeze teacher model
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        
        # Store KD hyperparameters
        self.T = float(T)
        self.alpha = float(alpha)

    def _move_model_to_device(self, model, device):
        """
        Override to prevent moving model to device.
        
        This is critical when using device_map="auto" as HuggingFace/Accelerate
        has already placed the model optimally across GPUs.
        """
        return model

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """
        Compute Knowledge Distillation loss during training.
        
        Args:
            model: Student model
            inputs: Batch of tokenized inputs
            return_outputs: Whether to return model outputs along with loss
            **kwargs: Additional arguments (for compatibility with newer HF versions)
        
        Returns:
            Loss tensor, or (loss, outputs) tuple if return_outputs=True
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # Student forward pass
        s_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )

        # Compute KD loss
        loss = kd_loss(
            s_out.logits, t_out.logits, labels, attention_mask,
            T=self.T, alpha=self.alpha
        )
        
        return (loss, s_out) if return_outputs else loss

    def evaluation_loop(
        self, 
        dataloader, 
        description, 
        prediction_loss_only=None, 
        ignore_keys=None, 
        metric_key_prefix="eval"
    ):
        """
        Custom evaluation loop that computes both loss and agreement metrics.
        
        This override allows us to:
        1. Run both student and teacher forward passes
        2. Compute KD loss on validation set
        3. Calculate agreement metrics between student and teacher
        
        Args:
            dataloader: Validation dataloader
            description: Description string for logging
            prediction_loss_only: Unused (for compatibility)
            ignore_keys: Unused (for compatibility)
            metric_key_prefix: Prefix for metric names (default: "eval")
        
        Returns:
            EvalLoopOutput containing metrics dictionary
        """
        model = self.model
        model.eval()
        
        all_losses = []
        all_agreement_metrics = defaultdict(list)
        
        print(f"\n[evaluation] Running {description}...")
        
        for step, inputs in enumerate(dataloader):
            # Move inputs to appropriate device
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                # Student forward pass
                s_out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False
                )
                
                # Teacher forward pass
                t_out = self.teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False
                )
                
                # Compute validation loss
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
            
            # Progress reporting
            if step % 10 == 0:
                print(f"  Eval step {step}/{len(dataloader)}")
        
        # Aggregate metrics across all batches
        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
        
        metrics_dict = {
            f"{metric_key_prefix}_loss": avg_loss,
        }
        
        for k, v_list in all_agreement_metrics.items():
            metrics_dict[f"{metric_key_prefix}_{k}"] = sum(v_list) / len(v_list) if v_list else 0.0
        
        # Print summary
        print(f"[evaluation] {description} complete:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Token Accuracy: {metrics_dict.get(f'{metric_key_prefix}_token_accuracy', 0):.4f}")
        print(f"  Top-5 Agreement: {metrics_dict.get(f'{metric_key_prefix}_top5_agreement', 0):.4f}")
        print(f"  KL Divergence: {metrics_dict.get(f'{metric_key_prefix}_kl_divergence', 0):.4f}")
        
        # Return proper EvalLoopOutput for HuggingFace Trainer compatibility
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics_dict,
            num_samples=len(dataloader.dataset) if hasattr(dataloader.dataset, '__len__') else len(all_losses)
        )


# -------------------------------------------------------------------------------------------------
# Command-line argument parsing
# -------------------------------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments for training configuration."""
    ap = argparse.ArgumentParser(
        description="Train student model with Knowledge Distillation on MBPP dataset"
    )

    # Precision options
    ap.add_argument(
        "--bf16", type=str, default=None,
        help="Use bfloat16 precision (True/False). Default: auto-detect if supported"
    )
    ap.add_argument(
        "--fp16", type=str, default=None,
        help="Use float16 precision (True/False). Default: opposite of bf16"
    )

    # Sequence length
    ap.add_argument(
        "--seq_len", type=int, default=None,
        help="Override max sequence length from config (default: from config)"
    )

    # Memory management
    ap.add_argument(
        "--max_memory_frac", type=float, default=0.90,
        help="Fraction of GPU memory to use per device (default: 0.90)"
    )
    ap.add_argument(
        "--teacher_4bit", type=str, default="False",
        help="Load teacher in 4-bit quantization (requires bitsandbytes)"
    )
    ap.add_argument(
        "--teacher_8bit", type=str, default="False",
        help="Load teacher in 8-bit quantization (requires bitsandbytes)"
    )

    # LoRA configuration
    ap.add_argument(
        "--lora", type=str, default="True",
        help="Use LoRA adapters for parameter-efficient training (True/False)"
    )

    # Logging and evaluation
    ap.add_argument(
        "--logging_steps", type=int, default=20,
        help="Log metrics every N training steps (default: 20)"
    )
    ap.add_argument(
        "--eval_steps", type=int, default=100,
        help="Run evaluation every N training steps (default: 100)"
    )
    ap.add_argument(
        "--use_mlflow_cb", type=str, default="True",
        help="Use MLflow callback for GPU monitoring if available (True/False)"
    )

    return ap.parse_args()


# -------------------------------------------------------------------------------------------------
# Main training function
# -------------------------------------------------------------------------------------------------
def main():
    """Main entry point for Knowledge Distillation training."""
    args_cli = parse_args()

    # Load project configuration
    cfg = Config.load().cfg
    P = cfg["paths"]
    M = cfg["models"]
    Tcfg = cfg["training"]

    # Override sequence length if specified
    if args_cli.seq_len is not None:
        Tcfg["seq_len"] = int(args_cli.seq_len)

    # Ensure output directories exist
    os.makedirs(P["outputs_dir"], exist_ok=True)

    print("\n" + "="*70)
    print("Knowledge Distillation Training on MBPP")
    print("="*70)

    # --------------------------
    # Step 1: Load tokenizer
    # --------------------------
    print("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(M["student_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded: {M['student_id']}")

    # --------------------------
    # Step 2: Configure precision (BF16/FP16/FP32)
    # --------------------------
    print("\n[2/7] Configuring precision...")
    if torch.cuda.is_available():
        # Auto-detect BF16 support
        auto_bf16 = torch.cuda.is_bf16_supported()
        force_bf16 = str2bool(args_cli.bf16)
        force_fp16 = str2bool(args_cli.fp16)

        # Determine BF16 usage
        if force_bf16 is True:
            use_bf16 = True
        elif force_bf16 is False:
            use_bf16 = False
        else:
            use_bf16 = auto_bf16

        # Determine FP16 usage
        if force_fp16 is True:
            use_fp16 = True
            use_bf16 = False  # FP16 takes precedence
        elif force_fp16 is False:
            use_fp16 = False
        else:
            use_fp16 = not use_bf16  # Use FP16 if not using BF16
    else:
        use_bf16, use_fp16 = False, False

    # Set dtype
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    print(f"✓ Precision: {dtype}")

    # --------------------------
    # Step 3: Configure memory limits per GPU
    # --------------------------
    max_memory = make_max_memory_map(frac=float(args_cli.max_memory_frac)) if torch.cuda.is_available() else None
    if max_memory:
        print(f"✓ Max memory per GPU: {list(max_memory.values())[0]}")

    # --------------------------
    # Step 4: Load teacher model (70B Llama-3.1)
    # --------------------------
    print("\n[3/7] Loading teacher model (70B)...")
    teacher_kwargs = dict(
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )

    # Configure quantization if requested
    teacher_4bit = str2bool(args_cli.teacher_4bit)
    teacher_8bit = str2bool(args_cli.teacher_8bit)

    if (teacher_4bit or teacher_8bit) and not _HAS_BNB:
        raise RuntimeError(
            "Quantization requested but bitsandbytes is not installed. "
            "Install with: pip install bitsandbytes"
        )

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

    # Load teacher model
    teacher = AutoModelForCausalLM.from_pretrained(M["teacher_id"], **teacher_kwargs)
    print(f"✓ Teacher loaded: {M['teacher_id']}")

    # --------------------------
    # Step 5: Load student model (8B Llama-3.1)
    # --------------------------
    print("\n[4/7] Loading student model (8B)...")
    student = AutoModelForCausalLM.from_pretrained(
        M["student_id"],
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )

    # Enable gradient checkpointing to save memory
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
    print(f"✓ Student loaded: {M['student_id']}")

    # --------------------------
    # Step 6: Apply LoRA adapters (optional)
    # --------------------------
    use_lora = str2bool(args_cli.lora) if args_cli.lora is not None else True
    if use_lora:
        print("\n[5/7] Applying LoRA adapters...")
        from peft import LoraConfig, get_peft_model, TaskType
        
        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha (scaling factor)
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj"  # MLP layers
            ],
            bias="none",
        )
        student = get_peft_model(student, lcfg)
        print("✓ LoRA applied")
    else:
        print("\n[5/7] Full-parameter training (no LoRA)")

    # Print trainable parameters
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"✓ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --------------------------
    # Step 7: Load MBPP datasets
    # --------------------------
    print("\n[6/7] Loading MBPP datasets...")
    train_files = [os.path.join(P["data_dir"], "mbpp_train.jsonl")]
    val_files = [os.path.join(P["data_dir"], "mbpp_val.jsonl")]
    
    # Verify files exist
    for f in train_files + val_files:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Missing {f}\n"
                f"Please run: python -m src.data_prep"
            )
    
    train_ds = ChatJsonlReader(train_files)
    val_ds = ChatJsonlReader(val_files)
    
    print(f"✓ Train: {len(train_ds)} examples")
    print(f"✓ Val: {len(val_ds)} examples")
    
    collator = CausalCollator(tokenizer, max_seq_len=Tcfg["seq_len"])

    # --------------------------
    # Step 8: Configure training arguments
    # --------------------------
    print("\n[7/7] Configuring training...")
    out_dir = os.path.join(
        P["outputs_dir"], 
        "llama31_8b_mbpp_kd_lora" if use_lora else "llama31_8b_mbpp_kd_full"
    )
    
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
        report_to=[],  # Disable external logging (wandb, tensorboard, etc.)
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
    # Step 9: Setup callbacks
    # --------------------------
    callbacks = [MetricsCallback(out_dir)]
    
    # Add MLflow callback if available and requested
    use_cb = str2bool(args_cli.use_mlflow_cb) if args_cli.use_mlflow_cb is not None else True
    if use_cb and _HAS_MLFLOW_CB:
        callbacks.append(LiteMLflowCallback(log_gpu_every_n_steps=int(args_cli.logging_steps)))

    # --------------------------
    # Step 10: Initialize trainer
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
    # Step 11: Train!
    # --------------------------
    trainer.train()

    # --------------------------
    # Step 12: Final evaluation
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
    
    # Save final metrics to JSON
    final_metrics_path = os.path.join(out_dir, "final_metrics.json")
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\n✓ Final metrics saved: {final_metrics_path}")

    # --------------------------
    # Step 13: Save trained model
    # --------------------------
    save_dir = P["lora_dir"] if use_lora else out_dir
    print(f"\n✓ Saving model to: {save_dir}")
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print("\n" + "="*70)
    print("✅ Knowledge Distillation Complete!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  Model: {save_dir}")
    print(f"  Metrics: {out_dir}/training_metrics.jsonl")
    print(f"  Final metrics: {final_metrics_path}")
    print("\nNext steps:")
    print("  1. Analyze metrics: python analyze_metrics.py")
    print("  2. Evaluate on test set: python src/eval_codebleu_hub.py ...")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Suppress minor warnings for cleaner output
    warnings.filterwarnings("once", category=UserWarning)
    main()