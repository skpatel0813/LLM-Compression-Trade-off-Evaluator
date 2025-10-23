#!/usr/bin/env python3
"""
train_kd_combined.py - Train KD on combined MBPP+HumanEval dataset

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_kd_combined.py \
    --bf16 True \
    --lora True \
    --epochs 5 \
    --seq_len 2048
"""

import sys
import argparse

# Import the existing train_kd module
from src.train_kd import *

def parse_combined_args():
    """Parse args specifically for combined dataset training."""
    ap = argparse.ArgumentParser(description="Train KD on combined MBPP+HumanEval")
    
    # Precision
    ap.add_argument("--bf16", type=str, default=None)
    ap.add_argument("--fp16", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=2048)
    
    # Memory
    ap.add_argument("--max_memory_frac", type=float, default=0.90)
    ap.add_argument("--teacher_4bit", type=str, default="False")
    ap.add_argument("--teacher_8bit", type=str, default="False")
    
    # LoRA
    ap.add_argument("--lora", type=str, default="True")
    
    # Training
    ap.add_argument("--epochs", type=int, default=5,
                    help="Number of training epochs (default: 5)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    
    # Logging
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--use_mlflow_cb", type=str, default="True")
    
    return ap.parse_args()


def main():
    """Train on combined dataset."""
    args_cli = parse_combined_args()
    
    print("\n" + "="*70)
    print("Knowledge Distillation on Combined MBPP+HumanEval Dataset")
    print("="*70)
    
    # Check if combined dataset exists
    required_files = [
        "data/combined_train.jsonl",
        "data/combined_val.jsonl",
        "data/combined_test.jsonl"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("\n❌ ERROR: Combined dataset not found!")
        print("Missing files:")
        for f in missing:
            print(f"  - {f}")
        print("\nRun this first:")
        print("  python prepare_combined_dataset.py")
        sys.exit(1)
    
    # Load base config
    cfg = Config.load().cfg
    P = cfg["paths"]
    M = cfg["models"]
    Tcfg = cfg["training"]
    
    # Override with CLI args
    Tcfg["seq_len"] = args_cli.seq_len
    Tcfg["epochs"] = args_cli.epochs
    Tcfg["per_device_batch_size"] = args_cli.batch_size
    Tcfg["grad_accum"] = args_cli.grad_accum
    Tcfg["learning_rate"] = args_cli.learning_rate
    
    # Override dataset paths to use combined data
    train_files = ["data/combined_train.jsonl"]
    val_files = ["data/combined_val.jsonl"]
    
    # Count examples
    def count_lines(fp):
        with open(fp) as f:
            return sum(1 for _ in f)
    
    n_train = count_lines(train_files[0])
    n_val = count_lines(val_files[0])
    
    print(f"\nDataset:")
    print(f"  Training:   {n_train} examples")
    print(f"  Validation: {n_val} examples")
    print(f"  Epochs:     {Tcfg['epochs']}")
    
    # Ensure output dirs exist
    os.makedirs(P["outputs_dir"], exist_ok=True)
    
    # Load tokenizer
    print("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(M["student_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded")
    
    # Configure precision
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
    
    # Memory limits
    max_memory = make_max_memory_map(frac=float(args_cli.max_memory_frac)) if torch.cuda.is_available() else None
    if max_memory:
        print(f"✓ Max memory per GPU: {list(max_memory.values())[0]}")
    
    # Load teacher
    print("\n[3/7] Loading teacher model (70B)...")
    teacher_kwargs = dict(
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )
    
    teacher_4bit = str2bool(args_cli.teacher_4bit)
    teacher_8bit = str2bool(args_cli.teacher_8bit)
    
    if teacher_4bit:
        teacher_kwargs.pop("dtype", None)
        teacher_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
        )
        print("✓ Using 4-bit quantization")
    elif teacher_8bit:
        teacher_kwargs.pop("dtype", None)
        teacher_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print("✓ Using 8-bit quantization")
    
    teacher = AutoModelForCausalLM.from_pretrained(M["teacher_id"], **teacher_kwargs)
    print(f"✓ Teacher loaded")
    
    # Load student
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
    print(f"✓ Student loaded")
    
    # Apply LoRA
    use_lora = str2bool(args_cli.lora) if args_cli.lora is not None else True
    if use_lora:
        print("\n[5/7] Applying LoRA adapters...")
        from peft import LoraConfig, get_peft_model, TaskType
        
        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
        student = get_peft_model(student, lcfg)
        print("✓ LoRA applied")
    else:
        print("\n[5/7] Full-parameter training (no LoRA)")
    
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"✓ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Load datasets
    print("\n[6/7] Loading datasets...")
    train_ds = ChatJsonlReader(train_files)
    val_ds = ChatJsonlReader(val_files)
    print(f"✓ Train: {len(train_ds)} examples")
    print(f"✓ Val: {len(val_ds)} examples")
    
    collator = CausalCollator(tokenizer, max_seq_len=Tcfg["seq_len"])
    
    # Configure training
    print("\n[7/7] Configuring training...")
    out_dir = os.path.join(
        P["outputs_dir"],
        "llama31_8b_combined_kd_lora" if use_lora else "llama31_8b_combined_kd_full"
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
        save_steps=200,
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
    
    print(f"✓ Output: {out_dir}")
    print(f"✓ Epochs: {Tcfg['epochs']}")
    print(f"✓ Batch size: {Tcfg['per_device_batch_size']}")
    print(f"✓ Grad accum: {Tcfg['grad_accum']}")
    print(f"✓ Learning rate: {Tcfg['learning_rate']}")
    
    # Setup callbacks
    callbacks = [MetricsCallback(out_dir)]
    use_cb = str2bool(args_cli.use_mlflow_cb) if args_cli.use_mlflow_cb is not None else True
    if use_cb and _HAS_MLFLOW_CB:
        callbacks.append(LiteMLflowCallback(log_gpu_every_n_steps=int(args_cli.logging_steps)))
    
    # Initialize trainer
    print("\n" + "="*70)
    print("Starting Training on Combined Dataset")
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
    
    # Train!
    trainer.train()
    
    # Final evaluation
    print("\n" + "="*70)
    print("Final Evaluation")
    print("="*70 + "\n")
    
    final_metrics = trainer.evaluate()
    
    print("\nFinal Metrics:")
    print(f"  Loss: {final_metrics['eval_loss']:.4f}")
    print(f"  Token Accuracy: {final_metrics['eval_token_accuracy']:.4f}")
    print(f"  Top-5 Agreement: {final_metrics['eval_top5_agreement']:.4f}")
    print(f"  KL Divergence: {final_metrics['eval_kl_divergence']:.4f}")
    
    # Save
    final_metrics_path = os.path.join(out_dir, "final_metrics.json")
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    save_dir = P["lora_dir"] if use_lora else out_dir
    print(f"\n✓ Saving model to: {save_dir}")
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"\nModel: {save_dir}")
    print(f"Metrics: {out_dir}/training_metrics.jsonl")
    print("\nNext: Evaluate all models")
    print("  bash evaluate_all_models.sh")
    print("="*70 + "\n")


if __name__ == "__main__":
    warnings.filterwarnings("once", category=UserWarning)
    main()