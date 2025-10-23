#!/usr/bin/env python3
"""
evaluate_combined.py - Evaluate a model on combined test set

This evaluates on the 10% test split from combined MBPP+HumanEval dataset
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
import argparse
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model path or ID")
    ap.add_argument("--lora_dir", default=None)
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--test_file", default="data/combined_test.jsonl")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.1)
    return ap.parse_args()


def load_model(args):
    """Load model and tokenizer."""
    print(f"Loading model: {args.model}")
    
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported() else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    if args.lora_dir:
        from peft import PeftModel
        print(f"Loading LoRA from: {args.lora_dir}")
        model = PeftModel.from_pretrained(model, args.lora_dir)
        try:
            model = model.merge_and_unload()
        except:
            pass
    
    model.eval()
    return model, tokenizer


def evaluate(model, tokenizer, test_file, args):
    """Run evaluation."""
    # Load test data
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Test examples: {len(test_data)}")
    
    # Generate predictions
    results = []
    correct = 0
    total = 0
    
    for example in tqdm(test_data, desc="Evaluating"):
        # Extract prompt and reference
        messages = example["messages"]
        prompt_messages = messages[:-1]  # All but last (assistant) message
        reference = messages[-1]["content"]
        
        # Format prompt
        formatted = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=(args.temperature > 0.1),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = full[len(formatted):].strip() if full.startswith(formatted) else full
        
        # Simple exact match check (you can improve this)
        if prediction.strip() == reference.strip():
            correct += 1
        total += 1
        
        results.append({
            "task_id": example.get("task_id"),
            "source": example.get("source"),
            "reference": reference,
            "prediction": prediction,
            "correct": (prediction.strip() == reference.strip())
        })
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    # Calculate per-source metrics
    humaneval_correct = sum(1 for r in results if r.get("source") == "humaneval" and r["correct"])
    humaneval_total = sum(1 for r in results if r.get("source") == "humaneval")
    
    mbpp_correct = sum(1 for r in results if r.get("source") == "mbpp" and r["correct"])
    mbpp_total = sum(1 for r in results if r.get("source") == "mbpp")
    
    metrics = {
        "model": args.model,
        "test_file": test_file,
        "total_examples": total,
        "overall_accuracy": accuracy,
        "humaneval_accuracy": humaneval_correct / humaneval_total if humaneval_total > 0 else 0,
        "humaneval_count": f"{humaneval_correct}/{humaneval_total}",
        "mbpp_accuracy": mbpp_correct / mbpp_total if mbpp_total > 0 else 0,
        "mbpp_count": f"{mbpp_correct}/{mbpp_total}"
    }
    
    return results, metrics


def main():
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model(args)
    
    # Evaluate
    results, metrics = evaluate(model, tokenizer, args.test_file, args)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    # Save metrics
    metrics_path = output_path.with_suffix('.metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
    print(f"HumanEval:        {metrics['humaneval_accuracy']:.4f} ({metrics['humaneval_count']})")
    print(f"MBPP:             {metrics['mbpp_accuracy']:.4f} ({metrics['mbpp_count']})")
    print("="*70)
    print(f"\nResults: {output_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()