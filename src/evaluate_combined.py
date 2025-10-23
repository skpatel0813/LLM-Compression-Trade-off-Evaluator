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


def normalize_code(code: str) -> str:
    """Normalize code for comparison."""
    import re
    # Remove comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    # Remove docstrings
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    # Remove extra whitespace
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    return '\n'.join(lines)


def code_similarity(pred: str, ref: str) -> float:
    """Compute simple code similarity (0-1)."""
    pred_norm = normalize_code(pred)
    ref_norm = normalize_code(ref)
    
    if not pred_norm or not ref_norm:
        return 0.0
    
    # Check if they're identical after normalization
    if pred_norm == ref_norm:
        return 1.0
    
    # Use token overlap
    pred_tokens = set(pred_norm.split())
    ref_tokens = set(ref_norm.split())
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    intersection = len(pred_tokens & ref_tokens)
    union = len(pred_tokens | ref_tokens)
    
    return intersection / union if union > 0 else 0.0


def evaluate(model, tokenizer, test_file, args):
    """Run evaluation."""
    # Load test data
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Test examples: {len(test_data)}")
    
    # Generate predictions
    results = []
    similarities = []
    
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
        
        # Compute similarity
        sim = code_similarity(prediction, reference)
        similarities.append(sim)
        
        results.append({
            "task_id": example.get("task_id"),
            "source": example.get("source"),
            "reference": reference,
            "prediction": prediction,
            "similarity": sim
        })
    
    # Calculate metrics
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    # Calculate per-source metrics
    humaneval_sims = [r["similarity"] for r in results if r.get("source") == "humaneval"]
    mbpp_sims = [r["similarity"] for r in results if r.get("source") == "mbpp"]
    
    metrics = {
        "model": args.model,
        "test_file": test_file,
        "total_examples": len(results),
        "overall_similarity": avg_similarity,
        "humaneval_similarity": sum(humaneval_sims) / len(humaneval_sims) if humaneval_sims else 0,
        "humaneval_count": len(humaneval_sims),
        "mbpp_similarity": sum(mbpp_sims) / len(mbpp_sims) if mbpp_sims else 0,
        "mbpp_count": len(mbpp_sims)
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