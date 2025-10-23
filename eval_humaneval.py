#!/usr/bin/env python3
"""
eval_humaneval.py - HumanEval benchmark evaluation for LLMs
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Check for human-eval package
try:
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
except ImportError:
    print("ERROR: human-eval package not found!")
    print("Install with: pip install human-eval")
    sys.exit(1)

# Optional: LoRA support
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# Optional: BitsAndBytes for quantization
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM on HumanEval benchmark")
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, default=None)
    
    # Quantization
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true")
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1)
    
    # Evaluation
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    
    return parser.parse_args()


def load_model_and_tokenizer(args):
    """Load model and tokenizer."""
    print(f"\n{'='*70}")
    print(f"Loading model: {args.model}")
    print(f"{'='*70}")
    
    # Determine dtype
    if args.load_in_4bit or args.load_in_8bit:
        dtype = None
    elif args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("Using bfloat16 precision")
    elif args.fp16:
        dtype = torch.float16
        print("Using float16 precision")
    else:
        dtype = torch.float32
        print("Using float32 precision")
    
    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    
    if dtype:
        model_kwargs["torch_dtype"] = dtype
    
    # Quantization
    if args.load_in_4bit:
        if not HAS_BNB:
            raise RuntimeError("4-bit needs bitsandbytes: pip install bitsandbytes")
        print("Using 4-bit quantization")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        if not HAS_BNB:
            raise RuntimeError("8-bit needs bitsandbytes: pip install bitsandbytes")
        print("Using 8-bit quantization")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    
    # LoRA
    if args.lora_dir:
        if not HAS_PEFT:
            raise RuntimeError("LoRA needs peft: pip install peft")
        print(f"Loading LoRA from: {args.lora_dir}")
        model = PeftModel.from_pretrained(model, args.lora_dir)
        try:
            model = model.merge_and_unload()
        except Exception as e:
            print(f"Warning: Could not merge LoRA: {e}")
    
    model.eval()
    print(f"Model loaded successfully!")
    print(f"{'='*70}\n")
    
    return model, tokenizer


def extract_code(completion: str) -> str:
    """Extract code from completion."""
    completion = completion.strip()
    
    # Try markdown fence
    if "```python" in completion:
        start = completion.find("```python") + len("```python")
        end = completion.find("```", start)
        if end != -1:
            return completion[start:end].strip()
    elif "```" in completion:
        start = completion.find("```") + 3
        end = completion.find("```", start)
        if end != -1:
            return completion[start:end].strip()
    
    # Remove explanation lines
    lines = completion.split('\n')
    code_lines = []
    skip = ['here is', 'here\'s', 'this function', 'explanation:']
    
    for line in lines:
        lower = line.strip().lower()
        if any(p in lower for p in skip) and len(line.strip()) < 100:
            continue
        code_lines.append(line)
    
    return '\n'.join(code_lines).strip()


def generate_completion(model, tokenizer, prompt: str, args) -> str:
    """Generate completion."""
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Complete the function. Only provide the implementation, no explanations."},
        {"role": "user", "content": prompt}
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if full.startswith(formatted):
        completion = full[len(formatted):].strip()
    else:
        completion = full
    
    return extract_code(completion)


def evaluate_humaneval(model, tokenizer, args):
    """Run evaluation."""
    print(f"\n{'='*70}")
    print("HumanEval Evaluation")
    print(f"{'='*70}")
    
    problems = read_problems()
    
    if args.limit:
        problems = dict(list(problems.items())[:args.limit])
        print(f"Limited to {args.limit} problems")
    
    print(f"Total problems: {len(problems)}")
    print(f"{'='*70}\n")
    
    results = []
    start = time.time()
    
    for task_id, problem in tqdm(problems.items(), desc="Generating"):
        prompt = problem["prompt"]
        
        for _ in range(args.num_samples):
            try:
                completion = generate_completion(model, tokenizer, prompt, args)
                results.append({"task_id": task_id, "completion": completion})
            except Exception as e:
                print(f"\nError on {task_id}: {e}")
                results.append({"task_id": task_id, "completion": ""})
    
    elapsed = time.time() - start
    print(f"\nGeneration: {elapsed:.1f}s ({elapsed/len(results):.2f}s per sample)")
    
    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output), results)
    print(f"Saved: {output}")
    
    # Evaluate
    print(f"\n{'='*70}")
    print("Evaluating...")
    print(f"{'='*70}\n")
    
    try:
        metrics = evaluate_functional_correctness(str(output))
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Pass@1: {metrics['pass@1']:.4f} ({metrics['pass@1']*100:.2f}%)")
        print(f"{'='*70}\n")
        
        # Save metrics
        metrics_file = output.with_suffix('.metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump({
                'model': args.model,
                'quantization': '4bit' if args.load_in_4bit else ('8bit' if args.load_in_8bit else 'none'),
                'metrics': metrics,
                'time_sec': elapsed,
            }, f, indent=2)
        print(f"Metrics saved: {metrics_file}\n")
        
        return metrics
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None


def main():
    args = parse_args()
    
    if args.load_in_4bit and args.load_in_8bit:
        print("ERROR: Cannot use both 4bit and 8bit")
        sys.exit(1)
    
    model, tokenizer = load_model_and_tokenizer(args)
    metrics = evaluate_humaneval(model, tokenizer, args)
    
    if metrics:
        print(f"✓ Complete! Pass@1: {metrics['pass@1']*100:.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
