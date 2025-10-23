#!/usr/bin/env python3
"""Fixed version with better code extraction."""

import argparse
import json
import os
import sys
import time
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
except ImportError:
    print("ERROR: pip install human-eval")
    sys.exit(1)

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_model_and_tokenizer(args):
    print(f"\n{'='*70}")
    print(f"Loading: {args.model}")
    print(f"{'='*70}")
    
    if args.load_in_4bit or args.load_in_8bit:
        dtype = None
    elif args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("Using bfloat16")
    else:
        dtype = torch.float32
    
    model_kwargs = {"device_map": "auto", "low_cpu_mem_usage": True}
    if dtype:
        model_kwargs["torch_dtype"] = dtype
    
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    print("Model loaded!\n")
    
    return model, tokenizer


def extract_code_better(text: str, prompt: str) -> str:
    """
    Better code extraction that preserves the actual function implementation.
    """
    # The completion should continue from the prompt
    # Prompt ends with the function signature, completion should be the body
    
    # Remove markdown fences if present
    if "```python" in text:
        match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1)
    elif "```" in text:
        match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1)
    
    # If the model re-generated the function signature, find where the prompt ends
    # and take everything after
    lines = text.split('\n')
    
    # Look for the start of actual code (indented lines after def)
    code_lines = []
    found_code = False
    
    for line in lines:
        # Skip empty lines at the start
        if not found_code and not line.strip():
            continue
        
        # Skip obvious explanation text
        lower = line.strip().lower()
        if any(skip in lower for skip in ['here is', 'here\'s', 'this function', 'explanation:', 'note:']):
            if len(line.strip()) < 100:  # Short explanation lines
                continue
        
        # Once we find actual code, keep everything
        if line.strip() and (line.startswith('    ') or line.startswith('\t') or 'def ' in line or 'return' in line):
            found_code = True
        
        if found_code:
            code_lines.append(line)
    
    result = '\n'.join(code_lines).strip()
    
    # If we got nothing, return original
    if not result:
        return text.strip()
    
    return result


def generate_completion(model, tokenizer, prompt: str, args) -> str:
    """Generate completion using direct prompt continuation (not chat)."""
    
    # For HumanEval, we want to continue the function directly
    # Not use chat format - just complete the code
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.temperature > 0 else None,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # The completion is everything after the prompt
    if full.startswith(prompt):
        completion = full[len(prompt):]
    else:
        # Sometimes the model doesn't exactly match, find the end of prompt
        completion = full
    
    # Extract just the code
    completion = extract_code_better(completion, prompt)
    
    return completion


def evaluate_humaneval(model, tokenizer, args):
    print(f"\n{'='*70}")
    print("HumanEval Evaluation")
    print(f"{'='*70}")
    
    problems = read_problems()
    if args.limit:
        problems = dict(list(problems.items())[:args.limit])
    
    print(f"Problems: {len(problems)}")
    print(f"{'='*70}\n")
    
    results = []
    start = time.time()
    
    for task_id, problem in tqdm(problems.items(), desc="Generating"):
        prompt = problem["prompt"]
        
        try:
            completion = generate_completion(model, tokenizer, prompt, args)
            results.append({"task_id": task_id, "completion": completion})
        except Exception as e:
            print(f"\nError on {task_id}: {e}")
            results.append({"task_id": task_id, "completion": ""})
    
    elapsed = time.time() - start
    print(f"\nGeneration: {elapsed:.1f}s")
    
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output), results)
    print(f"Saved: {output}")
    
    # Show sample
    print(f"\nSample completion (first problem):")
    print("="*70)
    print(results[0]['completion'][:200])
    print("...")
    print("="*70)
    
    print(f"\nEvaluating...")
    try:
        metrics = evaluate_functional_correctness(str(output))
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Pass@1: {metrics['pass@1']:.4f} ({metrics['pass@1']*100:.2f}%)")
        print(f"{'='*70}\n")
        
        metrics_file = output.with_suffix('.metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump({
                'model': args.model,
                'quantization': '4bit' if args.load_in_4bit else ('8bit' if args.load_in_8bit else 'none'),
                'metrics': metrics,
                'time_sec': elapsed,
            }, f, indent=2)
        
        return metrics
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)
    metrics = evaluate_humaneval(model, tokenizer, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
