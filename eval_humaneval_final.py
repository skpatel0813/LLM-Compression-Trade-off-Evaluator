#!/usr/bin/env python3
"""Final version with proper indentation fixing."""

import argparse
import json
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


def fix_indentation(code: str) -> str:
    """Fix indentation issues in generated code."""
    lines = code.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            fixed_lines.append(line)
            continue
        
        # If line has content but no indentation, add 4 spaces
        if line and not line[0].isspace():
            fixed_lines.append('    ' + line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def extract_code_clean(text: str) -> str:
    """Extract and clean code from completion."""
    # Remove markdown
    text = re.sub(r'```python\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    lines = text.split('\n')
    code_lines = []
    in_main_block = False
    
    for line in lines:
        # Stop at if __name__ == "__main__"
        if 'if __name__' in line:
            break
        
        # Skip explanation lines
        stripped = line.strip().lower()
        if any(skip in stripped for skip in [
            'here is', 'here\'s', 'this code', 'this function',
            'explanation:', 'note that', 'the above', 'test'
        ]) and len(stripped) < 100:
            continue
        
        code_lines.append(line)
    
    code = '\n'.join(code_lines)
    
    # Fix indentation
    code = fix_indentation(code)
    
    return code.strip()


def generate_completion(model, tokenizer, prompt: str, args) -> str:
    """Generate completion."""
    
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
    
    # Extract completion
    if full.startswith(prompt):
        completion = full[len(prompt):]
    else:
        completion = full
    
    # Clean and fix
    completion = extract_code_clean(completion)
    
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
    
    # Show sample with full function
    if results:
        first_problem = list(problems.values())[0]
        sample = first_problem['prompt'] + results[0]['completion']
        print(f"\nSample (full function for {results[0]['task_id']}):")
        print("="*70)
        print(sample[:400])
        if len(sample) > 400:
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
