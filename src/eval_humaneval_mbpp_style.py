#!/usr/bin/env python3
"""
eval_humaneval_mbpp_style.py - Evaluate MBPP-trained models on HumanEval

This version is specifically designed for models trained on MBPP format.
MBPP models generate complete functions, so we extract the body for HumanEval.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
except ImportError:
    print("ERROR: human-eval package not found!")
    print("Install with: pip install human-eval")
    sys.exit(1)

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def extract_function_body(full_code: str, func_name: str) -> str:
    """
    Extract the body of a function from complete code.
    
    The model generates a complete function, but HumanEval needs just the body.
    """
    if not full_code or not func_name:
        return ""
    
    lines = full_code.split('\n')
    body_lines = []
    in_function = False
    indent_level = None
    
    for line in lines:
        # Find the function definition
        if f'def {func_name}(' in line:
            in_function = True
            # Don't include the def line itself
            continue
        
        if not in_function:
            continue
        
        # Skip docstrings
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        
        # Detect indent level from first real line
        if indent_level is None and line and not line[0].isspace():
            # No indent means the next def started
            break
        
        if indent_level is None and line and line[0].isspace():
            # Count leading spaces
            indent_level = len(line) - len(line.lstrip())
        
        # Check if we've exited the function (dedent or new def)
        if indent_level is not None:
            current_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level
            
            # If we hit a line with less or equal indent (and it's not empty), we're done
            if line.strip() and current_indent < indent_level:
                break
            
            # If we hit another def at same level, we're done
            if current_indent == 0 and line.strip().startswith('def '):
                break
        
        body_lines.append(line)
    
    body = '\n'.join(body_lines)
    
    # Ensure it ends with newline
    if body and not body.endswith('\n'):
        body += '\n'
    
    return body


def clean_generated_code(text: str) -> str:
    """Remove markdown fences and explanatory text."""
    if not text:
        return ""
    
    # Remove markdown fences
    text = re.sub(r'```python\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Remove common prefixes
    lines = text.split('\n')
    cleaned = []
    started = False
    
    for line in lines:
        stripped = line.strip().lower()
        
        # Skip intro text
        if not started:
            if any(p in stripped for p in ['here is', 'here\'s', 'certainly', 'sure']):
                continue
            if any(line.strip().startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ']):
                started = True
        
        if started:
            cleaned.append(line)
    
    return '\n'.join(cleaned)


def load_model_and_tokenizer(args):
    """Load model and tokenizer."""
    print(f"\n{'='*70}")
    print(f"Loading model: {args.model}")
    print(f"{'='*70}")
    
    if args.load_in_4bit or args.load_in_8bit:
        dtype = None
    elif args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    print(f"Precision: {dtype if dtype else 'Quantized'}")
    
    model_kwargs = {"device_map": "auto", "low_cpu_mem_usage": True}
    
    if dtype:
        model_kwargs["torch_dtype"] = dtype
    
    if args.load_in_4bit:
        if not HAS_BNB:
            raise RuntimeError("Need bitsandbytes")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        if not HAS_BNB:
            raise RuntimeError("Need bitsandbytes")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    
    if args.lora_dir:
        if not HAS_PEFT:
            raise RuntimeError("Need peft")
        print(f"Loading LoRA from: {args.lora_dir}")
        model = PeftModel.from_pretrained(model, args.lora_dir)
        try:
            model = model.merge_and_unload()
        except:
            pass
    
    model.eval()
    print(f"{'='*70}\n")
    return model, tokenizer


def generate_completion(model, tokenizer, task_id: str, prompt: str, args) -> str:
    """
    Generate completion using MBPP-style prompting.
    
    The key insight: MBPP models were trained to write complete functions
    from problem descriptions. So we give them a problem description style prompt.
    """
    # Extract function name from prompt
    func_match = re.search(r'def\s+(\w+)\s*\(', prompt)
    if not func_match:
        return ""
    func_name = func_match.group(1)
    
    # Create MBPP-style problem description from the docstring
    doc_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if doc_match:
        description = doc_match.group(1).strip()
        # Remove example lines (>>> ...)
        desc_lines = []
        for line in description.split('\n'):
            if not line.strip().startswith('>>>') and not line.strip().startswith('...'):
                desc_lines.append(line)
        description = '\n'.join(desc_lines).strip()
    else:
        description = f"Write a Python function called {func_name}"
    
    # MBPP-style prompt: problem description asking for a function
    messages = [
        {"role": "system", "content": "You are a helpful Python coding assistant."},
        {"role": "user", "content": f"{description}\n\nWrite the complete function."}
    ]
    
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=(args.temperature > 0.1),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part
    if full_output.startswith(formatted):
        generated = full_output[len(formatted):].strip()
    else:
        generated = full_output
    
    # Clean markdown/explanations
    generated = clean_generated_code(generated)
    
    # Extract just the body of the function
    body = extract_function_body(generated, func_name)
    
    return body


def evaluate_humaneval(model, tokenizer, args):
    """Run HumanEval evaluation."""
    print(f"\n{'='*70}")
    print("HumanEval Evaluation (MBPP-style prompting)")
    print(f"{'='*70}")
    
    problems = read_problems()
    
    if args.limit:
        problems = dict(list(problems.items())[:args.limit])
    
    print(f"Total problems: {len(problems)}")
    print(f"{'='*70}\n")
    
    results = []
    start_time = time.time()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    for task_id, problem in tqdm(problems.items(), desc="Generating"):
        prompt = problem["prompt"]
        
        try:
            completion = generate_completion(model, tokenizer, task_id, prompt, args)
            results.append({"task_id": task_id, "completion": completion})
        except Exception as e:
            print(f"\nError on {task_id}: {e}")
            results.append({"task_id": task_id, "completion": ""})
    
    elapsed = time.time() - start_time
    print(f"\nGeneration: {elapsed:.1f}s ({elapsed/len(results):.2f}s/sample)")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_path), results)
    print(f"Saved: {output_path}")
    
    # Debug: show first completion
    print("\n" + "="*70)
    print("Sample Completion (first problem):")
    print("="*70)
    print(results[0]["completion"][:300])
    print("="*70 + "\n")
    
    print(f"\n{'='*70}")
    print("Running tests...")
    print(f"{'='*70}\n")
    
    try:
        metrics = evaluate_functional_correctness(str(output_path))
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Pass@1: {metrics['pass@1']:.4f} ({metrics['pass@1']*100:.2f}%)")
        print(f"{'='*70}\n")
        
        metrics_path = output_path.with_suffix('.metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'model': args.model,
                'method': 'mbpp_style_prompting',
                'num_problems': len(problems),
                'metrics': metrics,
                'generation_time_sec': elapsed,
            }, f, indent=2)
        
        return metrics
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)
    metrics = evaluate_humaneval(model, tokenizer, args)
    
    if metrics:
        print(f"\n✓ Complete! Pass@1: {metrics['pass@1']*100:.2f}%")
    else:
        print(f"\n⚠ Completed with errors")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())