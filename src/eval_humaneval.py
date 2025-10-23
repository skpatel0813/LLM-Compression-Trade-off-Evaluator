#!/usr/bin/env python3
"""
eval_humaneval.py - HumanEval benchmark evaluation for LLMs (FIXED VERSION)

This version has improved code extraction for better HumanEval pass rates.
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

# HumanEval imports
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
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    
    return parser.parse_args()


def extract_code_from_completion(completion: str, prompt: str = "") -> str:
    """
    Extract clean Python code from model completion.
    
    The model often regenerates the prompt, so we need to:
    1. Remove the prompt if it appears
    2. Remove "assistant" markers
    3. Extract only the NEW function definition after the prompt
    """
    if not completion:
        return ""
    
    completion = completion.strip()
    
    # Remove markdown code fences
    if "```python" in completion:
        start = completion.find("```python") + len("```python")
        end = completion.find("```", start)
        if end != -1:
            completion = completion[start:end].strip()
    elif "```" in completion:
        start = completion.find("```") + 3
        end = completion.find("```", start)
        if end != -1:
            completion = completion[start:end].strip()
    
    # Remove "assistant" markers that appear in the middle
    completion = completion.replace("assistant\n", "\n")
    completion = completion.replace("assistant ", "")
    
    # If we have the prompt, try to find where the NEW code starts after it
    if prompt:
        # Look for the function signature from the prompt
        lines = prompt.strip().split('\n')
        func_signature = None
        for line in lines:
            if line.strip().startswith('def '):
                func_signature = line.strip()
                break
        
        if func_signature:
            # Find the SECOND occurrence of the function def (the new one)
            # First occurrence is the prompt, second is the completion
            parts = completion.split(func_signature)
            if len(parts) > 2:
                # Found it twice - take everything after the second occurrence
                completion = func_signature + parts[2]
            elif len(parts) == 2:
                # Found it once - might be the new one
                # Check if there's actual implementation after it
                after = parts[1].strip()
                if after and not after.startswith('"""') and not after.startswith("'''"):
                    completion = func_signature + parts[1]
    
    # Split into lines and clean
    lines = completion.split('\n')
    code_lines = []
    in_function = False
    found_def = False
    skip_docstring = False
    docstring_char = None
    
    for line in lines:
        stripped = line.strip()
        
        # Track if we're in a function definition
        if stripped.startswith('def '):
            found_def = True
            in_function = True
            code_lines.append(line)
            continue
        
        if not found_def:
            # Skip everything before the def
            continue
        
        # Handle docstrings
        if not skip_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
            docstring_char = '"""' if stripped.startswith('"""') else "'''"
            skip_docstring = True
            if stripped.endswith(docstring_char) and len(stripped) > 6:
                # Single-line docstring
                skip_docstring = False
            continue
        
        if skip_docstring:
            if docstring_char in stripped:
                skip_docstring = False
            continue
        
        # Add code lines
        if in_function:
            code_lines.append(line)
    
    code = '\n'.join(code_lines).strip()
    
    # Ensure newline at end
    if code and not code.endswith('\n'):
        code += '\n'
    
    return code


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
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    print(f"Precision: {dtype if dtype else 'Quantized'}")
    
    # Model kwargs
    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    
    if dtype:
        model_kwargs["torch_dtype"] = dtype
    
    # Quantization
    if args.load_in_4bit:
        if not HAS_BNB:
            raise RuntimeError("4-bit requires bitsandbytes")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        if not HAS_BNB:
            raise RuntimeError("8-bit requires bitsandbytes")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    
    # Load LoRA
    if args.lora_dir:
        if not HAS_PEFT:
            raise RuntimeError("LoRA requires peft")
        print(f"Loading LoRA from: {args.lora_dir}")
        model = PeftModel.from_pretrained(model, args.lora_dir)
        try:
            model = model.merge_and_unload()
        except:
            pass
    
    model.eval()
    
    print(f"{'='*70}\n")
    return model, tokenizer


def generate_completion(model, tokenizer, prompt: str, args) -> str:
    """Generate completion for a HumanEval problem."""
    # Cleaner system message - emphasize completing ONLY the body
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Complete the function body only. Do not rewrite the function signature or docstring."},
        {"role": "user", "content": f"Complete this function:\n\n{prompt}"}
    ]
    
    # Format
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    # Generate (greedy decoding)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=(args.temperature > 0.0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract completion
    if full_output.startswith(formatted_prompt):
        completion = full_output[len(formatted_prompt):].strip()
    else:
        completion = full_output
    
    # Clean with prompt context for better extraction
    completion = extract_code_from_completion(completion, prompt)
    
    return completion


def evaluate_humaneval(model, tokenizer, args):
    """Run HumanEval evaluation."""
    print(f"\n{'='*70}")
    print("HumanEval Evaluation")
    print(f"{'='*70}")
    
    # Load problems
    problems = read_problems()
    
    if args.limit:
        problems = dict(list(problems.items())[:args.limit])
    
    print(f"Total problems: {len(problems)}")
    print(f"{'='*70}\n")
    
    # Generate
    results = []
    start_time = time.time()
    
    # Disable tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    for task_id, problem in tqdm(problems.items(), desc="Generating"):
        prompt = problem["prompt"]
        
        try:
            completion = generate_completion(model, tokenizer, prompt, args)
            results.append({
                "task_id": task_id,
                "completion": completion,
            })
        except Exception as e:
            print(f"\nError on {task_id}: {e}")
            results.append({
                "task_id": task_id,
                "completion": "",
            })
    
    elapsed = time.time() - start_time
    print(f"\nGeneration: {elapsed:.1f}s ({elapsed/len(results):.2f}s/sample)")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_path), results)
    print(f"Saved: {output_path}")
    
    # Evaluate
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
        
        # Save metrics
        metrics_path = output_path.with_suffix('.metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'model': args.model,
                'lora_dir': args.lora_dir,
                'num_problems': len(problems),
                'metrics': metrics,
                'generation_time_sec': elapsed,
            }, f, indent=2)
        print(f"Metrics: {metrics_path}")
        
        return metrics
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None


def main():
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Evaluate
    metrics = evaluate_humaneval(model, tokenizer, args)
    
    if metrics:
        print(f"\n✓ Complete! Pass@1: {metrics['pass@1']*100:.2f}%")
    else:
        print(f"\n⚠ Completed with errors")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())