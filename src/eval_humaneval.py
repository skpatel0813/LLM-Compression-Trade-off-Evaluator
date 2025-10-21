#!/usr/bin/env python3
"""
eval_humaneval.py - HumanEval benchmark evaluation for LLMs

Evaluates code generation models on the HumanEval benchmark with pass@k metrics.
Supports quantization, LoRA adapters, and various generation strategies.

Usage:
    # Evaluate 70B teacher
    python eval_humaneval.py \
        --model meta-llama/Meta-Llama-3.1-70B-Instruct \
        --output results/teacher_humaneval.jsonl
    
    # Evaluate 8B student with 4-bit quantization
    python eval_humaneval.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --load_in_4bit \
        --output results/student_4bit_humaneval.jsonl
    
    # Evaluate with LoRA adapter
    python eval_humaneval.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --lora_dir outputs/llama31_8b_kd_lora/lora \
        --output results/student_kd_humaneval.jsonl
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
    parser.add_argument("--model", type=str, required=True,
                        help="Model ID or path (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--lora_dir", type=str, default=None,
                        help="Path to LoRA adapter directory")
    
    # Quantization options
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit precision")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit precision")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 precision (default: True if supported)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 precision")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate per problem")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (use 0.1 for near-greedy)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling top-p")
    parser.add_argument("--do_sample", action="store_true",
                        help="Use sampling instead of greedy decoding")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per problem (for pass@k with k>1)")
    
    # Evaluation options
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of problems to evaluate (for testing)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation (default: 1 for safety)")
    
    # System options
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (default: auto)")
    
    return parser.parse_args()


def load_model_and_tokenizer(args):
    """Load model and tokenizer with specified configuration."""
    print(f"\n{'='*70}")
    print(f"Loading model: {args.model}")
    print(f"{'='*70}")
    
    # Determine dtype
    if args.load_in_4bit or args.load_in_8bit:
        dtype = None  # Quantization handles dtype
    elif args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    print(f"Precision: {dtype if dtype else 'Quantized'}")
    
    # Prepare model kwargs
    model_kwargs = {
        "device_map": args.device,
        "low_cpu_mem_usage": True,
    }
    
    if dtype:
        model_kwargs["torch_dtype"] = dtype
    
    # Quantization config
    if args.load_in_4bit:
        if not HAS_BNB:
            raise RuntimeError("4-bit quantization requires bitsandbytes. Install: pip install bitsandbytes")
        print("Using 4-bit quantization")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        if not HAS_BNB:
            raise RuntimeError("8-bit quantization requires bitsandbytes. Install: pip install bitsandbytes")
        print("Using 8-bit quantization")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For batch generation
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    
    # Load LoRA if specified
    if args.lora_dir:
        if not HAS_PEFT:
            raise RuntimeError("LoRA loading requires peft. Install: pip install peft")
        print(f"Loading LoRA adapter from: {args.lora_dir}")
        model = PeftModel.from_pretrained(model, args.lora_dir)
        # Merge adapter for faster inference
        try:
            model = model.merge_and_unload()
        except Exception as e:
            print(f"Warning: Could not merge LoRA adapter: {e}")
    
    model.eval()
    
    # Print model info
    if torch.cuda.is_available():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Try to estimate memory usage
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU memory allocated: {memory_allocated:.2f} GB")
        except:
            pass
    
    print(f"{'='*70}\n")
    
    return model, tokenizer


def extract_code_from_completion(completion: str) -> str:
    """
    Extract code from model completion, removing markdown fences and explanations.
    """
    # Remove common preambles
    completion = completion.strip()
    
    # Try to find code in markdown fences
    if "```python" in completion:
        # Extract code between ```python and ```
        start = completion.find("```python") + len("```python")
        end = completion.find("```", start)
        if end != -1:
            return completion[start:end].strip()
    elif "```" in completion:
        # Generic code fence
        start = completion.find("```") + 3
        end = completion.find("```", start)
        if end != -1:
            return completion[start:end].strip()
    
    # Remove common explanation patterns
    lines = completion.split('\n')
    code_lines = []
    skip_phrases = ['here is', 'here\'s', 'this function', 'this code', 'explanation:']
    
    for line in lines:
        lower = line.strip().lower()
        # Skip explanatory lines
        if any(phrase in lower for phrase in skip_phrases) and len(line.strip()) < 100:
            continue
        code_lines.append(line)
    
    return '\n'.join(code_lines).strip()


def generate_completion(model, tokenizer, prompt: str, args) -> str:
    """Generate a single completion for a prompt."""
    # Format prompt for chat models
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Complete the following function. Only provide the function implementation, no explanations."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the completion (remove prompt)
    if full_output.startswith(formatted_prompt):
        completion = full_output[len(formatted_prompt):].strip()
    else:
        completion = full_output
    
    # Clean the completion
    completion = extract_code_from_completion(completion)
    
    return completion


def evaluate_humaneval(model, tokenizer, args):
    """Run HumanEval evaluation."""
    print(f"\n{'='*70}")
    print("HumanEval Evaluation")
    print(f"{'='*70}")
    
    # Load problems
    problems = read_problems()
    
    # Limit for testing
    if args.limit:
        problems = dict(list(problems.items())[:args.limit])
        print(f"Limited to {args.limit} problems for testing")
    
    print(f"Total problems: {len(problems)}")
    print(f"Samples per problem: {args.num_samples}")
    print(f"{'='*70}\n")
    
    # Generate completions
    results = []
    start_time = time.time()
    
    for task_id, problem in tqdm(problems.items(), desc="Generating"):
        prompt = problem["prompt"]
        
        # Generate multiple samples if requested
        for sample_idx in range(args.num_samples):
            try:
                completion = generate_completion(model, tokenizer, prompt, args)
                
                results.append({
                    "task_id": task_id,
                    "completion": completion,
                })
            except Exception as e:
                print(f"\nError on {task_id} (sample {sample_idx}): {e}")
                # Add empty completion to maintain count
                results.append({
                    "task_id": task_id,
                    "completion": "",
                })
    
    elapsed = time.time() - start_time
    print(f"\nGeneration completed in {elapsed:.1f}s ({elapsed/len(results):.2f}s per sample)")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_path), results)
    print(f"Results saved to: {output_path}")
    
    # Evaluate
    print(f"\n{'='*70}")
    print("Running functional correctness evaluation...")
    print(f"{'='*70}\n")
    
    try:
        metrics = evaluate_functional_correctness(str(output_path))
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Pass@1: {metrics['pass@1']:.4f} ({metrics['pass@1']*100:.2f}%)")
        if args.num_samples > 1:
            for k in sorted([k for k in metrics.keys() if k.startswith('pass@')]):
                if k != 'pass@1':
                    print(f"{k}: {metrics[k]:.4f} ({metrics[k]*100:.2f}%)")
        print(f"{'='*70}\n")
        
        # Save metrics
        metrics_path = output_path.with_suffix('.metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'model': args.model,
                'lora_dir': args.lora_dir,
                'quantization': '4bit' if args.load_in_4bit else ('8bit' if args.load_in_8bit else 'none'),
                'num_problems': len(problems),
                'num_samples': args.num_samples,
                'metrics': metrics,
                'generation_time_sec': elapsed,
            }, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        
        return metrics
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Results were saved but could not be evaluated.")
        print("You may need to install additional dependencies or check the output format.")
        return None


def main():
    args = parse_args()
    
    # Validation
    if args.load_in_4bit and args.load_in_8bit:
        print("ERROR: Cannot use both 4-bit and 8-bit quantization")
        sys.exit(1)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Run evaluation
    metrics = evaluate_humaneval(model, tokenizer, args)
    
    if metrics:
        print("\n✓ Evaluation complete!")
        print(f"  Pass@1: {metrics['pass@1']*100:.2f}%")
    else:
        print("\n⚠ Evaluation completed with errors")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())