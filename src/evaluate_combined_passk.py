#!/usr/bin/env python3
"""
evaluate_combined_passk.py - Evaluate with Pass@k on combined test set

This uses actual code execution to test functional correctness.
"""

import json
import torch
import os
import tempfile
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora_dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--test_file", default="data/combined_test.jsonl")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--n_samples", type=int, default=1, help="Samples per problem for pass@k")
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


def execute_code(code: str, test_code: str, timeout: int = 5) -> bool:
    """Execute code with tests and return pass/fail."""
    full_code = code + "\n\n" + test_code
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        os.unlink(temp_file)
        
        # If no error and no assertion failures, it passes
        return result.returncode == 0
    
    except subprocess.TimeoutExpired:
        try:
            os.unlink(temp_file)
        except:
            pass
        return False
    except Exception as e:
        return False


def extract_code_from_completion(completion: str) -> str:
    """Extract Python code from model output."""
    import re
    
    # Remove markdown fences
    completion = re.sub(r'```python\s*', '', completion)
    completion = re.sub(r'```\s*', '', completion)
    
    # Find function definition
    lines = []
    in_function = False
    
    for line in completion.split('\n'):
        stripped = line.strip()
        
        # Start collecting at def
        if stripped.startswith('def '):
            in_function = True
        
        if in_function:
            lines.append(line)
            
            # Stop at next def or class at same indent level
            if line and not line[0].isspace() and stripped.startswith(('def ', 'class ')) and len(lines) > 1:
                lines.pop()  # Remove the new def/class
                break
    
    return '\n'.join(lines)


def evaluate(model, tokenizer, test_file, args):
    """Run Pass@k evaluation."""
    # Load test data
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Test examples: {len(test_data)}")
    print(f"Samples per problem: {args.n_samples}")
    
    results = []
    
    for example in tqdm(test_data, desc="Evaluating"):
        task_id = example.get("task_id")
        source = example.get("source", "unknown")
        messages = example["messages"]
        
        # Get test code
        test_code = example.get("test", "")
        if not test_code:
            # Try to extract from messages
            for msg in messages:
                if "assert" in msg.get("content", ""):
                    test_code = msg["content"]
                    break
        
        # Generate n_samples completions
        passed = False
        completions = []
        
        for _ in range(args.n_samples):
            # Format prompt
            prompt_messages = messages[:-1]
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
                    do_sample=(args.temperature > 0.0 and args.n_samples > 1),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = full[len(formatted):].strip() if full.startswith(formatted) else full
            
            # Extract code
            code = extract_code_from_completion(completion)
            completions.append(code)
            
            # Test if we have test code
            if test_code and code:
                if execute_code(code, test_code):
                    passed = True
                    break  # At least one passed
        
        results.append({
            "task_id": task_id,
            "source": source,
            "completions": completions,
            "passed": passed,
            "has_tests": bool(test_code)
        })
    
    # Calculate Pass@k metrics
    total = len(results)
    total_with_tests = sum(1 for r in results if r["has_tests"])
    passed = sum(1 for r in results if r["passed"])
    
    # Per-source
    humaneval_results = [r for r in results if r["source"] == "humaneval"]
    mbpp_results = [r for r in results if r["source"] == "mbpp"]
    
    he_passed = sum(1 for r in humaneval_results if r["passed"])
    mb_passed = sum(1 for r in mbpp_results if r["passed"])
    
    metrics = {
        "model": args.model,
        "test_file": test_file,
        "total_examples": total,
        "examples_with_tests": total_with_tests,
        "pass_at_k": passed / total_with_tests if total_with_tests > 0 else 0,
        "humaneval_pass_at_k": he_passed / len(humaneval_results) if humaneval_results else 0,
        "humaneval_count": f"{he_passed}/{len(humaneval_results)}",
        "mbpp_pass_at_k": mb_passed / len(mbpp_results) if mbpp_results else 0,
        "mbpp_count": f"{mb_passed}/{len(mbpp_results)}",
    }
    
    return results, metrics


def main():
    args = parse_args()
    
    model, tokenizer = load_model(args)
    results, metrics = evaluate(model, tokenizer, args.test_file, args)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    metrics_path = output_path.with_suffix('.metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print
    print("\n" + "="*70)
    print("RESULTS (Pass@k)")
    print("="*70)
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Examples with tests: {metrics['examples_with_tests']}")
    print(f"\nPass@{args.n_samples}: {metrics['pass_at_k']:.4f} ({metrics['pass_at_k']*100:.2f}%)")
    print(f"  HumanEval: {metrics['humaneval_pass_at_k']:.4f} ({metrics['humaneval_count']})")
    print(f"  MBPP:      {metrics['mbpp_pass_at_k']:.4f} ({metrics['mbpp_count']})")
    print("="*70)
    print(f"\nResults: {output_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()