#!/usr/bin/env python3
"""
eval_combined_pass_at_1.py - Execution-based Pass@1 evaluation on combined test set

This evaluates code by actually executing it against test cases.
"""

import json
import torch
import argparse
import sys
import os
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora_dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--test_file", default="data/combined_test.jsonl")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.1)
    return ap.parse_args()


def load_model(args):
    """Load model and tokenizer."""
    print(f"Loading: {args.model}")
    
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
        print(f"Loading LoRA: {args.lora_dir}")
        model = PeftModel.from_pretrained(model, args.lora_dir)
        try:
            model = model.merge_and_unload()
        except:
            pass
    
    model.eval()
    return model, tokenizer


def extract_code(text: str) -> str:
    """Extract code from model output."""
    import re
    
    # Remove markdown fences
    text = re.sub(r'```python\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Remove common preamble
    lines = text.split('\n')
    code_lines = []
    started = False
    
    for line in lines:
        stripped = line.strip().lower()
        
        # Skip intro text
        if not started:
            if any(p in stripped for p in ['here is', 'here\'s', 'certainly']):
                continue
            if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                started = True
        
        if started:
            code_lines.append(line)
    
    return '\n'.join(code_lines).strip()


def test_code(code: str, test_cases: list, timeout: int = 5) -> tuple:
    """
    Execute code with test cases and return (passed, total, error).
    
    Returns:
        (num_passed, num_total, error_message)
    """
    if not code or not test_cases:
        return 0, 0, "No code or tests"
    
    # Create test script
    test_script = code + "\n\n"
    
    # Add test cases
    passed = 0
    total = 0
    
    for test in test_cases:
        test = test.strip()
        if not test:
            continue
        
        # Ensure it's an assert statement
        if not test.startswith('assert'):
            test = f"assert {test}"
        
        total += 1
        
        # Try to execute this specific test
        full_script = test_script + test + "\n"
        
        try:
            # Execute in isolated environment
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_script)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    timeout=timeout,
                    text=True
                )
                
                if result.returncode == 0:
                    passed += 1
            finally:
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue
    
    return passed, total, None


def evaluate(model, tokenizer, test_file, args):
    """Run evaluation with execution testing."""
    # Load test data
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Test examples: {len(test_data)}")
    
    results = []
    total_passed = 0
    total_problems = 0
    
    for example in tqdm(test_data, desc="Evaluating"):
        # Extract components
        messages = example["messages"]
        source = example.get("source", "unknown")
        task_id = example.get("task_id", "unknown")
        
        # Get prompt (all messages except last)
        prompt_messages = messages[:-1]
        reference_code = messages[-1]["content"]
        
        # Extract test cases if available
        test_cases = []
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                # Look for test cases in user message
                import re
                test_matches = re.findall(r'assert\s+.*', content)
                test_cases.extend(test_matches)
        
        # Generate prediction
        formatted = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
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
        
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = full[len(formatted):].strip() if full.startswith(formatted) else full
        
        # Extract clean code
        prediction_code = extract_code(prediction)
        
        # Test the code
        if test_cases:
            passed, total_tests, error = test_code(prediction_code, test_cases)
            is_correct = (passed == total_tests and total_tests > 0)
        else:
            # No tests available - can't determine correctness
            passed, total_tests, error = 0, 0, "No tests"
            is_correct = False
        
        if is_correct:
            total_passed += 1
        total_problems += 1
        
        results.append({
            "task_id": task_id,
            "source": source,
            "prediction": prediction_code,
            "reference": reference_code,
            "tests_passed": passed,
            "tests_total": total_tests,
            "correct": is_correct
        })
    
    # Calculate metrics
    overall_pass_rate = total_passed / total_problems if total_problems > 0 else 0
    
    # Per-source metrics
    humaneval_correct = sum(1 for r in results if r["source"] == "humaneval" and r["correct"])
    humaneval_total = sum(1 for r in results if r["source"] == "humaneval")
    
    mbpp_correct = sum(1 for r in results if r["source"] == "mbpp" and r["correct"])
    mbpp_total = sum(1 for r in results if r["source"] == "mbpp")
    
    metrics = {
        "model": args.model,
        "test_file": test_file,
        "total_problems": total_problems,
        "overall_pass_at_1": overall_pass_rate,
        "humaneval_pass_at_1": humaneval_correct / humaneval_total if humaneval_total > 0 else 0,
        "humaneval_count": f"{humaneval_correct}/{humaneval_total}",
        "mbpp_pass_at_1": mbpp_correct / mbpp_total if mbpp_total > 0 else 0,
        "mbpp_count": f"{mbpp_correct}/{mbpp_total}"
    }
    
    return results, metrics


def main():
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model(args)
    
    # Evaluate
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
    print("RESULTS (Pass@1)")
    print("="*70)
    print(f"Overall:    {metrics['overall_pass_at_1']:.4f} ({metrics['overall_pass_at_1']*100:.2f}%)")
    print(f"HumanEval:  {metrics['humaneval_pass_at_1']:.4f} ({metrics['humaneval_count']})")
    print(f"MBPP:       {metrics['mbpp_pass_at_1']:.4f} ({metrics['mbpp_count']})")
    print("="*70)
    print(f"\nResults: {output_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()