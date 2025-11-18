#!/usr/bin/env python3
"""
eval_humaneval.py - HumanEval benchmark evaluation for LLMs (ENHANCED VERSION)

Features:
- Improved code extraction for better HumanEval pass rates
- Pass@k metrics: pass@1, pass@5, pass@10
- GPU & power consumption tracking with wandb
- Comprehensive metrics logging
- Metrics saved to dedicated metrics/ folder
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import threading

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

# Wandb for tracking
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("WARNING: wandb not found. Install with: pip install wandb")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM on HumanEval benchmark")
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path")
    parser.add_argument("--lora_dir", type=str, default=None,
                       help="Path to LoRA adapters")
    
    # Quantization
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit quantization")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                       help="Use float16 precision")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.01,
                       help="Sampling temperature (use 0.01 for near-greedy)")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples per problem for pass@k")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of problems (for testing)")
    
    # Wandb configuration
    parser.add_argument("--wandb_project", type=str, default="humaneval-llm-eval",
                       help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name (default: auto-generated)")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                       help="Wandb API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    
    return parser.parse_args()


class GPUMonitor:
    """Monitor GPU utilization and power consumption."""
    
    def __init__(self, log_interval: float = 1.0):
        """
        Args:
            log_interval: Seconds between measurements
        """
        self.log_interval = log_interval
        self.monitoring = False
        self.thread = None
        self.metrics = defaultdict(list)
        
        # Check if pynvml is available
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.has_nvml = True
            self.device_count = pynvml.nvmlDeviceGetCount()
        except:
            self.has_nvml = False
            print("WARNING: pynvml not available. Install with: pip install nvidia-ml-py3")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            timestamp = time.time()
            
            if self.has_nvml:
                for i in range(self.device_count):
                    try:
                        handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
                        
                        # GPU utilization
                        util = self.nvml.nvmlDeviceGetUtilizationRates(handle)
                        self.metrics[f'gpu_{i}_utilization'].append(util.gpu)
                        self.metrics[f'gpu_{i}_memory_utilization'].append(util.memory)
                        
                        # Memory usage
                        mem_info = self.nvml.nvmlDeviceGetMemoryInfo(handle)
                        mem_used_gb = mem_info.used / (1024**3)
                        mem_total_gb = mem_info.total / (1024**3)
                        self.metrics[f'gpu_{i}_memory_used_gb'].append(mem_used_gb)
                        self.metrics[f'gpu_{i}_memory_total_gb'].append(mem_total_gb)
                        
                        # Power consumption
                        try:
                            power_mw = self.nvml.nvmlDeviceGetPowerUsage(handle)
                            power_w = power_mw / 1000.0
                            self.metrics[f'gpu_{i}_power_watts'].append(power_w)
                        except:
                            pass
                        
                        # Temperature
                        try:
                            temp = self.nvml.nvmlDeviceGetTemperature(handle, self.nvml.NVML_TEMPERATURE_GPU)
                            self.metrics[f'gpu_{i}_temperature_c'].append(temp)
                        except:
                            pass
                        
                    except Exception as e:
                        print(f"Warning: Failed to get metrics for GPU {i}: {e}")
            
            # Also log timestamp
            self.metrics['timestamp'].append(timestamp)
            
            time.sleep(self.log_interval)
    
    def start(self):
        """Start monitoring in background thread."""
        if not self.has_nvml:
            return
        
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> Dict:
        """Stop monitoring and return aggregated metrics."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Aggregate metrics
        aggregated = {}
        for key, values in self.metrics.items():
            if values and key != 'timestamp':
                aggregated[f'{key}_mean'] = sum(values) / len(values)
                aggregated[f'{key}_max'] = max(values)
                aggregated[f'{key}_min'] = min(values)
        
        # Calculate total energy consumption
        for i in range(self.device_count if self.has_nvml else 0):
            power_key = f'gpu_{i}_power_watts'
            if power_key in self.metrics and self.metrics[power_key]:
                # Energy = average power * time
                avg_power = aggregated.get(f'{power_key}_mean', 0)
                duration = (self.metrics['timestamp'][-1] - self.metrics['timestamp'][0])
                energy_wh = (avg_power * duration) / 3600.0  # Watt-hours
                aggregated[f'gpu_{i}_energy_wh'] = energy_wh
        
        return aggregated
    
    def get_current_metrics(self) -> Dict:
        """Get latest measurements."""
        current = {}
        for key, values in self.metrics.items():
            if values and key != 'timestamp':
                current[key] = values[-1]
        return current


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
        print("Using 4-bit quantization")
    elif args.load_in_8bit:
        if not HAS_BNB:
            raise RuntimeError("8-bit requires bitsandbytes")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization")
    
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
            print("LoRA merged successfully")
        except:
            print("Could not merge LoRA (continuing with adapter)")
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print(f"{'='*70}\n")
    return model, tokenizer


def generate_completion(model, tokenizer, prompt: str, args, sample_idx: int = 0) -> str:
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
    
    # Adjust temperature for sampling
    # For pass@k, we want diversity when k>1
    temp = args.temperature if args.num_samples == 1 else max(0.8, args.temperature)
    do_sample = (temp > 0.0) and (args.num_samples > 1)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=temp,
            do_sample=do_sample,
            top_p=0.95 if do_sample else None,
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


def evaluate_humaneval(model, tokenizer, args, gpu_monitor: Optional[GPUMonitor] = None):
    """Run HumanEval evaluation with pass@k metrics."""
    print(f"\n{'='*70}")
    print("HumanEval Evaluation")
    print(f"{'='*70}")
    
    # Load problems
    problems = read_problems()
    
    if args.limit:
        problems = dict(list(problems.items())[:args.limit])
    
    print(f"Total problems: {len(problems)}")
    print(f"Samples per problem: {args.num_samples}")
    print(f"Total generations: {len(problems) * args.num_samples}")
    print(f"{'='*70}\n")
    
    # Start GPU monitoring
    if gpu_monitor:
        gpu_monitor.start()
    
    # Generate completions
    results = []
    start_time = time.time()
    
    # Disable tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    total_generations = len(problems) * args.num_samples
    
    with tqdm(total=total_generations, desc="Generating") as pbar:
        for task_id, problem in problems.items():
            prompt = problem["prompt"]
            
            # Generate multiple samples for pass@k
            for sample_idx in range(args.num_samples):
                try:
                    completion = generate_completion(model, tokenizer, prompt, args, sample_idx)
                    results.append({
                        "task_id": task_id,
                        "completion": completion,
                    })
                    
                    # Log to wandb periodically
                    if HAS_WANDB and wandb.run and gpu_monitor and len(results) % 10 == 0:
                        current_gpu = gpu_monitor.get_current_metrics()
                        wandb.log({
                            "generation_progress": len(results) / total_generations,
                            **current_gpu
                        })
                    
                except Exception as e:
                    print(f"\nError on {task_id} (sample {sample_idx}): {e}")
                    results.append({
                        "task_id": task_id,
                        "completion": "",
                    })
                
                pbar.update(1)
    
    elapsed = time.time() - start_time
    print(f"\nGeneration: {elapsed:.1f}s ({elapsed/len(results):.2f}s/sample)")
    
    # Stop GPU monitoring
    gpu_metrics = {}
    if gpu_monitor:
        gpu_metrics = gpu_monitor.stop()
        print("\nGPU Metrics:")
        for key, value in sorted(gpu_metrics.items()):
            if 'mean' in key or 'energy' in key:
                print(f"  {key}: {value:.2f}")
    
    # Save completions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_path), results)
    print(f"\nSaved completions: {output_path}")
    
    # Evaluate pass@k
    print(f"\n{'='*70}")
    print("Running tests...")
    print(f"{'='*70}\n")
    
    try:
        # Evaluate with different k values
        pass_at_k = {}
        k_values = [1, 5, 10] if args.num_samples >= 10 else [1]
        
        for k in k_values:
            if k <= args.num_samples:
                print(f"Evaluating pass@{k}...")
                metrics = evaluate_functional_correctness(
                    str(output_path),
                    k=[k],
                    n_workers=4,
                    timeout=3.0
                )
                pass_at_k[k] = metrics[f'pass@{k}']
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        for k, score in pass_at_k.items():
            print(f"Pass@{k}: {score:.4f} ({score*100:.2f}%)")
        print(f"{'='*70}\n")
        
        # Save comprehensive metrics to dedicated metrics/ folder
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_filename = output_path.stem + '.metrics.json'
        metrics_path = metrics_dir / metrics_filename
        
        all_metrics = {
            'model': args.model,
            'lora_dir': args.lora_dir,
            'num_problems': len(problems),
            'num_samples_per_problem': args.num_samples,
            'total_generations': len(results),
            'generation_time_sec': elapsed,
            'avg_time_per_sample': elapsed / len(results),
            'pass_at_k': {f'pass@{k}': v for k, v in pass_at_k.items()},
            'gpu_metrics': gpu_metrics,
            'config': {
                'temperature': args.temperature,
                'max_new_tokens': args.max_new_tokens,
                'quantization': '4bit' if args.load_in_4bit else ('8bit' if args.load_in_8bit else 'none'),
                'dtype': 'bf16' if args.bf16 else ('fp16' if args.fp16 else 'fp32'),
            }
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Metrics saved: {metrics_path}")
        
        # Log to wandb
        if HAS_WANDB and wandb.run:
            wandb.log({
                **{f'pass@{k}': v for k, v in pass_at_k.items()},
                'generation_time_sec': elapsed,
                'avg_time_per_sample': elapsed / len(results),
                **{f'final_{k}': v for k, v in gpu_metrics.items()},
            })
            
            # Upload files as artifacts
            artifact = wandb.Artifact('humaneval_results', type='evaluation')
            artifact.add_file(str(output_path))
            artifact.add_file(str(metrics_path))
            wandb.log_artifact(artifact)
        
        return pass_at_k
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def init_wandb(args):
    """Initialize wandb logging."""
    if args.no_wandb or not HAS_WANDB:
        return None
    
    # Set API key if provided
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    
    # Auto-generate run name if not provided
    run_name = args.wandb_run_name
    if not run_name:
        model_name = Path(args.model).name
        lora_suffix = "_lora" if args.lora_dir else ""
        quant_suffix = "_4bit" if args.load_in_4bit else ("_8bit" if args.load_in_8bit else "")
        run_name = f"{model_name}{lora_suffix}{quant_suffix}_humaneval"
    
    # Initialize
    try:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model,
                "lora_dir": args.lora_dir,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "num_samples": args.num_samples,
                "quantization": "4bit" if args.load_in_4bit else ("8bit" if args.load_in_8bit else "none"),
                "dtype": "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32"),
            }
        )
        print(f"✓ Wandb initialized: {wandb.run.url}")
        return wandb.run
    except Exception as e:
        print(f"WARNING: Could not initialize wandb: {e}")
        return None


def main():
    args = parse_args()
    
    # Initialize wandb
    wandb_run = init_wandb(args)
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(log_interval=1.0)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Evaluate
    pass_at_k = evaluate_humaneval(model, tokenizer, args, gpu_monitor)
    
    if pass_at_k:
        print(f"\n✓ Complete!")
        for k, score in pass_at_k.items():
            print(f"  Pass@{k}: {score*100:.2f}%")
    else:
        print(f"\n⚠ Completed with errors")
    
    # Finish wandb
    if wandb_run:
        wandb.finish()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
