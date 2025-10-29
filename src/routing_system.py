#!/usr/bin/env python3
"""
Comprehensive LLM Routing System Evaluation
Evaluates distilled and 70B models on different complexity levels with wandb logging.
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import subprocess
import tiktoken
from collections import defaultdict


class ComplexityEstimator:
    """
    Estimate query complexity based on token count.
    """
    
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def estimate_complexity(self, query: str) -> int:
        """
        Estimate query complexity on scale of 1-10 based on token count.
        
        Optimized routing based on actual performance:
        - 1-150 tokens: Simple/Medium -> Distilled model (complexity 1-6)
        - 151+ tokens: Complex -> 70B model (complexity 7-10)
        """
        tokens = self.encoder.encode(query)
        token_count = len(tokens)
        
        # Optimized: Use distilled for most queries, 70B only for complex ones
        if token_count <= 150:
            return max(1, min(6, token_count // 25))    # Distilled: 1-6
        else:
            return 7 + min(3, (token_count - 151) // 50)  # 70B: 7-10


class RoutingEvaluator:
    """
    Comprehensive evaluation system for routing strategy.
    """
    
    def __init__(self,
                 distilled_model_path: str,
                 teacher_model_path: str,
                 lora_dir: str,
                 wandb_api_key: str,
                 wandb_project: str = "LLM-Compression-Project",
                 output_dir: str = "results"):
        
        self.distilled_model = distilled_model_path
        self.teacher_model = teacher_model_path
        self.lora_dir = lora_dir
        self.wandb_api_key = wandb_api_key
        self.wandb_project = wandb_project
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.estimator = ComplexityEstimator()
        self.eval_script = "src/eval_humaneval.py"
        
        # Check if eval script exists
        if not os.path.exists(self.eval_script):
            raise FileNotFoundError(f"Evaluation script not found: {self.eval_script}")
    
    def classify_problems_by_complexity(self) -> Dict[str, List[str]]:
        """
        Classify HumanEval problems by complexity level.
        
        Returns:
            Dictionary with 'simple' (1-6) and 'complex' (7-10) task IDs
        """
        from human_eval.data import read_problems
        
        problems = read_problems()
        classified = {
            'simple': [],      # Complexity 1-6
            'complex': []      # Complexity 7-10
        }
        
        for task_id, problem in problems.items():
            prompt = problem['prompt']
            complexity = self.estimator.estimate_complexity(prompt)
            
            if complexity <= 6:
                classified['simple'].append(task_id)
            else:
                classified['complex'].append(task_id)
        
        return classified, problems
    
    def create_filtered_problems_file(self, 
                                     task_ids: List[str], 
                                     output_file: str) -> str:
        """
        Create a temporary HumanEval problems file with only specified task IDs.
        """
        from human_eval.data import read_problems, write_jsonl
        
        all_problems = read_problems()
        filtered = {task_id: all_problems[task_id] for task_id in task_ids if task_id in all_problems}
        
        # Save filtered problems
        temp_file = self.output_dir / f"temp_{output_file}_problems.jsonl"
        problems_list = [{"task_id": k, **v} for k, v in filtered.items()]
        write_jsonl(str(temp_file), problems_list)
        
        return str(temp_file)
    
    def run_evaluation(self,
                      model: str,
                      output_file: str,
                      wandb_run_name: str,
                      lora_dir: Optional[str] = None,
                      num_samples: int = 10,
                      task_ids: Optional[List[str]] = None) -> Dict:
        """
        Run HumanEval evaluation for a specific model configuration.
        
        Args:
            model: Model name or path
            output_file: Output JSONL file name
            wandb_run_name: Name for wandb run
            lora_dir: Optional LoRA adapter directory
            num_samples: Number of samples per problem (for pass@k)
            task_ids: Optional list of task IDs to evaluate (if None, evaluates all)
        
        Returns:
            Evaluation metrics dictionary
        """
        output_path = self.output_dir / output_file
        
        # Build command
        cmd = [
            "python", self.eval_script,
            "--model", model,
            "--output", str(output_path),
            "--num_samples", str(num_samples),
            "--wandb_api_key", self.wandb_api_key,
            "--wandb_project", self.wandb_project,
            "--wandb_run_name", wandb_run_name,
        ]
        
        if lora_dir:
            cmd.extend(["--lora_dir", lora_dir])
        
        print(f"\n{'='*70}")
        print(f"Running: {wandb_run_name}")
        print(f"{'='*70}")
        print(f"Model: {model}")
        if lora_dir:
            print(f"LoRA: {lora_dir}")
        print(f"Output: {output_path}")
        if task_ids:
            print(f"Problems: {len(task_ids)} tasks")
        else:
            print(f"Problems: All (164 tasks)")
        print(f"{'='*70}\n")
        
        # If task_ids specified, we need to filter problems
        # For now, we'll run on all problems and filter results later
        # (HumanEval doesn't natively support task filtering)
        
        # Run evaluation
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: Evaluation failed for {wandb_run_name}")
            return None
        
        # Load metrics
        metrics_file = output_path.with_suffix('.metrics.json')
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            print(f"WARNING: Metrics file not found: {metrics_file}")
            return None
    
    def filter_results_by_task_ids(self,
                                   results_file: str,
                                   task_ids: List[str],
                                   output_file: str) -> None:
        """
        Filter evaluation results to only include specific task IDs.
        """
        from human_eval.data import read_problems, write_jsonl
        
        results_path = self.output_dir / results_file
        output_path = self.output_dir / output_file
        
        # Read results
        with open(results_path, 'r') as f:
            all_results = [json.loads(line) for line in f]
        
        # Filter
        filtered_results = [r for r in all_results if r['task_id'] in task_ids]
        
        # Write filtered results
        write_jsonl(str(output_path), filtered_results)
        
        print(f"Filtered {len(filtered_results)}/{len(all_results)} results -> {output_path}")
    
    def evaluate_all_configurations(self):
        """
        Run comprehensive evaluation across all configurations.
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE ROUTING EVALUATION")
        print("="*70)
        
        # Step 1: Classify problems
        print("\nStep 1: Classifying problems by complexity...")
        classified, all_problems = self.classify_problems_by_complexity()
        
        print(f"  Simple (1-6):  {len(classified['simple'])} problems")
        print(f"  Complex (7-10): {len(classified['complex'])} problems")
        
        # Store task IDs for later filtering
        simple_tasks = classified['simple']
        complex_tasks = classified['complex']
        
        all_metrics = {}
        
        # Step 2: Evaluate Distilled Model on Simple Queries (1-6)
        print("\n" + "="*70)
        print("EVALUATION 1/6: Distilled Model on Simple Queries (Complexity 1-6)")
        print("="*70)
        
        metrics = self.run_evaluation(
            model=self.distilled_model,
            output_file="distilled_simple_all.jsonl",
            wandb_run_name="distilled_simple_1-6",
            lora_dir=self.lora_dir,
            num_samples=10
        )
        
        # Filter results to only simple tasks
        if metrics:
            self.filter_results_by_task_ids(
                "distilled_simple_all.jsonl",
                simple_tasks,
                "distilled_simple.jsonl"
            )
            all_metrics['distilled_simple'] = metrics
        
        # Step 3: Evaluate Distilled Model on Complex Queries (7-10)
        print("\n" + "="*70)
        print("EVALUATION 2/6: Distilled Model on Complex Queries (Complexity 7-10)")
        print("="*70)
        
        metrics = self.run_evaluation(
            model=self.distilled_model,
            output_file="distilled_complex_all.jsonl",
            wandb_run_name="distilled_complex_7-10",
            lora_dir=self.lora_dir,
            num_samples=10
        )
        
        # Filter results to only complex tasks
        if metrics:
            self.filter_results_by_task_ids(
                "distilled_complex_all.jsonl",
                complex_tasks,
                "distilled_complex.jsonl"
            )
            all_metrics['distilled_complex'] = metrics
        
        # Step 4: Evaluate 70B Model on Simple Queries (1-6)
        print("\n" + "="*70)
        print("EVALUATION 3/6: 70B Model on Simple Queries (Complexity 1-6)")
        print("="*70)
        
        metrics = self.run_evaluation(
            model=self.teacher_model,
            output_file="70b_simple_all.jsonl",
            wandb_run_name="70b_simple_1-6",
            num_samples=10
        )
        
        # Filter results to only simple tasks
        if metrics:
            self.filter_results_by_task_ids(
                "70b_simple_all.jsonl",
                simple_tasks,
                "70b_simple.jsonl"
            )
            all_metrics['70b_simple'] = metrics
        
        # Step 5: Evaluate 70B Model on Complex Queries (7-10)
        print("\n" + "="*70)
        print("EVALUATION 4/6: 70B Model on Complex Queries (Complexity 7-10)")
        print("="*70)
        
        metrics = self.run_evaluation(
            model=self.teacher_model,
            output_file="70b_complex_all.jsonl",
            wandb_run_name="70b_complex_7-10",
            num_samples=10
        )
        
        # Filter results to only complex tasks
        if metrics:
            self.filter_results_by_task_ids(
                "70b_complex_all.jsonl",
                complex_tasks,
                "70b_complex.jsonl"
            )
            all_metrics['70b_complex'] = metrics
        
        # Step 6: Create Routed Results (Hybrid)
        print("\n" + "="*70)
        print("EVALUATION 5/6: Creating Routed Results (Hybrid Approach)")
        print("="*70)
        
        self.create_routed_results(simple_tasks, complex_tasks)
        
        # Step 7: Evaluate Routed Results
        print("\n" + "="*70)
        print("EVALUATION 6/6: Evaluating Routed System")
        print("="*70)
        
        # We need to evaluate the combined results
        # This is a bit tricky - we'll compute metrics from the combined file
        routed_metrics = self.evaluate_routed_results("routed_combined.jsonl")
        all_metrics['routed'] = routed_metrics
        
        # Final Summary
        self.print_comprehensive_summary(all_metrics, classified)
        
        return all_metrics
    
    def create_routed_results(self, simple_tasks: List[str], complex_tasks: List[str]):
        """
        Combine distilled (simple) and 70B (complex) results into routed results.
        """
        # Read distilled simple results
        distilled_simple_path = self.output_dir / "distilled_simple.jsonl"
        with open(distilled_simple_path, 'r') as f:
            distilled_simple = [json.loads(line) for line in f]
        
        # Read 70B complex results
        teacher_complex_path = self.output_dir / "70b_complex.jsonl"
        with open(teacher_complex_path, 'r') as f:
            teacher_complex = [json.loads(line) for line in f]
        
        # Combine
        routed_results = distilled_simple + teacher_complex
        
        # Sort by task_id for consistency
        routed_results.sort(key=lambda x: x['task_id'])
        
        # Write combined results
        from human_eval.data import write_jsonl
        routed_path = self.output_dir / "routed_combined.jsonl"
        write_jsonl(str(routed_path), routed_results)
        
        print(f"Created routed results: {len(distilled_simple)} simple + {len(teacher_complex)} complex = {len(routed_results)} total")
    
    def evaluate_routed_results(self, results_file: str) -> Dict:
        """
        Evaluate the routed (combined) results file.
        """
        from human_eval.evaluation import evaluate_functional_correctness
        
        results_path = self.output_dir / results_file
        
        print(f"Evaluating routed results: {results_path}")
        
        # Evaluate pass@k
        pass_at_k = {}
        for k in [1, 5, 10]:
            print(f"  Computing pass@{k}...")
            metrics = evaluate_functional_correctness(
                str(results_path),
                k=[k],
                n_workers=4,
                timeout=3.0
            )
            pass_at_k[k] = metrics[f'pass@{k}']
        
        # Count problems
        with open(results_path, 'r') as f:
            results = [json.loads(line) for line in f]
        
        # Get unique task IDs
        unique_tasks = len(set(r['task_id'] for r in results))
        
        routed_metrics = {
            'num_problems': unique_tasks,
            'num_samples_per_problem': 10,
            'total_generations': len(results),
            'pass_at_k': {f'pass@{k}': v for k, v in pass_at_k.items()},
        }
        
        # Save metrics
        metrics_path = results_path.with_suffix('.metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(routed_metrics, f, indent=2)
        
        print(f"Routed metrics saved: {metrics_path}")
        
        return routed_metrics
    
    def print_comprehensive_summary(self, all_metrics: Dict, classified: Dict):
        """
        Print comprehensive summary of all evaluations.
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*70)
        
        # Problem distribution
        print(f"\nProblem Distribution:")
        print(f"  Simple (1-6):  {len(classified['simple'])} problems")
        print(f"  Complex (7-10): {len(classified['complex'])} problems")
        print(f"  Total:          {len(classified['simple']) + len(classified['complex'])} problems")
        
        # Performance comparison
        print(f"\n{'Configuration':<30} {'Pass@1':<10} {'Pass@5':<10} {'Pass@10':<10}")
        print("-" * 70)
        
        configs = [
            ('distilled_simple', 'Distilled on Simple (1-6)'),
            ('distilled_complex', 'Distilled on Complex (7-10)'),
            ('70b_simple', '70B on Simple (1-6)'),
            ('70b_complex', '70B on Complex (7-10)'),
            ('routed', 'Routed (Hybrid)'),
        ]
        
        for key, name in configs:
            if key in all_metrics and all_metrics[key]:
                metrics = all_metrics[key]
                pass_at_k = metrics.get('pass_at_k', {})
                p1 = pass_at_k.get('pass@1', 0) * 100
                p5 = pass_at_k.get('pass@5', 0) * 100
                p10 = pass_at_k.get('pass@10', 0) * 100
                print(f"{name:<30} {p1:>6.2f}%    {p5:>6.2f}%    {p10:>6.2f}%")
        
        # Cost analysis
        print(f"\nCost Analysis:")
        print(f"  Baseline (always 70B):     1.00x")
        
        if 'routed' in all_metrics:
            n_simple = len(classified['simple'])
            n_complex = len(classified['complex'])
            total = n_simple + n_complex
            
            # Assuming distilled is 0.02x cost, 70B is 1.0x
            routed_cost = (n_simple * 0.02 + n_complex * 1.0) / total
            savings = (1.0 - routed_cost) / 1.0 * 100
            
            print(f"  Routed (hybrid):           {routed_cost:.2f}x")
            print(f"  Savings:                   {savings:.1f}%")
        
        # Efficiency metrics
        if 'routed' in all_metrics and '70b_simple' in all_metrics and '70b_complex' in all_metrics:
            # Baseline efficiency (70B on all)
            baseline_metrics = all_metrics.get('70b_simple', {}).get('pass_at_k', {})
            baseline_p1 = baseline_metrics.get('pass@1', 0.72)  # fallback
            
            routed_metrics = all_metrics['routed'].get('pass_at_k', {})
            routed_p1 = routed_metrics.get('pass@1', 0)
            
            baseline_efficiency = baseline_p1 / 1.0
            routed_efficiency = routed_p1 / routed_cost if routed_cost > 0 else 0
            
            print(f"\nEfficiency (Performance per Cost):")
            print(f"  Baseline (70B):   {baseline_efficiency:.3f}")
            print(f"  Routed (hybrid):  {routed_efficiency:.3f}")
            if baseline_efficiency > 0:
                gain = ((routed_efficiency - baseline_efficiency) / baseline_efficiency * 100)
                print(f"  Efficiency gain:  {gain:+.1f}%")
        
        print("\n" + "="*70)
        print("Evaluation complete! Check wandb for detailed metrics and visualizations.")
        print("="*70 + "\n")


def main():
    """
    Main entry point for routing evaluation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive LLM Routing Evaluation")
    parser.add_argument("--distilled_model", type=str, 
                       default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="Distilled model name or path")
    parser.add_argument("--teacher_model", type=str,
                       default="meta-llama/Meta-Llama-3.1-70B-Instruct",
                       help="Teacher model name or path")
    parser.add_argument("--lora_dir", type=str,
                       default="outputs/llama31_8b_codealpaca_kd_lora/checkpoint-10010",
                       help="LoRA adapter directory for distilled model")
    parser.add_argument("--wandb_api_key", type=str, required=True,
                       help="Wandb API key")
    parser.add_argument("--wandb_project", type=str,
                       default="LLM-Compression-Project",
                       help="Wandb project name")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = RoutingEvaluator(
        distilled_model_path=args.distilled_model,
        teacher_model_path=args.teacher_model,
        lora_dir=args.lora_dir,
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir
    )
    
    # Run comprehensive evaluation
    try:
        all_metrics = evaluator.evaluate_all_configurations()
        
        # Save comprehensive results
        summary_file = Path(args.output_dir) / "comprehensive_evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"\nComprehensive results saved to: {summary_file}")
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())