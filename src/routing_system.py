#!/usr/bin/env python3
"""
Dynamic LLM Routing System for AFSC/EN
Routes queries to appropriate model based on token complexity and actual performance data.
"""

import json
from typing import Dict, Tuple, Optional
from pathlib import Path
import tiktoken


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
        - 1-150 tokens: Simple/Medium -> Distilled model (56.7% performance)
        - 151+ tokens: Complex -> 70B model (72.0% performance)
        """
        tokens = self.encoder.encode(query)
        token_count = len(tokens)
        
        # Optimized: Use distilled for most queries, 70B only for complex ones
        if token_count <= 150:
            return max(1, min(6, token_count // 25))    # Distilled: 1-6
        else:
            return 7 + min(3, (token_count - 151) // 50)  # 70B: 7-10


class ModelRouter:
    """
    Routes queries to appropriate model based on complexity and actual performance.
    """
    
    def __init__(self, 
                 complexity_thresholds: Dict[str, Tuple[int, int]] = None,
                 model_costs: Dict[str, float] = None):
        
        self.estimator = ComplexityEstimator()
        
        # Optimized thresholds based on actual performance data
        self.thresholds = complexity_thresholds or {
            'distilled': (1, 6),   # Simple/Medium queries: 1-150 tokens
            '70B': (7, 10),        # Complex queries: 151+ tokens
        }
        
        # Model names
        self.model_names = {
            'distilled': 'distilled-model',
            '70B': 'Meta-Llama-3.1-70B-Instruct'
        }
        
        # Relative costs
        self.costs = model_costs or {
            'distilled': 0.02,
            '70B': 1.0,
        }
        
        # Actual performance data from evaluation
        self.performance = {
            'distilled': 0.5671,  # 56.71% pass@1
            '70B': 0.7195,        # 71.95% pass@1
        }
    
    def route(self, query: str) -> Dict:
        """
        Determine which model to use for a query.
        """
        complexity = self.estimator.estimate_complexity(query)
        token_count = len(self.estimator.encoder.encode(query))
        
        for model_type, (min_score, max_score) in self.thresholds.items():
            if min_score <= complexity <= max_score:
                return {
                    'model': self.model_names[model_type],
                    'model_type': model_type,
                    'complexity': complexity,
                    'token_count': token_count,
                    'cost': self.costs[model_type],
                    'expected_performance': self.performance[model_type],
                    'reasoning': self._get_reasoning(complexity, token_count, model_type)
                }
        
        # Default to 70B for safety
        return {
            'model': self.model_names['70B'],
            'model_type': '70B',
            'complexity': complexity,
            'token_count': token_count,
            'cost': self.costs['70B'],
            'expected_performance': self.performance['70B'],
            'reasoning': f'Complex query ({token_count} tokens) -> Use 70B for best quality ({self.performance["70B"]*100:.1f}% pass@1)'
        }
    
    def _get_reasoning(self, complexity: int, token_count: int, model_type: str) -> str:
        """Explain routing decision based on token count."""
        performance_pct = self.performance[model_type] * 100
        if complexity <= 6:
            return f"Simple/Medium query ({token_count} tokens) -> Use distilled model for 98% cost savings ({performance_pct:.1f}% pass@1)"
        else:
            return f"Complex query ({token_count} tokens) -> Use 70B for best quality ({performance_pct:.1f}% pass@1)"


def evaluate_routing_strategy():
    """
    Evaluate routing strategy on HumanEval problems using actual token counts.
    """
    from human_eval.data import read_problems
    
    router = ModelRouter()
    problems = read_problems()
    
    # Analyze routing decisions based on actual token counts
    routing_decisions = {}
    
    for task_id, problem in problems.items():
        prompt = problem['prompt']
        decision = router.route(prompt)
        routing_decisions[task_id] = decision
    
    # Calculate statistics
    model_counts = {'distilled': 0, '70B': 0}
    token_stats = {'distilled': [], '70B': []}
    
    for decision in routing_decisions.values():
        model_type = decision['model_type']
        model_counts[model_type] += 1
        token_stats[model_type].append(decision['token_count'])
    
    total = len(routing_decisions)
    
    print("=" * 70)
    print("OPTIMIZED ROUTING STRATEGY EVALUATION")
    print("=" * 70)
    
    print(f"\nQuery Distribution ({total} problems):")
    for model_type in ['distilled', '70B']:
        count = model_counts[model_type]
        avg_tokens = sum(token_stats[model_type]) / len(token_stats[model_type]) if token_stats[model_type] else 0
        performance = router.performance[model_type] * 100
        print(f"  {model_type:10}: {count:3d} ({count/total*100:5.1f}%) - Avg: {avg_tokens:.0f} tokens, {performance:.1f}% pass@1")
    
    # Calculate weighted costs
    baseline_cost = 1.0 * total
    routed_cost = sum(router.costs[d['model_type']] for d in routing_decisions.values())
    savings = (baseline_cost - routed_cost) / baseline_cost * 100
    
    print(f"\nCost Analysis:")
    print(f"  Baseline (always 70B): {baseline_cost:.2f} units")
    print(f"  With routing:          {routed_cost:.2f} units")
    print(f"  Savings:               {savings:.1f}%")
    
    # Expected performance
    baseline_perf = router.performance['70B']
    expected_perf = sum(router.performance[d['model_type']] for d in routing_decisions.values()) / total
    perf_loss = (baseline_perf - expected_perf) * 100
    
    print(f"\nPerformance Analysis:")
    print(f"  Baseline (always 70B): {baseline_perf*100:.1f}% pass@1")
    print(f"  With routing:          {expected_perf*100:.1f}% pass@1")
    print(f"  Performance loss:      {perf_loss:.1f}pp")
    
    # Efficiency metrics
    baseline_efficiency = baseline_perf / 1.0
    routed_efficiency = expected_perf / (routed_cost / total)
    
    print(f"\nEfficiency Analysis:")
    print(f"  Baseline efficiency: {baseline_efficiency:.3f} performance/cost")
    print(f"  Routed efficiency:   {routed_efficiency:.3f} performance/cost")
    print(f"  Efficiency gain:     {((routed_efficiency - baseline_efficiency) / baseline_efficiency * 100):+.1f}%")
    
    # Show token distribution
    print(f"\nToken Count Statistics:")
    all_tokens = [d['token_count'] for d in routing_decisions.values()]
    print(f"  Min: {min(all_tokens)} tokens")
    print(f"  Max: {max(all_tokens)} tokens") 
    print(f"  Avg: {sum(all_tokens)/len(all_tokens):.1f} tokens")
    
    # Show some examples
    print(f"\nExample Routing Decisions:")
    print("-" * 70)
    for i, (task_id, decision) in enumerate(list(routing_decisions.items())[:5]):
        problem = problems[task_id]
        prompt_preview = problem['prompt'].split('\n')[0][:50]
        print(f"\n{i+1}. {task_id}")
        print(f"   Prompt: {prompt_preview}...")
        print(f"   Tokens: {decision['token_count']}")
        print(f"   -> {decision['model']} (complexity={decision['complexity']})")
        print(f"   -> {decision['reasoning']}")
    
    print("=" * 70)
    
    return routing_decisions


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AFSC/EN Optimized LLM Routing System")
    print("=" * 70)
    
    # Demo with test queries
    router = ModelRouter()
    
    test_queries = [
        "Write a function to check if a number is even",
        "Implement a binary search tree with insert, delete, and search operations",
        "Design a microservices architecture for a real-time analytics platform with fault tolerance and load balancing across multiple availability zones",
        "Print 'Hello, World!' to the console",
        "Create a RESTful API with JWT authentication and role-based access control using FastAPI with SQLAlchemy ORM and PostgreSQL backend",
    ]
    
    print("\n1. ROUTING EXAMPLES")
    print("-" * 70)
    for i, query in enumerate(test_queries, 1):
        decision = router.route(query)
        print(f"\n{i}. Query: {query[:60]}...")
        print(f"   -> Tokens: {decision['token_count']}")
        print(f"   -> Model: {decision['model']}")
        print(f"   -> Complexity: {decision['complexity']}/10")
        print(f"   -> Cost: {decision['cost']:.3f}x (vs 70B)")
        print(f"   -> Expected performance: {decision['expected_performance']*100:.1f}% pass@1")
        print(f"   -> {decision['reasoning']}")
    
    # Evaluate on HumanEval
    print("\n\n2. HUMANEVAL EVALUATION")
    print("-" * 70)
    try:
        evaluate_routing_strategy()
    except Exception as e:
        print(f"Note: Run after HumanEval results are available")
        print(f"Error: {e}")
    
    print("\nOptimized routing system evaluation complete.")