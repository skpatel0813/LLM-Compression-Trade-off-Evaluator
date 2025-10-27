#!/usr/bin/env python3
"""
Dynamic LLM Routing System for AFSC/EN
Routes queries to appropriate model based on token complexity.
"""

import re
import json
from typing import Dict, Tuple, Optional
from pathlib import Path
import tiktoken  # For token counting


class ComplexityEstimator:
    """
    Estimate query complexity based on token count and structural patterns.
    """
    
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")  # Same tokenizer used by Llama
    
    def estimate_complexity(self, query: str) -> int:
        """
        Estimate query complexity on scale of 1-10 based on token count.
        
        Token ranges based on HumanEval/MBPP analysis:
        - 1-50 tokens: Simple (1-3)
        - 51-150 tokens: Medium (4-7) 
        - 151+ tokens: Complex (8-10)
        """
        # Count tokens
        tokens = self.encoder.encode(query)
        token_count = len(tokens)
        
        # Map token count to complexity score
        if token_count <= 50:
            # Simple queries: basic functions, simple questions
            return max(1, min(3, token_count // 15))
        elif token_count <= 150:
            # Medium queries: moderate complexity
            return 4 + min(3, (token_count - 51) // 25)
        else:
            # Complex queries: long, detailed requirements
            return 8 + min(2, (token_count - 151) // 75)


class ModelRouter:
    """
    Routes queries to appropriate model based on complexity and cost.
    """
    
    def __init__(self, 
                 complexity_thresholds: Dict[str, Tuple[int, int]] = None,
                 model_costs: Dict[str, float] = None):
        """
        Args:
            complexity_thresholds: {model: (min_score, max_score)}
            model_costs: {model: relative_cost}
        """
        self.estimator = ComplexityEstimator()
        
        # Default thresholds based on token complexity
        self.thresholds = complexity_thresholds or {
            'distilled': (1, 3),   # Simple queries: 1-50 tokens
            '8B': (4, 7),         # Medium queries: 51-150 tokens  
            '70B': (8, 10),       # Complex queries: 151+ tokens
        }
        
        # Actual model names
        self.model_names = {
            'distilled': 'distilled-model',  # Replace with actual distilled model name
            '8B': 'Meta-Llama-3.1-8B-Instruct',
            '70B': 'Meta-Llama-3.1-70B-Instruct'
        }
        
        # Relative costs (based on actual inference costs)
        self.costs = model_costs or {
            'distilled': 0.02,  # ~2% of 70B cost
            '8B': 0.11,         # ~11% of 70B cost  
            '70B': 1.0,         # Baseline
        }
        
        # ACTUAL PERFORMANCE METRICS FROM YOUR EVALUATION
        self.performance = {
            'distilled': 0.5671,  # 56.71% pass@1 - YOUR ACTUAL DATA
            '8B': 0.3902,         # 39.02% pass@1 - YOUR ACTUAL DATA
            '70B': 0.7195,        # 71.95% pass@1 - YOUR ACTUAL DATA
        }
    
    def route(self, query: str) -> Dict:
        """
        Determine which model to use for a query.
        
        Returns:
            {
                'model': model name,
                'model_type': 'distilled' | '8B' | '70B',
                'complexity': int (1-10),
                'token_count': int,
                'cost': float,
                'expected_performance': float
            }
        """
        complexity = self.estimator.estimate_complexity(query)
        token_count = len(self.estimator.encoder.encode(query))
        
        # Select model based on complexity
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
        
        # Default to 70B if nothing matches
        return {
            'model': self.model_names['70B'],
            'model_type': '70B',
            'complexity': complexity,
            'token_count': token_count,
            'cost': self.costs['70B'],
            'expected_performance': self.performance['70B'],
            'reasoning': f'Complex query ({token_count} tokens) → Use 70B for best quality'
        }
    
    def _get_reasoning(self, complexity: int, token_count: int, model_type: str) -> str:
        """Explain routing decision based on token count."""
        performance_pct = self.performance[model_type] * 100
        if complexity <= 3:
            return f"Simple query ({token_count} tokens) → Use distilled model for 98% cost savings ({performance_pct:.1f}% pass@1)"
        elif complexity <= 7:
            return f"Medium query ({token_count} tokens) → Use 8B for 89% cost savings ({performance_pct:.1f}% pass@1)"
        else:
            return f"Complex query ({token_count} tokens) → Use 70B for best quality ({performance_pct:.1f}% pass@1)"


def load_humaneval_tokens() -> Dict[str, int]:
    """Load actual token counts from HumanEval problems."""
    from human_eval.data import read_problems
    import tiktoken
    
    encoder = tiktoken.get_encoding("cl100k_base")
    problems = read_problems()
    
    token_counts = {}
    for task_id, problem in problems.items():
        prompt = problem['prompt']
        tokens = encoder.encode(prompt)
        token_counts[task_id] = len(tokens)
    
    return token_counts


def evaluate_routing_strategy(humaneval_results_dir: str = "results"):
    """
    Evaluate routing strategy on HumanEval problems using actual token counts.
    """
    from human_eval.data import read_problems
    
    router = ModelRouter()
    problems = read_problems()
    
    # Analyze routing decisions based on actual token counts
    routing_decisions = {}
    token_counts = {}
    
    for task_id, problem in problems.items():
        prompt = problem['prompt']
        decision = router.route(prompt)
        routing_decisions[task_id] = decision
        token_counts[task_id] = decision['token_count']
    
    # Calculate statistics
    model_counts = {'distilled': 0, '8B': 0, '70B': 0}
    token_stats = {'distilled': [], '8B': [], '70B': []}
    performance_stats = {'distilled': [], '8B': [], '70B': []}
    
    for decision in routing_decisions.values():
        model_type = decision['model_type']
        model_counts[model_type] += 1
        token_stats[model_type].append(decision['token_count'])
        performance_stats[model_type].append(decision['expected_performance'])
    
    total = len(routing_decisions)
    
    print("="*70)
    print("ROUTING STRATEGY EVALUATION (Token-Based) - WITH REAL PERFORMANCE DATA")
    print("="*70)
    
    print(f"\nQuery Distribution ({total} problems):")
    for model_type in ['distilled', '8B', '70B']:
        count = model_counts[model_type]
        avg_tokens = sum(token_stats[model_type]) / len(token_stats[model_type]) if token_stats[model_type] else 0
        avg_performance = sum(performance_stats[model_type]) / len(performance_stats[model_type]) if performance_stats[model_type] else 0
        print(f"  {model_type:10}: {count:3d} ({count/total*100:5.1f}%) - Avg: {avg_tokens:.0f} tokens, {avg_performance*100:.1f}% pass@1")
    
    # Calculate weighted costs
    baseline_cost = 1.0 * total  # Always use 70B
    routed_cost = sum(router.costs[d['model_type']] for d in routing_decisions.values())
    savings = (baseline_cost - routed_cost) / baseline_cost * 100
    
    print(f"\nCost Analysis:")
    print(f"  Baseline (always 70B): {baseline_cost:.2f} units")
    print(f"  With routing:          {routed_cost:.2f} units")
    print(f"  Savings:               {savings:.1f}%")
    
    # Expected performance (weighted average)
    expected_perf = sum(
        router.performance[d['model_type']] for d in routing_decisions.values()
    ) / total
    
    baseline_perf = router.performance['70B']
    perf_loss = (baseline_perf - expected_perf) * 100
    
    print(f"\nPerformance Analysis:")
    print(f"  Baseline (always 70B): {baseline_perf*100:.1f}% pass@1")
    print(f"  With routing:          {expected_perf*100:.1f}% pass@1")
    print(f"  Performance loss:      {perf_loss:.1f}pp")
    
    # Calculate efficiency score (performance per cost unit)
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
    print("-"*70)
    for i, (task_id, decision) in enumerate(list(routing_decisions.items())[:5]):
        problem = problems[task_id]
        prompt_preview = problem['prompt'].split('\n')[0][:50]
        print(f"\n{i+1}. {task_id}")
        print(f"   Prompt: {promview}...")
        print(f"   Tokens: {decision['token_count']}")
        print(f"   → {decision['model']} (complexity={decision['complexity']})")
        print(f"   → {decision['reasoning']}")
    
    print("="*70)
    
    return routing_decisions


def simulate_production_deployment(queries_per_day: int = 10000,
                                   cost_per_70b_hour: float = 3.0):
    """
    Simulate annual costs for AFSC/EN deployment using actual token-based routing.
    """
    router = ModelRouter()
    
    # Model inference speeds (tokens/second) - based on actual benchmarks
    inference_speeds = {
        'distilled': 250,  # tokens/sec
        '8B': 150,         # tokens/sec  
        '70B': 30,         # tokens/sec
    }
    
    # Query token distribution (based on your HumanEval analysis)
    token_distribution = {
        'distilled': (1, 50, 0.049),    # 4.9% of queries, 1-50 tokens
        '8B': (51, 150, 0.665),         # 66.5% of queries, 51-150 tokens
        '70B': (151, 500, 0.287),       # 28.7% of queries, 151-500 tokens
    }
    
    days_per_year = 365
    
    # Baseline: Always use 70B
    avg_tokens = sum((max+min)/2 * prob for min, max, prob in token_distribution.values())
    baseline_seconds_per_query = avg_tokens / inference_speeds['70B']
    baseline_seconds_per_day = queries_per_day * baseline_seconds_per_query
    baseline_gpu_hours_per_year = (baseline_seconds_per_day * days_per_year) / 3600
    baseline_annual_cost = baseline_gpu_hours_per_year * cost_per_70b_hour
    
    # With routing: Weighted average based on distribution
    routed_seconds_per_query = 0
    for model_type, (min_tokens, max_tokens, probability) in token_distribution.items():
        avg_tokens_range = (min_tokens + max_tokens) / 2
        time_per_query = avg_tokens_range / inference_speeds[model_type]
        routed_seconds_per_query += time_per_query * probability
    
    routed_seconds_per_day = queries_per_day * routed_seconds_per_query
    routed_gpu_hours_per_year = (routed_seconds_per_day * days_per_year) / 3600
    routed_annual_cost = routed_gpu_hours_per_year * cost_per_70b_hour
    
    # Calculate expected performance
    baseline_performance = router.performance['70B']
    routed_performance = sum(
        router.performance[model_type] * probability 
        for model_type, (_, _, probability) in token_distribution.items()
    )
    
    savings = baseline_annual_cost - routed_annual_cost
    savings_pct = (savings / baseline_annual_cost) * 100
    performance_loss_pct = (baseline_performance - routed_performance) * 100
    
    print("="*70)
    print("PRODUCTION DEPLOYMENT COST SIMULATION - WITH REAL PERFORMANCE DATA")
    print("="*70)
    
    print(f"\nAssumptions:")
    print(f"  Queries per day: {queries_per_day:,}")
    print(f"  70B GPU cost: ${cost_per_70b_hour:.2f}/hour")
    print(f"  Actual performance - 70B: {router.performance['70B']*100:.1f}%, "
          f"8B: {router.performance['8B']*100:.1f}%, "
          f"Distilled: {router.performance['distilled']*100:.1f}%")
    
    print(f"\nBaseline (Always 70B):")
    print(f"  GPU hours/year: {baseline_gpu_hours_per_year:,.0f}")
    print(f"  Annual cost: ${baseline_annual_cost:,.2f}")
    print(f"  Expected performance: {baseline_performance*100:.1f}% pass@1")
    
    print(f"\nWith Token-Based Routing:")
    print(f"  GPU hours/year: {routed_gpu_hours_per_year:,.0f}")
    print(f"  Annual cost: ${routed_annual_cost:,.2f}")
    print(f"  Expected performance: {routed_performance*100:.1f}% pass@1")
    
    print(f"\n💰 SAVINGS & TRADE-OFF:")
    print(f"  Annual savings: ${savings:,.2f}")
    print(f"  Cost reduction: {savings_pct:.1f}%")
    print(f"  Performance trade-off: {performance_loss_pct:.1f}pp")
    
    # Calculate cost per percentage point of performance
    baseline_cost_per_perf = baseline_annual_cost / (baseline_performance * 100)
    routed_cost_per_perf = routed_annual_cost / (routed_performance * 100)
    
    print(f"\n📈 EFFICIENCY METRICS:")
    print(f"  Baseline cost per 1% performance: ${baseline_cost_per_perf:,.2f}")
    print(f"  Routed cost per 1% performance: ${routed_cost_per_perf:,.2f}")
    print(f"  Efficiency improvement: {((baseline_cost_per_perf - routed_cost_per_perf) / baseline_cost_per_perf * 100):.1f}%")
    
    print("="*70)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("AFSC/EN Dynamic LLM Routing System - WITH REAL PERFORMANCE DATA")
    print("="*70)
    
    # Demo: Route some example queries
    router = ModelRouter()
    
    test_queries = [
        "Write a function to check if a number is even",
        "Implement a binary search tree with insert, delete, and search operations",
        "Design a microservices architecture for a real-time analytics platform with fault tolerance and load balancing across multiple availability zones",
        "Print 'Hello, World!' to the console",
        "Create a RESTful API with JWT authentication and role-based access control using FastAPI with SQLAlchemy ORM and PostgreSQL backend",
    ]
    
    print("\n1. ROUTING EXAMPLES WITH REAL PERFORMANCE DATA")
    print("-"*70)
    for i, query in enumerate(test_queries, 1):
        decision = router.route(query)
        print(f"\n{i}. Query: {query[:60]}...")
        print(f"   → Tokens: {decision['token_count']}")
        print(f"   → Model: {decision['model']}")
        print(f"   → Complexity: {decision['complexity']}/10")
        print(f"   → Cost: {decision['cost']:.3f}× (vs 70B)")
        print(f"   → Expected performance: {decision['expected_performance']*100:.1f}% pass@1")
        print(f"   → {decision['reasoning']}")
    
    # Evaluate on HumanEval
    print("\n\n2. HUMANEVAL EVALUATION WITH REAL DATA")
    print("-"*70)
    try:
        evaluate_routing_strategy()
    except Exception as e:
        print(f"Note: Run after HumanEval results are available")
        print(f"Error: {e}")
    
    # Production simulation
    print("\n\n3. PRODUCTION COST SIMULATION WITH REAL DATA")
    print("-"*70)
    simulate_production_deployment(queries_per_day=10000, cost_per_70b_hour=3.0)
    
    print("\n✓ Routing system with REAL performance data complete!\n")