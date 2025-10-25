#!/usr/bin/env python3
"""
Dynamic LLM Routing System for AFSC/EN
Routes queries to appropriate model based on complexity.

Based on your Slide 4 formula:
C ∝ P·ε + p·(1-ε)

Where to route:
  - log(P/p) ≥ 1 → Use compressed model (cheaper)
  - Otherwise → Use full model (quality needed)
"""

import re
import json
from typing import Dict, Tuple, Optional
from pathlib import Path


class ComplexityEstimator:
    """
    Estimate query complexity to determine which model to use.
    
    Complexity factors:
    1. Query length (longer = more complex)
    2. Technical terms (more = more complex)
    3. Code patterns (classes, decorators = more complex)
    4. Ambiguity (multiple interpretations = more complex)
    """
    
    def __init__(self):
        # Keywords indicating high complexity
        self.complex_keywords = {
            'class', 'decorator', 'async', 'await', 'generator',
            'metaclass', 'abstract', 'interface', 'protocol',
            'concurrency', 'threading', 'multiprocessing',
            'optimization', 'algorithm', 'data structure',
            'design pattern', 'architecture', 'refactor'
        }
        
        # Simple task indicators
        self.simple_keywords = {
            'hello world', 'print', 'add', 'subtract', 'multiply',
            'divide', 'simple', 'basic', 'easy', 'beginner',
            'return', 'list', 'loop', 'if', 'else'
        }
    
    def estimate_complexity(self, query: str) -> int:
        """
        Estimate query complexity on scale of 1-10.
        
        1-3: Simple (use 4-bit)
        4-7: Medium (use 8B full)
        8-10: Complex (use 70B)
        """
        query_lower = query.lower()
        score = 5  # Start at medium
        
        # Factor 1: Length
        word_count = len(query.split())
        if word_count < 20:
            score -= 2
        elif word_count > 50:
            score += 2
        
        # Factor 2: Technical complexity
        complex_count = sum(1 for kw in self.complex_keywords if kw in query_lower)
        simple_count = sum(1 for kw in self.simple_keywords if kw in query_lower)
        score += complex_count
        score -= simple_count
        
        # Factor 3: Code patterns
        if 'class ' in query:
            score += 1
        if '@' in query:  # Decorators
            score += 1
        if re.search(r'def.*\(.*\).*->', query):  # Type hints
            score += 1
        
        # Factor 4: Multiple requirements
        if query.count('and') > 2:
            score += 1
        if query.count(',') > 3:
            score += 1
        
        # Clamp to 1-10
        return max(1, min(10, score))


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
        
        # Default thresholds
        self.thresholds = complexity_thresholds or {
            '4bit': (0, 3),   # Simple queries
            '8B': (3, 7),     # Medium queries
            '70B': (7, 10),   # Complex queries
        }
        
        # Default costs (relative to 70B)
        self.costs = model_costs or {
            '4bit': 0.03,
            '8B': 0.11,
            '70B': 1.0,
        }
        
        # Performance expectations (from your HumanEval results)
        self.performance = {
            '4bit': 0.476,  # 47.6% pass@1
            '8B': 0.524,    # 52.4% pass@1
            '70B': 0.683,   # 68.3% pass@1
        }
    
    def route(self, query: str) -> Dict:
        """
        Determine which model to use for a query.
        
        Returns:
            {
                'model': '4bit' | '8B' | '70B',
                'complexity': int (1-10),
                'cost': float,
                'expected_performance': float
            }
        """
        complexity = self.estimator.estimate_complexity(query)
        
        # Select model based on complexity
        for model, (min_score, max_score) in self.thresholds.items():
            if min_score <= complexity <= max_score:
                return {
                    'model': model,
                    'complexity': complexity,
                    'cost': self.costs[model],
                    'expected_performance': self.performance[model],
                    'reasoning': self._get_reasoning(complexity, model)
                }
        
        # Default to 70B if nothing matches
        return {
            'model': '70B',
            'complexity': complexity,
            'cost': self.costs['70B'],
            'expected_performance': self.performance['70B'],
            'reasoning': 'Default to highest quality'
        }
    
    def _get_reasoning(self, complexity: int, model: str) -> str:
        """Explain routing decision."""
        if complexity <= 3:
            return f"Simple query (complexity={complexity}) → Use 4-bit for 97% cost savings"
        elif complexity <= 7:
            return f"Medium query (complexity={complexity}) → Use 8B for 89% cost savings"
        else:
            return f"Complex query (complexity={complexity}) → Use 70B for best quality"


def evaluate_routing_strategy(humaneval_results_dir: str = "results"):
    """
    Evaluate routing strategy on HumanEval problems.
    Shows what the cost/performance would be with routing vs always-70B.
    """
    from human_eval.data import read_problems
    
    router = ModelRouter()
    problems = read_problems()
    
    # Load actual performance data
    results = {}
    for model_name in ['70B', '8B', '4bit']:
        metrics_file = Path(humaneval_results_dir) / f"teacher_{model_name}_final.metrics.json" \
                       if '70B' in model_name else \
                       Path(humaneval_results_dir) / f"student_{model_name}_final.metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file) as f:
                results[model_name] = json.load(f)
    
    # Analyze routing decisions
    routing_decisions = {}
    for task_id, problem in problems.items():
        prompt = problem['prompt']
        decision = router.route(prompt)
        routing_decisions[task_id] = decision
    
    # Calculate statistics
    model_counts = {'4bit': 0, '8B': 0, '70B': 0}
    for decision in routing_decisions.values():
        model_counts[decision['model']] += 1
    
    total = len(routing_decisions)
    
    print("="*70)
    print("ROUTING STRATEGY EVALUATION")
    print("="*70)
    
    print(f"\nQuery Distribution ({total} problems):")
    print(f"  4-bit:  {model_counts['4bit']:3d} ({model_counts['4bit']/total*100:5.1f}%)")
    print(f"  8B:     {model_counts['8B']:3d} ({model_counts['8B']/total*100:5.1f}%)")
    print(f"  70B:    {model_counts['70B']:3d} ({model_counts['70B']/total*100:5.1f}%)")
    
    # Calculate weighted costs
    baseline_cost = 1.0 * total  # Always use 70B
    routed_cost = sum(router.costs[d['model']] for d in routing_decisions.values())
    savings = (baseline_cost - routed_cost) / baseline_cost * 100
    
    print(f"\nCost Analysis:")
    print(f"  Baseline (always 70B): {baseline_cost:.2f} units")
    print(f"  With routing:          {routed_cost:.2f} units")
    print(f"  Savings:               {savings:.1f}%")
    
    # Expected performance (weighted average)
    expected_perf = sum(
        router.performance[d['model']] for d in routing_decisions.values()
    ) / total
    
    baseline_perf = router.performance['70B']
    perf_loss = (baseline_perf - expected_perf) * 100
    
    print(f"\nPerformance Analysis:")
    print(f"  Baseline (always 70B): {baseline_perf*100:.1f}% pass@1")
    print(f"  With routing:          {expected_perf*100:.1f}% pass@1")
    print(f"  Performance loss:      {perf_loss:.1f}pp")
    
    # Show some examples
    print(f"\nExample Routing Decisions:")
    print("-"*70)
    for i, (task_id, decision) in enumerate(list(routing_decisions.items())[:5]):
        problem = problems[task_id]
        prompt_preview = problem['prompt'].split('\n')[0][:50]
        print(f"\n{i+1}. {task_id}")
        print(f"   Prompt: {prompt_preview}...")
        print(f"   → {decision['model']} (complexity={decision['complexity']})")
        print(f"   → {decision['reasoning']}")
    
    print("="*70)
    
    return routing_decisions


def simulate_production_deployment(queries_per_day: int = 10000,
                                   cost_per_gpu_hour: float = 3.0):
    """
    Simulate annual costs for AFSC/EN deployment.
    """
    router = ModelRouter()
    
    # Assume query distribution matches HumanEval complexity distribution
    # For production, these would come from actual query logs
    sample_queries = {
        'simple': [
            "Write a function to add two numbers",
            "Print hello world",
            "Create a function that returns the sum of a list",
        ],
        'medium': [
            "Implement a binary search algorithm",
            "Create a REST API endpoint for user authentication",
            "Write a function to parse JSON and extract nested fields",
        ],
        'complex': [
            "Design a distributed caching system with automatic failover",
            "Implement a thread-safe singleton pattern with lazy initialization",
            "Create an async web scraper with rate limiting and retry logic",
        ]
    }
    
    # Estimate distribution (would use real logs in production)
    distribution = {
        'simple': 0.75,   # 75% of queries are simple
        'medium': 0.20,   # 20% are medium
        'complex': 0.05,  # 5% are complex
    }
    
    # Model inference times (seconds per query)
    inference_times = {
        '4bit': 0.4,
        '8B': 0.6,
        '70B': 5.0,
    }
    
    days_per_year = 365
    
    # Baseline: Always use 70B
    baseline_seconds_per_day = queries_per_day * inference_times['70B']
    baseline_gpu_hours_per_year = (baseline_seconds_per_day * days_per_year) / 3600
    baseline_annual_cost = baseline_gpu_hours_per_year * cost_per_gpu_hour
    
    # With routing
    routed_seconds_per_day = queries_per_day * (
        distribution['simple'] * inference_times['4bit'] +
        distribution['medium'] * inference_times['8B'] +
        distribution['complex'] * inference_times['70B']
    )
    routed_gpu_hours_per_year = (routed_seconds_per_day * days_per_year) / 3600
    routed_annual_cost = routed_gpu_hours_per_year * cost_per_gpu_hour
    
    savings = baseline_annual_cost - routed_annual_cost
    savings_pct = (savings / baseline_annual_cost) * 100
    
    print("="*70)
    print("PRODUCTION DEPLOYMENT COST SIMULATION")
    print("="*70)
    
    print(f"\nAssumptions:")
    print(f"  Queries per day: {queries_per_day:,}")
    print(f"  GPU cost: ${cost_per_gpu_hour:.2f}/hour")
    print(f"  Query distribution: {distribution}")
    
    print(f"\nBaseline (Always 70B):")
    print(f"  GPU hours/year: {baseline_gpu_hours_per_year:,.0f}")
    print(f"  Annual cost: ${baseline_annual_cost:,.2f}")
    
    print(f"\nWith Dynamic Routing:")
    print(f"  GPU hours/year: {routed_gpu_hours_per_year:,.0f}")
    print(f"  Annual cost: ${routed_annual_cost:,.2f}")
    
    print(f"\n💰 SAVINGS:")
    print(f"  Annual: ${savings:,.2f}")
    print(f"  Percentage: {savings_pct:.1f}%")
    
    print("="*70)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("AFSC/EN Dynamic LLM Routing System Demo")
    print("="*70)
    
    # Demo: Route some example queries
    router = ModelRouter()
    
    test_queries = [
        "Write a function to check if a number is even",
        "Implement a binary search tree with insert, delete, and search operations",
        "Design a microservices architecture for a real-time analytics platform with fault tolerance",
        "Print 'Hello, World!' to the console",
        "Create a RESTful API with JWT authentication and role-based access control",
    ]
    
    print("\n1. ROUTING EXAMPLES")
    print("-"*70)
    for i, query in enumerate(test_queries, 1):
        decision = router.route(query)
        print(f"\n{i}. Query: {query[:60]}...")
        print(f"   → Model: {decision['model']}")
        print(f"   → Complexity: {decision['complexity']}/10")
        print(f"   → Cost: {decision['cost']:.3f}× (vs 70B)")
        print(f"   → {decision['reasoning']}")
    
    # Evaluate on HumanEval
    print("\n\n2. HUMANEVAL EVALUATION")
    print("-"*70)
    try:
        evaluate_routing_strategy()
    except Exception as e:
        print(f"Note: Run after HumanEval results are available")
        print(f"Error: {e}")
    
    # Production simulation
    print("\n\n3. PRODUCTION COST SIMULATION")
    print("-"*70)
    simulate_production_deployment(queries_per_day=10000, cost_per_gpu_hour=3.0)
    
    print("\n✓ Demo complete!\n")