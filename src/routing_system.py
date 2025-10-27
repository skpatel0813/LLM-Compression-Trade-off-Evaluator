#!/usr/bin/env python3
"""
Dynamic LLM Routing System for AFSC/EN
Routes queries to appropriate model based on token complexity.
"""

import re
import json
import tiktoken  # For token counting
from typing import Dict, Tuple, Optional
from pathlib import Path


class TokenComplexityEstimator:
    """
    Estimate query complexity based on token count and content.
    
    Complexity factors:
    1. Token count (primary factor)
    2. Technical terms density
    3. Code patterns
    4. Structural complexity
    """
    
    def __init__(self):
        # Use GPT-4 tokenizer (similar to Llama tokenizer)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Technical keywords indicating high complexity
        self.complex_keywords = {
            'class', 'decorator', 'async', 'await', 'generator',
            'metaclass', 'abstract', 'interface', 'protocol',
            'concurrency', 'threading', 'multiprocessing',
            'optimization', 'algorithm', 'data structure',
            'design pattern', 'architecture', 'refactor',
            'distributed', 'microservice', 'kubernetes', 'docker',
            'database', 'api', 'rest', 'graphql', 'authentication',
            'authorization', 'encryption', 'security'
        }
        
        # Simple task indicators
        self.simple_keywords = {
            'hello world', 'print', 'add', 'subtract', 'multiply',
            'divide', 'simple', 'basic', 'easy', 'beginner',
            'return', 'list', 'loop', 'if', 'else', 'variable',
            'function', 'calculate', 'sum', 'average'
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def estimate_complexity(self, query: str) -> Tuple[int, int]:
        """
        Estimate query complexity based on tokens and content.
        
        Returns:
            Tuple[token_count, complexity_score(1-10)]
        """
        query_lower = query.lower()
        token_count = self.count_tokens(query)
        score = 5  # Start at medium
        
        # Factor 1: Token count (primary driver)
        if token_count < 50:
            score -= 2
        elif token_count < 100:
            score -= 1
        elif token_count > 300:
            score += 2
        elif token_count > 500:
            score += 3
        
        # Factor 2: Technical complexity density
        total_words = len(query.split())
        complex_count = sum(1 for kw in self.complex_keywords if kw in query_lower)
        simple_count = sum(1 for kw in self.simple_keywords if kw in query_lower)
        
        if total_words > 0:
            complexity_density = complex_count / total_words
            if complexity_density > 0.3:  # 30%+ technical terms
                score += 2
            elif complexity_density > 0.1:  # 10%+ technical terms
                score += 1
        
        score += complex_count
        score -= simple_count
        
        # Factor 3: Code patterns
        if 'class ' in query:
            score += 1
        if '@' in query:  # Decorators
            score += 1
        if re.search(r'def.*\(.*\).*->', query):  # Type hints
            score += 1
        if 'import ' in query:  # Imports
            score += 1
        
        # Factor 4: Multiple requirements
        if query.count('and') > 2 or query.count('or') > 2:
            score += 1
        if query.count(',') > 3:
            score += 1
        
        # Clamp to 1-10
        complexity_score = max(1, min(10, score))
        
        return token_count, complexity_score


class ModelRouter:
    """
    Routes queries to appropriate model based on token complexity and cost.
    """
    
    def __init__(self, 
                 complexity_thresholds: Dict[str, Tuple[int, int]] = None,
                 model_costs: Dict[str, float] = None):
        """
        Args:
            complexity_thresholds: {model: (min_score, max_score)}
            model_costs: {model: relative_cost}
        """
        self.estimator = TokenComplexityEstimator()
        
        # Default thresholds based on token complexity
        self.thresholds = complexity_thresholds or {
            'distilled': (0, 3),   # Simple queries, low tokens
            '8B': (3, 7),         # Medium queries
            '70B': (7, 10),       # Complex queries, high tokens
        }
        
        # Default costs (relative to 70B)
        self.costs = model_costs or {
            'distilled': 0.02,    # Even cheaper than 4bit
            '8B': 0.11,           # Student model
            '70B': 1.0,           # Teacher model
        }
        
        # Performance expectations (adjust based on your eval results)
        self.performance = {
            'distilled': 0.45,    # Estimated performance
            '8B': 0.524,          # From your data
            '70B': 0.683,         # From your data
        }
        
        # Token limits for context management
        self.token_limits = {
            'distilled': 2048,    # Smaller context
            '8B': 4096,           # Standard context
            '70B': 8192,          # Larger context
        }
    
    def route(self, query: str) -> Dict:
        """
        Determine which model to use for a query based on tokens.
        
        Returns:
            {
                'model': 'distilled' | '8B' | '70B',
                'token_count': int,
                'complexity': int (1-10),
                'cost': float,
                'expected_performance': float,
                'within_token_limit': bool,
                'reasoning': str
            }
        """
        token_count, complexity = self.estimator.estimate_complexity(query)
        
        # Select model based on complexity
        for model, (min_score, max_score) in self.thresholds.items():
            if min_score <= complexity <= max_score:
                within_limit = token_count <= self.token_limits[model]
                return {
                    'model': model,
                    'token_count': token_count,
                    'complexity': complexity,
                    'cost': self.costs[model],
                    'expected_performance': self.performance[model],
                    'within_token_limit': within_limit,
                    'reasoning': self._get_reasoning(token_count, complexity, model, within_limit)
                }
        
        # Default to 70B if nothing matches
        within_limit = token_count <= self.token_limits['70B']
        return {
            'model': '70B',
            'token_count': token_count,
            'complexity': complexity,
            'cost': self.costs['70B'],
            'expected_performance': self.performance['70B'],
            'within_token_limit': within_limit,
            'reasoning': 'Default to highest quality'
        }
    
    def _get_reasoning(self, token_count: int, complexity: int, model: str, within_limit: bool) -> str:
        """Explain routing decision with token info."""
        limit_status = "✓ within limit" if within_limit else "⚠ over limit"
        
        if complexity <= 3:
            return f"Simple query ({token_count}tokens, complexity={complexity}) → Use distilled for 98% cost savings {limit_status}"
        elif complexity <= 7:
            return f"Medium query ({token_count}tokens, complexity={complexity}) → Use 8B for 89% cost savings {limit_status}"
        else:
            return f"Complex query ({token_count}tokens, complexity={complexity}) → Use 70B for best quality {limit_status}"


def evaluate_routing_strategy_on_datasets(datasets_dir: str = "datasets"):
    """
    Evaluate routing strategy on HumanEval and MBPP problems with token analysis.
    """
    try:
        from human_eval.data import read_problems
        # For MBPP, you might need to import from your local dataset files
    except ImportError:
        print("HumanEval package not available, using simulated data")
        return
    
    router = ModelRouter()
    
    # Load HumanEval problems
    humaneval_problems = read_problems()
    
    # For MBPP, you would load similarly
    # mbpp_problems = load_mbpp_problems()
    
    print("="*80)
    print("TOKEN-BASED ROUTING EVALUATION")
    print("="*80)
    
    # Analyze routing decisions with token counts
    datasets = {
        'HumanEval': humaneval_problems,
        # 'MBPP': mbpp_problems  # Add when available
    }
    
    for dataset_name, problems in datasets.items():
        print(f"\n{dataset_name} Dataset Analysis:")
        print("-" * 50)
        
        routing_decisions = {}
        token_stats = []
        
        for task_id, problem in problems.items():
            prompt = problem['prompt']
            decision = router.route(prompt)
            routing_decisions[task_id] = decision
            token_stats.append(decision['token_count'])
        
        # Calculate statistics
        model_counts = {'distilled': 0, '8B': 0, '70B': 0}
        for decision in routing_decisions.values():
            model_counts[decision['model']] += 1
        
        total = len(routing_decisions)
        
        print(f"Total problems: {total}")
        print(f"Token statistics:")
        print(f"  Min: {min(token_stats)} tokens")
        print(f"  Max: {max(token_stats)} tokens")
        print(f"  Avg: {sum(token_stats)/len(token_stats):.1f} tokens")
        
        print(f"\nModel Distribution:")
        for model in ['distilled', '8B', '70B']:
            count = model_counts[model]
            percentage = count / total * 100
            print(f"  {model:10}: {count:3d} problems ({percentage:5.1f}%)")
        
        # Cost analysis
        baseline_cost = 1.0 * total  # Always use 70B
        routed_cost = sum(router.costs[d['model']] for d in routing_decisions.values())
        savings = (baseline_cost - routed_cost) / baseline_cost * 100
        
        print(f"\nCost Analysis:")
        print(f"  Baseline (always 70B): {baseline_cost:.2f} units")
        print(f"  With routing:          {routed_cost:.2f} units")
        print(f"  Savings:               {savings:.1f}%")
        
        # Show token-based examples
        print(f"\nToken-based Routing Examples:")
        print("-" * 50)
        for i, (task_id, decision) in enumerate(list(routing_decisions.items())[:3]):
            problem = problems[task_id]
            prompt_preview = problem['prompt'].split('\n')[0][:60]
            print(f"\n{i+1}. {task_id}")
            print(f"   Prompt: {prompt_preview}...")
            print(f"   Tokens: {decision['token_count']}")
            print(f"   → {decision['model']} (complexity={decision['complexity']})")
            print(f"   → {decision['reasoning']}")


def analyze_dataset_token_sizes():
    """
    Analyze token sizes of HumanEval and MBPP datasets.
    """
    try:
        from human_eval.data import read_problems
        # Import MBPP if available
    except ImportError:
        print("Required packages not available")
        return
    
    estimator = TokenComplexityEstimator()
    humaneval_problems = read_problems()
    
    print("\n" + "="*60)
    print("DATASET TOKEN SIZE ANALYSIS")
    print("="*60)
    
    # HumanEval analysis
    humaneval_tokens = []
    for task_id, problem in humaneval_problems.items():
        token_count = estimator.count_tokens(problem['prompt'])
        humaneval_tokens.append(token_count)
    
    print(f"\nHumanEval Dataset:")
    print(f"  Problems: {len(humaneval_tokens)}")
    print(f"  Avg tokens: {sum(humaneval_tokens)/len(humaneval_tokens):.1f}")
    print(f"  Min tokens: {min(humaneval_tokens)}")
    print(f"  Max tokens: {max(humaneval_tokens)}")
    
    # Token distribution
    bins = [0, 100, 200, 300, 400, 500, float('inf')]
    distribution = [0] * (len(bins) - 1)
    
    for tokens in humaneval_tokens:
        for i in range(len(bins) - 1):
            if bins[i] <= tokens < bins[i + 1]:
                distribution[i] += 1
                break
        if tokens >= bins[-2]:
            distribution[-1] += 1
    
    print(f"\nToken Distribution:")
    for i in range(len(distribution)):
        if i < len(distribution) - 1:
            range_str = f"{bins[i]}-{bins[i+1]-1}"
        else:
            range_str = f"{bins[-2]}+"
        print(f"  {range_str:8} tokens: {distribution[i]:3d} problems")


if __name__ == "__main__":
    # Install required package: pip install tiktoken
    
    print("\n" + "="*70)
    print("TOKEN-BASED LLM ROUTING SYSTEM")
    print("="*70)
    
    # Demo with token analysis
    router = ModelRouter()
    
    test_queries = [
        "Write a function to check if a number is even",
        "Implement a binary search tree with insert, delete, and search operations",
        "Design a microservices architecture for a real-time analytics platform with fault tolerance and load balancing across multiple availability zones",
        "Print 'Hello, World!' to the console",
        "Create a RESTful API with JWT authentication, role-based access control, rate limiting, and request validation using FastAPI",
    ]
    
    print("\n1. TOKEN-BASED ROUTING EXAMPLES")
    print("-"*70)
    for i, query in enumerate(test_queries, 1):
        decision = router.route(query)
        print(f"\n{i}. Query: {query[:50]}...")
        print(f"   → Tokens: {decision['token_count']}")
        print(f"   → Model: {decision['model']}")
        print(f"   → Complexity: {decision['complexity']}/10")
        print(f"   → Cost: {decision['cost']:.3f}× (vs 70B)")
        print(f"   → {decision['reasoning']}")
    
    # Analyze dataset token sizes
    print("\n\n2. DATASET TOKEN ANALYSIS")
    print("-"*70)
    analyze_dataset_token_sizes()
    
    # Evaluate routing strategy
    print("\n\n3. ROUTING STRATEGY EVALUATION")
    print("-"*70)
    evaluate_routing_strategy_on_datasets()
    
    print("\n✓ Token-based routing demo complete!\n")