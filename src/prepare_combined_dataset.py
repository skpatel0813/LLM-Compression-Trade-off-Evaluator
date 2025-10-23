#!/usr/bin/env python3
"""
prepare_combined_dataset.py - Combine MBPP and HumanEval into 80/10/10 split

This creates a unified training dataset from both benchmarks for better generalization.

Output:
  data/combined_train.jsonl (80%)
  data/combined_val.jsonl (10%)
  data/combined_test.jsonl (10%)
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any

# For reproducibility
random.seed(42)

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(filepath: str, data: List[Dict[str, Any]]):
    """Save JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_humaneval() -> List[Dict[str, Any]]:
    """Load HumanEval and convert to chat format."""
    try:
        from human_eval.data import read_problems
    except ImportError:
        print("ERROR: human-eval not installed. Run: pip install human-eval")
        return []
    
    problems = read_problems()
    dataset = []
    
    for task_id, problem in problems.items():
        # Create chat format
        messages = [
            {
                "role": "system",
                "content": "You are a helpful Python coding assistant. Write clean, working code."
            },
            {
                "role": "user",
                "content": f"Complete this Python function:\n\n{problem['prompt']}"
            },
            {
                "role": "assistant",
                "content": problem['canonical_solution'].strip()
            }
        ]
        
        dataset.append({
            "task_id": task_id,
            "source": "humaneval",
            "messages": messages
        })
    
    return dataset


def load_mbpp() -> List[Dict[str, Any]]:
    """Load MBPP data (already in chat format)."""
    mbpp_files = [
        "data/mbpp_train.jsonl",
        "data/mbpp_val.jsonl",
        "data/mbpp_test.jsonl"
    ]
    
    dataset = []
    for filepath in mbpp_files:
        if os.path.exists(filepath):
            data = load_jsonl(filepath)
            for item in data:
                item["source"] = "mbpp"
                dataset.append(item)
    
    return dataset


def create_split(data: List[Dict[str, Any]], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split data into train/val/test with stratification by source."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"
    
    # Separate by source for stratified split
    humaneval_data = [x for x in data if x.get("source") == "humaneval"]
    mbpp_data = [x for x in data if x.get("source") == "mbpp"]
    
    # Shuffle each source
    random.shuffle(humaneval_data)
    random.shuffle(mbpp_data)
    
    def split_list(items, train_r, val_r):
        """Split a list into train/val/test."""
        n = len(items)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        
        return (
            items[:train_end],
            items[train_end:val_end],
            items[val_end:]
        )
    
    # Split each source
    he_train, he_val, he_test = split_list(humaneval_data, train_ratio, val_ratio)
    mb_train, mb_val, mb_test = split_list(mbpp_data, train_ratio, val_ratio)
    
    # Combine
    train = he_train + mb_train
    val = he_val + mb_val
    test = he_test + mb_test
    
    # Shuffle combined sets
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def main():
    print("="*70)
    print("Combining MBPP + HumanEval with 80/10/10 Split")
    print("="*70)
    
    # Load datasets
    print("\n[1/4] Loading HumanEval...")
    humaneval = load_humaneval()
    print(f"  ✓ Loaded {len(humaneval)} HumanEval problems")
    
    print("\n[2/4] Loading MBPP...")
    mbpp = load_mbpp()
    print(f"  ✓ Loaded {len(mbpp)} MBPP problems")
    
    # Combine
    all_data = humaneval + mbpp
    total = len(all_data)
    print(f"\n  Total problems: {total}")
    print(f"    - HumanEval: {len(humaneval)} ({len(humaneval)/total*100:.1f}%)")
    print(f"    - MBPP: {len(mbpp)} ({len(mbpp)/total*100:.1f}%)")
    
    # Split
    print("\n[3/4] Creating 80/10/10 split...")
    train, val, test = create_split(all_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    print(f"  ✓ Train: {len(train)} examples ({len(train)/total*100:.1f}%)")
    print(f"  ✓ Val:   {len(val)} examples ({len(val)/total*100:.1f}%)")
    print(f"  ✓ Test:  {len(test)} examples ({len(test)/total*100:.1f}%)")
    
    # Show source distribution in each split
    for split_name, split_data in [("Train", train), ("Val", val), ("Test", test)]:
        he_count = sum(1 for x in split_data if x.get("source") == "humaneval")
        mb_count = sum(1 for x in split_data if x.get("source") == "mbpp")
        print(f"    {split_name}: HumanEval={he_count}, MBPP={mb_count}")
    
    # Save
    print("\n[4/4] Saving splits...")
    save_jsonl("data/combined_train.jsonl", train)
    save_jsonl("data/combined_val.jsonl", val)
    save_jsonl("data/combined_test.jsonl", test)
    
    print("  ✓ Saved data/combined_train.jsonl")
    print("  ✓ Saved data/combined_val.jsonl")
    print("  ✓ Saved data/combined_test.jsonl")
    
    # Show sample
    print("\n" + "="*70)
    print("Sample from training set:")
    print("="*70)
    sample = train[0]
    print(f"Source: {sample.get('source')}")
    print(f"Task ID: {sample.get('task_id')}")
    print("\nMessages:")
    for msg in sample.get('messages', [])[:2]:  # Show first 2 messages
        print(f"\n[{msg['role'].upper()}]")
        content = msg['content']
        if len(content) > 200:
            print(content[:200] + "...")
        else:
            print(content)
    
    print("\n" + "="*70)
    print("✅ Dataset preparation complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Train KD model: python -m src.train_kd_combined --epochs 5")
    print("  2. Evaluate all models: bash evaluate_all_models.sh")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()