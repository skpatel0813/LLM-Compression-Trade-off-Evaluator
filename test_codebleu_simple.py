#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

print("="*60)
print("Testing codebleu_compat.py")
print("="*60)

# Test 1: Import
print("\n[Test 1] Import...")
from src.codebleu_compat import calc_codebleu
print("✓ Import successful")

# Test 2: Simple examples
print("\n[Test 2] Simple code examples...")
refs = [
    "def add(a, b):\n    return a + b\n",
    "def multiply(x, y):\n    return x * y\n"
]
hyps = [
    "def add(x, y):\n    return x + y\n",
    "def multiply(a, b):\n    return a * b\n"
]

result = calc_codebleu(refs, hyps, lang="python")
print(f"✓ CodeBLEU computed!")
print(f"  Overall:  {result['codebleu']:.4f}")
print(f"  Syntax:   {result['syntax_match']:.4f}")

if result['syntax_match'] > 0:
    print("\n✓✓✓ SUCCESS! Syntax scoring is WORKING! ✓✓✓")
else:
    print("\n✗ WARNING: Syntax score is 0")

# Test 3: Real predictions
print("\n[Test 3] Testing with real predictions...")
import json
from pathlib import Path

predictions_file = "test_data/predictions_eval.jsonl"
if Path(predictions_file).exists():
    refs, hyps = [], []
    with open(predictions_file, "r") as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            try:
                rec = json.loads(line)
                ref = rec.get("ref", "").strip()
                hyp = rec.get("hyp", "").strip()
                if ref and hyp:
                    refs.append(ref)
                    hyps.append(hyp)
            except:
                continue
    
    if len(refs) > 0:
        print(f"Loaded {len(refs)} predictions")
        print(f"\nSample ref: {refs[0][:100]}...")
        print(f"Sample hyp: {hyps[0][:100]}...")
        
        result = calc_codebleu(refs, hyps, lang="python")
        print(f"\nReal data results:")
        print(f"  CodeBLEU: {result['codebleu']:.4f}")
        print(f"  Syntax:   {result['syntax_match']:.4f}")
        
        if result['syntax_match'] > 0:
            print("\n✓✓✓ Real data has non-zero syntax! ✓✓✓")
        else:
            print("\n⚠️  Syntax is 0 - predictions may need cleaning")
else:
    print(f"No predictions file at {predictions_file}")
    print("Download it first or test with sample data only")

print("\n" + "="*60)
