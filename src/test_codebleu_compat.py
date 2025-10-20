#!/usr/bin/env python3
"""
Full test of codebleu_compat.py with real predictions
"""
import json
import sys
from pathlib import Path

print("="*60)
print("Testing codebleu_compat.py")
print("="*60)

# Test 1: Import and basic functionality
print("\n[Test 1] Import codebleu_compat...")
try:
    from src.codebleu_compat import calc_codebleu
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Simple examples
print("\n[Test 2] Simple code examples...")
try:
    refs = [
        "def add(a, b):\n    return a + b\n",
        "def multiply(x, y):\n    return x * y\n"
    ]
    hyps = [
        "def add(x, y):\n    return x + y\n",
        "def multiply(a, b):\n    return a * b\n"
    ]
    
    result = calc_codebleu(refs, hyps, lang="python")
    print(f"✓ CodeBLEU computed successfully!")
    print(f"  Overall:         {result['codebleu']:.4f}")
    print(f"  N-gram:          {result['ngram_match']:.4f}")
    print(f"  Weighted N-gram: {result['weighted_ngram_match']:.4f}")
    print(f"  Syntax:          {result['syntax_match']:.4f}")
    print(f"  Dataflow:        {result['dataflow_match']:.4f}")
    
    if result['syntax_match'] > 0:
        print("\n✓✓✓ Syntax scoring is WORKING! ✓✓✓")
    else:
        print("\n✗ WARNING: Syntax score is 0")
        
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Download and test real predictions
print("\n[Test 3] Testing with real predictions...")
predictions_file = "test_data/predictions_eval.jsonl"

if not Path(predictions_file).exists():
    print(f"Downloading predictions file...")
    import urllib.request
    Path("test_data").mkdir(exist_ok=True)
    url = "https://huggingface.co/datasets/skpatel0813/llm-kd-evals/resolve/main/runs/teacher-70B-baseline-both-1018-2216/predictions_eval.jsonl"
    try:
        urllib.request.urlretrieve(url, predictions_file)
        print(f"✓ Downloaded to {predictions_file}")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("Skipping real data test...")
        sys.exit(0)

# Load real predictions
print(f"Loading predictions from {predictions_file}...")
refs, hyps = [], []
with open(predictions_file, "r") as f:
    for i, line in enumerate(f):
        if i >= 20:  # Test with first 20 examples
            break
        try:
            rec = json.loads(line)
            ref = rec.get("ref", "").strip()
            hyp = rec.get("hyp", "").strip()
            if ref and hyp:  # Only include non-empty pairs
                refs.append(ref)
                hyps.append(hyp)
        except Exception:
            continue

print(f"✓ Loaded {len(refs)} valid prediction pairs")

if len(refs) > 0:
    print("\nSample reference (first 150 chars):")
    print(refs[0][:150] + "...")
    print("\nSample hypothesis (first 150 chars):")
    print(hyps[0][:150] + "...")
    
    print(f"\n[Test 3] Computing CodeBLEU on {len(refs)} examples...")
    try:
        result = calc_codebleu(refs, hyps, lang="python")
        print(f"✓ Real data CodeBLEU computed!")
        print(f"  Overall:         {result['codebleu']:.4f}")
        print(f"  N-gram:          {result['ngram_match']:.4f}")
        print(f"  Weighted N-gram: {result['weighted_ngram_match']:.4f}")
        print(f"  Syntax:          {result['syntax_match']:.4f}")
        print(f"  Dataflow:        {result['dataflow_match']:.4f}")
        
        if result['syntax_match'] > 0:
            print("\n✓✓✓ SUCCESS! Real predictions have non-zero syntax score! ✓✓✓")
        else:
            print("\n⚠️  WARNING: Syntax score is 0 on real data")
            print("   This might be due to:")
            print("   - Invalid Python syntax in predictions")
            print("   - Markdown/text around code that needs cleaning")
            print("   - Empty predictions")
            print("\n   Try running: python src/clean_predictions_for_syntax.py")
            
    except Exception as e:
        print(f"✗ Real data test failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Testing complete!")
print("="*60)
print("\nNext steps:")
print("1. If syntax score > 0 on simple examples: ✓ Setup is correct")
print("2. If syntax score = 0 on real data: Clean predictions first")
print("   python src/clean_predictions_for_syntax.py test_data/predictions_eval.jsonl test_data/predictions_clean.jsonl")
print("3. Then re-run your evaluation script with the working codebleu_compat")