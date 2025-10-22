# src/data_prep.py
"""
MBPP dataset preparation for knowledge distillation training.

This script downloads the MBPP (Mostly Basic Programming Problems) dataset
and converts it into chat conversation format for training.

MBPP contains:
  - ~374 training examples
  - ~90 validation examples  
  - ~500 test examples
  
Total: ~974 Python programming problems with test cases

What we create:
  data/mbpp_train.jsonl
  data/mbpp_val.jsonl
  data/mbpp_test.jsonl

Environment variables (optional):
  MBPP_VARIANT="sanitized" or "full" (default: sanitized)
  HF_HOME="$PWD/hf_cache"
  HF_DATASETS_CACHE="$PWD/hf_cache/datasets"

Usage:
  python -m src.data_prep
  
  # Or with custom cache location:
  export HF_HOME="$PWD/hf_cache"
  export HF_DATASETS_CACHE="$PWD/hf_cache/datasets"
  python -m src.data_prep
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Any

# Where we'll save our conversation files
OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)

# Create a special folder just for our downloads to avoid mixing with others
ISOLATED_CACHE = os.path.join(os.getcwd(), "hf_cache_isolated")
os.makedirs(ISOLATED_CACHE, exist_ok=True)

# Respect custom HF caches if provided
for env_var in ("HF_HOME", "HF_DATASETS_CACHE"):
    v = os.environ.get(env_var)
    if v:
        os.makedirs(v, exist_ok=True)


def to_chat(prompt: str, code: str, test_cases: List[str]) -> Dict[str, Any]:
    """
    Turn MBPP examples into friendly chat conversations.
    
    Args:
        prompt: The problem description (e.g., "Write a function to find the minimum cost path")
        code: The solution code
        test_cases: List of test assertions (e.g., ["min_cost([[1,2,3],[4,5,6]], 2, 2) == 8"])
    
    Returns:
        Dictionary with 'messages' key containing the conversation
    """
    system = (
        "You are a helpful Python coding assistant. "
        "Write clean, working code and include simple tests when helpful."
    )
    
    # Build user message: problem description + test cases
    user_content = prompt.strip()
    
    # Add test cases if available (helps student understand requirements)
    if test_cases:
        user_content += "\n\nTest cases:\n"
        # Include up to 3 test cases for context
        for i, test_case in enumerate(test_cases[:3], 1):
            # Clean up test case (remove 'assert' if present)
            tc = test_case.strip()
            if not tc.startswith('assert'):
                tc = f"assert {tc}"
            user_content += f"{tc}\n"
    
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": code.strip()},
        ]
    }


def prepare_mbpp_data() -> None:
    """
    Download and prepare MBPP dataset in chat conversation format.
    
    MBPP has two variants:
      - "sanitized": Cleaned version with ~420 examples (recommended)
      - "full": Original version with ~1000 examples
    
    We use "sanitized" by default as it has better quality.
    """
    from datasets import load_dataset
    
    # Choose MBPP variant (sanitized is recommended)
    variant = os.environ.get("MBPP_VARIANT", "sanitized").lower()
    if variant not in ["sanitized", "full"]:
        print(f"[warning] Unknown MBPP_VARIANT '{variant}', using 'sanitized'")
        variant = "sanitized"
    
    print("="*70)
    print(f"MBPP Dataset Preparation (variant: {variant})")
    print("="*70)
    
    # Load MBPP dataset from Hugging Face
    print(f"\n[1/4] Downloading MBPP '{variant}' dataset from Hugging Face...")
    try:
        dataset = load_dataset(
            "mbpp",
            name=variant,
            cache_dir=ISOLATED_CACHE,
            download_mode="reuse_cache_if_exists",
        )
        print("✓ Dataset downloaded successfully")
        
        # Debug: Show what fields are available
        if len(dataset["train"]) > 0:
            print("\n[debug] Available fields in dataset:")
            sample = dataset["train"][0]
            for key in sample.keys():
                print(f"  - {key}: {type(sample[key])}")
            print()
        
    except Exception as e:
        print(f"✗ Failed to download MBPP dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Try: pip install datasets --upgrade")
        print("  3. Clear cache: rm -rf hf_cache_isolated/")
        raise
    
    # Process each split
    splits_to_process = [
        ("train", "mbpp_train.jsonl", "Training"),
        ("validation", "mbpp_val.jsonl", "Validation"),
        ("test", "mbpp_test.jsonl", "Test"),
    ]
    
    total_examples = 0
    
    for split_name, output_filename, display_name in splits_to_process:
        if split_name not in dataset:
            print(f"\n[warning] Split '{split_name}' not found in dataset, skipping...")
            continue
        
        print(f"\n[{splits_to_process.index((split_name, output_filename, display_name)) + 2}/4] Processing {display_name} split...")
        
        split_data = dataset[split_name]
        output_path = os.path.join(OUTDIR, output_filename)
        
        rows_written = 0
        skipped = 0
        
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, example in enumerate(split_data):
                # Debug first example to see actual field names
                if idx == 0:
                    print(f"  [debug] First example keys: {list(example.keys())}")
                    print(f"  [debug] Sample data:")
                    for k, v in example.items():
                        val_str = str(v)[:100] if v else "None"
                        print(f"    {k}: {val_str}")
                
                # Extract fields - MBPP uses different field names
                # Common field names in MBPP: text, code, test_list, task_id
                task_id = example.get("task_id", idx)
                
                # Try multiple possible field names for prompt
                prompt = (example.get("text") or 
                         example.get("prompt") or 
                         example.get("description") or "").strip()
                
                # Try multiple possible field names for code
                code = (example.get("code") or 
                       example.get("solution") or 
                       example.get("canonical_solution") or "").strip()
                
                # Try multiple possible field names for test cases
                test_list = (example.get("test_list") or 
                           example.get("test_cases") or 
                           example.get("tests") or [])
                
                # Skip examples with missing critical fields
                if not prompt:
                    if idx == 0:
                        print(f"  [debug] Skipping: no prompt found")
                    skipped += 1
                    continue
                    
                if not code:
                    if idx == 0:
                        print(f"  [debug] Skipping: no code found")
                    skipped += 1
                    continue
                
                # Convert to chat format
                try:
                    chat_record = to_chat(
                        prompt=prompt,
                        code=code,
                        test_cases=test_list if isinstance(test_list, list) else []
                    )
                    
                    # Add metadata for tracking
                    chat_record["task_id"] = task_id
                    chat_record["split"] = split_name
                    
                    # Write to JSONL file
                    f.write(json.dumps(chat_record, ensure_ascii=False) + "\n")
                    rows_written += 1
                    
                except Exception as e:
                    print(f"  [warning] Failed to process task_id={task_id}: {e}")
                    skipped += 1
                    continue
        
        total_examples += rows_written
        
        print(f"  ✓ Saved {output_path}")
        print(f"    - Written: {rows_written} examples")
        if skipped > 0:
            print(f"    - Skipped: {skipped} examples (missing fields)")
    
    # Summary
    print("\n" + "="*70)
    print("MBPP Dataset Preparation Complete!")
    print("="*70)
    print(f"\nTotal examples prepared: {total_examples}")
    print(f"\nOutput files in '{OUTDIR}/':")
    
    for _, filename, display_name in splits_to_process:
        filepath = os.path.join(OUTDIR, filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            # Count lines
            with open(filepath, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"  • {filename:25s} - {line_count:4d} examples ({size_kb:7.1f} KB)")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Train with: python -m src.train_kd --bf16 True --seq_len 2048")
    print("  2. Evaluate with: python src/eval_codebleu_hub.py ...")
    print("="*70 + "\n")


def verify_data_files() -> bool:
    """
    Verify that all required data files exist and are not empty.
    
    Returns:
        True if all files are present and valid, False otherwise
    """
    required_files = [
        "mbpp_train.jsonl",
        "mbpp_val.jsonl", 
        "mbpp_test.jsonl",
    ]
    
    all_valid = True
    
    for filename in required_files:
        filepath = os.path.join(OUTDIR, filename)
        
        if not os.path.exists(filepath):
            print(f"✗ Missing: {filepath}")
            all_valid = False
        elif os.path.getsize(filepath) == 0:
            print(f"✗ Empty: {filepath}")
            all_valid = False
        else:
            # Count lines to verify content
            with open(filepath, 'r') as f:
                line_count = sum(1 for _ in f)
            if line_count == 0:
                print(f"✗ No data: {filepath}")
                all_valid = False
            else:
                print(f"✓ Valid: {filepath} ({line_count} examples)")
    
    return all_valid


def show_sample_data() -> None:
    """
    Display a sample from the training data to verify format.
    """
    train_file = os.path.join(OUTDIR, "mbpp_train.jsonl")
    
    if not os.path.exists(train_file):
        print("No training file found to display sample")
        return
    
    print("\n" + "="*70)
    print("Sample Training Example")
    print("="*70 + "\n")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line:
            try:
                sample = json.loads(first_line)
                
                print("Task ID:", sample.get("task_id", "N/A"))
                print("\nMessages:")
                for msg in sample.get("messages", []):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    print(f"\n[{role.upper()}]")
                    # Truncate long content
                    if len(content) > 300:
                        print(content[:300] + "...")
                    else:
                        print(content)
                
            except json.JSONDecodeError:
                print("Failed to parse sample JSON")
    
    print("\n" + "="*70 + "\n")


def main():
    """
    Main entry point for MBPP data preparation.
    """
    print("\n" + "="*70)
    print("MBPP Data Preparation Script")
    print("="*70 + "\n")
    
    # Check if data already exists
    if verify_data_files():
        print("\n✓ All MBPP data files already exist and are valid!")
        print("\nOptions:")
        print("  1. Continue anyway (will re-download and overwrite)")
        print("  2. Skip preparation")
        
        user_input = os.environ.get("FORCE_REDOWNLOAD", "").lower()
        if user_input not in ["1", "yes", "true", "force"]:
            print("\nSkipping data preparation (files already exist)")
            print("To force re-download, set: export FORCE_REDOWNLOAD=1")
            
            # Show sample
            show_sample_data()
            return
        else:
            print("\nForcing re-download...")
    
    # Prepare the data
    try:
        prepare_mbpp_data()
        
        # Verify after preparation
        print("\n[Final] Verifying prepared data files...")
        if verify_data_files():
            print("\n✓✓✓ All data files verified successfully! ✓✓✓")
            
            # Show a sample
            show_sample_data()
        else:
            print("\n✗ Some data files are invalid or missing")
            print("Please check the error messages above and try again")
            
    except KeyboardInterrupt:
        print("\n\n[interrupted] Data preparation cancelled by user")
    except Exception as e:
        print(f"\n✗ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error above and try again")


if __name__ == "__main__":
    main()