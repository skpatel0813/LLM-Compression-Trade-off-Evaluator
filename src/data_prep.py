# src/data_prep.py
"""
MBPP dataset preparation for knowledge distillation training.

This script downloads the MBPP (Mostly Basic Python Problems) dataset and 
converts it into chat conversation format for training.

MBPP contains:
  - ~1,000 Python code generation problems
  - Function-level tasks with test cases
  - Clean, educational programming problems
  
What we create:
  data/mbpp_train.jsonl (~800 examples - 80%)
  data/mbpp_val.jsonl (~100 examples - 10%)
  data/mbpp_test.jsonl (~100 examples - 10%)

Environment variables (optional):
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
import random
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

# Set seed for reproducible splits
random.seed(42)


def to_chat(text: str, code: str, test_list: List[str]) -> Dict[str, Any]:
    """
    Turn MBPP examples into friendly chat conversations.
    
    Args:
        text: The problem description
        code: The solution code
        test_list: List of test cases
    
    Returns:
        Dictionary with 'messages' key containing the conversation
    """
    system = (
        "You are a helpful Python coding assistant. "
        "Write clean, working code based on the given instructions."
    )
    
    # Build user message: problem description + test cases
    user_content = text.strip()
    
    # Add test cases if available
    if test_list:
        user_content += "\n\nTest cases:\n"
        for test in test_list:
            user_content += f"{test}\n"
    
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": code.strip()},
        ]
    }


def prepare_mbpp_data() -> None:
    """
    Download and prepare MBPP dataset in chat conversation format.
    
    Splits the data into:
      - Train: 80% (~800 examples)
      - Validation: 10% (~100 examples)
      - Test: 10% (~100 examples)
    """
    from datasets import load_dataset
    
    print("="*70)
    print("MBPP Dataset Preparation")
    print("="*70)
    
    # Load MBPP dataset from Hugging Face
    print(f"\n[1/4] Downloading MBPP dataset from Hugging Face...")
    try:
        dataset = load_dataset(
            "mbpp",
            "sanitized",  # Use sanitized version for better quality
            cache_dir=ISOLATED_CACHE,
            download_mode="reuse_cache_if_exists",
        )
        print("✓ Dataset downloaded successfully")
        
        # Debug: Show what fields are available
        if "train" in dataset and len(dataset["train"]) > 0:
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
    
    # Combine all splits from MBPP
    all_examples = []
    for split_name in ["train", "validation", "test"]:
        if split_name in dataset:
            all_examples.extend(list(dataset[split_name]))
    
    total_count = len(all_examples)
    print(f"\n[2/4] Total examples in MBPP: {total_count}")
    
    # Shuffle for random splits
    random.shuffle(all_examples)
    
    # Calculate split sizes (80/10/10)
    train_size = int(0.8 * total_count)
    val_size = int(0.1 * total_count)
    # test_size = remaining
    
    splits = [
        (all_examples[:train_size], "mbpp_train.jsonl", "Training"),
        (all_examples[train_size:train_size+val_size], "mbpp_val.jsonl", "Validation"),
        (all_examples[train_size+val_size:], "mbpp_test.jsonl", "Test"),
    ]
    
    total_examples = 0
    
    print(f"\n[3/4] Creating splits:")
    print(f"  Train: {len(splits[0][0])} examples (80%)")
    print(f"  Validation: {len(splits[1][0])} examples (10%)")
    print(f"  Test: {len(splits[2][0])} examples (10%)")
    
    for split_data, output_filename, display_name in splits:
        print(f"\n[4/4] Processing {display_name} split...")
        
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
                
                # Extract fields - MBPP uses: text, code, test_list
                text = example.get("text", "").strip()
                code = example.get("code", "").strip()
                test_list = example.get("test_list", [])
                
                # Skip examples with missing critical fields
                if not text:
                    if idx == 0:
                        print(f"  [debug] Skipping: no text found")
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
                        text=text,
                        code=code,
                        test_list=test_list
                    )
                    
                    # Add metadata for tracking
                    chat_record["task_id"] = example.get("task_id", idx)
                    chat_record["example_id"] = idx
                    chat_record["split"] = display_name.lower()
                    
                    # Write to JSONL file
                    f.write(json.dumps(chat_record, ensure_ascii=False) + "\n")
                    rows_written += 1
                    
                except Exception as e:
                    print(f"  [warning] Failed to process example {idx}: {e}")
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
    
    for _, filename, display_name in splits:
        filepath = os.path.join(OUTDIR, filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            # Count lines
            with open(filepath, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"  • {filename:30s} - {line_count:5d} examples ({size_kb:8.1f} KB)")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Train with: python -m src.train_kd --bf16 True --seq_len 2048")
    print("  2. Evaluate with HumanEval (no changes needed)")
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
                print("Example ID:", sample.get("example_id", "N/A"))
                print("Split:", sample.get("split", "N/A"))
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