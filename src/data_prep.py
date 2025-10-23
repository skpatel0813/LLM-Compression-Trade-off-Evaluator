# src/data_prep.py
"""
CodeAlpaca dataset preparation for knowledge distillation training.

This script downloads the CodeAlpaca-20k dataset and converts it into 
chat conversation format for training.

CodeAlpaca contains:
  - ~20,000 Python code generation examples
  - Instruction-following format
  - Clean, diverse programming problems
  
What we create:
  data/codealpaca_train.jsonl (~16,000 examples - 80%)
  data/codealpaca_val.jsonl (~2,000 examples - 10%)
  data/codealpaca_test.jsonl (~2,000 examples - 10%)

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


def to_chat(instruction: str, input_text: str, output: str) -> Dict[str, Any]:
    """
    Turn CodeAlpaca examples into friendly chat conversations.
    
    Args:
        instruction: The task description
        input_text: Optional additional context/input
        output: The solution code
    
    Returns:
        Dictionary with 'messages' key containing the conversation
    """
    system = (
        "You are a helpful Python coding assistant. "
        "Write clean, working code based on the given instructions."
    )
    
    # Build user message: instruction + optional input
    user_content = instruction.strip()
    
    # Add input context if available
    if input_text and input_text.strip():
        user_content += f"\n\nInput:\n{input_text.strip()}"
    
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output.strip()},
        ]
    }


def prepare_codealpaca_data() -> None:
    """
    Download and prepare CodeAlpaca dataset in chat conversation format.
    
    Splits the data into:
      - Train: 80% (~16,000 examples)
      - Validation: 10% (~2,000 examples)
      - Test: 10% (~2,000 examples)
    """
    from datasets import load_dataset
    
    print("="*70)
    print("CodeAlpaca Dataset Preparation")
    print("="*70)
    
    # Load CodeAlpaca dataset from Hugging Face
    print(f"\n[1/4] Downloading CodeAlpaca-20k dataset from Hugging Face...")
    try:
        dataset = load_dataset(
            "sahil2801/CodeAlpaca-20k",
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
        print(f"✗ Failed to download CodeAlpaca dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Try: pip install datasets --upgrade")
        print("  3. Clear cache: rm -rf hf_cache_isolated/")
        raise
    
    # Get all examples (CodeAlpaca only has 'train' split)
    if "train" in dataset:
        all_examples = list(dataset["train"])
    else:
        raise ValueError("CodeAlpaca dataset has unexpected structure")
    
    total_count = len(all_examples)
    print(f"\n[2/4] Total examples in CodeAlpaca: {total_count}")
    
    # Shuffle for random splits
    random.shuffle(all_examples)
    
    # Calculate split sizes (80/10/10)
    train_size = int(0.8 * total_count)
    val_size = int(0.1 * total_count)
    # test_size = remaining
    
    splits = [
        (all_examples[:train_size], "codealpaca_train.jsonl", "Training"),
        (all_examples[train_size:train_size+val_size], "codealpaca_val.jsonl", "Validation"),
        (all_examples[train_size+val_size:], "codealpaca_test.jsonl", "Test"),
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
                
                # Extract fields - CodeAlpaca uses: instruction, input, output
                instruction = example.get("instruction", "").strip()
                input_text = example.get("input", "").strip()
                output = example.get("output", "").strip()
                
                # Skip examples with missing critical fields
                if not instruction:
                    if idx == 0:
                        print(f"  [debug] Skipping: no instruction found")
                    skipped += 1
                    continue
                    
                if not output:
                    if idx == 0:
                        print(f"  [debug] Skipping: no output found")
                    skipped += 1
                    continue
                
                # Convert to chat format
                try:
                    chat_record = to_chat(
                        instruction=instruction,
                        input_text=input_text,
                        output=output
                    )
                    
                    # Add metadata for tracking
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
    print("CodeAlpaca Dataset Preparation Complete!")
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
        "codealpaca_train.jsonl",
        "codealpaca_val.jsonl", 
        "codealpaca_test.jsonl",
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
    train_file = os.path.join(OUTDIR, "codealpaca_train.jsonl")
    
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
    Main entry point for CodeAlpaca data preparation.
    """
    print("\n" + "="*70)
    print("CodeAlpaca Data Preparation Script")
    print("="*70 + "\n")
    
    # Check if data already exists
    if verify_data_files():
        print("\n✓ All CodeAlpaca data files already exist and are valid!")
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
        prepare_codealpaca_data()
        
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