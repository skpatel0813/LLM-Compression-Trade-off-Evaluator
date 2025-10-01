# src/data_prep.py
"""
Reliable data preparation for shared computers where downloading datasets might crash.

This script safely downloads coding examples and turns them into chat conversations.
It has two ways to work:
  • Method 1: Download small pieces and keep them in memory (like reading a short book)
  • Method 2: Stream data like watching a video if Method 1 doesn't work

What we create:
  data/opencodeinstruct_python_train.jsonl
  data/codesearchnet_python_train.jsonl

Settings you can change:
  # For Method 1 (small pieces)
  OCI_PCT=1   CSN_PCT=1

  # For Method 2 (streaming with limits)
  OCI_MAX=200000  CSN_MAX=100000

To keep things tidy:
  export HF_HOME="$PWD/hf_cache"
  export HF_DATASETS_CACHE="$PWD/hf_cache/datasets"
"""

from __future__ import annotations
import os
import json
import itertools
from typing import Optional

# Where we'll save our conversation files
OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)

# Create a special folder just for our downloads to avoid mixing with others
ISOLATED_CACHE = os.path.join(os.getcwd(), "hf_cache_isolated")
os.makedirs(ISOLATED_CACHE, exist_ok=True)

def has_pyarrow() -> bool:
    """Check if we have the special tool that helps read data faster"""
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False

def env_int(name: str, default: int) -> int:
    """Read numbers from settings, use default if not provided"""
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

# How much data to take for Method 1 (like taking 1% of a big cake)
OCI_PCT = env_int("OCI_PCT", 1)    # Take 1% of OpenCodeInstruct
CSN_PCT = env_int("CSN_PCT", 1)    # Take 1% of CodeSearchNet

# How many examples to take for Method 2 (like counting how many candies to take)
OCI_MAX = env_int("OCI_MAX", 200_000)  # Maximum 200,000 from OpenCodeInstruct
CSN_MAX = env_int("CSN_MAX", 100_000)  # Maximum 100,000 from CodeSearchNet

def to_chat(instruction: str, _input: str, output: str) -> dict:
    """Turn coding examples into friendly chat conversations"""
    system = (
        "You are a helpful Python coding assistant. "
        "Write clean, working code and include simple tests when helpful."
    )
    user = (instruction or "")
    if _input:
        user += ("\n" + _input)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": output or ""},
        ]
    }

def is_python_lang(ex) -> bool:
    """Check if this example is written in Python (like checking if a book is in English)"""
    lang = (ex.get("language") or "").lower()
    return ("python" in lang) or (lang == "py") or (lang.endswith("/python"))

def delete_dataset_cache_like(substr_list) -> None:
    """
    Clean up old downloaded data that might cause problems.
    Like cleaning your room before starting a new project.
    """
    # Look in different places where data might be stored
    roots = []
    # First check if user told us where to look
    if os.environ.get("HF_DATASETS_CACHE"):
        roots.append(os.environ["HF_DATASETS_CACHE"])
    # Check our project folders
    roots.append(os.path.join(os.getcwd(), "hf_cache", "datasets"))
    roots.append(os.path.join(os.getcwd(), "hf_cache_isolated", "datasets"))
    # Check the usual hiding spot
    roots.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets"))

    seen = set()
    for root in roots:
        if not root or not os.path.isdir(root): 
            continue
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if not os.path.isdir(path): 
                continue
            low = name.lower()
            # If this folder matches what we're looking for, clean it up
            if any(s in low for s in substr_list):
                if path in seen: 
                    continue
                print(f"[info] Cleaning up old data: {path}")
                seen.add(path)
                try:
                    import shutil
                    shutil.rmtree(path)
                except Exception as e:
                    print(f"[warn] Couldn't clean {path}: {e}")

# ---------------------- METHOD 1: NON-STREAMING (like reading a book) -------------------------------------

def prepare_non_streaming() -> None:
    """
    Download small pieces of data and keep them in memory.
    Like borrowing a few books from the library instead of the whole shelf.
    """
    from datasets import load_dataset

    # 1) Get Python coding examples from OpenCodeInstruct
    oci_split = f"train[:{OCI_PCT}%]"
    print(f"[Method 1] Getting {OCI_PCT}% of OpenCodeInstruct examples (keeping in memory)")
    oci = load_dataset(
        "nvidia/OpenCodeInstruct",
        split=oci_split,
        cache_dir=ISOLATED_CACHE,
        keep_in_memory=True,               # <- Important: keep everything in memory
        download_mode="reuse_cache_if_exists",
    )
    # Only keep Python examples
    oci_py = oci.filter(is_python_lang)

    # Save as conversation file
    oci_out = os.path.join(OUTDIR, "opencodeinstruct_python_train.jsonl")
    rows_oci = 0
    with open(oci_out, "w", encoding="utf-8") as f:
        for ex in oci_py:
            rec = to_chat(ex.get("instruction"), ex.get("input"), ex.get("output"))
            f.write(json.dumps(rec) + "\n")
            rows_oci += 1
    print(f"Saved {oci_out} ({rows_oci} conversations)")

    # 2) Get more examples from CodeSearchNet
    csn_train_split = f"train[:{CSN_PCT}%]"
    csn_val_split   = f"validation[:{CSN_PCT}%]"
    print(f"[Method 1] Getting {CSN_PCT}% of CodeSearchNet examples (keeping in memory)")
    csn_tr = load_dataset(
        "code_search_net", "python", split=csn_train_split,
        cache_dir=ISOLATED_CACHE, keep_in_memory=True, download_mode="reuse_cache_if_exists",
    )
    csn_va = load_dataset(
        "code_search_net", "python", split=csn_val_split,
        cache_dir=ISOLATED_CACHE, keep_in_memory=True, download_mode="reuse_cache_if_exists",
    )

    def to_pair(ex):
        """Turn code examples into question-answer pairs"""
        doc  = (ex.get("func_documentation_string") or "").strip()  # The description
        code = (ex.get("func_code_string") or "").strip()           # The code answer
        if doc and code:
            return to_chat("Write the Python function described by this docstring:", doc, code)
        return None

    csn_out = os.path.join(OUTDIR, "codesearchnet_python_train.jsonl")
    rows_csn = 0
    with open(csn_out, "w", encoding="utf-8") as f:
        # Save training examples
        for ex in csn_tr:
            rec = to_pair(ex)
            if rec:
                f.write(json.dumps(rec) + "\n")
                rows_csn += 1
        # Save validation examples
        for ex in csn_va:
            rec = to_pair(ex)
            if rec:
                f.write(json.dumps(rec) + "\n")
                rows_csn += 1
    print(f"Saved {csn_out} ({rows_csn} conversations)")
    print("All done with Method 1! You can get more data by increasing OCI_PCT/CSN_PCT.")

# ---------------------- METHOD 2: STREAMING (like watching a video) ------------------------------------------

def prepare_streaming() -> None:
    """
    Stream data with limits. Clean up first to avoid problems.
    Like watching a movie online instead of downloading it.
    """
    from datasets import load_dataset

    # Clean up first to avoid problems
    delete_dataset_cache_like([
        "nvidia___open_code_instruct",
        "open_code_instruct",              # different name styles
        "code_search_net",
        "codesearchnet",
    ])

    # 1) Stream OpenCodeInstruct examples
    print("[Method 2] Streaming OpenCodeInstruct Python examples…")
    rows_oci = 0
    oci_out = os.path.join(OUTDIR, "opencodeinstruct_python_train.jsonl")
    try:
        oci_stream = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    except NotImplementedError:
        print("[warn] Streaming not working for OpenCodeInstruct; skipping this one.")
        oci_stream = None

    if oci_stream is not None:
        with open(oci_out, "w", encoding="utf-8") as f:
            for ex in oci_stream:
                if not is_python_lang(ex):
                    continue
                rec = to_chat(ex.get("instruction"), ex.get("input"), ex.get("output"))
                f.write(json.dumps(rec) + "\n")
                rows_oci += 1
                # Stop when we have enough
                if rows_oci >= OCI_MAX:
                    break
        print(f"Saved {oci_out} ({rows_oci} conversations)")
    else:
        print("[info] No OpenCodeInstruct data saved.")

    # 2) Stream CodeSearchNet examples
    print("[Method 2] Streaming CodeSearchNet Python examples…")
    rows_csn = 0
    csn_out = os.path.join(OUTDIR, "codesearchnet_python_train.jsonl")

    try:
        csn_tr = load_dataset("code_search_net", "python", split="train", streaming=True)
        csn_va = load_dataset("code_search_net", "python", split="validation", streaming=True)
    except NotImplementedError:
        print("[warn] Streaming not working for CodeSearchNet. Try Method 1 by installing pyarrow.")
        csn_tr = csn_va = None

    def csn_pairs(stream):
        """Get question-answer pairs from the stream"""
        for ex in stream:
            doc = (ex.get("func_documentation_string") or "").strip()
            code = (ex.get("func_code_string") or "").strip()
            if doc and code:
                yield to_chat("Write the Python function described by this docstring:", doc, code)

    with open(csn_out, "w", encoding="utf-8") as f:
        if csn_tr is not None:
            # Take half from training data
            for rec in itertools.islice(csn_pairs(csn_tr), CSN_MAX // 2):
                f.write(json.dumps(rec) + "\n"); rows_csn += 1
        if csn_va is not None:
            # Take the rest from validation data
            for rec in itertools.islice(csn_pairs(csn_va), CSN_MAX - rows_csn):
                f.write(json.dumps(rec) + "\n"); rows_csn += 1
    print(f"Saved {csn_out} ({rows_csn} conversations)")
    print(" All done with Method 2!")

# ---------------------- Starting Point ----------------------------------------------------

def main():
    # Try Method 1 first if we have the right tools
    if has_pyarrow():
        print("[choice] Using Method 1: Small pieces in memory")
        try:
            prepare_non_streaming()
            return
        except NotImplementedError as e:
            print(f"[warning] Method 1 didn't work: {e}\n"
                  f"          Trying Method 2 instead…")

    print("[choice] Using Method 2: Streaming with limits")
    prepare_streaming()

if __name__ == "__main__":
    main()
