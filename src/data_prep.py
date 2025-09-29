# src/data_prep.py
"""
Resilient data prep for shared clusters (DGX/KSU) where Hugging Face `datasets`
may crash with:
  NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.

Strategy:
  • Prefer NON-STREAMING small slices but force everything into memory
    (keep_in_memory=True) and isolate cache to a fresh local dir.
  • If that still fails due to FS quirks, fall back to STREAMING with caps and
    proactively delete only the relevant cached dataset dirs to unblock streaming.

Outputs (JSONL in data/):
  data/opencodeinstruct_python_train.jsonl
  data/codesearchnet_python_train.jsonl

Env knobs:
  # Non-streaming (preferred if pyarrow present)
  OCI_PCT=1   CSN_PCT=1

  # Streaming fallback (if non-streaming fails)
  OCI_MAX=200000  CSN_MAX=100000

Optional reliability (keep global cache clean):
  export HF_HOME="$PWD/hf_cache"
  export HF_DATASETS_CACHE="$PWD/hf_cache/datasets"
"""

from __future__ import annotations
import os
import json
import itertools
from typing import Optional

OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)

# Use a fresh, isolated cache inside the project to avoid collisions.
ISOLATED_CACHE = os.path.join(os.getcwd(), "hf_cache_isolated")
os.makedirs(ISOLATED_CACHE, exist_ok=True)

def has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False

def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

# Non-streaming slice sizes (percentages of each split)
OCI_PCT = env_int("OCI_PCT", 1)    # e.g., 1 => train[:1%]
CSN_PCT = env_int("CSN_PCT", 1)    # applied to both train and validation

# Streaming caps (row counts)
OCI_MAX = env_int("OCI_MAX", 200_000)
CSN_MAX = env_int("CSN_MAX", 100_000)

def to_chat(instruction: str, _input: str, output: str) -> dict:
    system = (
        "You are a precise Python coding assistant. "
        "Prefer minimal, runnable code and include brief tests when sensible."
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
    lang = (ex.get("language") or "").lower()
    return ("python" in lang) or (lang == "py") or (lang.endswith("/python"))

def delete_dataset_cache_like(substr_list) -> None:
    """
    Remove ONLY the specific dataset cache directories whose names contain any
    of the given substrings (e.g. 'nvidia___open_code_instruct', 'code_search_net').
    We search common cache roots, including project-local caches.
    """
    roots = []
    # Respect user-provided HF_DATASETS_CACHE first
    if os.environ.get("HF_DATASETS_CACHE"):
        roots.append(os.environ["HF_DATASETS_CACHE"])
    # Project-local caches
    roots.append(os.path.join(os.getcwd(), "hf_cache", "datasets"))
    roots.append(os.path.join(os.getcwd(), "hf_cache_isolated", "datasets"))
    # User default cache
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
            if any(s in low for s in substr_list):
                if path in seen: 
                    continue
                print(f"[info] Removing cached dataset dir to unblock: {path}")
                seen.add(path)
                try:
                    import shutil
                    shutil.rmtree(path)
                except Exception as e:
                    print(f"[warn] Failed to remove {path}: {e}")

# ---------------------- NON-STREAMING (preferred) -------------------------------------

def prepare_non_streaming() -> None:
    """
    Load small slices into memory (keep_in_memory=True) and isolate cache directory
    to avoid writing Arrow tables to problematic shared filesystems.
    """
    from datasets import load_dataset

    # 1) OpenCodeInstruct (Python-only)
    oci_split = f"train[:{OCI_PCT}%]"
    print(f"[non-streaming] Loading nvidia/OpenCodeInstruct split={oci_split} (in-memory)")
    oci = load_dataset(
        "nvidia/OpenCodeInstruct",
        split=oci_split,
        cache_dir=ISOLATED_CACHE,
        keep_in_memory=True,               # <- critical: avoid FS writes
        download_mode="reuse_cache_if_exists",
    )
    oci_py = oci.filter(is_python_lang)

    oci_out = os.path.join(OUTDIR, "opencodeinstruct_python_train.jsonl")
    rows_oci = 0
    with open(oci_out, "w", encoding="utf-8") as f:
        for ex in oci_py:
            rec = to_chat(ex.get("instruction"), ex.get("input"), ex.get("output"))
            f.write(json.dumps(rec) + "\n")
            rows_oci += 1
    print(f"✅ Wrote {oci_out} ({rows_oci} rows)")

    # 2) CodeSearchNet:python (train + val), also in-memory
    csn_train_split = f"train[:{CSN_PCT}%]"
    csn_val_split   = f"validation[:{CSN_PCT}%]"
    print(f"[non-streaming] Loading code_search_net:python train={csn_train_split} val={csn_val_split} (in-memory)")
    csn_tr = load_dataset(
        "code_search_net", "python", split=csn_train_split,
        cache_dir=ISOLATED_CACHE, keep_in_memory=True, download_mode="reuse_cache_if_exists",
    )
    csn_va = load_dataset(
        "code_search_net", "python", split=csn_val_split,
        cache_dir=ISOLATED_CACHE, keep_in_memory=True, download_mode="reuse_cache_if_exists",
    )

    def to_pair(ex):
        doc  = (ex.get("func_documentation_string") or "").strip()
        code = (ex.get("func_code_string") or "").strip()
        if doc and code:
            return to_chat("Write the Python function described by this docstring:", doc, code)
        return None

    csn_out = os.path.join(OUTDIR, "codesearchnet_python_train.jsonl")
    rows_csn = 0
    with open(csn_out, "w", encoding="utf-8") as f:
        for ex in csn_tr:
            rec = to_pair(ex)
            if rec:
                f.write(json.dumps(rec) + "\n")
                rows_csn += 1
        for ex in csn_va:
            rec = to_pair(ex)
            if rec:
                f.write(json.dumps(rec) + "\n")
                rows_csn += 1
    print(f"✅ Wrote {csn_out} ({rows_csn} rows)")
    print("✅ Data prep complete (non-streaming, in-memory). Increase OCI_PCT/CSN_PCT if resources allow.")

# ---------------------- STREAMING (fallback) ------------------------------------------

def prepare_streaming() -> None:
    """
    Streaming with row caps. Before streaming, delete only the relevant cached
    dataset dirs so `datasets` won’t hit local shards and crash.
    """
    from datasets import load_dataset

    # Remove stale caches that break streaming
    delete_dataset_cache_like([
        "nvidia___open_code_instruct",
        "open_code_instruct",              # catch variants
        "code_search_net",
        "codesearchnet",
    ])

    # 1) OpenCodeInstruct (stream)
    print("[streaming] Streaming nvidia/OpenCodeInstruct (Python-only)…")
    rows_oci = 0
    oci_out = os.path.join(OUTDIR, "opencodeinstruct_python_train.jsonl")
    try:
        oci_stream = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    except NotImplementedError:
        print("[warn] Streaming refused for OpenCodeInstruct; skipping OCI.")
        oci_stream = None

    if oci_stream is not None:
        with open(oci_out, "w", encoding="utf-8") as f:
            for ex in oci_stream:
                if not is_python_lang(ex):
                    continue
                rec = to_chat(ex.get("instruction"), ex.get("input"), ex.get("output"))
                f.write(json.dumps(rec) + "\n")
                rows_oci += 1
                if rows_oci >= OCI_MAX:
                    break
        print(f"✅ Wrote {oci_out} ({rows_oci} rows)")
    else:
        print("[info] Skipped OCI; no file written.")

    # 2) CodeSearchNet:python (stream)
    print("[streaming] Streaming code_search_net:python (train + validation)…")
    rows_csn = 0
    csn_out = os.path.join(OUTDIR, "codesearchnet_python_train.jsonl")

    try:
        csn_tr = load_dataset("code_search_net", "python", split="train", streaming=True)
        csn_va = load_dataset("code_search_net", "python", split="validation", streaming=True)
    except NotImplementedError:
        print("[warn] Streaming refused for CodeSearchNet even after cleanup. Try non-streaming small slices by installing pyarrow:\n"
              "    conda install -y -n llama311 -c conda-forge pyarrow\n"
              "Proceeding with zero CSN rows.")
        csn_tr = csn_va = None

    def csn_pairs(stream):
        for ex in stream:
            doc = (ex.get("func_documentation_string") or "").strip()
            code = (ex.get("func_code_string") or "").strip()
            if doc and code:
                yield to_chat("Write the Python function described by this docstring:", doc, code)

    with open(csn_out, "w", encoding="utf-8") as f:
        if csn_tr is not None:
            for rec in itertools.islice(csn_pairs(csn_tr), CSN_MAX // 2):
                f.write(json.dumps(rec) + "\n"); rows_csn += 1
        if csn_va is not None:
            for rec in itertools.islice(csn_pairs(csn_va), CSN_MAX - rows_csn):
                f.write(json.dumps(rec) + "\n"); rows_csn += 1
    print(f"✅ Wrote {csn_out} ({rows_csn} rows)")
    print("✅ Data prep complete (streaming fallback).")

# ---------------------- Entrypoint ----------------------------------------------------

def main():
    # Prefer non-streaming with in-memory slices if pyarrow is present.
    if has_pyarrow():
        print("[mode] pyarrow detected → non-streaming (in-memory) small slices.")
        try:
            prepare_non_streaming()
            return
        except NotImplementedError as e:
            print(f"[warn] Non-streaming still failed on this FS: {e}\n"
                  f"       Falling back to streaming with caps…")

    print("[mode] Using streaming fallback with caps.")
    prepare_streaming()

if __name__ == "__main__":
    main()
