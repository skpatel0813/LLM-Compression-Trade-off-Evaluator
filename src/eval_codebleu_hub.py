# -*- coding: utf-8 -*-
"""
Evaluate a causal LM (baseline or distilled) on Python generation and push results to the Hugging Face Hub.

What you get in the Hub repo (repo_type="dataset"):
- runs/<run_name>/metrics.json          # BLEU-4, CodeBLEU (+components), wall time, GPU stats
- runs/<run_name>/predictions.jsonl     # prompt / reference / prediction
- README.md is updated with a results table row for the run

Requirements (already in your env from earlier steps):
- transformers>=4.44
- torch with CUDA
- huggingface_hub
- sacrebleu
Optional:
- codebleu (pip install codebleu) for true CodeBLEU (otherwise we skip it gracefully)
- pynvml or nvidia-ml-py3 (for GPU power/temps; we fall back to mem-only if unavailable)

Usage example:
CUDA_VISIBLE_DEVICES=0 python -m src.eval_codebleu_hub \
  --base_id_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --eval_files data/codesearchnet_python_train.jsonl \
  --max_rows 200 --seq_len 2048 --max_new_tokens 256 \
  --bf16 True \
  --hub_repo_id "yourname/llm-kd-evals" \
  --run_name "baseline-8B-noKD"
"""

from __future__ import annotations
import os
import io
import re
import json
import time
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# BLEU-4
import sacrebleu

# Optional CodeBLEU
try:
    # pip install codebleu
    from codebleu import calc_codebleu  # type: ignore
    _HAS_CODEBLEU = True
except Exception:
    _HAS_CODEBLEU = False

# Optional GPU telemetry
try:
    import pynvml  # provided by nvidia-ml-py3 as well
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

from huggingface_hub import HfApi, HfFolder, create_repo, upload_file, upload_folder, hf_hub_url

# Project helper (uses your existing formatting for chat prompts)
from .utils import build_chat_text

# -------------------------
# Utilities
# -------------------------
def str2bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    return x.lower() in ("1", "true", "t", "yes", "y")

def get_device_dtype(bf16: Optional[bool], fp16: Optional[bool]) -> torch.dtype:
    if torch.cuda.is_available():
        auto_bf16 = torch.cuda.is_bf16_supported()
        use_bf16 = auto_bf16 if bf16 is None else bf16
        if fp16 is True:
            use_bf16 = False
        if use_bf16:
            return torch.bfloat16
        if fp16 or fp16 is None:
            return torch.float16
    return torch.float32

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# -------------------------
# GPU telemetry (safe fallbacks)
# -------------------------
def maybe_init_nvml():
    if _HAS_NVML:
        try:
            pynvml.nvmlInit()
        except Exception:
            pass

def maybe_shutdown_nvml():
    if _HAS_NVML:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

def gpu_snapshot() -> Dict[str, Any]:
    snap = {}
    if not torch.cuda.is_available():
        return snap
    n = torch.cuda.device_count()
    snap["gpu_count"] = n
    try:
        used_gib = float(torch.cuda.memory_allocated()) / (1024**3)
        total_gib = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
        snap["gpu_mem_used_GiB"] = round(used_gib, 3)
        snap["gpu_mem_total_GiB"] = round(total_gib, 1)
        snap["gpu_mem_frac"] = round(used_gib / total_gib, 3) if total_gib else None
    except Exception:
        pass

    if _HAS_NVML:
        try:
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            temps = []
            powers = []
            for h in handles:
                try:
                    temps.append(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    temps.append(None)
                try:
                    p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # watts
                    powers.append(p)
                except Exception:
                    powers.append(None)
            if temps:
                snap["gpu_temp_C_mean"] = float(sum(t for t in temps if t is not None) / max(1, sum(1 for t in temps if t is not None)))
            if powers:
                snap["gpu_power_W_mean"] = float(sum(p for p in powers if p is not None) / max(1, sum(1 for p in powers if p is not None)))
        except Exception:
            pass

    return snap

# -------------------------
# Data loading
# -------------------------
def load_jsonl(path: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows

# -------------------------
# Metrics
# -------------------------
def compute_bleu4(refs: List[str], hyps: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(hyps, [refs], smooth_method="exp")
    # Return BLEU-4 (percentage-ish scale like sacrebleu prints)
    return float(bleu.score)

def compute_codebleu(refs: List[str], hyps: List[str], lang: str = "python") -> Dict[str, float]:
    """
    Uses the official CodeBLEU implementation if available.
    Returns dictionary with overall and components. If codebleu package
    is not installed, returns empty dict.
    """
    if not _HAS_CODEBLEU:
        return {}
    # calc_codebleu returns (ngram, weighted_ngram, syntax_match, dataflow_match, overall)
    ngram, w_ngram, syntax, dataflow, overall = calc_codebleu.get_codebleu(hyps, refs, lang)
    return {
        "codebleu": float(overall * 100),                 # scale to [0,100] like BLEU
        "codebleu_ngram": float(ngram * 100),
        "codebleu_weighted_ngram": float(w_ngram * 100),
        "codebleu_syntax": float(syntax * 100),
        "codebleu_dataflow": float(dataflow * 100),
    }

# -------------------------
# Hub helpers
# -------------------------
def ensure_hub_repo(repo_id: str, token: Optional[str], private: bool = False, repo_type: str = "dataset") -> None:
    try:
        create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True, token=token)
    except Exception:
        # already exists or created; ignore
        pass

def update_readme_with_run(readme_old: str, row: Dict[str, Any]) -> str:
    """
    Ensures we have a results table in README and appends/updates a row.
    row keys: run_name, model_id, max_rows, seq_len, max_new_tokens, bleu4, codebleu, wall_time_sec, timestamp
    """
    header = (
        "# LLM KD Evaluations\n\n"
        "This repo stores evaluation runs (predictions + metrics) for our LLM compression experiments.\n\n"
        "## Results\n\n"
        "| run_name | model | rows | seq_len | max_new | BLEU-4 | CodeBLEU | wall_time_sec | timestamp |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    if not readme_old or "| run_name |" not in readme_old:
        base = header
    else:
        base = readme_old
        # If the header is missing, prepend
        if "| run_name |" not in base:
            base = header + "\n" + base

    # Remove any existing row for this run_name (simple regex)
    pattern = re.compile(rf"^\|{re.escape(row['run_name'])}\s*\|.*$", re.MULTILINE)
    base = re.sub(pattern, "", base)

    line = (
        f"| {row['run_name']} | {row['model_id']} | {row['max_rows']} | "
        f"{row['seq_len']} | {row['max_new_tokens']} | "
        f"{row.get('bleu4','-'):.2f} | {row.get('codebleu','-') if isinstance(row.get('codebleu'), (int,float)) else '-'} | "
        f"{row['wall_time_sec']:.2f} | {row['timestamp']} |\n"
    )

    # Append
    if not base.endswith("\n"):
        base += "\n"
    base += line
    return base

def push_run_to_hub(
    repo_id: str,
    token: Optional[str],
    run_name: str,
    local_run_dir: str,
    metrics: Dict[str, Any],
    table_info: Dict[str, Any],
    private: bool = False,
) -> None:
    ensure_hub_repo(repo_id, token, private=private, repo_type="dataset")

    # Upload the whole runs/<run_name> folder
    upload_folder(
        repo_id=repo_id,
        folder_path=local_run_dir,
        path_in_repo=f"runs/{run_name}",
        repo_type="dataset",
        token=token,
        commit_message=f"Add eval run {run_name}",
    )

    # Update README
    api = HfApi()
    try:
        readme = api.dataset_info(repo_id, token=token).cardData.get("content", "")
    except Exception:
        # try to fetch README.md content
        try:
            import requests
            url = hf_hub_url(repo_id, filename="README.md", repo_type="dataset")
            readme = requests.get(url).text
        except Exception:
            readme = ""

    new_readme = update_readme_with_run(readme, table_info)
    # Write temp README and upload
    tmp_readme = os.path.join(local_run_dir, "_README_tmp.md")
    with open(tmp_readme, "w", encoding="utf-8") as f:
        f.write(new_readme)

    upload_file(
        path_or_fileobj=tmp_readme,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Update README with run {run_name}",
    )

# -------------------------
# Generation helper
# -------------------------
@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool, top_p: float, temperature: float, device: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    return out[len(prompt):] if out.startswith(prompt) else out

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_id_or_path", required=True, help="Base model id or local path")
    ap.add_argument("--lora_dir", default=None, help="(Optional) merge an adapter before eval (not used here)")
    ap.add_argument("--eval_files", nargs="+", required=True, help="JSONL files with {'messages': [...], 'target': '...'}")
    ap.add_argument("--max_rows", type=int, default=200)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", type=str, default="False")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--bf16", type=str, default=None)
    ap.add_argument("--fp16", type=str, default=None)

    # Hub logging
    ap.add_argument("--hub_repo_id", required=True, help="e.g., yourname/llm-kd-evals (dataset repo)")
    ap.add_argument("--run_name", required=True, help="Row key for the run; files go to runs/<run_name>/")
    ap.add_argument("--hub_private", type=str, default="False")

    return ap.parse_args()

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
    hub_private = str2bool(args.hub_private) or False

    use_bf16 = str2bool(args.bf16)
    use_fp16 = str2bool(args.fp16)
    dtype = get_device_dtype(use_bf16, use_fp16)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_id_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading checkpoint shards...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_id_or_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    # Gather rows
    rows: List[Dict[str, Any]] = []
    for path in args.eval_files:
        rows.extend(load_jsonl(path, None))
        if len(rows) >= args.max_rows:
            rows = rows[: args.max_rows]
            break

    # Evaluate
    refs: List[str] = []
    hyps: List[str] = []
    preds_out: List[Dict[str, Any]] = []

    maybe_init_nvml()
    pre_snap = gpu_snapshot()

    t0 = time.time()
    for i, ex in enumerate(rows):
        # Your JSONL has {"messages": [...]} (and optionally "target" or "answer")
        # We’ll try several common keys for the reference:
        target = ex.get("target") or ex.get("answer") or ex.get("reference") or ""
        prompt = build_chat_text(ex["messages"])
        gen = generate_one(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=str2bool(args.do_sample) or False,
            top_p=args.top_p, temperature=args.temperature,
            device=device
        )
        preds_out.append({
            "idx": i,
            "prompt": prompt,
            "reference": target,
            "prediction": gen,
        })
        refs.append(target)
        hyps.append(gen)
    wall = time.time() - t0

    post_snap = gpu_snapshot()
    maybe_shutdown_nvml()

    # Metrics
    bleu4 = compute_bleu4(refs, hyps)
    codebleu = compute_codebleu(refs, hyps, lang="python") if _HAS_CODEBLEU else {}
    metrics: Dict[str, Any] = {
        "timestamp": now_ts(),
        "model_id": args.base_id_or_path,
        "max_rows": int(args.max_rows),
        "seq_len": int(args.seq_len),
        "max_new_tokens": int(args.max_new_tokens),
        "bleu4": float(bleu4),
        "wall_time_sec": float(wall),
        "pre_gpu_snapshot": pre_snap,
        "post_gpu_snapshot": post_snap,
        "has_codebleu": bool(_HAS_CODEBLEU),
    }
    metrics.update(codebleu)

    # Save locally (outputs/eval/<run_name>/)
    local_run_dir = os.path.join("outputs", "eval_hub", args.run_name)
    os.makedirs(local_run_dir, exist_ok=True)
    pred_path = os.path.join(local_run_dir, "predictions.jsonl")
    metrics_path = os.path.join(local_run_dir, "metrics.json")

    with open(pred_path, "w", encoding="utf-8") as f:
        for rec in preds_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== EVAL SUMMARY ===")
    print(f"samples         : {len(rows)}")
    print(f"BLEU-4 (avg)    : {metrics['bleu4']:.2f}")
    if metrics.get("has_codebleu"):
        print(f"CodeBLEU        : {metrics.get('codebleu'):.2f}")
        print(f"  • ngram             : {metrics.get('codebleu_ngram'):.2f}")
        print(f"  • weighted_ngram    : {metrics.get('codebleu_weighted_ngram'):.2f}")
        print(f"  • syntax            : {metrics.get('codebleu_syntax'):.2f}")
        print(f"  • dataflow          : {metrics.get('codebleu_dataflow'):.2f}")
    else:
        print("CodeBLEU        : (codebleu package not installed; run `pip install codebleu`)")

    print(f"wall_time_sec   : {metrics['wall_time_sec']:.2f}")
    print("pre_gpu_snapshot:", pre_snap)
    print("post_gpu_snapshot:", post_snap)
    print("predictions file:", os.path.abspath(pred_path))
    print("metrics file    :", os.path.abspath(metrics_path))

    # ---- Push to Hugging Face Hub ----
    print("\nPushing run to the Hugging Face Hub...")
    table_row = {
        "run_name": args.run_name,
        "model_id": args.base_id_or_path,
        "max_rows": int(args.max_rows),
        "seq_len": int(args.seq_len),
        "max_new_tokens": int(args.max_new_tokens),
        "bleu4": metrics["bleu4"],
        "codebleu": metrics.get("codebleu"),
        "wall_time_sec": metrics["wall_time_sec"],
        "timestamp": metrics["timestamp"],
    }
    push_run_to_hub(
        repo_id=args.hub_repo_id,
        token=token,
        run_name=args.run_name,
        local_run_dir=local_run_dir,
        metrics=metrics,
        table_info=table_row,
        private=hub_private,
    )
    print(f"✅ Done. Open https://huggingface.co/datasets/{args.hub_repo_id} to view.")
    print(f"   Your files are under runs/{args.run_name}/ and the README table was updated.")

if __name__ == "__main__":
    # friendlier matmul perf
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
