# src/eval_codebleu_hub.py
# --------------------------------------------------------------------------------------------------
# Evaluate a (base or LoRA-merged) model on JSONL chat data and push results to Hugging Face Hub.
#
# What you get per run (stored under outputs/eval/<run_name>/ and uploaded to the Hub):
#   - predictions_eval.jsonl      : each row has {"prompt", "ref", "hyp"}
#   - metrics_eval.json           : {"model", "rows", "bleu4", "codebleu", "...subscores", "dtype", "wall_time_sec"}
#   - gpu_trace.csv               : timestamp,gpu,util%,mem_used_MB,mem_total_MB,power_W (sampled each second)
#
# Key features:
#   - Works with Llama-3.x etc. via HF Transformers
#   - Optional LoRA merge (materialize adapters before eval)
#   - Robust CodeBLEU via a shim that uses tree_sitter_languages (no fragile per-lang wheels)
#   - Push artifacts to a Hugging Face *dataset* repo (recommended) with a simple folder layout
#
# Usage example:
#   export HUGGINGFACE_HUB_TOKEN=hf_xxx
#   CUDA_VISIBLE_DEVICES=0 "$CONDA_PREFIX/bin/python" -u -m src.eval_codebleu_hub \
#     --base_id_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --eval_files data/codesearchnet_python_train.jsonl \
#     --max_rows 200 \
#     --seq_len 2048 \
#     --max_new_tokens 256 \
#     --bf16 True \
#     --hub_repo_id "username/llm-kd-evals" \
#     --run_name "baseline-8B-noKD"
#
# Requirements in your env:
#   pip install -U transformers sacrebleu codebleu tree_sitter_languages huggingface_hub pynvml peft
# --------------------------------------------------------------------------------------------------

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# IMPORTANT: Make CodeBLEU work with tree_sitter_languages before importing calc_codebleu
# This file must exist at src/codebleu_shim.py (provided separately)
import src.codebleu_shim  # noqa: F401  (monkey-patches codebleu internals)
from codebleu import calc_codebleu
import sacrebleu

from huggingface_hub import HfApi, HfFolder, create_repo, upload_file

# Optional LoRA support (merge adapters into base weights)
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# Lightweight NVML sampler (provided separately at src/gpu_monitor.py)
from src.gpu_monitor import GPUMonitor


# ---------------------------------------
# Small helpers
# ---------------------------------------
def str2bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    return x.lower() in ("1", "true", "t", "yes", "y")


def load_jsonl(fp: str, max_rows: int) -> List[Dict[str, Any]]:
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        for _line_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if 0 < max_rows <= len(rows):
                break
    return rows


def extract_prompt_and_ref(tokenizer, ex: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    Expect JSONL lines of the shape:
      {"messages": [{"role": "user"/"assistant"/"system", "content": "..."} , ...]}

    We use tokenizer.apply_chat_template(messages, add_generation_prompt=True) to build the prompt.
    The reference 'ref' is considered the last assistant message (if present).
    """
    msgs: List[Dict[str, str]] = ex.get("messages", [])
    if not msgs:
        # Fallback: accept single-field examples
        user = ex.get("user", "") or ex.get("prompt", "")
        msgs = [{"role": "user", "content": user}]

    ref = ""
    if msgs and msgs[-1]["role"] == "assistant":
        ref = msgs[-1]["content"]
        msgs_for_prompt = msgs[:-1]
    else:
        msgs_for_prompt = msgs

    prompt = tokenizer.apply_chat_template(
        msgs_for_prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, ref, msgs_for_prompt


def maybe_merge_lora(model, lora_dir: Optional[str]) -> Any:
    """
    If a LoRA adapter dir is specified, load and try to merge it into the base model.
    If merge is unsupported, we still return a PEFT-wrapped model and proceed.
    """
    if not lora_dir:
        return model
    if not _HAS_PEFT:
        raise RuntimeError("Requested LoRA merge but `peft` is not installed.")
    model = PeftModel.from_pretrained(model, lora_dir)
    try:
        model = model.merge_and_unload()
    except Exception:
        # Keep as PEFT wrapper if not mergeable
        pass
    return model


def load_model_and_tokenizer(
    base_id_or_path: str,
    lora_dir: Optional[str],
    bf16: Optional[bool],
    fp16: Optional[bool],
    seq_len: int,
) -> Tuple[Any, Any, torch.dtype]:
    """
    Load tokenizer + model with auto device_map and selected dtype (bf16/fp16/fp32).
    Optionally merge LoRA adapters into the base model.
    """
    if torch.cuda.is_available():
        auto_bf16 = torch.cuda.is_bf16_supported()
        use_bf16 = bf16 if bf16 is not None else auto_bf16
        use_fp16 = (fp16 if fp16 is not None else not use_bf16)
    else:
        use_bf16 = use_fp16 = False

    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    tok = AutoTokenizer.from_pretrained(base_id_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = seq_len

    model = AutoModelForCausalLM.from_pretrained(
        base_id_or_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        dtype=dtype,
    )
    model = maybe_merge_lora(model, lora_dir)
    return tok, model, dtype


def generate_batch(
    tokenizer, model, prompts: List[str], max_new_tokens: int, do_sample: bool, top_p: float, temperature: float
) -> List[str]:
    """
    Tokenize prompts, generate, and return only the continuation (strip prompt prefix).
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    results: List[str] = []
    for prompt, full in zip(prompts, decoded):
        if full.startswith(prompt):
            results.append(full[len(prompt):].lstrip())
        else:
            results.append(full)
    return results


def compute_bleu4(refs: List[str], hyps: List[str]) -> float:
    """
    BLEU-4 via SacreBLEU. (one reference per hypothesis)
    """
    bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True, tokenize="intl")
    return float(bleu.score)


def compute_codebleu_scores(refs: List[str], hyps: List[str], lang: str = "python") -> Dict[str, float]:
    """
    CodeBLEU (overall + sub-scores) using patched codebleu (see codebleu_shim.py).
    Returns dict with keys:
      - codebleu
      - codebleu_ngram
      - codebleu_weighted_ngram
      - codebleu_syntax
      - codebleu_dataflow
    """
    out = calc_codebleu(refs, hyps, lang=lang)
    return {
        "codebleu": float(out["codebleu"]),
        "codebleu_ngram": float(out["ngram_match"]),
        "codebleu_weighted_ngram": float(out["weighted_ngram_match"]),
        "codebleu_syntax": float(out["syntax_match"]),
        "codebleu_dataflow": float(out["dataflow_match"]),
    }


def ensure_hf_token(token: Optional[str]) -> str:
    tok = token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or HfFolder.get_token()
    if not tok:
        raise RuntimeError("Missing Hugging Face token. Set HUGGINGFACE_HUB_TOKEN or use --hf_token.")
    return tok


def hf_upload_dir(repo_id: str, run_dir: str, local_dir: str, token: str):
    """
    Upload all files under `local_dir` into the dataset repo `repo_id` at path `run_dir/..`.
    If the dataset repo does not exist, create it (if it's actually a model repo, uploads still proceed).
    """
    api = HfApi()
    try:
        create_repo(repo_id, token=token, repo_type="dataset", exist_ok=True)
    except Exception:
        # Repo might already exist as a model repo, still fine to upload
        pass

    for root, _, files in os.walk(local_dir):
        for fn in files:
            local_path = os.path.join(root, fn)
            repo_path = f"{run_dir}/{os.path.relpath(local_path, local_dir)}"
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                token=token,
                repo_type="dataset",
            )


# ---------------------------------------
# CLI
# ---------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_id_or_path", required=True, help="HF model id or local path")
    ap.add_argument("--lora_dir", default=None, help="Optional LoRA adapter dir to merge before eval")

    ap.add_argument("--eval_files", required=True, nargs="+",
                    help="One or more JSONL files (each line has 'messages': [{role, content}, ...])")
    ap.add_argument("--max_rows", type=int, default=200, help="Evaluate on at most this many rows")

    ap.add_argument("--seq_len", type=int, default=2048, help="Tokenizer max length for prompts")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Tokens to generate")
    ap.add_argument("--do_sample", type=str, default="False")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)

    ap.add_argument("--bf16", type=str, default=None, help="True/False to force bfloat16 (default: auto)")
    ap.add_argument("--fp16", type=str, default=None, help="True/False to force float16 (default: opposite of bf16)")

    # HF Hub logging
    ap.add_argument("--hub_repo_id", required=True, help='e.g. "username/llm-kd-evals" (dataset repo recommended)')
    ap.add_argument("--run_name", default=None, help="Subfolder name; if omitted, use model+timestamp")
    ap.add_argument("--hf_token", default=None, help="Token override; else uses env or cached login")

    # GPU monitor
    ap.add_argument("--gpu_poll_sec", type=float, default=1.0, help="NVML polling interval in seconds")

    return ap.parse_args()


# ---------------------------------------
# Main
# ---------------------------------------
def main():
    args = parse_args()

    # Parse booleans
    bf16 = str2bool(args.bf16)
    fp16 = str2bool(args.fp16)
    do_sample = str2bool(args.do_sample) or False

    # Output locations
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{os.path.basename(args.base_id_or_path)}-{timestamp}"
    out_dir = os.path.join("outputs", "eval", f"{run_name}")
    os.makedirs(out_dir, exist_ok=True)

    preds_path = os.path.join(out_dir, "predictions_eval.jsonl")
    metrics_path = os.path.join(out_dir, "metrics_eval.json")
    gpu_csv = os.path.join(out_dir, "gpu_trace.csv")

    # Start GPU monitor
    mon = GPUMonitor(gpu_csv, interval_sec=float(args.gpu_poll_sec))
    mon.start()

    # Time the full evaluation
    t0 = time.time()

    # Load model/tokenizer (optional LoRA merge)
    tokenizer, model, dtype = load_model_and_tokenizer(
        args.base_id_or_path, args.lora_dir, bf16, fp16, args.seq_len
    )

    # Read evaluation rows (capped)
    rows: List[Dict[str, Any]] = []
    for fp in args.eval_files:
        take = max(0, args.max_rows - len(rows))
        if take == 0:
            break
        rows.extend(load_jsonl(fp, max_rows=take))

    prompts: List[str] = []
    refs: List[str] = []
    for ex in rows:
        p, r, _ = extract_prompt_and_ref(tokenizer, ex)
        prompts.append(p)
        refs.append(r)

    # Generate predictions in small batches
    B = 8
    hyps: List[str] = []
    for i in range(0, len(prompts), B):
        batch_prompts = prompts[i : i + B]
        out = generate_batch(
            tokenizer,
            model,
            batch_prompts,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        hyps.extend(out)

    # Save predictions JSONL
    with open(preds_path, "w", encoding="utf-8") as f:
        for p, r, h in zip(prompts, refs, hyps):
            f.write(json.dumps({"prompt": p, "ref": r, "hyp": h}, ensure_ascii=False) + "\n")

    # Compute metrics
    bleu4 = compute_bleu4(refs, hyps)
    codebleu_scores = compute_codebleu_scores(refs, hyps, lang="python")
    wall_time = time.time() - t0

    # Pack and save metrics
    metrics = {
        "model": args.base_id_or_path,
        "rows": len(hyps),
        "bleu4": round(bleu4, 4),
        "dtype": "bf16" if dtype == torch.bfloat16 else ("fp16" if dtype == torch.float16 else "fp32"),
        "wall_time_sec": round(wall_time, 2),
        **{k: round(v, 4) for k, v in codebleu_scores.items()},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Stop GPU monitor
    mon.stop()

    # Push artifacts to the Hub
    token = ensure_hf_token(args.hf_token)
    run_dir_in_repo = f"runs/{run_name}"
    hf_upload_dir(args.hub_repo_id, run_dir_in_repo, out_dir, token)

    # Console summary
    print("\n=== EVAL DONE ===")
    print("predictions :", preds_path)
    print("metrics     :", metrics_path)
    print("gpu trace   :", gpu_csv)
    print(f"pushed to   : hf://{args.hub_repo_id}/{run_dir_in_repo}/")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    # Small perf nicety (safe no-op if unsupported)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
