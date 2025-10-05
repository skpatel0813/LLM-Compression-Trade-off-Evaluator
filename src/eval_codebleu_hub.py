# src/eval_codebleu_hub.py
# --------------------------------------------------------------------------------------------------
# Baseline evaluation (no KD / no LoRA) with CodeBLEU + BLEU-4
# - Loads a base model (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)
# - Reads JSONL eval data where each line has: {"messages": [...]}
#   * We assume the LAST assistant message is the reference answer (gold).
#   * The prompt is all earlier messages (user/assistant) up to that reference, formatted
#     via utils.build_chat_text so it matches Llama-3 chat templates.
# - Generates predictions, computes BLEU-4 and CodeBLEU, captures GPU telemetry & wall-time.
# - Saves artifacts locally and uploads to a Hugging Face Dataset repo in a run subfolder.
#
# Usage example:
#   env HUGGINGFACE_HUB_TOKEN=hf_*** \
#   CUDA_VISIBLE_DEVICES=0 "$CONDA_PREFIX/bin/python" -u -m src.eval_codebleu_hub \
#     --base_id_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --eval_files data/codesearchnet_python_train.jsonl \
#     --max_rows 200 \
#     --seq_len 2048 \
#     --max_new_tokens 256 \
#     --bf16 True \
#     --hub_repo_id "skpatel0813/llm-kd-evals" \
#     --run_name "baseline-8B-noKD"
#
# Notes:
# - No quantization flags here: this is a *baseline* evaluator.
# - If your CodeBLEU install differs, the robust wrapper below adapts automatically.
# - If sacrebleu is unavailable, we fall back to nltk BLEU-4 (with smoothing).
# --------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import io
import re
import gc
import sys
import json
import time
import math
import argparse
import datetime as dt
from typing import Dict, List, Any, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Project helper (builds model-specific chat text)
try:
    from .utils import build_chat_text
except Exception:
    # Minimal fallback: if utils is not available, concatenate roles plainly.
    def build_chat_text(messages: List[Dict[str, str]]) -> str:
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            out.append(f"[{role.upper()}]\n{content}\n")
        out.append("[ASSISTANT]\n")
        return "\n".join(out)

# ----------------------------- Robust CodeBLEU import & wrapper -----------------------------
# We try multiple module names/APIs used across CodeBLEU releases.
try:
    from codebleu import calc_codebleu as _codebleu_api  # common path
    _HAS_CODEBLEU = True
except Exception:
    try:
        from codebleu import calc_code_bleu as _codebleu_api  # alt name in some forks
        _HAS_CODEBLEU = True
    except Exception:
        _codebleu_api = None
        _HAS_CODEBLEU = False


def compute_codebleu(refs: List[str], hyps: List[str], lang: str = "python") -> Dict[str, float]:
    """
    Returns a dict:
      codebleu, codebleu_ngram, codebleu_weighted_ngram, codebleu_syntax, codebleu_dataflow
    Falls back to {} if CodeBLEU is not available or fails.
    """
    if _codebleu_api is None:
        return {}
    try:
        # API variant A: module with .get_codebleu(...)
        if hasattr(_codebleu_api, "get_codebleu"):
            ngram, w_ngram, syntax, dataflow, overall = _codebleu_api.get_codebleu(hyps, refs, lang)
            return {
                "codebleu": float(overall),
                "codebleu_ngram": float(ngram),
                "codebleu_weighted_ngram": float(w_ngram),
                "codebleu_syntax": float(syntax),
                "codebleu_dataflow": float(dataflow),
            }

        # API variant B: callable returning a tuple or dict
        if callable(_codebleu_api):
            out = _codebleu_api(hyps, refs, lang)  # try direct call
            if isinstance(out, dict) and "codebleu" in out:
                return {
                    "codebleu": float(out["codebleu"]),
                    "codebleu_ngram": float(out.get("ngram", out.get("ngram_match", 0.0))),
                    "codebleu_weighted_ngram": float(out.get("weighted_ngram", out.get("weighted_ngram_match", 0.0))),
                    "codebleu_syntax": float(out.get("syntax", out.get("syntax_match", 0.0))),
                    "codebleu_dataflow": float(out.get("dataflow", out.get("dataflow_match", 0.0))),
                }
            if isinstance(out, (list, tuple)) and len(out) == 5:
                ngram, w_ngram, syntax, dataflow, overall = out
                return {
                    "codebleu": float(overall),
                    "codebleu_ngram": float(ngram),
                    "codebleu_weighted_ngram": float(w_ngram),
                    "codebleu_syntax": float(syntax),
                    "codebleu_dataflow": float(dataflow),
                }
    except Exception:
        pass
    return {}

# ----------------------------- BLEU-4 (sacrebleu preferred, else nltk) -----------------------------
try:
    import sacrebleu
    _HAS_SACREBLEU = True
except Exception:
    _HAS_SACREBLEU = False

try:
    # nltk BLEU-4 fallback
    import nltk

    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except Exception:
        corpus_bleu, SmoothingFunction = None, None
    _HAS_NLTK = corpus_bleu is not None
except Exception:
    _HAS_NLTK = False


def compute_bleu4(refs: List[str], hyps: List[str]) -> float:
    """
    Returns corpus BLEU-4. Uses sacrebleu if available (recommended),
    else falls back to nltk BLEU-4 with smoothing.
    """
    if _HAS_SACREBLEU:
        # sacrebleu expects: list of sys outputs, list of list of references
        # Here we have exactly one reference per hypothesis.
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        return float(bleu.score)

    if _HAS_NLTK:
        # Tokenize simply by whitespace; apply smoothing to avoid zero scores
        smoothie = SmoothingFunction().method1
        # nltk expects tokenized lists
        refs_tok = [[r.split()] for r in refs]  # list of list-of-refs
        hyps_tok = [h.split() for h in hyps]
        score = corpus_bleu(refs_tok, hyps_tok, smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25))
        return float(score * 100.0)  # to match sacrebleu's percentage scale
    return float("nan")

# ----------------------------- GPU telemetry (optional) -----------------------------
_GPU_OK = False
try:
    # Prefer modern package; if only pynvml exists, we still use it
    try:
        import nvidia_ml_py3 as _nvml  # type: ignore
        _GPU_OK = True
    except Exception:
        import pynvml as _nvml  # type: ignore
        _GPU_OK = True
except Exception:
    _GPU_OK = False


def snapshot_gpu() -> Dict[str, Any]:
    """
    Capture per-GPU telemetry: util %, mem GiB used/total, power W, temp C.
    Returns a dict with summary + per_device stats. Safe if no GPUs.
    """
    out = {"has_gpu": torch.cuda.is_available(), "num_devices": torch.cuda.device_count(), "devices": []}
    if not (out["has_gpu"] and _GPU_OK):
        return out

    try:
        _nvml.nvmlInit()
        for i in range(torch.cuda.device_count()):
            handle = _nvml.nvmlDeviceGetHandleByIndex(i)
            name = _nvml.nvmlDeviceGetName(handle).decode("utf-8") if hasattr(_nvml, "nvmlDeviceGetName") else str(
                torch.cuda.get_device_name(i)
            )
            util = _nvml.nvmlDeviceGetUtilizationRates(handle)
            mem = _nvml.nvmlDeviceGetMemoryInfo(handle)
            power = None
            temp = None
            try:
                power = _nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
            except Exception:
                pass
            try:
                temp = _nvml.nvmlDeviceGetTemperature(handle, 0)  # 0 = GPU
            except Exception:
                pass
            out["devices"].append(
                {
                    "index": i,
                    "name": name,
                    "util_percent": getattr(util, "gpu", None),
                    "mem_used_gib": round(mem.used / (1024**3), 3),
                    "mem_total_gib": round(mem.total / (1024**3), 3),
                    "power_w": power,
                    "temp_c": temp,
                }
            )
    except Exception:
        pass
    finally:
        try:
            _nvml.nvmlShutdown()
        except Exception:
            pass
    return out

# ----------------------------- HF Hub helpers -----------------------------
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file


def hf_make_run_dir(repo_id: str, run_name: str) -> str:
    stamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    safe_run = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip("-")
    return f"runs/{stamp}-{safe_run}"


def hf_upload_run(repo_id: str, local_files: Dict[str, str], run_dir: str, repo_type: str = "dataset") -> None:
    """
    Upload multiple local files into subpaths under the hub repo.
    local_files: {hub_subpath: local_path}
    """
    token = HfFolder.get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Set HUGGINGFACE_HUB_TOKEN or run `huggingface-cli login`."
        )

    # Ensure repo exists
    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, token=token)

    # Upload each file
    for hub_subpath, local_path in local_files.items():
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{run_dir}/{hub_subpath}",
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=f"Upload {hub_subpath}",
        )

    # Also drop a tiny README index for the run
    readme = f"# Evaluation Run: `{run_dir}`\n\nFiles:\n" + "\n".join(
        [f"- `{hub_subpath}`" for hub_subpath in local_files.keys()]
    ) + "\n"
    upload_file(
        path_or_fileobj=io.BytesIO(readme.encode("utf-8")),
        path_in_repo=f"{run_dir}/README.md",
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        commit_message="Add run README",
    )

# ----------------------------- Data loading -----------------------------
def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_prompt_and_ref(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    Given a chat with an 'assistant' final message (reference), create:
      - prompt: all messages up to BEFORE the last assistant
      - ref: the last assistant's content
    """
    if not messages:
        return "", ""
    # Find the last assistant as the reference
    last_ass_idx = None
    for idx in reversed(range(len(messages))):
        if messages[idx].get("role") == "assistant":
            last_ass_idx = idx
            break
    if last_ass_idx is None:
        # No assistant target; treat entire thread as prompt, empty ref.
        return build_chat_text(messages), ""

    ref = messages[last_ass_idx].get("content", "")
    prompt_msgs = messages[: last_ass_idx]  # everything before that assistant
    prompt = build_chat_text(prompt_msgs)
    return prompt, ref

# ----------------------------- Model loading -----------------------------
def load_model_and_tokenizer(
    base_id_or_path: str,
    use_bf16: bool = False,
    use_fp16: bool = False,
    seq_len: int = 2048,
):
    tok = AutoTokenizer.from_pretrained(base_id_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        base_id_or_path,
        device_map="auto",   # shard across visible GPUs automatically
        low_cpu_mem_usage=True,
        torch_dtype=dtype if dtype is not torch.float32 else None,  # prefer dtype arg (transforms will map)
    )

    # If the model supports it, enable gradient checkpointing OFF (we're evaluating)
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    return tok, model

# ----------------------------- CLI & main -----------------------------
def str2bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    return x.lower() in ("1", "true", "t", "yes", "y")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_id_or_path", required=True, type=str, help="Model id or local path.")
    ap.add_argument("--lora_dir", type=str, default=None, help="(Optional) LoRA adapters to merge for eval.")
    ap.add_argument("--eval_files", required=True, nargs="+", help="One or more JSONL eval files.")
    ap.add_argument("--max_rows", type=int, default=200, help="Max rows to evaluate total (across files).")
    ap.add_argument("--seq_len", type=int, default=2048, help="Prompt truncation length.")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Generation length.")
    ap.add_argument("--do_sample", type=str, default="False", help="Stochastic decoding.")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--bf16", type=str, default=None, help="Force bf16 (default=auto if supported).")
    ap.add_argument("--fp16", type=str, default=None, help="Force fp16.")
    # Hugging Face Hub logging
    ap.add_argument("--hub_repo_id", type=str, required=True, help="Dataset repo to upload results.")
    ap.add_argument("--run_name", type=str, default="eval-run", help="A friendly name for this run.")
    return ap.parse_args()


def main():
    args = parse_args()

    # Numerics selection
    if torch.cuda.is_available():
        auto_bf16 = torch.cuda.is_bf16_supported()
        force_bf16 = str2bool(args.bf16)
        force_fp16 = str2bool(args.fp16)

        if force_bf16 is True:
            use_bf16 = True
        elif force_bf16 is False:
            use_bf16 = False
        else:
            use_bf16 = auto_bf16

        if force_fp16 is True:
            use_fp16 = True
            use_bf16 = False
        elif force_fp16 is False:
            use_fp16 = False
        else:
            use_fp16 = not use_bf16
    else:
        use_bf16, use_fp16 = False, False

    # Resolve outputs dir (prefer ./outputs/eval)
    base_out_dir = os.path.join("outputs", "eval")
    os.makedirs(base_out_dir, exist_ok=True)

    # Load model & tokenizer
    tokenizer, model = load_model_and_tokenizer(
        args.base_id_or_path, use_bf16=use_bf16, use_fp16=use_fp16, seq_len=args.seq_len
    )
    model.eval()

    # Generation config
    do_sample = str2bool(args.do_sample) or False
    gen_kwargs = dict(
        max_new_tokens=int(args.max_new_tokens),
        do_sample=do_sample,
    )
    # Only pass top_p/temperature if sampling
    if do_sample:
        gen_kwargs.update(top_p=float(args.top_p), temperature=float(args.temperature))

    # Ingest eval rows
    rows = []
    for path in args.eval_files:
        rows.extend(read_jsonl(path))
        if len(rows) >= args.max_rows:
            rows = rows[: args.max_rows]
            break

    # Build prompts and refs
    prompts: List[str] = []
    refs: List[str] = []
    for r in rows:
        msgs = r.get("messages", [])
        prompt, ref = build_prompt_and_ref(msgs)
        # Safety fallback
        if not prompt:
            # if no structure, try to treat 'prompt'/'reference' keys if present
            prompt = r.get("prompt", "")
            ref = r.get("reference", ref)
        prompts.append(prompt)
        refs.append(ref if isinstance(ref, str) else str(ref))

    # Run generation
    hyps: List[str] = []
    t0 = time.time()
    gpu_before = snapshot_gpu()
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.seq_len,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(**enc, **gen_kwargs)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            # Try to slice out only the assistant continuation by removing the prompt prefix
            if text.startswith(prompt):
                text = text[len(prompt) :]
            hyps.append(text)
    gpu_after = snapshot_gpu()
    wall_time_sec = time.time() - t0

    # Compute metrics
    bleu4 = compute_bleu4(refs, hyps)
    codebleu_scores = compute_codebleu(refs, hyps, lang="python")
    metrics = {
        "bleu4": float(bleu4),
        **codebleu_scores,
        "num_examples": len(hyps),
        "wall_time_sec": float(wall_time_sec),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "dtype": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
        "model": args.base_id_or_path,
        "run_name": args.run_name,
        "do_sample": do_sample,
        "max_new_tokens": int(args.max_new_tokens),
        "seq_len": int(args.seq_len),
    }

    # Save local artifacts
    run_stamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    local_dir = os.path.join(base_out_dir, f"{run_stamp}-{re.sub(r'[^A-Za-z0-9._-]+','-', args.run_name)}")
    os.makedirs(local_dir, exist_ok=True)

    preds_path = os.path.join(local_dir, "predictions_eval.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for p, r, h in zip(prompts, refs, hyps):
            f.write(json.dumps({"prompt": p, "reference": r, "prediction": h}, ensure_ascii=False) + "\n")

    metrics_path = os.path.join(local_dir, "metrics_eval.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Minimal README for the local run folder
    with open(os.path.join(local_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            f"# Eval run: {args.run_name}\n\n"
            f"- Model: `{args.base_id_or_path}`\n"
            f"- Rows: {len(hyps)}\n"
            f"- BLEU-4: {metrics['bleu4']:.4f}\n"
            f"- CodeBLEU: {metrics.get('codebleu', float('nan')):.4f}\n"
            f"- Wall time (s): {metrics['wall_time_sec']:.2f}\n"
        )

    print(f"[local] wrote: {metrics_path}")
    print(f"[local] wrote: {preds_path}")

    # Upload to Hugging Face Hub (Dataset repo)
    run_dir = hf_make_run_dir(args.hub_repo_id, args.run_name)
    hub_files = {
        "predictions_eval.jsonl": preds_path,
        "metrics_eval.json": metrics_path,
    }
    # Also upload a compact metrics.txt for quick viewing in the UI
    short_metrics = (
        f"model={args.base_id_or_path}\n"
        f"rows={len(hyps)}\n"
        f"bleu4={metrics['bleu4']:.4f}\n"
        f"codebleu={metrics.get('codebleu', float('nan')):.4f}\n"
        f"wall_time_sec={metrics['wall_time_sec']:.2f}\n"
        f"dtype={metrics['dtype']}\n"
    )
    short_path = os.path.join(local_dir, "metrics.txt")
    with open(short_path, "w", encoding="utf-8") as f:
        f.write(short_metrics)
    hub_files["metrics.txt"] = short_path

    hf_upload_run(args.hub_repo_id, hub_files, run_dir, repo_type="dataset")
    print(f"[hub] uploaded to: https://huggingface.co/datasets/{args.hub_repo_id}/tree/main/{run_dir}")

    # Clean up a bit
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # keep warnings quieter
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    main()
