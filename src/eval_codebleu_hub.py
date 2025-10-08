# src/eval_codebleu_hub.py
# --------------------------------------------------------------------------------------------------
# Evaluate a (base or LoRA-merged) model on JSONL chat data and push results to the Hugging Face Hub.
#
# Artifacts per run (saved under outputs/eval/<run_name>/ and uploaded to the Hub):
#   - predictions_eval.jsonl : each line {"prompt", "ref", "hyp"}
#   - metrics_eval.json      : {"model","rows","bleu4","codebleu", sub-scores, "dtype","wall_time_sec"}
#   - gpu_trace.csv          : timestamp,gpu,util%,mem_used_MiB,mem_total_MiB,power_W (sampled each second)
#
# Features:
#   - Robust loader for decoder-only LLMs (e.g., Llama 3.x) with dtype auto-selection
#   - Optional LoRA merge (PEFT) before eval
#   - Left-padding for decoder-only generation (prevents HF warnings & subtle issues)
#   - CodeBLEU:
#       * First tries official `codebleu` (if correctly installed)
#       * Otherwise falls back to `src.codebleu_compat` (no external CodeXGLUE)
#   - Push to a Hugging Face *dataset* repo under `runs/<run_name>/...`
#
# Usage:
#   export HUGGINGFACE_HUB_TOKEN=hf_xxx
#   CUDA_VISIBLE_DEVICES=0 "$CONDA_PREFIX/bin/python" -u -m src.eval_codebleu_hub \
#     --base_id_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --eval_files data/codesearchnet_python_train.jsonl \
#     --max_rows 200 \
#     --seq_len 2048 \
#     --max_new_tokens 256 \
#     --bf16 True \
#     --do_sample False \
#     --hub_repo_id "skpatel0813/llm-kd-evals" \
#     --run_name "baseline-8B-noKD"
#
# Minimal deps:
#   pip install -U transformers sacrebleu huggingface_hub pynvml  # (pynvml only for nicer GPU csv)
#   # For fallback CodeBLEU (no CodeXGLUE):
#   pip install -U tree-sitter tree-sitter-python
#
# Notes:
#   - If official `codebleu` is installed and compatible, it will be used.
#   - If not, fallback module `src/codebleu_compat.py` (provided separately) is used automatically.
# --------------------------------------------------------------------------------------------------

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sacrebleu

from huggingface_hub import HfApi, create_repo, upload_file

# ---------------------------
# Optional LoRA merge support
# ---------------------------
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# ---------------------------
# CodeBLEU: try official first
# ---------------------------
_USE_OFFICIAL_CODEBLEU = True
try:
    from codebleu import calc_codebleu as _official_calc_codebleu
except Exception:
    _USE_OFFICIAL_CODEBLEU = False

# Fallback (dependency-light) CodeBLEU (you must have src/codebleu_compat.py in your repo)
try:
    from src.codebleu_compat import calc_codebleu as _compat_calc_codebleu  # noqa: F401
except Exception:
    _compat_calc_codebleu = None  # type: ignore


# ---------------------------
# Simple GPU monitor (nvidia-smi)
# ---------------------------
class GPUMonitor:
    """
    Samples GPU utilization/memory/power via nvidia-smi every `interval` seconds
    and writes to a CSV. If nvidia-smi is unavailable, it quietly does nothing.
    """
    def __init__(self, out_csv: str, interval: float = 1.0):
        self.out_csv = out_csv
        self.interval = interval
        self._proc = None

    def start(self):
        try:
            import subprocess
            os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
            with open(self.out_csv, "w", encoding="utf-8") as f:
                f.write("timestamp,gpu,util_pct,mem_used_MiB,mem_total_MiB,power_W\n")
            cmd = (
                "bash -lc 'while :; do "
                "d=$(date -Is); "
                "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw "
                "--format=csv,noheader,nounits | "
                "awk -v d=\"$d\" -F, '{printf \"%s,%s,%s,%s,%s,%s\\n\",d,$1,$2,$3,$4,$5}'; "
                f"sleep {self.interval}; done'"
            )
            self._proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
        except Exception:
            self._proc = None  # degrade quietly

    def stop(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass


# ---------------------------
# Helpers
# ---------------------------
def str2bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    return x.lower() in ("1", "true", "t", "yes", "y")


def load_jsonl(fp: str, max_rows: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            rows.append(ex)
            if 0 < max_rows <= len(rows):
                break
    return rows


def extract_prompt_and_ref(tokenizer, ex: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    Expect lines like:
      {"messages": [{"role":"user"/"assistant"/"system","content":"..."} , ...]}
    Reference = last assistant message (if provided).
    Prompt is built via tokenizer.apply_chat_template(..., add_generation_prompt=True).
    """
    msgs: List[Dict[str, str]] = ex.get("messages", [])
    if not msgs:
        # Fallback for single-field examples
        user = ex.get("user", "") or ex.get("prompt", "")
        msgs = [{"role": "user", "content": user}]

    ref = ""
    if msgs and msgs[-1].get("role") == "assistant":
        ref = msgs[-1].get("content", "")
        msgs_for_prompt = msgs[:-1]
    else:
        msgs_for_prompt = msgs

    prompt = tokenizer.apply_chat_template(
        msgs_for_prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, ref, msgs_for_prompt


def maybe_merge_lora(model, lora_dir: Optional[str]):
    if not lora_dir:
        return model
    if not _HAS_PEFT:
        raise RuntimeError("Requested LoRA merge but `peft` is not installed.")
    model = PeftModel.from_pretrained(model, lora_dir)
    try:
        model = model.merge_and_unload()
    except Exception:
        # If merge isn't supported, keep PEFT wrapper and continue
        pass
    return model


def load_model_and_tokenizer(
    base_id_or_path: str,
    lora_dir: Optional[str],
    bf16: Optional[bool],
    fp16: Optional[bool],
    seq_len: int,
):
    """
    Auto-select dtype, set left-padding for decoder-only models, and load.
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
    # Critical for decoder-only generation correctness with batched prompts:
    tok.padding_side = "left"
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
    tokenizer,
    model,
    prompts: List[str],
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
) -> List[str]:
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

    stripped: List[str] = []
    for prompt, full in zip(prompts, decoded):
        stripped.append(full[len(prompt):].lstrip() if full.startswith(prompt) else full)
    return stripped


def compute_bleu4(refs: List[str], hyps: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True, tokenize="intl")
    return float(bleu.score)


def compute_codebleu(refs: List[str], hyps: List[str]) -> Dict[str, Optional[float]]:
    """
    Returns normalized keys:
      codebleu, codebleu_ngram, codebleu_weighted_ngram, codebleu_syntax, codebleu_dataflow
    """
    # Try official package first
    if _USE_OFFICIAL_CODEBLEU:
        try:
            out = _official_calc_codebleu(refs, hyps, lang="python")
            return {
                "codebleu": float(out.get("codebleu", out.get("code_bleu", float("nan")))),
                "codebleu_ngram": float(out.get("ngram_match", out.get("ngram_match_score", float("nan")))),
                "codebleu_weighted_ngram": float(out.get("weighted_ngram_match", out.get("weighted_ngram_match_score", float("nan")))),
                "codebleu_syntax": float(out.get("syntax_match", out.get("syntax_match_score", float("nan")))),
                "codebleu_dataflow": float(out.get("dataflow_match", out.get("dataflow_match_score", float("nan")))),
            }
        except Exception as e:
            print(f"[WARN] Official CodeBLEU failed: {e}\n       Falling back to codebleu_compat.", file=sys.stderr)

    # Fallback: local compat (no fragile CodeXGLUE deps)
    if _compat_calc_codebleu is not None:
        try:
            out = _compat_calc_codebleu(refs, hyps, lang="python")
            return {
                "codebleu": float(out["codebleu"]),
                "codebleu_ngram": float(out["ngram_match"]),
                "codebleu_weighted_ngram": float(out["weighted_ngram_match"]),
                "codebleu_syntax": float(out["syntax_match"]),
                "codebleu_dataflow": float(out["dataflow_match"]),  # likely 0.0 in compat
            }
        except Exception as e:
            print(f"[WARN] codebleu_compat failed: {e}\n       Setting CodeBLEU metrics to NaN.", file=sys.stderr)

    # If both paths fail, return NaNs
    return {
        "codebleu": float("nan"),
        "codebleu_ngram": float("nan"),
        "codebleu_weighted_ngram": float("nan"),
        "codebleu_syntax": float("nan"),
        "codebleu_dataflow": float("nan"),
    }


def ensure_hf_token(token: Optional[str]) -> str:
    from huggingface_hub import HfFolder
    tok = token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or HfFolder.get_token()
    if not tok:
        raise RuntimeError("Missing Hugging Face token. Set HUGGINGFACE_HUB_TOKEN or pass --hf_token.")
    return tok


def hf_upload_dir(repo_id: str, run_dir: str, local_dir: str, token: str):
    api = HfApi()
    try:
        create_repo(repo_id, token=token, repo_type="dataset", exist_ok=True)
    except Exception:
        # If it already exists (dataset or model repo), keep going.
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


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_id_or_path", required=True, help="HF model id or local path")
    ap.add_argument("--lora_dir", default=None, help="Optional LoRA adapter dir to merge before eval")

    ap.add_argument("--eval_files", required=True, nargs="+",
                    help="One or more JSONL files (each line has 'messages': [{role, content}, ...])")
    ap.add_argument("--max_rows", type=int, default=200, help="Evaluate at most this many rows total")

    ap.add_argument("--seq_len", type=int, default=2048, help="Tokenizer max length for prompts")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Tokens to generate")
    ap.add_argument("--do_sample", type=str, default="False")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)

    ap.add_argument("--bf16", type=str, default=None, help="True/False; default auto if supported")
    ap.add_argument("--fp16", type=str, default=None, help="True/False; default opposite of bf16")

    # HF Hub
    ap.add_argument("--hub_repo_id", required=True, help='e.g. "username/llm-kd-evals" (dataset repo recommended)')
    ap.add_argument("--run_name", default=None, help="Subfolder name. If omitted, use model+timestamp")
    ap.add_argument("--hf_token", default=None, help="Token override; else env/cached login")

    # GPU monitor
    ap.add_argument("--gpu_poll_sec", type=float, default=1.0, help="nvidia-smi polling interval seconds")

    return ap.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    # Parse booleans
    bf16 = str2bool(args.bf16)
    fp16 = str2bool(args.fp16)
    do_sample = str2bool(args.do_sample) or False

    # Output paths
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{os.path.basename(args.base_id_or_path)}-{timestamp}"
    out_dir = os.path.join("outputs", "eval", run_name)
    os.makedirs(out_dir, exist_ok=True)

    preds_path = os.path.join(out_dir, "predictions_eval.jsonl")
    metrics_path = os.path.join(out_dir, "metrics_eval.json")
    gpu_csv = os.path.join(out_dir, "gpu_trace.csv")

    # Start GPU monitor
    mon = GPUMonitor(gpu_csv, interval=args.gpu_poll_sec)
    mon.start()

    t0 = time.time()

    # Load model/tokenizer (+ optional LoRA merge)
    tokenizer, model, dtype = load_model_and_tokenizer(
        args.base_id_or_path, args.lora_dir, bf16, fp16, args.seq_len
    )

    # Read & cap rows
    rows: List[Dict[str, Any]] = []
    remaining = args.max_rows
    for fp in args.eval_files:
        if remaining <= 0:
            break
        chunk = load_jsonl(fp, max_rows=remaining)
        rows.extend(chunk)
        remaining -= len(chunk)

    # Build prompts/refs
    prompts: List[str] = []
    refs: List[str] = []
    for ex in rows:
        p, r, _ = extract_prompt_and_ref(tokenizer, ex)
        prompts.append(p)
        refs.append(r)

    # Generate in batches
    hyps: List[str] = []
    B = 8
    for i in range(0, len(prompts), B):
        batch_prompts = prompts[i : i + B]
        batch_out = generate_batch(
            tokenizer, model, batch_prompts,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        hyps.extend(batch_out)

    # Save predictions
    with open(preds_path, "w", encoding="utf-8") as f:
        for p, r, h in zip(prompts, refs, hyps):
            f.write(json.dumps({"prompt": p, "ref": r, "hyp": h}, ensure_ascii=False) + "\n")

    # Metrics
    bleu4 = compute_bleu4(refs, hyps)
    cb = compute_codebleu(refs, hyps)
    wall = time.time() - t0

    metrics = {
        "model": args.base_id_or_path,
        "lora_dir": args.lora_dir or "",
        "rows": len(hyps),
        "bleu4": round(float(bleu4), 4),
        "dtype": "bf16" if dtype == torch.bfloat16 else ("fp16" if dtype == torch.float16 else "fp32"),
        "wall_time_sec": round(float(wall), 2),
        # CodeBLEU (handle NaNs safely)
        "codebleu": None if math.isnan(cb["codebleu"]) else float(cb["codebleu"]),
        "codebleu_ngram": None if math.isnan(cb["codebleu_ngram"]) else float(cb["codebleu_ngram"]),
        "codebleu_weighted_ngram": None if math.isnan(cb["codebleu_weighted_ngram"]) else float(cb["codebleu_weighted_ngram"]),
        "codebleu_syntax": None if math.isnan(cb["codebleu_syntax"]) else float(cb["codebleu_syntax"]),
        "codebleu_dataflow": None if math.isnan(cb["codebleu_dataflow"]) else float(cb["codebleu_dataflow"]),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Stop GPU monitor
    mon.stop()

    # Push artifacts to Hub
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
    # Small perf nicety
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
