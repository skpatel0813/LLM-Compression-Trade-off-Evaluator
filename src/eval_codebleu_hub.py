# src/eval_codebleu_hub.py
# --------------------------------------------------------------------------------------------------
# Evaluate a (base or LoRA-merged) model on JSONL chat data and push results to Hugging Face Hub.
#
# What you get per run (locally under outputs/eval/<run_name>/ and uploaded to the Hub):
#   - predictions_eval.jsonl      : each row has {"prompt", "ref", "hyp"}
#   - metrics_eval.json           : {"model","rows","bleu4","codebleu", sub-scores, "dtype","wall_time_sec"}
#   - gpu_trace.csv               : timestamp,gpu,util_pct,mem_used_MiB,mem_total_MiB,power_W (1 Hz)
#   - README.md                   : nice summary of metrics for the Hub UI
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
# Requirements:
#   pip install -U transformers sacrebleu codebleu tree_sitter_languages huggingface_hub pynvml peft
# --------------------------------------------------------------------------------------------------

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import threading
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CodeBLEU shim (must come BEFORE importing codebleu.calc_codebleu) -----------------------------
try:
    import src.codebleu_shim  # noqa: F401  (monkey-patches codebleu internals to use tree_sitter_languages)
except Exception as e:
    print(f"[WARN] codebleu_shim import failed: {e}\n"
          f"       CodeBLEU will try to run without the shim.", file=sys.stderr)

# CodeBLEU + BLEU
try:
    from codebleu import calc_codebleu as _codebleu_fn
    _HAS_CODEBLEU = True
except Exception as e:
    print(f"[WARN] CodeBLEU import failed: {e}\n"
          f"       CodeBLEU metrics will be set to NaN.", file=sys.stderr)
    _HAS_CODEBLEU = False

import sacrebleu

# HF Hub
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file

# LoRA (optional)
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# === Lightweight GPU monitor (pynvml if available, else nvidia-smi subprocess) ====================
class GPUMonitor:
    def __init__(self, csv_path: str, interval_sec: float = 1.0):
        self.csv_path = csv_path
        self.interval = float(interval_sec)
        self._stop = threading.Event()
        self._th = None
        self._use_nvml = False
        try:
            import pynvml  # noqa
            self._use_nvml = True
        except Exception:
            self._use_nvml = False

    def start(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("timestamp,gpu,util_pct,mem_used_MiB,mem_total_MiB,power_W\n")
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=5)

    def _loop(self):
        if self._use_nvml:
            self._loop_nvml()
        else:
            self._loop_nvidia_smi()

    def _loop_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            while not self._stop.is_set():
                ts = datetime.utcnow().isoformat()
                lines = []
                for i in range(n):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    power = 0.0
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    except Exception:
                        power = 0.0
                    lines.append(f"{ts},{i},{util},{mem.used//(1024*1024)},{mem.total//(1024*1024)},{power}\n")
                with open(self.csv_path, "a", encoding="utf-8") as f:
                    f.writelines(lines)
                time.sleep(self.interval)
        except Exception as e:
            print(f"[WARN] NVML monitor failed, falling back to nvidia-smi: {e}", file=sys.stderr)
            self._loop_nvidia_smi()

    def _loop_nvidia_smi(self):
        q = ["index,utilization.gpu,memory.used,memory.total,power.draw"]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu="+q[0], "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL,
                ).decode("utf-8").strip().splitlines()
                ts = datetime.utcnow().isoformat()
                with open(self.csv_path, "a", encoding="utf-8") as f:
                    for row in out:
                        parts = [p.strip() for p in row.split(",")]
                        if len(parts) != 5:
                            continue
                        f.write(f"{ts},{parts[0]},{parts[1]},{parts[2]},{parts[3]},{parts[4]}\n")
            except Exception:
                pass
            time.sleep(self.interval)


# === Helpers ======================================================================================
def str2bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    return x.lower() in ("1", "true", "t", "yes", "y")


def load_jsonl(fp: str, max_rows: int) -> List[Dict[str, Any]]:
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
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
    Expects:
      {"messages": [{"role": "user"/"assistant"/"system", "content": "..."} , ...]}
    Reference is the last assistant message (if present); the prompt is built from the rest.
    """
    msgs: List[Dict[str, str]] = ex.get("messages", [])
    if not msgs:
        # fallback single-field records
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


def maybe_merge_lora(model, lora_dir: Optional[str]) -> Any:
    if not lora_dir:
        return model
    if not _HAS_PEFT:
        raise RuntimeError("Requested LoRA merge but `peft` is not installed.")
    model = PeftModel.from_pretrained(model, lora_dir)
    try:
        model = model.merge_and_unload()
    except Exception:
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
    Sets left-padding (decoder-only) and merges LoRA if provided.
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
    tok.padding_side = "left"          # << important for decoder-only generation
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

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, top_p=top_p, temperature=temperature))
    else:
        gen_kwargs.update(dict(do_sample=False))

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    results: List[str] = []
    for prompt, full in zip(prompts, decoded):
        if full.startswith(prompt):
            results.append(full[len(prompt):].lstrip())
        else:
            results.append(full)
    return results


def compute_bleu4(refs: List[str], hyps: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True, tokenize="intl")
    return float(bleu.score)


def compute_codebleu_scores(refs: List[str], hyps: List[str], lang: str = "python") -> Dict[str, float]:
    if not _HAS_CODEBLEU:
        return {
            "codebleu": float("nan"),
            "codebleu_ngram": float("nan"),
            "codebleu_weighted_ngram": float("nan"),
            "codebleu_syntax": float("nan"),
            "codebleu_dataflow": float("nan"),
        }
    try:
        out = _codebleu_fn(refs, hyps, lang=lang)
        # PyPI codebleu returns keys: codebleu, ngram_match, weighted_ngram_match, syntax_match, dataflow_match
        return {
            "codebleu": float(out["codebleu"]),
            "codebleu_ngram": float(out["ngram_match"]),
            "codebleu_weighted_ngram": float(out["weighted_ngram_match"]),
            "codebleu_syntax": float(out["syntax_match"]),
            "codebleu_dataflow": float(out["dataflow_match"]),
        }
    except Exception as e:
        print(f"[WARN] CodeBLEU computation failed: {e}", file=sys.stderr)
        return {
            "codebleu": float("nan"),
            "codebleu_ngram": float("nan"),
            "codebleu_weighted_ngram": float("nan"),
            "codebleu_syntax": float("nan"),
            "codebleu_dataflow": float("nan"),
        }


def ensure_hf_token(token: Optional[str]) -> str:
    tok = token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or HfFolder.get_token()
    if not tok:
        raise RuntimeError("Missing Hugging Face token. Set HUGGINGFACE_HUB_TOKEN or use --hf_token.")
    return tok


def hf_upload_dir(repo_id: str, run_dir: str, local_dir: str, token: str):
    """
    Upload all files under `local_dir` into the dataset repo `repo_id` at path `run_dir/..`.
    If the dataset repo does not exist, create it (if it's a model repo, uploads still proceed).
    """
    api = HfApi()
    try:
        create_repo(repo_id, token=token, repo_type="dataset", exist_ok=True)
    except Exception:
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


def write_readme(path: str, metrics: Dict[str, Any], run_name: str):
    lines = [
        f"# {run_name}",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for k in ["bleu4", "codebleu", "codebleu_ngram", "codebleu_weighted_ngram",
              "codebleu_syntax", "codebleu_dataflow", "rows", "dtype", "wall_time_sec"]:
        if k in metrics:
            lines.append(f"| {k} | {metrics[k]} |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# === CLI ==========================================================================================
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
    ap.add_argument("--gpu_poll_sec", type=float, default=1.0, help="NVML/nvidia-smi polling interval in seconds")

    return ap.parse_args()


# === Main =========================================================================================
def main():
    # small perf nicety
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

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
    readme_path = os.path.join(out_dir, "README.md")

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
    B = int(os.environ.get("EVAL_BATCH_SIZE", "8"))
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
        "lora_dir": args.lora_dir or "",
        "rows": len(hyps),
        "bleu4": round(bleu4, 4),
        "dtype": "bf16" if dtype == torch.bfloat16 else ("fp16" if dtype == torch.float16 else "fp32"),
        "wall_time_sec": round(wall_time, 2),
        **{k: (None if isinstance(v, float) and (v != v) else round(v, 4)) for k, v in codebleu_scores.items()},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Stop GPU monitor
    mon.stop()

    # README for the Hub
    write_readme(readme_path, metrics, run_name)

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
    main()
