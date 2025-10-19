# src/eval_codebleu_hub.py
# --------------------------------------------------------------------------------------------------
# FIXED: Better error handling and debugging for CodeBLEU syntax scoring
# --------------------------------------------------------------------------------------------------

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import threading
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------------------------
# Import the CodeBLEU shim FIRST (if present)
# --------------------------------------------------------------------------------------------------
_SHIM = None
try:
    from src import codebleu_shim as _SHIM
except Exception:
    try:
        import codebleu_shim as _SHIM
    except Exception:
        _SHIM = None

if _SHIM is not None:
    print("[eval] CodeBLEU shim loaded", file=sys.stderr, flush=True)

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
# CodeBLEU options (official first, then fallback)
# ---------------------------
_USE_OFFICIAL_CODEBLEU = True
_official_calc_codebleu = None
try:
    from codebleu import calc_codebleu as _official_calc_codebleu
    print("[eval] Official CodeBLEU imported successfully", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[eval] Official CodeBLEU not available: {e}", file=sys.stderr, flush=True)
    _USE_OFFICIAL_CODEBLEU = False

_compat_calc_codebleu = None
try:
    from src.codebleu_compat import calc_codebleu as _compat_calc_codebleu
    print("[eval] Compat CodeBLEU imported successfully", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[eval] Compat CodeBLEU not available: {e}", file=sys.stderr, flush=True)


# ---------------------------
# Safe GPU monitor (Python thread)
# ---------------------------
class GPUMonitor:
    """GPU monitoring with pynvml or nvidia-smi fallback"""
    def __init__(self, out_csv: str, interval_sec: float = 1.0):
        self.out_csv = out_csv
        self.interval = float(interval_sec)
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)

    def _loop_pynvml(self):
        import datetime as _dt
        import pynvml as _nv
        with open(self.out_csv, "w", encoding="utf-8") as f:
            f.write("timestamp,gpu,util_pct,mem_used_MiB,mem_total_MiB,power_W\n")
            try:
                _nv.nvmlInit()
                n = _nv.nvmlDeviceGetCount()
                while not self._stop.is_set():
                    ts = _dt.datetime.now().isoformat()
                    for i in range(n):
                        h = _nv.nvmlDeviceGetHandleByIndex(i)
                        util = _nv.nvmlDeviceGetUtilizationRates(h)
                        mem = _nv.nvmlDeviceGetMemoryInfo(h)
                        try:
                            pwr = _nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                        except Exception:
                            pwr = float("nan")
                        f.write(f"{ts},{i},{util.gpu},{mem.used/1048576:.0f},{mem.total/1048576:.0f},{pwr:.1f}\n")
                    f.flush()
                    self._stop.wait(self.interval)
            except Exception:
                pass
            finally:
                try:
                    _nv.nvmlShutdown()
                except Exception:
                    pass

    def _loop_smi(self):
        with open(self.out_csv, "w", encoding="utf-8") as f:
            f.write("timestamp,gpu,util_pct,mem_used_MiB,mem_total_MiB,power_W\n")
            while not self._stop.is_set():
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
                         "--format=csv,noheader,nounits"],
                        text=True,
                    )
                    ts = datetime.now().isoformat()
                    for line in out.strip().splitlines():
                        cols = [c.strip() for c in line.split(",")]
                        if len(cols) == 5:
                            f.write(f"{ts},{cols[0]},{cols[1]},{cols[2]},{cols[3]},{cols[4]}\n")
                    f.flush()
                except Exception:
                    break
                self._stop.wait(self.interval)

    def start(self):
        try:
            import pynvml
            target = self._loop_pynvml
        except Exception:
            target = self._loop_smi
        self._thr = threading.Thread(target=target, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=2.0)


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
    """Extract prompt and reference from chat message format"""
    msgs: List[Dict[str, str]] = ex.get("messages", [])
    if not msgs:
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
        pass
    return model


def load_model_and_tokenizer(
    base_id_or_path: str,
    lora_dir: Optional[str],
    bf16: Optional[bool],
    fp16: Optional[bool],
    seq_len: int,
):
    """Load model and tokenizer with auto dtype selection"""
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
    Compute CodeBLEU with detailed error reporting.
    Returns normalized keys with proper debugging output.
    """
    print(f"\n[CodeBLEU] Computing for {len(refs)} examples", file=sys.stderr, flush=True)
    
    # Show sample data for debugging
    if refs and hyps:
        print(f"[CodeBLEU] Sample ref (first 100 chars): {refs[0][:100]!r}", file=sys.stderr, flush=True)
        print(f"[CodeBLEU] Sample hyp (first 100 chars): {hyps[0][:100]!r}", file=sys.stderr, flush=True)
    
    # Try official package first
    if _USE_OFFICIAL_CODEBLEU and _official_calc_codebleu is not None:
        try:
            print("[CodeBLEU] Trying official codebleu package...", file=sys.stderr, flush=True)
            out = _official_calc_codebleu(refs, hyps, lang="python")
            print(f"[CodeBLEU] Official result: {out}", file=sys.stderr, flush=True)
            
            result = {
                "codebleu": float(out.get("codebleu", out.get("code_bleu", float("nan")))),
                "codebleu_ngram": float(out.get("ngram_match_score", out.get("ngram_match", float("nan")))),
                "codebleu_weighted_ngram": float(out.get("weighted_ngram_match_score", out.get("weighted_ngram_match", float("nan")))),
                "codebleu_syntax": float(out.get("syntax_match_score", out.get("syntax_match", float("nan")))),
                "codebleu_dataflow": float(out.get("dataflow_match_score", out.get("dataflow_match", float("nan")))),
            }
            
            # Check for zero syntax score
            if result["codebleu_syntax"] == 0.0:
                print("[CodeBLEU] WARNING: Official package returned 0.0 for syntax_match!", file=sys.stderr, flush=True)
                print("[CodeBLEU]          This may indicate tree-sitter parsing issues.", file=sys.stderr, flush=True)
            
            return result
            
        except Exception as e:
            print(f"[CodeBLEU] Official package FAILED: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            print("[CodeBLEU] Falling back to codebleu_compat...", file=sys.stderr, flush=True)

    # Fallback: local compat
    if _compat_calc_codebleu is not None:
        try:
            print("[CodeBLEU] Using codebleu_compat fallback...", file=sys.stderr, flush=True)
            out = _compat_calc_codebleu(refs, hyps, lang="python")
            print(f"[CodeBLEU] Compat result: {out}", file=sys.stderr, flush=True)
            
            result = {
                "codebleu": float(out["codebleu"]),
                "codebleu_ngram": float(out["ngram_match"]),
                "codebleu_weighted_ngram": float(out["weighted_ngram_match"]),
                "codebleu_syntax": float(out["syntax_match"]),
                "codebleu_dataflow": float(out["dataflow_match"]),
            }
            
            # Check for zero syntax score
            if result["codebleu_syntax"] == 0.0:
                print("[CodeBLEU] WARNING: Compat returned 0.0 for syntax_match!", file=sys.stderr, flush=True)
                print("[CodeBLEU]          Check tree-sitter installation and code validity.", file=sys.stderr, flush=True)
            
            return result
            
        except Exception as e:
            print(f"[CodeBLEU] Compat FAILED: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)

    # Both failed
    print("[CodeBLEU] ERROR: Both official and compat failed! Returning NaNs.", file=sys.stderr, flush=True)
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

    ap.add_argument("--hub_repo_id", required=True, help='e.g. "username/llm-kd-evals"')
    ap.add_argument("--run_name", default=None, help="Subfolder name. If omitted, use model+timestamp")
    ap.add_argument("--hf_token", default=None, help="Token override; else env/cached login")

    ap.add_argument("--gpu_poll_sec", type=float, default=1.0, help="GPU polling interval seconds")

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
    mon = GPUMonitor(gpu_csv, interval_sec=args.gpu_poll_sec)
    mon.start()

    t0 = time.time()

    # Load model/tokenizer (+ optional LoRA merge)
    print(f"\n[eval] Loading model: {args.base_id_or_path}", flush=True)
    tokenizer, model, dtype = load_model_and_tokenizer(
        args.base_id_or_path, args.lora_dir, bf16, fp16, args.seq_len
    )
    print(f"[eval] Model loaded with dtype: {dtype}", flush=True)

    # Read & cap rows
    rows: List[Dict[str, Any]] = []
    remaining = args.max_rows
    for fp in args.eval_files:
        if remaining <= 0:
            break
        chunk = load_jsonl(fp, max_rows=remaining)
        rows.extend(chunk)
        remaining -= len(chunk)
    
    print(f"[eval] Loaded {len(rows)} examples for evaluation", flush=True)

    # Build prompts/refs
    prompts: List[str] = []
    refs: List[str] = []
    for ex in rows:
        p, r, _ = extract_prompt_and_ref(tokenizer, ex)
        prompts.append(p)
        refs.append(r)

    # Generate in batches
    print(f"[eval] Generating predictions (batch_size=8)...", flush=True)
    hyps: List[str] = []
    B = 8
    for i in range(0, len(prompts), B):
        if i % 50 == 0:
            print(f"[eval] Progress: {i}/{len(prompts)}", flush=True)
        batch_prompts = prompts[i : i + B]
        batch_out = generate_batch(
            tokenizer, model, batch_prompts,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        hyps.extend(batch_out)
    
    print(f"[eval] Generation complete: {len(hyps)} predictions", flush=True)

    # Save predictions
    with open(preds_path, "w", encoding="utf-8") as f:
        for p, r, h in zip(prompts, refs, hyps):
            f.write(json.dumps({"prompt": p, "ref": r, "hyp": h}, ensure_ascii=False) + "\n")
    print(f"[eval] Predictions saved to: {preds_path}", flush=True)

    # Metrics
    print("\n[eval] Computing BLEU-4...", flush=True)
    bleu4 = compute_bleu4(refs, hyps)
    print(f"[eval] BLEU-4: {bleu4:.4f}", flush=True)
    
    print("\n[eval] Computing CodeBLEU...", flush=True)
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
        "codebleu": None if math.isnan(cb["codebleu"]) else round(float(cb["codebleu"]), 4),
        "codebleu_ngram": None if math.isnan(cb["codebleu_ngram"]) else round(float(cb["codebleu_ngram"]), 4),
        "codebleu_weighted_ngram": None if math.isnan(cb["codebleu_weighted_ngram"]) else round(float(cb["codebleu_weighted_ngram"]), 4),
        "codebleu_syntax": None if math.isnan(cb["codebleu_syntax"]) else round(float(cb["codebleu_syntax"]), 4),
        "codebleu_dataflow": None if math.isnan(cb["codebleu_dataflow"]) else round(float(cb["codebleu_dataflow"]), 4),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[eval] Metrics saved to: {metrics_path}", flush=True)

    # Stop GPU monitor
    mon.stop()

    # Push artifacts to Hub
    try:
        token = ensure_hf_token(args.hf_token)
        run_dir_in_repo = f"runs/{run_name}"
        print(f"\n[eval] Uploading to HF Hub: {args.hub_repo_id}/{run_dir_in_repo}", flush=True)
        hf_upload_dir(args.hub_repo_id, run_dir_in_repo, out_dir, token)
        print(f"[eval] Upload complete!", flush=True)
    except Exception as e:
        print(f"[eval] Hub upload failed (continuing anyway): {e}", file=sys.stderr, flush=True)

    # Console summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Predictions : {preds_path}")
    print(f"Metrics     : {metrics_path}")
    print(f"GPU trace   : {gpu_csv}")
    print(f"Hub location: {args.hub_repo_id}/runs/{run_name}/")
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))
    print("="*60)


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()