# src/eval_codebleu_mlflow.py
# =================================================================================================
# Evaluate a (student) code model on your prepared JSONL files and log metrics to MLflow.
#
# What it does:
#   • Loads a base model (e.g., Llama-3.1-8B-Instruct) and optional LoRA adapters
#   • Builds prompts from your JSONL (messages array as in training)
#   • Generates code completions
#   • Computes a *simple* BLEU-4 score (pure-Python fallback) per sample + average
#     (If sacrebleu is installed, it uses that; otherwise uses a simple BLEU-4.)
#   • Saves detailed predictions to an artifact JSONL (prompt, ref, pred, bleu)
#   • Logs summary metrics + GPU telemetry + wall time to MLflow (if available)
#
# Notes:
#   • This is a lightweight evaluator suitable for quick relative comparisons.
#   • True CodeBLEU (https://arxiv.org/abs/2009.10297) is more complex; feel free to
#     swap in a proper CodeBLEU library if you’d like. This script’s interface remains
#     the same so it’s easy to replace the scoring function.
# =================================================================================================

from __future__ import annotations
import os
import sys
import json
import time
import math
import argparse
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional: PEFT for adapters
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None  # type: ignore
    _HAS_PEFT = False

# Optional: MLflow
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    mlflow = None  # type: ignore
    _HAS_MLFLOW = False

# Optional: sacrebleu for BLEU
try:
    import sacrebleu
    _HAS_SACREBLEU = True
except Exception:
    _HAS_SACREBLEU = False

# Optional: NVML telemetry
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    pynvml = None  # type: ignore
    _HAS_NVML = False

# Project helper for formatting the prompt
try:
    from .utils import build_chat_text
except Exception:
    # Minimal fallback if utils is not importable
    def build_chat_text(messages: List[Dict[str, str]]) -> str:
        """
        Very simple chat format fallback: just concatenates roles/content.
        """
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            out.append(f"[{role.upper()}]\n{content}")
        out.append("[ASSISTANT]\n")
        return "\n".join(out)


# -------------------------------------------------------------------------------------------------
# Simple BLEU-4 (fallback when sacrebleu is not available)
# -------------------------------------------------------------------------------------------------
def _ngram_counts(tokens: List[str], n: int) -> Dict[tuple, int]:
    return {tuple(tokens[i:i+n]): tokens[i:i+n].count(tokens[i]) for i in range(len(tokens)-n+1)}  # quick & dirty


def _simple_bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    A very naive BLEU-4 approximation (not brevity-penalized like sacrebleu).
    For relative comparisons only. Prefer sacrebleu if available.
    """
    ref_toks = reference.split()
    hyp_toks = hypothesis.split()
    if len(hyp_toks) == 0:
        return 0.0

    p_ns = []
    for n in range(1, max_n + 1):
        ref_ngrams = {}
        for i in range(len(ref_toks) - n + 1):
            ng = tuple(ref_toks[i:i+n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        match = 0
        total = max(1, len(hyp_toks) - n + 1)
        for i in range(len(hyp_toks) - n + 1):
            ng = tuple(hyp_toks[i:i+n])
            if ref_ngrams.get(ng, 0) > 0:
                match += 1
                ref_ngrams[ng] -= 1

        p_ns.append(match / total)

    # geometric mean of p_ns
    try:
        score = math.exp(sum(math.log(p + 1e-9) for p in p_ns) / len(p_ns))
    except Exception:
        score = 0.0

    # crude brevity penalty
    bp = 1.0 if len(hyp_toks) > len(ref_toks) else math.exp(1 - len(ref_toks) / max(1, len(hyp_toks)))
    return 100.0 * bp * score


def _bleu(reference: str, hypothesis: str) -> float:
    """Prefer sacrebleu if present; otherwise simple BLEU-4."""
    if _HAS_SACREBLEU:
        try:
            # sacrebleu expects list of references
            return float(sacrebleu.corpus_bleu([hypothesis], [[reference]]).score)
        except Exception:
            pass
    return _simple_bleu_score(reference, hypothesis, max_n=4)


# -------------------------------------------------------------------------------------------------
# GPU Telemetry helpers
# -------------------------------------------------------------------------------------------------
def _nvml_init_once() -> bool:
    if not _HAS_NVML:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def _collect_gpu_snapshot() -> Dict[str, float]:
    if not _HAS_NVML:
        return {}
    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return {}

    if count == 0:
        return {}

    util_vals, used, total, pwr, temps = [], [], [], [], []
    for i in range(count):
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            pow_mw = pynvml.nvmlDeviceGetPowerUsage(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)

            util_vals.append(float(util))
            used.append(float(mem.used) / (1024 ** 3))
            total.append(float(mem.total) / (1024 ** 3))
            pwr.append(float(pow_mw) / 1000.0)
            temps.append(float(temp))
        except Exception:
            continue

    if not util_vals:
        return {}

    def _mean(xs: List[float]) -> float:
        return sum(xs) / max(1, len(xs))

    mem_tot = sum(total)
    mem_used = sum(used)
    frac = mem_used / mem_tot if mem_tot > 0 else 0.0

    return {
        "gpu_count": float(len(util_vals)),
        "gpu_util_mean_pct": _mean(util_vals),
        "gpu_mem_used_GiB": mem_used,
        "gpu_mem_total_GiB": mem_tot,
        "gpu_mem_frac": frac,
        "gpu_power_W_mean": _mean(pwr) if pwr else 0.0,
        "gpu_temp_C_mean": _mean(temps) if temps else 0.0,
    }


# -------------------------------------------------------------------------------------------------
# Data loader (reads your JSONL produced by data_prep.py)
# -------------------------------------------------------------------------------------------------
def load_jsonl(path: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except Exception:
                continue
    return rows


def build_eval_pairs(examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    From each example with `messages`, create:
      - prompt text = build_chat_text(all messages EXCEPT the last assistant)
      - reference = last assistant content (if present)
    Skips examples lacking an assistant reference.
    """
    pairs = []
    for ex in examples:
        msgs = ex.get("messages", [])
        # find last assistant message as reference
        ref = None
        cut = len(msgs)
        for j in reversed(range(len(msgs))):
            if msgs[j].get("role") == "assistant":
                ref = msgs[j].get("content", "")
                cut = j  # prompt includes everything before ref
                break
        if ref is None or not ref.strip():
            continue
        prompt_msgs = msgs[:cut]
        prompt_text = build_chat_text(prompt_msgs)
        pairs.append({"prompt": prompt_text, "reference": ref})
    return pairs


# -------------------------------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------------------------------
def load_model_and_tokenizer(
    base_id_or_path: str,
    lora_dir: Optional[str],
    bf16: bool,
    fp16: bool,
    seq_len: int,
):
    """
    Load tokenizer and model (optionally with LoRA adapters).
    """
    tokenizer = AutoTokenizer.from_pretrained(base_id_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_id_or_path,
        torch_dtype=dtype,
        device_map="auto",               # let HF shard if multiple GPUs are visible
        low_cpu_mem_usage=True,
        offload_state_dict=True,
    )

    if lora_dir:
        if not _HAS_PEFT:
            raise RuntimeError("PEFT not installed but --lora_dir was provided.")
        # Load adapters; this keeps PEFT-ified model structure
        model = PeftModel.from_pretrained(model, lora_dir)

    model.eval()
    return tokenizer, model


# -------------------------------------------------------------------------------------------------
# Generation
# -------------------------------------------------------------------------------------------------
@torch.inference_mode()
def generate(model, tokenizer, prompts: List[str], max_new_tokens: int, do_sample: bool, top_p: float, temperature: float) -> List[str]:
    outs = []
    for p in prompts:
        # Tokenize with truncation to respect seq_len context
        enc = tokenizer(
            p,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # decode only the newly generated part
        full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # a simple heuristic: remove the original prompt text if it's included verbatim
        if full.startswith(p):
            outs.append(full[len(p):].strip())
        else:
            outs.append(full.strip())
    return outs


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate code model (BLEU proxy) + log to MLflow.")

    # Model & adapters
    ap.add_argument("--base_id_or_path", type=str, required=True,
                    help="Base model ID or local path (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct).")
    ap.add_argument("--lora_dir", type=str, default=None,
                    help="Optional LoRA adapters dir (e.g., outputs/llama31_8b_kd_lora/lora).")

    # Data
    ap.add_argument("--eval_files", type=str, nargs="+", required=True,
                    help="One or more JSONL files with {messages:[...]}.")
    ap.add_argument("--max_rows", type=int, default=200,
                    help="Max rows to evaluate (for speed).")

    # Generation
    ap.add_argument("--seq_len", type=int, default=2048, help="Tokenizer/model max length.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", type=str, default="False")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)

    # Precision
    ap.add_argument("--bf16", type=str, default=None)
    ap.add_argument("--fp16", type=str, default=None)

    # MLflow
    ap.add_argument("--log_to_mlflow", type=str, default="True")
    ap.add_argument("--experiment_name", type=str, default=None)
    ap.add_argument("--run_name", type=str, default=None)

    args = ap.parse_args()

    def _str2bool(x: Optional[str]) -> Optional[bool]:
        if x is None:
            return None
        return x.lower() in ("1", "true", "t", "yes", "y")

    bf16 = _str2bool(args.bf16)
    fp16 = _str2bool(args.fp16)
    log_to_mlflow = _str2bool(args.log_to_mlflow)

    # Set tokenizer max length (helps avoid super-long prompts)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Prepare data
    rows: List[Dict[str, Any]] = []
    for path in args.eval_files:
        rows.extend(load_jsonl(path))
    pairs = build_eval_pairs(rows)
    if args.max_rows is not None and args.max_rows > 0:
        pairs = pairs[: args.max_rows]
    if not pairs:
        print("No evaluable (prompt, reference) pairs found.")
        sys.exit(1)

    # Load model
    tokenizer, model = load_model_and_tokenizer(
        base_id_or_path=args.base_id_or_path,
        lora_dir=args.lora_dir,
        bf16=bool(bf16),
        fp16=bool(fp16),
        seq_len=args.seq_len,
    )
    # Respect seq_len override
    tokenizer.model_max_length = int(args.seq_len)

    # Init MLflow (optional)
    run_active = False
    if log_to_mlflow and _HAS_MLFLOW:
        try:
            if args.experiment_name:
                mlflow.set_experiment(args.experiment_name)
            mlflow.start_run(run_name=args.run_name or "eval-code")
            run_active = True
            mlflow.set_tags({
                "phase": "evaluation",
                "evaluator": "eval_codebleu_mlflow.py",
                "uses_lora": str(bool(args.lora_dir)),
            })
            mlflow.log_params({
                "base_id_or_path": args.base_id_or_path,
                "lora_dir": args.lora_dir or "",
                "seq_len": args.seq_len,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "bf16": bool(bf16),
                "fp16": bool(fp16),
                "num_eval_rows": len(pairs),
            })
        except Exception as e:
            print(f"[eval] MLflow initialization failed: {e}")
            run_active = False

    # Init NVML if available (for one-shot snapshots pre/post)
    nvml_ready = _nvml_init_once()
    pre_gpu = _collect_gpu_snapshot() if nvml_ready else {}

    # Evaluate
    t0 = time.time()
    prompts = [p["prompt"] for p in pairs]
    refs = [p["reference"] for p in pairs]

    preds = generate(
        model,
        tokenizer,
        prompts,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=_str2bool(args.do_sample) or False,
        top_p=float(args.top_p),
        temperature=float(args.temperature),
    )
    t1 = time.time()
    wall_s = t1 - t0

    # Compute per-sample BLEU and average
    per_bleu = []
    for ref, hyp in zip(refs, preds):
        per_bleu.append(_bleu(ref, hyp))
    bleu_avg = sum(per_bleu) / max(1, len(per_bleu))

    # Torch-reported peak memory on each visible GPU (if CUDA)
    cuda_mem_peak = {}
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    # bytes -> GiB
                    peak_gib = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
                    cuda_mem_peak[f"cuda{i}_max_mem_allocated_GiB"] = peak_gib
        except Exception:
            pass

    post_gpu = _collect_gpu_snapshot() if nvml_ready else {}

    # Save detailed predictions to an artifact file
    out_dir = os.path.join(os.getcwd(), "outputs", "eval")
    os.makedirs(out_dir, exist_ok=True)
    artifact_path = os.path.join(out_dir, "predictions_eval.jsonl")
    with open(artifact_path, "w", encoding="utf-8") as f:
        for p, r, h, b in zip(prompts, refs, preds, per_bleu):
            f.write(json.dumps({"prompt": p, "reference": r, "prediction": h, "bleu": b}, ensure_ascii=False) + "\n")

    # Print summary
    print("\n=== EVAL SUMMARY ===")
    print(f"samples          : {len(pairs)}")
    print(f"BLEU-4 (avg)     : {bleu_avg:.2f}")
    print(f"wall_time_sec    : {wall_s:.2f}")
    if pre_gpu:
        print(f"pre_gpu_snapshot : {pre_gpu}")
    if post_gpu:
        print(f"post_gpu_snapshot: {post_gpu}")
    if cuda_mem_peak:
        print(f"torch_peak_mem   : {cuda_mem_peak}")
    print(f"predictions file : {artifact_path}")

    # Log to MLflow
    if run_active and _HAS_MLFLOW:
        try:
            mlflow.log_metric("bleu4_avg", float(bleu_avg))
            mlflow.log_metric("wall_time_sec", float(wall_s))
            for k, v in pre_gpu.items():
                mlflow.log_metric(f"pre_{k}", float(v))
            for k, v in post_gpu.items():
                mlflow.log_metric(f"post_{k}", float(v))
            for k, v in cuda_mem_peak.items():
                mlflow.log_metric(k, float(v))
            mlflow.log_artifact(artifact_path, artifact_path="eval_outputs")
        except Exception as e:
            print(f"[eval] MLflow logging failed: {e}")
        finally:
            try:
                mlflow.end_run()
            except Exception:
                pass

    # Shutdown NVML
    if nvml_ready:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
