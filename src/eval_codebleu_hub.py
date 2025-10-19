# -*- coding: utf-8 -*-
"""
Evaluation script that:
- builds a strict "code-only" prompt
- generates with chat or causal models
- extracts first Python code block from output (robust fallback)
- computes BLEU-4 and CodeBLEU (syntax/dataflow included)
- saves predictions & metrics locally
- optionally uploads everything to the HF Hub under <hub_repo_id>/runs/<run_name>/

Fixes included:
- No 'offload_state_dict' usage (compatible with your env)
- Better code extraction -> higher syntax scores
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CodeBLEU wrapper (official or compat)
from .codebleu_shim import compute_codebleu

# Optional Hub upload
from huggingface_hub import HfApi, CommitOperationAdd, create_repo

# -----------------------------
# IO helpers
# -----------------------------
def read_jsonl(path: str, max_rows: int | None = None) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_str():
    return time.strftime("%Y%m%d-%H%M%S")

# -----------------------------
# Prompting & post-processing
# -----------------------------
SYS_STRICT = (
    "You are a Python code generator.\n"
    "Return ONLY valid Python code implementing the requested function/solution.\n"
    "Do not include explanations, comments, markdown, or backticks unless asked.\n"
    "If a function name/signature is implied by the prompt, implement it exactly."
)

def build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """Flatten messages to a single instruction for code-only gen."""
    user_bits = []
    for m in messages:
        if m.get("role") == "user":
            user_bits.append(m.get("content", ""))
    demand = "\n\n".join(user_bits).strip()
    # Ask explicitly for code-only in a fenced block to help extraction downstream
    return (
        f"{SYS_STRICT}\n\n"
        f"Task:\n{demand}\n\n"
        "Return only the Python code, enclosed exactly like this:\n"
        "```python\n# your code here\n```\n"
    )

_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*(?P<code>[\s\S]*?)```", re.IGNORECASE
)

def extract_python(code_like: str) -> str:
    """
    Extract the first ```python ... ``` block if present.
    If not present, try generic ``` ... ```; if still not, heuristically keep
    lines that look like code (starts with def/class/import/from/return/indent).
    """
    if not code_like:
        return ""

    m = _CODE_FENCE_RE.search(code_like)
    if m:
        return m.group("code").strip()

    # generic triple backticks
    mg = re.search(r"```([\s\S]*?)```", code_like)
    if mg:
        return mg.group(1).strip()

    # heuristic: keep likely-code lines
    kept = []
    for line in code_like.splitlines():
        tl = line.strip()
        if not tl:
            continue
        if (
            tl.startswith(("def ", "class ", "import ", "from ", "return ", "@"))
            or tl.startswith(("for ", "while ", "with ", "if ", "elif ", "else:"))
            or tl.startswith(("try:", "except", "finally:", "raise ", "yield"))
            or re.match(r"^[A-Za-z_]\w*\s*=\s*.+", tl)
            or tl.startswith(("    ", "\t"))
        ):
            kept.append(line)
    # if nothing matched, return original (last resort)
    return "\n".join(kept).strip() if kept else code_like.strip()

# -----------------------------
# BLEU-4 (quick, simple)
# -----------------------------
def _tok(s: str) -> List[str]:
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    return [t for t in re.split(r"(\W+)", s) if t and not t.isspace()]

def bleu4(refs: List[str], hyps: List[str], smooth_eps: float = 1e-9) -> float:
    assert len(refs) == len(hyps)
    def ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        out = {}
        for i in range(len(tokens) - n + 1):
            k = tuple(tokens[i:i+n])
            out[k] = out.get(k, 0) + 1
        return out

    precisions = []
    for n in (1,2,3,4):
        match = 0
        total = 0
        for ref, hyp in zip(refs, hyps):
            cr = ngram_counts(_tok(ref), n)
            ch = ngram_counts(_tok(hyp), n)
            overlap = 0
            for k, v in ch.items():
                overlap += min(v, cr.get(k, 0))
            match += overlap
            total += max(1, sum(ch.values()))
        precisions.append((match + smooth_eps) / (total + smooth_eps))

    # BP
    ref_len = sum(len(_tok(r)) for r in refs)
    hyp_len = sum(len(_tok(h)) for h in hyps)
    if hyp_len == 0:
        return 0.0
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / max(1, hyp_len))

    score = bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return 100.0 * score

# -----------------------------
# Model loading & generation
# -----------------------------
@dataclass
class EvalArgs:
    base_id_or_path: str
    lora_dir: str
    eval_files: List[str]
    max_rows: int
    seq_len: int
    max_new_tokens: int
    do_sample: bool
    top_p: float
    temperature: float
    bf16: bool | None
    fp16: bool | None
    hub_repo_id: str
    run_name: str
    hf_token: str | None
    gpu_poll_sec: float

def parse_args() -> EvalArgs:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_id_or_path", required=True)
    ap.add_argument("--lora_dir", default="")
    ap.add_argument("--eval_files", nargs="+", required=True)
    ap.add_argument("--max_rows", type=int, default=200)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", type=lambda x: x.lower() in ("1","true","t","yes","y"), default=False)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--bf16", type=lambda x: x.lower() in ("1","true","t","yes","y"), default=None)
    ap.add_argument("--fp16", type=lambda x: x.lower() in ("1","true","t","yes","y"), default=None)
    ap.add_argument("--hub_repo_id", required=True)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--hf_token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    ap.add_argument("--gpu_poll_sec", type=float, default=0.0)
    a = ap.parse_args()
    return EvalArgs(**vars(a))

def load_model_and_tokenizer(base_id: str, bf16: bool | None, fp16: bool | None):
    if torch.cuda.is_available():
        auto_bf16 = torch.cuda.is_bf16_supported()
        if bf16 is None and fp16 is None:
            use_bf16 = auto_bf16
            use_fp16 = not auto_bf16
        else:
            use_bf16 = bool(bf16) if bf16 is not None else False
            use_fp16 = bool(fp16) if fp16 is not None else (not use_bf16)
    else:
        use_bf16 = use_fp16 = False

    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return tok, model, dtype

def is_chat_model(tokenizer: AutoTokenizer) -> bool:
    # crude but works for Llama 3.x chat
    bos = getattr(tokenizer, "bos_token", None) or ""
    return "<|start_header_id|>" in (bos + "".join(getattr(tokenizer, "added_tokens_decoder", {}) or {}))

def batch_iter(lst: List[Any], bs: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    run = args.run_name or f"eval-{now_str()}"
    out_dir = os.path.join("outputs", "eval", run)
    ensure_dir(out_dir)
    preds_path   = os.path.join(out_dir, "predictions_eval.jsonl")
    metrics_path = os.path.join(out_dir, "metrics_eval.json")

    # Load and merge data
    rows = []
    for fp in args.eval_files:
        rows.extend(read_jsonl(fp, max_rows=None))
    rows = rows[: args.max_rows]

    # Model
    print("\n[eval] Loading model:", args.base_id_or_path)
    tok, model, dtype = load_model_and_tokenizer(args.base_id_or_path, args.bf16, args.fp16)
    print(f"[eval] Model loaded with dtype: {dtype}")
    print(f"[eval] Loaded {len(rows)} examples for evaluation")

    # Build prompts
    prompts = [build_prompt_from_messages(r.get("messages", [])) for r in rows]

    # Generate
    bsz = 8
    all_raw = []
    model.eval()
    print(f"[eval] Generating predictions (batch_size={bsz})...")
    for chunk in batch_iter(prompts, bsz):
        enc = tok(chunk, padding=True, truncation=True, max_length=args.seq_len, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample)
        if args.do_sample:
            gen_kwargs.update(top_p=args.top_p, temperature=args.temperature)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        texts = tok.batch_decode(out, skip_special_tokens=True)
        all_raw.extend(texts)

    # Extract code & save predictions
    preds = []
    with open(preds_path, "w", encoding="utf-8") as f:
        for raw, row in zip(all_raw, rows):
            code = extract_python(raw)
            ref  = row.get("reference", "") or row.get("answer", "") or row.get("output", "")
            obj = {"raw_output": raw, "prediction": code, "reference": ref}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            preds.append(code)

    print(f"[eval] Predictions saved to: {preds_path}")

    # Prepare references
    refs = []
    for r in rows:
        ref = r.get("reference", "") or r.get("answer", "") or r.get("output", "")
        refs.append(ref if isinstance(ref, str) else json.dumps(ref))

    # BLEU-4
    print("\n[eval] Computing BLEU-4...")
    b4 = bleu4(refs, preds)
    print(f"[eval] BLEU-4: {b4:.4f}")

    # CodeBLEU
    print("\n[eval] Computing CodeBLEU...")
    cb = compute_codebleu(preds, refs, lang="python")

    metrics = {
        "model": args.base_id_or_path,
        "lora_dir": "",
        "rows": len(rows),
        "bleu4": round(b4, 4),
        "dtype": "bf16" if dtype == torch.bfloat16 else ("fp16" if dtype == torch.float16 else "fp32"),
        "wall_time_sec": None,  # you can thread a timer if you want
        "codebleu": cb["codebleu"],
        "codebleu_ngram": cb["ngram_match_score"],
        "codebleu_weighted_ngram": cb["weighted_ngram_match_score"],
        "codebleu_syntax": cb["syntax_match_score"],
        "codebleu_dataflow": cb["dataflow_match_score"],
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[eval] Metrics saved to: {metrics_path}")

    # Hub upload
    print("\n[eval] Uploading to HF Hub:", f"{args.hub_repo_id}/runs/{run}")
    api = HfApi(token=args.hf_token)
    try:
        create_repo(repo_id=args.hub_repo_id, private=False, exist_ok=True, token=args.hf_token)
    except Exception:
        pass

    ops = []
    with open(preds_path, "rb") as f:
        ops.append(CommitOperationAdd(path_in_repo=f"runs/{run}/predictions_eval.jsonl", path_or_fileobj=f.read()))
    with open(metrics_path, "rb") as f:
        ops.append(CommitOperationAdd(path_in_repo=f"runs/{run}/metrics_eval.json", path_or_fileobj=f.read()))
    api.create_commit(
        repo_id=args.hub_repo_id,
        operations=ops,
        commit_message=f"Add eval run {run}",
        token=args.hf_token,
    )
    print("[eval] Upload complete!")

    print("\n==================== EVALUATION COMPLETE ====================")
    print("Predictions :", preds_path)
    print("Metrics     :", metrics_path)
    print("Hub location:", f"{args.hub_repo_id}/runs/{run}/")
    print("Metrics:\n", json.dumps(metrics, indent=2))
    print("=============================================================")

if __name__ == "__main__":
    main()
