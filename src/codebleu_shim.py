# src/codebleu_shim.py
# --------------------------------------------------------------------------------------
# Lightweight, robust CodeBLEU-style scorer with safe fallbacks.
# Computes:
#   - bleu4
#   - codebleu_ngram
#   - codebleu_weighted_ngram
#   - codebleu_syntax
#   - codebleu_dataflow
#   - codebleu (equal-weight aggregate)
#
# Notes:
#   * Syntax uses tree-sitter if available (via tree_sitter_languages or a custom .so).
#   * Dataflow uses a simple variable-name overlap proxy (not full dataflow graph).
#   * Weighted n-gram gives extra credit to Python keywords/operators.
#   * If a component can't be computed, it falls back to 0.0 and logs a warning.
#
# CLI:
#   python -m src.codebleu_shim --preds path.jsonl --refs path.jsonl
#   (each line: {"pred": "...", "ref": "..."})
# --------------------------------------------------------------------------------------

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

# ---------------------------
# Optional dependencies
# ---------------------------
_SACREBLEU = None
try:
    import sacrebleu  # type: ignore
    _SACREBLEU = sacrebleu
except Exception:
    _SACREBLEU = None

_TREE_SITTER_OK = False
_TS_PARSER = None
_TS_LANG = None
_TS_ERR = None

# First try tree_sitter_languages (prebuilt grammars)
try:
    from tree_sitter import Parser, Language  # type: ignore
    try:
        from tree_sitter_languages import get_language  # type: ignore
        _TS_LANG = get_language("python")
        _TS_PARSER = Parser()
        _TS_PARSER.set_language(_TS_LANG)
        _TREE_SITTER_OK = True
    except Exception as e:
        _TS_ERR = e
        # If user built a custom bundle at build/my-languages.so:
        so_path = os.path.join("build", "my-languages.so")
        if os.path.isfile(so_path):
            try:
                _TS_LANG = Language(so_path, "python")
                _TS_PARSER = Parser()
                _TS_PARSER.set_language(_TS_LANG)
                _TREE_SITTER_OK = True
            except Exception as e2:
                _TS_ERR = e2
except Exception as e:
    _TS_ERR = e


# ---------------------------
# Small helpers
# ---------------------------
PY_KEYWORDS = {
    # from keyword.kwlist (kept explicit to avoid importing keyword on hot path)
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else",
    "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
    "return", "try", "while", "with", "yield",
}

PY_OPERATORS = {
    "+", "-", "*", "/", "//", "%", "**", "=", "+=", "-=", "*=", "/=", "//=",
    "%=", "**=", "==", "!=", ">", "<", ">=", "<=", "&", "|", "^", "~", ">>", "<<",
    "and", "or", "not", "is", "in", "not in", "is not",
    ".", ":", ",", ";"
}

def _simple_tokenize_py(code: str) -> List[str]:
    """
    Very simple tokenizer that splits on whitespace & punctuation,
    keeping operators & keywords when possible.
    """
    out = []
    cur = []
    def flush():
        if cur:
            tok = "".join(cur)
            out.append(tok)
            cur.clear()

    for ch in code:
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            flush()
            if not ch.isspace():
                out.append(ch)
    flush()
    # Merge operator pairs/triples where reasonable
    merged = []
    i = 0
    while i < len(out):
        if i + 1 < len(out) and out[i] + out[i+1] in PY_OPERATORS:
            merged.append(out[i] + out[i+1])
            i += 2
        elif i + 2 < len(out) and (out[i] + out[i+1] + out[i+2]) in PY_OPERATORS:
            merged.append(out[i] + out[i+1] + out[i+2])
            i += 3
        else:
            merged.append(out[i])
            i += 1
    return merged


# ---------------------------
# BLEU-4
# ---------------------------
def _bleu4_sacrebleu(preds: List[str], refs: List[str]) -> float:
    # sacrebleu expects list of sys outputs, and list of reference sets
    res = _SACREBLEU.corpus_bleu(preds, [refs], lowercase=False, force=True, tokenize=None)
    return float(res.score)

def _bleu4_simple(preds: List[str], refs: List[str]) -> float:
    """
    Simple corpus BLEU-4 with tiny smoothing.
    Tokenizer: _simple_tokenize_py (ok for Python).
    Returns percentage in [0, 100].
    """
    def ngrams(seq, n):
        return list(zip(*[seq[i:] for i in range(n)]))

    weights = [0.25, 0.25, 0.25, 0.25]
    p_ns = [0.0, 0.0, 0.0, 0.0]
    hyp_len = 0
    ref_len = 0
    eps = 1e-9

    for hyp, ref in zip(preds, refs):
        h = _simple_tokenize_py(hyp)
        r = _simple_tokenize_py(ref)
        hyp_len += max(len(h), 1)
        ref_len += max(len(r), 1)

        for n in range(1, 5):
            h_ngrams = Counter(ngrams(h, n))
            r_ngrams = Counter(ngrams(r, n))
            overlap = 0
            total = 0
            for ng, c in h_ngrams.items():
                total += c
                overlap += min(c, r_ngrams.get(ng, 0))
            p = (overlap + eps) / (total + eps)
            p_ns[n-1] += math.log(p)

    # average log-precisions
    p_ns = [math.exp(p / max(len(preds), 1)) for p in p_ns]
    # brevity penalty
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - (ref_len + 1e-12) / (hyp_len + 1e-12))
    bleu = bp * math.exp(sum(w * math.log(p + 1e-12) for w, p in zip(weights, p_ns)))
    return float(bleu * 100.0)

def corpus_bleu4(preds: List[str], refs: List[str]) -> float:
    if _SACREBLEU is not None:
        try:
            return _bleu4_sacrebleu(preds, refs)
        except Exception:
            pass
    return _bleu4_simple(preds, refs)


# ---------------------------
# Weighted n-gram
# ---------------------------
def weighted_ngram_score(preds: List[str], refs: List[str]) -> float:
    """
    Boost n-gram matches that include Python keywords/operators.
    Scored similarly to BLEU-4 but each matched n-gram gets a weight:
      base=1.0; +0.5 if contains any PY_KEYWORDS; +0.25 if contains any PY_OPERATORS
    Returned as [0,1].
    """
    def ngrams(seq, n):
        return list(zip(*[seq[i:] for i in range(n)]))

    totals = 0.0
    hits = 0.0
    eps = 1e-12

    for hyp, ref in zip(preds, refs):
        htok = _simple_tokenize_py(hyp)
        rtok = _simple_tokenize_py(ref)
        for n in range(1, 5):
            h_ngrams = Counter(ngrams(htok, n))
            r_ngrams = Counter(ngrams(rtok, n))
            for ng, c in h_ngrams.items():
                weight = 1.0
                toks = set(ng)
                if any(t in PY_KEYWORDS for t in toks):
                    weight += 0.5
                if any(t in PY_OPERATORS for t in toks):
                    weight += 0.25
                totals += c * weight
                match = min(c, r_ngrams.get(ng, 0))
                if match > 0:
                    hits += match * weight

    return float((hits + eps) / (totals + eps))


# ---------------------------
# Syntax score (tree-sitter)
# ---------------------------
def _ts_node_types(root) -> Counter:
    """Return multiset of node types in the parsed tree."""
    stack = [root]
    cnt = Counter()
    while stack:
        node = stack.pop()
        cnt[node.type] += 1
        for i in range(node.named_child_count):
            stack.append(node.named_child(i))
    return cnt

def syntax_score(preds: List[str], refs: List[str]) -> float:
    """
    AST node-type multiset overlap (Jaccard variant) + parse-success prior.
    If parsing fails on either side, that pair contributes 0.0.
    Returns [0,1].
    """
    if not _TREE_SITTER_OK:
        warnings.warn(
            f"[codebleu_shim] tree-sitter not available — syntax component set to 0.0. "
            f"Last error: {_TS_ERR}",
            RuntimeWarning,
        )
        return 0.0

    scores = []
    for hyp, ref in zip(preds, refs):
        try:
            th = _TS_PARSER.parse(hyp.encode("utf-8"))
            tr = _TS_PARSER.parse(ref.encode("utf-8"))
            rh = th.root_node
            rr = tr.root_node
            # If the whole module failed, give zero
            if rh is None or rr is None:
                scores.append(0.0)
                continue
            H = _ts_node_types(rh)
            R = _ts_node_types(rr)
            # Jaccard on node-type *multiset* (min/max)
            keys = set(H) | set(R)
            inter = sum(min(H[k], R[k]) for k in keys)
            union = sum(max(H[k], R[k]) for k in keys) or 1
            j = inter / union
            # Slightly penalize trees that have explicit ERROR nodes
            if H.get("ERROR", 0) > 0:
                j *= 0.8
            scores.append(float(j))
        except Exception:
            scores.append(0.0)
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


# ---------------------------
# Dataflow proxy
# ---------------------------
def _extract_identifiers(code: str) -> Counter:
    """
    Extract simple identifiers (variables/function names).
    Very rough proxy for dataflow; returns a multiset.
    """
    toks = _simple_tokenize_py(code)
    idents = [t for t in toks if (t and (t[0].isalpha() or t[0] == "_") and t not in PY_KEYWORDS)]
    return Counter(idents)

def dataflow_score(preds: List[str], refs: List[str]) -> float:
    """
    Approximate dataflow similarity: Jaccard of identifier multisets.
    Returns [0,1].
    """
    scores = []
    for hyp, ref in zip(preds, refs):
        H = _extract_identifiers(hyp)
        R = _extract_identifiers(ref)
        keys = set(H) | set(R)
        inter = sum(min(H[k], R[k]) for k in keys)
        union = sum(max(H[k], R[k]) for k in keys) or 1
        scores.append(float(inter / union))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


# ---------------------------
# Public API
# ---------------------------
@dataclass
class CodeBLEUScores:
    bleu4: float                  # 0..100
    codebleu_ngram: float         # 0..1
    codebleu_weighted_ngram: float# 0..1
    codebleu_syntax: float        # 0..1
    codebleu_dataflow: float      # 0..1
    codebleu: float               # 0..1
    details: Dict[str, float]

def compute_codebleu(
    preds: List[str],
    refs: List[str],
    lang: str = "python",
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> CodeBLEUScores:
    """
    Compute CodeBLEU-style scores. `lang` is kept for API parity (currently only 'python' path is implemented).
    """
    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must be the same length (got {len(preds)} vs {len(refs)})")

    # BLEU-4 (0..100)
    bleu4 = corpus_bleu4(preds, refs)

    # CodeBLEU components (0..1)
    c_ngram = corpus_bleu4(preds, refs) / 100.0
    c_weighted = weighted_ngram_score(preds, refs)
    c_syntax = syntax_score(preds, refs)
    c_dataflow = dataflow_score(preds, refs)

    w1, w2, w3, w4 = weights
    codebleu = w1 * c_ngram + w2 * c_weighted + w3 * c_syntax + w4 * c_dataflow

    details = {
        "bleu4": bleu4,
        "codebleu_ngram": c_ngram,
        "codebleu_weighted_ngram": c_weighted,
        "codebleu_syntax": c_syntax,
        "codebleu_dataflow": c_dataflow,
        "codebleu": codebleu,
    }
    return CodeBLEUScores(
        bleu4=bleu4,
        codebleu_ngram=c_ngram,
        codebleu_weighted_ngram=c_weighted,
        codebleu_syntax=c_syntax,
        codebleu_dataflow=c_dataflow,
        codebleu=codebleu,
        details=details,
    )


# ---------------------------
# CLI for quick testing
# ---------------------------
def _read_jsonl_pairs(path: str) -> Tuple[List[str], List[str]]:
    preds, refs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # accept a few common shapes
            p = obj.get("pred") or obj.get("prediction") or obj.get("hyp") or obj.get("generated") or ""
            r = obj.get("ref") or obj.get("reference") or obj.get("gold") or obj.get("target") or ""
            preds.append(str(p))
            refs.append(str(r))
    return preds, refs

def _main():
    import argparse
    ap = argparse.ArgumentParser(description="CodeBLEU shim scorer (robust & self-contained).")
    ap.add_argument("--preds", type=str, help="JSONL with lines like {'pred': ..., 'ref': ...}", required=False)
    ap.add_argument("--refs", type=str, help="Optional separate JSONL with {'ref': ...}", required=False)
    ap.add_argument("--lang", type=str, default="python")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if args.preds and not args.refs:
        preds, refs = _read_jsonl_pairs(args.preds)
    elif args.preds and args.refs:
        # Combine when separate files are used; each line in --preds has {"pred": ...}
        # and each line in --refs has {"ref": ...}
        with open(args.preds, "r", encoding="utf-8") as f:
            preds = [json.loads(l).get("pred", "") for l in f if l.strip()]
        with open(args.refs, "r", encoding="utf-8") as f:
            refs = [json.loads(l).get("ref", "") for l in f if l.strip()]
        m = min(len(preds), len(refs))
        preds, refs = preds[:m], refs[:m]
    else:
        # tiny demo
        preds = ["def add(a,b):\n    return a+b\n"]
        refs  = ["def add(a, b):\n    return a + b\n"]

    scores = compute_codebleu(preds, refs, lang=args.lang)
    out = {
        "bleu4": scores.bleu4,
        "codebleu": scores.codebleu,
        "codebleu_ngram": scores.codebleu_ngram,
        "codebleu_weighted_ngram": scores.codebleu_weighted_ngram,
        "codebleu_syntax": scores.codebleu_syntax,
        "codebleu_dataflow": scores.codebleu_dataflow,
    }
    js = json.dumps(out, indent=2)
    print(js)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js + "\n")


# Compatibility flags for your logs
class ParserShimStatus:
    language = _TREE_SITTER_OK
class UtilsShimStatus:
    patched = True
class CodeBLEUShimStatus:
    patched = True

if __name__ == "__main__":
    _main()
