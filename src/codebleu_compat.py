# -*- coding: utf-8 -*-
"""
Compact, dependency-light CodeBLEU-compatible scorer.

This is NOT a perfect reimplementation of CodeBLEU. It:
- approximates n-gram & weighted n-gram via sentencepiece-free token n-grams
- approximates syntax/dataflow by (a) parsability via ast for Python and
  (b) simple identifier graph overlap.

The goal is robustness + directionally-correct signals when official package
is unavailable or fragile in your environment.
"""

from __future__ import annotations
import ast
import re
from collections import Counter
from typing import List, Dict, Any

def _tok_lines(s: str) -> List[str]:
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    return [t for t in re.split(r"(\W+)", s) if t and not t.isspace()]

def _ngram_counter(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(0, len(tokens)-n+1))

def _ngram_f1(ref: List[str], hyp: List[str], n: int) -> float:
    cr = _ngram_counter(ref, n)
    ch = _ngram_counter(hyp, n)
    overlap = sum((cr & ch).values())
    pr = sum(ch.values()) or 1
    re = sum(cr.values()) or 1
    precision = overlap / pr
    recall = overlap / re
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def _weighted_ngram_score(ref: List[str], hyp: List[str]) -> float:
    # simple weights favoring keywords/idents
    w = Counter()
    for t in ref:
        if re.match(r"[A-Za-z_]\w*$", t):
            w[t] += 3
        else:
            w[t] += 1
    hit = 0
    total = 0
    for i, t in enumerate(hyp):
        total += w.get(t, 1)
        if i < len(ref) and t == ref[i]:
            hit += w.get(t, 1)
    return hit / total if total else 0.0

def _syntax_score_py(ref: str, hyp: str) -> float:
    # 1. both parsable -> 1.0 ; only hyp parsable -> 0.5 ; else 0
    try:
        ast.parse(ref)
        ref_ok = True
    except Exception:
        ref_ok = False
    try:
        ast.parse(hyp)
        hyp_ok = True
    except Exception:
        hyp_ok = False
    if ref_ok and hyp_ok:
        return 1.0
    if hyp_ok:
        return 0.5
    return 0.0

def _idents(s: str) -> Counter:
    return Counter(re.findall(r"[A-Za-z_]\w*", s))

def _dataflow_score_py(ref: str, hyp: str) -> float:
    # crude identifier overlap Jaccard on multiset min/sum
    R = _idents(ref)
    H = _idents(hyp)
    if not R and not H:
        return 1.0
    inter = sum((R & H).values())
    denom = sum((R | H).values()) or 1
    return inter / denom

def codebleu_score(refs: List[str], hyps: List[str], lang: str = "python") -> Dict[str, Any]:
    assert len(refs) == len(hyps)
    n = len(refs)
    if n == 0:
        return {"codebleu": 0.0, "ngram": 0.0, "weighted_ngram": 0.0, "syntax": 0.0, "dataflow": 0.0}

    ngram_scores = []
    weighted_scores = []
    syntax_scores = []
    dataflow_scores = []

    for ref, hyp in zip(refs, hyps):
        tr = _tok_lines(ref)
        th = _tok_lines(hyp)

        # ngram avg F1 for n=1..4
        ng = sum(_ngram_f1(tr, th, k) for k in (1,2,3,4)) / 4.0
        wg = _weighted_ngram_score(tr, th)

        if lang == "python":
            syn = _syntax_score_py(ref, hyp)
            df  = _dataflow_score_py(ref, hyp)
        else:
            # fallback if other languages show up
            syn = 1.0 if hyp.strip() else 0.0
            df  = 0.0

        ngram_scores.append(ng)
        weighted_scores.append(wg)
        syntax_scores.append(syn)
        dataflow_scores.append(df)

    res = {
        "ngram": float(sum(ngram_scores) / n),
        "weighted_ngram": float(sum(weighted_scores) / n),
        "syntax": float(sum(syntax_scores) / n),
        "dataflow": float(sum(dataflow_scores) / n),
    }
    res["codebleu"] = 0.25 * (res["ngram"] + res["weighted_ngram"] + res["syntax"] + res["dataflow"])
    return res
