# -*- coding: utf-8 -*-
"""
A robust CodeBLEU wrapper that:
- prefers the official 'codebleu' package when available
- falls back to a bundled-compatible implementation if needed
- normalizes code and tolerates parser failures
"""

from __future__ import annotations
import re
from typing import List, Dict, Any

# Try the official package
_OFFICIAL_OK = False
try:
    from codebleu import calc_codebleu as _official_calc
    _OFFICIAL_OK = True
except Exception:
    _OFFICIAL_OK = False

# Try a compatible fallback (bundled alongside this file in your repo)
_FALLBACK_OK = False
try:
    from .codebleu_compat import codebleu_score as _compat_calc
    _FALLBACK_OK = True
except Exception:
    _FALLBACK_OK = False

def _normalize_code(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # strip trailing spaces
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    # collapse >1 blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def compute_codebleu(preds: List[str], refs: List[str], lang: str = "python") -> Dict[str, Any]:
    """
    Returns:
      {
        'codebleu': float,
        'ngram_match_score': float,
        'weighted_ngram_match_score': float,
        'syntax_match_score': float,
        'dataflow_match_score': float
      }
    """
    assert len(preds) == len(refs), "preds and refs must have same length"
    preds = [_normalize_code(p) for p in preds]
    refs  = [_normalize_code(r) for r in refs]

    if _OFFICIAL_OK:
        try:
            res = _official_calc.calc_code_bleu(refs, preds, lang,
                                                weights=(0.25, 0.25, 0.25, 0.25))
            return {
                "codebleu": float(res["codebleu"]),
                "ngram_match_score": float(res["ngram_match_score"]),
                "weighted_ngram_match_score": float(res["weighted_ngram_match_score"]),
                "syntax_match_score": float(res["syntax_match_score"]),
                "dataflow_match_score": float(res["dataflow_match_score"]),
            }
        except Exception:
            # fall through to compat
            pass

    if _FALLBACK_OK:
        res = _compat_calc(refs, preds, lang=lang)
        return {
            "codebleu": float(res["codebleu"]),
            "ngram_match_score": float(res["ngram"]),
            "weighted_ngram_match_score": float(res["weighted_ngram"]),
            "syntax_match_score": float(res["syntax"]),
            "dataflow_match_score": float(res["dataflow"]),
        }

    # If all else fails, give zeros with a warning-ish shape
    return {
        "codebleu": 0.0,
        "ngram_match_score": 0.0,
        "weighted_ngram_match_score": 0.0,
        "syntax_match_score": 0.0,
        "dataflow_match_score": 0.0,
    }
