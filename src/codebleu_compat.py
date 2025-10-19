# src/codebleu_compat.py
# ------------------------------------------------------------------------------------
# Minimal, dependency-light CodeBLEU for Python.
# - BLEU-4 via sacrebleu
# - Weighted n-gram using Python keywords as weights
# - Syntax match via tree_sitter (Python grammar)
# - Dataflow match is set to 0.0 (no packaged DFG extractor for Python)
#
# Returns a dict compatible with the 'codebleu' PyPI package:
#   {
#     "codebleu": float,                   # overall (avg of 4 parts)
#     "ngram_match": float,
#     "weighted_ngram_match": float,
#     "syntax_match": float,
#     "dataflow_match": float
#   }
# ------------------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Tuple
import keyword
import math

# sacrebleu for BLEU-4 / tokenization
try:
    import sacrebleu
except Exception as e:
    raise RuntimeError("codebleu_compat.py requires 'sacrebleu' (pip install sacrebleu).") from e

# tree-sitter Python grammar
try:
    from tree_sitter import Parser
    import tree_sitter_python as tsp  # provides tree-sitter Python grammar capsule
except Exception as e:
    raise RuntimeError(
        "codebleu_compat.py needs tree-sitter + tree-sitter-python. "
        "Install e.g.: pip install 'tree-sitter==0.22.3' 'tree-sitter-python==0.23.4'"
    ) from e


# ---------------------------
# Utility: n-grams & weights
# ---------------------------
PY_KEYWORDS = set(keyword.kwlist)

def tokenize_python(code: str) -> List[str]:
    # ultra-simple whitespace tokenization (good enough for BLEU-style counts)
    return code.strip().split()

def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1))]

def ngram_match_score(refs: List[str], hyps: List[str], max_n: int = 4) -> float:
    # Use sacrebleu BLEU-4, which already handles clipping etc.
    bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True, tokenize="intl", smooth_method="exp")
    return float(bleu.score / 100.0)

def weighted_ngram_match_score(refs: List[str], hyps: List[str], max_n: int = 4) -> float:
    # Assign extra weight to n-grams containing Python keywords
    def weight_of_ng(ng: Tuple[str, ...]) -> float:
        # weight base 1.0, +0.5 if contains keyword
        return 1.0 + (0.5 if any(tok in PY_KEYWORDS for tok in ng) else 0.0)

    ref_tokens = [tokenize_python(r) for r in refs]
    hyp_tokens = [tokenize_python(h) for h in hyps]

    scores = []
    for r_tok, h_tok in zip(ref_tokens, hyp_tokens):
        per_n_scores = []
        for n in range(1, max_n + 1):
            r_ngrams = ngrams(r_tok, n)
            h_ngrams = ngrams(h_tok, n)
            if not r_ngrams:
                # degenerate case
                per_n_scores.append(1.0 if not h_ngrams else 0.0)
                continue

            # weighted precision = sum(min(count_h, count_r) * w) / sum(count_h * w)
            from collections import Counter
            rc = Counter(r_ngrams)
            hc = Counter(h_ngrams)

            num = 0.0
            den = 0.0
            for ng, hcount in hc.items():
                w = weight_of_ng(ng)
                den += hcount * w
                num += min(hcount, rc.get(ng, 0)) * w

            p_n = (num / den) if den > 0 else 0.0
            per_n_scores.append(p_n)

        # geometric mean over up to 4 n-gram precisions (BLEU-style)
        gm = math.exp(sum((math.log(max(s, 1e-12)) for s in per_n_scores)) / max_n)
        scores.append(gm)

    return float(sum(scores) / max(len(scores), 1))


# ---------------------------
# Syntax match via tree-sitter
# ---------------------------
# We’ll parse both ref & hyp and compute a simple tree-structure similarity:
# similarity = 2 * |common node types| / (|types_ref| + |types_hyp|)
#
# This is a lightweight proxy to CodeXGLUE's more involved syntax match and is
# consistent & stable across environments (no custom shared objects needed).

def _python_language_capsule():
    # For tree-sitter-python>=0.23, tsp.language() returns a PyCapsule the Parser understands
    return tsp.language()

def _parser():
    p = Parser()
    # Newer tree_sitter allows setting capsule directly:
    #   parser.language = <Language or capsule>
    # If your installed wheel prefers set_language(), we’ll try that too.
    lang = _python_language_capsule()
    try:
        p.language = lang
    except Exception:
        p.set_language(lang)
    return p

def _node_types_from_code(code: str) -> List[str]:
    p = _parser()
    tree = p.parse(bytes(code, "utf-8"))
    types = []

    def walk(node):
        types.append(node.type)
        for i in range(node.child_count):
            walk(node.children[i])

    walk(tree.root_node)
    return types

def syntax_match_score(refs: List[str], hyps: List[str]) -> float:
    import collections
    scores = []
    for r, h in zip(refs, hyps):
        types_r = collections.Counter(_node_types_from_code(r))
        types_h = collections.Counter(_node_types_from_code(h))

        # multiset intersection size
        common = 0
        for t, cr in types_r.items():
            if t in types_h:
                common += min(cr, types_h[t])

        total = sum(types_r.values()) + sum(types_h.values())
        s = (2.0 * common / total) if total > 0 else 1.0
        scores.append(s)

    return float(sum(scores) / max(len(scores), 1))


# ---------------------------
# Dataflow match (stub=0.0)
# ---------------------------
def dataflow_match_score(refs: List[str], hyps: List[str]) -> float:
    # Proper DFG extraction requires a custom analyzer; we return 0.0 to keep the
    # metric defined. If you later integrate a DFG extractor, wire it here.
    return 0.0


# ---------------------------
# Public API (drop-in)
# ---------------------------
def calc_codebleu(refs: List[str], hyps: List[str], lang: str = "python", weights=(0.25, 0.25, 0.25, 0.25)) -> Dict[str, float]:
    if lang.lower() not in {"python", "py"}:
        raise NotImplementedError("codebleu_compat.py currently supports lang='python' only.")

    ngram = ngram_match_score(refs, hyps)
    weighted = weighted_ngram_match_score(refs, hyps)
    syntax = syntax_match_score(refs, hyps)
    dataflow = dataflow_match_score(refs, hyps)

    w1, w2, w3, w4 = weights
    overall = w1 * ngram + w2 * weighted + w3 * syntax + w4 * dataflow

    return {
        "codebleu": float(overall),
        "ngram_match": float(ngram),
        "weighted_ngram_match": float(weighted),
        "syntax_match": float(syntax),
        "dataflow_match": float(dataflow),
    }