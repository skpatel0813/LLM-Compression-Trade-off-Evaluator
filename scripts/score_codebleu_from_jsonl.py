#!/usr/bin/env python
import json, sys
from pathlib import Path

# ---- Try both possible import layouts for CodeBLEU
_calc = None
try:
    from codebleu.calc_codebleu import calc_codebleu as _calc
except Exception:
    try:
        from codebleu import calc_codebleu as _calc
    except Exception:
        _calc = None

# Try to patch CodeBLEU's language resolver to use tree_sitter_languages
_ts_ready = False
try:
    import codebleu.utils as _cb_utils
    from tree_sitter_languages import get_language as _get_ts_lang
    def _patched_get_tree_sitter_language(lang: str):
        alias = {
            "py": "python", "python": "python",
            "js": "javascript", "javascript": "javascript",
            "cpp": "cpp", "c++": "cpp",
            "java": "java", "go": "go",
        }.get(lang.lower(), lang.lower())
        return _get_ts_lang(alias)
    _cb_utils.get_tree_sitter_language = _patched_get_tree_sitter_language
    _ts_ready = True
except Exception:
    _ts_ready = False

def load_pairs(path):
    refs, hyps = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ref = r.get("reference") or r.get("target") or ""
            hyp = r.get("prediction") or r.get("output") or ""
            refs.append(ref)
            hyps.append(hyp)
    return refs, hyps

def main(inp, outp):
    refs, hyps = load_pairs(inp)
    lang = "python"
    weights = (0.25, 0.25, 0.25, 0.25)

    if _calc is not None:
        try:
            # full CodeBLEU (if syntax/dataflow are available via our patch)
            sco = _calc(refs, hyps, lang, weights=weights, tokenizer=None)
            metrics = {
                "codebleu": float(sco.get("codebleu", 0.0)),
                "codebleu_ngram": float(sco.get("ngram_match_score", 0.0)),
                "codebleu_weighted_ngram": float(sco.get("weighted_ngram_match_score", 0.0)),
                "codebleu_syntax": float(sco.get("syntax_match_score", 0.0)),
                "codebleu_dataflow": float(sco.get("dataflow_match_score", 0.0)),
                "bleu4": float(sco.get("bleu", 0.0)),
                "rows": len(refs),
                "predictions_in": inp,
                "syntax_engine": "tree_sitter_languages" if _ts_ready else "codebleu_default",
            }
        except Exception as e:
            # graceful fallback: BLEU + weighted/ngram only (syntax/dataflow=0)
            from codebleu import bleu, weighted_ngram_match
            bleu_score = bleu.corpus_bleu(refs, hyps, smooth=True)
            ngram_match_score = weighted_ngram_match.corpus_ngram_match(refs, hyps)
            weighted_ngram_score = weighted_ngram_match.corpus_weighted_ngram_match(refs, hyps)
            codebleu_no_syntax = 0.25*ngram_match_score + 0.25*weighted_ngram_score
            metrics = {
                "codebleu": float(codebleu_no_syntax),
                "codebleu_ngram": float(ngram_match_score),
                "codebleu_weighted_ngram": float(weighted_ngram_score),
                "codebleu_syntax": 0.0,
                "codebleu_dataflow": 0.0,
                "bleu4": float(bleu_score),
                "rows": len(refs),
                "predictions_in": inp,
                "syntax_engine": "fallback_no_syntax",
                "warning": f"syntax/dataflow disabled due to: {type(e).__name__}: {e}",
            }
    else:
        # No CodeBLEU at all — compute BLEU only, so you still get a result
        try:
            import sacrebleu
            bleu_score = sacrebleu.corpus_bleu(hyps, [refs]).score
        except Exception:
            bleu_score = 0.0
        metrics = {
            "codebleu": 0.0,
            "codebleu_ngram": 0.0,
            "codebleu_weighted_ngram": 0.0,
            "codebleu_syntax": 0.0,
            "codebleu_dataflow": 0.0,
            "bleu4": float(bleu_score),
            "rows": len(refs),
            "predictions_in": inp,
            "syntax_engine": "unavailable",
            "warning": "CodeBLEU not importable; BLEU only.",
        }

    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: score_codebleu_from_jsonl.py <clean_predictions.jsonl> <metrics.json>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
