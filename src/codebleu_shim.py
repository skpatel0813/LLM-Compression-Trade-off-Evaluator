# codebleu_shim.py
from __future__ import annotations
import re

# Try to get a ready-made Python grammar first
_LANG = None
try:
    from tree_sitter import Parser
    from tree_sitter_languages import get_language
    _LANG = get_language('python')
    _PARSER = Parser()
    _PARSER.set_language(_LANG)
except Exception:
    # Fallback: try a compiled .so if user built one (see step 1B)
    try:
        from tree_sitter import Language, Parser
        _LANG = Language('build/my-languages.so', 'python')
        _PARSER = Parser()
        _PARSER.set_language(_LANG)
    except Exception:
        _PARSER = None  # syntax component will be skipped

# Simple cleaners to avoid failing on fenced code blocks, trailing prompts, etc.
_FENCE_RE = re.compile(r"^```(?:python)?\s*|\s*```$", re.MULTILINE)
def _clean(code: str) -> str:
    return _FENCE_RE.sub("", code).strip()

def parse_ok(code: str) -> bool:
    if _PARSER is None:
        return False
    try:
        tree = _PARSER.parse(bytes(_clean(code), "utf8"))
        # A tiny sanity check: non-empty file should yield at least a root with children
        return tree.root_node is not None and len(tree.root_node.children) > 0
    except Exception:
        return False

# ---- Patch CodeBLEU internals ----
def patch_codebleu():
    # Import *after* we set globals so CodeBLEU sees our parser helpers
    import codebleu
    # Monkey-patch places CodeBLEU uses to compute the syntax score.
    # Most public CodeBLEU forks look for "calc_syntax_match" and/or run a parser behind the scenes.
    try:
        from codebleu import syntax_match

        _orig = syntax_match.calc_syntax_match

        def _calc_syntax_match_python(references, candidates, lang="python"):
            # If we don’t have a parser, return zeros so overall score is still computed.
            if _PARSER is None:
                return [0.0] * len(candidates)
            out = []
            for ref, hyp in zip(references, candidates):
                ok_ref = parse_ok(ref)
                ok_hyp = parse_ok(hyp)
                # Very simple signal: 1.0 only if both parse; else 0.0
                # (You can replace this with a tree-edit similarity if you’d like.)
                out.append(1.0 if (ok_ref and ok_hyp) else 0.0)
            return out

        def _wrapped(references, candidates, lang):
            if lang.lower() == "python":
                return _calc_syntax_match_python(references, candidates, lang="python")
            return _orig(references, candidates, lang)

        syntax_match.calc_syntax_match = _wrapped
    except Exception:
        # If API differs (other forks), fail soft: CodeBLEU main score will still run.
        pass

    return True
