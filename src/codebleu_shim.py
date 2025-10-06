# src/codebleu_shim.py
"""
CodeBLEU <-> Tree-Sitter compatibility shim.

Why: PyPI `codebleu` 0.7.0 tries to import per-language wheels (e.g. tree_sitter_python)
with an older API. Modern setups commonly use `tree_sitter_languages`, which bundles
grammars & exposes a single `get_language()` API. Mixing them causes the infamous:
  TypeError: an integer is required
This shim forces CodeBLEU to use `tree_sitter_languages` and reloads internals
so the change actually takes effect.

Usage:
    import src.codebleu_shim  # must happen BEFORE importing `from codebleu import calc_codebleu`
    from codebleu import calc_codebleu
"""

from importlib import reload

# import codebleu internal utils and tree_sitter_languages
try:
    import codebleu.utils as _cu
    from tree_sitter_languages import get_language as _ts_get_language
except Exception as e:
    raise RuntimeError(
        "Failed importing codebleu utils or tree_sitter_languages. "
        "Install with: pip install codebleu==0.7.0 tree_sitter_languages==1.10.2\n"
        f"Original error: {e}"
    )

# Map common names CodeBLEU expects -> keys accepted by tree_sitter_languages
_TS_LANG_MAP = {
    "python": "python",
    "java": "java",
    "javascript": "javascript",
    "typescript": "typescript",
    "c": "c",
    "cpp": "cpp",
    "go": "go",
    "ruby": "ruby",
    "php": "php",
    "rust": "rust",
    "csharp": "c_sharp",
    "scala": "scala",
    "kotlin": "kotlin",
}


def _patched_get_ts_language(lang: str):
    key = _TS_LANG_MAP.get(str(lang).lower(), str(lang).lower())
    return _ts_get_language(key)


# Monkey-patch, then reload modules that cached the old function.
_cu.get_tree_sitter_language = _patched_get_ts_language

import codebleu.codebleu as _cb  # noqa: E402
reload(_cb)

# Optional quick self-test when run directly.
if __name__ == "__main__":
    from codebleu import calc_codebleu
    refs = ["def add(a, b):\n    return a + b\n"]
    hyps = ["def add(a,b):\n    return a+b\n"]
    out = calc_codebleu(refs, hyps, lang="python")
    print("[shim self-test OK]", {k: round(v, 4) for k, v in out.items()})
