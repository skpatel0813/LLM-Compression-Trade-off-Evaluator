# src/codebleu_shim.py
# Make CodeBLEU (PyPI 0.7.0) work with modern tree-sitter wheels on Py3.11+
# - Replaces codebleu.utils.get_tree_sitter_language with a version that uses tree_sitter_languages
# - Adds a .language property to Parser for packages that only expose .set_language()

from __future__ import annotations

# 1) Patch Parser.language <-> set_language
try:
    from tree_sitter import Parser
except Exception as e:
    raise RuntimeError("tree_sitter is not installed. `pip install tree_sitter`") from e

if not hasattr(Parser, "language"):
    # Older/newer bindings don’t expose a .language property; emulate it.
    _Parser_set_language = getattr(Parser, "set_language", None)
    if _Parser_set_language is None:
        raise RuntimeError(
            "Your tree_sitter.Parser has neither `.language` nor `.set_language()`; "
            "please reinstall tree_sitter."
        )

    def _get_lang(self):  # type: ignore[override]
        # Not strictly needed by CodeBLEU, but helps debugging
        return getattr(self, "_ts_lang", None)

    def _set_lang(self, lang):  # type: ignore[override]
        _Parser_set_language(self, lang)
        setattr(self, "_ts_lang", lang)

    Parser.language = property(_get_lang, _set_lang)  # type: ignore[attr-defined]

# 2) Patch CodeBLEU’s language resolver
try:
    from tree_sitter_languages import get_language as _ts_get_language
except Exception as e:
    raise RuntimeError(
        "Missing dependency `tree_sitter_languages`. "
        "Install with: pip install tree_sitter_languages==1.10.2"
    ) from e

# Import AFTER patching Parser, then monkey-patch CodeBLEU utils
from codebleu import utils as _cb_utils  # type: ignore


_LANG_ALIASES = {
    "python": ("python", "py"),
    "java": ("java",),
    "javascript": ("javascript", "js"),
    "cpp": ("cpp", "c++"),
    "c": ("c",),
    "go": ("go", "golang"),
    "ruby": ("ruby", "rb"),
    "php": ("php",),
    "csharp": ("c_sharp", "csharp", "c#"),
    "rust": ("rust",),
    "scala": ("scala",),
    "swift": ("swift",),
    "kotlin": ("kotlin",),
    "typescript": ("typescript", "ts"),
}

def _normalize_lang(lang: str) -> str:
    key = (lang or "").strip().lower()
    for std, aliases in _LANG_ALIASES.items():
        if key == std or key in aliases:
            return std
    return key

def _patched_get_ts_language(lang: str):
    """
    Return a tree-sitter Language object using tree_sitter_languages,
    compatible with Parser.language = <Language>.
    """
    key = _normalize_lang(lang)
    try:
        return _ts_get_language(key)
    except Exception as e:
        raise RuntimeError(
            f"tree_sitter_languages.get_language failed for '{lang}' (normalized '{key}')."
        ) from e

# Monkey-patch CodeBLEU’s resolver
_cb_utils.get_tree_sitter_language = _patched_get_ts_language  # type: ignore[attr-defined]
