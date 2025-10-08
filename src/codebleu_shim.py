# src/codebleu_shim.py
# Make CodeBLEU (PyPI 0.7.0) work with modern tree-sitter wheels on Py3.11+.
# Patches BOTH codebleu.utils.get_tree_sitter_language and
# codebleu.codebleu.get_tree_sitter_language, and provides Parser.language property.

from __future__ import annotations

# --- Patch tree_sitter.Parser to have a .language property (maps to .set_language)
try:
    from tree_sitter import Parser
except Exception as e:
    raise RuntimeError("tree_sitter is not installed. `pip install tree_sitter`") from e

if not hasattr(Parser, "language"):
    _set_language = getattr(Parser, "set_language", None)
    if _set_language is None:
        raise RuntimeError(
            "Your tree_sitter.Parser has neither `.language` nor `.set_language()`; "
            "please reinstall tree_sitter."
        )

    def _get_lang(self):
        return getattr(self, "_ts_lang", None)

    def _set_lang(self, lang):
        _set_language(self, lang)
        setattr(self, "_ts_lang", lang)

    Parser.language = property(_get_lang, _set_lang)  # type: ignore[attr-defined]

# --- Use tree_sitter_languages for robust Language retrieval
try:
    from tree_sitter_languages import get_language as _ts_get_language
except Exception as e:
    raise RuntimeError(
        "Missing dependency `tree_sitter_languages`. "
        "Install with: pip install tree_sitter_languages==1.10.2"
    ) from e

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
    key = _normalize_lang(lang)
    try:
        return _ts_get_language(key)
    except Exception as e:
        raise RuntimeError(
            f"tree_sitter_languages.get_language failed for '{lang}' (normalized '{key}')."
        ) from e

# --- Patch BOTH modules inside CodeBLEU
import importlib
_cb_utils = importlib.import_module("codebleu.utils")
_cb_mod   = importlib.import_module("codebleu.codebleu")

# Replace the resolver everywhere CodeBLEU looks:
_cb_utils.get_tree_sitter_language = _patched_get_ts_language          # utils
_cb_mod.get_tree_sitter_language   = _patched_get_ts_language          # codebleu (bound at import)
