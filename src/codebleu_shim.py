# src/codebleu_shim.py
"""
Compat shim so CodeBLEU 0.7.0 runs on tree-sitter 0.20.x without tree_sitter_python.
- Replaces codebleu.utils.get_tree_sitter_language() with tree_sitter_languages.get_language()
- Adds a .language property on Parser that proxies to .set_language()
"""

from __future__ import annotations
import re

# 1) Use prebuilt grammars from tree_sitter_languages
try:
    from tree_sitter_languages import get_language as _ts_get_language
except Exception as e:
    raise RuntimeError(
        "Missing dependency `tree_sitter_languages`. Install with: "
        "pip install tree_sitter_languages==1.10.2"
    ) from e

# 2) Monkey-patch CodeBLEU's language getter
import codebleu.utils as _cb_utils

_LANG_ALIASES = {
    "py": "python",
    "python3": "python",
}

def _norm_lang(name: str) -> str:
    n = name.strip().lower()
    n = _LANG_ALIASES.get(n, n)
    n = re.sub(r"[^a-z0-9_+-]", "", n)
    return n

def _patched_get_ts_language(lang: str):
    key = _norm_lang(lang)
    try:
        return _ts_get_language(key)
    except Exception as e:
        raise RuntimeError(
            f"tree_sitter_languages.get_language failed for {lang!r} (normalized {key!r})."
        ) from e

_cb_utils.get_tree_sitter_language = _patched_get_ts_language  # <- patch applied

# 3) Give Parser a .language property that forwards to .set_language(...)
from tree_sitter import Parser as _Parser

def _get_lang(self):  # optional getter, not used by CodeBLEU
    return getattr(self, "_compat_lang", None)

def _set_lang(self, lang):
    # tree-sitter 0.20.x has Parser.set_language(Language)
    self.set_language(lang)
    self._compat_lang = lang

# Expose property so codebleu can do: parser.language = <Language>
_Parser.language = property(_get_lang, _set_lang)
