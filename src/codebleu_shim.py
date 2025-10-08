# src/codebleu_shim.py
"""
Compat shim so CodeBLEU 0.7.0 works without tree_sitter_python by using
tree_sitter_languages and a Parser.language property.

Patches BOTH:
  - codebleu.utils.get_tree_sitter_language
  - codebleu.codebleu.get_tree_sitter_language
"""

from __future__ import annotations
import re

# 1) get Language objects from tree_sitter_languages
try:
    from tree_sitter_languages import get_language as _ts_get_language
except Exception as e:
    raise RuntimeError(
        "Missing dependency `tree_sitter_languages`. Install with:\n"
        "  pip install tree_sitter_languages==1.10.2"
    ) from e

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

# 2) Apply patches to BOTH modules
import codebleu.utils as _cb_utils
_cb_utils.get_tree_sitter_language = _patched_get_ts_language

import codebleu.codebleu as _cb_main
_cb_main.get_tree_sitter_language = _patched_get_ts_language

# 3) Provide Parser.language write-only property that forwards to set_language
from tree_sitter import Parser as _Parser

def _set_lang(self, lang):
    # CodeBLEU does: parser.language = <Language>
    # For tree_sitter>=0.20, the supported API is set_language(lang)
    self.set_language(lang)

# only a setter; getter not needed
_Parser.language = property(fset=_set_lang)
