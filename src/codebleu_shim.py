# src/codebleu_shim.py
"""
Compatibility shim for codebleu==0.7.0 with modern tree-sitter wheels.

- Forces CodeBLEU to use `tree_sitter_languages.get_language(...)`
- Emulates the old `parser.language = lang` assignment that CodeBLEU expects,
  on top of the current `Parser.set_language(lang)` API.
"""

from __future__ import annotations

# 1) Patch how CodeBLEU loads languages
try:
    from tree_sitter_languages import get_language as _ts_get_language
except Exception as e:
    raise RuntimeError(
        "Missing dependency `tree_sitter_languages`. "
        "Install with: pip install tree_sitter_languages==1.10.2"
    ) from e

from codebleu import utils as _utils

def _patched_get_ts_language(lang: str):
    # normalize common names to tree-sitter keys; extend as you need
    alias = {
        "py": "python",
        "python": "python",
        "java": "java",
        "js": "javascript",
        "javascript": "javascript",
        "cpp": "cpp",
        "c++": "cpp",
        "c": "c",
        "go": "go",
        "ruby": "ruby",
        "rust": "rust",
        "php": "php",
        "scala": "scala",
        "csharp": "c_sharp",
        "c-sharp": "c_sharp",
        "c#": "c_sharp",
        "kotlin": "kotlin",
        "typescript": "typescript",
    }
    key = alias.get(lang.lower(), lang.lower())
    return _ts_get_language(key)

_utils.get_tree_sitter_language = _patched_get_ts_language  # monkey-patch


# 2) Patch Parser usage in modules that assume old API (parser.language = lang)
from tree_sitter import Parser as _Parser
from codebleu import syntax_match as _syntax
from codebleu import dataflow_match as _dataflow

class _ParserCompat:
    """
    Lightweight wrapper over tree_sitter.Parser that exposes a .language property
    setter, mapping it to .set_language() (new API).
    """
    def __init__(self):
        self._p = _Parser()

    # CodeBLEU writes: parser.language = <Language>
    @property
    def language(self):
        # Not used by CodeBLEU, but kept for completeness
        return getattr(self, "_lang", None)

    @language.setter
    def language(self, lang):
        self._p.set_language(lang)
        self._lang = lang

    # CodeBLEU calls: parser.parse(src_bytes)
    def parse(self, *args, **kwargs):
        return self._p.parse(*args, **kwargs)

# swap Parser in the two modules that instantiate it
_syntax.Parser = _ParserCompat
_dataflow.Parser = _ParserCompat
