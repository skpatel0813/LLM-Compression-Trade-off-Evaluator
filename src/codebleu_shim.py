# src/codebleu_shim.py
"""
Compatibility shim for CodeBLEU 0.7.0 with modern tree-sitter / tree-sitter-python.

What it does:
  1) Patches codebleu.utils.get_tree_sitter_language(lang) to return the right
     grammar object directly (e.g., tree_sitter_python.language()) instead of wrapping
     with tree_sitter.Language(...), which causes a TypeError with newer wheels.
  2) Adds a property `Parser.language` that proxies assignment to `Parser.set_language(...)`
     so CodeBLEU's `parser.language = ...` code continues to work.

Requirements:
  - codebleu==0.7.0
  - tree-sitter==0.22.3
  - tree-sitter-python==0.23.4
"""

from __future__ import annotations
import importlib
import sys

# 1) Patch Parser.language property to use set_language
try:
    from tree_sitter import Parser
except Exception as e:
    raise RuntimeError(f"Failed to import tree_sitter.Parser: {e}")

if not hasattr(Parser, "language"):
    # define a property that stores/returns the last set language
    def _get_lang(self):
        return getattr(self, "_shim_lang", None)

    def _set_lang(self, lang):
        # Newer API wants the grammar object directly
        self.set_language(lang)
        self._shim_lang = lang

    Parser.language = property(_get_lang, _set_lang)  # type: ignore[attr-defined]

# 2) Patch CodeBLEU's get_tree_sitter_language
try:
    import codebleu.utils as cb_utils
except Exception as e:
    raise RuntimeError(f"Failed to import codebleu.utils: {e}")

def _get_ts_lang(lang_name: str):
    """
    Return the correct grammar object for the requested language.
    CodeBLEU calls this and then does: parser.language = <returned value>
    which our property redirects to parser.set_language(...).
    """
    key = (lang_name or "").strip().lower()
    if key in {"py", "python"}:
        tsp = importlib.import_module("tree_sitter_python")
        return tsp.language()

    # You can extend this block with other grammars:
    #   - Java:     pip install tree_sitter_javascript / tree_sitter_java (if available)
    #   - C/C++:    pip install tree_sitter_c / tree_sitter_cpp
    # For now, raise a clear error for unsupported languages.
    raise NotImplementedError(
        f"CodeBLEU shim: language '{lang_name}' not supported. "
        f"Currently implemented: python"
    )

# Monkey-patch
cb_utils.get_tree_sitter_language = _get_ts_lang
