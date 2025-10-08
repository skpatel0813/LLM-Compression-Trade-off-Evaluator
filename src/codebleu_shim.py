# src/codebleu_shim.py
from __future__ import annotations
import importlib

# --- 1) Make Parser.language property work across tree-sitter versions ---
try:
    from tree_sitter import Parser
except Exception as e:
    raise RuntimeError(f"Failed to import tree_sitter.Parser: {e}")

# Some versions don’t ship a .language property; emulate it and redirect to set_language
if not hasattr(Parser, "language"):
    def _get_lang(self):
        return getattr(self, "_shim_lang", None)
    def _set_lang(self, lang):
        # Newer wheels expose LANGUAGE objects directly (e.g., tree_sitter_python.language())
        # Older code expects setting .language; we funnel both to set_language(...)
        self.set_language(lang)
        self._shim_lang = lang
    Parser.language = property(_get_lang, _set_lang)  # type: ignore[attr-defined]

# --- 2) Build a language resolver that returns the LANGUAGE object directly ---
def _get_ts_lang(lang_name: str):
    """
    Return the tree-sitter LANGUAGE object for the requested language.
    We currently implement only Python to keep dependencies minimal.
    """
    key = (lang_name or "").strip().lower()
    if key in {"py", "python"}:
        tsp = importlib.import_module("tree_sitter_python")
        # IMPORTANT: return the object, do NOT wrap in Language(...)
        return tsp.language()
    raise NotImplementedError(
        f"CodeBLEU shim: language '{lang_name}' not supported. "
        f"Currently implemented: python"
    )

# --- 3) Patch BOTH codebleu.utils and codebleu modules ---
# utils
import codebleu.utils as cb_utils
cb_utils.get_tree_sitter_language = _get_ts_lang

# codebleu (it imported the symbol at import time via `from .utils import ...`)
try:
    import codebleu
    # If codebleu already imported the old function, overwrite its binding too
    setattr(codebleu, "get_tree_sitter_language", _get_ts_lang)
except Exception:
    # If codebleu isn’t imported yet, the above will run next time this file is imported.
    pass
