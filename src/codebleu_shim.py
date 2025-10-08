from __future__ import annotations
import importlib

# --- A) Make Parser.language work across tree-sitter versions ---
try:
    from tree_sitter import Parser
except Exception as e:
    raise RuntimeError(f"Failed to import tree_sitter.Parser: {e}")

if not hasattr(Parser, "language"):
    def _get_lang(self):
        return getattr(self, "_shim_lang", None)
    def _set_lang(self, lang):
        self.set_language(lang)
        self._shim_lang = lang
    Parser.language = property(_get_lang, _set_lang)  # type: ignore[attr-defined]

# --- B) A resolver that returns a LANGUAGE object directly (Python only for now) ---
def _get_ts_lang(lang_name: str):
    key = (lang_name or "").strip().lower()
    if key in {"py", "python"}:
        tsp = importlib.import_module("tree_sitter_python")
        return tsp.language()  # IMPORTANT: return the LANGUAGE object
    raise NotImplementedError(
        f"CodeBLEU shim: language '{lang_name}' not supported. Only 'python' implemented."
    )

# --- C) Patch BOTH the utils module AND the codebleu.py local binding ---
import codebleu.utils as cb_utils
cb_utils.get_tree_sitter_language = _get_ts_lang

cb_code = importlib.import_module("codebleu.codebleu")
cb_code.get_tree_sitter_language = _get_ts_lang
