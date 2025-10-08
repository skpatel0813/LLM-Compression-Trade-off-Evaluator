# Replace your current codebleu_shim.py with this enhanced version

from __future__ import annotations
import sys

try:
    from tree_sitter import Parser, Language
except Exception as e:
    raise RuntimeError("tree_sitter is not installed. `pip install tree_sitter`") from e

# Patch Parser.language property if needed
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

    Parser.language = property(_get_lang, _set_lang)

# Try multiple approaches to get language
def _patched_get_ts_language(lang: str):
    key = lang.lower().strip()
    
    # Approach 1: Try tree_sitter_languages first
    try:
        from tree_sitter_languages import get_language as _ts_get_language
        return _ts_get_language(key)
    except Exception as e1:
        print(f"tree_sitter_languages failed: {e1}", file=sys.stderr)
    
    # Approach 2: Try manual loading for common languages
    try:
        if key == "python":
            from tree_sitter_languages.parsers.python import Python
            return Python
        elif key == "java":
            from tree_sitter_languages.parsers.java import Java
            return Java
        elif key == "javascript":
            from tree_sitter_languages.parsers.javascript import Javascript
            return Javascript
    except ImportError:
        pass
    
    # Approach 3: Last resort - build from source if available
    try:
        # This would require the tree-sitter-* repos to be cloned locally
        # For now, we'll raise a clear error
        raise RuntimeError(
            f"Could not load parser for '{lang}'. "
            f"Try: pip install --force-reinstall tree-sitter-languages "
            f"or manually install tree-sitter-{key}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to get tree-sitter language for '{lang}': {e}")

# Patch the CodeBLEU modules
import importlib
_cb_utils = importlib.import_module("codebleu.utils")
_cb_mod = importlib.import_module("codebleu.codebleu")

_cb_utils.get_tree_sitter_language = _patched_get_ts_language
_cb_mod.get_tree_sitter_language = _patched_get_ts_language

print("CodeBLEU shim loaded successfully", file=sys.stderr)