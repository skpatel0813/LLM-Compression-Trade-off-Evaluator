# src/codebleu_shim.py - Enhanced version for compatibility
from __future__ import annotations
import sys
import os

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

def _patched_get_ts_language(lang: str):
    key = lang.lower().strip()
    print(f"DEBUG: Loading language for '{key}'", file=sys.stderr)
    
    # Approach 1: Try tree_sitter_languages with error handling
    try:
        from tree_sitter_languages import get_language as _ts_get_language
        print("DEBUG: Using tree_sitter_languages", file=sys.stderr)
        return _ts_get_language(key)
    except TypeError as e:
        # This is the specific error we're seeing - try alternative approach
        print(f"DEBUG: tree_sitter_languages failed with TypeError: {e}", file=sys.stderr)
        
        # Try importing the language directly from the parsers module
        try:
            if key == "python":
                import tree_sitter_languages.parsers.python
                return tree_sitter_languages.parsers.python.Python
            elif key == "java":
                import tree_sitter_languages.parsers.java  
                return tree_sitter_languages.parsers.java.Java
            elif key == "javascript":
                import tree_sitter_languages.parsers.javascript
                return tree_sitter_languages.parsers.javascript.Javascript
            elif key == "cpp":
                import tree_sitter_languages.parsers.cpp
                return tree_sitter_languages.parsers.cpp.Cpp
            elif key == "c":
                import tree_sitter_languages.parsers.c
                return tree_sitter_languages.parsers.c.C
        except ImportError as ie:
            print(f"DEBUG: Direct import failed: {ie}", file=sys.stderr)
    
    except Exception as e:
        print(f"DEBUG: tree_sitter_languages failed with: {e}", file=sys.stderr)
    
    # Approach 2: Try building from source if we have the repositories
    try:
        # Check if we have any local tree-sitter repositories
        repo_path = f"tree-sitter-{key}"
        if os.path.exists(repo_path):
            print(f"DEBUG: Building {key} from local repo", file=sys.stderr)
            Language.build_library(f'build/{key}-language.so', [repo_path])
            return Language(f'build/{key}-language.so', key)
    except Exception as e:
        print(f"DEBUG: Build from source failed: {e}", file=sys.stderr)
    
    # Final fallback: Use a minimal implementation that skips syntax matching
    print(f"WARNING: Could not load tree-sitter parser for {key}. Syntax matching will be disabled.", file=sys.stderr)
    
    # Return a dummy language object that won't crash
    class DummyLanguage:
        def __init__(self):
            self.name = f"dummy_{key}"
    
    return DummyLanguage()

# Patch the CodeBLEU modules
import importlib
_cb_utils = importlib.import_module("codebleu.utils")
_cb_mod = importlib.import_module("codebleu.codebleu")

_cb_utils.get_tree_sitter_language = _patched_get_ts_language
_cb_mod.get_tree_sitter_language = _patched_get_ts_language

print("CodeBLEU shim loaded successfully", file=sys.stderr)