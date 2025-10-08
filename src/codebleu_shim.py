# src/codebleu_shim.py
# --------------------------------------------------------------------------------------
# Purpose
# -------
# CodeBLEU (PyPI: codebleu==0.7.0) internally expects a tree-sitter Language object,
# and historically wraps a PyCapsule returned by tree_sitter_python.language() with
# Language(...). Across tree-sitter versions this can raise: "TypeError: an integer
# is required" or similar ABI issues.
#
# This shim fixes that by:
# 1) Overriding codebleu.utils.get_tree_sitter_language to return the Python grammar
#    *capsule* directly (not Language(...)).
# 2) Patching tree_sitter.Parser to accept either a Language or the grammar capsule
#    by delegating to set_language() in a safe way.
#
# Usage
# -----
#   import codebleu_shim         # MUST be imported before 'codebleu'
#   from codebleu import calc_codebleu
#
# Supported language(s): python
# --------------------------------------------------------------------------------------

from __future__ import annotations

import importlib
import inspect
import sys

# ---- Import tree_sitter primitives (Parser, Language) --------------------------------
try:
    from tree_sitter import Parser, Language  # type: ignore
except Exception as e:
    raise RuntimeError(f"[codebleu_shim] Failed to import tree_sitter Parser/Language: {e}")

# ---- Helper: normalize a value to either Language or "capsule" -----------------------
def _split_lang_or_capsule(obj):
    """
    Return (lang_obj, capsule_or_none).

    - If 'obj' is a tree_sitter.Language → (obj, None)
    - Otherwise we treat it as a PyCapsule originating from tree_sitter_python.language()
      → (None, obj)
    """
    if isinstance(obj, Language):
        return obj, None
    return None, obj

# ---- Property hook: Parser.language setter that accepts both types -------------------
def _get_lang_prop(self: Parser):
    # CodeBLEU never reads this; present only for completeness.
    return None

def _set_lang_prop(self: Parser, val):
    """
    Accept either:
      - Language instance → self.set_language(Language)
      - PyCapsule (from tree_sitter_python.language()) → convert via tree_sitter_python and set
    """
    lang_obj, capsule = _split_lang_or_capsule(val)
    if lang_obj is not None:
        self.set_language(lang_obj)
        return

    # Capsule path: load tree_sitter_python and set via its language() (ABI-safe).
    try:
        tsp = importlib.import_module("tree_sitter_python")
    except Exception as e:
        raise RuntimeError(
            "[codebleu_shim] Missing 'tree_sitter_python'. Install with:\n"
            "  pip install tree-sitter-python\n"
            f"Original error: {e}"
        ) from e

    # Note: We do NOT try to wrap the capsule with Language(...).
    # We simply ask the official tree_sitter_python to provide the compatible object.
    self.set_language(tsp.language())

# Install the patched property once
if not hasattr(Parser, "_codebleu_shim_installed"):
    try:
        Parser.language = property(_get_lang_prop, _set_lang_prop)  # type: ignore[attr-defined]
        Parser._codebleu_shim_installed = True  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"[codebleu_shim] Failed to install Parser.language patch: {e}")

# ---- Patch codebleu.utils.get_tree_sitter_language -----------------------------------
def _get_ts_lang_python_only(lang: str):
    """
    Replacement for codebleu.utils.get_tree_sitter_language(lang).

    Returns the Python grammar *capsule* directly from tree_sitter_python.
    The Parser.language property we installed will handle this safely.
    """
    key = (lang or "").strip().lower()
    if key not in {"py", "python"}:
        raise NotImplementedError(
            f"[codebleu_shim] Only 'python' is supported by this shim. Got '{lang}'."
        )
    try:
        tsp = importlib.import_module("tree_sitter_python")
    except Exception as e:
        raise RuntimeError(
            "[codebleu_shim] Could not import 'tree_sitter_python'. Install with:\n"
            "  pip install tree-sitter-python"
        ) from e
    return tsp.language()  # return capsule; our Parser.language setter handles it

def _apply_patch_now():
    try:
        import codebleu.utils as u
    except Exception:
        return None
    try:
        u.get_tree_sitter_language = _get_ts_lang_python_only  # type: ignore[attr-defined]
        return u
    except Exception:
        return None

# Try patch immediately (if utils already imported)
_utils_mod = _apply_patch_now()

# Also register a lazy import hook: if codebleu.utils is imported later, re-apply patch.
class _CodeBLEUShimImporter:
    def find_spec(self, fullname, path, target=None):
        # We only care when codebleu.utils is being imported
        if fullname == "codebleu.utils":
            # After import, set our function
            spec = importlib.util.find_spec(fullname)
            if spec and spec.loader:
                original_exec = spec.loader.exec_module  # type: ignore[attr-defined]

                def _exec_and_patch(mod):
                    original_exec(mod)
                    try:
                        mod.get_tree_sitter_language = _get_ts_lang_python_only  # type: ignore[attr-defined]
                    except Exception:
                        pass

                spec.loader.exec_module = _exec_and_patch  # type: ignore[attr-defined]
            return spec
        return None

# Install the lazy patcher if needed
if _utils_mod is None:
    try:
        sys.meta_path.insert(0, _CodeBLEUShimImporter())
    except Exception:
        # Non-fatal; best effort
        pass

# ---- Self-check to confirm patch took effect (printed once to stderr) ----------------
try:
    import codebleu.utils as _uc
    src = inspect.getsource(_uc.get_tree_sitter_language)
    if "_get_ts_lang_python_only" in src:
        sys.stderr.write("[codebleu_shim] patched get_tree_sitter_language ✓\n")
    else:
        sys.stderr.write("[codebleu_shim] WARNING: patch may not be active. "
                         "Ensure 'import codebleu_shim' happens BEFORE 'from codebleu ...'\n")
except Exception:
    # Silent if codebleu not yet imported; lazy patch will handle it later.
    pass
