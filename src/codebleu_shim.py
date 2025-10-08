# src/codebleu_shim.py
# Patch CodeBLEU so it uses tree-sitter-python with tree-sitter==0.22.x
# and so codebleu's "parser.language = ..." line works by delegating to set_language(...).

from __future__ import annotations
import importlib
from tree_sitter import Parser

# --- Patch Parser.language property to forward to set_language(...) ---
# CodeBLEU does:
#   parser.language = <lang>
# In tree-sitter 0.22.x, official API is parser.set_language(<Language>).
# Also, tree_sitter_python.language() returns a PyCapsule, not a Language.
# We accept either and convert capsules to Language via tree_sitterPython.

def _to_language(obj):
    """
    Accept either a tree_sitter.Language or a PyCapsule from tree_sitter_python.language().
    If it's a capsule, use tree_sitter_python to set it directly.
    """
    # If it's already a Language, return as-is.
    from tree_sitter import Language
    if isinstance(obj, Language):
        return obj, None  # (Language, capsule)

    # Otherwise we expect a PyCapsule from tree_sitter_python.language()
    # We return (None, capsule) to let the setter handle it via tsp.
    return None, obj

@property
def _get_language_prop(self: Parser):
    # not used by CodeBLEU; return None to keep behavior minimal
    return None

@_get_language_prop.setter  # type: ignore[attr-defined]
def _set_language_prop(self: Parser, val):
    lang_obj, capsule = _to_language(val)
    if lang_obj is not None:
        # Normal: already a Language
        self.set_language(lang_obj)
        return
    # Capsule path: use tree_sitter_python to set the capsule directly.
    tsp = importlib.import_module("tree_sitter_python")
    # tree_sitter 0.22.x Parser has a C-API that accepts the capsule via set_language
    # (works when capsule comes from the same versioned wheels)
    self.set_language(tsp.language())

# Install the patched property
try:
    Parser.language = _get_language_prop  # type: ignore[attr-defined]
except Exception:
    pass

# --- Patch codebleu.utils.get_tree_sitter_language to return the python grammar ---
# CodeBLEU calls this and then assigns to parser.language (which we just hooked).
try:
    import codebleu.utils as _u
except Exception:
    _u = None

def _patched_get_ts_language(lang: str):
    key = (lang or "").strip().lower()
    if key in {"py", "python"}:
        tsp = importlib.import_module("tree_sitter_python")
        return tsp.language()  # PyCapsule; our setter handles it
    raise NotImplementedError(
        f"Only 'python' is implemented in codebleu_shim; requested '{lang}'."
    )

if _u is not None:
    try:
        _u.get_tree_sitter_language = _patched_get_ts_language  # type: ignore[attr-defined]
    except Exception:
        pass
