# src/codebleu_shim.py
# --------------------------------------------------------------------------------------
# CodeBLEU (codebleu==0.7.0) expects a tree_sitter.Language for Parser.language,
# but the canonical `tree_sitter_python.language()` returns a PyCapsule in many builds,
# which leads to: "TypeError: an integer is required" or similar ABI issues.
#
# This shim:
#   1) Uses `tree_sitter_languages` (prebuilt grammars) to obtain a REAL
#      `tree_sitter.Language` for Python (and later, other languages, if you add them).
#   2) Patches BOTH:
#        - codebleu.utils.get_tree_sitter_language
#        - codebleu.codebleu.get_tree_sitter_language
#      so that calc_codebleu() uses our safe function.
#   3) Verifies the patch on import and prints a short status line to stderr.
#
# Requirements (install in the SAME env as CodeBLEU):
#   pip install "tree_sitter==0.20.2" "tree_sitter_languages==1.10.2"
#   pip install "codebleu==0.7.0"
#
# Usage (MUST import shim *before* codebleu):
#   import codebleu_shim
#   from codebleu import calc_codebleu
#   scores = calc_codebleu(refs, hyps, lang="python")
# --------------------------------------------------------------------------------------

from __future__ import annotations
import sys
import inspect
import importlib
import importlib.util
from typing import Optional

# ---- Hard dependency we rely on: tree_sitter_languages gives us Language objects directly.
try:
    from tree_sitter_languages import get_language as _ts_get_language  # returns tree_sitter.Language
except Exception as e:
    raise RuntimeError(
        "[codebleu_shim] Failed importing `tree_sitter_languages`. Install with:\n"
        "  pip install tree_sitter_languages==1.10.2\n"
        "and ensure it's in the SAME environment as codebleu/transformers."
    ) from e


# ---- our safe replacement: return a real tree_sitter.Language for the requested lang
def _get_ts_language_safely(lang: Optional[str]):
    key = (lang or "").strip().lower()
    if key in {"py", "python"}:
        return _ts_get_language("python")   # <class 'tree_sitter.Language'>
    # Extend here as needed (examples):
    # elif key in {"js", "javascript"}:
    #     return _ts_get_language("javascript")
    # elif key in {"java"}:
    #     return _ts_get_language("java")
    raise NotImplementedError(
        f"[codebleu_shim] Only 'python' is supported right now. Got '{lang}'. "
        f"Add your language to _get_ts_language_safely if needed."
    )


def _patch_module_getter(mod) -> bool:
    """
    Replace mod.get_tree_sitter_language with our safe version.
    Returns True if patched, False otherwise.
    """
    try:
        setattr(mod, "get_tree_sitter_language", _get_ts_language_safely)
        return True
    except Exception:
        return False


def _eager_patch() -> tuple[bool, bool]:
    """
    Try to patch both codebleu.utils and codebleu.codebleu immediately (if already imported).
    Returns (patched_utils, patched_codebleu).
    """
    pu = pc = False
    try:
        import codebleu.utils as cu
        pu = _patch_module_getter(cu)
    except Exception:
        pass
    try:
        import codebleu.codebleu as ccb
        pc = _patch_module_getter(ccb)
    except Exception:
        pass
    return pu, pc


# ---- Patch already-imported modules (if any)
_pu, _pc = _eager_patch()

# ---- Also install a meta-path hook to patch future imports of those modules
class _ShimImporter:
    """
    If `codebleu.utils` or `codebleu.codebleu` are imported AFTER this shim,
    we intercept, let Python import them, then patch their `get_tree_sitter_language`.
    """
    TARGETS = {"codebleu.utils", "codebleu.codebleu"}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.TARGETS:
            return None
        spec = importlib.util.find_spec(fullname)
        if not spec or not spec.loader:
            return spec
        _orig_exec = spec.loader.exec_module  # type: ignore[attr-defined]

        def _exec_and_patch(module):
            _orig_exec(module)
            _patch_module_getter(module)

        spec.loader.exec_module = _exec_and_patch  # type: ignore[attr-defined]
        return spec

# install the hook once
if not any(isinstance(h, _ShimImporter) for h in sys.meta_path):
    sys.meta_path.insert(0, _ShimImporter())

# ---- Print a quick verification message
def _verify_and_note():
    msg = "[codebleu_shim] "
    try:
        import codebleu.utils as cu
        import codebleu.codebleu as ccb
        ok_u = "_get_ts_language_safely" in inspect.getsource(cu.get_tree_sitter_language)
        ok_c = "_get_ts_language_safely" in inspect.getsource(ccb.get_tree_sitter_language)
        if ok_u and ok_c:
            msg += "patched utils & codebleu ✓\n"
        elif ok_u:
            msg += "patched utils ✓ (codebleu will be patched on import)\n"
        elif ok_c:
            msg += "patched codebleu ✓ (utils will be patched on import)\n"
        else:
            msg += "ready to patch on import…\n"
    except Exception:
        msg += "ready to patch on import…\n"
    sys.stderr.write(msg)

_verify_and_note()
