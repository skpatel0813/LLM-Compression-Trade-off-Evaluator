"""
Robust shim for CodeBLEU on Python:

- Always returns a *tree_sitter.Language* object (never a PyCapsule).
- Patches BOTH:
    * codebleu.utils.get_tree_sitter_language
    * codebleu.codebleu.get_tree_sitter_language
- Does not depend on tree_sitter_languages (since that varied for you).

Test it with:
  PYTHONPATH="src:$PYTHONPATH" python - <<'PY'
  import codebleu_shim
  from codebleu import calc_codebleu
  refs = ["def add(a, b):\n    return a + b\n"]
  hyps = ["def add(a,b):\n    return a+b\n"]
  print(calc_codebleu(refs, hyps, lang="python"))
  PY
"""

from __future__ import annotations
import importlib

# We need these types available to manufacture a proper Language object
try:
    from tree_sitter import Language
except Exception as e:
    raise RuntimeError(f"[codebleu_shim] Failed to import tree_sitter.Language: {e}")

def _get_ts_lang(lang_name: str):
    """
    Return a **tree_sitter.Language** object for the requested language.

    We only implement Python here because that’s what you’re evaluating.
    If you later need more languages, add similar logic for their grammar wheels
    (e.g., tree_sitter_javascript, tree_sitter_cpp, etc.).
    """
    key = (lang_name or "").strip().lower()

    if key in {"py", "python"}:
        # 1) Prefer the modern API: module exposes a Language instance as `LANGUAGE`
        try:
            tsp = importlib.import_module("tree_sitter_python")
        except Exception as e:
            raise RuntimeError(
                "[codebleu_shim] Could not import 'tree_sitter_python'. "
                "Install it with: pip install tree-sitter-python"
            ) from e

        lang_obj = getattr(tsp, "LANGUAGE", None)
        if lang_obj is not None:
            # Already a proper tree_sitter.Language in current tree_sitter>=0.22
            return lang_obj

        # 2) Older API: module exposes a PyCapsule via `language()` → we must wrap it
        if hasattr(tsp, "language"):
            try:
                capsule = tsp.language()  # PyCapsule
                # Convert PyCapsule to a Language object (works for tree_sitter>=0.22)
                return Language(capsule)
            except Exception as e:
                raise RuntimeError(
                    "[codebleu_shim] Could not convert Python grammar capsule to tree_sitter.Language. "
                    "This usually means a binary mismatch between 'tree-sitter' and 'tree-sitter-python'. "
                    "Try reinstalling both in the SAME env, e.g.\n"
                    "  pip install -U 'tree-sitter==0.22.3' 'tree-sitter-python==0.23.4'\n"
                ) from e

        # 3) No known symbol found
        raise RuntimeError(
            "[codebleu_shim] tree_sitter_python does not expose 'LANGUAGE' or 'language()'. "
            "Please upgrade: pip install -U tree-sitter-python"
        )

    # Not implemented languages (extend as needed)
    raise NotImplementedError(
        f"[codebleu_shim] Language '{lang_name}' not supported in shim (only 'python' implemented)."
    )

# --- Patch BOTH places CodeBLEU reads the resolver ---
# (1) codebleu.utils.get_tree_sitter_language
import codebleu.utils as _cb_utils
_cb_utils.get_tree_sitter_language = _get_ts_lang

# (2) codebleu.codebleu.get_tree_sitter_language (local binding used by calc_codebleu)
_cb_code = importlib.import_module("codebleu.codebleu")
_cb_code.get_tree_sitter_language = _get_ts_lang
