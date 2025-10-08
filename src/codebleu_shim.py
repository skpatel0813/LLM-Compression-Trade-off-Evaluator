# src/codebleu_shim.py
# --------------------------------------------------------------------------------------
# Make CodeBLEU 0.7.0 work reliably with tree_sitter==0.20.x by:
#   1) Adding a Parser.language property that forwards to set_language(Language)
#      AND stores the assigned Language in a WeakKeyDictionary (C objects
#      won’t accept new Python attributes in 0.20.x).
#   2) Replacing codebleu.utils.get_tree_sitter_language with a version that
#      returns a real tree_sitter.Language via tree_sitter_languages for Python.
#   3) Also patching codebleu.codebleu in case it imports the original function.
#
# Supports: lang="python".
# Requires: pip install codebleu==0.7.0 tree_sitter_languages==1.10.2
# --------------------------------------------------------------------------------------

from __future__ import annotations
import importlib
import weakref

# ----- Patch Parser.language (for tree_sitter 0.20.x) -------------------------
try:
    from tree_sitter import Parser, Language as TS_Language  # type: ignore
except Exception as e:
    raise RuntimeError(
        "[codebleu_shim] Failed to import tree_sitter. Ensure it's installed in this env."
    ) from e

# Keep per-Parser assigned Language here (Parser objects can’t be given attrs).
_PARSER_LANG = weakref.WeakKeyDictionary()

if not hasattr(Parser, "language"):
    def _get_lang(self):
        # Return last set Language (or None if never set)
        return _PARSER_LANG.get(self)

    def _set_lang(self, lang):
        if not isinstance(lang, TS_Language):
            raise TypeError("language must be a tree_sitter.Language instance")
        # Call the native API and remember it
        self.set_language(lang)
        _PARSER_LANG[self] = lang

    Parser.language = property(_get_lang, _set_lang)  # type: ignore[attr-defined]
    _patched_parser = True
else:
    _patched_parser = False


# ----- Replacement get_tree_sitter_language (Python only) ---------------------
def _get_ts_lang_from_tsl(lang: str) -> TS_Language:
    """
    Replacement for codebleu.utils.get_tree_sitter_language(lang).
    Returns a real tree_sitter.Language for Python using tree_sitter_languages.
    """
    key = (lang or "").strip().lower()
    if key not in {"python", "py"}:
        raise NotImplementedError(
            f"[codebleu_shim] Only 'python' is supported by this shim. Got '{lang}'."
        )

    try:
        tsl = importlib.import_module("tree_sitter_languages")
    except Exception as e:
        raise RuntimeError(
            "[codebleu_shim] Missing dependency 'tree_sitter_languages'. Install with:\n"
            "  pip install tree_sitter_languages==1.10.2"
        ) from e

    try:
        lang_obj = tsl.get_language("python")  # -> tree_sitter.Language
    except Exception as e:
        raise RuntimeError(
            "[codebleu_shim] tree_sitter_languages.get_language('python') failed."
        ) from e

    if not isinstance(lang_obj, TS_Language):
        raise TypeError(
            "[codebleu_shim] Expected a tree_sitter.Language from tree_sitter_languages"
        )
    return lang_obj


# ----- Patch CodeBLEU to use our loader everywhere ----------------------------
_patched_utils = False
_patched_codebleu = False

try:
    utils = importlib.import_module("codebleu.utils")
    setattr(utils, "get_tree_sitter_language", _get_ts_lang_from_tsl)
    _patched_utils = True
except Exception:
    # CodeBLEU not imported yet or not present; that’s fine
    pass

try:
    cmod = importlib.import_module("codebleu.codebleu")
    if hasattr(cmod, "get_tree_sitter_language"):
        setattr(cmod, "get_tree_sitter_language", _get_ts_lang_from_tsl)
    _patched_codebleu = True
except Exception:
    # Same as above — safe to ignore
    pass

print(
    "[codebleu_shim] ready — "
    f"Parser.language patched={_patched_parser}, "
    f"utils.patched={_patched_utils}, "
    f"codebleu.patched={_patched_codebleu}"
)
