# src/codebleu_shim.py
# --------------------------------------------------------------------------------------
# Make CodeBLEU 0.7.0 work with tree_sitter==0.20.x by:
#   1) Adding a Parser.language property that forwards to set_language(Language)
#      and stores the assigned Language in a global dict keyed by id(parser).
#   2) Replacing codebleu.utils.get_tree_sitter_language to return a real
#      tree_sitter.Language for Python via tree_sitter_languages.
#   3) Also patching codebleu.codebleu in case it imports that symbol directly.
#
# Supports: lang="python".
# Requires: pip install codebleu==0.7.0 tree_sitter_languages==1.10.2
# --------------------------------------------------------------------------------------

from __future__ import annotations
import importlib

try:
    from tree_sitter import Parser, Language as TS_Language  # type: ignore
except Exception as e:
    raise RuntimeError(
        "[codebleu_shim] Failed to import tree_sitter. Install it in this env."
    ) from e

# --- Store assigned languages per Parser (Parser cannot hold new attrs; also not weakref'able) ---
_PARSER_LANG_BY_ID: dict[int, TS_Language] = {}

def _parser_get_lang(self) -> TS_Language | None:
    return _PARSER_LANG_BY_ID.get(id(self))

def _parser_set_lang(self, lang):
    if not isinstance(lang, TS_Language):
        raise TypeError("language must be a tree_sitter.Language instance")
    # call native setter
    self.set_language(lang)
    # remember it (keyed by id(self); tiny leak is acceptable for short-lived eval)
    _PARSER_LANG_BY_ID[id(self)] = lang

# Add property only if missing (tree_sitter 0.20.x)
if not hasattr(Parser, "language"):
    Parser.language = property(_parser_get_lang, _parser_set_lang)  # type: ignore[attr-defined]
    _patched_parser = True
else:
    _patched_parser = False


# --- Replacement for CodeBLEU's language loader (Python only) ---
def _get_ts_lang_from_tsl(lang: str) -> TS_Language:
    key = (lang or "").strip().lower()
    if key not in {"python", "py"}:
        raise NotImplementedError(
            f"[codebleu_shim] Only 'python' is supported by this shim. Got '{lang}'."
        )
    try:
        tsl = importlib.import_module("tree_sitter_languages")
    except Exception as e:
        raise RuntimeError(
            "[codebleu_shim] Missing 'tree_sitter_languages'. Install with:\n"
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


# --- Patch CodeBLEU modules to use our loader everywhere ---
_patched_utils = False
_patched_codebleu = False

try:
    utils = importlib.import_module("codebleu.utils")
    setattr(utils, "get_tree_sitter_language", _get_ts_lang_from_tsl)
    _patched_utils = True
except Exception:
    pass

try:
    cmod = importlib.import_module("codebleu.codebleu")
    if hasattr(cmod, "get_tree_sitter_language"):
        setattr(cmod, "get_tree_sitter_language", _get_ts_lang_from_tsl)
    _patched_codebleu = True
except Exception:
    pass

print(
    "[codebleu_shim] ready — "
    f"Parser.language patched={_patched_parser}, "
    f"utils.patched={_patched_utils}, "
    f"codebleu.patched={_patched_codebleu}"
)