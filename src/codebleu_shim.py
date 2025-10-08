"""
codebleu_shim.py
Make CodeBLEU 0.7.0 work with tree_sitter==0.20.x by:

- Adding a Parser.language property (so `parser.language = lang` works)
- Returning a real tree_sitter.Language via tree_sitter_languages.get_language(...)
- Patching codebleu.utils.get_tree_sitter_language and the duplicate in codebleu.codebleu
"""

from __future__ import annotations
import importlib
import types

# 1) Patch Parser.language for tree_sitter==0.20.x
try:
    from tree_sitter import Parser, Language as TS_Language  # type: ignore
except Exception as e:
    raise RuntimeError(
        "[codebleu_shim] Failed to import tree_sitter. Ensure it's installed in this env."
    ) from e

# Only add the property if it's not already present
if not hasattr(Parser, "language"):
    def _get_lang(self):
        # best-effort mirror; tree_sitter doesn't expose a getter in 0.20.x
        return getattr(self, "_shim_lang", None)

    def _set_lang(self, lang):
        if not isinstance(lang, TS_Language):
            raise TypeError("language must be assigned a tree_sitter.Language object")
        # Forward to the real API in 0.20.x
        self.set_language(lang)
        # Keep a Python-side reference for the getter
        self._shim_lang = lang

    # Attach property to the C-extension class; this works in 0.20.x
    Parser.language = property(_get_lang, _set_lang)  # type: ignore[attr-defined]
    _patched_parser = True
else:
    _patched_parser = False

# 2) Our replacement that returns a real Language for Python via tree_sitter_languages
def _get_ts_lang_from_tsl(lang: str) -> TS_Language:
    """
    Replacement for codebleu.utils.get_tree_sitter_language(lang).
    Uses tree_sitter_languages to fetch a real tree_sitter.Language.
    """
    key = (lang or "").strip().lower()
    if key not in {"py", "python"}:
        raise NotImplementedError(
            f"[codebleu_shim] Only 'python' supported. Got '{lang}'. "
            "Extend this shim if you need more languages."
        )
    try:
        tsl = importlib.import_module("tree_sitter_languages")
    except Exception as e:
        raise RuntimeError(
            "[codebleu_shim] Failed importing `tree_sitter_languages`. Install:\n"
            "  pip install tree_sitter_languages==1.10.2"
        ) from e
    try:
        lang_obj = tsl.get_language("python")  # returns a tree_sitter.Language
    except Exception as e:
        raise RuntimeError("[codebleu_shim] tree_sitter_languages.get_language('python') failed") from e
    if not isinstance(lang_obj, TS_Language):
        raise TypeError("[codebleu_shim] Expected tree_sitter.Language from tree_sitter_languages")
    return lang_obj

# 3) Patch CodeBLEU’s imports/entrypoints
_patched_utils = False
_patched_codebleu = False

try:
    # Patch codebleu.utils.get_tree_sitter_language
    utils = importlib.import_module("codebleu.utils")
    setattr(utils, "get_tree_sitter_language", _get_ts_lang_from_tsl)
    _patched_utils = True
except Exception:
    pass

try:
    # Patch the duplicate in codebleu.codebleu
    codebleu_mod = importlib.import_module("codebleu.codebleu")
    # Some versions alias the function at module scope
    if hasattr(codebleu_mod, "get_tree_sitter_language"):
        setattr(codebleu_mod, "get_tree_sitter_language", _get_ts_lang_from_tsl)
        _patched_codebleu = True
    else:
        # If it's imported via "from .utils import get_tree_sitter_language", our utils-patch is enough.
        _patched_codebleu = True
except Exception:
    pass

print(
    "[codebleu_shim] ready — "
    f"Parser.language patched={_patched_parser}, "
    f"utils.patched={_patched_utils}, "
    f"codebleu.patched={_patched_codebleu}"
)
