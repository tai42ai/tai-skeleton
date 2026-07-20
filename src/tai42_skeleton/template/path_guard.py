"""Lexical containment guard for logical template keys.

Every non-HTTP entrance that reaches the store with a caller-supplied template key
runs it through :func:`safe_template_path` — the HTTP router, the backup importer,
the builtin upload tool, and the read seam in ``ResourceManager.load``. The guard
is a single shared implementation so no entrance can drift from the others.
"""

from __future__ import annotations

import os


class UnsafeTemplatePathError(ValueError):
    """A logical template key that escapes the template root."""


# A virtual anchor the logical template key is resolved under. It never exists on
# disk, so ``os.path.realpath`` does purely LEXICAL normalization here — it
# collapses ``..`` and drops the anchor for an absolute key, so a key that escapes
# resolves OUT of the root and is rejected. It does NOT follow filesystem symlinks
# (the anchor is not the store's real path); a filesystem-backed store defends
# symlinks at its own real root.
_TEMPLATE_ROOT = "/tai-template-root"


def safe_template_path(path: object) -> str:
    """Reject a template key that escapes the template root.

    LEXICAL containment: the key is resolved with ``os.path.realpath`` under the
    virtual ``_TEMPLATE_ROOT`` anchor (which is not on disk), so realpath collapses
    ``..`` and drops an absolute key but does NOT follow filesystem symlinks — an
    absolute key or a ``..`` escape resolves outside the root and is refused
    loudly. A backslash is refused outright: a Windows-backed store treats it as a
    separator that POSIX ``realpath`` would leave as a literal, so a
    ``a\\..\\..\\x`` traversal would otherwise slip through.
    """
    if not isinstance(path, str) or not path:
        raise UnsafeTemplatePathError("path must be a non-empty string")
    if "\\" in path:
        raise UnsafeTemplatePathError(f"unsafe template path: {path!r}")
    root = os.path.realpath(_TEMPLATE_ROOT)
    try:
        resolved = os.path.realpath(os.path.join(root, path))
    except (ValueError, OSError) as exc:
        # An embedded NUL byte (or other malformed path) makes realpath raise
        # ValueError/OSError — that is malformed input, a loud rejection, never a 500.
        raise UnsafeTemplatePathError(f"unsafe template path: {path!r}") from exc
    if resolved != root and not resolved.startswith(root + os.sep):
        raise UnsafeTemplatePathError(f"unsafe template path: {path!r}")
    return path
