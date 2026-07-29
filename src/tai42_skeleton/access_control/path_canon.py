"""One canonical request-path form for every access-control path decision.

Every path check — the ``/api``/``/mcp`` exclusion, the reserved-set membership
test, and the route-table match — runs on ONE canonical string computed ONCE, so
two checks can never disagree on the shape of a path (the bypass class this closes).
The prefix helper is SEGMENT-aware: ``/api`` guards ``/api`` and ``/api/...`` but
never ``/apiary``.
"""

from __future__ import annotations

import re

# A residual percent-escape surviving the ASGI decode: ``%`` followed by two hex digits
# (``%2F``, ``%5C``, and every other ``%XX``). The ASGI server percent-decodes the path
# once into ``scope["path"]`` and the router matches on that once-decoded form, so a
# still-present escape is a double-encoded byte — decoding it here would put authz on a
# different form than the router matched (the parser differential this closes).
_RESIDUAL_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")


class MalformedPathError(ValueError):
    """A request path that cannot be reasoned about safely: it carries a NUL byte, an
    ASCII control char, a backslash, or a residual percent-escape (a ``%XX`` the ASGI
    decode did not resolve). The classifier treats it as fail-closed (never the SPA
    shell) and logs it; settings validation treats it as a config error."""


def canonicalize_path(path: str) -> str:
    """The single canonical form of ``path``, which is ALREADY once-decoded: the ASGI
    server percent-decodes the request path into ``scope["path"]`` (what ``conn.url.path``
    returns) and the router matches on that form. This function does NOT decode again — it
    keeps authz and routing on ONE form:

    1. Reject a residual percent-escape (``%2F``, ``%5C``, or any ``%`` followed by two
       hex digits). Decoding it a second time would make authz reason on a different form
       than the router matched (and would corrupt a segment that legitimately carries that
       byte as data), so a surviving escape raises :class:`MalformedPathError` — fail-closed,
       never re-decoded.
    2. Reject NUL, ASCII control chars (``\\x00``-``\\x1f``, ``\\x7f``), and backslash —
       a malformed path raises :class:`MalformedPathError` rather than being coerced
       into something servable.
    3. Collapse duplicate slashes and resolve ``.``/``..`` segments, never letting a
       ``..`` walk above root (an escaping ``..`` normalizes to root).

    The result is an absolute, slash-collapsed, dot-resolved path with no trailing
    slash (except root ``/``). Case is preserved — the API is mounted lowercase, so
    case-sensitivity is intentional.
    """
    if _RESIDUAL_ESCAPE.search(path):
        raise MalformedPathError(f"path {path!r} contains a residual percent-escape (a double-encoded byte)")
    for ch in path:
        code = ord(ch)
        if ch == "\\" or code < 0x20 or code == 0x7F:
            raise MalformedPathError(f"path {path!r} contains a NUL, control, or backslash byte")

    segments: list[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            # "" is a duplicate/leading/trailing slash; "." is the current dir.
            continue
        if segment == "..":
            if segments:
                segments.pop()
            # A "..'' with nothing to pop escapes root — drop it (normalize to root).
            continue
        segments.append(segment)
    return "/" + "/".join(segments)


def under_prefix(path: str, prefix: str) -> bool:
    """Whether ``path`` is ``prefix`` itself or a path-segment descendant of it —
    ``path == prefix or path.startswith(prefix + "/")``. Segment-aware, never a bare
    ``startswith`` (which would leak ``/apiary`` past a ``/api`` guard)."""
    return path == prefix or path.startswith(f"{prefix}/")
