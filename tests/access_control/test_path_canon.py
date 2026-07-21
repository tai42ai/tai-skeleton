"""Unit behavior of ``access_control.path_canon.canonicalize_path``.

The path arriving here is ALREADY once-decoded by the ASGI server (``scope["path"]`` /
``conn.url.path``), and the router matched on that same form. So canonicalization must
NOT decode again — it keeps authz and routing on ONE form — while still collapsing
slashes, resolving dot-segments, and rejecting anything unservable (NUL/control/backslash
or a residual percent-escape that a second decode would resolve).
"""

from __future__ import annotations

import pytest

from tai42_skeleton.access_control.path_canon import (
    MalformedPathError,
    canonicalize_path,
    under_prefix,
)

# -- no second decode --------------------------------------------------------


def test_no_second_decode_plain_path_is_identity():
    assert canonicalize_path("/agents/inner/view") == "/agents/inner/view"


def test_no_second_decode_bare_percent_kept_literal():
    # A lone ``%`` that is NOT a valid ``%XX`` escape is ordinary data — it is neither
    # decoded nor rejected (the router sees the same literal).
    assert canonicalize_path("/discount/50%") == "/discount/50%"
    assert canonicalize_path("/x/%zz") == "/x/%zz"


# -- residual percent-escape rejection (fail-closed, not re-decode) ----------


@pytest.mark.parametrize(
    "path",
    [
        "/api%2Fx",  # residual encoded slash — a second decode would forge a /api/ segment
        "/x%5Cy",  # residual encoded backslash
        "/%61pi/x",  # residual %XX that would decode to a real /api segment
        "/files/a%2Fb",  # a legitimate-looking encoded byte inside a segment
        "/foo%2fbar",  # lower-case hex is a residual escape too
    ],
)
def test_residual_percent_escape_is_rejected(path):
    # The double-encoded byte is denied loudly rather than decoded a second time (which
    # would put authz on a different form than the router matched).
    with pytest.raises(MalformedPathError, match="residual percent-escape"):
        canonicalize_path(path)


# -- NUL / control / backslash rejection -------------------------------------


@pytest.mark.parametrize("path", ["/agents\x00", "/agents\x1f", "/agents\x7f", "/agents\\admin"])
def test_control_and_backslash_rejected(path):
    with pytest.raises(MalformedPathError, match="NUL, control, or backslash"):
        canonicalize_path(path)


# -- slash collapse + dot-segment resolution ---------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("//api/x", "/api/x"),  # duplicate leading slash collapses
        ("/a//b///c", "/a/b/c"),  # interior duplicate slashes collapse
        ("/a/b/", "/a/b"),  # trailing slash dropped
        ("/", "/"),  # root preserved
        ("/a/./b", "/a/b"),  # current-dir segment removed
        ("/api/../agents", "/agents"),  # dot-resolution before any prefix check
        ("/agents/../api/secret", "/api/secret"),  # …and the reverse
        ("/../../etc", "/etc"),  # a ".." that escapes root normalizes to root
    ],
)
def test_slash_and_dot_normalization(raw, expected):
    assert canonicalize_path(raw) == expected


# -- segment-aware prefix helper ---------------------------------------------


def test_under_prefix_is_segment_aware():
    assert under_prefix("/api", "/api") is True
    assert under_prefix("/api/x", "/api") is True
    assert under_prefix("/apiary", "/api") is False
