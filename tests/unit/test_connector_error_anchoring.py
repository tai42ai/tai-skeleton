"""Anchored ``CONNECTOR_ERROR_PREFIX`` scan.

The previous ``text.find(CONNECTOR_ERROR_PREFIX)`` matched the prefix at
ANY position in the content block. A managed tool that echoed
user-controlled input into its error text could thereby plant
``tai-hub-err:{"code":"token_expired"}`` and trigger a phantom force-
refresh + retry from the adapter. The anchored regex below requires
the prefix to be either at position 0 (bare hub framing) or
immediately after fastmcp's ``Error calling tool '<name>': `` prefix
(the only framing fastmcp emits around hub errors).
"""

from __future__ import annotations

import json
import re

import mcp.types

import tai_skeleton.app.instance  # noqa: F401 — bind app
from tai_skeleton.connectors.token_injection import (
    _CONNECTOR_ERROR_FRAMING_RE,
    CONNECTOR_ERROR_PREFIX,
    extract_connector_error_payload,
)


def _result(text: str) -> mcp.types.CallToolResult:
    return mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type="text", text=text)],
        isError=True,
    )


def _payload(code: str = "token_expired") -> str:
    return json.dumps({"code": code})


# -- accepted framings --------------------------------------------------------


def test_bare_prefix_at_start_matches():
    text = f"{CONNECTOR_ERROR_PREFIX}{_payload()}"
    out = extract_connector_error_payload(_result(text))
    assert out is not None
    assert out["code"] == "token_expired"


def test_fastmcp_framing_matches():
    text = f"Error calling tool 'list_messages': {CONNECTOR_ERROR_PREFIX}{_payload()}"
    out = extract_connector_error_payload(_result(text))
    assert out is not None
    assert out["code"] == "token_expired"


# -- echo-attack rejections ---------------------------------------------------


def test_user_echo_inside_message_does_not_match():
    """A managed tool that mirrors user input back into the error text
    cannot trigger the retry path: the prefix appears mid-string, not
    at the start of the content block and not after fastmcp's framing."""
    text = f"User said: '{CONNECTOR_ERROR_PREFIX}{_payload()}' — request rejected."
    assert extract_connector_error_payload(_result(text)) is None


def test_leading_whitespace_blocks_match():
    """The regex anchors with ``^`` (no re.MULTILINE) so a leading
    space disqualifies the match. fastmcp does not emit leading
    whitespace; anything that does is not the framing we trust."""
    text = f" {CONNECTOR_ERROR_PREFIX}{_payload()}"
    assert extract_connector_error_payload(_result(text)) is None


def test_lookalike_framing_with_different_tool_quote_char_does_not_match():
    """The framing regex requires single quotes around the tool name.
    A managed tool that echoed a double-quoted lookalike like
    ``Error calling tool "x": tai-hub-err:{...}`` MUST NOT match."""
    text = f'Error calling tool "x": {CONNECTOR_ERROR_PREFIX}{_payload()}'
    assert extract_connector_error_payload(_result(text)) is None


# -- regex contract -----------------------------------------------------------


def test_framing_regex_is_anchored_at_start():
    """``^`` anchor must remain so a future re-build cannot drop the
    anchor accidentally."""
    # The module exposes this lazily via __getattr__ (typed ``object``); it is a
    # compiled regex, so narrow before reading ``.pattern``.
    assert isinstance(_CONNECTOR_ERROR_FRAMING_RE, re.Pattern)
    assert _CONNECTOR_ERROR_FRAMING_RE.pattern.startswith("^")
