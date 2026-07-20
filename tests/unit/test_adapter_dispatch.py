"""Unit tests for transport-aware dispatch + token_expired retry.

These cover the pure helpers — ``_detect_transport`` (general, in the adapter)
and ``_merge_http_auth`` / ``extract_connector_error_payload`` (connector glue,
in ``connectors.token_injection``). The end-to-end wrapper exercise (resolver →
transport branch → meta injection → retry) lands in the integration suite
where the fastmcp client stack runs in-process.
"""

from __future__ import annotations

import mcp.types
import pytest
from tai_contract.connectors.models import ConnectorRef
from tai_contract.manifest import MCPConfig, TaiMCPConfig

import tai_skeleton.app.instance  # noqa: F401 — binds app
from tai_skeleton.connectors.runtime.resolver import ManagedAuth
from tai_skeleton.connectors.token_injection import (
    CONNECTOR_ERROR_PREFIX,
    _merge_http_auth,
    extract_connector_error_payload,
)
from tai_skeleton.tools.adapters.mcp_tool_to_func import _detect_transport

# -- _detect_transport --------------------------------------------------------


def test_detect_transport_http():
    cfg = MCPConfig(type="http", url="https://x.test/sse")
    assert _detect_transport(cfg) == "http"


def test_detect_transport_stdio():
    cfg = MCPConfig(type="stdio", command="uvx", args=["tool"])
    assert _detect_transport(cfg) == "stdio"


def test_detect_transport_uds():
    cfg = MCPConfig(type="uds", uds="/tmp/mcp.sock")
    assert _detect_transport(cfg) == "uds"


def test_detect_transport_ambiguous_raises():
    # MCPConfig._exactly_one_transport now rejects this at
    # construction; bypass it via model_construct to verify _detect_transport
    # is still defensive (programmatic mutation could in principle bypass
    # the model validator).
    cfg = MCPConfig.model_construct(
        type="http",
        url="https://x.test",
        command="uvx",
    )
    with pytest.raises(RuntimeError, match="ambiguous"):
        _detect_transport(cfg)


def test_detect_transport_empty_raises():
    cfg = MCPConfig(type=None)
    with pytest.raises(RuntimeError, match="no transport"):
        _detect_transport(cfg)


# -- _merge_http_auth ---------------------------------------------------------


_CONN_ID = "11111111-1111-1111-1111-111111111111"


def _http_managed(headers: dict[str, str] | None = None) -> TaiMCPConfig:
    return TaiMCPConfig(
        title="t",
        config=MCPConfig(type="http", url="https://x.test/sse", headers=headers or {}),
        managed=ConnectorRef(
            connection_id=_CONN_ID,
            provider_id="google",
            sub_service="gmail",
        ),
    )


def test_merge_http_auth_adds_authorization():
    cfg = _http_managed()
    merged = _merge_http_auth(cfg, ManagedAuth(access_token="tok"))
    assert merged.config.headers == {"authorization": "Bearer tok"}


def test_merge_http_auth_canonicalizes_case_and_overrides_manifest():
    cfg = _http_managed({"Authorization": "Bearer manifest", "X-Other": "v"})
    merged = _merge_http_auth(cfg, ManagedAuth(access_token="resolved"))
    # manifest Authorization collapses with resolver's via lowercase canon,
    # resolver wins; sibling headers preserved.
    assert merged.config.headers == {
        "authorization": "Bearer resolved",
        "x-other": "v",
    }


def test_merge_http_auth_returns_new_object_not_mutated_in_place():
    cfg = _http_managed({"X": "v"})
    merged = _merge_http_auth(cfg, ManagedAuth(access_token="t"))
    assert merged is not cfg
    assert cfg.config.headers == {"X": "v"}  # original unchanged


# -- extract_connector_error_payload ------------------------------------------


def _text_result(text: str, is_error: bool = True) -> mcp.types.CallToolResult:
    return mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type="text", text=text)],
        isError=is_error,
    )


def test_extract_payload_recognises_prefixed_sentinel():
    result = _text_result(
        f'Error calling tool \'list_messages\': {CONNECTOR_ERROR_PREFIX}{{"code":"token_expired","message":"401"}}',
    )
    payload = extract_connector_error_payload(result)
    assert payload == {"code": "token_expired", "message": "401"}


def test_extract_payload_recognises_bare_sentinel():
    # If a future fastmcp variant stops prefixing the tool name, the parser
    # must still find the sentinel anywhere in the text.
    result = _text_result(f"{CONNECTOR_ERROR_PREFIX}" + '{"code":"rate_limited","retry_after":7}')
    payload = extract_connector_error_payload(result)
    assert payload == {"code": "rate_limited", "retry_after": 7}


def test_extract_payload_returns_none_for_success():
    result = mcp.types.CallToolResult(content=[], isError=False)
    assert extract_connector_error_payload(result) is None


def test_extract_payload_returns_none_for_unrelated_error():
    # Plain ToolError from a hand-authored MCP server — not a hub-encoded
    # sentinel; must not be misinterpreted.
    result = _text_result("Error calling tool 'X': boom")
    assert extract_connector_error_payload(result) is None


def test_extract_payload_returns_none_on_malformed_json():
    result = _text_result(f"{CONNECTOR_ERROR_PREFIX}not-json")
    assert extract_connector_error_payload(result) is None


def test_extract_payload_returns_none_when_code_missing():
    result = _text_result(f'{CONNECTOR_ERROR_PREFIX}{{"detail":"x"}}')
    assert extract_connector_error_payload(result) is None
