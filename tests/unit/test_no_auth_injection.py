"""No-auth token-injection paths in _prepare_request + the resolver wrapper.

OAuth: token injected as http Bearer / stdio _meta. No-auth with client config:
headers merged (http) or env merged (stdio), NO token, NO _meta. No-auth without
config: resolver returns None → inject nothing, and the empty-token guard does
NOT fire.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from tai42_contract.connectors.models import ConnectorRef
from tai42_contract.manifest import MCPConfig, TaiMCPConfig

# Bind app before importing the adapter seam.
import tai42_skeleton.app.instance  # noqa: F401
from tai42_skeleton.connectors.runtime.resolver import ManagedAuth
from tai42_skeleton.connectors.token_injection import (
    CONNECTOR_META_TOKEN_KEY,
    _prepare_request,
    resolve_managed_auth_for_config,
)

CONN_ID = "11111111-1111-1111-1111-111111111111"
_RESOLVER = "tai42_skeleton.connectors.token_injection.resolve_managed_auth"


def _config(*, transport: str = "http") -> TaiMCPConfig:
    if transport == "http":
        inner = MCPConfig(type="http", url="https://acme.test/mcp", headers={})
    else:
        inner = MCPConfig(type="stdio", command="uvx", args=["x"])
    return TaiMCPConfig(
        title="acme_api_prod",
        config=inner,
        managed=ConnectorRef(
            connection_id=CONN_ID,
            provider_id="acme",
            sub_service="api",
        ),
    )


def test_prepare_request_none_injects_nothing():
    cfg = _config()
    out_cfg, meta = _prepare_request(cfg, None, "http")
    assert out_cfg is cfg
    assert meta is None


def test_prepare_request_oauth_http_bearer():
    cfg = _config(transport="http")
    out_cfg, meta = _prepare_request(cfg, ManagedAuth(access_token="tok"), "http")
    assert meta is None
    assert out_cfg.config.headers is not None
    assert out_cfg.config.headers["authorization"] == "Bearer tok"


def test_prepare_request_oauth_stdio_meta():
    cfg = _config(transport="stdio")
    _out_cfg, meta = _prepare_request(cfg, ManagedAuth(access_token="tok"), "stdio")
    assert meta == {CONNECTOR_META_TOKEN_KEY: "tok"}


def test_prepare_request_no_auth_http_headers():
    cfg = _config(transport="http")
    auth = ManagedAuth(headers={"x-api-key": "k-123"})
    out_cfg, meta = _prepare_request(cfg, auth, "http")
    assert meta is None
    assert out_cfg.config.headers is not None
    assert out_cfg.config.headers["x-api-key"] == "k-123"
    # No bearer token injected.
    assert "authorization" not in out_cfg.config.headers


def test_prepare_request_no_auth_stdio_env():
    cfg = _config(transport="stdio")
    auth = ManagedAuth(env={"API_KEY": "k-123"})
    out_cfg, meta = _prepare_request(cfg, auth, "stdio")
    # No _meta token for no-auth stdio.
    assert meta is None
    assert out_cfg.config.env is not None
    assert out_cfg.config.env["API_KEY"] == "k-123"


def test_resolver_wrapper_passes_through_no_auth_none():
    cfg = _config()
    with patch(_RESOLVER, new=AsyncMock(return_value=None)):
        auth = asyncio.run(resolve_managed_auth_for_config(cfg))
    # None is legitimate for a no-auth-no-config entry — not rejected.
    assert auth is None


def test_resolver_wrapper_passes_through_no_auth_headers():
    cfg = _config()
    expected = ManagedAuth(headers={"x-api-key": "k"})
    with patch(_RESOLVER, new=AsyncMock(return_value=expected)):
        auth = asyncio.run(resolve_managed_auth_for_config(cfg))
    assert auth is expected


def test_resolver_wrapper_rejects_empty_oauth_token():
    cfg = _config()
    with (
        patch(_RESOLVER, new=AsyncMock(return_value=ManagedAuth(access_token=""))),
        pytest.raises(RuntimeError, match="empty access_token"),
    ):
        asyncio.run(resolve_managed_auth_for_config(cfg))
