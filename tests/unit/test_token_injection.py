"""Unit tests for the connector managed-auth wrappers.

The injection seam (``set_resolver`` / ``set_force_refresher`` / sentinels)
is gone; ``resolve_managed_auth`` / ``force_refresh`` are store-backed runtime
functions taking ``(connection_id, provider_id, sub_service)`` directly. The
token-injection module wraps them in ``resolve_managed_auth_for_config(config)``
/ ``_force_refresh(config)`` which unwrap ``config.managed``, gate on
``config.is_managed``, and reject an empty access_token (no silent
unauthenticated requests on the wire).

Covers:
- managed config → wrapper passes the ConnectorRef fields to the runtime
  resolver and returns its ManagedAuth.
- hand-authored (non-managed) config → wrapper returns None without calling
  the runtime resolver.
- empty / unusable ManagedAuth from the resolver → wrapper raises.
- force_refresh wrapper passes the ConnectorRef fields through.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from tai_contract.connectors.models import ConnectorRef
from tai_contract.manifest import MCPConfig, TaiMCPConfig

# Bind app before importing the adapter, so the adapter import chain
# resolves against a constructed app.
import tai_skeleton.app.instance  # noqa: F401
from tai_skeleton.connectors.runtime.resolver import ManagedAuth
from tai_skeleton.connectors.token_injection import (
    _force_refresh,
    resolve_managed_auth_for_config,
)

CONN_ID = "11111111-1111-1111-1111-111111111111"

_RESOLVER = "tai_skeleton.connectors.token_injection.resolve_managed_auth"
_REFRESHER = "tai_skeleton.connectors.token_injection.force_refresh"


def _managed_config(*, transport: str = "http") -> TaiMCPConfig:
    if transport == "http":
        inner = MCPConfig(type="http", url="https://gmail.test/sse", headers={})
    else:
        inner = MCPConfig(type="stdio", command="uvx", args=["tai-mcp-google-gmail"])
    return TaiMCPConfig(
        title="google_gmail_work",
        config=inner,
        managed=ConnectorRef(
            connection_id=CONN_ID,
            provider_id="google",
            sub_service="gmail",
        ),
    )


def _hand_authored_config() -> TaiMCPConfig:
    return TaiMCPConfig(
        title="manual",
        config=MCPConfig(type="http", url="https://example.com/sse"),
    )


# -- resolve_managed_auth_for_config ------------------------------------------


def test_managed_config_resolves_via_runtime_resolver():
    with patch(
        _RESOLVER,
        new=AsyncMock(return_value=ManagedAuth(access_token="raw-token")),
    ) as resolver:
        auth = asyncio.run(resolve_managed_auth_for_config(_managed_config()))

    assert auth == ManagedAuth(access_token="raw-token")
    resolver.assert_awaited_once_with(CONN_ID, "google", "gmail")


def test_hand_authored_config_returns_none_without_resolving():
    with patch(_RESOLVER, new=AsyncMock()) as resolver:
        auth = asyncio.run(resolve_managed_auth_for_config(_hand_authored_config()))

    assert auth is None
    resolver.assert_not_awaited()


def test_resolver_returning_empty_managed_auth_raises():
    with (
        patch(
            _RESOLVER,
            new=AsyncMock(return_value=ManagedAuth(access_token="")),
        ),
        pytest.raises(RuntimeError, match="empty access_token"),
    ):
        asyncio.run(resolve_managed_auth_for_config(_managed_config()))


# -- _force_refresh -----------------------------------------------------------


def test_force_refresh_passes_connection_id_and_failed_token():
    with patch(
        _REFRESHER,
        new=AsyncMock(return_value=ManagedAuth(access_token="fresh-token")),
    ) as refresher:
        auth = asyncio.run(_force_refresh(_managed_config(), "dead-token"))

    assert auth.access_token == "fresh-token"
    # The wrapper keys on connection_id and fences on the failing token.
    refresher.assert_awaited_once_with(CONN_ID, failed_access_token="dead-token")


def test_force_refresh_returning_empty_managed_auth_raises():
    with (
        patch(
            _REFRESHER,
            new=AsyncMock(return_value=ManagedAuth(access_token="")),
        ),
        pytest.raises(RuntimeError, match="empty access_token"),
    ):
        asyncio.run(_force_refresh(_managed_config(), "dead-token"))
