"""Shared offline fixtures + descriptor/record builders for the connectors suite.

Every backend (Redis / Postgres / HTTP / MCP) is faked at the kit ``client_ctx``
seam; no real network or process is touched. Crypto/state secrets are injected
via env and the settings caches are reset around each test so a ``setenv`` in one
test never bleeds into the next.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest
from pydantic import SecretStr
from tai42_contract.connectors.models import AuthHealthState, ConnectionRecord
from tai42_contract.connectors.providers import (
    ConfigFieldSpec,
    McpServerDescriptor,
    OAuthEndpoints,
    ProviderDescriptor,
    SubServiceDescriptor,
)
from tai42_kit.settings import reset_all_settings

# Deterministic test secrets: a 32-byte KEK and a 32-byte state-HMAC key.
TEST_KEK_B64 = base64.b64encode(bytes(range(32))).decode("ascii")
TEST_HMAC_B64 = base64.b64encode(bytes(range(32, 64))).decode("ascii")

CID = "11111111-1111-4111-8111-111111111111"
CID2 = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def _reset_settings_caches() -> Iterator[None]:
    """Drop every cached settings accessor around each test."""
    reset_all_settings()
    yield
    reset_all_settings()


@pytest.fixture(autouse=True)
def crypto_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provision the two engine secrets + a permissive redirect allow-list.

    Autouse so any test exercising encrypt/sign/authorize has them; a test that
    wants them absent deletes the var and resets the cache itself.
    """
    monkeypatch.setenv("CONNECTORS_KEK", TEST_KEK_B64)
    monkeypatch.setenv("CONNECTORS_STATE_HMAC_KEY", TEST_HMAC_B64)
    monkeypatch.setenv(
        "CONNECTORS_REDIRECT_URI_ALLOWLIST",
        "https://app.example.com,http://localhost:5173",
    )
    reset_all_settings()


# -- Descriptor / record builders -------------------------------------------


def make_oauth_descriptor(
    *,
    provider_id: str = "acme",
    revoke: str | None = "https://acme.test/revoke",
    extra_authorize_params: dict[str, str] | None = None,
) -> ProviderDescriptor:
    """An http-transport OAuth provider with two sub-services."""
    return ProviderDescriptor(
        id=provider_id,
        display_name="Acme",
        icon_url="https://acme.test/icon.png",
        kind="oauth",
        origin="system",
        category="productivity",
        oauth=OAuthEndpoints(
            authorize="https://acme.test/authorize",
            token="https://acme.test/token",
            revoke=revoke,
        ),
        client_id_env="ACME_CLIENT_ID",
        client_secret_env="ACME_CLIENT_SECRET",
        extra_authorize_params=extra_authorize_params or {},
        sub_services={
            "mail": SubServiceDescriptor(
                id="mail",
                display_name="Mail",
                scopes=["mail.read", "mail.send"],
                mcp_server=McpServerDescriptor(type="http", url="https://acme.test/mcp/mail"),
            ),
            "cal": SubServiceDescriptor(
                id="cal",
                display_name="Calendar",
                scopes=["cal.read"],
                mcp_server=McpServerDescriptor(type="http", url="https://acme.test/mcp/cal"),
            ),
        },
    )


def make_noauth_stdio_descriptor(
    *,
    provider_id: str = "widgets",
    origin: Literal["system", "community"] = "system",
) -> ProviderDescriptor:
    """A no-auth provider whose single sub-service is pkg-launched (uvx)."""
    return ProviderDescriptor(
        id=provider_id,
        display_name="Widgets",
        icon_url="https://widgets.test/icon.png",
        kind="none",
        origin=origin,
        category="data",
        pkg_manager="uvx",
        pkg_version="1.2.3",
        config_fields=[
            ConfigFieldSpec(key="api_key", label="API Key", target="env", secret=True),
        ],
        sub_services={
            "search": SubServiceDescriptor(
                id="search",
                display_name="Search",
                entry_point="tai-mcp-widgets",
            ),
        },
    )


def make_noauth_http_descriptor(
    *,
    provider_id: str = "httpsvc",
) -> ProviderDescriptor:
    """A no-auth provider whose sub-service is a fully-declared http server."""
    return ProviderDescriptor(
        id=provider_id,
        display_name="HttpSvc",
        icon_url="https://httpsvc.test/icon.png",
        kind="none",
        origin="system",
        category="data",
        config_fields=[
            ConfigFieldSpec(key="token", label="Token", target="header"),
        ],
        sub_services={
            "main": SubServiceDescriptor(
                id="main",
                display_name="Main",
                mcp_server=McpServerDescriptor(type="http", url="https://httpsvc.test/mcp"),
            ),
        },
    )


def make_oauth_record(
    *,
    connection_id: str = CID,
    provider_id: str = "acme",
    alias: str = "work",
    expires_in_seconds: int = 3600,
    health: AuthHealthState = AuthHealthState.HEALTHY,
    enabled_sub_services: list[str] | None = None,
    granted_scopes: list[str] | None = None,
) -> ConnectionRecord:
    now = datetime.now(UTC)
    return ConnectionRecord(
        connection_id=connection_id,
        provider_id=provider_id,
        kind="oauth",
        alias=alias,
        account_identity="user@acme.test",
        enabled_sub_services=enabled_sub_services or ["mail"],
        granted_scopes=granted_scopes or ["mail.read", "mail.send"],
        access_token=SecretStr("access-tok"),
        refresh_token=SecretStr("refresh-tok"),
        access_token_expires_at=now + timedelta(seconds=expires_in_seconds),
        auth_health_state=health,
        created_at=now,
    )


def make_noauth_record(
    *,
    connection_id: str = CID,
    provider_id: str = "widgets",
    alias: str = "main",
    enabled_sub_services: list[str] | None = None,
    config_values: dict[str, str] | None = None,
) -> ConnectionRecord:
    values = config_values if config_values is not None else {"api_key": "k"}
    return ConnectionRecord(
        connection_id=connection_id,
        provider_id=provider_id,
        kind="none",
        alias=alias,
        enabled_sub_services=enabled_sub_services or ["search"],
        config_values={k: SecretStr(v) for k, v in values.items()},
        auth_health_state=AuthHealthState.HEALTHY,
        created_at=datetime.now(UTC),
    )


# -- Fake HTTP client --------------------------------------------------------


class FakeHttpResponse:
    """Minimal httpx.Response stand-in for the OAuth client."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_body=None,
        content: bytes = b"",
        raise_on_json: bool = False,
    ) -> None:
        self.status_code = status_code
        self._json = json_body
        self.content = content if content else (b"x" if json_body is None else b"{}")
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise ValueError("not json")
        return self._json


class FakeHttp:
    """Records posts and returns a queued response (or raises a queued error)."""

    def __init__(self, responses) -> None:
        # responses: list of FakeHttpResponse | Exception, consumed in order.
        self._responses = list(responses)
        self.posts: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, data=None, timeout=None):
        self.posts.append((url, dict(data or {})))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def oauth_client_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the operator-supplied OAuth client credentials."""
    monkeypatch.setenv("ACME_CLIENT_ID", "client-123")
    monkeypatch.setenv("ACME_CLIENT_SECRET", "secret-xyz")
