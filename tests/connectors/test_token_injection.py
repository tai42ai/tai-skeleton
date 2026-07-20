"""Connector auth glue: resolve, inject, error framing, force-refresh retry."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast

import mcp.types
import pytest
from tai42_contract.connectors.models import ConnectorRef
from tai42_contract.manifest import MCPConfig, TaiMCPConfig
from tai42_kit.clients.impl.mcp import FastMCPClient

import tai42_skeleton.connectors.token_injection as ti
from tai42_skeleton.connectors.runtime.resolver import ConnectorAuthExpiredError, ManagedAuth
from tai42_skeleton.connectors.token_injection import (
    CONNECTOR_ERROR_PREFIX,
    CONNECTOR_META_TOKEN_KEY,
    check_managed_transport,
    extract_connector_error_payload,
    is_token_expired,
    resolve_managed_auth_for_config,
)

from .conftest import CID


def _managed_http(url="https://acme.test/mcp", headers=None):
    return TaiMCPConfig(
        title="acme_mail_work",
        config=MCPConfig(type="http", url=url, headers=headers or {}),
        managed=ConnectorRef(connection_id=CID, provider_id="acme", sub_service="mail"),
    )


def _managed_stdio(env=None):
    return TaiMCPConfig(
        title="widgets_search_main",
        config=MCPConfig(type="stdio", command="uvx", args=["x"], env=env or {}),
        managed=ConnectorRef(connection_id=CID, provider_id="widgets", sub_service="search"),
    )


def _unmanaged():
    return TaiMCPConfig(title="hand", config=MCPConfig(type="http", url="https://x/mcp"))


def _result(*, is_error, texts) -> mcp.types.CallToolResult:
    # extract_connector_error_payload reads only .isError, .content and each
    # block's .text. CallToolResult is a concrete pydantic model whose content
    # requires typed ContentBlocks, so the lightweight SimpleNamespace fake is
    # cast to it.
    return cast(
        mcp.types.CallToolResult,
        SimpleNamespace(
            isError=is_error,
            content=[SimpleNamespace(text=t) for t in texts],
        ),
    )


# -- resolve_managed_auth_for_config -----------------------------------------


async def test_resolve_unmanaged_returns_none(monkeypatch):
    assert await resolve_managed_auth_for_config(_unmanaged()) is None


async def test_resolve_oauth_returns_auth(monkeypatch):
    async def fake_resolve(cid, pid, ss):
        return ManagedAuth(access_token="tok")

    monkeypatch.setattr(ti, "resolve_managed_auth", fake_resolve)
    auth = await resolve_managed_auth_for_config(_managed_http())
    assert auth is not None
    assert auth.access_token == "tok"


async def test_resolve_oauth_empty_token_rejected(monkeypatch):
    async def fake_resolve(cid, pid, ss):
        return ManagedAuth(access_token="")

    monkeypatch.setattr(ti, "resolve_managed_auth", fake_resolve)
    # An empty-but-present access_token is never sent as if authenticated.
    with pytest.raises(RuntimeError, match="empty access_token"):
        await resolve_managed_auth_for_config(_managed_http())


async def test_resolve_no_auth_returns_auth(monkeypatch):
    async def fake_resolve(cid, pid, ss):
        return ManagedAuth(env={"api_key": "k"})

    monkeypatch.setattr(ti, "resolve_managed_auth", fake_resolve)
    auth = await resolve_managed_auth_for_config(_managed_stdio())
    assert auth is not None
    assert auth.env == {"api_key": "k"}


async def test_resolve_returns_none_when_resolver_none(monkeypatch):
    async def fake_resolve(cid, pid, ss):
        return None

    monkeypatch.setattr(ti, "resolve_managed_auth", fake_resolve)
    assert await resolve_managed_auth_for_config(_managed_stdio()) is None


# -- token injection / _prepare_request --------------------------------------


def test_prepare_request_no_auth_injects_nothing():
    config = _managed_http()
    out_config, meta = ti._prepare_request(config, None, "http")
    assert out_config is config
    assert meta is None


def test_prepare_request_oauth_stdio_sets_meta():
    config = _managed_stdio()
    _out_config, meta = ti._prepare_request(config, ManagedAuth(access_token="tok"), "stdio")
    assert meta == {CONNECTOR_META_TOKEN_KEY: "tok"}


def test_prepare_request_oauth_http_sets_bearer():
    config = _managed_http()
    out_config, meta = ti._prepare_request(config, ManagedAuth(access_token="tok"), "http")
    assert meta is None
    assert out_config.config.headers is not None
    assert out_config.config.headers["authorization"] == "Bearer tok"


def test_prepare_request_no_auth_stdio_merges_env():
    config = _managed_stdio(env={"BASE": "1"})
    out_config, meta = ti._prepare_request(config, ManagedAuth(env={"api_key": "k"}), "stdio")
    assert out_config.config.env == {"BASE": "1", "api_key": "k"}
    assert meta is None


def test_prepare_request_no_auth_http_merges_headers():
    config = _managed_http(headers={"X-Base": "1"})
    out_config, _meta = ti._prepare_request(config, ManagedAuth(headers={"X-Tok": "v"}), "http")
    assert out_config.config.headers == {"x-base": "1", "x-tok": "v"}


def test_merge_http_auth_lowercases_existing_headers():
    config = _managed_http(headers={"Authorization": "old", "X-Keep": "1"})
    merged = ti._merge_http_auth(config, ManagedAuth(access_token="new"))
    assert merged.config.headers is not None
    assert merged.config.headers["authorization"] == "Bearer new"
    assert merged.config.headers["x-keep"] == "1"


# -- check_managed_transport -------------------------------------------------


def test_check_managed_transport_allows_http():
    check_managed_transport(_managed_http(), "http")  # no raise


def test_check_managed_transport_rejects_unsupported():
    with pytest.raises(RuntimeError, match="not supported"):
        check_managed_transport(_managed_http(), "websocket")


def test_check_managed_transport_ignores_unmanaged():
    check_managed_transport(_unmanaged(), "websocket")  # no raise


# -- extract_connector_error_payload -----------------------------------------


def test_extract_returns_none_when_not_error():
    assert extract_connector_error_payload(_result(is_error=False, texts=["x"])) is None


def test_extract_parses_framed_payload():
    payload = json.dumps({"code": "token_expired"})
    result = _result(is_error=True, texts=[f"{CONNECTOR_ERROR_PREFIX}{payload}"])
    assert extract_connector_error_payload(result) == {"code": "token_expired"}


def test_extract_parses_after_fastmcp_framing():
    payload = json.dumps({"code": "boom"})
    text = f"Error calling tool 'mytool': {CONNECTOR_ERROR_PREFIX}{payload}"
    result = _result(is_error=True, texts=[text])
    assert extract_connector_error_payload(result) == {"code": "boom"}


def test_extract_rejects_forged_prefix_mid_string():
    payload = json.dumps({"code": "token_expired"})
    # prefix not at an anchored position → not trusted
    result = _result(is_error=True, texts=[f"user said: {CONNECTOR_ERROR_PREFIX}{payload}"])
    assert extract_connector_error_payload(result) is None


def test_extract_skips_bad_json():
    result = _result(is_error=True, texts=[f"{CONNECTOR_ERROR_PREFIX}not-json"])
    assert extract_connector_error_payload(result) is None


def test_extract_skips_payload_without_code():
    payload = json.dumps({"detail": "x"})
    result = _result(is_error=True, texts=[f"{CONNECTOR_ERROR_PREFIX}{payload}"])
    assert extract_connector_error_payload(result) is None


def test_extract_skips_non_text_blocks():
    # A non-text block (text=None) is not expressible as a real ContentBlock, so
    # the SimpleNamespace fake is cast to the concrete CallToolResult model.
    result = cast(
        mcp.types.CallToolResult,
        SimpleNamespace(isError=True, content=[SimpleNamespace(text=None)]),
    )
    assert extract_connector_error_payload(result) is None


# -- is_token_expired --------------------------------------------------------


def test_is_token_expired():
    assert is_token_expired({"code": "token_expired"}) is True
    assert is_token_expired({"code": "other"}) is False
    assert is_token_expired(None) is False


# -- call_with_auth / handle_token_expired -----------------------------------


class _FakeMcpClient:
    def __init__(self, result) -> None:
        self._result = result
        self.calls: list = []
        self.closed: list = []

    def current(self, *, config):
        client = self

        @asynccontextmanager
        async def _cm():
            yield client

        return _cm()

    async def call_tool_mcp(self, tool_name, arguments, meta=None, timeout=None):
        self.calls.append((tool_name, arguments, meta, timeout))
        return self._result

    async def close(self, *, config):
        self.closed.append(config)


async def test_call_with_auth_passes_meta():
    ok = _result(is_error=False, texts=["done"])
    client = _FakeMcpClient(ok)
    out = await ti.call_with_auth(
        _managed_stdio(),
        ManagedAuth(access_token="tok"),
        "stdio",
        "send",
        {"a": 1},
        cast(FastMCPClient, client),
    )
    assert out is ok
    assert client.calls[0][2] == {CONNECTOR_META_TOKEN_KEY: "tok"}


async def test_call_with_auth_passes_call_timeout():
    # The dispatch hands fastmcp the settings-backed call-time budget so a
    # downstream that accepts the request then stalls cannot hang the caller.
    from tai42_skeleton.settings.mcp_settings import mcp_dispatch_settings

    ok = _result(is_error=False, texts=["done"])
    client = _FakeMcpClient(ok)
    await ti.call_with_auth(
        _managed_stdio(),
        ManagedAuth(access_token="tok"),
        "stdio",
        "send",
        {"a": 1},
        cast(FastMCPClient, client),
    )
    assert client.calls[0][3] == mcp_dispatch_settings().call_timeout_seconds


async def test_handle_token_expired_retry_succeeds(monkeypatch):
    async def fake_force(cid, *, failed_access_token=None):
        return ManagedAuth(access_token="fresh")

    monkeypatch.setattr(ti, "force_refresh", fake_force)
    ok = _result(is_error=False, texts=["done"])
    client = _FakeMcpClient(ok)
    out = await ti.handle_token_expired(
        _managed_http(),
        "http",
        "send",
        {},
        cast(FastMCPClient, client),
        ManagedAuth(access_token="old"),
    )
    assert out is ok


async def test_handle_token_expired_second_expiry_raises(monkeypatch):
    async def fake_force(cid, *, failed_access_token=None):
        return ManagedAuth(access_token="fresh")

    monkeypatch.setattr(ti, "force_refresh", fake_force)
    payload = json.dumps({"code": "token_expired"})
    again = _result(is_error=True, texts=[f"{CONNECTOR_ERROR_PREFIX}{payload}"])
    client = _FakeMcpClient(again)
    with pytest.raises(ConnectorAuthExpiredError):
        await ti.handle_token_expired(
            _managed_http(),
            "http",
            "send",
            {},
            cast(FastMCPClient, client),
            ManagedAuth(access_token="old"),
        )


# -- pooled-session eviction on token rotation --------------------------------


class _FakeSession:
    """One pooled MCP session, keyed in :class:`_FakePool` by its connection config."""

    def __init__(self, result) -> None:
        self.result = result
        self.close_calls = 0
        self.close_raises = False

    async def call_tool_mcp(self, tool_name, arguments, meta=None, timeout=None):
        return self.result

    async def _close(self) -> None:
        self.close_calls += 1
        if self.close_raises:
            raise RuntimeError("close failed")


class _FakePool:
    """Stand-in for the kit pooled client: a per-key session map with the
    exact-key ``close`` evict :func:`handle_token_expired` relies on. ``current``
    reuses an existing key's session or opens a fresh one, mirroring the real pool.
    """

    def __init__(self, new_session_result) -> None:
        self._new_session_result = new_session_result
        self.pool: dict[str, _FakeSession] = {}

    @staticmethod
    def _key(config) -> str:
        return json.dumps({"config": config}, sort_keys=True)

    def seed(self, config_dump, session: _FakeSession) -> None:
        self.pool[self._key(config_dump)] = session

    def current(self, *, config):
        key = self._key(config)
        session = self.pool.get(key)
        if session is None:
            session = _FakeSession(self._new_session_result)
            self.pool[key] = session

        @asynccontextmanager
        async def _cm():
            yield session

        return _cm()

    async def close(self, *, config) -> None:
        session = self.pool.pop(self._key(config), None)
        if session is not None:
            await session._close()


def _effective_dump(config: TaiMCPConfig, auth: ManagedAuth, transport: str) -> dict:
    """The pool-key config a dispatch with ``auth`` opens — the pure preparation
    the eviction path itself uses to compute the superseded/fresh keys."""
    return ti._prepare_request(config, auth, transport)[0].model_dump()


def _fresh_refresher(monkeypatch, token: str = "fresh"):
    async def fake_force(cid, *, failed_access_token=None):
        return ManagedAuth(access_token=token)

    monkeypatch.setattr(ti, "force_refresh", fake_force)


async def test_http_rotation_evicts_superseded_keeps_fresh(monkeypatch):
    _fresh_refresher(monkeypatch)
    config = _managed_http()
    superseded = ManagedAuth(access_token="old")
    ok = _result(is_error=False, texts=["done"])
    pool = _FakePool(ok)
    old_session = _FakeSession(_result(is_error=True, texts=["stale"]))
    pool.seed(_effective_dump(config, superseded, "http"), old_session)

    out = await ti.handle_token_expired(
        config, "http", "send", {}, cast(FastMCPClient, pool), superseded, failed_access_token="old"
    )

    assert out is ok
    # Superseded key closed and evicted; only the fresh key survives.
    assert old_session.close_calls == 1
    fresh_key = pool._key(_effective_dump(config, ManagedAuth(access_token="fresh"), "http"))
    assert list(pool.pool) == [fresh_key]


async def test_stdio_rotation_closes_nothing(monkeypatch):
    _fresh_refresher(monkeypatch)
    config = _managed_stdio()
    superseded = ManagedAuth(access_token="old")
    ok = _result(is_error=False, texts=["done"])
    pool = _FakePool(ok)
    # stdio carries the token in _meta, so rotation keeps the same pool key: the
    # live session is seeded there and the retry reuses it.
    live_dump = _effective_dump(config, superseded, "stdio")
    live = _FakeSession(ok)
    pool.seed(live_dump, live)

    out = await ti.handle_token_expired(
        config, "stdio", "send", {}, cast(FastMCPClient, pool), superseded, failed_access_token="old"
    )

    assert out is ok
    # Equal keys → nothing closed, the live just-retried session survives.
    assert live.close_calls == 0
    assert list(pool.pool) == [pool._key(live_dump)]


async def test_close_error_is_non_fatal(monkeypatch, caplog):
    _fresh_refresher(monkeypatch)
    config = _managed_http()
    superseded = ManagedAuth(access_token="old")
    ok = _result(is_error=False, texts=["done"])
    pool = _FakePool(ok)
    old_session = _FakeSession(ok)
    old_session.close_raises = True
    pool.seed(_effective_dump(config, superseded, "http"), old_session)

    with caplog.at_level(logging.WARNING):
        out = await ti.handle_token_expired(
            config, "http", "send", {}, cast(FastMCPClient, pool), superseded, failed_access_token="old"
        )

    # A failing close never disturbs the retry's result.
    assert out is ok
    assert old_session.close_calls == 1
    assert "failed to close superseded MCP session after token rotation" in caplog.text


async def test_eviction_fires_on_error_result(monkeypatch):
    _fresh_refresher(monkeypatch)
    config = _managed_http()
    superseded = ManagedAuth(access_token="old")
    err = _result(is_error=True, texts=["plain tool error"])  # not a token_expired sentinel
    pool = _FakePool(err)
    old_session = _FakeSession(err)
    pool.seed(_effective_dump(config, superseded, "http"), old_session)

    out = await ti.handle_token_expired(
        config, "http", "send", {}, cast(FastMCPClient, pool), superseded, failed_access_token="old"
    )

    assert out is err
    assert old_session.close_calls == 1


async def test_eviction_fires_on_second_token_expired_raise(monkeypatch):
    _fresh_refresher(monkeypatch)
    config = _managed_http()
    superseded = ManagedAuth(access_token="old")
    payload = json.dumps({"code": "token_expired"})
    again = _result(is_error=True, texts=[f"{CONNECTOR_ERROR_PREFIX}{payload}"])
    pool = _FakePool(again)
    old_session = _FakeSession(again)
    pool.seed(_effective_dump(config, superseded, "http"), old_session)

    with pytest.raises(ConnectorAuthExpiredError):
        await ti.handle_token_expired(
            config, "http", "send", {}, cast(FastMCPClient, pool), superseded, failed_access_token="old"
        )

    # Evicted while ConnectorAuthExpiredError propagates out of the finally.
    assert old_session.close_calls == 1


async def test_eviction_never_before_retry(monkeypatch):
    async def failing_force(cid, *, failed_access_token=None):
        raise RuntimeError("refresh boom")

    monkeypatch.setattr(ti, "force_refresh", failing_force)
    config = _managed_http()
    superseded = ManagedAuth(access_token="old")
    pool = _FakePool(_result(is_error=False, texts=["unused"]))
    superseded_dump = _effective_dump(config, superseded, "http")
    old_session = _FakeSession(_result(is_error=False, texts=["unused"]))
    pool.seed(superseded_dump, old_session)

    with pytest.raises(RuntimeError, match="refresh boom"):
        await ti.handle_token_expired(
            config, "http", "send", {}, cast(FastMCPClient, pool), superseded, failed_access_token="old"
        )

    # The refresh failed before the retry ran, so no eviction happened.
    assert old_session.close_calls == 0
    assert list(pool.pool) == [pool._key(superseded_dump)]


# -- Import hygiene -----------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "tai42_skeleton.connectors.token_injection",
        "tai42_skeleton.tools.adapters.mcp_tool_to_func",
    ],
)
def test_module_imports_alone_in_fresh_interpreter(module: str) -> None:
    """Each dispatch module must import cleanly as the FIRST skeleton import in a
    fresh interpreter. ``mcp_tool_to_func`` imports ``token_injection`` and both
    read ``settings.mcp_settings``; importing either one first must not re-enter a
    half-initialized peer. A subprocess gives the pristine ``sys.modules`` an
    in-process test (collected after the package graph is already warm) cannot."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
