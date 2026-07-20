"""MCP liveness probe + verbose verify over a faked pooled FastMCP client."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import tai42_skeleton.connectors.runtime.probe as probe_mod
from tai42_skeleton.connectors.runtime.probe import probe, verify

from .conftest import make_noauth_stdio_descriptor, make_oauth_descriptor


class _FakeMcpClient:
    def __init__(self, *, tools=None, error: Exception | None = None) -> None:
        self._tools = tools or []
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self):
        if self._error is not None:
            raise self._error
        return self._tools


@pytest.fixture
def install_client(monkeypatch):
    captured = {}

    def _install(*, tools=None, error=None):
        client = _FakeMcpClient(tools=tools, error=error)

        @asynccontextmanager
        async def fake_client_ctx(client_cls, **kwargs):
            captured["kwargs"] = kwargs
            yield client

        monkeypatch.setattr(probe_mod, "client_ctx", fake_client_ctx)
        return captured

    return _install


# -- probe -------------------------------------------------------------------


async def test_probe_unknown_sub_service_is_false(install_client):
    desc = make_oauth_descriptor()
    assert await probe(desc, "nonexistent") is False


async def test_probe_live_returns_true(install_client):
    install_client(tools=[SimpleNamespace(name="t", description="d")])
    desc = make_oauth_descriptor()
    assert await probe(desc, "mail", access_token="at") is True


async def test_probe_unreachable_returns_false(install_client):
    install_client(error=RuntimeError("connect failed"))
    desc = make_oauth_descriptor()
    assert await probe(desc, "mail", access_token="at") is False


async def test_probe_stdio_builds_env_from_config_values(install_client):
    captured = install_client(tools=[])
    desc = make_noauth_stdio_descriptor()
    assert await probe(desc, "search", config_values={"api_key": "k"}) is True
    cfg = captured["kwargs"]["config"]
    assert cfg["config"]["type"] == "stdio"
    assert cfg["config"]["env"]["api_key"] == "k"


# -- verify ------------------------------------------------------------------


async def test_verify_unknown_sub_service(install_client):
    desc = make_oauth_descriptor()
    result = await verify(desc, "nope")
    assert result.ok is False
    assert result.error is not None
    assert "unknown sub_service" in result.error


async def test_verify_success_returns_tools(install_client):
    install_client(
        tools=[
            SimpleNamespace(name="send", description="Send a message"),
            SimpleNamespace(name="read", description=None),
        ]
    )
    desc = make_oauth_descriptor()
    result = await verify(desc, "mail", access_token="at")
    assert result.ok is True
    assert [t.name for t in result.tools] == ["send", "read"]
    assert result.tools[1].description == ""  # None coalesced


async def test_verify_timeout(install_client, monkeypatch):
    install_client(error=TimeoutError())
    desc = make_oauth_descriptor()
    result = await verify(desc, "mail", access_token="at")
    assert result.ok is False
    assert result.error is not None
    assert "timeout" in result.error


async def test_verify_transport_error_returns_fixed_reason(install_client):
    # The raw exception text is attacker-influenceable, so verify() must return a
    # fixed reason and never echo the exception type or its message to the caller.
    install_client(error=ValueError("bad handshake from upstream"))
    desc = make_oauth_descriptor()
    result = await verify(desc, "mail", access_token="at")
    assert result.ok is False
    assert result.error == "transport error: could not complete the MCP handshake"
    assert "ValueError" not in result.error
    assert "bad handshake" not in result.error


async def test_verify_http_sets_bearer_header(install_client):
    captured = install_client(tools=[])
    desc = make_oauth_descriptor()
    await verify(desc, "mail", access_token="my-token")
    cfg = captured["kwargs"]["config"]
    assert cfg["config"]["headers"]["Authorization"] == "Bearer my-token"
