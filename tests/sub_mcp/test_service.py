"""The sub-MCP write service: the store-write-FIRST durability contract, up-front
validation (invalid input never reaches the store), and the store+local unregister.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tai_contract.app import tai_app

from tai_skeleton.sub_mcp import service
from tai_skeleton.sub_mcp import store as store_mod
from tai_skeleton.sub_mcp.store import InMemorySubMcpStore


class _FakeRouter:
    def __init__(self, routes=None, register_error: Exception | None = None):
        self.routes = dict(routes or {})
        self.registered: list[tuple] = []
        self.unregistered: list[str] = []
        self._register_error = register_error

    async def register_sub_mcp_app(self, slug, tools, transport="http"):
        if self._register_error is not None:
            raise self._register_error
        self.registered.append((slug, tools, transport))
        from tai_contract.sub_mcp import RouteConfig

        self.routes[slug] = RouteConfig(tools=tools, transport=transport)

    async def unregister_sub_mcp_app(self, slug):
        self.unregistered.append(slug)
        self.routes.pop(slug, None)


@pytest.fixture
def wired(monkeypatch):
    """Install a fresh in-memory store + a fake router behind ``tai_app``."""

    def _wire(router: _FakeRouter) -> InMemorySubMcpStore:
        fresh = InMemorySubMcpStore()
        monkeypatch.setattr(store_mod, "_IN_MEMORY_STORE", fresh)
        monkeypatch.setattr(tai_app, "_impl", SimpleNamespace(sub_app=SimpleNamespace(mcp_sub_app_router=router)))
        return fresh

    return _wire


async def test_register_writes_store_before_router(wired):
    router = _FakeRouter()
    store = wired(router)
    await service.register_sub_mcp_app("weather", ["get_forecast"], transport="sse")
    # Both halves landed, store first.
    assert (await store.get_route("weather")).tools == ["get_forecast"]
    assert router.registered == [("weather", ["get_forecast"], "sse")]


async def test_router_swap_failure_leaves_registration_recoverable(wired):
    # A crash in the in-process router swap AFTER the store write must leave the
    # registration DURABLE — the store-write-FIRST contract — so the next rehydrate
    # re-materializes it rather than losing it forever.
    router = _FakeRouter(register_error=RuntimeError("router boom"))
    store = wired(router)
    with pytest.raises(RuntimeError, match="router boom"):
        await service.register_sub_mcp_app("weather", ["get_forecast"])
    # The store write happened before the failing swap, so the registration survives.
    assert (await store.get_route("weather")).tools == ["get_forecast"]


async def test_invalid_slug_never_reaches_the_store(wired):
    # Validation runs BEFORE the store write, so a malformed slug raises without
    # persisting a garbage entry (and without touching the router).
    router = _FakeRouter()
    store = wired(router)
    with pytest.raises(ValueError, match="must match"):
        await service.register_sub_mcp_app("bad/slug", ["get_forecast"])
    assert await store.list_routes() == {}
    assert router.registered == []


async def test_invalid_transport_never_reaches_the_store(wired):
    router = _FakeRouter()
    store = wired(router)
    with pytest.raises(ValueError, match="transport"):
        await service.register_sub_mcp_app("weather", ["get_forecast"], transport="carrier-pigeon")
    assert await store.list_routes() == {}


async def test_unregister_removes_from_store_and_router(wired):
    from tai_contract.sub_mcp import RouteConfig

    router = _FakeRouter(routes={"weather": RouteConfig(tools=["get_forecast"])})
    store = wired(router)
    await store.save_route("weather", RouteConfig(tools=["get_forecast"]))
    removed = await service.unregister_sub_mcp_app("weather")
    assert removed is True
    assert await store.get_route("weather") is None
    assert router.unregistered == ["weather"]


async def test_unregister_store_only_slug_returns_true(wired):
    from tai_contract.sub_mcp import RouteConfig

    router = _FakeRouter()
    store = wired(router)
    await store.save_route("remote", RouteConfig(tools=["get_forecast"]))
    removed = await service.unregister_sub_mcp_app("remote")
    assert removed is True
    assert await store.get_route("remote") is None
    # Nothing was bound here, so the local router was left alone.
    assert router.unregistered == []


async def test_unregister_absent_slug_returns_false(wired):
    router = _FakeRouter()
    wired(router)
    assert await service.unregister_sub_mcp_app("ghost") is False
    assert router.unregistered == []
