"""The durable sub-MCP registration store: both impls' round-trip surface, the
malformed-value loud failure, mode selection by ``SUB_MCP_REDIS_URL``, and the
derived settings keys / ``in_memory`` flag.

The Redis impl runs against the shared ``fake_redis`` + ``fake_client_ctx``
fixtures (the same offline seam the interactions/hooks suites use), extended with
the hash ``hget``/``hset``/``hdel`` commands the store needs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tai42_contract.sub_mcp import RouteConfig
from tai42_kit.settings import reset_all_settings

from tai42_skeleton.sub_mcp import store as store_mod
from tai42_skeleton.sub_mcp.settings import SubMcpRedisSettings, SubMcpSettings, sub_mcp_settings
from tai42_skeleton.sub_mcp.store import (
    InMemorySubMcpStore,
    RedisSubMcpStore,
    get_sub_mcp_store,
)


@pytest.fixture
def redis_store(monkeypatch, fake_client_ctx) -> RedisSubMcpStore:
    """A ``RedisSubMcpStore`` whose pooled client opens the shared fake redis."""
    monkeypatch.setattr(store_mod, "client_ctx", fake_client_ctx)
    return RedisSubMcpStore(SubMcpSettings(redis=SubMcpRedisSettings(redis_url="redis://localhost:6379/0")))


# -- Redis impl ---------------------------------------------------------------


async def test_redis_round_trip(redis_store):
    assert await redis_store.get_route("weather") is None
    assert await redis_store.list_routes() == {}

    await redis_store.save_route("weather", RouteConfig(tools=["get_forecast"], transport="sse"))
    await redis_store.save_route("news", RouteConfig(tools=["headlines"]))

    got = await redis_store.get_route("weather")
    assert got == RouteConfig(tools=["get_forecast"], transport="sse")
    assert await redis_store.list_routes() == {
        "weather": RouteConfig(tools=["get_forecast"], transport="sse"),
        "news": RouteConfig(tools=["headlines"], transport="http"),
    }


async def test_redis_save_overwrites(redis_store):
    await redis_store.save_route("weather", RouteConfig(tools=["old"]))
    await redis_store.save_route("weather", RouteConfig(tools=["new"]))
    got = await redis_store.get_route("weather")
    assert got.tools == ["new"]


async def test_redis_delete_reports_existence(redis_store):
    await redis_store.save_route("weather", RouteConfig(tools=["get_forecast"]))
    assert await redis_store.delete_route("weather") is True
    # The field is gone, so a second delete reports it did not exist.
    assert await redis_store.delete_route("weather") is False
    assert await redis_store.get_route("weather") is None


async def test_redis_malformed_value_raises(redis_store, fake_redis):
    # A hand-corrupted stored value must raise loudly on read, never skip-and-continue.
    await fake_redis.hset(redis_store._settings.routes_key, "weather", "not json")
    with pytest.raises(ValidationError):
        await redis_store.get_route("weather")
    with pytest.raises(ValidationError):
        await redis_store.list_routes()


# -- in-memory impl (same surface) --------------------------------------------


async def test_in_memory_round_trip():
    store = InMemorySubMcpStore()
    assert await store.get_route("weather") is None
    assert await store.list_routes() == {}

    await store.save_route("weather", RouteConfig(tools=["get_forecast"], transport="sse"))
    assert await store.get_route("weather") == RouteConfig(tools=["get_forecast"], transport="sse")
    assert await store.list_routes() == {"weather": RouteConfig(tools=["get_forecast"], transport="sse")}

    assert await store.delete_route("weather") is True
    assert await store.delete_route("weather") is False
    assert await store.get_route("weather") is None


async def test_in_memory_list_returns_a_copy():
    # list_routes returns a fresh dict so a concurrent write can't mutate a caller's
    # iteration (the rehydrate-mid-register hazard).
    store = InMemorySubMcpStore()
    await store.save_route("a", RouteConfig(tools=["t"]))
    snapshot = await store.list_routes()
    await store.save_route("b", RouteConfig(tools=["t"]))
    assert snapshot == {"a": RouteConfig(tools=["t"], transport="http")}


# -- mode selection -----------------------------------------------------------


def test_selects_in_memory_when_no_redis_url(monkeypatch):
    monkeypatch.delenv("SUB_MCP_REDIS_URL", raising=False)
    sub_mcp_settings.cache_clear()
    try:
        assert isinstance(get_sub_mcp_store(), InMemorySubMcpStore)
    finally:
        sub_mcp_settings.cache_clear()


def test_selects_redis_when_redis_url_set(monkeypatch):
    monkeypatch.setenv("SUB_MCP_REDIS_URL", "redis://localhost:6379/0")
    sub_mcp_settings.cache_clear()
    try:
        assert isinstance(get_sub_mcp_store(), RedisSubMcpStore)
    finally:
        sub_mcp_settings.cache_clear()


async def test_in_memory_store_survives_settings_reset(monkeypatch):
    # The in-memory store is deliberately reset-EXEMPT: its whole purpose is to
    # survive reload_config's settings-reset wipe within the process. A route saved
    # in in-memory mode must still be there after reset_all_settings() — the single
    # most load-bearing durability property of the reset-exempt singleton.
    monkeypatch.delenv("SUB_MCP_REDIS_URL", raising=False)
    sub_mcp_settings.cache_clear()
    store = get_sub_mcp_store()
    assert isinstance(store, InMemorySubMcpStore)
    try:
        await store.save_route("survivor", RouteConfig(tools=["get_forecast"], transport="sse"))
        reset_all_settings()
        # Same durable singleton, route intact — the reset did not clear it.
        after = get_sub_mcp_store()
        assert after is store
        assert await after.get_route("survivor") == RouteConfig(tools=["get_forecast"], transport="sse")
    finally:
        await store.delete_route("survivor")
        sub_mcp_settings.cache_clear()


def test_in_memory_singleton_is_stable(monkeypatch):
    # In-memory mode returns the SAME module-level singleton every call, so a
    # registration survives across accessor calls within a process (the whole point
    # of the reset-exempt store).
    monkeypatch.delenv("SUB_MCP_REDIS_URL", raising=False)
    sub_mcp_settings.cache_clear()
    try:
        assert get_sub_mcp_store() is get_sub_mcp_store()
    finally:
        sub_mcp_settings.cache_clear()


# -- settings -----------------------------------------------------------------


def test_in_memory_flag_reflects_redis_url():
    assert SubMcpSettings().in_memory is True
    with_url = SubMcpSettings(redis=SubMcpRedisSettings(redis_url="redis://localhost:6379/0"))
    assert with_url.in_memory is False


def test_routes_key_derived_from_prefix():
    assert SubMcpSettings().routes_key == "sub_mcp:routes"
    assert SubMcpSettings(prefix="custom").routes_key == "custom:routes"
