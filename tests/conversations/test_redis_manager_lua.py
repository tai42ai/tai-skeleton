"""End-to-end execution of the routing-table manager's Redis Lua scripts.

``test_redis_manager`` drives a Python re-implementation of ``_PUT_LUA`` / ``_DELETE_LUA``;
this module runs the REAL script text against ``fakeredis[lua]``, so the atomic name-index
step each one carries is exercised, not just a stand-in.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fakeredis import aioredis
from tai42_contract.conversations import ConversationRoute

from tai42_skeleton.conversations.managers import redis_conversations_manager as redis_module
from tai42_skeleton.conversations.managers.redis_conversations_manager import RedisConversationsManager
from tai42_skeleton.conversations.settings import ConversationsSettings

_NAMES_KEY = "conversations:route_names"


@pytest.fixture(autouse=True)
def _conversations_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONVERSATIONS_REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
async def lua_redis(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[aioredis.FakeRedis]:
    client = aioredis.FakeRedis(decode_responses=True)

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        yield client

    monkeypatch.setattr(redis_module, "client_ctx", fake_client_ctx)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def manager(lua_redis: aioredis.FakeRedis) -> RedisConversationsManager:
    return RedisConversationsManager(ConversationsSettings())


def _api_route(name: str = "support", **over: Any) -> ConversationRoute:
    fields: dict[str, Any] = {
        "route_name": name,
        "door": "api",
        "agent_name": "triage",
        "execution_key": "svc",
        "callback_url": "https://example.com/cb",
        "execution_key_fingerprint": "fp-1",
        "callback_secret": "sec-1",
    }
    fields.update(over)
    return ConversationRoute(**fields)


async def test_put_keys_and_indexes_the_row_atomically(manager, lua_redis):
    # A fresh create returns True and the SADD indexes the name; a replace returns False and
    # leaves exactly one index member.
    assert await manager.put_route(_api_route()) is True
    assert await lua_redis.smembers(_NAMES_KEY) == {"support"}
    assert await lua_redis.get(ConversationsSettings().route_key("support")) is not None

    assert await manager.put_route(_api_route(agent_name="other")) is False
    assert await lua_redis.smembers(_NAMES_KEY) == {"support"}
    got = await manager.get_route("support")
    assert got is not None
    assert got.agent_name == "other"


async def test_delete_removes_the_row_and_unindexes_it_atomically(manager, lua_redis):
    await manager.put_route(_api_route())

    assert await manager.delete_route("support") is True
    assert await lua_redis.exists(ConversationsSettings().route_key("support")) == 0
    assert await lua_redis.smembers(_NAMES_KEY) == set()
    # A second delete removes nothing and says so, and the SREM is harmless on an absent name.
    assert await manager.delete_route("support") is False


async def test_list_reads_the_index_and_the_rows_in_lockstep(manager, lua_redis):
    await manager.put_route(_api_route("a"))
    await manager.put_route(_api_route("b"))

    listed = await manager.list_routes()
    assert set(listed) == {"a", "b"}
    assert await lua_redis.smembers(_NAMES_KEY) == {"a", "b"}
