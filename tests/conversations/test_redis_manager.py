"""``RedisConversationsManager`` put / get / delete / list against the faked redis
string + set seam."""

from __future__ import annotations

from typing import Any

from tai42_contract.conversations import ConversationRoute

from tai42_skeleton.conversations.managers import redis_conversations_manager as redis_module
from tai42_skeleton.conversations.managers.redis_conversations_manager import RedisConversationsManager
from tai42_skeleton.conversations.settings import ConversationsSettings

from .conftest import FakeRedis, make_client_ctx


def _manager(monkeypatch, fake) -> RedisConversationsManager:
    monkeypatch.setattr(redis_module, "client_ctx", make_client_ctx(fake))
    return RedisConversationsManager(ConversationsSettings())


def _api_route(name: str = "support", **over) -> ConversationRoute:
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


def _channel_route(name: str = "line", **over) -> ConversationRoute:
    fields: dict[str, Any] = {
        "route_name": name,
        "door": "channel",
        "agent_name": "triage",
        "execution_key": "svc",
        "channel": "twilio",
        "our_identity": "+15550001111",
        "execution_key_fingerprint": "fp-1",
    }
    fields.update(over)
    return ConversationRoute(**fields)


async def test_put_creates_then_replaces(monkeypatch):
    fake = FakeRedis()
    manager = _manager(monkeypatch, fake)

    assert await manager.put_route(_api_route()) is True  # newly created
    assert await manager.put_route(_api_route(agent_name="other")) is False  # replace

    got = await manager.get_route("support")
    assert got is not None
    assert got.agent_name == "other"
    # The name index and the per-route key stay in lockstep.
    assert fake._sets["conversations:route_names"] == {"support"}


async def test_get_returns_none_for_unknown(monkeypatch):
    manager = _manager(monkeypatch, FakeRedis())
    assert await manager.get_route("nope") is None


async def test_get_carries_the_callback_secret_for_internal_consumers(monkeypatch):
    manager = _manager(monkeypatch, FakeRedis())
    await manager.put_route(_api_route(callback_secret="the-secret"))
    got = await manager.get_route("support")
    assert got is not None
    assert got.callback_secret == "the-secret"


async def test_delete_removes_and_reports(monkeypatch):
    fake = FakeRedis()
    manager = _manager(monkeypatch, fake)
    await manager.put_route(_api_route())

    assert await manager.delete_route("support") is True
    assert await manager.get_route("support") is None
    assert await manager.delete_route("support") is False
    assert fake._sets["conversations:route_names"] == set()


async def test_list_routes_returns_every_row(monkeypatch):
    manager = _manager(monkeypatch, FakeRedis())
    await manager.put_route(_api_route("a"))
    await manager.put_route(_channel_route("b"))

    listed = await manager.list_routes()
    assert set(listed) == {"a", "b"}
    assert listed["a"].door == "api"
    assert listed["b"].door == "channel"


async def test_list_routes_skips_an_indexed_but_missing_row(monkeypatch):
    fake = FakeRedis()
    manager = _manager(monkeypatch, fake)
    await manager.put_route(_api_route("live"))
    await manager.put_route(_api_route("orphan"))
    # Drop only the row key, leaving the name in the index behind.
    del fake._strings["conversations:route:orphan"]

    listed = await manager.list_routes()
    assert set(listed) == {"live"}


async def test_list_routes_empty_when_no_index(monkeypatch):
    manager = _manager(monkeypatch, FakeRedis())
    assert await manager.list_routes() == {}
