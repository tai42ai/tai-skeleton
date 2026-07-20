"""Per-topic verifier binding store, parametrized over BOTH hooks manager
backends (in-memory and redis) through their shared interface."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tai_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai_skeleton.hooks.managers.redis_hooks_manager import RedisHooksManager
from tai_skeleton.hooks.settings import HooksSettings

_BINDING = {"verifier": "shared_secret", "config": {"header": "X-Token", "secret_env": "WH_SECRET"}}


def _in_memory() -> InMemoryHooksManager:
    return InMemoryHooksManager(HooksSettings())


def _redis(fake_redis, monkeypatch, make_ctx) -> RedisHooksManager:
    from tai_skeleton.hooks.managers import redis_hooks_manager

    monkeypatch.setattr(redis_hooks_manager, "client_ctx", make_ctx(fake_redis))
    # A redis_url makes HooksSettings.in_memory False; the connection is faked.
    monkeypatch.setenv("HOOKS_REDIS_URL", "redis://localhost:6379/0")
    return RedisHooksManager(HooksSettings())


@pytest.fixture(params=["in_memory", "redis"])
def manager(request, fake_redis, monkeypatch, make_ctx):
    if request.param == "in_memory":
        return _in_memory()
    return _redis(fake_redis, monkeypatch, make_ctx)


async def test_set_get_round_trip(manager) -> None:
    assert await manager.get_topic_verifier("orders") is None
    await manager.set_topic_verifier("orders", _BINDING)
    assert await manager.get_topic_verifier("orders") == _BINDING


async def test_replace(manager) -> None:
    await manager.set_topic_verifier("orders", _BINDING)
    replacement = {"verifier": "github", "config": {"secret_env": "GH"}}
    await manager.set_topic_verifier("orders", replacement)
    assert await manager.get_topic_verifier("orders") == replacement


async def test_all_topic_verifiers(manager) -> None:
    await manager.set_topic_verifier("orders", _BINDING)
    await manager.set_topic_verifier("alerts", {"verifier": "github", "config": {}})
    everything = await manager.all_topic_verifiers()
    assert everything == {"orders": _BINDING, "alerts": {"verifier": "github", "config": {}}}


async def test_delete(manager) -> None:
    await manager.set_topic_verifier("orders", _BINDING)
    assert await manager.delete_topic_verifier("orders") is True
    assert await manager.get_topic_verifier("orders") is None
    # A second delete reports nothing removed.
    assert await manager.delete_topic_verifier("orders") is False


@pytest.mark.parametrize("bad", [{"config": {}}, {"verifier": 123}, {"verifier": "x", "config": "nope"}])
async def test_set_rejects_wrong_shape(manager, bad) -> None:
    # Both backends validate the binding shape against ``TopicVerifierBinding`` on
    # write, so a malformed binding can never be stored.
    with pytest.raises(ValidationError):
        await manager.set_topic_verifier("orders", bad)


async def test_redis_reads_reject_wrong_shape_binding(fake_redis, monkeypatch, make_ctx) -> None:
    # A valid-JSON but wrong-shape value already in the store raises loudly at
    # BOTH read boundaries (single-topic and all-topics), matching every other
    # read in the module — never flows untyped into the ingress.
    manager = _redis(fake_redis, monkeypatch, make_ctx)
    fake_redis._hashes[manager.settings.topic_verifiers_key] = {"orders": '{"missing": "verifier"}'}

    with pytest.raises(ValidationError):
        await manager.get_topic_verifier("orders")
    with pytest.raises(ValidationError):
        await manager.all_topic_verifiers()
