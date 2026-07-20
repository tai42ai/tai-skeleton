"""Conformance + behavior tests for the copied hooks feature.

Conformance asserts the local managers satisfy the ``tai_contract.hooks``
``HooksManager`` protocol they are meant to implement. Behavior exercises the
in-memory registry (register / list / unregister) plus jq validation at
registration time — none of which depend on the template manager (firing, which
renders condition/expr via the template impl, is verified at integration).
"""

import pytest
from tai_contract.hooks import HookParams, HooksManager

from tai_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai_skeleton.hooks.managers.redis_hooks_manager import RedisHooksManager
from tai_skeleton.hooks.settings import HooksSettings


def _settings() -> HooksSettings:
    return HooksSettings()


def test_in_memory_manager_satisfies_contract_protocol():
    manager = InMemoryHooksManager(_settings())
    assert isinstance(manager, HooksManager)


def test_redis_manager_satisfies_contract_protocol():
    manager = RedisHooksManager(_settings())
    assert isinstance(manager, HooksManager)


async def test_in_memory_register_list_unregister():
    manager = InMemoryHooksManager(_settings())
    hook = HookParams(name="h1", topic="orders", tool="ship", condition='.status == "paid"')

    assert await manager.register(hook) is True

    assert set((await manager.list_hooks()).keys()) == {"h1"}
    by_topic = await manager.list_hooks_by_topic("orders")
    assert by_topic["h1"].tool == "ship"
    assert await manager.list_hooks_by_topic("other") == {}

    assert await manager.unregister("h1") is True
    assert await manager.list_hooks() == {}
    assert await manager.unregister("h1") is False


async def test_register_rejects_invalid_jq():
    manager = InMemoryHooksManager(_settings())
    bad = HookParams(name="bad", topic="t", tool="noop", condition="this is ( not jq")

    with pytest.raises(ValueError, match="not valid jq"):
        await manager.register(bad)
