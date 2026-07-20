"""Edge branches of the ``InMemoryHooksManager`` registry: removing one hook from
a topic that still holds another leaves the topic bucket in place, and
re-registering a name under a new topic moves it out of the old topic's bucket.
"""

from __future__ import annotations

from tai_contract.hooks import HookParams

from tai_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai_skeleton.hooks.settings import HooksSettings


async def test_unregister_one_of_two_keeps_topic_bucket():
    manager = InMemoryHooksManager(HooksSettings())
    await manager.register(HookParams(name="a", topic="t", tool="x"))
    await manager.register(HookParams(name="b", topic="t", tool="y"))

    assert await manager.unregister("a") is True
    # The topic bucket survives because "b" is still registered under it.
    assert set((await manager.list_hooks_by_topic("t")).keys()) == {"b"}

    assert await manager.unregister("b") is True
    assert await manager.list_hooks_by_topic("t") == {}


async def test_reregister_under_new_topic_moves_hook(make_app):
    # Re-registering the same name under a new topic must drop the old topic's
    # bucket entry: the old topic fires nothing, the new topic fires the hook,
    # and unregister still works afterwards.
    app = make_app()
    manager = InMemoryHooksManager(HooksSettings())
    await manager.register(HookParams(name="mv", topic="topic-a", tool="mv_tool"))
    await manager.register(HookParams(name="mv", topic="topic-b", tool="mv_tool"))

    await manager.on_event("topic-a", {})
    assert app.tools.runs == []

    await manager.on_event("topic-b", {})
    assert app.tools.runs == [("mv_tool", {})]

    assert await manager.unregister("mv") is True
    assert await manager.list_hooks() == {}
    assert await manager.list_hooks_by_topic("topic-b") == {}
