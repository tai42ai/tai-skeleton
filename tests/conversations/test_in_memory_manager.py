"""The in-memory routing-row store refuses every operation with a loud 501 — a durable,
backed-up routing table cannot be per-process, restart-volatile state."""

from __future__ import annotations

import pytest
from tai42_contract.conversations import ConversationRoute

from tai42_skeleton.conversations.managers.in_memory_conversations_manager import InMemoryConversationsManager
from tai42_skeleton.conversations.settings import ConversationsSettings
from tai42_skeleton.operations.errors import NotSupportedError


def _manager() -> InMemoryConversationsManager:
    return InMemoryConversationsManager(ConversationsSettings())


def _route() -> ConversationRoute:
    return ConversationRoute(
        route_name="r",
        door="api",
        agent_name="a",
        execution_key="svc",
        callback_url="https://example.com/cb",
        execution_key_fingerprint="fp",
    )


async def test_every_operation_raises_not_supported():
    manager = _manager()
    with pytest.raises(NotSupportedError):
        await manager.put_route(_route())
    with pytest.raises(NotSupportedError):
        await manager.get_route("r")
    with pytest.raises(NotSupportedError):
        await manager.delete_route("r")
    with pytest.raises(NotSupportedError):
        await manager.list_routes()


def test_not_supported_maps_to_501():
    assert NotSupportedError("x").status == 501
