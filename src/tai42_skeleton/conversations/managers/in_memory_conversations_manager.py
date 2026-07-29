from tai42_contract.conversations import ConversationRoute

from tai42_skeleton.conversations.managers.base_conversations_manager import BaseConversationsManager
from tai42_skeleton.operations.errors import NotSupportedError

# A capability the deployment does not provide (501), not a transient outage (503).
_IN_MEMORY_REFUSAL = "conversation routes require the redis conversations backend"


class InMemoryConversationsManager(BaseConversationsManager):
    """The in-memory routing-row store — a null backend refusing every operation with a
    typed 501, because a durable routing table cannot live per-process."""

    async def put_route(self, route: ConversationRoute) -> bool:
        raise NotSupportedError(_IN_MEMORY_REFUSAL)

    async def get_route(self, route_name: str) -> ConversationRoute | None:
        raise NotSupportedError(_IN_MEMORY_REFUSAL)

    async def delete_route(self, route_name: str) -> bool:
        raise NotSupportedError(_IN_MEMORY_REFUSAL)

    async def list_routes(self) -> dict[str, ConversationRoute]:
        raise NotSupportedError(_IN_MEMORY_REFUSAL)
