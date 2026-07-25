from abc import ABC, abstractmethod

from tai42_contract.conversations import ConversationRoute

from tai42_skeleton.conversations.settings import ConversationsSettings


class BaseConversationsManager(ABC):
    """The routing-row store — keyspace 4 of the conversation bridge: the durable,
    backed-up mapping from an inbound door to the agent a turn runs and the execution key
    it runs AS. The in-memory backend refuses every operation with a loud 501.
    """

    def __init__(self, settings: ConversationsSettings) -> None:
        self.settings = settings

    @abstractmethod
    async def put_route(self, route: ConversationRoute) -> bool:
        """Store ``route`` (an upsert — create or replace), keeping the name index in
        lockstep. Return ``True`` when the row is newly created, ``False`` when it
        replaced an existing row of the same name."""
        ...

    @abstractmethod
    async def get_route(self, route_name: str) -> ConversationRoute | None:
        """The stored row for ``route_name`` (its ``callback_secret`` included, for the
        delivery executor), or ``None`` when no such route exists."""
        ...

    @abstractmethod
    async def delete_route(self, route_name: str) -> bool:
        """Remove the row for ``route_name``, keeping the name index in lockstep. Return
        ``True`` when a row was removed, ``False`` when none existed."""
        ...

    @abstractmethod
    async def list_routes(self) -> dict[str, ConversationRoute]:
        """Every stored routing row keyed by route name (each ``callback_secret``
        included, for internal consumers)."""
        ...
