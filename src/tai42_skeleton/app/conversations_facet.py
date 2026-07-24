"""The ``app.conversations`` namespace, forwarding to the conversation bridge in
:mod:`tai42_skeleton.conversations`.

``accept`` turns a received message into an agent turn and returns the new message's id;
``record_delivery_status`` is the out-of-band sink an adapter calls when a provider later
reports an outbound message's terminal fate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tai42_contract.conversations import DeliveryReceipt

if TYPE_CHECKING:
    from tai42_skeleton.app.server import TaiMCP


class ConversationsFacet:
    """``app.conversations`` — the bridge's inbound + delivery-receipt entry surface
    (``AppConversations``)."""

    __slots__ = ("_app",)

    def __init__(self, app: TaiMCP) -> None:
        self._app = app

    async def accept(
        self,
        channel: str,
        our_identity: str,
        client_address: str,
        text: str,
        provider_message_id: str,
    ) -> str:
        return await self._app._conversation_accept(channel, our_identity, client_address, text, provider_message_id)

    async def record_delivery_status(self, channel: str, provider_message_id: str, status: DeliveryReceipt) -> None:
        await self._app._conversation_record_delivery_status(channel, provider_message_id, status)
