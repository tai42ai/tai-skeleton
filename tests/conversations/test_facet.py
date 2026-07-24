"""The ``app.conversations`` facet as a channel adapter sees it.

A medium adapter reaches the bridge only through ``tai42_app.conversations``, which must
satisfy the contract's :class:`AppConversations` Protocol and forward calls through to
the app's core with the arguments intact.
"""

from __future__ import annotations

import pytest
from tai42_contract.app.facets import AppConversations
from tai42_contract.conversations import DeliveryReceipt

from tai42_skeleton.app.conversations_facet import ConversationsFacet


class _FakeApp:
    """Stands in for ``TaiMCP`` — records what the facet forwards to the core."""

    def __init__(self) -> None:
        self.accepted: list[tuple] = []
        self.receipts: list[tuple] = []

    async def _conversation_accept(self, channel, our_identity, client_address, text, provider_message_id) -> str:
        self.accepted.append((channel, our_identity, client_address, text, provider_message_id))
        return "mid-1"

    async def _conversation_record_delivery_status(self, channel, provider_message_id, status) -> None:
        self.receipts.append((channel, provider_message_id, status))


def test_the_facet_satisfies_the_contract_protocol():
    # A channel adapter type-checks against the contract Protocol alone; the runtime
    # conformance check stands in for that here.
    facet = ConversationsFacet(_FakeApp())  # type: ignore[arg-type]
    assert isinstance(facet, AppConversations)


async def test_channel_side_accept_forwards_to_the_core():
    app = _FakeApp()
    facet: AppConversations = ConversationsFacet(app)  # type: ignore[arg-type]
    message_id = await facet.accept("twilio", "+15550001111", "+15550002222", "hi", "PID1")
    assert message_id == "mid-1"
    assert app.accepted == [("twilio", "+15550001111", "+15550002222", "hi", "PID1")]


async def test_channel_side_record_delivery_status_forwards_to_the_core():
    app = _FakeApp()
    facet: AppConversations = ConversationsFacet(app)  # type: ignore[arg-type]
    await facet.record_delivery_status("twilio", "out-9", DeliveryReceipt.FAILED)
    assert app.receipts == [("twilio", "out-9", DeliveryReceipt.FAILED)]


def test_a_non_conforming_object_is_not_the_protocol():
    # Non-vacuous: the Protocol actually discriminates — a bare object is not it.
    assert not isinstance(object(), AppConversations)


@pytest.mark.parametrize("method", ["accept", "record_delivery_status"])
def test_the_facet_exposes_exactly_the_channel_surface(method):
    facet = ConversationsFacet(_FakeApp())  # type: ignore[arg-type]
    assert callable(getattr(facet, method))
