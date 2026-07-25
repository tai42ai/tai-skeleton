"""Shared cross-suite helpers for the interactions and channel tests."""

from __future__ import annotations

import asyncio

from tai42_contract.channels import ChannelNotification


class DeliverOnlyChannel:
    """Base for delivery-focused channel fakes: satisfies the full ``Channel``
    protocol by declaring it cannot notify — ``notify`` raises
    ``NotImplementedError``, exactly as the contract prescribes for a channel
    without a notify capability. Subclasses implement ``deliver``."""

    async def notify(self, notification: ChannelNotification) -> list[str]:
        raise NotImplementedError


async def await_add_event(fake_redis, store, timeout: float = 2.0) -> tuple[str, str]:
    """Poll the events stream until an ``interaction.add`` event appears and
    return its ``(interaction_id, group_id)``; fail the test on timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        for _entry_id, fields in await fake_redis.xrange(store.events_key):
            if fields.get("type") == "interaction.add":
                return fields["interaction_id"], fields["group_id"]
        await asyncio.sleep(0.01)
    raise AssertionError("no interaction.add event was written")
