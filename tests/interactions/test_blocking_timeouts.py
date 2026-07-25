"""The reply BLPOP (a legitimately-blocking command) runs with the socket read
timeout stripped, but a black-holed Redis is still bounded by an outer
``asyncio.wait_for``: a stall raises a loud, DISTINCT ``RuntimeError`` rather than
hanging the loop task forever. The normal no-answer path (BLPOP
nil -> ``None``) is unchanged. (The SSE keepalive XREAD tail's equivalent bound
is covered under ``tests/routers`` where the app handle is bound.)"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from redis.asyncio import Redis

from tai42_skeleton.interactions import settings as settings_module
from tai42_skeleton.interactions.store import InteractionStore


class _StallBlpopRedis:
    """A redis whose BLPOP never resolves — a black-holed connection."""

    async def blpop(self, keys, timeout=0):
        await asyncio.Event().wait()  # pragma: no cover - cancelled by wait_for


async def test_wait_for_reply_stall_raises_distinct_runtime_error():
    store = InteractionStore("t:")
    stall = cast(Redis, _StallBlpopRedis())
    with pytest.raises(RuntimeError, match="BLPOP"):
        await store.wait_for_reply(stall, store.reply_key("i1"), timeout_seconds=0.02, grace_seconds=0.02)


async def test_wait_for_reply_normal_nil_still_returns_none(fake_redis):
    # No answer pushed: BLPOP returns nil within budget -> None, WELL before the
    # outer budget+grace fires. This must stay distinct from the stall RuntimeError.
    store = InteractionStore("t:")
    result = await store.wait_for_reply(fake_redis, store.reply_key("i1"), timeout_seconds=0.02, grace_seconds=5)
    assert result is None


def test_settings_module_still_exposes_grace_field():
    # Guard: the field lives on the feature settings, read by the router tail.
    assert "blocking_grace_seconds" in settings_module.InteractionsSettings.model_fields
