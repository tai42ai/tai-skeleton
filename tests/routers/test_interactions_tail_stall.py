"""The SSE keepalive XREAD tail (a legitimately-blocking command) runs with the
socket read timeout stripped, but a black-holed Redis is bounded by an outer
``asyncio.wait_for``: a stalled XREAD raises a loud ``RuntimeError`` so the SSE
generator dies loudly and the client reconnects."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import cast

import pytest
from starlette.requests import Request

from tai42_skeleton.interactions.settings import InteractionsSettings
from tai42_skeleton.interactions.store import InteractionStore
from tai42_skeleton.routers import interactions as router


class _StallTailRedis:
    """A redis with an empty backlog whose keepalive XREAD never resolves."""

    async def xrevrange(self, key, count=1):
        return []

    async def zrange(self, key, start, end):
        return []

    async def xread(self, streams, block=None):
        await asyncio.Event().wait()  # pragma: no cover - cancelled by wait_for


class _FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


async def test_sse_tail_stall_raises_runtime_error(monkeypatch):
    stall = _StallTailRedis()

    @asynccontextmanager
    async def _ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield stall

    monkeypatch.setattr(router, "client_ctx", _ctx)
    # Tiny keepalive so the outer wait_for (keepalive + grace) fires promptly.
    monkeypatch.setattr(router, "_KEEPALIVE_SECONDS", 0)
    settings = InteractionsSettings(blocking_grace_seconds=0.05)
    store = InteractionStore(settings.key_prefix)

    gen = router._stream_events(cast(Request, _FakeRequest()), store, settings)
    # Empty backlog -> first frame is the backlog_done marker.
    assert "backlog_done" in await gen.__anext__()
    # Next pull drives into the tail loop; the stalled XREAD trips the outer bound.
    with pytest.raises(RuntimeError, match="XREAD"):
        await gen.__anext__()
