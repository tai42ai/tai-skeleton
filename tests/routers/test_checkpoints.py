"""Checkpoints router: the enveloped sweep result and the admin-only fence.

The handler is driven directly (the router-test pattern); the operation's kit
dependencies (settings + checkpoint registry) are faked. The sweep is a
deployment-wide destructive memory purge, so its route is ``action="fenced"`` —
admin only, denied to every non-admin regardless of granted level.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.requests import Request

from tai42_skeleton.operations import checkpoints as checkpoints_ops
from tai42_skeleton.routers import checkpoints as router


def _req() -> Request:
    return cast(Request, SimpleNamespace(path_params={}, reload_gated=False))


def _json(resp: Any) -> dict:
    return json.loads(bytes(resp.body))


class _FakeSaver:
    def __init__(self, threads: dict[str, list[str]]) -> None:
        self._threads = threads
        self.deleted: list[str] = []

    async def alist(self, config: object) -> AsyncIterator[Any]:
        for thread_id, timestamps in list(self._threads.items()):
            for ts in timestamps:
                yield SimpleNamespace(config={"configurable": {"thread_id": thread_id}}, checkpoint={"ts": ts})

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)
        self._threads.pop(thread_id, None)


async def test_sweep_route_envelopes_result(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)
    saver = _FakeSaver({"bridge:sms:old": [(now - timedelta(hours=2)).isoformat()]})
    monkeypatch.setattr(
        checkpoints_ops,
        "llm_provider_settings",
        lambda: SimpleNamespace(
            checkpoint="postgres", checkpoint_ttl_minutes=60, checkpoint_conn_string="postgresql://u@h/db"
        ),
    )

    async def _get_checkpointer(*, provider: str, conn_string: str) -> object:
        return saver

    monkeypatch.setattr(
        checkpoints_ops, "checkpoint_registry", lambda: SimpleNamespace(get_checkpointer=_get_checkpointer)
    )

    resp = await router.sweep_checkpoints(_req())
    assert resp.status_code == 200
    body = _json(resp)
    assert body["data"]["swept_count"] == 1
    assert body["data"]["swept_threads"] == ["bridge:sms:old"]


def test_sweep_route_is_admin_fenced() -> None:
    from tai42_skeleton.access_control.role_gate import (
        DenialCause,
        grant_map_admits,
        reset_route_index,
        resolve_route_meta,
    )

    reset_route_index()
    meta = resolve_route_meta("/api/checkpoints/sweep", "POST")
    assert meta is not None
    assert meta.action == "fenced"
    # No per-tag level opens a fenced route: even a role granted write on the tag is denied.
    allowed, cause = grant_map_admits(meta, "POST", {"checkpoints": "write"})
    assert allowed is False
    assert cause is DenialCause.HARD_FENCE
