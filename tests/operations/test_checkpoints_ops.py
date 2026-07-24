"""Op-level oracles for the checkpoint retention sweep.

The sweep deletes threads whose newest checkpoint is older than the configured
idle TTL, leaves fresh threads untouched, is a no-op for the TTL-less / non-DB
cases, and surfaces a deletion failure loudly. It also projects as a tool, so it
is schedulable through the existing ``/api/schedules`` create door.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from tai42_contract.app import tai42_app
from tai42_contract.manifest import ApiToolsConfig

from tai42_skeleton.operations import OperationRegistry, operation_metadata_of
from tai42_skeleton.operations import checkpoints as checkpoints_ops
from tai42_skeleton.operations import schedules as schedules_ops
from tai42_skeleton.operations.projection import project_operations


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class _FakeSaver:
    """A saver over an in-memory ``{thread_id: [checkpoint_ts, ...]}`` map."""

    def __init__(self, threads: dict[str, list[str]]) -> None:
        self._threads = threads
        self.deleted: list[str] = []

    async def alist(self, config: object) -> AsyncIterator[Any]:
        for thread_id, timestamps in list(self._threads.items()):
            for ts in timestamps:
                yield SimpleNamespace(
                    config={"configurable": {"thread_id": thread_id}},
                    checkpoint={"ts": ts},
                )

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)
        self._threads.pop(thread_id, None)


def _install(monkeypatch: pytest.MonkeyPatch, *, provider: str, ttl_minutes: int | None, saver: object) -> None:
    monkeypatch.setattr(
        checkpoints_ops,
        "llm_provider_settings",
        lambda: SimpleNamespace(
            checkpoint=provider,
            checkpoint_ttl_minutes=ttl_minutes,
            checkpoint_conn_string="postgresql://u@h/db",
        ),
    )

    async def _get_checkpointer(*, provider: str, conn_string: str) -> object:
        return saver

    monkeypatch.setattr(
        checkpoints_ops, "checkpoint_registry", lambda: SimpleNamespace(get_checkpointer=_get_checkpointer)
    )


async def test_sweeps_stale_and_keeps_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)
    saver = _FakeSaver(
        {
            "bridge:sms:old": [_iso(now - timedelta(hours=3)), _iso(now - timedelta(hours=2))],
            "bridge:sms:fresh": [_iso(now - timedelta(minutes=5))],
        }
    )
    _install(monkeypatch, provider="postgres", ttl_minutes=60, saver=saver)

    result = await checkpoints_ops.sweep_checkpoints()

    assert result["swept_count"] == 1
    assert result["swept_threads"] == ["bridge:sms:old"]
    assert saver.deleted == ["bridge:sms:old"]
    assert result["provider"] == "postgres"
    assert result["ttl_minutes"] == 60


async def test_swept_thread_gone_on_next_list(monkeypatch: pytest.MonkeyPatch) -> None:
    # After the sweep the deleted thread is gone from the store, so the client's next
    # message on that thread_id starts a fresh conversation.
    now = datetime.now(UTC)
    saver = _FakeSaver({"bridge:sms:old": [_iso(now - timedelta(hours=2))]})
    _install(monkeypatch, provider="postgres", ttl_minutes=60, saver=saver)

    await checkpoints_ops.sweep_checkpoints()

    remaining = [tup.config["configurable"]["thread_id"] async for tup in saver.alist(None)]
    assert remaining == []


async def test_deletion_failure_propagates_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)

    class _BoomSaver(_FakeSaver):
        async def adelete_thread(self, thread_id: str) -> None:
            raise RuntimeError("delete failed")

    saver = _BoomSaver({"bridge:sms:old": [_iso(now - timedelta(hours=2))]})
    _install(monkeypatch, provider="postgres", ttl_minutes=60, saver=saver)

    with pytest.raises(RuntimeError, match="delete failed"):
        await checkpoints_ops.sweep_checkpoints()


async def test_noop_when_ttl_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    saver = _FakeSaver({"bridge:sms:old": ["2000-01-01T00:00:00+00:00"]})
    _install(monkeypatch, provider="postgres", ttl_minutes=None, saver=saver)

    result = await checkpoints_ops.sweep_checkpoints()

    assert result["swept_count"] == 0
    assert saver.deleted == []
    assert "retention disabled" in result["skipped"]


@pytest.mark.parametrize("provider", ["redis", "memory"])
async def test_noop_for_non_db_provider(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    # A saver whose alist would raise proves the sweep never touches the store for
    # these providers.
    class _Unusable:
        async def alist(self, config: object) -> AsyncIterator[Any]:
            raise AssertionError("must not walk the store for a non-DB provider")
            yield

    _install(monkeypatch, provider=provider, ttl_minutes=60, saver=_Unusable())

    result = await checkpoints_ops.sweep_checkpoints()

    assert result["swept_count"] == 0
    assert result["provider"] == provider
    assert "no swept store" in result["skipped"]


def test_projects_as_a_schedulable_tool() -> None:
    reg = OperationRegistry()
    reg.register(operation_metadata_of(checkpoints_ops.sweep_checkpoints))

    class _Rec:
        def __init__(self) -> None:
            self.registered: dict[str, dict] = {}

        def tool(self, *, force: bool, name: str, tags: set, annotations: object) -> Any:
            self.registered[name] = {"annotations": annotations}
            return lambda fn: fn

    app = SimpleNamespace(tools=_Rec())
    names = project_operations(app, ApiToolsConfig(expose_destructive=True), registry=reg)
    assert "sweep_checkpoints" in names  # projected → dispatchable by name
    assert app.tools.registered["sweep_checkpoints"]["annotations"].destructiveHint is True


async def test_schedulable_via_create_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    # A scheduler-capable backend (marker tools present) plus the projected sweep tool:
    # the create-schedule door dispatches it by name, so the sweep is schedulable.
    class _FakeTools:
        def __init__(self, registered: set[str]) -> None:
            self._registered = registered

        async def get_tools(self) -> dict:
            return {name: SimpleNamespace(name=name) for name in self._registered}

        async def run_tool(self, key: str, arguments: dict) -> object:
            if key not in self._registered:
                raise RuntimeError(f"No such tool: {key}.")
            return {"scheduled": key, "arguments": arguments}

    fake = _FakeTools({"backend_list_schedules", "backend_delete_schedule", "sweep_checkpoints"})
    monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(tools=fake))

    # The submitted-tool authorization runs the full HTTP-edge decision against the live
    # caller; there is no caller identity in this op-level unit, so stub it to an allow —
    # the dispatch-by-name path is what this test pins, not the authz seam (covered by
    # ``tests/operations/test_schedules_ops.py``).
    async def _allow(tool_name: str, arguments: dict) -> None:
        return None

    monkeypatch.setattr(schedules_ops, "authorize_submitted_tool", _allow)

    result = await schedules_ops.create_schedule("sweep_checkpoints", {}, {"cron": "0 3 * * *"})
    assert result == {"scheduled": "sweep_checkpoints", "arguments": {"cron": "0 3 * * *"}}
