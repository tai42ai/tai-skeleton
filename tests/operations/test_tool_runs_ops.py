"""Op-level oracles for the background tool-run operations.

These pin ``submit_run`` / ``get_run`` / ``list_tool_runs`` behavior DIRECTLY
through the operation functions (flat params, typed raises) — independent of the
route adapter that the router tests drive — and pin the declared metadata
(destructive, the tier-1 meta-executor block, reload gate, error classes). Redis
is the focused in-memory fake wired at the operation module's ``client_ctx`` seam.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from tai_contract.app import tai_app

from tai_skeleton.operations import BadRequestError, ForbiddenError, NotFoundError, UnavailableError
from tai_skeleton.operations import tool_runs as ops
from tai_skeleton.operations.decorator import operation_metadata_of
from tai_skeleton.operations.tool_runs import ToolRunStore
from tai_skeleton.routers.tool_runs_settings import ToolRunsSettings
from tests._fakes.tool_runs_redis import FakeRedis


class _FakeTools:
    def __init__(self, registered: set[str] | None = None) -> None:
        self.result: object = None
        self.calls: list[tuple] = []
        self._registered = registered if registered is not None else {"alpha"}

    async def get_tools(self):
        return {name: SimpleNamespace(name=name) for name in self._registered}

    async def run_tool(self, key, arguments, *, offload_sync=False):
        self.calls.append((key, arguments, offload_sync))
        return self.result


@pytest.fixture
def wired(monkeypatch):
    fake = FakeRedis()
    settings = ToolRunsSettings()

    @asynccontextmanager
    async def ctx(client_cls, s=None, *, fresh=False, **kwargs):
        yield fake

    monkeypatch.setattr(ops, "client_ctx", ctx)
    monkeypatch.setattr(ops, "tool_runs_settings", lambda: settings)
    monkeypatch.setattr(ops, "_now", lambda: datetime(2026, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(ops, "_ACTIVE_RUNS", 0)

    def install(registered: set[str] | None = None) -> _FakeTools:
        tools = _FakeTools(registered)
        monkeypatch.setattr(tai_app, "_impl", SimpleNamespace(tools=tools))
        return tools

    yield SimpleNamespace(
        fake=fake, settings=settings, install=install, monkeypatch=monkeypatch, store=ToolRunStore(settings.key_prefix)
    )

    for task in list(ops._SUPERVISORS):
        task.cancel()


async def _drain() -> None:
    tasks = list(ops._SUPERVISORS)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2.0)


async def test_submit_returns_run_id_and_runs_through_the_offload_seam(wired):
    tools = wired.install()
    tools.result = {"ok": 1}
    out = await ops.submit_run("alpha", {"x": 2})
    assert isinstance(out["run_id"], str)
    await _drain()
    # Background path runs through the same seam with the sync offload gate on.
    assert tools.calls == [("alpha", {"x": 2}, True)]
    record = await wired.store.get_run(wired.fake, out["run_id"])
    assert record["status"] == "succeeded"


async def test_submit_unknown_tool_raises_not_found_before_any_record(wired):
    wired.install(registered={"alpha"})
    with pytest.raises(NotFoundError, match="unknown tool: nope"):
        await ops.submit_run("nope", {})
    assert list(ops._SUPERVISORS) == []


async def test_submit_at_capacity_raises_unavailable(wired):
    wired.install(registered={"slow"})
    wired.monkeypatch.setattr(ops, "tool_runs_settings", lambda: ToolRunsSettings(max_concurrent_runs=1))
    wired.monkeypatch.setattr(ops, "_ACTIVE_RUNS", 1)  # the only slot is taken
    with pytest.raises(UnavailableError, match="tool-run capacity reached"):
        await ops.submit_run("slow", {})


async def test_get_run_unknown_raises_not_found(wired):
    with pytest.raises(NotFoundError, match="not found"):
        await ops.get_run("does-not-exist")


async def test_get_run_returns_running_view(wired):
    await wired.store.create_run(wired.fake, "r1", "alpha", "2026-01-01T00:00:00", 1.0, wired.settings)
    view = await ops.get_run("r1")
    assert view == {"run_id": "r1", "tool_name": "alpha", "status": "running", "started_at": "2026-01-01T00:00:00"}


async def test_list_tool_runs_empty_for_unknown_tool(wired):
    assert await ops.list_tool_runs("alpha") == []


async def test_list_tool_runs_returns_present_records(wired):
    await wired.store.create_run(wired.fake, "r1", "alpha", "2026-01-01T00:00:00", 1.0, wired.settings)
    entries = await ops.list_tool_runs("alpha")
    assert [e["run_id"] for e in entries] == ["r1"]
    assert "result" not in entries[0]
    assert "error" not in entries[0]


def test_metadata_declares_the_tier1_destructive_submit_and_read_ops():
    submit = operation_metadata_of(ops.submit_run)
    assert submit.destructive is True
    assert submit.meta_executor is True  # a "run any tool by name" door — never MCP-projected
    assert submit.reload_gated is True
    assert set(submit.error_classes) == {BadRequestError, NotFoundError, UnavailableError}

    get = operation_metadata_of(ops.get_run)
    assert get.destructive is False
    assert get.meta_executor is False
    assert set(get.error_classes) == {ForbiddenError, NotFoundError}

    listing = operation_metadata_of(ops.list_tool_runs)
    assert listing.destructive is False
    assert listing.meta_executor is False
    assert set(listing.error_classes) == {BadRequestError}
