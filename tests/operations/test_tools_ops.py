"""Op-level oracles for the tool-surface operations.

Covers ``run_tool`` / ``reload_tool`` / ``remove_tool``. ``run_tool`` takes a
``tool_name`` argument (the route's request-model shape) and shares the
``/api/run-tool`` route's typed error surface — an unknown tool is a
:class:`NotFoundError` (404) and a tool that raises DURING execution is an
:class:`OperationFailed` (500). ``reload_tool`` / ``remove_tool`` apply locally when
this worker is a target, then broadcast on the bus; the response is the per-origin
fleet report. Projection: ``run_tool`` is tier-1 hardcode-blocked; ``reload_tool`` /
``remove_tool`` project with ``destructiveHint``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from tai42_contract.app import tai42_app
from tai42_contract.manifest import ApiToolsConfig

from tai42_skeleton.app import instance
from tai42_skeleton.app.bus import LocalApplyResult, OpOutcome
from tai42_skeleton.operations import (
    BadRequestError,
    NotFoundError,
    OperationFailed,
    OperationRegistry,
    operation_metadata_of,
)
from tai42_skeleton.operations import tools as tools_ops
from tai42_skeleton.operations.projection import project_operations
from tai42_skeleton.tools.binding import UnknownToolError
from tests._fakes.bus import FakeBus


class _Tools:
    def __init__(self, registered: set[str], *, run_result: object = None, run_exc: Exception | None = None) -> None:
        self._registered = registered
        self._run_result = run_result
        self._run_exc = run_exc
        self.run_calls: list[tuple] = []

    async def get_tool(self, key: str) -> object:
        if key not in self._registered:
            raise UnknownToolError(key)
        return SimpleNamespace(name=key)

    async def run_tool(self, key: str, arguments: dict, *, offload_sync: bool = False) -> object:
        self.run_calls.append((key, arguments, offload_sync))
        if self._run_exc is not None:
            raise self._run_exc
        return self._run_result


class _Admin:
    def __init__(self, reload_result: object = None) -> None:
        self.reload_calls: list[tuple[str, str, str]] = []
        self._reload_result = reload_result

    async def run_tool_reload(self, kind: str, action: str, name: str) -> object:
        self.reload_calls.append((kind, action, name))
        return self._reload_result


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tools: _Tools | None = None,
    admin: _Admin | None = None,
    bus: FakeBus | None = None,
) -> FakeBus:
    impl = SimpleNamespace(tools=tools, admin=admin, backends=SimpleNamespace(backend=None))
    monkeypatch.setattr(tai42_app, "_impl", impl)
    bus = bus or FakeBus()
    monkeypatch.setattr(instance.app, "_bus", bus)
    return bus


# -- run_tool --------------


async def test_run_tool_delegates_and_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = _Tools({"calc"}, run_result={"answer": 42})
    _install(monkeypatch, tools=tools)

    result = await tools_ops.run_tool("calc", {"a": 1})

    assert result == {"answer": 42}
    # The sync door offloads a sync tool body onto a worker thread.
    assert tools.run_calls == [("calc", {"a": 1}, True)]


async def test_run_tool_unknown_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, tools=_Tools(set()))
    with pytest.raises(NotFoundError, match="unknown tool: missing"):
        await tools_ops.run_tool("missing", {})


async def test_run_tool_raise_during_execution_is_500(monkeypatch: pytest.MonkeyPatch) -> None:
    # The op maps a raise DURING execution to a structured OperationFailed (500)
    # carrying the message.
    tools = _Tools({"calc"}, run_exc=RuntimeError("kaboom"))
    _install(monkeypatch, tools=tools)
    with pytest.raises(OperationFailed, match="kaboom"):
        await tools_ops.run_tool("calc", {})


async def test_run_tool_resolve_unrelated_runtime_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    # A RuntimeError from get_tool that is NOT the unknown-tool error must propagate
    # loudly, never be masked into a 404.
    class _Reg:
        async def get_tool(self, key: str) -> object:
            raise RuntimeError("registry backend unreachable")

    monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(tools=_Reg()))
    with pytest.raises(RuntimeError, match="registry backend unreachable"):
        await tools_ops.run_tool("calc", {})


async def test_run_tool_vanished_after_resolve_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resolved above, then vanished before the run (a concurrent reload) — still 404.
    tools = _Tools({"calc"}, run_exc=UnknownToolError("calc"))
    _install(monkeypatch, tools=tools)
    with pytest.raises(NotFoundError, match="unknown tool: calc"):
        await tools_ops.run_tool("calc", {})


# -- reload_tool / remove_tool -----------------------
#
# Runtime registry ops (class a): apply locally when this worker is a target, then
# broadcast; the response is the per-origin fleet report. A local-apply raise aborts
# before anything is broadcast.


async def test_reload_tool_untargeted_applies_locally_and_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(reload_result={"status": "ok"})
    bus = _install(monkeypatch, admin=admin)

    result = await tools_ops.reload_tool("flow", "f1")

    # Local apply ran, then the op broadcast untargeted with the local result as the
    # self-entry payload.
    assert admin.reload_calls == [("flow", "reload", "f1")]
    assert bus.publish_calls == [
        (
            {"op": "reload_tool", "kind": "flow", "name": "f1"},
            None,
            LocalApplyResult(outcome=OpOutcome.applied, payload={"status": "ok"}),
        )
    ]
    assert result["op"] == "reload_tool"
    assert result["results"][0]["outcome"] == "applied"
    assert result["results"][0]["payload"] == {"status": "ok"}


async def test_reload_tool_targeted_to_remote_skips_local_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(reload_result={"status": "ok"})
    bus = _install(monkeypatch, admin=admin, bus=FakeBus(remotes=["serve-w1"]))

    result = await tools_ops.reload_tool("flow", "f1", ["serve-w1"])

    # Targets exclude this worker → no local apply, broadcast to the named worker.
    assert admin.reload_calls == []
    assert bus.validate_calls == [["serve-w1"]]
    assert bus.publish_calls == [({"op": "reload_tool", "kind": "flow", "name": "f1"}, ["serve-w1"], None)]
    assert {r["origin"]: r["outcome"] for r in result["results"]} == {"serve-w1": "applied"}


async def test_reload_tool_unknown_target_raises_before_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(reload_result={"status": "ok"})
    bus = _install(monkeypatch, admin=admin)

    with pytest.raises(BadRequestError, match="unknown fleet targets"):
        await tools_ops.reload_tool("flow", "f1", ["ghost"])
    # Validation precedes side effects: nothing applied, nothing broadcast.
    assert admin.reload_calls == []
    assert bus.publish_calls == []


async def test_remove_tool_untargeted_applies_locally_and_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(reload_result={"status": "removed"})
    bus = _install(monkeypatch, admin=admin)

    result = await tools_ops.remove_tool("flow", "f1")

    assert admin.reload_calls == [("flow", "remove", "f1")]
    assert bus.publish_calls == [
        (
            {"op": "remove_tool", "kind": "flow", "name": "f1"},
            None,
            LocalApplyResult(outcome=OpOutcome.applied, payload={"status": "removed"}),
        )
    ]
    assert result["results"][0]["payload"] == {"status": "removed"}


async def test_reload_tool_local_apply_raise_aborts_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailAdmin(_Admin):
        async def run_tool_reload(self, kind: str, action: str, name: str) -> object:
            raise RuntimeError("reload failed")

    bus = _install(monkeypatch, admin=_FailAdmin())
    with pytest.raises(RuntimeError, match="reload failed"):
        await tools_ops.reload_tool("flow", "f1")
    # Abort-before-publish: a failed local apply broadcasts nothing.
    assert bus.publish_calls == []


# -- reads --------------------------------------------------------------------


async def test_tool_schema_unknown_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Reg:
        async def get_tools(self) -> dict:
            return {}

    monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(tools=_Reg()))
    with pytest.raises(NotFoundError, match="not registered"):
        await tools_ops.tool_schema("ghost")


# -- projection ---------------------------------------------------------------


class _Rec:
    def __init__(self) -> None:
        self.registered: dict[str, Any] = {}

    def tool(self, *, force, name, tags, annotations):
        self.registered[name] = {"annotations": annotations}
        return lambda fn: fn


def test_run_tool_is_tier1_never_projected() -> None:
    reg = OperationRegistry()
    reg.register(operation_metadata_of(tools_ops.run_tool))
    app = SimpleNamespace(tools=_Rec())
    # Even named in include, a tier-1 meta-executor is hardcode-blocked.
    names = project_operations(app, ApiToolsConfig(include=["run_tool"]), registry=reg)
    assert names == []
    assert "run_tool" not in app.tools.registered


def test_reload_and_remove_tool_project_with_destructive_hint() -> None:
    reg = OperationRegistry()
    reg.register(operation_metadata_of(tools_ops.reload_tool))
    reg.register(operation_metadata_of(tools_ops.remove_tool))
    app = SimpleNamespace(tools=_Rec())
    names = project_operations(app, ApiToolsConfig(expose_destructive=True), registry=reg)
    assert set(names) == {"reload_tool", "remove_tool"}
    assert app.tools.registered["reload_tool"]["annotations"].destructiveHint is True
    assert app.tools.registered["remove_tool"]["annotations"].destructiveHint is True
