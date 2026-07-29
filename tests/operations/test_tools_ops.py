"""Op-level oracles for the tool-surface operations.

Covers ``run_tool`` / ``reload_tool`` / ``remove_tool``. ``run_tool`` takes a
``tool_name`` argument (the route's request-model shape) and shares the
``/api/run-tool`` route's typed error surface — an unknown tool is a
:class:`NotFoundError` (404), a typed error from the dispatch seam (a
:class:`PermissionDenied` 403) passes through untouched, and any other raise DURING
execution is an :class:`OperationFailed` (500). ``reload_tool`` / ``remove_tool`` apply
locally when this worker is a target, then broadcast on the bus; the response is the per-origin
fleet report. Projection: ``run_tool`` is tier-1 hardcode-blocked; ``reload_tool`` /
``remove_tool`` project with ``destructiveHint``.
"""

from __future__ import annotations

import logging
import re
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
from tai42_skeleton.operations.errors import PermissionDenied
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


def _assert_logged_server_side(caplog: pytest.LogCaptureFixture, detail: str) -> None:
    """Exactly one ERROR record from the tools ops logger, carrying the caught exception,
    with ``detail`` present in the formatted log — the full server-side record, traceback
    included, of a failure the envelope reports as a message alone."""
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == tools_ops.logger.name]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
    # Formatting the record renders its own message and appends the attached traceback,
    # so ``detail`` is pinned to THAT record rather than to anything else the capture
    # happens to hold.
    assert detail in logging.Formatter().format(errors[0])


def _assert_warned_server_side(caplog: pytest.LogCaptureFixture, door: str, tool_name: str) -> None:
    """Exactly one WARNING record from the tools ops logger, naming the door and the tool
    that did not resolve at the dispatch."""
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == tools_ops.logger.name]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert door in message
    assert tool_name in message


def _assert_nothing_logged_server_side(caplog: pytest.LogCaptureFixture) -> None:
    """No record at all from the tools ops logger, at any level. Silence is the contract on
    all three quiet branches: a name that never resolved at all is the caller's own typo,
    answered in full by the 404; the typed-``OperationError`` passthrough is the tool's own
    answer handed to the caller intact — a routine refusal such as a 403, not a server-side
    incident; and a resolve that fails for anything OTHER than an unknown tool leaves the
    door untouched, so the raise surfaces as itself rather than as a door's log line.
    Pinning the absence keeps a later log line from reporting any of them as a failure."""
    assert [r for r in caplog.records if r.name == tools_ops.logger.name] == []


# -- run_tool --------------


async def test_run_tool_delegates_and_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = _Tools({"calc"}, run_result={"answer": 42})
    _install(monkeypatch, tools=tools)

    result = await tools_ops.run_tool("calc", {"a": 1})

    assert result == {"answer": 42}
    # The sync door offloads a sync tool body onto a worker thread.
    assert tools.run_calls == [("calc", {"a": 1}, True)]


async def test_run_tool_unknown_is_404(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    # A name that never resolved at all is the caller's own typo, answered in full by the
    # 404 — so the door leaves no server record.
    _install(monkeypatch, tools=_Tools(set()))
    with (
        caplog.at_level(logging.DEBUG, logger=tools_ops.logger.name),
        pytest.raises(NotFoundError, match="unknown tool: missing"),
    ):
        await tools_ops.run_tool("missing", {})
    _assert_nothing_logged_server_side(caplog)


async def test_run_tool_raise_during_execution_is_500(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    # The op maps a raise DURING execution to a structured OperationFailed (500)
    # carrying the message, and records the failure server-side: one ERROR from this
    # module's logger carrying the caught exception, so the traceback survives whatever
    # the envelope shows.
    tools = _Tools({"calc"}, run_exc=RuntimeError("kaboom at 10.0.0.7:6379"))
    _install(monkeypatch, tools=tools)
    with caplog.at_level(logging.ERROR, logger=tools_ops.logger.name), pytest.raises(OperationFailed) as caught:
        await tools_ops.run_tool("calc", {})
    assert caught.value.message == "kaboom at 10.0.0.7:6379"
    _assert_logged_server_side(caplog, "kaboom at 10.0.0.7:6379")


async def test_run_tool_resolve_unrelated_runtime_error_propagates(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    # A RuntimeError from get_tool that is NOT the unknown-tool error must propagate
    # loudly, never be masked into a 404 — and the door leaves it untouched, neither
    # enveloped nor logged, so a registry that cannot answer surfaces as itself.
    class _Reg:
        async def get_tool(self, key: str) -> object:
            raise RuntimeError("registry backend unreachable")

    monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(tools=_Reg()))
    with (
        caplog.at_level(logging.DEBUG, logger=tools_ops.logger.name),
        pytest.raises(RuntimeError, match="registry backend unreachable"),
    ):
        await tools_ops.run_tool("calc", {})
    _assert_nothing_logged_server_side(caplog)


async def test_run_tool_vanished_after_resolve_is_404(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    # Resolved above, then did not resolve at the dispatch (a concurrent reload would do
    # that) — still 404. The caller sees a plain 404, so the anomaly is recorded
    # server-side as a WARNING naming the door and the tool.
    tools = _Tools({"calc"}, run_exc=UnknownToolError("calc"))
    _install(monkeypatch, tools=tools)
    with (
        caplog.at_level(logging.WARNING, logger=tools_ops.logger.name),
        pytest.raises(NotFoundError, match="unknown tool: calc"),
    ):
        await tools_ops.run_tool("calc", {})
    _assert_warned_server_side(caplog, "run-tool", "calc")


async def test_run_tool_maps_inner_unknown_tool_to_structured_500(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    # The requested tool resolved and ran; an ``UnknownToolError`` naming a DIFFERENT
    # tool escaped its body. That is the inner dispatch's failure, not "the requested
    # tool does not exist", so it takes the execution-failure path — a structured
    # OperationFailed (500) carrying the inner error — never a 404 for ``calc``. The
    # failure is recorded server-side as one ERROR carrying the caught exception.
    tools = _Tools({"calc"}, run_exc=UnknownToolError("some_other_tool"))
    _install(monkeypatch, tools=tools)
    with (
        caplog.at_level(logging.ERROR, logger=tools_ops.logger.name),
        pytest.raises(OperationFailed, match=re.escape("No such tool: some_other_tool.")),
    ):
        await tools_ops.run_tool("calc", {})
    _assert_logged_server_side(caplog, "No such tool: some_other_tool.")


async def test_run_tool_permission_denied_passes_through(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    # A ``PermissionDenied`` taken by the tool-dispatch seam is already the caller's
    # answer (403); it passes through typed rather than being flattened into a 500, and
    # leaves no server record — it is the tool's own answer, not a failure.
    tools = _Tools({"calc"}, run_exc=PermissionDenied("access denied: insufficient scope"))
    _install(monkeypatch, tools=tools)
    with (
        caplog.at_level(logging.DEBUG, logger=tools_ops.logger.name),
        pytest.raises(PermissionDenied, match="insufficient scope"),
    ):
        await tools_ops.run_tool("calc", {})
    _assert_nothing_logged_server_side(caplog)


async def test_run_tool_message_less_raise_names_the_exception_class(monkeypatch: pytest.MonkeyPatch) -> None:
    # A bare raise stringifies to "", which would emit ``{"error": ""}``; the door falls
    # back to the class name so the envelope always carries something.
    _install(monkeypatch, tools=_Tools({"calc"}, run_exc=RuntimeError()))
    with pytest.raises(OperationFailed, match="RuntimeError") as caught:
        await tools_ops.run_tool("calc", {})
    assert caught.value.message == "RuntimeError"


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
