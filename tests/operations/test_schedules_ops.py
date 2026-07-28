"""Op-level oracles for the scheduling operations.

The route oracles (``tests/routers/test_schedules.py``) pin the enveloped surface;
these pin the ops directly, including the paths all four doors share. Every door
dispatches a NAMED tool, so every door discriminates an ``UnknownToolError`` by name:
one naming a DIFFERENT tool than the door asked for is the inner dispatch's own
failure, so it becomes a structured ``OperationFailed`` (500) like any other raise
during execution, never a verdict about the tool the door asked for — including when
the name is the door's SIBLING marker tool, which is still not the tool that door
asked for. One naming the door's OWN tool is that tool's absence, and each door answers
it honestly: ``NotSupportedError`` (501) for ``list_schedules`` / ``delete_schedule``
(their marker tool passed the presence pre-check and vanished before the dispatch) and
for ``server_datetime`` (no pre-check at all — the dispatch is how it learns), and
``NotFoundError`` (404) for ``create_schedule``, whose named tool is the caller's and is
never covered by the marker pre-check. A typed ``OperationError`` from the dispatch seam
passes through untouched — a ``PermissionDenied`` 403, or the retriable
``OperationSurfaceUnsettledError`` (503) the seam's execution authorization raises while
the operation surface is being rebuilt.

These doors are authed but not admin-fenced, so the 500s pin a BOUNDED message: the
door's own sentence plus the failure's exception class (or the inner tool's name),
never the caught exception's text. The compensating control is pinned alongside it —
the detail the caller is denied lands SERVER-side, as one ERROR record from the ops
logger carrying the original exception, so the traceback still names what the bounded
message drops (a backend's host and port). List's and delete's 501 leaves its own server
record — a WARNING naming the door and the marker tool that passed the pre-check and did
not resolve at the dispatch. Server-datetime's 501, create's 404 and every typed
passthrough pin the opposite: no record at all — neither absence is an anomaly, and a
passthrough is the tool's own answer rather than a failure of the door.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from tai42_contract.app import tai42_app

from tai42_skeleton.authz.resolver import OperationSurfaceUnsettledError
from tai42_skeleton.operations import NotFoundError, NotSupportedError, OperationFailed, UnavailableError
from tai42_skeleton.operations import schedules as schedules_ops
from tai42_skeleton.operations.errors import PermissionDenied
from tai42_skeleton.tools.binding import UnknownToolError


class _FakeTools:
    def __init__(self, registered: set[str], *, run_exc: Exception | None = None, run_result: object = None) -> None:
        self._registered = registered
        self._run_exc = run_exc
        self._run_result = run_result

    async def get_tools(self) -> dict:
        return {name: SimpleNamespace(name=name) for name in self._registered}

    async def run_tool(self, key: str, arguments: dict) -> object:
        if self._run_exc is not None:
            raise self._run_exc
        if key not in self._registered:
            raise UnknownToolError(key)
        return self._run_result


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch):
    def _install(fake: _FakeTools) -> _FakeTools:
        monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(tools=fake))
        return fake

    return _install


_MARKERS = {"backend_list_schedules", "backend_delete_schedule"}


def _assert_logged_server_side(caplog: pytest.LogCaptureFixture, detail: str) -> None:
    """Exactly one ERROR record from the schedules ops logger, carrying the caught
    exception, with ``detail`` present in the formatted log. ``detail`` is asserted
    against the FULL server-side record — the caught exception's own text, reaching the
    log through the attached traceback — which covers both what the bounded client
    message withholds (a backend's host and port) and what it does show (the inner tool's
    name on a mismatched unknown-tool error)."""
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == schedules_ops.logger.name]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
    # Formatting the record renders its own message and appends the attached traceback,
    # so ``detail`` is pinned to THAT record rather than to anything else the capture
    # happens to hold.
    assert detail in logging.Formatter().format(errors[0])


def _assert_warned_server_side(caplog: pytest.LogCaptureFixture, door: str, tool_name: str) -> None:
    """Exactly one WARNING record from the schedules ops logger, naming the door and the
    tool that did not resolve at the dispatch."""
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == schedules_ops.logger.name]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert door in message
    assert tool_name in message


def _assert_nothing_logged_server_side(caplog: pytest.LogCaptureFixture) -> None:
    """No record at all from the schedules ops logger, at any level. Silence is the
    contract wherever the outcome is not an anomaly: an unregistered ``current_time_info``
    is steady-state configuration (the toolbox extra is not installed), so a record there
    would repeat on every request; a caller naming a tool that is not registered is an
    ordinary caller error the 404 already answers in full; and a typed ``OperationError``
    from the dispatch seam is the tool's own answer delivered to the caller intact, a
    routine refusal such as a 403 rather than a server-side failure. Pinning the absence
    keeps a later log line from turning any of them into per-request noise."""
    assert [r for r in caplog.records if r.name == schedules_ops.logger.name] == []


async def test_list_501_without_backend(install) -> None:
    install(_FakeTools(set()))
    with pytest.raises(NotSupportedError, match="no installed backend"):
        await schedules_ops.list_schedules()


async def test_list_raise_during_execution_is_500(install, caplog) -> None:
    # The backend is installed and its list tool ran; a raise from that body is a
    # structured OperationFailed (500) naming the door and the exception class, never an
    # opaque 500 and never the caught exception's own text — which lands in the log.
    install(_FakeTools(_MARKERS, run_exc=RuntimeError("boom listing at 10.0.0.7:6379")))
    with caplog.at_level(logging.ERROR, logger=schedules_ops.logger.name), pytest.raises(OperationFailed) as caught:
        await schedules_ops.list_schedules()
    assert caught.value.message == "schedule listing failed (RuntimeError)"
    _assert_logged_server_side(caplog, "10.0.0.7:6379")


async def test_list_vanished_marker_is_501(install, caplog) -> None:
    # The marker tool passed the presence pre-check and did not resolve at the dispatch.
    # It never ran, so the door answers the 501 the next request would answer — never a
    # 500 blaming the tool for "raising during execution". The caller sees a plain 501,
    # so the anomaly is recorded server-side as a WARNING naming the door and the tool.
    install(_FakeTools(_MARKERS, run_exc=UnknownToolError(schedules_ops._LIST_TOOL)))
    with (
        caplog.at_level(logging.WARNING, logger=schedules_ops.logger.name),
        pytest.raises(NotSupportedError, match="no installed backend"),
    ):
        await schedules_ops.list_schedules()
    _assert_warned_server_side(caplog, "list-schedules", schedules_ops._LIST_TOOL)


async def test_list_maps_unknown_tool_raised_for_another_name_to_structured_500(install, caplog) -> None:
    # The marker tool ran and an ``UnknownToolError`` naming a DIFFERENT tool escaped its
    # body — that inner dispatch's failure, so a structured 500, never a 501. The caught
    # error itself stays server-side on an ERROR record.
    install(_FakeTools(_MARKERS, run_exc=UnknownToolError("some_inner_tool")))
    with caplog.at_level(logging.ERROR, logger=schedules_ops.logger.name), pytest.raises(OperationFailed) as caught:
        await schedules_ops.list_schedules()
    assert caught.value.message == "schedule listing failed (unknown tool some_inner_tool)"
    _assert_logged_server_side(caplog, "No such tool: some_inner_tool.")


async def test_list_sibling_marker_unknown_tool_is_structured_500(install) -> None:
    # The DELETE marker is not the tool the LIST door asked for, so an ``UnknownToolError``
    # naming it is the inner dispatch's failure like any other foreign name — a structured
    # 500, never the 501 reserved for this door's own marker vanishing.
    install(_FakeTools(_MARKERS, run_exc=UnknownToolError(schedules_ops._DELETE_TOOL)))
    with pytest.raises(OperationFailed) as caught:
        await schedules_ops.list_schedules()
    assert caught.value.message == "schedule listing failed (unknown tool backend_delete_schedule)"


async def test_list_permission_denied_passes_through(install, caplog) -> None:
    # A ``PermissionDenied`` taken by the tool-dispatch seam is already the caller's
    # answer (403); it passes through typed rather than being flattened into a 500, and
    # records nothing — it is the tool's own answer, not a failure of this door.
    install(_FakeTools(_MARKERS, run_exc=PermissionDenied("access denied: listing refused")))
    with (
        caplog.at_level(logging.DEBUG, logger=schedules_ops.logger.name),
        pytest.raises(PermissionDenied, match="listing refused"),
    ):
        await schedules_ops.list_schedules()
    _assert_nothing_logged_server_side(caplog)


async def test_list_unsettled_operation_surface_is_503(install, caplog) -> None:
    # Mid-rebuild the dispatch seam's execution authorization cannot decide either way, so
    # it refuses with the retriable ``OperationSurfaceUnsettledError`` — an
    # ``UnavailableError`` (503) and a typed ``OperationError``, so it reaches the caller
    # retriable rather than flattened into a 500, and records nothing.
    unsettled = OperationSurfaceUnsettledError("the operation surface is being rebuilt — retry shortly")
    install(_FakeTools(_MARKERS, run_exc=unsettled))
    with (
        caplog.at_level(logging.DEBUG, logger=schedules_ops.logger.name),
        pytest.raises(UnavailableError, match="retry shortly") as caught,
    ):
        await schedules_ops.list_schedules()
    assert caught.value.status == 503
    _assert_nothing_logged_server_side(caplog)


async def test_server_datetime_501_when_tool_absent(install, caplog) -> None:
    # This door has no presence pre-check, so the dispatch is where it learns
    # ``current_time_info`` is not registered — answered as the door's own honest 501. The
    # absence leaves no server record: an uninstalled toolbox extra is steady-state
    # configuration, so a log here would repeat on every request.
    install(_FakeTools(set()))
    with (
        caplog.at_level(logging.DEBUG, logger=schedules_ops.logger.name),
        pytest.raises(NotSupportedError, match="current_time_info tool is not available"),
    ):
        await schedules_ops.server_datetime()
    _assert_nothing_logged_server_side(caplog)


async def test_server_datetime_maps_unknown_tool_raised_for_another_name_to_structured_500(install, caplog) -> None:
    # ``current_time_info`` is registered and ran; the ``UnknownToolError`` that escaped
    # names a different tool, so it is that inner dispatch's failure — a structured 500
    # naming that tool, never "the time tool is not available" (501). The caught error
    # itself stays server-side on an ERROR record.
    install(_FakeTools({schedules_ops._TIME_TOOL}, run_exc=UnknownToolError("not_current_time_info")))
    with caplog.at_level(logging.ERROR, logger=schedules_ops.logger.name), pytest.raises(OperationFailed) as caught:
        await schedules_ops.server_datetime()
    assert caught.value.message == "server-datetime lookup failed (unknown tool not_current_time_info)"
    _assert_logged_server_side(caplog, "No such tool: not_current_time_info.")


async def test_server_datetime_raise_during_execution_is_500(install, caplog) -> None:
    # The time tool is registered and ran; a raise from its body is a structured
    # OperationFailed (500) naming the door and the exception class, never "the time tool
    # is not available" (501) and never the caught exception's own text — which lands in
    # the log.
    install(_FakeTools({schedules_ops._TIME_TOOL}, run_exc=RuntimeError("boom clock at 10.0.0.7:6379")))
    with caplog.at_level(logging.ERROR, logger=schedules_ops.logger.name), pytest.raises(OperationFailed) as caught:
        await schedules_ops.server_datetime()
    assert caught.value.message == "server-datetime lookup failed (RuntimeError)"
    _assert_logged_server_side(caplog, "10.0.0.7:6379")


async def test_server_datetime_permission_denied_passes_through(install, caplog) -> None:
    # A ``PermissionDenied`` taken by the tool-dispatch seam is already the caller's
    # answer (403); it passes through typed rather than being flattened into a 500, and
    # records nothing — it is the tool's own answer, not a failure of this door.
    install(_FakeTools({schedules_ops._TIME_TOOL}, run_exc=PermissionDenied("access denied: clock refused")))
    with (
        caplog.at_level(logging.DEBUG, logger=schedules_ops.logger.name),
        pytest.raises(PermissionDenied, match="clock refused"),
    ):
        await schedules_ops.server_datetime()
    _assert_nothing_logged_server_side(caplog)


async def test_server_datetime_unsettled_operation_surface_is_503(install, caplog) -> None:
    # Mid-rebuild the dispatch seam's execution authorization cannot decide either way, so
    # it refuses with the retriable ``OperationSurfaceUnsettledError`` — an
    # ``UnavailableError`` (503) and a typed ``OperationError``, so it reaches the caller
    # retriable rather than flattened into a 500, and records nothing.
    unsettled = OperationSurfaceUnsettledError("the operation surface is being rebuilt — retry shortly")
    install(_FakeTools({schedules_ops._TIME_TOOL}, run_exc=unsettled))
    with (
        caplog.at_level(logging.DEBUG, logger=schedules_ops.logger.name),
        pytest.raises(UnavailableError, match="retry shortly") as caught,
    ):
        await schedules_ops.server_datetime()
    assert caught.value.status == 503
    _assert_nothing_logged_server_side(caplog)


async def test_create_maps_unknown_tool_raised_for_another_name_to_structured_500(install, caplog) -> None:
    # Same discrimination on the create door: an ``UnknownToolError`` for another name
    # becomes a structured 500 naming that inner tool, not a 404 for the caller-named
    # tool. The caught error itself stays server-side on an ERROR record.
    install(_FakeTools(_MARKERS | {"send"}, run_exc=UnknownToolError("some_inner_tool")))
    with caplog.at_level(logging.ERROR, logger=schedules_ops.logger.name), pytest.raises(OperationFailed) as caught:
        await schedules_ops.create_schedule("send", {}, {})
    assert caught.value.message == "schedule creation failed (unknown tool some_inner_tool)"
    _assert_logged_server_side(caplog, "No such tool: some_inner_tool.")


async def test_create_raise_during_execution_is_500(install, caplog) -> None:
    # The caller-named tool resolved and was scheduled; a raise from that dispatch is a
    # structured OperationFailed (500) naming the door and the exception class, never a
    # 404 for the caller-named tool and never the caught exception's own text — which
    # lands in the log.
    install(_FakeTools(_MARKERS | {"send"}, run_exc=RuntimeError("boom scheduling at 10.0.0.7:6379")))
    with caplog.at_level(logging.ERROR, logger=schedules_ops.logger.name), pytest.raises(OperationFailed) as caught:
        await schedules_ops.create_schedule("send", {}, {})
    assert caught.value.message == "schedule creation failed (RuntimeError)"
    _assert_logged_server_side(caplog, "10.0.0.7:6379")


async def test_create_permission_denied_from_the_dispatch_passes_through(install, caplog) -> None:
    # Distinct from the pre-dispatch authorization denial below: this ``PermissionDenied``
    # comes from the tool-dispatch seam itself and is already the caller's answer (403),
    # so it passes through typed rather than being flattened into a 500, and records
    # nothing — it is the tool's own answer, not a failure of this door.
    install(_FakeTools(_MARKERS | {"send"}, run_exc=PermissionDenied("access denied: dispatch refused")))
    with (
        caplog.at_level(logging.DEBUG, logger=schedules_ops.logger.name),
        pytest.raises(PermissionDenied, match="dispatch refused"),
    ):
        await schedules_ops.create_schedule("send", {}, {})
    _assert_nothing_logged_server_side(caplog)


async def test_create_authorizes_the_submitted_tool_with_merged_arguments(install, monkeypatch) -> None:
    # The submitted tool is authorized against the live caller — on the exact merged
    # arguments the dispatch fires (schedule keys win on collision) — before scheduling.
    install(_FakeTools(_MARKERS | {"send"}, run_result="ok"))
    seen: list[tuple] = []

    async def _spy(tool_name, arguments):
        seen.append((tool_name, dict(arguments)))

    monkeypatch.setattr(schedules_ops, "authorize_submitted_tool", _spy)
    out = await schedules_ops.create_schedule("send", {"to": "x"}, {"cron": "* * * * *"})
    assert out == "ok"
    assert seen == [("send", {"to": "x", "cron": "* * * * *"})]


async def test_create_denied_tool_is_refused_before_scheduling(install, monkeypatch) -> None:
    # A denial from the submitted-tool authorization is the caller's 403, raised before the
    # tool is ever dispatched to the scheduling backend.
    install(_FakeTools(_MARKERS | {"write_env"}, run_exc=RuntimeError("must not be scheduled")))

    async def _deny(tool_name, arguments):
        raise PermissionDenied("access denied: POST /api/config/env is not permitted")

    monkeypatch.setattr(schedules_ops, "authorize_submitted_tool", _deny)
    with pytest.raises(PermissionDenied, match="not permitted"):
        await schedules_ops.create_schedule("write_env", {"k": "v"}, {"cron": "* * * * *"})


async def test_create_unsettled_operation_surface_is_503(install, monkeypatch) -> None:
    # Mid-rebuild the submitted-tool authorization cannot decide either way, so it refuses
    # with the retriable ``OperationSurfaceUnsettledError`` — an ``UnavailableError``
    # (503), the second source of the status this door declares. It fires before the
    # dispatch, so nothing is scheduled, and it reaches the caller retriable rather than
    # flattened into a 500.
    install(_FakeTools(_MARKERS | {"send"}, run_exc=RuntimeError("must not be scheduled")))

    async def _unsettled(tool_name, arguments):
        raise OperationSurfaceUnsettledError("the operation surface is being rebuilt — retry shortly")

    monkeypatch.setattr(schedules_ops, "authorize_submitted_tool", _unsettled)
    with pytest.raises(UnavailableError, match="retry shortly") as caught:
        await schedules_ops.create_schedule("send", {"to": "x"}, {"cron": "* * * * *"})
    assert caught.value.status == 503


async def test_create_unknown_tool_is_404(install, caplog) -> None:
    # The marker pre-check tests the MARKER tools, never the caller-named one, so an
    # unregistered caller-named tool surfaces here as a 404 — and is not logged: naming a
    # tool that is not registered is an ordinary caller error the response answers in full.
    install(_FakeTools(_MARKERS))  # markers present, target tool absent
    with (
        caplog.at_level(logging.DEBUG, logger=schedules_ops.logger.name),
        pytest.raises(NotFoundError, match="unknown tool: typo"),
    ):
        await schedules_ops.create_schedule("typo", {}, {})
    _assert_nothing_logged_server_side(caplog)


async def test_delete_501_without_backend(install) -> None:
    install(_FakeTools(set()))
    with pytest.raises(NotSupportedError, match="no installed backend"):
        await schedules_ops.delete_schedule("nightly")


async def test_delete_raise_during_execution_is_500(install, caplog) -> None:
    # The backend is installed and its delete tool ran; a raise from that body is a
    # structured OperationFailed (500) naming the door and the exception class, never an
    # opaque 500 and never the caught exception's own text — which lands in the log.
    install(_FakeTools(_MARKERS, run_exc=RuntimeError("boom deleting at 10.0.0.7:6379")))
    with caplog.at_level(logging.ERROR, logger=schedules_ops.logger.name), pytest.raises(OperationFailed) as caught:
        await schedules_ops.delete_schedule("nightly")
    assert caught.value.message == "schedule deletion failed (RuntimeError)"
    _assert_logged_server_side(caplog, "10.0.0.7:6379")


async def test_delete_vanished_marker_is_501(install, caplog) -> None:
    # The marker tool passed the presence pre-check and did not resolve at the dispatch.
    # It never ran, so the door answers the 501 the next request would answer — never a
    # 500 blaming the tool for "raising during execution". The caller sees a plain 501,
    # so the anomaly is recorded server-side as a WARNING naming the door and the tool.
    install(_FakeTools(_MARKERS, run_exc=UnknownToolError(schedules_ops._DELETE_TOOL)))
    with (
        caplog.at_level(logging.WARNING, logger=schedules_ops.logger.name),
        pytest.raises(NotSupportedError, match="no installed backend"),
    ):
        await schedules_ops.delete_schedule("nightly")
    _assert_warned_server_side(caplog, "delete-schedule", schedules_ops._DELETE_TOOL)


async def test_delete_maps_unknown_tool_raised_for_another_name_to_structured_500(install, caplog) -> None:
    # The marker tool ran and an ``UnknownToolError`` naming a DIFFERENT tool escaped its
    # body — that inner dispatch's failure, so a structured 500, never a 501. The caught
    # error itself stays server-side on an ERROR record.
    install(_FakeTools(_MARKERS, run_exc=UnknownToolError("some_inner_tool")))
    with caplog.at_level(logging.ERROR, logger=schedules_ops.logger.name), pytest.raises(OperationFailed) as caught:
        await schedules_ops.delete_schedule("nightly")
    assert caught.value.message == "schedule deletion failed (unknown tool some_inner_tool)"
    _assert_logged_server_side(caplog, "No such tool: some_inner_tool.")


async def test_delete_sibling_marker_unknown_tool_is_structured_500(install) -> None:
    # The LIST marker is not the tool the DELETE door asked for, so an ``UnknownToolError``
    # naming it is the inner dispatch's failure like any other foreign name — a structured
    # 500, never the 501 reserved for this door's own marker vanishing.
    install(_FakeTools(_MARKERS, run_exc=UnknownToolError(schedules_ops._LIST_TOOL)))
    with pytest.raises(OperationFailed) as caught:
        await schedules_ops.delete_schedule("nightly")
    assert caught.value.message == "schedule deletion failed (unknown tool backend_list_schedules)"


async def test_delete_permission_denied_passes_through(install, caplog) -> None:
    # A ``PermissionDenied`` taken by the tool-dispatch seam is already the caller's
    # answer (403); it passes through typed rather than being flattened into a 500, and
    # records nothing — it is the tool's own answer, not a failure of this door.
    install(_FakeTools(_MARKERS, run_exc=PermissionDenied("access denied: deletion refused")))
    with (
        caplog.at_level(logging.DEBUG, logger=schedules_ops.logger.name),
        pytest.raises(PermissionDenied, match="deletion refused"),
    ):
        await schedules_ops.delete_schedule("nightly")
    _assert_nothing_logged_server_side(caplog)


async def test_delete_unsettled_operation_surface_is_503(install, caplog) -> None:
    # Mid-rebuild the dispatch seam's execution authorization cannot decide either way, so
    # it refuses with the retriable ``OperationSurfaceUnsettledError`` — an
    # ``UnavailableError`` (503) and a typed ``OperationError``, so it reaches the caller
    # retriable rather than flattened into a 500, and records nothing.
    unsettled = OperationSurfaceUnsettledError("the operation surface is being rebuilt — retry shortly")
    install(_FakeTools(_MARKERS, run_exc=unsettled))
    with (
        caplog.at_level(logging.DEBUG, logger=schedules_ops.logger.name),
        pytest.raises(UnavailableError, match="retry shortly") as caught,
    ):
        await schedules_ops.delete_schedule("nightly")
    assert caught.value.status == 503
    _assert_nothing_logged_server_side(caplog)
