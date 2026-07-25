"""Op-level oracles for the scheduling operations.

The route oracles (``tests/routers/test_schedules.py``) pin the enveloped surface;
these pin the ops directly, including the defensive branch every door shares — a
RuntimeError that is NOT the run-tool seam's unknown-tool error must propagate
loudly, never be swallowed into a 404/501.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tai42_contract.app import tai42_app

from tai42_skeleton.operations import NotFoundError, NotSupportedError
from tai42_skeleton.operations import schedules as schedules_ops
from tai42_skeleton.operations.errors import PermissionDenied


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
            raise RuntimeError(f"No such tool: {key}.")
        return self._run_result


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch):
    def _install(fake: _FakeTools) -> _FakeTools:
        monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(tools=fake))
        return fake

    return _install


_MARKERS = {"backend_list_schedules", "backend_delete_schedule"}


async def test_list_501_without_backend(install) -> None:
    install(_FakeTools(set()))
    with pytest.raises(NotSupportedError, match="no installed backend"):
        await schedules_ops.list_schedules()


async def test_server_datetime_reraises_unrelated_runtime_error(install) -> None:
    install(_FakeTools({schedules_ops._TIME_TOOL}, run_exc=RuntimeError("boom from the tool body")))
    with pytest.raises(RuntimeError, match="boom from the tool body"):
        await schedules_ops.server_datetime()


async def test_create_reraises_unrelated_runtime_error(install) -> None:
    install(_FakeTools(_MARKERS | {"send"}, run_exc=RuntimeError("boom from the tool body")))
    with pytest.raises(RuntimeError, match="boom from the tool body"):
        await schedules_ops.create_schedule("send", {}, {})


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


async def test_create_unknown_tool_is_404(install) -> None:
    install(_FakeTools(_MARKERS))  # markers present, target tool absent
    with pytest.raises(NotFoundError, match="unknown tool: typo"):
        await schedules_ops.create_schedule("typo", {}, {})


async def test_delete_501_without_backend(install) -> None:
    install(_FakeTools(set()))
    with pytest.raises(NotSupportedError, match="no installed backend"):
        await schedules_ops.delete_schedule("nightly")
