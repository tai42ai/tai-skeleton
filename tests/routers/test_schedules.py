"""Schedules router: route-to-tool mapping plus the no-backend honesty logic.

Handlers are driven directly (the router-test pattern); the ``tai_app.tools``
facet is faked by swapping the bound app impl for a stand-in exposing ``tools``.
The fake's ``get_tools()`` decides backend presence, and ``run_tool`` raises the
binding's real unknown-tool ``RuntimeError`` for any name it does not know.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest
from starlette.requests import Request
from tai_contract.app import tai_app

from tai_skeleton.routers import schedules as router

_MARKERS = {"backend_list_schedules", "backend_delete_schedule"}


def _req(**path_params) -> Request:
    return cast(Request, SimpleNamespace(path_params=path_params))


def _body_req(body: bytes) -> Request:
    scope = {"type": "http", "method": "POST", "path": "/api/schedules", "headers": [], "query_string": b""}
    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _json(resp) -> dict:
    return json.loads(bytes(resp.body))


class _FakeTools:
    """A tool registry where ``registered`` names run and return ``run_result``;
    any other name raises the binding's unknown-tool ``RuntimeError``."""

    def __init__(self, registered: set[str], run_result=None):
        self._registered = registered
        self._run_result = run_result
        self.run_calls: list[tuple] = []

    async def get_tools(self):
        return {name: SimpleNamespace(name=name) for name in self._registered}

    async def run_tool(self, key, arguments):
        if key not in self._registered:
            raise RuntimeError(f"No such tool: {key}.")
        self.run_calls.append((key, arguments))
        return self._run_result


@pytest.fixture
def install(monkeypatch):
    def _install(fake_tools: _FakeTools):
        monkeypatch.setattr(tai_app, "_impl", SimpleNamespace(tools=fake_tools))
        return fake_tools

    return _install


# -- GET /api/schedules ------------------------------------------------------


async def test_list_calls_backend_tool(install):
    fake = install(_FakeTools(_MARKERS, run_result=[{"name": "nightly"}]))
    resp = await router.list_schedules(_req())
    assert resp.status_code == 200
    assert _json(resp) == {"data": [{"name": "nightly"}]}
    assert fake.run_calls == [("backend_list_schedules", {})]


async def test_list_501_when_backend_absent(install):
    install(_FakeTools(set()))
    resp = await router.list_schedules(_req())
    assert resp.status_code == 501
    assert _json(resp) == {"error": "no installed backend exposes scheduling tools"}


# -- DELETE /api/schedules/{name} --------------------------------------------


async def test_delete_calls_backend_tool(install):
    fake = install(_FakeTools(_MARKERS, run_result={"removed": True}))
    resp = await router.delete_schedule(_req(schedule_name="nightly"))
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"removed": True}}
    assert fake.run_calls == [("backend_delete_schedule", {"name": "nightly"})]


async def test_delete_501_when_backend_absent(install):
    install(_FakeTools(set()))
    resp = await router.delete_schedule(_req(schedule_name="nightly"))
    assert resp.status_code == 501
    assert _json(resp) == {"error": "no installed backend exposes scheduling tools"}


# -- POST /api/schedules -----------------------------------------------------


async def test_create_merges_kwargs_schedule_wins(install):
    fake = install(_FakeTools(_MARKERS | {"send_report"}, run_result={"scheduled": True}))
    body = (
        b'{"tool_name": "send_report", "tool_kwargs": {"to": "a", "cron": "tool"},'
        b' "schedule_kwargs": {"cron": "sched"}}'
    )
    resp = await router.create_schedule(_body_req(body))
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"scheduled": True}}
    assert fake.run_calls == [("send_report", {"to": "a", "cron": "sched"})]


async def test_create_defaults_kwargs_to_empty(install):
    fake = install(_FakeTools(_MARKERS | {"send_report"}, run_result=None))
    resp = await router.create_schedule(_body_req(b'{"tool_name": "send_report"}'))
    assert resp.status_code == 200
    assert fake.run_calls == [("send_report", {})]


async def test_create_501_when_backend_absent(install):
    install(_FakeTools({"send_report"}))
    resp = await router.create_schedule(_body_req(b'{"tool_name": "send_report"}'))
    assert resp.status_code == 501
    assert _json(resp) == {"error": "no installed backend exposes scheduling tools"}


async def test_create_404_when_client_tool_unknown(install):
    # Backend present, but the caller named a tool that is not registered. The
    # fake raises the legacy ``RuntimeError("No such tool: ...")`` message, which
    # the dual-catch still recognizes as unknown-tool.
    install(_FakeTools(_MARKERS))
    resp = await router.create_schedule(_body_req(b'{"tool_name": "typo_tool"}'))
    assert resp.status_code == 404
    assert _json(resp) == {"error": "unknown tool: typo_tool"}


async def test_create_404_when_client_tool_unknown_typed_error(install, monkeypatch):
    # The dual-catch also recognizes the TYPED ``UnknownToolError`` the binding
    # raises, not only the legacy message string.
    from tai_skeleton.tools.binding import UnknownToolError

    fake = install(_FakeTools(_MARKERS))

    async def _run_tool(key, arguments):
        raise UnknownToolError(key)

    monkeypatch.setattr(fake, "run_tool", _run_tool)
    resp = await router.create_schedule(_body_req(b'{"tool_name": "typo_tool"}'))
    assert resp.status_code == 404
    assert _json(resp) == {"error": "unknown tool: typo_tool"}


async def test_create_bad_json_400(install):
    install(_FakeTools(_MARKERS))
    resp = await router.create_schedule(_body_req(b"not json"))
    assert resp.status_code == 400
    assert "invalid JSON" in _json(resp)["error"]


async def test_create_non_object_400(install):
    install(_FakeTools(_MARKERS))
    resp = await router.create_schedule(_body_req(b"[1, 2]"))
    assert resp.status_code == 400
    assert "JSON object" in _json(resp)["error"]


async def test_create_missing_tool_name_400(install):
    install(_FakeTools(_MARKERS))
    resp = await router.create_schedule(_body_req(b'{"tool_kwargs": {}}'))
    assert resp.status_code == 400
    assert "tool_name" in _json(resp)["error"]


async def test_create_tool_kwargs_not_object_400(install):
    install(_FakeTools(_MARKERS))
    resp = await router.create_schedule(_body_req(b'{"tool_name": "t", "tool_kwargs": []}'))
    assert resp.status_code == 400
    assert "tool_kwargs" in _json(resp)["error"]


async def test_create_schedule_kwargs_not_object_400(install):
    install(_FakeTools(_MARKERS))
    resp = await router.create_schedule(_body_req(b'{"tool_name": "t", "schedule_kwargs": 5}'))
    assert resp.status_code == 400
    assert "schedule_kwargs" in _json(resp)["error"]


# -- GET /api/schedules/server-datetime --------------------------------------


async def test_server_datetime_happy(install):
    fake = install(_FakeTools({"current_time_info"}, run_result={"iso": "2026-07-05T00:00:00Z"}))
    resp = await router.server_datetime(_req())
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"iso": "2026-07-05T00:00:00Z"}}
    assert fake.run_calls == [("current_time_info", {})]


async def test_server_datetime_501_when_tool_absent(install):
    # Scheduling backend present, but the time tool is not — independent 501.
    install(_FakeTools(_MARKERS))
    resp = await router.server_datetime(_req())
    assert resp.status_code == 501
    assert _json(resp) == {"error": "current_time_info tool is not available"}


async def test_server_datetime_independent_of_backend(install):
    # No scheduling backend at all, but the time tool is present -> still 200.
    fake = install(_FakeTools({"current_time_info"}, run_result="now"))
    resp = await router.server_datetime(_req())
    assert resp.status_code == 200
    assert _json(resp) == {"data": "now"}
    assert fake.run_calls == [("current_time_info", {})]
