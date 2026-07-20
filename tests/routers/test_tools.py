"""Tool router: name listing, per-tool + bulk schema, and tool execution.

Handlers are driven directly (the router-test pattern); the ``tai_app.tools``
facet is faked by swapping the bound app impl for a stand-in exposing ``tools``.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import cast

import pytest
from starlette.requests import Request
from tai_contract.app import tai_app

from tai_skeleton.routers import tools as router
from tai_skeleton.tools.binding import UnknownToolError


def _req(**path_params) -> Request:
    return cast(Request, SimpleNamespace(path_params=path_params))


def _body_req(body: bytes) -> Request:
    scope = {"type": "http", "method": "POST", "path": "/api/run-tool", "headers": [], "query_string": b""}
    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _json(resp) -> dict:
    return json.loads(bytes(resp.body))


_IMPL = object()


def _tool(name, *, fn=_IMPL, tags=()):
    return SimpleNamespace(
        name=name,
        parameters={"type": "object", "properties": {}},
        output_schema={"type": "string"},
        description=f"{name} desc",
        fn=fn,
        tags=set(tags),
    )


class _FakeTools:
    def __init__(self, tools: dict, run_result=None, run_exc=None):
        self._tools = tools
        self._run_result = run_result
        self._run_exc = run_exc
        self.run_calls: list[tuple] = []

    async def get_tools(self):
        return self._tools

    async def get_tool(self, key):
        tool = self._tools.get(key)
        if tool is None:
            raise UnknownToolError(key)
        return tool

    async def run_tool(self, key, arguments, *, offload_sync=False):
        self.run_calls.append((key, arguments, offload_sync))
        if self._run_exc is not None:
            raise self._run_exc
        return self._run_result


@pytest.fixture
def install(monkeypatch):
    def _install(fake_tools: _FakeTools):
        monkeypatch.setattr(tai_app, "_impl", SimpleNamespace(tools=fake_tools))
        return fake_tools

    return _install


# -- GET /api/tools ----------------------------------------------------------


async def test_list_tools_sorted(install):
    install(_FakeTools({"beta": _tool("beta"), "alpha": _tool("alpha")}))
    resp = await router.list_tools(_req())
    assert resp.status_code == 200
    assert _json(resp) == {"data": ["alpha", "beta"]}


async def test_list_tools_empty(install):
    install(_FakeTools({}))
    resp = await router.list_tools(_req())
    assert _json(resp) == {"data": []}


# -- GET /api/tools/tags -----------------------------------------------------


async def test_tool_tags_map(install):
    install(_FakeTools({"beta": _tool("beta", tags=("z", "a")), "alpha": _tool("alpha")}))
    resp = await router.tool_tags(_req())
    assert resp.status_code == 200
    # One entry per tool, tools sorted by name, each tool's tags sorted.
    assert _json(resp) == {"data": [{"name": "alpha", "tags": []}, {"name": "beta", "tags": ["a", "z"]}]}


async def test_tool_tags_empty(install):
    install(_FakeTools({}))
    resp = await router.tool_tags(_req())
    assert _json(resp) == {"data": []}


# -- GET /api/tools/{tool_name}/schema ---------------------------------------


async def test_tool_schema_happy(install):
    install(_FakeTools({"alpha": _tool("alpha")}))
    resp = await router.tool_schema(_req(tool_name="alpha"))
    assert resp.status_code == 200
    assert _json(resp)["data"] == {
        "input": {"type": "object", "properties": {}},
        "output": {"type": "string"},
        "description": "alpha desc",
    }


async def test_tool_schema_unknown_404(install):
    install(_FakeTools({"alpha": _tool("alpha")}))
    resp = await router.tool_schema(_req(tool_name="nope"))
    assert resp.status_code == 404
    assert "not registered" in _json(resp)["error"]


async def test_tool_schema_serves_fn_less_tool(install):
    # A schema view needs no callable body, so the per-tool route serves the schema
    # of a registered tool whose ``fn`` is ``None`` (parity with the bulk route,
    # which serves-and-logs an fn-less tool) rather than 404-ing it.
    install(_FakeTools({"broken": _tool("broken", fn=None)}))
    resp = await router.tool_schema(_req(tool_name="broken"))
    assert resp.status_code == 200
    assert _json(resp)["data"] == {
        "input": {"type": "object", "properties": {}},
        "output": {"type": "string"},
        "description": "broken desc",
    }


# -- GET /api/tools-schema ---------------------------------------------------


async def test_tools_schema_all(install):
    install(_FakeTools({"alpha": _tool("alpha"), "beta": _tool("beta")}))
    resp = await router.tools_schema(_req())
    assert resp.status_code == 200
    data = _json(resp)["data"]
    assert set(data) == {"alpha", "beta"}
    assert data["alpha"]["description"] == "alpha desc"


async def test_tools_schema_serves_all_and_logs_none_impl(install, caplog):
    # A schema view needs no implementation, so an fn-less tool's schema IS served
    # (matching the per-tool route) — but its missing implementation is still
    # logged once as a registry-health signal, never silently dropped.
    install(_FakeTools({"alpha": _tool("alpha"), "broken": _tool("broken", fn=None)}))
    with caplog.at_level("WARNING"):
        resp = await router.tools_schema(_req())
    data = _json(resp)["data"]
    assert set(data) == {"alpha", "broken"}
    assert "broken" in caplog.text
    assert "no implementation" in caplog.text


async def test_tools_schema_empty(install):
    install(_FakeTools({}))
    resp = await router.tools_schema(_req())
    assert _json(resp) == {"data": {}}


# -- POST /api/run-tool ------------------------------------------------------


async def test_run_tool_happy(install):
    fake = install(_FakeTools({"alpha": _tool("alpha")}, run_result={"ok": 1}))
    resp = await router.run_tool(_body_req(b'{"tool_name": "alpha", "arguments": {"x": 2}}'))
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"ok": 1}}
    # The sync door offloads a sync tool body onto a worker thread.
    assert fake.run_calls == [("alpha", {"x": 2}, True)]


async def test_run_tool_defaults_arguments_to_empty(install):
    fake = install(_FakeTools({"alpha": _tool("alpha")}, run_result=None))
    await router.run_tool(_body_req(b'{"tool_name": "alpha"}'))
    assert fake.run_calls == [("alpha", {}, True)]


async def test_run_tool_missing_tool_name_400(install):
    install(_FakeTools({}))
    resp = await router.run_tool(_body_req(b'{"arguments": {}}'))
    assert resp.status_code == 400
    assert "tool_name" in _json(resp)["error"]


async def test_run_tool_bad_json_400(install):
    install(_FakeTools({}))
    resp = await router.run_tool(_body_req(b"not json"))
    assert resp.status_code == 400
    assert "invalid JSON" in _json(resp)["error"]


async def test_run_tool_unknown_tool_404(install):
    # An unregistered name is a loud 404 (matching the schema route), not an
    # opaque 500.
    install(_FakeTools({}))
    resp = await router.run_tool(_body_req(b'{"tool_name": "nope"}'))
    assert resp.status_code == 404
    assert "unknown tool" in _json(resp)["error"]


async def test_run_tool_unknown_tool_404_legacy_runtime_error(install):
    # Defensive dual-catch: even if the binding raises a plain
    # ``RuntimeError("No such tool: ...")`` instead of the typed error, the door
    # still answers 404, never a masked 500.
    fake = _FakeTools({})

    async def _legacy_get_tool(key):
        raise RuntimeError(f"No such tool: {key}.")

    fake.get_tool = _legacy_get_tool  # type: ignore[method-assign]
    install(fake)
    resp = await router.run_tool(_body_req(b'{"tool_name": "nope"}'))
    assert resp.status_code == 404
    assert "unknown tool" in _json(resp)["error"]


async def test_run_tool_raised_error_is_structured_500(install):
    # A tool that raises DURING execution returns a structured ``{"error": ...}``
    # 500 carrying the caught message, never the opaque "Internal Server Error".
    install(_FakeTools({"alpha": _tool("alpha")}, run_exc=RuntimeError("kaboom")))
    resp = await router.run_tool(_body_req(b'{"tool_name": "alpha"}'))
    assert resp.status_code == 500
    assert _json(resp)["error"] == "kaboom"


async def test_run_tool_sync_body_does_not_stall_the_loop(install):
    # The sync door must offload a blocking sync tool so a concurrently awaited
    # task keeps running (mirrors the tool-runs offload regression test).
    ticks: list[int] = []

    class _BlockingTools(_FakeTools):
        async def run_tool(self, key, arguments, *, offload_sync=False):
            assert offload_sync is True
            # Block a worker thread; if it ran inline the loop would freeze and the
            # ticker below could not advance.
            return await asyncio.to_thread(self._blocking)

        def _blocking(self):
            time.sleep(0.3)
            return {"done": True}

    fake = _BlockingTools({"slow": _tool("slow")})
    install(fake)

    async def ticker():
        for _ in range(30):
            ticks.append(1)
            await asyncio.sleep(0.01)

    tick_task = asyncio.ensure_future(ticker())
    resp = await router.run_tool(_body_req(b'{"tool_name": "slow"}'))
    await tick_task
    assert _json(resp) == {"data": {"done": True}}
    # The loop kept ticking while the sync body blocked its thread.
    assert len(ticks) == 30
