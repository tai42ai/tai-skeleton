"""The awaitable reload gate.

Proves the reload lock's three surfaces: a gated route answers the retriable 503
while a reload holds the gate; the serving loop is NOT frozen during a reload
(health + a read-only GET answer concurrently while the reload sleeps on its
worker thread); the FastMCP session middleware rejects a tool call the same way;
after release everything answers again; and the reload-through-gate path runs its
reload on a worker thread and returns the reload result under a ``{"data": ...}``
envelope.

Handlers are driven directly (the router-test pattern); the gate is the
process-wide :data:`reload_gate` singleton the routers import.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastmcp.exceptions import ToolError
from starlette.requests import Request
from tai42_contract.app import tai42_app

from tai42_skeleton.app import instance
from tai42_skeleton.app.reload_gate import REJECT_MESSAGE, ReloadGate, reload_gate
from tai42_skeleton.app.server import TaiMCP
from tai42_skeleton.app.sessions import ReloadRejectionMiddleware
from tai42_skeleton.routers import agents as agents_router
from tai42_skeleton.routers import backup as backup_router
from tai42_skeleton.routers import config as config_router
from tai42_skeleton.routers import health as health_router
from tai42_skeleton.routers import manifest as manifest_router
from tai42_skeleton.routers import presets as presets_router
from tai42_skeleton.routers import schedules as schedules_router
from tai42_skeleton.routers import sub_mcp as sub_mcp_router
from tai42_skeleton.routers import tool_runs as tool_runs_router
from tai42_skeleton.routers import tools as tools_router
from tests._fakes.bus import FakeBus


def _dummy_req(**path_params: Any) -> Request:
    """A stand-in request for a handler that rejects before touching the request
    (the gate check short-circuits ahead of any body/path-param read)."""
    return cast(Request, SimpleNamespace(path_params=path_params))


def _json(resp: Any) -> dict[str, Any]:
    return json.loads(bytes(resp.body))


def _body_req(body: bytes, path: str) -> Request:
    scope = {"type": "http", "method": "POST", "path": path, "headers": [], "query_string": b""}
    delivered = {"done": False}

    async def receive() -> dict[str, Any]:
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


class _FakeTools:
    def __init__(self, tools: dict[str, Any]) -> None:
        self._tools = tools

    async def get_tools(self) -> dict[str, Any]:
        return self._tools


# -- reject_response shape ---------------------------------------------------


async def test_reject_response_is_retriable_503() -> None:
    resp = reload_gate.reject_response()
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "5"
    assert _json(resp) == {"error": REJECT_MESSAGE, "reloading": True}


# -- run() offloads to a worker thread ---------------------------------------


async def test_run_offloads_to_worker_thread_and_returns_result() -> None:
    """The heavy body runs on a worker thread (not the serving loop's thread) and
    ``run`` returns its result; the lock is held during and released after."""
    gate = ReloadGate()
    loop_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def body() -> dict[str, str]:
        seen["thread"] = threading.get_ident()
        return {"status": "ok"}

    assert not gate.locked
    result = await gate.run(body)
    assert result == {"status": "ok"}
    assert seen["thread"] != loop_thread
    assert not gate.locked


# -- gated routes reject while the gate is held ------------------------------


async def test_gated_routes_reject_while_reloading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every run / live-registry-mutation / reload-trigger route answers the
    retriable 503 while a reload holds the gate, then answers normally after."""

    async def _assert_rejected(resp: Any) -> None:
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == "5"
        assert _json(resp) == {"error": REJECT_MESSAGE, "reloading": True}

    async with reload_gate.lock:
        assert reload_gate.locked
        # A run surface.
        await _assert_rejected(await tools_router.run_tool(_body_req(b"{}", "/api/run-tool")))
        await _assert_rejected(await tool_runs_router.submit_run(_body_req(b"{}", "/api/tool-runs")))
        await _assert_rejected(await agents_router.run_agent(_dummy_req(name="a")))
        await _assert_rejected(await agents_router.run_authored_agent(_dummy_req(name="a")))
        # A live-registry mutation.
        await _assert_rejected(await sub_mcp_router.register_sub_mcp(_body_req(b"{}", "/api/sub-mcp")))
        await _assert_rejected(await sub_mcp_router.unregister_sub_mcp(_dummy_req(slug="s")))
        await _assert_rejected(await presets_router.create_preset(_body_req(b"{}", "/api/presets")))
        await _assert_rejected(await backup_router.import_backup(_body_req(b"{}", "/api/backup/import")))
        # The schedule create/delete doors dispatch a tool run through the same
        # run-tool seam, so they are gated too (the read-only list/datetime are not).
        await _assert_rejected(await schedules_router.create_schedule(_body_req(b"{}", "/api/schedules")))
        await _assert_rejected(await schedules_router.delete_schedule(_dummy_req(schedule_name="s")))
        # A reload-trigger route — a second reload gets the same 503, never a queue.
        await _assert_rejected(await config_router.write_env(_body_req(b"{}", "/api/config/env")))
        await _assert_rejected(await manifest_router.set_mcp_config(_body_req(b"{}", "/api/mcp-config")))
        await _assert_rejected(await manifest_router.reload_mcp(_dummy_req(title="gh")))

    # Released: a run surface is past the gate again — the request reaches normal
    # handling (here a 400 for the missing tool name), not the 503 rejection.
    assert not reload_gate.locked
    monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(tools=_FakeTools({})))
    resp = await tools_router.run_tool(_body_req(b"{}", "/api/run-tool"))
    assert resp.status_code == 400


# -- the serving loop is NOT frozen during a reload --------------------------


async def test_serving_loop_not_frozen_during_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    """While a reload holds the gate and sleeps on its worker thread, /health and a
    read-only GET both answer 200 on the still-running serving loop."""
    monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(tools=_FakeTools({"a": object(), "b": object()})))
    loop_thread = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    worker: dict[str, int] = {}

    def blocking_reload() -> dict[str, Any]:
        worker["thread"] = threading.get_ident()
        entered.set()
        # Sleep on the worker thread until the test lets go, leaving the serving
        # loop free the whole time.
        assert release.wait(timeout=5)
        return {"status": "ok", "env_keys": 3}

    task = asyncio.create_task(reload_gate.run(blocking_reload))
    try:
        for _ in range(500):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        assert reload_gate.locked
        # The serving loop keeps answering while the reload runs on its thread.
        health_resp = await health_router.health_check(cast(Request, object()))
        assert health_resp.status_code == 200
        assert bytes(health_resp.body) == b"OK"
        listing = await tools_router.list_tools(_dummy_req())
        assert listing.status_code == 200
        assert _json(listing) == {"data": ["a", "b"]}
    finally:
        release.set()
        result = await task

    assert result == {"status": "ok", "env_keys": 3}
    assert worker["thread"] != loop_thread  # the reload really ran off the serving thread
    assert not reload_gate.locked


# -- FastMCP session middleware ----------------------------------------------


async def test_middleware_rejects_session_tool_call_while_reloading() -> None:
    """A session ``tools/call`` raises the retriable reloading error while the gate
    is held, and passes through once released — never dispatching while held."""
    middleware = ReloadRejectionMiddleware(reload_gate)
    calls: list[str] = []

    async def call_next(context: Any) -> str:
        calls.append("dispatched")
        return "tool-result"

    async with reload_gate.lock:
        with pytest.raises(ToolError) as excinfo:
            await middleware.on_call_tool(cast(Any, object()), call_next)
        assert str(excinfo.value) == REJECT_MESSAGE
    assert calls == []  # the held call never reached dispatch

    # Released: the call dispatches normally.
    result = await middleware.on_call_tool(cast(Any, object()), call_next)
    assert result == "tool-result"
    assert calls == ["dispatched"]


def test_reload_rejection_middleware_registered_on_server() -> None:
    """A constructed server wires ``ReloadRejectionMiddleware`` onto its FastMCP
    middleware stack — dropping the ``add_middleware`` call would fail here, not
    just the isolated on_call_tool behaviour above."""
    server = TaiMCP(name="reload-gate-under-test", version="1.0")
    assert any(isinstance(mw, ReloadRejectionMiddleware) for mw in server.fastmcp.middleware)


# -- real reload-through-gate result shape -----------------------------------


class _FakeConfigManager:
    def __init__(self) -> None:
        self._env: dict[str, str] = {}
        self.written: list[dict[str, str]] = []

    def read_env(self) -> dict[str, str]:
        return self._env

    def read_manifest_preserved(self) -> dict[str, str]:
        # No backend registered, so the env-change invariant has nothing to reject.
        return {}

    def write_env(self, config: dict[str, str]) -> None:
        self.written.append(config)
        self._env = {**self._env, **config}


class _FakeAdmin:
    def __init__(self, manager: _FakeConfigManager) -> None:
        self._manager = manager
        self.reloads = 0
        self.reload_thread: int | None = None

    def reload_config(self) -> dict[str, Any]:
        self.reloads += 1
        self.reload_thread = threading.get_ident()
        return {"status": "ok", "env_keys": len(self._manager._env)}


async def test_write_env_reloads_through_gate_same_result_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """``POST /api/config/env`` runs its reload through the gate on a worker thread
    and returns the reload result under a ``{"data": <reload result>}`` envelope."""
    manager = _FakeConfigManager()
    admin = _FakeAdmin(manager)
    monkeypatch.setattr(
        tai42_app,
        "_impl",
        SimpleNamespace(
            config=SimpleNamespace(config_manager=manager),
            admin=admin,
            backends=SimpleNamespace(backend=None),
        ),
    )
    monkeypatch.setattr(instance.app, "_bus", FakeBus(origin="serve-x"))

    assert not reload_gate.locked
    resp = await config_router.write_env(_body_req(b'{"NEW": "val"}', "/api/config/env"))

    assert resp.status_code == 200
    assert _json(resp) == {
        "data": {
            "status": "ok",
            "env_keys": 1,
            "fanout": {"mode": "local-only", "note": "no worker bus configured; only this worker reloaded"},
        }
    }
    assert manager.written == [{"NEW": "val"}]
    assert admin.reloads == 1
    # The reload ran on a worker thread, not the serving loop's thread.
    assert admin.reload_thread is not None
    assert admin.reload_thread != threading.get_ident()
    assert not reload_gate.locked  # released after the route returns
