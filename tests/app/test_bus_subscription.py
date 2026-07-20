"""The app-owned worker-bus subscription seam in ``lifecycle.py``.

The lifecycle joins ONE long-lived worker-bus subscription per process, built once
and never rejoined: ``app_context`` builds the bus (``serve``/``backend`` origin kind or
the no-op local variant), subscribes with ``_apply_bus_op`` as the callback and
``_resync_on_ready`` as the on-ready self-resync, and cancels it at shutdown.

* ``_apply_bus_op`` dispatches an op to its local admin primitive AND fires the
  ``on_fleet_op_applied`` handlers with the op NAME — the op has not fully applied
  until they finish, so a raising handler fails the op;
* ``_resync_on_ready`` routes the reconnect self-resync through the same apply
  path, so the hook fires for the resync reload too (a celery worker's prefork
  children must not stay stale on the exact path the resync heals);
* the no-op ``WorkerBus.local`` variant is used when no bus is configured, and the
  real process ``app`` proves ``app_context`` builds it, spawns the subscription,
  and cancels it cleanly at shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from tai_contract.app import tai_app

from tai_skeleton.app.bus import OriginKind, WorkerBus
from tai_skeleton.app.instance import app
from tai_skeleton.app.lifecycle import TaiMCPLifecycleMixin
from tai_skeleton.manifest import Manifest

tai_app.bind(app)


class _Mixin(TaiMCPLifecycleMixin):
    """Concrete-enough mixin for the hook logic: no server, no config manager."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preset_manager = cast("Any", SimpleNamespace(reconcile_bases=AsyncMock()))

    def _mcp_tools(self, config, tools):  # pragma: no cover - unused here
        self._mcp_bound_tools[config.title] = {f"{config.title}_t"}


# -- _build_bus: local when unconfigured, real when configured ----------------


def test_build_bus_is_local_without_a_redis_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from tai_kit.settings import reset_all_settings

    monkeypatch.delenv("TAI_BUS_REDIS_URL", raising=False)
    reset_all_settings()
    m = _Mixin()
    bus = m._build_bus(OriginKind.serve)
    assert isinstance(bus, WorkerBus)
    assert bus._local is True
    assert bus.origin.kind is OriginKind.serve


def test_build_bus_is_real_with_a_redis_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from tai_kit.settings import reset_all_settings

    monkeypatch.setenv("TAI_BUS_REDIS_URL", "redis://localhost:6379/0")
    reset_all_settings()
    try:
        m = _Mixin()
        bus = m._build_bus(OriginKind.backend)
        assert bus._local is False
        assert bus.origin.kind is OriginKind.backend
    finally:
        reset_all_settings()


# -- _apply_bus_op fires the on_fleet_op_applied hook with the op name ---------


async def test_apply_bus_op_dispatches_then_fires_hook_with_op_name(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _Mixin()
    sentinel = {"status": "ok", "from": "dispatch"}

    async def fake_dispatch(op: dict[str, Any]) -> Any:
        return sentinel

    monkeypatch.setattr(m, "_dispatch_bus_op", fake_dispatch)

    fired: list[str] = []
    m._on_fleet_op_applied(lambda name: fired.append(name))

    result = await m._apply_bus_op({"op": "reload_mcp", "title": "svc"})

    # The dispatch result becomes the op's terminal payload, and the hook fired
    # AFTER the op applied, with the op name so a handler can filter by op.
    assert result is sentinel
    assert fired == ["reload_mcp"]


async def test_a_raising_hook_fails_the_op(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _Mixin()

    async def fake_dispatch(op: dict[str, Any]) -> Any:
        return {"status": "ok"}

    monkeypatch.setattr(m, "_dispatch_bus_op", fake_dispatch)

    def boom(_name: str) -> None:
        raise RuntimeError("prefork turnover failed")

    m._on_fleet_op_applied(boom)

    # The post-apply hook is part of the op applying, so a raising handler makes
    # the op fail loudly (the subscriber turns it into a terminal ``failed``).
    with pytest.raises(RuntimeError, match="prefork turnover failed"):
        await m._apply_bus_op({"op": "list_failed_mcps"})


async def test_self_resync_routes_reload_config_through_the_apply_path(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _Mixin()
    dispatched: list[str] = []

    async def fake_dispatch(op: dict[str, Any]) -> Any:
        dispatched.append(op["op"])
        return {"status": "ok"}

    monkeypatch.setattr(m, "_dispatch_bus_op", fake_dispatch)

    fired: list[str] = []
    m._on_fleet_op_applied(lambda name: fired.append(name))

    await m._resync_on_ready()

    # The reconnect self-resync is a ``reload_config`` routed through the SAME apply
    # path as a delivered op, so the hook fires for it too.
    assert dispatched == ["reload_config"]
    assert fired == ["reload_config"]


async def test_self_resync_swallows_a_failing_apply_and_stays_live(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    m = _Mixin()
    dispatched: list[str] = []

    async def flaky_dispatch(op: dict[str, Any]) -> Any:
        dispatched.append(op["op"])
        # Only the resync apply (the first call) blows up; a later delivered op
        # applies cleanly, proving the subscription stayed live.
        if len(dispatched) == 1:
            raise RuntimeError("resync reload blew up")
        return {"status": "ok"}

    monkeypatch.setattr(m, "_dispatch_bus_op", flaky_dispatch)

    # A failing resync must NOT propagate — it runs before the message loop, so a
    # propagating error would kill the subscription with no reconnect. Instead it is
    # ERROR-logged and swallowed.
    with caplog.at_level(logging.ERROR):
        await m._resync_on_ready()

    assert any(r.levelno == logging.ERROR and "self-resync" in r.getMessage() for r in caplog.records)
    # The subscription is still live: a subsequently delivered op still applies.
    result = await m._apply_bus_op({"op": "reload_config"})
    assert result == {"status": "ok"}
    assert dispatched == ["reload_config", "reload_config"]


async def test_self_resync_propagates_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _Mixin()

    async def cancel_dispatch(_op: dict[str, Any]) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(m, "_dispatch_bus_op", cancel_dispatch)

    # Cancellation (shutdown) is NOT the swallowed self-resync failure — it must
    # propagate untouched so the subscription task tears down cleanly.
    with pytest.raises(asyncio.CancelledError):
        await m._resync_on_ready()


async def test_a_successful_self_resync_latches_boot_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _Mixin()

    async def ok_dispatch(_op: dict[str, Any]) -> Any:
        return {"status": "ok"}

    monkeypatch.setattr(m, "_dispatch_bus_op", ok_dispatch)

    # Not ready until the first self-resync completes: the tool registry is still
    # being (re)built, so a consumer awaiting readiness must stay blocked.
    assert not m._boot_ready.is_set()

    await m._resync_on_ready()

    # The registry is now rebuilt and stable — boot-ready is latched and
    # ``wait_until_ready`` resolves at once.
    assert m._boot_ready.is_set()
    await asyncio.wait_for(m._wait_until_ready(), timeout=1.0)


async def test_a_failing_self_resync_does_not_latch_boot_ready(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    m = _Mixin()
    calls = 0

    async def flaky_dispatch(_op: dict[str, Any]) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("resync reload blew up")
        return {"status": "ok"}

    monkeypatch.setattr(m, "_dispatch_bus_op", flaky_dispatch)

    # A failed resync leaves the registry possibly half-built — it must NOT latch
    # ready, so the consumer keeps waiting (and fails loudly on its own timeout)
    # rather than forking work against it.
    with caplog.at_level(logging.ERROR):
        await m._resync_on_ready()
    assert not m._boot_ready.is_set()

    # The next resync succeeds and latches ready — the one-way latch is set once and
    # stays set from here.
    await m._resync_on_ready()
    assert m._boot_ready.is_set()


def test_on_fleet_op_applied_is_registered_through_the_lifecycle_facet() -> None:
    fired: list[str] = []

    @app.lifecycle.on_fleet_op_applied
    def _handler(op_name: str) -> None:  # pragma: no cover - registration only
        fired.append(op_name)

    key = f"{_handler.__module__}.{_handler.__qualname__}"
    try:
        assert key in app._fleet_op_applied_handlers
        assert app._fleet_op_applied_handlers[key] is _handler
    finally:
        app._fleet_op_applied_handlers.pop(key, None)


# -- app_context builds the local bus + spawns/cancels the subscription --------


async def test_app_context_builds_local_bus_and_manages_the_subscription(monkeypatch: pytest.MonkeyPatch) -> None:
    from tai_kit.settings import reset_all_settings

    # Single-worker, file-mode, no-backend, no-bus: the supported busless shape runs
    # on WorkerBus.local() — app_context builds it, spawns the subscription, and
    # cancels it cleanly at shutdown.
    monkeypatch.delenv("TAI_BUS_REDIS_URL", raising=False)
    reset_all_settings()

    async with app.app_context(Manifest.model_validate({})):
        assert app.bus._local is True
        task = app._bus_subscription_task
        assert task is not None
        assert not task.done()
        # The local subscribe parks; a foreign op is still applied through the shared
        # callback + hook path.
        await asyncio.sleep(0)

    assert app._bus_subscription_task is None
    assert task.cancelled()
