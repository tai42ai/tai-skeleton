"""Lifecycle coverage for ``TaiMCPLifecycleMixin`` + the ``TaiMCP`` boot path.

Two layers:

* A network-free ``_Mixin`` subclass exercises handler registration/running, the
  tool-reloader dispatch, failed-MCP recording, the reload/deregister result
  shapes, and the blocking-loop helper — all without an event server.
* The real process ``app`` is driven through ``app_context``/``update`` with
  fixture manifests (and a faked MCP probe) to cover ``start`` →
  ``_initialize_components``, the MCP load/probe seam, ``reload_config``, and
  ``live_mcp_status``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from tai42_contract.app import tai42_app
from tai42_contract.manifest import MCPConfig, TaiMCPConfig

from tai42_skeleton.app import kind_status as ks
from tai42_skeleton.app.instance import app
from tai42_skeleton.app.lifecycle import TaiMCPLifecycleMixin
from tai42_skeleton.app.route_defaults import DEFAULT_API_ROUTERS, STUDIO_SPA_ROUTER
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.monitoring.registry import reset_monitoring
from tai42_skeleton.template import ResourceManager

if TYPE_CHECKING:
    from fastmcp import FastMCP


class _FakeMcpTool:
    name = "ping"
    description = "ping"
    inputSchema: ClassVar[dict] = {"type": "object", "properties": {}}
    outputSchema: ClassVar[dict] = {}


class _NoManifestConfig:
    """Embedded/test runtime with no external manifest file: ``read_manifest``
    raises ``FileNotFoundError`` so ``_refresh_manifest_mcp`` keeps its in-memory
    rows."""

    def read_manifest(self):
        raise FileNotFoundError("no external manifest")


class _StubPresetManager:
    """A no-op preset manager: the network-free ``_Mixin`` binds no presets, so the
    post-reload/deregister reconciliation has nothing to do. Records the calls so a
    test can assert the reconciliation ran."""

    def __init__(self) -> None:
        self.reconciled: list[set[str]] = []

    async def reconcile_bases(self, affected_bases: set[str]) -> None:
        self.reconciled.append(set(affected_bases))


class _Mixin(TaiMCPLifecycleMixin):
    """A concrete-enough mixin: ``_mcp_tools`` records bound tool names without a
    real server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config_manager = _NoManifestConfig()  # pyright: ignore[reportAttributeAccessIssue]
        self.preset_manager = cast("Any", _StubPresetManager())

    def _mcp_tools(self, config, tools):
        self._mcp_bound_tools[config.title] = {f"{config.title}_t"}


def _cfg(title="svc"):
    return TaiMCPConfig(title=title, include=[], config=MCPConfig(type="http", url="http://x/mcp"))


# -- handler registration + running ------------------------------------------


def test_handlers_register_and_run_sync_and_async():
    m = _Mixin()
    order: list[str] = []

    @m._on_startup
    def s1():
        order.append("s1")

    @m._on_shutdown
    def d1():
        order.append("d1")

    @m._on_reload
    def r1():
        order.append("r1")

    assert s1 in m._startup_handlers.values()
    assert d1 in m._shutdown_handlers.values()
    assert r1 in m._reload_handlers.values()

    async def a():
        order.append("async")

    asyncio.run(m._run_handlers([s1, a]))
    assert order == ["s1", "async"]


def test_run_handlers_swallows_for_shutdown_but_raises_when_asked(caplog):
    m = _Mixin()

    async def boom():
        raise RuntimeError("handler boom")

    # Default (the shutdown path): recover, but log loudly — never a
    # truly-silent drop.
    with caplog.at_level(logging.ERROR):
        asyncio.run(m._run_handlers([boom]))
    assert "handler boom" in caplog.text
    # Startup/reload paths: surface loudly.
    with pytest.raises(RuntimeError, match=r"lifecycle handlers failed.*boom"):
        asyncio.run(m._run_handlers([boom], raise_on_error=True))


# -- tool reloader dispatch ---------------------------------------------------


def test_tool_reloader_sync_and_default_result():
    m = _Mixin()

    @m._tool_reloader("flow")
    def _reload(action, name):
        return None  # falsy -> default result dict synthesized

    out = asyncio.run(m._run_tool_reload("flow", "reload", "f1"))
    assert out == {"kind": "flow", "action": "reload", "name": "f1", "status": "ok"}


def test_tool_reloader_async_passthrough_result():
    m = _Mixin()

    @m._tool_reloader("flow")
    async def _reload(action, name):
        return {"custom": True}

    assert asyncio.run(m._run_tool_reload("flow", "remove", "f1")) == {"custom": True}


def test_run_tool_reload_unknown_kind_and_bad_action_raise():
    m = _Mixin()
    with pytest.raises(ValueError, match="delete"):
        asyncio.run(m._run_tool_reload("flow", "delete", "x"))
    with pytest.raises(RuntimeError, match="no_such_kind"):
        asyncio.run(m._run_tool_reload("no_such_kind", "reload", "x"))


# -- failed-MCP recording + ignore set ----------------------------------------


def test_record_and_list_failed_mcps_strip_to_title_status():
    m = _Mixin()
    m._record_failed_mcp(_cfg("redis"), "TimeoutError")
    assert m._failed_mcps == {"redis": "unavailable"}
    assert m._list_failed_mcps() == [{"title": "redis", "status": "unavailable"}]


def test_missing_tools_ignore_maps_failed_titles_to_tool_names():
    m = _Mixin()
    m._failed_mcps = {"svc": "unavailable"}

    class _M:
        include_title_mcp_tools_map: ClassVar[dict[str, set[str]]] = {"svc": {"svc_a", "svc_b"}}

    # _missing_tools_ignore only reads include_title_mcp_tools_map; the pydantic
    # Manifest cannot be structurally matched by this minimal stand-in.
    m._manifest = cast("Manifest", _M())
    assert m._missing_tools_ignore() == frozenset({"svc_a", "svc_b"})


# -- reload / deregister ------------------------------------------------------


def test_reload_mcp_unknown_title_is_structured_error():
    m = _Mixin()
    m._manifest = Manifest.model_validate({})
    out = m._reload_mcp("nope")
    assert out["title"] == "nope"
    assert out["status"] == "error"
    assert "Unknown MCP" in out["error"]


def test_reload_mcp_success_binds_and_clears_failed():
    m = _Mixin()
    m._manifest = Manifest.model_validate({"mcp": [_cfg("svc").model_dump()]})
    m._failed_mcps = {"svc": "unavailable"}
    m._probe_mcp = AsyncMock(return_value=[_FakeMcpTool()])
    out = m._reload_mcp("svc")
    assert out["status"] == "ok"
    assert out["tools"] == ["svc_t"]
    assert "svc" not in m._failed_mcps


def test_reload_mcp_probe_failure_records_unavailable():
    m = _Mixin()
    m._manifest = Manifest.model_validate({"mcp": [_cfg("svc").model_dump()]})
    m._probe_mcp = AsyncMock(side_effect=TimeoutError("slow"))
    out = m._reload_mcp("svc")
    assert out == {"title": "svc", "status": "unavailable"}
    assert m._failed_mcps["svc"] == "unavailable"


def test_reload_failed_mcps_keeps_siblings_on_one_failure():
    m = _Mixin()
    m._manifest = Manifest.model_validate({"mcp": [_cfg("a").model_dump(), _cfg("b").model_dump()]})
    m._failed_mcps = {"a": "unavailable", "b": "unavailable"}
    m._probe_mcp = AsyncMock(return_value=[_FakeMcpTool()])

    # The batch probes concurrently but applies binds one server at a time; a
    # failure applying one must not discard the other's result.
    async def fake_apply(title, config, tools):
        if title == "b":
            raise RuntimeError("rebind blew up")
        return {"title": title, "status": "ok"}

    m._apply_reloaded_mcp = fake_apply
    out = m._reload_failed_mcps()
    assert {"title": "a", "status": "ok"} in out
    assert {"title": "b", "status": "error"} in out


def test_deregister_mcp_absent_is_idempotent():
    m = _Mixin()
    assert m._deregister_mcp("never-bound") == {"title": "never-bound", "status": "absent"}


def test_deregister_mcp_bound_removes_and_reports():
    m = _Mixin()
    m._mcp_bound_tools = {"svc": {"svc_t"}}
    out = m._deregister_mcp("svc")
    assert out == {"title": "svc", "status": "ok", "removed": ["svc_t"]}
    assert "svc" not in m._mcp_bound_tools


def test_deregister_reconcile_marshals_onto_the_serving_loop():
    """The deregister reconcile takes the ``PresetManager`` locks on the SERVING
    loop (via ``run_coroutine_threadsafe``), not the throwaway ``_run_blocking``
    worker loop — so it can never contend cross-loop with a serving-loop preset
    route on the same name."""
    m = _Mixin()
    m._mcp_bound_tools = {"svc": {"svc_t"}}
    recorded: list[asyncio.AbstractEventLoop] = []

    async def record_reconcile(affected_bases: set[str]) -> None:
        recorded.append(asyncio.get_running_loop())

    m.preset_manager.reconcile_bases = record_reconcile  # pyright: ignore[reportAttributeAccessIssue]

    # Stand up a serving loop on a background thread and point the mixin at it.
    serving_loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run_serving_loop() -> None:
        asyncio.set_event_loop(serving_loop)
        serving_loop.call_soon(ready.set)
        serving_loop.run_forever()

    thread = threading.Thread(target=_run_serving_loop, daemon=True)
    thread.start()
    ready.wait()
    m._serving_loop = serving_loop
    try:
        # Called from a thread that is NOT the serving loop, as reload_gate.run's
        # worker thread would call it.
        out = m._deregister_mcp("svc")
    finally:
        serving_loop.call_soon_threadsafe(serving_loop.stop)
        thread.join()
        serving_loop.close()

    assert out == {"title": "svc", "status": "ok", "removed": ["svc_t"]}
    # The reconcile ran ON the serving loop, not the _run_blocking worker loop.
    assert recorded == [serving_loop]


def test_reload_closes_llm_registries_on_the_serving_loop():
    """A settings reset refuses to drop a checkpoint/store registry that still holds
    live resources on a running loop (it would leak the pools). An agent run leaves
    such resources on the serving loop, so the reload path closes them there first —
    marshalled onto the serving loop from the reload worker thread — so a reload
    after an agent run does not crash. Without this close, ``reset`` raises."""
    from tai42_kit.llm.checkpoint.checkpoint_registry import checkpoint_registry

    m = _Mixin()

    serving_loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run_serving_loop() -> None:
        asyncio.set_event_loop(serving_loop)
        serving_loop.call_soon(ready.set)
        serving_loop.run_forever()

    thread = threading.Thread(target=_run_serving_loop, daemon=True)
    thread.start()
    ready.wait()
    m._serving_loop = serving_loop

    async def _open() -> bool:
        await checkpoint_registry().get_checkpointer("memory", "e2e")
        return checkpoint_registry().has_live_resources

    async def _has_live() -> bool:
        return checkpoint_registry().has_live_resources

    def _on_serving_loop(coro) -> bool:
        return asyncio.run_coroutine_threadsafe(coro, serving_loop).result()

    try:
        # A memory checkpoint saver opened on the serving loop is a live, loop-bound
        # resource — exactly what an agent run leaves behind.
        assert _on_serving_loop(_open()) is True

        # Called from a thread that is NOT the serving loop, as reload_gate.run's
        # worker thread would call it: it closes the pools on the serving loop.
        m._close_llm_registries_on_serving_loop()

        # The loop-bound registry no longer holds live resources, so the settings
        # reset that follows in ``_reload_config`` can drop it cleanly.
        assert _on_serving_loop(_has_live()) is False
    finally:
        serving_loop.call_soon_threadsafe(serving_loop.stop)
        thread.join()
        serving_loop.close()


def test_reload_config_closes_live_registries_before_resetting_settings(monkeypatch):
    """``_reload_config`` must close the loop-bound checkpoint/store pools on the
    serving loop BEFORE it calls ``reset_all_settings``. The real reset refuses to
    drop a registry that still holds live resources on a running loop (dropping it
    would leak the open pools), so with a live agent-run resource open on the
    serving loop the reset raises unless the close ran first and in that order.

    This guards the call site itself (not just the helper): the real
    ``reset_all_settings`` runs underneath, so removing the close call — or moving
    it after the reset — makes this reload raise the "still hold live resources"
    error and fail the test."""
    from tai42_kit.llm.checkpoint.checkpoint_registry import checkpoint_registry

    class _StubConfig:
        """Minimal config source so ``_reload_config`` refreshes an empty env and
        re-inits from an empty manifest — the reset/close ordering is what's under
        test, not the config read."""

        def read_env(self) -> dict[str, str]:
            return {}

        def read_manifest(self) -> dict:
            return {}

    m = _Mixin()
    m._config_manager = cast("Any", _StubConfig())
    # Isolate the reset-vs-close ordering: the soft re-init (``_update``) is covered
    # by the reload_config reinitialize tests, so stub it to a no-op here.
    monkeypatch.setattr(m, "_update", lambda manifest: None)

    serving_loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run_serving_loop() -> None:
        asyncio.set_event_loop(serving_loop)
        serving_loop.call_soon(ready.set)
        serving_loop.run_forever()

    thread = threading.Thread(target=_run_serving_loop, daemon=True)
    thread.start()
    ready.wait()
    m._serving_loop = serving_loop

    async def _open() -> bool:
        await checkpoint_registry().get_checkpointer("memory", "e2e")
        return checkpoint_registry().has_live_resources

    try:
        # A memory checkpoint saver opened on the serving loop is a live, loop-bound
        # resource — exactly what an agent run leaves behind before a reload.
        assert asyncio.run_coroutine_threadsafe(_open(), serving_loop).result() is True

        # Called from a worker thread that is NOT the serving loop, as reload_gate.run
        # would call it. The real ``reset_all_settings`` runs inside ``_reload_config``
        # and raises unless the live pools were closed on the serving loop first.
        out = m._reload_config()
        assert out == {"status": "ok", "env_keys": 0}
    finally:
        serving_loop.call_soon_threadsafe(serving_loop.stop)
        thread.join()
        serving_loop.close()


def test_deregister_reconcile_falls_back_to_worker_loop_without_serving_loop():
    """With no serving loop bound (a pure-sync boot) nothing contends, so the
    deregister reconcile runs on the ``_run_blocking`` worker loop and still
    completes."""
    m = _Mixin()
    m._mcp_bound_tools = {"svc": {"svc_t"}}
    recorded: list[asyncio.AbstractEventLoop] = []

    async def record_reconcile(affected_bases: set[str]) -> None:
        recorded.append(asyncio.get_running_loop())

    m.preset_manager.reconcile_bases = record_reconcile  # pyright: ignore[reportAttributeAccessIssue]

    assert m._serving_loop is None
    out = m._deregister_mcp("svc")
    assert out == {"title": "svc", "status": "ok", "removed": ["svc_t"]}
    # Ran on the throwaway worker loop — there is no serving loop to marshal onto.
    assert len(recorded) == 1


def test_reconcile_facets_on_serving_loop_raise_instead_of_deadlocking():
    """A reconcile-driving admin facet called from a coroutine ON the serving loop
    would freeze that loop in ``_run_blocking`` while the marshaled reconcile waits
    for it — a deadlock. The guard raises loudly instead."""
    m = _Mixin()
    m._mcp_bound_tools = {"svc": {"svc_t"}}

    async def drive():
        m._serving_loop = asyncio.get_running_loop()
        for call in (
            lambda: m._reload_mcp("svc"),
            lambda: m._reload_failed_mcps(),
            lambda: m._deregister_mcp("svc"),
        ):
            with pytest.raises(RuntimeError, match="must not be called from the serving loop"):
                call()

    asyncio.run(drive())


# -- _run_blocking / _registry_names_sync -------------------------------------


def test_run_blocking_runs_coroutine_to_completion():
    m = _Mixin()

    async def coro():
        return 21 * 2

    assert m._run_blocking(coro) == 42


def test_run_blocking_shuts_down_pooled_clients(monkeypatch):
    # The ephemeral loop must close its pooled clients before teardown, otherwise
    # reload handlers that open pooled clients leak one pool per reload.
    m = _Mixin()
    calls: list[str] = []

    async def fake_shutdown():
        calls.append("shutdown")

    monkeypatch.setattr("tai42_skeleton.app.lifecycle.shutdown_all_clients", fake_shutdown)

    async def coro():
        calls.append("body")
        return "done"

    assert m._run_blocking(coro) == "done"
    assert calls == ["body", "shutdown"]


def test_run_blocking_cleanup_failure_does_not_mask_result(monkeypatch):
    # A failing client shutdown is logged, never raised — the coroutine's result
    # stands.
    m = _Mixin()

    async def boom_shutdown():
        raise RuntimeError("shutdown failed")

    monkeypatch.setattr("tai42_skeleton.app.lifecycle.shutdown_all_clients", boom_shutdown)

    async def coro():
        return 7

    assert m._run_blocking(coro) == 7


def test_registry_names_sync_on_empty_server():
    m = _Mixin()
    assert m._registry_names_sync()["tool"] == set()


def test_registry_names_sync_raises_when_list_tools_fails():
    # A failing list_tools() must propagate through the off-loop runner — never
    # leave the caller blocked.
    m = _Mixin()
    m._fast_mcp = cast(
        "FastMCP",
        MagicMock(
            list_tools=AsyncMock(side_effect=RuntimeError("list_tools boom")),
            list_prompts=AsyncMock(return_value=[]),
            list_resources=AsyncMock(return_value=[]),
        ),
    )
    with pytest.raises(RuntimeError, match="list_tools boom"):
        m._registry_names_sync()


# -- resource teardown --------------------------------------------------------


def _teardown_mixin(monkeypatch):
    """A mixin with a fake clients facet plus the kit registries and monitoring
    stubbed in the lifecycle module namespace, so ``_teardown_resources`` is
    observable without real pools."""
    m = _Mixin()
    clients = MagicMock(shutdown_clients=AsyncMock())
    m.clients = clients
    checkpoint = MagicMock(close_all=AsyncMock())
    store = MagicMock(close_all=AsyncMock())
    writer = MagicMock()
    monkeypatch.setattr("tai42_skeleton.app.lifecycle.checkpoint_registry", lambda: checkpoint)
    monkeypatch.setattr("tai42_skeleton.app.lifecycle.store_registry", lambda: store)
    monkeypatch.setattr("tai42_skeleton.app.lifecycle.get_monitoring", lambda: MagicMock(writer=writer))
    return m, clients, checkpoint, store, writer


def test_teardown_resources_closes_pools_and_flushes(monkeypatch):
    m, clients, checkpoint, store, writer = _teardown_mixin(monkeypatch)
    asyncio.run(m._teardown_resources())
    clients.shutdown_clients.assert_awaited_once()
    checkpoint.close_all.assert_awaited_once()
    store.close_all.assert_awaited_once()
    writer.flush.assert_called_once_with()


def test_teardown_resources_runs_every_step_then_raises_group(monkeypatch):
    # One failing step must not skip the rest; collected failures surface as an
    # ExceptionGroup rather than being swallowed.
    m, clients, checkpoint, store, writer = _teardown_mixin(monkeypatch)
    checkpoint.close_all.side_effect = RuntimeError("checkpoint boom")

    with pytest.raises(ExceptionGroup) as ei:
        asyncio.run(m._teardown_resources())

    # Every other step still ran despite the checkpoint failure.
    clients.shutdown_clients.assert_awaited_once()
    store.close_all.assert_awaited_once()
    writer.flush.assert_called_once_with()
    assert "shutdown teardown failed" in str(ei.value)
    assert any(isinstance(e, RuntimeError) for e in ei.value.exceptions)


def test_app_context_startup_handler_failure_raises(monkeypatch):
    # The startup path runs the handlers with raise_on_error: a failed startup
    # handler must abort the boot loudly, never yield a healthy-looking
    # half-initialized app.
    m, *_ = _teardown_mixin(monkeypatch)
    monkeypatch.setattr(m, "start", lambda manifest: None)

    @m._on_startup
    async def boom():
        raise RuntimeError("startup boom")

    async def run():
        async with m.app_context(Manifest.model_validate({})):
            pass  # pragma: no cover — startup fails before the body runs

    with pytest.raises(RuntimeError, match=r"lifecycle handlers failed.*startup boom"):
        asyncio.run(run())


def test_live_mcp_status_snapshot():
    m = _Mixin()
    m._mcp_bound_tools = {"svc": {"b", "a"}}
    m._failed_mcps = {"down": "unavailable"}
    status = m._live_mcp_status()
    assert status["bound"] == {"svc": ["a", "b"]}
    assert status["failed"] == [{"title": "down", "status": "unavailable"}]


# -- real-app integration -----------------------------------------------------


def test_start_binds_global_handle_and_loads_tools():
    manifest = Manifest.model_validate(
        {"tools": [{"title": "fxt", "module": "tests.app._fixtures.tools_a", "include": ["greet"]}]}
    )

    async def run():
        async with app.app_context(manifest):
            # start() claims the global handle, binding it to this app impl.
            assert object.__getattribute__(tai42_app, "_impl") is app
            tools = await app.tools.get_tools()
            assert "greet" in tools

    asyncio.run(run())


def test_start_clears_cached_resource_manager():
    # A reload re-imports the storage module and rebuilds the storage provider;
    # start() must drop the cached resource manager so it cannot keep rendering
    # against (and pinning open) the previous provider.
    async def run():
        app._resource_manager_cache = cast("ResourceManager", "stale")
        async with app.app_context(Manifest.model_validate({})):
            assert app._resource_manager_cache is None

    asyncio.run(run())


def test_module_handlers_and_middleware_idempotent_across_update():
    # A lifecycle module registers a startup/shutdown handler + a middleware on
    # import. Each start()/update() re-imports it and re-fires the decorators;
    # the qualname-keyed registries must keep the counts at exactly one, never
    # 1->2->3 (which would re-run shutdowns N+1 times and duplicate middleware).
    manifest = Manifest.model_validate({"lifecycle_modules": ["tests.app._fixtures.lifecycle_reg"]})

    def _counts() -> tuple[int, int, int]:
        startups = sum(1 for k in app._startup_handlers if k.endswith(".startup_marker"))
        shutdowns = sum(1 for k in app._shutdown_handlers if k.endswith(".shutdown_marker"))
        middlewares = sum(1 for k in app._http_surface._middlewares if k.endswith(".MarkerMiddleware"))
        return startups, shutdowns, middlewares

    async def run():
        async with app.app_context(manifest):
            assert _counts() == (1, 1, 1)
            for _ in range(3):
                app._update(manifest)
                assert _counts() == (1, 1, 1)

    asyncio.run(run())


def test_update_drops_old_tools_and_reruns_reload_handlers():
    base = Manifest.model_validate(
        {"tools": [{"title": "fxt", "module": "tests.app._fixtures.tools_a", "include": ["greet"]}]}
    )
    empty = Manifest.model_validate({})
    reloaded: list[bool] = []

    async def run():
        async with app.app_context(base):

            @app.lifecycle.on_reload
            def _mark():
                reloaded.append(True)

            assert "greet" in await app.tools.get_tools()
            app._update(empty)
            # The greet tool is gone after re-init to an empty manifest.
            assert "greet" not in await app.tools.get_tools()
            # The reload handler ran during update().
            assert reloaded

    asyncio.run(run())


def test_failed_update_leaves_previous_tool_set_live():
    # A reload to a manifest with a broken module must fail loudly, but the
    # worker's previous tool surface is restored (re-added) rather than left empty
    # — a bad module bricks nothing.
    base = Manifest.model_validate(
        {"tools": [{"title": "fxt", "module": "tests.app._fixtures.tools_a", "include": ["greet"]}]}
    )
    broken = Manifest.model_validate({"lifecycle_modules": ["totally_bogus_pkg_xyz"]})

    async def run():
        async with app.app_context(base):
            assert "greet" in await app.tools.get_tools()
            with pytest.raises(ImportError, match="totally_bogus_pkg_xyz"):
                app._update(broken)
            # The previous tool surface is restored after the failed reload.
            assert "greet" in await app.tools.get_tools()

    asyncio.run(run())


def test_webhook_verifier_modules_import_registers_verifier():
    # A manifest ``webhook_verifier_modules`` entry is imported at app load like
    # the lifecycle modules; its import-time register(...) side-effect lands in
    # the verifier registry, and a reload (which resets the registry, then
    # re-imports) re-registers cleanly rather than tripping the duplicate guard.
    manifest = Manifest.model_validate({"webhook_verifier_modules": ["tests.app._fixtures.webhook_verifier_mod"]})

    async def run():
        async with app.app_context(manifest):
            assert app.webhook_verifiers.get("fixture_verifier") is not None
            app._update(manifest)
            assert app.webhook_verifiers.get("fixture_verifier") is not None

    asyncio.run(run())


def test_webhook_verifier_modules_import_failure_is_loud():
    # A broken verifier module must fail the boot loudly, exactly as a broken
    # lifecycle module does — never a silently unregistered verifier.
    manifest = Manifest.model_validate({"webhook_verifier_modules": ["totally_bogus_verifier_pkg"]})

    async def run():
        async with app.app_context(manifest):
            pass

    with pytest.raises(ImportError, match="totally_bogus_verifier_pkg"):
        asyncio.run(run())


def test_channel_modules_import_registers_channel():
    # A manifest ``channel_modules`` entry is imported at app load like the
    # verifier modules; its import-time register(...) side-effect lands in the
    # channel registry, and a reload (which resets the registry, then
    # re-imports) re-registers cleanly rather than tripping the duplicate guard.
    manifest = Manifest.model_validate({"channel_modules": ["tests.app._fixtures.channel_mod"]})

    async def run():
        async with app.app_context(manifest):
            assert app.channels.get("fixture_channel") is not None
            app._update(manifest)
            assert app.channels.get("fixture_channel") is not None

    asyncio.run(run())


def test_channel_modules_import_failure_is_loud():
    # A broken channel module must fail the boot loudly, exactly as a broken
    # verifier module does — never a silently undeliverable channel.
    manifest = Manifest.model_validate({"channel_modules": ["totally_bogus_channel_pkg"]})

    async def run():
        async with app.app_context(manifest):
            pass

    with pytest.raises(ImportError, match="totally_bogus_channel_pkg"):
        asyncio.run(run())


def test_reload_dropping_channel_module_unregisters_channel():
    # The per-start reset is real: a reload to a manifest without the channel
    # module leaves the dropped channel unresolvable, never lingering.
    with_channel = Manifest.model_validate({"channel_modules": ["tests.app._fixtures.channel_mod"]})
    empty = Manifest.model_validate({})

    async def run():
        async with app.app_context(with_channel):
            assert app.channels.get("fixture_channel") is not None
            app._update(empty)
            with pytest.raises(KeyError, match="unknown channel"):
                app.channels.get("fixture_channel")

    asyncio.run(run())


def test_reload_dropping_prompt_module_removes_its_prompts():
    # Reload removes stale prompts symmetrically with tools: a reload dropping the
    # prompt-owning module leaves NONE of its prompts live.
    with_prompt = Manifest.model_validate({"lifecycle_modules": ["tests.app._fixtures.prompt_mod"]})
    empty = Manifest.model_validate({})

    async def run():
        async with app.app_context(with_prompt):
            names = {p.name for p in await app._fast_mcp.list_prompts()}
            assert "fixture_prompt" in names
            app._update(empty)
            names = {p.name for p in await app._fast_mcp.list_prompts()}
            assert "fixture_prompt" not in names

    asyncio.run(run())


def test_reload_mcp_unregisters_vanished_tool():
    # A re-probed MCP that no longer serves a previously-bound tool must drop it
    # from the base registry too (symmetry with deregister_mcp) — not leave it
    # stale until a full reload.
    m = _Mixin()
    m._manifest = Manifest.model_validate({"mcp": [_cfg("svc").model_dump()]})
    m._probe_mcp = AsyncMock(return_value=[_FakeMcpTool()])
    # Two tools bound before; _Mixin._mcp_tools rebinds only ``svc_t`` on reload,
    # so ``svc_old`` vanished.
    m._mcp_bound_tools = {"svc": {"svc_old", "svc_t"}}
    unregistered: list[str] = []
    m._tool_registry.unregister_tool_base = lambda name: (unregistered.append(name), [])[1]  # type: ignore[method-assign]

    out = m._reload_mcp("svc")

    assert out["status"] == "ok"
    assert out["tools"] == ["svc_t"]
    assert unregistered == ["svc_old"]


def test_reload_with_connector_plugin_is_reload_safe():
    # A manifest carrying a connector plugin module: start() imports it, running
    # register_connector(...). update() re-imports the same module — without the
    # start()-time registry reset the duplicate guard would crash the reload.
    from tai42_skeleton.connectors.providers import registry as conn_registry

    provider_id = "fixture_conn"
    manifest = Manifest.model_validate({"lifecycle_modules": ["tests.app._fixtures.connector_plugin"]})

    saved = dict(conn_registry._REGISTRY)
    conn_registry._REGISTRY.clear()

    async def run():
        async with app.app_context(manifest):
            assert conn_registry.get_provider(provider_id).id == provider_id
            # Reload re-imports the plugin; the registry reset makes the repeated
            # register_connector(...) safe instead of a duplicate-id crash.
            app._update(manifest)
            assert conn_registry.get_provider(provider_id).id == provider_id

    try:
        asyncio.run(run())
    finally:
        conn_registry._REGISTRY.clear()
        conn_registry._REGISTRY.update(saved)


def test_reload_config_refreshes_env_and_reinitializes(monkeypatch):
    import os

    captured_env = {"NEW_KEY": "v1", "OTHER": "v2"}
    reinit_manifests: list[Manifest] = []

    async def run():
        async with app.app_context(Manifest.model_validate({})):
            monkeypatch.setattr(app.config.config_manager, "read_env", lambda: captured_env)
            monkeypatch.setattr(app.config.config_manager, "read_manifest", dict)

            # Spy the reinitialize seam so the "and_reinitializes" half is asserted,
            # not just the env refresh: reload_config MUST drive _update (the soft
            # re-init) with a manifest freshly built from read_manifest, while the
            # real re-init still runs underneath.
            real_update = app._update

            def spy_update(manifest):
                reinit_manifests.append(manifest)
                return real_update(manifest)

            monkeypatch.setattr(app, "_update", spy_update)

            out = app.admin.reload_config()
            assert out == {"status": "ok", "env_keys": 2}
            # Env refreshed into the process environment.
            assert os.environ["NEW_KEY"] == "v1"
            assert os.environ["OTHER"] == "v2"
            # Reinitialized: _update ran exactly once with a freshly-built Manifest.
            assert len(reinit_manifests) == 1
            assert isinstance(reinit_manifests[0], Manifest)

    try:
        asyncio.run(run())
    finally:
        os.environ.pop("NEW_KEY", None)
        os.environ.pop("OTHER", None)


def test_reload_config_drops_env_keys_removed_from_source(monkeypatch):
    import os

    async def run():
        async with app.app_context(Manifest.model_validate({})):
            monkeypatch.setattr(app.config.config_manager, "read_manifest", dict)

            monkeypatch.setattr(app.config.config_manager, "read_env", lambda: {"K1_RC": "a", "K2_RC": "b"})
            app.admin.reload_config()
            assert os.environ["K1_RC"] == "a"
            assert os.environ["K2_RC"] == "b"

            # K2 removed from the source env: the next reload must drop it, not
            # leave it lingering as stale config.
            monkeypatch.setattr(app.config.config_manager, "read_env", lambda: {"K1_RC": "a"})
            app.admin.reload_config()
            assert os.environ["K1_RC"] == "a"
            assert "K2_RC" not in os.environ

    try:
        asyncio.run(run())
    finally:
        os.environ.pop("K1_RC", None)
        os.environ.pop("K2_RC", None)


def test_initialize_components_loads_and_records_failed_mcp(monkeypatch):
    # A manifest MCP whose probe times out is skipped + recorded, not fatal.
    manifest = Manifest.model_validate({"mcp": [_cfg("downsvc").model_dump()]})
    monkeypatch.setattr(app, "_probe_mcp", AsyncMock(side_effect=TimeoutError("slow")))

    async def run():
        async with app.app_context(manifest):
            assert app.admin.list_failed_mcps() == [{"title": "downsvc", "status": "unavailable"}]

    asyncio.run(run())


def test_initialize_components_binds_probed_mcp_tools(monkeypatch):
    manifest = Manifest.model_validate({"mcp": [_cfg("upsvc").model_dump()]})
    monkeypatch.setattr(app, "_probe_mcp", AsyncMock(return_value=[_FakeMcpTool()]))

    async def run():
        async with app.app_context(manifest):
            status = app.admin.live_mcp_status()
            assert "upsvc" in status["bound"]

    asyncio.run(run())


def test_initialize_helpers_require_started():
    m = _Mixin()  # _manifest is None
    with pytest.raises(RuntimeError, match="not started"):
        m._initialize_registries()
    with pytest.raises(RuntimeError, match="not started"):
        m._initialize_components()


def test_load_mcps_early_returns_without_mcp():
    m = _Mixin()
    m._manifest = Manifest.model_validate({})
    assert asyncio.run(m._load_mcps()) == ([], [])


def test_start_imports_lifecycle_router_and_middleware_modules():
    # The lifecycle/routers/middlewares module loops each import their listed
    # packages (pointed at a neutral fixture package here).
    manifest = Manifest.model_validate(
        {
            "lifecycle_modules": ["tests.app._fixtures.neutral"],
            "routers_modules": ["tests.app._fixtures.neutral"],
            "middlewares_modules": ["tests.app._fixtures.neutral"],
            # "none" keeps this focused on the loop importing the listed module,
            # not the default set.
            "default_routers": "none",
        }
    )

    async def run():
        async with app.app_context(manifest):
            assert app._manifest is manifest

    asyncio.run(run())


def test_start_raises_on_broken_manifest_module():
    # A manifest naming a module that fails to import is corrupt configuration:
    # start() aborts loudly, naming the module, instead of booting a
    # silently-degraded server.
    manifest = Manifest.model_validate({"lifecycle_modules": ["totally_bogus_pkg_xyz"]})

    async def run():
        async with app.app_context(manifest):
            pass  # pragma: no cover — start() fails before the body runs

    with pytest.raises(ImportError, match="totally_bogus_pkg_xyz"):
        asyncio.run(run())


def test_refresh_manifest_mcp_grafts_reread_rows(monkeypatch):
    async def run():
        async with app.app_context(Manifest.model_validate({})):
            fresh = Manifest.model_validate({"mcp": [_cfg("late").model_dump()]})
            monkeypatch.setattr(app.config.config_manager, "read_manifest", lambda: fresh.model_dump())
            app._refresh_manifest_mcp()
            assert app._manifest is not None
            assert "late" in (app._manifest.mcp_map or {})

    asyncio.run(run())


def test_probe_mcp_uses_pooled_client_seam(monkeypatch):
    # The real ``_probe_mcp`` opens a fresh pooled FastMCPClient and lists tools.
    class _Client:
        async def list_tools(self):
            return [_FakeMcpTool()]

    @asynccontextmanager
    async def fake_ctx(client_cls, *args, **kwargs):
        yield _Client()

    async def run():
        async with app.app_context(Manifest.model_validate({})):
            monkeypatch.setattr(app.clients, "client_ctx", fake_ctx)
            tools = await app._probe_mcp(_cfg("svc"))
            assert [t.name for t in tools] == ["ping"]

    asyncio.run(run())


def test_start_logs_kind_summary_and_warns_once_on_noop_monitoring(monkeypatch, caplog):
    # A real boot must render the [kinds] summary and, with NoOp monitoring as the
    # active recorder, fire the once-per-process warning exactly once — the manual
    # smoke path made load-bearing. Reset the once-per-process guard and force the
    # monitoring registry back to its NoOp default so the boot sees "not configured".
    monkeypatch.setattr(ks, "_NOOP_WARNED", False)
    reset_monitoring()

    async def run():
        async with app.app_context(Manifest.model_validate({})):
            pass

    with caplog.at_level(logging.INFO, logger="tai42_skeleton.app.lifecycle"):
        asyncio.run(run())

    messages = [r.getMessage() for r in caplog.records if r.name == "tai42_skeleton.app.lifecycle"]
    assert "[kinds]" in messages
    assert any("monitoring: default" in m for m in messages)
    noop_warnings = [r for r in caplog.records if "monitoring: OFF" in r.getMessage()]
    assert len(noop_warnings) == 1


def test_start_fails_when_kind_status_collector_raises(monkeypatch):
    # Task 2's contract: a collector exception during the startup summary must fail
    # the boot loudly, never a silently degraded server with a missing table.
    def _boom():
        raise RuntimeError("collector exploded")

    monkeypatch.setattr("tai42_skeleton.app.lifecycle.collect_kind_status", _boom)

    async def run():
        async with app.app_context(Manifest.model_validate({})):
            pass  # pragma: no cover — start() fails before the body runs

    with pytest.raises(RuntimeError, match="collector exploded"):
        asyncio.run(run())


# -- default-router composition (_effective_router_modules) -------------------


def _effective(default_routers=None, routers_modules=None) -> list[str]:
    m = _Mixin()
    body: dict[str, Any] = {}
    if default_routers is not None:
        body["default_routers"] = default_routers
    if routers_modules is not None:
        body["routers_modules"] = routers_modules
    m._manifest = Manifest.model_validate(body)
    return m._effective_router_modules()


def test_all_with_empty_list_is_defaults_then_catch_all_last():
    # "all" is the default when default_routers is omitted.
    eff = _effective(routers_modules=[])
    assert eff == [*DEFAULT_API_ROUTERS, STUDIO_SPA_ROUTER]
    assert eff[-1] == STUDIO_SPA_ROUTER


def test_all_with_extra_appends_extra_before_catch_all():
    eff = _effective(default_routers="all", routers_modules=["some.extra.router"])
    assert eff == [*DEFAULT_API_ROUTERS, "some.extra.router", STUDIO_SPA_ROUTER]


def test_all_dedups_a_redundantly_listed_core_router_no_double_mount():
    # A manifest that still lists a defaulted core router imports it exactly once.
    core = DEFAULT_API_ROUTERS[0]
    eff = _effective(default_routers="all", routers_modules=[core])
    assert eff.count(core) == 1
    assert eff[-1] == STUDIO_SPA_ROUTER
    assert eff == [*DEFAULT_API_ROUTERS, STUDIO_SPA_ROUTER]


def test_all_never_double_appends_an_explicitly_listed_catch_all():
    # An operator listing the catch-all among extras under "all" gets it exactly
    # once, still last — never in the middle, never twice.
    eff = _effective(default_routers="all", routers_modules=[STUDIO_SPA_ROUTER])
    assert eff.count(STUDIO_SPA_ROUTER) == 1
    assert eff == [*DEFAULT_API_ROUTERS, STUDIO_SPA_ROUTER]


def test_api_mounts_defaults_without_catch_all():
    eff = _effective(default_routers="api", routers_modules=["some.extra.router"])
    assert eff == [*DEFAULT_API_ROUTERS, "some.extra.router"]
    assert STUDIO_SPA_ROUTER not in eff


def test_api_honors_an_explicitly_listed_catch_all_last():
    # Under "api" the loader never adds the catch-all, but an operator who lists it
    # explicitly is honored — placed last.
    eff = _effective(default_routers="api", routers_modules=[STUDIO_SPA_ROUTER])
    assert eff == [*DEFAULT_API_ROUTERS, STUDIO_SPA_ROUTER]


def test_none_is_the_verbatim_manual_surface_with_catch_all_last():
    eff = _effective(
        default_routers="none",
        routers_modules=["a.router", STUDIO_SPA_ROUTER, "b.router"],
    )
    # No defaults; the operator list is authoritative, catch-all moved to last.
    assert eff == ["a.router", "b.router", STUDIO_SPA_ROUTER]


def test_none_with_empty_list_is_empty_mcp_only():
    assert _effective(default_routers="none", routers_modules=[]) == []


def test_effective_router_modules_requires_started():
    m = _Mixin()  # _manifest is None
    with pytest.raises(RuntimeError, match="not started"):
        m._effective_router_modules()


def test_default_boot_mounts_the_studio_and_cli_page_routes():
    """A DEFAULT ("all") boot registers the specific Studio/CLI-consumed endpoints.
    Asserted by LITERAL path (not derived from DEFAULT_API_ROUTERS) against the
    routes the boot registered, so the composition is checked against real
    materialized routes a partial router list is prone to omit."""
    manifest = Manifest.model_validate({"default_routers": "all"})

    async def run():
        async with app.app_context(manifest):
            routes = app._fast_mcp._additional_http_routes
            registered = {getattr(route, "path", None) for route in routes}
            # channels backs the Interactions ChannelsCard, sub-mcp the Manifest
            # SubMcpTab, resources/get the `tai resources get` CLI.
            assert "/api/channels" in registered
            assert "/api/sub-mcp" in registered
            assert "/api/resources/get" in registered
            # A representative privileged page a partial manifest could omit.
            assert "/api/marketplace/install" in registered
            assert "/api/backup/export" in registered
            # The SPA catch-all is mounted AND is the LAST-registered route — its
            # ``/{spa_path:path}`` matches any path, so anything registered after it
            # would be shadowed. Assert its terminal position, not mere membership.
            assert getattr(routes[-1], "path", None) == "/{spa_path:path}"

    asyncio.run(run())


# -- the shared router importer is manifest-aware -----------------------------


def test_started_none_boot_serves_only_the_curated_routers(monkeypatch):
    """Regression: an access-control-enabled boot with ``default_routers="none"`` and a
    curated two-module ``routers_modules`` must serve ONLY those modules' routes. The
    access-control startup audits enumerate the surface through ``load_all_routes``/``load_api_routes``;
    that shared importer must NOT pull the whole ``tai42_skeleton.routers`` package into the
    started app's live route table — doing so silently serves every router the manifest
    excluded.

    Asserted on the PER-INSTANCE served table of a FRESH app: the process-global
    ``route_registry`` is legitimately polluted by other tests, and the singleton's served
    table accumulates across boots. ``marketplace``/``connectors`` are popped from
    ``sys.modules`` first, so the whole-package importer — if it ran — WOULD re-execute
    their ``@custom_route`` decorators into this app's live table; this test asserts that importer does not run."""
    import sys

    from tai42_skeleton.access_control.startup import (
        check_accounts_providers_configured,
        check_always_public_routes,
        check_fenced_routes_resolvable,
        check_route_actions,
        check_spa_shell_public,
        probe_identity_provider,
        seed_roles,
    )
    from tai42_skeleton.app.server import TaiMCP

    instance = TaiMCP(name="curation-under-test")
    # Wire the access-control startup audits exactly as the real build does when access
    # control is enabled, so every audit's route enumeration runs during this boot.
    for audit in (
        probe_identity_provider,
        seed_roles,
        check_always_public_routes,
        check_spa_shell_public,
        check_route_actions,
        check_fenced_routes_resolvable,
        check_accounts_providers_configured,
    ):
        instance.lifecycle.on_startup(audit)

    excluded = ("tai42_skeleton.routers.marketplace", "tai42_skeleton.routers.connectors")
    saved_modules = {name: sys.modules.pop(name, None) for name in excluded}

    manifest = Manifest.model_validate(
        {
            "default_routers": "none",
            "routers_modules": ["tai42_skeleton.routers.tools", "tai42_skeleton.routers.health"],
        }
    )

    async def run() -> set[str | None]:
        async with instance.app_context(manifest):
            return {getattr(route, "path", None) for route in instance._fast_mcp._additional_http_routes}

    try:
        # ``app_context`` binds the instance under test; scoping that bind restores
        # whatever this process had bound when the boot is over.
        with tai42_app.bound(None):
            served = asyncio.run(run())
    finally:
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module

    # The curated modules' routes ARE served ...
    assert "/api/tools" in served
    assert "/health" in served
    # ... and NOTHING from the excluded routers reached the live route table.
    assert not any((path or "").startswith("/api/marketplace") for path in served)
    assert not any((path or "").startswith("/api/connectors") for path in served)


def test_load_all_routes_uses_the_effective_set_when_started_and_whole_package_offline(monkeypatch):
    """The shared importer chooses its universe: a bound+STARTED app enumerates its
    manifest's effective router set and must NEVER fall back to the whole-package importer;
    an unbound (offline spec-harness) process MUST use it."""
    from tai42_skeleton.app import route_registry as rr

    # Started case: the whole-package importer must not run — the effective set is the
    # universe. Patched inside the context so start()'s own router import is untouched.
    def _forbidden() -> None:
        raise AssertionError("_import_all_router_modules must not run in a started process")

    manifest = Manifest.model_validate({"default_routers": "none", "routers_modules": ["tai42_skeleton.routers.tools"]})

    async def run_started() -> None:
        async with app.app_context(manifest):
            monkeypatch.setattr(rr, "_import_all_router_modules", _forbidden)
            rr.load_all_routes()  # no raise: the effective set is enumerated directly

    asyncio.run(run_started())

    # Offline case: with no deployment bound, the whole-package importer MUST run.
    calls: list[int] = []
    monkeypatch.setattr(rr, "_import_all_router_modules", lambda: calls.append(1))
    # Unbound for the call: no started deployment answers effective_router_modules().
    with tai42_app.bound(None):
        rr.load_all_routes()
    assert calls == [1]


def test_load_all_routes_leaves_the_bound_app_exactly_as_it_found_it(monkeypatch):
    """The enumeration is a READ: it must never replace the process's app binding.

    A bound impl that answers no router universe — a partially-faked app, a harness
    stand-in — takes the offline branch, whose router import runs under the ``_SpecApp``
    stand-in for that import alone. Overwriting the binding instead would strand
    whichever component owns the real one, and an unbound process must come back
    unbound rather than silently acquiring a spec app.
    """
    from types import SimpleNamespace

    from tai42_skeleton.app import route_registry as rr

    imports: list[str] = []

    def _record_universe_app() -> None:
        # The offline import needs a stand-in exposing ``http``; record which impl is
        # bound while it runs.
        imports.append(type(tai42_app.http._app).__name__)  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setattr(rr, "_import_all_router_modules", _record_universe_app)

    partial = SimpleNamespace(storage="fake-storage")
    with tai42_app.bound(partial):
        rr.load_all_routes()
        assert tai42_app.storage == "fake-storage"

    with tai42_app.bound(None):
        rr.load_all_routes()
        with pytest.raises(AttributeError, match="accessed before bind"):
            _ = tai42_app.storage

    assert imports == ["_SpecApp", "_SpecApp"]
