import asyncio
import inspect
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mcp
from fastmcp import FastMCP
from fastmcp.prompts import Prompt
from fastmcp.resources import Resource
from fastmcp.resources.template import ResourceTemplate
from fastmcp.tools import Tool
from tai_contract.access_control.registry import reset_registry as reset_identity_registry
from tai_contract.accounts import reset_registry as reset_accounts_registry
from tai_contract.app import tai_app
from tai_contract.manifest import TaiMCPConfig
from tai_kit.clients import shutdown_all_clients
from tai_kit.clients.impl.mcp import FastMCPClient
from tai_kit.llm.checkpoint.checkpoint_registry import checkpoint_registry
from tai_kit.llm.store.store_registry import store_registry
from tai_kit.settings import reset_all_settings

from tai_skeleton.app.boot_rules import require_bus_for_backend, require_bus_for_k8s
from tai_skeleton.app.bus import OriginKind, WorkerBus, make_origin
from tai_skeleton.app.bus_settings import bus_settings
from tai_skeleton.app.importer import import_or_reload_package
from tai_skeleton.app.kind_status import collect_kind_status, warn_if_noop_monitoring
from tai_skeleton.app.reload_gate import reload_gate
from tai_skeleton.connectors.providers.registry import reset_registry
from tai_skeleton.extensions import ExtensionRegistry
from tai_skeleton.manifest import Manifest
from tai_skeleton.monitoring import get_monitoring
from tai_skeleton.operations.projection import project_operations
from tai_skeleton.operations.registry import operation_registry
from tai_skeleton.settings.cache import mcp_probe_timeout
from tai_skeleton.settings.settings import CoreSettings
from tai_skeleton.tools import ToolRegistry

if TYPE_CHECKING:
    from tai_contract.config.manager import ConfigManager

    from tai_skeleton.agent.binding import AgentBinding
    from tai_skeleton.app.clients import ClientsFacet
    from tai_skeleton.app.sessions import SessionRegistry
    from tai_skeleton.app.sub_mcp_app import SubMcpAppRouter
    from tai_skeleton.channels.registry import ChannelRegistry
    from tai_skeleton.presets.manager import PresetManager
    from tai_skeleton.template import ResourceManager
    from tai_skeleton.webhooks.registry import WebhookVerifierRegistry

logger = logging.getLogger(__name__)


def _op_field(op: dict[str, Any], key: str) -> Any:
    """A required field of a delivered fleet op, raising loudly when absent —
    a malformed fleet op must fail its confirmation, never apply a partial op."""
    value = op.get(key)
    if value is None:
        raise ValueError(f"{op.get('op')} fleet op missing {key!r}")
    return value


@dataclass
class _CapturedComponents:
    """The live, RE-ADDABLE FastMCP components snapshotted before a reload.

    Precisely typed so the ``add_*`` restore is statically checked: ``list_*``
    returns wire types ``add_*`` cannot re-add, so each surface holds the FastMCP
    component objects (``Tool`` by name, ``Prompt`` by name, ``Resource`` by URI)
    the restore path re-adds. A name that vanished between ``list_*`` and
    ``get_*`` is skipped at capture, so a ``None`` is never stored.
    """

    tools: dict[str, Tool] = field(default_factory=dict)
    prompts: dict[str, Prompt] = field(default_factory=dict)
    resources: dict[str, Resource] = field(default_factory=dict)


class TaiMCPLifecycleMixin(ABC):
    def __init__(self, *args, **kwargs):
        # None until start() loads a manifest; the registration decorators
        # truthiness-check it to no-op when a tool module is imported before
        # the app boots.
        self._manifest: Manifest | None = None

        # Empty placeholders until start() rebuilds them from the manifest; an
        # empty registry is a valid registry, so these are never None.
        self._tool_registry: ToolRegistry = ToolRegistry(set[str](), {})
        self._extension_registry: ExtensionRegistry = ExtensionRegistry(frozenset[str]())

        # ``on_duplicate="error"`` (server-wide: tools, prompts, resources) makes a
        # duplicate registration raise instead of the default warn-then-replace
        # (last-write-win). Every
        # legitimate rebind removes the name first, so an in-boot duplicate is
        # always a genuine collision (two modules claiming one tool name, a
        # normalized-name MCP collision, or an agent run tool shadowing a tool).
        self._fast_mcp: FastMCP = FastMCP(*args, on_duplicate="error", **kwargs)

        # Keyed by qualified name so a module re-import (each start() re-imports
        # the lifecycle modules) replaces rather than accumulates its handler,
        # while a construction-time (build_app) handler — registered once and
        # never re-imported — persists across reloads.
        self._startup_handlers: dict[str, Callable] = {}
        self._shutdown_handlers: dict[str, Callable] = {}

        # Env keys applied by the last _reload_config so the next reload can drop
        # keys that were removed from the source env (a merge-only update would
        # leave them lingering as stale config).
        self._loaded_env_keys: set[str] = set()

        # Dynamic tool loaders re-run after every re-init (update() drops all
        # tools). Keyed by qualified name so a module re-import replaces rather
        # than accumulates.
        self._reload_handlers: dict[str, Callable] = {}

        # Per-kind tool reloaders (e.g. "flow" / "agentic_flow"), registered
        # by the host app via @tool_reloader; fleet reload_tool/remove_tool
        # ops dispatch through run_tool_reload on every worker.
        self._tool_reloaders: dict[str, Callable] = {}

        # Manifest MCP servers that failed their viability check and were
        # skipped instead of crashing startup. title -> "unavailable".
        self._failed_mcps: dict[str, str] = {}
        # Tools bound on the live server per MCP title, so a targeted reload
        # can cleanly replace them.
        self._mcp_bound_tools: dict[str, set[str]] = {}
        # Per-title names a scoped MCP (re)bind REFUSED because a registered preset
        # already owns them — the returning server must not clobber the preset, so
        # the reload surfaces the conflict rather than binding over it.
        self._mcp_preset_conflicts: dict[str, set[str]] = {}

        # The one app-owned worker-bus subscription (cross-worker fleet ops). The
        # bus is internal infrastructure that SURVIVES reloads (it is not a manifest
        # plugin), so this process joins once in app_context and keeps a single
        # long-lived subscription for its whole lifetime — a reload re-imports the
        # backend but never tears down and rejoins the subscription.
        self._bus: WorkerBus | None = None
        self._bus_subscription_task: asyncio.Task[None] | None = None
        # Latched once this process's first boot self-resync completes: the tool
        # registry is then fully built and stable for the run. A forking/consuming
        # backend runtime awaits this (``wait_until_ready``) before its work loop
        # accepts a job, so a work-horse forked at boot never inherits a half-built
        # registry. One-way for the process lifetime — a reconnect self-resync
        # re-runs while the app is already live and must not un-ready it — so it is
        # NOT reset by ``start()``/reload.
        self._boot_ready = asyncio.Event()
        # The serving loop, remembered so a reload running on a worker thread can
        # marshal loop-affine work (preset reconcile, checkpoint/store close) back
        # onto it.
        self._serving_loop: asyncio.AbstractEventLoop | None = None

        # Handlers fired with the OP NAME after every applied bus op AND after the
        # reconnect self-resync reload. Keyed by qualified name so a module re-import
        # replaces rather than accumulates (mirrors the startup/reload registries). A
        # backend whose worker model needs post-op work registers here (celery's
        # prefork pool turnover is the known consumer).
        self._fleet_op_applied_handlers: dict[str, Callable] = {}

        # Lifespan-owned exponential-backoff task that re-probes failed-at-boot
        # MCP servers so a deploy-order race (the MCP pod comes up after the
        # skeleton) self-heals without a manual reload. Distinct from the
        # worker-bus subscription task above.
        self._reprobe_task: asyncio.Task[None] | None = None

    def _on_startup(self, func: Callable):
        self._startup_handlers[f"{func.__module__}.{func.__qualname__}"] = func
        return func

    def _on_reload(self, func: Callable):
        """Register a handler to re-run after every in-place re-init."""
        self._reload_handlers[f"{func.__module__}.{func.__qualname__}"] = func
        return func

    def _on_fleet_op_applied(self, func: Callable):
        """Register a handler fired with the OP NAME after every applied bus op and
        after the reconnect self-resync reload. Keyed by qualified name so a module
        re-import replaces rather than accumulates."""
        self._fleet_op_applied_handlers[f"{func.__module__}.{func.__qualname__}"] = func
        return func

    def _tool_reloader(self, kind: str) -> Callable:
        """Register an ``(action, name) -> dict`` reloader for one tool kind.
        Last registration wins (module re-imports re-run the decorator)."""

        def decorator(func: Callable) -> Callable:
            self._tool_reloaders[kind] = func
            return func

        return decorator

    async def _run_tool_reload(self, kind: str, action: str, name: str) -> dict[str, Any]:
        """Apply one tool reload/remove via the registered reloader. Raises on
        an unknown kind/action or a failing reloader."""
        if action not in ("reload", "remove"):
            raise ValueError(f"Unknown tool-reload action {action!r} (expected 'reload' or 'remove')")
        reloader = self._tool_reloaders.get(kind)
        if reloader is None:
            raise RuntimeError(
                f"No tool reloader registered for kind {kind!r} (registered: {sorted(self._tool_reloaders)})"
            )
        if inspect.iscoroutinefunction(reloader):
            result = await reloader(action, name)
        else:
            result = reloader(action, name)
        return result or {"kind": kind, "action": action, "name": name, "status": "ok"}

    def _on_shutdown(self, func: Callable):
        self._shutdown_handlers[f"{func.__module__}.{func.__qualname__}"] = func
        return func

    async def _run_handlers(self, handlers: list[Callable], raise_on_error: bool = False):
        """Run lifecycle handlers, always attempting every handler. The shutdown
        path swallows-and-logs so teardown reaches every handler; the startup and
        reload paths pass ``raise_on_error`` so a failed handler surfaces loudly
        instead of leaving a healthy-looking half-initialized app or reporting a
        successful reload with missing tools."""
        errors: list[tuple] = []
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.error(f"Error in lifecycle handler {handler.__name__}: {e}", exc_info=True)
                errors.append((handler.__name__, e))
        if raise_on_error and errors:
            raise RuntimeError("lifecycle handlers failed: " + ", ".join(f"{name}: {exc!r}" for name, exc in errors))

    @asynccontextmanager
    async def app_context(self, manifest: Manifest, *, origin_kind: OriginKind = OriginKind.serve):
        # Bus/boot invariant at the one seam both `tai serve` and `tai backend`
        # cross: a process with a registered backend, or a k8s-mode deployment, must
        # have the worker bus configured — otherwise sibling workers or sibling pods
        # serve stale config after a reload. Refuse loudly before the app starts.
        require_bus_for_k8s()
        require_bus_for_backend(manifest)
        try:
            self.start(manifest)
            # raise_on_error: a failed startup handler must abort the boot
            # loudly, never leave a healthy-looking half-initialized app.
            await self._run_handlers(list(self._startup_handlers.values()), raise_on_error=True)
            # Bind the reload gate + remember the serving loop, so a reload running
            # on a worker thread can marshal loop-affine work (preset reconcile,
            # checkpoint/store close) back onto the serving loop.
            self._serving_loop = asyncio.get_running_loop()
            reload_gate.bind_to_running_loop()
            # Join the worker bus: construct this process's one bus (origin kind
            # ``serve`` or ``backend``) and open its single long-lived subscription.
            # The subscription registers presence and self-resyncs (reload_config)
            # through on_ready on every (re)connect.
            self._bus = self._build_bus(origin_kind)
            self._spawn_bus_subscription()
            # Start the failed-MCP re-probe task so a server that failed its boot
            # probe self-heals on an exponential backoff without a manual reload.
            self._spawn_reprobe_task()
            yield self
        finally:
            await self._cancel_bus_subscription()
            await self._cancel_reprobe_task()
            # Shutdown keeps swallow-and-log so teardown runs every handler.
            await self._run_handlers(list(self._shutdown_handlers.values()))
            await self._teardown_resources()

    # --- worker-bus subscription (cross-worker fleet ops) ---

    @property
    def bus(self) -> WorkerBus:
        """This process's worker bus, built in ``app_context``. The runtime-op
        publishers and the fleet census route reach the fleet through it. Raises if
        accessed before ``app_context`` builds it."""
        bus = self._bus
        if bus is None:
            raise RuntimeError("the worker bus is not built — enter app_context first")
        return bus

    def _build_bus(self, origin_kind: OriginKind) -> WorkerBus:
        """Construct this process's one worker bus. With ``TAI_BUS_REDIS_URL`` set the
        real bus joins the fleet; otherwise the no-op ``WorkerBus.local`` variant —
        legal only under the boot rules that permit a busless deployment (single
        worker, file mode, no backend)."""
        origin = make_origin(origin_kind)
        settings = bus_settings()
        if settings.enabled:
            return WorkerBus(settings, origin)
        return WorkerBus.local(origin)

    def _spawn_bus_subscription(self) -> None:
        """Start the one long-lived bus subscription on the serving loop. Owned by
        ``app_context``; runs until cancelled at shutdown. The subscription reconnects
        with backoff internally and fires ``on_ready`` (the self-resync) after
        subscribe+presence-register on every (re)connect."""
        bus = self._bus
        if bus is None:
            raise RuntimeError("bus subscription spawned before the bus was built")
        self._bus_subscription_task = asyncio.create_task(
            bus.subscribe(bus.origin, self._apply_bus_op, on_ready=self._resync_on_ready),
            name="tai-worker-bus-subscription",
        )
        self._bus_subscription_task.add_done_callback(self._on_perpetual_task_done)

    async def _resync_on_ready(self) -> None:
        """Self-resync run after subscribe, before presence-register, on every
        (re)connect: a local ``reload_config`` re-reads persisted state so a broadcast
        missed while this worker was away self-heals. Routed through the same apply
        path as a delivered op so ``on_fleet_op_applied`` handlers fire — else a
        reconnecting celery worker would resync only its main process while its prefork
        children stay stale on the exact path the resync heals.

        A failing resync is non-fatal to the subscription. This runs inside
        ``subscribe`` BEFORE the message loop, so a propagating error would kill the
        subscription task with no reconnect — silently ejecting this worker from the
        fleet for the process lifetime. Instead the failure is ERROR-logged and
        swallowed so the subscription stays live (future broadcasts still reach this
        worker) and the next reconnect re-attempts the resync — an explicit, logged
        recovery. Cancellation propagates untouched.

        A SUCCESSFUL resync latches boot-ready (:meth:`_mark_boot_ready`): the tool
        registry is rebuilt and stable, so a backend runtime awaiting
        ``wait_until_ready`` may now consume work. A failed resync does NOT latch — a
        consumer stays blocked (and fails loudly on its own timeout) rather than
        forking against a registry a broken reload left half-built."""
        try:
            await self._apply_bus_op({"op": "reload_config"})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                "worker bus: self-resync reload on (re)connect failed — subscription stays live, "
                "resync retried on the next reconnect",
                exc_info=True,
            )
        else:
            self._mark_boot_ready()

    def _mark_boot_ready(self) -> None:
        """Latch the boot-ready signal on the FIRST successful self-resync. One-way:
        a reconnect resync re-enters here on a live app but must never clear the
        latch, so a consumer already past it is never retroactively un-readied."""
        if not self._boot_ready.is_set():
            logger.info("app boot-ready: first self-resync complete — tool registry built and stable")
            self._boot_ready.set()

    async def _wait_until_ready(self) -> None:
        """Backs ``app.lifecycle.wait_until_ready``: block until the first boot
        self-resync has latched boot-ready."""
        await self._boot_ready.wait()

    async def _cancel_bus_subscription(self) -> None:
        """Cancel the bus subscription and await its termination — the shutdown
        counterpart of ``_spawn_bus_subscription``.

        A task that died with a non-``CancelledError`` exception was already surfaced
        at ERROR by its done-callback, so it is awaited-and-swallowed here rather than
        re-raised — this runs inside ``app_context``'s shutdown ``finally``, and one
        dead task must not skip the remaining teardown."""
        task = self._bus_subscription_task
        self._bus_subscription_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Already logged at ERROR by the done-callback; swallowed so a dead
            # subscription cannot abort the remaining shutdown steps.
            pass

    async def _apply_bus_op(self, op: dict[str, Any]) -> Any:
        """Apply one fleet op delivered from a sibling worker (or the self-resync),
        then fire the post-apply hooks.

        The sibling-worker counterpart of a route-received admin call: it maps each op
        to the local admin primitive. The heavy sync ops run on a worker thread through
        the reload gate, identically to the HTTP path; the tool reload/remove ops
        dispatch through the async per-kind reloader; ``list_failed_mcps`` is a plain
        in-process read. An unrecognized op raises loudly.

        After the op applies, ``on_fleet_op_applied`` handlers fire with the op name —
        the op has not fully "applied" until they finish (a celery worker must re-fork
        its prefork pool before it reports applied), and a raising handler fails the op.
        The returned value becomes the op's terminal ``applied`` payload; the publisher
        echo-skips its own broadcast, so this never re-applies a self-op."""
        result = await self._dispatch_bus_op(op)
        await self._run_fleet_op_applied_handlers(_op_field(op, "op"))
        return result

    async def _dispatch_bus_op(self, op: dict[str, Any]) -> Any:
        """Map one fleet op to its local admin primitive and return its result."""
        op_name = op.get("op")
        if op_name == "reload_config":
            return await reload_gate.run(tai_app.admin.reload_config)
        if op_name == "reload_mcp":
            title = _op_field(op, "title")
            return await reload_gate.run(lambda: tai_app.admin.reload_mcp(title))
        if op_name == "deregister_mcp":
            title = _op_field(op, "title")
            return await reload_gate.run(lambda: tai_app.admin.deregister_mcp(title))
        if op_name in ("reload_tool", "remove_tool"):
            action = "reload" if op_name == "reload_tool" else "remove"
            return await tai_app.admin.run_tool_reload(_op_field(op, "kind"), action, _op_field(op, "name"))
        if op_name == "reload_failed_mcps":
            # Return the BARE list — the same shape the publisher's own self-apply
            # (operations/manifest.py) rides on its self entry — so every origin's
            # payload in one FleetResult is uniform (a consumer never special-cases
            # self vs remote for this op).
            return await reload_gate.run(tai_app.admin.reload_failed_mcps)
        if op_name == "list_failed_mcps":
            # Bare list, matching the self-apply shape — see reload_failed_mcps above.
            return tai_app.admin.list_failed_mcps()
        raise ValueError(f"unknown fleet op {op_name!r}")

    async def _run_fleet_op_applied_handlers(self, op_name: str) -> None:
        """Fire every ``on_fleet_op_applied`` handler with the op name. A raising
        handler propagates (fail-loud), turning the op's terminal reply into
        ``failed`` — the post-apply obligation is part of the op applying."""
        for handler in list(self._fleet_op_applied_handlers.values()):
            if inspect.iscoroutinefunction(handler):
                await handler(op_name)
            else:
                handler(op_name)

    @staticmethod
    def _log_task_exception(task: asyncio.Task[Any]) -> bool:
        """Log a lifespan-owned background task's terminal exception at ERROR.

        Cancellation (shutdown) is the normal stop and stays silent;
        any other exception means the task stopped doing its job, so log it at
        ERROR with the task's name. Returns True when the task instead returned a
        value cleanly (no cancellation, no exception), letting a caller treat an
        unexpected clean return as its own failure."""
        if task.cancelled():
            return False
        exc = task.exception()
        if exc is not None:
            logger.error("background task %r terminated with an exception", task.get_name(), exc_info=exc)
            return False
        return True

    @classmethod
    def _on_perpetual_task_done(cls, task: asyncio.Task[Any]) -> None:
        """Done-callback for a run-until-cancelled lifespan-owned task (the
        worker-bus subscription and the failed-MCP re-probe loop).

        A clean cancellation stays silent and a runtime exception is logged at
        ERROR — and, unlike a bounded task, a NORMAL return is ALSO logged at
        ERROR: these tasks are contractually perpetual, so returning means the
        worker silently stopped doing its job (a subscription that returns stops
        receiving sibling reloads; the re-probe loop stops self-healing)."""
        if cls._log_task_exception(task):
            logger.error(
                "perpetual background task %r returned unexpectedly; it must run until cancelled",
                task.get_name(),
            )

    async def _teardown_resources(self) -> None:
        """Release process-wide resources at shutdown.

        Runs on ``app_context`` — the single seam both the served (HTTP / stdio)
        and backend-worker entrypoints cross — so neither path leaks pooled
        clients or the langgraph store/checkpoint pools, nor loses buffered
        monitoring spans. Each step runs independently so one failure cannot skip
        the rest; collected failures are re-raised together so they surface
        loudly rather than being swallowed.
        """
        errors: list[Exception] = []

        async def _guard(label: str, teardown: Callable[[], Any]) -> None:
            try:
                await teardown()
            except Exception as e:
                logger.error("Error during %s teardown: %s", label, e, exc_info=True)
                errors.append(e)

        await _guard("pooled clients", self.clients.shutdown_clients)
        await _guard("checkpoint registry", lambda: checkpoint_registry().close_all())
        await _guard("store registry", lambda: store_registry().close_all())

        # Flush buffered monitoring spans last, so spans emitted during the
        # teardown above are captured before the process exits rather than lost
        # to a SIGTERM / container restart ahead of the SDK's periodic flush.
        try:
            get_monitoring().writer.flush()
        except Exception as e:
            logger.error("Error during monitoring flush: %s", e, exc_info=True)
            errors.append(e)

        if errors:
            raise ExceptionGroup("shutdown teardown failed", errors)

    def start(self, manifest: Manifest):
        # Booting is the composition root: this is where the running impl claims
        # the global ``tai_app`` handle. Constructing a ``TaiMCP`` must not — only
        # start()/app_context binds, so building a throwaway app can't hijack it.
        tai_app.bind(self)
        self._manifest = manifest
        self._failed_mcps = {}
        self._mcp_bound_tools = {}
        # Reset so a dropped agent doesn't linger across update()/reload —
        # importer.py re-fires the @tai_app.agents.agent decorator each start().
        self._agent_binding.reset()

        # Reset so a dropped webhook verifier doesn't linger across update()/reload
        # — the manifest's verifier modules re-run their register() call each
        # start(). Mirrors the agent reset above.
        self._webhook_verifier_registry.reset()

        # Reset so a dropped channel doesn't linger across update()/reload — the
        # manifest's channel modules re-run their register() call each start().
        # Mirrors the webhook-verifier reset above.
        self._channel_registry.reset()

        # Drop the cached resource manager: a reload re-imports the storage
        # module and rebuilds the storage provider, so a stale cache would keep
        # rendering against (and pin open) the previous provider's pool.
        self._resource_manager_cache = None

        # Clear the code-built connector registry before _initialize_components()
        # re-imports the manifest's connector plugin modules, which re-run their
        # module-level register_connector(...) calls; the registry's duplicate
        # guard would otherwise crash every reload. Mirrors the _agents reset.
        reset_registry()

        # Clear the identity-provider registry before _initialize_components()
        # re-imports the manifest's identity-plugin modules, which re-run their
        # module-level register_identity_provider(...) calls; the duplicate guard
        # would otherwise crash every reload. Mirrors the connector reset above.
        # The skeleton ships NO concrete identity provider: a deployment names one
        # (e.g. tai_identity_redis.redis_api_key_provider, the default in the example
        # manifest) in its manifest lifecycle_modules, which _initialize_components
        # imports below — that import-only registration is the sole home, exactly as
        # the shared_secret webhook verifier registers.
        reset_identity_registry()

        # Clear the accounts-provider registry beside the identity reset: an accounts
        # provider registers into BOTH registries on import, so without this mirror an
        # in-process reload would clear identity but not accounts, and the duplicate
        # guard in register_accounts_provider would crash the reload once an accounts
        # plugin is installed.
        reset_accounts_registry()

        # Drop sub-MCP routes + cached sub-apps so a re-init stops serving
        # sub-apps from the previous generation; the reload handlers below
        # re-register the current ones. Their lifespans tear down on the loop
        # that owns them.
        self._mcp_sub_app_router.reset()

        # Clear the live tool/prompt/resource/template surface before
        # _initialize_components() re-imports the manifest modules and re-fires their
        # module-level @tai_app.tools.tool (and any module-registered prompt /
        # resource / resource-template) decorators, so each re-registration lands in
        # a clean surface and never trips on_duplicate="error". Mirrors the
        # agent/webhook/channel/connector/identity resets above: tools are the same
        # shape (a module decorator re-fired every reload), so they get the same
        # reset-before-reimport treatment rather than relying on the CALLER's removal
        # being atomic with the reimport — an interleaving reload whose caller-side
        # removal had not fully cleared the surface would otherwise re-add a
        # still-present tool and crash the reload with "Component already exists".
        self._reset_component_surface()

        # Clear the operation registry before _initialize_components() re-imports
        # the router modules, which re-fire their module-level @operation
        # decorators; without this a reload would trip the duplicate-name guard.
        # Mirrors the agent/webhook/channel resets above.
        operation_registry.clear()

        self._initialize_registries()
        self._initialize_components()

        logger.info("[tools]")
        for t in sorted(self._registry_names_sync()["tool"]):
            logger.info(f"\t. {t}")

        # The pluggable-kind summary: one line per kind's live active/default/off
        # state (a broken registry raises here and fails the boot, never a silent
        # partial table), plus the once-per-process warning when NoOp monitoring is
        # the active recorder — the point where "not configured" becomes
        # distinguishable from "no traffic".
        kind_rows = collect_kind_status()
        logger.info("[kinds]")
        for row in kind_rows:
            suffix = f" ({row.plugin})" if row.plugin else ""
            logger.info(f"\t. {row.kind}: {row.state}{suffix} — {row.detail}")
        warn_if_noop_monitoring(kind_rows, logger)

    def _update(self, manifest: Manifest):
        # Reload-time re-check of the backend-needs-bus invariant, BEFORE registries
        # rebuild: a reload whose new manifest registers a backend while the bus is
        # unconfigured is refused here, catching an env-materialized backend or an
        # out-of-band manifest edit. The k8s rule needs no reload twin — its settings
        # are boot-fixed env.
        require_bus_for_backend(manifest)
        # Capture the live, RE-ADDABLE FastMCP components (tools + prompts +
        # resources) — component objects ``add_tool`` / ``add_prompt`` /
        # ``add_resource`` can restore, NOT the wire types ``list_*`` return. This
        # snapshot is the previous known-good surface the failure path restores and
        # the baseline the list_changed diff compares against (both keyed on the
        # tool/prompt/resource surfaces this snapshot holds); the actual clearing of
        # the old generation happens inside ``start()`` (``_reset_component_surface``),
        # right before the reimport re-registers the current ones.
        before = self._capture_components_sync()
        before_names = {"tool": set(before.tools), "prompt": set(before.prompts), "resource": set(before.resources)}

        try:
            self.start(manifest)
            # start() only registers manifest-module tools; dynamically-loaded
            # tools (DB flows, sub-MCP apps) come back via the reload handlers.
            # raise_on_error: a failed reload handler must fail the op loudly, not
            # leave the worker silently short of tools while reporting success.
            self._run_blocking(lambda: self._run_handlers(list(self._reload_handlers.values()), raise_on_error=True))
        except Exception:
            # Restore ONLY the tool/prompt/resource surface. start() reset the
            # agent/webhook/connector/sub-MCP registries at its top, BEFORE the
            # failing step, so those stay reset — the re-raise is loud and the
            # operator must re-reload. Re-adding the captured components keeps the
            # worker serving the previous known-good tool/prompt/resource set rather
            # than a bricked (empty) surface.
            self._restore_components(before)
            raise

        # Diff-guarded list_changed: emit exactly once per registry whose name
        # set actually changed across the reload, so a no-op reload notifies
        # nothing. Runs AFTER rehydration has re-registered, so it reflects the
        # settled tool/prompt/resource set — never per-register spam mid-reload.
        after = self._registry_names_sync()
        for kind in ("tool", "prompt", "resource"):
            if before_names[kind] != after[kind]:
                self._session_registry.schedule_list_changed(kind)

    def _restore_components(self, captured: _CapturedComponents) -> None:
        """Restore the tool/prompt/resource surface to ``captured`` after a failed
        ``_update``.

        A failed ``start()`` can leave a PARTIAL new surface (some modules imported
        before the failure), so the current tools/prompts/resources are cleared
        first, then the captured previous-generation components are re-added. This
        restores only the FastMCP-served surface — NOT a full known-good generation:
        the base ``_tool_registry`` reflects the attempted (failed) manifest, which
        is the acknowledged cost of the loud re-raise."""
        current = self._capture_components_sync()
        for name in current.tools:
            self._fast_mcp.local_provider.remove_tool(name)
        for name in current.prompts:
            self._fast_mcp.local_provider.remove_prompt(name)
        for uri in current.resources:
            self._fast_mcp.local_provider.remove_resource(uri)
        for tool in captured.tools.values():
            self._fast_mcp.add_tool(tool)
        for prompt in captured.prompts.values():
            self._fast_mcp.add_prompt(prompt)
        for resource in captured.resources.values():
            self._fast_mcp.add_resource(resource)

    def _initialize_registries(self):
        if self._manifest is None:
            raise RuntimeError("TaiMCP is not started — call start()/app_context first.")
        self._tool_registry = ToolRegistry(
            requested_tools=self._manifest.tools_list,
            tool_extensions=self._manifest.tool_extensions,
        )
        self._extension_registry = ExtensionRegistry(self._tool_registry.used_extensions)

    def _initialize_components(self):
        if self._manifest is None:
            raise RuntimeError("TaiMCP is not started — call start()/app_context first.")

        # Repopulate the operation registry that start() cleared at its top. A leaf
        # module declares its operations with @operation at IMPORT, and that
        # decorator fires exactly once per interpreter; a plain re-import of a router
        # that only ``from operations.<domain> import <op>`` never re-registers a
        # leaf that stayed cached in sys.modules, so without this the projection
        # below would find an empty registry and project nothing. The first call
        # re-imports the leaves to fire each @operation into the registry (never the
        # operations package or its infra, so the registry singleton the projection
        # and authz hold is preserved) and snapshots the records; a reload replays that
        # stable in-memory snapshot instead of re-importing, so the reload adds no
        # sys.modules churn on the loop-affine path. Runs BEFORE the routers below so
        # each route re-attaches its template + method to the metadata record already IN
        # the registry (the record the projection reads and the tool-edge authorization
        # synthesizes its concrete path from).
        #
        # Imported here rather than at module scope: the app import chain pulls the
        # operations package in, so a module-level import of it would be circular.
        from tai_skeleton.operations import reregister_operations

        reregister_operations()

        for module in self._manifest.lifecycle_modules or []:
            import_or_reload_package(module)

        # Import-only verifier plugins: each import runs the module's
        # ``tai_app.webhook_verifiers.register(...)`` side-effect. Imported like
        # the lifecycle modules (loud on failure) — the registry was reset at the
        # top of start(), so every (re)load re-registers cleanly.
        for module in self._manifest.webhook_verifier_modules or []:
            import_or_reload_package(module)

        # Import-only channel plugins: each import runs the module's
        # ``tai_app.channels.register(...)`` side-effect and binds the plugin's
        # inbound HTTP route. Imported like the verifier modules (loud on
        # failure) — the registry was reset at the top of start(), so every
        # (re)load re-registers cleanly.
        for module in self._manifest.channel_modules or []:
            import_or_reload_package(module)

        for module in self._manifest.routers_modules or []:
            import_or_reload_package(module)

        for module in self._manifest.middlewares_modules or []:
            import_or_reload_package(module)

        import_or_reload_package(self._manifest.backend_module)
        import_or_reload_package(self._manifest.storage_module)
        import_or_reload_package(self._manifest.monitoring_module)

        for extension in self._manifest.extensions_modules or []:
            import_or_reload_package(extension)
        self._extension_registry.validation()

        for cfg in self._manifest.tools:
            import_or_reload_package(cfg.module)

        # Importing an agents-module fires its @tai_app.agents.agent decorator, which
        # registers the agent + auto-generates its run tool. Done after tools so
        # an agent tool can reference a base tool already loaded above.
        for cfg in self._manifest.agents:
            import_or_reload_package(cfg.module)

        if self._manifest.mcp:
            successes, failures = self._run_blocking(self._load_mcps)

            for cfg, tools in successes:
                self._mcp_tools(cfg, tools)

            for cfg, kind in failures:
                self._record_failed_mcp(cfg, kind)

        # Project the operation surface into MCP tools. Runs AFTER the router
        # modules registered their operations and AFTER base tools/MCPs bound (so
        # extension combos over a projected op resolve at bind time), and BEFORE
        # validation and the preset-rehydration reload handlers re-bake — the
        # pinned order registries/routes -> operations -> projection -> extension
        # wraps -> preset rebakes.
        project_operations(self, self._manifest.api_tools)

        self._tool_registry.validation(ignore=self._missing_tools_ignore())

    def _registry_names_sync(self) -> dict[str, set[str]]:
        """Snapshot the live server's tool / prompt / resource names off-loop —
        used by the reload path (which runs outside any event loop) and by
        ``start()``'s tool log. Keyed by the SINGULAR kind so the reload diff can
        drive ``schedule_list_changed`` per changed registry. Runs through the one
        off-loop ``_run_blocking`` runner, safe from a loop-less caller and from
        inside the server loop."""

        async def snapshot() -> dict[str, set[str]]:
            return {
                "tool": {t.name for t in await self._fast_mcp.list_tools()},
                "prompt": {p.name for p in await self._fast_mcp.list_prompts()},
                "resource": {str(r.uri) for r in await self._fast_mcp.list_resources()},
            }

        return self._run_blocking(snapshot)

    def _reset_component_surface(self) -> None:
        """Clear every tool / prompt / resource / resource template off the live
        ``local_provider``.

        Called at the top of ``start()`` (before ``_initialize_components``
        re-imports the manifest modules) so a module-level ``@tai_app.tools.tool``
        — or any module-registered prompt / resource / resource-template — decorator
        that re-fires on the reimport always adds into a clean surface, never
        tripping ``on_duplicate="error"``.

        Enumerates the provider's OWN stored components, deliberately NOT the server
        ``list_*`` views: those filter (enabled / visibility / auth) and synthesize
        (prefab renderer resources computed on demand, resident on no provider), so
        a filtered-out component would survive the reset and re-collide on the
        re-fire, and a synthetic resource URI has nothing to remove and would raise.
        The raw provider surface is exactly the set the re-fired decorators
        (re-)populate. ``ResourceTemplate`` is a distinct component kind (not a
        ``Resource`` subclass) under its own key namespace, so it is cleared on its
        own branch. Names/URIs are de-duplicated so a versioned/unversioned mix
        cannot double-remove one name (``remove_*`` clears all versions by name in
        one call). Synchronous — a plain dict read and sync ``remove_*`` calls,
        needing no event loop, so it runs inline wherever ``start()`` runs (the
        serving loop at cold boot, a worker thread on reload)."""
        provider = self._fast_mcp.local_provider
        components = list(provider._components.values())
        for name in {c.name for c in components if isinstance(c, Tool)}:
            provider.remove_tool(name)
        for name in {c.name for c in components if isinstance(c, Prompt)}:
            provider.remove_prompt(name)
        # ResourceTemplate is a distinct component kind, NOT a Resource subclass, so
        # the Resource branch never sweeps it — it needs its own removal branch (the
        # four kinds are disjoint, so the branch order is immaterial).
        for uri_template in {c.uri_template for c in components if isinstance(c, ResourceTemplate)}:
            provider.remove_template(uri_template)
        for uri in {str(c.uri) for c in components if isinstance(c, Resource)}:
            provider.remove_resource(uri)

    def _capture_components_sync(self) -> _CapturedComponents:
        """Snapshot the live, RE-ADDABLE FastMCP components off-loop, keyed by name
        (tools/prompts) or URI (resources).

        ``list_*`` returns wire types (``mcp.types.Tool``/``Prompt``/``Resource``)
        that ``add_*`` cannot re-add, so each name is re-fetched via
        ``get_tool`` / ``get_prompt`` / ``get_resource`` — the FastMCP component
        objects the reload restore path re-adds. A name that vanished between the
        ``list_*`` snapshot and the ``get_*`` re-fetch (``get_*`` returns ``None``)
        is skipped, so the captured surface never stores a ``None``. Runs through
        the one off-loop ``_run_blocking`` runner."""

        async def snapshot() -> _CapturedComponents:
            captured = _CapturedComponents()
            for tool in await self._fast_mcp.list_tools():
                got = await self._fast_mcp.get_tool(tool.name)
                if got is not None:
                    captured.tools[tool.name] = got
            for prompt in await self._fast_mcp.list_prompts():
                got = await self._fast_mcp.get_prompt(prompt.name)
                if got is not None:
                    captured.prompts[prompt.name] = got
            for resource in await self._fast_mcp.list_resources():
                uri = str(resource.uri)
                got = await self._fast_mcp.get_resource(uri)
                if got is not None:
                    captured.resources[uri] = got
            return captured

        return self._run_blocking(snapshot)

    async def _probe_mcp(self, config: TaiMCPConfig) -> list["mcp.types.Tool"]:
        """Connect to one MCP server and list its tools, bounded by
        ``mcp_probe_timeout``. Raises on failure/timeout; callers decide
        whether to skip-and-record or surface the error. The probe runs through
        the pooled ``FastMCPClient`` (one-shot, off-pool) so no raw fastmcp
        ``Client`` is opened by the app."""

        async def _do() -> list["mcp.types.Tool"]:
            async with self.clients.client_ctx(FastMCPClient, fresh=True, config=config.model_dump()) as client:
                return await client.list_tools()

        return await asyncio.wait_for(_do(), timeout=mcp_probe_timeout())

    async def _load_mcps(
        self,
    ) -> tuple[list[tuple[TaiMCPConfig, Any]], list[tuple[TaiMCPConfig, str]]]:
        """Probe every manifest MCP server concurrently, each isolated.

        Returns ``(successes, failures)`` and never raises for one server, so a
        dead/slow MCP can't abort startup. Driven off-loop through
        ``_run_blocking`` so a sync (Celery/RQ) caller and the serving loop both
        reach it safely.
        """
        manifest = self._manifest
        if manifest is None or not manifest.mcp:
            return [], []

        async def run_one(config: TaiMCPConfig):
            try:
                tools = await self._probe_mcp(config)
                return config, tools, None
            except Exception as e:
                return config, None, type(e).__name__

        results = await asyncio.gather(*(run_one(cfg) for cfg in manifest.mcp))
        successes, failures = [], []
        for config, tools, kind in results:
            if kind is None:
                successes.append((config, tools))
            else:
                failures.append((config, kind))
        return successes, failures

    def _record_failed_mcp(self, config: TaiMCPConfig, kind: str) -> None:
        """Record a failed MCP as ``unavailable`` and log it.

        Stores only the title + coarse status, never the exception text or
        config — ``list_failed_mcps`` is LLM-callable and the config carries
        credentials. Only ``kind`` (exception class name) reaches the log.
        """
        self._failed_mcps[config.title] = "unavailable"
        logger.error(
            "MCP server '%s' unavailable — skipped, recorded for reload (%s)",
            config.title,
            kind,
        )

    def _missing_tools_ignore(self) -> frozenset[str]:
        """Tool names the failed MCP servers were to provide — legitimately
        absent (server down), so ``tools.validation`` must not raise for them.

        Matched by base name (``:``-extension stripped), so validation is
        slightly under-strict for a missing tool sharing a base name with a
        failed MCP's tool — collisions are unlikely and not crashing wins.
        """
        ignore: set[str] = set()
        title_map = getattr(self._manifest, "include_title_mcp_tools_map", {}) or {}
        for title in self._failed_mcps:
            ignore |= set(title_map.get(title, set()))
        return frozenset(ignore)

    # --- targeted single-MCP reload (boot recovery + agent tools) ---

    def _list_failed_mcps(self) -> list[dict[str, str]]:
        """MCP servers skipped due to a failed viability check: ``title`` +
        coarse ``status`` only. No config, no exception text — this is
        LLM-callable, logged and broadcast, and the config carries
        credentials. Per-process: in a multi-worker backend this reflects only
        the current process.

        Reads race a reload worker thread mutating ``_failed_mcps`` (this read is
        deliberately not reload-gated, so status keeps answering mid-reload), so
        the dict is snapshot-copied — a single C-level op, atomic under the GIL —
        before iterating."""
        return [{"title": title, "status": status} for title, status in dict(self._failed_mcps).items()]

    def _require_live_manifest(self) -> Manifest:
        """The live in-process manifest, or a loud error if the app is not started
        — the typed accessor behind ``app.admin.live_manifest_typed``."""
        manifest = self._manifest
        if manifest is None:
            raise RuntimeError("TaiMCP is not started — call start()/app_context first.")
        return manifest

    def _refresh_manifest_mcp(self) -> None:
        """Graft the manifest's current MCP rows into the boot-time snapshot so
        a row written after boot is reloadable / a removed one deregisterable.
        When no external manifest is readable (embedded/test runtimes) the
        in-memory copy stays authoritative."""
        try:
            fresh = Manifest.model_validate(self._config_manager.read_manifest())
        except FileNotFoundError:
            logger.warning(
                "no external manifest to re-read; using the in-memory MCP rows",
                exc_info=True,
            )
            return
        if self._manifest is None:
            raise RuntimeError("TaiMCP is not started — call start()/app_context first.")
        self._manifest.replace_mcp(fresh.mcp)

    async def _reload_mcp_async(self, title: str) -> dict[str, Any]:
        self._refresh_manifest_mcp()
        manifest = self._manifest
        mcp_map = (manifest.mcp_map if manifest else {}) or {}
        if title not in mcp_map:
            return {
                "title": title,
                "status": "error",
                "error": f"Unknown MCP '{title}' — not present in the current manifest.",
            }

        config = mcp_map[title]
        try:
            tools = await self._probe_mcp(config)
        except Exception as e:
            self._record_failed_mcp(config, type(e).__name__)
            return {"title": title, "status": "unavailable"}

        return await self._apply_reloaded_mcp(title, config, tools)

    async def _apply_reloaded_mcp(self, title: str, config: TaiMCPConfig, tools: list[Any]) -> dict[str, Any]:
        """Bind a freshly-probed MCP server's tools and reconcile dependent presets —
        the registry-mutating half of a reload, split from the probe.

        This half synchronously rewrites the process-wide tool registry and then
        marshals the preset reconcile onto the serving loop, so two servers applying
        at once would race each other's registry mutation across the probe and serving
        threads. ``_reload_failed_mcps_async`` therefore probes servers concurrently
        but calls this ONE server at a time; the single-server reload calls it once.
        """
        # Clean reload: drop any tools this MCP previously bound, then rebind.
        old_bound = set(self._mcp_bound_tools.get(title, set()))
        for name in sorted(old_bound):
            try:
                self._fast_mcp.local_provider.remove_tool(name)
            except Exception:
                logger.warning("reload_mcp: could not remove stale tool %s", name, exc_info=True)

        self._mcp_tools(config, tools)
        new_bound = set(self._mcp_bound_tools.get(title, set()))

        # Symmetry with _deregister_mcp: a tool this MCP no longer serves must ALSO
        # leave the base registry, or a re-probed server drops T from the wire while
        # T lingers in _requested_tools/_tools (its self-entry keeps it out of
        # missing_tools) — a registry-derived surface stays stale until a full
        # reload.
        for name in sorted(old_bound - new_bound):
            self._tool_registry.unregister_tool_base(name)

        await self._reconcile_after_mcp_reload(title, old_bound, new_bound)

        self._failed_mcps.pop(title, None)
        result: dict[str, Any] = {
            "title": title,
            "status": "ok",
            "tools": sorted(new_bound),
        }
        # A name the (re)bind refused because a registered preset owns it — surfaced
        # loudly so the caller sees the returning server did NOT clobber the preset.
        conflicts = sorted(self._mcp_preset_conflicts.get(title, set()))
        if conflicts:
            result["preset_conflicts"] = conflicts
        return result

    async def _reconcile_after_mcp_reload(self, title: str, old_bound: set[str], new_bound: set[str]) -> None:
        """Reconcile dependent presets after a targeted MCP reload rebinds
        ``title``'s tools and the vanished-tool unregistration has settled the base
        registry.

        A preset over a base this MCP rebound is a ``TransformedTool`` whose closure
        still holds the OLD wrapper/config, so it is re-registered from its in-memory
        spec to track the freshly-bound base; a preset whose base vanished across the
        reload is quarantined (its store row surfaces as ``conflicted``).
        ``old_bound`` / ``new_bound`` are the tool names this MCP bound before and
        after the rebind,
        so their union is exactly the set of bases whose bindings changed."""
        await self._reconcile_bases_on_serving_loop(old_bound | new_bound)

    async def _reconcile_bases_on_serving_loop(self, affected_bases: set[str]) -> None:
        """Reconcile base-dependent presets with the ``PresetManager`` per-name locks
        taken on the serving loop.

        Those locks are ``asyncio.Lock``s, valid on a single loop only. The preset
        mutation routes take them on the serving loop, so every reconcile takes them
        there too: a route mutating a preset and a reconcile touching the same name
        contend on ONE loop, never two (a cross-loop contended acquire raises, and a
        lock touched from two threads gives no mutual exclusion). The reprobe pass
        already runs this ON the serving loop and awaits directly; an admin
        reload/deregister runs its body on a ``_run_blocking`` worker loop and
        marshals the coroutine back onto the serving loop, awaiting the cross-loop
        result without blocking the worker loop. With no serving loop bound (a
        pure-sync boot) nothing else contends, so the reconcile runs on the current
        loop.
        """
        loop = self._serving_loop
        if loop is None or asyncio.get_running_loop() is loop:
            await self.preset_manager.reconcile_bases(affected_bases)
            return
        future = asyncio.run_coroutine_threadsafe(self.preset_manager.reconcile_bases(affected_bases), loop)
        await asyncio.wrap_future(future)

    async def _reload_failed_mcps_async(self) -> list[dict[str, Any]]:
        """Re-probe every currently-failed MCP concurrently, then apply the binds
        ONE server at a time.

        Probing is network-bound and mutates no shared state, so all servers are
        probed at once — N down servers cost ~one probe timeout, not N. Applying a
        probe result rewrites the process-wide tool/preset registry and marshals the
        reconcile onto the serving loop, so applications are serialized: running two
        at once would let one server's bind race another's reconcile across the probe
        and serving threads. Titles are snapshotted before any mutation.
        """
        self._refresh_manifest_mcp()
        manifest = self._manifest
        mcp_map = (manifest.mcp_map if manifest else {}) or {}
        titles = list(self._failed_mcps.keys())
        known = [t for t in titles if t in mcp_map]
        unknown = [t for t in titles if t not in mcp_map]

        probes = await asyncio.gather(*(self._probe_mcp(mcp_map[t]) for t in known), return_exceptions=True)

        out: list[dict[str, Any]] = []
        for title, probe in zip(known, probes, strict=True):
            config = mcp_map[title]
            if isinstance(probe, BaseException):
                self._record_failed_mcp(config, type(probe).__name__)
                out.append({"title": title, "status": "unavailable"})
                continue
            try:
                out.append(await self._apply_reloaded_mcp(title, config, probe))
            except Exception:
                # A post-probe bind/reconcile failure must not lose the other titles'
                # results — log the trace loudly, then surface this one coarsely.
                logger.error("reload_failed_mcps: applying reloaded MCP %r failed after probe", title, exc_info=True)
                out.append({"title": title, "status": "error"})
        for title in unknown:
            out.append(
                {
                    "title": title,
                    "status": "error",
                    "error": f"Unknown MCP '{title}' — not present in the current manifest.",
                }
            )
        return out

    @staticmethod
    def _run_blocking(coro_factory: Callable[[], Any]) -> Any:
        """Run a coroutine to completion regardless of the caller's context.

        The single off-loop runner: a private event loop on a worker thread, safe
        from a loop-less caller and from inside the server loop. Every off-loop
        snapshot / probe / reload-handler run goes through here.
        """

        async def _run_and_cleanup() -> Any:
            try:
                return await coro_factory()
            finally:
                # asyncio.run tears down this throwaway loop without closing the
                # per-loop pooled clients the coroutine opened (reload handlers
                # open pooled clients via app.clients.client_ctx), so close them
                # here before teardown. A cleanup failure is logged loudly but
                # must not replace the coroutine's result or its exception.
                try:
                    await shutdown_all_clients()
                except Exception:
                    logger.exception("Error closing pooled clients after _run_blocking")

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(_run_and_cleanup())).result()

    def _raise_if_on_serving_loop(self, op: str) -> None:
        """Refuse a reconcile-driving admin call issued from a coroutine already on
        the serving loop.

        ``reload_mcp`` / ``reload_failed_mcps`` / ``deregister_mcp`` run their async
        body through ``_run_blocking`` and marshal the preset reconcile back onto the
        serving loop. Called from the serving loop itself, ``_run_blocking`` would
        freeze that loop on its blocking wait, so the marshaled reconcile could never
        run — a silent deadlock. Raise loudly instead. The supported callers —
        ``reload_gate.run``'s worker thread and a loop-less sync caller — have no
        running loop here (or a different one) and pass through.
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            return
        if running is self._serving_loop:
            raise RuntimeError(
                f"{op} must not be called from the serving loop; drive it through "
                "reload_gate.run(...) or an off-loop/sync caller"
            )

    def _reload_mcp(self, title: str) -> dict[str, Any]:
        """Re-probe one MCP server by title and, if viable, (re)attach its tools.
        Synchronous and context-agnostic, mirroring ``update``.

        On failure the server is re-recorded in ``list_failed_mcps`` and its
        existing tools are left intact (transient blips self-heal). Only a
        successful reload replaces tools; a mid-rebind failure can leave the set
        partially updated — rerun after fixing the config.
        """
        self._raise_if_on_serving_loop("reload_mcp")
        return self._run_blocking(lambda: self._reload_mcp_async(title))

    def _reload_failed_mcps(self) -> list[dict[str, Any]]:
        """Re-probe every MCP currently in the failed list; attach the ones
        that are now viable. Synchronous and context-agnostic."""
        self._raise_if_on_serving_loop("reload_failed_mcps")
        return self._run_blocking(self._reload_failed_mcps_async)

    # --- lifespan-owned failed-MCP backoff re-probe ---

    def _spawn_reprobe_task(self) -> None:
        """Start the failed-MCP re-probe loop on the serving loop. Owned by
        ``app_context``; runs until cancelled at shutdown."""
        self._reprobe_task = asyncio.create_task(
            self._reprobe_failed_mcps_loop(),
            name="tai-failed-mcp-reprobe",
        )
        # Backstop a silent death OR an unexpected normal return loudly, mirroring
        # the worker-bus subscription: both are run-until-cancelled tasks.
        self._reprobe_task.add_done_callback(self._on_perpetual_task_done)

    async def _cancel_reprobe_task(self) -> None:
        """Cancel the re-probe task and await its termination — the shutdown
        counterpart of ``_spawn_reprobe_task``.

        A task that died with a non-``CancelledError`` exception was already
        surfaced at ERROR by its done-callback, so it is awaited-and-swallowed
        here rather than re-raised — this runs inside ``app_context``'s shutdown
        ``finally``, and re-raising would skip the remaining teardown."""
        task = self._reprobe_task
        self._reprobe_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Already logged at ERROR by the done-callback; swallowed so a dead
            # re-probe task cannot abort the remaining shutdown steps.
            pass

    async def _reprobe_sleep(self, seconds: float) -> None:
        """The re-probe loop's inter-pass sleep, isolated so tests can drive the
        backoff with a controllable clock instead of real time."""
        await asyncio.sleep(seconds)

    async def _reprobe_failed_mcps_loop(self) -> None:
        """Re-probe failed-at-boot MCP servers on an exponential backoff.

        Each pass sleeps the current interval, then — only when a server is
        currently failed — holds the reload gate and drives the existing
        ``_reload_failed_mcps_async`` path (probe → drop stale → rebind recovered
        tools → clear from the failed set), logging the outcome at INFO. The
        interval starts at ``mcp_reprobe_initial_seconds``, doubles (capped at
        ``mcp_reprobe_max_seconds``) after a pass where every probed server stayed
        down, and resets to the initial value the moment any server recovers or a
        new one appears in the failed set. An empty failed set probes nothing.

        A cancellation (shutdown) propagates for a clean exit; any other per-pass
        error is logged loudly and the loop survives to the next pass — a silently
        dead recovery task is the exact failure mode this task removes.
        """
        interval = CoreSettings().mcp_reprobe_initial_seconds
        # Titles known-failed as of the previous pass; a title present now but
        # absent here is a fresh failure and resets the backoff to probe promptly.
        # Seeded under the reload-gate lock on the first pass (below) so the
        # snapshot never races a worker-thread reload mutating the failed set.
        known_failed: set[str] | None = None
        while True:
            await self._reprobe_sleep(interval)
            try:
                settings = CoreSettings()
                initial = settings.mcp_reprobe_initial_seconds
                # Every read/write of _failed_mcps happens under the reload-gate
                # lock: a worker-thread reload holds the same lock, so this is the
                # only place the failed set is touched concurrency-safely (a bare
                # snapshot would race a concurrent mutation → dict-changed-size).
                async with reload_gate.lock:
                    current_failed = set(self._failed_mcps)
                    if known_failed is None:
                        known_failed = current_failed
                    if not current_failed:
                        interval = initial
                        known_failed = set()
                        continue
                    newly_appeared = not current_failed <= known_failed
                    results = await self._reload_failed_mcps_async()
                    still_failed = set(self._failed_mcps)
                recovered = sorted(r["title"] for r in results if r.get("status") == "ok")
                logger.info(
                    "failed-MCP re-probe pass: probed=%s recovered=%s still_failed=%s",
                    sorted(current_failed),
                    recovered,
                    sorted(still_failed),
                )
                if recovered or newly_appeared:
                    interval = initial
                else:
                    interval = min(interval * 2, settings.mcp_reprobe_max_seconds)
                known_failed = still_failed
            except Exception:
                # ``CancelledError`` is a BaseException and passes this handler for
                # a clean shutdown cancel; every other error is logged and the loop
                # continues at the current backoff.
                logger.error("failed-MCP re-probe pass failed; retrying next interval", exc_info=True)

    def _deregister_mcp(self, title: str) -> dict[str, Any]:
        """Detach one MCP server's tools — the removal counterpart of
        ``reload_mcp``. Idempotent: a process that never bound the title
        reports ``absent``, not an error."""
        self._raise_if_on_serving_loop("deregister_mcp")
        self._refresh_manifest_mcp()
        bound = sorted(self._mcp_bound_tools.pop(title, set()))
        failed = self._failed_mcps.pop(title, None) is not None
        if not bound and not failed:
            return {"title": title, "status": "absent"}
        for name in bound:
            try:
                self._fast_mcp.local_provider.remove_tool(name)
            except Exception:
                logger.warning("deregister_mcp: could not remove tool %s", name, exc_info=True)
            self._tool_registry.unregister_tool_base(name)
        # Reconcile presets that depended on the just-removed bases: a dependent
        # preset is quarantined (its store row surfaces as ``conflicted``) — no
        # dependent preset is left bound to a base that no longer exists.
        # This method is synchronous (context-agnostic, mirroring ``update``), so the
        # async reconciliation runs through the off-loop blocking runner, which
        # marshals the ``PresetManager`` locks onto the serving loop.
        if bound:
            self._run_blocking(lambda: self._reconcile_bases_on_serving_loop(set(bound)))
        return {"title": title, "status": "ok", "removed": bound}

    def _reload_config(self) -> dict[str, Any]:
        """Soft restart: refresh env into ``os.environ``, reset settings
        caches, then re-init from the manifest so rebuilt clients read the
        fresh config. Heavy but in-process — no pod restart."""
        env = self._config_manager.read_env()
        # Reconcile removed keys: a key present in the previously applied env but
        # absent now must be dropped from os.environ, or the stale value would
        # linger. Only keys THIS method applied are removed — never unrelated
        # process env.
        for key in self._loaded_env_keys - set(env):
            os.environ.pop(key, None)
        os.environ.update(env)
        self._loaded_env_keys = set(env)
        # Release the loop-bound langgraph checkpoint/store pools before resetting
        # settings. ``reset_all_settings`` drops the per-loop registries so they
        # rebuild against the fresh config, but it refuses to drop a registry that
        # still holds live resources on a running loop (dropping it would leak the
        # open pools). An agent run on this worker leaves live checkpoint/store
        # resources behind, so close them first — the same release the shutdown
        # teardown performs, marshalled onto the serving loop the pools are bound
        # to since this body runs on a reload worker thread.
        self._close_llm_registries_on_serving_loop()
        reset_all_settings()
        manifest = Manifest.model_validate(self._config_manager.read_manifest())
        self._update(manifest)
        return {"status": "ok", "env_keys": len(env)}

    async def _close_llm_registries(self) -> None:
        """Close the langgraph checkpoint + store resource pools — the release a
        settings reset requires before it can drop the per-loop registries."""
        await checkpoint_registry().close_all()
        await store_registry().close_all()

    def _close_llm_registries_on_serving_loop(self) -> None:
        """Close the loop-bound checkpoint/store pools on the serving loop.

        The pools bind to the serving loop that opened them (an agent run), so they
        must be closed there. A reload body runs on a ``reload_gate`` /
        ``_run_blocking`` worker thread (no running loop of its own), so the close is
        marshalled onto the serving loop and awaited. Two paths skip the marshal:
        with no serving loop bound (a pure-sync context) nothing holds these pools on
        a running loop; and a reload driven synchronously ON the serving loop itself
        (a pure-sync / test caller) cannot block on a coroutine only that loop can
        run — such a caller holds no agent-run resources, so the following settings
        reset drops the empty registries cleanly (and raises loudly if that ever does
        not hold, rather than hiding it)."""
        loop = self._serving_loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            return
        asyncio.run_coroutine_threadsafe(self._close_llm_registries(), loop).result()

    def _live_mcp_status(self) -> dict[str, Any]:
        """Snapshot the in-process MCP-binding state.

        Returns ``{"bound": {title: [tool, ...]}, "failed": [{title, status}]}``
        (consumed by ``GET /api/mcp_status``).

        Reads race a reload worker thread mutating ``_mcp_bound_tools`` (this
        read is deliberately not reload-gated, so status keeps answering
        mid-reload), so the dict and each per-title tool set are snapshot-copied
        — single C-level ops, atomic under the GIL — before iterating.
        """
        return {
            "bound": {title: sorted(set(tools)) for title, tools in dict(self._mcp_bound_tools).items()},
            "failed": self._list_failed_mcps(),
        }

    @abstractmethod
    def _mcp_tools(self, config: TaiMCPConfig, tools):
        raise NotImplementedError

    # Members the concrete subclass (``TaiMCP``) supplies; declared here so this
    # mixin's methods can reference them with a known type.
    clients: "ClientsFacet"
    preset_manager: "PresetManager"
    _config_manager: "ConfigManager"
    _agent_binding: "AgentBinding"
    _mcp_sub_app_router: "SubMcpAppRouter"
    _resource_manager_cache: "ResourceManager | None"
    _webhook_verifier_registry: "WebhookVerifierRegistry"
    _channel_registry: "ChannelRegistry"
    _session_registry: "SessionRegistry"
