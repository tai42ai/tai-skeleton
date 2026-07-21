import logging
import os
from contextlib import asynccontextmanager

from tai42_kit.logging import logging_settings, setup_logging

from tai42_skeleton.access_control.adapter import AuthAdapter
from tai42_skeleton.access_control.settings import access_control_settings
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
from tai42_skeleton.connectors.meta_log_redactor import install_meta_log_redactor
from tai42_skeleton.connectors.providers.registry import list_providers
from tai42_skeleton.connectors.store.catalog_store import refresh_catalog
from tai42_skeleton.plugins.registry import rebuild_studio_plugin_registry
from tai42_skeleton.versioning import versioned_store_configured

logger = logging.getLogger(__name__)

_app: TaiMCP | None = None

# Env prefixes that carry connector configuration; a non-empty value under
# either means the operator wired up the connector engine.
_CONNECTOR_ENV_PREFIXES = ("CONNECTORS_", "CONNECTOR_STORE_")


def connectors_in_use() -> bool:
    """Whether this deployment uses connectors at all.

    True when the loaded manifest carries a connector-managed MCP entry, a
    manifest-named plugin module registered a provider, or any non-empty
    ``CONNECTORS_*`` / ``CONNECTOR_STORE_*`` env var is set. Runs inside a
    startup/reload handler, i.e. after ``start()`` bound the manifest and
    imported its plugin modules.
    """
    manifest = build_app()._manifest
    if manifest is not None and any(entry.managed is not None for entry in manifest.mcp):
        return True
    if list_providers():
        return True
    return any(key.startswith(_CONNECTOR_ENV_PREFIXES) and value for key, value in os.environ.items())


async def refresh_catalog_if_connectors_in_use() -> None:
    """Startup/reload handler: load the connector catalog only when connectors
    are in play. A deployment with no managed manifest entries, no registered
    providers, and no connector env config has nothing the catalog could serve,
    and refreshing it would only stall boot retrying an absent Postgres. When
    connectors ARE in use the refresh runs and an unreachable Postgres fails it
    loudly; a skipped refresh still fails loudly at first real connector use via
    the ``get_provider`` miss or the store's connection error.
    """
    if not connectors_in_use():
        logger.info(
            "connectors: skipping catalog refresh — no managed manifest entries, no "
            "registered providers, and no CONNECTORS_*/CONNECTOR_STORE_* env configuration"
        )
        return
    await refresh_catalog()


def versioned_store_in_use() -> bool:
    """Whether this deployment uses the versioned-document store at all.

    Thin alias for :func:`tai42_skeleton.versioning.versioned_store_configured` (the
    single source of truth) so the boot/reload hook skips the Postgres open when no
    ``VERSIONING_STORE_*`` env is set, mirroring the connector-catalog gate.
    """
    return versioned_store_configured()


async def rehydrate_versioned_presets_if_store_in_use() -> None:
    """Startup/reload handler: re-register every versioned preset from the store,
    only when the versioned-document store is configured.

    ``reload_config()`` (and a cold boot) wipes the runtime tool registry, so this
    reloads the persisted presets from their active bodies — clearing the
    ``PresetManager`` spec map + quarantine set wholesale first, then re-registering
    every preset from its store row.
    A stale preset (foreign name / missing or preset-owned base) is QUARANTINED
    and surfaced as ``conflicted`` rather than raising, so one bad name never
    bricks boot; a genuinely unreachable Postgres still fails the boot/op loudly
    under the handler's ``raise_on_error``. The load is skipped when connectors'
    sibling store is unconfigured (see :func:`versioned_store_in_use`), so a
    deployment with no versioned documents never opens a Postgres connection at
    boot.
    """
    if not versioned_store_in_use():
        logger.info("presets: skipping versioned-preset rehydration — no VERSIONING_STORE_* env configuration")
        return
    await build_app().preset_manager.rehydrate()


async def rehydrate_sub_mcp_apps() -> None:
    """Startup/reload handler: re-materialize every persisted sub-MCP app from the
    shared store into THIS worker's in-process router.

    ``reset()`` (run on every ``start()``/``reload_config()``) wipes the router's
    per-worker route cache, so this reloads the durable registrations — the store is
    the source of truth and the router is its cache. Each route is bound through the
    router directly (NOT the write service): rehydration must not re-write the store
    it is reading. The store's route list is snapshotted before iterating so the
    in-memory store can never mutate mid-iteration. No env gate (unlike the presets
    handler): the in-memory store needs no connection, and in Redis mode an
    unreachable Redis fails the boot/op loudly under the handler's ``raise_on_error``.
    """
    from tai42_skeleton.sub_mcp.store import get_sub_mcp_store

    router = build_app().sub_app.mcp_sub_app_router
    routes = await get_sub_mcp_store().list_routes()
    for slug, config in list(routes.items()):
        await router.register_sub_mcp_app(slug, config.tools, config.transport)


async def _apply_preset_tool_reload(action: str, name: str) -> None:
    """Worker-bus dispatch target for the ``"preset"`` tool kind: rebind or tear
    down one preset on THIS worker.

    ``reload`` re-reads the preset's ACTIVE store body and rebinds it — a missing
    store row raises :class:`~tai42_contract.presets.errors.PresetNotFoundError`, so a
    bogus fan-out fails loudly, and a sibling name conflict propagating out of
    register surfaces as a real fleet inconsistency.

    ``remove`` mirrors the delete route's own conflicted/non-conflicted split so the
    fan-out never touches a foreign tool: a QUARANTINED name was never registered as
    a preset (its name may be owned by a foreign tool), so only its quarantine entry
    is dropped — the foreign tool stays bound; any other name is a live preset whose
    registration is torn down (idempotent for an absent name). Registered in
    :func:`build_app` and invoked through ``TaiMCP._run_tool_reload``, which rejects
    any other action."""
    manager = build_app().preset_manager
    if action == "reload":
        await manager.reload(name)
    elif manager.is_quarantined(name):
        manager.drop_quarantine(name)
    else:
        await manager.remove(name)
        manager.drop_quarantine(name)


def apply_logging_settings() -> None:
    """Reload handler: re-apply the root logger config from the fresh settings.

    A config reload runs ``reset_all_settings()`` before the reload handlers, so
    ``logging_settings()`` re-reads the current ``TAI_LOG_LEVEL`` and
    ``setup_logging``'s ``force=True`` swaps the root handler config to the new
    level without a process restart. A sync handler — ``_run_handlers`` supports
    those. Registered as a reload handler only from the CLI entrypoints (via
    :func:`register_cli_logging_reload`), which own their process's root logger; an
    embedded app leaves the host's logging untouched. No ``on_startup``
    counterpart: process start is covered by the CLI entrypoints' own
    ``setup_logging`` call.
    """
    setup_logging(logging_settings())


def register_cli_logging_reload() -> None:
    """Register the root-logger re-apply as a config-reload handler.

    Called only from the CLI entrypoints — a process whose root logger the CLI
    configured keeps it in sync across reloads; an embedded app never touches the
    host's logging. Idempotent: handlers are keyed by their ``module.qualname``, so
    a repeat registration replaces the same entry.
    """
    build_app().lifecycle.on_reload(apply_logging_settings)


@asynccontextmanager
async def lifespan(app_):
    # Process-wide resource teardown (pooled clients, store/checkpoint pools,
    # monitoring flush) runs on ``app_context`` — the seam both this served path
    # and the backend worker cross — so it is not duplicated here. See
    # ``TaiMCPLifecycleMixin._teardown_resources``.
    app = build_app()
    async with app.sub_app.mcp_sub_app_router.lifespan(app_):
        yield


def build_app() -> TaiMCP:
    """Build (once) the process app singleton.

    Reads access-control settings, so the CLI entrypoints call this from
    ``main()`` AFTER the env bootstrap (``load_dotenv``); the settings cache then
    captures the post-``.env`` value rather than a pre-bootstrap default. Mirrors
    ``cli.metrics.create_app`` — construction is deferred out of import time.

    Idempotent: the first call builds the app and wires the no-auth connector
    catalog refresh as a startup+reload handler; later calls return the same app.
    The catalog loads the in-memory provider cache at MCP startup AND after every
    in-place re-init (``update()`` drops process state, and a community add
    propagates fleet-wide through the backend reload dispatch), so the resolver's
    sync ``get_provider`` sees catalog providers on the tool-call hot path. The
    handler skips the load entirely when connectors are not in use (see
    :func:`refresh_catalog_if_connectors_in_use`), so a connector-less deployment
    never opens a Postgres connection at boot. Both at startup and on reload the
    handlers run with ``raise_on_error``, so a failed load fails the boot / the
    op loudly — never a silent half-loaded catalog.

    The root-logger re-apply on config reload is NOT wired here: it is a CLI-seam
    concern registered via :func:`register_cli_logging_reload`, so an embedded app
    that builds this singleton never reconfigures the host's root logger.
    ``install_meta_log_redactor`` is wired here regardless of caller, at its default
    ``"tai"`` scope: its fail-closed connector-secret redaction covers tai's own log
    records in embed mode too, while a host app's own records pass through untouched.
    The CLI entrypoints widen it to process scope beside their logging-reload
    registration.
    """
    global _app
    if _app is None:
        install_meta_log_redactor()
        settings = access_control_settings()
        auth_adapter = AuthAdapter(settings) if settings.enable else None
        app = TaiMCP(name="Tai", auth=auth_adapter, lifespan=lifespan)
        if settings.enable:
            # The configured identity providers probe their OWN record stores once at
            # startup, so a deployment against a backend a provider cannot use fails
            # LOUDLY at boot instead of on the first authenticated request.
            app.lifecycle.on_startup(probe_identity_provider)
            # The control-plane role templates (admin/editor/viewer) are seeded before
            # traffic, so a bootstrap ``apply_role`` can never KeyError on a fresh deploy.
            app.lifecycle.on_startup(seed_roles)
            # The always-public login surface is enumerated (visible at every boot) and
            # an accidental authed mount under it fails the boot closed.
            app.lifecycle.on_startup(check_always_public_routes)
            # The SPA-shell public fallback surface is audited: the derived reserved set
            # is printed, an unacknowledged public-by-declaration non-/api GET route fails
            # the boot closed, and the control-plane terminal-deny invariant is confirmed.
            app.lifecycle.on_startup(check_spa_shell_public)
            # Every gated route must resolve to an authorization action-class
            # (read/write/fenced/secret) — an unclassifiable route fails the boot closed,
            # so allow-by-omission can never reach the enforcement path.
            app.lifecycle.on_startup(check_route_actions)
            # Every registered fenced/secret route must resolve back to itself through the
            # gate's resolver, so the admin-only fence can never silently fail open.
            app.lifecycle.on_startup(check_fenced_routes_resolvable)
            # A registered accounts provider left out of the resolution chain would mint
            # sessions that never authenticate — refuse to boot instead.
            app.lifecycle.on_startup(check_accounts_providers_configured)
        app.lifecycle.on_startup(refresh_catalog_if_connectors_in_use)
        app.lifecycle.on_reload(refresh_catalog_if_connectors_in_use)
        # The Studio-plugin registry is a config-derived catalog like the
        # connector catalog: built at startup AND rebuilt on reload (both with
        # ``raise_on_error``, so a listed package missing its ``studio/`` dist
        # fails the boot/op loudly), so a reload that changes ``studio_plugins``
        # reflects without a process restart.
        app.lifecycle.on_startup(rebuild_studio_plugin_registry)
        app.lifecycle.on_reload(rebuild_studio_plugin_registry)
        # Versioned presets load at boot AND re-load on every in-place reload
        # (``reload_config()`` wipes the runtime tool registry), so a persisted
        # preset survives a restart and a reload. Gated + ``raise_on_error`` like
        # the connector catalog: the handler skips the Postgres open when the store
        # is unconfigured, and an unreachable Postgres fails the boot/op loudly.
        app.lifecycle.on_startup(rehydrate_versioned_presets_if_store_in_use)
        app.lifecycle.on_reload(rehydrate_versioned_presets_if_store_in_use)
        # Durable sub-MCP registrations rehydrate into this worker's router at boot
        # AND on every in-place reload (``reset()`` wipes the per-worker route cache),
        # so a registration survives a restart/reload and a sibling's registration
        # becomes visible on this worker's next reload. Registered AFTER the preset
        # handlers (handler dicts run in insertion order): a sub-MCP app may expose a
        # preset tool, whose registration must exist first. No env gate — the
        # in-memory store needs no connection and Redis mode fails loudly under
        # ``raise_on_error``.
        app.lifecycle.on_startup(rehydrate_sub_mcp_apps)
        app.lifecycle.on_reload(rehydrate_sub_mcp_apps)
        # Preset create/save/rollback/delete fan out on the worker bus
        # (``reload_tool``/``remove_tool`` with ``kind="preset"``); each subscribed
        # worker dispatches the op through this reloader so the fleet converges.
        app.admin.tool_reloader("preset")(_apply_preset_tool_reload)
        _app = app
    return _app


def __getattr__(name: str):
    # Expose the process app singleton as ``instance.app`` for importers
    # (``from tai42_skeleton.app.instance import app``) while deferring the heavy
    # build + settings read to first access (after the CLI env bootstrap) rather
    # than running it at import.
    if name == "app":
        return build_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
