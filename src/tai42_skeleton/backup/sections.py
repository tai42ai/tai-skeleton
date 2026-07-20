"""The host's own backup sections — the skeleton as the first consumer of its
own ``app.backup`` facet.

Each section is a THIN exporter/importer pair over the owning subsystem's
existing read/write seam; no backup logic lives inside the subsystems. An
exporter returns a JSON-safe payload; an importer applies a payload and returns a
section report ``{"created", "updated", "skipped", "errors"}`` (the
``access_control`` importer additionally carries ``"new_api_keys"``; the ``manifest``
and ``env`` importers additionally carry ``"fanout"`` — the per-origin fleet report of
the reload their pipeline apply broadcast). A section
whose backing subsystem is absent lets the underlying seam raise, and the HTTP
router catches it per-section into the report; the registry and these functions
never swallow it.

The exporters/importers reference the live subsystems through the ``tai42_app``
handle (or a feature accessor) at call time, so a section is a pure closure over
the running app, not a snapshot taken at registration.

The manifest section exports the preserved-``!ENV`` view, so an ``!ENV``
placeholder never leaves the host as a resolved secret; a value the operator wrote
as a literal (not an ``!ENV`` placeholder) exports as-is — the same exposure as
``manifest.yml`` itself — so anything that must never leave the host belongs behind
an ``!ENV`` placeholder.

The connector tables are backed up at the SQL layer through
``connectors.store.backup`` (Postgres is their source of truth, so the section
reads/writes rows directly rather than through the network-gated service layer):
``connector_catalog`` carries the categories, the full catalog including disabled
rows, and the allowed-source list; ``connector_connections`` carries each token
record's ciphertext verbatim under the same KEK constraint that module documents.

The ``schedules`` section is opaque: it exports through the scheduling backend's
own ``backend_export_schedules`` door and imports through
``backend_import_schedules``, carrying the backend's document as-is without
parsing schedule internals. When no scheduling backend is installed those tools
are absent, so ``run_tool`` raises the unknown-tool error and the backup router
records it per-section — a bound backend round-trips, an unbound one reports the
absence loudly.
"""

from __future__ import annotations

import logging
from typing import Any

from tai42_contract.app import tai42_app

logger = logging.getLogger(__name__)

# The report shape every importer returns (access_control extends it with
# ``new_api_keys``). Built fresh per import so no state leaks between runs.
_SectionReport = dict[str, Any]


def _empty_report() -> _SectionReport:
    return {"created": 0, "updated": 0, "skipped": 0, "errors": []}


# -- manifest ----------------------------------------------------------------


def _export_manifest() -> dict[str, Any]:
    # Export the preserved-tag view: each ``!ENV <expr>`` reference travels as its
    # literal ``"!ENV <expr>"`` marker string (JSON-safe), not the resolved secret,
    # so the non-secret manifest section never carries a live secret. The importer's
    # dump re-emits the marker as a genuine ``!ENV`` tag.
    return tai42_app.config.config_manager.read_manifest_preserved()


async def _import_manifest(payload: dict[str, Any]) -> _SectionReport:
    from tai42_skeleton.config.service import ConfigService

    # The imported section is the PRESERVED-view manifest (``!ENV`` markers intact),
    # so it replaces the persisted manifest as a whole through the pipeline: the
    # document is validated on its resolved projection, persisted, locally reloaded,
    # and broadcast to the fleet. A validation failure raises here (nothing persisted)
    # and the router records it as this section's error.
    result = await ConfigService.from_app().apply_replace(payload)
    report = _empty_report()
    # The manifest is a single document; a restore always replaces the live one.
    report["updated"] = 1
    # The awaited fleet broadcast rides the section report so a restore's fleet-wide
    # propagation is visible per-origin.
    report["fanout"] = result.fanout
    return report


# -- env ---------------------------------------------------------------------


def _export_env() -> dict[str, str]:
    try:
        return tai42_app.config.config_manager.read_env()
    except FileNotFoundError:
        # No env file yet is a normal empty state, not an error.
        return {}


async def _import_env(payload: dict[str, str]) -> _SectionReport:
    from tai42_skeleton.config.service import ConfigService

    config_manager = tai42_app.config.config_manager
    try:
        existing = config_manager.read_env()
    except FileNotFoundError:
        existing = {}
    report = _empty_report()
    report["created"] = sum(1 for key in payload if key not in existing)
    report["updated"] = sum(1 for key in payload if key in existing)
    # Apply through the pipeline so a restored env is validated (the effective config
    # the change produces), locally reloaded, and broadcast to the fleet.
    result = await ConfigService.from_app().apply_env_change(payload)
    report["fanout"] = result.fanout
    return report


# -- access_control ----------------------------------------------------------


async def _export_access_control() -> dict[str, Any]:
    from tai42_skeleton.access_control import management

    return {
        # ``scopes`` carries EVERY route mapping url -> value, including explicit
        # public routes (value ``settings.public_resource_id``). Using the full
        # mapping — not the non-public-only ``get_all_existing_scopes`` — keeps a
        # public route from silently restoring as protected.
        "scopes": await management.get_all_route_mappings(),
        "patterns": await management.get_all_existing_patterns(),
        "tokens": await management.get_all_existing_tokens_payload(),
    }


async def _import_access_control(payload: dict[str, Any]) -> _SectionReport:
    from tai42_skeleton.access_control import management
    from tai42_skeleton.access_control.settings import access_control_settings

    report = _empty_report()
    report["new_api_keys"] = []

    # Replay the route -> scope mappings first so the token restore below finds
    # every scope it references already provisioned. A url carrying a dynamic
    # route pattern is restored WITH that pattern, so pattern-scoped routes
    # re-authorize exactly as before the backup. An explicit public route (value
    # ``settings.public_resource_id``) replays through the same setter, re-creating
    # the public mapping so it does not silently come back protected.
    marker = access_control_settings().public_resource_id
    patterns = payload.get("patterns") or {}
    for url, scope_id in (payload.get("scopes") or {}).items():
        if scope_id == marker:
            # A public route restores through the dedicated public-pin writer, never
            # ``add_url_to_scope`` — the marker is a column value, not a scope, and the
            # pin functions are its only writers.
            await management.pin_route_public(url, patterns.get(url))
        else:
            await management.add_url_to_scope(scope_id, url, patterns.get(url))
        report["created"] += 1

    # API-key hashes are one-way, so a restore mints BRAND-NEW keys and surfaces
    # each plaintext in ``new_api_keys`` for the operator to redistribute.
    for token in payload.get("tokens") or []:
        user_id = token.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            # A corrupt/hand-edited backup with a missing or empty user id is a
            # loud per-token rejection, never a record written under a blank id.
            report["errors"].append(f"token with missing or empty user_id: {user_id!r}")
            report["skipped"] += 1
            continue
        description = token.get("description", "")
        try:
            api_key, _committed_body = await management.add_user_api_key(
                user_id,
                description,
                token.get("scopes") or [],
                token.get("policy_data"),
                token.get("condition"),
                token.get("condition_id"),
                token.get("condition_kwargs"),
            )
        except ValueError as exc:
            # A collided user id or a scope that never materialized is a per-token
            # failure surfaced loudly in the report — the rest still restore.
            report["errors"].append(f"token {user_id!r}: {exc}")
            report["skipped"] += 1
            continue
        report["created"] += 1
        report["new_api_keys"].append({"user_id": user_id, "description": description, "api_key": api_key})

    await management.bump_policy_version()
    return report


# -- sub_mcp -----------------------------------------------------------------


async def _export_sub_mcp() -> dict[str, Any]:
    # Export from the durable store (the source of truth), so a backup captures
    # every registration coherently — not just this worker's in-process cache.
    from tai42_skeleton.sub_mcp.store import get_sub_mcp_store

    routes = await get_sub_mcp_store().list_routes()
    return {slug: config.model_dump() for slug, config in routes.items()}


async def _import_sub_mcp(payload: dict[str, Any]) -> _SectionReport:
    from tai42_skeleton.sub_mcp import service
    from tai42_skeleton.sub_mcp.store import get_sub_mcp_store

    store = get_sub_mcp_store()
    report = _empty_report()
    for slug, config in payload.items():
        # The created/updated counts reflect the DURABLE state (store presence), not
        # this worker's local cache — consistent with the store-backed export above.
        existed = await store.get_route(slug) is not None
        if not isinstance(config, dict) or "tools" not in config:
            # A hand-edited backup entry missing its ``tools`` mapping is a loud
            # per-slug rejection in the report, so one malformed entry never aborts
            # the whole restore.
            report["errors"].append(
                f"sub-MCP app {slug!r}: malformed backup entry (expected a mapping with a 'tools' key)"
            )
            report["skipped"] += 1
            continue
        try:
            # The service validates the slug shape + transport BEFORE its store
            # write, so a malformed slug from a hand-edited backup is a loud per-slug
            # rejection in the report — never a silently-minted phantom route or a
            # garbage store entry — then persists (store-first) and binds locally.
            await service.register_sub_mcp_app(slug, config["tools"], config.get("transport", "http"))
        except ValueError as exc:
            report["errors"].append(f"sub-MCP app {slug!r}: {exc}")
            report["skipped"] += 1
            continue
        if existed:
            report["updated"] += 1
        else:
            report["created"] += 1
    return report


# -- webhooks ----------------------------------------------------------------


async def _export_webhooks() -> list[dict[str, Any]]:
    from tai42_skeleton.hooks.cache import get_hooks_manager

    hooks = await get_hooks_manager().list_hooks()
    return [params.model_dump(mode="json") for params in hooks.values()]


async def _import_webhooks(payload: list[dict[str, Any]]) -> _SectionReport:
    from tai42_contract.hooks import HookParams

    from tai42_skeleton.hooks.cache import get_hooks_manager

    manager = get_hooks_manager()
    existing = await manager.list_hooks()
    report = _empty_report()
    for item in payload:
        params = HookParams.model_validate(item)
        await manager.register(params)
        if params.name in existing:
            report["updated"] += 1
        else:
            report["created"] += 1
    return report


# -- templates ---------------------------------------------------------------


async def _export_templates() -> dict[str, str]:
    resource_manager = tai42_app.storage.resource_manager
    paths = await resource_manager.list_resources()
    return {path: await resource_manager.fetch_template(path) for path in paths}


async def _import_templates(payload: dict[str, str]) -> _SectionReport:
    from tai42_skeleton.template.path_guard import UnsafeTemplatePathError, safe_template_path

    resource_manager = tai42_app.storage.resource_manager
    existing = set(await resource_manager.list_resources())
    report = _empty_report()
    for path, content in payload.items():
        try:
            # An untrusted backup can carry a traversal key (``../../etc/app/x``)
            # that writes outside the store root; guard each key BEFORE the upload.
            # A bad key is a loud per-path rejection in the report + log, never a
            # silent drop and never an aborted restore.
            safe_template_path(path)
        except UnsafeTemplatePathError as exc:
            report["errors"].append(f"template {path!r}: {exc}")
            report["skipped"] += 1
            logger.warning("backup restore skipped unsafe template path %r: %s", path, exc)
            continue
        await resource_manager.upload_template(path, content)
        if path in existing:
            report["updated"] += 1
        else:
            report["created"] += 1
    return report


# -- schedules ---------------------------------------------------------------


async def _export_schedules() -> Any:
    # The scheduling backend owns the schedule document shape; the skeleton
    # carries it opaquely. An unbound backend has no export tool, so run_tool
    # raises the unknown-tool error and the router records it per-section.
    return await tai42_app.tools.run_tool("backend_export_schedules", {})


async def _import_schedules(payload: Any) -> _SectionReport:
    return await tai42_app.tools.run_tool("backend_import_schedules", {"schedules": payload})


# -- connector_catalog / connector_connections -------------------------------


async def _export_connector_catalog() -> dict[str, Any]:
    from tai42_skeleton.connectors.store.backup import export_connector_catalog

    return await export_connector_catalog()


async def _import_connector_catalog(payload: dict[str, Any]) -> _SectionReport:
    from tai42_skeleton.connectors.store.backup import import_connector_catalog

    return await import_connector_catalog(payload)


async def _export_connector_connections() -> list[dict[str, Any]]:
    from tai42_skeleton.connectors.store.backup import export_connector_connections

    return await export_connector_connections()


async def _import_connector_connections(payload: list[dict[str, Any]]) -> _SectionReport:
    from tai42_skeleton.connectors.store.backup import import_connector_connections

    return await import_connector_connections(payload)


# -- versioned_documents (the kind-agnostic versioned-document store) ---------


async def _export_versioned_documents() -> dict[str, Any]:
    from tai42_skeleton.versioning.backup import export_versioned_documents

    return await export_versioned_documents()


async def _import_versioned_documents(payload: dict[str, Any]) -> _SectionReport:
    from tai42_skeleton.versioning.backup import import_versioned_documents

    return await import_versioned_documents(payload)


# -- registration ------------------------------------------------------------


def register_core_sections(registry: Any) -> None:
    """Register the skeleton's built-in sections on ``registry``.

    Called once when the app object is constructed (the registry is built per
    ``TaiMCP``), so an in-place reload — which re-imports modules but keeps the
    same app object — never re-registers and never trips the duplicate guard.
    """
    registry.register_section("manifest", _export_manifest, _import_manifest)
    registry.register_section("env", _export_env, _import_env, secret=True)
    registry.register_section("access_control", _export_access_control, _import_access_control, secret=True)
    registry.register_section("sub_mcp", _export_sub_mcp, _import_sub_mcp)
    registry.register_section("webhooks", _export_webhooks, _import_webhooks)
    registry.register_section("templates", _export_templates, _import_templates)
    # Registered unconditionally: an unbound scheduling backend surfaces as a
    # per-section error through the router, never a hidden gap in the section list.
    registry.register_section("schedules", _export_schedules, _import_schedules, secret=True)
    registry.register_section("connector_catalog", _export_connector_catalog, _import_connector_catalog)
    registry.register_section(
        "connector_connections", _export_connector_connections, _import_connector_connections, secret=True
    )
    # The versioned-document store is body-opaque and kind-agnostic, so ONE
    # section covers every kind (presets, AC policies, authored agents, ...).
    # secret=True: at least one covered kind is secret-bearing — a preset's
    # ``fixed_kwargs`` can embed credentials and an AC-policy condition body is
    # sensitive — so the whole opaque section is treated as a secret.
    registry.register_section(
        "versioned_documents", _export_versioned_documents, _import_versioned_documents, secret=True
    )
