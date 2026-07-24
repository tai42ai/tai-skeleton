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

from pydantic import ValidationError
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
            api_key, _committed_body, _fingerprint = await management.add_user_api_key(
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


async def _export_webhooks() -> dict[str, Any]:
    from tai42_skeleton.hooks.cache import get_hooks_manager
    from tai42_skeleton.hooks.trigger_links import export_trigger_links

    manager = get_hooks_manager()
    hooks = await manager.list_hooks()
    # Envelope: hooks + per-topic verifier bindings + trigger-link records/index/tombstones
    # (hashes and metadata only — never a raw token; a binding's secret is an env var NAME).
    # The bindings must travel with the hooks or a verified topic restores as a public door.
    return {
        "hooks": [params.model_dump(mode="json") for params in hooks.values()],
        "topic_verifiers": await manager.all_topic_verifiers(),
        **(await export_trigger_links()),
    }


async def _import_webhooks(payload: list[dict[str, Any]] | dict[str, Any]) -> _SectionReport:
    from tai42_contract.hooks import HookParams

    from tai42_skeleton.authz.execution import ExecutionKeyAuthorityError, ExecutionKeyScan
    from tai42_skeleton.authz.token_free import TokenFreeConditionError
    from tai42_skeleton.hooks.cache import get_hooks_manager
    from tai42_skeleton.hooks.trigger_links import (
        TriggerLinkError,
        bound_hashes_by_name,
        restore_tombstone,
        restore_trigger_link,
    )

    manager = get_hooks_manager()
    report = _empty_report()
    # One scan for the whole restore: each distinct execution key is read and rendered once.
    scan = ExecutionKeyScan()

    async def _restore_hooks(hooks: list[dict[str, Any]], unlocked_topics: frozenset[str] = frozenset()) -> None:
        existing = await manager.list_hooks()
        for item in hooks:
            name = item.get("name") if isinstance(item, dict) else None
            try:
                params = HookParams.model_validate(item)
            except ValidationError as exc:
                # Per-hook rejection (which forces the import to report failure), never a
                # hook written without a bounded identity nor an aborted restore of the rest.
                report["errors"].append(f"hook {name!r}: {exc}")
                report["skipped"] += 1
                continue
            if params.topic in unlocked_topics:
                # Lock absent: writing the hook would hand an unverified public door this
                # hook's bound execution key.
                report["errors"].append(
                    f"hook {name!r}: topic {params.topic!r} is not restored — its verifier binding failed, "
                    "which would leave this hook live on an unverified public door"
                )
                report["skipped"] += 1
                continue
            try:
                # Every stored hook must name a usable, token-free-evaluable execution key,
                # so the import asserts it exactly as the register door does. The pass-role
                # half is not asserted: this route is admin-only fenced.
                await scan.assert_usable(params.execution_key, bound_fingerprint=params.execution_key_fingerprint)
            except (ExecutionKeyAuthorityError, TokenFreeConditionError) as exc:
                # Only these two types are per-record faults; other types from the
                # policy-store read propagate as the section's own failure.
                report["errors"].append(f"hook {name!r}: {exc}")
                report["skipped"] += 1
                continue
            try:
                await manager.register(params)
            except ValueError as exc:
                # register() compiles the inline condition/expr jq: a non-compiling
                # expression is a per-hook rejection. Store/transport failures raise other
                # types and propagate as the section's own failure.
                report["errors"].append(f"hook {name!r}: {exc}")
                report["skipped"] += 1
                continue
            if params.name in existing:
                report["updated"] += 1
            else:
                report["created"] += 1

    # A bare LIST is the hooks-only backup shape (no trigger-link envelope); restore the hooks directly.
    if isinstance(payload, list):
        await _restore_hooks(payload)
        return report

    if not isinstance(payload, dict):
        raise ValueError(
            f"webhooks section payload must be a list (old shape) or an envelope dict, got {type(payload)}"
        )
    # A hand-edited envelope MISSING any key, or carrying a non-list value for one,
    # raises loudly BEFORE any write — never default-empty a section silently, never
    # let a bad type fail ungracefully deeper in.
    for key in ("hooks", "trigger_links", "tombstones"):
        if key not in payload:
            raise ValueError(f"webhooks envelope is missing the required {key!r} key")
        if not isinstance(payload[key], list):
            raise ValueError(f"webhooks envelope {key!r} must be a list")
    if "topic_verifiers" not in payload:
        raise ValueError("webhooks envelope is missing the required 'topic_verifiers' key")
    if not isinstance(payload["topic_verifiers"], dict):
        raise ValueError("webhooks envelope 'topic_verifiers' must be a mapping of topic to binding")
    hooks = payload["hooks"]
    topic_verifiers = payload["topic_verifiers"]
    trigger_links = payload["trigger_links"]
    tombstones = payload["tombstones"]

    # PRE-WRITE whole-section scan: refuse a payload binding ONE hash under TWO
    # different names — internally, or against the LIVE index (a corrupt backup would
    # otherwise leave a permanent zombie name index after one is revoked). The live
    # index enumerates EVERY name:* binding on the store, orphans included, so a new
    # name binding an already-orphaned hash is refused too (revoke would later destroy
    # its live record). This runs BEFORE any write (hooks included) so the refusal
    # touches zero keys. A NAME appearing twice is NOT refused — it resolves last-wins
    # through displacement.
    _reject_duplicate_hash_binding(trigger_links, await bound_hashes_by_name())

    # Ingress locks go back BEFORE the records they gate: no window in which a restored
    # hook is reachable through a door the backup had verified. The bound verifier NAME is
    # not resolved here — the door resolves it live per delivery and denies what it cannot
    # resolve. Topics whose lock failed are carried forward and every record ON them is
    # refused below, so no record is ever written onto a door whose lock did not restore.
    unlocked_topics: set[str] = set()
    existing_verifiers = await manager.all_topic_verifiers()
    for topic, binding in topic_verifiers.items():
        if not isinstance(topic, str) or not topic:
            report["errors"].append(f"topic verifier with missing or empty topic: {topic!r}")
            report["skipped"] += 1
            continue
        try:
            # Validates the binding shape on write: a hand-edited entry is a per-topic
            # rejection, never a stored binding the ingress door would choke on.
            await manager.set_topic_verifier(topic, binding)
        except ValidationError as exc:
            report["errors"].append(f"topic verifier {topic!r}: {exc}")
            report["skipped"] += 1
            unlocked_topics.add(topic)
            continue
        if topic in existing_verifiers:
            report["updated"] += 1
        else:
            report["created"] += 1

    await _restore_hooks(hooks, frozenset(unlocked_topics))

    # Tombstones first, then records — a tombstoned hash then refuses its own record
    # in-script (the tombstone wins over any imported record).
    for token_hash in tombstones:
        try:
            await restore_tombstone(token_hash)
        except TriggerLinkError as exc:
            report["errors"].append(f"tombstone {token_hash!r}: {exc.message}")
            report["skipped"] += 1

    for item in trigger_links:
        name = item.get("name") if isinstance(item, dict) else None
        try:
            if not isinstance(item, dict):
                raise TriggerLinkError(400, "trigger link entry must be a JSON object")
            record = item["record"]
            topic = record.get("topic") if isinstance(record, dict) else None
            if topic in unlocked_topics:
                # Lock absent: restoring the link would put it back into service on a door
                # the backup had verified.
                raise TriggerLinkError(
                    400,
                    f"topic {topic!r} is not restored — its verifier binding failed, which would take this "
                    "link back into service on an unverified public door",
                )
            outcome = await restore_trigger_link(
                name=item["name"], token_hash=item["token_hash"], record=record, scan=scan
            )
        except (TriggerLinkError, KeyError) as exc:
            message = exc.message if isinstance(exc, TriggerLinkError) else f"missing key {exc}"
            report["errors"].append(f"trigger link {name!r}: {message}")
            report["skipped"] += 1
            continue
        if outcome in ("skipped_expired", "skipped_tombstoned"):
            report["skipped"] += 1
        elif outcome == "updated":
            report["updated"] += 1
        else:
            report["created"] += 1
    return report


def _reject_duplicate_hash_binding(trigger_links: Any, live_by_name: dict[str, str]) -> None:
    """Raise if any single token hash is bound under two DIFFERENT names, either
    within the payload or against the live store index — zero keys written.
    ``live_by_name`` maps every ``name:*`` binding on the store (orphans included) to
    its hash, so a payload binding a new name to an already-orphaned hash is refused
    too (revoke treats an orphan binding as authoritative and would later destroy the
    new name's live record). ``trigger_links`` is a list — the envelope-key check
    validates that before this runs."""
    # Only well-formed entries participate; malformed ones are left to the per-item
    # restore to reject loudly in the report.
    hash_to_name: dict[str, str] = {}
    for item in trigger_links:
        if not isinstance(item, dict):
            continue
        name, token_hash = item.get("name"), item.get("token_hash")
        if not isinstance(name, str) or not isinstance(token_hash, str):
            continue
        prior = hash_to_name.get(token_hash)
        if prior is not None and prior != name:
            raise ValueError(
                f"webhooks envelope binds token hash {token_hash!r} under two names ({prior!r} and {name!r})"
            )
        hash_to_name[token_hash] = name

    live_by_hash = {token_hash: name for name, token_hash in live_by_name.items()}
    for token_hash, name in hash_to_name.items():
        live_name = live_by_hash.get(token_hash)
        if live_name is not None and live_name != name:
            raise ValueError(
                f"import binds token hash {token_hash!r} to {name!r} but it is already live under {live_name!r}"
            )


# -- conversations -----------------------------------------------------------


async def _export_conversations() -> dict[str, Any]:
    from tai42_skeleton.conversations.backup import export_conversation_routes

    return await export_conversation_routes()


async def _import_conversations(payload: dict[str, Any]) -> _SectionReport:
    from tai42_skeleton.conversations.backup import import_conversation_routes

    return await import_conversation_routes(payload)


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

    Registration order IS an import's replay order: a section deciding its records against
    live state another section restores must be registered AFTER that section.
    """
    registry.register_section("manifest", _export_manifest, _import_manifest)
    registry.register_section("env", _export_env, _import_env, secret=True)
    registry.register_section("access_control", _export_access_control, _import_access_control, secret=True)
    registry.register_section("sub_mcp", _export_sub_mcp, _import_sub_mcp)
    # Before ``webhooks``/``conversations``: their token-free scan renders policy
    # conditions carried by ``condition_id``, which are templates this section restores.
    registry.register_section("templates", _export_templates, _import_templates)
    # AFTER ``access_control`` and ``templates``: every record is decided against the LIVE
    # policy store (its bound execution key must exist and be token-free-evaluable), which
    # those two sections restore.
    # secret=True: the bulk export aggregates the whole surface's records — the hook
    # ``tool_kwargs`` and the trigger-link envelope's full token hashes — which is
    # broader than the grantable per-record list read.
    registry.register_section("webhooks", _export_webhooks, _import_webhooks, secret=True)
    # AFTER ``access_control`` and ``templates``, for the same reason as ``webhooks``.
    # secret=False: the export equals the grantable route-list read (``callback_secret``
    # is excluded and re-minted on import).
    registry.register_section("conversations", _export_conversations, _import_conversations)
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
