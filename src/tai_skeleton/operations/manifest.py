"""Manifest + MCP-status operations — ``/api/manifest*``, ``/api/mcp-config*``,
``/api/mcp-status*``.

A thin skin over the live-manifest admin surface (``tai_app.admin``), the config
manager, the reload gate, and the worker bus (``instance.app.bus``). Two groups:

Reads (each returns its shape directly; the adapter envelopes it):

* ``get_manifest`` — the live manifest's MCP section + user tools.
* ``get_mcp_config_schema`` — the JSON Schema for one MCP-config entry.
* ``get_mcp_status`` — the live MCP binding snapshot.
* ``list_failed_mcps`` — the MCP servers skipped by the viability check; a query op
  over the bus, so every origin's list arrives as its per-origin report payload.

Mutations cross the single :class:`~tai_skeleton.config.service.ConfigService`
pipeline (validate → persist → local reload → broadcast) or, for pure runtime ops,
the shared :func:`~tai_skeleton.operations._broadcast.broadcast` primitive. Each is
``destructive`` + ``reload_gated`` and its response embeds the per-origin fleet
report as a ``fanout`` summary:

* ``set_mcp_config`` — replace the manifest's MCP section, persist, and reload the fleet.
* ``reload_mcp`` — re-probe a single MCP server by title (all workers, or only
  ``targets``). An unknown title is a loud 404.
* ``update_manifest`` — replace the WHOLE persisted manifest and reload the whole
  fleet. Authority-changing (it governs ``api_tools`` + module loading), so it is
  tier-2 (off the default MCP surface, includable).
* ``reload_failed_mcps`` — re-probe every failed MCP server.
* ``deregister_mcp`` — detach a single MCP server's tools by title.

The one response shape every ConfigService writer returns from an
:class:`~tai_skeleton.config.service.ApplyResult` is built by
:func:`~tai_skeleton.operations._broadcast.apply_response`.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel
from tai_contract.app import tai_app
from tai_kit.utils.data import load_manifest

from tai_skeleton.app.boot_rules import BackendNeedsBusError
from tai_skeleton.app.reload_gate import reload_gate
from tai_skeleton.config.service import ConfigService
from tai_skeleton.manifest import TaiMCPConfig
from tai_skeleton.operations import BadRequestError, NotFoundError, operation
from tai_skeleton.operations._broadcast import apply_response, broadcast


class McpConfigUpdate(BaseModel):
    """A replacement MCP config section — the full list of MCP server entries
    that overwrites the manifest's ``mcp`` list before the reload."""

    mcp: list[TaiMCPConfig]


class McpTargets(BaseModel):
    """An optional fleet fan-out restriction — a ``targets`` list naming the workers
    a single-server MCP action applies to (all workers when omitted)."""

    targets: list[str] | None = None


class ManifestReplace(BaseModel):
    """A full-manifest replacement carrying the manifest TEXT verbatim — the
    PRESERVED view (``!ENV`` markers intact). The server loads it to the preserved
    document and persists it through the pipeline, so it owns resolution and no
    secret bakes to disk. There is no ``targets``: a persisted replacement reaches
    the whole fleet."""

    manifest_text: str


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


@operation(summary="Read the live registries' manifest MCP section and user tools", tags=["manifest"])
async def get_manifest() -> dict:
    """Return the MCP section and user tools of the LIVE in-process manifest — what the
    running registries currently serve, as re-derived at the last reload. With every
    mutation crossing the ConfigService pipeline (persist → reload → broadcast), the
    live view tracks the persisted store through all supported paths; asserting the
    live-vs-persisted agreement is deferred to a later plan.
    """
    live = tai_app.admin.live_manifest
    # user_tools serializes from a set; JSON needs a stable list.
    user_tools = sorted(live.get("user_tools", []))
    return {"mcp": live.get("mcp", []), "user_tools": user_tools}


@operation(summary="Get the JSON schema for one MCP-config entry", tags=["manifest"])
async def get_mcp_config_schema() -> dict:
    return TaiMCPConfig.model_json_schema()


@operation(summary="Snapshot the live MCP binding status", tags=["manifest"])
async def get_mcp_status() -> dict:
    return tai_app.admin.live_mcp_status()


@operation(summary="List MCP servers skipped by the viability check", tags=["manifest"])
async def list_failed_mcps(targets: list[str] | None = None) -> Any:
    """List MCP servers skipped due to a failed viability check (server down or
    slow at boot or last reload). Use ``reload_mcp`` to re-attach one once healthy.

    Each entry is ``{"title": <name>, "status": "unavailable"}`` — title plus a
    coarse status only. A query op rides the same fan-out primitive as a mutation:
    every origin's list arrives as its per-origin ``payload`` in the fleet report
    (this worker's list on its own self entry); ``targets`` optionally restricts the
    query to specific workers.
    """

    async def _apply() -> Any:
        # A read, so no reload gate — this worker's failed-MCP list rides its self
        # entry as the payload.
        return tai_app.admin.list_failed_mcps()

    return await broadcast({"op": "list_failed_mcps"}, targets, _apply)


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


@operation(
    summary="Replace the MCP config section and hot-reload",
    tags=["manifest"],
    destructive=True,
    reload_gated=True,
    errors=[BadRequestError],
    request_model=McpConfigUpdate,
)
async def set_mcp_config(mcp: list[Any]) -> dict:
    # Replace the manifest's ``mcp`` section through the pipeline: the mutator edits
    # the PRESERVED document in place, then ConfigService validates, persists, reloads
    # locally, and broadcasts the reload to the whole fleet. A malformed entry fails
    # validation inside the transaction (nothing persisted) and maps to a loud 400.
    # (No docstring here, so the route description in projection falls back to the
    # operation summary.)
    def mutator(document: dict[str, Any]) -> None:
        document["mcp"] = mcp

    try:
        result = await ConfigService.from_app().apply_change(mutator)
    except BackendNeedsBusError as exc:
        # The invariant is a RuntimeError (a boot-time refusal must still crash loudly),
        # so the mutate-time path maps it explicitly to a loud, actionable 400 naming
        # TAI_BUS_REDIS_URL rather than letting it escape as a 500.
        raise BadRequestError(str(exc)) from exc
    except ValueError as exc:
        raise BadRequestError(f"invalid mcp config: {exc}") from exc
    return apply_response(result)


@operation(
    summary="Reload a single MCP server by title",
    tags=["manifest"],
    destructive=True,
    reload_gated=True,
    errors=[NotFoundError],
    request_model=McpTargets,
)
async def reload_mcp(title: str, targets: list[str] | None = None) -> Any:
    # Re-probe a single MCP server by title (unknown title → loud 404), applied on
    # this worker through the gate and broadcast to the fleet (all workers, or only
    # ``targets``); the response embeds the per-origin fleet report. A pure runtime
    # op: if the local re-probe raises, nothing is broadcast. (No docstring here, so
    # the route description in projection falls back to the operation summary.)
    live = tai_app.admin.live_manifest
    titles = {entry.get("title") for entry in live.get("mcp", [])}
    if title not in titles:
        raise NotFoundError(f"unknown mcp title: {title!r}")
    return await broadcast(
        {"op": "reload_mcp", "title": title},
        targets,
        lambda: reload_gate.run(lambda: tai_app.admin.reload_mcp(title)),
    )


@operation(
    summary="Replace the whole manifest, persist, and reload the fleet",
    tags=["manifest"],
    destructive=True,
    reload_gated=True,
    authority_changing=True,
    errors=[BadRequestError],
    request_model=ManifestReplace,
)
async def update_manifest(manifest_text: str) -> Any:
    """Replace the WHOLE manifest fleet-wide and persist it.

    The posted ``manifest_text`` is the PRESERVED view (``!ENV`` markers intact); the
    server loads it to the preserved document and pushes it through the pipeline —
    validate the RESOLVED projection, persist verbatim (no secret bakes to disk),
    reload locally, and broadcast the reload so every worker re-reads the persisted
    store. The response embeds the per-origin fleet report as its ``fanout`` summary.
    A persisted replacement reaches the whole fleet, so there is no ``targets``.

    Authority-changing — the manifest governs ``api_tools`` + module loading — so it
    is off the default MCP surface (tier 2), projectable via an explicit
    ``api_tools.include``.
    """
    try:
        document = cast("dict[str, Any]", load_manifest(manifest_text))
    except Exception as exc:
        raise BadRequestError(f"invalid manifest: {exc}") from exc
    try:
        result = await ConfigService.from_app().apply_replace(document)
    except BackendNeedsBusError as exc:
        # The invariant is a RuntimeError (a boot-time refusal must still crash loudly),
        # so the mutate-time path maps it explicitly to a loud, actionable 400 naming
        # TAI_BUS_REDIS_URL rather than letting it escape as a 500.
        raise BadRequestError(str(exc)) from exc
    except ValueError as exc:
        raise BadRequestError(f"invalid manifest: {exc}") from exc
    return apply_response(result)


@operation(
    summary="Re-probe every failed MCP server",
    tags=["manifest"],
    destructive=True,
    reload_gated=True,
    request_model=McpTargets,
)
async def reload_failed_mcps(targets: list[str] | None = None) -> Any:
    """Re-probe every MCP server currently in the failed list and attach the ones now
    viable. Applied on this worker through the gate and broadcast to the fleet (all
    workers, or only ``targets``); the response embeds the per-origin fleet report.
    """
    # Run the heavy sync re-probe pass on a worker thread through the gate.
    return await broadcast(
        {"op": "reload_failed_mcps"},
        targets,
        lambda: reload_gate.run(tai_app.admin.reload_failed_mcps),
    )


@operation(
    summary="Detach a single MCP server's tools by title",
    tags=["manifest"],
    destructive=True,
    reload_gated=True,
    request_model=McpTargets,
)
async def deregister_mcp(title: str, targets: list[str] | None = None) -> Any:
    """Detach a single MCP server's tools (by manifest title) without touching the
    other servers — the removal counterpart of ``reload_mcp``. Applied on this worker
    through the gate and broadcast to the fleet (all workers, or only ``targets``);
    the response embeds the per-origin fleet report.
    """
    # Run the heavy sync detach on a worker thread through the gate.
    return await broadcast(
        {"op": "deregister_mcp", "title": title},
        targets,
        lambda: reload_gate.run(lambda: tai_app.admin.deregister_mcp(title)),
    )
