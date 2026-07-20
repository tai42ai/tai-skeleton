"""Manifest + MCP HTTP surface for the Studio manifest feature.

AUTHED thin adapters over operations in ``tai_skeleton.operations.manifest`` (the
live manifest can embed connector tokens in MCP headers/env, so this whole surface
stays behind the credential):

- ``GET  /api/manifest``                     — the LIVE registries' manifest MCP section + user tools.
- ``POST /api/manifest/replace``             — replace the WHOLE persisted manifest, fleet-wide (tier-2).
- ``POST /api/mcp-config``                   — replace the MCP section (persist + reload).
- ``GET  /api/mcp-config/schema``            — the JSON Schema for one MCP-config entry.
- ``GET  /api/mcp-status``                   — live MCP binding snapshot.
- ``GET  /api/mcp-status/failed``            — the MCP servers skipped by the viability check.
- ``POST /api/mcp-status/reload-failed``     — re-probe every failed MCP server.
- ``POST /api/mcp-status/{title}/reload``    — reload a single MCP server by title.
- ``POST /api/mcp-status/{title}/deregister``— detach a single MCP server's tools by title.

There is no ``probe-mcp`` route: the skeleton has no MCP-probe primitive, only the
connectors sub-service probe.

Success bodies are ``{"data": ...}``; failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

from json import JSONDecodeError
from typing import Any

from pydantic import ValidationError
from starlette.requests import Request
from tai_contract.app import tai_app

from tai_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai_skeleton.operations.manifest import ManifestReplace
from tai_skeleton.operations.manifest import deregister_mcp as _deregister_mcp_op
from tai_skeleton.operations.manifest import get_manifest as _get_manifest_op
from tai_skeleton.operations.manifest import get_mcp_config_schema as _get_mcp_config_schema_op
from tai_skeleton.operations.manifest import get_mcp_status as _get_mcp_status_op
from tai_skeleton.operations.manifest import list_failed_mcps as _list_failed_mcps_op
from tai_skeleton.operations.manifest import reload_failed_mcps as _reload_failed_mcps_op
from tai_skeleton.operations.manifest import reload_mcp as _reload_mcp_op
from tai_skeleton.operations.manifest import set_mcp_config as _set_mcp_config_op
from tai_skeleton.operations.manifest import update_manifest as _update_manifest_op


async def _json_object(request: Request) -> dict:
    """Read a JSON-object request body, mapping a malformed/non-object body to the
    door's explicit 400 (a plain request-model parse would answer 422)."""
    try:
        body = await request.json()
    except (JSONDecodeError, ValueError) as exc:
        raise BadRequestError(str(exc)) from exc
    if not isinstance(body, dict):
        raise BadRequestError("request body must be a JSON object")
    return body


async def _extract_mcp_config(request: Request) -> dict[str, Any]:
    """The MCP-config replacement body → the operation's flat ``mcp`` kwarg. The
    hand-authored 400s (malformed body, missing ``mcp`` list) are preserved here;
    a malformed ENTRY is validated by the operation (also a 400)."""
    body = await _json_object(request)
    if "mcp" not in body:
        raise BadRequestError("body must carry an 'mcp' list")
    return {"mcp": body["mcp"]}


async def _optional_targets(request: Request) -> list[str] | None:
    """The optional ``targets`` fan-out restriction from a POST body, tolerating an
    absent/empty body (no body → ``targets=None`` → the unchanged single-worker path)."""
    try:
        body = await request.json()
    except (JSONDecodeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    return body.get("targets")


async def _extract_targets(request: Request) -> dict[str, Any]:
    """A body carrying only the optional ``targets`` fan-out restriction (reload,
    reload-failed, deregister); ``title`` is a path param the adapter supplies."""
    return {"targets": await _optional_targets(request)}


async def _extract_failed_query(request: Request) -> dict[str, Any]:
    """The optional ``targets`` fan-out restriction from the query string (a GET
    carries it as a repeated ``?targets=`` param, never a body); absent → ``None``."""
    targets = request.query_params.getlist("targets")
    return {"targets": targets or None}


async def _extract_manifest_replace(request: Request) -> dict[str, Any]:
    """The full-manifest replacement body → the operation's flat ``manifest_text``.

    The body carries the manifest TEXT verbatim (the PRESERVED view — ``!ENV`` markers
    intact); the operation loads and validates it. A body missing ``manifest_text`` (or
    a non-string one) is a loud 400 rather than the adapter's default 422."""
    body = await _json_object(request)
    try:
        model = ManifestReplace.model_validate(body)
    except ValidationError as exc:
        raise BadRequestError(f"invalid manifest: {exc}") from exc
    return {"manifest_text": model.manifest_text}


get_manifest = register_operation_route(
    tai_app,
    operation_metadata_of(_get_manifest_op),
    path="/api/manifest",
    method="GET",
)

update_manifest = register_operation_route(
    tai_app,
    operation_metadata_of(_update_manifest_op),
    path="/api/manifest/replace",
    method="POST",
    context_extractor=_extract_manifest_replace,
)

set_mcp_config = register_operation_route(
    tai_app,
    operation_metadata_of(_set_mcp_config_op),
    path="/api/mcp-config",
    method="POST",
    context_extractor=_extract_mcp_config,
)

get_mcp_config_schema = register_operation_route(
    tai_app,
    operation_metadata_of(_get_mcp_config_schema_op),
    path="/api/mcp-config/schema",
    method="GET",
)

get_mcp_status = register_operation_route(
    tai_app,
    operation_metadata_of(_get_mcp_status_op),
    path="/api/mcp-status",
    method="GET",
)

list_failed_mcps = register_operation_route(
    tai_app,
    operation_metadata_of(_list_failed_mcps_op),
    path="/api/mcp-status/failed",
    method="GET",
    context_extractor=_extract_failed_query,
)

reload_failed_mcps = register_operation_route(
    tai_app,
    operation_metadata_of(_reload_failed_mcps_op),
    path="/api/mcp-status/reload-failed",
    method="POST",
    context_extractor=_extract_targets,
)

reload_mcp = register_operation_route(
    tai_app,
    operation_metadata_of(_reload_mcp_op),
    path="/api/mcp-status/{title}/reload",
    method="POST",
    context_extractor=_extract_targets,
)

deregister_mcp = register_operation_route(
    tai_app,
    operation_metadata_of(_deregister_mcp_op),
    path="/api/mcp-status/{title}/deregister",
    method="POST",
    context_extractor=_extract_targets,
)
