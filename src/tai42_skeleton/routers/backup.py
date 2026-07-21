"""HTTP surface for the backup/restore feature — the Studio's backup UI doors.

Three AUTHED doors over the live ``tai42_app.backup`` registry. The registry lists
its sections (host sections plus any a plugin registered), and each section runs
its own subsystem's export/import through a thin exporter/importer pair. An
export whose section carries secrets stays behind this credential exactly as the
config-env surface does.

- ``GET  /api/backup/sections`` — the registered sections as ``{name, secret}``,
  so the UI renders one checkbox per live section (plugins included).
- ``POST /api/backup/export`` — body ``{"sections": [names]}``. Requesting a
  section that is not registered is a loud 400. The response is a downloadable
  JSON document ``{"version", "created_at", "sections": {name: payload},
  "errors": {name: message}}``: a section that exported cleanly lands under
  ``sections``; a section whose backing subsystem is absent (its exporter raises)
  lands under ``errors`` and is omitted from ``sections`` — the export still
  returns the document as a download, never a 500 and never a silent drop.
- ``POST /api/backup/import`` — body ``{"document": <export document>,
  "sections": [names to import]}``. A document whose ``version`` is not 1 is a
  loud 400. Each SELECTED section is imported and its report collected; a section
  that fails imports nothing and carries its error, and a selected section that
  is unknown (not registered) or absent from the document carries an error too —
  none of these is a transport error, because the request itself is well-formed.
  The response is HTTP 200 ``{"data": {"ok": <bool>, "sections": {name: report}}}``;
  ``access_control`` surfaces its freshly-minted keys in its section report.

``list_sections`` and ``import_backup`` are thin adapters over operations in
``tai42_skeleton.operations.backup`` (import's envelope shape is validated here at
the HTTP edge). The export door is a downloadable attachment — the raw document
(not a ``{"data": ...}`` envelope) so a saved ``.json`` file is exactly what
``import`` consumes — so it stays a handler. Success bodies for the other doors
are ``{"data": ...}``; failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from tai42_contract.app import tai42_app

from tai42_skeleton.app.http import http_surface
from tai42_skeleton.app.route_registry import DeclaredRouteMetadata
from tai42_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai42_skeleton.operations.backup import _DOCUMENT_VERSION, _maybe_await, _registered_section_names
from tai42_skeleton.operations.backup import import_backup as _import_backup_op
from tai42_skeleton.operations.backup import list_sections as _list_sections_op

logger = logging.getLogger(__name__)


class BackupExport(BaseModel):
    """Export request — the backup section names to include."""

    sections: list[str]


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _require_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return value


list_sections = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_sections_op),
    path="/api/backup/sections",
    method="GET",
    action="read",
)


@http_surface().custom_route(
    "/api/backup/export",
    methods=["POST"],
    summary="Export the named backup sections",
    tags=["backup"],
    request_model=BackupExport,
    response_model=None,
    action="fenced",
    declared=DeclaredRouteMetadata(
        reload_gated=False,
        reads_body=True,
        error_statuses=(400, 401, 500),
        success_status=200,
    ),
)
async def export_backup(request: Request) -> Response:
    try:
        body = await request.json()
    except ValueError:
        return _error("invalid JSON body", 400)
    if not isinstance(body, dict):
        return _error("body must be a JSON object", 400)
    requested = _require_string_list(body.get("sections"))
    if requested is None:
        return _error("body must contain a list of section-name strings 'sections'", 400)

    registered = _registered_section_names()
    unknown = [name for name in requested if name not in registered]
    if unknown:
        return _error(f"unknown section(s): {', '.join(sorted(unknown))}", 400)

    sections: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name in requested:
        try:
            payload = await _maybe_await(tai42_app.backup.export_section(name))
        except Exception as exc:  # a section whose subsystem is absent — record, don't 500
            # Still surfaced in the returned report; also logged so a genuine
            # exporter code bug is visible server-side, not just an absent
            # subsystem indistinguishable from it.
            logger.warning("backup export of section %r failed: %s", name, exc, exc_info=True)
            errors[name] = str(exc)
            continue
        sections[name] = payload

    document = {
        "version": _DOCUMENT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "sections": sections,
        "errors": errors,
    }
    filename = f"tai-backup-{datetime.now(UTC).strftime('%Y-%m-%dT%H-%M-%SZ')}.json"
    return JSONResponse(document, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


async def _extract_import(request: Request) -> dict:
    """Parse + structurally validate the import body into the operation's flat
    ``document``/``sections`` arguments, rejecting a malformed envelope with a loud
    400 before the operation runs (the document CONTENT — version, its sections map
    — is the operation's own validation)."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object") from None
    document = body.get("document")
    if not isinstance(document, dict):
        raise BadRequestError("body must contain the export document object 'document'") from None
    selected = _require_string_list(body.get("sections"))
    if selected is None:
        raise BadRequestError("body must contain a list of section-name strings 'sections'") from None
    return {"document": document, "sections": selected}


import_backup = register_operation_route(
    tai42_app,
    operation_metadata_of(_import_backup_op),
    path="/api/backup/import",
    method="POST",
    context_extractor=_extract_import,
    action="fenced",
)
