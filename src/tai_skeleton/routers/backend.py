"""HTTP surface for backend identity + the fleet doors (all AUTHED).

- ``GET /api/backend`` — backend identity (or ``present: false``, 200, so the UI
  can render the empty state without treating it as an error).
- ``GET /api/fleet/workers`` — the live worker fleet from the bus presence census.
- ``POST /api/fleet/reload-config`` — soft-restart all (or ``targets``) workers.

``list_workers`` carries no try/except: a presence-store failure must surface as a
500, never an empty ``[]``. The reload door applies locally then broadcasts and
embeds the per-origin fleet report; a failed LOCAL apply re-surfaces as a 500 with
the report attached, while a sibling that fails/goes missing is reported in the
embedded fleet report (and logged), never raised.

All three doors are thin adapters over operations in
``tai_skeleton.operations.backend``. The reload ``targets`` list is a body input the
route validates itself with typed 400s, so it is parsed here at the HTTP edge
rather than by the adapter's plain request-model parse. Success bodies are
``{"data": ...}``; failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from tai_contract.app import tai_app

from tai_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai_skeleton.operations.backend import backend_info as _backend_info_op
from tai_skeleton.operations.backend import fleet_reload_config as _fleet_reload_config_op
from tai_skeleton.operations.backend import list_workers as _list_workers_op

backend_info = register_operation_route(
    tai_app,
    operation_metadata_of(_backend_info_op),
    path="/api/backend",
    method="GET",
)

list_workers = register_operation_route(
    tai_app,
    operation_metadata_of(_list_workers_op),
    path="/api/fleet/workers",
    method="GET",
)


async def _reload_targets(request: Request) -> dict[str, Any]:
    """Parse the reload-config body into the operation's flat ``targets`` argument,
    rejecting a malformed body / non-string-list targets with a loud 400 before the
    operation runs."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object") from None
    targets = body.get("targets")
    if targets is not None and (not isinstance(targets, list) or not all(isinstance(t, str) for t in targets)):
        raise BadRequestError("'targets' must be a list of worker names") from None
    return {"targets": targets}


reload_config = register_operation_route(
    tai_app,
    operation_metadata_of(_fleet_reload_config_op),
    path="/api/fleet/reload-config",
    method="POST",
    context_extractor=_reload_targets,
)
