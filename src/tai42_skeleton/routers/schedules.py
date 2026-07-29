"""HTTP surface for the scheduling feature — the Studio's schedules UI doors.

Four doors, all AUTHED, thin adapters over operations in
``tai42_skeleton.operations.schedules``:

- ``GET /api/schedules`` — list schedules via ``backend_list_schedules``.
- ``GET /api/schedules/server-datetime`` — the server's current time via
  ``current_time_info`` (a toolbox tool, available independently of any scheduling
  backend).
- ``POST /api/schedules`` — schedule a caller-chosen tool run; the body names the
  ``tool_name`` and carries its ``tool_kwargs`` and the ``schedule_kwargs`` the
  backend's scheduling tool consumes (schedule keys win on collision).
- ``DELETE /api/schedules/{schedule_name}`` — remove a schedule by name via
  ``backend_delete_schedule``.

Availability is detected at REQUEST time, never probed at import. When no installed
backend registers the scheduling marker tools the list/create/delete doors answer a
loud 501; ``server-datetime`` reports its own 501 when ``current_time_info`` is
absent. Success bodies are ``{"data": ...}``; failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from tai42_contract.app import tai42_app

from tai42_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai42_skeleton.operations.schedules import create_schedule as _create_schedule_op
from tai42_skeleton.operations.schedules import delete_schedule as _delete_schedule_op
from tai42_skeleton.operations.schedules import list_schedules as _list_schedules_op
from tai42_skeleton.operations.schedules import server_datetime as _server_datetime_op


async def _extract_create(request: Request) -> dict[str, Any]:
    """Parse and validate the create body at the HTTP edge, preserving the door's
    hand-authored 400 messages (a plain request-model parse would answer 422 with a
    different shape). Yields the operation's flat ``tool_name`` / ``tool_kwargs`` /
    ``schedule_kwargs`` kwargs."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise BadRequestError("invalid JSON body") from exc
    if not isinstance(body, dict):
        raise BadRequestError("body must be a JSON object")

    tool_name = body.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        raise BadRequestError("body must contain a non-empty string 'tool_name'")
    tool_kwargs = body.get("tool_kwargs", {})
    if not isinstance(tool_kwargs, dict):
        raise BadRequestError("'tool_kwargs' must be a JSON object")
    schedule_kwargs = body.get("schedule_kwargs", {})
    if not isinstance(schedule_kwargs, dict):
        raise BadRequestError("'schedule_kwargs' must be a JSON object")
    return {"tool_name": tool_name, "tool_kwargs": tool_kwargs, "schedule_kwargs": schedule_kwargs}


list_schedules = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_schedules_op),
    path="/api/schedules",
    method="GET",
    action="read",
)

server_datetime = register_operation_route(
    tai42_app,
    operation_metadata_of(_server_datetime_op),
    path="/api/schedules/server-datetime",
    method="GET",
    action="read",
)

create_schedule = register_operation_route(
    tai42_app,
    operation_metadata_of(_create_schedule_op),
    path="/api/schedules",
    method="POST",
    context_extractor=_extract_create,
    action="write",
)

delete_schedule = register_operation_route(
    tai42_app,
    operation_metadata_of(_delete_schedule_op),
    path="/api/schedules/{schedule_name}",
    method="DELETE",
    action="write",
)
