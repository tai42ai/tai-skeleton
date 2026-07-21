"""HTTP routes for background tool runs — ``/api/tool-runs*`` (all AUTHED).

Three thin adapters over operations in ``tai42_skeleton.operations.tool_runs`` — no
run/supervisor/store logic lives here:

* ``POST /api/tool-runs`` — body ``{tool_name, arguments}`` (parsed by the same
  helper the sync door uses, at the HTTP edge here); returns ``202 {"data":
  {"run_id": ...}}`` at once and executes the tool as an in-process background
  task through the ``tai42_app.tools.run_tool`` seam.
* ``GET /api/tool-runs/{run_id}`` — the run record; an unknown/expired id is a
  loud 404, and a restricted caller reading another identity's run gets a loud
  403 (never a 404 — the run exists, it is simply not the caller's).
* ``GET /api/tool-runs?tool_name=...`` — the recent runs for one tool, newest
  first; a restricted caller reads its own per-identity slice, an unrestricted
  caller the full shared window.

The submit body's ``{tool_name, arguments}`` shape and the list's required
``tool_name`` query param are validated here at the HTTP edge with typed 400s
(producing the operation's flat arguments) rather than by the adapter's plain
request-model parse. Success bodies are ``{"data": ...}``; failures are
``{"error": "<message>"}``.
"""

from __future__ import annotations

from starlette.requests import Request
from tai42_contract.app import tai42_app

from tai42_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai42_skeleton.operations.tool_runs import get_run as _get_run_op
from tai42_skeleton.operations.tool_runs import list_tool_runs as _list_tool_runs_op
from tai42_skeleton.operations.tool_runs import submit_run as _submit_run_op
from tai42_skeleton.routers._tool_call import ToolCallRequestError, read_tool_call


async def _extract_submission(request: Request) -> dict:
    """Parse the tool-call body into the operation's flat ``tool_name``/``arguments``
    arguments, mapping the shared parser's loud ``ToolCallRequestError`` to the same
    explicit 400 (the adapter's plain parse would yield 422)."""
    try:
        tool_name, arguments = await read_tool_call(request)
    except ToolCallRequestError as exc:
        raise BadRequestError(exc.message) from exc
    return {"tool_name": tool_name, "arguments": arguments}


async def _extract_list_query(request: Request) -> dict:
    """Read the required ``tool_name`` query param into the operation's flat argument,
    rejecting its absence with the explicit 400 (never a GET body)."""
    tool_name = request.query_params.get("tool_name")
    if not tool_name:
        raise BadRequestError("query param 'tool_name' is required")
    return {"tool_name": tool_name}


submit_run = register_operation_route(
    tai42_app,
    operation_metadata_of(_submit_run_op),
    path="/api/tool-runs",
    method="POST",
    context_extractor=_extract_submission,
    success_status=202,
    action="write",
)

get_run = register_operation_route(
    tai42_app,
    operation_metadata_of(_get_run_op),
    path="/api/tool-runs/{run_id}",
    method="GET",
    action="read",
)

list_tool_runs = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_tool_runs_op),
    path="/api/tool-runs",
    method="GET",
    context_extractor=_extract_list_query,
    action="read",
)
