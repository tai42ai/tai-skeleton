"""Scheduling operations — a thin skin over the run-tool seam that reports honestly
when no scheduling backend is installed.

* ``list_schedules`` lists schedules via ``backend_list_schedules``.
* ``server_datetime`` reports the server clock via the ``current_time_info`` toolbox
  tool, available independently of any scheduling backend.
* ``create_schedule`` schedules a caller-chosen tool run (``tool_name`` +
  ``tool_kwargs`` + ``schedule_kwargs``; schedule keys win on collision).
* ``delete_schedule`` removes a schedule by name via ``backend_delete_schedule``.

Availability is detected at CALL time, never probed at import: when no installed
backend registers the scheduling marker tools the list/create/delete ops raise a loud
:class:`NotSupportedError` (501) rather than answering an empty list;
``server_datetime`` raises its own 501 when ``current_time_info`` is absent,
independent of the scheduling backend. An unknown caller-named tool on create is a
:class:`NotFoundError` (404). ``create_schedule`` and ``delete_schedule`` mutate live
scheduling state, so they are ``destructive`` (create) / a DELETE (delete).

The schedule's authorization is checked at schedule CREATION (setting the schedule
up); its later recurring firing has no live caller and runs anonymous/system, exactly
as every scheduled job does today — the HTTP middleware / tool-edge authz are
untouched by this extraction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from tai42_contract.app import tai42_app

from tai42_skeleton.operations import BadRequestError, NotFoundError, NotSupportedError, operation
from tai42_skeleton.tools.binding import is_unknown_tool_error

# The tools an installed scheduling backend registers; their presence is the marker
# that scheduling is available. ``run_tool`` on the caller's target tool also raises
# the same unknown-tool error, so create() distinguishes by name.
_MARKER_TOOLS = ("backend_list_schedules", "backend_delete_schedule")
_NO_BACKEND_MESSAGE = "no installed backend exposes scheduling tools"
_TIME_TOOL = "current_time_info"


class ScheduleCreate(BaseModel):
    """Create a schedule that periodically runs ``tool_name`` with ``tool_kwargs``
    on the cadence in ``schedule_kwargs``."""

    tool_name: str = Field(min_length=1)
    tool_kwargs: dict[str, Any] = {}
    schedule_kwargs: dict[str, Any] = {}


def _is_unknown_tool_error(exc: RuntimeError, tool_name: str) -> bool:
    """Whether ``exc`` is the run-tool seam's unknown-tool error for ``tool_name``.

    Recognizes the typed ``UnknownToolError`` the binding raises AND, defensively,
    the legacy ``RuntimeError("No such tool: ...")`` message — so an unknown-tool
    failure is told apart from any other RuntimeError (which must still surface
    loudly) whichever shape the binding raises."""
    return is_unknown_tool_error(exc, tool_name)


async def _scheduling_backend_present() -> bool:
    """Whether an installed backend registers the scheduling marker tools."""
    tools = await tai42_app.tools.get_tools()
    return all(name in tools for name in _MARKER_TOOLS)


@operation(summary="List schedules", tags=["schedules"], errors=[NotSupportedError])
async def list_schedules() -> Any:
    if not await _scheduling_backend_present():
        raise NotSupportedError(_NO_BACKEND_MESSAGE)
    return await tai42_app.tools.run_tool("backend_list_schedules", {})


@operation(summary="Get the server date and time", tags=["schedules"], errors=[NotSupportedError])
async def server_datetime() -> Any:
    try:
        return await tai42_app.tools.run_tool(_TIME_TOOL, {})
    except RuntimeError as exc:
        if _is_unknown_tool_error(exc, _TIME_TOOL):
            raise NotSupportedError(f"{_TIME_TOOL} tool is not available") from exc
        raise


@operation(
    summary="Create a schedule",
    tags=["schedules"],
    destructive=True,
    reload_gated=True,
    errors=[BadRequestError, NotFoundError, NotSupportedError],
    request_model=ScheduleCreate,
)
async def create_schedule(tool_name: str, tool_kwargs: dict[str, Any], schedule_kwargs: dict[str, Any]) -> Any:
    """Schedule a caller-named tool to run on a cadence — a run-ANY-tool door.

    The caller supplies ``tool_name``, so reaching this is arbitrary-tool-execution
    privilege (the recurring firing runs the named tool with real side effects)."""
    if not await _scheduling_backend_present():
        raise NotSupportedError(_NO_BACKEND_MESSAGE)
    # Schedule keys win on collision so the backend's scheduling parameters cannot be
    # shadowed by the tool's own arguments.
    arguments: dict[str, Any] = {**tool_kwargs, **schedule_kwargs}
    try:
        return await tai42_app.tools.run_tool(tool_name, arguments)
    except RuntimeError as exc:
        if _is_unknown_tool_error(exc, tool_name):
            raise NotFoundError(f"unknown tool: {tool_name}") from exc
        raise


@operation(summary="Delete a schedule", tags=["schedules"], reload_gated=True, errors=[NotSupportedError])
async def delete_schedule(schedule_name: str) -> Any:
    if not await _scheduling_backend_present():
        raise NotSupportedError(_NO_BACKEND_MESSAGE)
    return await tai42_app.tools.run_tool("backend_delete_schedule", {"name": schedule_name})
