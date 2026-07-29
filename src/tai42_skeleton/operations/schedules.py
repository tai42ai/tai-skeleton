"""Scheduling operations — a thin skin over the run-tool seam, reporting honestly when
no scheduling backend is installed.

Availability is detected at CALL time, never probed at import: list/create/delete
pre-check that an installed backend registers the marker tools (``_MARKER_TOOLS``) and
raise :class:`NotSupportedError` (501) when it does not. ``server_datetime`` has no
pre-check — it dispatches ``current_time_info`` and learns of its absence from the
dispatch itself, so its 501 is independent of the scheduling backend. An unknown
caller-named tool on create is :class:`NotFoundError` (404) instead.

Every door dispatches a NAMED inner tool and wraps that dispatch identically:

* an :class:`~tai42_skeleton.tools.binding.UnknownToolError` naming the tool the door
  itself asked for is that tool's absence — the door's own verdict (501 for
  list/delete/server-datetime, 404 for create);
* an ``UnknownToolError`` naming a DIFFERENT tool escaped the running tool's own body —
  a structured :class:`OperationFailed` (500);
* a typed :class:`OperationError` (most sharply ``PermissionDenied``) passes through as
  the answer it already is;
* any other exception becomes a structured ``OperationFailed`` (500), never an opaque
  "Internal Server Error".

Only list's and delete's absent-marker-tool branch logs (``logger.warning``): the
marker passed the presence pre-check moments earlier, so failing to resolve at dispatch
is an ANOMALY worth a trace, and the caller sees only a plain 501. server-datetime's 501
and create's 404 stay silent — an uninstalled toolbox extra and an unregistered
caller-named tool are both steady-state/ordinary, and logging either would repeat every
request. Both 500 branches always ``logger.exception``.

``UnavailableError`` (503) on every door: the tool-dispatch seam — and for create,
``authorize_submitted_tool`` — refuses mid-rebuild with the retriable
``OperationSurfaceUnsettledError``.

These doors are authed but NOT admin-fenced: an UNTYPED failure's exception text never
reaches the caller (it can carry internal detail, e.g. a dialled host:port) and stays
server-side in the log — the caller gets only the exception CLASS. A typed
``OperationError`` is the deliberate exception: the tool raised it AS the client-facing
answer, so its message passes through untouched. The inner tool NAME on a mismatched
``UnknownToolError`` IS reported — it is always a registry identifier (never free text
from the caught exception, never third-party data), so naming it is a bounded, accepted
disclosure. The admin-fenced ``run_tool`` door keeps the full error message; these
doors do not.

The caller-named tool's authorization is decided at schedule CREATION, run through the
full tool-edge decision against the live submitter (a fenced/secret target is
admin-only) — the later recurring firing has no live caller and runs anonymous/system,
so creation is the only edge the inner tool reaches.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from tai42_contract.app import tai42_app

from tai42_skeleton.operations import (
    BadRequestError,
    NotFoundError,
    NotSupportedError,
    OperationError,
    OperationFailed,
    PermissionDenied,
    UnavailableError,
    operation,
)
from tai42_skeleton.operations._submitted_tool_authz import authorize_submitted_tool
from tai42_skeleton.tools.binding import UnknownToolError

logger = logging.getLogger(__name__)

# The tools an installed scheduling backend registers; their presence is the marker
# that scheduling is available.
_LIST_TOOL = "backend_list_schedules"
_DELETE_TOOL = "backend_delete_schedule"
_MARKER_TOOLS = (_LIST_TOOL, _DELETE_TOOL)
_NO_BACKEND_MESSAGE = "no installed backend exposes scheduling tools"
_TIME_TOOL = "current_time_info"


class ScheduleCreate(BaseModel):
    """Create a schedule that periodically runs ``tool_name`` with ``tool_kwargs``
    on the cadence in ``schedule_kwargs``."""

    tool_name: str = Field(min_length=1)
    tool_kwargs: dict[str, Any] = {}
    schedule_kwargs: dict[str, Any] = {}


async def _scheduling_backend_present() -> bool:
    """Whether an installed backend registers the scheduling marker tools."""
    tools = await tai42_app.tools.get_tools()
    return all(name in tools for name in _MARKER_TOOLS)


@operation(
    summary="List schedules",
    tags=["schedules"],
    errors=[NotSupportedError, PermissionDenied, UnavailableError, OperationFailed],
)
async def list_schedules() -> Any:
    if not await _scheduling_backend_present():
        raise NotSupportedError(_NO_BACKEND_MESSAGE)
    try:
        return await tai42_app.tools.run_tool(_LIST_TOOL, {})
    except UnknownToolError as exc:
        # Discriminate by NAME — see module docstring for the shared dispatch-wrap contract.
        if exc.tool_name == _LIST_TOOL:
            logger.warning(
                "list-schedules: %r passed the presence pre-check but did not resolve at dispatch; answering 501",
                _LIST_TOOL,
            )
            raise NotSupportedError(_NO_BACKEND_MESSAGE) from exc
        logger.exception("list-schedules %r raised unknown-tool %r during execution", _LIST_TOOL, exc.tool_name)
        raise OperationFailed(f"schedule listing failed (unknown tool {exc.tool_name})") from exc
    except OperationError:
        raise
    except Exception as exc:
        logger.exception("list-schedules %r raised during execution", _LIST_TOOL)
        raise OperationFailed(f"schedule listing failed ({type(exc).__name__})") from exc


@operation(
    summary="Get the server date and time",
    tags=["schedules"],
    errors=[NotSupportedError, PermissionDenied, UnavailableError, OperationFailed],
)
async def server_datetime() -> Any:
    try:
        return await tai42_app.tools.run_tool(_TIME_TOOL, {})
    except UnknownToolError as exc:
        if exc.tool_name == _TIME_TOOL:
            raise NotSupportedError(f"{_TIME_TOOL} tool is not available") from exc
        logger.exception("server-datetime %r raised unknown-tool %r during execution", _TIME_TOOL, exc.tool_name)
        raise OperationFailed(f"server-datetime lookup failed (unknown tool {exc.tool_name})") from exc
    except OperationError:
        raise
    except Exception as exc:
        logger.exception("server-datetime %r raised during execution", _TIME_TOOL)
        raise OperationFailed(f"server-datetime lookup failed ({type(exc).__name__})") from exc


@operation(
    summary="Create a schedule",
    tags=["schedules"],
    destructive=True,
    reload_gated=True,
    meta_executor=True,
    errors=[
        BadRequestError,
        NotFoundError,
        NotSupportedError,
        PermissionDenied,
        UnavailableError,
        OperationFailed,
    ],
    request_model=ScheduleCreate,
)
async def create_schedule(tool_name: str, tool_kwargs: dict[str, Any], schedule_kwargs: dict[str, Any]) -> Any:
    """Schedule a caller-named tool to run on a cadence — a run-ANY-tool door.

    The caller supplies ``tool_name``, so reaching this is arbitrary-tool-execution
    privilege (the recurring firing runs the named tool with real side effects). As a
    "run any tool by name" door it is a tier-1 meta-executor, never projected to the MCP
    surface — matching ``run_tool`` and ``submit_run``."""
    if not await _scheduling_backend_present():
        raise NotSupportedError(_NO_BACKEND_MESSAGE)
    # Schedule keys win on collision so the backend's scheduling parameters cannot be
    # shadowed by the tool's own arguments.
    arguments: dict[str, Any] = {**tool_kwargs, **schedule_kwargs}
    # The recurring firing has no live caller, so this creation is the ONLY edge the inner
    # tool reaches — decide it here, over the exact arguments the dispatch below fires.
    await authorize_submitted_tool(tool_name, arguments)
    try:
        return await tai42_app.tools.run_tool(tool_name, arguments)
    except UnknownToolError as exc:
        if exc.tool_name == tool_name:
            raise NotFoundError(f"unknown tool: {tool_name}") from exc
        logger.exception("create-schedule %r raised unknown-tool %r during execution", tool_name, exc.tool_name)
        raise OperationFailed(f"schedule creation failed (unknown tool {exc.tool_name})") from exc
    except OperationError:
        raise
    except Exception as exc:
        logger.exception("create-schedule %r raised during execution", tool_name)
        raise OperationFailed(f"schedule creation failed ({type(exc).__name__})") from exc


@operation(
    summary="Delete a schedule",
    tags=["schedules"],
    reload_gated=True,
    errors=[NotSupportedError, PermissionDenied, UnavailableError, OperationFailed],
)
async def delete_schedule(schedule_name: str) -> Any:
    if not await _scheduling_backend_present():
        raise NotSupportedError(_NO_BACKEND_MESSAGE)
    try:
        return await tai42_app.tools.run_tool(_DELETE_TOOL, {"name": schedule_name})
    except UnknownToolError as exc:
        if exc.tool_name == _DELETE_TOOL:
            logger.warning(
                "delete-schedule: %r passed the presence pre-check but did not resolve at dispatch; answering 501",
                _DELETE_TOOL,
            )
            raise NotSupportedError(_NO_BACKEND_MESSAGE) from exc
        logger.exception("delete-schedule %r raised unknown-tool %r during execution", _DELETE_TOOL, exc.tool_name)
        raise OperationFailed(f"schedule deletion failed (unknown tool {exc.tool_name})") from exc
    except OperationError:
        raise
    except Exception as exc:
        logger.exception("delete-schedule %r raised during execution", _DELETE_TOOL)
        raise OperationFailed(f"schedule deletion failed ({type(exc).__name__})") from exc
