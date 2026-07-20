"""HTTP routes for the agents surface — ``/api/agents`` (all AUTHED).

Four doors, bound to the :class:`~tai42_contract.agent.Agent` CONTRACT only (never
to a concrete agent implementation):

* ``GET /api/agents`` — every registered agent as ``{name, description,
  tool_name, input_schema, spec_runnable}``. ``input_schema`` is the agent's
  ``ToolInput`` JSON schema — the SAME source the agent-binding derives the
  auto-generated ``run`` tool from, so the list schema and the run-tool schema
  never diverge. ``spec_runnable`` is the agent's own capability marker (read, not
  inferred): ``True`` means baking the composable spec fields as fixed kwargs
  yields an authored agent.
* ``GET /api/agents/spec-runnable`` — only the authorable agents (``spec_runnable``
  is ``True``), the same shape as the list door. The compose UI's base-agent picker
  consumes exactly this; an empty list means no authoring is possible.
* ``POST /api/agents/{name}/runs`` — the agent's input (validated against that
  same ``ToolInput`` model; 400 on mismatch, 404 on an unknown agent). The
  validated input is mapped to ``run`` kwargs through the agent's
  ``from_tool_input`` — the SAME mapping the binding applies — and driven through
  ``agent.astream``. The response is an SSE stream: one frame per
  :class:`~tai42_contract.agent.events.StreamEvent`, a terminal ``stream.end`` when
  the iterator completes, or a terminal ``stream.error`` (and a logged traceback)
  when the agent raises. A client disconnect cancels the underlying run so an
  abandoned request never keeps executing.
* ``POST /api/agents/authored/{name}/runs`` — stream a run of an AUTHORED agent (a
  preset over an agent's run tool whose baked fields the agent honors). The baked spec
  is resolved from
  the ``PresetManager`` in-memory map (the one source of truth for every registered
  preset). The request supplies only the non-baked
  ``ToolInput`` fields; naming a baked field is a loud 400 (never a silent
  override). The baked spec plus the request fields are combined, validated against
  ``ToolInput``, mapped through ``from_tool_input``, and driven through
  ``astream`` — the SAME field-combine → validate → map path the tool face applies.
  The SSE framing is identical to the plain run door.

The two list doors are thin adapters over operations in
``tai42_skeleton.operations.agents``; the two run doors are SSE streams
(transport-shaped) that stay handlers. Success bodies are ``{"data": {...}}``;
failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from pydantic import RootModel, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from tai42_contract.agent import Agent
from tai42_contract.app import tai42_app
from tai42_contract.presets.errors import PresetNotFoundError

from tai42_skeleton.app import instance
from tai42_skeleton.app.http import http_surface
from tai42_skeleton.app.reload_gate import reload_gate
from tai42_skeleton.app.route_registry import DeclaredRouteMetadata
from tai42_skeleton.operations import agents as agent_ops
from tai42_skeleton.operations import operation_metadata_of, register_operation_route
from tai42_skeleton.operations.agents import list_agents as _list_agents_op
from tai42_skeleton.operations.agents import list_spec_runnable_agents as _list_spec_runnable_agents_op

logger = logging.getLogger(__name__)

# Idle SSE keep-alive cadence — mirrors the interactions stream so a proxy does
# not drop a connection while an agent thinks between events.
_KEEPALIVE_SECONDS = 15
_KEEPALIVE_FRAME = ": keepalive\n\n"

# How often the disconnect monitor checks whether the client is gone. A dropped
# client cancels the underlying run within this window even mid-stream, so an
# abandoned run never keeps executing.
_DISCONNECT_POLL_SECONDS = 1
# Bound the producer→consumer buffer so a fast agent feeding a stalled client
# applies backpressure (``_produce`` awaits ``queue.put``, which blocks when full)
# instead of accumulating events in memory until the disconnect poll fires.
_MAX_QUEUED_EVENTS = 256

# Terminal frames. ``json.dumps`` (not a hand-written literal) so the payloads
# stay well-formed and any interpolated value is escaped.
_END_FRAME = json.dumps({"type": "stream.end"})

# Same headers the interactions stream route sets: no caching, keep-alive, and
# the nginx no-buffering hint so frames flush as they are produced.
_STREAM_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


class AgentRunInput(RootModel[dict[str, object]]):
    """The input object for an agent run. Fields are agent-specific — validated at
    runtime against the target agent's dynamic ``ToolInput`` model — so the request
    body is a free-form JSON object here."""


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _validation_detail(exc: ValidationError) -> str:
    """Render a validation failure as ``loc: message`` parts WITHOUT echoing the
    input values. The authored-run input embeds the baked ``fixed_kwargs`` — which
    can carry credentials the runner is not entitled to see — so the surfaced error
    omits the input values that ``str(exc)`` would otherwise include."""
    parts: list[str] = []
    for err in exc.errors(include_url=False, include_input=False):
        loc = ".".join(str(part) for part in err["loc"])
        parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return "; ".join(parts)


def _sse(data: str) -> str:
    """One SSE frame carrying ``data``. The event ``type`` is the client's
    discriminator and rides inside the JSON, so no ``event:`` line is emitted."""
    return f"data: {data}\n\n"


def _error_frame(exc: BaseException) -> str:
    return _sse(json.dumps({"type": "stream.error", "message": str(exc)}))


# -- list routes (operation adapters) ----------------------------------------

list_agents = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_agents_op),
    path="/api/agents",
    method="GET",
)

list_spec_runnable_agents = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_spec_runnable_agents_op),
    path="/api/agents/spec-runnable",
    method="GET",
)


# -- run route (SSE) ---------------------------------------------------------


async def _produce(agent: Agent, run_kwargs: dict[str, Any], queue: asyncio.Queue[tuple[str, Any]]) -> None:
    """Drain ``agent.astream`` into ``queue`` as ``(kind, payload)`` items:
    ``("event", StreamEvent)`` per event, then one terminal ``("end", None)`` or
    ``("error", exc)``. A cancellation (client disconnect) propagates into
    ``astream`` and re-raises so the abandoned run stops."""
    try:
        async for event in agent.astream(**run_kwargs):
            await queue.put(("event", event))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Surfaced to the client as a terminal stream.error frame and logged with
        # its traceback where it is drained — never a silent close.
        await queue.put(("error", exc))
    else:
        await queue.put(("end", None))


async def _wait_until_disconnected(request: Request) -> None:
    """Complete once the client has disconnected, polling on a fixed cadence.
    ``is_disconnected`` is cheap (a non-blocking receive peek), so a poll loop is
    both correct and low-cost."""
    while not await request.is_disconnected():
        await asyncio.sleep(_DISCONNECT_POLL_SECONDS)


async def _agent_event_stream(request: Request, agent: Agent, run_kwargs: dict[str, Any]) -> AsyncIterator[str]:
    """Yield SSE frames for one agent run: one frame per ``StreamEvent`` (via
    ``model_dump_json(fallback=str)`` so a live non-JSON-native payload serializes
    instead of crashing the stream), a keep-alive comment on an idle gap, and a
    terminal ``stream.end``/``stream.error`` frame.

    A disconnect monitor races the event feed; when the client drops, the run's
    producer task is cancelled in ``finally``, which propagates cancellation into
    ``astream`` so the abandoned run stops."""
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=_MAX_QUEUED_EVENTS)
    producer = asyncio.ensure_future(_produce(agent, run_kwargs, queue))
    monitor = asyncio.ensure_future(_wait_until_disconnected(request))
    # A persistent get-task across keep-alive timeouts: cancelling and re-issuing
    # a fresh ``queue.get()`` each idle tick could drop an item that arrived at the
    # cancellation boundary, so the same task is kept until it resolves.
    get_task: asyncio.Future[tuple[str, Any]] | None = None
    try:
        while True:
            if get_task is None:
                get_task = asyncio.ensure_future(queue.get())
            done, _pending = await asyncio.wait(
                {get_task, monitor}, timeout=_KEEPALIVE_SECONDS, return_when=asyncio.FIRST_COMPLETED
            )
            if monitor in done:
                # Client gone — stop; ``finally`` cancels the run.
                return
            if get_task not in done:
                yield _KEEPALIVE_FRAME
                continue
            kind, payload = get_task.result()
            get_task = None
            if kind == "event":
                yield _sse(payload.model_dump_json(fallback=str))
            elif kind == "end":
                yield _sse(_END_FRAME)
                return
            else:
                logger.error("agent run failed while streaming the response", exc_info=payload)
                yield _error_frame(payload)
                return
    finally:
        for task in (get_task, monitor, producer):
            if task is not None:
                task.cancel()
        for task in (get_task, monitor, producer):
            if task is not None:
                with suppress(asyncio.CancelledError):
                    await task


@http_surface().custom_route(
    "/api/agents/{name}/runs",
    methods=["POST"],
    summary="Stream a run of an agent",
    tags=["agents"],
    request_model=AgentRunInput,
    response_model=None,
    declared=DeclaredRouteMetadata(
        reload_gated=True,
        reads_body=True,
        error_statuses=(400, 401, 404, 503),
        success_status=200,
    ),
)
async def run_agent(request: Request) -> Response:
    # An agent run dispatched against registries a reload is tearing down would
    # race the rebuild — reject while the gate is held (retriable).
    if reload_gate.locked:
        return reload_gate.reject_response()
    name = request.path_params["name"]
    agent = agent_ops._agents_registry().get(name)
    if agent is None:
        return _error(f"no such agent: {name!r}", 404)

    try:
        body = await request.json()
    except ValueError:
        return _error("invalid JSON body", 400)
    if not isinstance(body, dict):
        return _error("body must be a JSON object of agent input", 400)
    # Reject a body key that is not a ``ToolInput`` field: pydantic's default
    # ``extra="ignore"`` would silently drop a typo'd field and run with its
    # default, so a loud 400 names the offending key(s) instead.
    unknown = sorted(set(body) - set(agent.ToolInput.model_fields))
    if unknown:
        return _error(f"unknown agent input field(s): {', '.join(unknown)}", 400)
    # Validate, then apply the same validated-input -> run-kwargs mapping the binding
    # applies before calling the agent (a raw pass-through diverges for any agent whose
    # ``from_tool_input`` renames or derives kwargs). ``from_tool_input`` raises a
    # ``ValueError`` on a genuinely conflicting input (e.g. two fields that map to the
    # same run kwarg) — surfaced as a loud 400, never a silent drop.
    try:
        validated = agent.ToolInput(**body)
        run_kwargs = agent.from_tool_input(validated)
    except ValidationError as exc:
        return _error(f"invalid agent input: {exc}", 400)
    except ValueError as exc:
        return _error(f"invalid agent input: {exc}", 400)
    return StreamingResponse(
        _agent_event_stream(request, agent, run_kwargs),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )


# -- authored-agent run route (SSE) ------------------------------------------


@http_surface().custom_route(
    "/api/agents/authored/{name}/runs",
    methods=["POST"],
    summary="Stream a run of an authored agent",
    tags=["agents"],
    request_model=AgentRunInput,
    response_model=None,
    declared=DeclaredRouteMetadata(
        reload_gated=True,
        reads_body=True,
        error_statuses=(400, 401, 404, 503),
        success_status=200,
    ),
)
async def run_authored_agent(request: Request) -> Response:
    """Stream a run of an authored agent — a preset over an agent's run tool whose
    baked fields the agent honors.

    The baked spec is read from the ``PresetManager`` in-memory map (the single
    source of truth for every registered preset — versioning never gates
    streamability). The request body supplies ONLY the remaining,
    non-baked ``ToolInput`` fields; naming a baked field is a loud 400 rather than a
    silent override. The baked fixed kwargs plus the request fields are combined at
    the FIELD level, validated against the agent's ``ToolInput``, and mapped through
    ``from_tool_input`` before ``astream`` — never a raw splat that would bypass
    validation and the field mapping."""
    # An agent run dispatched against registries a reload is tearing down would
    # race the rebuild — reject while the gate is held (retriable).
    if reload_gate.locked:
        return reload_gate.reject_response()
    name = request.path_params["name"]
    try:
        spec = instance.app.preset_manager.get_spec(name)
    except PresetNotFoundError:
        return _error(f"no such authored agent: {name!r}", 404)

    # An authored agent is a preset whose base_tool is an agent's run tool (bound
    # under the agent's registration name). A preset over a plain tool is not
    # streamable through this door.
    agent = agent_ops._agents_registry().get(spec.base_tool)
    if agent is None:
        return _error(f"{name!r} is a tool preset, not an authored agent", 400)

    try:
        body = await request.json()
    except ValueError:
        return _error("invalid JSON body", 400)
    if not isinstance(body, dict):
        return _error("body must be a JSON object of agent input", 400)

    # Reject a body key that is not a ``ToolInput`` field: pydantic's default
    # ``extra="ignore"`` would silently drop a typo'd field, so name it in a loud
    # 400 instead.
    unknown = sorted(set(body) - set(agent.ToolInput.model_fields))
    if unknown:
        return _error(f"unknown agent input field(s): {', '.join(unknown)}", 400)

    # The baked spec fields are FIXED (baked as hidden, non-overridable constants).
    # A request that names one is rejected loudly — baked never silently wins over a
    # request key, and a request never silently loses to a baked key.
    baked = spec.fixed_kwargs
    overridden = sorted(set(body) & set(baked))
    if overridden:
        return _error(f"cannot override the fixed field {overridden[0]!r} baked into this authored agent", 400)

    # Field-combine (baked spec + remaining request fields) THEN validate THEN map —
    # the same order the tool face applies. The baked fixed kwargs are authoritative
    # and the request fills only the remaining fields.
    try:
        validated = agent.ToolInput(**{**body, **baked})
        run_kwargs = agent.from_tool_input(validated)
    except ValidationError as exc:
        return _error(f"invalid agent input: {_validation_detail(exc)}", 400)
    except ValueError as exc:
        # ``from_tool_input`` rejects a conflicting input (e.g. a request field that
        # maps to the same run kwarg as a baked field) — a loud 400, never a silent
        # override. The message is hand-authored (no input values), so it is safe to
        # surface verbatim.
        return _error(f"invalid agent input: {exc}", 400)
    return StreamingResponse(
        _agent_event_stream(request, agent, run_kwargs),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )
