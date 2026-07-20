"""Tool-surface operations — ``/api/tools*`` and ``/api/run-tool``, plus the live
app-management tools (run, reload, and remove a tool).

Reads:

* ``list_tools`` — the sorted registered tool names.
* ``tool_tags`` — the per-tool native-``tags`` map.
* ``tool_schema`` — one tool's input/output/description (unknown name → 404).
* ``tools_schema`` — the same schema view for every tool, keyed by name.

Mutations (each applied on this worker and broadcast to the fleet over the bus):

* ``run_tool`` — execute an arbitrary registered tool with REAL side effects. A
  "run any tool by name" META-EXECUTOR: hardcode-blocked from the MCP surface
  (``meta_executor=True``, tier 1) and admin-fenced at the HTTP edge; it is reachable
  only as the route + ``tai tools run`` CLI + internal dispatch. Its argument is
  ``tool_name`` (the route's request-model shape).
* ``reload_tool`` — re-register one app tool from its stored definition.
* ``remove_tool`` — remove one app tool from the live registry.

``run_tool`` / ``reload_tool`` / ``remove_tool`` mutate the live registry, so they are
``destructive`` and honor the reload gate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from tai_contract.app import tai_app

from tai_skeleton.operations import BadRequestError, NotFoundError, OperationFailed, operation
from tai_skeleton.operations._broadcast import broadcast
from tai_skeleton.tools.binding import UnknownToolError, is_unknown_tool_error

if TYPE_CHECKING:
    from fastmcp.tools import Tool

logger = logging.getLogger(__name__)


class RunToolRequest(BaseModel):
    """A synchronous tool-run request: the ``tool_name`` and its keyword
    ``arguments``. Mirrors the shape ``read_tool_call`` enforces at runtime."""

    tool_name: str = Field(min_length=1, description="Registered tool name.")
    arguments: dict[str, object] = Field(default_factory=dict, description="Tool keyword arguments.")


class ToolReloadRequest(BaseModel):
    """Re-register or remove one app tool by ``kind`` and ``name``, optionally
    restricting the fleet fan-out to specific ``targets``."""

    kind: str = Field(min_length=1, description='The tool kind (e.g. "flow").')
    name: str = Field(min_length=1, description="The tool name.")
    targets: list[str] | None = Field(default=None, description="Workers to restrict the fan-out to.")


def _tool_schema(tool: Tool) -> dict[str, object]:
    """The input/output/description view of a single tool."""
    return {
        "input": tool.parameters,
        "output": tool.output_schema,
        "description": tool.description,
    }


@operation(summary="List the registered tool names", tags=["tools"])
async def list_tools() -> list[str]:
    tools = await tai_app.tools.get_tools()
    return sorted(tools.keys())


@operation(summary="List each tool's native tags", tags=["tools"])
async def tool_tags() -> list[dict]:
    """The per-tool native-``tags`` map — one ``{name, tags}`` entry per registered
    tool, ``tags`` sorted for a stable wire order. Additive to the flat names
    contract; a tool with no tags carries an empty list."""
    tools = await tai_app.tools.get_tools()
    return [{"name": name, "tags": sorted(tool.tags)} for name, tool in sorted(tools.items())]


@operation(summary="Get one tool's input/output schema", tags=["tools"], errors=[NotFoundError])
async def tool_schema(tool_name: str) -> dict:
    tools = await tai_app.tools.get_tools()
    tool = tools.get(tool_name)
    if tool is None:
        raise NotFoundError(f"Tool {tool_name!r} not registered")
    return _tool_schema(tool)


@operation(summary="Get the input/output schema of every tool", tags=["tools"])
async def tools_schema() -> dict:
    tools = await tai_app.tools.get_tools()
    # A schema view needs no callable body, so every registered tool's schema is served
    # here — identical to the per-tool route, which 404s only an unknown name and serves
    # any registered tool's schema. A tool whose ``fn`` is ``None`` is a registry-health
    # signal (a registered name with no backing implementation), so it is logged once
    # per response, never silently dropped.
    missing_impl = [name for name, tool in tools.items() if getattr(tool, "fn", None) is None]
    if missing_impl:
        logger.warning(
            "tools-schema: %d registered tool(s) have no implementation: %s", len(missing_impl), missing_impl
        )
    return {name: _tool_schema(tool) for name, tool in tools.items()}


@operation(
    summary="Run a registered tool synchronously",
    tags=["tools"],
    destructive=True,
    reload_gated=True,
    meta_executor=True,
    errors=[BadRequestError, NotFoundError, OperationFailed],
    request_model=RunToolRequest,
)
async def run_tool(tool_name: str, arguments: dict[str, object]) -> Any:
    """Execute an arbitrary registered tool with REAL side effects.

    Reaching this route is full-execution privilege — the Studio key runs any
    registered tool. Per-tool scoped keys are not supported.
    """
    # Resolve the name first: an unknown tool is a loud 404 (matching the schema route),
    # told apart from a tool that raises DURING execution — which becomes a structured
    # 500 carrying the caught error, never an opaque "Internal Server Error". Dual-catch
    # the typed ``UnknownToolError`` AND the legacy ``RuntimeError("No such tool: ...")``
    # message so a binding rewrite that drops the typed error cannot silently turn the
    # 404 back into a raw 500.
    try:
        await tai_app.tools.get_tool(tool_name)
    except RuntimeError as exc:
        if is_unknown_tool_error(exc, tool_name):
            raise NotFoundError(f"unknown tool: {tool_name}") from exc
        raise
    try:
        return await tai_app.tools.run_tool(tool_name, arguments, offload_sync=True)
    except UnknownToolError as exc:
        # The tool was resolved above but vanished before the run (a concurrent
        # reload) — still an unknown-tool 404, not a masked 500.
        raise NotFoundError(f"unknown tool: {tool_name}") from exc
    except Exception as exc:
        logger.exception("run-tool %s raised during execution", tool_name)
        raise OperationFailed(str(exc)) from exc


@operation(
    summary="Reload one app tool from its stored definition",
    tags=["tools"],
    destructive=True,
    reload_gated=True,
    request_model=ToolReloadRequest,
)
async def reload_tool(kind: str, name: str, targets: list[str] | None = None) -> Any:
    """Re-register one app tool (e.g. kind "flow") from its current stored definition.

    Applied on this worker and broadcast to the fleet (all workers, or only
    ``targets``); each worker re-reads the definition itself, so the op carries only
    the kind and name. The response embeds the per-origin fleet report.
    """
    return await broadcast(
        {"op": "reload_tool", "kind": kind, "name": name},
        targets,
        lambda: tai_app.admin.run_tool_reload(kind, "reload", name),
    )


@operation(
    summary="Remove one app tool from the live registry",
    tags=["tools"],
    destructive=True,
    reload_gated=True,
    request_model=ToolReloadRequest,
)
async def remove_tool(kind: str, name: str, targets: list[str] | None = None) -> Any:
    """Remove one app tool (e.g. kind "flow") from the live registry.

    Applied on this worker and broadcast to the fleet (all workers, or only
    ``targets``); the response embeds the per-origin fleet report.
    """
    return await broadcast(
        {"op": "remove_tool", "kind": kind, "name": name},
        targets,
        lambda: tai_app.admin.run_tool_reload(kind, "remove", name),
    )
