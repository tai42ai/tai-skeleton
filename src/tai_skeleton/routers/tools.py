"""HTTP routes for the tool surface — ``/api/tools*`` and ``/api/run-tool``.

Thin adapters over operations in ``tai_skeleton.operations.tools``, all AUTHED:

* ``GET /api/tools`` — the sorted list of registered tool names.
* ``GET /api/tools/tags`` — the per-tool native-``tags`` map.
* ``GET /api/tools/{tool_name}/schema`` — one tool's input/output JSON schema.
* ``GET /api/tools-schema`` — the same schema view for every registered tool.
* ``POST /api/run-tool`` — execute an arbitrary registered tool with REAL side
  effects (full-execution privilege; hardcode-blocked from the MCP surface and
  admin-fenced at the HTTP edge).
* ``POST /api/tools/reload`` — re-register one app tool from its stored definition.
* ``POST /api/tools/remove`` — remove one app tool from the live registry.

Success bodies are ``{"data": ...}``; failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from tai_contract.app import tai_app

from tai_skeleton.operations import BadRequestError, operation_metadata_of, register_operation_route
from tai_skeleton.operations.tools import list_tools as _list_tools_op
from tai_skeleton.operations.tools import reload_tool as _reload_tool_op
from tai_skeleton.operations.tools import remove_tool as _remove_tool_op
from tai_skeleton.operations.tools import run_tool as _run_tool_op
from tai_skeleton.operations.tools import tool_schema as _tool_schema_op
from tai_skeleton.operations.tools import tool_tags as _tool_tags_op
from tai_skeleton.operations.tools import tools_schema as _tools_schema_op
from tai_skeleton.routers._tool_call import ToolCallRequestError, read_tool_call


async def _extract_run_tool(request: Request) -> dict[str, Any]:
    """Parse the run-tool body ``{tool_name, arguments}`` at the HTTP edge via the
    shared tool-call parser, mapping its loud 4xx to a typed error. Yields the
    operation's flat ``tool_name`` / ``arguments`` kwargs."""
    try:
        tool_name, arguments = await read_tool_call(request)
    except ToolCallRequestError as exc:
        raise BadRequestError(exc.message) from exc
    return {"tool_name": tool_name, "arguments": arguments}


list_tools = register_operation_route(
    tai_app,
    operation_metadata_of(_list_tools_op),
    path="/api/tools",
    method="GET",
)

tool_tags = register_operation_route(
    tai_app,
    operation_metadata_of(_tool_tags_op),
    path="/api/tools/tags",
    method="GET",
)

tool_schema = register_operation_route(
    tai_app,
    operation_metadata_of(_tool_schema_op),
    path="/api/tools/{tool_name}/schema",
    method="GET",
)

tools_schema = register_operation_route(
    tai_app,
    operation_metadata_of(_tools_schema_op),
    path="/api/tools-schema",
    method="GET",
)

run_tool = register_operation_route(
    tai_app,
    operation_metadata_of(_run_tool_op),
    path="/api/run-tool",
    method="POST",
    context_extractor=_extract_run_tool,
)

reload_tool = register_operation_route(
    tai_app,
    operation_metadata_of(_reload_tool_op),
    path="/api/tools/reload",
    method="POST",
)

remove_tool = register_operation_route(
    tai_app,
    operation_metadata_of(_remove_tool_op),
    path="/api/tools/remove",
    method="POST",
)
