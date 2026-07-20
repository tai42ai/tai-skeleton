"""The FastMCP middleware that authorizes every projected-operation dispatch.

``AuthzMiddleware`` runs at tool DISPATCH — caller-side, before the tool's
extension/transform chain (so before any backend enqueue) — on EVERY MCP-serving
FastMCP instance (the main server and every sub-MCP mount). It resolves the
dispatched tool name to its base operation and, for a projected operation, runs
:func:`authz.check`. A denial raises a :class:`~fastmcp.exceptions.ToolError`
backed by :class:`PermissionDenied`; a non-operation tool passes straight
through (its authorization is a separate concern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware, MiddlewareContext

from tai42_skeleton.authz.check import check
from tai42_skeleton.authz.identity import resolve_caller_identity
from tai42_skeleton.authz.resolver import resolve_base_operation
from tai42_skeleton.operations.errors import PermissionDenied

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class AuthzMiddleware(Middleware):
    """Tool-edge authorization for projected operations. Installed on the main
    ``_fast_mcp`` and on every sub-MCP FastMCP via ``add_middleware``."""

    def __init__(self, app: Any) -> None:
        self._app = app

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: Callable[[MiddlewareContext[Any]], Awaitable[Any]],
    ) -> Any:
        name = context.message.name
        arguments = dict(context.message.arguments or {})

        op = resolve_base_operation(
            name,
            tool_registry=getattr(self._app, "_tool_registry", None),
            preset_manager=getattr(self._app, "preset_manager", None),
        )
        if op is not None:
            identity = resolve_caller_identity()
            try:
                await check(identity, op, arguments)
            except PermissionDenied as exc:
                raise ToolError(str(exc)) from exc

        return await call_next(context)
