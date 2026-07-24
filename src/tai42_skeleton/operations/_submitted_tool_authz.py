"""Authorize the LIVE caller to dispatch a caller-SUBMITTED tool name at an async
run-any-tool door.

``submit_run`` and ``create_schedule`` run a caller-supplied ``tool_name`` DETACHED from
the request, so the inner tool reaches no other edge: the execution-identity seam is a
no-op with no fire bound and ``AuthzMiddleware`` runs only on MCP. Their doors must
therefore make the decision themselves, before recording or scheduling the run.

:func:`authorize_submitted_tool` runs the SAME full edge decision ``AuthzMiddleware``
makes, against the request caller's own identity — never a bound execution identity.
"""

from __future__ import annotations

from typing import Any

from tai42_contract.app import tai42_app

from tai42_skeleton.authz.check import check
from tai42_skeleton.authz.identity import resolve_caller_identity
from tai42_skeleton.authz.resolver import resolve_dispatch


async def authorize_submitted_tool(tool_name: str, arguments: dict[str, Any]) -> None:
    """Authorize the current caller to dispatch ``tool_name`` with ``arguments``.

    Resolves the name through the live registries (chasing presets and extension
    branches, folding in what each preset bakes), then :func:`check`s the caller against
    it. A non-operation (capability) target carries no per-call decision and passes.

    Raises ``PermissionDenied`` (403) on a deny and the retriable
    ``OperationSurfaceUnsettledError`` (503) mid-rebuild — the MCP edge's own refusals.
    """
    resolved = resolve_dispatch(
        tool_name,
        arguments,
        tool_registry=getattr(tai42_app, "_tool_registry", None),
        preset_manager=getattr(tai42_app, "preset_manager", None),
    )
    if resolved is None:
        return
    await check(resolve_caller_identity(), resolved.operation, resolved.call_arguments)
