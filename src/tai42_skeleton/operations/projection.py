"""Project the operation surface into MCP tools.

At startup (after manifest load) and on every reload this iterates the
:class:`OperationRegistry` and registers each SURVIVING operation as a
first-class tool through the normal tool binding, so extensions wrap it, presets
bake over it, ``user_tools`` curates it, and ``tai tools run`` / agents dispatch
it. Curation is by ``api_tools`` only (``enabled`` / ``include`` / ``exclude`` /
``expose_destructive``); it is registered ``force=True`` so the per-module
``tools[]`` gate never double-filters a projected tool.

Two exclusion tiers guard the surface:

* **Tier 1 — hardcoded, never projectable.** A meta-executor (``run_tool`` — a
  "run any tool by name" operation) is a universal authz bypass; a caller-context
  op (``get_me`` — its params are the caller's OWN edge-derived identity) would be
  an identity spoof if a caller supplied those params itself. Both are skipped
  unconditionally, even when named in ``include``, with a loud log.
* **Tier 2 — configurable default-exclude.** An authority-changing operation
  (an ``/api/auth/*`` route, the backup-import op, ``update_manifest``, …) is off
  the default surface but projectable via an explicit ``api_tools.include``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from mcp.types import ToolAnnotations

from tai42_skeleton.app.reload_gate import REJECT_MESSAGE, reload_gate
from tai42_skeleton.operations.errors import OperationError
from tai42_skeleton.operations.registry import OperationRegistry, operation_registry

if TYPE_CHECKING:
    from tai42_contract.manifest import ApiToolsConfig

    from tai42_skeleton.operations.registry import OperationMetadata

logger = logging.getLogger(__name__)

# Tier 1: names hardcode-blocked from projection regardless of config. A
# meta-executor is also caught by the ``meta_executor`` metadata flag; this named
# set is the belt-and-braces default so the invariant holds even for an operation
# whose author forgets the flag.
TIER1_META_EXECUTORS: frozenset[str] = frozenset({"run_tool"})

# Tier 2: the route-prefix that marks an authority-changing operation family.
# The backup-import op and ``update_manifest`` carry the ``authority_changing``
# metadata flag instead (they own no shared prefix); an authority-changing op that
# shares no route prefix is marked by setting that same flag.
_TIER2_ROUTE_PREFIX = "/api/auth/"


def _tier1_reason(op: OperationMetadata) -> str | None:
    """The reason ``op`` is hardcode-blocked from projection, or ``None`` if it is not.

    Named so the block log states WHY: a meta-executor is a universal authz bypass; a
    caller-context op takes the caller's OWN edge-derived identity as params, which an
    MCP caller would supply itself to spoof another principal."""
    if op.meta_executor or op.name in TIER1_META_EXECUTORS:
        return "meta-executor"
    if op.caller_context:
        return "caller-context identity op"
    return None


def is_tier1(op: OperationMetadata) -> bool:
    """Whether ``op`` is hardcode-blocked from projection (never projectable)."""
    return _tier1_reason(op) is not None


def is_tier2(op: OperationMetadata) -> bool:
    """Whether ``op`` is default-excluded (authority-changing; includable)."""
    if op.authority_changing:
        return True
    return op.route_template is not None and op.route_template.startswith(_TIER2_ROUTE_PREFIX)


def _make_tool(op: OperationMetadata) -> Callable[..., Awaitable[object]]:
    """A projected-tool wrapper carrying ``op``'s flat typed signature.

    It honors the reload gate on the tool edge (a reload-gated op racing a
    registry teardown raises the same retriable rejection its route answers) and
    maps the operation's declared errors to loud ``ToolError``s carrying the same
    message. The success payload is the operation's own value — no ``{"data": …}``
    envelope (that is the HTTP-edge concern only).
    """
    import inspect

    from fastmcp.exceptions import ToolError

    func = op.func

    async def projected(**kwargs: Any) -> object:
        if op.reload_gated and reload_gate.locked:
            raise ToolError(REJECT_MESSAGE)
        try:
            return await func(**kwargs)
        except OperationError as exc:
            raise ToolError(exc.message) from exc

    # ``eval_str=True`` resolves string annotations (every operation module carries
    # ``from __future__ import annotations``, so its annotations are strings) against
    # the operation FUNCTION's own module globals — the wrapper lives in this module,
    # whose namespace lacks the operation's imported types (e.g. ``ExtensionElement``),
    # so the tool schema fastmcp builds from the flat signature would otherwise fail
    # to resolve them.
    projected.__name__ = op.name
    projected.__qualname__ = op.name
    # The tool's description is the operation's own docstring, falling back to its
    # declared ``summary`` — always present — so a projected tool is never
    # description-less (a client-tool conversion rejects a tool with neither).
    projected.__doc__ = func.__doc__ or op.summary
    projected.__signature__ = inspect.signature(func, eval_str=True)  # type: ignore[attr-defined]
    projected.__annotations__ = inspect.get_annotations(func, eval_str=True)
    return projected


def _selected_operations(
    registry: OperationRegistry, config: ApiToolsConfig
) -> tuple[list[OperationMetadata], list[str]]:
    """The operations to project plus the tier-1 names skipped, given ``config``.

    Raises loudly if ``include`` names an operation that is not registered — a
    manifest that curates a nonexistent op fails boot rather than silently doing
    nothing.
    """
    known = registry.names()
    unknown_includes = sorted(set(config.include) - known)
    if unknown_includes:
        raise ValueError(
            f"api_tools.include names operation(s) not registered: {unknown_includes!r}; "
            f"registered operations are {sorted(known)!r}"
        )

    include = set(config.include)
    exclude = set(config.exclude)
    selected: list[OperationMetadata] = []
    tier1_skipped: list[str] = []

    for op in registry.all():
        reason = _tier1_reason(op)
        if reason is not None:
            # Never projectable — not even when named in include.
            tier1_skipped.append(op.name)
            if op.name in include:
                logger.warning(
                    "operations projection: %r is a hardcode-blocked %s (tier 1) and is NOT "
                    "projected despite appearing in api_tools.include",
                    op.name,
                    reason,
                )
            continue
        if op.name in exclude:
            logger.info("operations projection: %r excluded by api_tools.exclude", op.name)
            continue
        if op.name in include:
            selected.append(op)
            continue
        if is_tier2(op):
            logger.info(
                "operations projection: %r is authority-changing (tier 2) and is off the default "
                "surface; add it to api_tools.include to project it",
                op.name,
            )
            continue
        if op.destructive and not config.expose_destructive:
            logger.info(
                "operations projection: %r is destructive and api_tools.expose_destructive is false; not projected",
                op.name,
            )
            continue
        selected.append(op)

    return selected, tier1_skipped


def project_operations(app: Any, config: ApiToolsConfig, *, registry: OperationRegistry | None = None) -> list[str]:
    """Register every surviving operation as an MCP tool; return the projected names.

    ``config`` is the manifest's ``api_tools`` block. With ``enabled=False`` the
    surface is empty. Each projected tool is bound ``force=True`` (so ``api_tools``
    is the only gate), carries a ``destructiveHint`` annotation for a destructive
    op, and is logged.
    """
    reg = registry if registry is not None else operation_registry

    # The hardcoded tier-1 block is always enforced and always logged — even when
    # projection is disabled — so the invariant "run_tool is never a tool" is
    # visible in every boot log.
    for op in reg.all():
        reason = _tier1_reason(op)
        if reason is not None:
            logger.info(
                "operations projection: %r is hardcode-blocked from the MCP surface (tier 1 %s)",
                op.name,
                reason,
            )

    if not config.enabled:
        logger.info("operations projection: api_tools.enabled is false — no operations projected")
        return []

    selected, _ = _selected_operations(reg, config)

    projected_names: list[str] = []
    for op in selected:
        annotations = ToolAnnotations(destructiveHint=True) if op.destructive else None
        app.tools.tool(
            force=True,
            name=op.name,
            tags=set(op.tags),
            annotations=annotations,
        )(_make_tool(op))
        projected_names.append(op.name)

    if projected_names:
        logger.info("operations projection: projected %d tool(s): %s", len(projected_names), sorted(projected_names))
    else:
        logger.info("operations projection: no operations projected")
    return projected_names
