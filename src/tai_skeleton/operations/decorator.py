"""The ``@operation`` decorator — the single declaration point for an operation.

A decorated function is a plain typed async function; the decorator records its
metadata into the :data:`operation_registry` and stamps the metadata onto the
function object (``__operation__``) so the route adapter can pick it up from the
function alone.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from tai_skeleton.operations.registry import OperationMetadata, OperationRegistry, operation_registry

if TYPE_CHECKING:
    from tai_skeleton.operations.errors import OperationError

# The attribute the metadata record is stamped onto the decorated function.
OPERATION_ATTR = "__operation__"

AsyncOp = Callable[..., Awaitable[object]]

# The decorator returns the function UNCHANGED, so it is typed identity-preserving:
# a decorated operation keeps its precise ``(params) -> Awaitable[ReturnType]``
# signature for callers (a direct ``await op(...)`` in an op-level test sees the
# real return type, not the erased ``object`` bound).
_AsyncOpT = TypeVar("_AsyncOpT", bound=AsyncOp)


def operation(
    *,
    name: str | None = None,
    summary: str,
    tags: list[str] | None = None,
    destructive: bool = False,
    reload_gated: bool = False,
    meta_executor: bool = False,
    caller_context: bool = False,
    authority_changing: bool = False,
    errors: list[type[OperationError]] | None = None,
    request_model: type[BaseModel] | None = None,
    response_model: type[BaseModel] | None = None,
    registry: OperationRegistry | None = None,
) -> Callable[[_AsyncOpT], _AsyncOpT]:
    """Declare a function as an operation and register it.

    ``name`` defaults to the function name. ``errors`` names the typed error
    classes the operation may raise (the adapter maps them to statuses; the
    coverage gate asserts each is exercised). ``meta_executor`` marks a
    "run a tool/agent by name" operation — hardcode-blocked from MCP projection
    (tier 1). ``caller_context`` marks an operation whose parameters are the
    caller's OWN identity, injected at the HTTP edge from the authenticated request
    — also hardcode-blocked from projection (tier 1), because as an MCP tool the
    caller would supply those identity params itself and spoof another principal.
    ``authority_changing`` marks an operation that mints/scopes keys, edits policy,
    replaces the manifest, or restores/runs unshipped state — off the default MCP
    surface (tier 2), includable by an explicit ``api_tools``.
    """

    target_registry = registry if registry is not None else operation_registry

    def decorator(func: _AsyncOpT) -> _AsyncOpT:
        op_name = name or func.__name__
        metadata = OperationMetadata(
            name=op_name,
            func=func,
            summary=summary,
            tags=tuple(tags or ()),
            destructive=destructive,
            reload_gated=reload_gated,
            meta_executor=meta_executor,
            caller_context=caller_context,
            authority_changing=authority_changing,
            error_classes=tuple(errors or ()),
            request_model=request_model,
            response_model=response_model,
        )
        target_registry.register(metadata)
        setattr(func, OPERATION_ATTR, metadata)
        return func

    return decorator


def operation_metadata_of(func: object) -> OperationMetadata:
    """The metadata stamped onto a decorated operation function, or raise."""
    metadata = getattr(func, OPERATION_ATTR, None)
    if not isinstance(metadata, OperationMetadata):
        raise TypeError(f"{func!r} is not an @operation-decorated function")
    return metadata
