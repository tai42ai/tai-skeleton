"""Resolve a dispatched tool name to the operation it runs, with the arguments it runs
with.

A projected operation can be reached under a DIFFERENT name: an extension branch
(``_extend_tools``) or a preset baked over it (``PresetManager._specs[name]
.base_tool``), possibly layered. The tool-edge authorization keys on operation
metadata AND on the call's arguments — the concrete resource path a dispatch targets
is synthesized from them — so it must chase both derivative maps, recursively, back
to the base while collecting what each preset BAKES along the way.

If the base resolves to a projected operation, its metadata and the effective
arguments are returned. If it resolves to a NON-operation tool (toolbox, plugin,
keep-set), ``None`` is returned — per-tool authorization for those is out of scope.
Neither answer is valid unless the operation surface is SETTLED, so mid-rebuild there
is a third answer, a loud refusal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from tai42_skeleton.app.reload_gate import REJECT_MESSAGE
from tai42_skeleton.operations.errors import UnavailableError
from tai42_skeleton.operations.registry import operation_registry

if TYPE_CHECKING:
    from tai42_skeleton.operations.registry import OperationMetadata


class OperationSurfaceUnsettledError(UnavailableError):
    """A dispatched name cannot be resolved to an authorizable target, because the
    operation surface is being rebuilt.

    A reload clears the registry, repopulates it, and only then lets the routers re-attach
    each record's route, all while the serving loop keeps dispatching. Inside that window
    neither an absent name nor a present record is evidence, so the resolver refuses with
    the run surface's retriable ``reloading`` contract (same message, same ``503``).
    """


class ResolvedDispatch(NamedTuple):
    """What a dispatched tool name actually runs: the base operation, and the arguments
    that reach it once every preset in the chain has applied its baked kwargs."""

    operation: OperationMetadata
    call_arguments: dict[str, Any]


def _assert_capability_decidable(name: str, base: str, registry: Any) -> None:
    """Assert that ``registry`` not holding ``base`` really does mean ``base`` is a
    capability tool, rather than the operation surface being mid-rebuild.

    Keyed on the registry's own SETTLED flag — no rebuild in flight AND non-empty. A
    reload that holds the process reload gate without touching the operation surface is
    not evidence here and must not refuse a dispatch that resolves fine.
    """
    if registry.settled:
        return
    raise OperationSurfaceUnsettledError(
        f"cannot authorize the dispatch of {name!r}: the operation surface is being rebuilt, so {base!r} "
        f"being absent from the operation registry does not mean it is a capability tool — {REJECT_MESSAGE}"
    )


def _assert_record_usable(name: str, base: str, registry: Any) -> None:
    """Assert that a record ``registry`` DOES hold for ``base`` is one a decision may be
    made against, rather than a half-rebuilt one.

    A record is usable only once the routers have re-attached its route template and
    method, which happens after the replay puts it back. Read inside that window it
    carries no template, or the PREVIOUS generation's — so present-but-torn takes the same
    retriable refusal as absent.
    """
    if registry.settled:
        return
    raise OperationSurfaceUnsettledError(
        f"cannot authorize the dispatch of {name!r}: the operation surface is being rebuilt, so the registry's "
        f"record for {base!r} may not yet carry the route this dispatch targets — {REJECT_MESSAGE}"
    )


def resolve_dispatch(
    name: str,
    call_arguments: dict[str, Any],
    *,
    tool_registry: Any | None,
    preset_manager: Any | None,
    registry: Any = None,
) -> ResolvedDispatch | None:
    """The operation a dispatched tool ``name`` ultimately runs and the arguments it
    ultimately runs with, or ``None`` when the settled base is a registered
    NON-operation tool.

    Raises :class:`OperationSurfaceUnsettledError` whenever the operation surface is
    mid-rebuild — neither its silence about the base nor a record it does hold is an
    answer there, and reading either as one would leave a projected operation
    unauthorized or key the decision on a route the dispatch is not taking.

    Chases the preset map (``PresetManager``) and the extension-branch map
    (``ToolRegistry._extend_tools`` via ``base_of``) alternately until the name stops
    moving, then returns the operation metadata if the settled base is a projected
    operation. A resolution cycle terminates on the visited set.

    A preset's ``fixed_kwargs`` are constants a caller can neither supply nor override, so
    they are applied OVER the arguments handed in and the decision keys on what the
    dispatch actually fires, including a path parameter only the bake supplies. An
    extension branch bakes nothing.
    """
    reg = registry if registry is not None else operation_registry
    current = name
    arguments = dict(call_arguments)
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        # A preset links through PresetManager, not _extend_tools — consult it
        # first so a preset baked over an op (or over a branch) is followed.
        if preset_manager is not None and preset_manager.is_registered(current):
            spec = preset_manager.get_spec(current)
            arguments.update(spec.fixed_kwargs)
            current = spec.base_tool
            continue
        # An extension branch links through _extend_tools; base_of returns the
        # origin base (or the name itself for a base — the loop then settles).
        if tool_registry is not None:
            base = tool_registry.base_of(current)
            if base != current:
                current = base
                continue
        break

    if reg.has(current):
        _assert_record_usable(name, current, reg)
        return ResolvedDispatch(reg.get(current), arguments)
    _assert_capability_decidable(name, current, reg)
    return None
