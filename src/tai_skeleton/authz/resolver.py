"""Resolve a dispatched tool name to its underlying operation.

A projected operation can be reached under a DIFFERENT name: an extension branch
(``_extend_tools``) or a preset baked over it (``PresetManager._specs[name]
.base_tool``), possibly layered. The tool-edge authorization keys on operation
metadata, so it must chase both derivative maps — recursively — back to the base.

If the base resolves to a projected operation, its metadata is returned and the
authorization check applies. If it resolves to a NON-operation tool (toolbox,
plugin, keep-set), ``None`` is returned — per-tool authorization for those is a
separate concern, out of this scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tai_skeleton.operations.registry import operation_registry

if TYPE_CHECKING:
    from tai_skeleton.operations.registry import OperationMetadata


def resolve_base_operation(
    name: str,
    *,
    tool_registry: Any | None,
    preset_manager: Any | None,
    registry: Any = None,
) -> OperationMetadata | None:
    """The operation metadata a dispatched tool ``name`` ultimately runs, or ``None``.

    Chases the preset map (``PresetManager``) and the extension-branch map
    (``ToolRegistry._extend_tools`` via ``base_of``) alternately until the name
    stops moving, then returns the operation metadata if the settled base is a
    projected operation. A resolution cycle terminates on the visited set.
    """
    reg = registry if registry is not None else operation_registry
    current = name
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        # A preset links through PresetManager, not _extend_tools — consult it
        # first so a preset baked over an op (or over a branch) is followed.
        if preset_manager is not None and preset_manager.is_registered(current):
            current = preset_manager.get_spec(current).base_tool
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
        return reg.get(current)
    return None
