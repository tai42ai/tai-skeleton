"""System-kinds operations — the authed pluggable-kind introspection read.

``list_system_kinds`` returns the live active/default/off state of every
pluggable kind (identity, accounts, monitoring, storage, backend, channels,
webhook verifiers, config, studio plugins), as computed by
:func:`tai42_skeleton.app.kind_status.collect_kind_status`. Read-only.
"""

from __future__ import annotations

from tai42_skeleton.app.kind_status import collect_kind_status
from tai42_skeleton.operations import operation


@operation(summary="List pluggable-kind status", tags=["system"])
async def list_system_kinds() -> list[dict]:
    return [row.model_dump() for row in collect_kind_status()]
