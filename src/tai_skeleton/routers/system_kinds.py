"""HTTP surface for pluggable-kind status — the authed introspection door.

- ``GET /api/system/kinds`` (AUTHED) — the live active/default/off state of every
  pluggable kind (identity, accounts, monitoring, storage, backend, channels,
  webhook verifiers, config, studio plugins), as computed by
  :func:`tai_skeleton.app.kind_status.collect_kind_status`. Read-only; success
  bodies are ``{"data": [...]}``.

The route names installed plugins, so it is AUTHED and never pinned public. It is
a thin adapter over the :func:`list_system_kinds` operation; the operation logic
lives in ``tai_skeleton.operations.system_kinds``.
"""

from __future__ import annotations

from tai_contract.app import tai_app

from tai_skeleton.operations import operation_metadata_of, register_operation_route
from tai_skeleton.operations.system_kinds import list_system_kinds as _list_system_kinds_op

list_system_kinds = register_operation_route(
    tai_app,
    operation_metadata_of(_list_system_kinds_op),
    path="/api/system/kinds",
    method="GET",
    authed=True,
)
