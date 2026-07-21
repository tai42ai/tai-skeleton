"""HTTP route for the extension surface — ``GET /api/extensions``.

One AUTHED door: the flat list of every registered extension for the UI's
extension picker. Each item is ``{"name": str, "kind": str}`` (the extension's
kind — WRAPPER / TRANSFORMER / BACKEND — carried as its lowercase enum value so
the UI can group and single-select the non-stackable BACKEND kind).

The route is a thin adapter over the :func:`list_extensions` operation; the
operation logic lives in ``tai42_skeleton.operations.extensions``. Success bodies
are ``{"data": ...}``; failures are ``{"error": "<message>"}``.
"""

from __future__ import annotations

from tai42_contract.app import tai42_app

from tai42_skeleton.operations import operation_metadata_of, register_operation_route
from tai42_skeleton.operations.extensions import list_extensions as _list_extensions_op

list_extensions = register_operation_route(
    tai42_app,
    operation_metadata_of(_list_extensions_op),
    path="/api/extensions",
    method="GET",
    action="read",
)
