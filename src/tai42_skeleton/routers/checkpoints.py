"""HTTP surface for checkpoint retention — one thin adapter over the operation.

``POST /api/checkpoints/sweep`` runs ``sweep_checkpoints``. It is a deployment-wide
destructive memory purge, so it is ``action="fenced"`` — admin only.
"""

from __future__ import annotations

from tai42_contract.app import tai42_app

from tai42_skeleton.operations import operation_metadata_of, register_operation_route
from tai42_skeleton.operations.checkpoints import sweep_checkpoints as _sweep_checkpoints_op

sweep_checkpoints = register_operation_route(
    tai42_app,
    operation_metadata_of(_sweep_checkpoints_op),
    path="/api/checkpoints/sweep",
    method="POST",
    action="fenced",
)
