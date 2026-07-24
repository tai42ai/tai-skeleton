"""A router module carrying one probe operation per authorization CLASS, so the
execution-seam matrix can drive each end to end.

``exec_probe_fenced`` sits at a ``fenced`` route (admin-only, enforced by the per-tag LEVEL
pass alone); ``exec_probe_read`` at a grantable ``read`` route decided by the caller's
scope. Real fenced operations either mutate the process or are blocked from projection, so
these echo instead, letting the ALLOW half of each pair assert a real result. The fenced
probe's path parameter is what a concrete resource path is synthesized from.
"""

from __future__ import annotations

from tai42_contract.app import tai42_app

from tai42_skeleton.operations import operation, operation_metadata_of, register_operation_route

# Every probe body that actually RAN, in order. Reset by each boot, since the router
# module is re-imported on every start.
calls: list[tuple[str, str]] = []


@operation(summary="Echo a mark through a fenced route", tags=["tools"])
async def exec_probe_fenced(target: str, mark: str = "") -> str:
    """Echo ``target``/``mark``; reached only on an allow."""
    calls.append(("fenced", mark))
    return f"fenced:{target}:{mark}"


@operation(summary="Echo a mark through a grantable read route", tags=["tools"])
async def exec_probe_read(mark: str = "") -> str:
    """Echo ``mark``; reached only on an allow."""
    calls.append(("read", mark))
    return f"read:{mark}"


fenced_route = register_operation_route(
    tai42_app,
    operation_metadata_of(exec_probe_fenced),
    path="/api/exec-probe/{target}/fenced",
    method="POST",
    action="fenced",
)

read_route = register_operation_route(
    tai42_app,
    operation_metadata_of(exec_probe_read),
    path="/api/exec-probe/read",
    method="GET",
    action="read",
)
