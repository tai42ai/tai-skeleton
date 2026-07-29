"""A fixture operation + its route adapter, imported via ``routers_modules`` so
the operations seam can be exercised end-to-end in a real app context. Kept out
of the product route set so it never disturbs the byte-identical spec pin.
"""

from __future__ import annotations

from pydantic import BaseModel
from tai42_contract.app import tai42_app

from tai42_skeleton.operations import operation, register_operation_route
from tai42_skeleton.operations.decorator import operation_metadata_of
from tai42_skeleton.operations.errors import NotFoundError


class GreetBody(BaseModel):
    name: str


@operation(summary="Greet by name", tags=["sample"], request_model=GreetBody, errors=[NotFoundError])
async def sample_greet(name: str) -> dict:
    """Greet someone by name."""
    if name == "missing":
        raise NotFoundError("no such name")
    return {"greeting": f"hello {name}"}


register_operation_route(
    tai42_app,
    operation_metadata_of(sample_greet),
    path="/api/sample/greet",
    method="POST",
    action="write",
)
