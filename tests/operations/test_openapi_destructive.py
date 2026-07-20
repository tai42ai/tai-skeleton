"""The emitter's ``x-destructive`` output, the declared-metadata dual path, and
the emitter-determinism pin.

The product spec now carries ``x-destructive`` on the mutating routes converted to
operations (the first routers migrated to the operations layer): a read never
declares it, so the invariant the emitter pins is ``x-destructive`` appears only on
a non-GET method. The emitter is still deterministic — rebuilding the spec twice is
byte-stable."""

from __future__ import annotations

import json

from pydantic import BaseModel

from tai42_skeleton.app.route_registry import _SpecApp as SpecApp
from tai42_skeleton.app.route_registry import load_api_routes, route_registry
from tai42_skeleton.cli.openapi import build_openapi_spec, version
from tai42_skeleton.operations import OperationRegistry, operation, register_operation_route
from tai42_skeleton.operations.decorator import operation_metadata_of
from tai42_skeleton.operations.errors import ConflictError


class _Body(BaseModel):
    x: int


def test_destructive_only_on_mutating_routes():
    """A read is never destructive: every product route carrying ``x-destructive``
    is a mutating (non-GET) method, and no GET declares it."""
    spec = build_openapi_spec()
    for path_item in spec["paths"].values():
        for method, op in path_item.items():
            if "x-destructive" in op:
                assert method.lower() != "get", f"GET route declares x-destructive: {op['operationId']}"


def test_emitter_is_deterministic():
    """The registry/emitter rework must not make the product surface unstable:
    rebuilding the spec twice is byte-identical."""
    first = json.dumps(build_openapi_spec(), sort_keys=True, indent=2)
    second = json.dumps(build_openapi_spec(), sort_keys=True, indent=2)
    assert first == second


def test_declared_destructive_route_emits_x_destructive():
    reg = OperationRegistry()

    @operation(summary="Wipe it", tags=["things"], destructive=True, request_model=_Body, registry=reg)
    async def wipe(x: int) -> dict:
        """Wipe."""
        return {}

    register_operation_route(SpecApp(), operation_metadata_of(wipe), path="/api/things/wipe", method="POST")

    # The emitter reads the registry; find the recorded route directly and emit.
    from tai42_skeleton.cli.openapi import _operation as emit_operation

    meta = next(r for r in route_registry.routes() if r.path == "/api/things/wipe")
    op = emit_operation(meta, "POST", {})
    assert op["x-destructive"] is True


def test_declared_metadata_drives_statuses_not_ast():
    """An adapter route takes its error statuses from the operation's declared
    error classes, not an AST scan of the adapter closure."""
    reg = OperationRegistry()

    @operation(summary="Do", tags=["things"], errors=[ConflictError], request_model=_Body, registry=reg)
    async def doit(x: int) -> dict:
        """Do it."""
        return {}

    register_operation_route(SpecApp(), operation_metadata_of(doit), path="/api/things/do", method="POST")
    meta = next(r for r in route_registry.routes() if r.path == "/api/things/do")
    # 409 (declared ConflictError) + 401 (authed). No AST-divined 500.
    assert set(meta.error_statuses) == {401, 409}
    assert meta.destructive is False


def test_spec_version_present():
    """Sanity: the emitter still stamps the package version (unchanged path)."""
    spec = build_openapi_spec()
    assert spec["info"]["version"] == version("tai42-skeleton")


def test_converted_route_appears_in_api_routes():
    reg = OperationRegistry()

    @operation(summary="Ping", tags=["things"], request_model=_Body, registry=reg)
    async def ping(x: int) -> dict:
        """Ping."""
        return {}

    register_operation_route(SpecApp(), operation_metadata_of(ping), path="/api/things/ping", method="POST")
    paths = {r.path for r in load_api_routes()}
    assert "/api/things/ping" in paths
