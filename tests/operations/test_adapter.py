"""The route-adapter helper: gate, parse, envelope, error mapping, and the
DELETE-forces-destructive / GET-declaring-destructive rules."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from typing import Any, cast

import pytest
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

from tai42_skeleton.app.reload_gate import reload_gate
from tai42_skeleton.app.route_registry import _SpecApp as SpecApp
from tai42_skeleton.app.route_registry import route_registry
from tai42_skeleton.operations import OperationRegistry, operation, register_operation_route
from tai42_skeleton.operations.decorator import operation_metadata_of
from tai42_skeleton.operations.errors import ConflictError, NotFoundError


def _run(awaitable: Awaitable[Response]) -> Response:
    return asyncio.run(cast("Any", awaitable))


def _request(method: str, path: str, *, body: dict | None = None, path_params: dict | None = None) -> Request:
    payload = json.dumps(body or {}).encode()
    sent = False

    async def receive():
        nonlocal sent
        sent = True
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "path_params": path_params or {},
    }
    return Request(scope, receive)


class GreetRequest(BaseModel):
    name: str


def _register(reg, method="POST", path="/api/sample/greet", **op_kwargs):
    @operation(summary="Greet", tags=["sample"], request_model=GreetRequest, registry=reg, **op_kwargs)
    async def greet(name: str) -> dict:
        """Greet by name."""
        if name == "missing":
            raise NotFoundError("no such name")
        if name == "conflict":
            raise ConflictError("already greeted")
        return {"greeting": f"hello {name}"}

    handler = register_operation_route(SpecApp(), operation_metadata_of(greet), path=path, method=method)
    return greet, handler


def test_success_wraps_in_data_envelope():
    reg = OperationRegistry()
    _, handler = _register(reg)
    resp = _run(handler(_request("POST", "/api/sample/greet", body={"name": "ann"})))
    assert resp.status_code == 200
    assert json.loads(bytes(resp.body)) == {"data": {"greeting": "hello ann"}}


def test_success_status_defaults_to_200_and_is_overridable():
    reg = OperationRegistry()

    @operation(summary="Accept", tags=["sample"], request_model=GreetRequest, registry=reg)
    async def accept(name: str) -> dict:
        return {"queued": name}

    meta = operation_metadata_of(accept)
    handler = register_operation_route(SpecApp(), meta, path="/api/sample/accept", method="POST", success_status=202)
    resp = _run(handler(_request("POST", "/api/sample/accept", body={"name": "ann"})))
    # The enveloped success answers the declared 202 (accepted-but-detached), not 200.
    assert resp.status_code == 202
    assert json.loads(bytes(resp.body)) == {"data": {"queued": "ann"}}
    # The route metadata records the same success status for the spec.
    recorded = route_registry._routes["/api/sample/accept", ("POST",)]
    assert recorded.success_status == 202

    # A route that omits ``success_status`` keeps the 200 default.
    _, default_handler = _register(reg, path="/api/sample/greet")
    default_resp = _run(default_handler(_request("POST", "/api/sample/greet", body={"name": "ann"})))
    assert default_resp.status_code == 200


def test_declared_error_maps_to_status():
    reg = OperationRegistry()
    _, handler = _register(reg)
    resp = _run(handler(_request("POST", "/api/sample/greet", body={"name": "missing"})))
    assert resp.status_code == 404
    assert json.loads(bytes(resp.body)) == {"error": "no such name"}

    resp = _run(handler(_request("POST", "/api/sample/greet", body={"name": "conflict"})))
    assert resp.status_code == 409


def test_validation_error_is_422():
    reg = OperationRegistry()
    _, handler = _register(reg)
    resp = _run(handler(_request("POST", "/api/sample/greet", body={})))
    assert resp.status_code == 422


def test_malformed_body_is_400():
    reg = OperationRegistry()
    _, handler = _register(reg)

    async def bad_receive():
        return {"type": "http.request", "body": b"not json", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/sample/greet",
        "query_string": b"",
        "headers": [],
        "path_params": {},
    }
    resp = _run(handler(Request(scope, bad_receive)))
    assert resp.status_code == 400


def test_reload_gate_honored_on_route_edge():
    reg = OperationRegistry()
    _, handler = _register(reg, reload_gated=True)

    async def run():
        reload_gate.bind_to_running_loop()
        async with reload_gate.lock:
            return await handler(_request("POST", "/api/sample/greet", body={"name": "ann"}))

    resp = asyncio.run(run())
    assert resp.status_code == 503
    assert json.loads(bytes(resp.body))["reloading"] is True


def test_path_params_passed_through():
    reg = OperationRegistry()

    class Empty(BaseModel):
        pass

    @operation(summary="Detach", tags=["mcp"], registry=reg)
    async def detach(title: str) -> dict:
        """Detach a server."""
        return {"detached": title}

    meta = operation_metadata_of(detach)
    handler = register_operation_route(SpecApp(), meta, path="/api/mcp-status/{title}/deregister", method="POST")
    assert meta.path_params == ("title",)
    resp = _run(handler(_request("POST", "/api/mcp-status/srv/deregister", path_params={"title": "srv"})))
    assert json.loads(bytes(resp.body)) == {"data": {"detached": "srv"}}


def test_delete_forces_destructive_and_records_route():
    reg = OperationRegistry()

    @operation(summary="Remove", tags=["tools"], registry=reg)
    async def remove_thing() -> dict:
        """Remove a thing."""
        return {"removed": True}

    meta = operation_metadata_of(remove_thing)
    register_operation_route(SpecApp(), meta, path="/api/things/remove", method="DELETE")
    assert meta.destructive is True
    recorded = next(r for r in route_registry.routes() if r.path == "/api/things/remove")
    assert recorded.destructive is True


def test_get_declaring_destructive_is_registration_error():
    reg = OperationRegistry()

    @operation(summary="List", tags=["tools"], destructive=True, registry=reg)
    async def list_things() -> dict:
        """List things."""
        return {}

    with pytest.raises(ValueError, match="never destructive"):
        register_operation_route(SpecApp(), operation_metadata_of(list_things), path="/api/things", method="GET")


def test_basemodel_result_is_serialized():
    reg = OperationRegistry()

    class Out(BaseModel):
        value: int

    @operation(summary="Make", tags=["things"], registry=reg)
    async def make_thing() -> Out:
        """Make a thing."""
        return Out(value=7)

    handler = register_operation_route(
        SpecApp(), operation_metadata_of(make_thing), path="/api/things/make", method="POST"
    )
    resp = _run(handler(_request("POST", "/api/things/make")))
    assert json.loads(bytes(resp.body)) == {"data": {"value": 7}}


@pytest.mark.parametrize(
    ("media", "wire_type"),
    [
        ("image", "image"),
        ("audio", "audio"),
        ("file", "resource"),
    ],
)
def test_media_result_is_serialized_to_its_wire_block(media: str, wire_type: str):
    # A fastmcp media return (``Image`` / ``Audio`` / ``File``) is not JSON-native;
    # the adapter converts it to its MCP content block so an operation returning
    # ``str | MediaBlock`` serves media over the HTTP envelope.
    from fastmcp.utilities.types import Audio, File, Image

    payloads = {
        "image": Image(data=b"\x89PNG\r\n", format="png"),
        "audio": Audio(data=b"RIFF....", format="wav"),
        "file": File(data=b"payload", format="bin"),
    }

    reg = OperationRegistry()

    @operation(summary="Load", tags=["things"], registry=reg)
    async def load_media() -> object:
        """Load media."""
        return payloads[media]

    handler = register_operation_route(
        SpecApp(), operation_metadata_of(load_media), path="/api/things/media", method="POST"
    )
    resp = _run(handler(_request("POST", "/api/things/media")))
    block = json.loads(bytes(resp.body))["data"]
    assert block["type"] == wire_type


def test_context_extractor_supplies_kwargs():
    reg = OperationRegistry()

    @operation(summary="Echo", tags=["sample"], registry=reg)
    async def echo(token: str) -> dict:
        """Echo a token."""
        return {"token": token}

    async def extractor(request):
        return {"token": request.headers.get("x-token", "")}

    meta = operation_metadata_of(echo)
    handler = register_operation_route(SpecApp(), meta, path="/api/echo", method="POST", context_extractor=extractor)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/echo",
        "query_string": b"",
        "headers": [(b"x-token", b"abc")],
        "path_params": {},
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    resp = _run(handler(Request(scope, receive)))
    assert json.loads(bytes(resp.body)) == {"data": {"token": "abc"}}


def test_context_extractor_error_maps_to_status():
    reg = OperationRegistry()

    @operation(summary="Guard", tags=["sample"], registry=reg)
    async def guarded(token: str) -> dict:
        """Never reached when the extractor rejects."""
        return {"token": token}

    async def extractor(_request):
        raise NotFoundError("no token presented")

    meta = operation_metadata_of(guarded)
    handler = register_operation_route(SpecApp(), meta, path="/api/guard", method="POST", context_extractor=extractor)
    resp = _run(handler(_request("POST", "/api/guard")))
    assert resp.status_code == 404
    assert json.loads(bytes(resp.body)) == {"error": "no token presented"}


def test_get_reads_query_params():
    reg = OperationRegistry()

    class Query(BaseModel):
        limit: int = 10

    @operation(summary="List", tags=["tools"], request_model=Query, registry=reg)
    async def list_things(limit: int) -> dict:
        """List things."""
        return {"limit": limit}

    meta = operation_metadata_of(list_things)
    handler = register_operation_route(SpecApp(), meta, path="/api/things", method="GET")

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/things",
        "query_string": b"limit=5",
        "headers": [],
        "path_params": {},
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    resp = _run(handler(Request(scope, receive)))
    assert json.loads(bytes(resp.body)) == {"data": {"limit": 5}}
