"""Route-metadata registry tests — the record-time behavior both gates depend on.

Each route DECLARES its behavioral OpenAPI properties (``reload_gated`` /
``reads_body`` / error statuses / success status) through
:class:`DeclaredRouteMetadata`; a route that declares nothing records trivial
defaults. The per-method success content type is derived from the responder each
method uses. These tests pin the declared/default handling, the media-type
derivation, and the record-time minimum-bar enforcement.
"""

from __future__ import annotations

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from tai_skeleton.app.route_registry import DeclaredRouteMetadata, RouteRegistry, load_api_routes


async def _plain(request: Request) -> Response:
    """A plain read."""
    return JSONResponse({"data": []})


async def _streams(request: Request) -> Response:
    return Response("", media_type="text/event-stream")


async def _callback_get(request: Request) -> Response:
    from starlette.responses import HTMLResponse

    return HTMLResponse("<h1>Confirm</h1>")


async def _delegates_to_callback_get(request: Request) -> Response:
    return await _callback_get(request)


async def _method_dispatching_callback(request: Request) -> Response:
    # GET serves the HTML confirm page; POST answers the ``{"data": ...}`` envelope —
    # the two methods must derive different media types from one registration.
    if request.method == "GET":
        return await _callback_get(request)
    return JSONResponse({"data": {}})


def _csv_response(rows: list) -> Response:
    return Response("", media_type="text/csv", headers={"Content-Disposition": "attachment"})


async def _csv_or_json_export(request: Request) -> Response:
    # One method serving two content types: a CSV body or a JSON download.
    if request.query_params.get("format") == "json":
        return Response("{}", media_type="application/json", headers={"Content-Disposition": "attachment"})
    return _csv_response([])


# -- Declared / default behavioral metadata ----------------------------------


def test_declared_metadata_populates_the_record() -> None:
    registry = RouteRegistry()
    registry.record(
        path="/api/thing",
        methods=["POST"],
        name="submit",
        handler=_plain,
        summary="Thing",
        tags=["t"],
        authed=True,
        request_model=None,
        response_model=None,
        declared=DeclaredRouteMetadata(
            reload_gated=True,
            reads_body=True,
            error_statuses=(400, 401, 503),
            success_status=202,
        ),
    )
    (meta,) = registry.routes()
    assert meta.reload_gated is True
    assert meta.reads_body is True
    assert meta.error_statuses == (400, 401, 503)
    assert meta.success_status == 202
    assert meta.name == "submit"


def test_undeclared_route_records_trivial_defaults() -> None:
    # A route that declares nothing (one outside the /api/* spec surface) records
    # trivial defaults — its behavioral metadata is never emitted.
    registry = RouteRegistry()
    registry.record(
        path="/health",
        methods=["GET"],
        name=None,
        handler=_plain,
        summary="Health",
        tags=["t"],
        authed=False,
        request_model=None,
        response_model=None,
    )
    (meta,) = registry.routes()
    assert meta.reload_gated is False
    assert meta.reads_body is False
    assert meta.error_statuses == ()
    assert meta.success_status == 200


def test_docstring_becomes_description() -> None:
    registry = RouteRegistry()
    registry.record(
        path="/api/plain",
        methods=["GET"],
        name=None,
        handler=_plain,
        summary="Plain",
        tags=["t"],
        authed=False,
        request_model=None,
        response_model=None,
    )
    (meta,) = registry.routes()
    assert meta.description == "A plain read."


# -- Success media-type derivation -------------------------------------------


def test_streaming_media_type_derivation() -> None:
    registry = RouteRegistry()
    registry.record(
        path="/api/stream",
        methods=["GET"],
        name=None,
        handler=_streams,
        summary="Stream",
        tags=["t"],
        authed=True,
        request_model=None,
        response_model=None,
    )
    (meta,) = registry.routes()
    assert meta.success_media_types == {"GET": ("text/event-stream",)}


def test_media_type_follows_a_delegated_responder() -> None:
    registry = RouteRegistry()
    registry.record(
        path="/api/callback",
        methods=["GET"],
        name=None,
        handler=_delegates_to_callback_get,
        summary="Callback",
        tags=["t"],
        authed=False,
        request_model=None,
        response_model=None,
    )
    (meta,) = registry.routes()
    # The handler returns no response class of its own — the ``text/html`` surface
    # is derived from the name of the delegated ``_callback_get`` responder.
    assert meta.success_media_types == {"GET": ("text/html",)}


def test_media_type_is_derived_per_method() -> None:
    registry = RouteRegistry()
    registry.record(
        path="/api/callback",
        methods=["GET", "POST"],
        name=None,
        handler=_method_dispatching_callback,
        summary="Callback",
        tags=["t"],
        authed=False,
        request_model=None,
        response_model=None,
    )
    (meta,) = registry.routes()
    # One registration, two methods: the GET branch serves the confirm page while
    # POST answers the JSON envelope, so each method derives its own media type.
    assert meta.success_media_types == {"GET": ("text/html",), "POST": ("application/json",)}


def test_a_method_serving_two_content_types_lists_both() -> None:
    registry = RouteRegistry()
    registry.record(
        path="/api/export",
        methods=["GET"],
        name=None,
        handler=_csv_or_json_export,
        summary="Export",
        tags=["t"],
        authed=True,
        request_model=None,
        response_model=None,
    )
    (meta,) = registry.routes()
    # A single method that serves either a CSV body or a JSON download documents
    # both content types rather than being falsely pinned to one.
    assert meta.success_media_types == {"GET": ("text/csv", "application/octet-stream")}


# -- Record-time minimum bar --------------------------------------------------


def test_missing_summary_is_rejected_loudly() -> None:
    registry = RouteRegistry()
    with pytest.raises(ValueError, match="summary"):
        registry.record(
            path="/api/x",
            methods=["GET"],
            name=None,
            handler=_plain,
            summary="",
            tags=["t"],
            authed=True,
            request_model=None,
            response_model=None,
        )


def test_missing_tags_is_rejected_loudly() -> None:
    registry = RouteRegistry()
    with pytest.raises(ValueError, match="tag"):
        registry.record(
            path="/api/x",
            methods=["GET"],
            name=None,
            handler=_plain,
            summary="X",
            tags=[],
            authed=True,
            request_model=None,
            response_model=None,
        )


def test_load_api_routes_returns_only_api_paths() -> None:
    routes = load_api_routes()
    assert routes
    assert all(meta.path.startswith("/api/") for meta in routes)
    # /health and /metrics are registered but are not part of the /api surface.
    assert not any(meta.path in ("/health", "/metrics") for meta in routes)
