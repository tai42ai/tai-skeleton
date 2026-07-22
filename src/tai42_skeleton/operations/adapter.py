"""The route-adapter helper — turns an operation into a thin HTTP route.

An adapter route is declarative: this helper registers the HTTP surface over
:meth:`HttpSurface.custom_route`, and the generated handler parses/validates the
flat typed parameters, calls the operation, wraps the success payload in the
``{"data": ...}`` envelope, and maps the operation's declared errors to statuses.
It honors the reload gate per the operation's metadata and NEVER performs
authorization — that is a separate consumer (the HTTP middleware at the route
edge, ``AuthzMiddleware`` at the tool edge).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from fastmcp.utilities.types import Audio, File, Image
from pydantic import BaseModel, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from tai42_skeleton.app.reload_gate import reload_gate
from tai42_skeleton.app.route_registry import DeclaredRouteMetadata, RouteAction
from tai42_skeleton.operations.errors import OperationError

if TYPE_CHECKING:
    from tai42_skeleton.operations.registry import OperationMetadata

# Methods whose adapter reads a JSON request body; a GET reads its typed
# parameters from the query string instead (never a GET body).
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# A request-edge extractor: derives an operation's flat kwargs from the raw
# request (header credentials, or a body the operation validates itself). It may
# raise an :class:`OperationError` to reject a malformed input with its status.
ContextExtractor = Callable[[Request], Awaitable[dict[str, Any]]]

# Path-template segment (``{name}`` / ``{name:path}``) → the argument name.
_PATH_PARAM = re.compile(r"\{([^}:]+)(?::[^}]+)?\}")


def _path_param_names(path: str) -> tuple[str, ...]:
    return tuple(_PATH_PARAM.findall(path))


def _declared_metadata(
    op: OperationMetadata, method: str, *, authed: bool, success_status: int
) -> DeclaredRouteMetadata:
    """The route's behavioral properties taken from the operation, not divined."""
    statuses: set[int] = {cls.status for cls in op.error_classes}
    if authed:
        statuses.add(401)
    if op.reload_gated:
        statuses.add(503)
    reads_body = op.request_model is not None and method in _BODY_METHODS
    return DeclaredRouteMetadata(
        reload_gated=op.reload_gated,
        reads_body=reads_body,
        error_statuses=tuple(sorted(statuses)),
        success_status=success_status,
    )


def _build_handler(
    op: OperationMetadata,
    method: str,
    path_params: tuple[str, ...],
    context_extractor: ContextExtractor | None = None,
    response_headers: dict[str, str] | None = None,
    success_status: int = 200,
) -> Callable[[Request], Awaitable[Response]]:
    reads_body = method in _BODY_METHODS

    async def handler(request: Request) -> Response:
        # An adapter route racing a registry teardown answers a retriable
        # 503 — the reload gate is honored on the route edge.
        if op.reload_gated and reload_gate.locked:
            return reload_gate.reject_response()

        kwargs: dict[str, Any] = {name: request.path_params[name] for name in path_params}

        if context_extractor is not None:
            # A request-shaped input (header credentials, a body the operation
            # validates itself with its own error classes) is derived here at the
            # HTTP edge and passed to the operation as flat kwargs — the operation
            # stays request-free. The extractor owns body/header reads, so it
            # REPLACES the default request-model parse; ``request_model`` is then
            # metadata only (the emitted spec's requestBody). A malformed input the
            # extractor rejects raises a typed error mapped to its status here.
            try:
                extra = await context_extractor(request)
            except OperationError as exc:
                return JSONResponse({"error": exc.message, **exc.extra}, status_code=exc.status)
            kwargs.update(extra)
        elif op.request_model is not None:
            try:
                if reads_body:
                    raw = await request.json()
                else:
                    raw = dict(request.query_params)
            except Exception:
                return JSONResponse({"error": "malformed request body"}, status_code=400)
            try:
                model = op.request_model.model_validate(raw)
            except ValidationError as exc:
                return JSONResponse({"error": exc.errors(include_url=False)}, status_code=422)
            kwargs.update(model.model_dump())

        try:
            result = await op.func(**kwargs)
        except OperationError as exc:
            # ``exc.extra`` carries any additional error-body fields the operation
            # opted into (e.g. a stable UI ``code``); empty for the common error.
            return JSONResponse({"error": exc.message, **exc.extra}, status_code=exc.status)

        # ``response_headers`` ride the SUCCESS response only (a caching directive
        # the route's read carries); an error response is uncached and unadorned.
        # ``success_status`` is the operation's own success code — 200 for the
        # common enveloped read/write, or an accepted-but-detached ``202`` (the
        # background tool-run submit door answers 202, not 200).
        return JSONResponse({"data": _serialize(result)}, status_code=success_status, headers=response_headers)

    handler.__name__ = op.name
    handler.__doc__ = op.func.__doc__
    return handler


def _serialize(result: object) -> Any:
    # A live fastmcp media object (``Image`` / ``Audio`` / ``File``) is not
    # JSON-native — fastmcp's own media-to-content conversion runs only on the MCP
    # tool edge, which the HTTP envelope bypasses. Convert it to its MCP content
    # model first (the same wire shape the direct tool-run path emits), so an
    # operation returning ``str | MediaBlock`` (``get_resource_by_id``) serves media
    # over HTTP as ``{"type": "image"|"audio"|"resource", ...}`` rather than raising.
    # Every existing operation returns dicts/lists/models, so this branch is a no-op
    # for them.
    if isinstance(result, Image):
        result = result.to_image_content()
    elif isinstance(result, Audio):
        result = result.to_audio_content()
    elif isinstance(result, File):
        result = result.to_resource_content()
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    return result


def register_operation_route(
    app: Any,
    op: OperationMetadata,
    *,
    path: str,
    method: str,
    tags: list[str] | None = None,
    authed: bool = True,
    context_extractor: ContextExtractor | None = None,
    response_headers: dict[str, str] | None = None,
    success_status: int = 200,
    action: RouteAction | None = None,
) -> Callable[[Request], Awaitable[Response]]:
    """Register ``op`` as an HTTP route at ``path``/``method`` and return the handler.

    A DELETE route auto-forces the ``destructive`` spec-surface bool on the operation;
    a GET route that declares ``destructive`` is a registration-time error (a read is
    never destructive). ``action`` is the ORTHOGONAL authorization action-class
    (``read``/``write`` derived from the method, or the admin-only ``fenced``/``secret``
    fence declared explicitly) — it never touches the ``destructive`` bool, so a
    ``secret`` GET never trips the GET-destructive guard. The operation's route template
    + method + path params are attached to its metadata so the tool-edge authorization
    can synthesize the concrete resource path.

    ``context_extractor`` derives the operation's non-path kwargs from the raw
    request when they are not a plain request-model parse (header credentials, or a
    body the operation validates itself with its own error classes); when given it
    REPLACES the default request-model parse and ``request_model`` becomes spec
    metadata only.

    ``response_headers`` are static headers stamped on the SUCCESS response only (a
    caching directive the route's read carries, e.g. ``cache-control: no-cache``) —
    the enveloping/status stay the adapter's, so the route surface is unchanged
    apart from the declared headers.

    ``success_status`` is the HTTP status the enveloped success answers with;
    it defaults to ``200`` and is set to ``202`` for an accepted-but-detached
    submission (the background tool-run submit door), so that door answers
    ``202`` rather than ``200``.
    """
    method_upper = method.upper()
    if method_upper == "GET" and op.destructive:
        raise ValueError(
            f"operation {op.name!r} is registered on GET {path} but declares destructive=True; "
            "a read operation is never destructive."
        )
    if method_upper == "DELETE":
        op.destructive = True

    path_params = _path_param_names(path)
    op.route_template = path
    op.http_method = method_upper
    op.path_params = path_params

    handler = _build_handler(op, method_upper, path_params, context_extractor, response_headers, success_status)
    route_tags = tags if tags is not None else list(op.tags)
    # Register over the concrete HttpSurface directly: the concrete app exposes it
    # as ``_http_surface``, the offline spec harness as ``.http``; either way the
    # resolved surface carries the ``declared`` / ``destructive`` metadata seam.
    surface = getattr(app, "_http_surface", None) or app.http
    decorator = surface.custom_route(
        path,
        methods=[method_upper],
        name=op.name,
        summary=op.summary,
        tags=route_tags,
        request_model=op.request_model,
        response_model=op.response_model,
        authed=authed,
        destructive=op.destructive,
        action=action,
        declared=_declared_metadata(op, method_upper, authed=authed, success_status=success_status),
    )
    return decorator(handler)
