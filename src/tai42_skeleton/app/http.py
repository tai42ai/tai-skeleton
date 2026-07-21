"""HTTP delivery surface — the impl body behind the ``app.http`` facet.

Owns the ASGI middleware stack registered via ``@app.http.middleware`` and the
custom-route passthrough; :meth:`finalize` applies the stack (plus the sub-MCP
mount) around whichever ASGI app the launch surface builds.
"""

import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response

from tai42_skeleton.app.route_registry import route_registry

if TYPE_CHECKING:
    from tai42_skeleton.app.route_registry import DeclaredRouteMetadata, RouteAction
    from tai42_skeleton.app.server import TaiMCP


class HttpSurface:
    """Middleware + custom-route registration over the app's FastMCP server."""

    def __init__(self, app: "TaiMCP") -> None:
        self._app = app
        # Keyed by the middleware class's qualified name so a module re-import
        # (each start() re-imports the middleware modules) replaces rather than
        # accumulates its entry — insertion order, hence stack order, is kept —
        # while a construction-time (build_app) middleware persists across
        # reloads.
        self._middlewares: dict[str, Middleware] = {}

    def _register_middleware(self, cls: type, options: dict[str, Any]) -> None:
        # starlette's ``Middleware`` over-constrains ``cls`` to its private
        # ``_MiddlewareFactory`` protocol; a user middleware class is a valid
        # factory at runtime.
        key = f"{cls.__module__}.{cls.__qualname__}"
        self._middlewares[key] = Middleware(cast(Any, cls), **options)

    def middleware(self, cls: type | None = None, **options: Any):
        if cls and inspect.isclass(cls):
            self._register_middleware(cls, options)
            return cls

        def decorator(inner_cls):
            self._register_middleware(inner_cls, options)
            return inner_cls

        return decorator

    def custom_route(
        self,
        path: str,
        methods: list[str],
        name: str | None = None,
        include_in_schema: bool = True,
        *,
        summary: str,
        tags: list[str],
        response_model: type[BaseModel] | None,
        request_model: type[BaseModel] | None = None,
        authed: bool = True,
        destructive: bool = False,
        action: "RouteAction | None" = None,
        declared: "DeclaredRouteMetadata | None" = None,
    ) -> Callable[[Callable[[Request], Awaitable[Response]]], Callable[[Request], Awaitable[Response]]]:
        """Register the handler with FastMCP AND record its OpenAPI metadata.

        The route serves exactly as before; the added metadata (summary, tags,
        request/response models, auth) is the source of truth the OpenAPI emitter
        and its coverage gate read. See :class:`tai42_contract.app.facets.AppHttp`.

        A route registered through the operations adapter passes ``destructive``
        (emitted as ``x-destructive``) and ``declared`` (its behavioral
        properties, taken from the operation's metadata rather than divined from
        the adapter closure's source).
        """
        fastmcp_route = self._app._fast_mcp.custom_route(path, methods, name, include_in_schema)

        def decorator(fn: Callable[[Request], Awaitable[Response]]) -> Callable[[Request], Awaitable[Response]]:
            route_registry.record(
                path=path,
                methods=methods,
                name=name,
                handler=fn,
                summary=summary,
                tags=tags,
                authed=authed,
                request_model=request_model,
                response_model=response_model,
                destructive=destructive,
                action=action,
                declared=declared,
            )
            return fastmcp_route(fn)

        return decorator

    def finalize(self, app):
        """Mount the sub-MCP router and wrap the registered middleware stack.

        ``app`` is the FastMCP-built ASGI app whose lifespan drives the
        streamable-http session-manager task group. Middleware wrappers are plain
        ASGI callables that expose neither ``app``'s ``router`` nor its lifespan,
        so the lifespan-bearing app is recorded on the returned object as
        ``mcp_lifespan_app``. A caller that must enter the lifespan by hand (the
        mounted worker, whose dispatch swallows the lifespan scope) reads it there
        so the FastMCP lifespan is entered regardless of any middleware wrapping.
        """
        lifespan_app = app
        app.mount(
            self._app._mcp_sub_app_router.root_prefix,
            self._app._mcp_sub_app_router,
        )

        for mw in self._middlewares.values():
            cls, args, kwargs = mw
            app = cls(app, *args, **kwargs)

        app.mcp_lifespan_app = lifespan_app
        return app


def http_surface() -> HttpSurface:
    """The bound concrete :class:`HttpSurface`, carrying the skeleton-typed
    ``declared`` / ``destructive`` metadata seam.

    A native ``/api/*`` handler registers through this so it can declare its
    OpenAPI metadata explicitly (its ``reload_gated`` / ``reads_body`` / error
    statuses / success status), the same way the operation adapter reaches the
    surface. The contract ``app.http`` facet intentionally omits the seam (it
    cannot reference the skeleton ``DeclaredRouteMetadata`` type), so the routers
    reach the concrete surface directly here. The offline spec harness exposes it
    as ``tai42_app.http``; the concrete server exposes it as ``_http_surface``.
    """
    from tai42_contract.app import tai42_app

    # The concrete server exposes the surface as ``_http_surface``; the offline spec
    # harness binds an app whose ``.http`` IS the ``HttpSurface`` directly. Either
    # way the resolved object is a concrete ``HttpSurface`` carrying the seam.
    surface = getattr(tai42_app, "_http_surface", None)
    if surface is None:
        surface = tai42_app.http
    return cast("HttpSurface", surface)
