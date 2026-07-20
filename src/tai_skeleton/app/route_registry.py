"""The route-metadata registry — the single source of truth for the app's
self-describing HTTP surface.

Every ``@tai_app.http.custom_route(...)`` registration records a
:class:`RouteMetadata` entry here (see :mod:`tai_skeleton.app.http`). Two
consumers read the registry through the shared enumeration primitive
:func:`load_api_routes`:

* the OpenAPI 3.1 emitter (:mod:`tai_skeleton.cli.openapi`) and its coverage
  gate, which turns the registry into a spec and asserts every ``/api/*`` route
  self-describes; and
* the CLI↔route parity gate, which asserts every ``/api/*`` route has a terminal
  command.

The registry is populated purely by importing the router modules — no database,
Redis, or booted server — so the spec emits OFFLINE.

Each route DECLARES its behavioral OpenAPI metadata (``reload_gated``,
``reads_body``, ``error_statuses``, ``success_status``) through
:class:`DeclaredRouteMetadata`: a route registered through the operations adapter
supplies it from its operation's metadata, and a native ``/api/*`` handler
passes it explicitly at its registration. A handler that declares nothing (a
route outside the ``/api/*`` spec surface, e.g. ``/health`` or ``/metrics``)
records trivial defaults, since its behavioral metadata is never emitted.

The per-method success CONTENT TYPE is derived from each handler's source: the
default JSON surface answers the ``{"data": ...}`` envelope, while a streaming,
CSV, HTML, or asset-serving route answers its own media type, which the emitter
documents faithfully.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

Handler = Callable[[Request], Awaitable[Response]]

# Success content types, derived from markers in the handler source. The default
# JSON surface answers the ``{"data": ...}`` envelope; a streaming, CSV, HTML, or
# asset-serving route answers its own media type instead, which the emitter must
# document faithfully (no ``{"data": ...}`` wrapper). Each marker is a token whose
# presence in the (method-scoped) handler source — its own body or the name of a
# shared responder it calls — contributes that media type. A method that matches
# several markers documents several content types (the runs export serves CSV or a
# JSON download from one method); a method that matches none answers JSON.
_JSON_MEDIA_TYPE = "application/json"
_MEDIA_TYPE_MARKERS = (
    ("text/event-stream", "text/event-stream"),
    ("text/csv", "text/csv"),
    ("_csv_response", "text/csv"),
    ("asset_content_type", "application/octet-stream"),
    ("HTML_CONTENT_TYPE", "text/html"),
    # The interactions callback door delegates its GET branch to ``_callback_get``,
    # which serves the browser confirm page; the delegated responder's name marks
    # the HTML surface (the derivation follows the responder the handler calls, not
    # only its own inline responses).
    ("_callback_get", "text/html"),
    # A downloadable attachment (the backup export, and the runs export's JSON
    # format) — a file, not the enveloped JSON surface.
    ("Content-Disposition", "application/octet-stream"),
)

# Marks the branch of a multi-method handler that dispatches on the request method,
# so each method's success media type is derived from only the code that serves it.
_METHOD_GUARD = re.compile(r"""request\.method\s*==\s*['"]([A-Za-z]+)['"]""")


@dataclass(frozen=True)
class RouteMetadata:
    """One self-describing route: its wire shape plus the OpenAPI metadata the
    emitter and the coverage/parity gates consume."""

    path: str
    methods: tuple[str, ...]
    name: str
    summary: str
    description: str
    tags: tuple[str, ...]
    authed: bool
    request_model: type[BaseModel] | None
    response_model: type[BaseModel] | None
    reload_gated: bool
    reads_body: bool
    error_statuses: tuple[int, ...]
    success_status: int
    success_media_types: dict[str, tuple[str, ...]]
    destructive: bool = False


@dataclass(frozen=True)
class DeclaredRouteMetadata:
    """The behavioral OpenAPI properties a route DECLARES.

    A route registered through the operations adapter supplies this
    from its operation's metadata + declared error classes; a native ``/api/*``
    handler passes it explicitly at its ``custom_route`` registration. Its
    ``reload_gated`` / ``reads_body`` / error statuses / success status feed the
    emitted spec and the coverage/parity gates.
    """

    reload_gated: bool
    reads_body: bool
    error_statuses: tuple[int, ...]
    success_status: int


def _handler_source(func: Callable[..., object]) -> str:
    return inspect.getsource(func)


def _method_scoped_source(source: str, method: str) -> str:
    """The handler source as ``method`` sees it: shared lines plus the block guarded
    by ``if request.method == "<method>"``, dropping the blocks that guard a
    DIFFERENT method. A handler that never dispatches on ``request.method`` (the
    common case) yields its whole source unchanged, so single-method routes and
    multi-method routes that share one code path are untouched."""
    kept: list[str] = []
    foreign_indent: int | None = None
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if foreign_indent is not None:
            if stripped and indent <= foreign_indent:
                foreign_indent = None  # the block guarding another method has closed
            else:
                continue  # still inside a block that serves a different method
        guard = _METHOD_GUARD.search(line) if stripped.startswith(("if ", "elif ")) else None
        if guard is not None and guard.group(1).upper() != method.upper():
            foreign_indent = indent
            continue
        kept.append(line)
    return "".join(kept)


def _method_media_types(source: str) -> tuple[str, ...]:
    matched = tuple(dict.fromkeys(media for token, media in _MEDIA_TYPE_MARKERS if token in source))
    return matched or (_JSON_MEDIA_TYPE,)


def _success_media_types(source: str, methods: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Map each method to the content type(s) its success response serves, derived
    from the method-scoped handler source so a route whose methods answer different
    media types (the callback door: GET serves HTML, POST serves the JSON envelope)
    documents each method faithfully."""
    return {method: _method_media_types(_method_scoped_source(source, method)) for method in methods}


class RouteRegistry:
    """In-memory map of every registered route, keyed by ``(path, methods)``.

    Populated as a side effect of importing the router modules. Recording the
    same ``(path, methods)`` twice replaces the entry (a module re-import is
    idempotent), never accumulates duplicates.
    """

    def __init__(self) -> None:
        self._routes: dict[tuple[str, tuple[str, ...]], RouteMetadata] = {}

    def record(
        self,
        *,
        path: str,
        methods: list[str],
        name: str | None,
        handler: Handler,
        summary: str,
        tags: list[str],
        authed: bool,
        request_model: type[BaseModel] | None,
        response_model: type[BaseModel] | None,
        destructive: bool = False,
        declared: DeclaredRouteMetadata | None = None,
    ) -> None:
        """Record one route's metadata.

        A route in the ``/api/*`` spec surface supplies ``declared`` — the
        operation adapter passes an operation's metadata, and a native handler
        passes its own — so its behavioral properties come from a declaration,
        never a divination. A handler that declares nothing (a route outside the
        spec surface) records trivial defaults, since its behavioral metadata is
        never emitted. The per-method success media type is always derived from
        the handler source. Raises loudly on a missing minimum-bar field so a
        route that fails to self-describe is caught at import, not in the gate."""
        if not summary:
            raise ValueError(f"route {'/'.join(methods)} {path} is missing a non-empty summary")
        if not tags:
            raise ValueError(f"route {'/'.join(methods)} {path} is missing at least one tag")
        source = _handler_source(handler)
        method_key = tuple(sorted(m.upper() for m in methods))
        if declared is None:
            reload_gated = False
            reads_body = False
            error_statuses: tuple[int, ...] = ()
            success_status = 200
        else:
            reload_gated = declared.reload_gated
            reads_body = declared.reads_body
            error_statuses = declared.error_statuses
            success_status = declared.success_status
        self._routes[path, method_key] = RouteMetadata(
            path=path,
            methods=method_key,
            name=name or handler.__name__,
            summary=summary,
            description=inspect.cleandoc(handler.__doc__ or ""),
            tags=tuple(tags),
            authed=authed,
            request_model=request_model,
            response_model=response_model,
            reload_gated=reload_gated,
            reads_body=reads_body,
            error_statuses=error_statuses,
            success_status=success_status,
            success_media_types=_success_media_types(source, method_key),
            destructive=destructive,
        )

    def routes(self) -> list[RouteMetadata]:
        """Every recorded route, ordered by path then methods for a stable spec."""
        return [self._routes[key] for key in sorted(self._routes)]


# The one process-wide registry. ``HttpSurface.custom_route`` records into it;
# the emitter and the parity gate read it via ``load_api_routes``.
route_registry = RouteRegistry()


class _SpecFastMCP:
    """A no-op stand-in for FastMCP's ``custom_route`` used only for OFFLINE
    metadata capture — it returns the handler unchanged, so importing the router
    modules records their metadata without a booted server."""

    def custom_route(
        self, path: str, methods: list[str], name: str | None, include_in_schema: bool
    ) -> Callable[[Handler], Handler]:
        return lambda fn: fn


class _SpecLifecycle:
    """A no-op stand-in for the app's ``lifecycle`` seam used only for OFFLINE
    metadata capture — a router module registering a startup/shutdown/reload
    handler at import time gets the handler back unchanged, so no handler is
    wired and no server is needed."""

    def on_startup(self, func: Callable[..., object]) -> Callable[..., object]:
        return func

    def on_shutdown(self, func: Callable[..., object]) -> Callable[..., object]:
        return func

    def on_reload(self, func: Callable[..., object]) -> Callable[..., object]:
        return func


class _SpecApp:
    """Minimal ``tai_app`` impl exposing only the ``http`` and ``lifecycle`` seams
    the router modules touch at import time, so metadata capture needs no database,
    Redis, or config."""

    def __init__(self) -> None:
        from tai_skeleton.app.http import HttpSurface

        self._fast_mcp = _SpecFastMCP()
        self.http = HttpSurface(self)  # type: ignore[arg-type]
        self.lifecycle = _SpecLifecycle()


def _tai_app_bound() -> bool:
    from tai_contract.app import tai_app

    try:
        tai_app.http  # noqa: B018 — attribute access probes the bind state
    except AttributeError:
        return False
    return True


def _ensure_routers_imported() -> None:
    """Import every router module so its routes record into the registry.

    When ``tai_app`` is unbound (a CLI/test process that never booted a server),
    a minimal offline harness is bound first so ``tai_app.http.custom_route``
    resolves. When a server is already running, its real binding stays and the
    already-imported modules simply keep their registry entries.
    """
    import importlib
    import pkgutil

    from tai_contract.app import tai_app

    import tai_skeleton.routers as routers_pkg

    if not _tai_app_bound():
        tai_app.bind(_SpecApp())

    for module_info in pkgutil.iter_modules(routers_pkg.__path__, routers_pkg.__name__ + "."):
        importlib.import_module(module_info.name)


def load_api_routes() -> list[RouteMetadata]:
    """The shared route-enumeration primitive: every registered ``/api/*`` route.

    Imports the router modules (offline) if needed, then returns their metadata.
    Both the OpenAPI emitter/coverage gate and the CLI↔route parity gate call
    this so they enumerate the API surface identically.
    """
    _ensure_routers_imported()
    return [meta for meta in route_registry.routes() if meta.path.startswith("/api/")]
