"""The route-metadata registry — the single source of truth for the app's
self-describing HTTP surface.

Every ``@tai42_app.http.custom_route(...)`` registration records a
:class:`RouteMetadata` entry here (see :mod:`tai42_skeleton.app.http`). Two
consumers read the registry through the shared enumeration primitive
:func:`load_api_routes`:

* the OpenAPI 3.1 emitter (:mod:`tai42_skeleton.cli.openapi`) and its coverage
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
from typing import Literal, Protocol, cast

from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response
from tai42_contract.app import DeclaredRouteMetadata

Handler = Callable[[Request], Awaitable[Response]]

# The route action-class — the SINGLE authoritative source of a route's
# authorization character:
#
# * ``read`` / ``write`` are the GRANTABLE classes: a role's per-tag level on the
#   route's feature tag decides reach, and the class equals the method's derived
#   action (:func:`method_to_action`).
# * ``fenced`` is the admin-only MUTATION fence: no per-tag level opens it, admin
#   only. It is DISTINCT from the ``RouteMetadata.destructive`` spec-surface bool
#   (which the adapter auto-forces on every DELETE) — sourcing the fence from that
#   bool would over-fence editor-reachable DELETEs, so the two never interact.
# * ``secret`` is the admin-only bulk-secret READ fence: a GET whose payload is
#   admin-equivalent, admin only and non-grantable like ``fenced``.
RouteAction = Literal["read", "write", "fenced", "secret"]
_VALID_ROUTE_ACTIONS: frozenset[str] = frozenset(("read", "write", "fenced", "secret"))
_GRANTABLE_ROUTE_ACTIONS: frozenset[str] = frozenset(("read", "write"))
_FENCED_ROUTE_ACTIONS: frozenset[str] = frozenset(("fenced", "secret"))

_READ_METHODS: frozenset[str] = frozenset(("GET", "HEAD", "OPTIONS"))
_WRITE_METHODS: frozenset[str] = frozenset(("POST", "PUT", "PATCH", "DELETE"))


def method_to_action(method: str) -> Literal["read", "write"]:
    """Map an HTTP method to its derived action-class — the ONE place the read/write
    action is derived from the method. ``GET``/``HEAD``/``OPTIONS`` → ``read``;
    ``POST``/``PUT``/``PATCH``/``DELETE`` → ``write``.

    An unknown/empty method raises loudly (fail-closed) — the derivation never
    defaults to ``read``, so an unclassifiable method is caught at registration/boot
    rather than silently admitted as a read."""
    upper = method.upper()
    if upper in _READ_METHODS:
        return "read"
    if upper in _WRITE_METHODS:
        return "write"
    raise ValueError(f"unclassifiable HTTP method {method!r}: cannot derive a read/write action")


def derive_route_action(methods: tuple[str, ...]) -> Literal["read", "write"]:
    """The grantable action-class a route's method set derives to: ``write`` when the
    route serves ANY write method, else ``read``. Enforcement re-derives per-method
    from the live request method (:func:`method_to_action`), so this coarse label
    only classes the route as a whole (the UI grouping and the boot validation)."""
    return "write" if any(method_to_action(method) == "write" for method in methods) else "read"


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
    additional_success_statuses: tuple[int, ...]
    success_media_types: dict[str, tuple[str, ...]]
    action: RouteAction
    destructive: bool = False


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


def _resolve_route_action(action: RouteAction | None, methods: tuple[str, ...], path: str, authed: bool) -> RouteAction:
    """The route's authoritative action-class. Every AUTHED route DECLARES its class
    explicitly: a grantable ``read``/``write``, or the admin-only ``fenced``/``secret``
    fence. A declared ``read``/``write`` is VALIDATED against every method's derived action,
    so a misdeclared class (``read`` on a write method) is refused at registration. An
    authed route that declares NOTHING BOOT-FAILS here — auto-deriving read/write for an
    undeclared authed route is fail-open (a forgotten fence silently becomes grantable), so
    the omission is refused rather than divined. A PUBLIC route (``authed=False``) never
    enforces an action, so it is exempt and its class is derived from the method for the
    spec — and a ``fenced``/``secret`` class on a public route is itself a contradiction
    (the fence is enforced only in the authenticated path, so a public fence silently opens
    an admin-only door), refused here symmetric with the authed-without-action raise. An
    unknown method raises out of the derivation (fail-closed)."""
    if action in _FENCED_ROUTE_ACTIONS:
        if not authed:
            raise ValueError(
                f"public route {'/'.join(methods)} {path} declares action={action!r}; a fence is enforced only "
                "in the authenticated path, so a fenced/secret class on an authed=False route silently opens it — "
                "a public route must be read/write (or authed=True to keep the fence)"
            )
        return action  # type: ignore[return-value]
    derived = derive_route_action(methods)
    if action is None:
        if authed:
            raise ValueError(
                f"authed route {'/'.join(methods)} {path} declares no action-class; every authed route must "
                "declare read/write/fenced/secret explicitly — allow-by-omission is a fail-open fence"
            )
        return derived
    if action not in _GRANTABLE_ROUTE_ACTIONS:
        raise ValueError(f"route {'/'.join(methods)} {path} declares unknown action {action!r}")
    # An explicit read/write must match the method-derived class for EVERY method.
    for method in methods:
        if method_to_action(method) != action:
            raise ValueError(
                f"route {'/'.join(methods)} {path} declares action={action!r} but method {method!r} "
                f"derives {method_to_action(method)!r}; a grantable route's class must equal its method"
            )
    return action


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
        action: RouteAction | None = None,
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
        resolved_action = _resolve_route_action(action, method_key, path, authed)
        if declared is None:
            reload_gated = False
            reads_body = False
            error_statuses: tuple[int, ...] = ()
            success_status = 200
            additional_success_statuses: tuple[int, ...] = ()
        else:
            reload_gated = declared.reload_gated
            reads_body = declared.reads_body
            error_statuses = declared.error_statuses
            success_status = declared.success_status
            additional_success_statuses = declared.additional_success_statuses
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
            additional_success_statuses=additional_success_statuses,
            success_media_types=_success_media_types(source, method_key),
            action=resolved_action,
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
    """Minimal ``tai42_app`` impl exposing only the ``http`` and ``lifecycle`` seams
    the router modules touch at import time, so metadata capture needs no database,
    Redis, or config."""

    def __init__(self) -> None:
        from tai42_skeleton.app.http import HttpSurface

        self._fast_mcp = _SpecFastMCP()
        self.http = HttpSurface(self)  # type: ignore[arg-type]
        self.lifecycle = _SpecLifecycle()

    def effective_router_modules(self) -> None:
        """No deployment is served under the spec harness, so the enumeration
        universe is the whole ``tai42_skeleton.routers`` package — signalled by
        ``None`` (never a curated started-manifest set)."""
        return None


class _RouterUniverseSource(Protocol):
    """The one method the shared importer asks of the currently-bound app to choose its
    enumeration universe. The forwarding ``tai42_app`` handle is typed as the assembled
    facade (which carries only namespace facets), so this Protocol types the flat method
    the handle forwards to the bound impl — the started ``TaiMCP`` (answers its manifest's
    effective router set) and the offline ``_SpecApp`` (answers ``None``) both satisfy it."""

    def effective_router_modules(self) -> list[str] | None: ...


def _started_router_modules() -> list[str] | None:
    """The EFFECTIVE router set of a started deployment, or ``None`` when none answers one
    (unbound process, offline spec harness, partially-faked app). The probe must stay a
    ``hasattr`` on the forwarding handle — a facet probe reads a partially-faked app as
    unbound."""
    from tai42_contract.app import tai42_app

    if not hasattr(tai42_app, "effective_router_modules"):
        return None
    return cast("_RouterUniverseSource", tai42_app).effective_router_modules()


def _import_all_router_modules() -> None:
    """Import EVERY module under ``tai42_skeleton.routers`` — the whole-package
    enumeration universe used only offline (CLI spec/parity tools, unbooted tests)."""
    import importlib
    import pkgutil

    import tai42_skeleton.routers as routers_pkg

    for module_info in pkgutil.iter_modules(routers_pkg.__path__, routers_pkg.__name__ + "."):
        importlib.import_module(module_info.name)


def _ensure_routers_imported() -> None:
    """Import the router modules that define the enumeration universe.

    There are exactly two universes. In a STARTED process the bound app answers its
    manifest's EFFECTIVE router set, and THAT set is the universe: ``start()`` already
    imported those modules, so re-importing them is an idempotent no-op and no other
    router is pulled into the live route table. Offline — an unbound CLI/test process,
    or a process whose bound impl answers no router set — the whole
    ``tai42_skeleton.routers`` package is the universe, so an offline enumeration sees
    every route with no server to boot.

    A router module resolves ``tai42_app.http.custom_route`` at import, so the offline
    import runs under the ``_SpecApp`` stand-in bound for that import ALONE — this
    enumeration is a read and must leave the process's app binding as it found it.
    """
    from tai42_contract.app import tai42_app

    effective = _started_router_modules()
    if effective is not None:
        # A STARTED deployment: its effective router set IS the universe.
        # start() already imported these; re-import is an idempotent no-op.
        import importlib

        for module in effective:
            importlib.import_module(module)
        return
    with tai42_app.bound(_SpecApp()):
        _import_all_router_modules()


def load_api_routes() -> list[RouteMetadata]:
    """The shared route-enumeration primitive: every registered ``/api/*`` route.

    Imports the enumeration universe if needed, then returns its metadata. In a STARTED
    process that universe is the deployment's effective router set (so this enumerates
    exactly the served surface); offline it is the whole router package. Both the OpenAPI
    emitter/coverage gate and the CLI↔route parity gate call this so they enumerate the
    API surface identically.
    """
    _ensure_routers_imported()
    return [meta for meta in route_registry.routes() if meta.path.startswith("/api/")]


def load_all_routes() -> list[RouteMetadata]:
    """Every registered route — the whole self-describing HTTP surface, ``/api/*`` and
    the non-``/api`` operational routes (``/health``, ``/ready``, ``/metrics``, …) alike.

    Imports the enumeration universe if needed, then returns its metadata. In a STARTED
    process that universe is the deployment's effective router set, so this enumerates
    exactly the served surface and never pulls an un-mounted router into the live route
    table; offline it is the whole router package. The access-control resolver derives its
    SPA-shell reserved set from the non-``/api`` GET routes surfaced here, so a route added
    to a router joins the reserved set with no static list to maintain.
    """
    _ensure_routers_imported()
    return route_registry.routes()


def route_action_violations() -> list[str]:
    """Every GATED route whose action-class fails the audit — the boot gate's fail
    list (empty means clean). A route is an offender when its ``action`` is not one of
    the four valid classes (allow-by-omission is dead) or, for a grantable
    ``read``/``write`` route, when the declared class disagrees with the method-derived
    action. ``fenced``/``secret`` are the explicit admin-only fence classes and are
    exempt from the method-equals-action rule. Public (``authed=False``) routes are not
    gated — their action never enforces — so they are not audited here.

    Enumerates through :func:`load_all_routes` so the enumeration universe is imported
    before the audit runs — in a started process that is the deployment's served router
    surface, so the audit judges exactly what the deployment serves; iterating the raw
    registry could pass VACUOUSLY (an empty loop is a silent no-op) had the routers not
    yet been imported."""
    violations: list[str] = []
    for meta in load_all_routes():
        if not meta.authed:
            continue
        if meta.action not in _VALID_ROUTE_ACTIONS:
            violations.append(f"{'/'.join(meta.methods)} {meta.path}: unclassified action {meta.action!r}")
            continue
        if meta.action in _GRANTABLE_ROUTE_ACTIONS:
            try:
                derived = derive_route_action(meta.methods)
            except ValueError as exc:
                violations.append(f"{'/'.join(meta.methods)} {meta.path}: {exc}")
                continue
            if derived != meta.action:
                violations.append(
                    f"{'/'.join(meta.methods)} {meta.path}: declared action={meta.action!r} "
                    f"disagrees with the method-derived {derived!r}"
                )
    return violations
