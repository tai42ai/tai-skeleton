"""One structured audit line per HTTP request that reaches it — who, what, when,
outcome.

Registered INSIDE the access-control stack (``TaiMCP._base_middleware``), after
the gate has resolved the caller, so THIS middleware writes the line for every
request the gate ADMITS. A
refusal writes its OWN line at the refusal site through the shared
:func:`emit_audit_line`, so failed authz is on the trail too — the route-
authorization denials (401/403) at
``tai42_skeleton.access_control.middleware.ResourceGuardMiddleware._deny`` (the
REAL caller id on an authenticated-but-unauthorized deny, else ``unauthenticated``,
with a ``reason=`` naming the deny), the auth-backend errors (401/403) at
``tai42_skeleton.access_control.adapter.handle_auth_error`` (a ``DenialCause`` as
``reason=`` when known), and the 429 at
``tai42_skeleton.middleware.rate_limit.RateLimitMiddleware`` (``unauthenticated``).
The ``reason=`` field and the ``unauthenticated`` principal are what tell a refusal
line from an accept line. The accept-line principal here is READ from the gate-bound
caller id (:func:`tai42_contract.access_control.context.get_current_user_id`), never
re-resolved from the credential; no bound caller (public route, or gate off) audits
as ``anonymous``.

The line carries request FACTS only: principal, method, route, status, duration,
timestamp — no body, no query string, no header. The route is the only request
text recorded, bounded by whether the request MATCHED a route. A match records the
path TEMPLATE, every path parameter written as its ``{name}`` — so a path-borne
capability (``/trigger/{token}``, ``/api/interactions/callback/{ticket}``) records
the template, never the secret. Under the default Studio SPA catch-all
(``default_routers="all"``), ``/{spa_path:path}`` matches every GET and
part-matches every other method, so a near-miss records ``/{spa_path}`` with its
tail swallowed by the parameter. A request NO route matched — no catch-all
(``default_routers`` ``"api"``/``"none"``), or a request the body cap refuses
above the router — has no template: it records the bare ``<unmatched>``
constant, no path text at all.

A request handed to a MOUNTED app (the sub-MCP router under ``/app``) is the third
outcome: the mount matched only its own PREFIX, and what the mounted app did with
the rest is invisible from here — routing inside it writes the same scope marks a
route match writes, with nothing recording which happened last. So the prefix
(templated) is the whole of what a crossed mount may state: the line reads
``/app/<unmatched>``, and the tail is never written even when the mounted app
served it. Exception: a mounted app that records the matched route OBJECT
(FastAPI) states its own template — otherwise a mounted app's own routing trail
belongs to that app: the sub-MCP router (``/app``) DOES register this middleware
for itself, so a sub-MCP request records TWO lines — this base app's
``/app/<unmatched>`` and the sub-app's own inner route. The mount is
recognised only by the marks Starlette's ``Mount`` writes (grown ``root_path``,
the ``app_root_path`` stamped on first match), which cannot see a ``Host`` route
(marks no path) or a ``Mount("/")`` inside an app an outer mount already stamped —
both read as a route match instead. Neither is reachable through this package's
route surface.

A failure inside the emit is caught and re-reported as a loud ``audit_log:`` ERROR
line (distinct from ``audit:``, so a failure can never read as a trail entry)
carrying the same bounded route text; the request continues either way. There is
no per-request switch: ``TAI_AUDIT_LOG_ENABLE=false`` leaves this middleware out of
the stack entirely.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import NamedTuple

from starlette.types import ASGIApp, Message, Receive, Scope, Send
from tai42_contract.access_control.context import get_current_user_id

logger = logging.getLogger(__name__)

# The principal recorded when no caller is bound (public route, gate off, or an
# admitted request whose caller id the gate resolved to nothing).
ANONYMOUS = "anonymous"

# The principal recorded on a refusal with NO authenticated caller (a 401 auth
# failure, a 429 at the outer door): distinct from ``anonymous``, which the accept
# trail uses for a bound-less ADMITTED request.
UNAUTHENTICATED = "unauthenticated"

# The status recorded when the inner app raised before sending a response head —
# the outer ``ServerErrorMiddleware`` answers that case with 500, which is what
# the client sees.
_UNSTARTED_STATUS = 500

# Recorded when no route matched — the path is caller text end to end, so none of
# it is written, not even truncated. A refusal site records it too when it has no
# clean template for the request (module docstring has the refusal-line contract).
UNMATCHED_ROUTE = "<unmatched>"

# Placeholder for an ABSENT scope key, so absent-then-written reads as a change
# like any other value; a fresh object is never equal to what a router writes.
_ABSENT = object()

# Fallback route text for the emit-failure ERROR line when the failure was the
# route derivation itself — never the concrete path.
_UNRESOLVED_ROUTE = "<unresolved>"


class _MatchMarks(NamedTuple):
    """Routing marks captured on a scope at one moment: the matched route
    (FastAPI), the matched endpoint (Starlette), and the two marks only a
    ``Mount`` writes — ``root_path`` (grown by the consumed prefix) and
    ``app_root_path`` (stamped on first mount match; the only trace of a mount
    consuming an EMPTY prefix)."""

    route: object
    endpoint: object
    root_path: str
    app_root_path: object


def _match_marks(scope: Scope) -> _MatchMarks:
    return _MatchMarks(
        scope.get("route"),
        scope.get("endpoint"),
        scope.get("root_path", ""),
        scope.get("app_root_path", _ABSENT),
    )


def _crossed_a_mount(entered: _MatchMarks, matched: _MatchMarks) -> bool:
    """Whether routing handed the request to a MOUNTED app while the inner app ran.

    ``root_path``/``app_root_path`` are used rather than the endpoint mark: a
    ``Route`` may declare an ASGI app as its endpoint (FastMCP's ``/mcp``), so what
    the endpoint mark holds says nothing about mount vs. route. An ASGI middleware
    that rewrites ``root_path`` without any routing reads as a crossed mount too —
    the safe way to be wrong.
    """
    return entered.root_path != matched.root_path or entered.app_root_path != matched.app_root_path


def _templated_path(path: str, path_params: Mapping[str, object]) -> str:
    """``path`` with every matched path-parameter value rewritten to its ``{name}``.

    A value is matched as a whole run of path SEGMENTS — never cut out
    mid-segment — and a ``{name:path}`` value spanning several segments collapses
    to one ``{name}``. Values are taken longest-first, so one value containing
    another cannot shadow it.

    A value is removed EVERYWHERE it appears as such a run, not only where the
    router matched it — the two are indistinguishable here, and over-replacing is
    the safe direction: a literal segment equal to the value
    ("/api/things/things" for ``{name}`` = ``things``) yields an extra placeholder
    rather than the value itself. A value the router converted away from its raw
    text matches no run and is left as-is — parameters are only ever needles to
    remove, never the source of the returned text.
    """
    segments = path.split("/")
    named = [(name, str(value)) for name, value in path_params.items() if str(value)]
    for name, value in sorted(named, key=lambda item: len(item[1]), reverse=True):
        needle = value.split("/")
        start = 0
        while start + len(needle) <= len(segments):
            if segments[start : start + len(needle)] == needle:
                segments[start : start + len(needle)] = ["{" + name + "}"]
            start += 1
    return "/".join(segments)


def emit_audit_line(
    principal: str,
    method: str,
    route: str,
    status: int,
    duration_ms: int,
    ts: str,
    reason: str | None = None,
) -> None:
    """Write ONE ``audit:`` trail line — the single place the line format lives, so
    the accept middleware and the refusal sites (401/403/429) all render the same
    shape. ``reason`` is the refusal cause (a ``DenialCause`` value); it is rendered
    only when set, so an accept line omits it.

    On ANY failure the line is dropped and a loud ``audit_log:`` ERROR line is
    written instead (distinct prefix, so a failure can never read as a trail entry);
    the caller's request is never broken."""
    try:
        if reason is None:
            logger.info(
                "audit: principal=%s method=%s route=%s status=%d duration_ms=%d ts=%s",
                principal,
                method,
                route,
                status,
                duration_ms,
                ts,
            )
        else:
            logger.info(
                "audit: principal=%s method=%s route=%s status=%d duration_ms=%d ts=%s reason=%s",
                principal,
                method,
                route,
                status,
                duration_ms,
                ts,
                reason,
            )
    except Exception:
        logger.exception("audit_log: failed to write the audit line for %s %s", method, route)


class AuditLogMiddleware:
    """Writes the ``audit:`` line for every http request that reaches it;
    non-http scopes (lifespan, websocket) pass straight through untouched —
    neither carries a response status."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        # Stamped at arrival: the line dates the REQUEST, not the post-response write.
        ts = datetime.now(UTC).isoformat()
        status: int | None = None
        # Marks already on the scope at entry; ``_route`` treats a CHANGE against
        # these as the match signal, not mere presence.
        entered = _match_marks(scope)

        async def audited_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, audited_send)
        finally:
            # ``finally``: audits even an unhandled exception, and never swallows it.
            self._emit(scope, entered, status, int((time.perf_counter() - started) * 1000), ts)

    @staticmethod
    def _route(scope: Scope, entered: _MatchMarks) -> str:
        """The MATCHED route's path TEMPLATE, or ``<unmatched>`` (module docstring
        has the three outcomes and what each may state).

        Read AFTER the inner app runs: a mark is read as a CHANGE against
        ``entered``, never mere presence — a host mounting this app may
        pre-populate ``endpoint`` before entry.

        Check order: a matched route OBJECT wins outright (also through a mount,
        since that mark is the same proof wherever routed from); then a crossed
        mount (:func:`_crossed_a_mount`) ends templating at the mount prefix; then a
        Starlette ``endpoint`` change rebuilds the template via
        :func:`_templated_path` (a parameterless route is already its own
        template); otherwise ``<unmatched>``.

        Rebuilding groups every caller of a ``{param}`` route under one key, with
        one wrinkle: two parameters given the SAME value collapse onto the first
        parameter's name (``/{a}/{b}`` reached with ``x``/``x`` reads
        ``/{a}/{a}``), so one route can appear under more than one key —
        over-replacing, never writing the value out.
        """
        matched = _match_marks(scope)
        if matched.route is not entered.route:
            template = getattr(matched.route, "path", None)
            if isinstance(template, str):
                return template
        path_params = scope.get("path_params") or {}
        if _crossed_a_mount(entered, matched):
            # ``rstrip``: a root path declared with a trailing slash must not double
            # the separator this joins with.
            prefix = _templated_path(matched.root_path, path_params).rstrip("/")
            return f"{prefix}/{UNMATCHED_ROUTE}" if prefix else UNMATCHED_ROUTE
        if matched.endpoint is not entered.endpoint:
            return _templated_path(scope["path"], path_params)
        return UNMATCHED_ROUTE

    def _emit(self, scope: Scope, entered: _MatchMarks, status: int | None, duration_ms: int, ts: str) -> None:
        """Write the accept line without breaking the request on failure (module
        docstring has the failure contract).

        Route derivation is its OWN guarded step: a derivation failure has no safe
        route text, so it names the placeholder in a loud ERROR line and writes NO
        trail line — never the concrete path, which would leak the credential into
        the very line templating exists to keep it out of."""
        route = _UNRESOLVED_ROUTE
        try:
            route = self._route(scope, entered)
            emit_audit_line(
                get_current_user_id() or ANONYMOUS,
                scope["method"],
                route,
                _UNSTARTED_STATUS if status is None else status,
                duration_ms,
                ts,
            )
        except Exception:
            logger.exception("audit_log: failed to write the audit line for %s %s", scope.get("method"), route)
