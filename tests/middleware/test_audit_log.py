"""Tests for ``AuditLogMiddleware`` (see ``tai42_skeleton.middleware.audit_log``
for the full contract).

Coverage: the line's fields and what they exclude (body, query, header, token);
the three route outcomes — matched (template, never the raw path-borne value),
unmatched (bare ``<unmatched>``, driven for real via a 404, a trailing-slash
redirect, and a decoded ``%2F``, both standalone and behind the served app's own
``Mount``), and crossed-mount (prefix + ``<unmatched>`` regardless of what the
mounted app did with the tail); and that an emit failure is loud and never breaks
the request, while a disabled toggle leaves the middleware unregistered. The
sub-MCP router (``/app``) registers this middleware for ITS OWN sub-app, so a
sub-MCP request records two lines (outer ``/app/<unmatched>`` + the sub-app's own
inner route); the env prefix is ``TAI_AUDIT_LOG_``."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route
from starlette.testclient import TestClient
from starlette.types import Receive, Scope, Send
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id

from tai42_skeleton.middleware import audit_log, body_limit
from tai42_skeleton.middleware.audit_log import AuditLogMiddleware
from tai42_skeleton.middleware.body_limit import BodyLimitMiddleware
from tai42_skeleton.settings.audit_log import AuditLogSettings
from tai42_skeleton.settings.body_limit import BodyLimitSettings

# The line's exact shape and field ORDER — the record a log pipeline parses. An
# accept line ends after ``ts`` (no ``reason=``); a trailing ``reason=`` fails this
# fullmatch, which is what keeps accept lines free of a refusal cause.
_AUDIT_LINE = re.compile(
    r"^audit: principal=(?P<principal>\S+) method=(?P<method>\S+) route=(?P<route>\S+) "
    r"status=(?P<status>\d+) duration_ms=(?P<duration_ms>\d+) ts=(?P<ts>\S+)$"
)

# A refusal line shares the shape but may carry a trailing ``reason=`` (the
# DenialCause). The optional group is ``None`` when the refusal had no cause.
_REJECT_LINE = re.compile(
    r"^audit: principal=(?P<principal>\S+) method=(?P<method>\S+) route=(?P<route>\S+) "
    r"status=(?P<status>\d+) duration_ms=(?P<duration_ms>\d+) ts=(?P<ts>\S+)"
    r"(?: reason=(?P<reason>\S+))?$"
)

# The credential/body/query material a request carries and the line must never echo.
_TOKEN = "sk-live-supersecret"
_QUERY_SECRET = "querysecret"
_BODY_SECRET = "bodysecret"


def _scope(
    path: str = "/api/things/42",
    method: str = "GET",
    query_string: bytes = b"",
    headers: list[tuple[bytes, bytes]] | None = None,
    scope_type: str = "http",
) -> Scope:
    return {
        "type": scope_type,
        "method": method,
        "path": path,
        "query_string": query_string,
        "headers": headers or [],
    }


async def _run(app, scope: Scope, body: bytes = b"") -> list[dict]:
    """Drive ``app`` with a single-chunk body; collect every ASGI message it sends."""
    sent: list[dict] = []
    pending = [body]

    async def receive():
        if pending:
            return {"type": "http.request", "body": pending.pop(0), "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(dict(message))

    await app(scope, receive, send)
    return sent


async def _ok(scope: Scope, receive: Receive, send: Send) -> None:
    """Inner app: drains the body, answers 200."""
    await receive()
    await PlainTextResponse("ok")(scope, receive, send)


async def _routed_ok(scope: Scope, receive: Receive, send: Send) -> None:
    """``_ok``, plus the ``endpoint`` mark a Starlette route match leaves — a
    hand-built scope carries no routing of its own, so a scope standing in for a
    routed request has to say so itself."""
    scope["endpoint"] = _ok
    await _ok(scope, receive, send)


def _audit_lines(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith("audit: ")]


def _one_audit_line(caplog) -> re.Match[str]:
    lines = _audit_lines(caplog)
    assert len(lines) == 1, lines
    match = _AUDIT_LINE.fullmatch(lines[0])
    assert match is not None, lines[0]
    return match


async def test_authenticated_request_names_the_gate_bound_principal(caplog):
    # Principal is READ from the gate-bound caller id, never re-resolved.
    token = set_request_user_id("key_abc123")
    try:
        with caplog.at_level(logging.INFO):
            sent = await _run(AuditLogMiddleware(_routed_ok), _scope(path="/api/things/42", method="POST"))
    finally:
        reset_request_user_id(token)

    assert sent[0]["status"] == 200
    fields = _one_audit_line(caplog).groupdict()
    assert fields["principal"] == "key_abc123"
    assert fields["method"] == "POST"
    assert fields["route"] == "/api/things/42"
    assert fields["status"] == "200"
    assert int(fields["duration_ms"]) >= 0
    # ts is an ISO-8601 instant carrying its UTC offset, not a naive local stamp.
    assert datetime.fromisoformat(fields["ts"]).tzinfo is not None


async def test_unauthenticated_request_is_audited_as_anonymous(caplog):
    # A public route (and a deployment with the gate off) leaves no caller bound.
    with caplog.at_level(logging.INFO):
        await _run(AuditLogMiddleware(_routed_ok), _scope(path="/health"))

    fields = _one_audit_line(caplog).groupdict()
    assert fields["principal"] == "anonymous"
    assert fields["route"] == "/health"


async def test_the_audit_record_is_emitted_at_info(caplog):
    # INFO, not WARNING (not page-worthy) or DEBUG (a prod log config could drop it).
    with caplog.at_level(logging.DEBUG):
        await _run(AuditLogMiddleware(_ok), _scope())

    (record,) = [r for r in caplog.records if r.getMessage().startswith("audit: ")]
    assert record.levelno == logging.INFO


async def test_route_template_wins_over_the_concrete_path(caplog):
    # A route OBJECT on the scope (FastAPI's) wins outright — no reconstruction attempted.
    async def routed(scope: Scope, receive: Receive, send: Send) -> None:
        scope["route"] = SimpleNamespace(path="/api/things/{thing_id}")
        await _ok(scope, receive, send)

    with caplog.at_level(logging.INFO):
        await _run(AuditLogMiddleware(routed), _scope(path="/api/things/42"))

    assert _one_audit_line(caplog).group("route") == "/api/things/{thing_id}"


# -- Real routers: the path-borne credential never reaches the line -----------
#
# Drives the REAL middleware through a REAL router (Starlette's, which records no
# route object) and asserts both halves: the template shows up, the concrete value
# does not.

_TICKET = "tkt_9f3a2b7c"


def _module_log_text(caplog) -> str:
    """Everything this middleware's logger emitted — formatted message AND raw args, so
    a secret riding along as an unformatted argument is caught too. Scoped to the
    module's own records: the test client's HTTP logger names the URL by design."""
    return "\n".join(
        record.getMessage() + repr(record.args) for record in caplog.records if record.name == audit_log.logger.name
    )


async def _hello(request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _starlette_app() -> Starlette:
    return Starlette(
        routes=[
            Route("/trigger/{token}", _hello, methods=["GET", "POST"]),
            Route("/api/interactions/callback/{ticket}", _hello, methods=["GET"]),
            Route("/health", _hello, methods=["GET"]),
            Route("/files/{rest:path}", _hello, methods=["GET"]),
            Route("/api/things/{name}", _hello, methods=["GET"]),
            Route("/x/{q}/{p:path}", _hello, methods=["GET"]),
            Mount("/mnt", routes=[Route("/x/{sub}", _hello, methods=["GET"])]),
        ]
    )


@pytest.mark.parametrize(
    ("url", "route", "value"),
    [
        # The trigger link: the token IS the capability, so it may never be logged.
        (f"/trigger/{_TOKEN}", "/trigger/{token}", _TOKEN),
        # The interactions callback ticket: same, a bearer secret in the path.
        (f"/api/interactions/callback/{_TICKET}", "/api/interactions/callback/{ticket}", _TICKET),
        # A ``:path`` parameter spanning several segments collapses to one placeholder.
        ("/files/a/b/c.txt", "/files/{rest}", "a/b/c.txt"),
        # A value equal to a LITERAL route segment: over-replacing wins, both
        # segments become the placeholder.
        ("/trigger/trigger", "/{token}/{token}", "trigger"),
    ],
)
def test_starlette_router_audits_the_template_never_the_value(caplog, url: str, route: str, value: str):
    app = _starlette_app()
    app.add_middleware(AuditLogMiddleware)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        assert client.get(url).status_code == 200

    assert _one_audit_line(caplog).group("route") == route
    assert value not in _module_log_text(caplog)


def test_starlette_router_audits_a_parameterless_route_as_its_own_path(caplog):
    # Nothing to template: a static route IS its own template and is recorded verbatim.
    app = _starlette_app()
    app.add_middleware(AuditLogMiddleware)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert _one_audit_line(caplog).group("route") == "/health"


@pytest.mark.parametrize(
    ("url", "route"),
    [
        # Matched as a whole SEGMENT run, never a substring: ``thing`` sits inside
        # the route's own literal ``things``; a plain replace would wrongly cut it
        # mid-segment.
        ("/api/things/thing", "/api/things/{name}"),
        # Longest value FIRST: ``{p:path}`` holds ``a/b``, ``{q}`` holds ``a``;
        # short-first would strand a ``b`` segment in the line.
        ("/x/a/a/b", "/x/{q}/{p}"),
        # An EMPTY ``:path`` value is no needle: unskipped, it would match every
        # empty segment and stamp placeholders over the path's own edges.
        ("/files/", "/files/"),
    ],
)
def test_starlette_router_rebuilds_the_template_from_whole_segments(caplog, url: str, route: str):
    app = _starlette_app()
    app.add_middleware(AuditLogMiddleware)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        assert client.get(url).status_code == 200

    assert _one_audit_line(caplog).group("route") == route


def test_a_route_whose_endpoint_is_an_asgi_app_still_audits_its_template(caplog):
    # FastMCP declares ``/mcp`` this way — an ASGI app as the endpoint — which is
    # why the endpoint mark alone can't prove mount vs. route. Full match, so it
    # keeps its template like any other.
    class _AsgiEndpoint:
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            await PlainTextResponse("ok")(scope, receive, send)

    app = Starlette(routes=[Route("/mcp", endpoint=_AsgiEndpoint(), methods=["GET"])])
    app.add_middleware(AuditLogMiddleware)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        assert client.get("/mcp").status_code == 200

    assert _one_audit_line(caplog).group("route") == "/mcp"


def test_fastapi_router_audits_the_route_objects_template(caplog):
    # FastAPI records the matched route object, so its own ``path`` is the
    # template.
    app = FastAPI()

    @app.get("/api/things/{thing_id}")
    async def thing(thing_id: str) -> dict[str, str]:
        return {"thing_id": thing_id}

    app.add_middleware(AuditLogMiddleware)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        assert client.get(f"/api/things/{_TOKEN}").status_code == 200

    assert _one_audit_line(caplog).group("route") == "/api/things/{thing_id}"
    assert _TOKEN not in _module_log_text(caplog)


# -- No route matched: not one segment of the path reaches the line -----------
#
# Near-misses in the shapes a path-borne credential actually arrives in.


@pytest.mark.parametrize(
    ("url", "status"),
    [
        # One segment too many: the token is still spelled out in the path.
        (f"/trigger/{_TOKEN}/extra", 404),
        # Trailing slash: the router redirects (client follows), token still live.
        (f"/trigger/{_TOKEN}/", 307),
        # ``%2F`` decodes before routing, growing a segment break that matches nothing.
        (f"/trigger/{_TOKEN}%2Fextra", 404),
    ],
    ids=["extra-segment-404", "trailing-slash-redirect", "percent-encoded-slash"],
)
def test_an_unmatched_path_audits_as_the_constant_never_its_own_text(caplog, url: str, status: int):
    app = _starlette_app()
    app.add_middleware(AuditLogMiddleware)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        assert client.get(url, follow_redirects=False).status_code == status

    assert _one_audit_line(caplog).group("route") == "<unmatched>"
    assert _TOKEN not in _module_log_text(caplog)


def test_a_mounted_deployment_still_tells_a_match_from_a_miss(caplog):
    # The shape this app actually serves in (``asgi.create_app`` mounts it under
    # ``Mount("/")``): the mount pre-populates ``endpoint`` before this middleware
    # runs, so presence alone proves nothing — only a mark inner routing replaces does.
    def hosted() -> Starlette:
        inner = _starlette_app()
        inner.add_middleware(AuditLogMiddleware)
        return Starlette(routes=[Mount("/", app=inner)])

    with caplog.at_level(logging.INFO), TestClient(hosted()) as client:
        assert client.get(f"/trigger/{_TOKEN}").status_code == 200
    assert _one_audit_line(caplog).group("route") == "/trigger/{token}"
    assert _TOKEN not in _module_log_text(caplog)

    caplog.clear()
    with caplog.at_level(logging.INFO), TestClient(hosted()) as client:
        assert client.get(f"/trigger/{_TOKEN}/extra").status_code == 404
    assert _one_audit_line(caplog).group("route") == "<unmatched>"
    assert _TOKEN not in _module_log_text(caplog)


# -- A crossed mount ends the templating at the mount's own prefix ------------
#
# The served app's ``/app/{slug}`` sub-MCP router is exactly this shape: prefix
# stated, tail is caller text whenever the slug is unknown.


class _OpaqueSubApp:
    """A mounted ASGI app that routes for itself and records NOTHING on the scope —
    the sub-MCP router's shape. It serves ``/served`` tails and turns the rest away,
    so a test can drive both and see the audit line stay the same either way."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.seen.append(scope["path"])
        status = 200 if scope["path"].endswith("/served") else 404
        await PlainTextResponse("sub", status_code=status)(scope, receive, send)


@pytest.mark.parametrize(
    ("tail", "status"),
    [("served", 200), ("missing", 404)],
    ids=["sub-app-served-it", "sub-app-turned-it-away"],
)
def test_a_mounted_opaque_app_audits_its_prefix_and_never_the_tail(caplog, tail: str, status: int):
    sub = _OpaqueSubApp()
    app = Starlette(routes=[Route("/health", _hello), Mount("/app", app=sub)])
    app.add_middleware(AuditLogMiddleware)

    url = f"/app/{_TOKEN}/{tail}"
    with caplog.at_level(logging.INFO), TestClient(app) as client:
        assert client.get(url).status_code == status

    assert sub.seen == [url]  # the sub-app really was handed the whole tail
    assert _one_audit_line(caplog).group("route") == "/app/<unmatched>"
    assert _TOKEN not in _module_log_text(caplog)


@pytest.mark.parametrize(
    ("url", "status"),
    [("/mnt/x/" + _TOKEN, 200), ("/mnt/static", 200), ("/mnt/" + _TOKEN, 404)],
    ids=["inner-route-with-a-param", "inner-static-route", "inner-router-turned-it-away"],
)
def test_a_mounted_starlette_router_audits_the_prefix_not_its_inner_template(caplog, url: str, status: int):
    # A mounted Starlette router that MATCHES is indistinguishable from one where
    # nothing inside it matched — both carry the mount's grown root path and an
    # endpoint mark. The inner template isn't provable, so the tail is dropped
    # either way.
    app = Starlette(routes=[Mount("/mnt", routes=[Route("/x/{sub}", _hello), Route("/static", _hello)])])
    app.add_middleware(AuditLogMiddleware)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        assert client.get(url).status_code == status

    assert _one_audit_line(caplog).group("route") == "/mnt/<unmatched>"
    assert _TOKEN not in _module_log_text(caplog)


def test_a_mounted_fastapi_router_keeps_its_full_template(caplog):
    # The one mounted router whose match IS provable: FastAPI's route-object mark
    # is written by nothing else, so a mounted FastAPI app keeps full templating.
    sub = FastAPI()

    @sub.get("/thing/{thing_id}")
    async def thing(thing_id: str) -> dict[str, str]:
        return {"thing_id": thing_id}

    app = Starlette(routes=[Mount("/fa", app=sub)])
    app.add_middleware(AuditLogMiddleware)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        assert client.get(f"/fa/thing/{_TOKEN}").status_code == 200

    assert _one_audit_line(caplog).group("route") == "/thing/{thing_id}"
    assert _TOKEN not in _module_log_text(caplog)


@pytest.mark.parametrize(
    ("routes", "url", "route"),
    [
        # A PARAMETERISED mount point: the prefix templates too, tenant value as ``{name}``.
        (
            [Mount("/t/{tenant}", app=_OpaqueSubApp())],
            "/t/acme-corp/" + _TOKEN,
            "/t/{tenant}/<unmatched>",
        ),
        # Nested mounts: the root path the innermost mount left is the whole prefix.
        (
            [Mount("/nest", routes=[Mount("/inner", app=_OpaqueSubApp())])],
            "/nest/inner/" + _TOKEN,
            "/nest/inner/<unmatched>",
        ),
        # A zero-width mount has nothing to state — bare constant. It grows no root
        # path, caught instead by the ``app_root_path`` mark it stamps.
        ([Mount("/", app=_OpaqueSubApp())], "/" + _TOKEN, "<unmatched>"),
    ],
    ids=["parameterised-mount-point", "nested-mounts", "zero-width-mount"],
)
def test_the_mount_prefix_is_route_text_and_is_stated_as_such(caplog, routes, url: str, route: str):
    app = Starlette(routes=list(routes))
    app.add_middleware(AuditLogMiddleware)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        assert client.get(url).status_code == 404

    assert _one_audit_line(caplog).group("route") == route
    assert _TOKEN not in _module_log_text(caplog)
    assert "acme-corp" not in _module_log_text(caplog)


def test_a_mounted_deployment_audits_the_mount_prefix_under_its_host_prefix(caplog):
    # The two mounts stack: the HOST's (already on the scope at entry) and the
    # app's OWN — the line names both and stops there.
    def hosted() -> Starlette:
        inner = Starlette(routes=[Route("/health", _hello), Mount("/app", app=_OpaqueSubApp())])
        inner.add_middleware(AuditLogMiddleware)
        return Starlette(routes=[Mount("/x", app=inner)])

    with caplog.at_level(logging.INFO), TestClient(hosted()) as client:
        assert client.get("/x/health").status_code == 200
    assert _one_audit_line(caplog).group("route") == "/x/health"

    caplog.clear()
    with caplog.at_level(logging.INFO), TestClient(hosted()) as client:
        assert client.get(f"/x/app/{_TOKEN}").status_code == 404
    assert _one_audit_line(caplog).group("route") == "/x/app/<unmatched>"
    assert _TOKEN not in _module_log_text(caplog)


async def test_a_failed_route_derivation_reports_the_placeholder_not_the_path(caplog, monkeypatch):
    # When the DERIVATION itself fails there is no safe route text — the placeholder
    # is named instead of the concrete path, which would leak the credential into
    # the very error line templating exists to keep it out of.
    def _explode(*args, **kwargs):
        raise RuntimeError("route derivation failed")

    monkeypatch.setattr(AuditLogMiddleware, "_route", _explode)

    with caplog.at_level(logging.INFO):
        sent = await _run(AuditLogMiddleware(_ok), _scope(path=f"/trigger/{_TOKEN}"))

    assert sent[0]["status"] == 200  # the request is served regardless
    assert _audit_lines(caplog) == []
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "<unresolved>" in errors[0].getMessage()
    assert _TOKEN not in _module_log_text(caplog)


async def test_status_and_duration_report_the_real_outcome(caplog):
    async def teapot(scope: Scope, receive: Receive, send: Send) -> None:
        await PlainTextResponse("nope", status_code=418)(scope, receive, send)

    with caplog.at_level(logging.INFO):
        await _run(AuditLogMiddleware(teapot), _scope())

    assert _one_audit_line(caplog).group("status") == "418"


async def test_unhandled_exception_is_audited_and_still_propagates(caplog):
    # Dies before a response head: audited as the 500 ServerErrorMiddleware answers
    # with, and the exception still propagates.
    async def boom(scope: Scope, receive: Receive, send: Send) -> None:
        raise RuntimeError("inner app failed")

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="inner app failed"):
        await _run(AuditLogMiddleware(boom), _scope())

    assert _one_audit_line(caplog).group("status") == "500"


async def test_a_raise_after_the_response_head_audits_the_status_the_client_saw(caplog):
    # A head already went out when the app died mid-body; ServerErrorMiddleware
    # re-raises rather than overwriting an already-started response, so the
    # audited outcome is the SENT status, not 500.
    async def half_sent(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("died after the head")

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="died after the head"):
        await _run(AuditLogMiddleware(half_sent), _scope())

    assert _one_audit_line(caplog).group("status") == "200"


async def test_non_http_scope_passes_through_unaudited(caplog):
    # A websocket carries no response status and lifespan is not a request.
    seen: list[str] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    with caplog.at_level(logging.INFO):
        await _run(AuditLogMiddleware(inner), _scope(scope_type="websocket"))

    assert seen == ["websocket"]
    assert _audit_lines(caplog) == []


async def test_no_credential_body_or_query_material_reaches_the_line(caplog):
    # Query string, header, and body all present at once: the PATH is the only
    # request text the record may contain.
    scope = _scope(
        path="/api/things/42",
        method="POST",
        query_string=f"api_key={_QUERY_SECRET}&q=hello".encode(),
        headers=[
            (b"authorization", f"Bearer {_TOKEN}".encode()),
            (b"cookie", b"session=cookiesecret"),
            (b"content-type", b"application/json"),
        ],
    )
    token = set_request_user_id("key_abc123")
    try:
        with caplog.at_level(logging.INFO):
            await _run(AuditLogMiddleware(_routed_ok), scope, body=f'{{"password": "{_BODY_SECRET}"}}'.encode())
    finally:
        reset_request_user_id(token)

    records = [record for record in caplog.records if record.getMessage().startswith("audit: ")]
    assert len(records) == 1
    # Both the formatted line and the raw record args are checked: a secret must not
    # ride along as an unformatted argument a structured handler would emit.
    haystack = records[0].getMessage() + repr(records[0].args)
    for secret in (_TOKEN, _QUERY_SECRET, _BODY_SECRET, "cookiesecret", "api_key", "authorization", "Bearer"):
        assert secret not in haystack


async def test_emit_failure_is_loud_and_the_request_still_succeeds(caplog, monkeypatch):
    def _explode(*args, **kwargs):
        raise RuntimeError("log sink unavailable")

    monkeypatch.setattr(audit_log.logger, "info", _explode)

    with caplog.at_level(logging.INFO):
        sent = await _run(AuditLogMiddleware(_ok), _scope())

    assert sent[0]["status"] == 200  # the request is served regardless
    assert _audit_lines(caplog) == []  # nothing pretends to be an audit record
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert errors[0].getMessage().startswith("audit_log: failed to write the audit line")
    assert errors[0].exc_info is not None  # the cause is carried, not just named


def _capped_trigger_app(reached: list[str]) -> Starlette:
    """``/trigger/{token}`` behind the body cap, audited outermost — the base-app order.

    ``reached`` records that the endpoint RAN, which is what tells the cap's two
    rejects apart: both answer with the same JSON, so the response body doesn't.
    """

    async def eats_body(request) -> PlainTextResponse:
        reached.append(request.url.path)
        await request.body()
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/trigger/{token}", eats_body, methods=["POST"])])
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(AuditLogMiddleware)  # outermost, as in the base-app stack
    return app


def test_an_up_front_over_cap_413_audits_the_unmatched_constant(caplog, monkeypatch):
    # The audit middleware wraps the body cap, so a reject is still audited with the
    # REAL status (413), not the inner app's would-be outcome. This is the UP-FRONT
    # reject: a declared ``Content-Length`` already over the cap refuses before the
    # router runs (``reached`` stays empty, pinning the branch) — so ``<unmatched>``
    # is earned by answering above the router, not by there being nothing to match.
    # The sibling test below drives the OTHER reject, which trips mid-body after a match.
    monkeypatch.setattr(body_limit, "body_limit_settings", lambda: BodyLimitSettings(max_body_bytes=8))

    reached: list[str] = []
    with caplog.at_level(logging.INFO), TestClient(_capped_trigger_app(reached)) as client:
        response = client.post(f"/trigger/{_TOKEN}", content=b"x" * 64)

    assert response.status_code == 413
    assert response.request.headers["content-length"] == "64"
    assert reached == []  # answered above the router: the up-front reject, not the escape

    fields = _one_audit_line(caplog).groupdict()
    assert fields["status"] == "413"
    assert fields["route"] == "<unmatched>"
    assert _TOKEN not in _module_log_text(caplog)


def test_an_over_the_wire_over_cap_413_audits_the_matched_template(caplog, monkeypatch):
    # The cap's OTHER reject, MID-BODY: a chunked body declares no ``Content-Length``,
    # so the up-front branch has nothing to refuse and the request is ROUTED first
    # (``reached`` says so) — the cap only trips while the endpoint reads the body.
    # So this 413 carries the matched route's TEMPLATE, not ``<unmatched>``.
    monkeypatch.setattr(body_limit, "body_limit_settings", lambda: BodyLimitSettings(max_body_bytes=8))

    def chunked():
        for _ in range(8):
            yield b"x" * 16

    reached: list[str] = []
    with caplog.at_level(logging.INFO), TestClient(_capped_trigger_app(reached)) as client:
        response = client.post(f"/trigger/{_TOKEN}", content=chunked())

    assert response.status_code == 413
    assert "content-length" not in response.request.headers  # chunked: nothing to refuse up front
    assert reached == [f"/trigger/{_TOKEN}"]  # the router ran: the escape, not the up-front reject

    fields = _one_audit_line(caplog).groupdict()
    assert fields["status"] == "413"
    assert fields["route"] == "/trigger/{token}"
    assert _TOKEN not in _module_log_text(caplog)


def test_enabled_by_default_the_middleware_is_registered(monkeypatch):
    # Registration, not behavior: the audit middleware is IN the base-app stack, and
    # OUTSIDE the body cap so an over-cap 413 is audited like any other outcome.
    from tai42_skeleton.app import instance, server

    monkeypatch.setattr(server, "audit_log_settings", lambda: AuditLogSettings())
    classes = [middleware.cls for middleware in instance.build_app()._base_middleware(None)]

    assert classes[0] is AuditLogMiddleware
    assert classes[1] is BodyLimitMiddleware


def test_disabled_toggle_leaves_the_middleware_unregistered(monkeypatch):
    # Off means ABSENT from the stack — zero overhead — never a registered no-op.
    from tai42_skeleton.app import instance, server

    monkeypatch.setattr(server, "audit_log_settings", lambda: AuditLogSettings(enable=False))
    classes = [middleware.cls for middleware in instance.build_app()._base_middleware(None)]

    assert AuditLogMiddleware not in classes


@pytest.mark.parametrize("launch", ["http_app", "sse_app"])
def test_the_served_app_stack_puts_the_audit_middleware_inside_the_gate(launch: str):
    # The audit middleware sits INSIDE the gate, pinned on the REAL stack the app
    # that actually serves traffic builds, not the middleware list handed to the builder.
    from tai42_skeleton.access_control.adapter import AuthAdapter
    from tai42_skeleton.access_control.settings import AccessControlSettings
    from tai42_skeleton.app.server import TaiMCP

    gate = AuthAdapter(AccessControlSettings(enable=True))
    gate_classes = [middleware.cls for middleware in gate.get_middleware()]
    assert gate_classes, "the gate registered no middleware — this pins nothing"

    served = getattr(TaiMCP(name="audit-stack-under-test", auth=gate), launch)()
    # ``finalize`` wraps the base app in the outer rate limiter and records the
    # lifespan-bearing base app, whose Starlette stack is the one under test.
    classes = [middleware.cls for middleware in served.mcp_lifespan_app.user_middleware]

    assert AuditLogMiddleware in classes, classes
    audit_at = classes.index(AuditLogMiddleware)
    for cls in gate_classes:
        assert classes.index(cls) < audit_at, classes
    # And it stays OUTSIDE the body cap, so an over-cap 413 is audited.
    assert classes.index(BodyLimitMiddleware) == audit_at + 1, classes


# -- The TAI_AUDIT_LOG_ env prefix --------------------------------------------


def test_the_tai_audit_log_env_prefix_controls_the_enable_toggle(monkeypatch):
    # The switch reads ``TAI_AUDIT_LOG_ENABLE`` (the TAI_BODY_LIMIT_/TAI_RATE_LIMIT_
    # family).
    monkeypatch.setenv("TAI_AUDIT_LOG_ENABLE", "false")
    assert AuditLogSettings().enable is False
    monkeypatch.setenv("TAI_AUDIT_LOG_ENABLE", "true")
    assert AuditLogSettings().enable is True


# -- The sub-MCP router registers the audit middleware for its OWN sub-app -----


async def _built_sub_app(router, slug: str):
    await router.register_sub_mcp_app(slug, [], transport="http")
    sub_app = await router._get_or_build_app(slug)
    assert sub_app is not None
    return sub_app


async def test_sub_mcp_sub_app_registers_the_audit_middleware_outside_the_body_cap(monkeypatch):
    # Task 2: the sub-MCP router audits ITS OWN sub-app, gated on the SAME switch and
    # positioned OUTSIDE the body cap (audit wraps the cap, the base-app order).
    from tai42_skeleton.app import instance, sub_mcp_app
    from tai42_skeleton.manifest import Manifest

    monkeypatch.setattr(sub_mcp_app, "audit_log_settings", lambda: AuditLogSettings())

    async with instance.app.app_context(Manifest.model_validate({})):
        router = instance.app.sub_app.mcp_sub_app_router
        async with router.lifespan(None):  # type: ignore[arg-type]
            sub_app = await _built_sub_app(router, "svc")
            classes = [m.cls for m in sub_app.user_middleware]
            assert AuditLogMiddleware in classes, classes
            assert BodyLimitMiddleware in classes, classes
            assert classes.index(AuditLogMiddleware) < classes.index(BodyLimitMiddleware), classes


async def test_sub_mcp_sub_app_omits_the_audit_middleware_when_disabled(monkeypatch):
    # Off means ABSENT from the sub-app's own stack — never a registered no-op — while
    # the body cap (always on) stays.
    from tai42_skeleton.app import instance, sub_mcp_app
    from tai42_skeleton.manifest import Manifest

    monkeypatch.setattr(sub_mcp_app, "audit_log_settings", lambda: AuditLogSettings(enable=False))

    async with instance.app.app_context(Manifest.model_validate({})):
        router = instance.app.sub_app.mcp_sub_app_router
        async with router.lifespan(None):  # type: ignore[arg-type]
            sub_app = await _built_sub_app(router, "svc")
            classes = [m.cls for m in sub_app.user_middleware]
            assert AuditLogMiddleware not in classes, classes
            assert BodyLimitMiddleware in classes, classes


async def test_a_sub_mcp_request_records_both_the_outer_and_the_inner_audit_line(caplog, monkeypatch):
    # Task 2 end to end: one sub-MCP request writes TWO audit lines — the base-app
    # (outer) line templated to ``/app/<unmatched>`` (the mount's whole tail is the
    # sub-app's business), and the sub-app's OWN (inner) line for its own route. Drives
    # the REAL router + REAL built sub-app under an outer AuditLogMiddleware + a real
    # ``Mount("/app")`` so the outer line earns its crossed-mount ``/app/<unmatched>``.
    from tai42_skeleton.app import instance, sub_mcp_app
    from tai42_skeleton.manifest import Manifest

    monkeypatch.setattr(sub_mcp_app, "audit_log_settings", lambda: AuditLogSettings())

    async with instance.app.app_context(Manifest.model_validate({})):
        router = instance.app.sub_app.mcp_sub_app_router
        async with router.lifespan(None):  # type: ignore[arg-type]
            await _built_sub_app(router, "svc")
            outer = AuditLogMiddleware(Starlette(routes=[Mount("/app", app=router)]))

            with caplog.at_level(logging.INFO):
                await _run(outer, _scope(path="/app/svc/anything", method="GET"))

    lines = _audit_lines(caplog)
    assert len(lines) == 2, lines
    routes = {_AUDIT_LINE.fullmatch(line).group("route") for line in lines}  # type: ignore[union-attr]
    # The outer base-app line names the mount prefix and stops; the inner line is the
    # sub-app's OWN and is a different route (never the base app's prefix marker).
    assert "/app/<unmatched>" in routes
    assert routes != {"/app/<unmatched>"}
    assert _TOKEN not in _module_log_text(caplog)


# -- Refusal lines at the refusal sites (401 / 403; 429 lives with the limiter) ---


def _one_reject_line(caplog) -> re.Match[str]:
    lines = _audit_lines(caplog)
    assert len(lines) == 1, lines
    match = _REJECT_LINE.fullmatch(lines[0])
    assert match is not None, lines[0]
    return match


def _driven_auth_error(caplog, backend):
    """Drive a request through the REAL ``AuthenticationMiddleware`` with
    ``handle_auth_error`` as its ``on_error`` and the given failing ``backend``.
    Returns ``(refusal_line_match, response)`` — the single refusal audit line the
    site emitted plus the HTTP response the caller saw."""
    from starlette.middleware.authentication import AuthenticationMiddleware

    from tai42_skeleton.access_control.adapter import handle_auth_error

    app = Starlette(routes=[Route("/whatever", _hello, methods=["GET"])])
    app.add_middleware(AuthenticationMiddleware, backend=backend, on_error=handle_auth_error)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        response = client.get("/whatever")
    return _one_reject_line(caplog), response


def test_a_401_auth_failure_emits_one_refusal_line_marked_unauthenticated(caplog):
    from starlette.authentication import AuthenticationBackend, AuthenticationError

    class _Failing(AuthenticationBackend):
        async def authenticate(self, conn):
            raise AuthenticationError("Invalid API key")

    match, response = _driven_auth_error(caplog, _Failing())
    assert response.status_code == 401
    # No caller bound at a 401 — the explicit ``unauthenticated`` marker, never
    # ``anonymous`` (which the accept trail uses for a bound-less admitted request).
    assert match.group("principal") == "unauthenticated"
    assert match.group("status") == "401"
    assert match.group("reason") is None  # a 401 carries no DenialCause


def test_a_403_authorization_denial_emits_one_refusal_line_carrying_the_cause(caplog):
    from starlette.authentication import AuthenticationBackend

    from tai42_skeleton.access_control.backend import AuthorizationError
    from tai42_skeleton.access_control.role_gate import DenialCause

    class _Denied(AuthenticationBackend):
        async def authenticate(self, conn):
            raise AuthorizationError("Access Denied", cause=DenialCause.HARD_FENCE)

    match, response = _driven_auth_error(caplog, _Denied())
    assert response.status_code == 403
    # A 403 is authenticated-but-denied; the caller id is not bound at this pre-gate
    # site, so it reads as ``anonymous`` while the ``reason=`` (DenialCause) is what
    # marks the line a refusal.
    assert match.group("status") == "403"
    assert match.group("reason") == DenialCause.HARD_FENCE.value


def test_a_403_without_a_cause_omits_the_reason_field(caplog):
    # Not every AuthorizationError carries a DenialCause (a no-policy/disabled deny does
    # not) — the ``reason=`` field is then absent, not rendered empty.
    from starlette.authentication import AuthenticationBackend

    from tai42_skeleton.access_control.backend import AuthorizationError

    class _Denied(AuthenticationBackend):
        async def authenticate(self, conn):
            raise AuthorizationError("Access Denied")

    match, response = _driven_auth_error(caplog, _Denied())
    assert response.status_code == 403
    assert match.group("reason") is None
