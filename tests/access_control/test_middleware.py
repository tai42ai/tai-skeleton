"""ASGI behavior of ``ResourceGuardMiddleware``: scope passthrough, the unknown
/ public / protected route decisions, scope checks, and the request-user context
set/reset around a successful downstream call.
"""

from __future__ import annotations

import pytest
from fastmcp.server.auth import AccessToken
from starlette.authentication import AuthCredentials, AuthenticationError, UnauthenticatedUser
from tai42_contract.access_control.context import get_current_user_id
from tai42_contract.access_control.identity import AuthIdentity, IdentityProvider

from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.middleware import ResourceGuardMiddleware
from tai42_skeleton.access_control.policy import PolicyEnforcer
from tai42_skeleton.access_control.roles import EDITOR_JQ
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.access_control.user import TaiUser
from tai42_skeleton.access_control.verifier import AccessControlVerifier

from .conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

PUBLIC_ID = "public"


class _NoIdentityProvider(IdentityProvider):
    """A provider that authenticates nobody — used to drive the real verifier
    inside the guard for unauthenticated end-to-end checks."""

    async def validate_token(self, token: str) -> AuthIdentity | None:
        return None


class _FakeVerifier(AccessControlVerifier):
    """Subclasses the real verifier so it is accepted where one is expected;
    only ``resolve_resource_ids`` (the method the middleware calls) is faked."""

    def __init__(self, resource_ids: list[str]) -> None:
        self._ids = resource_ids

    async def resolve_resource_ids(self, path: str, *, policy_version: int | None = None) -> list[str]:
        return self._ids


class _RaisingVerifier(AccessControlVerifier):
    """Stands in for a verifier whose backend fetch fails (fail-closed by raise)."""

    def __init__(self) -> None:
        pass

    async def resolve_resource_ids(self, path: str, *, policy_version: int | None = None) -> list[str]:
        raise RuntimeError("redis down")


def _http_scope(path="/x", user=None, auth=None) -> dict:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "query_string": b"",
        "headers": [],
    }
    if user is not None:
        scope["user"] = user
    if auth is not None:
        scope["auth"] = auth
    return scope


def _ws_scope(path="/x", user=None, auth=None) -> dict:
    scope = {
        "type": "websocket",
        "path": path,
        "query_string": b"",
        "headers": [],
    }
    if user is not None:
        scope["user"] = user
    if auth is not None:
        scope["auth"] = auth
    return scope


async def _drive(mw: ResourceGuardMiddleware, scope, app_probe=None):
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    return sent


def _status(sent: list[dict]) -> int:
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def _body(sent: list[dict]) -> bytes:
    return next(m["body"] for m in sent if m["type"] == "http.response.body")


def _make_app(captured):
    async def app(scope, receive, send):
        captured["called"] = True
        captured["user_id_in_ctx"] = get_current_user_id()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return app


def _authed_user(user_id="u1") -> TaiUser:
    return TaiUser(AccessToken(token="t", client_id=user_id, scopes=[], claims={}))


async def test_non_http_scope_passes_through():
    captured: dict = {}
    mw = ResourceGuardMiddleware(_make_app(captured), _FakeVerifier(["x"]), PUBLIC_ID)
    sent: list = []

    async def receive():
        return {}

    async def send(m):
        sent.append(m)

    await mw({"type": "lifespan"}, receive, send)
    assert captured.get("called") is True


DISABLE_HINT = "set ACCESS_CONTROL_ENABLE=false to disable access control for local development"


async def test_unknown_route_is_forbidden(caplog):
    mw = ResourceGuardMiddleware(_make_app({}), _FakeVerifier([]), PUBLIC_ID)
    with caplog.at_level("WARNING"):
        sent = await _drive(mw, _http_scope())
    assert _status(sent) == 403
    # The server-side log names the kill switch; the client body stays generic.
    assert DISABLE_HINT in caplog.text
    assert DISABLE_HINT.encode() not in _body(sent)


async def test_public_route_allows_unauthenticated():
    captured: dict = {}
    mw = ResourceGuardMiddleware(_make_app(captured), _FakeVerifier([PUBLIC_ID]), PUBLIC_ID)
    scope = _http_scope(user=UnauthenticatedUser(), auth=AuthCredentials())
    sent = await _drive(mw, scope)
    assert captured["called"] is True
    assert _status(sent) == 200


async def test_route_matching_public_and_protected_is_treated_protected():
    # Deny wins: a path is public only when public is the ONLY resolved id. A
    # path that also matched a protected route must not be opened by an
    # over-broad public pattern — unauthenticated callers are challenged.
    mw = ResourceGuardMiddleware(_make_app({}), _FakeVerifier([PUBLIC_ID, "protected"]), PUBLIC_ID)
    scope = _http_scope(user=UnauthenticatedUser(), auth=AuthCredentials())
    sent = await _drive(mw, scope)
    assert _status(sent) == 401


async def test_protected_route_requires_authentication(caplog):
    mw = ResourceGuardMiddleware(_make_app({}), _FakeVerifier(["u1"]), PUBLIC_ID)
    scope = _http_scope(user=UnauthenticatedUser(), auth=AuthCredentials())
    with caplog.at_level("WARNING"):
        sent = await _drive(mw, scope)
    assert _status(sent) == 401
    assert DISABLE_HINT in caplog.text


async def test_protected_route_missing_scope_is_forbidden(caplog):
    mw = ResourceGuardMiddleware(_make_app({}), _FakeVerifier(["needed"]), PUBLIC_ID)
    scope = _http_scope(user=_authed_user(), auth=AuthCredentials(["other"]))
    with caplog.at_level("WARNING"):
        sent = await _drive(mw, scope)
    assert _status(sent) == 403
    # The client body is generic and does not disclose the required resource name.
    assert _body(sent) == b'{"error":"Forbidden"}'
    assert b"needed" not in _body(sent)
    # The required scope is logged server-side for operators.
    assert "needed" in caplog.text
    # The deny log also names the kill switch for local development.
    assert DISABLE_HINT in caplog.text


async def test_protected_route_with_matching_scope_runs_app_and_sets_context():
    captured: dict = {}
    mw = ResourceGuardMiddleware(_make_app(captured), _FakeVerifier(["res-a"]), PUBLIC_ID)
    scope = _http_scope(user=_authed_user("u7"), auth=AuthCredentials(["res-a"]))
    sent = await _drive(mw, scope)
    assert _status(sent) == 200
    assert captured["called"] is True
    assert captured["user_id_in_ctx"] == "u7"
    # Context is reset after the downstream call returns.
    assert get_current_user_id() is None


async def test_plugin_visible_read_sees_caller_mid_request_and_none_after():
    """A plugin reads the caller through the shared ``tai42_contract`` accessor, not
    a skeleton-internal one. Prove the guard writes the SAME context the contract
    exposes: an arbitrary downstream reader sees the caller mid-request and ``None``
    once the request unwinds."""
    from tai42_contract.access_control import get_current_user_id as contract_read

    seen: dict[str, str | None] = {}

    async def plugin_reader_app(scope, receive, send):
        # Stands in for any plugin resolving "who is calling right now".
        seen["mid_request"] = contract_read()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = ResourceGuardMiddleware(plugin_reader_app, _FakeVerifier(["res-a"]), PUBLIC_ID)
    scope = _http_scope(user=_authed_user("plugin-caller"), auth=AuthCredentials(["res-a"]))
    sent = await _drive(mw, scope)
    assert _status(sent) == 200
    assert seen["mid_request"] == "plugin-caller"
    # The binding does not outlive the request.
    assert contract_read() is None


async def test_context_is_reset_when_downstream_raises():
    """The request-user context must be reset even when the downstream app raises,
    so a failed request never leaks an authenticated identity into the next one."""

    async def raising_app(scope, receive, send):
        raise RuntimeError("downstream boom")

    mw = ResourceGuardMiddleware(raising_app, _FakeVerifier(["res-a"]), PUBLIC_ID)
    scope = _http_scope(user=_authed_user("u9"), auth=AuthCredentials(["res-a"]))
    with pytest.raises(RuntimeError, match="downstream boom"):
        await _drive(mw, scope)
    # The finally-reset ran on the exception path.
    assert get_current_user_id() is None


async def test_wildcard_scope_grants_any_resource():
    captured: dict = {}
    mw = ResourceGuardMiddleware(_make_app(captured), _FakeVerifier(["res-a"]), PUBLIC_ID)
    scope = _http_scope(user=_authed_user(), auth=AuthCredentials(["*"]))
    sent = await _drive(mw, scope)
    assert _status(sent) == 200
    assert captured["called"] is True


async def test_multiple_protected_ids_require_all_scopes(caplog):
    # Deny wins: a path resolving to several protected resources (e.g. a broad
    # tier plus a more-specific override) requires the caller to hold EVERY one —
    # a broad tier's scope alone must not open the restricted route.
    mw = ResourceGuardMiddleware(_make_app({}), _FakeVerifier(["broad", "restricted"]), PUBLIC_ID)
    scope = _http_scope(user=_authed_user(), auth=AuthCredentials(["broad"]))
    with caplog.at_level("WARNING"):
        sent = await _drive(mw, scope)
    assert _status(sent) == 403


async def test_multiple_protected_ids_allow_when_all_scopes_held():
    captured: dict = {}
    mw = ResourceGuardMiddleware(_make_app(captured), _FakeVerifier(["broad", "restricted"]), PUBLIC_ID)
    scope = _http_scope(user=_authed_user(), auth=AuthCredentials(["broad", "restricted"]))
    sent = await _drive(mw, scope)
    assert _status(sent) == 200
    assert captured["called"] is True


async def test_resolve_error_fails_closed_with_403_on_http():
    """A verifier backend error must fail closed as a clean 403 deny, never leak
    out of the middleware as a raw 500."""
    mw = ResourceGuardMiddleware(_make_app({}), _RaisingVerifier(), PUBLIC_ID)
    sent = await _drive(mw, _http_scope())
    assert _status(sent) == 403


async def test_websocket_unknown_route_closes_with_policy_violation():
    """A deny on a websocket scope must send a websocket.close frame (1008), not
    an http.response.start (which would be a malformed close)."""
    mw = ResourceGuardMiddleware(_make_app({}), _FakeVerifier([]), PUBLIC_ID)
    sent = await _drive(mw, _ws_scope())
    assert sent == [{"type": "websocket.close", "code": 1008}]


async def test_websocket_protected_route_unauthenticated_closes():
    mw = ResourceGuardMiddleware(_make_app({}), _FakeVerifier(["u1"]), PUBLIC_ID)
    scope = _ws_scope(user=UnauthenticatedUser(), auth=AuthCredentials())
    sent = await _drive(mw, scope)
    assert sent == [{"type": "websocket.close", "code": 1008}]


async def test_public_and_protected_route_denies_unauthenticated_end_to_end(monkeypatch):
    # M3 end-to-end through the REAL verifier: a path that is both a public exact
    # match and covered by a protected dynamic pattern resolves to both ids, so the
    # guard keeps it protected and challenges the unauthenticated request with 401.
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_route("/mixed", PUBLIC_ID)
    pg.add_route("/protected-template", "protected", pattern=r"^/mixed$")
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(FakeRedis()))
    real_verifier = AccessControlVerifier(settings, providers=[_NoIdentityProvider()])
    mw = ResourceGuardMiddleware(_make_app({}), real_verifier, PUBLIC_ID)
    scope = _http_scope(path="/mixed", user=UnauthenticatedUser(), auth=AuthCredentials())
    sent = await _drive(mw, scope)
    assert _status(sent) == 401


async def test_websocket_resolve_error_fails_closed_with_close():
    mw = ResourceGuardMiddleware(_make_app({}), _RaisingVerifier(), PUBLIC_ID)
    sent = await _drive(mw, _ws_scope())
    assert sent == [{"type": "websocket.close", "code": 1008}]


# -- authenticated-always-allowed carve-out ----------------------------------


class _SpyResolveVerifier(AccessControlVerifier):
    """Records every path it is asked to resolve, so a test can prove the carve-out is
    decided BEFORE (and without) route resolution."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def resolve_resource_ids(self, path: str, *, policy_version: int | None = None) -> list[str]:
        self.calls.append(path)
        return []


async def test_carve_out_authenticated_reaches_app_without_route_rows():
    # An authenticated caller reaches a carve-out path with NO route rows, and the
    # verifier's resolution is never consulted (the store is never queried).
    captured: dict = {}
    spy = _SpyResolveVerifier()
    mw = ResourceGuardMiddleware(_make_app(captured), spy, PUBLIC_ID, ("/api/auth/me",))
    scope = _http_scope(path="/api/auth/me", user=_authed_user("u1"), auth=AuthCredentials(["read"]))
    sent = await _drive(mw, scope)
    assert _status(sent) == 200
    assert captured["called"] is True
    # The identity contextvar is bound for the carved request (the /me handler reads it).
    assert captured["user_id_in_ctx"] == "u1"
    assert spy.calls == []


async def test_carve_out_unauthenticated_is_401(caplog):
    mw = ResourceGuardMiddleware(_make_app({}), _FakeVerifier([]), PUBLIC_ID, ("/api/auth/me",))
    scope = _http_scope(path="/api/auth/me", user=UnauthenticatedUser(), auth=AuthCredentials())
    with caplog.at_level("WARNING"):
        sent = await _drive(mw, scope)
    assert _status(sent) == 401
    assert DISABLE_HINT in caplog.text


async def test_carve_out_is_exact_path_not_prefix():
    # A DIFFERENT unmapped /api/auth path still 403s (CASE A) — the carve-out is
    # exact-path, so it can never swallow a future sibling route.
    mw = ResourceGuardMiddleware(_make_app({}), _FakeVerifier([]), PUBLIC_ID, ("/api/auth/me",))
    scope = _http_scope(path="/api/auth/xyz", user=_authed_user(), auth=AuthCredentials(["*"]))
    sent = await _drive(mw, scope)
    assert _status(sent) == 403


async def test_carve_out_is_exact_matching_the_jq_fence():
    # The carve-out membership test uses the EXACT request path (no trailing-slash
    # normalization) so it admits exactly the shape the companion role jq fence admits:
    # ``/api/auth/me`` exact is carved, but ``/api/auth/me/`` is NOT — it falls through to
    # resolution (here unmapped → 403), mirroring EDITOR_JQ's exact-match, so the two never
    # disagree on the trailing-slash variant — a normalizing carve-out would admit
    # ``/api/auth/me/`` here (200) yet the jq fence denies it (403).
    captured: dict = {}
    mw = ResourceGuardMiddleware(_make_app(captured), _FakeVerifier([]), PUBLIC_ID, ("/api/auth/me",))
    exact = _http_scope(path="/api/auth/me", user=_authed_user(), auth=AuthCredentials(["read"]))
    assert _status(await _drive(mw, exact)) == 200
    assert captured["called"] is True

    mw2 = ResourceGuardMiddleware(_make_app({}), _FakeVerifier([]), PUBLIC_ID, ("/api/auth/me",))
    slashed = _http_scope(path="/api/auth/me/", user=_authed_user(), auth=AuthCredentials(["read"]))
    assert _status(await _drive(mw2, slashed)) == 403

    # jq parity: the editor fence admits the exact path and denies the trailing-slash one,
    # exactly as the carve-out now does.
    enforcer = PolicyEnforcer(AccessControlSettings())
    await enforcer.enforce({"request": {"path": "/api/auth/me", "method": "GET"}}, EDITOR_JQ)
    with pytest.raises(AuthenticationError):
        await enforcer.enforce({"request": {"path": "/api/auth/me/", "method": "GET"}}, EDITOR_JQ)


async def test_no_carve_out_configured_leaves_path_to_resolution():
    # With an empty carve-out set (the default ctor value), the path falls through to the
    # normal resolution path and 403s as an unknown route.
    mw = ResourceGuardMiddleware(_make_app({}), _FakeVerifier([]), PUBLIC_ID)
    scope = _http_scope(path="/api/auth/me", user=_authed_user(), auth=AuthCredentials(["*"]))
    sent = await _drive(mw, scope)
    assert _status(sent) == 403
