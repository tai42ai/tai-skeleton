"""``authz.check``: the AC-disabled/internal/no-identity gates, path synthesis,
the scope + jq-fence decision, and HTTP↔MCP parity for the same operation."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from fastmcp.server.auth import AccessToken
from starlette.middleware.authentication import AuthenticationMiddleware
from tai_contract.access_control import OWNER_USER_ID_CLAIM

from tai_skeleton.access_control.adapter import handle_auth_error
from tai_skeleton.access_control.backend import AccessControlAuthBackend
from tai_skeleton.access_control.middleware import ResourceGuardMiddleware
from tai_skeleton.access_control.settings import AccessControlSettings
from tai_skeleton.access_control.verifier import AccessControlVerifier
from tai_skeleton.authz import check, synthesize_path
from tai_skeleton.authz.identity import INTERNAL_PRINCIPAL, CallerIdentity, resolve_caller_identity
from tai_skeleton.operations import OperationRegistry, operation
from tai_skeleton.operations.errors import PermissionDenied


def _op(reg, route="/api/things/wipe", method="POST"):
    @operation(name="wipe", summary="Wipe", tags=["things"], registry=reg)
    async def wipe(**_):
        return {}

    meta = reg.get("wipe")
    meta.route_template = route
    meta.http_method = method
    return meta


# -- path synthesis ----------------------------------------------------------


def test_synthesize_path_substitutes_path_args():
    reg = OperationRegistry()
    meta = _op(reg, route="/api/mcp-status/{title}/deregister")
    assert synthesize_path(meta, {"title": "srv"}) == "/api/mcp-status/srv/deregister"


def test_synthesize_path_preserves_slash_in_path_converter_value():
    reg = OperationRegistry()
    meta = _op(reg, route="/api/resources/{resource_id:path}")
    assert synthesize_path(meta, {"resource_id": "a/b/c"}) == "/api/resources/a/b/c"


def test_synthesize_path_missing_arg_denies():
    reg = OperationRegistry()
    meta = _op(reg, route="/api/x/{id}")
    with pytest.raises(PermissionDenied):
        synthesize_path(meta, {})


# -- the gates ---------------------------------------------------------------


def test_access_control_disabled_allows_everything():
    reg = OperationRegistry()
    meta = _op(reg)
    settings = AccessControlSettings(enable=False)
    asyncio.run(check(CallerIdentity(user_id=None), meta, {}, settings=settings))  # no raise


def test_internal_principal_allowed():
    reg = OperationRegistry()
    meta = _op(reg)
    asyncio.run(check(INTERNAL_PRINCIPAL, meta, {}, settings=AccessControlSettings()))  # no raise


def test_external_no_identity_denied():
    reg = OperationRegistry()
    meta = _op(reg)
    with pytest.raises(PermissionDenied, match="no caller identity"):
        asyncio.run(check(CallerIdentity(user_id=None), meta, {}, settings=AccessControlSettings()))


def test_unknown_route_denied(ac_env, bound_app):
    reg = OperationRegistry()
    meta = _op(reg)  # no route seeded in pg
    with pytest.raises(PermissionDenied, match="no resource configured"):
        asyncio.run(check(CallerIdentity(user_id="alice"), meta, {}, settings=AccessControlSettings()))


def test_public_route_allowed(ac_env, bound_app):
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", settings.public_resource_id)
    reg = OperationRegistry()
    meta = _op(reg)
    asyncio.run(check(CallerIdentity(user_id="alice"), meta, {}, settings=settings))  # no raise


def test_scoped_caller_allowed_unscoped_denied(ac_env, bound_app):
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.add_policy("alice", scopes=["things"])
    ac_env.add_policy("bob", scopes=[])
    reg = OperationRegistry()
    meta = _op(reg)

    asyncio.run(check(CallerIdentity(user_id="alice"), meta, {}, settings=settings))
    with pytest.raises(PermissionDenied, match="insufficient scope"):
        asyncio.run(check(CallerIdentity(user_id="bob"), meta, {}, settings=settings))


def test_wildcard_scope_allowed(ac_env, bound_app):
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.add_policy("root", scopes=["*"])
    reg = OperationRegistry()
    meta = _op(reg)
    asyncio.run(check(CallerIdentity(user_id="root"), meta, {}, settings=settings))


def test_disabled_principal_denied(ac_env, bound_app):
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.add_policy("alice", scopes=["things"], policy_data={"disabled": True})
    reg = OperationRegistry()
    meta = _op(reg)
    with pytest.raises(PermissionDenied, match="disabled"):
        asyncio.run(check(CallerIdentity(user_id="alice"), meta, {}, settings=settings))


def test_jq_fence_denies_over_synthesized_path(ac_env, bound_app):
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    # A fence that only allows GET; a POST synthesized path is rejected.
    ac_env.add_policy("alice", scopes=["things"], condition='.request.method == "GET"')
    reg = OperationRegistry()
    meta = _op(reg, method="POST")
    with pytest.raises(PermissionDenied, match="policy condition rejected"):
        asyncio.run(check(CallerIdentity(user_id="alice"), meta, {}, settings=settings))


def test_jq_fence_allows_matching_path(ac_env, bound_app):
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.add_policy("alice", scopes=["things"], condition='.request.path == "/api/things/wipe"')
    reg = OperationRegistry()
    meta = _op(reg, method="POST")
    asyncio.run(check(CallerIdentity(user_id="alice"), meta, {}, settings=settings))


def test_synthesize_path_without_route_template_raises():
    reg = OperationRegistry()
    meta = _op(reg)
    meta.route_template = None
    with pytest.raises(ValueError, match="no route template"):
        synthesize_path(meta, {})


def test_route_resolution_failure_denies_fail_closed(ac_env, bound_app):
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.fault = ("SELECT scope_id FROM access_control_routes WHERE url", RuntimeError("redis down"))
    reg = OperationRegistry()
    meta = _op(reg)
    with pytest.raises(PermissionDenied, match="access denied"):
        asyncio.run(check(CallerIdentity(user_id="alice"), meta, {}, settings=settings))


def test_policy_fetch_failure_denies_fail_closed(ac_env, bound_app):
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.fault = ("SELECT scopes, policy_data, condition", RuntimeError("policy read failed"))
    reg = OperationRegistry()
    meta = _op(reg)
    with pytest.raises(PermissionDenied, match="access denied"):
        asyncio.run(check(CallerIdentity(user_id="alice"), meta, {}, settings=settings))


def test_enforcement_error_denies_fail_closed(ac_env, bound_app, monkeypatch):
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.add_policy("alice", scopes=["things"], condition=".request.path")

    async def _boom(**_):
        raise ValueError("render blew up")

    monkeypatch.setattr(bound_app.storage.resource_manager, "render_by_id_or_content", _boom)
    reg = OperationRegistry()
    meta = _op(reg)
    with pytest.raises(PermissionDenied, match="access denied"):
        asyncio.run(check(CallerIdentity(user_id="alice"), meta, {}, settings=settings))


# -- HTTP <-> MCP parity -----------------------------------------------------


class _TokenVerifier:
    """Maps a bearer token to a user id for the HTTP backend."""

    def __init__(self, valid: dict[str, str]) -> None:
        self._valid = valid

    async def verify_token(self, token: str) -> AccessToken | None:
        user = self._valid.get(token)
        if user is None:
            return None
        return AccessToken(token=token, client_id=user, scopes=[], claims={"sub": user})


async def _http_allows(headers: dict[str, str], path: str, settings: AccessControlSettings) -> bool:
    """Run the REAL HTTP edge (auth backend + resource guard) at ``path`` and
    report whether the request reached the app (allowed) or was denied."""
    reached = {"v": False}

    async def sentinel(scope, receive, send):
        reached["v"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    verifier = AccessControlVerifier(settings)
    guard = ResourceGuardMiddleware(sentinel, verifier, settings.public_resource_id)
    backend = AccessControlAuthBackend(
        cast("AccessControlVerifier", _TokenVerifier({"tok-alice": "alice", "tok-bob": "bob"})), settings
    )
    app = AuthenticationMiddleware(guard, backend=backend, on_error=handle_auth_error)

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        return None

    await app(scope, receive, send)
    return reached["v"]


def test_http_mcp_parity_for_same_operation(ac_env, bound_app):
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.add_policy("alice", scopes=["things"])
    ac_env.add_policy("bob", scopes=[])
    reg = OperationRegistry()
    meta = _op(reg, method="POST")

    async def mcp_allows(user: str) -> bool:
        try:
            await check(CallerIdentity(user_id=user), meta, {}, settings=settings)
            return True
        except PermissionDenied:
            return False

    async def run():
        # Allowed caller: allowed on BOTH edges.
        http_alice = await _http_allows({"X-Api-Key": "tok-alice"}, "/api/things/wipe", settings)
        mcp_alice = await mcp_allows("alice")
        # Denied caller: denied on BOTH edges.
        http_bob = await _http_allows({"X-Api-Key": "tok-bob"}, "/api/things/wipe", settings)
        mcp_bob = await mcp_allows("bob")
        return http_alice, mcp_alice, http_bob, mcp_bob

    http_alice, mcp_alice, http_bob, mcp_bob = asyncio.run(run())
    assert http_alice is True
    assert mcp_alice is True
    assert http_bob is False
    assert mcp_bob is False


# -- owned-key OWNER-attenuation parity (HTTP <-> MCP) ------------------------


class _OwnedKeyVerifier:
    """Maps ``tok-<key>`` to an OWNED-key access token whose claims carry the owner,
    so the HTTP backend applies owner attenuation (effective scopes = key ∩ owner)."""

    def __init__(self, key: str, owner: str) -> None:
        self._key = key
        self._owner = owner

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != f"tok-{self._key}":
            return None
        return AccessToken(token=token, client_id=self._key, scopes=[], claims={OWNER_USER_ID_CLAIM: self._owner})


async def _run_owned_key_request(
    entry_path: str,
    verifier: _OwnedKeyVerifier,
    settings: AccessControlSettings,
    inside=None,
) -> dict:
    """Drive an owned-key request through the REAL HTTP edge (auth backend + resource
    guard) to ``entry_path``; on reaching the guarded app, run ``inside()`` WITHIN the
    bound request context (where the guard has bound the caller id + the auth backend's
    effective scopes) and record its result. Returns whether the app was reached and the
    inside-context result."""
    reached: dict = {"v": False, "inside": None}

    async def sentinel(scope, receive, send):
        reached["v"] = True
        if inside is not None:
            reached["inside"] = await inside()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    guard = ResourceGuardMiddleware(sentinel, AccessControlVerifier(settings), settings.public_resource_id)
    backend = AccessControlAuthBackend(cast("AccessControlVerifier", verifier), settings)
    app = AuthenticationMiddleware(guard, backend=backend, on_error=handle_auth_error)

    scope = {
        "type": "http",
        "method": "POST",
        "path": entry_path,
        "query_string": b"",
        "headers": [(b"x-api-key", f"tok-{verifier._key}".encode())],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        return None

    await app(scope, receive, send)
    return reached


def test_owned_key_attenuation_parity_http_mcp(ac_env, bound_app):
    """An owned key whose own scopes ATTENUATE to the owner's must reach the SAME
    allow/deny at BOTH edges: denied for an operation its effective (attenuated) scope
    forbids, allowed for one it permits — the MCP edge consuming the HTTP edge's
    owner-attenuated scopes, never the key's unattenuated policy scopes."""
    settings = AccessControlSettings()
    # An MCP transport entry the owned key can reach (so the guard binds its context),
    # plus a WRITE op the attenuation forbids and a READ op it permits.
    ac_env.add_route("/mcp", "read")
    ac_env.add_route("/api/things/write", "write")
    ac_env.add_route("/api/things/read", "read")
    # The key's OWN scopes are broad; the owner's are narrow → effective = {"read"}.
    ac_env.add_policy("key1", scopes=["read", "write"], policy_data={OWNER_USER_ID_CLAIM: "owner1"})
    ac_env.add_policy("owner1", scopes=["read"])

    reg = OperationRegistry()

    @operation(name="write_thing", summary="Write", tags=["things"], registry=reg)
    async def _write(**_):
        return {}

    @operation(name="read_thing", summary="Read", tags=["things"], registry=reg)
    async def _read(**_):
        return {}

    op_write = reg.get("write_thing")
    op_write.route_template = "/api/things/write"
    op_write.http_method = "POST"
    op_read = reg.get("read_thing")
    op_read.route_template = "/api/things/read"
    op_read.http_method = "POST"

    verifier = _OwnedKeyVerifier("key1", "owner1")

    async def mcp_allows(op_meta) -> bool:
        # Runs INSIDE the bound HTTP request context, reading the identity+scopes the
        # guard bound — exactly how AuthzMiddleware.on_call_tool resolves the caller.
        async def inside() -> bool:
            identity = resolve_caller_identity()
            try:
                await check(identity, op_meta, {}, settings=settings)
                return True
            except PermissionDenied:
                return False

        reached = await _run_owned_key_request("/mcp", verifier, settings, inside=inside)
        assert reached["v"] is True  # the owned key reaches the MCP transport
        return reached["inside"]

    async def http_allows(path: str) -> bool:
        reached = await _run_owned_key_request(path, verifier, settings)
        return reached["v"]

    async def unattenuated_mcp_allows(op_meta) -> bool:
        # The pre-fix behavior: without the carried attenuation the check falls back to
        # the key's OWN (unattenuated) policy scopes. Proves the test is non-vacuous —
        # the write op would be ALLOWED over MCP without the owner-attenuation carry.
        try:
            await check(CallerIdentity(user_id="key1"), op_meta, {}, settings=settings)
            return True
        except PermissionDenied:
            return False

    async def run():
        return (
            await http_allows("/api/things/write"),
            await mcp_allows(op_write),
            await http_allows("/api/things/read"),
            await mcp_allows(op_read),
            await unattenuated_mcp_allows(op_write),
        )

    http_write, mcp_write, http_read, mcp_read, unattenuated_write = asyncio.run(run())

    # The attenuation FORBIDS write: denied on BOTH edges (parity).
    assert http_write is False
    assert mcp_write is False
    # The attenuation PERMITS read: allowed on BOTH edges (parity).
    assert http_read is True
    assert mcp_read is True
    # Non-vacuous: without the owner-attenuation carry the MCP edge would have ALLOWED
    # write (the key's unattenuated scopes include it) — the gap the carry closes.
    assert unattenuated_write is True


# -- owned-key OWNER-CONDITION parity (HTTP <-> MCP) --------------------------


def test_owned_key_owner_condition_parity_http_mcp(ac_env, bound_app):
    """An owned/delegated key whose OWNER's policy carries a path-sensitive jq
    condition (permits the transport path, DENIES an operation path) must reach the
    SAME allow/deny at BOTH edges: the MCP tool edge runs the owner's condition as a
    SECOND enforce pass over the synthesized operation path, exactly as the HTTP edge
    does. The write deny comes purely from the owner CONDITION (both key and owner
    scopes permit the op), isolating the owner-condition gap from scope attenuation."""
    settings = AccessControlSettings()
    # A transport the owned key can reach (so the guard binds its context), plus a
    # WRITE op the owner condition FORBIDS and a READ op it PERMITS.
    ac_env.add_route("/mcp", "read")
    ac_env.add_route("/api/things/write", "write")
    ac_env.add_route("/api/things/read", "read")
    # Key + owner both hold read+write, so effective scopes permit BOTH ops — the
    # write deny is decided ONLY by the owner's path-sensitive jq condition.
    ac_env.add_policy("key1", scopes=["read", "write"], policy_data={OWNER_USER_ID_CLAIM: "owner1"})
    ac_env.add_policy("owner1", scopes=["read", "write"], condition='.request.path != "/api/things/write"')

    reg = OperationRegistry()

    @operation(name="write_thing", summary="Write", tags=["things"], registry=reg)
    async def _write(**_):
        return {}

    @operation(name="read_thing", summary="Read", tags=["things"], registry=reg)
    async def _read(**_):
        return {}

    op_write = reg.get("write_thing")
    op_write.route_template = "/api/things/write"
    op_write.http_method = "POST"
    op_read = reg.get("read_thing")
    op_read.route_template = "/api/things/read"
    op_read.http_method = "POST"

    verifier = _OwnedKeyVerifier("key1", "owner1")

    async def mcp_allows(op_meta) -> bool:
        # Runs INSIDE the bound HTTP request context, reading the identity (claims →
        # owner reference) + scopes the guard bound — exactly how AuthzMiddleware.
        # on_call_tool resolves the caller.
        async def inside() -> bool:
            identity = resolve_caller_identity()
            try:
                await check(identity, op_meta, {}, settings=settings)
                return True
            except PermissionDenied:
                return False

        reached = await _run_owned_key_request("/mcp", verifier, settings, inside=inside)
        assert reached["v"] is True  # the owned key reaches the MCP transport
        return reached["inside"]

    async def http_allows(path: str) -> bool:
        reached = await _run_owned_key_request(path, verifier, settings)
        return reached["v"]

    async def no_claims_mcp_allows(op_meta) -> bool:
        # The pre-fix behavior: effective scopes carried, but NO claims → the owner
        # reference never surfaces, so the owner condition is never enforced. Proves
        # the test is non-vacuous — write would be ALLOWED over MCP (out-permitting HTTP).
        try:
            await check(
                CallerIdentity(user_id="key1", effective_scopes=("read", "write"), claims=None),
                op_meta,
                {},
                settings=settings,
            )
            return True
        except PermissionDenied:
            return False

    async def run():
        return (
            await http_allows("/api/things/write"),
            await mcp_allows(op_write),
            await http_allows("/api/things/read"),
            await mcp_allows(op_read),
            await no_claims_mcp_allows(op_write),
        )

    http_write, mcp_write, http_read, mcp_read, no_claims_write = asyncio.run(run())

    # The owner condition FORBIDS write: denied on BOTH edges (parity).
    assert http_write is False
    assert mcp_write is False
    # The owner condition PERMITS read: allowed on BOTH edges (parity).
    assert http_read is True
    assert mcp_read is True
    # Non-vacuous: without the claims carry (owner reference) the MCP edge would have
    # ALLOWED write — the owner-condition gap the second-pass enforce closes.
    assert no_claims_write is True


def test_owned_key_denied_when_owner_disabled(ac_env, bound_app):
    """The owner second-pass mirrors the HTTP backend fail-closed: a delegated key
    whose owner is DISABLED is denied at the tool edge (before the scope check)."""
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.add_policy("key1", scopes=["things"])
    ac_env.add_policy("owner1", scopes=["things"], policy_data={"disabled": True})
    reg = OperationRegistry()
    meta = _op(reg, method="POST")
    identity = CallerIdentity(user_id="key1", effective_scopes=("things",), claims={OWNER_USER_ID_CLAIM: "owner1"})
    with pytest.raises(PermissionDenied, match="owner is disabled"):
        asyncio.run(check(identity, meta, {}, settings=settings))


def test_owned_key_denied_when_owner_has_no_policy(ac_env, bound_app):
    """The owner second-pass mirrors the HTTP backend fail-closed: a delegated key
    whose owner has NO policy (no scopes, no condition) is denied at the tool edge."""
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.add_policy("key1", scopes=["things"])
    ac_env.add_policy("owner1", scopes=[])
    reg = OperationRegistry()
    meta = _op(reg, method="POST")
    identity = CallerIdentity(user_id="key1", effective_scopes=("things",), claims={OWNER_USER_ID_CLAIM: "owner1"})
    with pytest.raises(PermissionDenied, match="owner has no policy"):
        asyncio.run(check(identity, meta, {}, settings=settings))


# -- identity-claims parity (HTTP <-> MCP) ------------------------------------


class _ClaimsVerifier:
    """Maps ``tok-<user>`` to an access token carrying arbitrary verified claims, so
    a policy condition referencing ``.identity.*`` evaluates over real claims."""

    def __init__(self, user: str, claims: dict) -> None:
        self._user = user
        self._claims = claims

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != f"tok-{self._user}":
            return None
        return AccessToken(token=token, client_id=self._user, scopes=[], claims=self._claims)


async def _http_reaches(verifier, headers: dict[str, str], path: str, settings: AccessControlSettings) -> bool:
    """Drive a request through the REAL HTTP edge (auth backend + resource guard) with
    ``verifier`` and report whether it reached the guarded app (allowed vs denied)."""
    reached = {"v": False}

    async def sentinel(scope, receive, send):
        reached["v"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    guard = ResourceGuardMiddleware(sentinel, AccessControlVerifier(settings), settings.public_resource_id)
    backend = AccessControlAuthBackend(cast("AccessControlVerifier", verifier), settings)
    app = AuthenticationMiddleware(guard, backend=backend, on_error=handle_auth_error)

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        return None

    await app(scope, receive, send)
    return reached["v"]


def test_identity_claims_parity_http_mcp(ac_env, bound_app):
    """A policy condition referencing ``.identity.*`` (a NEGATIVE predicate an empty
    identity would satisfy) must reach the SAME decision at BOTH edges: the MCP tool
    edge builds its jq context with the caller's real token claims, exactly as the HTTP
    edge does — never an empty identity that flips a negative predicate deny→allow."""
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    # A negative predicate: a null (empty-identity) ``.identity.suspended`` would satisfy
    # it (null != true), so an empty identity flips a suspended caller's deny to allow.
    ac_env.add_policy("alice", scopes=["things"], condition=".identity.suspended != true")
    reg = OperationRegistry()
    meta = _op(reg, method="POST")

    suspended = {"sub": "alice", "suspended": True}
    active = {"sub": "alice", "suspended": False}

    async def mcp_allows(claims) -> bool:
        try:
            await check(
                CallerIdentity(user_id="alice", effective_scopes=("things",), claims=claims),
                meta,
                {},
                settings=settings,
            )
            return True
        except PermissionDenied:
            return False

    headers = {"X-Api-Key": "tok-alice"}
    path = "/api/things/wipe"

    async def run():
        return (
            await _http_reaches(_ClaimsVerifier("alice", suspended), headers, path, settings),
            await mcp_allows(suspended),
            await _http_reaches(_ClaimsVerifier("alice", active), headers, path, settings),
            await mcp_allows(active),
            await mcp_allows(None),
        )

    http_susp, mcp_susp, http_active, mcp_active, empty_identity_susp = asyncio.run(run())

    # Suspended: denied on BOTH edges (parity).
    assert http_susp is False
    assert mcp_susp is False
    # Active: allowed on BOTH edges (parity).
    assert http_active is True
    assert mcp_active is True
    # Non-vacuous: an EMPTY identity (the pre-fix seam) flips the negative predicate to
    # allow — the suspended caller would have been ALLOWED over MCP.
    assert empty_identity_susp is True
