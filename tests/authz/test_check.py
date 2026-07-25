"""``authz.check``: the AC-disabled/internal/no-identity gates, path synthesis,
the scope + jq-fence decision, and HTTP↔MCP parity for the same operation."""

from __future__ import annotations

import asyncio
import importlib
import re
from contextlib import contextmanager
from typing import cast

import pytest
from fastmcp.server.auth import AccessToken
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import Response
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_kit.settings import reset_all_settings

from tai42_skeleton.access_control import role_grants as role_grants_module
from tai42_skeleton.access_control.adapter import handle_auth_error
from tai42_skeleton.access_control.backend import AccessControlAuthBackend
from tai42_skeleton.access_control.middleware import ResourceGuardMiddleware
from tai42_skeleton.access_control.path_canon import canonicalize_path
from tai42_skeleton.access_control.role_gate import reset_route_index, resolve_route_meta
from tai42_skeleton.access_control.roles import ROLE_POINTER_KEY, grantable_feature_tags
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.access_control.verifier import AccessControlVerifier, is_always_public_prefix
from tai42_skeleton.app.route_registry import load_all_routes, route_registry
from tai42_skeleton.authz import check, synthesize_path
from tai42_skeleton.authz.identity import INTERNAL_PRINCIPAL, CallerIdentity, resolve_caller_identity
from tai42_skeleton.operations import OperationRegistry, operation
from tai42_skeleton.operations.errors import PermissionDenied
from tests.authz.conftest import FENCED_TEMPLATE_ROUTE, PROBE_ROUTES, SHADOW_ROUTE

# alru caches are held across the several ``asyncio.run`` loops a test opens — a benign
# loop-reset artifact, since a real process serves one loop for its lifetime.
pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")


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


def test_synthesize_path_refuses_a_value_spanning_more_than_its_segment():
    # A plain ``{name}`` names ONE segment; a ``/`` would re-shape the path off the route.
    reg = OperationRegistry()
    meta = _op(reg, route="/api/x/{id}")
    with pytest.raises(PermissionDenied, match="spans more than one path segment"):
        synthesize_path(meta, {"id": "a/b"})


@pytest.mark.parametrize("value", ["", ".", ".."])
def test_synthesize_path_refuses_an_empty_or_dot_segment(value):
    # Empty collapses the segment out of the path; a dot segment re-parents it.
    reg = OperationRegistry()
    meta = _op(reg, route="/api/x/{id}")
    with pytest.raises(PermissionDenied, match="is not a path segment"):
        synthesize_path(meta, {"id": value})


@pytest.mark.parametrize("value", ["../escape", "a/../b", "a/./b", "a//b", ""])
def test_synthesize_path_refuses_a_dot_segment_in_a_path_converter_value(value):
    # A ``:path`` value may span segments, but a dot or empty segment inside it still
    # re-parents or collapses the path.
    reg = OperationRegistry()
    meta = _op(reg, route="/api/resources/{resource_id:path}")
    with pytest.raises(PermissionDenied, match="is not a path segment"):
        synthesize_path(meta, {"resource_id": value})


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


def test_owner_policy_fetch_failure_denies_fail_closed(ac_env, bound_app, monkeypatch):
    # A fault on the owner's policy row must deny, not skip the owner's terms — otherwise a
    # delegated key escapes its owner's fence whenever that read faults.
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.add_policy("key1", scopes=["things"])
    ac_env.add_policy("owner1", scopes=["things"])
    reg = OperationRegistry()
    meta = _op(reg)
    identity = CallerIdentity(user_id="key1", effective_scopes=("things",), claims={OWNER_USER_ID_CLAIM: "owner1"})

    from tai42_skeleton.access_control import policy as policy_module

    real_get_policy_at = policy_module.PolicyEnforcer.get_policy_at

    async def _fault_on_the_owner(self, user_id, version):
        if user_id == "owner1":
            raise RuntimeError("owner policy read failed")
        return await real_get_policy_at(self, user_id, version)

    monkeypatch.setattr(policy_module.PolicyEnforcer, "get_policy_at", _fault_on_the_owner)
    with pytest.raises(PermissionDenied, match="access denied"):
        asyncio.run(check(identity, meta, {}, settings=settings))


def test_level_pass_infra_fault_denies_fail_closed(ac_env, bound_app, monkeypatch):
    # The LEVEL pass is what hard-fences a fenced/secret operation, so an allow-on-fault
    # would open every fenced operation while the grant store was unreachable.
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/wipe", "things")
    ac_env.add_policy("alice", scopes=["things"])
    reg = OperationRegistry()
    meta = _op(reg)

    # By module, not through the package facade: ``authz.check`` there is the FUNCTION.
    check_module = importlib.import_module("tai42_skeleton.authz.check")

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("grant resolution failed")

    monkeypatch.setattr(check_module, "role_level_decision_for_route", _boom)
    with pytest.raises(PermissionDenied, match="access denied"):
        asyncio.run(check(CallerIdentity(user_id="alice"), meta, {}, settings=settings))


# -- a caller-supplied path argument cannot steer the target ------------------


def _fenced_op(reg):
    """An operation at the admin-only TEMPLATED route, whose ``{target}`` segment a caller's
    path argument fills."""

    @operation(name="fence_probe", summary="Fenced probe", tags=["things"], registry=reg)
    async def fence_probe(**_):
        return {}

    meta = reg.get("fence_probe")
    meta.route_template = FENCED_TEMPLATE_ROUTE
    meta.http_method = "POST"
    return meta


def test_a_traversing_path_argument_is_denied_on_a_fenced_operation(ac_env, bound_app, fenced_template_route):
    """``../login/z`` re-parents the synthesized path onto the always-public login surface,
    whose short-circuit precedes the scope test, both jq passes and the fence. The synthesis
    refuses the value, so no layer is asked about a target this dispatch is not for."""
    settings = AccessControlSettings()
    # The caller holds the fenced route's resource but is not admin, so the fence decides.
    ac_env.add_route("/api/things/deploy/fenced", "things")
    ac_env.add_policy("narrow", scopes=["things"])
    reg = OperationRegistry()
    meta = _fenced_op(reg)

    # Non-vacuous: the canonical form of the synthesized path really is on the pre-auth
    # login surface, so without the refusal ``check`` returns before any policy layer.
    steered = canonicalize_path("/api/things/../login/z/fenced")
    assert is_always_public_prefix(steered, settings) is True

    with pytest.raises(PermissionDenied, match="path argument 'target'"):
        asyncio.run(check(CallerIdentity(user_id="narrow"), meta, {"target": "../login/z"}, settings=settings))


def test_a_path_argument_spanning_segments_is_denied_on_a_fenced_operation(ac_env, bound_app, fenced_template_route):
    """A ``/`` makes the synthesized path miss the operation's templated route — a miss that
    reads as "not a gated route" and drops the fence, while the scope test still passes off
    the route TABLE's subtree rows. The synthesis refuses the value instead."""
    settings = AccessControlSettings()
    ac_env.add_route("/api/things", "things")
    ac_env.add_policy("narrow", scopes=["things"])
    reg = OperationRegistry()
    meta = _fenced_op(reg)
    with pytest.raises(PermissionDenied, match="spans more than one path segment"):
        asyncio.run(check(CallerIdentity(user_id="narrow"), meta, {"target": "a/b"}, settings=settings))


@pytest.mark.parametrize("value", ["", ".", ".."])
def test_an_empty_or_dot_path_argument_is_denied(ac_env, bound_app, fenced_template_route, value):
    settings = AccessControlSettings()
    ac_env.add_route("/api/things", "things")
    ac_env.add_policy("narrow", scopes=["things"])
    reg = OperationRegistry()
    meta = _fenced_op(reg)
    with pytest.raises(PermissionDenied, match="is not a path segment"):
        asyncio.run(check(CallerIdentity(user_id="narrow"), meta, {"target": value}, settings=settings))


def test_a_single_segment_path_argument_dispatches_the_operation(ac_env, bound_app, fenced_template_route):
    """Over-refusal guard: the legitimate shape runs for an admin, and the non-admin holding
    the same scope is denied by the LEVEL pass rather than by the synthesis."""
    settings = AccessControlSettings()
    ac_env.add_route("/api/things/deploy/fenced", "things")
    ac_env.add_policy("root", scopes=["*"])
    ac_env.add_policy("narrow", scopes=["things"])
    reg = OperationRegistry()
    meta = _fenced_op(reg)

    asyncio.run(check(CallerIdentity(user_id="root"), meta, {"target": "deploy"}, settings=settings))

    with pytest.raises(PermissionDenied, match="is not permitted"):
        asyncio.run(check(CallerIdentity(user_id="narrow"), meta, {"target": "deploy"}, settings=settings))


def test_an_argument_steering_onto_another_registered_route_is_denied(ac_env, bound_app, fenced_template_route):
    """A single-segment argument can still STEER: the path it synthesizes is registered as
    its own grantable route shadowing the templated one. The decision is pinned to the route
    the operation is REGISTERED at, so it denies rather than reading the fence off the shadow."""
    settings = AccessControlSettings()
    ac_env.add_route(SHADOW_ROUTE, "things")
    ac_env.add_policy("narrow", scopes=["things"])
    reg = OperationRegistry()
    meta = _fenced_op(reg)

    # Non-vacuous: the shadow really is what the path resolves to, and it is grantable.
    resolved = resolve_route_meta(SHADOW_ROUTE, "POST")
    assert resolved is not None
    assert resolved.action == "write"

    with pytest.raises(PermissionDenied, match="does not resolve to the route"):
        asyncio.run(check(CallerIdentity(user_id="narrow"), meta, {"target": "shadow"}, settings=settings))


def test_an_operation_whose_route_is_not_registered_is_denied(ac_env, bound_app):
    """An operation with a route template must resolve back to a registered route or be
    denied: a miss is what a steered path reads as, and treating it as "not a gated route"
    drops the fence on the one input a caller controls."""
    settings = AccessControlSettings()
    unregistered = "/api/things/vanished"
    ac_env.add_route(unregistered, "things")
    ac_env.add_route(PROBE_ROUTES[0], "things")
    ac_env.add_policy("alice", scopes=["things"])
    reg = OperationRegistry()

    with pytest.raises(PermissionDenied, match="does not resolve to the route"):
        asyncio.run(check(CallerIdentity(user_id="alice"), _op(reg, route=unregistered), {}, settings=settings))

    # Non-vacuous: the same caller and policy on a REGISTERED route is allowed.
    asyncio.run(check(CallerIdentity(user_id="alice"), _op(OperationRegistry()), {}, settings=settings))


async def _reload_added_handler(request):
    """The handler of a reload-added route; never served. It exists so the recorded row is
    the one a real adapter registration produces."""
    return Response()


def test_a_route_added_by_an_in_place_reload_is_dispatchable_without_a_restart():
    """A reload drops the route index BEFORE re-importing the routers, so a request served
    in that window can freeze the index against the pre-reimport surface. ``start()`` must
    drop it AGAIN once the reimport completes, or a reload-added route stays undispatchable
    until restart. Only ``start()`` clears the frozen index here, so the assertion fails if
    that post-reimport drop is absent."""
    from tai42_skeleton.app.instance import app
    from tai42_skeleton.authz.check import _own_route
    from tai42_skeleton.manifest import Manifest

    manifest = Manifest.model_validate({})
    added = "/api/things/reload-added"

    async def run() -> None:
        async with app.app_context(manifest):
            reset_all_settings()  # the reload's settings reset drops the index
            # A request in the window rebuilds the index before the route is registered.
            assert resolve_route_meta(added, "POST") is None
            # The reimport completes: the reload-added router records its route.
            route_registry.record(
                path=added,
                methods=["POST"],
                name="reload_added",
                handler=_reload_added_handler,
                summary="A route contributed by a reload-added router",
                tags=["things"],
                authed=True,
                request_model=None,
                response_model=None,
                action="write",
            )
            try:
                # Non-vacuous: the frozen index still answers nothing for the new route.
                assert resolve_route_meta(added, "POST") is None
                # No manual reset: ``start()``'s own drop is what must clear the index.
                app.start(manifest)
                meta = _op(OperationRegistry(), route=added)
                assert _own_route(meta, added, "POST").path == added
            finally:
                route_registry._routes.pop((added, ("POST",)), None)
                reset_route_index()

    asyncio.run(run())


def test_every_registered_route_resolves_back_to_itself_from_its_own_template():
    """The route pin denies a synthesized path that resolves elsewhere, so every served
    route must resolve back to itself once instantiated — else the pin denies a legitimate
    dispatch. Filled with a single segment and a multi-segment (``:path``) value."""
    param = re.compile(r"\{([^}:]+)(?::([^}]+))?\}")
    shadowed = []
    for meta in load_all_routes():
        for method in meta.methods:
            for filler in ("seg", "seg/two"):
                concrete = param.sub(lambda m, fill=filler: fill if m.group(2) == "path" else "seg", meta.path)
                resolved = resolve_route_meta(canonicalize_path(concrete), method)
                if resolved is None or canonicalize_path(resolved.path) != canonicalize_path(meta.path):
                    shadowed.append(f"{method} {meta.path} -> {concrete} -> {resolved.path if resolved else None}")
    assert shadowed == []


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


async def _http_reaches(
    verifier, headers: dict[str, str], path: str, settings: AccessControlSettings, method: str = "POST"
) -> bool:
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
        "method": method,
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


# -- the always-public login surface (HTTP <-> MCP parity) --------------------

_LOGIN_PATH = "/api/login/methods"


async def _login_route_handler(request):
    """Handler of the always-public login-methods route; never served here."""
    return Response()


@contextmanager
def _recorded_login_route():
    """Record the always-public login route for the test, restoring the prior surface after.

    ``check``'s route pin resolves through the global table, so the row must be present on
    its own footing rather than depending on the login router's once-per-process import."""
    key = (_LOGIN_PATH, ("GET",))
    prior = route_registry._routes.get(key)
    route_registry.record(
        path=_LOGIN_PATH,
        methods=["GET"],
        name="login_methods",
        handler=_login_route_handler,
        summary="The always-public login-methods route",
        tags=["login"],
        authed=False,
        request_model=None,
        response_model=None,
    )
    reset_route_index()
    try:
        yield
    finally:
        if prior is None:
            route_registry._routes.pop(key, None)
        else:
            route_registry._routes[key] = prior
        reset_route_index()


def test_always_public_operation_short_circuits_every_policy_layer(ac_env, bound_app, monkeypatch):
    """An always-public route is admitted at the tool edge before any policy read, jq pass
    or LEVEL pass, exactly as the auth backend admits it at step 0.

    A route-table public PIN is the other kind of publicness: there the HTTP edge does run
    all three passes for an authenticated caller, so the tool edge does too."""
    settings = AccessControlSettings()
    reg = OperationRegistry()
    meta = _op(reg, route=_LOGIN_PATH, method="GET")

    # The maximal non-admin grant map: ``write`` on every grantable tag. The login surface
    # is ``authed=False``, so its tag is not grantable and the LEVEL pass still denies.
    async def _max_grants(role_name: str, version: int):
        return dict.fromkeys(grantable_feature_tags(), "write")

    monkeypatch.setattr(role_grants_module, "resolve_role_grants", _max_grants)
    ac_env.add_policy("viewer1", scopes=[], policy_data={ROLE_POINTER_KEY: "editor"})
    identity = CallerIdentity(user_id="viewer1", effective_scopes=())

    async def run():
        # No route row and no scope: the short-circuit precedes both.
        await check(identity, meta, {}, settings=settings)

        http = await _http_reaches(
            _ClaimsVerifier("viewer1", {"sub": "viewer1"}),
            {"X-Api-Key": "tok-viewer1"},
            _LOGIN_PATH,
            settings,
            method="GET",
        )

        # Non-vacuous: with the always-public family emptied the path is an ordinary
        # route-table public pin, and the LEVEL pass denies.
        pinned = AccessControlSettings(always_public_path_prefixes=())
        ac_env.add_route(_LOGIN_PATH, pinned.public_resource_id)
        with pytest.raises(PermissionDenied, match=f"GET {_LOGIN_PATH} is not permitted"):
            await check(identity, meta, {}, settings=pinned)
        return http

    # The same path over HTTP is public for everyone: the two edges agree.
    with _recorded_login_route():
        assert asyncio.run(run()) is True
