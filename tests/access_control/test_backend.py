"""Behavior of ``AccessControlAuthBackend``: credential extraction order, the
unauthenticated / invalid / valid token outcomes, and the full authenticate flow
that fetches policy + context, renders the condition, and enforces it.
"""

from __future__ import annotations

import pytest
from fastmcp.server.auth import AccessToken
from starlette.authentication import (
    AuthenticationError,
    UnauthenticatedUser,
)
from starlette.requests import Request
from tai42_contract.access_control import OWNER_USER_ID_CLAIM

from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control.backend import (
    AccessControlAuthBackend,
    AuthorizationError,
    effective_scopes,
)
from tai42_skeleton.access_control.path_canon import canonicalize_path
from tai42_skeleton.access_control.roles import EDITOR_JQ
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.access_control.user import TaiUser
from tai42_skeleton.access_control.verifier import is_always_public_prefix

from .conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx


@pytest.fixture(autouse=True)
def store_pg(monkeypatch) -> FakeAccessControlPg:
    """The enforcer reads the POLICY body from the PG store; default it to an empty
    fake so no test opens a real Postgres. A test that needs a policy row seeds this
    same instance (request it by name)."""
    fake = FakeAccessControlPg()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(fake))
    return fake


class _FakeVerifier:
    """Verifies a fixed set of valid tokens; optionally raises for one token."""

    def __init__(self, valid: dict[str, str], raise_for: set[str] | None = None) -> None:
        self._valid = valid
        self._raise_for = raise_for or set()

    async def verify_token(self, token: str) -> AccessToken | None:
        if token in self._raise_for:
            raise RuntimeError("verifier blew up")
        user_id = self._valid.get(token)
        if not user_id:
            return None
        return AccessToken(token=token, client_id=user_id, scopes=[], claims={"sub": user_id})


class _OwnedKeyVerifier:
    """Returns a fixed identity carrying an owner claim (an owned key)."""

    def __init__(self, user_id: str, owner: str) -> None:
        self._user_id = user_id
        self._owner = owner

    async def verify_token(self, token: str) -> AccessToken | None:
        return AccessToken(
            token=token,
            client_id=self._user_id,
            scopes=[],
            claims={"sub": self._user_id, OWNER_USER_ID_CLAIM: self._owner},
        )


class _SpyVerifier:
    """Records whether verify_token was called (for the public short-circuit pin)."""

    def __init__(self) -> None:
        self.called = 0

    async def verify_token(self, token: str) -> AccessToken | None:
        self.called += 1
        return AccessToken(token=token, client_id="u", scopes=[], claims={})


def _conn(headers: dict[str, str] | None = None, path="/x", method="GET") -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "query_string": b"",
            "headers": raw,
        }
    )


def _backend(verifier, settings=None) -> AccessControlAuthBackend:
    return AccessControlAuthBackend(verifier, settings or AccessControlSettings())


async def test_no_credentials_is_unauthenticated():
    backend = _backend(_FakeVerifier({}))
    creds, user = await backend.authenticate(_conn())
    assert isinstance(user, UnauthenticatedUser)
    assert "unauthenticated" in creds.scopes


async def test_invalid_credentials_raise():
    backend = _backend(_FakeVerifier({}))
    with pytest.raises(AuthenticationError, match="Invalid API key"):
        await backend.authenticate(_conn({"Authorization": "Bearer nope"}))


async def test_bearer_token_is_split_from_scheme(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("u1", scopes=["res-a"])
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_FakeVerifier({"good": "u1"}), settings)
    _creds, user = await backend.authenticate(_conn({"Authorization": "Bearer good"}))
    assert isinstance(user, TaiUser)
    assert user.token.client_id == "u1"


async def test_authorization_without_space_is_used_whole(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("u2", scopes=["res-a"])
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_FakeVerifier({"rawtoken": "u2"}), settings)
    _creds, user = await backend.authenticate(_conn({"Authorization": "rawtoken"}))
    assert isinstance(user, TaiUser)
    assert user.token.client_id == "u2"


async def test_basic_scheme_is_never_a_bearer_candidate():
    # The verifier would accept "zzz" — but a Basic credential must never be
    # tried as a bearer token, so no candidate exists and the caller stays
    # unauthenticated.
    backend = _backend(_FakeVerifier({"zzz": "u9"}))
    creds, user = await backend.authenticate(_conn({"Authorization": "Basic zzz"}))
    assert isinstance(user, UnauthenticatedUser)
    assert "unauthenticated" in creds.scopes


async def test_bearer_token_with_extra_space_is_stripped(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("u5", scopes=["res-a"])
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_FakeVerifier({"tok": "u5"}), settings)
    # A double space after the scheme leaves a leading space on the token; the
    # stripped value is what gets verified.
    _creds, user = await backend.authenticate(_conn({"Authorization": "Bearer  tok"}))
    assert isinstance(user, TaiUser)
    assert user.token.client_id == "u5"


async def test_x_api_key_header_is_used(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("u3", scopes=["res-a"])
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_FakeVerifier({"key123": "u3"}), settings)
    _creds, user = await backend.authenticate(_conn({"X-Api-Key": "key123"}))
    assert isinstance(user, TaiUser)
    assert user.token.client_id == "u3"


async def test_authorization_header_wins_over_x_api_key(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("auth-user", scopes=["res-a"])
    store_pg.add_policy("api-user", scopes=["res-a"])
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    # Both credentials are independently valid but map to different users.
    backend = _backend(_FakeVerifier({"auth-tok": "auth-user", "api-tok": "api-user"}), settings)
    _creds, user = await backend.authenticate(_conn({"Authorization": "Bearer auth-tok", "X-Api-Key": "api-tok"}))
    assert isinstance(user, TaiUser)
    # Authorization is candidate #1, so its identity wins over the also-valid
    # X-Api-Key — the priority order, not just presence, is what's asserted.
    assert user.token.client_id == "auth-user"


async def test_verifier_error_falls_through_to_next_candidate(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("u4", scopes=["res-a"])
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    verifier = _FakeVerifier({"good-key": "u4"}, raise_for={"boom-bearer"})
    backend = _backend(verifier, settings)
    # Authorization raises in the verifier; X-Api-Key is the valid fallback.
    _creds, user = await backend.authenticate(_conn({"Authorization": "Bearer boom-bearer", "X-Api-Key": "good-key"}))
    assert isinstance(user, TaiUser)
    assert user.token.client_id == "u4"


async def test_authenticate_attaches_policy_scopes_when_condition_passes(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("u1", scopes=["res-a", "res-b"], condition='.sub == "u1"')
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_FakeVerifier({"good": "u1"}), settings)
    creds, user = await backend.authenticate(_conn({"Authorization": "Bearer good"}))
    assert isinstance(user, TaiUser)
    assert creds.scopes == ["res-a", "res-b"]
    assert user.token.scopes == ["res-a", "res-b"]
    # The condition was rendered through the bound template manager.
    assert bound_app.storage.resource_manager.calls


async def test_credential_for_a_principal_with_no_policy_is_denied(monkeypatch, bound_app):
    # A revoke deletes the policy row before the identity record, so a fault in between
    # leaves a credential that still verifies against a principal with no policy.
    settings = AccessControlSettings()
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_FakeVerifier({"ghost": "revoked-user"}), settings)
    with pytest.raises(AuthorizationError):
        await backend.authenticate(_conn({"Authorization": "Bearer ghost"}))


async def test_authenticate_fails_closed_when_live_context_unavailable(monkeypatch, caplog):
    """A live-context outage must deny, not authenticate against empty data. The
    fetch error is wrapped into a clean ``AuthorizationError`` (a fail-closed deny
    the AuthenticationMiddleware renders as 403) rather than leaking out as a raw
    500 or being masked as an empty, possibly-permissive context.

    The client-facing message stays generic — the underlying "redis down" detail
    must never reach the caller — while the full detail is logged server-side.
    """
    settings = AccessControlSettings()
    fake = FakeRedis(raise_hgetall=RuntimeError("redis down"))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(fake))
    backend = _backend(_FakeVerifier({"good": "u1"}), settings)
    with caplog.at_level("ERROR"), pytest.raises(AuthorizationError) as excinfo:
        await backend.authenticate(_conn({"Authorization": "Bearer good"}))
    # Client-facing message is generic and discloses no internal detail.
    assert str(excinfo.value) == "Access Denied"
    assert "redis down" not in str(excinfo.value)
    # The underlying detail is retained server-side for operators.
    assert "redis down" in caplog.text


async def test_authenticate_allows_when_no_condition_configured(monkeypatch, bound_app, store_pg):
    # A policy with scopes but NO condition (neither condition nor condition_id)
    # is enforced as a no-op allow: the caller is authenticated with its scopes.
    settings = AccessControlSettings()
    store_pg.add_policy("u1", scopes=["res-a", "res-b"])
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_FakeVerifier({"good": "u1"}), settings)
    creds, user = await backend.authenticate(_conn({"Authorization": "Bearer good"}))
    assert isinstance(user, TaiUser)
    assert creds.scopes == ["res-a", "res-b"]


async def test_authenticate_denies_when_configured_condition_renders_empty(monkeypatch, bound_app, caplog, store_pg):
    # A condition WAS configured (via condition_id) but renders to empty. This must
    # fail closed as a deny, never be treated as "no condition" (which would let
    # the caller through against a condition that never passed).
    settings = AccessControlSettings()
    store_pg.add_policy("u1", scopes=["res-a"], condition_id="renders-to-nothing")
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_FakeVerifier({"good": "u1"}), settings)
    with caplog.at_level("ERROR"), pytest.raises(AuthorizationError) as excinfo:
        await backend.authenticate(_conn({"Authorization": "Bearer good"}))
    # The caller only ever sees the generic message.
    assert str(excinfo.value) == "Access Denied"
    # The enforcement failure is logged server-side.
    assert "policy enforcement failed" in caplog.text


async def test_authenticate_denies_when_condition_fails(monkeypatch, bound_app, caplog, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("u1", scopes=["res-a"], condition='.sub == "someone-else"')
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_FakeVerifier({"good": "u1"}), settings)
    with caplog.at_level("ERROR"), pytest.raises(AuthorizationError) as excinfo:
        await backend.authenticate(_conn({"Authorization": "Bearer good"}))
    # A policy denial surfaces only the generic message to the caller.
    assert str(excinfo.value) == "Access Denied"
    # The enforcement failure is logged server-side.
    assert "policy enforcement failed" in caplog.text


# -- effective-scopes helper (owned-key attenuation) -------------------------


def test_effective_scopes_intersection():
    assert effective_scopes(["a", "b", "c"], ["b", "c", "d"]) == ["b", "c"]


def test_effective_scopes_disjoint_is_empty():
    assert effective_scopes(["a"], ["b"]) == []


def test_effective_scopes_star_owner_caps_nothing():
    assert effective_scopes(["a", "b"], ["*"]) == ["a", "b"]


def test_effective_scopes_star_key_collapses_to_owner():
    assert effective_scopes(["*"], ["a", "b"]) == ["a", "b"]


# -- owned-key attenuation (full authenticate flow) --------------------------


async def test_owned_key_scopes_intersect_owner(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("key1", scopes=["a", "b"])
    store_pg.add_policy("owner1", scopes=["b", "c"])
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_OwnedKeyVerifier("key1", "owner1"), settings)
    creds, user = await backend.authenticate(_conn({"X-Api-Key": "k"}))
    assert isinstance(user, TaiUser)
    assert creds.scopes == ["b"]  # ["a","b"] ∩ ["b","c"]


async def test_owned_key_star_collapses_to_owner_scopes(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("key1", scopes=["*"])
    store_pg.add_policy("owner1", scopes=["read"])
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_OwnedKeyVerifier("key1", "owner1"), settings)
    creds, _user = await backend.authenticate(_conn({"X-Api-Key": "k"}))
    assert creds.scopes == ["read"]


async def test_owned_key_denied_when_owner_disabled(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("key1", scopes=["a"])
    store_pg.add_policy("owner1", scopes=["a"], policy_data={"disabled": True})
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_OwnedKeyVerifier("key1", "owner1"), settings)
    with pytest.raises(AuthorizationError):
        await backend.authenticate(_conn({"X-Api-Key": "k"}))


async def test_owned_key_denied_when_owner_policy_empty(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("key1", scopes=["a"])
    # No owner policy row → empty AccessPolicy() default → deny.
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_OwnedKeyVerifier("key1", "owner1"), settings)
    with pytest.raises(AuthorizationError):
        await backend.authenticate(_conn({"X-Api-Key": "k"}))


async def test_direct_disabled_principal_denied(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("u1", scopes=["a"], policy_data={"disabled": True})
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_FakeVerifier({"good": "u1"}), settings)
    with pytest.raises(AuthorizationError):
        await backend.authenticate(_conn({"Authorization": "Bearer good"}))


async def test_owned_key_owner_condition_denies_while_key_passes(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("key1", scopes=["a"])  # no key condition
    store_pg.add_policy("owner1", scopes=["a"], condition='.request.path == "/never"')
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_OwnedKeyVerifier("key1", "owner1"), settings)
    with pytest.raises(AuthorizationError):
        await backend.authenticate(_conn({"X-Api-Key": "k"}, path="/x"))


async def test_owned_key_both_conditions_pass_allows(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("key1", scopes=["a"], condition='.request.path == "/x"')
    store_pg.add_policy("owner1", scopes=["a"], condition='.request.path == "/x"')
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_OwnedKeyVerifier("key1", "owner1"), settings)
    creds, user = await backend.authenticate(_conn({"X-Api-Key": "k"}, path="/x"))
    assert isinstance(user, TaiUser)
    assert creds.scopes == ["a"]


async def test_owned_key_owner_editor_role_reaches_me(monkeypatch, bound_app, store_pg):
    # An owned key whose OWNER carries the seeded EDITOR_JQ passes GET /api/auth/me: the
    # owner-condition second pass admits the capability-projection route via its carve-in,
    # so a scoped delegated key can still introspect its own capabilities.
    settings = AccessControlSettings()
    store_pg.add_policy("key1", scopes=["a"])  # no key condition
    store_pg.add_policy("owner1", scopes=["*"], condition=EDITOR_JQ)
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_OwnedKeyVerifier("key1", "owner1"), settings)
    _creds, user = await backend.authenticate(_conn({"X-Api-Key": "k"}, path="/api/auth/me"))
    assert isinstance(user, TaiUser)


async def test_owned_key_owner_editor_role_denied_on_admin_area(monkeypatch, bound_app, store_pg):
    # The mirror: the same owner EDITOR_JQ still fences the access-control admin area, so
    # the owned key is denied a non-carved /api/auth route (the owner second pass denies).
    settings = AccessControlSettings()
    store_pg.add_policy("key1", scopes=["a"])
    store_pg.add_policy("owner1", scopes=["*"], condition=EDITOR_JQ)
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_OwnedKeyVerifier("key1", "owner1"), settings)
    with pytest.raises(AuthorizationError):
        await backend.authenticate(_conn({"X-Api-Key": "k"}, path="/api/auth/public-routes"))


async def test_owner_condition_sees_owner_scopes_not_attenuated(monkeypatch, bound_app, store_pg):
    # The owner condition references ``.scopes`` and gates on the OWNER's scope count (5).
    # The key's scopes attenuate to key∩owner (3), so the condition passes ONLY if the
    # owner-condition pass is judged against the OWNER's scopes, not the attenuated set.
    settings = AccessControlSettings()
    store_pg.add_policy("key1", scopes=["a", "b", "c"])  # no key condition
    store_pg.add_policy("owner1", scopes=["a", "b", "c", "d", "e"], condition="(.scopes | length) == 5")
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_OwnedKeyVerifier("key1", "owner1"), settings)
    creds, user = await backend.authenticate(_conn({"X-Api-Key": "k"}))
    assert isinstance(user, TaiUser)
    # The finalized scopes stay the attenuated key∩owner set.
    assert creds.scopes == ["a", "b", "c"]


async def test_owner_condition_denies_when_gated_on_attenuated_length(monkeypatch, bound_app, store_pg):
    # The mirror image: the owner condition gates on the ATTENUATED length (3). Because the
    # owner-condition pass now sees the OWNER's 5 scopes, it evaluates false and DENIES —
    # proving the pass is not judged against the attenuated set (which would have passed).
    settings = AccessControlSettings()
    store_pg.add_policy("key1", scopes=["a", "b", "c"])
    store_pg.add_policy("owner1", scopes=["a", "b", "c", "d", "e"], condition="(.scopes | length) == 3")
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_OwnedKeyVerifier("key1", "owner1"), settings)
    with pytest.raises(AuthorizationError):
        await backend.authenticate(_conn({"X-Api-Key": "k"}))


async def test_public_path_short_circuits_without_verifying(monkeypatch):
    settings = AccessControlSettings()
    spy = _SpyVerifier()
    backend = _backend(spy, settings)
    # A garbage credential on the always-public login surface yields the
    # unauthenticated result with the verifier NEVER called.
    creds, user = await backend.authenticate(_conn({"Authorization": "Bearer garbage"}, path="/api/login/methods"))
    assert isinstance(user, UnauthenticatedUser)
    assert "unauthenticated" in creds.scopes
    assert spy.called == 0


@pytest.mark.parametrize("path", ["/api//login/methods", "/api/login/./methods", "/api/login/"])
async def test_step0_decides_the_login_surface_on_the_canonical_form(path):
    """Step 0 and the resource guard ask ONE predicate over ONE canonical form, so a
    non-canonical login path cannot be public to the guard yet credential-verified here."""
    settings = AccessControlSettings()
    spy = _SpyVerifier()
    backend = _backend(spy, settings)

    # Non-vacuous: the guard's own resolution really does call these paths public.
    assert is_always_public_prefix(canonicalize_path(path), settings) is True

    creds, user = await backend.authenticate(_conn({"Authorization": "Bearer garbage"}, path=path))
    assert isinstance(user, UnauthenticatedUser)
    assert "unauthenticated" in creds.scopes
    assert spy.called == 0


@pytest.mark.parametrize("path", ["/api/login/../auth/api-keys", "/apiary/login", "/api/login\\x"])
async def test_step0_never_admits_a_path_that_only_looks_like_the_login_surface(path):
    """A traversal out of the login prefix, a neighbour the segment-aware match must not
    swallow, and a path with no canonical form: none may skip credential verification."""
    settings = AccessControlSettings()
    backend = _backend(_SpyVerifier(), settings)
    assert backend._is_always_public_path(path) is False


async def test_same_credential_on_protected_path_still_verifies(monkeypatch, bound_app, store_pg):
    settings = AccessControlSettings()
    store_pg.add_policy("u1", scopes=["a"])
    spy = _SpyVerifier()

    async def _verify(token):
        spy.called += 1
        return AccessToken(token=token, client_id="u1", scopes=[], claims={})

    spy.verify_token = _verify  # type: ignore[method-assign]
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(spy, settings)
    await backend.authenticate(_conn({"Authorization": "Bearer good"}, path="/x"))
    assert spy.called == 1


# -- step 5b: the per-tag level pass denies through the full authenticate flow -


async def test_step5b_level_miss_and_hard_fence_deny_with_cause(monkeypatch, bound_app, store_pg):
    # Drive authenticate() end-to-end for a non-admin: a grantable route the governing role
    # holds no level for is a LEVEL_MISS 403, and an admin-only fenced route is a HARD_FENCE
    # 403 — confirming step 5b actually invokes role_level_decision and sets the internal
    # DenialCause (the external response is a generic AuthorizationError either way).
    import tai42_skeleton.versioning as versioning_module
    from tai42_skeleton.access_control import role_grants as role_grants_module
    from tai42_skeleton.access_control.role_gate import DenialCause, reset_route_index
    from tai42_skeleton.access_control.roles import seed_default_roles

    from .test_policy_store import _MemStore

    # The offline route harness rebinds a minimal app when the bound app exposes no
    # ``http`` seam; give the fake app one so importing routes (via resolve_route_meta)
    # keeps the storage-bearing fake bound for the enforce step.
    bound_app.http = object()

    mem = _MemStore()
    monkeypatch.setattr(versioning_module, "versioned_store", lambda: mem)
    monkeypatch.setattr(versioning_module, "versioned_store_configured", lambda: True)
    role_grants_module.reset_role_grants_cache()
    reset_route_index()
    await seed_default_roles()
    # A non-admin role: the editor base jq admits non-/api/auth paths (so step 5's condition
    # passes and we REACH step 5b), but its grant map is empty, so a grantable route misses.
    await mem.create(
        "role",
        "narrow",
        {
            "name": "narrow",
            "description": "",
            "scopes": ["*"],
            "base_tier": "editor",
            "grants": {},
            "condition": EDITOR_JQ,
            "allow_all": False,
        },
    )

    settings = AccessControlSettings()
    store_pg.add_policy("ed", scopes=["*"], condition=EDITOR_JQ, policy_data={"role": "narrow"})
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(FakeRedis()))
    backend = _backend(_FakeVerifier({"tok": "ed"}), settings)

    # A grantable read route the narrow role holds no tag for → LEVEL_MISS.
    with pytest.raises(AuthorizationError) as level:
        await backend.authenticate(_conn({"Authorization": "Bearer tok"}, path="/api/tools"))
    assert str(level.value) == "Access Denied"
    assert level.value.cause is DenialCause.LEVEL_MISS

    # An admin-only fenced route → HARD_FENCE regardless of any level. The editor base jq
    # admits this non-/api/auth path, so enforcement reaches step 5b and the fence denies.
    with pytest.raises(AuthorizationError) as fence:
        await backend.authenticate(_conn({"Authorization": "Bearer tok"}, path="/api/run-tool", method="POST"))
    assert fence.value.cause is DenialCause.HARD_FENCE
