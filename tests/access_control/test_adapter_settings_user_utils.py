"""Unit coverage for the small pieces: the ``AuthAdapter`` composition + error
handler, settings pattern compilation + cached accessor, ``TaiUser``, and the
api-key / context-var utilities.
"""

from __future__ import annotations

import pytest
from fastmcp.server.auth import AccessToken
from starlette.authentication import AuthenticationError
from starlette.middleware import Middleware
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from tai_contract.access_control import OWNER_USER_ID_CLAIM
from tai_contract.access_control.context import (
    get_current_user_id,
    reset_request_user_id,
    set_request_user_id,
)
from tai_contract.access_control.identity import AuthIdentity, IdentityProvider
from tai_contract.access_control.models import AccessPolicy
from tai_contract.access_control.registry import register_identity_provider
from tai_identity_redis.redis_api_key_provider import RedisApiKeyProvider
from tai_kit.utils.data.string_util import hash_api_key

from tai_skeleton.access_control.adapter import AuthAdapter, handle_auth_error
from tai_skeleton.access_control.backend import AccessControlAuthBackend, AuthorizationError
from tai_skeleton.access_control.request_scopes import (
    reset_request_identity_claims,
    set_request_identity_claims,
)
from tai_skeleton.access_control.settings import (
    AccessControlSettings,
    access_control_settings,
)
from tai_skeleton.access_control.user import (
    TaiUser,
    is_admin_policy,
    request_identity,
    restricted_identity,
)

# -- AuthAdapter -------------------------------------------------------------


class _StubProvider(IdentityProvider):
    def __init__(self, settings: AccessControlSettings) -> None:
        self.settings = settings

    async def validate_token(self, token: str) -> AuthIdentity | None:
        return AuthIdentity(user_id="stub-user", claims={"src": "stub"})


def test_adapter_builds_middleware_stack_without_resolving_provider():
    # Deferred resolution: constructing the adapter must NOT bind concrete providers
    # (build_app runs before start() populates the registry). The provider list is
    # bound lazily on the first verify_token, so it is None at construction.
    adapter = AuthAdapter(AccessControlSettings())
    assert adapter._internal_verifier._providers is None
    stack = adapter.get_middleware()
    assert len(stack) == 3
    assert all(isinstance(m, Middleware) for m in stack)


def test_adapter_get_identity_providers_resolves_redis_from_registry():
    # The tai-identity-redis plugin registers "redis" in the module-level registry at
    # import (ensured by the suite's default-provider fixture), so the adapter's
    # registry lookup builds the plugin's RedisApiKeyProvider on demand.
    adapter = AuthAdapter(AccessControlSettings())
    providers = adapter._get_identity_providers()
    assert len(providers) == 1
    assert isinstance(providers[0], RedisApiKeyProvider)


def test_adapter_resolves_every_configured_provider_in_order():
    # A two-provider chain resolves both names in configured order.
    register_identity_provider("first-stub", _StubProvider)
    register_identity_provider("second-stub", _StubProvider)
    adapter = AuthAdapter(AccessControlSettings(auth_providers=["first-stub", "second-stub"]))
    providers = adapter._get_identity_providers()
    assert len(providers) == 2
    assert all(isinstance(p, _StubProvider) for p in providers)


def test_adapter_returns_empty_stack_when_disabled():
    adapter = AuthAdapter(AccessControlSettings(enable=False))
    assert adapter.get_middleware() == []


def test_adapter_construction_does_not_crash_for_unregistered_provider():
    # THE TIMING TRAP: a provider not yet in the registry (the manifest plugin has
    # not been imported) must NOT crash at construction — build_app runs before
    # start() registers it. Construction stays quiet; resolution is deferred.
    adapter = AuthAdapter(AccessControlSettings(auth_providers=["not-registered-yet"]))
    assert adapter._internal_verifier._providers is None


async def test_adapter_resolves_provider_lazily_after_registration():
    # THE TIMING TRAP, closed: build the adapter BEFORE the provider is registered
    # (no crash), then register it (as start()'s plugin import would), then the
    # first verify_token resolves through the lazily-selected provider.
    adapter = AuthAdapter(AccessControlSettings(auth_providers=["lazy-stub"]))
    register_identity_provider("lazy-stub", _StubProvider)
    token = await adapter.verify_token("raw")
    assert token is not None
    assert token.client_id == "stub-user"
    assert token.claims == {"src": "stub"}
    # The provider list is now bound (memoized) on the verifier.
    bound_providers = adapter._internal_verifier._providers
    assert bound_providers is not None
    assert isinstance(bound_providers[0], _StubProvider)


async def test_adapter_unknown_provider_raises_loudly_on_first_verify():
    # An unknown provider name is a fail-closed LOUD raise on first use, NOT a boot
    # crash (construction succeeded above) and NOT a silent allow.
    adapter = AuthAdapter(AccessControlSettings(auth_providers=["never-registered"]))
    with pytest.raises(KeyError, match="never-registered"):
        await adapter.verify_token("raw")


async def test_unknown_provider_denies_via_backend_fail_closed():
    # The loud raise from an unknown provider surfaces as a fail-closed DENY through
    # the backend's per-candidate catch: credentials were provided, verification
    # errored, so every candidate is exhausted and the caller is denied — never a
    # silent allow, never a leaked 500.
    settings = AccessControlSettings(auth_providers=["never-registered"])
    adapter = AuthAdapter(settings)
    backend = AccessControlAuthBackend(adapter._internal_verifier, settings)
    conn = HTTPConnection({"type": "http", "headers": [(b"x-api-key", b"some-key")]})
    with pytest.raises(AuthenticationError):
        await backend._get_access_token(conn)


async def test_provider_resolved_once_and_memoized():
    # Resolution runs the factory exactly once; every later verify_token reuses the
    # memoized provider rather than rebuilding it.
    calls = {"n": 0}

    def factory(settings: AccessControlSettings) -> IdentityProvider:
        calls["n"] += 1
        return _StubProvider(settings)

    register_identity_provider("memo-stub", factory)
    adapter = AuthAdapter(AccessControlSettings(auth_providers=["memo-stub"]))
    await adapter.verify_token("a")
    await adapter.verify_token("b")
    assert calls["n"] == 1


async def test_adapter_verify_token_delegates_to_internal_verifier(monkeypatch):
    adapter = AuthAdapter(AccessControlSettings())
    sentinel = AccessToken(token="t", client_id="u", scopes=[], claims={})
    received: dict[str, str] = {}

    async def fake_verify(token):
        received["token"] = token
        return sentinel

    monkeypatch.setattr(adapter._internal_verifier, "verify_token", fake_verify)
    assert await adapter.verify_token("anything") is sentinel
    # The token argument is passed straight through to the internal verifier.
    assert received["token"] == "anything"


def _conn() -> HTTPConnection:
    return HTTPConnection({"type": "http", "headers": []})


def test_handle_auth_error_returns_generic_401_and_hides_detail(caplog):
    with caplog.at_level("ERROR"):
        response = handle_auth_error(_conn(), RuntimeError("redis://secret-host:6379 down"))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 401
    # The internal detail must never reach the client body.
    assert b"secret-host" not in response.body
    assert response.body == b'{"error":"Unauthorized"}'
    # But it is logged server-side for operators.
    assert "secret-host" in caplog.text


def test_handle_auth_error_renders_authorization_error_as_generic_403():
    response = handle_auth_error(_conn(), AuthorizationError("Access Denied"))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 403
    assert response.body == b'{"error":"Forbidden"}'


# -- settings ----------------------------------------------------------------


def test_settings_compile_path_patterns():
    settings = AccessControlSettings(path_patterns={r"^/a/\d+$": "/a/{id}"})
    assert len(settings.compiled_patterns) == 1
    pattern, template = settings.compiled_patterns[0]
    assert pattern.match("/a/5")
    assert template == "/a/{id}"


def test_access_control_settings_accessor_returns_settings():
    assert isinstance(access_control_settings(), AccessControlSettings)


def test_auth_providers_defaults_preserve_single_redis():
    # The default is byte-for-byte today's single-provider behavior.
    assert AccessControlSettings().auth_providers == ["redis"]
    assert AccessControlSettings().always_public_path_prefixes == ("/api/login",)


def test_auth_providers_parses_json_list_from_env(monkeypatch):
    from tai_kit.settings import reset_all_settings

    monkeypatch.setenv("ACCESS_CONTROL_AUTH_PROVIDERS", '["accounts-postgres", "redis"]')
    reset_all_settings()
    try:
        assert AccessControlSettings().auth_providers == ["accounts-postgres", "redis"]
    finally:
        reset_all_settings()


def test_empty_auth_providers_with_enable_raises():
    with pytest.raises(ValueError, match="auth_providers is empty"):
        AccessControlSettings(enable=True, auth_providers=[])


def test_empty_auth_providers_allowed_when_disabled():
    # With the gate off there is no provider to resolve, so an empty list is fine.
    assert AccessControlSettings(enable=False, auth_providers=[]).auth_providers == []


def test_reserved_and_always_public_prefixes_must_be_disjoint():
    with pytest.raises(ValueError, match="overlaps reserved prefix"):
        AccessControlSettings(
            reserved_public_pin_prefixes=("/api/auth",),
            always_public_path_prefixes=("/api/auth/login",),
        )


def test_disjoint_prefixes_are_accepted():
    settings = AccessControlSettings(
        reserved_public_pin_prefixes=("/api/auth",),
        always_public_path_prefixes=("/api/login",),
    )
    assert settings.always_public_path_prefixes == ("/api/login",)


# -- authenticated-always-allowed carve-out ----------------------------------


def test_authenticated_always_allowed_paths_default():
    assert AccessControlSettings().authenticated_always_allowed_paths == ("/api/auth/me",)


def test_authenticated_always_allowed_paths_parse_json_from_env(monkeypatch):
    from tai_kit.settings import reset_all_settings

    monkeypatch.setenv("ACCESS_CONTROL_AUTHENTICATED_ALWAYS_ALLOWED_PATHS", '["/api/auth/me", "/api/auth/session"]')
    reset_all_settings()
    try:
        assert AccessControlSettings().authenticated_always_allowed_paths == (
            "/api/auth/me",
            "/api/auth/session",
        )
    finally:
        reset_all_settings()


def test_authenticated_always_allowed_under_always_public_raises():
    # A path cannot be both public-anonymous and authenticated-only.
    with pytest.raises(ValueError, match="falls under always-public prefix"):
        AccessControlSettings(
            always_public_path_prefixes=("/api/login",),
            authenticated_always_allowed_paths=("/api/login/me",),
        )


def test_authenticated_always_allowed_non_slash_entry_raises():
    with pytest.raises(ValueError, match="must be an absolute path"):
        AccessControlSettings(authenticated_always_allowed_paths=("api/auth/me",))


# -- claim-link settings -----------------------------------------------------


def test_claim_link_settings_defaults():
    settings = AccessControlSettings()
    assert settings.claim_prefix == "ac:claim:"
    assert settings.claim_link_ttl_seconds == 600
    assert settings.claim_link_max_ttl_seconds == 3600


def test_claim_link_ttl_above_ceiling_raises():
    with pytest.raises(ValueError, match="claim_link_ttl_seconds"):
        AccessControlSettings(claim_link_ttl_seconds=7200, claim_link_max_ttl_seconds=3600)


def test_claim_link_ttl_non_positive_raises():
    with pytest.raises(ValueError, match="claim_link_ttl_seconds"):
        AccessControlSettings(claim_link_ttl_seconds=0)


def test_claim_link_ttl_at_ceiling_is_accepted():
    settings = AccessControlSettings(claim_link_ttl_seconds=3600, claim_link_max_ttl_seconds=3600)
    assert settings.claim_link_ttl_seconds == 3600


# -- is_admin_policy discriminator -------------------------------------------


def test_is_admin_policy_true_for_condition_free_wildcard():
    assert is_admin_policy(AccessPolicy(scopes=["*"]), None) is True


def test_is_admin_policy_false_for_conditioned_wildcard():
    assert is_admin_policy(AccessPolicy(scopes=["*"], condition='.request.method == "GET"'), None) is False
    assert is_admin_policy(AccessPolicy(scopes=["*"], condition_id="tmpl"), None) is False


def test_is_admin_policy_false_for_owned_wildcard_key():
    # A condition-free ["*"] key owned by someone is NOT admin (the you-plus escalation).
    assert is_admin_policy(AccessPolicy(scopes=["*"]), "owner-1") is False


def test_is_admin_policy_false_for_scoped_policy():
    assert is_admin_policy(AccessPolicy(scopes=["read"]), None) is False


# -- TaiUser -----------------------------------------------------------------


def test_tai_user_exposes_identity():
    user = TaiUser(AccessToken(token="t", client_id="u42", scopes=[], claims={}))
    assert user.is_authenticated is True
    assert user.display_name == "u42"
    assert user.identity == "u42"


# -- restricted_identity / request_identity seam -----------------------------


def test_restricted_identity_and_request_identity_for_owned_key():
    # An owned key carries OWNER_USER_ID_CLAIM, so it is restricted — but confined to
    # its OWN id, not its owner's (each key is its own island). The owner claim here is
    # deliberately DIFFERENT from the key's own id, so the key-own model is exercised:
    # restricted_identity returns the key's own id, and request_identity is (self, self).
    claims_token = set_request_identity_claims({OWNER_USER_ID_CLAIM: "owner-9"})
    uid_token = set_request_user_id("key-1")
    try:
        assert restricted_identity() == "key-1"
        assert request_identity() == ("key-1", "key-1")
    finally:
        reset_request_user_id(uid_token)
        reset_request_identity_claims(claims_token)


def test_restricted_identity_none_for_unrestricted_key():
    # Claims are bound but carry no owner claim (an ownerless/unrestricted key):
    # restricted_identity is None, and request_identity's isolation half is None while
    # the caller id still resolves.
    claims_token = set_request_identity_claims({"src": "stub"})
    uid_token = set_request_user_id("key-2")
    try:
        assert restricted_identity() is None
        assert request_identity() == ("key-2", None)
    finally:
        reset_request_user_id(uid_token)
        reset_request_identity_claims(claims_token)


def test_restricted_identity_none_for_empty_claims():
    # Empty claims (present but no owner reference) restrict nobody.
    claims_token = set_request_identity_claims({})
    try:
        assert restricted_identity() is None
    finally:
        reset_request_identity_claims(claims_token)


def test_restricted_identity_and_request_identity_when_no_claims_bound():
    # No claims bound at all (gate off / unauthenticated): no identity to restrict,
    # and request_identity is (None, None).
    assert restricted_identity() is None
    assert request_identity() == (None, None)


def test_restricted_identity_raises_when_owner_claim_but_no_own_id():
    # A caller whose claims carry an owner is always authenticated, so its own id is
    # always bound. An owner claim with no bound own id is a broken invariant, raised
    # loudly rather than silently confined to None (which would open the full view).
    claims_token = set_request_identity_claims({OWNER_USER_ID_CLAIM: "owner-9"})
    try:
        with pytest.raises(RuntimeError):
            restricted_identity()
    finally:
        reset_request_identity_claims(claims_token)


# -- utils -------------------------------------------------------------------


def test_hash_api_key_is_deterministic_sha256():
    digest = hash_api_key("sk-abc")
    assert digest == hash_api_key("sk-abc")
    assert len(digest) == 64
    assert digest != hash_api_key("sk-xyz")


def test_user_id_context_set_get_reset():
    assert get_current_user_id() is None
    token = set_request_user_id("u-ctx")
    assert get_current_user_id() == "u-ctx"
    reset_request_user_id(token)
    assert get_current_user_id() is None
