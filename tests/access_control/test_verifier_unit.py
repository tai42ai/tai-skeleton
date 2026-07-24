"""Behavior of ``AccessControlVerifier``.

Covers token->identity translation and the route-resolution ladder (exact,
auto-normalized, explicit patterns, dynamic patterns). The route + dynamic-pattern
reads come from the PG store (the ``FakeAccessControlPg`` seeded per test), while the
policy-version read stays plain-Redis. The store
fetchers fail closed by RAISING: a backend error (or a corrupt stored pattern)
propagates rather than being cached as a degraded empty result, so the downstream
guard denies the request loudly instead of silently.
"""

from __future__ import annotations

import pytest
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.identity import ApiKeyIdentityProvider, AuthIdentity, IdentityProvider

from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.access_control.verifier import AccessControlVerifier

from .conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx


class _Provider(IdentityProvider):
    def __init__(self, identity: AuthIdentity | None) -> None:
        self._identity = identity

    async def validate_token(self, token: str) -> AuthIdentity | None:
        return self._identity


class _SpyProvider(IdentityProvider):
    """Records that it was consulted, and answers a fixed identity/None or raises."""

    def __init__(self, identity: AuthIdentity | None = None, *, raises: Exception | None = None) -> None:
        self._identity = identity
        self._raises = raises
        self.called = 0

    async def validate_token(self, token: str) -> AuthIdentity | None:
        self.called += 1
        if self._raises is not None:
            raise self._raises
        return self._identity


class _MintProvider(ApiKeyIdentityProvider):
    """An api-key (mint-capable) provider whose owner claim is NOT stripped."""

    def __init__(self, identity: AuthIdentity) -> None:
        self._identity = identity

    async def validate_token(self, token: str) -> AuthIdentity | None:
        return self._identity

    async def provision(
        self, user_id: str, description: str, *, owner_user_id: str | None = None
    ) -> str:  # pragma: no cover - unused
        return "sk-x"

    async def revoke(self, user_id: str) -> bool:  # pragma: no cover - unused
        return False

    async def update_description(self, user_id: str, description: str) -> bool:  # pragma: no cover - unused
        return False

    async def list_identities(self) -> list[tuple[str, str]]:  # pragma: no cover - unused
        return []


@pytest.fixture(autouse=True)
def _isolate_registered_reserved_memo():
    """Drop the module-level SPA-shell reserved-set memo around each test: it outlives any
    single verifier instance, so a warmed or faked surface would leak into the next test."""
    verifier_module.reset_registered_reserved_paths()
    try:
        yield
    finally:
        verifier_module.reset_registered_reserved_paths()


def _verifier(settings: AccessControlSettings | None = None, identity=None) -> AccessControlVerifier:
    return AccessControlVerifier(settings or AccessControlSettings(), providers=[_Provider(identity)])


def _wire(monkeypatch, pg: FakeAccessControlPg, redis: FakeRedis | None = None) -> None:
    """Route/pattern reads → the fake PG store; the version read → the fake Redis."""
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis or FakeRedis()))


async def test_verify_token_returns_access_token_for_identity():
    v = _verifier(identity=AuthIdentity(user_id="u1", claims={"email": "a@b"}))
    token = await v.verify_token("raw")
    assert token is not None
    assert token.client_id == "u1"
    assert token.scopes == []
    assert token.claims == {"email": "a@b"}


async def test_verify_token_returns_none_when_no_identity():
    v = _verifier(identity=None)
    assert await v.verify_token("raw") is None


# -- provider chain ----------------------------------------------------------


async def test_chain_first_provider_wins_second_never_called():
    first = _SpyProvider(AuthIdentity(user_id="u1", claims={}))
    second = _SpyProvider(AuthIdentity(user_id="u2", claims={}))
    v = AccessControlVerifier(AccessControlSettings(), providers=[first, second])
    token = await v.verify_token("raw")
    assert token is not None
    assert token.client_id == "u1"
    assert first.called == 1
    assert second.called == 0  # short-circuit on first non-None


async def test_chain_falls_through_on_none():
    first = _SpyProvider(None)
    second = _SpyProvider(AuthIdentity(user_id="u2", claims={}))
    v = AccessControlVerifier(AccessControlSettings(), providers=[first, second])
    token = await v.verify_token("raw")
    assert token is not None
    assert token.client_id == "u2"
    assert first.called == 1
    assert second.called == 1


async def test_chain_all_none_returns_none():
    first = _SpyProvider(None)
    second = _SpyProvider(None)
    v = AccessControlVerifier(AccessControlSettings(), providers=[first, second])
    assert await v.verify_token("raw") is None


async def test_chain_error_propagates_and_never_reaches_next_provider():
    # A provider error propagates even when a later provider would match — an
    # unreachable primary must never silently shift auth onto a weaker provider.
    first = _SpyProvider(raises=RuntimeError("provider store down"))
    second = _SpyProvider(AuthIdentity(user_id="u2", claims={}))
    v = AccessControlVerifier(AccessControlSettings(), providers=[first, second])
    with pytest.raises(RuntimeError, match="provider store down"):
        await v.verify_token("raw")
    assert second.called == 0


async def test_owner_claim_stripped_from_non_mint_provider():
    # An external-issuer (non-ApiKeyIdentityProvider) that returns an owner claim has
    # it stripped centrally, so downstream attenuation never honors it.
    provider = _SpyProvider(AuthIdentity(user_id="u1", claims={OWNER_USER_ID_CLAIM: "victim", "email": "a@b"}))
    v = AccessControlVerifier(AccessControlSettings(), providers=[provider])
    token = await v.verify_token("raw")
    assert token is not None
    assert OWNER_USER_ID_CLAIM not in token.claims
    assert token.claims == {"email": "a@b"}


async def test_owner_claim_kept_for_mint_provider():
    # The mint path legitimately carries the owner claim: it is preserved.
    provider = _MintProvider(AuthIdentity(user_id="k1", claims={OWNER_USER_ID_CLAIM: "owner-1"}))
    v = AccessControlVerifier(AccessControlSettings(), providers=[provider])
    token = await v.verify_token("raw")
    assert token is not None
    assert token.claims[OWNER_USER_ID_CLAIM] == "owner-1"


async def test_always_public_path_short_circuits_without_store_query(monkeypatch):
    # An always-public path returns exactly [public] and NEVER queries the store or
    # the version counter — proven by wiring both to raise on any access.
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.fault = ("SELECT", RuntimeError("store must not be queried"))
    redis = FakeRedis(raise_get=RuntimeError("version must not be read"))
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/api/login/methods") == [settings.public_resource_id]
    assert await v.resolve_resource_ids("/api/login") == [settings.public_resource_id]
    assert pg.executed == []


async def test_resolve_exact_route_match(monkeypatch):
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_route("/orders", "orders")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/orders") == ["orders"]


async def test_resolve_strips_trailing_slash(monkeypatch):
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_route("/orders", "orders")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/orders/") == ["orders"]


async def test_resolve_auto_normalizes_uuid_and_digit(monkeypatch):
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_route("/items/{id}", "items")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/items/42") == ["items"]


async def test_resolve_explicit_compiled_patterns(monkeypatch):
    settings = AccessControlSettings(path_patterns={r"^/files/.*$": "/files/template"})
    pg = FakeAccessControlPg()
    pg.add_route("/files/template", "files")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/files/anything/here") == ["files"]


async def test_explicit_patterns_use_fullmatch_not_prefix(monkeypatch):
    # ``fullmatch`` semantics: a pattern for /api/x/<digits> must not let a
    # longer, more-privileged path inherit the shorter path's resource id.
    settings = AccessControlSettings(path_patterns={r"/api/x/\d+": "/api/x/template"})
    pg = FakeAccessControlPg()
    pg.add_route("/api/x/template", "xroute")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/api/x/123") == ["xroute"]
    assert await v.resolve_resource_ids("/api/x/123/delete") == []


async def test_resolve_dynamic_patterns(monkeypatch):
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    # The dynamic pattern lives on the template url's row: /dyn/N matches the regex
    # and resolves through /dyn/template to its scope.
    pg.add_route("/dyn/template", "dynroute", pattern=r"^/dyn/\d+$")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/dyn/7") == ["dynroute"]


async def test_resolve_pattern_loops_skip_non_match_and_missing_route(monkeypatch):
    # One explicit pattern does not match the path; another matches but its template
    # has no stored route -> both loops add nothing.
    settings = AccessControlSettings(
        path_patterns={r"^/zzz$": "/t1", r"^/files/.*$": "/explicit-no-route"},
    )
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/files/x") == []


async def test_reserved_prefix_path_never_resolves_public_via_route_row(monkeypatch):
    # A reserved management path pinned public (directly on its route row) must NOT
    # resolve public: the verifier drops the marker so the control plane stays
    # authenticated regardless of the route table. An otherwise-unmapped reserved path
    # then resolves to nothing and is denied downstream.
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_route("/api/auth/api-keys", settings.public_resource_id)
    # A path equal to the reserved prefix itself (not just a route beneath it) is also
    # dropped — the exact-match branch of the reserved check, not only the child branch.
    pg.add_route("/api/auth", settings.public_resource_id)
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/api/auth/api-keys") == []
    assert await v.resolve_resource_ids("/api/auth") == []


async def test_reserved_prefix_path_never_resolves_public_via_pattern(monkeypatch):
    # The pattern channel cannot open the control plane either: an unreserved url pinned
    # public with a dynamic pattern that fullmatches a reserved path resolves the marker
    # for that reserved path, but the verifier drops it — the reserved prefix is never
    # public no matter which write channel produced the mapping.
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_route("/decoy", settings.public_resource_id, pattern=r"^/api/auth/.*$")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/api/auth/api-keys") == []
    # A non-reserved path the same pattern would not match is unaffected by the drop.
    assert await v.resolve_resource_ids("/decoy") == [settings.public_resource_id]


async def test_reserved_prefix_drop_leaves_protected_id(monkeypatch):
    # Dropping the marker never removes a protected id: a reserved path that also
    # resolves a real scope stays protected (the drop only strips the public marker).
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_route("/api/auth/api-keys", settings.public_resource_id)
    pg.add_route("/protected-template", "admin", pattern=r"^/api/auth/api-keys$")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/api/auth/api-keys") == ["admin"]


async def test_public_exact_and_protected_pattern_both_resolve(monkeypatch):
    # Cross-tier deny-wins: a path that is BOTH a public exact match AND covered
    # by a protected dynamic pattern must resolve to BOTH ids. A short-circuit on
    # the exact tier would drop the protected id, and the guard would serve the
    # route as public-only with no auth.
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_route("/mixed", "public")
    pg.add_route("/protected-template", "protected", pattern=r"^/mixed$")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert set(await v.resolve_resource_ids("/mixed")) == {"public", "protected"}


async def test_public_auto_normalized_and_protected_explicit_pattern_both_resolve(monkeypatch):
    # Same deny-wins guarantee across the auto-normalized tier and an explicit
    # pattern: an auto-normalized public match must not short-circuit past a
    # protected explicit-pattern match on the same path.
    settings = AccessControlSettings(path_patterns={r"^/items/\d+$": "/protected-template"})
    pg = FakeAccessControlPg()
    pg.add_route("/items/{id}", "public")
    pg.add_route("/protected-template", "protected")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert set(await v.resolve_resource_ids("/items/42")) == {"public", "protected"}


async def test_resolve_unknown_route_returns_empty(monkeypatch):
    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/nope") == []


def test_normalize_auto_substitutions():
    v = _verifier()
    uuid = "/u/123e4567-e89b-12d3-a456-426614174000"
    assert v._normalize_auto(uuid) == "/u/{uuid}"
    assert v._normalize_auto("/n/55") == "/n/{id}"


def test_normalize_auto_uppercase_uuid():
    # UUID matching is case-insensitive: an uppercase UUID segment normalizes
    # to /{uuid} just like a lowercase one.
    v = _verifier()
    assert v._normalize_auto("/u/123E4567-E89B-12D3-A456-426614174000") == "/u/{uuid}"


async def test_dynamic_patterns_empty_when_no_hash(monkeypatch):
    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    assert await v._raw_fetch_dynamic_patterns() == []


async def test_dynamic_patterns_raises_on_uncompilable_regex(monkeypatch):
    """A corrupt stored pattern is malformed config, not an empty result: it is
    surfaced loudly rather than silently dropped from the matched set."""
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_route("/bad", "s1", pattern="(")
    pg.add_route("/ok", "s2", pattern=r"^/ok$")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    with pytest.raises(ValueError, match="malformed dynamic route pattern"):
        await v._raw_fetch_dynamic_patterns()


async def test_dynamic_patterns_raises_on_error_is_fail_closed(monkeypatch):
    """A dynamic-pattern fetch error fails closed by RAISING, so alru never caches
    a degraded empty result and the request is denied loudly downstream."""
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.fault = ("SELECT pattern, url FROM access_control_routes", RuntimeError("pg down"))
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    with pytest.raises(RuntimeError, match="pg down"):
        await v._raw_fetch_dynamic_patterns()


async def test_raw_fetch_route_hit(monkeypatch):
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_route("/x", "routex")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert await v._raw_fetch_route("/x") == "routex"


async def test_raw_fetch_route_raises_on_error_is_fail_closed(monkeypatch):
    """A route-map fetch error fails closed by RAISING, so alru never caches a
    degraded ``None`` and the request is denied loudly rather than silently."""
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.fault = ("SELECT scope_id FROM access_control_routes", RuntimeError("pg down"))
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    with pytest.raises(RuntimeError, match="pg down"):
        await v._raw_fetch_route("/x")


async def test_route_repoint_visible_to_warm_cache_after_version_bump(monkeypatch):
    """A route re-point via management is visible to a second reader the instant
    the policy version is bumped, WITHOUT waiting out the cache ttl — the verifier
    route cache is version-aware, mirroring the policy cache."""
    from tai42_skeleton.access_control import management

    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    redis = FakeRedis(strings={})
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(redis))

    await management.add_url_to_scope("weak", "/admin")
    v = _verifier(settings)
    # Warm the route cache at the current version.
    assert await v.resolve_resource_ids("/admin") == ["weak"]

    # Operator locks the route down to a stronger scope (overwrites the mapping)
    # WITHOUT bumping the version: the warm cache still serves the old scope
    # (proves the cache is actually warm — a bounded fail-open without the fix).
    await management.add_url_to_scope("strong", "/admin")
    assert await v.resolve_resource_ids("/admin") == ["weak"]

    # Every scope route bumps the version → cross-worker cache miss, the re-point
    # is visible immediately without waiting out the ttl.
    await management.bump_policy_version()
    assert await v.resolve_resource_ids("/admin") == ["strong"]


async def test_dynamic_pattern_change_visible_after_version_bump(monkeypatch):
    """The dynamic-pattern cache is version-aware too: a pattern registered after
    the cache warmed is visible once the version is bumped, without the ttl."""
    from tai42_skeleton.access_control import management

    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    redis = FakeRedis(strings={})
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(redis))

    v = _verifier(settings)
    # Warm the dynamic-pattern cache while empty.
    assert await v.resolve_resource_ids("/dyn/7") == []

    # Register a dynamic pattern + its route WITHOUT bumping: the warm empty cache
    # still resolves nothing.
    await management.add_url_to_scope("dynroute", "/dyn/template", pattern=r"^/dyn/\d+$")
    assert await v.resolve_resource_ids("/dyn/7") == []

    # Bump the version → both the route and the dynamic-pattern caches miss and
    # re-read, so the new pattern-scoped route resolves.
    await management.bump_policy_version()
    assert await v.resolve_resource_ids("/dyn/7") == ["dynroute"]


# -- SPA-shell public fallback + H1 canonicalization -------------------------
#
# The last resolution tier: a GET to an UNMAPPED, non-/api, non-/mcp canonical path
# that is NOT a registered route resolves public so a deep-link refresh reaches the
# dataless SPA shell. Every classification runs on ONE canonical path form.


async def test_spa_fallback_unmapped_get_is_public(monkeypatch):
    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/agents", method="GET") == [settings.public_resource_id]
    # Any inner Studio route, too.
    assert await v.resolve_resource_ids("/agents/123/detail", method="GET") == [settings.public_resource_id]


async def test_spa_fallback_only_fires_for_get(monkeypatch):
    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    # A non-GET method never opens the shell (it would be a mutation on an unmapped path).
    assert await v.resolve_resource_ids("/agents", method="POST") == []
    # An unspecified method is fail-closed: the fallback never fires.
    assert await v.resolve_resource_ids("/agents", method=None) == []
    assert await v.resolve_resource_ids("/agents") == []


async def test_spa_fallback_skips_registered_operational_routes(monkeypatch):
    # /health, /ready, /metrics are registered non-/api GET routes → in the DERIVED
    # reserved set → the fallback skips them; they never gain the public id from this tier.
    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    for path in ("/health", "/ready", "/metrics"):
        assert await v.resolve_resource_ids(path, method="GET") == []


async def test_spa_fallback_skips_openapi_via_supplement(monkeypatch):
    # /openapi.json is not a live registered route; the supplement keeps the fallback
    # from serving the shell for the conventional OpenAPI document path.
    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/openapi.json", method="GET") == []


async def test_spa_fallback_never_opens_api(monkeypatch):
    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/api/anything", method="GET") == []


async def test_spa_fallback_deny_wins_over_explicit_protected_mapping(monkeypatch):
    # An operator who explicitly mapped /agents to a protected scope keeps it protected:
    # the fallback fires ONLY when the route table resolved nothing.
    settings = AccessControlSettings()
    pg = FakeAccessControlPg()
    pg.add_route("/agents", "agents-scope")
    _wire(monkeypatch, pg)
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/agents", method="GET") == ["agents-scope"]


async def test_spa_fallback_reserved_prefix_stays_gated(monkeypatch):
    # A reserved-prefix path (also under /api) never opens via the fallback.
    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/api/auth/whatever", method="GET") == []


async def test_spa_fallback_master_switch_off(monkeypatch):
    settings = AccessControlSettings(spa_shell_public=False)
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    assert await v.resolve_resource_ids("/agents", method="GET") == []


async def test_spa_reserved_set_is_derived_not_static(monkeypatch):
    # A NEWLY registered non-/api GET route is automatically in the derived reserved set
    # (the fallback skips it) with NO settings change — pins the no-static-list property.
    from types import SimpleNamespace

    fake_routes = [
        SimpleNamespace(path="/newpage", methods=("GET",)),
        SimpleNamespace(path="/api/agents", methods=("GET",)),  # /api excluded
        SimpleNamespace(path="/webhook/{topic}", methods=("GET",)),  # templated excluded
    ]
    monkeypatch.setattr(verifier_module, "load_all_routes", lambda: fake_routes)
    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    # The freshly declared route is reserved (gated) with no static list edit.
    assert await v.resolve_resource_ids("/newpage", method="GET") == []
    # A genuinely unmapped, unregistered path still reaches the shell.
    assert await v.resolve_resource_ids("/otherpage", method="GET") == [settings.public_resource_id]


async def test_reload_added_route_not_served_anonymously_after_reset(monkeypatch):
    # The HTTP-edge verifier is built once and baked into the ASGI stack, so the same
    # instance answers across a reload: its reserved set must not stay frozen on the
    # pre-reload surface, or a reload-added route is served the anonymous SPA shell.
    from types import SimpleNamespace

    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)  # one persistent verifier across the reload

    # Pre-reload surface: /status is NOT a registered route, so a GET reaches the shell.
    monkeypatch.setattr(verifier_module, "load_all_routes", list)
    assert await v.resolve_resource_ids("/status", method="GET") == [settings.public_resource_id]

    # The reload re-imports a router that registers an authed GET /status.
    monkeypatch.setattr(verifier_module, "load_all_routes", lambda: [SimpleNamespace(path="/status", methods=("GET",))])

    # Until the reset drops it the memo is stale, so /status still gets the anonymous shell.
    assert await v.resolve_resource_ids("/status", method="GET") == [settings.public_resource_id]

    # After the reload's drop the same verifier derives /status into the reserved set.
    verifier_module.reset_registered_reserved_paths()
    assert await v.resolve_resource_ids("/status", method="GET") == []


# -- H1 bypass corpus (MUST): one canonical form, segment-aware prefixes ------


async def test_h1_bypass_corpus(monkeypatch):
    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    public = [settings.public_resource_id]

    # A residual percent-escape (the ASGI decode already ran) is a double-encoded byte:
    # canonicalize REJECTS it rather than re-decoding, so it is denied (empty), never public.
    assert await v.resolve_resource_ids("/%61pi/x", method="GET") == []
    # Duplicate leading slash collapses to /api/x — gated.
    assert await v.resolve_resource_ids("//api/x", method="GET") == []
    # Dot-resolution first: /api/../agents is genuinely /agents — shell-public.
    assert await v.resolve_resource_ids("/api/../agents", method="GET") == public
    # …and /agents/../api/secret is genuinely /api/secret — gated.
    assert await v.resolve_resource_ids("/agents/../api/secret", method="GET") == []
    # A residual encoded slash (%2F) is NOT decoded again — it is REJECTED as a
    # double-encoded byte and denied, rather than canonicalized into a /api/ segment.
    assert await v.resolve_resource_ids("/api%2Fx", method="GET") == []
    # Case-sensitive: /API is not the lowercase /api mount, so it never reaches API
    # DATA; shell-or-deny is acceptable (here: the shell).
    assert await v.resolve_resource_ids("/API/x", method="GET") == public


async def test_h1_malformed_paths_denied(monkeypatch):
    # A NUL / control / backslash path is malformed → fail-closed deny, never the shell.
    settings = AccessControlSettings()
    _wire(monkeypatch, FakeAccessControlPg())
    v = _verifier(settings)
    for bad in ("/agents\x00", "/agents\\admin", "/agents\x1f"):
        assert await v.resolve_resource_ids(bad, method="GET") == []
