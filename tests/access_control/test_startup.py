"""The access-control startup checks (``access_control.startup``).

``probe_identity_provider`` resolves EVERY configured provider through the registry
and awaits each ``healthcheck()`` (any failure fails the boot loudly);
``check_accounts_providers_configured`` refuses to boot when a registered accounts
provider is left out of the resolution chain; ``check_always_public_routes`` enumerates
the always-public login surface and refuses an authed mount beneath it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tai42_contract.access_control import registry
from tai42_contract.access_control.identity import AuthIdentity, IdentityProvider
from tai42_contract.accounts import registry as accounts_registry
from tai42_contract.accounts.models import LoginMethod
from tai42_contract.accounts.provider import AccountsProvider
from tai42_kit.settings import reset_all_settings

import tai42_skeleton.access_control.startup as startup
from tai42_skeleton.access_control.startup import (
    check_accounts_providers_configured,
    check_always_public_routes,
    check_fenced_routes_resolvable,
    check_spa_shell_public,
    probe_identity_provider,
    seed_roles,
)

# -- provider healthcheck probe ----------------------------------------------


class _SpyProvider(IdentityProvider):
    """A provider whose ``healthcheck`` records that it ran (or raises to fail boot)."""

    def __init__(self, settings, *, fail: Exception | None = None) -> None:
        self._fail = fail
        self.ran = False

    async def validate_token(self, token: str) -> AuthIdentity | None:
        return None

    async def healthcheck(self) -> None:
        self.ran = True
        if self._fail is not None:
            raise self._fail


def _bind_providers(monkeypatch: pytest.MonkeyPatch, providers: dict[str, _SpyProvider], names: list[str]) -> None:
    # Point the configured chain at the given spies and reset the settings cache so the
    # probe resolves them. The autouse registry fixture restores the baseline afterwards.
    for name, spy in providers.items():
        registry._REGISTRY[name] = lambda _settings, spy=spy: spy
    import json

    monkeypatch.setenv("ACCESS_CONTROL_AUTH_PROVIDERS", json.dumps(names))
    reset_all_settings()


async def test_provider_probe_awaits_every_configured_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _SpyProvider(None)
    second = _SpyProvider(None)
    _bind_providers(monkeypatch, {"spy1": first, "spy2": second}, ["spy1", "spy2"])
    try:
        await probe_identity_provider()  # no raise
    finally:
        reset_all_settings()
    assert first.ran is True
    assert second.ran is True


async def test_provider_probe_first_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    # A provider whose own storage is unusable fails the boot loudly; a later provider
    # is never reached (the first failure propagates).
    first = _SpyProvider(None, fail=RuntimeError("provider store unreachable"))
    second = _SpyProvider(None)
    _bind_providers(monkeypatch, {"spy1": first, "spy2": second}, ["spy1", "spy2"])
    try:
        with pytest.raises(RuntimeError, match="provider store unreachable"):
            await probe_identity_provider()
    finally:
        reset_all_settings()
    assert first.ran is True
    assert second.ran is False


# -- registered-vs-configured accounts check ---------------------------------


class _FakeAccountsProvider(AccountsProvider):
    def __init__(self, settings) -> None:
        self.settings = settings

    async def validate_token(self, token: str) -> AuthIdentity | None:
        return None

    def login_methods(self) -> list[LoginMethod]:
        return []

    async def needs_bootstrap(self) -> bool:
        return False

    async def revoke_session(self, token: str) -> bool:
        return False


async def test_registered_but_unconfigured_accounts_provider_fails_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    # A registered accounts provider missing from the chain would mint sessions that
    # never authenticate — boot must fail loudly naming it.
    accounts_registry._REGISTRY["acct"] = _FakeAccountsProvider
    monkeypatch.setenv("ACCESS_CONTROL_AUTH_PROVIDERS", '["redis"]')
    reset_all_settings()
    try:
        with pytest.raises(RuntimeError, match="acct"):
            await check_accounts_providers_configured()
    finally:
        accounts_registry._REGISTRY.pop("acct", None)
        reset_all_settings()


async def test_configured_accounts_provider_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    accounts_registry._REGISTRY["acct"] = _FakeAccountsProvider
    monkeypatch.setenv("ACCESS_CONTROL_AUTH_PROVIDERS", '["acct"]')
    reset_all_settings()
    try:
        await check_accounts_providers_configured()  # no raise
    finally:
        accounts_registry._REGISTRY.pop("acct", None)
        reset_all_settings()


# -- role seeding gate -------------------------------------------------------


async def test_seed_roles_seeds_when_versioned_store_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    # Access control is enabled and a versioned store is wired, so the boot seeds the
    # default role templates into it.
    import tai42_skeleton.access_control.roles as roles
    import tai42_skeleton.versioning as versioning

    seeded = False

    async def _seed() -> None:
        nonlocal seeded
        seeded = True

    monkeypatch.setattr(versioning, "versioned_store_configured", lambda: True)
    monkeypatch.setattr(roles, "seed_default_roles", _seed)
    await seed_roles()
    assert seeded is True


async def test_seed_roles_skips_when_no_versioned_store(monkeypatch: pytest.MonkeyPatch) -> None:
    # Roles live in the versioned document store, so a deployment without one seeds
    # nothing and never opens a Postgres connection at boot.
    import tai42_skeleton.access_control.roles as roles
    import tai42_skeleton.versioning as versioning

    async def _seed() -> None:
        raise AssertionError("seed_default_roles must not run without a versioned store")

    monkeypatch.setattr(versioning, "versioned_store_configured", lambda: False)
    monkeypatch.setattr(roles, "seed_default_roles", _seed)
    await seed_roles()  # no raise, no seed


# -- always-public route guard -----------------------------------------------


def _bind_public_check(monkeypatch: pytest.MonkeyPatch, routes: list, prefixes=("/api/login",)) -> None:
    """Point the always-public check at fixed route metadata and prefixes. The check
    reads only ``.path``/``.methods``/``.authed`` off each entry, so lightweight
    stand-ins suffice."""
    from tai42_skeleton.app import route_registry as rr

    monkeypatch.setattr(rr.route_registry, "routes", lambda: routes)
    monkeypatch.setattr(
        startup, "access_control_settings", lambda: SimpleNamespace(always_public_path_prefixes=prefixes)
    )


def _meta(path: str, methods: tuple[str, ...], authed: bool):
    return SimpleNamespace(path=path, methods=methods, authed=authed)


async def test_check_always_public_routes_raises_on_authed_offender(monkeypatch: pytest.MonkeyPatch) -> None:
    # A route under an always-public prefix that declares authed=True is a
    # credential-front-door contradiction: the boot must REFUSE, naming the path.
    _bind_public_check(monkeypatch, [_meta("/api/login/methods", ("POST",), True)])
    with pytest.raises(RuntimeError, match="/api/login/methods"):
        await check_always_public_routes()


async def test_check_always_public_routes_passes_and_logs_public_route(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    # A public route (authed=False) under the prefix passes and is enumerated in the
    # single info line so an accidental mount stays visible at boot.
    _bind_public_check(monkeypatch, [_meta("/api/login/methods", ("GET",), False)])
    with caplog.at_level("INFO"):
        await check_always_public_routes()  # no raise
    assert "always-public routes (no auth)" in caplog.text
    assert "GET /api/login/methods" in caplog.text


async def test_check_always_public_routes_ignores_routes_outside_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    # A route OUTSIDE the always-public prefix is ignored entirely — even authed=True is
    # legal there — so the check neither raises nor enumerates it.
    _bind_public_check(monkeypatch, [_meta("/api/tools/run", ("POST",), True)])
    await check_always_public_routes()  # no raise: the offender is not under the prefix


# -- SPA-shell public fallback boot audit (H3/H4/H7) -------------------------


def _bind_spa_check(
    monkeypatch: pytest.MonkeyPatch,
    routes: list,
    *,
    derived: set[str],
    acknowledged: tuple[str, ...] = (),
    spa_shell_public: bool = True,
) -> None:
    """Point ``check_spa_shell_public`` at fixed route metadata + a fixed derived reserved
    set + fixed settings. The audit reads only ``.path``/``.methods``/``.authed`` per route."""
    import tai42_skeleton.access_control.verifier as verifier_module
    from tai42_skeleton.app import route_registry as rr

    monkeypatch.setattr(rr.route_registry, "routes", lambda: routes)
    monkeypatch.setattr(verifier_module, "registered_reserved_get_paths", lambda: frozenset(derived))
    monkeypatch.setattr(
        startup,
        "access_control_settings",
        lambda: SimpleNamespace(spa_shell_public=spa_shell_public, acknowledged_public_routes=acknowledged),
    )


async def test_spa_check_fails_on_unacknowledged_public_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    # An authed=False non-/api GET route not on the allowlist HALTS boot (H3): a boot-log
    # flag is not a control — the reviewer must consciously acknowledge it.
    _bind_spa_check(monkeypatch, [_meta("/dashboard", ("GET",), False)], derived={"/dashboard"})
    with pytest.raises(RuntimeError, match="/dashboard"):
        await check_spa_shell_public()


async def test_spa_check_passes_acknowledged_and_logs_surface(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    # Acknowledged public-by-declaration routes pass; the derived reserved set and the
    # acknowledged routes are both printed (H7) so the surface is reviewable at boot.
    _bind_spa_check(
        monkeypatch,
        [_meta("/health", ("GET",), False), _meta("/metrics", ("GET",), False)],
        derived={"/health", "/metrics"},
        acknowledged=("/health", "/metrics"),
    )
    with caplog.at_level("INFO"):
        await check_spa_shell_public()  # no raise
    assert "SPA-shell public fallback ON" in caplog.text
    assert "/health" in caplog.text
    assert "/metrics" in caplog.text
    assert "acknowledged public-by-declaration" in caplog.text


async def test_spa_check_fails_on_authed_route_invisible_to_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    # An authed=True non-/api GET route the fallback derivation cannot see would resolve
    # public via the shell — a contradiction the boot refuses.
    _bind_spa_check(monkeypatch, [_meta("/secretpage", ("GET",), True)], derived=set())
    with pytest.raises(RuntimeError, match="/secretpage"):
        await check_spa_shell_public()


async def test_spa_check_excludes_api_and_mcp_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    # /api + /mcp routes — concrete AND templated — are excluded from the audit: serve_spa
    # 404s them, so the shell tier can never reach them. None of these should halt boot,
    # even unacknowledged.
    _bind_spa_check(
        monkeypatch,
        [
            _meta("/api/tools", ("GET",), False),
            _meta("/mcp/x", ("GET",), False),
            _meta("/api/plugins/{name}/studio/{path:path}", ("GET",), False),
            _meta("/mcp/{tool}", ("GET",), True),
        ],
        derived=set(),
    )
    await check_spa_shell_public()  # no raise


async def test_spa_check_fails_on_templated_authed_route(monkeypatch: pytest.MonkeyPatch) -> None:
    # A TEMPLATED authed=True non-/api GET route is structurally not derivable into the
    # concrete reserved set: a concrete request matching its pattern with no route row
    # would be served the public shell. The exhaustive audit refuses to skip it and FAILS
    # the boot, forcing the author to /api-prefix it or acknowledge it.
    _bind_spa_check(monkeypatch, [_meta("/reports/{id}", ("GET",), True)], derived=set())
    with pytest.raises(RuntimeError, match=r"/reports/\{id\}"):
        await check_spa_shell_public()


async def test_spa_check_fails_on_unacknowledged_templated_public_route(monkeypatch: pytest.MonkeyPatch) -> None:
    # A TEMPLATED authed=False non-/api GET route (the webhook door) is no longer skipped:
    # unacknowledged, it halts boot exactly like a concrete public-by-declaration route.
    _bind_spa_check(monkeypatch, [_meta("/universal_webhook/{topic}", ("GET",), False)], derived=set())
    with pytest.raises(RuntimeError, match=r"/universal_webhook/\{topic\}"):
        await check_spa_shell_public()


async def test_spa_check_passes_acknowledged_templated_route(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    # Acknowledged by its REGISTERED template string, the same templated public route passes
    # CONSCIOUSLY and is printed in the acknowledged surface line.
    _bind_spa_check(
        monkeypatch,
        [_meta("/universal_webhook/{topic}", ("GET",), False), _meta("/{spa_path:path}", ("GET",), False)],
        derived=set(),
        acknowledged=("/universal_webhook/{topic}", "/{spa_path:path}"),
    )
    with caplog.at_level("INFO"):
        await check_spa_shell_public()  # no raise
    assert "/universal_webhook/{topic}" in caplog.text
    assert "/{spa_path:path}" in caplog.text


# -- fenced-route resolvability boot guarantee -------------------------------


async def test_check_fenced_routes_resolvable_passes_on_real_surface() -> None:
    # Every registered fenced/secret route resolves back to itself through the gate's
    # resolver, so the real surface passes the boot guarantee.
    from tai42_skeleton.access_control.role_gate import reset_route_index

    reset_route_index()
    await check_fenced_routes_resolvable()


async def test_check_fenced_routes_resolvable_raises_when_a_fence_does_not_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If a REGISTERED fenced/secret route fails to resolve, the fence would silently fail
    # open — the boot must refuse. Force the resolver to miss on one real fenced route.
    from tai42_skeleton.access_control import role_gate
    from tai42_skeleton.app import route_registry as rr

    rr.load_all_routes()
    target = next(m for m in rr.route_registry.routes() if m.action in ("fenced", "secret"))
    original = role_gate.resolve_route_meta

    def _fake_resolve(path, method):
        if path == target.path:
            return None
        return original(path, method)

    monkeypatch.setattr(role_gate, "resolve_route_meta", _fake_resolve)
    with pytest.raises(RuntimeError, match="fail open"):
        await check_fenced_routes_resolvable()
