"""Startup checks for access control.

The policy RULES live in Postgres and the live-context/version-counter surfaces are
plain Redis reads that fail closed at request time, so neither needs a boot probe.
What DOES get boot-time treatment lives here, run once at startup when access control
is enabled: the configured identity providers' OWN storage is probed; the roles the
control plane hands out are seeded; the always-public login surface is enumerated and
guarded against an accidental authed mount; and a registered accounts provider left
out of the resolution chain fails the boot rather than minting dead sessions.
"""

from __future__ import annotations

import logging

from tai42_contract.access_control.registry import get_identity_provider_factory
from tai42_contract.accounts import iter_accounts_provider_factories

from tai42_skeleton.access_control.settings import access_control_settings

logger = logging.getLogger(__name__)


async def probe_identity_provider() -> None:
    """Probe EVERY configured identity provider's OWN storage at startup.

    Resolves each name in ``auth_providers`` through the module-level registry (the
    same deferred path the runtime auth adapter uses — the plugins register on import
    during ``start()``, before this startup handler runs) and awaits its
    ``healthcheck()``. A provider whose storage needs no boot probe inherits the
    contract's default no-op; a key-minting provider probes its own record store. ANY
    provider's failure propagates, so a deployment against a backend a provider cannot
    use fails LOUDLY at boot rather than on the first authenticated request.
    """
    settings = access_control_settings()
    for name in settings.auth_providers:
        provider = get_identity_provider_factory(name)(settings)
        await provider.healthcheck()


async def seed_roles() -> None:
    """Seed the default role templates (admin/editor/viewer) at startup whenever access
    control is enabled.

    Idempotent create-only: an operator-edited template is never overwritten. Runs
    before the server accepts traffic so a bootstrap ``apply_role(user_id, "admin")``
    can never ``KeyError`` on a fresh deployment. A seeding failure fails the boot
    loudly, the same posture as the provider probe.

    The templates live in the versioned document store, so seeding is skipped when no
    versioned store is configured — the same store-configured gate the versioned-preset
    rehydration handler applies, so an access-control deployment without a versioned
    store never opens a Postgres connection at boot.
    """
    from tai42_skeleton.versioning import versioned_store_configured

    if not versioned_store_configured():
        return
    from tai42_skeleton.access_control.roles import seed_default_roles

    await seed_default_roles()


async def check_always_public_routes() -> None:
    """Enumerate the always-public login surface and refuse an authed mount under it.

    After routes are registered, walk the route registry: for every registered route
    whose path falls under ``always_public_path_prefixes`` emit ONE info line naming
    them (so an accidental mount under the public namespace is VISIBLE at every boot),
    and FAIL CLOSED — raise — if any such route carries ``authed=True``. A route that
    resolves public at runtime yet declares itself authed is a credential-front-door
    contradiction the boot must REFUSE, never silently serve public.
    """
    from tai42_skeleton.app.route_registry import route_registry

    settings = access_control_settings()
    prefixes = settings.always_public_path_prefixes

    public_routes = []
    authed_offenders = []
    for meta in route_registry.routes():
        if not _under_prefixes(meta.path, prefixes):
            continue
        for method in meta.methods:
            public_routes.append(f"{method} {meta.path}")
        if meta.authed:
            authed_offenders.append(meta.path)

    if authed_offenders:
        raise RuntimeError(
            "access_control: routes under an always-public prefix declare authed=True — a public "
            f"route must not declare itself authed: {sorted(set(authed_offenders))}"
        )

    if public_routes:
        logger.info("access_control: always-public routes (no auth): %s", ", ".join(sorted(public_routes)))


async def check_accounts_providers_configured() -> None:
    """Refuse to boot when a registered accounts provider is left out of the chain.

    A registered accounts provider still advertises its login methods and mints
    sessions, but if it is missing from ``auth_providers`` the verifier chain never
    consults it — every minted session then 401s as a clean "unknown token" with
    nothing logging the cause. That is misconfiguration, not a legal state, so the boot
    fails loudly naming the missing providers and the fix.
    """
    settings = access_control_settings()
    configured = set(settings.auth_providers)
    missing = [name for name, _factory in iter_accounts_provider_factories() if name not in configured]
    if missing:
        raise RuntimeError(
            "access_control: registered accounts provider(s) are missing from the resolution chain: "
            f"{missing} — add them to ACCESS_CONTROL_AUTH_PROVIDERS or their minted sessions will never "
            "authenticate"
        )


def _under_prefixes(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)
