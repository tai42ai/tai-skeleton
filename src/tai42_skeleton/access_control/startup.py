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


async def check_spa_shell_public() -> None:
    """Audit the non-``/api`` GET route surface against the SPA-shell public fallback.

    Driven by the ROUTE REGISTRY, never a static list, and EXHAUSTIVE: it iterates EVERY
    registered non-``/api``/non-``/mcp`` GET route — CONCRETE AND TEMPLATED — and never
    silently skips one. Each such route must fall into exactly ONE bucket, or the boot
    FAILS closed:

    * consciously ACKNOWLEDGED public — its REGISTERED path (the template string for a
      templated route) is in ``acknowledged_public_routes``. This is a registry-level
      match: templates are ordinary keys. The operational probes, the webhook ingress
      door, and the SPA shell catch-all itself are the app's acknowledged public GETs;
    * gated by AUTHZ (``authed=True``) AND visible to the fallback derivation — a CONCRETE
      authed route is in the DERIVED reserved set (the shell fallback skips it, so an
      unmapped request terminal-denies rather than being served the shell). A TEMPLATED
      ``authed=True`` route is structurally NOT derivable into the concrete reserved set:
      a concrete request matching its pattern with no route row would be served the public
      shell, so the boot FAILS it (the author must ``/api``-prefix it — control-plane
      excluded — or consciously acknowledge it), the same posture as the always-public
      authed refusal;
    * ``authed=False`` and NOT acknowledged → the boot FAILS: a boot-log flag is not a
      control; a publicly declared non-API GET must be a consciously reviewed decision.

    ``/api``/``/mcp`` routes (concrete and templated alike) are excluded: ``serve_spa``
    404s them, so the shell tier can never reach them regardless of auth or templating.
    The fallback state and the derived + acknowledged surfaces are printed so drift and the
    public-by-declaration vs shell-fallback split stay reviewable in ops logs. The audit's
    exhaustiveness is what lets the runtime fallback (concrete-match only) rely on it for
    templated routes rather than build a second matcher — see ``resolve_resource_ids``.

    Finally it confirms the terminal-deny exclusion: a ``/api``/``/mcp`` probe is
    segment-under the excluded prefixes, so the GET fallback can never open the control
    plane. (Exercised end-to-end by the terminal-deny + route-walk tests.)
    """
    from tai42_skeleton.access_control.path_canon import canonicalize_path, under_prefix
    from tai42_skeleton.access_control.verifier import registered_reserved_get_paths
    from tai42_skeleton.app.route_registry import route_registry

    settings = access_control_settings()
    derived = registered_reserved_get_paths()
    acknowledged = frozenset(settings.acknowledged_public_routes)

    logger.info(
        "access_control: SPA-shell public fallback %s; derived reserved (gated) non-/api GET routes: %s",
        "ON" if settings.spa_shell_public else "OFF",
        ", ".join(sorted(derived)) or "(none)",
    )

    invisible_authed: list[str] = []
    unacknowledged: list[str] = []
    acknowledged_present: list[str] = []
    for meta in route_registry.routes():
        if "GET" not in meta.methods:
            continue
        registered = meta.path
        # The control plane is excluded structurally: serve_spa 404s /api and /mcp, so the
        # shell tier never reaches them. The literal REGISTERED prefix decides (registered
        # paths carry clean, un-encoded prefixes), so a templated /api route is excluded too.
        if under_prefix(registered, "/api") or under_prefix(registered, "/mcp"):
            continue
        templated = "{" in registered
        # Acknowledgment is REGISTRY-LEVEL: the REGISTERED path — the template string for a
        # templated route — is an ordinary key compared against acknowledged_public_routes.
        # A consciously-public route passes here whatever its concrete/templated shape.
        if registered in acknowledged:
            acknowledged_present.append(registered)
            continue
        if meta.authed:
            # Gated by authz ONLY if the fallback derivation can SEE it. A CONCRETE authed
            # route is in the derived reserved set (the shell skips it). A TEMPLATED authed
            # route is structurally not derivable, so a concrete request matching its pattern
            # with no route row would be served the public shell — the boot must refuse it.
            if templated or canonicalize_path(registered) not in derived:
                invisible_authed.append(registered)
        else:
            # authed=False and not acknowledged: public by declaration with no conscious review.
            unacknowledged.append(registered)

    if invisible_authed:
        raise RuntimeError(
            "access_control: authed=True non-/api GET route(s) would be served the public SPA shell — they are "
            f"not visible in the derived reserved set: {sorted(set(invisible_authed))}. A concrete route must "
            "register so it joins the derived set; a TEMPLATED route is structurally not derivable, so /api-prefix "
            "it (control-plane excluded) or add its registered template to ACCESS_CONTROL_ACKNOWLEDGED_PUBLIC_ROUTES "
            "if it is genuinely public"
        )
    if unacknowledged:
        raise RuntimeError(
            "access_control: authed=False non-/api GET route(s) are public by declaration but not acknowledged: "
            f"{sorted(set(unacknowledged))} — add each (the registered path, or the template string for a templated "
            "route) to ACCESS_CONTROL_ACKNOWLEDGED_PUBLIC_ROUTES if it is intentionally public, else set authed=True "
            "or remove the route"
        )
    if acknowledged_present:
        logger.info(
            "access_control: acknowledged public-by-declaration non-/api GET routes: %s",
            ", ".join(sorted(set(acknowledged_present))),
        )

    # Terminal-deny confirmation: the resolver structurally excludes the control plane
    # from the shell tier, so no unmatched /api or /mcp path can ever reach the SPA shell.
    for probe in ("/api/__boot_probe__", "/mcp/__boot_probe__"):
        if not (under_prefix(probe, "/api") or under_prefix(probe, "/mcp")):
            raise RuntimeError(
                f"access_control: control-plane probe {probe!r} is not excluded from the SPA-shell tier — "
                "the terminal-deny invariant is broken"
            )


async def check_route_actions() -> None:
    """Boot-fail on a route whose authorization action-class cannot be resolved.

    Every gated route carries a required action-class (``read``/``write``/``fenced``/
    ``secret``) — the SINGLE source of its authorization character. A route the registry
    cannot classify, or a grantable ``read``/``write`` route whose declared class
    disagrees with its HTTP method, is a fail-closed contradiction the boot REFUSES
    (allow-by-omission is dead), mirroring the ``summary``/``tags`` registration raises.
    Runs after the routers register so the whole surface is audited at once.
    """
    from tai42_skeleton.app.route_registry import route_action_violations

    violations = route_action_violations()
    if violations:
        raise RuntimeError(
            "access_control: gated route(s) failed the action-class audit — every gated route must "
            f"resolve to a read/write/fenced/secret action-class: {sorted(violations)}"
        )


async def check_fenced_routes_resolvable() -> None:
    """Boot-fail if a registered fenced/secret route does not resolve back to itself.

    The admin-only fence is enforced ONLY where ``resolve_route_meta`` returns the route:
    a genuinely-unregistered path resolves to ``None`` and the per-tag gate correctly
    does not act on it, but a REGISTERED fenced/secret route that fails to resolve would
    be a SILENT fail-open — the fence would never fire. This check closes that by
    construction: every ``fenced``/``secret`` route must resolve via
    ``resolve_route_meta(path, method)`` to ITSELF for each of its methods, or the boot
    refuses to start. Runs after the routers register so the whole surface is audited.

    The resolver's route index is rebuilt first so it reflects the routes that just
    registered — the boot then validates the live surface, and leaves the runtime index
    consistent with what it verified rather than trusting a possibly-stale earlier build.

    Enumerates through ``load_all_routes`` so the router modules are imported before the
    audit runs — iterating the raw registry could pass VACUOUSLY (an empty loop verifies
    nothing) had the routers not yet been imported.
    """
    from tai42_skeleton.access_control.role_gate import reset_route_index, resolve_route_meta
    from tai42_skeleton.app.route_registry import load_all_routes

    reset_route_index()
    unresolvable: list[str] = []
    for meta in load_all_routes():
        if meta.action not in ("fenced", "secret"):
            continue
        for method in meta.methods:
            if resolve_route_meta(meta.path, method) is not meta:
                unresolvable.append(f"{method} {meta.path}")

    if unresolvable:
        raise RuntimeError(
            "access_control: fenced/secret route(s) do not resolve back to themselves via resolve_route_meta — "
            f"the admin-only fence would silently fail open for: {sorted(unresolvable)}"
        )


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
