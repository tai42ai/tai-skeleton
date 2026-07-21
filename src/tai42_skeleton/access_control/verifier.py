import logging
import re
from collections.abc import Callable
from re import Pattern

from async_lru import alru_cache
from fastmcp.server.auth import AccessToken, TokenVerifier
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.identity import ApiKeyIdentityProvider, IdentityProvider
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.redis import RedisClient

from tai42_skeleton.access_control.path_canon import MalformedPathError, canonicalize_path, under_prefix
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.access_control.store import access_control_store
from tai42_skeleton.app.route_registry import load_all_routes

logger = logging.getLogger(__name__)

UUID_PATTERN = re.compile(r"/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
DIGIT_PATTERN = re.compile(r"/\d+")


def registered_reserved_get_paths() -> frozenset[str]:
    """The DERIVED SPA-shell reserved set: the canonical path of every CONCRETE,
    non-``/api``, non-``/mcp`` registered GET route (``/health``, ``/ready``,
    ``/metrics``, and any future such route).

    A path that IS a registered route is not the SPA shell, so the GET fallback must
    skip it — deriving the set from the route registry means a newly registered route
    joins it automatically, with no static list to go stale. Templated catch-all/mount
    paths (``/{spa_path:path}``, ``/universal_webhook/{topic}``) are excluded: they are
    not single concrete public URLs and the fallback matches concrete request paths."""
    paths: set[str] = set()
    for meta in load_all_routes():
        if "GET" not in meta.methods or "{" in meta.path:
            continue
        canonical = canonicalize_path(meta.path)
        if under_prefix(canonical, "/api") or under_prefix(canonical, "/mcp"):
            continue
        paths.add(canonical)
    return frozenset(paths)


class AccessControlVerifier(TokenVerifier):
    def __init__(
        self,
        settings: AccessControlSettings,
        providers: list[IdentityProvider] | None = None,
        provider_factories: Callable[[], list[IdentityProvider]] | None = None,
    ):
        super().__init__()
        self.settings = settings

        # Provider resolution is DEFERRED (see AuthAdapter for the timing trap): the
        # concrete provider LIST is bound on the FIRST verify_token and memoized, never
        # at construction. A caller may inject concrete providers directly (tests); the
        # adapter injects a factory that reads the module-level registry, which start()
        # has populated by the first request.
        self._providers = providers
        self._provider_factories = provider_factories

        # Route + dynamic-pattern reads are cached per (key, policy_version): the
        # version participates only in the cache key, so an operator route re-point
        # (or a pattern change) that bumps the version yields a fresh cache slot — a
        # cross-worker miss that re-reads the edited route/patterns instead of
        # serving the stale copy, without waiting out the ttl. Mirrors
        # PolicyEnforcer's version-aware policy cache; the same bump that busts the
        # policy cache busts these.
        self._fetch_route_data = alru_cache(maxsize=settings.cache_size, ttl=settings.cache_ttl_seconds)(
            self._raw_fetch_route_versioned
        )

        self._get_dynamic_patterns = alru_cache(maxsize=1, ttl=settings.cache_ttl_seconds)(
            self._raw_fetch_dynamic_patterns_versioned
        )

        # The DERIVED SPA-shell reserved set, computed once on first use and reused per
        # request (never read from the registry per request). The route registry is
        # import-populated and immutable at runtime — routes are recorded only as router
        # modules import at startup — so unlike the dynamic route table (redis, version-
        # keyed) this needs no version key: a given process's registered routes never
        # change. A fresh verifier re-derives it, so a test registering a route sees it.
        self._registered_route_paths_cache: frozenset[str] | None = None

    def _resolve_providers(self) -> list[IdentityProvider]:
        # Bind the provider list on first use and memoize (deferred resolution — see
        # __init__). Directly-injected providers short-circuit; otherwise the
        # adapter-supplied factory reads the registry. An unknown provider name
        # raises loudly out of the factory here, failing closed downstream.
        if self._providers is None:
            if self._provider_factories is None:
                raise ValueError("no identity providers or provider factories configured")
            self._providers = self._provider_factories()
        return self._providers

    async def verify_token(self, token: str) -> AccessToken | None:
        # Try each configured provider in order: the FIRST to return a non-None
        # AuthIdentity wins; a clean None moves to the next provider; any exception
        # PROPAGATES rather than falling through — an unreachable primary store must
        # never silently shift authentication onto a weaker provider. The backend's
        # per-candidate catch (``AccessControlAuthBackend._get_access_token``) then logs
        # a propagated error and denies, so a provider error can only ever end in a
        # deny, never an allow. The two catches are different axes: the backend iterates
        # CREDENTIALS (two headers), this iterates PROVIDERS for one credential.
        for provider in self._resolve_providers():
            identity = await provider.validate_token(token)
            if identity is None:
                continue

            # Central reserved-claim strip: only the mint path (an
            # ``ApiKeyIdentityProvider``) legitimately carries an owner claim. Strip
            # OWNER_USER_ID_CLAIM from any other provider's claims (an accounts session
            # or an external-issuer validator) BEFORE wrapping them, so owner
            # attenuation can never be driven by a non-mint provider — the enforced
            # guarantee, independent of any per-plugin strip.
            claims = identity.claims
            if not isinstance(provider, ApiKeyIdentityProvider) and OWNER_USER_ID_CLAIM in claims:
                claims = {k: v for k, v in claims.items() if k != OWNER_USER_ID_CLAIM}

            # Return the pure identity token; scopes are injected later by PolicyEnforcer.
            return AccessToken(
                token=token,
                client_id=identity.user_id,
                scopes=[],
                claims=claims,
            )
        return None

    async def resolve_resource_ids(
        self, path: str, method: str | None = None, *, policy_version: int | None = None
    ) -> list[str]:
        # Canonicalize ONCE at the top so EVERY tier — the always-public short-circuit,
        # the route-table lookups, the reserved-drop, and the SPA-shell fallback — decides
        # on the SAME path form (percent-decoded once, slash-collapsed, dot-resolved, no
        # trailing slash). Divergent path forms between tiers are exactly the bypass class
        # the single canonical form closes. A NUL/control/backslash path is malformed: it
        # is denied fail-closed (never reaches the shell) and logged loudly, never coerced
        # into something servable.
        try:
            path = canonicalize_path(path)
        except MalformedPathError:
            logger.warning(
                "access_control: rejected malformed request path %r "
                "(NUL/control/backslash or residual percent-escape) — denying",
                path,
            )
            return []

        # Always-public prefixes short-circuit BEFORE any route-table read: the
        # pre-auth login surface answers the public resource id unconditionally, so it
        # is reachable on a fresh deployment with no route rows. The always-public and
        # reserved prefix sets are validated disjoint at settings construction, so such a
        # path is never also reserved and the reserved-drop below can never contradict this.
        if self._is_always_public_prefix(path):
            return [self.settings.public_resource_id]

        # Read the current policy version once (a single cheap GET) and thread it
        # through every cached route/pattern read below, so a management route
        # re-point that bumped the version is a cross-worker cache miss here and is
        # visible immediately rather than up to the ttl later. A caller that already
        # snapshotted the version for a whole batch (the projection build resolving
        # many paths against one version) passes it in to skip the redundant per-path
        # read; the request-path gate omits it and reads once per request.
        version = policy_version if policy_version is not None else await self._current_policy_version()

        # Accumulate matches from EVERY tier (exact, auto-normalized, explicit and
        # dynamic patterns) into one set. Deny wins across tiers: a path that is
        # both a public exact/auto match AND covered by a protected pattern must
        # resolve to BOTH ids, so the guard sees more than the public id and keeps
        # the route protected. Short-circuiting on the first tier would drop the
        # protected id and open the route.
        found_ids: set[str] = set()

        # 1. Exact Match
        if route := await self._fetch_route_data(path, version):
            found_ids.add(route)

        # 2. Automatic Normalization
        normalized_auto = self._normalize_auto(path)
        if normalized_auto != path and (route := await self._fetch_route_data(normalized_auto, version)):
            found_ids.add(route)

        # 3a. Explicit Patterns. ``fullmatch`` — a prefix match would let a
        # longer, more-privileged path inherit the shorter path's resource id.
        for pattern, template in self.settings.compiled_patterns:
            if pattern.fullmatch(path) and (route := await self._fetch_route_data(template, version)):
                found_ids.add(route)

        # 3b. Dynamic Patterns
        dynamic_patterns = await self._get_dynamic_patterns(version)
        if dynamic_patterns:
            for pattern, template in dynamic_patterns:
                if pattern.fullmatch(path) and (route := await self._fetch_route_data(template, version)):
                    found_ids.add(route)

        # The reserved management prefixes are never public: drop the public marker
        # for a reserved-prefix path even if a route row or a dynamic pattern resolved
        # it, so the control plane can never be served unauthenticated regardless of
        # what the route table holds (a route row, a pattern whose template names a
        # reserved url, or a public pattern that fullmatches a reserved path). An
        # otherwise-unmapped reserved path then resolves to nothing and is denied.
        public = self.settings.public_resource_id
        if public in found_ids and self._is_reserved_prefix(path):
            found_ids.discard(public)

        # SPA-shell public fallback (GET-only, last tier): a GET to an UNMAPPED,
        # non-/api, non-/mcp canonical path that is NOT a registered route is served by
        # the SPA catch-all as the dataless index.html shell — treat it as public so a
        # deep-link refresh reaches the shell. Deny wins: this fires ONLY when the route
        # table resolved NOTHING and the path is not a registered route, so an explicit
        # protected mapping OR any registered operational route still wins, and it never
        # opens a mutation (GET only) nor the API/control-plane surface. ``path`` is the
        # single canonical form computed at the top of this method.
        #
        # The registered-route check below is CONCRETE-only: ``_registered_route_paths()``
        # holds the canonical paths of concrete non-/api GET routes, so a concrete request
        # matching a TEMPLATED route's pattern (e.g. ``/reports/5`` for ``/reports/{id}``)
        # is not seen here and — absent a route row — would be served the shell. The boot
        # audit (``check_spa_shell_public``) closes that gap by construction: it refuses to
        # start when any templated ``authed=True`` non-/api GET route exists that is neither
        # /api-prefixed (control-plane-excluded above) nor consciously acknowledged, so no
        # such route can reach this tier. This fallback's safety for templated routes rests
        # on that audit; the code deliberately does not build a second (shadow) matcher.
        if (
            self.settings.spa_shell_public
            and method == "GET"
            # ``not found_ids`` is the deny-wins guard: an operator who explicitly mapped
            # this path to a protected scope keeps it protected.
            and not found_ids
            # Segment-aware, mirroring ``serve_spa``'s own /api,/mcp 404 guard, so the
            # public surface is exactly the shell surface: the control plane never opens.
            and not under_prefix(path, "/api")
            and not under_prefix(path, "/mcp")
            # The DERIVED reserved check (both sides canonical): any real registered
            # non-/api GET route (health/ready/metrics/…) resolves via its own path,
            # never via the shell fallback — no static list.
            and path not in self._registered_route_paths()
            # Operational paths the registry does not surface as concrete GET routes.
            and path not in self.settings.reserved_operational_supplement
            # Belt-and-suspenders: keeps ``/api/auth`` (and any reserved prefix) gated.
            and not self._is_reserved_prefix(path)
        ):
            found_ids.add(public)

        return list(found_ids)

    def _registered_route_paths(self) -> frozenset[str]:
        """The DERIVED reserved set (cached once per verifier). See
        :func:`registered_reserved_get_paths`."""
        if self._registered_route_paths_cache is None:
            self._registered_route_paths_cache = registered_reserved_get_paths()
        return self._registered_route_paths_cache

    def _is_reserved_prefix(self, path: str) -> bool:
        """Whether ``path`` is the access-control management surface that must never
        resolve public (equal to a reserved prefix or a route beneath it)."""
        return any(under_prefix(path, prefix) for prefix in self.settings.reserved_public_pin_prefixes)

    def _is_always_public_prefix(self, path: str) -> bool:
        """Whether ``path`` is the pre-auth login surface that always resolves public
        (equal to an always-public prefix or a route beneath it)."""
        return any(under_prefix(path, prefix) for prefix in self.settings.always_public_path_prefixes)

    async def _current_policy_version(self) -> int:
        # A backend error here fails closed by RAISING (it surfaces out of the
        # resource guard as a clean deny), never a silent default: swallowing it to
        # a fixed version would pin the route/pattern caches to one slot and serve a
        # stale route map for the ttl. A successful read with no key yet is version
        # 0. Mirrors PolicyEnforcer._current_policy_version.
        async with client_ctx(RedisClient, self.settings.redis) as r:
            raw = await r.get(self.settings.policy_version_key)
        return int(raw) if raw is not None else 0

    def _normalize_auto(self, path: str) -> str:
        path = UUID_PATTERN.sub("/{uuid}", path)
        path = DIGIT_PATTERN.sub("/{id}", path)
        return path

    async def _raw_fetch_route_versioned(self, path: str, version: int) -> str | None:
        # ``version`` participates only in the cache key (see __init__); the actual
        # fetch is version-independent.
        return await self._raw_fetch_route(path)

    async def _raw_fetch_dynamic_patterns_versioned(self, version: int) -> list[tuple[Pattern, str]]:
        # ``version`` participates only in the cache key; the fetch is
        # version-independent.
        return await self._raw_fetch_dynamic_patterns()

    async def _raw_fetch_dynamic_patterns(self) -> list[tuple[Pattern, str]]:
        # A genuine backend/parse error must fail closed by RAISING, never by
        # returning a degraded (empty/partial) result: the alru cache only stores
        # successful returns, so a swallowed error would otherwise be cached and
        # stick for the ttl. The error propagates to ResourceGuardMiddleware,
        # which denies the request -- fail-closed and loud, never silent.
        raw_patterns = await access_control_store().fetch_dynamic_patterns()
        # A successful read with no stored patterns is legitimately empty.
        if not raw_patterns:
            return []

        compiled = []
        for regex, template in raw_patterns.items():
            try:
                compiled.append((re.compile(regex), template))
            except re.error as e:
                # A malformed stored pattern is corrupt config, not an empty
                # result: surface it loudly rather than silently dropping the
                # pattern (which would quietly narrow the matched resource set).
                raise ValueError(f"malformed dynamic route pattern {regex!r}: {e}") from e
        return compiled

    async def _raw_fetch_route(self, path: str) -> str | None:
        # A genuine backend error must fail closed by RAISING (see
        # _raw_fetch_dynamic_patterns): a swallowed error would be cached by alru
        # and stick for the ttl. A successful read with no mapping returns None
        # (legitimately-unknown route -> denied downstream).
        return await access_control_store().fetch_route(path)
