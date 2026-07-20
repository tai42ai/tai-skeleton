"""The derived capability projection — what the caller can actually reach.

``GET /api/auth/me`` answers this projection: the concrete (path, method) surface,
dynamic route patterns, sub-MCP mounts, tools, and agents an authenticated caller can
reach RIGHT NOW, derived — never stored, never a second ACL. The one invariant is
**projection ⊆ gate**: every projected surface is one the real middleware + backend
stack would ADMIT, so the projection can never advertise a door the gate would slam.

How the invariant is held:

- **Scopes are READ, never recomputed** — ``build_projection`` consumes the effective
  (owner-attenuated) scopes the backend already committed to the request, so the
  projection filters against the SAME scope set the edge enforces.
- **Reachability is the gate's own resolution** — the authenticated-always-allowed
  carve-out short-circuits BEFORE resolution (exactly as ``ResourceGuardMiddleware``
  checks it); otherwise every candidate path is resolved through
  :meth:`AccessControlVerifier.resolve_resource_ids` and coverage-checked exactly as the
  middleware does (deny wins: ALL resolved protected ids covered, or ``"*"``).
- **jq is exact, per (path, method)** — every reachable candidate is evaluated through
  the REAL :class:`PolicyEnforcer` with the SAME two-pass context the backend builds (the
  key's condition, then the owner's condition for an owned key), so a jq fence that denies
  a route at the edge denies it in the projection too. An admin (the condition-free
  ``"*"`` discriminator) skips the jq pass — its policy carries no condition by
  definition.

**Point-in-time:** both ``.context.*`` (a single ``get_live_context`` read) and
``.system.time`` (a single ``time.time()`` read, baked into every pass by
``_prepare_pass``) are snapshotted once per build, so a cached projection reports them
as of build time and is point-in-time within the ttl. This is informational only: the
enforcement gate evaluates ``.context.*`` and ``.system.time`` fresh per request, so a
condition whose truth turns on live time can read stale in the projection yet is always
evaluated live at the gate.

**Failure doctrine:** any store/registry/render/jq INFRASTRUCTURE error propagates
(surfaces as a 500) — never a partial projection. A caller's own condition legitimately
DENYING a candidate is not an error (it is exactly what the gate does) and simply omits
that candidate.

**Cache:** keyed on ``(user_id, policy_version, sorted(effective_scopes),
claims_digest)`` — fuller than the policy cache's ``(user_id, version)`` because the
projection also depends on the caller's effective scopes and identity claims, neither
reconstructable from ``user_id`` alone. Every scope/route/policy mutation bumps the
version, so a route-table edit invalidates projections with no new machinery.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from re import _parser as _re_parser  # type: ignore[attr-defined]
from typing import Any

from async_lru import alru_cache
from pydantic import BaseModel
from starlette.authentication import AuthenticationError
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.models import AccessPolicy, JqAuthContext
from tai42_contract.app import tai42_app
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.redis import RedisClient
from tai42_kit.settings import register_settings_reset

from tai42_skeleton.access_control import management
from tai42_skeleton.access_control.policy import PolicyEnforcer
from tai42_skeleton.access_control.settings import AccessControlSettings, access_control_settings
from tai42_skeleton.access_control.user import is_admin_policy
from tai42_skeleton.access_control.verifier import AccessControlVerifier
from tai42_skeleton.app.route_registry import RouteMetadata, load_api_routes
from tai42_skeleton.app.sub_mcp_app import ROOT_PREFIX
from tai42_skeleton.sub_mcp.store import get_sub_mcp_store

logger = logging.getLogger(__name__)

# The synthetic caller id used when the gate is OFF: there is no principal to project,
# so the route returns a total projection under this named identity.
NO_AUTH_USER_ID = "__no_auth__"

# The global tool-execution doors: iff the caller can reach one of these, the whole
# registry tool surface is projected (there is no per-tool ACL). Located by route, not
# divined.
_TOOL_RUN_DOORS: frozenset[tuple[str, str]] = frozenset({("POST", "/api/run-tool"), ("POST", "/api/tool-runs")})


# -- Response models ---------------------------------------------------------


class RouteEntry(BaseModel):
    """A concrete route the caller can reach, with the methods that pass its jq."""

    path: str
    methods: list[str]


class PatternEntry(BaseModel):
    """A dynamic route pattern the caller can reach — a mount/pattern surface that is
    NOT enumerable into concrete paths, projected only when its scope AND jq admit it."""

    pattern: str
    scope_id: str


class SubMcpEntry(BaseModel):
    """A sub-MCP mount the caller can reach."""

    slug: str
    tools: list[str]
    transport: str


class ProjectionResult(BaseModel):
    """The caller's derived capability projection — every field derived, never stored."""

    user_id: str
    owner_user_id: str | None
    admin: bool
    scopes: list[str]
    routes: list[RouteEntry]
    route_patterns: list[PatternEntry]
    sub_mcp: list[SubMcpEntry]
    tools: list[str]
    agents: list[str]
    mintable: bool


# -- Live-source seams (monkeypatch points for unit tests) -------------------


def _registry_routes() -> list[RouteMetadata]:
    """Every registered ``/api/*`` route's (path, methods) metadata."""
    return load_api_routes()


async def _sub_mcp_routes() -> dict[str, Any]:
    """The durable sub-MCP registrations as ``{slug: RouteConfig}`` (coherent across
    workers, not this worker's in-process cache)."""
    return await get_sub_mcp_store().list_routes()


async def _all_registry_tools() -> list[str]:
    """Every registered tool name."""
    return sorted((await tai42_app.tools.get_tools()).keys())


def _all_agent_names() -> list[str]:
    """Every registered agent name."""
    from tai42_skeleton.app import instance

    return sorted(instance.app.agents.all_agents().keys())


# -- claims digest + frozen wrapper (the cache mechanism) --------------------


def _claims_digest(claims: Mapping[str, Any]) -> str:
    """Collapse the whole claims mapping into one hashable token so the cache key
    captures EVERYTHING a jq ``.identity.*`` condition could read (not just the owner
    claim) and can never serve a stale or wrong-identity projection. A non-serializable
    claim value RAISES loudly (the failure doctrine), never a silent digest of a partial
    view."""
    canonical = json.dumps(dict(claims), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _FrozenClaims:
    """A frozen wrapper carrying the full (unhashable) claims INTO the cached body while
    hashing/comparing ONLY on the claims digest.

    ``alru_cache`` hashes every argument, so the raw claims dict cannot be a cache arg;
    this wrapper keys solely on the digest (which already captures the whole claims), so
    the cached body reads ``wrapper.claims`` to build the jq contexts without the dict
    itself entering the hash."""

    __slots__ = ("claims", "digest")

    def __init__(self, claims: Mapping[str, Any], digest: str) -> None:
        self.claims = claims
        self.digest = digest

    def __hash__(self) -> int:
        return hash(self.digest)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FrozenClaims) and self.digest == other.digest


class _FrozenScopes:
    """Carries the ORIGINAL-order effective scopes into the cached build while
    hashing/comparing ONLY on the SORTED tuple.

    The cache key must stay order-independent (a given caller's effective-scope order is
    a deterministic function of its policy + version, so two order-variants of one set
    are the same slot), yet the build must evaluate jq against the SAME scope ORDER the
    backend gate does — the backend enforces over ``resolved_scopes`` in original order,
    so an order-sensitive condition (``.scopes[0]``) would otherwise diverge between
    projection and gate. This wrapper keys on the sorted tuple but hands the build the
    unsorted list."""

    __slots__ = ("_key", "scopes")

    def __init__(self, scopes: list[str]) -> None:
        self.scopes = scopes
        self._key = tuple(sorted(scopes))

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FrozenScopes) and self._key == other._key


# -- representative-path derivation for dynamic patterns ---------------------
#
# A dynamic route pattern is a regex, non-enumerable into concrete paths. To jq-check it
# a single concrete path that PROVABLY matches the pattern is derived from the
# regex AST and validated with ``fullmatch`` before use — a pattern whose representative
# cannot be derived is EXCLUDED (under-showing is safe; over-showing is the topology-leak
# bug), never emitted unfiltered.

_NEGATE_CANDIDATES = "abcdefghijkxyz0123456789-_"


def _category_sample(name: str) -> str | None:
    return {
        "CATEGORY_DIGIT": "1",
        "CATEGORY_WORD": "a",
        "CATEGORY_SPACE": " ",
        "CATEGORY_NOT_DIGIT": "a",
        "CATEGORY_NOT_WORD": "-",
        "CATEGORY_NOT_SPACE": "x",
    }.get(name)


def _category_matches(name: str, char: str) -> bool:
    if name == "CATEGORY_DIGIT":
        return char.isdigit()
    if name == "CATEGORY_WORD":
        return char.isalnum() or char == "_"
    if name == "CATEGORY_SPACE":
        return char.isspace()
    if name == "CATEGORY_NOT_DIGIT":
        return not char.isdigit()
    if name == "CATEGORY_NOT_WORD":
        return not (char.isalnum() or char == "_")
    if name == "CATEGORY_NOT_SPACE":
        return not char.isspace()
    return False


def _member_matches(members: list[tuple[Any, Any]], char: str) -> bool:
    for op, arg in members:
        if op.name == "LITERAL" and arg == ord(char):
            return True
        if op.name == "RANGE" and arg[0] <= ord(char) <= arg[1]:
            return True
        if op.name == "CATEGORY" and _category_matches(arg.name, char):
            return True
    return False


def _first_member_char(members: list[tuple[Any, Any]]) -> str | None:
    for op, arg in members:
        if op.name == "LITERAL":
            return chr(arg)
        if op.name == "RANGE":
            return chr(arg[0])
        if op.name == "CATEGORY":
            return _category_sample(arg.name)
    return None


def _emit_in(items: list[tuple[Any, Any]]) -> str | None:
    negate = bool(items) and items[0][0].name == "NEGATE"
    members = items[1:] if negate else items
    if negate:
        for candidate in _NEGATE_CANDIDATES:
            if not _member_matches(members, candidate):
                return candidate
        return None
    return _first_member_char(members)


def _emit_node(name: str, arg: Any) -> str | None:
    if name == "LITERAL":
        return chr(arg)
    if name == "NOT_LITERAL":
        return "a" if arg != ord("a") else "b"
    if name == "ANY":
        return "x"
    if name == "IN":
        return _emit_in(arg)
    if name in ("MAX_REPEAT", "MIN_REPEAT"):
        minimum, _maximum, subpattern = arg
        sub = _emit_seq(subpattern)
        if sub is None:
            return None
        return sub * (minimum if minimum > 0 else 1)
    if name == "SUBPATTERN":
        return _emit_seq(arg[3])
    if name == "BRANCH":
        for branch in arg[1]:
            emitted = _emit_seq(branch)
            if emitted is not None:
                return emitted
        return None
    if name == "AT":
        return ""
    if name == "CATEGORY":
        return _category_sample(arg.name)
    if name == "RANGE":
        return chr(arg[0])
    # Backreferences, look-arounds, and any opcode without a concrete sample are
    # unsupported: the pattern is excluded rather than guessed.
    return None


def _emit_seq(seq: Any) -> str | None:
    parts: list[str] = []
    for op, arg in seq:
        piece = _emit_node(op.name, arg)
        if piece is None:
            return None
        parts.append(piece)
    return "".join(parts)


def _sample_path_for_pattern(regex: str) -> str | None:
    """A concrete path that PROVABLY matches ``regex`` (validated with ``fullmatch``), or
    ``None`` when no representative can be safely derived."""
    try:
        parsed = _re_parser.parse(regex)
        compiled = re.compile(regex)
    except re.error:
        return None
    sample = _emit_seq(parsed)
    if sample is None:
        return None
    return sample if compiled.fullmatch(sample) is not None else None


# -- gate-faithful reachability + jq -----------------------------------------


async def _path_reachable(
    verifier: AccessControlVerifier,
    settings: AccessControlSettings,
    scope_set: set[str],
    carve_out: frozenset[str],
    version: int,
    path: str,
) -> bool:
    """Whether ``path`` clears the route-resolution + scope-coverage gate (or the
    authenticated-always-allowed carve-out) — the SAME decision ``ResourceGuardMiddleware``
    reaches, jq excluded (jq is a separate per-method pass)."""
    # The carve-out is checked BEFORE resolution, exactly as the middleware does, so a
    # carve-out path that ALSO carries a route row is not under-shown by falling through
    # to a scope-coverage test the middleware never reaches.
    if path in carve_out:
        return True
    ids = await verifier.resolve_resource_ids(path, policy_version=version)
    if ids:
        public = settings.public_resource_id
        if set(ids) == {public}:
            return True
        protected = [rid for rid in ids if rid != public]
        return "*" in scope_set or all(rid in scope_set for rid in protected)
    # No resolved id and not carved: unreachable.
    return False


class _PreparedPass:
    """A single enforce pass (the key's, or the owner's) with everything that is
    INVARIANT across the whole build pre-computed once: the rendered condition string,
    the configured flag, and the ``JqAuthContext`` body sans the per-probe ``request``.

    Only ``.request`` varies per ``(path, method)`` probe, so a probe shallow-copies the
    base body and substitutes ``request`` rather than re-rendering the condition (a
    storage read for a ``condition_id``) and rebuilding+dumping the whole context on
    every pass — mirroring how the backend renders once per request."""

    __slots__ = ("base", "condition", "configured")

    def __init__(self, condition: str | None, configured: bool, base: dict[str, Any]) -> None:
        self.condition = condition
        self.configured = configured
        self.base = base


async def _prepare_pass(
    policy: AccessPolicy,
    scopes: list[str],
    claims: Mapping[str, Any],
    live_ctx: dict[str, Any],
    user_id: str,
    now: float,
) -> _PreparedPass:
    """Render ``policy``'s condition ONCE and build the invariant jq-context body ONCE
    for reuse across every per-probe pass in this build."""
    condition = await tai42_app.storage.resource_manager.render_by_id_or_content(
        content=policy.condition, template_id=policy.condition_id, kwargs=policy.condition_kwargs
    )
    configured = policy.condition is not None or policy.condition_id is not None
    base = JqAuthContext(
        sub=user_id,
        scopes=list(scopes),
        identity=dict(claims),
        policy=policy.policy_data,
        context=live_ctx,
        request={},
        system={"time": now},
    ).model_dump()
    return _PreparedPass(condition, configured, base)


async def _jq_admits(
    enforcer: PolicyEnforcer,
    key_pass: _PreparedPass,
    owner_pass: _PreparedPass | None,
    path: str,
    method: str,
) -> bool:
    """Whether the caller's (and, for an owned key, the owner's) jq condition admits
    ``(path, method)`` — the SAME two-pass evaluation the backend runs, so a fenced route
    is denied here exactly as it is at the edge. A genuine policy DENY returns ``False``;
    a jq/render/store INFRASTRUCTURE fault (a ``PolicyEvaluationError``, which is NOT an
    ``AuthenticationError``) propagates loudly rather than being swallowed as a deny that
    would silently drop the route from a 200 projection."""
    request = {"method": method, "path": path}
    try:
        await enforcer.enforce(
            {**key_pass.base, "request": request}, key_pass.condition, condition_configured=key_pass.configured
        )
        if owner_pass is not None:
            await enforcer.enforce(
                {**owner_pass.base, "request": request},
                owner_pass.condition,
                condition_configured=owner_pass.configured,
            )
    except AuthenticationError:
        return False
    return True


# -- version read + cache ----------------------------------------------------


async def _read_policy_version(settings: AccessControlSettings) -> int:
    """The current policy version (a cheap single-key GET), mirroring the policy cache's
    own read. A backend error RAISES (fail-closed) rather than pinning the cache to one
    slot; a successful read with no key yet is version 0."""
    async with client_ctx(RedisClient, settings.redis) as r:
        raw = await r.get(settings.policy_version_key)
    return int(raw) if raw is not None else 0


_CachedBuilder = Callable[[str, int, _FrozenScopes, _FrozenClaims], Awaitable[ProjectionResult]]
_cached_builder: _CachedBuilder | None = None


def _get_cached_builder(settings: AccessControlSettings) -> _CachedBuilder:
    """The memoized ``alru_cache``-wrapped builder, mirroring ``PolicyEnforcer``'s cache
    (same ``cache_size`` / ``cache_ttl_seconds`` bound). Version participates in the key,
    so a mutation-driven version bump yields a fresh slot — a cross-worker miss."""
    global _cached_builder
    if _cached_builder is None:
        _cached_builder = alru_cache(maxsize=settings.cache_size, ttl=settings.cache_ttl_seconds)(_build_uncached)
    return _cached_builder


@register_settings_reset
def reset_projection_cache() -> None:
    """Drop the memoized builder so a fresh settings object (or a test) rebuilds it.

    Registered with the settings-reset registry so a config reload (which runs
    ``reset_all_settings()``) rebuilds it against the new ``cache_size`` /
    ``cache_ttl_seconds`` bound instead of serving from a builder bound to the stale
    settings snapshot, mirroring the sibling ``@register_settings_reset`` caches."""
    global _cached_builder
    _cached_builder = None


# -- the build ---------------------------------------------------------------


async def build_projection(user_id: str, effective_scopes: list[str], claims: Mapping[str, Any]) -> ProjectionResult:
    """The caller's capability projection (cached, version-keyed).

    ``user_id``, ``effective_scopes``, and ``claims`` come from the request; the caller's
    policy — and, for an owned key, the owner's policy — are fetched INTERNALLY through
    the version-keyed policy cache, so the handler never fetches policy itself."""
    settings = access_control_settings()
    version = await _read_policy_version(settings)
    wrapper = _FrozenClaims(dict(claims), _claims_digest(claims))
    builder = _get_cached_builder(settings)
    return await builder(user_id, version, _FrozenScopes(list(effective_scopes)), wrapper)


async def _build_uncached(
    user_id: str, version: int, scopes: _FrozenScopes, wrapper: _FrozenClaims
) -> ProjectionResult:
    settings = access_control_settings()
    # ORIGINAL-order scopes (the wrapper keys the cache on the sorted tuple, but jq must
    # evaluate the exact order the gate does — see ``_FrozenScopes``).
    effective_scopes = scopes.scopes
    scope_set = set(effective_scopes)
    claims = wrapper.claims
    carve_out = frozenset(settings.authenticated_always_allowed_paths)

    enforcer = PolicyEnforcer(settings)
    verifier = AccessControlVerifier(settings, providers=[])

    policy = await enforcer.get_policy(user_id)
    stored_owner = policy.policy_data.get(OWNER_USER_ID_CLAIM)
    admin = is_admin_policy(policy, stored_owner)

    owner_from_claims = claims.get(OWNER_USER_ID_CLAIM)
    owner_policy = await enforcer.get_policy(owner_from_claims) if owner_from_claims is not None else None

    # One point-in-time live-context read for every jq pass in this build.
    live_ctx = await enforcer.get_live_context(user_id)

    # Render each condition and build each jq-context body ONCE per build (they are
    # invariant across every path/method probe — only ``.request`` varies), then reuse
    # them for every probe. The owner pass exists only for an owned key whose owner
    # carries a condition, matching the backend's key-then-owner two-pass enforce.
    now = time.time()
    key_pass: _PreparedPass | None = None
    owner_pass: _PreparedPass | None = None
    if not admin:
        key_pass = await _prepare_pass(policy, effective_scopes, claims, live_ctx, user_id, now)
        if owner_policy is not None and (owner_policy.condition is not None or owner_policy.condition_id is not None):
            owner_pass = await _prepare_pass(owner_policy, list(owner_policy.scopes), claims, live_ctx, user_id, now)

    async def admits(path: str, method: str) -> bool:
        if admin:
            return True
        assert key_pass is not None  # built for every non-admin caller above
        return await _jq_admits(enforcer, key_pass, owner_pass, path, method)

    # Routes: every registry route whose resolution+scope gate admits it, then jq-filtered
    # per method.
    routes: list[RouteEntry] = []
    projected_pairs: set[tuple[str, str]] = set()
    for meta in _registry_routes():
        # A templated registry path (``/api/agents/{name}/runs``) is a dynamic surface, not
        # a concrete route: ``resolve_resource_ids`` would ``fullmatch`` the brace-literal
        # under a dynamic-pattern row and emit a bogus RouteEntry with literal braces that
        # also double-lists a surface the route_patterns loop already carries. Such routes
        # are represented ONLY via route_patterns (and the sub_mcp/agents lists).
        if "{" in meta.path:
            continue
        if not await _path_reachable(verifier, settings, scope_set, carve_out, version, meta.path):
            continue
        allowed = [method for method in meta.methods if await admits(meta.path, method)]
        if allowed:
            routes.append(RouteEntry(path=meta.path, methods=sorted(allowed)))
            projected_pairs.update((method, meta.path) for method in allowed)
    routes.sort(key=lambda entry: entry.path)

    # Route patterns: scope- AND jq-filtered exactly like routes, via a representative
    # path. A pattern with no derivable representative is excluded (logged), never leaked.
    patterns = await management.get_all_existing_patterns()
    mappings = await management.get_all_route_mappings()
    route_patterns: list[PatternEntry] = []
    for template_url, regex in sorted(patterns.items()):
        scope_id = mappings.get(template_url)
        if scope_id is None:
            continue
        representative = _sample_path_for_pattern(regex)
        if representative is None:
            logger.info("access_control: projection excluding non-sampleable route pattern %r", regex)
            continue
        if not await _path_reachable(verifier, settings, scope_set, carve_out, version, representative):
            continue
        if not await admits(representative, "GET"):
            continue
        route_patterns.append(PatternEntry(pattern=regex, scope_id=scope_id))

    # Sub-MCP mounts: scope- AND jq-filtered exactly like every other surface — coverage
    # on the mount root the gate resolves PLUS a jq GET-probe of the mount root the gate
    # would see, so a mount whose jq condition denies it is not topology-leaked. Only a
    # mount admitted by BOTH is projected, and only its tools fold into the union below.
    sub_mcp: list[SubMcpEntry] = []
    sub_routes = await _sub_mcp_routes()
    for slug in sorted(sub_routes):
        config = sub_routes[slug]
        mount_root = f"{ROOT_PREFIX}/{slug}"
        if not await _path_reachable(verifier, settings, scope_set, carve_out, version, mount_root):
            continue
        if not await admits(f"{mount_root}/", "GET"):
            continue
        sub_mcp.append(SubMcpEntry(slug=slug, tools=list(config.tools), transport=config.transport))

    # Tools: every registry tool iff a global tool-run door is projected; otherwise the
    # union of the allowed sub-MCP mounts' tools. No per-tool ACL exists or is invented.
    if any(door in projected_pairs for door in _TOOL_RUN_DOORS):
        tools = await _all_registry_tools()
    else:
        tool_names: set[str] = set()
        for entry in sub_mcp:
            tool_names.update(entry.tools)
        tools = sorted(tool_names)

    # Agents: each agent whose per-agent run door passes the gate (resolution + jq POST),
    # so a path-specific jq fence projects per-agent truthfully.
    agents: list[str] = []
    for name in _all_agent_names():
        run_path = f"/api/agents/{name}/runs"
        if await _path_reachable(verifier, settings, scope_set, carve_out, version, run_path) and await admits(
            run_path, "POST"
        ):
            agents.append(name)

    return ProjectionResult(
        user_id=user_id,
        owner_user_id=owner_from_claims,
        admin=admin,
        scopes=effective_scopes,
        routes=routes,
        route_patterns=route_patterns,
        sub_mcp=sub_mcp,
        tools=tools,
        agents=agents,
        mintable=_mintable(),
    )


def _mintable() -> bool:
    """Whether any configured identity provider can mint keys — independent of
    ``settings.enable``, so a gate-off deployment whose provider physically cannot mint
    reports ``False``."""
    return any(mintable for _name, mintable in management.provider_capabilities())


def synthetic_full_projection() -> ProjectionResult:
    """The gate-OFF total projection: there is no identity to project, so every surface
    is reachable. ``admin=True`` + ``scopes=["*"]`` under the named ``__no_auth__``
    identity; the list fields are explicitly EMPTY (the Studio renders everything off the
    full-projection flag). ``mintable`` is still DERIVED — a provider that physically
    cannot mint reports ``False`` even here."""
    return ProjectionResult(
        user_id=NO_AUTH_USER_ID,
        owner_user_id=None,
        admin=True,
        scopes=["*"],
        routes=[],
        route_patterns=[],
        sub_mcp=[],
        tools=[],
        agents=[],
        mintable=_mintable(),
    )
