import json
from typing import Any

from async_lru import alru_cache
from starlette.authentication import AuthenticationError
from tai42_contract.access_control.models import AccessPolicy
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.redis import RedisClient, hgetall
from tai42_kit.utils.data import run_jq_first

from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.access_control.store import access_control_store


class PolicyEvaluationError(Exception):
    """An INFRASTRUCTURE fault while evaluating a policy condition — a jq timeout, a
    render/template fault, an eval error — as distinct from a genuine policy DENY
    (which stays an ``AuthenticationError``).

    Deliberately NOT an ``AuthenticationError`` subclass: a caller that narrowly
    catches the deny type to fail closed (the projection build) then lets an
    infrastructure fault PROPAGATE loudly instead of silently swallowing it as a
    deny — a vanished route in an otherwise-200 projection. The runtime gate
    (``backend``/``authz``) catches broad ``Exception`` and so still fails closed on
    it."""


def policy_is_empty(policy: AccessPolicy) -> bool:
    """Whether ``policy`` grants nothing at all — no scope and no condition.

    The ONE spelling of "this principal has no policy": an unknown or deleted key reads
    back as exactly this, so every layer that must refuse such a principal (the tokenless
    identity build, the tool edge's live re-read, the execution-key bind door, the HTTP
    backend's owner check) asks the same question and can never disagree about which
    keys exist."""
    return not policy.scopes and policy.condition is None and policy.condition_id is None


class PolicyEnforcer:
    def __init__(self, settings: AccessControlSettings):
        self.settings = settings

        # Cache for Policy (Static Rules). The cache is keyed on (user_id,
        # version): ``version`` is a value the fetch ignores, so a bumped version
        # simply yields a fresh cache slot — a cross-worker miss that re-reads the
        # edited policy from redis instead of serving the stale cached copy.
        self._fetch_policy = alru_cache(maxsize=settings.cache_size, ttl=settings.cache_ttl_seconds)(
            self._raw_fetch_policy_versioned
        )

    async def get_policy(self, user_id: str) -> AccessPolicy:
        """Fetch policy (scopes + rules) for a specific user (Cached).

        Reads the current policy version first (a cheap single-key GET) and mixes
        it into the cache key. In a multi-worker deployment a management edit
        bumps that version, so the stale per-worker cache entry is bypassed on the
        next read without waiting out the ttl.
        """
        return await self.get_policy_at(user_id, await self._current_policy_version())

    async def get_policy_at(self, user_id: str, version: int) -> AccessPolicy:
        """Fetch policy for a specific user at an ALREADY-READ ``version`` — the form
        :meth:`get_policy` is built on, for a decision that reads several policies (a
        key's and its owner's) and then keys a further pass on the same version.

        ``version`` is a CACHE key, not a store coordinate: the fetch always reads the
        row as it stands, and a bumped version simply lands on a fresh slot. Threading
        one version through a whole decision therefore buys two things — one version
        round trip instead of several, and every cache that version keys (the policy
        cache here, the role-grant cache) answering from the same generation, so no layer
        of the decision serves a pre-bump cached copy while another serves a post-bump
        one. The underlying reads stay independent and live, which is what a fire needs.
        """
        return await self._fetch_policy(user_id, version)

    async def current_policy_version(self) -> int:
        """The current policy version (a cheap single-key GET) — the cache key the LIVE
        per-tag grant resolution mixes in, so a role edit's version bump busts it. A
        backend error fails closed by RAISING (surfaces as a clean deny)."""
        return await self._current_policy_version()

    async def _current_policy_version(self) -> int:
        # A backend error here must fail closed by RAISING (surfaces out of
        # ``authenticate`` as a clean deny), never a silent default: swallowing it
        # to a fixed version would pin the whole cache to one slot and serve stale
        # policy for the ttl. A successful read with no key yet is version 0.
        async with client_ctx(RedisClient, self.settings.redis) as r:
            raw = await r.get(self.settings.policy_version_key)
        return int(raw) if raw is not None else 0

    async def _raw_fetch_policy_versioned(self, user_id: str, version: int) -> AccessPolicy:
        # ``version`` participates only in the cache key (see ``__init__``); the
        # actual fetch is version-independent.
        return await self._raw_fetch_policy(user_id)

    async def _raw_fetch_policy(self, user_id: str) -> AccessPolicy:
        # A genuine backend error must fail closed by RAISING, not by returning an
        # empty policy: the alru cache only stores successful returns, so a
        # swallowed error would be cached and stick for the ttl. The error
        # propagates out of ``authenticate``, which turns it into a clean deny.
        data = await access_control_store().get_policy_body(user_id)
        # A successful read with no stored policy is legitimately empty.
        if not data:
            return AccessPolicy()
        return AccessPolicy(**data)

    async def get_live_context(self, user_id: str) -> dict[str, Any]:
        """
        Fetches dynamic context (usage, counters) - ALWAYS fresh from Redis.
        No caching here to ensure security enforcement is based on live data.

        The context is stored as a Redis HASH at ``ac:context:{user_id}``: each
        field is a context field name and each value is that value ``json.dumps``-
        encoded. A bare integer's JSON encoding is its plain digits, so counters an
        external metering writer maintains with ``HINCRBY ac:context:{user_id} used
        1`` are valid JSON numbers — the two writer styles compose. On read the
        per-field ``json.loads`` reassembles the real typed values (ints, objects,
        …) into a plain dict, so a jq allow-condition like
        ``.context.used < .policy.limit`` compares against a real JSON number, not a
        string.

        A malformed field value makes ``json.loads`` RAISE; it propagates out of
        ``authenticate`` and the auth decision fails closed. This is deliberate: a
        fetch/decode failure must NOT be masked as an empty context. Substituting
        ``{}`` makes a missing field read as ``null``, which can satisfy a ``<``
        comparison and flip a deny into an allow -- a real fail-open.

        A *successful* read with no stored context is different: ``HGETALL`` of a
        missing key answers ``{}``, which means there is genuinely no live data yet
        (e.g. no usage recorded), so the empty dict is the true live state and the
        condition is correctly evaluated against it.
        """
        context_key = f"{self.settings.context_prefix}{user_id}"
        async with client_ctx(RedisClient, self.settings.redis) as r:
            raw = await hgetall(r, context_key)
        return {field: json.loads(value) for field, value in raw.items()}

    async def get_auth_data(self, user_id: str) -> tuple[AccessPolicy, dict[str, Any]]:
        policy = await self.get_policy(user_id)
        context = await self.get_live_context(user_id)
        return policy, context

    async def enforce(self, context: dict[str, Any], expression: str | None, *, condition_configured: bool = False):
        if not expression:
            # Distinguish "no condition configured" from "a condition was
            # configured but rendered to empty". When a condition WAS configured
            # (e.g. a jinja ``{% if %}`` whose branch is false, or an undefined
            # var) yet renders to an empty string, treating it as "no condition"
            # would fail open: the caller gets allowed against a condition that
            # never actually passed. So a configured-but-empty condition DENIES;
            # only a genuinely absent condition is a no-op allow.
            if condition_configured:
                raise AuthenticationError("Policy violation: configured condition rendered empty")
            return

        try:
            # Evaluated off-loop under a wall-clock budget (JQ_TIMEOUT_SECONDS) so a
            # hostile/buggy policy expression cannot block the auth path's loop.
            result = await run_jq_first(expression, context)

            if result is not True:
                raise AuthenticationError("Policy violation")

        except AuthenticationError:
            # A genuine policy DENY — re-raise as-is so callers can distinguish it from
            # an infrastructure fault below.
            raise
        except Exception as e:
            # An infrastructure/evaluation fault (jq timeout, render/eval error) — a
            # DISTINCT type so a build-time caller lets it propagate loudly rather than
            # swallowing it as a deny, while the runtime gate's broad ``except`` still
            # fails closed on it.
            raise PolicyEvaluationError(f"Policy error: {e!s}") from e
