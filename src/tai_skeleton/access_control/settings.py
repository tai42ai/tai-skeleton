import re
from re import Pattern
from typing import Any

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai_kit.clients import RedisConnectionSettings
from tai_kit.settings import TaiBaseSettings, settings_cache


def _prefix_overlaps(a: str, b: str) -> bool:
    """Whether path prefixes ``a`` and ``b`` overlap — equal, or one nested under the
    other (a prefix that is a path-segment ancestor of the other)."""
    return a == b or a.startswith(f"{b}/") or b.startswith(f"{a}/")


class AccessControlRedisSettings(RedisConnectionSettings):
    """Redis connection for the auth gate, composed from the kit connection shape.

    The gate runs on every request, so the connection tunes a short socket read
    timeout plus timeout-retry: a black-holed redis fails fast instead of hanging
    the auth path. Connection values come from the ``ACCESS_CONTROL_`` redis env
    (``ACCESS_CONTROL_REDIS_URL`` …); only the resilience defaults are set here.
    """

    model_config = SettingsConfigDict(env_prefix="ACCESS_CONTROL_")

    redis_url: str | None = "redis://localhost:6379/0"
    redis_max_connections: int | None = 10
    # Bound the connect phase too, so a black-holed redis fails the auth path fast
    # instead of hanging on connect (``socket_timeout`` already bounds each read).
    # Must be positive.
    socket_connect_timeout: float | None = Field(default=5, gt=0)
    socket_timeout: float | None = 5
    retry_on_timeout: bool = True


class AccessControlSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(env_prefix="ACCESS_CONTROL_")

    enable: bool = True

    # The ordered token-resolution chain: the verifier tries each named provider in
    # turn and the first to recognize a credential wins (a provider error propagates,
    # never falls through). The default preserves the single-``redis`` behavior. Parsed
    # from env as a JSON list (``ACCESS_CONTROL_AUTH_PROVIDERS='["redis"]'``); an empty
    # list with the gate enabled is a misconfiguration rejected in ``model_post_init``.
    auth_providers: list[str] = ["redis"]  # noqa: RUF012

    # Runtime slot, NOT configuration: the application installs its
    # ``AccountsAdminServices`` implementation here (at ``AuthAdapter`` construction)
    # before any accounts-provider factory receives this settings object, so a plugin
    # reaches it only as ``settings.admin`` (the contract's ``AccountsProviderSettings``
    # Protocol) without importing the application. ``exclude=True`` keeps it out of the
    # serialized model, and env parsing never populates it.
    admin: Any = Field(default=None, exclude=True)

    # Infra: the redis connection is composed from the kit (a field, not a base),
    # so the feature config declares no connection fields of its own.
    redis: AccessControlRedisSettings = Field(default_factory=AccessControlRedisSettings)

    cache_size: int = 5000
    cache_ttl_seconds: int = 60

    key_prefix: str = "ac:key:"
    context_prefix: str = "ac:context:"

    # The claim-link store prefix: a one-time claim record lives at
    # ``ac:claim:<sha256(token)>`` as a TTL-bound Redis STRING holding the raw key it
    # hands out. That record is the single at-rest home of a raw key in the system —
    # findable only by hash (never by the raw token), single-use, and TTL-erased — so
    # the prefix is a settings field like every other ``ac:`` family, never a literal.
    claim_prefix: str = "ac:claim:"

    # Claim-link lifetimes. A claim record carries raw key material, so a longer-lived
    # record is a standing hazard: creation defaults to ``claim_link_ttl_seconds`` and
    # accepts an optional per-link ttl capped at ``claim_link_max_ttl_seconds`` (the
    # over-cap request is a loud 400, never a silent clamp). Both must be positive and
    # the default must not exceed the ceiling — enforced in ``model_post_init``.
    claim_link_ttl_seconds: int = 600
    claim_link_max_ttl_seconds: int = 3600

    # A monotonically-bumped counter that gates the policy cache: every read of a
    # user's policy mixes the current value of this key into its cache key, so a
    # management edit that increments it forces a cross-worker cache miss on the
    # next read. Cheap single-key GET on the auth path; INCR on an edit.
    policy_version_key: str = "ac:policy_version"

    public_resource_id: str = "public"

    # Url prefixes that are never public: the access-control management surface must not
    # be usable to de-authenticate itself. Enforced on BOTH sides — ``pin_route_public``
    # rejects a public pin of a route under one of these prefixes with a loud error, and
    # the verifier drops the public marker for a reserved-prefix request path at
    # resolution, so the control plane stays authenticated regardless of what the route
    # table holds (a pinned row, a dynamic pattern, or a direct write). A leaked or
    # coerced operator key therefore cannot durably open the control plane (including the
    # pin door and key minting) to unauthenticated callers. Ingress/sub-app routes that
    # operators legitimately pin public (e.g. ``/universal_webhook/...``) live outside
    # this set; extend it to reserve further prefixes.
    reserved_public_pin_prefixes: tuple[str, ...] = ("/api/auth",)

    # The pre-auth login surface: url prefixes that are ALWAYS public. Resolution
    # answers the public resource id for a path under one of these prefixes WITHOUT
    # consulting the route table (the mirror image of ``reserved_public_pin_prefixes``:
    # reserved = never public regardless of the table, always-public = public
    # regardless of the table), so the login/recovery page is reachable on a fresh
    # deployment with no route rows seeded. NEVER mount a post-auth route under one of
    # these prefixes — a route that resolves public at runtime yet declares itself
    # authed is rejected loudly at boot.
    always_public_path_prefixes: tuple[str, ...] = ("/api/login",)

    # The third prefix/path family: EXACT request paths reachable by ANY authenticated
    # identity regardless of the ROUTE TABLE. Where ``reserved_public_pin_prefixes`` is
    # never-public and ``always_public_path_prefixes`` is public-regardless-of-table,
    # this is allowed-for-any-authenticated-identity-regardless-of-table: an
    # authenticated-always-allowed path is reachable for any authenticated caller even
    # when no route row maps it, so an identity-introspection route works on a fresh
    # deployment with zero rows. EXACT paths (not prefixes) so the set can never
    # accidentally swallow a future sibling route. The carve-out bypasses ONLY the
    # route-table check: the backend's jq enforcement runs BEFORE the guard middleware,
    # so a seeded-role jq condition still applies (the role carve-in in ``roles.py`` is
    # the mandatory companion), and an unauthenticated caller is still denied 401.
    authenticated_always_allowed_paths: tuple[str, ...] = ("/api/auth/me",)

    path_patterns: dict[str, str] = {}  # noqa: RUF012
    compiled_patterns: list[tuple[Pattern, str]] = []  # noqa: RUF012

    def model_post_init(self, __context):
        if self.enable and not self.auth_providers:
            raise ValueError(
                "auth_providers is empty while access control is enabled — an enabled gate "
                "with no identity provider is a misconfiguration; set ACCESS_CONTROL_AUTH_PROVIDERS"
            )

        # A path cannot be both never-public (reserved) and always-public: the two
        # prefix sets must be disjoint, where a prefix equal to or nested under a member
        # of the other set is an overlap. This is what lets ``resolve_resource_ids``
        # short-circuit an always-public path unconditionally — the reserved-drop can
        # never contradict it.
        for reserved in self.reserved_public_pin_prefixes:
            for always in self.always_public_path_prefixes:
                if _prefix_overlaps(reserved, always):
                    raise ValueError(
                        f"prefix {always!r} in always_public_path_prefixes overlaps reserved prefix "
                        f"{reserved!r} — a path cannot be both never-public and always-public"
                    )

        # Every authenticated-always-allowed path must be an absolute path, and none may
        # fall under an always-public prefix: a path cannot be both public-anonymous and
        # authenticated-only. (An entry under a reserved prefix is EXPECTED — the route
        # returns identity-derived data and ``/api/auth`` is reserved-never-public — so
        # there is deliberately no check against ``reserved_public_pin_prefixes``.)
        for path in self.authenticated_always_allowed_paths:
            if not path.startswith("/"):
                raise ValueError(
                    f"authenticated_always_allowed_paths entry {path!r} must be an absolute path starting with '/'"
                )
            for always in self.always_public_path_prefixes:
                if path == always or path.startswith(f"{always}/"):
                    raise ValueError(
                        f"authenticated_always_allowed_paths entry {path!r} falls under always-public prefix "
                        f"{always!r} — a path cannot be both public-anonymous and authenticated-only"
                    )

        # A claim record holds raw key material, so its lifetime must be a bounded,
        # positive window: the default is rejected if it is non-positive or exceeds the
        # hard ceiling (the two numbers a creation request is clamped against).
        if not 0 < self.claim_link_ttl_seconds <= self.claim_link_max_ttl_seconds:
            raise ValueError(
                f"claim_link_ttl_seconds ({self.claim_link_ttl_seconds}) must be > 0 and <= "
                f"claim_link_max_ttl_seconds ({self.claim_link_max_ttl_seconds})"
            )

        if self.path_patterns:
            self.compiled_patterns = [
                (re.compile(pattern), template) for pattern, template in self.path_patterns.items()
            ]


@settings_cache
def access_control_settings() -> AccessControlSettings:
    return AccessControlSettings()
