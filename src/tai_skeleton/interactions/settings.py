"""``INTERACTIONS_*`` config for the ask_user capability.

Settings are co-located with the impl and de-mixed: the Redis connection is a
field composed from the kit connection shape (not a base the feature config
extends), so the feature settings declare only feature fields. Connection values
read from the ``INTERACTIONS_REDIS_*`` env; feature values from ``INTERACTIONS_*``.
"""

from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict
from tai_kit.clients import RedisConnectionSettings
from tai_kit.settings import TaiBaseSettings, settings_cache

# Hosts for which an http:// (non-TLS) public base URL is accepted — local
# development only; every other host must be https:// (the callback URL built
# from it is a bearer capability and the POST door carries sensitive data).
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1"})


class InteractionsRedisSettings(RedisConnectionSettings):
    """Per-deploy Redis holding the group streams, state hashes, pending index,
    reply channels, and the events tail. Connection values come from the
    ``INTERACTIONS_REDIS_*`` env (``INTERACTIONS_REDIS_URL`` ...); defaults to
    local dev."""

    model_config = SettingsConfigDict(env_prefix="INTERACTIONS_")

    redis_url: str | None = "redis://localhost:6379/0"
    redis_max_connections: int | None = 10

    # A black-holed Redis fails the interactions op loudly within 5s instead of
    # hanging the request/loop task: the connect phase and each command read are
    # both bounded. The two legitimately-blocking commands (BLPOP, keepalive
    # XREAD) open their connection with ``socket_timeout`` stripped and take an
    # explicit outer bound instead (see ``blocking_grace_seconds``). Must be
    # positive.
    socket_connect_timeout: float | None = Field(default=5, gt=0)
    socket_timeout: float | None = Field(default=5, gt=0)


class InteractionsSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(env_prefix="INTERACTIONS_")

    # Infra: the redis connection is composed from the kit (a field, not a base),
    # so the feature config declares no connection fields of its own.
    redis: InteractionsRedisSettings = Field(default_factory=InteractionsRedisSettings)

    # Namespace prefix for every interactions key.
    key_prefix: str = "interactions:"

    # Bound on the internal notifications sink feed (the list ``notify_user`` with
    # no channel writes). The feed is a newest-first ring buffer: each write LTRIMs
    # it to this many entries, keeping the newest N and evicting older ones by
    # design, so the feed key cannot grow without limit. A deliberate, documented
    # retention cap — not a silent truncation. Must be positive.
    notifications_feed_max: int = Field(default=1000, gt=0)

    # Default wait budget for a blocked ask_user before it raises (1h); a caller
    # may override per call. Must be positive.
    answer_timeout_seconds: int = Field(default=3600, gt=0)

    # Idle TTL on a group's stream + state + index entry, refreshed on each new
    # question; a group with no open questions expires after this (24h). Must be
    # positive — a non-positive TTL would delete keys on write.
    idle_ttl_seconds: int = Field(default=86400, gt=0)

    # Public base URL of the host serving the interactions routes; required for
    # external-format questions (the callback URL is built from it). Must be
    # https:// — the callback URL is a bearer capability and the POST door carries
    # sensitive data. http:// is rejected loudly at settings load unless the host
    # is localhost/127.0.0.1 (local development).
    public_base_url: str | None = None

    # Hard cap (bytes) on any request body the interactions doors read into
    # memory — the callback doors (body + query string) and the /answer door.
    # Oversized -> 413, loudly — never truncated. Must be positive.
    callback_max_body_bytes: int = Field(default=65536, gt=0)

    # Open-questions guard: refuse new ask_user calls once this many questions
    # are open platform-wide. None = unlimited; a set value must be positive.
    max_concurrent: int | None = Field(default=None, gt=0)

    # Slack past a legitimately-blocking command's own server-side block window
    # after which its connection is presumed stalled: the BLPOP reply wait and the
    # keepalive XREAD tail run with no socket read timeout, so this bounds them via
    # an outer ``asyncio.wait_for`` (budget/keepalive + grace). Must be positive.
    blocking_grace_seconds: float = Field(default=5, gt=0)

    @field_validator("public_base_url")
    @classmethod
    def _require_tls(cls, value: str | None) -> str | None:
        """Reject a non-TLS public base URL unless it points at
        localhost/127.0.0.1 (local dev) — loud at settings load, before any
        callback URL is ever minted from it."""
        if value is None:
            return None
        parsed = urlparse(value)
        if parsed.scheme == "https":
            return value
        if parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS:
            return value
        raise ValueError(
            f"INTERACTIONS_PUBLIC_BASE_URL must be https:// (or http:// for localhost/127.0.0.1), got {value!r}"
        )


@settings_cache
def interactions_settings() -> InteractionsSettings:
    return InteractionsSettings()
