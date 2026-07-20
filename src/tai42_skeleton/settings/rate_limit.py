"""``TAI_RATE_LIMIT_*`` config for the app-level public-door rate limiter.

The limiter is an app middleware applied to the two PUBLIC door families —
the interactions callback route and ``universal_webhook/*`` — leaving authed
routes untouched (the credential is the gate there). Each family has its own
per-minute limit + 10-second burst and an enable switch, so a flood on one door
cannot exhaust the other's budget. Settings are read at call time.
"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai42_kit.clients import RedisConnectionSettings
from tai42_kit.settings import TaiBaseSettings, settings_cache


class RateLimitRedisSettings(RedisConnectionSettings):
    """Redis holding the per-bucket fixed-window counters. Connection values come
    from the ``TAI_RATE_LIMIT_REDIS_*`` env; defaults to local dev."""

    model_config = SettingsConfigDict(env_prefix="TAI_RATE_LIMIT_")

    redis_url: str | None = "redis://localhost:6379/0"
    redis_max_connections: int | None = 10

    # A black-holed Redis fails the rate-limit counter op loudly within 5s instead
    # of hanging the request: the connect phase and each command read are both
    # bounded. Must be positive.
    socket_connect_timeout: float | None = Field(default=5, gt=0)
    socket_timeout: float | None = Field(default=5, gt=0)


class RateLimitSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAI_RATE_LIMIT_")

    # Infra: the redis connection is composed from the kit (a field, not a base),
    # so this config declares no connection fields of its own.
    redis: RateLimitRedisSettings = Field(default_factory=RateLimitRedisSettings)

    # Namespace prefix for every rate-limit counter key.
    key_prefix: str = "ratelimit:"

    # The public ``universal_webhook/*`` door family. Requests allowed per client
    # bucket per minute window, and per 10-second burst window. Must be positive.
    webhook_enabled: bool = True
    webhook_limit: int = Field(default=60, gt=0)
    webhook_burst: int = Field(default=10, gt=0)

    # The public interactions callback door family. Same two-window shape as the
    # webhook family (per-minute limit + 10-second burst). Must be positive.
    interactions_callback_enabled: bool = True
    interactions_callback_limit: int = Field(default=60, gt=0)
    interactions_callback_burst: int = Field(default=10, gt=0)

    # Reverse proxies whose X-Forwarded-For may be trusted for the client-IP
    # resolution. Empty (default) = trust no proxy: the direct peer is the client.
    trusted_proxies: list[str] = Field(default_factory=list)


@settings_cache
def rate_limit_settings() -> RateLimitSettings:
    return RateLimitSettings()
