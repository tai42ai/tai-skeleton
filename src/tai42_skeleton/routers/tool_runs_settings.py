"""``TAI_TOOL_RUNS_*`` config for the background tool-run surface.

Settings are co-located with the router and de-mixed like the interactions
config: the Redis connection is a field composed from the kit connection shape
(not a base the feature config extends), so the feature settings declare only
feature fields. Connection values read from the ``TAI_TOOL_RUNS_REDIS_*`` env;
feature values from ``TAI_TOOL_RUNS_*``.
"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai42_kit.clients import RedisConnectionSettings
from tai42_kit.settings import TaiBaseSettings, settings_cache


class ToolRunsRedisSettings(RedisConnectionSettings):
    """Per-deploy Redis holding the run records, per-run liveness keys, and the
    per-tool recent-runs index. Connection values come from the
    ``TAI_TOOL_RUNS_REDIS_*`` env (``TAI_TOOL_RUNS_REDIS_URL`` ...); defaults to
    local dev."""

    model_config = SettingsConfigDict(env_prefix="TAI_TOOL_RUNS_")

    redis_url: str | None = "redis://localhost:6379/0"
    redis_max_connections: int | None = 10

    # A black-holed Redis fails the tool-run store op loudly within 5s instead of
    # hanging the request/loop task: the connect phase and each command read are
    # both bounded. Must be positive.
    socket_connect_timeout: float | None = Field(default=5, gt=0)
    socket_timeout: float | None = Field(default=5, gt=0)


class ToolRunsSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAI_TOOL_RUNS_")

    # Infra: the redis connection is composed from the kit (a field, not a base),
    # so the feature config declares no connection fields of its own.
    redis: ToolRunsRedisSettings = Field(default_factory=ToolRunsRedisSettings)

    # Namespace prefix for every tool-run key.
    key_prefix: str = "tool_runs:"

    # Per-tool recent-runs index size: the ZSET is trimmed to the newest N runs
    # on every submit. Must be positive.
    recent_runs_limit: int = Field(default=50, gt=0)

    # TTL on a run record hash; a finished run's result/error is retrievable for
    # this long after it started (24h). Must be positive — a non-positive TTL
    # would delete the record on write.
    result_ttl_seconds: int = Field(default=86400, gt=0)

    # TTL on a run's liveness key; the supervisor re-sets it every
    # ``liveness_ttl_seconds / 3`` while the tool runs. A still-``running`` record
    # whose liveness key has expired is a dead run (its supervisor's terminal
    # write never landed) and is reconciled to ``lost`` on the next read. Must be
    # positive.
    liveness_ttl_seconds: int = Field(default=30, gt=0)

    # Per uvicorn worker cap on in-flight background runs; at capacity the submit
    # door answers 503 rather than piling up unbounded detached run stacks. The
    # cap is per-process (each worker counts its own runs). Must be positive.
    max_concurrent_runs: int = Field(default=32, gt=0)

    # How long the shutdown handler waits for cancelled supervisors to write their
    # terminal records before teardown proceeds to close pooled clients. A run that
    # does not finish its terminal write in this window reconciles to ``lost``
    # later. Must be positive.
    shutdown_drain_seconds: float = Field(default=10, gt=0)


@settings_cache
def tool_runs_settings() -> ToolRunsSettings:
    return ToolRunsSettings()
