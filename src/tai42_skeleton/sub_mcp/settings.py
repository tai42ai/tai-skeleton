"""``SUB_MCP_*`` config for the durable sub-MCP registration store.

Sub-MCP app registrations (``slug -> {tools, transport}``) are per-uvicorn-worker
in-process routing state. To survive a reload and to be visible across workers,
the registrations are persisted to a shared store selected here: a
``RedisSubMcpStore`` when ``SUB_MCP_REDIS_URL`` is set, else a per-process
in-memory store.

A multi-uvicorn-worker deployment (``tai42-skeleton -w N``, ``N > 1``) REQUIRES
``SUB_MCP_REDIS_URL`` — the in-memory store is per-process, so without a shared
Redis a registration made on one worker is invisible to every sibling and no
worker rehydrates another's routes. Only a single-worker deployment may run on the
in-memory store.

Settings are co-located with the store and de-mixed: the Redis connection is a
field composed from the kit connection shape (not a base the feature extends), so
the feature settings declare only feature fields. Connection values read from the
``SUB_MCP_REDIS_*`` env; feature values from ``SUB_MCP_*``.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai42_kit.clients import RedisConnectionSettings
from tai42_kit.settings import TaiBaseSettings, settings_cache


class SubMcpRedisSettings(RedisConnectionSettings):
    """Redis connection for the sub-MCP registration store, composed from the kit
    connection shape. Connection values come from the ``SUB_MCP_REDIS_*`` env
    (``SUB_MCP_REDIS_URL`` …); with no ``redis_url`` the store runs in-memory."""

    model_config = SettingsConfigDict(env_prefix="SUB_MCP_")


class SubMcpSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUB_MCP_",
        frozen=True,
    )

    # Infra: the redis connection is composed from the kit (a field, not a base),
    # so the feature config declares no connection fields of its own.
    redis: SubMcpRedisSettings = Field(default_factory=SubMcpRedisSettings)

    prefix: str = "sub_mcp"

    @property
    def in_memory(self) -> bool:
        return not self.redis.redis_url

    @property
    def routes_key(self) -> str:
        # The single Redis hash holding every registration: field = slug, value =
        # ``RouteConfig`` JSON.
        return f"{self.prefix}:routes"


@settings_cache
def sub_mcp_settings() -> SubMcpSettings:
    return SubMcpSettings()
