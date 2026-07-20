"""``TAI_BUS_*`` config for the internal worker bus.

The worker bus is app-owned INTERNAL infrastructure — the fan-out primitive that
updates every worker (HTTP server and backend runtime alike) — NOT a plugin: none
of these settings name a registrable or swappable surface, and no manifest field
selects a bus implementation. There is exactly one bus.

Settings are de-mixed the same way the other Redis-backed features are: the Redis
connection is a field composed from the kit connection shape (not a base the group
extends), so the group declares only its own fields. Connection values read from
the ``TAI_BUS_REDIS_*`` env; feature values from ``TAI_BUS_*``.

``TAI_BUS_REDIS_URL`` unset means the bus is OFF — the single-worker/file-mode
process runs on :meth:`WorkerBus.local`, and the boot rules name this var when they
refuse a deployment that requires a bus (multi-worker, a registered backend, or
``TAI_CONFIG_MODE=k8s``).

Namespacing (``TAI_BUS_NAMESPACE``, default ``tai``) prefixes the control channel,
every ephemeral reply channel, and every presence key. Redis pub/sub is
server-global (it is NOT scoped by the numeric db), so two stacks sharing one Redis
MUST diverge by namespace or they cross-deliver each other's fleet ops.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai42_kit.clients import RedisConnectionSettings
from tai42_kit.settings import TaiBaseSettings, settings_cache


class BusRedisSettings(RedisConnectionSettings):
    """Redis connection for the worker bus, composed from the kit connection shape.

    Connection values come from the ``TAI_BUS_REDIS_*`` env (``TAI_BUS_REDIS_URL``
    …); with no ``redis_url`` the bus is off and the process runs on
    :meth:`WorkerBus.local`."""

    model_config = SettingsConfigDict(env_prefix="TAI_BUS_")


class BusSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAI_BUS_", frozen=True)

    # Infra: the redis connection is composed from the kit (a field, not a base),
    # so the group declares no connection fields of its own.
    redis: BusRedisSettings = Field(default_factory=BusRedisSettings)

    # Prefixes the control channel, every reply channel, and every presence key so
    # co-tenant stacks on one server-global pub/sub Redis do not cross-deliver.
    namespace: str = "tai"

    # Short liveness deadline: reaching it only ends the brief ack wait — the
    # collector stops waiting on the fast ``received`` acks and budgets the rest of
    # its wait toward the apply/report cut. The missing/departed presence re-check
    # and verdict assignment happen once, in the finalize pass at the report cut
    # (apply deadline) — not here.
    ack_timeout: float = Field(default=2.0, gt=0)

    # Long apply deadline — the report cut. Budgets a worst-case reload (MCP
    # re-probes included) AND the post-apply ``on_fleet_op_applied`` hook. An origin
    # that acked but exceeds this is re-checked against presence: expired ⇒
    # departed, alive ⇒ timed_out.
    apply_timeout: float = Field(default=30.0, gt=0)

    # Presence-key TTL; a subscriber refreshes it at a third of this, so a frozen or
    # killed worker leaves the census within one TTL instead of blocking every op.
    heartbeat_ttl: float = Field(default=15.0, gt=0)

    @property
    def enabled(self) -> bool:
        """True when a Redis URL is configured; False means the bus is off."""
        return bool(self.redis.redis_url)

    @property
    def channel(self) -> str:
        """The single control channel every subscriber consumes."""
        return f"{self.namespace}:bus:control"

    @property
    def reply_prefix(self) -> str:
        """Prefix for the per-dispatch ephemeral reply channels."""
        return f"{self.namespace}:bus:reply:"

    @property
    def presence_prefix(self) -> str:
        """Prefix for the per-origin presence keys the census scans."""
        return f"{self.namespace}:bus:presence:"

    @property
    def presence_pattern(self) -> str:
        """Glob the census scans to enumerate live presence keys."""
        return f"{self.presence_prefix}*"

    def presence_key(self, origin: str) -> str:
        """The presence key for one origin id."""
        return f"{self.presence_prefix}{origin}"


@settings_cache
def bus_settings() -> BusSettings:
    """Return the process-wide :class:`BusSettings`, cached after first load."""
    return BusSettings()
