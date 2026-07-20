from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai42_kit.clients import RedisConnectionSettings
from tai42_kit.settings import TaiBaseSettings


class HooksRedisSettings(RedisConnectionSettings):
    """Redis connection for the hooks registry, composed from the kit connection
    shape. Connection values come from the ``HOOKS_REDIS_*`` env (``HOOKS_REDIS_URL``
    …); with no ``redis_url`` the registry runs in-memory.

    With ``HOOKS_REDIS_URL`` set, hook registrations and deliveries live in Redis
    and are shared across every worker. With it unset the registry runs in-memory,
    per-process — valid only for a single worker; siblings do not see each other's
    hooks (see ``HooksSettings``)."""

    model_config = SettingsConfigDict(env_prefix="HOOKS_")

    # A black-holed Redis fails the hooks-registry op loudly within 5s instead of
    # hanging the request/loop task: the connect phase and each command read are
    # both bounded. Must be positive.
    socket_connect_timeout: float | None = Field(default=5, gt=0)
    socket_timeout: float | None = Field(default=5, gt=0)


class HooksSettings(TaiBaseSettings):
    """Hooks-registry configuration (``HOOKS_*`` env).

    Backend selection follows ``in_memory``: with ``HOOKS_REDIS_URL`` set the
    registry is Redis-backed and shared across all workers; with it unset the
    registry is in-memory and per-process, which is valid ONLY for a single worker
    — sibling workers (and a separate backend worker) do not see each other's
    registrations or deliveries. Set ``HOOKS_REDIS_URL`` for shared state whenever
    more than one worker runs."""

    model_config = SettingsConfigDict(
        env_prefix="HOOKS_",
        frozen=True,
    )

    # Infra: the redis connection is composed from the kit (a field, not a base),
    # so the feature config declares no connection fields of its own.
    redis: HooksRedisSettings = Field(default_factory=HooksRedisSettings)

    # Global bound on in-flight hook executions per manager: the manager creates
    # ONE semaphore of this size at construction, shared across every event's
    # fan-out, so a burst of concurrent events cannot multiply the bound. Must be
    # positive — there is no unbounded mode.
    max_workers: int = Field(default=10, gt=0)
    prefix: str = "hooks"

    @property
    def in_memory(self) -> bool:
        return not self.redis.redis_url

    def get_hook_key(self, topic: str) -> str:
        # Distinct namespace segment so no topic name can collide with the
        # ``name_trigger_map`` key below.
        return f"{self.prefix}:topic:{topic}"

    @property
    def name_trigger_map_key(self) -> str:
        return f"{self.prefix}:name_trigger_map"

    @property
    def topic_verifiers_key(self) -> str:
        # Distinct namespace segment (a hash of topic -> verifier binding JSON) so
        # no topic name can collide with the per-topic hook keys above.
        return f"{self.prefix}:topic_verifiers"
