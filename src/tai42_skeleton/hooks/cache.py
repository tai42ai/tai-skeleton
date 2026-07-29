import logging
from threading import RLock

from cachetools import LRUCache, cached
from tai42_kit.settings import register_settings_reset

from tai42_skeleton.hooks.managers.base_hooks_manager import BaseHooksManager
from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai42_skeleton.hooks.managers.redis_hooks_manager import RedisHooksManager
from tai42_skeleton.hooks.settings import HooksSettings

logger = logging.getLogger(__name__)

# Named module-level cache + lock (rather than an anonymous ``@cached`` closure) so
# the settings-reset hook below can PEEK the live singleton to report an in-memory
# loss before clearing it.
_HOOKS_MANAGER_KEY = "hooks_manager_singleton"
_HOOKS_MANAGER_CACHE: LRUCache = LRUCache(maxsize=1)
_HOOKS_MANAGER_LOCK = RLock()


@cached(_HOOKS_MANAGER_CACHE, key=lambda *args, **kwargs: _HOOKS_MANAGER_KEY, lock=_HOOKS_MANAGER_LOCK)
def get_hooks_manager() -> BaseHooksManager:
    """The process-wide hooks manager, selected from ``HOOKS_*`` config.

    A single instance is cached: an ``InMemoryHooksManager`` when no
    ``HOOKS_REDIS_URL`` is configured, else a ``RedisHooksManager``. The singleton
    follows the settings reset — a ``reload_config`` drops it so the next call
    rebuilds against the new ``HOOKS_*`` config. In in-memory mode that means
    registered hooks live only until the next reload or restart (they are dropped
    with a loud warning, see ``_reset_hooks_manager``); set ``HOOKS_REDIS_URL`` for
    durable hooks.
    """
    settings = HooksSettings()
    if settings.in_memory:
        # The cached singleton makes this fire once per worker process — exactly the
        # surface where in-memory hooks silently no-op across siblings.
        logger.warning(
            "hooks are running IN-MEMORY because HOOKS_REDIS_URL is not set: "
            "registrations and deliveries are per-process only, so with more than "
            "one worker (or a separate backend worker) sibling processes will not "
            "see them; set HOOKS_REDIS_URL for shared state (in-memory is valid only "
            "for a single worker)."
        )
        return InMemoryHooksManager(settings)
    return RedisHooksManager(settings)


@register_settings_reset
def _reset_hooks_manager() -> None:
    # The singleton is bound to a snapshot of ``HooksSettings`` (backend choice +
    # pooled redis connection). A settings reload must drop it so the next call
    # rebuilds against the new config: the hooks manager exists to HONOR ``HOOKS_*``
    # config, so it is reset-registered. This is the deliberate opposite of the
    # in-memory sub-MCP store, which is reset-EXEMPT because its purpose is to
    # survive a reload — the asymmetry is intentional, not an inconsistency.
    #
    # Clearing an in-memory manager drops every in-memory-registered hook. That loss
    # must not be silent, so peek the live singleton first and warn (naming the hook
    # count) before clearing. Redis-backed hooks live in Redis, so a Redis manager
    # reset loses nothing and emits no warning.
    with _HOOKS_MANAGER_LOCK:
        manager = _HOOKS_MANAGER_CACHE.get(_HOOKS_MANAGER_KEY)
        if isinstance(manager, InMemoryHooksManager) and manager.hook_count:
            logger.warning(
                "hooks: config reload is discarding the in-memory hooks manager with %d "
                "registered hook(s); in-memory hooks do not survive a reload or restart — "
                "set HOOKS_REDIS_URL for durable hooks",
                manager.hook_count,
            )
        _HOOKS_MANAGER_CACHE.clear()
