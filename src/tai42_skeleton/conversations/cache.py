import logging
from threading import RLock

from cachetools import LRUCache, cached
from tai42_kit.settings import register_settings_reset

from tai42_skeleton.conversations.managers.base_conversations_manager import BaseConversationsManager
from tai42_skeleton.conversations.managers.in_memory_conversations_manager import InMemoryConversationsManager
from tai42_skeleton.conversations.managers.redis_conversations_manager import RedisConversationsManager
from tai42_skeleton.conversations.settings import ConversationsSettings

logger = logging.getLogger(__name__)

# Named module-level cache + lock so the settings-reset hook can clear the live
# singleton on a config reload.
_CONVERSATIONS_MANAGER_KEY = "conversations_manager_singleton"
_CONVERSATIONS_MANAGER_CACHE: LRUCache = LRUCache(maxsize=1)
_CONVERSATIONS_MANAGER_LOCK = RLock()


@cached(
    _CONVERSATIONS_MANAGER_CACHE,
    key=lambda *args, **kwargs: _CONVERSATIONS_MANAGER_KEY,
    lock=_CONVERSATIONS_MANAGER_LOCK,
)
def get_conversations_manager() -> BaseConversationsManager:
    """The process-wide routing-row manager, selected from ``CONVERSATIONS_*`` config: the
    null in-memory backend without ``CONVERSATIONS_REDIS_URL``, else the Redis one. Cached
    over a settings snapshot and rebuilt on a settings reload.
    """
    settings = ConversationsSettings()
    if settings.in_memory:
        # Cached, so this fires once per worker process.
        logger.warning(
            "conversation routes are UNAVAILABLE because CONVERSATIONS_REDIS_URL is not set: "
            "the routing-row store has no durable backend and every routing operation refuses "
            "with a 501; set CONVERSATIONS_REDIS_URL to enable conversation routes."
        )
        return InMemoryConversationsManager(settings)
    return RedisConversationsManager(settings)


@register_settings_reset
def _reset_conversations_manager() -> None:
    # The rows live in Redis, so dropping the settings-snapshotting singleton loses no state.
    with _CONVERSATIONS_MANAGER_LOCK:
        _CONVERSATIONS_MANAGER_CACHE.clear()
