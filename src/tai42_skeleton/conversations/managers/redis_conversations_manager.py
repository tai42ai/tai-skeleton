import logging
from typing import Any

from tai42_contract.conversations import ConversationRoute
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.redis import RedisClient

from tai42_skeleton.conversations.managers.base_conversations_manager import BaseConversationsManager
from tai42_skeleton.utils.redis_typing import awaited, eval_script

logger = logging.getLogger(__name__)

# The per-route key write and the name-index add must be ONE atomic unit, or a create
# racing a delete of the same name leaves the row keyed but unindexed, or indexed but
# keyless. Both keys are passed in from ``ConversationsSettings``.
#
# put: KEYS[1]=names index, KEYS[2]=the route's own key, ARGV = route_name, route_json.
# Returns 1 when the row already existed (a replace), 0 when it is newly created.
_PUT_LUA = """
-- conversations:route:put:atomic
local names_key, route_key = KEYS[1], KEYS[2]
local route_name, route_json = ARGV[1], ARGV[2]
local existed = redis.call('EXISTS', route_key)
redis.call('SET', route_key, route_json)
redis.call('SADD', names_key, route_name)
return existed
"""

# delete: KEYS[1]=names index, KEYS[2]=the route's own key, ARGV = route_name. Returns 1
# when a row was removed, 0 when none existed.
_DELETE_LUA = """
-- conversations:route:delete:atomic
local names_key, route_key = KEYS[1], KEYS[2]
local route_name = ARGV[1]
local removed = redis.call('DEL', route_key)
redis.call('SREM', names_key, route_name)
return removed
"""


def _as_str(value: Any) -> str:
    """Normalize Redis's ``bytes``-or-``str`` return to ``str``."""
    return value.decode() if isinstance(value, bytes) else value


class RedisConversationsManager(BaseConversationsManager):
    async def put_route(self, route: ConversationRoute) -> bool:
        async with client_ctx(RedisClient, self.settings.redis) as r:
            existed = await eval_script(
                r,
                _PUT_LUA,
                2,
                self.settings.route_names_key,
                self.settings.route_key(route.route_name),
                route.route_name,
                route.model_dump_json(),
            )
        # ``existed`` truthy ⇒ a replace; falsy ⇒ a fresh create.
        return not bool(existed)

    async def get_route(self, route_name: str) -> ConversationRoute | None:
        async with client_ctx(RedisClient, self.settings.redis) as r:
            raw = await awaited(r.get(self.settings.route_key(route_name)))
        if raw is None:
            return None
        return ConversationRoute.model_validate_json(_as_str(raw))

    async def delete_route(self, route_name: str) -> bool:
        async with client_ctx(RedisClient, self.settings.redis) as r:
            removed = await eval_script(
                r,
                _DELETE_LUA,
                2,
                self.settings.route_names_key,
                self.settings.route_key(route_name),
                route_name,
            )
        return bool(removed)

    async def list_routes(self) -> dict[str, ConversationRoute]:
        routes: dict[str, ConversationRoute] = {}
        async with client_ctx(RedisClient, self.settings.redis) as r:
            names = await awaited(r.smembers(self.settings.route_names_key))
            if not names:
                return {}
            name_list = sorted(_as_str(name) for name in names)
            raws = await awaited(r.mget([self.settings.route_key(name) for name in name_list]))
        for name, raw in zip(name_list, raws, strict=True):
            if raw is None:
                # Indexed name with no row: a corrupt state (the row key never expires).
                logger.warning("conversations: route name %r is indexed but has no row; skipping", name)
                continue
            routes[name] = ConversationRoute.model_validate_json(_as_str(raw))
        return routes
