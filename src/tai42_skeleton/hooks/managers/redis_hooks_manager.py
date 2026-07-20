from typing import Any

from tai42_contract.hooks.models import HookParams, TopicVerifierBinding
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.redis import RedisClient

from tai42_skeleton.hooks.managers.base_hooks_manager import BaseHooksManager
from tai42_skeleton.hooks.settings import HooksSettings
from tai42_skeleton.utils.redis_typing import awaited, eval_script

# The name->topic map read and the per-topic hash writes must be ONE atomic unit,
# or two concurrent moves of the same name (or a register racing an unregister)
# can leave a hook double-listed under a stale topic or orphaned. Each op is a
# single server-side Lua step. KEYS[1] is the name->topic map; the per-topic hook
# hash keys are built inside Lua from the ``prefix`` ARGV to mirror
# ``HooksSettings.get_hook_key`` (``f"{prefix}:topic:{topic}"``) — the prior
# topic's key is not known until the map is read.
#
# register: KEYS[1]=map, ARGV = prefix, topic, name, hook_json
_REGISTER_LUA = """
-- hooks:register:atomic
local map_key = KEYS[1]
local prefix, topic, name, hook_json = ARGV[1], ARGV[2], ARGV[3], ARGV[4]
local prev_topic = redis.call('HGET', map_key, name)
if prev_topic and prev_topic ~= topic then
  redis.call('HDEL', prefix .. ':topic:' .. prev_topic, name)
end
redis.call('HSET', prefix .. ':topic:' .. topic, name, hook_json)
redis.call('HSET', map_key, name, topic)
return 1
"""

# unregister: KEYS[1]=map, ARGV = prefix, name
_UNREGISTER_LUA = """
-- hooks:unregister:atomic
local map_key = KEYS[1]
local prefix, name = ARGV[1], ARGV[2]
local topic = redis.call('HGET', map_key, name)
if not topic then
  return 0
end
local removed = redis.call('HDEL', prefix .. ':topic:' .. topic, name)
local removed_map = redis.call('HDEL', map_key, name)
if removed > 0 or removed_map > 0 then
  return 1
end
return 0
"""


class RedisHooksManager(BaseHooksManager):
    def __init__(self, settings: HooksSettings):
        super().__init__(settings)

    async def register(self, params: HookParams) -> bool:
        self.validate_jq_fields(params)
        async with client_ctx(RedisClient, self.settings.redis) as r:
            await eval_script(
                r,
                _REGISTER_LUA,
                1,
                self.settings.name_trigger_map_key,
                self.settings.prefix,
                params.topic,
                params.name,
                params.model_dump_json(),
            )
        return True

    async def unregister(self, name: str) -> bool:
        async with client_ctx(RedisClient, self.settings.redis) as r:
            removed = await eval_script(
                r,
                _UNREGISTER_LUA,
                1,
                self.settings.name_trigger_map_key,
                self.settings.prefix,
                name,
            )
        return bool(removed)

    async def list_hooks_by_topic(self, topic: str) -> dict[str, HookParams]:
        key = self.settings.get_hook_key(topic)
        async with client_ctx(RedisClient, self.settings.redis) as r:
            data = await awaited(r.hgetall(key))
            return {name: HookParams.model_validate_json(hook_json) for name, hook_json in data.items()} if data else {}

    async def list_hooks(self) -> dict[str, HookParams]:
        hooks: dict[str, HookParams] = {}
        async with client_ctx(RedisClient, self.settings.redis) as r:
            name_topic_map = await awaited(r.hgetall(self.settings.name_trigger_map_key))
            if not name_topic_map:
                return {}

            pipe = r.pipeline()
            names = []

            for name, topic in name_topic_map.items():
                key = self.settings.get_hook_key(topic)
                pipe.hget(key, name)
                names.append(name)

            results = await pipe.execute()

            for name, hook_json in zip(names, results, strict=True):
                if hook_json:
                    hooks[name] = HookParams.model_validate_json(hook_json)
        return hooks

    async def set_topic_verifier(self, topic: str, binding: dict[str, Any]) -> None:
        # Validate the shape on write so a malformed binding can never be stored;
        # persist the canonical JSON.
        model = TopicVerifierBinding.model_validate(binding)
        async with client_ctx(RedisClient, self.settings.redis) as r:
            await awaited(r.hset(self.settings.topic_verifiers_key, topic, model.model_dump_json()))

    async def get_topic_verifier(self, topic: str) -> dict[str, Any] | None:
        async with client_ctx(RedisClient, self.settings.redis) as r:
            raw = await awaited(r.hget(self.settings.topic_verifiers_key, topic))
        if not raw:
            return None
        # Validate on read: a wrong-shape stored value raises here (loud), and the
        # ingress consumer gets a shape-guaranteed dict.
        return TopicVerifierBinding.model_validate_json(raw).model_dump()

    async def delete_topic_verifier(self, topic: str) -> bool:
        async with client_ctx(RedisClient, self.settings.redis) as r:
            removed = await awaited(r.hdel(self.settings.topic_verifiers_key, topic))
        return removed > 0

    async def all_topic_verifiers(self) -> dict[str, dict[str, Any]]:
        async with client_ctx(RedisClient, self.settings.redis) as r:
            data = await awaited(r.hgetall(self.settings.topic_verifiers_key))
        # Validate every entry on read (loud on a wrong shape), matching the
        # single-topic read.
        return {
            topic: TopicVerifierBinding.model_validate_json(binding).model_dump() for topic, binding in data.items()
        }
