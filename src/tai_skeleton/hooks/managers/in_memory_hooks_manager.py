from typing import Any

from tai_contract.hooks.models import HookParams, TopicVerifierBinding

from tai_skeleton.hooks.managers.base_hooks_manager import BaseHooksManager
from tai_skeleton.hooks.settings import HooksSettings


class InMemoryHooksManager(BaseHooksManager):
    def __init__(self, settings: HooksSettings):
        super().__init__(settings)
        self._hooks: dict[str, dict[str, HookParams]] = {}
        self._name_topic_map: dict[str, str] = {}
        self._topic_verifiers: dict[str, dict[str, Any]] = {}

    @property
    def hook_count(self) -> int:
        """Number of live in-memory hooks across every topic bucket.

        A synchronous read (no async API needed) so the settings-reset hook can name
        how many hooks a config reload is about to drop from this manager.
        """
        return sum(len(bucket) for bucket in self._hooks.values())

    async def register(self, params: HookParams) -> bool:
        self.validate_jq_fields(params)
        key = self.settings.get_hook_key(params.topic)

        prev_topic = self._name_topic_map.get(params.name)
        if prev_topic is not None and prev_topic != params.topic:
            # Re-registering under a new topic: drop the entry from the old
            # topic's bucket, or the stale hook keeps firing there forever.
            prev_key = self.settings.get_hook_key(prev_topic)
            bucket = self._hooks.get(prev_key)
            if bucket is not None:
                bucket.pop(params.name, None)
                if not bucket:
                    del self._hooks[prev_key]

        self._hooks.setdefault(key, {})[params.name] = params
        self._name_topic_map[params.name] = params.topic
        return True

    async def unregister(self, name: str) -> bool:
        topic = self._name_topic_map.pop(name, None)
        if not topic:
            return False

        key = self.settings.get_hook_key(topic)

        if key in self._hooks and name in self._hooks[key]:
            del self._hooks[key][name]
            if not self._hooks[key]:
                del self._hooks[key]
        return True

    async def list_hooks_by_topic(self, topic: str) -> dict[str, HookParams]:
        key = self.settings.get_hook_key(topic)
        # Return a copy, not the live bucket: ``on_event`` iterates this map
        # across an await (condition render), so a concurrent register/unregister
        # mutating the live dict mid-iteration would raise "dictionary changed
        # size during iteration".
        return dict(self._hooks.get(key, {}))

    async def list_hooks(self) -> dict[str, HookParams]:
        all_hooks: dict[str, HookParams] = {}
        for _, hooks in self._hooks.items():
            all_hooks.update(hooks)
        return all_hooks

    async def set_topic_verifier(self, topic: str, binding: dict[str, Any]) -> None:
        # Validate the shape on write (loud on a wrong shape) so both backends
        # enforce the same binding contract; store the canonical dict.
        self._topic_verifiers[topic] = TopicVerifierBinding.model_validate(binding).model_dump()

    async def get_topic_verifier(self, topic: str) -> dict[str, Any] | None:
        binding = self._topic_verifiers.get(topic)
        # Return a copy so a caller cannot mutate the stored binding in place.
        return dict(binding) if binding is not None else None

    async def delete_topic_verifier(self, topic: str) -> bool:
        return self._topic_verifiers.pop(topic, None) is not None

    async def all_topic_verifiers(self) -> dict[str, dict[str, Any]]:
        return {topic: dict(binding) for topic, binding in self._topic_verifiers.items()}
