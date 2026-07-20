"""Fakes for the redis-backed hooks feature.

``FakeRedis`` covers exactly the hash + pipeline operations
``RedisHooksManager`` calls (``hset``/``hdel``/``hget``/``hgetall`` direct and
queued through a pipeline) plus the two atomic register/unregister ``eval``
scripts. ``bound_app`` binds a fake ``tai_app`` impl exposing the template
manager + tool runner the firing path reaches.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest


class _FakePipeline:
    """Queues commands; ``execute`` runs them in order and returns their results,
    matching how ``RedisHooksManager`` builds register/unregister/list batches."""

    def __init__(self, redis: FakeRedis) -> None:
        self._r = redis
        self._queue: list = []

    def hset(self, *a, **k):
        self._queue.append(("hset", a, k))
        return self

    def hdel(self, *a, **k):
        self._queue.append(("hdel", a, k))
        return self

    def hget(self, *a, **k):
        self._queue.append(("hget", a, k))
        return self

    async def execute(self) -> list:
        results = [getattr(self._r, "_" + name)(*a, **k) for name, a, k in self._queue]
        self._queue.clear()
        return results


class FakeRedis:
    def __init__(self) -> None:
        self._hashes: dict[str, dict[str, str]] = {}

    # internal command impls shared by direct calls + pipeline
    def _hset(self, key, field=None, value=None, **_kw) -> int:
        self._hashes.setdefault(key, {})[str(field)] = str(value)
        return 1

    def _hdel(self, key, *fields) -> int:
        h = self._hashes.get(key, {})
        return sum(1 for f in fields if h.pop(f, None) is not None)

    def _hget(self, key, field):
        return self._hashes.get(key, {}).get(field)

    # direct async surface
    async def hget(self, key, field):
        return self._hget(key, field)

    async def hgetall(self, key) -> dict:
        return dict(self._hashes.get(key, {}))

    async def hset(self, key, field=None, value=None, **kw) -> int:
        return self._hset(key, field, value, **kw)

    async def hdel(self, key, *fields) -> int:
        return self._hdel(key, *fields)

    async def eval(self, script: str, numkeys: int, *keys_and_args) -> int:
        """Emulate the atomic register/unregister Lua scripts.

        The real manager runs them server-side; the fake recognizes each by its
        marker comment and runs the equivalent Python — atomic here because the
        fake is single-threaded async. Signature-compatible with
        ``redis.eval(script, numkeys, *keys, *args)``.
        """
        if "hooks:register:atomic" in script:
            map_key, prefix, topic, name, hook_json = keys_and_args
            prev = self._hget(map_key, name)
            if prev and prev != topic:
                self._hdel(f"{prefix}:topic:{prev}", name)
            self._hset(f"{prefix}:topic:{topic}", name, hook_json)
            self._hset(map_key, name, topic)
            return 1
        if "hooks:unregister:atomic" in script:
            map_key, prefix, name = keys_and_args
            topic = self._hget(map_key, name)
            if not topic:
                return 0
            removed = self._hdel(f"{prefix}:topic:{topic}", name)
            removed_map = self._hdel(map_key, name)
            return 1 if (removed > 0 or removed_map > 0) else 0
        raise NotImplementedError("FakeRedis.eval only emulates the register/unregister scripts")

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self)


def make_client_ctx(fake: FakeRedis):
    @asynccontextmanager
    async def _ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield fake

    return _ctx


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def make_ctx():
    return make_client_ctx


class _FakeResourceManager:
    """Renders by returning inline ``content`` (or a per-id mapping), recording
    every call so a test can assert the firing path rendered condition + expr."""

    def __init__(self, by_id: dict | None = None) -> None:
        self._by_id = by_id or {}
        self.calls: list = []

    async def render_by_id_or_content(self, *, content, template_id, kwargs):
        self.calls.append((content, template_id, kwargs))
        if template_id is not None:
            return self._by_id.get(template_id)
        return content


class _FakeTools:
    def __init__(self, raise_for: set[str] | None = None) -> None:
        self.runs: list = []
        self._raise_for = raise_for or set()

    async def run_tool(self, name, tool_input):
        self.runs.append((name, tool_input))
        if name in self._raise_for:
            raise RuntimeError(f"tool {name} failed")
        return {"ok": True}


class _FakeStorage:
    def __init__(self, resource_manager: _FakeResourceManager) -> None:
        self.resource_manager = resource_manager


class _FakeApp:
    def __init__(self, *, by_id: dict | None = None, raise_tools: set[str] | None = None) -> None:
        self.resource_manager = _FakeResourceManager(by_id)
        self.storage = _FakeStorage(self.resource_manager)
        self.tools = _FakeTools(raise_tools)


@pytest.fixture
def make_app():
    """Factory binding a fake ``tai_app`` impl; unbinds after the test."""
    from tai_contract.app import tai_app

    created: list = []

    def _make(*, by_id: dict | None = None, raise_tools: set[str] | None = None) -> _FakeApp:
        app = _FakeApp(by_id=by_id, raise_tools=raise_tools)
        tai_app.bind(app)
        created.append(app)
        return app

    try:
        yield _make
    finally:
        tai_app.bind(None)
