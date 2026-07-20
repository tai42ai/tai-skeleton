"""In-memory async fake of the redis surface the tool-runs store exercises.

Covers hashes (hset/hgetall), strings (set with EX / get), sorted sets
(zadd/zrevrange/zremrangebyrank/zrem), key deletion, lazily-enforced key TTLs,
a scripted ``eval`` emulating the store's atomic terminal compare-and-set, and a
minimal buffered ``pipeline`` (the store batches its non-branching writes and its
per-run reads). Values are strings, matching the pooled ``RedisClient``'s
``decode_responses=True`` default.

Expiry is driven by ``self.time``. By default that is a manual clock the test
advances explicitly (``advance``) so TTL never depends on wall-clock; pass a
``clock`` callable (e.g. ``time.monotonic``) to drive expiry off real elapsed
time instead — the tool-runs supervisor refreshes liveness on a real
``asyncio.sleep`` cadence, so an end-to-end "slow run stays alive" test needs its
liveness TTL to expire on the same real clock.

The buffered pipeline is non-transactional (no WATCH/MULTI): the store's batched
commands never branch on one another, so queueing and running them in order —
exactly what real redis-py's pipeline does — is sufficient, and the fake being
single-threaded async makes each ``eval`` and ``execute`` naturally atomic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Marker comment identifying the store's terminal compare-and-set script (the
# fake recognizes the script by this line, as the interactions fake does).
_TERMINAL_CAS_MARKER = "tool_runs:terminal-cas"


class _FakePipeline:
    """A buffered pipeline: every queued command runs, in order, on ``execute``
    and its result is collected — the shape real redis-py drives without WATCH."""

    def __init__(self, redis: FakeRedis) -> None:
        self._r = redis
        self._queue: list[tuple[str, tuple, dict]] = []

    def _enqueue(self, name: str, args: tuple, kwargs: dict) -> _FakePipeline:
        self._queue.append((name, args, kwargs))
        return self

    def hset(self, *a, **k) -> _FakePipeline:
        return self._enqueue("_hset", a, k)

    def hgetall(self, *a, **k) -> _FakePipeline:
        return self._enqueue("_hgetall", a, k)

    def set(self, *a, **k) -> _FakePipeline:
        return self._enqueue("_set", a, k)

    def get(self, *a, **k) -> _FakePipeline:
        return self._enqueue("_get", a, k)

    def zadd(self, *a, **k) -> _FakePipeline:
        return self._enqueue("_zadd", a, k)

    def zremrangebyrank(self, *a, **k) -> _FakePipeline:
        return self._enqueue("_zremrangebyrank", a, k)

    def expire(self, *a, **k) -> _FakePipeline:
        return self._enqueue("_expire", a, k)

    async def execute(self) -> list:
        results = [getattr(self._r, name)(*args, **kwargs) for name, args, kwargs in self._queue]
        self._queue.clear()
        return results


class FakeRedis:
    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._hashes: dict[str, dict[str, str]] = {}
        self._strings: dict[str, str] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        # Absolute expiry (in ``self.time`` units) per key; expiry is lazily
        # enforced on read.
        self._ttls: dict[str, float] = {}
        # Manual clock (advanced by ``advance``) unless an external ``clock`` is
        # injected, in which case ``self.time`` reads that instead.
        self._clock = clock
        self._manual = 0.0

    @property
    def time(self) -> float:
        return self._clock() if self._clock is not None else self._manual

    def advance(self, seconds: float) -> None:
        if self._clock is not None:
            raise RuntimeError("advance() is unavailable under an injected clock")
        self._manual += seconds

    def ttl_of(self, key: str) -> float | None:
        """Remaining TTL for ``key`` (test helper) — the absolute expiry minus the
        current clock, or ``None`` when the key carries no TTL."""
        exp = self._ttls.get(key)
        return None if exp is None else exp - self.time

    def _stores(self) -> tuple[dict, ...]:
        return (self._hashes, self._strings, self._zsets)

    def _expired(self, key: str) -> bool:
        exp = self._ttls.get(key)
        if exp is not None and self.time >= exp:
            for store in self._stores():
                store.pop(key, None)
            self._ttls.pop(key, None)
            return True
        return False

    # -- command impls (shared by direct calls + pipeline) -------------------

    def _hset(self, key: str, mapping: dict) -> int:
        h = self._hashes.setdefault(key, {})
        h.update({str(k): str(v) for k, v in mapping.items()})
        return len(mapping)

    def _hgetall(self, key: str) -> dict:
        self._expired(key)
        return dict(self._hashes.get(key, {}))

    def _set(self, key: str, value: str, ex: int | None = None) -> bool:
        self._strings[key] = str(value)
        # Mirror ``SET`` without KEEPTTL: a new value discards any prior TTL.
        if ex is not None:
            self._ttls[key] = self.time + ex
        else:
            self._ttls.pop(key, None)
        return True

    def _get(self, key: str) -> str | None:
        self._expired(key)
        return self._strings.get(key)

    def _zadd(self, key: str, mapping: dict) -> int:
        z = self._zsets.setdefault(key, {})
        z.update({str(m): float(s) for m, s in mapping.items()})
        return len(mapping)

    def _zrevrange(self, key: str, start: int, stop: int) -> list[str]:
        self._expired(key)
        # Descending by score; ties broken by reverse-lexicographic member, as
        # real Redis ZREVRANGE does.
        members = sorted(self._zsets.get(key, {}).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        ordered = [m for m, _ in members]
        if stop == -1:
            return ordered[start:]
        return ordered[start : stop + 1]

    def _zremrangebyrank(self, key: str, start: int, stop: int) -> int:
        z = self._zsets.get(key, {})
        # Ascending rank by score (ties by member), matching Redis rank ordering.
        ordered = [m for m, _ in sorted(z.items(), key=lambda kv: (kv[1], kv[0]))]
        n = len(ordered)
        lo = max(start + n if start < 0 else start, 0)
        hi = min(stop + n if stop < 0 else stop, n - 1)
        if lo > hi:
            return 0
        doomed = ordered[lo : hi + 1]
        for m in doomed:
            del z[m]
        return len(doomed)

    def _zrem(self, key: str, *members: str) -> int:
        z = self._zsets.get(key, {})
        return sum(1 for m in members if z.pop(str(m), None) is not None)

    def _expire(self, key: str, ttl: int) -> bool:
        # Mirror redis: EXPIRE on a missing key is a no-op returning False.
        if not any(key in store for store in self._stores()):
            return False
        self._ttls[key] = self.time + ttl
        return True

    def _delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if any(key in store for store in self._stores()):
                for store in self._stores():
                    store.pop(key, None)
                self._ttls.pop(key, None)
                removed += 1
        return removed

    # -- direct (non-pipeline) async surface ---------------------------------

    async def hset(self, key: str, mapping: dict) -> int:
        return self._hset(key, mapping)

    async def hgetall(self, key: str) -> dict:
        return self._hgetall(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        return self._set(key, value, ex=ex)

    async def get(self, key: str) -> str | None:
        return self._get(key)

    async def zadd(self, key: str, mapping: dict) -> int:
        return self._zadd(key, mapping)

    async def zrevrange(self, key: str, start: int, stop: int) -> list[str]:
        return self._zrevrange(key, start, stop)

    async def zremrangebyrank(self, key: str, start: int, stop: int) -> int:
        return self._zremrangebyrank(key, start, stop)

    async def zrem(self, key: str, *members: str) -> int:
        return self._zrem(key, *members)

    async def expire(self, key: str, ttl: int) -> bool:
        return self._expire(key, ttl)

    async def delete(self, *keys: str) -> int:
        return self._delete(*keys)

    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> int:
        """Emulate the store's atomic terminal compare-and-set. The real store runs
        ``_TERMINAL_CAS_LUA`` server-side; the fake recognizes it by its marker
        comment and runs the equivalent Python — a read-then-conditional-write that
        is atomic here because the fake is single-threaded async. Transitions the
        record only while its stored ``status`` is still ``running`` (returns 1),
        otherwise a no-op (returns 0). Signature-compatible with
        ``redis.eval(script, numkeys, *keys, *args)``."""
        if _TERMINAL_CAS_MARKER not in script:
            raise NotImplementedError("FakeRedis.eval only emulates the terminal compare-and-set script")
        run_key = keys_and_args[:numkeys][0]
        args = keys_and_args[numkeys:]
        ttl = int(args[0])
        field_pairs = args[1:]
        self._expired(run_key)
        if self._hashes.get(run_key, {}).get("status") != "running":
            return 0
        h = self._hashes.setdefault(run_key, {})
        for i in range(0, len(field_pairs), 2):
            h[str(field_pairs[i])] = str(field_pairs[i + 1])
        self._ttls[run_key] = self.time + ttl
        return 1

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self)
