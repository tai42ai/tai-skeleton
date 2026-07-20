"""In-memory async fake of the redis surface the interactions store, the
``ask_user`` helper, and the interactions router exercise.

Covers hashes, strings (incr/decr/set with EX), streams (xadd/xrange/xrevrange/
xread), sorted sets (zadd with ``GT``, zrem/zremrangebyscore/zcard/zrange/
zrangebyscore), lists (lpush/rpush/lrange/ltrim) with a blocking ``blpop``, key TTLs with ``EXPIRE``
``NX``/``GT`` options (an injectable clock so expiry is driven explicitly, never
by wall-clock), a scripted ``eval`` emulating the store's atomic
pending-deadline purge, and a pipeline honoring the watch/multi transaction
shape ``record_answer`` / ``prune_pending`` rely on. Values are strings, matching
the pooled ``RedisClient``'s ``decode_responses=True`` default.
"""

from __future__ import annotations

import asyncio

from redis.exceptions import WatchError


class _FakePipeline:
    """A pipeline with two modes: queued by default (everything runs on
    ``execute``), and immediate between ``watch`` and ``multi`` (reads run at
    once and return values), exactly as redis-py drives a WATCH/MULTI block.

    WATCH is enforced like real Redis: the versions of the watched keys are
    snapshotted at ``watch`` time and re-checked at ``execute``; a concurrent
    write to any watched key aborts the transaction with ``WatchError`` and runs
    none of the queued commands. This lets tests catch a store that watches the
    wrong key set."""

    def __init__(self, redis: FakeRedis) -> None:
        self._r = redis
        self._queue: list[tuple[str, tuple, dict]] = []
        self._watching = False
        self._in_multi = False
        self._watched: dict[str, int] = {}

    async def __aenter__(self) -> _FakePipeline:
        return self

    async def __aexit__(self, *exc) -> bool:
        await self.reset()
        return False

    def _immediate(self) -> bool:
        return self._watching and not self._in_multi

    async def watch(self, *keys) -> bool:
        self._watching = True
        self._watched.update({k: self._r._versions.get(k, 0) for k in keys})
        return True

    def multi(self) -> None:
        self._in_multi = True

    async def reset(self) -> None:
        self._queue.clear()
        self._watching = False
        self._in_multi = False
        self._watched.clear()

    def _dispatch(self, name: str, args: tuple, kwargs: dict):
        if self._immediate():

            async def _run():
                return getattr(self._r, "_" + name)(*args, **kwargs)

            return _run()
        self._queue.append((name, args, kwargs))
        return self

    def xadd(self, *a, **k):
        return self._dispatch("xadd", a, k)

    def hset(self, *a, **k):
        return self._dispatch("hset", a, k)

    def hget(self, *a, **k):
        return self._dispatch("hget", a, k)

    def hgetall(self, *a, **k):
        return self._dispatch("hgetall", a, k)

    def incr(self, *a, **k):
        return self._dispatch("incr", a, k)

    def decr(self, *a, **k):
        return self._dispatch("decr", a, k)

    def get(self, *a, **k):
        return self._dispatch("get", a, k)

    def set(self, *a, **k):
        return self._dispatch("set", a, k)

    def zadd(self, *a, **k):
        return self._dispatch("zadd", a, k)

    def zrem(self, *a, **k):
        return self._dispatch("zrem", a, k)

    def zremrangebyscore(self, *a, **k):
        return self._dispatch("zremrangebyscore", a, k)

    def rpush(self, *a, **k):
        return self._dispatch("rpush", a, k)

    def lpush(self, *a, **k):
        return self._dispatch("lpush", a, k)

    def ltrim(self, *a, **k):
        return self._dispatch("ltrim", a, k)

    def expire(self, *a, **k):
        return self._dispatch("expire", a, k)

    def delete(self, *a, **k):
        return self._dispatch("delete", a, k)

    async def execute(self) -> list:
        # A watched key changed since ``watch`` → abort, running no queued command.
        for key, version in self._watched.items():
            if self._r._versions.get(key, 0) != version:
                await self.reset()
                raise WatchError()
        results = []
        for name, args, kwargs in self._queue:
            results.append(getattr(self._r, "_" + name)(*args, **kwargs))
        await self.reset()
        return results


class FakeRedis:
    def __init__(self) -> None:
        self._hashes: dict[str, dict[str, str]] = {}
        self._strings: dict[str, str] = {}
        self._streams: dict[str, list[tuple[str, dict]]] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        self._lists: dict[str, list[str]] = {}
        # Absolute expiry (in ``self.time`` units) per key; expiry is lazily
        # enforced on read. Tests advance ``self.time`` to drive TTL.
        self._ttls: dict[str, float] = {}
        self.time = 0.0
        # Monotonic per-stream entry counter — survives MAXLEN trimming so ids
        # never repeat.
        self._stream_seq: dict[str, int] = {}
        # Monotonic per-key write version, bumped on every mutation; the pipeline's
        # WATCH snapshots and re-checks these to detect concurrent writes.
        self._versions: dict[str, int] = {}

    def advance(self, seconds: float) -> None:
        self.time += seconds

    def _bump(self, key: str) -> None:
        self._versions[key] = self._versions.get(key, 0) + 1

    def _expired(self, key: str) -> bool:
        exp = self._ttls.get(key)
        if exp is not None and self.time >= exp:
            self._drop(key)
            return True
        return False

    def _drop(self, key: str) -> None:
        for store in (self._hashes, self._strings, self._streams, self._zsets, self._lists):
            store.pop(key, None)
        self._ttls.pop(key, None)
        self._bump(key)

    # -- internal command impls (shared by direct calls + pipeline) ----------

    def _xadd(self, key, fields, maxlen=None, approximate=False) -> str:
        stream = self._streams.setdefault(key, [])
        # 1-based monotonic ids (independent of the current length, which shrinks
        # on trim) so the "0-0" tail sentinel (which assumes stream ids are
        # always greater than 0-0, as real Redis millisecond ids are) delivers
        # every entry rather than dropping the first.
        seq = self._stream_seq[key] = self._stream_seq.get(key, 0) + 1
        entry_id = f"{seq}-0"
        stream.append((entry_id, {str(k): str(v) for k, v in fields.items()}))
        # MAXLEN trims the oldest entries; the fake trims exactly (a valid
        # realization of approximate trimming, which only permits keeping MORE).
        if maxlen is not None and len(stream) > maxlen:
            del stream[: len(stream) - maxlen]
        self._bump(key)
        return entry_id

    def _hset(self, key, field=None, value=None, mapping=None) -> int:
        # Mirror redis-py ``hset``: either a single field/value pair or a ``mapping``
        # (or both). The interactions store uses the ``mapping=`` form; the sub-MCP
        # store uses the field/value form.
        h = self._hashes.setdefault(key, {})
        items = dict(mapping) if mapping else {}
        if field is not None:
            items[field] = value
        h.update({str(k): str(v) for k, v in items.items()})
        self._bump(key)
        return len(items)

    def _hget(self, key, field):
        self._expired(key)
        return self._hashes.get(key, {}).get(field)

    def _hdel(self, key, *fields) -> int:
        h = self._hashes.get(key, {})
        removed = sum(1 for f in fields if h.pop(f, None) is not None)
        self._bump(key)
        return removed

    def _hgetall(self, key) -> dict:
        self._expired(key)
        return dict(self._hashes.get(key, {}))

    def _incr(self, key) -> int:
        self._expired(key)
        value = int(self._strings.get(key, "0")) + 1
        self._strings[key] = str(value)
        self._bump(key)
        return value

    def _decr(self, key) -> int:
        self._expired(key)
        value = int(self._strings.get(key, "0")) - 1
        self._strings[key] = str(value)
        self._bump(key)
        return value

    def _get(self, key):
        self._expired(key)
        return self._strings.get(key)

    def _set(self, key, value, ex=None) -> bool:
        self._strings[key] = str(value)
        # Mirror ``SET`` without KEEPTTL: a new value discards any prior TTL.
        if ex is not None:
            self._ttls[key] = self.time + ex
        else:
            self._ttls.pop(key, None)
        self._bump(key)
        return True

    def _zadd(self, key, mapping, gt=False) -> int:
        z = self._zsets.setdefault(key, {})
        added = 0
        for member, score in mapping.items():
            member = str(member)
            score = float(score)
            if member not in z:
                z[member] = score
                added += 1
            elif not gt or score > z[member]:
                # ``GT`` updates an existing member only when the new score is
                # greater; without ``GT`` every write updates unconditionally.
                z[member] = score
        self._bump(key)
        return added

    def _zrem(self, key, *members) -> int:
        z = self._zsets.get(key, {})
        removed = sum(1 for m in members if z.pop(str(m), None) is not None)
        self._bump(key)
        return removed

    def _zremrangebyscore(self, key, min_score, max_score) -> int:
        z = self._zsets.get(key, {})
        doomed = [m for m, s in z.items() if min_score <= s <= max_score]
        for m in doomed:
            del z[m]
        self._bump(key)
        return len(doomed)

    def _zrangebyscore(self, key, min_score, max_score) -> list[str]:
        z = self._zsets.get(key, {})
        matched = sorted(((m, s) for m, s in z.items() if min_score <= s <= max_score), key=lambda kv: kv[1])
        return [m for m, _ in matched]

    def _rpush(self, key, *values) -> int:
        lst = self._lists.setdefault(key, [])
        lst.extend(str(v) for v in values)
        self._bump(key)
        return len(lst)

    def _lpush(self, key, *values) -> int:
        lst = self._lists.setdefault(key, [])
        # redis LPUSH prepends each value in turn, so the last argument ends up at
        # the head — mirror that ordering exactly.
        for value in values:
            lst.insert(0, str(value))
        self._bump(key)
        return len(lst)

    def _lrange(self, key, start, stop) -> list[str]:
        lst = self._lists.get(key, [])
        if stop == -1:
            return lst[start:]
        return lst[start : stop + 1]

    def _ltrim(self, key, start, stop) -> bool:
        # Keep only the inclusive ``[start, stop]`` slice, mirroring redis LTRIM
        # (``stop == -1`` means through the tail); a key trimmed empty is dropped.
        lst = self._lists.get(key)
        if lst is None:
            return True
        trimmed = lst[start:] if stop == -1 else lst[start : stop + 1]
        if trimmed:
            self._lists[key] = trimmed
        else:
            self._lists.pop(key, None)
        self._bump(key)
        return True

    def _expire(self, key, ttl, nx=False, gt=False) -> bool:
        # Mirror redis: EXPIRE on a missing key is a no-op returning False.
        if not any(key in store for store in (self._hashes, self._strings, self._streams, self._zsets, self._lists)):
            return False
        new_exp = self.time + ttl
        current = self._ttls.get(key)  # None == no expiry, treated as infinity
        if nx and current is not None:
            # NX sets a TTL only when the key currently has none.
            return False
        if gt and (current is None or new_exp <= current):
            # GT sets only when the new expiry is greater than the current one; a
            # key with no expiry counts as infinity, so GT never fires on it.
            return False
        self._ttls[key] = new_exp
        self._bump(key)
        return True

    def _delete(self, *keys) -> int:
        removed = 0
        for key in keys:
            present = any(
                key in store for store in (self._hashes, self._strings, self._streams, self._zsets, self._lists)
            )
            if present:
                self._drop(key)
                removed += 1
        return removed

    # -- direct (non-pipeline) async surface ---------------------------------

    async def xrange(self, key) -> list[tuple[str, dict]]:
        self._expired(key)
        return list(self._streams.get(key, []))

    async def xrevrange(self, key, count=None) -> list[tuple[str, dict]]:
        entries = list(reversed(self._streams.get(key, [])))
        return entries[:count] if count is not None else entries

    async def xread(self, streams: dict, block=None):
        result = []
        for key, cursor in streams.items():
            entries = [(eid, fields) for eid, fields in self._streams.get(key, []) if _entry_gt(eid, cursor)]
            if entries:
                result.append((key, entries))
        return result

    async def hgetall(self, key) -> dict:
        return self._hgetall(key)

    async def hget(self, key, field):
        return self._hget(key, field)

    async def hset(self, key, field=None, value=None, mapping=None) -> int:
        return self._hset(key, field, value, mapping)

    async def hdel(self, key, *fields) -> int:
        return self._hdel(key, *fields)

    async def get(self, key):
        return self._get(key)

    async def zadd(self, key, mapping, gt=False) -> int:
        return self._zadd(key, mapping, gt=gt)

    async def zrem(self, key, *members) -> int:
        return self._zrem(key, *members)

    async def zremrangebyscore(self, key, min_score, max_score) -> int:
        return self._zremrangebyscore(key, min_score, max_score)

    async def zrangebyscore(self, key, min_score, max_score) -> list[str]:
        return self._zrangebyscore(key, min_score, max_score)

    async def eval(self, script, numkeys, *keys_and_args):
        """Emulate the store's server-side Lua scripts, recognized by their marker
        comments. Signature-compatible with ``redis.eval(script, numkeys, *keys,
        *args)``.

        * ``interactions:pending-deadline-purge`` — RE-READS the deadline index at
          eval time (the atomicity the deterministic revive test relies on) before
          each per-group index delete; it leaves the count key untouched.
        * ``interactions:open-slot-reserve`` — purges stale open members, then
          admits (ZADD) the member only while the live count is below the limit,
          all in one call (the atomicity the concurrency-cap test relies on)."""
        keys = keys_and_args[:numkeys]
        args = keys_and_args[numkeys:]
        if "interactions:pending-deadline-purge" in script:
            pending_deadline_key, pending_key = keys[0], keys[1]
            now_ms, current = float(args[0]), str(args[1])
            expired = self._zrangebyscore(pending_deadline_key, 0, now_ms)
            purged = 0
            for group in expired:
                if group == current:
                    # The current add's group is skipped — it is becoming live.
                    continue
                self._zrem(pending_key, group)
                self._zrem(pending_deadline_key, group)
                purged += 1
            return purged
        if "interactions:open-slot-reserve" in script:
            open_key = keys[0]
            now_ms, limit, score, member = float(args[0]), int(args[1]), float(args[2]), str(args[3])
            self._zremrangebyscore(open_key, 0, now_ms)
            if len(self._zsets.get(open_key, {})) >= limit:
                return 0
            self._zadd(open_key, {member: score})
            return 1
        raise NotImplementedError("FakeRedis.eval only emulates the interactions Lua scripts")

    async def zcard(self, key) -> int:
        return len(self._zsets.get(key, {}))

    async def zrange(self, key, start, stop) -> list[str]:
        members = sorted(self._zsets.get(key, {}).items(), key=lambda kv: kv[1])
        ordered = [m for m, _ in members]
        if stop == -1:
            return ordered[start:]
        return ordered[start : stop + 1]

    async def blpop(self, keys, timeout=0):
        # The timeout is satisfied by EITHER clock: real elapsed time (tests
        # that await concurrent work) or the injectable ``self.time`` (tests
        # that drive expiry with ``advance()``), so both styles can time out
        # a blocked waiter.
        wall_start = asyncio.get_event_loop().time()
        fake_start = self.time
        while True:
            for key in keys:
                lst = self._lists.get(key)
                if lst:
                    return key, lst.pop(0)
            wall_elapsed = asyncio.get_event_loop().time() - wall_start
            if wall_elapsed >= timeout or (self.time - fake_start) >= timeout:
                return None
            await asyncio.sleep(0.005)

    async def lpush(self, key, *values) -> int:
        self._expired(key)
        return self._lpush(key, *values)

    async def lrange(self, key, start, stop) -> list[str]:
        self._expired(key)
        return self._lrange(key, start, stop)

    async def ltrim(self, key, start, stop) -> bool:
        self._expired(key)
        return self._ltrim(key, start, stop)

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self)


def _entry_gt(entry_id: str, cursor: str) -> bool:
    """Stream id ordering for ``xread``: entries strictly after ``cursor``."""

    def parse(eid: str) -> tuple[int, int]:
        a, _, b = eid.partition("-")
        return int(a), int(b or 0)

    return parse(entry_id) > parse(cursor)
