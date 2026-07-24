"""An in-memory fake of the redis surface the answer/record store uses.

Covers the HASH + STRING commands the record store calls, the LIST commands the send
ledger calls, the ZSET commands the per-status record index uses, plus the record Lua
scripts (dispatched by marker comment, re-implemented here). Single-threaded async, so
each faked ``eval`` runs atomically.

``ttl_ms`` records the expiry each command applied, so a test can assert the retention
TTL lands only on a terminal record.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class FakeRecordRedis:
    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._lists: dict[str, list[str]] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        # The live expiry (in milliseconds) each key carries, so a test can assert the
        # retention TTL lands only on a terminal transition.
        self.ttl_ms: dict[str, int] = {}

    def seed_hash(self, key: str, fields: dict[str, str]) -> None:
        """Plant a raw hash under ``key`` — the seam a test uses to put a row no store
        write could produce (a corrupt or foreign row) in the record keyspace."""
        self._hashes[key] = dict(fields)

    # -- sorted-set commands (the per-status record index) -------------------
    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        target = self._zsets.setdefault(key, {})
        added = sum(1 for member in mapping if member not in target)
        target.update(mapping)
        return added

    async def zrem(self, key: str, *members: str) -> int:
        target = self._zsets.get(key, {})
        return sum(1 for member in members if target.pop(member, None) is not None)

    async def zrange(self, key: str, start: int, end: int) -> list[str]:
        ordered = [m for m, _ in sorted(self._zsets.get(key, {}).items(), key=lambda item: (item[1], item[0]))]
        stop = len(ordered) if end == -1 else end + 1
        return ordered[start:stop]

    async def zremrangebyscore(self, key: str, minimum: str | float, maximum: str | float) -> int:
        low, high = float(minimum), float(maximum)
        target = self._zsets.get(key, {})
        doomed = [member for member, score in target.items() if low <= score <= high]
        for member in doomed:
            del target[member]
        return len(doomed)

    # -- list commands (the channel send ledger) -----------------------------
    async def rpush(self, key: str, *values: str) -> int:
        target = self._lists.setdefault(key, [])
        target.extend(values)
        return len(target)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        entries = self._lists.get(key, [])
        stop = len(entries) if end == -1 else end + 1
        return entries[start:stop]

    async def expire(self, key: str, seconds: int) -> bool:
        self.ttl_ms[key] = seconds * 1000
        return True

    async def delete(self, key: str) -> int:
        self.ttl_ms.pop(key, None)
        held = [store.pop(key, None) for store in (self._strings, self._hashes, self._lists)]
        return 1 if any(value is not None for value in held) else 0

    # -- string commands -----------------------------------------------------
    async def get(self, key: str) -> str | None:
        return self._strings.get(key)

    async def set(self, key: str, value: str, ex: int | None = None, nx: bool = False) -> bool | None:
        if nx and key in self._strings:
            return None
        self._strings[key] = value
        if ex is not None:
            self.ttl_ms[key] = ex * 1000
        return True

    # -- hash commands -------------------------------------------------------
    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    async def hget(self, key: str, field: str) -> str | None:
        return self._hashes.get(key, {}).get(field)

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        target = self._hashes.setdefault(key, {})
        new = int(target.get(field, "0")) + amount
        target[field] = str(new)
        return new

    def scan_iter(self, match: str) -> AsyncIterator[str]:
        prefix = match.rstrip("*")

        async def _iter() -> AsyncIterator[str]:
            for key in list(self._hashes):
                if key.startswith(prefix):
                    yield key

        return _iter()

    # -- Lua eval (re-implements the record scripts by marker) ---------------
    @staticmethod
    def _foreign_lease(hashed: dict[str, str] | None, *, now: str, token: str) -> bool:
        """Whether a DIFFERENT worker holds a live delivery lease on the row."""
        claim = (hashed or {}).get("claim", "")
        if not claim:
            return False
        ctoken, cexp = claim.split(":", 1)
        return ctoken != token and float(cexp) > float(now)

    def _reindex(self, index_keys: list[str], member: str, score: str) -> None:
        """Move ``member`` into the target index (the LAST key) and out of every other, as
        the scripts' shared reindex step does."""
        for index_key in index_keys[:-1]:
            self._zsets.get(index_key, {}).pop(member, None)
        self._zsets.setdefault(index_keys[-1], {})[member] = float(score)

    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        keys = [str(k) for k in keys_and_args[:numkeys]]
        key = keys[0]
        indexes = keys[1:]
        argv = [str(a) for a in keys_and_args[numkeys:]]
        if "conversations:dedupe:claim" in script:
            existing = self._strings.get(key)
            if existing is not None:
                return existing
            self._strings[key] = argv[0]
            return argv[0]
        h = self._hashes.get(key)
        status = h.get("delivery_status") if h else None
        if "conversations:record:create" in script:
            self._hashes[key] = {
                "data": argv[0],
                "delivery_status": argv[1],
                "outbound_ids": argv[2],
                "attempts": argv[3],
                "claim": "",
                "grace_deadline": "",
                "updated_at": argv[4],
                "intake_claim": argv[6],
            }
            if argv[5]:
                self.ttl_ms[key] = int(argv[5])
            self._reindex(indexes, argv[7], argv[8])
            return 1
        if "conversations:record:complete_turn" in script:
            if status is None:
                return -1
            if status != "accepted":
                return 0
            h.update(data=argv[0], delivery_status="pending_delivery", updated_at=argv[1], intake_claim="")  # type: ignore[union-attr]
            self._reindex(indexes, argv[2], argv[3])
            return 1
        if "conversations:record:intake_claim" in script:
            now, lease, token = float(argv[0]), float(argv[1]), argv[2]
            if status is None:
                return -1
            if status != "accepted":
                return -2
            claim = h.get("intake_claim", "") if h else ""
            if claim:
                ctoken, cexp = claim.split(":", 1)
                if ctoken != token and float(cexp) > now:
                    return 0
            h["intake_claim"] = f"{token}:{now + lease}"  # type: ignore[index]
            return 1
        if "conversations:record:claim" in script:
            now, lease, token = float(argv[0]), float(argv[1]), argv[2]
            if status is None:
                return -1
            if status == "accepted":
                return -2
            if status != "pending_delivery":
                return 0
            claim = h.get("claim", "") if h else ""
            if claim:
                ctoken, cexp = claim.split(":", 1)
                if ctoken != token and float(cexp) > now:
                    return 0
            h["claim"] = f"{token}:{now + lease}"  # type: ignore[index]
            return 1
        if "conversations:record:provisional" in script:
            if status is None:
                return -1
            if status in ("delivered", "failed", "shed"):
                return 0
            if self._foreign_lease(h, now=argv[3], token=argv[4]):
                return -3
            h.update(  # type: ignore[union-attr]
                delivery_status="provisional",
                outbound_ids=argv[0],
                attempts=argv[1],
                grace_deadline=argv[2],
                updated_at=argv[3],
                claim="",
            )
            self._reindex(indexes, argv[5], argv[6])
            return 1
        if "conversations:record:delivered" in script:
            if status is None:
                return -1
            if status in ("failed", "shed"):
                return -2
            if status == "delivered":
                return 0
            if self._foreign_lease(h, now=argv[2], token=argv[4]):
                return -3
            h.update(  # type: ignore[union-attr]
                delivery_status="delivered",
                outbound_ids=argv[0],
                attempts=argv[1],
                updated_at=argv[2],
                claim="",
                grace_deadline="",
            )
            self.ttl_ms[key] = int(argv[3])
            self._reindex(indexes, argv[5], argv[6])
            return 1
        if "conversations:record:failed" in script:
            if status is None:
                return -1
            if status in ("delivered", "shed", "provisional"):
                return -2
            if status == "failed":
                return 0
            if self._foreign_lease(h, now=argv[1], token=argv[3]):
                return -3
            h.update(delivery_status="failed", attempts=argv[0], updated_at=argv[1], claim="", grace_deadline="")  # type: ignore[union-attr]
            self.ttl_ms[key] = int(argv[2])
            self._reindex(indexes, argv[4], argv[5])
            return 1
        if "conversations:record:receipt" in script:
            target = argv[0]
            if status is None:
                return -1
            if status == target:
                return 0
            if status in ("delivered", "failed", "shed"):
                return -2
            if status != "provisional":
                return -3
            h.update(delivery_status=target, updated_at=argv[1], claim="", grace_deadline="")  # type: ignore[union-attr]
            self.ttl_ms[key] = int(argv[2])
            self._reindex(indexes, argv[3], argv[4])
            return 1
        if "conversations:record:delete" in script:
            removed = await self.delete(key)
            for index_key in indexes:
                self._zsets.get(index_key, {}).pop(argv[0], None)
            return removed
        raise NotImplementedError(f"FakeRecordRedis.eval: unknown script {script[:60]!r}")


def make_record_client_ctx(fake: FakeRecordRedis):
    @asynccontextmanager
    async def _ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield fake

    return _ctx
