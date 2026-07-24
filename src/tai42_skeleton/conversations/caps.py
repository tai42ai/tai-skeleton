"""The turn-engine cost bounds — three per-worker caps:

* a GLOBAL in-flight-turn ceiling of ``max_concurrent_turns``;
* a token bucket per ``bucket_key`` — an opaque key its door composes — refilling at
  ``per_address_turns_per_hour``, whose over-limit verdict pays ONE slow-down reply per
  refill window and then drops silently. The buckets live in a cache bounded on both size
  and idle time;
* a per-thread FIFO of depth ``thread_queue_depth``, serialized in arrival order, refusing
  LOUDLY rather than growing an unbounded backlog.

The caps are a per-worker singleton. A settings reload is applied to that ONE instance, so
a turn in flight and one accepted after the reload always meet in the same FIFO and under
the same ceiling.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum, auto
from threading import RLock

from cachetools import LRUCache, TTLCache, cached
from tai42_kit.settings import register_settings_reset

from tai42_skeleton.conversations.settings import ConversationsSettings
from tai42_skeleton.operations.errors import UnavailableError

logger = logging.getLogger(__name__)

# Seconds a per-address bucket is kept after its last message. A bucket refills fully in
# one hour at any configured rate, so one idle this long holds nothing a fresh one would not.
_BUCKET_IDLE_SECONDS = 3600.0


def _now() -> float:
    """The monotonic clock the caps read — one seam, so bucket refill and cache expiry
    always advance together."""
    return time.monotonic()


class ThreadQueueOverflowError(UnavailableError):
    """The per-thread FIFO is full — a loud, retriable 503 rather than an unbounded
    backlog."""


class AddressRateLimitedError(UnavailableError):
    """A bucket key is over its per-hour turn cap on a synchronous door: a loud, retriable
    refusal instead of the channel door's paid slow-down reply."""


class AddressAdmission(Enum):
    """The token bucket's verdict for one bucket key: run the turn, run none but owe one
    slow-down reply (first over-limit hit of a window), or run none and send nothing."""

    ADMIT = auto()
    SHED_WITH_REPLY = auto()
    SHED_SILENT = auto()


class _TokenBucket:
    """One bucket key's token bucket plus its slow-down-reply cooldown. Capacity is one
    hour's worth of turns; it refills continuously at the per-hour rate."""

    __slots__ = ("capacity", "cooldown_until", "refill_per_second", "tokens", "updated")

    def __init__(self, per_hour: int, now: float) -> None:
        self.capacity = float(per_hour)
        self.tokens = float(per_hour)
        self.refill_per_second = per_hour / 3600.0
        self.updated = now
        # Next moment a slow-down reply may be paid; in the past, so the FIRST over-limit
        # hit is always the paid one.
        self.cooldown_until = 0.0

    def _refill(self, now: float) -> None:
        elapsed = now - self.updated
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
            self.updated = now


class _ConcurrencyGate:
    """The global in-flight-turn ceiling, admitting waiters in arrival order. The limit is
    re-set in place rather than by rebuilding the gate, so a raised or lowered ceiling
    counts the turns already running against it instead of granting them a second one."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._in_flight = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    def set_limit(self, limit: int) -> None:
        self._limit = limit
        self._admit_waiting()

    @asynccontextmanager
    async def held(self) -> AsyncIterator[None]:
        await self._acquire()
        try:
            yield
        finally:
            self._release()

    async def _acquire(self) -> None:
        if not self._waiters and self._in_flight < self._limit:
            self._in_flight += 1
            return
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                # Admitted and cancelled together: hand the slot straight back.
                self._release()
            else:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(waiter)
            raise

    def _release(self) -> None:
        self._in_flight -= 1
        self._admit_waiting()

    def _admit_waiting(self) -> None:
        while self._waiters and self._in_flight < self._limit:
            waiter = self._waiters.popleft()
            if waiter.done():
                continue
            self._in_flight += 1
            waiter.set_result(None)


class TurnCaps:
    """The per-worker turn caps. One instance backs every turn on the worker."""

    def __init__(self, settings: ConversationsSettings) -> None:
        self.settings = settings
        self._gate = _ConcurrencyGate(settings.max_concurrent_turns)
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._thread_waiters: dict[str, int] = {}
        # Bounded on both axes: idle expiry drops a bucket indistinguishable from a fresh
        # one, and the size bound caps a flood of never-seen keys.
        self._buckets: TTLCache[str, _TokenBucket] = TTLCache(
            maxsize=settings.address_bucket_max_entries, ttl=_BUCKET_IDLE_SECONDS, timer=_now
        )

    def reconfigure(self, settings: ConversationsSettings) -> None:
        """Adopt a reloaded settings snapshot IN PLACE. The FIFO, its locks, the spent
        buckets and the in-flight ceiling are live state a running turn is held by, so they
        carry over; rebuilding them would run a second cohort of turns outside them."""
        if settings.address_bucket_max_entries != self.settings.address_bucket_max_entries:
            resized: TTLCache[str, _TokenBucket] = TTLCache(
                maxsize=settings.address_bucket_max_entries, ttl=_BUCKET_IDLE_SECONDS, timer=_now
            )
            resized.update(self._buckets)
            self._buckets = resized
        if settings.per_address_turns_per_hour != self.settings.per_address_turns_per_hour:
            # Re-rate the live buckets in place so a reload changes the limit an actively
            # sending key already sees; never hand back tokens (tokens are only ever clamped
            # down), and keep the refill clock so no in-flight refill is corrupted.
            new_rate = settings.per_address_turns_per_hour
            for bucket in self._buckets.values():
                bucket.capacity = float(new_rate)
                bucket.refill_per_second = new_rate / 3600.0
                bucket.tokens = min(bucket.tokens, float(new_rate))
        self.settings = settings
        self._gate.set_limit(settings.max_concurrent_turns)

    # -- the per-key token bucket --------------------------------------------

    def admit_address(self, bucket_key: str) -> AddressAdmission:
        """Consume one token for ``bucket_key`` and answer whether its turn is admitted,
        and if not whether this hit still owes a paid slow-down reply. The key is opaque
        here: each door composes one naming the party its cap holds accountable."""
        now = _now()
        bucket = self._buckets.get(bucket_key)
        if bucket is None:
            bucket = _TokenBucket(self.settings.per_address_turns_per_hour, now)
        bucket._refill(now)
        # Written back on EVERY hit, not just a miss: the write restarts the idle window,
        # so a key that keeps sending keeps its spent bucket.
        self._buckets[bucket_key] = bucket
        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return AddressAdmission.ADMIT
        if now >= bucket.cooldown_until:
            bucket.cooldown_until = now + (1.0 / bucket.refill_per_second)
            return AddressAdmission.SHED_WITH_REPLY
        return AddressAdmission.SHED_SILENT

    # -- per-thread FIFO + global semaphore ----------------------------------

    def reserve_thread_slot(self, thread_id: str) -> None:
        """Reserve a place in ``thread_id``'s FIFO, raising
        :class:`ThreadQueueOverflowError` at ``thread_queue_depth``. Synchronous, so an
        overflow is refused before any state is written for the message. Every reservation
        must be released by exactly one :meth:`run_reserved` or
        :meth:`release_thread_slot`."""
        waiting = self._thread_waiters.get(thread_id, 0)
        if waiting >= self.settings.thread_queue_depth:
            raise ThreadQueueOverflowError(
                f"conversation thread {thread_id!r} already has {waiting} turns queued "
                f"(limit {self.settings.thread_queue_depth}); retry once it drains"
            )
        self._thread_waiters[thread_id] = waiting + 1

    @asynccontextmanager
    async def run_reserved(self, thread_id: str):
        """Run a turn holding ``thread_id``'s FIFO lock and a global concurrency slot,
        releasing the reservation :meth:`reserve_thread_slot` took when the body exits."""
        lock = self._thread_locks.setdefault(thread_id, asyncio.Lock())
        try:
            # Per-thread lock FIRST (arrival order), global ceiling SECOND and inside it,
            # so a turn queued behind a busy thread holds no global slot while it waits.
            async with lock, self._gate.held():
                yield
        finally:
            self.release_thread_slot(thread_id)

    def release_thread_slot(self, thread_id: str) -> None:
        """Give back one reservation :meth:`reserve_thread_slot` took, so an aborted accept
        leaves the FIFO exactly as it found it."""
        remaining = self._thread_waiters.get(thread_id, 0) - 1
        if remaining <= 0:
            # Thread drained: drop its lock and counter so the maps do not grow one entry
            # per distinct thread forever. A lock still held is never dropped.
            self._thread_waiters.pop(thread_id, None)
            lock = self._thread_locks.get(thread_id)
            if lock is not None and not lock.locked():
                self._thread_locks.pop(thread_id, None)
        else:
            self._thread_waiters[thread_id] = remaining


_CAPS_KEY = "conversations_turn_caps_singleton"
_CAPS_CACHE: LRUCache = LRUCache(maxsize=1)
_CAPS_LOCK = RLock()
_CAPS_STALE = False


@cached(_CAPS_CACHE, key=lambda *args, **kwargs: _CAPS_KEY, lock=_CAPS_LOCK)
def _build_turn_caps() -> TurnCaps:
    return TurnCaps(ConversationsSettings())


def get_turn_caps() -> TurnCaps:
    """The per-worker turn caps, built from ``CONVERSATIONS_*`` config and held for the
    process. New bounds from a settings reload are applied to the SAME instance."""
    global _CAPS_STALE
    with _CAPS_LOCK:
        caps = _build_turn_caps()
        if _CAPS_STALE:
            caps.reconfigure(ConversationsSettings())
            _CAPS_STALE = False
        return caps


@register_settings_reset
def _reset_turn_caps() -> None:
    """Mark the caps for reconfiguration on the next read. The instance is NOT dropped: a
    replacement would carry an empty FIFO and a second ceiling, so every turn in flight
    would lose its serialization the moment config reloads."""
    global _CAPS_STALE
    with _CAPS_LOCK:
        _CAPS_STALE = True


__all__ = [
    "AddressAdmission",
    "AddressRateLimitedError",
    "ThreadQueueOverflowError",
    "TurnCaps",
    "get_turn_caps",
]
