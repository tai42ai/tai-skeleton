"""The turn-engine cost bounds: the per-address token bucket with its paid
slow-down / silent-drop cooldown, the per-thread FIFO overflow, and the global
concurrency semaphore."""

from __future__ import annotations

import asyncio

import pytest

from tai42_skeleton.conversations import caps as caps_module
from tai42_skeleton.conversations.caps import AddressAdmission, ThreadQueueOverflowError, TurnCaps
from tai42_skeleton.conversations.settings import ConversationsSettings


@pytest.fixture
def small_settings(monkeypatch) -> ConversationsSettings:
    monkeypatch.setenv("CONVERSATIONS_PER_ADDRESS_TURNS_PER_HOUR", "2")
    monkeypatch.setenv("CONVERSATIONS_THREAD_QUEUE_DEPTH", "2")
    monkeypatch.setenv("CONVERSATIONS_MAX_CONCURRENT_TURNS", "1")
    return ConversationsSettings()


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


def test_token_bucket_admits_then_sheds_with_reply_then_silent(monkeypatch, small_settings):
    clock = _Clock()
    monkeypatch.setattr(caps_module.time, "monotonic", clock.monotonic)
    caps = TurnCaps(small_settings)

    # Capacity is one hour's worth of turns (2 here): the first two are admitted.
    assert caps.admit_address("+1") is AddressAdmission.ADMIT
    assert caps.admit_address("+1") is AddressAdmission.ADMIT
    # The next over-limit hit is paid ONE slow-down reply; later hits in the window drop.
    assert caps.admit_address("+1") is AddressAdmission.SHED_WITH_REPLY
    assert caps.admit_address("+1") is AddressAdmission.SHED_SILENT
    assert caps.admit_address("+1") is AddressAdmission.SHED_SILENT

    # A different address has its own independent bucket.
    assert caps.admit_address("+2") is AddressAdmission.ADMIT

    # After the cooldown window (one token's refill interval) a fresh slow-down reply is
    # paid again — one paid reply per window.
    clock.now += 3600.0 / small_settings.per_address_turns_per_hour + 0.01
    # That interval also refilled ~one token, so the address is admitted again first.
    assert caps.admit_address("+1") is AddressAdmission.ADMIT
    assert caps.admit_address("+1") is AddressAdmission.SHED_WITH_REPLY


def test_a_spent_bucket_survives_and_a_quiet_one_is_dropped(monkeypatch, small_settings):
    # The bucket map is keyed by a sender-chosen value, so it is evicted: a still-sending
    # address keeps its SPENT bucket (dropping it hands back unearned tokens); one quiet
    # for a full refill window is dropped.
    clock = _Clock()
    monkeypatch.setattr(caps_module.time, "monotonic", clock.monotonic)
    caps = TurnCaps(small_settings)
    assert caps.admit_address("+1") is AddressAdmission.ADMIT
    assert caps.admit_address("+1") is AddressAdmission.ADMIT
    assert caps.admit_address("+1") is AddressAdmission.SHED_WITH_REPLY
    assert len(caps._buckets) == 1

    # Inside the window (one token takes 1800s to refill at 2/hour), the spent bucket is
    # still there and still shedding.
    clock.now += 1000.0
    assert caps.admit_address("+1") is AddressAdmission.SHED_SILENT
    assert len(caps._buckets) == 1

    # Quiet for a full refill window: the entry is gone, and the address that DID keep
    # sending is unaffected by that eviction.
    clock.now += caps_module._BUCKET_IDLE_SECONDS + 1
    assert len(caps._buckets) == 0
    assert caps.admit_address("+1") is AddressAdmission.ADMIT


def test_the_bucket_map_is_bounded_under_address_churn(monkeypatch):
    # A flood of never-seen addresses cannot grow the map past its bound: each bucket is
    # live for a refill hour, so without the bound one worker holds one bucket per address
    # sent in that hour.
    monkeypatch.setenv("CONVERSATIONS_PER_ADDRESS_TURNS_PER_HOUR", "2")
    monkeypatch.setenv("CONVERSATIONS_ADDRESS_BUCKET_MAX_ENTRIES", "16")
    caps = TurnCaps(ConversationsSettings())
    for index in range(500):
        assert caps.admit_address(f"+{index}") is AddressAdmission.ADMIT
    assert len(caps._buckets) == 16


def test_thread_queue_overflow_is_loud(small_settings):
    caps = TurnCaps(small_settings)
    caps.reserve_thread_slot("t")  # 1
    caps.reserve_thread_slot("t")  # 2 (depth == 2)
    with pytest.raises(ThreadQueueOverflowError):
        caps.reserve_thread_slot("t")
    # A different thread is independent.
    caps.reserve_thread_slot("other")


def test_release_thread_slot_is_the_abort_mirror(small_settings):
    # An aborted accept must give its reservation back: no ``run_reserved`` will consume
    # it, so the FIFO would otherwise leak a slot per abort.
    caps = TurnCaps(small_settings)
    caps.reserve_thread_slot("t")
    caps.reserve_thread_slot("t")  # depth == 2, the thread is full
    caps.release_thread_slot("t")
    caps.reserve_thread_slot("t")  # the abandoned reservation came back
    with pytest.raises(ThreadQueueOverflowError):
        caps.reserve_thread_slot("t")


async def test_run_reserved_releases_the_slot(small_settings):
    caps = TurnCaps(small_settings)
    caps.reserve_thread_slot("t")
    async with caps.run_reserved("t"):
        pass
    # The slot is released, so the thread can be reserved to its full depth again.
    caps.reserve_thread_slot("t")
    caps.reserve_thread_slot("t")


async def test_global_semaphore_bounds_concurrency(small_settings):
    caps = TurnCaps(small_settings)  # max_concurrent_turns == 1
    entered = asyncio.Event()
    release = asyncio.Event()
    second_entered = asyncio.Event()

    async def first():
        caps.reserve_thread_slot("a")
        async with caps.run_reserved("a"):
            entered.set()
            await release.wait()

    async def second():
        await entered.wait()
        caps.reserve_thread_slot("b")
        async with caps.run_reserved("b"):
            second_entered.set()

    t1 = asyncio.create_task(first())
    t2 = asyncio.create_task(second())
    await entered.wait()
    # The single global slot is held by ``first``; ``second`` cannot enter yet.
    await asyncio.sleep(0.02)
    assert not second_entered.is_set()
    release.set()
    await asyncio.gather(t1, t2)
    assert second_entered.is_set()


async def test_same_thread_runs_serialize_in_arrival_order(monkeypatch):
    # Two turns for the SAME thread never overlap; the global ceiling is raised so the
    # serialization under test is the per-thread lock, not the semaphore.
    monkeypatch.setenv("CONVERSATIONS_MAX_CONCURRENT_TURNS", "8")
    caps = TurnCaps(ConversationsSettings())
    order: list[str] = []
    first_in = asyncio.Event()
    let_first_finish = asyncio.Event()

    async def first():
        caps.reserve_thread_slot("t")
        async with caps.run_reserved("t"):
            order.append("first-enter")
            first_in.set()
            await let_first_finish.wait()
            order.append("first-exit")

    async def second():
        await first_in.wait()
        caps.reserve_thread_slot("t")
        async with caps.run_reserved("t"):
            order.append("second-enter")

    t1 = asyncio.create_task(first())
    t2 = asyncio.create_task(second())
    await first_in.wait()
    await asyncio.sleep(0.02)
    # ``second`` is queued behind the per-thread lock and has not entered.
    assert order == ["first-enter"]
    let_first_finish.set()
    await asyncio.gather(t1, t2)
    assert order == ["first-enter", "first-exit", "second-enter"]


async def test_distinct_threads_run_concurrently(monkeypatch):
    # Serialization is PER THREAD only: two different threads run at once under a global
    # ceiling that admits both, so one slow thread never blocks another conversation.
    monkeypatch.setenv("CONVERSATIONS_MAX_CONCURRENT_TURNS", "8")
    caps = TurnCaps(ConversationsSettings())
    both_in = asyncio.Event()
    count = 0

    async def run(thread: str):
        nonlocal count
        caps.reserve_thread_slot(thread)
        async with caps.run_reserved(thread):
            count += 1
            if count == 2:
                both_in.set()
            await both_in.wait()

    await asyncio.wait_for(asyncio.gather(run("a"), run("b")), timeout=1.0)
    assert count == 2


@pytest.fixture
def fresh_caps_singleton():
    """A caps singleton built inside the test and dropped after it, so the process-wide
    instance a reload now KEEPS never leaks between tests."""
    caps_module._CAPS_CACHE.clear()
    try:
        yield
    finally:
        caps_module._CAPS_CACHE.clear()


async def test_a_settings_reload_keeps_a_live_thread_serialized(monkeypatch, fresh_caps_singleton):
    # The reload hook must not hand the next message a fresh FIFO: two turns for one
    # thread would then run at once against a single agent checkpoint.
    monkeypatch.setenv("CONVERSATIONS_MAX_CONCURRENT_TURNS", "4")
    caps = caps_module.get_turn_caps()
    inside: list[str] = []
    first_in = asyncio.Event()
    let_first_finish = asyncio.Event()

    async def first():
        caps.reserve_thread_slot("t")
        async with caps.run_reserved("t"):
            inside.append("A")
            first_in.set()
            await let_first_finish.wait()
            inside.remove("A")

    async def second():
        reloaded = caps_module.get_turn_caps()
        reloaded.reserve_thread_slot("t")
        async with reloaded.run_reserved("t"):
            inside.append("B")

    t1 = asyncio.create_task(first())
    await first_in.wait()
    monkeypatch.setenv("CONVERSATIONS_MAX_CONCURRENT_TURNS", "8")
    caps_module._reset_turn_caps()

    t2 = asyncio.create_task(second())
    await asyncio.sleep(0.02)
    assert inside == ["A"]
    # Same instance, new bounds: the reload took effect without dropping live state.
    assert caps_module.get_turn_caps() is caps
    assert caps.settings.max_concurrent_turns == 8

    let_first_finish.set()
    await asyncio.gather(t1, t2)
    assert inside == ["B"]


async def test_a_reload_that_lowers_the_ceiling_counts_the_turns_already_running(monkeypatch):
    monkeypatch.setenv("CONVERSATIONS_MAX_CONCURRENT_TURNS", "2")
    caps = TurnCaps(ConversationsSettings())
    running = 0
    peak = 0
    let_finish = asyncio.Event()

    async def run(thread: str):
        nonlocal running, peak
        caps.reserve_thread_slot(thread)
        async with caps.run_reserved(thread):
            running += 1
            peak = max(peak, running)
            await let_finish.wait()
            running -= 1

    first = [asyncio.create_task(run(t)) for t in ("a", "b")]
    await asyncio.sleep(0.02)
    assert running == 2

    monkeypatch.setenv("CONVERSATIONS_MAX_CONCURRENT_TURNS", "1")
    caps.reconfigure(ConversationsSettings())
    third = asyncio.create_task(run("c"))
    await asyncio.sleep(0.02)
    # The two already running count against the lowered ceiling, so the third waits.
    assert running == 2

    let_finish.set()
    await asyncio.gather(*first, third)
    assert peak == 2


def test_reconfigure_re_rates_a_live_bucket(monkeypatch):
    # Lowering the per-hour rate and reloading must bind the address that keeps sending —
    # the very one the reduction was made for — not just addresses first seen after it.
    clock = _Clock()
    monkeypatch.setattr(caps_module.time, "monotonic", clock.monotonic)
    monkeypatch.setenv("CONVERSATIONS_PER_ADDRESS_TURNS_PER_HOUR", "1000")
    caps = TurnCaps(ConversationsSettings())
    for _ in range(40):
        assert caps.admit_address("abuser") is AddressAdmission.ADMIT

    monkeypatch.setenv("CONVERSATIONS_PER_ADDRESS_TURNS_PER_HOUR", "5")
    caps.reconfigure(ConversationsSettings())

    bucket = caps._buckets["abuser"]
    assert bucket.capacity == 5.0
    assert bucket.refill_per_second == 5 / 3600.0
    # Tokens (~960 under the old rate) are clamped DOWN to the new capacity; none handed back.
    assert bucket.tokens == 5.0
    for _ in range(5):
        assert caps.admit_address("abuser") is AddressAdmission.ADMIT
    assert caps.admit_address("abuser") is AddressAdmission.SHED_WITH_REPLY
