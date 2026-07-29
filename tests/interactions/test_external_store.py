"""Store behavior for the external-interactions additions: the callback ticket
key + TTL, the open-index ZSET (ZADD on add for all formats, ZREM on answer, the
unconditional stale purge), ``count_open``, ``resolve_ticket``, the ticket TTL
refresh on claim, and ``prune_pending`` (decrement / at-zero cleanup / the
answered- and missing-state no-ops / the WATCH-retry path / the removed event).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from redis.exceptions import WatchError
from tai42_contract.interactions import (
    AnswerFormat,
    InteractionRequest,
    InteractionResponse,
)

from tai42_skeleton.interactions import InteractionStore


def _request(iid: str, gid: str, store: InteractionStore, budget: int = 60) -> InteractionRequest:
    now = datetime.now(UTC)
    return InteractionRequest(
        interaction_id=iid,
        group_id=gid,
        question="proceed?",
        answer_format=AnswerFormat.TEXT,
        reply_to=store.reply_key(iid),
        created_at=now,
        timeout_at=now + timedelta(seconds=budget),
    )


def _response(iid: str) -> InteractionResponse:
    return InteractionResponse(interaction_id=iid, answer="ok", answered_by="tester", answered_at=datetime.now(UTC))


async def _removed_events(fake_redis, store: InteractionStore) -> list[str]:
    events = await fake_redis.xrange(store.events_key)
    return [f["interaction_id"] for _id, f in events if f.get("type") == "interaction.removed"]


# -- ticket + open index -----------------------------------------------------


async def test_add_writes_ticket_open_member_and_ttl(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100, ticket="tk", ticket_ttl=60)

    assert await store.resolve_ticket(fake_redis, "tk") == "i1"
    assert "i1" in fake_redis._zsets[store.open_key]
    # The ticket carries a TTL ≈ the budget: still resolvable before it, gone after.
    fake_redis.advance(59)
    assert await store.resolve_ticket(fake_redis, "tk") == "i1"
    fake_redis.advance(2)
    assert await store.resolve_ticket(fake_redis, "tk") is None


async def test_add_ticket_requires_ttl(fake_redis):
    store = InteractionStore("t:")
    with pytest.raises(ValueError, match="ticket_ttl"):
        await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100, ticket="tk")


async def test_open_member_added_for_all_formats_without_ticket(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)
    assert await store.count_open(fake_redis) == 1


async def test_add_purges_stale_open_members_unconditionally(fake_redis):
    store = InteractionStore("t:")
    # A SIGKILLed waiter's member with a long-past deadline (score 1ms).
    fake_redis._zadd(store.open_key, {"stale": 1.0})
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)
    assert "stale" not in fake_redis._zsets[store.open_key]
    assert "i1" in fake_redis._zsets[store.open_key]


async def test_count_open_purges_then_counts(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)
    await store.add(fake_redis, _request("i2", "g", store), idle_ttl=100)
    fake_redis._zadd(store.open_key, {"stale": 1.0})
    assert await store.count_open(fake_redis) == 2


async def test_resolve_ticket_unknown_is_none(fake_redis):
    store = InteractionStore("t:")
    assert await store.resolve_ticket(fake_redis, "nope") is None


async def test_fake_set_without_ex_clears_ttl(fake_redis):
    # Fake fidelity: SET without EX discards a prior TTL (mirrors real Redis, no
    # KEEPTTL), so the ticket/state TTL semantics the store relies on are honest.
    fake_redis._set("k", "v1", ex=10)
    fake_redis._set("k", "v2")  # no ex → TTL cleared
    fake_redis.advance(100)
    assert await fake_redis.get("k") == "v2"


# -- record_answer additions -------------------------------------------------


async def test_record_answer_removes_open_and_refreshes_ticket(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100, ticket="tk", ticket_ttl=60)

    claimed = await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60, ticket="tk", ticket_ttl=200)
    assert claimed is True
    # Open member gone; ticket still resolves (never deleted) with a refreshed TTL.
    assert await store.count_open(fake_redis) == 0
    fake_redis.advance(120)  # past the original 60s ticket TTL
    assert await store.resolve_ticket(fake_redis, "tk") == "i1"
    fake_redis.advance(120)  # past the refreshed 200s TTL
    assert await store.resolve_ticket(fake_redis, "tk") is None


async def test_record_answer_refreshes_state_ttl_with_ticket(fake_redis):
    # The idempotent-200 path resolves the ticket AND reads the state; the claim
    # must refresh the state TTL to the ticket window so a late retry (after the
    # state's original deadline) still finds the answered state, not a 404.
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100, ticket="tk", ticket_ttl=60)
    await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60, ticket="tk", ticket_ttl=200)

    fake_redis.advance(150)  # past the state's original 100s TTL
    state = await store.get_state(fake_redis, "i1")
    assert state is not None  # refreshed to 200s alongside the ticket
    assert state.status == "answered"
    assert await store.resolve_ticket(fake_redis, "tk") == "i1"


async def test_record_answer_at_zero_cleans_count_and_pending(fake_redis):
    # Answering the group's last open question deletes the count key and removes
    # the group from the pending index (the at-zero branch, symmetric to prune).
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)
    assert "g" in fake_redis._zsets[store.pending_key]

    assert await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60) is True
    assert store.count_key("g") not in fake_redis._strings
    assert "g" not in fake_redis._zsets.get(store.pending_key, {})


async def test_record_answer_emits_answered_event(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)
    await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60)

    events = await fake_redis.xrange(store.events_key)
    answered = [f for _id, f in events if f.get("type") == "interaction.answered"]
    assert len(answered) == 1
    assert answered[0]["interaction_id"] == "i1"
    assert answered[0]["group_id"] == "g"


async def test_record_answer_ticket_requires_ttl(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100, ticket="tk", ticket_ttl=60)
    with pytest.raises(ValueError, match="ticket_ttl"):
        await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60, ticket="tk")


# -- prune_pending -----------------------------------------------------------


async def test_prune_decrements_then_cleans_up_last(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)
    await store.add(fake_redis, _request("i2", "g", store), idle_ttl=100)

    assert await store.prune_pending(fake_redis, "i1", "g") is True
    assert fake_redis._strings[store.count_key("g")] == "1"
    assert "g" in fake_redis._zsets[store.pending_key]
    assert "i1" not in fake_redis._zsets[store.open_key]
    assert await _removed_events(fake_redis, store) == ["i1"]

    assert await store.prune_pending(fake_redis, "i2", "g") is True
    assert store.count_key("g") not in fake_redis._strings
    assert "g" not in fake_redis._zsets.get(store.pending_key, {})
    assert await store.count_open(fake_redis) == 0


async def test_prune_noop_on_answered(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)
    await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60)

    assert await store.prune_pending(fake_redis, "i1", "g") is False
    # No removed event was emitted for an answered interaction.
    assert await _removed_events(fake_redis, store) == []


async def test_prune_noop_on_missing(fake_redis):
    store = InteractionStore("t:")
    assert await store.prune_pending(fake_redis, "ghost", "g") is False


async def test_record_answer_count_watch_detects_concurrent_add(fake_redis):
    # A genuine conflict: a concurrent add() to the group INCRs the count key
    # between record_answer's WATCH and its EXEC. The fake enforces WATCH like
    # real Redis, so this aborts+retries ONLY because the store watches count_key
    # — a store that dropped that watch would not retry here.
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)

    real_pipeline = fake_redis.pipeline
    state = {"fired": False, "watch_calls": 0}

    class _ConcurrentAddOnce:
        def __init__(self, inner):
            self._inner = inner

        async def __aenter__(self):
            await self._inner.__aenter__()
            return self

        async def __aexit__(self, *exc):
            return await self._inner.__aexit__(*exc)

        async def watch(self, *keys):
            state["watch_calls"] += 1
            result = await self._inner.watch(*keys)
            if not state["fired"]:
                state["fired"] = True
                # Concurrent sibling add: INCR the group count AFTER the snapshot,
                # so EXEC sees the count key changed and raises WatchError.
                fake_redis._incr(store.count_key("g"))
            return result

        def __getattr__(self, name):
            return getattr(self._inner, name)

    fake_redis.pipeline = lambda: _ConcurrentAddOnce(real_pipeline())

    assert await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60) is True
    # A retry MUST have happened — proving the store watches count_key. A store
    # that dropped that watch would EXEC on the first pass (watch_calls == 1).
    assert state["watch_calls"] >= 2
    answered = await store.get_state(fake_redis, "i1")
    assert answered is not None
    assert answered.status == "answered"


# -- count_key idle_ttl basis (sibling survives a decrement-to-zero) ----------


async def test_sibling_pending_survives_at_zero_via_record_answer(fake_redis):
    # The count key rides ``idle_ttl`` (the state basis), not the shorter question
    # deadline. A sibling still-open question therefore survives a decrement even
    # after the answered question's own deadline has passed — the old deadline-TTL
    # basis would have expired the count key, read remaining as -1, and dropped the
    # group from the pending index while the sibling was still pending.
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store, budget=60), idle_ttl=100000)
    await store.add(fake_redis, _request("i2", "g", store, budget=60), idle_ttl=100000)
    fake_redis.advance(120)  # past the 60s deadline, far under idle_ttl

    assert await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60) is True
    # The sibling keeps the group alive: count decremented to 1, group retained.
    assert fake_redis._strings[store.count_key("g")] == "1"
    assert "g" in fake_redis._zsets[store.pending_key]


async def test_sibling_pending_survives_at_zero_via_prune_pending(fake_redis):
    # The BYTE-IDENTICAL hole in ``prune_pending``: same idle_ttl basis, same
    # survival guarantee for a sibling still open past the question deadline.
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store, budget=60), idle_ttl=100000)
    await store.add(fake_redis, _request("i2", "g", store, budget=60), idle_ttl=100000)
    fake_redis.advance(120)

    assert await store.prune_pending(fake_redis, "i1", "g") is True
    assert fake_redis._strings[store.count_key("g")] == "1"
    assert "g" in fake_redis._zsets[store.pending_key]
    assert "i2" in fake_redis._zsets[store.open_key]


async def test_group_revive_after_purge_keeps_consistent_count(fake_redis):
    # F.1 desync: a group's waiter dies, an unrelated add purges the group from the
    # pending indexes, then the group revives with a new question while the stale
    # state survives. Because the purge no longer tears down the count key (it rides
    # idle_ttl), the revive's INCR builds on the live count, so pruning the stale
    # sibling leaves the group intact while the new question is still open. The old
    # purge-DELs-count behavior desynced: the revive re-seeded count to 1 and the
    # stale prune then read remaining 0 and dropped the live question.
    store = InteractionStore("t:")
    # q1: a past-deadline question whose waiter died without running cleanup.
    await store.add(fake_redis, _request("i1", "g", store, budget=-30), idle_ttl=100000)
    # An unrelated add fires the atomic purge; g's deadline is past, so g is dropped
    # from both pending indexes. Its count key is left to idle_ttl.
    await store.add(fake_redis, _request("other", "g_other", store, budget=60), idle_ttl=100000)
    assert "g" not in fake_redis._zsets.get(store.pending_key, {})
    assert store.count_key("g") in fake_redis._strings

    # q2 revives group g while q1's stale state still survives.
    await store.add(fake_redis, _request("i2", "g", store, budget=60), idle_ttl=100000)
    assert "g" in fake_redis._zsets[store.pending_key]

    # Pruning the stale q1 must NOT drop g — q2 is still open.
    assert await store.prune_pending(fake_redis, "i1", "g") is True
    assert "g" in fake_redis._zsets[store.pending_key]
    assert "i2" in fake_redis._zsets[store.open_key]
    assert fake_redis._strings[store.count_key("g")] == "1"


# -- atomic concurrency cap (reserve_open_slot) ------------------------------


async def test_reserve_open_slot_atomic_cap_refuses_burst(fake_redis):
    # A concurrent burst of reservations past the cap: the atomic ZCARD+ZADD guard
    # admits EXACTLY ``limit`` callers and refuses the N+1th (and beyond) — no
    # check-then-act overshoot the way a separate count-then-add pair would allow.
    store = InteractionStore("t:")
    limit = 3
    reqs = [_request(f"i{n}", f"g{n}", store) for n in range(limit + 2)]
    results = await asyncio.gather(*(store.reserve_open_slot(fake_redis, req, limit) for req in reqs))

    assert sum(results) == limit  # exactly `limit` True, the rest refused
    assert await store.count_open(fake_redis) == limit


async def test_prune_retries_on_watch_conflict_and_returns_false(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)

    real_pipeline = fake_redis.pipeline
    state = {"raised": False}

    class _AnswerThenConflict:
        def __init__(self, inner):
            self._inner = inner

        async def __aenter__(self):
            await self._inner.__aenter__()
            return self

        async def __aexit__(self, *exc):
            return await self._inner.__aexit__(*exc)

        async def watch(self, *keys):
            if not state["raised"]:
                state["raised"] = True
                # An answer commits between the status read and EXEC: flip the
                # state to answered, then fire the WatchError the real WATCH would.
                fake_redis._hashes[store.state_key("i1")]["status"] = "answered"
                raise WatchError()
            return await self._inner.watch(*keys)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    fake_redis.pipeline = lambda: _AnswerThenConflict(real_pipeline())

    # The retry reads ``answered`` and returns False cleanly — no WatchError escapes.
    assert await store.prune_pending(fake_redis, "i1", "g") is False
    assert state["raised"] is True
    assert await _removed_events(fake_redis, store) == []
