"""The store-owned SSE backlog read: ``InteractionStore.backlog`` returns
the pending questions to replay — in group creation order then stream order —
AND owns the reconciliation side effects (phantom-group prune, abandoned
past-deadline prune, answered/missing skip). The returned sequence AND the
post-read store key state are both pinned
here, since the pruning is a store-state side effect invisible in the returned
list alone. A second test pins the batching: one pipeline per group, not an N+1
of per-question state reads.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tai_contract.interactions import (
    AnswerFormat,
    InteractionRequest,
    InteractionResponse,
)

from tai_skeleton.interactions import InteractionStore


def _dated(store: InteractionStore, iid: str, gid: str, created: datetime, timeout: datetime) -> InteractionRequest:
    return InteractionRequest(
        interaction_id=iid,
        group_id=gid,
        question="?",
        answer_format=AnswerFormat.TEXT,
        reply_to=store.reply_key(iid),
        created_at=created,
        timeout_at=timeout,
    )


async def _removed_ids(fake_redis, store: InteractionStore) -> list[str]:
    events = await fake_redis.xrange(store.events_key)
    return [f["interaction_id"] for _id, f in events if f.get("type") == "interaction.removed"]


async def test_backlog_returns_order_and_reconciles_state(fake_redis):
    store = InteractionStore("t:")
    now = datetime.now(UTC)
    # Group A: a live question, an abandoned one (past deadline, SIGKILLed waiter),
    # and an answered one — added live, dead, done so A's last add (done) is now.
    await store.add(fake_redis, _dated(store, "live", "A", now, now + timedelta(seconds=60)), idle_ttl=86400)
    await store.add(
        fake_redis,
        _dated(store, "dead", "A", now - timedelta(seconds=120), now - timedelta(seconds=60)),
        idle_ttl=86400,
    )
    await store.add(fake_redis, _dated(store, "done", "A", now, now + timedelta(seconds=60)), idle_ttl=86400)
    done_resp = InteractionResponse(interaction_id="done", answer="x", answered_by="t", answered_at=now)
    await store.record_answer(fake_redis, done_resp, "A", reply_ttl=60)
    # Group B: one live question, created later so it sorts after A.
    await store.add(
        fake_redis,
        _dated(store, "b1", "B", now + timedelta(seconds=5), now + timedelta(seconds=65)),
        idle_ttl=86400,
    )
    # Group C: a phantom — in both indexes but its stream vanished.
    fake_redis._zadd(store.pending_key, {"C": 1.0})
    fake_redis._zadd(store.pending_deadline_key, {"C": 1.0})

    backlog = await store.backlog(fake_redis)

    # Only the live questions surface, in group-creation then stream order.
    assert [r.interaction_id for r in backlog] == ["live", "b1"]

    # Phantom group C pruned from BOTH indexes.
    assert "C" not in fake_redis._zsets.get(store.pending_key, {})
    assert "C" not in fake_redis._zsets.get(store.pending_deadline_key, {})
    # Abandoned "dead" pruned (state gone, exactly one removed event, count decremented).
    assert await store.get_state(fake_redis, "dead") is None
    assert await _removed_ids(fake_redis, store) == ["dead"]
    assert fake_redis._strings[store.count_key("A")] == "1"  # only "live" remains open
    assert "A" in fake_redis._zsets[store.pending_key]
    # Answered "done" untouched — not pruned, no removed event.
    done_state = await store.get_state(fake_redis, "done")
    assert done_state is not None
    assert done_state.status == "answered"


async def test_backlog_batches_state_reads_one_pipeline_per_group(fake_redis):
    store = InteractionStore("t:")
    now = datetime.now(UTC)
    # One group with three open questions — the N+1 candidate.
    for i in range(3):
        await store.add(fake_redis, _dated(store, f"i{i}", "g", now, now + timedelta(seconds=60)), idle_ttl=86400)

    pipelines = {"count": 0}
    real_pipeline = fake_redis.pipeline

    def counting_pipeline():
        pipelines["count"] += 1
        return real_pipeline()

    fake_redis.pipeline = counting_pipeline

    backlog = await store.backlog(fake_redis)
    assert [r.interaction_id for r in backlog] == ["i0", "i1", "i2"]
    # Exactly ONE pipeline batches the group's three state reads — not three
    # separate HGETALL round trips (the N+1 the refactor removed).
    assert pipelines["count"] == 1
