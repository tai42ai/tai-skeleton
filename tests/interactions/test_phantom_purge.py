"""Phantom self-heal for the pending index: the parallel
``pending_deadline`` index plus the atomic Lua purge that drops a group whose
furthest question deadline has passed from ``pending_key`` /
``pending_deadline_key`` — driven by any unrelated ``add``. The purge leaves
``count_key`` to its ``idle_ttl`` (death and revival stay symmetric via the shared
TTL basis). Covers the SIGKILL phantom, the deterministic revive-during-purge
atomicity (re-read inside the script), the GT extend-only guarantee (a shorter
later deadline never shortens), the count-key ``idle_ttl`` refresh basis, and the
invariant that the purge never rescores ``pending_key`` (scored by each group's
most-recent question ``created_at``) so it stays in creation-timestamp order and
the reconnect backlog order is unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from tai42_contract.interactions import AnswerFormat, InteractionRequest

from tai42_skeleton.interactions import InteractionStore
from tai42_skeleton.interactions import store as store_module


def _req(iid: str, gid: str, store: InteractionStore, *, created_offset: float = 0, timeout_offset: float = 60):
    now = datetime.now(UTC)
    return InteractionRequest(
        interaction_id=iid,
        group_id=gid,
        question="?",
        answer_format=AnswerFormat.TEXT,
        reply_to=store.reply_key(iid),
        created_at=now + timedelta(seconds=created_offset),
        timeout_at=now + timedelta(seconds=timeout_offset),
    )


async def test_unrelated_add_purges_expired_phantom_group(fake_redis):
    store = InteractionStore("t:")
    # A SIGKILLed waiter's group: recorded with a deadline already in the past and
    # never cleaned up. Its own add skips itself in the purge, so it lingers in
    # pending / pending_deadline — a phantom.
    dead = _req("dead", "gdead", store, created_offset=-120, timeout_offset=-60)
    await store.add(fake_redis, dead, idle_ttl=100)
    assert "gdead" in fake_redis._zsets[store.pending_key]
    assert "gdead" in fake_redis._zsets[store.pending_deadline_key]
    assert store.count_key("gdead") in fake_redis._strings

    # An unrelated add to a DIFFERENT group triggers the atomic purge, which drops
    # the phantom from both indexes even though no cleanup code ever ran for it.
    await store.add(fake_redis, _req("live", "glive", store, timeout_offset=60), idle_ttl=100)

    assert "gdead" not in fake_redis._zsets.get(store.pending_key, {})
    assert "gdead" not in fake_redis._zsets.get(store.pending_deadline_key, {})
    # The count key is left to its ``idle_ttl`` (the purge no longer touches it):
    # a surviving state keeps a live count, so death/revival stay symmetric.
    assert store.count_key("gdead") in fake_redis._strings
    # The live group is untouched.
    assert "glive" in fake_redis._zsets[store.pending_key]
    assert "glive" in fake_redis._zsets[store.pending_deadline_key]


async def test_purge_atomically_spares_revived_group(fake_redis):
    """The purge CONTRACT spares a group revived between snapshot and delete: it
    re-reads the expired-deadline set at eval time rather than acting on an
    earlier snapshot, so a group that gained a future deadline in the interim is
    no longer in the expired set and survives. This asserts that contract as
    modeled by ``FakeRedis.eval``, which emulates the script in Python (re-reading
    the deadline index before each delete). The fake does not execute the real
    Lua; the real ``_PENDING_PURGE_LUA`` runs server-side, and Redis's single-
    threaded script execution is what makes that re-read atomic in production."""
    store = InteractionStore("t:")
    dead = _req("i1", "g", store, created_offset=-120, timeout_offset=-60)
    await store.add(fake_redis, dead, idle_ttl=100)

    now_ms = store_module._now_ms()
    # g's deadline is in the past: it is currently in the expired set the purge
    # would act on.
    expired_before = await fake_redis.zrangebyscore(store.pending_deadline_key, 0, now_ms)
    assert expired_before == ["g"]

    # Revive g BEFORE the delete step, exactly as a concurrent add() would: extend
    # its deadline (ZADD GT into the future), re-add it to the pending index, bump
    # its count. Its earlier position in the expired set is now stale.
    future_ms = store_module._now_ms() + 60_000
    await fake_redis.zadd(store.pending_deadline_key, {"g": future_ms}, gt=True)
    await fake_redis.zadd(store.pending_key, {"g": store_module._now_ms()})
    fake_redis._incr(store.count_key("g"))
    # The CURRENT expired set is empty — an eval-time re-read is what spares g.
    assert await fake_redis.zrangebyscore(store.pending_deadline_key, 0, store_module._now_ms()) == []

    # Run the purge with a non-matching skip group ("__none__") so the current-add
    # skip branch cannot be what spares g; only the eval-time re-read of the
    # deadline set can. The fake re-reads that set before deleting (emulating the
    # script), so g — no longer expired — is left intact.
    purged = await fake_redis.eval(
        store_module._PENDING_PURGE_LUA,
        2,
        store.pending_deadline_key,
        store.pending_key,
        store_module._now_ms(),
        "__none__",
    )
    assert purged == 0
    assert "g" in fake_redis._zsets[store.pending_key]
    assert "g" in fake_redis._zsets[store.pending_deadline_key]
    assert store.count_key("g") in fake_redis._strings


async def test_purge_skips_current_add_group(fake_redis):
    """The ``if group ~= current`` skip branch: a purge invoked for an add whose
    OWN group is g leaves g in both indexes and keeps its count, even when g's
    recorded deadline is already past. This add is about to make g live (its future
    deadline is written by the pipeline that follows the purge), so purging it on
    the stale past deadline would drop a group that is gaining a live question —
    invariant (b)."""
    store = InteractionStore("t:")
    # Seed g as an expired-deadline member of both indexes with a live count — the
    # residue a group carries when its recorded deadline has already passed.
    dead = _req("i1", "g", store, created_offset=-120, timeout_offset=-60)
    await store.add(fake_redis, dead, idle_ttl=100)
    assert await fake_redis.zrangebyscore(store.pending_deadline_key, 0, store_module._now_ms()) == ["g"]
    count_before = fake_redis._strings[store.count_key("g")]

    # Invoke the purge exactly as add() does for a NEW question in group g:
    # current == "g", so the skip branch must spare g despite its past deadline.
    purged = await fake_redis.eval(
        store_module._PENDING_PURGE_LUA,
        2,
        store.pending_deadline_key,
        store.pending_key,
        store_module._now_ms(),
        "g",
    )
    assert purged == 0
    assert "g" in fake_redis._zsets[store.pending_key]
    assert "g" in fake_redis._zsets[store.pending_deadline_key]
    assert fake_redis._strings[store.count_key("g")] == count_before


async def test_gt_keeps_longer_deadline_on_shorter_later_add(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _req("i1", "g", store, timeout_offset=600), idle_ttl=100)
    long_score = fake_redis._zsets[store.pending_deadline_key]["g"]

    # A later question with a SHORTER deadline must not shorten the deadline index
    # score (invariant b: never drop a live group early). The count key no longer
    # tracks the deadline at all — it rides ``idle_ttl`` (asserted separately).
    await store.add(fake_redis, _req("i2", "g", store, timeout_offset=60), idle_ttl=100)
    assert fake_redis._zsets[store.pending_deadline_key]["g"] == long_score


async def test_count_key_ttl_tracks_idle_ttl_not_deadline(fake_redis):
    store = InteractionStore("t:")
    # idle_ttl is deliberately far larger than the question deadline so the two are
    # distinguishable: the count key rides ``idle_ttl`` (the state/stream basis),
    # never the shorter deadline, so a surviving state always has a live count.
    await store.add(fake_redis, _req("i1", "g", store, timeout_offset=60), idle_ttl=100000)
    assert fake_redis._ttls[store.count_key("g")] == pytest.approx(100000, abs=1)

    # A later question refreshes it to ``idle_ttl`` again (not to its own deadline).
    await store.add(fake_redis, _req("i2", "g", store, timeout_offset=300), idle_ttl=100000)
    assert fake_redis._ttls[store.count_key("g")] == pytest.approx(100000, abs=1)


async def test_pending_key_stays_creation_ordered_not_deadline(fake_redis):
    store = InteractionStore("t:")
    now = datetime.now(UTC)
    # Three groups added in creation order g0, g1, g2 but with deadlines that would
    # REVERSE the order if pending_key were (wrongly) scored by deadline.
    for i in range(3):
        req = InteractionRequest(
            interaction_id=f"i{i}",
            group_id=f"g{i}",
            question="?",
            answer_format=AnswerFormat.TEXT,
            reply_to=store.reply_key(f"i{i}"),
            created_at=now + timedelta(seconds=i),
            timeout_at=now + timedelta(seconds=1000 - i),
        )
        await store.add(fake_redis, req, idle_ttl=100)

    # The reconnect backlog replays pending_key in SCORE (creation) order.
    assert await fake_redis.zrange(store.pending_key, 0, -1) == ["g0", "g1", "g2"]
    # The parallel deadline index carries the reversed (deadline) order — proving
    # the two indexes are distinct and pending_key was never rescored.
    assert await fake_redis.zrange(store.pending_deadline_key, 0, -1) == ["g2", "g1", "g0"]
