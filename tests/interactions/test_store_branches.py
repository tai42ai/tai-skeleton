"""Branch coverage for ``InteractionStore``: the bytes-decoding normalizer, the
sibling-TTL refresh on a second question in a group, a multi-question group that
stays pending after one answer, the WATCH-conflict retry, and missing state.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from redis.exceptions import WatchError
from tai_contract.interactions import (
    AnswerFormat,
    InteractionRequest,
    InteractionResponse,
)

from tai_skeleton.interactions import InteractionStore
from tai_skeleton.interactions import store as store_module


def _request(interaction_id: str, group_id: str, store: InteractionStore) -> InteractionRequest:
    now = datetime.now(UTC)
    return InteractionRequest(
        interaction_id=interaction_id,
        group_id=group_id,
        question="proceed?",
        answer_format=AnswerFormat.TEXT,
        reply_to=store.reply_key(interaction_id),
        created_at=now,
        timeout_at=now + timedelta(seconds=60),
    )


def _response(interaction_id: str) -> InteractionResponse:
    return InteractionResponse(
        interaction_id=interaction_id,
        answer="ok",
        answered_by="tester",
        answered_at=datetime.now(UTC),
    )


def test_as_str_normalizes_bytes_and_passthrough():
    assert store_module.as_str(b"hello") == "hello"
    assert store_module.as_str(bytearray(b"hi")) == "hi"
    assert store_module.as_str("plain") == "plain"
    assert store_module.as_str(None) is None


async def test_get_state_returns_none_when_missing(fake_redis):
    store = InteractionStore("t:")
    assert await store.get_state(fake_redis, "ghost") is None


async def test_second_question_refreshes_sibling_ttls(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)
    # The second add reads the group's existing entries and refreshes each
    # sibling's state TTL — the loop only runs when a sibling is present.
    await store.add(fake_redis, _request("i2", "g", store), idle_ttl=100)

    assert (await store.get_state(fake_redis, "i1")) is not None
    assert (await store.get_state(fake_redis, "i2")) is not None


async def test_sibling_refresh_skips_entry_without_interaction_id(fake_redis):
    store = InteractionStore("t:")
    # Seed the group stream with a malformed entry carrying no interaction_id;
    # the sibling-refresh loop must skip it rather than expire a bogus state key.
    fake_redis._streams.setdefault(store.group_key("g"), []).append(("0-0", {"junk": "x"}))

    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)

    assert (await store.get_state(fake_redis, "i1")) is not None


async def test_group_stays_pending_after_one_of_two_answers(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)
    await store.add(fake_redis, _request("i2", "g", store), idle_ttl=100)

    assert await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60) is True

    # One question remains open, so the group is not dropped from the pending
    # index and its count survives (decremented, not deleted).
    assert "g" in fake_redis._zsets[store.pending_key]
    assert fake_redis._strings[store.count_key("g")] == "1"


async def test_record_answer_retries_on_watch_conflict(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store), idle_ttl=100)

    real_pipeline = fake_redis.pipeline
    state = {"raised": False}

    class _WatchOnce:
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
                raise WatchError()
            return await self._inner.watch(*keys)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    fake_redis.pipeline = lambda: _WatchOnce(real_pipeline())

    assert await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60) is True
    assert state["raised"] is True
    answered = await store.get_state(fake_redis, "i1")
    assert answered is not None
    assert answered.status == "answered"


async def test_record_answer_on_missing_interaction_returns_false(fake_redis):
    store = InteractionStore("t:")
    assert await store.record_answer(fake_redis, _response("ghost"), "g", reply_ttl=60) is False
