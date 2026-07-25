"""Sensitive-answer persistence opt-out at the store layer: the ``sensitive``
flag round-trips through ``add`` into the state hash and the reconstructed
request; an answered sensitive question persists ONLY the status (never the
response body) while the blocked waiter still receives the full answer; and a
late duplicate to a sensitive question takes the already-answered path with no
body — by design, not a bug. The non-sensitive path keeps its response body
byte-for-byte as before (regression).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tai42_contract.interactions import (
    AnswerFormat,
    InteractionRequest,
    InteractionResponse,
)

from tai42_skeleton.interactions import InteractionStore


def _request(iid: str, gid: str, store: InteractionStore, *, sensitive: bool) -> InteractionRequest:
    now = datetime.now(UTC)
    return InteractionRequest(
        interaction_id=iid,
        group_id=gid,
        question="secret?",
        answer_format=AnswerFormat.TEXT,
        reply_to=store.reply_key(iid),
        created_at=now,
        timeout_at=now + timedelta(seconds=60),
        sensitive=sensitive,
    )


def _response(iid: str, answer: str = "s3cr3t") -> InteractionResponse:
    return InteractionResponse(interaction_id=iid, answer=answer, answered_by="tester", answered_at=datetime.now(UTC))


async def test_sensitive_flag_round_trips_through_add(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store, sensitive=True), idle_ttl=100)

    # Denormalized flag lives in the hash so record_answer can gate on one hget.
    assert fake_redis._hashes[store.state_key("i1")]["sensitive"] == "1"
    # And it survives on the reconstructed request (model_dump_json carries it).
    state = await store.get_state(fake_redis, "i1")
    assert state is not None
    assert state.request.sensitive is True


async def test_sensitive_answer_persists_status_only_no_body(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store, sensitive=True), idle_ttl=100)

    assert await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60) is True

    # The durable hash records the answered status but NEVER the response body.
    hash_fields = fake_redis._hashes[store.state_key("i1")]
    assert hash_fields["status"] == "answered"
    assert "response" not in hash_fields
    state = await store.get_state(fake_redis, "i1")
    assert state is not None
    assert state.status == "answered"
    assert state.response is None


async def test_sensitive_waiter_still_receives_full_answer(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store, sensitive=True), idle_ttl=100)
    await store.record_answer(fake_redis, _response("i1", answer="my-password"), "g", reply_ttl=60)

    # The reply channel (which wakes the blocked caller) carries the full answer —
    # only the persisted record drops the body.
    delivered = await store.wait_for_reply(fake_redis, store.reply_key("i1"), timeout_seconds=1, grace_seconds=5)
    assert delivered is not None
    assert delivered.answer == "my-password"


async def test_sensitive_late_duplicate_already_answered_no_body(fake_redis):
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store, sensitive=True), idle_ttl=100)
    assert await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60) is True

    # A late duplicate callback is a lost race: nothing re-written, and because the
    # first answer stored no body, the already-answered record still has none.
    assert await store.record_answer(fake_redis, _response("i1"), "g", reply_ttl=60) is False
    assert "response" not in fake_redis._hashes[store.state_key("i1")]


async def test_non_sensitive_answer_keeps_response_body(fake_redis):
    # Regression: the default path is unchanged — the response body is persisted
    # and reconstructs on read exactly as before.
    store = InteractionStore("t:")
    await store.add(fake_redis, _request("i1", "g", store, sensitive=False), idle_ttl=100)
    assert "sensitive" not in fake_redis._hashes[store.state_key("i1")]

    await store.record_answer(fake_redis, _response("i1", answer="visible"), "g", reply_ttl=60)
    state = await store.get_state(fake_redis, "i1")
    assert state is not None
    assert state.response is not None
    assert state.response.answer == "visible"
