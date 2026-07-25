"""Op-level oracles for the human answer operation.

These pin ``answer_interaction``'s store-logic branches DIRECTLY through the
operation (typed raises, not the route's JSON responses) — independent of the
router adapter and its body extractor — plus the format-validation helper's
server-bug guard and the declared destructive/error-class metadata. Redis is the
shared in-memory fake wired at the operation module's ``client_ctx`` seam.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from tai42_contract.interactions import AnswerFormat, InteractionRequest

from tai42_skeleton.interactions import InteractionStore
from tai42_skeleton.interactions.settings import InteractionsSettings
from tai42_skeleton.operations import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PayloadTooLargeError,
)
from tai42_skeleton.operations import interactions as ops
from tai42_skeleton.operations.decorator import operation_metadata_of


@pytest.fixture
def wired(monkeypatch, fake_redis, fake_client_ctx):
    settings = InteractionsSettings()
    monkeypatch.setattr(ops, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(ops, "interactions_settings", lambda: settings)
    store = InteractionStore(settings.key_prefix)
    return SimpleNamespace(settings=settings, store=store, fake=fake_redis)


def _req(store, fmt, *, iid="p1", gid="pg", payload=None) -> InteractionRequest:
    now = datetime.now(UTC)
    return InteractionRequest(
        interaction_id=iid,
        group_id=gid,
        question="?",
        answer_format=fmt,
        format_payload=payload,
        reply_to=store.reply_key(iid),
        created_at=now,
        timeout_at=now + timedelta(seconds=60),
    )


async def test_answer_unknown_interaction_raises_not_found(wired):
    with pytest.raises(NotFoundError, match="Interaction not found"):
        await ops.answer_interaction("ghost", "hi")


async def test_answer_external_raises_bad_request(wired):
    await wired.store.add(wired.fake, _req(wired.store, AnswerFormat.EXTERNAL, payload={"url": "x"}), idle_ttl=86400)
    with pytest.raises(BadRequestError, match="callback URL"):
        await ops.answer_interaction("p1", "x")


async def test_answer_text_success_then_conflict(wired):
    await wired.store.add(wired.fake, _req(wired.store, AnswerFormat.TEXT), idle_ttl=86400)
    assert await ops.answer_interaction("p1", "hello") == {"interaction_id": "p1", "status": "answered"}
    with pytest.raises(ConflictError, match="already answered"):
        await ops.answer_interaction("p1", "again")


async def test_answer_invalid_value_raises_bad_request(wired):
    await wired.store.add(wired.fake, _req(wired.store, AnswerFormat.CONFIRM), idle_ttl=86400)
    with pytest.raises(BadRequestError, match="must be a boolean"):
        await ops.answer_interaction("p1", "not-a-bool")


def test_validate_answer_external_is_a_server_bug(wired):
    # EXTERNAL is rejected by the answer door before ``_validate_answer`` runs, so a
    # direct call is the defensive server-bug path — a loud 500, never a 4xx.
    req = _req(wired.store, AnswerFormat.EXTERNAL, payload={"url": "x"})
    with pytest.raises(RuntimeError, match="unhandled answer_format"):
        ops._validate_answer(req, "x")


def test_metadata_declares_destructive_and_the_full_error_set():
    meta = operation_metadata_of(ops.answer_interaction)
    assert meta.destructive is True
    assert meta.meta_executor is False
    assert meta.reload_gated is False
    assert set(meta.error_classes) == {
        BadRequestError,
        ConflictError,
        ForbiddenError,
        NotFoundError,
        PayloadTooLargeError,
    }
