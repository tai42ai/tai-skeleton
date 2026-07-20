"""Builtin ``ask_user`` tool: a thin shim over the interactions helper. It must
forward every argument verbatim, return the helper's answer, and never swallow a
timeout.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime

import pytest
from tai_contract.access_control import OWNER_USER_ID_CLAIM
from tai_contract.access_control.context import reset_request_user_id, set_request_user_id
from tai_contract.interactions import InteractionResponse

from tai_skeleton.access_control.request_scopes import (
    reset_request_identity_claims,
    set_request_identity_claims,
)
from tai_skeleton.access_control.user import CrossIdentityAudienceError
from tai_skeleton.interactions import InteractionStore, InteractionTimeoutError
from tai_skeleton.interactions import helper as helper_module
from tai_skeleton.tools.builtin import interactions as builtin_interactions
from tests._helpers import await_add_event


@contextmanager
def _restricted(own_id: str, owner: str | None = None) -> Iterator[None]:
    """Bind a RESTRICTED owned-key caller isolated to its OWN id ``own_id``. The owner
    claim (a DISTINCT ``owner-of-{own_id}`` by default) is what MARKS the caller
    restricted, but the isolation identity is its own id — each key is its own island —
    so the write clamp scopes writes to ``own_id``, never the owner."""
    owner_claim = owner if owner is not None else f"owner-of-{own_id}"
    claims_token = set_request_identity_claims({OWNER_USER_ID_CLAIM: owner_claim})
    uid_token = set_request_user_id(own_id)
    try:
        yield
    finally:
        reset_request_user_id(uid_token)
        reset_request_identity_claims(claims_token)


class _RecordingHelper:
    def __init__(self, *, answer: object = None, raise_exc: Exception | None = None) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self._answer = answer
        self._raise = raise_exc

    async def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        if self._raise is not None:
            raise self._raise
        return self._answer


async def test_ask_user_forwards_arguments_and_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _RecordingHelper(answer="blue")
    monkeypatch.setattr(builtin_interactions, "_ask_user", helper)

    result = await builtin_interactions.ask_user(
        "Favourite colour?",
        answer_format="select",
        options=["red", "blue"],
        schema=None,
        group_id="g1",
        timeout=30.0,
        link=None,
    )

    assert result == "blue"
    assert helper.calls == [
        (
            ("Favourite colour?",),
            {
                "answer_format": "select",
                "options": ["red", "blue"],
                "schema": None,
                "group_id": "g1",
                "timeout": 30.0,
                "link": None,
                "channel": None,
                "recipient": None,
                "audience": None,
                "media": None,
            },
        )
    ]


async def test_ask_user_defaults_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _RecordingHelper(answer="hi")
    monkeypatch.setattr(builtin_interactions, "_ask_user", helper)

    result = await builtin_interactions.ask_user("Anything?")

    assert result == "hi"
    assert helper.calls == [
        (
            ("Anything?",),
            {
                "answer_format": "text",
                "options": None,
                "schema": None,
                "group_id": None,
                "timeout": None,
                "link": None,
                "channel": None,
                "recipient": None,
                "audience": None,
                "media": None,
            },
        )
    ]


async def test_ask_user_forwards_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _RecordingHelper(answer="ok")
    monkeypatch.setattr(builtin_interactions, "_ask_user", helper)

    result = await builtin_interactions.ask_user("Ping?", channel="telegram")

    assert result == "ok"
    assert helper.calls[0][1]["channel"] == "telegram"


async def test_ask_user_forwards_recipient(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _RecordingHelper(answer="ok")
    monkeypatch.setattr(builtin_interactions, "_ask_user", helper)

    result = await builtin_interactions.ask_user("Ping?", channel="telegram", recipient="@ops")

    assert result == "ok"
    assert helper.calls[0][1]["channel"] == "telegram"
    assert helper.calls[0][1]["recipient"] == "@ops"


async def test_ask_user_forwards_media(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _RecordingHelper(answer="ok")
    monkeypatch.setattr(builtin_interactions, "_ask_user", helper)

    media = [{"kind": "image", "url": "https://cdn.example/p.png", "caption": "A product"}]
    result = await builtin_interactions.ask_user("Which?", media=media)

    assert result == "ok"
    assert helper.calls[0][1]["media"] == media


async def test_ask_user_propagates_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _RecordingHelper(raise_exc=InteractionTimeoutError("no answer"))
    monkeypatch.setattr(builtin_interactions, "_ask_user", helper)

    with pytest.raises(InteractionTimeoutError, match="no answer"):
        await builtin_interactions.ask_user("Still there?")


# -- write-side isolation clamp: a restricted caller touches ONLY its own slice ---
# Exercised end-to-end through the agent-facing builtin shim over the REAL helper
# (no monkeypatched helper), so the clamp is proven on the surface an agent uses.


async def _persisted_audience(fake_redis, store: InteractionStore, *, audience: str | None) -> str | None:
    """Drive the blocking shim to completion with ``audience``: answer the question
    the moment it is persisted, capture the ``audience`` off the stored request, and
    return it (asserting the shim returned the recorded answer)."""
    captured: dict[str, str | None] = {}

    async def answer_when_asked() -> None:
        interaction_id, group_id = await await_add_event(fake_redis, store)
        state = await store.get_state(fake_redis, interaction_id)
        assert state is not None
        captured["audience"] = state.request.audience
        await store.record_answer(
            fake_redis,
            InteractionResponse(
                interaction_id=interaction_id,
                answer="ok",
                answered_by="tester",
                answered_at=datetime.now(UTC),
            ),
            group_id,
            reply_ttl=60,
        )

    answerer = asyncio.create_task(answer_when_asked())
    result = await builtin_interactions.ask_user("proceed?", audience=audience, timeout=5)
    await answerer
    assert result == "ok"
    return captured["audience"]


async def test_restricted_ask_user_rejects_other_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    # A restricted bob addressing alice is a cross-identity inject/exfil attempt: it
    # is rejected loudly as an authorization denial (CrossIdentityAudienceError, the
    # write-side mirror of the answer door's 403) and NO state is written (no redis
    # connection is even opened, so no InteractionRequest with audience=alice can land
    # in alice's slice).
    calls: list = []

    @asynccontextmanager
    async def tracking_ctx(*args, **kwargs):
        calls.append((args, kwargs))
        yield None

    monkeypatch.setattr(helper_module, "client_ctx", tracking_ctx)
    with (
        _restricted("bob"),
        pytest.raises(CrossIdentityAudienceError, match="a restricted caller may address only its own identity"),
    ):
        await builtin_interactions.ask_user("proceed?", audience="alice")
    assert calls == []


async def test_restricted_ask_user_rejects_own_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    # Under key-keyed isolation each key is its own island, so the caller's OWN OWNER
    # is a FOREIGN target with no write privilege. A restricted bob (own id bob,
    # owner-claim alice) addressing its owner alice is rejected loudly as a
    # CrossIdentityAudienceError and NO redis connection is even opened, so no request
    # with audience=alice can land in the owner's slice. Pins that the owner is no
    # longer a privileged write target; an owner-privileged model would let it through.
    calls: list = []

    @asynccontextmanager
    async def tracking_ctx(*args, **kwargs):
        calls.append((args, kwargs))
        yield None

    monkeypatch.setattr(helper_module, "client_ctx", tracking_ctx)
    with (
        _restricted("bob", owner="alice"),
        pytest.raises(CrossIdentityAudienceError, match="a restricted caller may address only its own identity"),
    ):
        await builtin_interactions.ask_user("proceed?", audience="alice")
    assert calls == []


async def test_restricted_ask_user_scopes_unset_audience_to_self(monkeypatch, fake_redis, fake_client_ctx) -> None:
    # An unset audience is scoped to the restricted caller's OWN identity, never its owner.
    monkeypatch.setattr(helper_module, "client_ctx", fake_client_ctx)
    store = InteractionStore(helper_module.interactions_settings().key_prefix)
    with _restricted("bob"):
        persisted = await _persisted_audience(fake_redis, store, audience=None)
    assert persisted == "bob"


async def test_restricted_ask_user_allows_own_identity(monkeypatch, fake_redis, fake_client_ctx) -> None:
    # Addressing its OWN identity passes unchanged.
    monkeypatch.setattr(helper_module, "client_ctx", fake_client_ctx)
    store = InteractionStore(helper_module.interactions_settings().key_prefix)
    with _restricted("bob"):
        persisted = await _persisted_audience(fake_redis, store, audience="bob")
    assert persisted == "bob"


async def test_unrestricted_ask_user_may_address_any_identity(monkeypatch, fake_redis, fake_client_ctx) -> None:
    # Regression guard: an unrestricted caller (no bound owner claim) is NOT clamped
    # — it may address any identity, exactly as before.
    monkeypatch.setattr(helper_module, "client_ctx", fake_client_ctx)
    store = InteractionStore(helper_module.interactions_settings().key_prefix)
    persisted = await _persisted_audience(fake_redis, store, audience="alice")
    assert persisted == "alice"
