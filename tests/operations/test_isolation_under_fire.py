"""The isolated operation doors under a background FIRE.

Every isolated surface consumes ``request_identity`` / ``clamp_write_audience``, so a
fire must reach them as the key it is authorized as. Each test binds a FOREIGN triggering
caller's request-scope vars alongside the fire — a ``BackgroundTask`` really does run
inside the triggering request's context — and the doors must still isolate as the key.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id
from tai42_contract.app import tai42_app
from tai42_contract.interactions import AnswerFormat, InteractionRequest

from tai42_skeleton.access_control.request_scopes import (
    reset_request_identity_claims,
    set_request_identity_claims,
)
from tai42_skeleton.authz.execution_identity import reset_execution_identity, set_execution_identity
from tai42_skeleton.authz.identity import CallerIdentity
from tai42_skeleton.channels import notifications_sink
from tai42_skeleton.interactions import InteractionStore
from tai42_skeleton.interactions.settings import InteractionsSettings
from tai42_skeleton.operations import ForbiddenError
from tai42_skeleton.operations import interactions as interactions_ops
from tai42_skeleton.operations import notifications as notifications_ops
from tai42_skeleton.operations import tool_runs as tool_runs_ops
from tai42_skeleton.operations.tool_runs import ToolRunStore
from tai42_skeleton.routers.tool_runs_settings import ToolRunsSettings
from tests._fakes.tool_runs_redis import FakeRedis as ToolRunsFakeRedis

KEY = "k-fire"
OWNER = "alice"
RINGER = "k-bob"


@contextmanager
def _fire(*, owned: bool) -> Iterator[None]:
    """Run the body as execution key ``KEY``, with the foreign caller ``RINGER`` — a
    RESTRICTED owned key of its own — still bound in the inherited request context.

    ``owned`` picks the claims ``build_execution_identity`` would build: the owner
    reference for an owned key, an empty mapping for an ownerless one.
    """
    claims = {OWNER_USER_ID_CLAIM: OWNER} if owned else {}
    ringer_claims_token = set_request_identity_claims({OWNER_USER_ID_CLAIM: "bob"})
    ringer_uid_token = set_request_user_id(RINGER)
    fire_token = set_execution_identity(CallerIdentity(user_id=KEY, effective_scopes=("*",), claims=claims))
    try:
        yield
    finally:
        reset_execution_identity(fire_token)
        reset_request_user_id(ringer_uid_token)
        reset_request_identity_claims(ringer_claims_token)


# -- read clamp: background tool runs -----------------------------------------


class _FakeTools:
    async def get_tools(self):
        return {"alpha": SimpleNamespace(name="alpha")}

    async def run_tool(self, key, arguments, *, offload_sync=False):
        return {"ok": 1}


@pytest.fixture
def runs_wired(monkeypatch):
    """The tool-run operations over an in-memory Redis, with the tool registry stubbed
    so ``submit_run`` reaches the real record write."""
    fake = ToolRunsFakeRedis()
    settings = ToolRunsSettings()

    @asynccontextmanager
    async def ctx(client_cls, s=None, *, fresh=False, **kwargs):
        yield fake

    monkeypatch.setattr(tool_runs_ops, "client_ctx", ctx)
    monkeypatch.setattr(tool_runs_ops, "tool_runs_settings", lambda: settings)
    monkeypatch.setattr(tool_runs_ops, "_now", lambda: datetime(2026, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(tool_runs_ops, "_ACTIVE_RUNS", 0)
    monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(tools=_FakeTools()))

    yield SimpleNamespace(fake=fake, settings=settings, store=ToolRunStore(settings.key_prefix))

    for task in list(tool_runs_ops._SUPERVISORS):
        task.cancel()


async def _seed_run(runs_wired, run_id: str, own_id: str, score: float) -> None:
    await runs_wired.store.create_run(
        runs_wired.fake, run_id, "alpha", "2026-01-01T00:00:00+00:00", score, runs_wired.settings, user_id=own_id
    )


async def test_owned_fire_submits_into_the_keys_own_slice(runs_wired) -> None:
    # The run is stamped and indexed under the KEY — not the triggering caller, and not
    # the key's owner (each key is its own island).
    with _fire(owned=True):
        run_id = (await tool_runs_ops.submit_run("alpha", {}))["run_id"]

    record = await runs_wired.store.get_run(runs_wired.fake, run_id)
    assert record["user_id"] == KEY
    assert run_id in await runs_wired.fake.zrevrange(runs_wired.store.recent_key("alpha", KEY), 0, -1)
    assert await runs_wired.fake.zrevrange(runs_wired.store.recent_key("alpha", RINGER), 0, -1) == []
    assert await runs_wired.fake.zrevrange(runs_wired.store.recent_key("alpha", OWNER), 0, -1) == []


async def test_ownerless_fire_attributes_its_run_to_the_key(runs_wired) -> None:
    # An ownerless key is UNRESTRICTED, but its run is still owned by the KEY — an
    # unrestricted fire must not write into the triggering caller's slice either.
    with _fire(owned=False):
        run_id = (await tool_runs_ops.submit_run("alpha", {}))["run_id"]

    record = await runs_wired.store.get_run(runs_wired.fake, run_id)
    assert record["user_id"] == KEY
    assert await runs_wired.fake.zrevrange(runs_wired.store.recent_key("alpha", RINGER), 0, -1) == []


async def test_owned_fire_reads_only_its_own_runs(runs_wired) -> None:
    # get_run admits the key's own run and denies the triggering caller's with the
    # cross-identity 403; list_tool_runs returns the key's slice alone.
    await _seed_run(runs_wired, "r-key", KEY, 1.0)
    await _seed_run(runs_wired, "r-ringer", RINGER, 2.0)

    with _fire(owned=True):
        assert (await tool_runs_ops.get_run("r-key"))["run_id"] == "r-key"
        with pytest.raises(ForbiddenError, match="belongs to another identity"):
            await tool_runs_ops.get_run("r-ringer")
        assert [e["run_id"] for e in await tool_runs_ops.list_tool_runs("alpha")] == ["r-key"]


async def test_ownerless_fire_reads_the_shared_run_index(runs_wired) -> None:
    # An ownerless key holds no owner claim, so it is unrestricted and reads the shared
    # index unchanged — the fire is not confined to the triggering caller's slice.
    await _seed_run(runs_wired, "r-key", KEY, 1.0)
    await _seed_run(runs_wired, "r-ringer", RINGER, 2.0)

    with _fire(owned=False):
        assert (await tool_runs_ops.get_run("r-ringer"))["run_id"] == "r-ringer"
        assert [e["run_id"] for e in await tool_runs_ops.list_tool_runs("alpha")] == ["r-ringer", "r-key"]


# -- read clamp: the interaction answer gate ----------------------------------


@pytest.fixture
def interactions_wired(monkeypatch, fake_redis, fake_client_ctx):
    settings = InteractionsSettings()
    monkeypatch.setattr(interactions_ops, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(interactions_ops, "interactions_settings", lambda: settings)
    return SimpleNamespace(settings=settings, store=InteractionStore(settings.key_prefix), fake=fake_redis)


async def _seed_question(interactions_wired, interaction_id: str, audience: str) -> None:
    now = datetime.now(UTC)
    await interactions_wired.store.add(
        interactions_wired.fake,
        InteractionRequest(
            interaction_id=interaction_id,
            group_id="g",
            question="?",
            answer_format=AnswerFormat.TEXT,
            reply_to=interactions_wired.store.reply_key(interaction_id),
            created_at=now,
            timeout_at=now + timedelta(seconds=60),
            audience=audience,
        ),
        idle_ttl=86400,
    )


async def test_owned_fire_answers_only_its_own_question(interactions_wired) -> None:
    # A restricted fire may answer only a question addressed to the KEY; the triggering
    # caller's pending question is a loud 403, not an answerable one.
    await _seed_question(interactions_wired, "q-key", KEY)
    await _seed_question(interactions_wired, "q-ringer", RINGER)

    with _fire(owned=True):
        assert await interactions_ops.answer_interaction("q-key", "hi") == {
            "interaction_id": "q-key",
            "status": "answered",
        }
        with pytest.raises(ForbiddenError, match="addressed to another identity"):
            await interactions_ops.answer_interaction("q-ringer", "hi")


async def test_ownerless_fire_may_answer_any_question(interactions_wired) -> None:
    # An ownerless key is unrestricted — the operator class that can unblock any stuck
    # question — so the gate lets it through.
    await _seed_question(interactions_wired, "q-ringer", RINGER)

    with _fire(owned=False):
        assert await interactions_ops.answer_interaction("q-ringer", "hi") == {
            "interaction_id": "q-ringer",
            "status": "answered",
        }


# -- read clamp + write clamp: notifications ----------------------------------


@pytest.fixture
def sink_redis(monkeypatch, fake_redis):
    """Point the internal notifications sink's Redis at the shared fake so the real
    helper's writes land somewhere readable back in the test."""

    @asynccontextmanager
    async def _ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield fake_redis

    monkeypatch.setattr(notifications_sink, "client_ctx", _ctx)
    return fake_redis


async def test_owned_fire_notifies_only_its_own_feed(sink_redis) -> None:
    # The write clamp scopes an unset audience to the KEY's feed and rejects every other
    # identity, the triggering caller and the key's owner included.
    with _fire(owned=True):
        await notifications_ops.notify_user("status")
        for foreign in (RINGER, OWNER, "victim"):
            with pytest.raises(ForbiddenError, match="may address only its own identity"):
                await notifications_ops.notify_user("hi", audience=foreign)

    own = await notifications_sink.read_notifications(audience=KEY)
    assert [record["audience"] for record in own] == [KEY]
    for foreign in (RINGER, OWNER, "victim"):
        assert await notifications_sink.read_notifications(audience=foreign) == []


async def test_owned_fire_lists_only_its_own_feed(sink_redis) -> None:
    await notifications_sink.record_notification("for the key", audience=KEY)
    await notifications_sink.record_notification("for the ringer", audience=RINGER)

    with _fire(owned=True):
        listed = (await notifications_ops.list_notifications())["notifications"]

    assert [record["message"] for record in listed] == ["for the key"]


async def test_ownerless_fire_is_unclamped_and_reads_the_shared_feed(sink_redis) -> None:
    # An ownerless key is unrestricted: it may address any identity and reads the shared
    # feed — the fire is neither confined to nor attributed to the triggering caller.
    await notifications_sink.record_notification("for the ringer", audience=RINGER)

    with _fire(owned=False):
        await notifications_ops.notify_user("hi victim", audience="victim")
        listed = (await notifications_ops.list_notifications())["notifications"]

    assert [record["audience"] for record in await notifications_sink.read_notifications(audience="victim")] == [
        "victim"
    ]
    assert {record["message"] for record in listed} == {"for the ringer", "hi victim"}
