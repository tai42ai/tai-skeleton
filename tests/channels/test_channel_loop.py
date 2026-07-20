"""The fake-channel end-to-end loop, offline: ``ask_user(channel=...)``
delivers through a registered channel, the recorded ``callback_url`` receives
the human's reply as ``{"answer": <value>}`` through the public callback door
(which validates it against the stored format), and the blocked caller returns
the typed value — deliver -> inbound bridge -> callback claim -> resume, proven
with no network and no external stack.

Handlers are driven directly (the router-test pattern); Redis is the shared
in-memory fake, wired at both the router's and the helper's ``client_ctx``
seams so the blocked ``ask_user`` caller and the callback door share one store.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.requests import Request
from tai42_contract.app import tai42_app
from tai42_contract.channels import ChannelDelivery, ChannelDeliveryError
from tai42_contract.interactions import MediaItem

from tai42_skeleton.app.instance import app
from tai42_skeleton.interactions import InteractionStore, ask_user
from tai42_skeleton.interactions import helper as helper_module
from tai42_skeleton.interactions.settings import InteractionsSettings
from tai42_skeleton.routers import interactions as router
from tests._helpers import DeliverOnlyChannel, await_add_event


class FakeChannel(DeliverOnlyChannel):
    """Records every delivery; the test then plays the plugin's part by POSTing
    the human's typed answer to the recorded ``callback_url``."""

    def __init__(self) -> None:
        self.deliveries: list[ChannelDelivery] = []

    async def deliver(self, delivery: ChannelDelivery) -> None:
        self.deliveries.append(delivery)


class FailingChannel(DeliverOnlyChannel):
    async def deliver(self, delivery: ChannelDelivery) -> None:
        raise ChannelDeliveryError("provider unreachable")


class BuggyChannel(DeliverOnlyChannel):
    async def deliver(self, delivery: ChannelDelivery) -> None:
        raise RuntimeError("plugin bug")


@pytest.fixture
def wired(monkeypatch, fake_redis, fake_client_ctx):
    settings = InteractionsSettings(public_base_url="https://cb.example")
    monkeypatch.setattr(router, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(router, "interactions_settings", lambda: settings)
    monkeypatch.setattr(helper_module, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(helper_module, "interactions_settings", lambda: settings)
    store = InteractionStore(settings.key_prefix)
    return SimpleNamespace(settings=settings, store=store, fake=fake_redis)


@pytest.fixture
def fake_channel():
    channel = FakeChannel()
    app._channel_registry.reset()
    tai42_app.channels.register("fake", channel)
    yield channel
    app._channel_registry.reset()


def _make_request(method, *, path_params=None, query="", body=b"", headers=None):
    hdrs = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": method,
        "path": "/api/interactions/callback/x",
        "query_string": query.encode(),
        "headers": hdrs,
        "client": ("1.2.3.4", 1111),
        "path_params": path_params or {},
    }
    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


async def _await_delivery(channel: FakeChannel, timeout: float = 2.0) -> ChannelDelivery:
    """Poll until ``deliver`` ran (it runs after the add event, inside the asking
    task) — fail the test on timeout, never hang."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if channel.deliveries:
            return channel.deliveries[0]
        await asyncio.sleep(0.01)
    raise AssertionError("channel.deliver was never called")


def _ticket(delivery: ChannelDelivery) -> str:
    return delivery.callback_url.rsplit("/", 1)[-1]


def _empty(fake_redis) -> bool:
    # All five FakeRedis stores, ``_lists`` included (the reply channel is a list).
    return not (
        fake_redis._hashes or fake_redis._streams or fake_redis._zsets or fake_redis._strings or fake_redis._lists
    )


# -- the loop ------------------------------------------------------------------


async def test_fake_channel_text_loop(wired, fake_channel):
    # 1. an agent-side ask blocks, bound to the registered channel
    task = asyncio.create_task(ask_user("What is the magic word?", channel="fake", timeout=5))
    iid, _gid = await await_add_event(wired.fake, wired.store)

    # 2. deliver() ran exactly once with the full delivery contract
    delivery = await _await_delivery(fake_channel)
    assert len(fake_channel.deliveries) == 1
    assert delivery.interaction_id == iid
    assert delivery.question == "What is the magic word?"
    assert delivery.answer_format == "text"
    assert delivery.options is None
    assert delivery.recipient is None  # no caller address -> the plugin's default
    assert delivery.callback_url.startswith("https://cb.example/api/interactions/callback/")

    # 3. the persisted request carries the channel; the add frame emits it
    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.request.channel == "fake"
    assert router._add_data(state.request)["channel"] == "fake"

    # 4. play the plugin: forward the human's reply as {"answer": <value>}
    resp = await router.callback(
        _make_request("POST", path_params={"ticket": _ticket(delivery)}, body=b'{"answer": "please"}')
    )
    assert resp.status_code == 200
    assert json.loads(bytes(resp.body))["data"]["status"] == "answered"

    # 5. the blocked ask_user returns the TYPED answer (str) — loop closed
    assert await task == "please"


async def test_fake_channel_confirm_loop(wired, fake_channel):
    task = asyncio.create_task(ask_user("Deploy?", answer_format="confirm", channel="fake", timeout=5))
    await await_add_event(wired.fake, wired.store)
    delivery = await _await_delivery(fake_channel)
    assert delivery.answer_format == "confirm"

    # Tap flow: the GET-confirm page's form POSTs an empty body → recorded True.
    resp = await router.callback(_make_request("POST", path_params={"ticket": _ticket(delivery)}, body=b""))
    assert resp.status_code == 200
    assert await task is True  # the TYPED bool


async def test_add_frame_omits_channel_when_unset(wired):
    # No channel → the add frame has NO ``channel`` key (conditional, like
    # ``server_verified``) and the persisted request carries None.
    task = asyncio.create_task(ask_user("q", timeout=5))
    iid, _gid = await await_add_event(wired.fake, wired.store)
    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.request.channel is None
    assert "channel" not in router._add_data(state.request)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_delivery_runs_after_persist(wired, fake_channel):
    # The callback ticket must be claimable at deliver time: the persisted state
    # already exists when the channel is handed the delivery.
    seen: dict = {}

    class ProbingChannel(DeliverOnlyChannel):
        async def deliver(self, delivery: ChannelDelivery) -> None:
            seen["state"] = await wired.store.get_state(wired.fake, delivery.interaction_id)
            seen["delivery"] = delivery

    app._channel_registry.reset()
    tai42_app.channels.register("probe", ProbingChannel())
    try:
        task = asyncio.create_task(ask_user("q", channel="probe", timeout=5))
        _iid, _gid = await await_add_event(wired.fake, wired.store)
        deadline = asyncio.get_event_loop().time() + 2.0
        while "delivery" not in seen and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.01)

        assert seen["state"] is not None  # persisted BEFORE deliver
        resp = await router.callback(
            _make_request("POST", path_params={"ticket": _ticket(seen["delivery"])}, body=b'{"answer": "done"}')
        )
        assert resp.status_code == 200
        assert await task == "done"
    finally:
        app._channel_registry.reset()


async def test_channel_external_url_is_callback_url(wired, fake_channel):
    # For a channel-delivered external ask, the stored tappable url IS the
    # callback door — no link builder involved.
    task = asyncio.create_task(ask_user("Approve?", answer_format="external", channel="fake", timeout=5))
    iid, _gid = await await_add_event(wired.fake, wired.store)
    delivery = await _await_delivery(fake_channel)

    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.request.format_payload is not None
    assert state.request.format_payload["url"] == delivery.callback_url

    # EXTERNAL keeps the verbatim-dict semantics — no {"answer": ...} envelope.
    resp = await router.callback(
        _make_request("POST", path_params={"ticket": _ticket(delivery)}, body=b'{"approved": true}')
    )
    assert resp.status_code == 200
    assert await task == {"approved": True}


async def test_recipient_forwarded_in_delivery(wired, fake_channel):
    # The caller-requested address rides the delivery verbatim; the skeleton
    # resolves nothing — allowlist validation is the plugin's job.
    task = asyncio.create_task(ask_user("Ping?", channel="fake", recipient="123456", timeout=5))
    await await_add_event(wired.fake, wired.store)
    delivery = await _await_delivery(fake_channel)

    assert delivery.recipient == "123456"

    resp = await router.callback(
        _make_request("POST", path_params={"ticket": _ticket(delivery)}, body=b'{"answer": "pong"}')
    )
    assert resp.status_code == 200
    assert await task == "pong"


async def test_channel_delivery_omits_media_inbox_keeps_it(wired, fake_channel):
    # Display-only media is NOT forwarded to a channel: the ChannelDelivery handed
    # to the plugin carries no media, while the persisted question keeps it and the
    # inbox add frame renders it. A documented limit, never a silent drop.
    media: list[MediaItem | dict[str, Any]] = [
        {"kind": "image", "url": "https://cdn.example/p.png", "caption": "A product"}
    ]
    task = asyncio.create_task(ask_user("Which?", channel="fake", media=media, timeout=5))
    iid, _gid = await await_add_event(wired.fake, wired.store)
    delivery = await _await_delivery(fake_channel)

    assert not hasattr(delivery, "media")  # the delivery contract carries no media

    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert router._add_data(state.request)["media"] == media

    resp = await router.callback(
        _make_request("POST", path_params={"ticket": _ticket(delivery)}, body=b'{"answer": "ok"}')
    )
    assert resp.status_code == 200
    assert await task == "ok"


async def test_channel_select_delivery_carries_options(wired, fake_channel):
    task = asyncio.create_task(
        ask_user("Pick one", answer_format="select", options=["red", "blue"], channel="fake", timeout=5)
    )
    await await_add_event(wired.fake, wired.store)
    delivery = await _await_delivery(fake_channel)

    assert delivery.answer_format == "select"
    assert delivery.options == ["red", "blue"]

    resp = await router.callback(
        _make_request("POST", path_params={"ticket": _ticket(delivery)}, body=b'{"answer": "blue"}')
    )
    assert resp.status_code == 200
    assert await task == "blue"  # the TYPED chosen option, not a dict


# -- loud validation (nothing persisted) ----------------------------------------


async def test_unknown_channel_rejected_up_front(wired):
    app._channel_registry.reset()
    with pytest.raises(ValueError, match="unknown channel"):
        await ask_user("q", channel="nope", timeout=5)
    assert _empty(wired.fake)  # rejected BEFORE any state was written


async def test_channel_forbids_link_non_external(wired, fake_channel):
    with pytest.raises(ValueError, match="link is forbidden when a channel is set"):
        await ask_user("q", channel="fake", link="{callback_url}", timeout=5)
    assert _empty(wired.fake)


async def test_channel_forbids_link_external(wired, fake_channel):
    with pytest.raises(ValueError, match="link is forbidden when a channel is set"):
        await ask_user("q", answer_format="external", channel="fake", link="{callback_url}", timeout=5)
    assert _empty(wired.fake)


async def test_channel_forbids_verifier(wired, fake_channel):
    # A channel's forward is unsigned — a bound verifier would 401 every reply.
    with pytest.raises(ValueError, match="verifier is forbidden when a channel is set"):
        await ask_user(
            "q", answer_format="external", channel="fake", verifier={"name": "github", "config": {}}, timeout=5
        )
    assert _empty(wired.fake)


async def test_channel_rejects_form(wired, fake_channel):
    # No single-reply mapping for a multi-field form on a chat/SMS medium.
    with pytest.raises(ValueError, match="'form' is not supported over a channel"):
        await ask_user("q", answer_format="form", schema={"type": "object"}, channel="fake", timeout=5)
    assert _empty(wired.fake)


async def test_recipient_without_channel_rejected(wired):
    # An address is meaningless without a channel to send on.
    with pytest.raises(ValueError, match="recipient requires a channel"):
        await ask_user("q", recipient="123456", timeout=5)
    assert _empty(wired.fake)


@pytest.mark.parametrize("bad_recipient", ["", "   "])
async def test_blank_recipient_with_channel_rejected(wired, fake_channel, bad_recipient):
    # Rejected up-front as a clean ValueError — never a post-persist pydantic
    # error from the delivery frame's recipient validator.
    with pytest.raises(ValueError, match="recipient must be a non-empty address"):
        await ask_user("q", channel="fake", recipient=bad_recipient, timeout=5)
    assert _empty(wired.fake)


async def test_channel_rejects_non_select_options(wired, fake_channel):
    # Rejected up-front as a clean ValueError — never a post-persist pydantic error.
    with pytest.raises(ValueError, match="options are only valid with answer_format 'select'"):
        await ask_user("q", channel="fake", options=["a", "b"], timeout=5)
    assert _empty(wired.fake)


async def test_channel_requires_public_base_url(monkeypatch, fake_redis, fake_client_ctx, fake_channel):
    settings = InteractionsSettings(public_base_url=None)
    monkeypatch.setattr(helper_module, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(helper_module, "interactions_settings", lambda: settings)
    with pytest.raises(RuntimeError, match="INTERACTIONS_PUBLIC_BASE_URL"):
        await ask_user("q", channel="fake", timeout=5)


# -- failed delivery: prune + raise, never a silent zombie question -------------


async def test_delivery_failure_prunes_and_raises(wired):
    app._channel_registry.reset()
    tai42_app.channels.register("boom", FailingChannel())
    try:
        with pytest.raises(ChannelDeliveryError, match="provider unreachable"):
            await ask_user("q", channel="boom", timeout=5)

        # The persisted question was pruned: nothing open, nothing claimable.
        assert await wired.store.count_open(wired.fake) == 0
    finally:
        app._channel_registry.reset()


async def test_delivery_bug_prunes_and_raises(wired):
    app._channel_registry.reset()
    tai42_app.channels.register("buggy", BuggyChannel())
    try:
        with pytest.raises(RuntimeError, match="plugin bug"):
            await ask_user("q", channel="buggy", timeout=5)

        assert await wired.store.count_open(wired.fake) == 0
    finally:
        app._channel_registry.reset()


async def test_failed_delivery_ticket_unclaimable(wired):
    # After the prune, the minted ticket resolves to a dead interaction: the
    # callback door answers the uniform 404 — no late answer can resurrect it.
    captured: dict = {}

    class CapturingFailer(DeliverOnlyChannel):
        async def deliver(self, delivery: ChannelDelivery) -> None:
            captured["delivery"] = delivery
            raise ChannelDeliveryError("send failed")

    app._channel_registry.reset()
    tai42_app.channels.register("capfail", CapturingFailer())
    try:
        with pytest.raises(ChannelDeliveryError):
            await ask_user("q", channel="capfail", timeout=5)
    finally:
        app._channel_registry.reset()

    resp = await router.callback(
        _make_request("POST", path_params={"ticket": _ticket(captured["delivery"])}, body=b'{"answer": "late"}')
    )
    assert resp.status_code == 404


async def test_delivery_failure_after_recorded_answer_falls_through(wired):
    # A fast reply can land before the delivery failure surfaces (e.g. the
    # provider accepted the message, the human answered, and only then the send
    # call errored). ``prune_pending`` then reports already-answered — the
    # recorded answer is returned, never discarded, and nothing raises.
    class AnswerThenFail(DeliverOnlyChannel):
        async def deliver(self, delivery: ChannelDelivery) -> None:
            resp = await router.callback(
                _make_request("POST", path_params={"ticket": _ticket(delivery)}, body=b'{"answer": "fast"}')
            )
            assert resp.status_code == 200
            raise ChannelDeliveryError("send failed after the reply landed")

    app._channel_registry.reset()
    tai42_app.channels.register("racy", AnswerThenFail())
    try:
        assert await ask_user("q", channel="racy", timeout=5) == "fast"
    finally:
        app._channel_registry.reset()


async def test_hung_delivery_times_out_prunes_and_raises(wired):
    # ``deliver`` is one sub-second send attempt bounded by the ask's whole
    # timeout budget; a deliver still running at that deadline is a hung plugin,
    # so the ask fails loudly instead of blocking forever on the send.
    class HungChannel(DeliverOnlyChannel):
        async def deliver(self, delivery: ChannelDelivery) -> None:
            await asyncio.sleep(60)

    app._channel_registry.reset()
    tai42_app.channels.register("hung", HungChannel())
    try:
        with pytest.raises(ChannelDeliveryError, match=r"delivery timed out after 0\.05s"):
            await ask_user("q", channel="hung", timeout=0.05)

        # The persisted question was pruned: nothing open, nothing claimable.
        assert await wired.store.count_open(wired.fake) == 0
    finally:
        app._channel_registry.reset()


async def test_hung_delivery_after_recorded_answer_falls_through(wired):
    # The reply lands while deliver is still hanging: the timeout's prune
    # reports already-answered, so the recorded answer is returned — the same
    # fall-through as any other post-answer delivery failure.
    class AnswerThenHang(DeliverOnlyChannel):
        async def deliver(self, delivery: ChannelDelivery) -> None:
            resp = await router.callback(
                _make_request("POST", path_params={"ticket": _ticket(delivery)}, body=b'{"answer": "fast"}')
            )
            assert resp.status_code == 200
            await asyncio.sleep(60)

    app._channel_registry.reset()
    tai42_app.channels.register("hang", AnswerThenHang())
    try:
        assert await ask_user("q", channel="hang", timeout=0.3) == "fast"
    finally:
        app._channel_registry.reset()
