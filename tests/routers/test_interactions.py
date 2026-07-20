"""Interactions router: the human answer door, the two callback doors (data +
redirect), byte-constant HTML + headers, webhook-verifier binding on the
callback door, and the SSE backlog/tail. (The public-door rate limiter lives
in the app-level ``RateLimitMiddleware``; its tests are in ``tests/middleware``.)

Handlers are driven directly (the existing router-test pattern); Redis is the
shared in-memory fake, wired at both the router's and the helper's ``client_ctx``
seams so a blocked ``ask_user`` caller and the callback door share one store.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError
from starlette.requests import Request
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id
from tai42_contract.interactions import (
    AnswerFormat,
    InteractionRequest,
    InteractionResponse,
    MediaItem,
    MediaKind,
)

from tai42_skeleton.access_control.request_scopes import (
    reset_request_identity_claims,
    set_request_identity_claims,
)
from tai42_skeleton.interactions import InteractionStore, ask_user
from tai42_skeleton.interactions import helper as helper_module
from tai42_skeleton.interactions.settings import InteractionsSettings
from tai42_skeleton.operations import interactions as ops
from tai42_skeleton.routers import interactions as router
from tests._helpers import await_add_event


@pytest.fixture
def wired(monkeypatch, fake_redis, fake_client_ctx):
    settings = InteractionsSettings(public_base_url="https://cb.example")
    monkeypatch.setattr(router, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(router, "interactions_settings", lambda: settings)
    # The answer door is an operation in ``operations.interactions``; it reads
    # ``client_ctx``/``interactions_settings`` from that module, so the same seams
    # are wired there too (the router patches still cover the callback/stream
    # handlers that stay in the router).
    monkeypatch.setattr(ops, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(ops, "interactions_settings", lambda: settings)
    monkeypatch.setattr(helper_module, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(helper_module, "interactions_settings", lambda: settings)
    monkeypatch.setattr(helper_module.secrets, "token_urlsafe", lambda n: "TKT")
    store = InteractionStore(settings.key_prefix)
    return SimpleNamespace(settings=settings, store=store, fake=fake_redis, monkeypatch=monkeypatch)


# -- request builder ---------------------------------------------------------


def make_request(method, *, path_params=None, query="", body=b"", headers=None, client=("1.2.3.4", 1111)):
    hdrs = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": method,
        "path": "/api/interactions/callback/x",
        "query_string": query.encode(),
        "headers": hdrs,
        "client": client,
        "path_params": path_params or {},
    }
    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _external_request(store, iid="i1", gid="g1", schema=None, budget=60, verifier=None) -> InteractionRequest:
    now = datetime.now(UTC)
    payload = {"url": "https://ext.example/resource"}
    if schema is not None:
        payload["schema"] = schema
    if verifier is not None:
        payload["verifier"] = verifier
    return InteractionRequest(
        interaction_id=iid,
        group_id=gid,
        question="Sign?",
        answer_format=AnswerFormat.EXTERNAL,
        format_payload=payload,
        reply_to=store.reply_key(iid),
        created_at=now,
        timeout_at=now + timedelta(seconds=budget),
    )


async def _seed(w, *, ticket="TKT", schema=None, budget=60, iid="i1", gid="g1", verifier=None) -> str:
    request = _external_request(w.store, iid, gid, schema, budget, verifier)
    await w.store.add(w.fake, request, idle_ttl=86400, ticket=ticket, ticket_ttl=budget)
    return iid


def _json(resp) -> dict:
    return json.loads(bytes(resp.body))


# -- callback POST (route 3) -------------------------------------------------


async def test_post_valid_body_wakes_caller(wired):
    task = asyncio.create_task(ask_user("Sign?", answer_format="external", link="{callback_url}", timeout=5))
    iid, _gid = await await_add_event(wired.fake, wired.store)

    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"signed":true}'))
    assert resp.status_code == 200
    assert _json(resp)["data"]["status"] == "answered"

    result = await task
    assert result == {"signed": True}
    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.response is not None
    assert state.response.answered_by == "external-callback"


async def test_post_unknown_ticket_404(wired):
    resp = await router.callback(make_request("POST", path_params={"ticket": "NOPE"}, body=b"{}"))
    assert resp.status_code == 404
    assert _json(resp) == {"error": "not found"}
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["x-content-type-options"] == "nosniff"


async def test_post_expired_ticket_404_identical(wired):
    await _seed(wired, budget=60)
    wired.fake.advance(61)  # ticket TTL elapsed
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b"{}"))
    assert resp.status_code == 404
    assert _json(resp) == {"error": "not found"}


async def test_post_state_missing_after_resolve_404(wired):
    await _seed(wired)
    # A cancel/timeout prune deletes state while the ticket lives out its TTL.
    await wired.store.prune_pending(wired.fake, "i1", "g1")
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b"{}"))
    assert resp.status_code == 404
    assert _json(resp) == {"error": "not found"}


async def test_post_answered_idempotent_200_and_ticket_survives(wired):
    await _seed(wired)
    prior = InteractionResponse(
        interaction_id="i1", answer={"x": 1}, answered_by="external-callback", answered_at=datetime.now(UTC)
    )
    await wired.store.record_answer(wired.fake, prior, "g1", reply_ttl=60, ticket="TKT", ticket_ttl=86400)

    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"y":2}'))
    assert resp.status_code == 200
    assert _json(resp)["data"]["status"] == "already_answered"
    # The ticket still resolves (never deleted).
    assert await wired.store.resolve_ticket(wired.fake, "TKT") == "i1"


@pytest.mark.parametrize("body", [b"[1,2]", b'"scalar"', b"not json at all"])
async def test_post_non_object_body_400(wired, body):
    await _seed(wired)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=body))
    assert resp.status_code == 400
    # Caller stays blocked: the interaction is still pending.
    state = await wired.store.get_state(wired.fake, "i1")
    assert state is not None
    assert state.status == "pending"


async def test_post_schema_invalid_then_valid(wired):
    schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}}
    task = asyncio.create_task(
        ask_user("Sign?", answer_format="external", link="{callback_url}", schema=schema, timeout=5)
    )
    await await_add_event(wired.fake, wired.store)

    bad = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"nope":1}'))
    assert bad.status_code == 400
    assert not task.done()

    good = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"x":7}'))
    assert good.status_code == 200
    assert await task == {"x": 7}


async def test_post_oversized_body_413(wired):
    wired.settings.callback_max_body_bytes = 10
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b"x" * 100))
    assert resp.status_code == 413


async def test_post_oversized_query_413(wired):
    wired.settings.callback_max_body_bytes = 10
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, query="a=" + "z" * 100))
    assert resp.status_code == 413


async def test_post_large_body_small_content_length_413(wired):
    wired.settings.callback_max_body_bytes = 10
    # A lying/absent Content-Length must not bypass the ACTUAL-byte cap.
    resp = await router.callback(
        make_request("POST", path_params={"ticket": "TKT"}, body=b"x" * 100, headers={"content-length": "2"})
    )
    assert resp.status_code == 413


async def test_post_empty_body_uses_query_params(wired):
    task = asyncio.create_task(ask_user("Sign?", answer_format="external", link="{callback_url}", timeout=5))
    await await_add_event(wired.fake, wired.store)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, query="a=1&tag=x&tag=y"))
    assert resp.status_code == 200
    assert await task == {"a": "1", "tag": ["x", "y"]}


async def test_post_body_wins_over_query(wired):
    task = asyncio.create_task(ask_user("Sign?", answer_format="external", link="{callback_url}", timeout=5))
    await await_add_event(wired.fake, wired.store)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, query="a=1", body=b'{"b":2}'))
    assert resp.status_code == 200
    assert await task == {"b": 2}


async def test_post_deeply_nested_body_400_not_500(wired):
    # A deeply-nested JSON OBJECT (not an array) isolates the PARSE-step recursion
    # catch: it would parse to a dict if json.loads didn't blow the recursion limit
    # first, so it can't 400 via the non-object guard — only via the parse except.
    await _seed(wired)
    body = (b'{"a":' * 6000) + b"1" + (b"}" * 6000)  # under the 64KiB cap
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=body))
    assert resp.status_code == 400


async def test_post_recursive_schema_400_not_500(wired):
    # A self-referential schema recurses to the body's depth; a deeply-nested body
    # under the size cap must yield 400, not a 500 from the validator blowup.
    schema = {"type": "object", "properties": {"a": {"$ref": "#"}}}
    await _seed(wired, schema=schema)
    nested: dict = {}
    cur = nested
    for _ in range(3000):
        cur["a"] = {}
        cur = cur["a"]
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=json.dumps(nested).encode()))
    assert resp.status_code == 400


async def test_post_json_headers_nosniff_nostore(wired):
    await _seed(wired)
    schema = {"type": "object", "required": ["x"]}
    req2 = _external_request(wired.store, "i2", "g2", schema)
    await wired.store.add(wired.fake, req2, idle_ttl=86400, ticket="TK2", ticket_ttl=60)
    for ticket, body, expect in [("TKT", b"{}", 200), ("TK2", b"{}", 400), ("NONE", b"{}", 404)]:
        resp = await router.callback(make_request("POST", path_params={"ticket": ticket}, body=body))
        assert resp.status_code == expect
        assert resp.headers["cache-control"] == "no-store"
        assert resp.headers["x-content-type-options"] == "nosniff"


# -- callback GET (route 4) --------------------------------------------------


async def test_get_pending_confirm_page_no_state_change(wired):
    task = asyncio.create_task(ask_user("Sign?", answer_format="external", link="{callback_url}", timeout=5))
    await await_add_event(wired.fake, wired.store)

    for _ in range(2):
        resp = await router.callback(make_request("GET", path_params={"ticket": "TKT"}))
        assert resp.status_code == 200
        assert b'<form method="post">' in bytes(resp.body)
        assert not task.done()  # GET never mutates

    # Finish so the task doesn't dangle.
    await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"ok":1}'))
    assert await task == {"ok": 1}


async def test_get_answered_done_page(wired):
    await _seed(wired)
    prior = InteractionResponse(
        interaction_id="i1", answer={}, answered_by="external-callback", answered_at=datetime.now(UTC)
    )
    await wired.store.record_answer(wired.fake, prior, "g1", reply_ttl=60, ticket="TKT", ticket_ttl=86400)
    resp = await router.callback(make_request("GET", path_params={"ticket": "TKT"}))
    assert resp.status_code == 200
    assert b"already been answered" in bytes(resp.body)


async def test_get_unknown_ticket_plain_404(wired):
    resp = await router.callback(make_request("GET", path_params={"ticket": "NOPE"}))
    assert resp.status_code == 404
    assert b"<form" not in bytes(resp.body)


async def test_get_state_missing_plain_404(wired):
    await _seed(wired)
    await wired.store.prune_pending(wired.fake, "i1", "g1")
    resp = await router.callback(make_request("GET", path_params={"ticket": "TKT"}))
    assert resp.status_code == 404
    assert b"<form" not in bytes(resp.body)


async def test_get_html_is_byte_constant_and_headers(wired):
    await _seed(wired)
    resp = await router.callback(
        make_request("GET", path_params={"ticket": "TKT"}, query='x=<script>alert(1)</script>&y="><b>')
    )
    body = bytes(resp.body)
    assert b"<script>" not in body
    assert b'"><b>' not in body
    # The page is byte-for-byte the module constant — no request-derived value
    # (query, ticket, or path) is interpolated anywhere.
    assert body == router._CONFIRM_PAGE.encode()
    assert resp.headers["content-security-policy"] == "default-src 'none'; style-src 'unsafe-inline'"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "no-referrer"
    assert resp.headers["cache-control"] == "no-store"


async def test_get_single_and_multi_query_values(wired):
    task = asyncio.create_task(ask_user("Sign?", answer_format="external", link="{callback_url}", timeout=5))
    await await_add_event(wired.fake, wired.store)
    # The confirm form POSTs back to the same URL with the query string.
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, query="a=1&tag=a&tag=b"))
    assert resp.status_code == 200
    assert await task == {"a": "1", "tag": ["a", "b"]}


# -- human answer door (route 2) ---------------------------------------------


async def test_answer_unknown_interaction_404(wired):
    req = make_request("POST", path_params={"interaction_id": "ghost"}, body=b'{"answer":"hi"}')
    resp = await router.answer(req)
    assert resp.status_code == 404


async def test_answer_external_rejected_400(wired):
    await _seed(wired)
    req = make_request("POST", path_params={"interaction_id": "i1"}, body=b'{"answer":"x"}')
    resp = await router.answer(req)
    assert resp.status_code == 400
    assert "callback URL" in _json(resp)["error"]


async def test_answer_text_then_second_is_409(wired):
    now = datetime.now(UTC)
    req_model = InteractionRequest(
        interaction_id="t1",
        group_id="g",
        question="?",
        answer_format=AnswerFormat.TEXT,
        reply_to=wired.store.reply_key("t1"),
        created_at=now,
        timeout_at=now + timedelta(seconds=60),
    )
    await wired.store.add(wired.fake, req_model, idle_ttl=86400)

    first = await router.answer(make_request("POST", path_params={"interaction_id": "t1"}, body=b'{"answer":"hello"}'))
    assert first.status_code == 200
    second = await router.answer(make_request("POST", path_params={"interaction_id": "t1"}, body=b'{"answer":"again"}'))
    assert second.status_code == 409


async def test_answer_oversized_body_413(wired):
    wired.settings.callback_max_body_bytes = 10
    req = make_request("POST", path_params={"interaction_id": "ghost"}, body=b"x" * 100)
    resp = await router.answer(req)
    assert resp.status_code == 413


async def test_answer_invalid_value_400(wired):
    now = datetime.now(UTC)
    req_model = InteractionRequest(
        interaction_id="c1",
        group_id="g",
        question="?",
        answer_format=AnswerFormat.CONFIRM,
        reply_to=wired.store.reply_key("c1"),
        created_at=now,
        timeout_at=now + timedelta(seconds=60),
    )
    await wired.store.add(wired.fake, req_model, idle_ttl=86400)
    req = make_request("POST", path_params={"interaction_id": "c1"}, body=b'{"answer":"not-a-bool"}')
    resp = await router.answer(req)
    assert resp.status_code == 400


def _plain_request(store, fmt, iid="p1", gid="pg", payload=None, audience=None) -> InteractionRequest:
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
        audience=audience,
    )


@contextmanager
def _identity(*, user_id: str | None = None, owner: str | None = None) -> Iterator[None]:
    """Bind a caller identity for a door/stream call: ``owner`` set makes it a
    RESTRICTED owned key (isolated to its OWN id ``user_id`` — NOT its owner; each key
    is its own island), ``owner=None`` an unrestricted caller. Tests pass an ``owner``
    DIFFERENT from ``user_id`` so the key-own-vs-owner distinction is exercised."""
    claims: dict[str, str] = {} if owner is None else {OWNER_USER_ID_CLAIM: owner}
    uid_token = set_request_user_id(user_id) if user_id is not None else None
    claims_token = set_request_identity_claims(claims)
    try:
        yield
    finally:
        reset_request_identity_claims(claims_token)
        if uid_token is not None:
            reset_request_user_id(uid_token)


async def test_answer_text_non_string_400(wired):
    await wired.store.add(wired.fake, _plain_request(wired.store, AnswerFormat.TEXT), idle_ttl=86400)
    resp = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b'{"answer":123}'))
    assert resp.status_code == 400


async def test_answer_select_valid_and_invalid(wired):
    req = _plain_request(wired.store, AnswerFormat.SELECT, payload={"options": ["a", "b"]})
    await wired.store.add(wired.fake, req, idle_ttl=86400)
    bad = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b'{"answer":"c"}'))
    assert bad.status_code == 400
    good = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b'{"answer":"a"}'))
    assert good.status_code == 200


async def test_answer_form_branches(wired):
    schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}}
    req = _plain_request(wired.store, AnswerFormat.FORM, payload={"schema": schema})
    await wired.store.add(wired.fake, req, idle_ttl=86400)
    non_dict = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b'{"answer":"str"}'))
    assert non_dict.status_code == 400
    bad = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b'{"answer":{"y":1}}'))
    assert bad.status_code == 400
    ok = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b'{"answer":{"x":1}}'))
    assert ok.status_code == 200


async def test_answer_confirm_valid(wired):
    await wired.store.add(wired.fake, _plain_request(wired.store, AnswerFormat.CONFIRM), idle_ttl=86400)
    resp = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b'{"answer":true}'))
    assert resp.status_code == 200


async def test_answer_form_malformed_stored_schema_400(wired):
    # A stored form question whose schema is truthy-but-not-a-dict is rejected
    # loudly rather than waving the answer through.
    req = _plain_request(wired.store, AnswerFormat.FORM, payload={"schema": "not-a-dict"})
    await wired.store.add(wired.fake, req, idle_ttl=86400)
    resp = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b'{"answer":{"x":1}}'))
    assert resp.status_code == 400


async def test_answer_lost_race_is_409(wired, monkeypatch):
    await wired.store.add(wired.fake, _plain_request(wired.store, AnswerFormat.TEXT), idle_ttl=86400)

    async def _lose(*args, **kwargs):
        return False

    monkeypatch.setattr(InteractionStore, "record_answer", _lose)
    resp = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b'{"answer":"hi"}'))
    assert resp.status_code == 409


async def test_post_lost_race_is_already_answered(wired, monkeypatch):
    await _seed(wired)

    async def _lose(*args, **kwargs):
        return False

    monkeypatch.setattr(InteractionStore, "record_answer", _lose)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"ok":1}'))
    assert resp.status_code == 200
    assert _json(resp)["data"]["status"] == "already_answered"


async def test_post_three_repeated_query_values(wired):
    task = asyncio.create_task(ask_user("Sign?", answer_format="external", link="{callback_url}", timeout=5))
    await await_add_event(wired.fake, wired.store)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, query="tag=a&tag=b&tag=c"))
    assert resp.status_code == 200
    assert await task == {"tag": ["a", "b", "c"]}


async def test_answer_deeply_nested_object_400_not_500(wired):
    # The human door shares the callback door's serialization guard: a FORM answer
    # that passes a permissive schema but blows the serializer must 400, not 500.
    schema = {"type": "object"}  # permissive: validates any object in O(1)
    req = _plain_request(wired.store, AnswerFormat.FORM, payload={"schema": schema})
    await wired.store.add(wired.fake, req, idle_ttl=86400)
    nested: dict = {}
    cur = nested
    for _ in range(3000):
        cur["a"] = {}
        cur = cur["a"]
    body = json.dumps({"answer": nested}).encode()
    resp = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=body))
    assert resp.status_code == 400


async def test_answer_invalid_json_body_400(wired):
    resp = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b"not json"))
    assert resp.status_code == 400
    assert _json(resp)["error"] == "invalid JSON body"


async def test_answer_missing_answer_key_400(wired):
    resp = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b"{}"))
    assert resp.status_code == 400


# -- SSE stream (route 1) ----------------------------------------------------


async def _collect_stream(gen) -> list[str]:
    frames = []
    async for frame in gen:
        frames.append(frame)
    return frames


def _stream_request() -> Request:
    """A GET request that yields one empty receive then disconnects — the tail-driven
    stream harness (backlog + one live-tail iteration, then the disconnect ends it)."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/s",
        "query_string": b"",
        "headers": [],
        "client": ("1.2.3.4", 1),
    }
    msgs = iter([{}, {"type": "http.disconnect"}])

    async def receive():
        try:
            return next(msgs)
        except StopIteration:
            return {"type": "http.disconnect"}

    return Request(scope, receive)


class _AliveRequest:
    """A request that reports connected for exactly ``alive`` tail iterations, then
    disconnects — lets a clock-driven test step the tail a fixed number of windows."""

    def __init__(self, alive: int) -> None:
        self._alive = alive

    async def is_disconnected(self) -> bool:
        if self._alive > 0:
            self._alive -= 1
            return False
        return True


def _add_ids(frames: list[str]) -> list[str]:
    return [
        json.loads(f.split("data: ", 1)[1])["interaction_id"] for f in frames if f.startswith("event: interaction.add")
    ]


def _event_ids(frames: list[str], event: str) -> list[str]:
    prefix = f"event: {event}"
    return [json.loads(f.split("data: ", 1)[1])["interaction_id"] for f in frames if f.startswith(prefix)]


async def test_stream_backlog_add_carries_external_url(wired):
    await _seed(wired)
    req = make_request("GET")
    frames = await _collect_stream(router._stream_events(req, wired.store, wired.settings))
    add_frames = [f for f in frames if f.startswith("event: interaction.add")]
    assert add_frames, frames
    payload = json.loads(add_frames[0].split("data: ", 1)[1].strip())
    assert payload["format_payload"]["url"] == "https://ext.example/resource"
    assert any("interaction.backlog_done" in f for f in frames)


async def test_stream_route_returns_streaming_response(wired):
    from starlette.responses import StreamingResponse

    resp = await router.stream(make_request("GET"))
    assert isinstance(resp, StreamingResponse)


async def test_stream_backlog_prunes_phantom_group(wired):
    # A group in the pending index whose stream expired must be pruned, not counted.
    wired.fake._zadd(wired.store.pending_key, {"gone": 1.0})
    frames = await _collect_stream(router._stream_events(make_request("GET"), wired.store, wired.settings))
    assert "gone" not in wired.fake._zsets.get(wired.store.pending_key, {})
    assert any("interaction.backlog_done" in f for f in frames)


def _plain_request_dated(store, iid, gid, created, timeout) -> InteractionRequest:
    return InteractionRequest(
        interaction_id=iid,
        group_id=gid,
        question="?",
        answer_format=AnswerFormat.TEXT,
        reply_to=store.reply_key(iid),
        created_at=created,
        timeout_at=timeout,
    )


async def test_stream_backlog_reconciles_abandoned_past_deadline(wired):
    # A SIGKILLed waiter leaves a pending state past its deadline; the backlog must
    # NOT surface it and must reconcile the count/pending index (it self-heals). An
    # already-answered sibling is simply skipped.
    now = datetime.now(UTC)
    dead = _plain_request_dated(wired.store, "dead", "g", now - timedelta(seconds=120), now - timedelta(seconds=60))
    live = _plain_request_dated(wired.store, "live", "g", now, now + timedelta(seconds=60))
    done = _plain_request_dated(wired.store, "done", "g", now, now + timedelta(seconds=60))
    await wired.store.add(wired.fake, dead, idle_ttl=86400)
    await wired.store.add(wired.fake, live, idle_ttl=86400)
    await wired.store.add(wired.fake, done, idle_ttl=86400)
    done_resp = InteractionResponse(interaction_id="done", answer="x", answered_by="tester", answered_at=now)
    await wired.store.record_answer(wired.fake, done_resp, "g", reply_ttl=60)

    frames = await _collect_stream(router._stream_events(make_request("GET"), wired.store, wired.settings))
    add_ids = [
        json.loads(f.split("data: ", 1)[1])["interaction_id"] for f in frames if f.startswith("event: interaction.add")
    ]
    assert add_ids == ["live"]  # dead abandoned, done answered — only live surfaces

    # Reconciled: a removed event for the dead one, count decremented to the live
    # sibling (dead + done both gone from the count of 3), group still pending.
    events = await wired.fake.xrange(wired.store.events_key)
    removed_ids = [f["interaction_id"] for _id, f in events if f.get("type") == "interaction.removed"]
    assert "dead" in removed_ids
    assert wired.fake._strings[wired.store.count_key("g")] == "1"
    assert "g" in wired.fake._zsets[wired.store.pending_key]
    assert await wired.store.get_state(wired.fake, "dead") is None
    # The answered sibling is skipped, NOT pruned — it stays answered, no removed event.
    assert "done" not in removed_ids
    done_state = await wired.store.get_state(wired.fake, "done")
    assert done_state is not None
    assert done_state.status == "answered"


def _sensitive_request(store, iid="s1", gid="sg", budget=60) -> InteractionRequest:
    now = datetime.now(UTC)
    return InteractionRequest(
        interaction_id=iid,
        group_id=gid,
        question="Paste your API key",
        answer_format=AnswerFormat.TEXT,
        reply_to=store.reply_key(iid),
        created_at=now,
        timeout_at=now + timedelta(seconds=budget),
        sensitive=True,
    )


async def test_stream_backlog_add_carries_sensitive_flag(wired):
    # The sensitive flag must reach the client on the backlog add frame so the UI
    # can label the answered state — not just live in a render fixture.
    await wired.store.add(wired.fake, _sensitive_request(wired.store), idle_ttl=86400)
    frames = await _collect_stream(router._stream_events(make_request("GET"), wired.store, wired.settings))
    add_frames = [f for f in frames if f.startswith("event: interaction.add")]
    assert add_frames, frames
    payload = json.loads(add_frames[0].split("data: ", 1)[1].strip())
    assert payload["sensitive"] is True


async def test_stream_non_sensitive_add_carries_sensitive_false(wired):
    # Regression: an ordinary question serializes sensitive: false, never omitted.
    await _seed(wired)
    frames = await _collect_stream(router._stream_events(make_request("GET"), wired.store, wired.settings))
    add_frames = [f for f in frames if f.startswith("event: interaction.add")]
    assert add_frames, frames
    payload = json.loads(add_frames[0].split("data: ", 1)[1].strip())
    assert payload["sensitive"] is False


async def test_stream_tail_add_carries_sensitive_flag(wired):
    # The live tail re-fetches the state and forwards the add frame; the sensitive
    # flag must ride that frame too (added after cursor capture).
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/s",
        "query_string": b"",
        "headers": [],
        "client": ("1.2.3.4", 1),
    }
    msgs = iter([{}, {"type": "http.disconnect"}])

    async def receive():
        try:
            return next(msgs)
        except StopIteration:
            return {"type": "http.disconnect"}

    store = wired.store
    # Seed a sensitive question, then move its events out of the way so capture
    # sees an empty tail; the live tail re-fetches the pending state and forwards.
    await store.add(wired.fake, _sensitive_request(store, iid="s7", gid="sg7"), idle_ttl=86400)
    events = wired.fake._streams.pop(store.events_key, [])
    gen = router._stream_events(Request(scope, receive), store, wired.settings)
    frames: list[str] = []
    async for frame in gen:
        frames.append(frame)
        if "backlog_done" in frame:
            break
    wired.fake._streams[store.events_key] = events
    async for frame in gen:
        frames.append(frame)
    add_frames = [f for f in frames if f.startswith("event: interaction.add")]
    assert add_frames, frames
    payload = json.loads(add_frames[-1].split("data: ", 1)[1].strip())
    assert payload["interaction_id"] == "s7"
    assert payload["sensitive"] is True


async def test_stream_tail_forwards_add(wired):
    # A pending interaction whose add-event lands after cursor capture is
    # re-fetched and forwarded by the tail.
    await _seed(wired, iid="i5", gid="g5")
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/s",
        "query_string": b"",
        "headers": [],
        "client": ("1.2.3.4", 1),
    }
    msgs = iter([{}, {"type": "http.disconnect"}])

    async def receive():
        try:
            return next(msgs)
        except StopIteration:
            return {"type": "http.disconnect"}

    # Fresh store with an empty events stream at capture; then inject the add event.
    empty_settings = wired.settings
    store = InteractionStore(empty_settings.key_prefix)
    # Move the existing add-event out of the way so capture sees an empty tail.
    events = wired.fake._streams.pop(store.events_key, [])
    gen = router._stream_events(Request(scope, receive), store, empty_settings)
    frames: list[str] = []
    async for frame in gen:
        frames.append(frame)
        if "backlog_done" in frame:
            break
    # Restore the add event so the tail re-fetches and forwards it.
    wired.fake._streams[store.events_key] = events
    async for frame in gen:
        frames.append(frame)
    assert any(f.startswith("event: interaction.add") for f in frames)


# -- display-only media: the add frame -------------------------------------


_MEDIA_ITEMS: list[MediaItem] = [
    MediaItem(kind=MediaKind.IMAGE, url="https://cdn.example/p.png", caption="A product"),
    MediaItem(kind=MediaKind.LINK, url="https://shop.example/p"),
]
# The wire form is exclude_none: the caption-less link carries no ``caption`` key.
_EXPECTED_MEDIA_FRAME = [
    {"kind": "image", "url": "https://cdn.example/p.png", "caption": "A product"},
    {"kind": "link", "url": "https://shop.example/p"},
]


def _store_empty(fake_redis) -> bool:
    # All five FakeRedis stores empty — nothing was written (``_lists`` is the reply channel).
    return not (
        fake_redis._hashes or fake_redis._streams or fake_redis._zsets or fake_redis._strings or fake_redis._lists
    )


def _media_request(
    store, iid="m1", gid="mg", media: list[MediaItem] | None = None, audience=None, budget=60
) -> InteractionRequest:
    now = datetime.now(UTC)
    return InteractionRequest(
        interaction_id=iid,
        group_id=gid,
        question="Pick a product",
        answer_format=AnswerFormat.TEXT,
        reply_to=store.reply_key(iid),
        created_at=now,
        timeout_at=now + timedelta(seconds=budget),
        media=media,
        audience=audience,
    )


async def _tail_add_frames(wired, request: InteractionRequest) -> list[dict]:
    """Drive the live-tail path for ``request`` in isolation: start the stream against
    an empty backlog and add ``request`` only AFTER ``backlog_done`` — so the backlog
    phase emits nothing for it and the add frame can come ONLY from the live tail's
    XREAD (never a backlog replay). Returns the parsed add payloads."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/s",
        "query_string": b"",
        "headers": [],
        "client": ("1.2.3.4", 1),
    }
    msgs = iter([{}, {"type": "http.disconnect"}])

    async def receive():
        try:
            return next(msgs)
        except StopIteration:
            return {"type": "http.disconnect"}

    store = wired.store
    gen = router._stream_events(Request(scope, receive), store, wired.settings)
    frames: list[str] = []
    async for frame in gen:
        frames.append(frame)
        if "backlog_done" in frame:
            break
    # The add-event lands only now — after the backlog read and the cursor capture —
    # so the frame the tail forwards is unambiguously a live-tail frame, not a replay.
    await store.add(wired.fake, request, idle_ttl=86400)
    async for frame in gen:
        frames.append(frame)
    return [json.loads(f.split("data: ", 1)[1].strip()) for f in frames if f.startswith("event: interaction.add")]


async def test_stream_backlog_add_carries_media(wired):
    # The media list rides the backlog add frame as plain JSON dicts (exclude_none).
    await wired.store.add(wired.fake, _media_request(wired.store, media=_MEDIA_ITEMS), idle_ttl=86400)
    frames = await _collect_stream(router._stream_events(make_request("GET"), wired.store, wired.settings))
    add_frames = [f for f in frames if f.startswith("event: interaction.add")]
    assert add_frames, frames
    payload = json.loads(add_frames[0].split("data: ", 1)[1].strip())
    assert payload["media"] == _EXPECTED_MEDIA_FRAME


async def test_stream_tail_add_carries_media(wired):
    # The live tail re-fetches the pending state and forwards the media on the add frame.
    payloads = await _tail_add_frames(wired, _media_request(wired.store, iid="m7", gid="mg7", media=_MEDIA_ITEMS))
    assert payloads, "no add frame forwarded by the tail"
    assert payloads[-1]["interaction_id"] == "m7"
    assert payloads[-1]["media"] == _EXPECTED_MEDIA_FRAME


async def test_stream_add_without_media_omits_key(wired):
    # A question without media carries NO ``media`` key (absent, not null).
    await _seed(wired)
    frames = await _collect_stream(router._stream_events(make_request("GET"), wired.store, wired.settings))
    add_frames = [f for f in frames if f.startswith("event: interaction.add")]
    assert add_frames, frames
    payload = json.loads(add_frames[0].split("data: ", 1)[1].strip())
    assert "media" not in payload


def test_add_data_media_is_conditional(wired):
    # The add frame carries ``media`` when the request has it and omits the key otherwise.
    with_media = _media_request(wired.store, media=_MEDIA_ITEMS)
    assert router._add_data(with_media)["media"] == _EXPECTED_MEDIA_FRAME
    without_media = _media_request(wired.store)
    assert "media" not in router._add_data(without_media)


async def test_ask_user_invalid_media_raises_before_persist(wired):
    # An invalid media item fails at the InteractionRequest build, before any state
    # is written: the request never persists and the open index stays empty.
    with pytest.raises(ValidationError):
        await ask_user("q", media=[{"kind": "image", "url": "javascript:alert(1)"}], timeout=5)
    assert await wired.store.count_open(wired.fake) == 0
    assert _store_empty(wired.fake)


async def test_ask_user_media_persists_and_frame_carries_it(wired):
    # End to end through the real helper with DICT-form media (an agent emits dicts):
    # the ask persists, answers normally (media never touches the answer), and the
    # stored request's add frame carries the coerced media — the display-only round trip.
    captured: dict = {}

    async def answer_when_asked() -> None:
        iid, gid = await await_add_event(wired.fake, wired.store)
        state = await wired.store.get_state(wired.fake, iid)
        assert state is not None
        captured["frame"] = router._add_data(state.request)
        response = InteractionResponse(
            interaction_id=iid, answer="chosen", answered_by="tester", answered_at=datetime.now(UTC)
        )
        await wired.store.record_answer(wired.fake, response, gid, reply_ttl=60)

    media: list[MediaItem | dict[str, Any]] = [
        {"kind": "image", "url": "https://cdn.example/p.png", "caption": "A product"},
        {"kind": "link", "url": "https://shop.example/p"},
    ]
    answerer = asyncio.create_task(answer_when_asked())
    result = await ask_user("Pick a product", answer_format="text", media=media, timeout=5)
    await answerer

    assert result == "chosen"  # answer unchanged by the presence of media
    assert captured["frame"]["media"] == _EXPECTED_MEDIA_FRAME


async def test_restricted_stream_media_frame_filtered_by_audience(wired):
    # Media rides the frame identically under the audience filter: a restricted caller
    # sees its OWN addressed media question (media intact) and NOT another identity's.
    mine = _media_request(wired.store, iid="mine", gid="gm", media=_MEDIA_ITEMS, audience="keyA")
    other = _media_request(wired.store, iid="other", gid="go", media=_MEDIA_ITEMS, audience="bob")
    await wired.store.add(wired.fake, mine, idle_ttl=86400)
    await wired.store.add(wired.fake, other, idle_ttl=86400)
    with _identity(user_id="keyA", owner="alice"):
        frames = await _collect_stream(router._stream_events(make_request("GET"), wired.store, wired.settings))
    add_frames = [json.loads(f.split("data: ", 1)[1].strip()) for f in frames if f.startswith("event: interaction.add")]
    assert [p["interaction_id"] for p in add_frames] == ["mine"]
    assert add_frames[0]["media"] == _EXPECTED_MEDIA_FRAME


async def test_unrestricted_stream_media_frame_broadcast(wired):
    # An unrestricted operator sees a broadcast (unaddressed) media question, media intact.
    broadcast = _media_request(wired.store, iid="cast", gid="gc", media=_MEDIA_ITEMS, audience=None)
    await wired.store.add(wired.fake, broadcast, idle_ttl=86400)
    with _identity(user_id="op1", owner=None):
        frames = await _collect_stream(router._stream_events(make_request("GET"), wired.store, wired.settings))
    add_frames = [json.loads(f.split("data: ", 1)[1].strip()) for f in frames if f.startswith("event: interaction.add")]
    assert [p["interaction_id"] for p in add_frames] == ["cast"]
    assert add_frames[0]["media"] == _EXPECTED_MEDIA_FRAME


def test_reply_ttl_clamps_past_deadline(wired):
    # A question already past its deadline still gets a positive reply TTL (the
    # 1s floor) so the RPUSH/EXPIRE pair stays valid and the key dies at once.
    req = _external_request(wired.store, budget=-30)
    assert router._reply_ttl(req) == 1


async def test_stream_tail_keepalive_then_disconnect(wired):
    # An idle tail whose keepalive deadline is due emits the SSE keepalive comment,
    # and a client disconnect ends the generator instead of blocking forever. A
    # zero window makes the deadline due on the first idle iteration.
    wired.monkeypatch.setattr(router, "_KEEPALIVE_SECONDS", 0)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/s",
        "query_string": b"",
        "headers": [],
        "client": ("1.2.3.4", 1),
    }
    msgs = iter([{}, {"type": "http.disconnect"}])

    async def receive():
        try:
            return next(msgs)
        except StopIteration:
            return {"type": "http.disconnect"}

    frames = await _collect_stream(router._stream_events(Request(scope, receive), wired.store, wired.settings))
    assert frames == ["event: interaction.backlog_done\ndata: {}\n\n", ": keepalive\n\n"]


async def test_keepalive_is_deadline_driven_not_reset_by_filtered_events(wired):
    # A restricted caller's keepalive cadence is independent of other identities'
    # volume, and the keepalive is DEADLINE-driven — not emitted once per window that
    # yields nothing to this caller. Two clock-controlled tail windows, no real waits:
    #   window 1: a bob-addressed add the alice caller filters out lands BEFORE the
    #             armed deadline (clock 1000 -> 1003, deadline 1010). No keepalive may
    #             fire — a per-non-yielding-window keepalive here would leak other
    #             identities' activity timing.
    #   window 2: an idle window crosses the deadline (clock 1003 -> 1011 >= 1010).
    #             EXACTLY ONE keepalive fires — and only because the filtered window
    #             left the 1010 deadline untouched (a reset to 1013 would suppress it).
    # Asserting on the full frame list makes any spurious keepalive visible.
    clock = {"t": 1000.0}
    wired.monkeypatch.setattr(router, "_now", lambda: clock["t"])
    wired.monkeypatch.setattr(router, "_KEEPALIVE_SECONDS", 10)

    real_xread = wired.fake.xread
    advances = iter([3, 8])  # window 1 stays before the deadline; window 2 crosses it

    async def _advancing_xread(streams, block=None):
        result = await real_xread(streams, block=block)
        clock["t"] += next(advances, 0)  # model time elapsed while blocked in this window
        return result

    wired.monkeypatch.setattr(wired.fake, "xread", _advancing_xread)

    with _identity(user_id="keyA", owner="alice"):
        gen = router._stream_events(cast(Request, _AliveRequest(alive=2)), wired.store, wired.settings)
        # Empty backlog at capture -> the first frame is the backlog_done marker, which
        # also arms the deadline at _now() + 10 == 1010.
        assert "backlog_done" in await gen.__anext__()
        # A bob-addressed add lands live (after cursor capture): window 1's XREAD returns
        # a NON-empty result the restricted alice caller filters out entirely.
        await _seed_addressed(wired, "b1", "gb", "bob")
        tail = [frame async for frame in gen]
    # The filtered window emitted nothing and left the deadline at 1010; only window 2's
    # deadline crossing emits a keepalive. A dropped `_now() >= next_keepalive` guard
    # would instead fire a keepalive every non-yielding window, doubling the count.
    assert tail == [": keepalive\n\n"]


async def test_keepalive_deadline_rearmed_by_delivered_frame(wired):
    # A frame delivered to THIS caller re-arms the keepalive deadline, so an idle
    # window that follows soon after does NOT emit a keepalive. Two clock-controlled
    # tail windows, no real waits:
    #   window 1: a keyA-addressed (entitled) add lands BEFORE the armed deadline
    #             (clock 1000 -> 1008, deadline 1010). Delivering it re-arms the
    #             deadline to _now() + 10 == 1018.
    #   window 2: an idle window advances past the ORIGINAL deadline but not the
    #             re-armed one (clock 1008 -> 1012; 1010 < 1012 < 1018). No keepalive
    #             may fire — were the delivered frame not to re-arm the deadline, this
    #             window would spuriously emit one.
    # Asserting on the full frame list makes any spurious keepalive visible.
    clock = {"t": 1000.0}
    wired.monkeypatch.setattr(router, "_now", lambda: clock["t"])
    wired.monkeypatch.setattr(router, "_KEEPALIVE_SECONDS", 10)

    real_xread = wired.fake.xread
    advances = iter([8, 4])  # window 1 stays before the deadline; window 2 crosses only the original

    async def _advancing_xread(streams, block=None):
        result = await real_xread(streams, block=block)
        clock["t"] += next(advances, 0)  # model time elapsed while blocked in this window
        return result

    wired.monkeypatch.setattr(wired.fake, "xread", _advancing_xread)

    with _identity(user_id="keyA", owner="alice"):
        gen = router._stream_events(cast(Request, _AliveRequest(alive=2)), wired.store, wired.settings)
        # Empty backlog at capture -> the backlog_done marker arms the deadline at 1010.
        assert "backlog_done" in await gen.__anext__()
        # A keyA-addressed add lands live (after cursor capture): window 1's XREAD
        # returns it and the entitled keyA caller delivers it.
        await _seed_addressed(wired, "a1", "ga", "keyA")
        tail = [frame async for frame in gen]
    # Only the entitled add rides the tail; the following idle window stays silent
    # because delivering the add re-armed the deadline to 1018. Dropping the
    # `if yielded` reset would leave the deadline at 1010 and emit a spurious keepalive.
    assert len(tail) == 1, tail
    assert _add_ids(tail) == ["a1"]
    assert ": keepalive\n\n" not in tail


async def test_stream_tail_forwards_answered_and_removed(wired):
    # Events injected AFTER the cursor is captured (empty-at-capture -> "0-0") are
    # delivered by the tail: answered + removed frames both forward.
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/s",
        "query_string": b"",
        "headers": [],
        "client": ("1.2.3.4", 1),
    }
    msgs = iter([{}, {"type": "http.disconnect"}])

    async def receive():
        try:
            return next(msgs)
        except StopIteration:
            return {"type": "http.disconnect"}

    req = Request(scope, receive)
    gen = router._stream_events(req, wired.store, wired.settings)

    frames: list[str] = []
    async for frame in gen:
        frames.append(frame)
        if "backlog_done" in frame:
            break
    # Now inject the events the live tail must forward.
    wired.fake._xadd(wired.store.events_key, {"type": "interaction.answered", "interaction_id": "i9", "group_id": "g9"})
    wired.fake._xadd(wired.store.events_key, {"type": "interaction.removed", "interaction_id": "i9", "group_id": "g9"})
    async for frame in gen:
        frames.append(frame)

    assert any(f.startswith("event: interaction.answered") for f in frames)
    assert any(f.startswith("event: interaction.removed") for f in frames)


async def test_stream_tail_skips_malformed_event_without_crashing(wired):
    # A stream entry missing a required field (partial XADD / older-or-newer schema
    # / seeded frame) must be SKIPPED, not tear down the whole SSE tail: a valid
    # event queued behind it is still delivered.
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/s",
        "query_string": b"",
        "headers": [],
        "client": ("1.2.3.4", 1),
    }
    msgs = iter([{}, {"type": "http.disconnect"}])

    async def receive():
        try:
            return next(msgs)
        except StopIteration:
            return {"type": "http.disconnect"}

    gen = router._stream_events(Request(scope, receive), wired.store, wired.settings)
    frames: list[str] = []
    async for frame in gen:
        frames.append(frame)
        if "backlog_done" in frame:
            break
    # A malformed frame (missing group_id) followed by a well-formed answered frame.
    wired.fake._xadd(wired.store.events_key, {"type": "interaction.answered", "interaction_id": "bad"})
    wired.fake._xadd(wired.store.events_key, {"type": "interaction.answered", "interaction_id": "i9", "group_id": "g9"})
    async for frame in gen:
        frames.append(frame)

    answered = [f for f in frames if f.startswith("event: interaction.answered")]
    # The malformed entry was skipped; only the well-formed one surfaced.
    assert len(answered) == 1
    assert json.loads(answered[0].split("data: ", 1)[1])["interaction_id"] == "i9"


# -- audience isolation: stream filtering ------------------------------------


async def _seed_addressed(wired, iid, gid, audience) -> None:
    await wired.store.add(
        wired.fake, _plain_request(wired.store, AnswerFormat.TEXT, iid=iid, gid=gid, audience=audience), idle_ttl=86400
    )


async def test_restricted_stream_backlog_shows_only_addressed(wired):
    await _seed_addressed(wired, "a1", "ga", "keyA")
    await _seed_addressed(wired, "b1", "gb", "bob")
    await _seed_addressed(wired, "own1", "gown", "alice")  # addressed to keyA's OWNER
    await _seed_addressed(wired, "o1", "go", None)  # broadcast/operator
    with _identity(user_id="keyA", owner="alice"):
        frames = await _collect_stream(router._stream_events(make_request("GET"), wired.store, wired.settings))
    # Only keyA's OWN-addressed question — not bob's, not the unaddressed broadcast, and
    # NOT the one addressed to its owner "alice" (key-keyed, not owner-keyed).
    assert _add_ids(frames) == ["a1"]


async def test_unrestricted_stream_backlog_shows_all(wired):
    await _seed_addressed(wired, "a1", "ga", "alice")
    await _seed_addressed(wired, "o1", "go", None)
    # An unrestricted (authenticated, no owner) caller keeps today's full inbox.
    with _identity(user_id="op1", owner=None):
        frames = await _collect_stream(router._stream_events(make_request("GET"), wired.store, wired.settings))
    assert sorted(_add_ids(frames)) == ["a1", "o1"]


async def test_restricted_stream_suppresses_other_identities_answered_removed(wired):
    # keyA's own addressed question rides the backlog (entering the visible set); a
    # bob-addressed interaction is answered AND removed while keyA streams. keyA
    # receives NEITHER of bob's terminal frames (its id was never in its set) but
    # DOES receive its own terminal frame. In reality exactly one terminal event
    # fires per interaction; a1's answered frame delivers and then discards a1 from
    # the visible set, so the duplicate removed frame is fail-closed suppressed —
    # never a leak, never a dropped entitled frame.
    await _seed_addressed(wired, "a1", "ga", "keyA")
    req = _stream_request()
    with _identity(user_id="keyA", owner="alice"):
        gen = router._stream_events(req, wired.store, wired.settings)
        frames: list[str] = []
        async for frame in gen:
            frames.append(frame)
            if "backlog_done" in frame:
                break
        for iid, gid in (("b1", "gb"), ("a1", "ga")):
            wired.fake._xadd(
                wired.store.events_key, {"type": "interaction.answered", "interaction_id": iid, "group_id": gid}
            )
            wired.fake._xadd(
                wired.store.events_key, {"type": "interaction.removed", "interaction_id": iid, "group_id": gid}
            )
        async for frame in gen:
            frames.append(frame)
    # keyA's single entitled terminal (answered) is delivered; the duplicate removed
    # for the same id is suppressed after the discard, and bob's are never entitled.
    assert _event_ids(frames, "interaction.answered") == ["a1"]
    assert _event_ids(frames, "interaction.removed") == []


async def test_restricted_stream_live_add_other_identity_not_emitted(wired):
    # A live ADD (arriving after cursor capture) addressed to bob must NOT reach the
    # restricted keyA caller — the live-add path filters on the record's audience.
    req = _stream_request()
    with _identity(user_id="keyA", owner="alice"):
        gen = router._stream_events(req, wired.store, wired.settings)
        frames: list[str] = []
        async for frame in gen:
            frames.append(frame)
            if "backlog_done" in frame:
                break
        # Add bob's addressed interaction AFTER backlog_done, so its add-event rides
        # the live tail (empty at capture -> "0-0").
        await _seed_addressed(wired, "b1", "gb", "bob")
        async for frame in gen:
            frames.append(frame)
    assert _add_ids(frames) == []


async def test_restricted_stream_live_add_own_identity_emitted_and_terminal_delivered(wired):
    # A live ADD addressed to keyA IS emitted, and the live-add path adds its id to
    # the visible set — proven by the subsequent terminal frame being delivered
    # (a terminal for an id never seen on an addressed add would be suppressed).
    req = _stream_request()
    with _identity(user_id="keyA", owner="alice"):
        gen = router._stream_events(req, wired.store, wired.settings)
        frames: list[str] = []
        async for frame in gen:
            frames.append(frame)
            if "backlog_done" in frame:
                break
        # Empty backlog at capture: keyA's addressed add arrives live, then its
        # single terminal (answered) frame follows.
        await _seed_addressed(wired, "a1", "ga", "keyA")
        wired.fake._xadd(
            wired.store.events_key, {"type": "interaction.answered", "interaction_id": "a1", "group_id": "ga"}
        )
        async for frame in gen:
            frames.append(frame)
    assert _add_ids(frames) == ["a1"]
    assert _event_ids(frames, "interaction.answered") == ["a1"]


async def test_stream_add_frame_carries_no_ticket_field(wired):
    # The ticket-containment invariant: no add frame carries a ticket for ANY caller,
    # so a restricted caller consuming the stream can never obtain one — the callback
    # door (channel-only ticket delivery) stays the sole ticket-bearing surface.
    assert "ticket" not in router._add_data(_external_request(wired.store))
    assert "ticket" not in router._add_data(_plain_request(wired.store, AnswerFormat.TEXT, audience="alice"))


# -- audience isolation: the answer-door matrix ------------------------------


async def test_audience_holder_answers_addressed_200(wired):
    await _seed_addressed(wired, "ad", "gad", "keyA")
    with _identity(user_id="keyA", owner="alice"):
        resp = await router.answer(make_request("POST", path_params={"interaction_id": "ad"}, body=b'{"answer":"x"}'))
    assert resp.status_code == 200


async def test_other_restricted_answers_addressed_403(wired):
    # Addressed to keyA; a DIFFERENT key keyB (even were it same-owner) is denied.
    await _seed_addressed(wired, "ad", "gad", "keyA")
    with _identity(user_id="keyB", owner="bob"):
        resp = await router.answer(make_request("POST", path_params={"interaction_id": "ad"}, body=b'{"answer":"x"}'))
    assert resp.status_code == 403
    assert _json(resp)["error"] == "interaction is addressed to another identity"


async def test_unrestricted_answers_addressed_200(wired):
    # The operator can always unblock a stuck addressed question.
    await _seed_addressed(wired, "ad", "gad", "alice")
    with _identity(user_id="op1", owner=None):
        resp = await router.answer(make_request("POST", path_params={"interaction_id": "ad"}, body=b'{"answer":"x"}'))
    assert resp.status_code == 200


async def test_unrestricted_answers_unaddressed_200(wired):
    await _seed_addressed(wired, "un", "gun", None)
    with _identity(user_id="op1", owner=None):
        resp = await router.answer(make_request("POST", path_params={"interaction_id": "un"}, body=b'{"answer":"x"}'))
    assert resp.status_code == 200


async def test_restricted_answers_unaddressed_403(wired):
    await _seed_addressed(wired, "un", "gun", None)
    with _identity(user_id="keyA", owner="alice"):
        resp = await router.answer(make_request("POST", path_params={"interaction_id": "un"}, body=b'{"answer":"x"}'))
    assert resp.status_code == 403
    assert _json(resp)["error"] == "restricted identities may answer only interactions addressed to them"


async def test_callback_answers_addressed_external_regardless_of_audience(wired):
    # The unauth ticket callback is NOT audience-gated: the ticket is the capability.
    req = _external_request(wired.store, iid="ext", gid="gext").model_copy(update={"audience": "alice"})
    await wired.store.add(wired.fake, req, idle_ttl=86400, ticket="TKT", ticket_ttl=60)
    # No caller identity bound — an external answerer holds only the ticket.
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"approved":true}'))
    assert resp.status_code == 200
    assert _json(resp)["data"]["status"] == "answered"


async def test_answer_no_auth_records_namespaced_sentinel(wired):
    # With access control disabled no caller identity is bound; the recorded
    # ``answered_by`` is the reserved namespaced sentinel, not a bare "anonymous"
    # that could collide with a real user id.
    await wired.store.add(wired.fake, _plain_request(wired.store, AnswerFormat.TEXT), idle_ttl=86400)
    resp = await router.answer(make_request("POST", path_params={"interaction_id": "p1"}, body=b'{"answer":"hi"}'))
    assert resp.status_code == 200
    state = await wired.store.get_state(wired.fake, "p1")
    assert state is not None
    assert state.response is not None
    assert state.response.answered_by == "system:no-auth"
    assert state.response.answered_by == ops._NO_AUTH_ANSWERED_BY
    # The sentinel is namespaced (``system:`` prefix), so it can't be a real id.
    assert ops._NO_AUTH_ANSWERED_BY.startswith("system:")


# -- callback door verifier binding (Door 2) ---------------------------------


class _FakeVerifier:
    """Records each verify call; optionally raises. Order-tracks against a shared
    list so a test can pin verify-before-record."""

    post_only = True

    def __init__(self, *, raise_exc=None, order=None):
        self._raise = raise_exc
        self._order = order
        self.calls: list = []

    async def verify(self, body, headers, config):
        if self._order is not None:
            self._order.append("verify")
        self.calls.append((body, dict(headers), config))
        if self._raise is not None:
            raise self._raise


@pytest.fixture
def verifier_registry():
    """The process app's verifier registry with ``tai42_app`` bound to it; the
    callback route resolves verifiers from here. Cleared after the test."""
    from tai42_contract.app import tai42_app

    from tai42_skeleton.app.instance import build_app

    app = build_app()
    tai42_app.bind(app)
    reg = app._webhook_verifier_registry
    try:
        yield reg
    finally:
        reg.reset()


_BINDING = {"name": "prov", "config": {"secret_env": "WH"}}


async def test_callback_bound_verifies_before_record(wired, verifier_registry):
    order: list[str] = []
    verifier_registry.register("prov", _FakeVerifier(order=order))
    await _seed(wired, verifier=_BINDING)

    # The route builds its own store instance, so spy on the class method: the
    # ticket/state reads share the fake redis, only record_answer is order-tracked.
    orig_record = router.InteractionStore.record_answer

    async def spy_record(self, *a, **k):
        order.append("record")
        return await orig_record(self, *a, **k)

    wired.monkeypatch.setattr(router.InteractionStore, "record_answer", spy_record)

    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"signed":true}'))
    assert resp.status_code == 200
    # Verify ran, and BEFORE the answer was recorded.
    assert order == ["verify", "record"]


async def test_callback_bound_verify_failure_401_ticket_unconsumed(wired, verifier_registry):
    from tai42_contract.webhooks import WebhookVerificationError

    verifier_registry.register("prov", _FakeVerifier(raise_exc=WebhookVerificationError("bad sig")))
    iid = await _seed(wired, verifier=_BINDING)

    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b"x"))
    assert resp.status_code == 401
    assert _json(resp)["error"] == "webhook verification failed"
    # Nothing recorded — the question stays pending.
    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.status == "pending"
    # The ticket is NOT consumed — it still resolves, so a legitimate retry works.
    assert await wired.store.resolve_ticket(wired.fake, "TKT") == iid


async def test_callback_bound_missing_verifier_500(wired, verifier_registry):
    # Binding names a verifier that is not registered -> fail closed 500.
    await _seed(wired, verifier={"name": "ghost", "config": {}})
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b"x"))
    assert resp.status_code == 500


async def test_callback_get_bound_question_no_confirm_page(wired, verifier_registry):
    verifier_registry.register("prov", _FakeVerifier())
    await _seed(wired, verifier=_BINDING)
    resp = await router.callback(make_request("GET", path_params={"ticket": "TKT"}))
    # A verifier-bound question is server-to-server only: no confirm page.
    assert resp.status_code == 404
    assert b"<form" not in bytes(resp.body)
    assert b"server-to-server" in bytes(resp.body)


async def test_unbound_external_callback_unchanged(wired):
    # No verifier binding -> today's ticket-only behavior, GET serves the confirm.
    await _seed(wired)
    resp = await router.callback(make_request("GET", path_params={"ticket": "TKT"}))
    assert resp.status_code == 200
    assert bytes(resp.body) == router._CONFIRM_PAGE.encode()


async def test_callback_post_only_empty_body_query_answer_denied(wired, verifier_registry):
    # A body-signature (post_only) verifier signs only the body. An empty-body POST
    # with a query-string answer must be DENIED — a replayed signature over an empty
    # body must never let ``?approved=true`` inject an answer.
    verifier_registry.register("prov", _FakeVerifier())  # post_only=True
    iid = await _seed(wired, verifier=_BINDING)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, query="approved=true"))
    assert resp.status_code == 400
    assert _json(resp)["error"] == router._POST_ONLY_EMPTY_BODY_DENY
    # Nothing recorded: the question stays pending, the ticket is unconsumed.
    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.status == "pending"
    assert await wired.store.resolve_ticket(wired.fake, "TKT") == iid


async def test_callback_non_post_only_empty_body_query_answer_accepted(wired, verifier_registry):
    # A header-signature (post_only=False) verifier keeps the empty-body/query path:
    # the verifier passes over the empty body and the query answer is accepted.
    header_verifier = _FakeVerifier()
    header_verifier.post_only = False
    verifier_registry.register("prov", header_verifier)
    iid = await _seed(wired, verifier=_BINDING)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, query="approved=true"))
    assert resp.status_code == 200
    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.status == "answered"
    assert state.response is not None
    assert state.response.answer == {"approved": "true"}


async def test_callback_post_only_signed_body_accepted(wired, verifier_registry):
    # A passing post_only verifier over a NON-empty (signed) body is accepted — the
    # answer rides the signed body, so post_only only blocks the empty-body path.
    verifier_registry.register("prov", _FakeVerifier())  # post_only=True
    iid = await _seed(wired, verifier=_BINDING)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"signed":true}'))
    assert resp.status_code == 200
    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.status == "answered"
    assert state.response is not None
    assert state.response.answer == {"signed": True}


def test_add_frame_strips_verifier_and_flags_server_verified(wired):
    req = _external_request(wired.store, "iv", "gv", verifier=_BINDING)
    data = router._add_data(req)
    # The verifier config is stripped from the client frame...
    assert "verifier" not in (data["format_payload"] or {})
    # ...and replaced with a server_verified flag.
    assert data["server_verified"] is True
    # A non-verified external question carries no server_verified key.
    plain = router._add_data(_external_request(wired.store, "ip", "gp"))
    assert "server_verified" not in plain


# -- callback POST: channel-delivered typed (non-external) formats -------------


def _typed_request(store, fmt, iid="t1", gid="tg1", options=None, budget=60, channel="chan") -> InteractionRequest:
    now = datetime.now(UTC)
    payload = {"options": options} if options is not None else None
    return InteractionRequest(
        interaction_id=iid,
        group_id=gid,
        question="Q?",
        answer_format=fmt,
        format_payload=payload,
        reply_to=store.reply_key(iid),
        created_at=now,
        timeout_at=now + timedelta(seconds=budget),
        channel=channel,
    )


async def _seed_typed(w, fmt, *, options=None, ticket="TKT") -> str:
    request = _typed_request(w.store, fmt, options=options)
    await w.store.add(w.fake, request, idle_ttl=86400, ticket=ticket, ticket_ttl=60)
    return "t1"


async def test_callback_text_records_typed_str(wired):
    iid = await _seed_typed(wired, AnswerFormat.TEXT)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"answer": "yes"}'))
    assert resp.status_code == 200
    assert _json(resp)["data"]["status"] == "answered"
    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.response is not None
    assert state.response.answer == "yes"  # the TYPED str, not the envelope dict


async def test_callback_confirm_records_typed_bool(wired):
    iid = await _seed_typed(wired, AnswerFormat.CONFIRM)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"answer": false}'))
    assert resp.status_code == 200
    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.response is not None
    assert state.response.answer is False


async def test_callback_select_records_chosen_option(wired):
    iid = await _seed_typed(wired, AnswerFormat.SELECT, options=["red", "blue"])
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"answer": "blue"}'))
    assert resp.status_code == 200
    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.response is not None
    assert state.response.answer == "blue"


async def test_callback_select_rejects_unknown_option(wired):
    await _seed_typed(wired, AnswerFormat.SELECT, options=["red", "blue"])
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"answer": "green"}'))
    assert resp.status_code == 400
    state = await wired.store.get_state(wired.fake, "t1")
    assert state is not None
    assert state.status == "pending"  # caller stays blocked


async def test_callback_typed_wrong_type_400(wired):
    await _seed_typed(wired, AnswerFormat.TEXT)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"answer": 7}'))
    assert resp.status_code == 400
    assert _json(resp) == {"error": "answer must be a string"}


async def test_callback_typed_missing_answer_key_400(wired):
    await _seed_typed(wired, AnswerFormat.TEXT)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b'{"text": "yes"}'))
    assert resp.status_code == 400
    assert _json(resp) == {"error": "body must contain 'answer'"}


async def test_callback_confirm_empty_body_records_true(wired):
    # The GET-confirm page's form POSTs an empty body: an affirmative tap.
    iid = await _seed_typed(wired, AnswerFormat.CONFIRM)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b""))
    assert resp.status_code == 200
    state = await wired.store.get_state(wired.fake, iid)
    assert state is not None
    assert state.response is not None
    assert state.response.answer is True


@pytest.mark.parametrize(("fmt", "options"), [(AnswerFormat.TEXT, None), (AnswerFormat.SELECT, ["red", "blue"])])
async def test_callback_non_confirm_empty_body_400(wired, fmt, options):
    await _seed_typed(wired, fmt, options=options)
    resp = await router.callback(make_request("POST", path_params={"ticket": "TKT"}, body=b""))
    assert resp.status_code == 400
    assert _json(resp) == {"error": "body must contain 'answer'"}


@pytest.mark.parametrize(("fmt", "options"), [(AnswerFormat.TEXT, None), (AnswerFormat.SELECT, ["red", "blue"])])
async def test_get_typed_value_format_serves_reply_page(wired, fmt, options):
    # These formats reject the confirm form's empty-body POST, so GET serves
    # the byte-constant awaiting-reply page — no form, no action that can fail.
    await _seed_typed(wired, fmt, options=options)
    resp = await router.callback(make_request("GET", path_params={"ticket": "TKT"}))
    assert resp.status_code == 200
    body = bytes(resp.body)
    assert body == router._REPLY_PAGE.encode()
    assert b"<form" not in body
    assert resp.headers["content-security-policy"] == "default-src 'none'; style-src 'unsafe-inline'"
    state = await wired.store.get_state(wired.fake, "t1")
    assert state is not None
    assert state.status == "pending"  # GET never mutates


async def test_get_channel_confirm_serves_confirm_page(wired):
    # confirm keeps the tappable page: its empty-body POST records True.
    await _seed_typed(wired, AnswerFormat.CONFIRM)
    resp = await router.callback(make_request("GET", path_params={"ticket": "TKT"}))
    assert resp.status_code == 200
    assert bytes(resp.body) == router._CONFIRM_PAGE.encode()


async def test_add_data_channel_is_conditional(wired):
    # The add frame carries ``channel`` when the request has one and omits the
    # key otherwise — the exact ``server_verified`` conditional-key shape.
    with_channel = _typed_request(wired.store, AnswerFormat.TEXT)
    assert router._add_data(with_channel)["channel"] == "chan"
    without_channel = _external_request(wired.store)
    assert "channel" not in router._add_data(without_channel)
