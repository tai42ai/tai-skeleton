"""The ``ask_user`` external-format flow in the helper: the link/options/schema
combo validation, up-front schema rejection (before the builder runs), the
template/callable link resolution, the exact callback URL, ``public_base_url``
scheme validation, the ``max_concurrent`` guard, cancel-cleanup, and the
timeout-path prune.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import cast

import pytest
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from tai42_contract.interactions import InteractionResponse

from tai42_skeleton.interactions import InteractionStore, ask_user
from tai42_skeleton.interactions import helper as helper_module
from tai42_skeleton.interactions.helper import InteractionLimitError, InteractionTimeoutError
from tai42_skeleton.interactions.settings import InteractionsSettings
from tests._helpers import await_add_event


def _wire(monkeypatch, fake_redis, fake_client_ctx, **settings_kw) -> InteractionsSettings:
    settings_kw.setdefault("public_base_url", "https://cb.example")
    settings = InteractionsSettings(**settings_kw)
    monkeypatch.setattr(helper_module, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(helper_module, "interactions_settings", lambda: settings)
    return settings


def _empty(fake_redis) -> bool:
    return not (fake_redis._hashes or fake_redis._streams or fake_redis._zsets or fake_redis._strings)


# -- combo validation (fails before any redis / builder work) ----------------


async def test_external_requires_link():
    with pytest.raises(ValueError, match="requires a link"):
        await ask_user("q", answer_format="external")


async def test_link_forbidden_without_external():
    with pytest.raises(ValueError, match="only valid with answer_format 'external'"):
        await ask_user("q", link="{callback_url}")


async def test_verifier_forbidden_without_external():
    with pytest.raises(ValueError, match="verifier is only valid with answer_format 'external'"):
        await ask_user("q", verifier={"name": "github", "config": {}})


async def test_external_rejects_options(monkeypatch, fake_redis, fake_client_ctx):
    _wire(monkeypatch, fake_redis, fake_client_ctx)
    with pytest.raises(ValueError, match="does not accept options"):
        await ask_user("q", answer_format="external", link="{callback_url}", options=["a"])


async def test_bad_schema_rejected_before_builder(monkeypatch, fake_redis, fake_client_ctx):
    _wire(monkeypatch, fake_redis, fake_client_ctx)
    called: list[str] = []

    async def builder(url: str) -> str:
        called.append(url)
        return "https://external.example/resource"

    with pytest.raises(ValueError, match="pydantic model or a JSON-schema dict"):
        await ask_user("q", answer_format="external", link=builder, schema=cast("dict", 123), timeout=60)
    assert called == []  # up-front validation ran before external work
    assert _empty(fake_redis)


# -- link resolution ---------------------------------------------------------


async def _answer_plain(fake_redis, store, answer):
    iid, gid = await await_add_event(fake_redis, store)
    await store.record_answer(
        fake_redis,
        InteractionResponse(interaction_id=iid, answer=answer, answered_by="t", answered_at=datetime.now(UTC)),
        gid,
        reply_ttl=60,
    )


async def _answer_with_capture(fake_redis, store, captured, answer):
    iid, gid = await await_add_event(fake_redis, store)
    state = await store.get_state(fake_redis, iid)
    assert state is not None
    captured["url"] = state.request.format_payload["url"]
    captured["interaction_id"] = iid
    await store.record_answer(
        fake_redis,
        InteractionResponse(interaction_id=iid, answer=answer, answered_by="ext", answered_at=datetime.now(UTC)),
        gid,
        reply_ttl=60,
    )


async def test_template_substitution_preserves_other_braces(monkeypatch, fake_redis, fake_client_ctx):
    settings = _wire(monkeypatch, fake_redis, fake_client_ctx)
    store = InteractionStore(settings.key_prefix)
    captured: dict = {}
    template = "https://sign.example/doc/{callback_url}?keep={other}"

    answerer = asyncio.create_task(_answer_with_capture(fake_redis, store, captured, {"ok": True}))
    result = await ask_user("q", answer_format="external", link=template, timeout=5)
    await answerer

    assert result == {"ok": True}
    assert captured["url"].startswith("https://sign.example/doc/https://cb.example/api/interactions/callback/")
    # The unrelated ``{other}`` brace survived (replace, not format).
    assert captured["url"].endswith("?keep={other}")


async def test_missing_placeholder_errors(monkeypatch, fake_redis, fake_client_ctx):
    _wire(monkeypatch, fake_redis, fake_client_ctx)
    with pytest.raises(ValueError, match=r"must contain \{callback_url\}"):
        await ask_user("q", answer_format="external", link="https://no-placeholder.example", timeout=5)
    assert _empty(fake_redis)


async def test_builder_receives_exact_callback_url(monkeypatch, fake_redis, fake_client_ctx):
    settings = _wire(monkeypatch, fake_redis, fake_client_ctx)
    monkeypatch.setattr(helper_module.secrets, "token_urlsafe", lambda n: "TICKET123")
    store = InteractionStore(settings.key_prefix)
    seen: list[str] = []

    async def builder(url: str) -> str:
        seen.append(url)
        return "https://external.example/go"

    answerer = asyncio.create_task(_answer_with_capture(fake_redis, store, {}, "done"))
    await ask_user("q", answer_format="external", link=builder, timeout=5)
    await answerer

    assert seen == ["https://cb.example/api/interactions/callback/TICKET123"]


async def test_builder_exception_propagates_nothing_persisted(monkeypatch, fake_redis, fake_client_ctx):
    _wire(monkeypatch, fake_redis, fake_client_ctx)

    async def builder(url: str) -> str:
        raise RuntimeError("resource creation failed")

    with pytest.raises(RuntimeError, match="resource creation failed"):
        await ask_user("q", answer_format="external", link=builder, timeout=5)
    assert _empty(fake_redis)


async def test_builder_non_url_return_errors(monkeypatch, fake_redis, fake_client_ctx):
    _wire(monkeypatch, fake_redis, fake_client_ctx)

    async def builder(url: str) -> str:
        return "ftp://not-http.example"

    with pytest.raises(ValueError, match="must return an http"):
        await ask_user("q", answer_format="external", link=builder, timeout=5)
    assert _empty(fake_redis)


# -- public_base_url validation ----------------------------------------------


async def test_missing_public_base_url_raises(monkeypatch, fake_redis, fake_client_ctx):
    _wire(monkeypatch, fake_redis, fake_client_ctx, public_base_url=None)
    with pytest.raises(RuntimeError, match="INTERACTIONS_PUBLIC_BASE_URL"):
        await ask_user("q", answer_format="external", link="{callback_url}", timeout=5)


def test_http_non_localhost_rejected(monkeypatch, fake_redis, fake_client_ctx):
    # The settings validator rejects a non-TLS URL at construction — before any
    # ask_user call can ever mint a callback URL from it.
    with pytest.raises(PydanticValidationError, match="must be https"):
        _wire(monkeypatch, fake_redis, fake_client_ctx, public_base_url="http://evil.example")


@pytest.mark.parametrize("base", ["http://localhost", "http://127.0.0.1"])
async def test_http_localhost_accepted(monkeypatch, fake_redis, fake_client_ctx, base):
    settings = _wire(monkeypatch, fake_redis, fake_client_ctx, public_base_url=base)
    store = InteractionStore(settings.key_prefix)
    answerer = asyncio.create_task(_answer_plain(fake_redis, store, "ok"))
    result = await ask_user("q", answer_format="external", link="{callback_url}", timeout=5)
    await answerer
    assert result == "ok"


# -- concurrency guard -------------------------------------------------------


async def test_max_concurrent_admits_below_limit(monkeypatch, fake_redis, fake_client_ctx):
    settings = _wire(monkeypatch, fake_redis, fake_client_ctx, max_concurrent=2)
    store = InteractionStore(settings.key_prefix)
    # One already-open member (future score).
    fake_redis._zadd(store.open_key, {"other": 9_999_999_999_999.0})

    answerer = asyncio.create_task(_answer_plain(fake_redis, store, "hi"))
    result = await ask_user("q", timeout=5)
    await answerer
    assert result == "hi"


async def test_max_concurrent_trips_at_limit(monkeypatch, fake_redis, fake_client_ctx):
    settings = _wire(monkeypatch, fake_redis, fake_client_ctx, max_concurrent=1)
    store = InteractionStore(settings.key_prefix)
    fake_redis._zadd(store.open_key, {"other": 9_999_999_999_999.0})

    with pytest.raises(InteractionLimitError) as exc:
        await ask_user("q", timeout=5)
    # The message carries the open count and the setting name.
    assert "1" in str(exc.value)
    assert "max_concurrent" in str(exc.value)


async def test_max_concurrent_purges_stale_then_admits(monkeypatch, fake_redis, fake_client_ctx):
    settings = _wire(monkeypatch, fake_redis, fake_client_ctx, max_concurrent=1)
    store = InteractionStore(settings.key_prefix)
    # A stale member (past deadline) must be purged and not count against the cap.
    fake_redis._zadd(store.open_key, {"stale": 1.0})

    answerer = asyncio.create_task(_answer_plain(fake_redis, store, "ok"))
    result = await ask_user("q", timeout=5)
    await answerer
    assert result == "ok"


async def test_max_concurrent_atomic_under_burst(monkeypatch, fake_redis, fake_client_ctx):
    # A concurrent burst past the cap: the reserve-and-check is atomic, so EXACTLY
    # ``max_concurrent`` calls are admitted (they block on the reply, then time out)
    # and every excess call is refused with ``InteractionLimitError`` — no
    # check-then-act overshoot admits more than the cap.
    _wire(monkeypatch, fake_redis, fake_client_ctx, max_concurrent=2)

    results = await asyncio.gather(
        *(ask_user("q", timeout=0.1) for _ in range(5)),
        return_exceptions=True,
    )

    refused = [e for e in results if isinstance(e, InteractionLimitError)]
    admitted_then_timed_out = [e for e in results if isinstance(e, InteractionTimeoutError)]
    assert len(refused) == 3  # 5 launched minus 2 admitted
    assert len(admitted_then_timed_out) == 2  # the 2 admitted block, then time out


# -- cancel + timeout cleanup ------------------------------------------------


async def test_cancel_prunes_and_reraises(monkeypatch, fake_redis, fake_client_ctx):
    settings = _wire(monkeypatch, fake_redis, fake_client_ctx)
    store = InteractionStore(settings.key_prefix)

    task = asyncio.create_task(ask_user("q", timeout=60))
    iid, _gid = await await_add_event(fake_redis, store)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await store.get_state(fake_redis, iid) is None
    assert await store.count_open(fake_redis) == 0


async def test_timeout_prunes(monkeypatch, fake_redis, fake_client_ctx):
    settings = _wire(monkeypatch, fake_redis, fake_client_ctx)
    store = InteractionStore(settings.key_prefix)

    with pytest.raises(InteractionTimeoutError):
        await ask_user("q", timeout=0.05)

    assert await store.count_open(fake_redis) == 0


class _Form(BaseModel):
    name: str


async def test_external_pydantic_schema_stored(monkeypatch, fake_redis, fake_client_ctx):
    settings = _wire(monkeypatch, fake_redis, fake_client_ctx)
    store = InteractionStore(settings.key_prefix)
    captured: dict = {}

    async def grab():
        iid, gid = await await_add_event(fake_redis, store)
        state = await store.get_state(fake_redis, iid)
        assert state is not None
        captured["payload"] = state.request.format_payload
        resp = InteractionResponse(
            interaction_id=iid, answer={"name": "x"}, answered_by="e", answered_at=datetime.now(UTC)
        )
        await store.record_answer(fake_redis, resp, gid, reply_ttl=60)

    answerer = asyncio.create_task(grab())
    await ask_user("q", answer_format="external", link="{callback_url}", schema=_Form, timeout=5)
    await answerer

    assert captured["payload"]["schema"] == _Form.model_json_schema()
    assert "url" in captured["payload"]
