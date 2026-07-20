"""The ``ask_external`` transformer tool extension: wrap-time validation, the
composed concrete signature, a clean bind through the real apply site (proving
the transformer schema rule), and an end-to-end answer delivered via the helper.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tai_contract.interactions import InteractionResponse

from tai_skeleton.exceptions.exceptions import TaiValidationError
from tai_skeleton.extensions.builtin.ask_external import ask_external
from tai_skeleton.interactions import InteractionStore
from tai_skeleton.interactions import helper as helper_module
from tai_skeleton.interactions.settings import InteractionsSettings
from tests._helpers import await_add_event

_BUILTIN_MODULE = "tai_skeleton.extensions.builtin.ask_external"


@pytest.fixture(autouse=True)
def _clean_server():
    """Clear the singleton FastMCP server's tools around each test — it outlives
    one ``app_context``, so a tool a prior apply-site test bound would collide
    with this test's bind under ``on_duplicate="error"``."""
    from tai_skeleton.app.instance import app

    async def _clear() -> None:
        provider = app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    yield
    asyncio.run(_clear())


async def _answer_first_add(fake_redis, store: InteractionStore, answer) -> None:
    """Wait for the add event, then record ``answer`` for that interaction."""
    iid, gid = await await_add_event(fake_redis, store)
    await store.record_answer(
        fake_redis,
        InteractionResponse(
            interaction_id=iid,
            answer=answer,
            answered_by="external-callback",
            answered_at=datetime.now(UTC),
        ),
        gid,
        reply_ttl=60,
    )


# -- wrap-time validation ----------------------------------------------------


def test_wrap_requires_callback_url():
    async def no_cb(*, city: str) -> str:
        return "https://x.example"

    with pytest.raises(TaiValidationError, match="must accept callback_url"):
        ask_external(no_cb, "no_cb", "desc")


@pytest.mark.parametrize("bad", ["question", "answer_schema", "timeout"])
def test_wrap_rejects_control_param_collision(bad):
    async def collide(*, callback_url: str, **_kw) -> str:
        return "https://x.example"

    # Rebuild the signature so the tool declares the colliding control param.
    params = [
        inspect.Parameter("callback_url", inspect.Parameter.KEYWORD_ONLY, annotation=str),
        inspect.Parameter(bad, inspect.Parameter.KEYWORD_ONLY, annotation=str),
    ]
    collide.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]

    with pytest.raises(TaiValidationError, match=f"'{bad}'"):
        ask_external(collide, "collide", "desc")


def test_wrap_rejects_positional_only_param():
    def tool(document, /, *, callback_url):  # positional-only document
        return "https://x.example"

    with pytest.raises(TaiValidationError, match="cannot use parameter 'document'"):
        ask_external(tool, "tool", "desc")


def test_wrap_rejects_var_positional():
    def tool(*args, callback_url):
        return "https://x.example"

    with pytest.raises(TaiValidationError, match="cannot use parameter 'args'"):
        ask_external(tool, "tool", "desc")


def test_wrap_rejects_var_keyword():
    def tool(*, callback_url, **kwargs):
        return "https://x.example"

    with pytest.raises(TaiValidationError, match="cannot use parameter 'kwargs'"):
        ask_external(tool, "tool", "desc")


def test_wrap_rejects_positional_only_callback_url():
    def tool(callback_url, /):  # positional-only callback_url would crash at call time
        return "https://x.example"

    with pytest.raises(TaiValidationError, match="cannot use parameter 'callback_url'"):
        ask_external(tool, "tool", "desc")


# -- composed signature ------------------------------------------------------


def test_composed_signature_is_concrete_without_callback_url():
    async def tool(*, document: str, callback_url: str) -> str:
        return "https://x.example"

    composed = ask_external(tool, "tool", "desc")
    params = inspect.signature(composed).parameters

    assert "callback_url" not in params
    assert "document" in params  # the tool's own input survives
    assert {"question", "answer_schema", "timeout"} <= set(params)
    # ``verifier`` is author-bound via config — it must NEVER be an LLM-facing
    # tool param the agent could set, drop, or forge.
    assert "verifier" not in params
    # Concrete, not a bare (*args, **kwargs) passthrough.
    assert not all(p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for p in params.values())
    assert composed.__name__ == "tool_ask_external"


# -- apply-site bind ---------------------------------------------------------


def test_binds_as_transformer_branch_at_apply_site():
    from tai_skeleton.app.instance import app
    from tai_skeleton.manifest import Manifest

    manifest = Manifest.model_validate(
        {
            "extensions_modules": [_BUILTIN_MODULE],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.extensions._fixtures.tools_external",
                    "include": ["make_signature"],
                    "extensions": {"make_signature": [["ask_external"]]},
                }
            ],
        }
    )

    async def run() -> None:
        async with app.app_context(manifest):
            tools = await app.tools.get_tools()
            assert {"make_signature", "make_signature_ask_external"} <= set(tools)

    asyncio.run(run())


def test_binds_with_author_bound_config_at_apply_site():
    # The manifest dict combo element ``{"name", "config"}`` binds the verifier as
    # author config; it must thread through tool_extensions -> binding -> the
    # ask_external factory and produce the branch tool, with NO verifier param.
    from tai_skeleton.app.instance import app
    from tai_skeleton.manifest import Manifest

    manifest = Manifest.model_validate(
        {
            "extensions_modules": [_BUILTIN_MODULE],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.extensions._fixtures.tools_external",
                    "include": ["make_signature"],
                    "extensions": {
                        "make_signature": [
                            [{"name": "ask_external", "config": {"verifier": {"name": "github", "config": {}}}}]
                        ]
                    },
                }
            ],
        }
    )

    async def run() -> None:
        async with app.app_context(manifest):
            tools = await app.tools.get_tools()
            assert {"make_signature", "make_signature_ask_external"} <= set(tools)
            branch = tools["make_signature_ask_external"]
            assert "verifier" not in (branch.parameters.get("properties", {}))

    asyncio.run(run())


def test_factory_accepts_config_detects_config_param():
    from tai_skeleton.extensions.registry import factory_accepts_config

    def three(func, name, desc): ...
    def positional_config(func, name, desc, config=None): ...
    def keyword_only_config(func, name, desc, *, config): ...
    def fourth_not_config(func, name, desc, extra): ...

    assert factory_accepts_config(three) is False
    assert factory_accepts_config(positional_config) is True
    # A keyword-only ``config`` is detected too (the apply site passes it by keyword).
    assert factory_accepts_config(keyword_only_config) is True
    # A fourth parameter not named ``config`` is not config — the factory is agnostic.
    assert factory_accepts_config(fourth_not_config) is False


def test_config_on_config_agnostic_extension_rejected_at_apply_site():
    # A three-argument (config-agnostic) factory is called without config; binding a
    # non-empty config to it would silently drop the author's intent, so the apply
    # site raises loudly.
    from tai_skeleton.app.instance import app
    from tai_skeleton.manifest import Manifest

    manifest = Manifest.model_validate(
        {
            "extensions_modules": ["tests.extensions._fixtures.ext_noconfig"],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.extensions._fixtures.tools_external",
                    "include": ["make_signature"],
                    "extensions": {"make_signature": [[{"name": "noconfig", "config": {"foo": 1}}]]},
                }
            ],
        }
    )

    async def run() -> None:
        with pytest.raises(ValueError, match="does not accept config"):
            async with app.app_context(manifest):
                await app.tools.get_tools()

    asyncio.run(run())


# -- end-to-end --------------------------------------------------------------


async def test_end_to_end_answer_delivered(monkeypatch, fake_redis, fake_client_ctx):
    settings = InteractionsSettings(public_base_url="https://cb.example")
    monkeypatch.setattr(helper_module, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(helper_module, "interactions_settings", lambda: settings)
    store = InteractionStore(settings.key_prefix)

    async def make_url(*, city: str, callback_url: str) -> str:
        return f"https://ext.example/{city}?cb={callback_url}"

    composed = ask_external(make_url, "make_url", "desc")

    task = asyncio.create_task(_answer_first_add(fake_redis, store, {"signed": True}))
    result = await composed(city="NYC", question="Sign the document?", timeout=5)
    await task

    assert result == {"signed": True}


async def test_sync_wrapped_tool_works(monkeypatch, fake_redis, fake_client_ctx):
    # A synchronous wrapped tool (returns a plain str, not a coroutine) must work
    # too — the wrapper awaits only when the result is awaitable.
    settings = InteractionsSettings(public_base_url="https://cb.example")
    monkeypatch.setattr(helper_module, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(helper_module, "interactions_settings", lambda: settings)
    store = InteractionStore(settings.key_prefix)

    def make_url(*, callback_url):  # sync tool
        return f"https://ext.example/go?cb={callback_url}"

    composed = ask_external(make_url, "make_url", "desc")

    task = asyncio.create_task(_answer_first_add(fake_redis, store, "ok"))
    result = await composed(question="Approve?", timeout=5)
    await task
    assert result == "ok"


async def test_author_bound_verifier_lands_in_format_payload(monkeypatch, fake_redis, fake_client_ctx):
    # The AUTHOR-BOUND ``verifier`` (supplied as extension config at build time)
    # must be threaded through ask_user -> _build_payload into the persisted
    # external ``format_payload`` — regardless of any agent input, since it is not
    # an LLM-facing param.
    settings = InteractionsSettings(public_base_url="https://cb.example")
    monkeypatch.setattr(helper_module, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(helper_module, "interactions_settings", lambda: settings)
    store = InteractionStore(settings.key_prefix)

    # ask_user validates the verifier name against the registry at bind time; stub
    # the lookup so the passthrough plumbing can be exercised without a live app.
    class _Registry:
        def get(self, name: str) -> object:
            if name == "github":
                return object()
            raise KeyError(name)

    monkeypatch.setattr(helper_module, "tai_app", SimpleNamespace(webhook_verifiers=_Registry()))

    async def make_url(*, callback_url: str) -> str:
        return f"https://ext.example/go?cb={callback_url}"

    binding = {"name": "github", "config": {"secret_env": "GH"}}
    # The verifier is closed over from author config — never a call-time param.
    composed = ask_external(make_url, "make_url", "desc", {"verifier": binding})
    assert "verifier" not in inspect.signature(composed).parameters

    async def _capture_then_answer():
        iid, gid = await await_add_event(fake_redis, store)
        state = await store.get_state(fake_redis, iid)
        assert state is not None
        # The author-bound verifier rides the persisted external format_payload.
        assert (state.request.format_payload or {}).get("verifier") == binding
        await store.record_answer(
            fake_redis,
            InteractionResponse(
                interaction_id=iid, answer="ok", answered_by="external-callback", answered_at=datetime.now(UTC)
            ),
            gid,
            reply_ttl=60,
        )

    task = asyncio.create_task(_capture_then_answer())
    result = await composed(question="Sign?", timeout=5)
    await task
    assert result == "ok"


def test_no_config_binds_no_verifier():
    # A plain-string combo element (no config) closes over no verifier — the
    # composed tool exposes no verifier param and rides the open ticket path.
    async def make_url(*, callback_url: str) -> str:
        return f"https://ext.example/go?cb={callback_url}"

    composed = ask_external(make_url, "make_url", "desc")
    assert "verifier" not in inspect.signature(composed).parameters
    # An explicit empty config is equivalent to no config.
    composed_empty = ask_external(make_url, "make_url", "desc", {})
    assert "verifier" not in inspect.signature(composed_empty).parameters


def test_unknown_config_key_rejected_loudly():
    async def make_url(*, callback_url: str) -> str:
        return f"https://ext.example/go?cb={callback_url}"

    with pytest.raises(TaiValidationError, match="unknown key"):
        ask_external(make_url, "make_url", "desc", {"verfier": {"name": "github"}})


async def test_verifier_rejected_at_ask_time_when_malformed_or_unknown(monkeypatch, fake_redis, fake_client_ctx):
    # A non-dict verifier, or a name that does not resolve in the registry, is
    # rejected LOUDLY at ask time — never stashed to silently degrade the question
    # into an open, unverified one at the callback door.
    from tai_skeleton.interactions.helper import ask_user

    settings = InteractionsSettings(public_base_url="https://cb.example")
    monkeypatch.setattr(helper_module, "client_ctx", fake_client_ctx)
    monkeypatch.setattr(helper_module, "interactions_settings", lambda: settings)

    class _Empty:
        def get(self, name: str) -> object:
            raise KeyError(name)

    monkeypatch.setattr(helper_module, "tai_app", SimpleNamespace(webhook_verifiers=_Empty()))

    with pytest.raises(ValueError, match="verifier must be a dict"):
        await ask_user(
            "Sign?", answer_format="external", link="https://x/{callback_url}", verifier=cast(Any, "not-a-dict")
        )
    with pytest.raises(ValueError, match="unknown webhook verifier"):
        await ask_user("Sign?", answer_format="external", link="https://x/{callback_url}", verifier={"name": "nope"})
