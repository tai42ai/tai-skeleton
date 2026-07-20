"""Op-level oracles for the hook-management operations.

Covers ``list_hooks`` / ``register_hook`` / ``unregister_hook``: ``register_hook``
takes FLAT ``HookParams`` fields and returns ``{"registered", "name"}``;
``unregister_hook`` returns ``{"removed", "name"}`` and raises ``NotFoundError`` on a
miss; ``list_hooks`` returns ``{"items", "total", "topic_verifiers"}``. The
remaining verifier-binding operations and the destructive projection carry their
own coverage.
"""

from __future__ import annotations

import pytest
from tai42_contract.manifest import ApiToolsConfig

from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai42_skeleton.hooks.settings import HooksSettings
from tai42_skeleton.operations import (
    BadRequestError,
    NotFoundError,
    OperationRegistry,
    operation_metadata_of,
)
from tai42_skeleton.operations import hooks as hooks_ops
from tai42_skeleton.operations.projection import project_operations


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> InMemoryHooksManager:
    """Pin the hook operations to a single real in-memory manager instance."""
    mgr = InMemoryHooksManager(HooksSettings())
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: mgr)
    return mgr


class _FakeVerifier:
    post_only = False

    async def verify(self, body, headers, config):
        return None


@pytest.fixture
def registry():
    """The process app's webhook-verifier registry bound to ``tai42_app`` so
    ``set_topic_verifier`` resolves bind-time verifier names; cleared after."""
    from tai42_contract.app import tai42_app

    from tai42_skeleton.app.instance import build_app

    app = build_app()
    tai42_app.bind(app)
    reg = app._webhook_verifier_registry
    try:
        yield reg
    finally:
        reg.reset()


# -- register / list / unregister


async def test_register_then_list_and_unregister(manager: InMemoryHooksManager) -> None:
    # Flat fields in, {"registered", "name"} out.
    assert await hooks_ops.register_hook(name="h1", topic="orders", tool="ship", condition='.status == "paid"') == {
        "registered": True,
        "name": "h1",
    }

    # {"items", "total", "topic_verifiers"} — an enveloped listing, not a bare dict.
    listed = await hooks_ops.list_hooks()
    assert listed["total"] == 1
    assert {item["name"] for item in listed["items"]} == {"h1"}
    assert listed["topic_verifiers"] == {}

    by_topic = await hooks_ops.list_hooks(topic="orders")
    assert by_topic["items"][0]["tool"] == "ship"
    assert (await hooks_ops.list_hooks(topic="other"))["items"] == []

    # {"removed", "name"} out.
    assert await hooks_ops.unregister_hook(name="h1") == {"removed": True, "name": "h1"}
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_register_rejects_invalid_jq_condition(manager: InMemoryHooksManager) -> None:
    # The manager's jq compile failure is a client-input 400, not a raw crash.
    with pytest.raises(BadRequestError, match="not valid jq"):
        await hooks_ops.register_hook(name="bad", topic="t", tool="noop", condition="this is ( not jq")


async def test_register_rejects_invalid_flat_params(manager: InMemoryHooksManager) -> None:
    # A flat-field validation failure reaching the op from the MCP/CLI edge (no HTTP
    # extractor) is a loud BadRequestError, never an escaped ValidationError.
    with pytest.raises(BadRequestError, match="invalid hook params"):
        await hooks_ops.register_hook(name="x", topic="t", tool="noop", condition=5)  # type: ignore[arg-type]


async def test_unregister_missing_is_not_found(manager: InMemoryHooksManager) -> None:
    # A miss is a loud 404, not a False return.
    with pytest.raises(NotFoundError, match="hook not found"):
        await hooks_ops.unregister_hook(name="nope")


# -- verifier bindings --------------------------------------------------------


async def test_set_topic_verifier_binds_and_lists(manager: InMemoryHooksManager, registry) -> None:
    registry.register("prov", _FakeVerifier())
    assert await hooks_ops.set_topic_verifier(topic="orders", verifier="prov", config={"k": "v"}) == {
        "topic": "orders",
        "verifier": "prov",
    }
    # The bound verifier now rides list_hooks' topic_verifiers.
    listed = await hooks_ops.list_hooks()
    assert listed["topic_verifiers"] == {"orders": {"verifier": "prov", "config": {"k": "v"}}}


async def test_set_topic_verifier_unknown_name_is_bad_request(manager: InMemoryHooksManager, registry) -> None:
    with pytest.raises(BadRequestError, match="unknown webhook verifier"):
        await hooks_ops.set_topic_verifier(topic="orders", verifier="nope")


async def test_delete_topic_verifier_removes_and_missing_is_404(manager: InMemoryHooksManager, registry) -> None:
    registry.register("prov", _FakeVerifier())
    await hooks_ops.set_topic_verifier(topic="orders", verifier="prov")
    assert await hooks_ops.delete_topic_verifier(topic="orders") == {"removed": True, "topic": "orders"}
    with pytest.raises(NotFoundError, match="no verifier bound to topic"):
        await hooks_ops.delete_topic_verifier(topic="orders")


async def test_list_verifiers_returns_sorted_names(registry) -> None:
    registry.register("zebra", _FakeVerifier())
    registry.register("alpha", _FakeVerifier())
    assert await hooks_ops.list_verifiers() == ["alpha", "zebra"]


# -- projection ---------------------------------------------------------------


def test_hook_mutations_project_with_destructive_hint() -> None:
    # register_hook / set_topic_verifier are destructive — off the default surface
    # but includable; when projected they carry the destructiveHint annotation.
    reg = OperationRegistry()
    reg.register(operation_metadata_of(hooks_ops.register_hook))
    reg.register(operation_metadata_of(hooks_ops.set_topic_verifier))
    reg.register(operation_metadata_of(hooks_ops.list_hooks))

    class _Rec:
        def __init__(self) -> None:
            self.registered: dict[str, dict] = {}

        def tool(self, *, force, name, tags, annotations):
            self.registered[name] = {"annotations": annotations}
            return lambda fn: fn

    class _App:
        def __init__(self) -> None:
            self.tools = _Rec()

    app = _App()
    names = project_operations(app, ApiToolsConfig(expose_destructive=True), registry=reg)
    assert {"register_hook", "set_topic_verifier", "list_hooks"} <= set(names)
    assert app.tools.registered["register_hook"]["annotations"].destructiveHint is True
    assert app.tools.registered["set_topic_verifier"]["annotations"].destructiveHint is True
    # list_hooks is a read — no destructiveHint annotation.
    assert app.tools.registered["list_hooks"]["annotations"] is None
