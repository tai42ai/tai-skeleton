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


# -- trigger-link operations --------------------------------------------------


@pytest.fixture
def gate_off(monkeypatch: pytest.MonkeyPatch):
    """Access control OFF — the identity-less caller shape (created_by → None)."""
    from types import SimpleNamespace

    monkeypatch.setattr(hooks_ops, "access_control_settings", lambda: SimpleNamespace(enable=False))


@pytest.fixture
def capture_create(monkeypatch: pytest.MonkeyPatch):
    """Capture the kwargs the op forwards to the trigger-link store, returning a
    canned create result."""
    seen: dict = {}

    async def _create(**kwargs):
        seen.update(kwargs)
        return {
            "name": kwargs["name"] or "trg-link-deadbeef",
            "trigger_path": "/trigger/tok",
            "token": "tok",
            "topic": kwargs["topic"],
            "expires_at": None,
        }

    monkeypatch.setattr(hooks_ops.trigger_links, "create_trigger_link", _create)
    return seen


async def test_create_trigger_link_ambient_created_by_gate_on(monkeypatch, capture_create) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(hooks_ops, "access_control_settings", lambda: SimpleNamespace(enable=True))
    monkeypatch.setattr(hooks_ops, "get_current_user_id", lambda: "alice")
    await hooks_ops.create_trigger_link(topic="t", name="n", ttl_seconds=None, tool_kwargs=None)
    assert capture_create["created_by"] == "alice"


async def test_create_trigger_link_identity_less_created_by_null(gate_off, capture_create) -> None:
    await hooks_ops.create_trigger_link(topic="t", name="n", ttl_seconds=None, tool_kwargs=None)
    assert capture_create["created_by"] is None


async def test_create_trigger_link_gate_on_unset_identity_raises(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(hooks_ops, "access_control_settings", lambda: SimpleNamespace(enable=True))
    monkeypatch.setattr(hooks_ops, "get_current_user_id", lambda: None)
    # A refactor must not quietly return None under gate-on — it RAISES (a 500).
    with pytest.raises(RuntimeError, match="caller user id is unset"):
        await hooks_ops.create_trigger_link(topic="t", name="n", ttl_seconds=None, tool_kwargs=None)


@pytest.mark.parametrize(
    ("status", "error_name"),
    [(400, "BadRequestError"), (409, "ConflictError"), (501, "NotSupportedError")],
)
async def test_create_trigger_link_status_mapping(monkeypatch, gate_off, status, error_name) -> None:
    from tai42_skeleton.hooks.trigger_links import TriggerLinkError

    async def _raise(**kwargs):
        raise TriggerLinkError(status, "boom")

    from tai42_skeleton.operations.errors import OperationError

    monkeypatch.setattr(hooks_ops.trigger_links, "create_trigger_link", _raise)
    with pytest.raises(OperationError) as ei:
        await hooks_ops.create_trigger_link(topic="t", name="n", ttl_seconds=None, tool_kwargs=None)
    assert type(ei.value).__name__ == error_name
    assert ei.value.status == status


async def test_delete_trigger_link_404_and_501(monkeypatch, gate_off) -> None:
    from tai42_skeleton.hooks.trigger_links import TriggerLinkError

    async def _raise404(name):
        raise TriggerLinkError(404, "unknown trigger link")

    monkeypatch.setattr(hooks_ops.trigger_links, "revoke_trigger_link", _raise404)
    with pytest.raises(NotFoundError):
        await hooks_ops.delete_trigger_link(name="x")

    async def _ok(name):
        return None

    monkeypatch.setattr(hooks_ops.trigger_links, "revoke_trigger_link", _ok)
    assert await hooks_ops.delete_trigger_link(name="x") == {"removed": True, "name": "x"}


def test_mutating_trigger_ops_are_authority_changing_and_excluded_from_projection() -> None:
    from tai42_skeleton.operations.projection import is_tier2

    create_meta = operation_metadata_of(hooks_ops.create_trigger_link)
    delete_meta = operation_metadata_of(hooks_ops.delete_trigger_link)
    list_meta = operation_metadata_of(hooks_ops.list_trigger_links)
    # Direct flag pin (survives any future projection-tier reshuffle).
    assert create_meta.authority_changing is True
    assert delete_meta.authority_changing is True
    assert list_meta.authority_changing is False
    # And the default surface excludes both mutating ops.
    assert is_tier2(create_meta)
    assert is_tier2(delete_meta)

    reg = OperationRegistry()
    reg.register(create_meta)
    reg.register(delete_meta)
    reg.register(list_meta)

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
    assert "create_trigger_link" not in names
    assert "delete_trigger_link" not in names
    assert "list_trigger_links" in names  # an ordinary read tool
