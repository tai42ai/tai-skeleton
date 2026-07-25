"""Op-level oracles for the hook-management operations.

Covers ``list_hooks`` / ``register_hook`` / ``unregister_hook``: ``register_hook``
takes FLAT ``HookParams`` fields and returns ``{"registered", "name"}``;
``unregister_hook`` returns ``{"removed", "name"}`` and raises ``NotFoundError`` on a
miss; ``list_hooks`` returns ``{"items", "total", "topic_verifiers", "trigger_auth"}``. The
remaining verifier-binding operations and the destructive projection carry their
own coverage.
"""

from __future__ import annotations

import pytest
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM, OWNER_USER_ID_CLAIM
from tai42_contract.access_control.models import AccessPolicy
from tai42_contract.manifest import ApiToolsConfig

from tai42_skeleton.access_control.policy import policy_is_empty
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.authz import execution
from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai42_skeleton.hooks.settings import HooksSettings
from tai42_skeleton.operations import (
    BadRequestError,
    ForbiddenError,
    NotFoundError,
    OperationFailed,
    OperationRegistry,
    operation_metadata_of,
)
from tai42_skeleton.operations import _authority as authority
from tai42_skeleton.operations import hooks as hooks_ops
from tai42_skeleton.operations.projection import project_operations
from tai42_skeleton.template import TemplateNotFoundError


@pytest.fixture(autouse=True)
def bind_gate_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Access control OFF by default here, so each test exercises the operation itself.

    The gate lives at two seams (``operations._authority`` and ``authz.execution``);
    both must be flipped together or a record write sees a half-open gate."""
    monkeypatch.setattr(authority, "access_control_settings", lambda: AccessControlSettings(enable=False))
    monkeypatch.setattr(execution, "access_control_settings", lambda: AccessControlSettings(enable=False))


# A condition-free ``"*"`` policy that is not an owned key: the admin discriminator.
_ADMIN_POLICY = AccessPolicy(scopes=["*"])


def _gate_on(
    monkeypatch: pytest.MonkeyPatch,
    *,
    caller_id: str | None,
    policies: dict[str, AccessPolicy],
) -> None:
    """Turn the gate ON over a canned policy store; an absent id resolves to the empty
    policy an unknown key has. Every half of the gate reads that one store."""
    monkeypatch.setattr(authority, "access_control_settings", lambda: AccessControlSettings(enable=True))
    monkeypatch.setattr(execution, "access_control_settings", lambda: AccessControlSettings(enable=True))
    monkeypatch.setattr(authority, "get_current_user_id", lambda: caller_id)

    # Stamp the per-mint ``KEY_FINGERPRINT_CLAIM`` a real mint writes; an EMPTY policy is
    # left bare — it stands for an absent key, refused before any fingerprint is read.
    stamped = {
        uid: (
            pol
            if policy_is_empty(pol) or KEY_FINGERPRINT_CLAIM in (pol.policy_data or {})
            else pol.model_copy(update={"policy_data": {**(pol.policy_data or {}), KEY_FINGERPRINT_CLAIM: "fp"}})
        )
        for uid, pol in policies.items()
    }

    class _Enforcer:
        def __init__(self, settings: AccessControlSettings) -> None:
            self.settings = settings

        async def current_policy_version(self) -> int:
            return 0

        async def get_policy(self, user_id: str) -> AccessPolicy:
            return stamped.get(user_id, AccessPolicy(scopes=[]))

        async def get_policy_at(self, user_id: str, _version: int) -> AccessPolicy:
            return stamped.get(user_id, AccessPolicy(scopes=[]))

    monkeypatch.setattr(authority, "PolicyEnforcer", _Enforcer)
    monkeypatch.setattr(execution, "PolicyEnforcer", _Enforcer)


def _owned_by(owner: str, *, condition: str | None = None) -> AccessPolicy:
    """The stored policy of a live key owned by ``owner``, optionally carrying an
    authorization ``condition``."""
    return AccessPolicy(scopes=["hooks"], policy_data={OWNER_USER_ID_CLAIM: owner}, condition=condition)


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
    ``set_topic_verifier`` resolves bind-time verifier names; the registry is cleared
    and the previous binding restored after."""
    from tai42_contract.app import tai42_app

    from tai42_skeleton.app.instance import build_app

    app = build_app()
    reg = app._webhook_verifier_registry
    with tai42_app.bound(app):
        try:
            yield reg
        finally:
            reg.reset()


# -- register / list / unregister


async def test_register_then_list_and_unregister(manager: InMemoryHooksManager) -> None:
    # Flat fields in, {"registered", "name"} out.
    assert await hooks_ops.register_hook(
        name="h1", topic="orders", tool="ship", execution_key="k-fire", condition='.status == "paid"'
    ) == {
        "registered": True,
        "name": "h1",
    }

    # Enveloped listing: {"items", "total", "topic_verifiers", "trigger_auth"}.
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
        await hooks_ops.register_hook(
            name="bad", topic="t", tool="noop", execution_key="k-fire", condition="this is ( not jq"
        )


async def test_register_rejects_invalid_flat_params(manager: InMemoryHooksManager) -> None:
    # A flat-field validation failure reaching the op from the MCP/CLI edge (no HTTP
    # extractor) is a loud BadRequestError, never an escaped ValidationError.
    with pytest.raises(BadRequestError, match="invalid hook params"):
        await hooks_ops.register_hook(name="x", topic="t", tool="noop", execution_key="k-fire", condition=5)  # type: ignore[arg-type]


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


async def test_listed_trigger_auth_is_derived_from_the_live_verifier_bindings(
    manager: InMemoryHooksManager, registry
) -> None:
    # ``public`` until a verifier is bound, then ``verifier`` — derived at every read
    # from the bindings just fetched, never stored (the binding mutates independently).
    registry.register("prov", _FakeVerifier())
    await hooks_ops.register_hook(name="h", topic="orders", tool="ship", execution_key="k-fire")
    assert (await hooks_ops.list_hooks())["trigger_auth"] == {"orders": "public"}

    await hooks_ops.set_topic_verifier(topic="orders", verifier="prov")
    assert (await hooks_ops.list_hooks())["trigger_auth"] == {"orders": "verifier"}

    # A topic carrying a binding but no hook yet is still reported — the door exists.
    await hooks_ops.set_topic_verifier(topic="alerts", verifier="prov")
    assert (await hooks_ops.list_hooks())["trigger_auth"] == {"alerts": "verifier", "orders": "verifier"}

    await hooks_ops.delete_topic_verifier(topic="orders")
    assert (await hooks_ops.list_hooks())["trigger_auth"]["orders"] == "public"


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
    # register_hook and set_topic_verifier are both destructive AND authority_changing,
    # so both are tier-2: default-excluded whatever expose_destructive says, includable
    # by name, and carrying destructiveHint once projected. list_hooks is the read.
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
    assert set(names) == {"list_hooks"}
    # list_hooks is a read — no destructiveHint annotation.
    assert app.tools.registered["list_hooks"]["annotations"] is None

    included = _App()
    names = project_operations(
        included,
        ApiToolsConfig(expose_destructive=True, include=["register_hook", "set_topic_verifier"]),
        registry=reg,
    )
    assert {"register_hook", "set_topic_verifier"} <= set(names)
    assert included.tools.registered["register_hook"]["annotations"].destructiveHint is True
    assert included.tools.registered["set_topic_verifier"]["annotations"].destructiveHint is True


# -- trigger-link operations --------------------------------------------------


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
    _gate_on(
        monkeypatch,
        caller_id="alice",
        policies={"alice": _ADMIN_POLICY, "k-fire": _ADMIN_POLICY},
    )
    await hooks_ops.create_trigger_link(topic="t", execution_key="k-fire", name="n", ttl_seconds=None, tool_kwargs=None)
    assert capture_create["created_by"] == "alice"


async def test_create_trigger_link_identity_less_created_by_null(capture_create) -> None:
    await hooks_ops.create_trigger_link(topic="t", execution_key="k-fire", name="n", ttl_seconds=None, tool_kwargs=None)
    assert capture_create["created_by"] is None


async def test_create_trigger_link_gate_on_unset_identity_raises(monkeypatch) -> None:
    _gate_on(monkeypatch, caller_id=None, policies={})
    # A refactor must not quietly return None under gate-on — it RAISES (a typed 500).
    with pytest.raises(OperationFailed, match="internal authority-resolution failure"):
        await hooks_ops.create_trigger_link(
            topic="t", execution_key="k-fire", name="n", ttl_seconds=None, tool_kwargs=None
        )


@pytest.mark.parametrize(
    ("status", "error_name"),
    [(400, "BadRequestError"), (409, "ConflictError"), (501, "NotSupportedError")],
)
async def test_create_trigger_link_status_mapping(monkeypatch, status, error_name) -> None:
    from tai42_skeleton.hooks.trigger_links import TriggerLinkError

    async def _raise(**kwargs):
        raise TriggerLinkError(status, "boom")

    from tai42_skeleton.operations.errors import OperationError

    monkeypatch.setattr(hooks_ops.trigger_links, "create_trigger_link", _raise)
    with pytest.raises(OperationError) as ei:
        await hooks_ops.create_trigger_link(
            topic="t", execution_key="k-fire", name="n", ttl_seconds=None, tool_kwargs=None
        )
    assert type(ei.value).__name__ == error_name
    assert ei.value.status == status


async def test_delete_trigger_link_404_and_501(monkeypatch) -> None:
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


# -- the execution-key bind gate ---------------------------------------------
#
# Pass-role (caller owns the key, or is admin) plus a token-free-evaluable scan of the
# key's and its owner's condition. Both run BEFORE the write, so every refusal below
# must also leave the store untouched.


class _FakeResourceManager:
    """Condition renderer reached through ``tai42_app``: inline ``content`` renders to
    itself; an unknown ``template_id`` raises, as an unresolvable condition does."""

    def __init__(self, templates: dict[str, str]) -> None:
        self._templates = templates

    async def render_by_id_or_content(self, *, content, template_id, kwargs) -> str:
        if template_id is not None:
            if template_id not in self._templates:
                raise TemplateNotFoundError(f"no such template: {template_id!r}")
            return self._templates[template_id]
        return content or ""


@pytest.fixture
def renderer(request: pytest.FixtureRequest):
    """Bind a minimal app carrying the condition renderer onto ``tai42_app``."""
    from types import SimpleNamespace

    from tai42_contract.app import tai42_app

    templates = getattr(request, "param", {})
    app = SimpleNamespace(storage=SimpleNamespace(resource_manager=_FakeResourceManager(templates)))
    with tai42_app.bound(app):
        yield app


async def test_register_hook_non_admin_binding_an_owned_key_is_allowed(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    _gate_on(
        monkeypatch,
        caller_id="alice",
        policies={"alice": AccessPolicy(scopes=["hooks"]), "k-fire": _owned_by("alice")},
    )
    assert await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-fire") == {
        "registered": True,
        "name": "h",
    }


async def test_register_hook_non_admin_binding_a_key_it_does_not_own_is_forbidden(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    _gate_on(
        monkeypatch,
        caller_id="alice",
        policies={"alice": AccessPolicy(scopes=["hooks"]), "k-bob": _owned_by("bob")},
    )
    with pytest.raises(ForbiddenError, match="only bind your own identity or an execution key you own"):
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-bob")
    # Refused before the upsert: nothing was stored.
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_register_hook_non_admin_binding_an_unknown_key_is_the_same_refusal(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    # Absent key and someone else's key answer the SAME 403 with the SAME message: this
    # door must not be an existence oracle for api keys.
    _gate_on(monkeypatch, caller_id="alice", policies={"alice": AccessPolicy(scopes=["hooks"])})
    with pytest.raises(ForbiddenError, match="only bind your own identity or an execution key you own"):
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="ghost")
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_register_hook_admin_binds_a_key_it_does_not_own(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    # Admin passes pass-role without owning the key; the key still has to be one a fire
    # could run under (exists, enabled, owner enabled).
    _gate_on(
        monkeypatch,
        caller_id="root",
        policies={"root": _ADMIN_POLICY, "k-bob": _owned_by("bob"), "bob": AccessPolicy(scopes=["hooks"])},
    )
    assert (await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-bob"))["registered"]


async def test_register_hook_non_admin_binding_its_own_identity_is_allowed(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    # Self-bind is a branch of its own: a top-level key carries no owner claim pointing
    # at itself, so the ownership comparison never admits it.
    _gate_on(monkeypatch, caller_id="alice", policies={"alice": AccessPolicy(scopes=["hooks"])})
    assert (await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="alice"))["registered"]


async def test_register_hook_binding_a_provisioned_but_empty_key_is_refused(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    # An empty policy row grants nothing, so every fire under it dies in the identity
    # build; the bind door asks the same question and refuses rather than store it.
    _gate_on(monkeypatch, caller_id="root", policies={"root": _ADMIN_POLICY, "k-empty": AccessPolicy(scopes=[])})
    with pytest.raises(NotFoundError, match="user not found"):
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-empty")
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_register_hook_admin_binding_an_unknown_key_is_not_found(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    # Refused for an admin too, as a 404 — an admin may already enumerate keys, so the
    # existence disclosure costs nothing.
    _gate_on(monkeypatch, caller_id="root", policies={"root": _ADMIN_POLICY})
    with pytest.raises(NotFoundError, match="user not found"):
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="ghost")
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_register_hook_binding_a_disabled_key_is_refused(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    # A disabled key is neither absent nor empty, and its fires still die: the bind door
    # refuses the whole set the fire refuses, not existence alone.
    _gate_on(
        monkeypatch,
        caller_id="root",
        policies={"root": _ADMIN_POLICY, "k-off": AccessPolicy(scopes=["hooks"], policy_data={"disabled": True})},
    )
    with pytest.raises(BadRequestError, match="execution key 'k-off' is disabled"):
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-off")
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_register_hook_binding_a_key_whose_owner_is_disabled_is_refused(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    # An owned key's fire is attenuated by its OWNER's policy, so a disabled owner is the
    # same dead record. The owner is named only because the caller is an admin.
    _gate_on(
        monkeypatch,
        caller_id="root",
        policies={
            "root": _ADMIN_POLICY,
            "k-bob": _owned_by("bob"),
            "bob": AccessPolicy(scopes=["hooks"], policy_data={"disabled": True}),
        },
    )
    with pytest.raises(BadRequestError, match="owner 'bob' of execution key 'k-bob' is disabled"):
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-bob")
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_register_hook_withholds_a_THIRD_principals_id_from_a_non_admin(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    # Pass-role admits the self-bind, but the refusal is about alice's OWNER: the defect
    # is disclosed so alice can act on it, the owner's id is not.
    _gate_on(
        monkeypatch,
        caller_id="alice",
        policies={"alice": _owned_by("carol"), "carol": AccessPolicy(scopes=[])},
    )
    with pytest.raises(BadRequestError, match="the owner of execution key 'alice' has no policy") as raised:
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="alice")
    assert "carol" not in str(raised.value)
    assert (await hooks_ops.list_hooks())["items"] == []


@pytest.mark.parametrize(
    "condition",
    [
        '.identity.description == "ops"',
        ".identity | keys",
        '.. | select(. == "x")',
        '.["identity"].team',
    ],
)
async def test_register_hook_refuses_a_key_whose_condition_needs_absent_claims(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager, renderer, condition: str
) -> None:
    _gate_on(
        monkeypatch,
        caller_id="root",
        policies={"root": _ADMIN_POLICY, "k-fire": AccessPolicy(scopes=["hooks"], condition=condition)},
    )
    with pytest.raises(BadRequestError, match="unusable at a fire"):
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-fire")
    assert (await hooks_ops.list_hooks())["items"] == []


@pytest.mark.parametrize(
    "condition",
    ['.identity.owner_user_id == "alice"', '.sub != "banned"', '.request.method == "POST"', '.policy.tier == "gold"'],
)
async def test_register_hook_accepts_a_key_evaluable_without_a_token(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager, renderer, condition: str
) -> None:
    _gate_on(
        monkeypatch,
        caller_id="root",
        policies={"root": _ADMIN_POLICY, "k-fire": AccessPolicy(scopes=["hooks"], condition=condition)},
    )
    assert (await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-fire"))["registered"]


async def test_register_hook_refuses_a_key_whose_OWNER_condition_needs_absent_claims(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager, renderer
) -> None:
    # The owner's condition is a second pass at every fire, so it is scanned too.
    owned = AccessPolicy(scopes=["hooks"], policy_data={OWNER_USER_ID_CLAIM: "alice"})
    _gate_on(
        monkeypatch,
        caller_id="root",
        policies={
            "root": _ADMIN_POLICY,
            "k-fire": owned,
            "alice": AccessPolicy(scopes=["hooks"], condition='.identity.mfa == "yes"'),
        },
    )
    with pytest.raises(BadRequestError, match="'alice' is unusable at a fire"):
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-fire")
    assert (await hooks_ops.list_hooks())["items"] == []


@pytest.mark.parametrize("renderer", [{"known": ".sub"}], indirect=True)
async def test_register_hook_refuses_a_key_whose_condition_does_not_render(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager, renderer
) -> None:
    # An unresolvable stored condition must never be read as "no condition".
    _gate_on(
        monkeypatch,
        caller_id="root",
        policies={"root": _ADMIN_POLICY, "k-fire": AccessPolicy(scopes=["hooks"], condition_id="missing")},
    )
    with pytest.raises(BadRequestError, match="does not render"):
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-fire")
    assert (await hooks_ops.list_hooks())["items"] == []


# -- what a refusal is allowed to say about a condition -----------------------
#
# A stored jq condition is readable only by an admin or the key's owner, and the scan's
# diagnostic quotes it: pass-role must be decided FIRST, then disclosure by role.

# Unevaluable, and short enough that the scan's bounded excerpt quotes it whole.
_SECRET_CONDITION = '.identity.dept == "classified"'
_SECRET_WORDS = ("dept", "classified")


async def test_register_hook_pass_role_is_decided_before_the_condition_scan(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager, renderer
) -> None:
    # ``k-bob`` is bob's and its condition is unevaluable: a caller that fails pass-role
    # gets the uniform 403 only. Scanning first would quote bob's condition in a 400.
    _gate_on(
        monkeypatch,
        caller_id="alice",
        policies={
            "alice": AccessPolicy(scopes=["hooks"]),
            "k-bob": _owned_by("bob", condition=_SECRET_CONDITION),
        },
    )
    with pytest.raises(ForbiddenError, match="only bind your own identity or an execution key you own") as raised:
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-bob")
    assert not any(word in str(raised.value) for word in _SECRET_WORDS)
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_create_trigger_link_pass_role_is_decided_before_the_condition_scan(
    monkeypatch: pytest.MonkeyPatch, capture_create, renderer
) -> None:
    _gate_on(
        monkeypatch,
        caller_id="alice",
        policies={
            "alice": AccessPolicy(scopes=["hooks"]),
            "k-bob": _owned_by("bob", condition=_SECRET_CONDITION),
        },
    )
    with pytest.raises(ForbiddenError, match="only bind your own identity or an execution key you own") as raised:
        await hooks_ops.create_trigger_link(
            topic="t", execution_key="k-bob", name="n", ttl_seconds=None, tool_kwargs=None
        )
    assert not any(word in str(raised.value) for word in _SECRET_WORDS)
    assert capture_create == {}


async def test_register_hook_withholds_a_THIRD_principals_condition_from_a_non_admin(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager, renderer
) -> None:
    # Pass-role admits the self-bind, but the scan reads alice's OWNER's condition: the
    # refusal names the owner and the rule, and quotes no condition text.
    _gate_on(
        monkeypatch,
        caller_id="alice",
        policies={
            "alice": _owned_by("boss"),
            "boss": AccessPolicy(scopes=["hooks"], condition=_SECRET_CONDITION),
        },
    )
    with pytest.raises(BadRequestError, match="policy condition of 'boss' is not evaluable") as raised:
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="alice")
    assert not any(word in str(raised.value) for word in _SECRET_WORDS)
    assert "'alice' can be bound" in str(raised.value)
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_register_hook_carries_the_diagnostic_for_a_key_the_caller_owns(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager, renderer
) -> None:
    # Other side of the rule: alice already reads its own key's condition, so the full
    # scan diagnostic is answered.
    _gate_on(
        monkeypatch,
        caller_id="alice",
        policies={
            "alice": AccessPolicy(scopes=["hooks"]),
            "k-fire": _owned_by("alice", condition=_SECRET_CONDITION),
        },
    )
    with pytest.raises(BadRequestError, match="'k-fire' is unusable at a fire") as raised:
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-fire")
    assert all(word in str(raised.value) for word in _SECRET_WORDS)
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_register_hook_carries_the_diagnostic_to_an_admin(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager, renderer
) -> None:
    # An admin reads every stored condition, so the diagnostic is never withheld from
    # one — including for a third principal's condition (the owner's here).
    _gate_on(
        monkeypatch,
        caller_id="root",
        policies={
            "root": _ADMIN_POLICY,
            "k-fire": _owned_by("boss"),
            "boss": AccessPolicy(scopes=["hooks"], condition=_SECRET_CONDITION),
        },
    )
    with pytest.raises(BadRequestError, match="'boss' is unusable at a fire") as raised:
        await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-fire")
    assert all(word in str(raised.value) for word in _SECRET_WORDS)
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_create_trigger_link_refused_bind_never_reaches_the_store(
    monkeypatch: pytest.MonkeyPatch, capture_create
) -> None:
    _gate_on(
        monkeypatch,
        caller_id="alice",
        policies={"alice": AccessPolicy(scopes=["hooks"]), "k-bob": _owned_by("bob")},
    )
    with pytest.raises(ForbiddenError, match="only bind your own identity or an execution key you own"):
        await hooks_ops.create_trigger_link(
            topic="t", execution_key="k-bob", name="n", ttl_seconds=None, tool_kwargs=None
        )
    # No link minted: the gate runs before the store is touched at all.
    assert capture_create == {}


async def test_create_trigger_link_forwards_the_bound_execution_key(
    monkeypatch: pytest.MonkeyPatch, capture_create
) -> None:
    _gate_on(
        monkeypatch,
        caller_id="alice",
        policies={"alice": AccessPolicy(scopes=["hooks"]), "k-fire": _owned_by("alice")},
    )
    await hooks_ops.create_trigger_link(topic="t", execution_key="k-fire", name="n", ttl_seconds=None, tool_kwargs=None)
    assert capture_create["execution_key"] == "k-fire"


# -- the pass-role gate under a background fire ------------------------------
#
# A record writer reached from a fire is decided against the EXECUTION key, never
# against the principal of whichever request happened to trigger the fire.


async def test_register_hook_under_a_fire_binds_a_key_the_execution_key_owns(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    # ``mallory`` rang the door and owns nothing here; the bind is decided against
    # ``k-exec``, which owns the bound key, so it is allowed.
    _gate_on(
        monkeypatch,
        caller_id="mallory",
        policies={
            "mallory": AccessPolicy(scopes=["hooks"]),
            "k-exec": AccessPolicy(scopes=["hooks"]),
            "k-owned": _owned_by("k-exec"),
        },
    )
    async with execution.bind_execution_identity("k-exec", bound_fingerprint="fp"):
        registered = await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-owned")
    assert registered == {"registered": True, "name": "h"}


async def test_register_hook_under_a_fire_refuses_a_key_the_execution_key_does_not_own(
    monkeypatch: pytest.MonkeyPatch, manager: InMemoryHooksManager
) -> None:
    # The triggering request's principal is an ADMIN; the fire must not inherit that
    # authority — ``k-exec`` does not own ``k-bob``, so the bind is refused.
    _gate_on(
        monkeypatch,
        caller_id="root",
        policies={
            "root": _ADMIN_POLICY,
            "k-exec": AccessPolicy(scopes=["hooks"]),
            "k-bob": _owned_by("bob"),
        },
    )
    with pytest.raises(ForbiddenError, match="only bind your own identity or an execution key you own"):
        async with execution.bind_execution_identity("k-exec", bound_fingerprint="fp"):
            await hooks_ops.register_hook(name="h", topic="t", tool="noop", execution_key="k-bob")
    assert (await hooks_ops.list_hooks())["items"] == []


async def test_create_trigger_link_under_a_fire_is_created_by_the_execution_key(
    monkeypatch: pytest.MonkeyPatch, capture_create
) -> None:
    # No request-scope caller at all (the ordinary fire): the acting principal is the
    # execution key, so the mint is both decided and stamped against it.
    _gate_on(
        monkeypatch,
        caller_id=None,
        policies={"k-exec": AccessPolicy(scopes=["hooks"]), "k-owned": _owned_by("k-exec")},
    )
    async with execution.bind_execution_identity("k-exec", bound_fingerprint="fp"):
        await hooks_ops.create_trigger_link(
            topic="t", execution_key="k-owned", name="n", ttl_seconds=None, tool_kwargs=None
        )
    assert capture_create["created_by"] == "k-exec"
