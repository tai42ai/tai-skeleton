"""The token-free-evaluable rule re-asserted AT THE FIRE, on the rendered condition text.

The bind-time scan is early rejection only: the invariant is about RENDERED text, and a
``condition_id`` re-renders at every fire, so a template edit changes the effective
condition with no policy-row write for any mutation-site guard to see. The hole is on the
allow path — ``.identity.X != v`` evaluates TRUE against the absent claim and ALLOWS. The
re-assert is gated on the ``execution_identity`` contextvar, so ordinary authed requests,
which carry full claims, are untouched.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM, OWNER_USER_ID_CLAIM
from tai42_contract.app import tai42_app

import tai42_skeleton.versioning as versioning_module
from tai42_skeleton.access_control import management
from tai42_skeleton.access_control.policy import PolicyEnforcer
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.authz.check import check
from tai42_skeleton.authz.execution import assert_execution_key_evaluable, bind_execution_identity
from tai42_skeleton.authz.identity import CallerIdentity
from tai42_skeleton.authz.token_free import TokenFreeConditionError
from tai42_skeleton.operations import OperationRegistry, operation
from tai42_skeleton.operations import _authority as authority
from tai42_skeleton.operations import api_keys as api_keys_ops
from tai42_skeleton.operations.errors import PermissionDenied
from tai42_skeleton.template import TemplateNotFoundError
from tests.access_control.conftest import FakeRedis, make_client_ctx
from tests.access_control.test_policy_store import _MemStore

# alru caches are held across the several ``asyncio.run`` loops a test opens — a benign
# loop-reset artifact, since a real process serves one loop for its lifetime.
pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")

_ROUTE = "/api/things/wipe"
_SCOPE = "things"

_EVALUABLE = '.sub != "banned"'  # a condition a tokenless fire can evaluate
# A negative identity predicate: the absent claim satisfies ``!= true``, so the jq pass
# would ALLOW — the fail-open the fire-time assertion closes.
_FAIL_OPEN = ".identity.suspended != true"


class _MutableResourceManager:
    """The condition renderer with a MUTABLE template map, so a test can edit a stored
    template without touching any policy row. Inline ``content`` renders to itself."""

    def __init__(self) -> None:
        self.templates: dict[str, str] = {}

    async def render_by_id_or_content(self, *, content, template_id, kwargs) -> str:
        if template_id is not None:
            try:
                return self.templates[template_id]
            except KeyError as exc:
                # The real manager's answer for an id with no stored template.
                raise TemplateNotFoundError(f"Template '{template_id}' not found.") from exc
        return content or ""


@pytest.fixture
def renderer() -> Iterator[_MutableResourceManager]:
    """Bind an app carrying the mutable renderer onto ``tai42_app`` for the test."""
    from types import SimpleNamespace

    manager = _MutableResourceManager()
    with tai42_app.bound(SimpleNamespace(storage=SimpleNamespace(resource_manager=manager))):
        yield manager


@pytest.fixture
def probe_op():
    """A registered operation whose route the seeded policies are scoped to."""
    registry = OperationRegistry()

    @operation(name="wipe", summary="Wipe", tags=["things"], registry=registry)
    async def _wipe(**_):
        return {}

    meta = registry.get("wipe")
    meta.route_template = _ROUTE
    meta.http_method = "POST"
    return meta


@pytest.fixture
def settings() -> AccessControlSettings:
    return AccessControlSettings()


@pytest.fixture(autouse=True)
def _execution_gate_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from tai42_skeleton.authz import execution as execution_module

    monkeypatch.setattr(execution_module, "access_control_settings", lambda: AccessControlSettings(enable=True))


def _bind_enforcer() -> PolicyEnforcer:
    """A fresh enforcer, the way each bind door builds one for the whole gate it runs."""
    return PolicyEnforcer(AccessControlSettings(enable=True))


async def _fire(key: str, meta, settings: AccessControlSettings) -> None:
    """Authorize one dispatch AS ``key``: bind the execution identity, then run the same
    ``check`` the tool edge runs."""
    async with bind_execution_identity(key, bound_fingerprint=f"fp-{key}") as identity:
        await check(identity, meta, {}, settings=settings)


async def _request(identity: CallerIdentity, meta, settings: AccessControlSettings) -> None:
    """Authorize one ORDINARY request: the same ``check``, with no execution identity bound."""
    await check(identity, meta, {}, settings=settings)


# -- the baseline: a key that passes the bind scan fires cleanly --------------


def test_a_key_whose_condition_is_evaluable_binds_and_fires(ac_env, renderer, probe_op, settings) -> None:
    ac_env.add_route(_ROUTE, _SCOPE)
    ac_env.add_policy("k-fire", scopes=[_SCOPE], condition=_EVALUABLE, policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-fire"})

    asyncio.run(assert_execution_key_evaluable(_bind_enforcer(), "k-fire"))  # the bind scan passes
    asyncio.run(_fire("k-fire", probe_op, settings))  # and so does the fire


# -- mutation shape (i): the POLICY ROW, edited through the real door ----------


def test_a_policy_row_edited_after_the_bind_is_denied_at_the_next_fire(
    monkeypatch: pytest.MonkeyPatch, ac_env, renderer, probe_op, settings
) -> None:
    # ``edit_api_key`` knows nothing about any record that bound this key, and carries no
    # guard: the fire is what refuses.
    ac_env.add_route(_ROUTE, _SCOPE)
    ac_env.add_policy("k-fire", scopes=[_SCOPE], condition=_EVALUABLE, policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-fire"})
    _wire_edit_door(monkeypatch, ac_env)

    asyncio.run(_fire("k-fire", probe_op, settings))  # allowed before the edit

    asyncio.run(api_keys_ops.edit_api_key(user_id="k-fire", updates={"condition": _FAIL_OPEN}))

    # The stored row really did change — the edit landed.
    assert ac_env.policy("k-fire")["condition"] == _FAIL_OPEN

    with pytest.raises(PermissionDenied, match="policy condition for 'k-fire' is not evaluable at a fire"):
        asyncio.run(_fire("k-fire", probe_op, settings))


def test_the_bind_scan_and_the_fire_assertion_are_one_rule(ac_env, renderer, probe_op, settings) -> None:
    # One implementation: a condition that cannot fire also cannot be freshly bound.
    ac_env.add_route(_ROUTE, _SCOPE)
    ac_env.add_policy("k-fire", scopes=[_SCOPE], condition=_FAIL_OPEN, policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-fire"})

    with pytest.raises(TokenFreeConditionError, match="unusable at a fire"):
        asyncio.run(assert_execution_key_evaluable(_bind_enforcer(), "k-fire"))
    with pytest.raises(PermissionDenied, match="not evaluable at a fire"):
        asyncio.run(_fire("k-fire", probe_op, settings))


# -- mutation shape (ii): the TEMPLATE, with no policy-row write at all --------


def test_a_template_edited_after_the_bind_is_denied_with_no_policy_row_write(
    ac_env, renderer, probe_op, settings
) -> None:
    # The policy row is byte-identical before and after; only the template it points at changed.
    ac_env.add_route(_ROUTE, _SCOPE)
    ac_env.add_policy("k-fire", scopes=[_SCOPE], condition_id="cond", policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-fire"})
    renderer.templates["cond"] = _EVALUABLE

    asyncio.run(assert_execution_key_evaluable(_bind_enforcer(), "k-fire"))
    asyncio.run(_fire("k-fire", probe_op, settings))

    row_before = dict(ac_env.policy("k-fire"))
    renderer.templates["cond"] = _FAIL_OPEN  # a template edit — no policy write

    assert ac_env.policy("k-fire") == row_before

    with pytest.raises(PermissionDenied, match="policy condition for 'k-fire' is not evaluable at a fire"):
        asyncio.run(_fire("k-fire", probe_op, settings))


def test_the_owners_condition_is_re_asserted_too(ac_env, renderer, probe_op, settings) -> None:
    # The owner's condition is a second pass at every fire and an equally mutable surface;
    # the refusal names the OWNER.
    ac_env.add_route(_ROUTE, _SCOPE)
    ac_env.add_policy(
        "k-fire", scopes=[_SCOPE], policy_data={OWNER_USER_ID_CLAIM: "alice", KEY_FINGERPRINT_CLAIM: "fp-k-fire"}
    )
    ac_env.add_policy("alice", scopes=[_SCOPE], condition_id="owner-cond")
    renderer.templates["owner-cond"] = _EVALUABLE

    asyncio.run(_fire("k-fire", probe_op, settings))

    renderer.templates["owner-cond"] = _FAIL_OPEN

    with pytest.raises(PermissionDenied, match="policy condition for 'alice' is not evaluable at a fire"):
        asyncio.run(_fire("k-fire", probe_op, settings))


# -- the polarity asymmetry, and the contextvar gate --------------------------


def test_the_negative_predicate_would_have_allowed_the_fire(ac_env, renderer, probe_op, settings) -> None:
    """Non-vacuity for every deny above: with the gate off (an ordinary request) the same
    condition ALLOWS, because the absent ``.identity.suspended`` satisfies ``!= true``."""
    ac_env.add_route(_ROUTE, _SCOPE)
    ac_env.add_policy("k-fire", scopes=[_SCOPE], condition=_FAIL_OPEN, policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-fire"})

    # No execution identity bound: the assertion does not run and the predicate allows.
    asyncio.run(_request(CallerIdentity(user_id="k-fire", effective_scopes=(_SCOPE,), claims={}), probe_op, settings))

    # Bound: same condition, same policy, same route — DENIED.
    with pytest.raises(PermissionDenied, match="not evaluable at a fire"):
        asyncio.run(_fire("k-fire", probe_op, settings))


def test_a_normal_authed_request_with_the_same_condition_is_unaffected(ac_env, renderer, probe_op, settings) -> None:
    # The assertion is contextvar-gated, so a token-carrying caller is decided by the
    # condition itself, both ways.
    ac_env.add_route(_ROUTE, _SCOPE)
    ac_env.add_policy("alice", scopes=[_SCOPE], condition=_FAIL_OPEN)

    active = CallerIdentity(user_id="alice", effective_scopes=(_SCOPE,), claims={"suspended": False})
    suspended = CallerIdentity(user_id="alice", effective_scopes=(_SCOPE,), claims={"suspended": True})

    asyncio.run(_request(active, probe_op, settings))  # allowed by the condition
    with pytest.raises(PermissionDenied, match="policy condition rejected"):
        asyncio.run(_request(suspended, probe_op, settings))  # denied by the condition


def test_a_condition_that_no_longer_renders_denies_the_fire(ac_env, renderer, probe_op, settings) -> None:
    # An unresolvable ``condition_id`` is never read as "no condition". The refusal must NAME
    # the principal: a generic "access denied" is also what a swallowed render error produces.
    ac_env.add_route(_ROUTE, _SCOPE)
    ac_env.add_policy("k-fire", scopes=[_SCOPE], condition_id="cond", policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-fire"})
    renderer.templates["cond"] = _EVALUABLE

    asyncio.run(_fire("k-fire", probe_op, settings))

    del renderer.templates["cond"]  # the template is deleted out from under the key

    with pytest.raises(PermissionDenied, match="the policy condition of 'k-fire' does not render"):
        asyncio.run(_fire("k-fire", probe_op, settings))

    # The render failure is not a fire-only rule: a real request denies identically.
    with pytest.raises(PermissionDenied, match="the policy condition of 'k-fire' does not render"):
        asyncio.run(
            _request(CallerIdentity(user_id="k-fire", effective_scopes=(_SCOPE,), claims={}), probe_op, settings)
        )


def _wire_edit_door(monkeypatch: pytest.MonkeyPatch, ac_env) -> None:
    """Wire the real ``edit_api_key`` door over the faked policy store: an admin caller, the
    management redis the version bump writes to, and a versioned store for the history."""
    ac_env.add_policy("root", scopes=["*"])
    monkeypatch.setattr(authority, "access_control_settings", lambda: AccessControlSettings(enable=True))
    monkeypatch.setattr(authority, "get_current_user_id", lambda: "root")
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(FakeRedis(strings={})))
    store = _MemStore()
    monkeypatch.setattr(versioning_module, "versioned_store", lambda: store)
    monkeypatch.setattr(versioning_module, "versioned_store_configured", lambda: True)
