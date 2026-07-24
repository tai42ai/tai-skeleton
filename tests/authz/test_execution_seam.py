"""The tool-dispatch seam on the live stack: the decision a dispatch takes under a bound
execution identity.

Boots the real registries over a faked policy store and dispatches through the real
``tools/binding.py`` seams (``run_tool`` and the ``get_client_tools`` runnable). An
OPERATION tool takes the full route-edge decision (scope/jq pass AND per-tag LEVEL pass);
a CAPABILITY tool (non-operation) takes liveness only. With no identity bound the seam is
a strict no-op. The two operation classes come from ``tests.authz._fixtures.execution_probe``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM, OWNER_USER_ID_CLAIM
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id
from tai42_contract.app import tai42_app
from tai42_contract.hooks import HookParams

import tai42_skeleton.versioning as versioning_module
from tai42_skeleton.access_control import management
from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import role_grants as role_grants_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.policy_store import ac_policy_store
from tai42_skeleton.access_control.role_gate import reset_route_index, resolve_route_meta
from tai42_skeleton.access_control.roles import ROLE_POINTER_KEY, role_store
from tai42_skeleton.access_control.settings import access_control_settings
from tai42_skeleton.app.instance import app
from tai42_skeleton.app.reload_gate import reload_gate
from tai42_skeleton.app.route_registry import route_registry
from tai42_skeleton.authz.execution import bind_execution_identity
from tai42_skeleton.authz.execution_identity import get_execution_identity
from tai42_skeleton.authz.resolver import OperationSurfaceUnsettledError
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.operations import api_keys as api_keys_ops
from tai42_skeleton.operations.errors import PermissionDenied
from tai42_skeleton.operations.registry import operation_registry
from tai42_skeleton.tools.binding import CLIENT_TOOL_NAME_MAX_LEN
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx
from tests.access_control.test_policy_store import _MemStore

# The enforcer's alru cache is created and used across boots — a benign loop-reset artifact.
pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")

_PROBE_ROUTER = "tests.authz._fixtures.execution_probe"

_FENCED_OP = "exec_probe_fenced"
_FENCED_PATH = "/api/exec-probe/deploy/fenced"
_FENCED_PATH_ALT = "/api/exec-probe/rollback/fenced"  # same operation, other path argument
_GRANTABLE_OP = "exec_probe_read"
_GRANTABLE_PATH = "/api/exec-probe/read"
_CAPABILITY_TOOL = "shout"  # a manifest tool, not an operation

_PROBE_SCOPE = "probe"  # the scope both probe routes are mapped to
_KEPT_SCOPE = "misc"  # kept through every reduction, so a narrowed policy stays non-empty


def _manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "api_tools": {"enabled": True},
            "tools": [{"title": "fxt", "module": "tests.app._fixtures.tools_b", "include": ["shout"]}],
            "agents": [{"title": "agents", "module": "tests.agent._fixtures", "include": ["nested_tools"]}],
            # Probe router only: it carries both action-classes the matrix needs.
            "routers_modules": [_PROBE_ROUTER],
            "default_routers": "none",
        }
    )


@pytest.fixture(autouse=True)
def _isolate_registries(preset_manager_restored):
    """Snapshot/restore the process-global operation and route registries around each test.

    The ``tai42_app.bound(None)`` scoping is load-bearing too: an app left bound reports
    ``effective_routers`` as the probe router alone, pinning every later ``load_*_routes()``
    in the process to the registry restored here."""
    routes_snapshot = dict(route_registry._routes)
    ops_snapshot = dict(operation_registry._operations)
    with tai42_app.bound(None):
        try:
            yield
        finally:
            route_registry._routes = routes_snapshot
            operation_registry._operations = ops_snapshot


@pytest.fixture
def ac(monkeypatch: pytest.MonkeyPatch) -> FakeAccessControlPg:
    """A faked policy store carrying the probe routes and the keys the matrix drives.

    ``k-admin`` is admin (unowned ``"*"``); ``k-scoped`` holds both probe routes' scope but
    is NOT admin, so any deny it takes comes from the LEVEL pass alone; ``k-narrow`` holds
    only an unrelated scope; ``k-owned``/``owner`` are the owned-key attenuation pair.
    """
    pg = FakeAccessControlPg()
    redis = FakeRedis()
    pg.add_route(_FENCED_PATH, _PROBE_SCOPE)
    pg.add_route(_FENCED_PATH_ALT, _PROBE_SCOPE)
    pg.add_route(_GRANTABLE_PATH, _PROBE_SCOPE)
    pg.add_policy("k-admin", scopes=["*"], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-admin"})
    pg.add_policy("k-scoped", scopes=[_PROBE_SCOPE, _KEPT_SCOPE], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-scoped"})
    pg.add_policy("k-narrow", scopes=[_KEPT_SCOPE], policy_data={KEY_FINGERPRINT_CLAIM: "fp-k-narrow"})
    pg.add_policy(
        "k-owned",
        scopes=[_PROBE_SCOPE, _KEPT_SCOPE],
        policy_data={OWNER_USER_ID_CLAIM: "owner", KEY_FINGERPRINT_CLAIM: "fp-k-owned"},
    )
    pg.add_policy("owner", scopes=[_PROBE_SCOPE, _KEPT_SCOPE])
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(redis))
    # Management writes reach redis through its own binding; same fake => one version counter.
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(redis))
    # Route index and grant cache are process-global; rebuild against this test's registry.
    reset_route_index()
    role_grants_module.reset_role_grants_cache()
    return pg


# ``target`` is the path parameter the concrete resource path is synthesized from.
_FENCED_ARGS = {"target": "deploy", "mark": "m"}
_GRANTABLE_ARGS = {"mark": "m"}


async def _run_as(key: str, tool: str, arguments: dict | None = None):
    """Dispatch ``tool`` through the real ``run_tool`` seam AS ``key``."""
    async with bind_execution_identity(key, bound_fingerprint=f"fp-{key}"):
        return await app.tools.run_tool(tool, arguments or {})


async def _invoke_client_tool_as(key: str, tool: str, arguments: dict | None = None):
    """Resolve ``tool`` as a client tool and invoke it AS ``key`` — the agent-facing seam."""
    async with bind_execution_identity(key, bound_fingerprint=f"fp-{key}"):
        [client_tool] = await app.tools.get_client_tools([tool])
        return await client_tool.ainvoke(arguments or {})


def test_the_probe_routes_carry_the_action_classes_the_matrix_assumes(ac) -> None:
    # If the fenced probe were retagged, every deny below would pass for the wrong reason.
    async def run() -> None:
        async with app.app_context(_manifest()):
            fenced = resolve_route_meta(_FENCED_PATH, "POST")
            grantable = resolve_route_meta(_GRANTABLE_PATH, "GET")
            assert fenced is not None
            assert fenced.action == "fenced"
            assert grantable is not None
            assert grantable.action == "read"

    asyncio.run(run())


# -- the granularity + the full edge decision ---------------------------------


def test_a_fenced_operation_is_denied_under_a_non_admin_key(ac) -> None:
    # The key HOLDS the route's scope, so the scope/jq pass allows: the deny is the LEVEL pass.
    async def run() -> None:
        async with app.app_context(_manifest()):
            with pytest.raises(PermissionDenied, match=f"POST {_FENCED_PATH} is not permitted"):
                await _run_as("k-scoped", _FENCED_OP, _FENCED_ARGS)

    asyncio.run(run())


def test_a_fenced_operation_runs_under_an_admin_key(ac) -> None:
    # ALLOW parity, so the deny above is not vacuous.
    async def run() -> None:
        async with app.app_context(_manifest()):
            assert await _run_as("k-admin", _FENCED_OP, _FENCED_ARGS) == "fenced:deploy:m"

    asyncio.run(run())


def test_the_fenced_deny_keys_on_the_path_the_call_actually_synthesizes(ac) -> None:
    # The decision keys on the path synthesized from the arguments being FIRED, not a
    # template: the refusal names the alternate path, which is mapped and in-scope.
    async def run() -> None:
        async with app.app_context(_manifest()):
            with pytest.raises(PermissionDenied, match=f"POST {_FENCED_PATH_ALT} is not permitted"):
                await _run_as("k-scoped", _FENCED_OP, {"target": "rollback", "mark": "m"})

            # ALLOW parity: the deny above is the fence, not an unresolved resource.
            assert await _run_as("k-admin", _FENCED_OP, {"target": "rollback", "mark": "m"}) == "fenced:rollback:m"

    asyncio.run(run())


def test_a_preset_is_decided_on_the_arguments_it_actually_fires(ac) -> None:
    # A preset bakes kwargs the caller cannot supply, so the decision must merge the baked
    # ``target`` in before synthesizing the resource path.
    async def run() -> None:
        async with app.app_context(_manifest()):
            await app.preset_manager.register("probe_deploy", _FENCED_OP, {"target": "deploy"}, [], [], "d")

            assert await _run_as("k-admin", "probe_deploy", {"mark": "m"}) == "fenced:deploy:m"

            with pytest.raises(PermissionDenied, match=f"POST {_FENCED_PATH} is not permitted"):
                await _run_as("k-scoped", "probe_deploy", {"mark": "m"})

    asyncio.run(run())


def test_a_grantable_operation_runs_when_the_key_holds_it_and_is_denied_when_it_does_not(ac) -> None:
    async def run() -> None:
        async with app.app_context(_manifest()):
            assert await _run_as("k-scoped", _GRANTABLE_OP, _GRANTABLE_ARGS) == "read:m"

            with pytest.raises(PermissionDenied, match="insufficient scope"):
                await _run_as("k-narrow", _GRANTABLE_OP, _GRANTABLE_ARGS)

    asyncio.run(run())


def test_a_capability_tool_runs_and_is_never_admin_gated(ac) -> None:
    # A non-operation has no route, so no per-call scope model exists for it at any edge;
    # its reach is bounded by EXPOSURE.
    async def run() -> None:
        async with app.app_context(_manifest()):
            assert await _run_as("k-scoped", _CAPABILITY_TOOL, {"text": "hi"}) == "hi"
            assert await _run_as("k-narrow", _CAPABILITY_TOOL, {"text": "hi"}) == "hi"

    asyncio.run(run())


def test_a_key_that_no_longer_exists_never_reaches_a_capability_tool(ac) -> None:
    # A capability tool is not scope-checked, so the identity build is the only refusal
    # keeping it bounded.
    async def run() -> None:
        async with app.app_context(_manifest()):
            with pytest.raises(PermissionDenied, match="has no policy"):
                await _run_as("ghost", _CAPABILITY_TOOL, {"text": "hi"})

    asyncio.run(run())


# -- the reload window: a torn surface must never read as "capability" ---------


@contextmanager
def _rebuilding_operation_surface() -> Iterator[None]:
    """The window a config reload opens: the operation registry is cleared while dispatch
    continues, so a tool stays resolvable while the surface cannot say what it is."""
    registered = dict(operation_registry._operations)
    operation_registry.clear()
    try:
        yield
    finally:
        operation_registry._operations.update(registered)


def test_a_fenced_operation_is_refused_while_the_operation_surface_rebuilds(ac) -> None:
    # Fail-closed: mid-rebuild an absent name must not be classified as a capability tool
    # and waved through. The refusal is retriable, not a verdict on the key.
    async def run() -> None:
        async with app.app_context(_manifest()):
            from tests.authz._fixtures import execution_probe

            with _rebuilding_operation_surface():
                execution_probe.calls.clear()
                with pytest.raises(OperationSurfaceUnsettledError, match="retry shortly"):
                    await _run_as("k-scoped", _FENCED_OP, _FENCED_ARGS)
                assert execution_probe.calls == []

            # Non-vacuous: once settled, the same dispatches take the real decision again.
            with pytest.raises(PermissionDenied, match=f"POST {_FENCED_PATH} is not permitted"):
                await _run_as("k-scoped", _FENCED_OP, _FENCED_ARGS)
            assert await _run_as("k-admin", _FENCED_OP, _FENCED_ARGS) == "fenced:deploy:m"

    asyncio.run(run())


def test_the_client_tool_seam_is_refused_while_the_operation_surface_rebuilds(ac) -> None:
    # A client runnable captured its target before the tear, so it never takes ``run_tool``'s
    # resolution wait — the discriminator itself is what refuses on this path.
    async def run() -> None:
        async with app.app_context(_manifest()):
            from tests.authz._fixtures import execution_probe

            runnable = app._tool_binding._client_runnable(await app.tools.get_tool(_FENCED_OP))

            with _rebuilding_operation_surface():
                execution_probe.calls.clear()
                async with bind_execution_identity("k-scoped", bound_fingerprint="fp-k-scoped"):
                    with pytest.raises(OperationSurfaceUnsettledError, match="retry shortly"):
                        await runnable(target="deploy", mark="m")
                assert execution_probe.calls == []

            async with bind_execution_identity("k-admin", bound_fingerprint="fp-k-admin"):
                assert await runnable(target="deploy", mark="m") == "fenced:deploy:m"

    asyncio.run(run())


def test_a_rebuilding_surface_leaves_an_unbound_dispatch_untouched(ac) -> None:
    # The strict no-op holds through the window; also proves the refusals above come from
    # the discriminator and not from the tool having become unreachable.
    async def run() -> None:
        async with app.app_context(_manifest()):
            with _rebuilding_operation_surface():
                assert get_execution_identity() is None
                assert await app.tools.run_tool(_FENCED_OP, _FENCED_ARGS) == "fenced:deploy:m"
                assert await app.tools.run_tool(_CAPABILITY_TOOL, {"text": "hi"}) == "hi"

    asyncio.run(run())


def test_a_capability_tool_is_refused_only_while_the_surface_is_unsettled(ac) -> None:
    # While torn, a capability tool is indistinguishable from an unregistered operation, so
    # it takes the same retriable refusal — a property of the surface, not the tool or key.
    async def run() -> None:
        async with app.app_context(_manifest()):
            with _rebuilding_operation_surface(), pytest.raises(OperationSurfaceUnsettledError):
                await _run_as("k-scoped", _CAPABILITY_TOOL, {"text": "hi"})

            assert await _run_as("k-scoped", _CAPABILITY_TOOL, {"text": "hi"}) == "hi"

    asyncio.run(run())


def test_a_fire_waits_out_an_in_flight_reload_instead_of_being_dropped(ac) -> None:
    # A background fire has no client to retry, so the decision waits for the reload holding
    # the gate and decides again; otherwise a fire landing in a reload window is lost.
    async def run() -> None:
        async with app.app_context(_manifest()):
            registered = dict(operation_registry._operations)

            async def reload_window() -> None:
                async with reload_gate.lock:
                    with operation_registry.rebuilding():
                        operation_registry.clear()
                        await asyncio.sleep(0.05)
                        operation_registry._operations.update(registered)

            window = asyncio.create_task(reload_window())
            await asyncio.sleep(0)  # let the reload take the gate before the fire dispatches

            assert await _run_as("k-admin", _FENCED_OP, _FENCED_ARGS) == "fenced:deploy:m"
            await window

    asyncio.run(run())


def test_a_fire_still_refuses_when_the_surface_is_unsettled_with_no_reload_to_wait_for(ac) -> None:
    # The wait is not a swallow: with the gate uncontended the refusal stands rather than
    # the dispatch being waved through as a capability tool.
    async def run() -> None:
        async with app.app_context(_manifest()):
            from tests.authz._fixtures import execution_probe

            with _rebuilding_operation_surface():
                execution_probe.calls.clear()
                with pytest.raises(OperationSurfaceUnsettledError, match="retry shortly"):
                    await _run_as("k-admin", _FENCED_OP, _FENCED_ARGS)
                assert execution_probe.calls == []

    asyncio.run(run())


# -- one policy-version round trip per decision ----------------------------------


class _VersionCountingRedis(FakeRedis):
    """Counts reads of the policy-version key — the round trip a decision pins its snapshot to."""

    def __init__(self) -> None:
        super().__init__()
        self.version_reads = 0

    async def get(self, key):
        if key == access_control_settings().policy_version_key:
            self.version_reads += 1
        return await super().get(key)


def test_one_dispatch_decides_on_one_policy_snapshot(monkeypatch: pytest.MonkeyPatch, ac) -> None:
    # An owned key exercises all four versioned reads of one decision (route→resource, the
    # key's policy, the owner's, the LEVEL pass's grants). One round trip is the proof that
    # the version is read once and threaded through, so all four see one snapshot.
    redis = _VersionCountingRedis()
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))

    async def run() -> None:
        async with app.app_context(_manifest()), bind_execution_identity("k-owned", bound_fingerprint="fp-k-owned"):
            redis.version_reads = 0
            assert await app.tools.run_tool(_GRANTABLE_OP, _GRANTABLE_ARGS) == "read:m"
            assert redis.version_reads == 1

            # Nothing is carried across dispatches — that is what makes revocation land.
            assert await app.tools.run_tool(_GRANTABLE_OP, _GRANTABLE_ARGS) == "read:m"
            assert redis.version_reads == 2

    asyncio.run(run())


# -- the client-tool seam (the agent-facing dispatch) --------------------------


def test_the_client_tool_seam_applies_the_same_decision(ac) -> None:
    # A tool handed to an agent runs through ``_client_runnable``, not ``run_tool``.
    async def run() -> None:
        async with app.app_context(_manifest()):
            with pytest.raises(PermissionDenied, match=f"POST {_FENCED_PATH} is not permitted"):
                await _invoke_client_tool_as("k-scoped", _FENCED_OP, _FENCED_ARGS)

            assert await _invoke_client_tool_as("k-admin", _FENCED_OP, _FENCED_ARGS) == "fenced:deploy:m"

    asyncio.run(run())


def test_the_client_tool_seam_authorizes_the_full_name_the_client_label_truncates(ac) -> None:
    # ``get_client_tools`` truncates the name it hands the LLM, but the seam must resolve the
    # FULL registered name: a truncated label resolves to nothing and would read as a
    # CAPABILITY tool. Extension branches carry no length bound, so >64 chars is reachable.
    long_name = "exec_probe_fenced_branch_" + "b" * 50

    async def run() -> None:
        async with app.app_context(_manifest()):

            @app.tools.tool(force=True, name=long_name)
            async def _branch(target: str, mark: str) -> str:
                """An extension branch over the fenced probe."""
                return f"branch:{target}:{mark}"

            app._tool_binding._tool_registry.register_extend_tool(tool_name=_FENCED_OP, extend_tool_name=long_name)
            assert len(long_name) > CLIENT_TOOL_NAME_MAX_LEN

            async with bind_execution_identity("k-scoped", bound_fingerprint="fp-k-scoped"):
                [client_tool] = await app.tools.get_client_tools([long_name])
                assert client_tool.name == long_name[:CLIENT_TOOL_NAME_MAX_LEN]
                with pytest.raises(PermissionDenied, match=f"POST {_FENCED_PATH} is not permitted"):
                    await client_tool.ainvoke(_FENCED_ARGS)

            # Non-vacuous: the deny above is the LEVEL pass, not a broken registration.
            assert await _invoke_client_tool_as("k-admin", long_name, _FENCED_ARGS) == "branch:deploy:m"

    asyncio.run(run())


def test_the_client_tool_seam_binds_positional_arguments_before_deciding(ac) -> None:
    # The decision keys arguments by PARAMETER NAME, so a positional path parameter must be
    # bound through the signature first — positional and keyword spellings must refuse alike.
    async def run() -> None:
        async with app.app_context(_manifest()):
            runnable = app._tool_binding._client_runnable(await app.tools.get_tool(_FENCED_OP))
            async with bind_execution_identity("k-scoped", bound_fingerprint="fp-k-scoped"):
                with pytest.raises(PermissionDenied, match=f"POST {_FENCED_PATH_ALT} is not permitted") as positional:
                    await runnable("rollback", "m")
                with pytest.raises(PermissionDenied) as keyword:
                    await runnable(target="rollback", mark="m")
            assert str(positional.value) == str(keyword.value)

    asyncio.run(run())


def test_a_tool_an_agent_resolves_mid_turn_is_governed_by_the_seam(ac) -> None:
    # An agent resolves its own tools by name mid-turn, which no wrapping at the agent's call
    # site can reach — the seam is the only place that dispatch can be governed.
    async def run() -> None:
        async with app.app_context(_manifest()):
            nested = {"tool_name": _FENCED_OP, "arguments": _FENCED_ARGS}
            with pytest.raises(PermissionDenied, match=f"POST {_FENCED_PATH} is not permitted"):
                await _run_as("k-scoped", "nested_tools", nested)

            # Non-vacuous: the deny above is the decision, not a broken fixture.
            assert await _run_as("k-admin", "nested_tools", nested) == "fenced:deploy:m"

    asyncio.run(run())


# -- the whole fire path, end to end ------------------------------------------


def test_a_hook_firing_a_fenced_operation_is_denied_under_a_non_admin_key(ac, caplog) -> None:
    # End to end through the real ``on_event`` fan-out. The denial surfaces as a per-hook
    # error outcome, so one refused hook never takes its topic's siblings down with it.
    from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
    from tai42_skeleton.hooks.settings import HooksSettings

    async def run() -> list[tuple[str, str]]:
        async with app.app_context(_manifest()):
            from tests.authz._fixtures import execution_probe

            manager = InMemoryHooksManager(HooksSettings())
            await manager.register(
                HookParams(
                    name="escalate",
                    topic="t",
                    tool=_FENCED_OP,
                    tool_kwargs={"target": "deploy", "mark": "escalate"},
                    execution_key="k-scoped",
                    execution_key_fingerprint="fp-k-scoped",
                )
            )
            await manager.register(
                HookParams(
                    name="operator",
                    topic="t",
                    tool=_FENCED_OP,
                    tool_kwargs={"target": "deploy", "mark": "operator"},
                    execution_key="k-admin",
                    execution_key_fingerprint="fp-k-admin",
                )
            )

            with caplog.at_level(logging.ERROR):
                await manager.on_event("t", {})
            return list(execution_probe.calls)

    ran = asyncio.run(run())

    assert ran == [("fenced", "operator")]
    denials = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("escalate" in message for message in denials)
    assert not any("operator" in message for message in denials)


# -- the load-bearing regression: no identity bound => strict no-op ------------


def test_with_no_execution_identity_the_seam_does_nothing(ac) -> None:
    # Access control is ENABLED, but with nothing bound the seam short-circuits on one
    # contextvar read: ordinary requests and in-process agent runs are this case.
    async def run() -> None:
        async with app.app_context(_manifest()):
            assert get_execution_identity() is None
            assert await app.tools.run_tool(_CAPABILITY_TOOL, {"text": "hi"}) == "hi"
            assert await app.tools.run_tool(_FENCED_OP, _FENCED_ARGS) == "fenced:deploy:m"

            [client_tool] = await app.tools.get_client_tools([_FENCED_OP])
            assert await client_tool.ainvoke(_FENCED_ARGS) == "fenced:deploy:m"

            nested = {"tool_name": _FENCED_OP, "arguments": _FENCED_ARGS}
            assert await app.tools.run_tool("nested_tools", nested) == "fenced:deploy:m"

    asyncio.run(run())


def test_the_binding_is_released_so_later_dispatches_are_unguarded(ac) -> None:
    # A DENIED fire must not leave the identity bound for whatever runs next on this context.
    async def run() -> None:
        async with app.app_context(_manifest()):
            with pytest.raises(PermissionDenied, match=f"POST {_FENCED_PATH} is not permitted"):
                await _run_as("k-scoped", _FENCED_OP, _FENCED_ARGS)
            assert get_execution_identity() is None
            assert await app.tools.run_tool(_FENCED_OP, _FENCED_ARGS) == "fenced:deploy:m"

    asyncio.run(run())


# -- automatic revocation, with ZERO revocation-specific code -----------------


def test_de_scoping_the_owner_denies_the_next_fire(ac) -> None:
    # Nothing is cascaded and no record is rewritten: the decision attenuates the key against
    # the owner's CURRENT scopes on every fire.
    async def run() -> None:
        async with app.app_context(_manifest()):
            assert await _run_as("k-owned", _GRANTABLE_OP, _GRANTABLE_ARGS) == "read:m"

            ac.policy("owner")["scopes"] = [_KEPT_SCOPE]

            with pytest.raises(PermissionDenied, match="insufficient scope"):
                await _run_as("k-owned", _GRANTABLE_OP, _GRANTABLE_ARGS)

    asyncio.run(run())


def test_rolling_the_policy_back_denies_the_next_fire(monkeypatch: pytest.MonkeyPatch, ac) -> None:
    # A second authority axis through the real door: ``rollback_policy`` re-points the
    # enforced policy at an earlier version, and the fire reads it live.
    store = _MemStore()
    monkeypatch.setattr(versioning_module, "versioned_store", lambda: store)
    monkeypatch.setattr(versioning_module, "versioned_store_configured", lambda: True)

    async def run() -> None:
        async with app.app_context(_manifest()):
            # Version 1 narrow, version 2 broad — the body the enforced row carries.
            history = ac_policy_store()
            await history.write("k-scoped", _policy_body([_KEPT_SCOPE]))
            await history.write("k-scoped", _policy_body([_PROBE_SCOPE, _KEPT_SCOPE]))

            assert await _run_as("k-scoped", _GRANTABLE_OP, _GRANTABLE_ARGS) == "read:m"

            version_before = await _stored_policy_version()
            # The door is admin-only and resolves its principal from the request-scoped caller.
            token = set_request_user_id("k-admin")
            try:
                await api_keys_ops.rollback_policy("k-scoped", 1)
            finally:
                reset_request_user_id(token)

            # The ENFORCED row must carry the restored body and the version must be bumped;
            # advancing only the history pointer would leave the pre-rollback grants in force.
            assert ac.policy("k-scoped")["scopes"] == [_KEPT_SCOPE]
            assert await _stored_policy_version() > version_before

            with pytest.raises(PermissionDenied, match="insufficient scope"):
                await _run_as("k-scoped", _GRANTABLE_OP, _GRANTABLE_ARGS)

    asyncio.run(run())


def test_de_scoping_the_key_denies_the_next_dispatch_of_a_fire_already_running(ac) -> None:
    # The bound identity carries NO scope set: scopes are re-derived live on every dispatch,
    # so a reduction lands mid-fire instead of waiting out the turn.
    async def run() -> None:
        async with app.app_context(_manifest()), bind_execution_identity("k-scoped", bound_fingerprint="fp-k-scoped"):
            assert await app.tools.run_tool(_GRANTABLE_OP, _GRANTABLE_ARGS) == "read:m"

            ac.policy("k-scoped")["scopes"] = [_KEPT_SCOPE]

            with pytest.raises(PermissionDenied, match="insufficient scope"):
                await app.tools.run_tool(_GRANTABLE_OP, _GRANTABLE_ARGS)

    asyncio.run(run())


def test_de_scoping_the_owner_denies_the_next_dispatch_of_a_fire_already_running(ac) -> None:
    # The attenuation cap is equally live: re-applied on every dispatch, not only the first.
    async def run() -> None:
        async with app.app_context(_manifest()), bind_execution_identity("k-owned", bound_fingerprint="fp-k-owned"):
            assert await app.tools.run_tool(_GRANTABLE_OP, _GRANTABLE_ARGS) == "read:m"

            ac.policy("owner")["scopes"] = [_KEPT_SCOPE]

            with pytest.raises(PermissionDenied, match="insufficient scope"):
                await app.tools.run_tool(_GRANTABLE_OP, _GRANTABLE_ARGS)

    asyncio.run(run())


def test_disabling_the_key_denies_the_next_fire(ac) -> None:
    async def run() -> None:
        async with app.app_context(_manifest()):
            assert await _run_as("k-scoped", _GRANTABLE_OP, _GRANTABLE_ARGS) == "read:m"

            ac.policy("k-scoped")["policy_data"] = {"disabled": True}

            with pytest.raises(PermissionDenied, match="is disabled"):
                await _run_as("k-scoped", _GRANTABLE_OP, _GRANTABLE_ARGS)

    asyncio.run(run())


def test_deleting_the_key_denies_the_next_dispatch_of_a_fire_already_running(ac) -> None:
    # Revoking DELETES the policy row, and the bound identity is built once — only a store
    # read per dispatch can stop a key deleted mid-turn.
    async def run() -> None:
        async with app.app_context(_manifest()), bind_execution_identity("k-scoped", bound_fingerprint="fp-k-scoped"):
            assert await app.tools.run_tool(_GRANTABLE_OP, _GRANTABLE_ARGS) == "read:m"

            ac.policies.remove(ac.policy("k-scoped"))

            with pytest.raises(PermissionDenied, match="principal has no policy"):
                await app.tools.run_tool(_GRANTABLE_OP, _GRANTABLE_ARGS)

    asyncio.run(run())


def test_deleting_the_key_denies_the_next_CAPABILITY_dispatch_too(ac) -> None:
    # A capability tool takes no authority decision, but "no authority model" is not "no
    # liveness": a key deleted mid-turn must stop it on its next dispatch.
    async def run() -> None:
        async with app.app_context(_manifest()), bind_execution_identity("k-scoped", bound_fingerprint="fp-k-scoped"):
            assert await app.tools.run_tool(_CAPABILITY_TOOL, {"text": "hi"}) == "hi"

            ac.policies.remove(ac.policy("k-scoped"))

            with pytest.raises(PermissionDenied, match="execution key 'k-scoped' has no policy"):
                await app.tools.run_tool(_CAPABILITY_TOOL, {"text": "hi"})

    asyncio.run(run())


def test_disabling_the_key_denies_the_next_CAPABILITY_dispatch_too(ac) -> None:
    # The other half of the liveness set, so the capability branch cannot drift to deletion only.
    async def run() -> None:
        async with app.app_context(_manifest()), bind_execution_identity("k-scoped", bound_fingerprint="fp-k-scoped"):
            assert await app.tools.run_tool(_CAPABILITY_TOOL, {"text": "hi"}) == "hi"

            ac.policy("k-scoped")["policy_data"] = {"disabled": True}

            with pytest.raises(PermissionDenied, match="execution key 'k-scoped' is disabled"):
                await app.tools.run_tool(_CAPABILITY_TOOL, {"text": "hi"})

    asyncio.run(run())


def test_reminting_the_key_denies_the_next_OPERATION_dispatch_of_a_running_fire(ac) -> None:
    # A remint of the same user_id writes a fresh fingerprint into the live policy. The
    # OPERATION branch must deny on the fingerprint mismatch, not authorize against the
    # reminted key's authority — the reminted "*" grants would otherwise clear the fence.
    async def run() -> None:
        async with app.app_context(_manifest()), bind_execution_identity("k-scoped", bound_fingerprint="fp-k-scoped"):
            # A matching fingerprint authorizes operations.
            assert await app.tools.run_tool(_GRANTABLE_OP, _GRANTABLE_ARGS) == "read:m"

            # The same user_id, reminted: a new fingerprint and admin grants.
            ac.policy("k-scoped")["scopes"] = ["*"]
            ac.policy("k-scoped")["policy_data"] = {KEY_FINGERPRINT_CLAIM: "fp-k-scoped-2"}

            with pytest.raises(PermissionDenied, match="no longer matches the bound key identity"):
                await app.tools.run_tool(_FENCED_OP, _FENCED_ARGS)

    asyncio.run(run())


def test_dropping_the_governing_roles_grant_denies_the_next_fire(monkeypatch: pytest.MonkeyPatch, ac) -> None:
    # A third authority axis: an owned key inherits its owner's role, so dropping that role's
    # grant to ``none`` denies the key's next fire with nothing written to the key's record.
    store = _MemStore()
    monkeypatch.setattr(versioning_module, "versioned_store", lambda: store)
    monkeypatch.setattr(versioning_module, "versioned_store_configured", lambda: True)

    async def run() -> None:
        async with app.app_context(_manifest()):
            tags = _grantable_probe_tags()
            await role_store().seed("ops", _role_body(tags, "write"))
            ac.policy("owner")["policy_data"] = {ROLE_POINTER_KEY: "ops"}
            assert await _run_as("k-owned", _GRANTABLE_OP, _GRANTABLE_ARGS) == "read:m"

            # The version bump is required: the grant cache is version-keyed.
            await role_store().update("ops", _role_body(tags, "none"))
            await management.bump_policy_version()

            with pytest.raises(PermissionDenied, match=f"GET {_GRANTABLE_PATH} is not permitted"):
                await _run_as("k-owned", _GRANTABLE_OP, _GRANTABLE_ARGS)

    asyncio.run(run())


def _grantable_probe_tags() -> list[str]:
    """The grantable probe route's feature tags, read from the live route table so a retagged
    probe cannot silently turn the grant edit into a no-op."""
    meta = resolve_route_meta(_GRANTABLE_PATH, "GET")
    assert meta is not None, f"{_GRANTABLE_PATH} is not a registered route"
    return list(meta.tags)


def _role_body(tags: list[str], level: str) -> dict:
    """A non-admin role granting ``level`` on every tag of the grantable probe route."""
    return {
        "name": "ops",
        "description": "the probe route's governing role",
        "scopes": ["*"],
        "grants": dict.fromkeys(tags, level),
        "condition": "true",
        "allow_all": False,
    }


def _policy_body(scopes: list[str]) -> dict:
    """A policy history body in the shape enforcement reads from the policy store."""
    return {
        "scopes": list(scopes),
        "policy_data": {KEY_FINGERPRINT_CLAIM: "fp-k-scoped"},
        "condition": None,
        "condition_id": None,
        "condition_kwargs": None,
    }


async def _stored_policy_version() -> int:
    """The stored policy version — the cache-invalidation counter, read as the enforcer reads it."""
    return await policy_module.PolicyEnforcer(access_control_settings()).current_policy_version()
