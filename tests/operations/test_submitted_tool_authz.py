""":func:`authorize_submitted_tool`, which ``submit_run`` and ``create_schedule`` call
before recording a caller-named tool, driven end to end on the LIVE stack.

A fenced operation is admin-only (the non-admin submitter HOLDS the route's scope, so the
deny is the per-tag level pass alone); a capability tool carries no per-call decision and
passes; a grantable operation passes for a caller holding its scope.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id
from tai42_contract.app import tai42_app

from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import role_grants as role_grants_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.role_gate import reset_route_index
from tai42_skeleton.app.instance import app
from tai42_skeleton.app.route_registry import route_registry
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.operations._submitted_tool_authz import authorize_submitted_tool
from tai42_skeleton.operations.errors import PermissionDenied
from tai42_skeleton.operations.registry import operation_registry
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

# The enforcer's alru cache is created and used across boots — a benign loop-reset test
# artifact, exactly as the other AC-e2e suites document.
pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")

_PROBE_ROUTER = "tests.authz._fixtures.execution_probe"
_FENCED_OP = "exec_probe_fenced"
_FENCED_PATH = "/api/exec-probe/deploy/fenced"
_GRANTABLE_OP = "exec_probe_read"
_GRANTABLE_PATH = "/api/exec-probe/read"
# A plain manifest tool: not an operation, so a capability tool.
_CAPABILITY_TOOL = "shout"
_PROBE_SCOPE = "probe"

_FENCED_ARGS = {"target": "deploy", "mark": "m"}
_GRANTABLE_ARGS = {"mark": "m"}


def _manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "api_tools": {"enabled": True},
            "tools": [{"title": "fxt", "module": "tests.app._fixtures.tools_b", "include": ["shout"]}],
            # Exactly the probe router: it carries both action-classes the matrix needs.
            "routers_modules": [_PROBE_ROUTER],
            "default_routers": "none",
        }
    )


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Snapshot and restore the process-global operation/route registries around each
    test — overriding the operations conftest's rebuild, since this suite boots a full app
    whose start() repopulates both from the probe manifest — so the probe router's
    operations and routes can never leak into another suite's registry view."""
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
    """A faked policy store carrying the probe routes and the two submitter keys the
    matrix drives: ``k-admin`` (condition-free ``"*"``, the admin discriminator) and
    ``k-scoped`` (holds the scope both probe routes are mapped to, and is NOT admin)."""
    pg = FakeAccessControlPg()
    redis = FakeRedis()
    pg.add_route(_FENCED_PATH, _PROBE_SCOPE)
    pg.add_route(_GRANTABLE_PATH, _PROBE_SCOPE)
    pg.add_policy("k-admin", scopes=["*"])
    pg.add_policy("k-scoped", scopes=[_PROBE_SCOPE])
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(redis))
    # The route index and the grant cache are process-global; rebuild both against this
    # test's registry so no earlier boot's index or grant map leaks in.
    reset_route_index()
    role_grants_module.reset_role_grants_cache()
    return pg


@contextmanager
def _as_caller(user_id: str):
    """Bind ``user_id`` as the request caller for the block — the identity
    :func:`authorize_submitted_tool` reads through ``resolve_caller_identity``."""
    token = set_request_user_id(user_id)
    try:
        yield
    finally:
        reset_request_user_id(token)


def test_a_fenced_tool_is_denied_for_a_non_admin_submitter(ac) -> None:
    # The submitter HOLDS the route's scope, so the scope pass allows: the deny is purely
    # the per-tag LEVEL pass fencing the operation to an admin.
    async def run() -> None:
        async with app.app_context(_manifest()):
            with _as_caller("k-scoped"), pytest.raises(PermissionDenied, match=f"POST {_FENCED_PATH} is not permitted"):
                await authorize_submitted_tool(_FENCED_OP, _FENCED_ARGS)

    asyncio.run(run())


def test_a_fenced_tool_is_allowed_for_an_admin_submitter(ac) -> None:
    # ALLOW parity, so the deny above is not vacuous: the same submission under an admin
    # key is authorized (no raise).
    async def run() -> None:
        async with app.app_context(_manifest()):
            with _as_caller("k-admin"):
                await authorize_submitted_tool(_FENCED_OP, _FENCED_ARGS)

    asyncio.run(run())


def test_a_capability_tool_passes_for_either_submitter(ac) -> None:
    # A non-operation has no route, so no per-call scope model exists for it at any edge;
    # it passes for the scope-holder and the admin alike.
    async def run() -> None:
        async with app.app_context(_manifest()):
            with _as_caller("k-scoped"):
                await authorize_submitted_tool(_CAPABILITY_TOOL, {"text": "hi"})
            with _as_caller("k-admin"):
                await authorize_submitted_tool(_CAPABILITY_TOOL, {"text": "hi"})

    asyncio.run(run())


def test_a_grantable_operation_the_caller_holds_passes(ac) -> None:
    # A grantable read the non-admin holds the route's scope for is authorized — the fix
    # denies the fence, not every submission.
    async def run() -> None:
        async with app.app_context(_manifest()):
            with _as_caller("k-scoped"):
                await authorize_submitted_tool(_GRANTABLE_OP, _GRANTABLE_ARGS)

    asyncio.run(run())
