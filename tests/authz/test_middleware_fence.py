"""The admin-only FENCE on the MCP dispatch edge, over the real fenced/secret routes.

Boots the live stack with the config router (a ``secret`` GET, a ``fenced`` POST and a
grantable ``read`` GET on neighbouring paths) over a faked policy store and dispatches
through the real ``AuthzMiddleware.on_call_tool``. A fenced/secret operation is admin-only
over MCP exactly as over its route, with the same hard-fence cause; the grantable operation
beside it still runs.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, cast

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import MiddlewareContext
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id

from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import role_grants as role_grants_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.request_scopes import (
    reset_request_effective_scopes,
    reset_request_identity_claims,
    set_request_effective_scopes,
    set_request_identity_claims,
)
from tai42_skeleton.access_control.role_gate import reset_route_index
from tai42_skeleton.access_control.roles import EDITOR_JQ
from tai42_skeleton.app.instance import app
from tai42_skeleton.authz.middleware import AuthzMiddleware
from tai42_skeleton.authz.resolver import resolve_dispatch
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.operations.errors import PermissionDenied
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

# The enforcer's alru cache is created and used across boots — a benign loop-reset artifact.
pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")

_SECRET_OP = "read_env"  # GET /api/config/env — the secret env dump
_SECRET_PATH = "/api/config/env"
_FENCED_OP = "write_env"  # POST /api/config/env
_GRANTABLE_OP = "read_mode"  # GET /api/config/mode
_GRANTABLE_PATH = "/api/config/mode"

_CONFIG_SCOPE = "config"  # the scope both config routes are mapped to


def _manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "api_tools": {"enabled": True},
            # Config router only: it carries the secret/fenced/grantable triple.
            "routers_modules": ["tai42_skeleton.routers.config"],
            "default_routers": "none",
        }
    )


@pytest.fixture
def ac(monkeypatch: pytest.MonkeyPatch) -> FakeAccessControlPg:
    """A faked policy store carrying the config routes and the two callers.

    ``editor-key`` clears both base-tier terms on every config path, so any deny it takes is
    the per-tag LEVEL pass alone; ``root`` is admin (unowned, condition-free ``"*"``).
    """
    pg = FakeAccessControlPg()
    redis = FakeRedis()
    pg.add_route(_SECRET_PATH, _CONFIG_SCOPE)
    pg.add_route(_GRANTABLE_PATH, _CONFIG_SCOPE)
    pg.add_policy("editor-key", scopes=["*"], condition=EDITOR_JQ)
    pg.add_policy("root", scopes=["*"])
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(redis))
    # Route index and grant cache are process-global; rebuild against this test's registry.
    reset_route_index()
    role_grants_module.reset_role_grants_cache()
    return pg


@dataclass
class _Msg:
    name: str
    arguments: dict


@dataclass
class _Ctx:
    message: _Msg


class _Dispatcher:
    """Drives a tool name through the real ``AuthzMiddleware`` with a ``call_next`` spy.
    ``on_call_tool`` runs before the extension/transform chain, so a deny precedes the body."""

    def __init__(self) -> None:
        self._mw = next(m for m in app._fast_mcp.middleware if isinstance(m, AuthzMiddleware))
        self.reached = 0

    async def _call_next(self, _ctx: Any) -> str:
        self.reached += 1
        return "reached"

    async def dispatch(self, name: str, arguments: dict | None = None) -> str:
        self.reached = 0
        ctx = cast("MiddlewareContext[Any]", _Ctx(_Msg(name, arguments or {})))
        return await self._mw.on_call_tool(ctx, self._call_next)


def _bind_caller(user_id: str, scopes: tuple[str, ...]):
    """Bind a caller as the resource guard does — id, effective scopes and verified claims
    as one set — returning the reset thunk."""
    tokens = (
        set_request_user_id(user_id),
        set_request_effective_scopes(scopes),
        set_request_identity_claims({"sub": user_id}),
    )

    def _reset() -> None:
        reset_request_user_id(tokens[0])
        reset_request_effective_scopes(tokens[1])
        reset_request_identity_claims(tokens[2])

    return _reset


async def _dispatch_as(user_id: str, scopes: tuple[str, ...], tool: str, arguments: dict | None = None):
    reset = _bind_caller(user_id, scopes)
    try:
        return await _Dispatcher().dispatch(tool, arguments)
    finally:
        reset()


async def _denied_as(user_id: str, scopes: tuple[str, ...], tool: str, arguments: dict | None = None) -> ToolError:
    """Dispatch and assert the deny is a ``PermissionDenied``-backed ``ToolError`` raised
    BEFORE ``call_next``; returns it for inspection."""
    dispatcher = _Dispatcher()
    reset = _bind_caller(user_id, scopes)
    try:
        with pytest.raises(ToolError) as excinfo:
            await dispatcher.dispatch(tool, arguments)
    finally:
        reset()
    assert isinstance(excinfo.value.__cause__, PermissionDenied), excinfo.value.__cause__
    assert dispatcher.reached == 0, f"{tool} reached call_next on a deny"
    return excinfo.value


def test_the_config_operations_project_as_tools_with_the_action_classes_the_matrix_assumes(ac) -> None:
    # If these stopped projecting, every dispatch below would pass through as a
    # non-operation and the denies would be vacuous.
    async def run() -> None:
        async with app.app_context(_manifest()):
            projected = await app.tools.get_tools()
            for name in (_SECRET_OP, _FENCED_OP, _GRANTABLE_OP):
                assert name in projected, f"{name} is not projected as an MCP tool"
                assert resolve_dispatch(name, {}, tool_registry=None, preset_manager=None) is not None, (
                    f"{name} does not resolve to an operation at the MCP edge"
                )

            from tai42_skeleton.access_control.role_gate import resolve_route_meta

            secret = resolve_route_meta(_SECRET_PATH, "GET")
            fenced = resolve_route_meta(_SECRET_PATH, "POST")
            grantable = resolve_route_meta(_GRANTABLE_PATH, "GET")
            assert secret is not None
            assert secret.action == "secret"
            assert fenced is not None
            assert fenced.action == "fenced"
            assert grantable is not None
            assert grantable.action == "read"

    asyncio.run(run())


def test_a_secret_operation_is_denied_over_the_mcp_edge_for_a_non_admin(ac, caplog) -> None:
    # Both base-tier terms allow for this caller, so the deny is the per-tag LEVEL pass,
    # carrying the same hard-fence cause the HTTP edge records.
    async def run() -> ToolError:
        async with app.app_context(_manifest()):
            with caplog.at_level(logging.WARNING):
                return await _denied_as("editor-key", ("*",), _SECRET_OP)

    error = asyncio.run(run())

    assert f"GET {_SECRET_PATH} is not permitted for 'editor-key'" in str(error)
    assert any("hard-fence" in record.getMessage() for record in caplog.records)


def test_a_fenced_operation_is_denied_over_the_mcp_edge_for_a_non_admin(ac) -> None:
    async def run() -> ToolError:
        async with app.app_context(_manifest()):
            return await _denied_as("editor-key", ("*",), _FENCED_OP, {"env": {"A": "1"}})

    assert f"POST {_SECRET_PATH} is not permitted for 'editor-key'" in str(asyncio.run(run()))


def test_a_secret_operation_runs_over_the_mcp_edge_for_an_admin(ac) -> None:
    # ALLOW parity, so the denies above are not vacuous.
    async def run() -> str:
        async with app.app_context(_manifest()):
            return await _dispatch_as("root", ("*",), _SECRET_OP)

    assert asyncio.run(run()) == "reached"


def test_a_grantable_operation_the_caller_holds_still_runs(ac) -> None:
    # The fence closes the admin-only classes and nothing else.
    async def run() -> str:
        async with app.app_context(_manifest()):
            return await _dispatch_as("editor-key", ("*",), _GRANTABLE_OP)

    assert asyncio.run(run()) == "reached"


def test_a_caller_without_the_scope_is_still_denied_at_the_scope_layer(ac) -> None:
    # The base tier is untouched by the fence: a scope shortfall denies before any
    # level question is asked.
    ac.add_policy("narrow", scopes=["other"])

    async def run() -> ToolError:
        async with app.app_context(_manifest()):
            return await _denied_as("narrow", ("other",), _GRANTABLE_OP)

    assert "insufficient scope" in str(asyncio.run(run()))
