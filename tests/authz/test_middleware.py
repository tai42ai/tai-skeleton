"""``AuthzMiddleware``: the ``on_call_tool`` deny/allow/pass-through behavior and
the install assertions on the main server and every sub-MCP app."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import MiddlewareContext

from tai_skeleton.access_control.settings import AccessControlSettings
from tai_skeleton.authz.middleware import AuthzMiddleware
from tai_skeleton.operations import OperationRegistry, operation


@dataclass
class _Msg:
    name: str
    arguments: dict


@dataclass
class _Ctx:
    message: _Msg


class _FakeToolRegistry:
    def base_of(self, name: str) -> str:
        return name


@dataclass
class _Body:
    base_tool: str


class _FakePresetManager:
    """Models ``PresetManager`` for the resolver: a registered preset maps to its
    ``base_tool`` (which the plan allows to be a projected operation)."""

    def __init__(self, specs: dict[str, str]) -> None:
        self._specs = {n: _Body(base_tool=b) for n, b in specs.items()}

    def is_registered(self, name: str) -> bool:
        return name in self._specs

    def get_spec(self, name: str) -> _Body:
        return self._specs[name]


class _FakeApp:
    """An app exposing the two derivative maps the resolver consults, plus an
    operation registry patched onto the resolver module for the test."""

    def __init__(self, preset_manager: Any = None) -> None:
        self._tool_registry = _FakeToolRegistry()
        self.preset_manager = preset_manager


def _op_registry() -> OperationRegistry:
    reg = OperationRegistry()

    @operation(name="wipe", summary="s", tags=["t"], registry=reg)
    async def wipe(**_):
        return {"ok": True}

    reg.get("wipe").route_template = "/api/things/wipe"
    reg.get("wipe").http_method = "POST"
    return reg


async def _call(mw: AuthzMiddleware, name: str):
    called = {"v": False}

    async def call_next(_ctx):
        called["v"] = True
        return "reached"

    result = await mw.on_call_tool(cast("MiddlewareContext[Any]", _Ctx(_Msg(name, {}))), call_next)
    return called["v"], result


def test_projected_op_denied_without_identity(monkeypatch):
    import sys

    import tai_skeleton.authz.resolver as resolver_mod

    check_mod = sys.modules["tai_skeleton.authz.check"]

    reg = _op_registry()
    monkeypatch.setattr(resolver_mod, "operation_registry", reg)
    monkeypatch.setattr(check_mod, "access_control_settings", lambda: AccessControlSettings())

    mw = AuthzMiddleware(_FakeApp())
    with pytest.raises(ToolError, match="no caller identity"):
        asyncio.run(_call(mw, "wipe"))


def test_projected_op_allowed_when_access_control_disabled(monkeypatch):
    import sys

    import tai_skeleton.authz.resolver as resolver_mod

    check_mod = sys.modules["tai_skeleton.authz.check"]

    reg = _op_registry()
    monkeypatch.setattr(resolver_mod, "operation_registry", reg)
    monkeypatch.setattr(check_mod, "access_control_settings", lambda: AccessControlSettings(enable=False))

    mw = AuthzMiddleware(_FakeApp())
    reached, result = asyncio.run(_call(mw, "wipe"))
    assert reached is True
    assert result == "reached"


def test_non_operation_tool_passes_through(monkeypatch):
    import sys

    import tai_skeleton.authz.resolver as resolver_mod

    check_mod = sys.modules["tai_skeleton.authz.check"]

    reg = _op_registry()
    monkeypatch.setattr(resolver_mod, "operation_registry", reg)
    monkeypatch.setattr(check_mod, "access_control_settings", lambda: AccessControlSettings())

    mw = AuthzMiddleware(_FakeApp())
    # A tool name that resolves to no operation is not op-authz's concern.
    reached, result = asyncio.run(_call(mw, "some_toolbox_tool"))
    assert reached is True
    assert result == "reached"


def test_preset_over_projected_op_resolves_and_runs_authz_on_base(monkeypatch):
    """The preset half of the MCP-parity guarantee: a preset whose ``base_tool`` is a
    projected operation, dispatched via the MCP ``on_call_tool`` edge, resolves to
    that base op and runs ``authz.check`` against the BASE op's metadata — DENY
    parity with dispatching the base op directly (real check, no identity)."""
    import sys

    import tai_skeleton.authz.middleware as mw_mod
    import tai_skeleton.authz.resolver as resolver_mod

    check_mod = sys.modules["tai_skeleton.authz.check"]

    reg = _op_registry()
    monkeypatch.setattr(resolver_mod, "operation_registry", reg)
    monkeypatch.setattr(check_mod, "access_control_settings", lambda: AccessControlSettings())

    # Capture which operation metadata the check runs against, delegating to the REAL
    # check so the deny/allow outcome is the base op's, not a stub's.
    seen: list[str] = []
    real_check = mw_mod.check

    async def spy_check(identity, op, arguments):
        seen.append(op.name)
        return await real_check(identity, op, arguments)

    monkeypatch.setattr(mw_mod, "check", spy_check)

    app = _FakeApp(preset_manager=_FakePresetManager({"my_preset": "wipe"}))
    mw = AuthzMiddleware(app)

    # Dispatching the PRESET denies (no identity) — the base op's decision.
    with pytest.raises(ToolError, match="no caller identity"):
        asyncio.run(_call(mw, "my_preset"))
    # The check keyed on the BASE op's metadata (name "wipe"), not the preset name.
    assert seen == ["wipe"]

    # Parity: dispatching the base op DIRECTLY denies identically.
    seen.clear()
    with pytest.raises(ToolError, match="no caller identity"):
        asyncio.run(_call(mw, "wipe"))
    assert seen == ["wipe"]


def test_preset_over_projected_op_allowed_when_access_control_disabled(monkeypatch):
    """ALLOW parity: with access control disabled, dispatching the preset over a
    projected op passes straight through to ``call_next`` — exactly as the base op."""
    import sys

    import tai_skeleton.authz.resolver as resolver_mod

    check_mod = sys.modules["tai_skeleton.authz.check"]

    reg = _op_registry()
    monkeypatch.setattr(resolver_mod, "operation_registry", reg)
    monkeypatch.setattr(check_mod, "access_control_settings", lambda: AccessControlSettings(enable=False))

    app = _FakeApp(preset_manager=_FakePresetManager({"my_preset": "wipe"}))
    mw = AuthzMiddleware(app)
    reached, result = asyncio.run(_call(mw, "my_preset"))
    assert reached is True
    assert result == "reached"


def test_authz_middleware_installed_on_main_server():
    from tai_skeleton.app.instance import app

    assert any(isinstance(m, AuthzMiddleware) for m in app._fast_mcp.middleware)


async def test_authz_middleware_installed_on_sub_mcp_app(monkeypatch):
    """Every sub-MCP FastMCP built by ``_build_sub_app`` re-adds ``AuthzMiddleware``
    (the main server's middleware never reaches a sub-mount)."""
    import tai_skeleton.app.sub_mcp_app as sub_mod
    from tai_skeleton.app.instance import app
    from tai_skeleton.app.sub_mcp_app import SubMcpAppRouter
    from tai_skeleton.manifest import Manifest

    instances: list = []
    real_fastmcp = sub_mod.FastMCP

    class _RecordingFastMCP(real_fastmcp):
        def __init__(self, *a, **k) -> None:
            super().__init__(*a, **k)
            instances.append(self)

    monkeypatch.setattr(sub_mod, "FastMCP", _RecordingFastMCP)

    manifest = Manifest.model_validate(
        {"tools": [{"title": "fxt", "module": "tests.app._fixtures.tools_a", "include": ["greet"]}]}
    )

    async with app.app_context(manifest):
        from starlette.applications import Starlette

        router = cast("SubMcpAppRouter", app.sub_app.mcp_sub_app_router)
        async with router.lifespan(cast("Starlette", None)):
            await router.register_sub_mcp_app("http_svc", ["greet"], transport="http")
            built = await router._get_or_build_app("http_svc")
            assert built is not None

    assert instances, "no sub-MCP FastMCP was built"
    assert any(any(isinstance(m, AuthzMiddleware) for m in inst.middleware) for inst in instances), (
        "AuthzMiddleware not installed on any sub-MCP app"
    )
