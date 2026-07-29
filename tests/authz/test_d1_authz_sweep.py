"""D.1 universal-authz sweep — the core LIVE-stack security proof.

Boots the app through the real ``app.app_context`` harness with an ``api_tools``
manifest (no management tool modules loaded, projection enabled) that also wires
every dispatch-derivative the sweep must cover, then dispatches each modality
through the REAL ``AuthzMiddleware.on_call_tool`` installed on the booted
FastMCP surfaces — proving that, with access control ENABLED, a caller with no
resolvable identity (or insufficient scope) is DENIED and the denial surfaces as a
``ToolError`` specifically backed by :class:`PermissionDenied` (never a generic
error, never a silent allow), with ALLOW parity for a sufficient identity and for
``ACCESS_CONTROL_ENABLE=false``.

The modalities (cases a-e):
  (a) a plain projected op dispatched via MCP (``remove_tool``);
  (a-priv) an INCLUDED tier-2 privileged op (``update_manifest``) — so the sweep
      is never vacuous (a real privileged op is on the surface);
  (b) a BACKEND-kind combo over a projected op (``remove_tool`` + ``backendswap``)
      — the execution-relocating case, proving the check fires caller-side BEFORE
      the extension/transform chain (before any backend enqueue), not worker-side;
  (c) an in-process combo over a projected op (``remove_tool`` + ``argswrap``);
  (d) a PRESET baked over a projected op (proving the name→base-op resolver
      consults ``PresetManager``, not just the extension-branch map);
  (e) the same projected op reachable through a SUB-MCP MOUNT (``/app/{slug}``),
      proving ``AuthzMiddleware`` is installed on the sub-app's FastMCP, not only
      the main server.

For every modality the denial is asserted to be the ``PermissionDenied``-backed
``ToolError`` SPECIFICALLY (type + message), and ``call_next`` is proven NOT
reached on a deny — mechanically enforcing "no projected tool is dispatchable
externally without a check, on any MCP surface, through any derivative".
"""

from __future__ import annotations

import asyncio
import pkgutil
import sys
from dataclasses import dataclass
from typing import Any, cast

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import MiddlewareContext
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id

import tai42_skeleton.routers as _routers_pkg
from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.request_scopes import (
    reset_request_effective_scopes,
    reset_request_identity_claims,
    set_request_effective_scopes,
    set_request_identity_claims,
)
from tai42_skeleton.access_control.role_gate import reset_route_index
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.app.instance import app
from tai42_skeleton.authz.middleware import AuthzMiddleware
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.operations.errors import PermissionDenied
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

# The enforcer's alru cache is created and used across boots — a benign loop-reset
# test artifact, exactly as the other AC-e2e suite documents.
pytestmark = pytest.mark.filterwarnings("ignore::async_lru.AlruCacheLoopResetWarning")


# Infra router modules carrying no projectable op; excluded so the stack loads only
# the operation-bearing routers (importing prometheus/metrics mutates process-global
# multiproc state that would leak into the metrics-CLI tests).
_INFRA_ROUTERS = frozenset(
    {"_tool_call", "health", "metrics", "metrics_settings", "observability_support", "prometheus", "tool_runs_settings"}
)


def _all_router_modules() -> list[str]:
    return [
        info.name
        for info in pkgutil.iter_modules(_routers_pkg.__path__, _routers_pkg.__name__ + ".")
        if info.name.rsplit(".", 1)[-1] not in _INFRA_ROUTERS
    ]


# Two routes carry the whole sweep: the plain/derivative ops all key on
# ``/api/tools/remove`` (``remove_tool`` and every derivative over it), and the
# included tier-2 op keys on ``/api/manifest/replace``. ``admin`` is the scope both
# guard. Both routes are FENCED, so the per-tag level pass admits only the admin
# discriminator: ``alice`` is an unowned condition-free ``"*"``, ``bob`` holds nothing.
def _seed_ac(monkeypatch: pytest.MonkeyPatch) -> FakeAccessControlPg:
    pg = FakeAccessControlPg()
    redis = FakeRedis()
    pg.add_route("/api/tools/remove", "admin")
    pg.add_route("/api/manifest/replace", "admin")
    pg.add_policy("alice", scopes=["*"])
    pg.add_policy("bob", scopes=[])
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(redis))
    # The route index is process-global; rebuild it or an earlier boot's index un-fences a route.
    reset_route_index()
    return pg


def _sweep_manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "api_tools": {
                "enabled": True,
                # (a-priv) an included tier-2 privileged op, so a real privileged op
                # is on the surface to sweep — never vacuous.
                "include": ["update_manifest"],
                # (b) a BACKEND-kind combo and (c) an in-process combo, both over the
                # projected op ``remove_tool``.
                "extensions": {"remove_tool": [["backendswap"], ["argswrap"]]},
            },
            "extensions_modules": ["tests.app._fixtures.ext_kinds"],
            "routers_modules": _all_router_modules(),
            # "none" keeps the surface to exactly the operation-bearing routers
            # listed here (infra routers deliberately excluded), so the default core
            # set does not re-introduce the prometheus/metrics multiproc-state imports.
            "default_routers": "none",
        }
    )


@dataclass
class _Msg:
    name: str
    arguments: dict


@dataclass
class _Ctx:
    message: _Msg


class _Dispatcher:
    """Drives a tool name through a real ``AuthzMiddleware`` instance with a
    ``call_next`` spy — the exact caller-side path ``on_call_tool`` runs before the
    tool's extension/transform chain, so a deny here is a deny BEFORE any enqueue."""

    def __init__(self, mw: AuthzMiddleware) -> None:
        self._mw = mw
        self.reached = 0

    async def _call_next(self, _ctx: Any) -> str:
        self.reached += 1
        return "reached"

    async def dispatch(self, name: str, arguments: dict | None = None) -> str:
        self.reached = 0
        ctx = cast("MiddlewareContext[Any]", _Ctx(_Msg(name, arguments or {})))
        return await self._mw.on_call_tool(ctx, self._call_next)

    async def assert_denied(self, name: str) -> None:
        """The dispatch raises the ``PermissionDenied``-backed ``ToolError``
        SPECIFICALLY and never reaches ``call_next`` (before any enqueue)."""
        with pytest.raises(ToolError) as excinfo:
            await self.dispatch(name)
        assert isinstance(excinfo.value.__cause__, PermissionDenied), excinfo.value.__cause__
        assert self.reached == 0, f"{name} reached call_next on a deny"

    async def assert_allowed(self, name: str) -> None:
        result = await self.dispatch(name)
        assert result == "reached"
        assert self.reached == 1


def _main_dispatcher() -> _Dispatcher:
    mw = next(m for m in app._fast_mcp.middleware if isinstance(m, AuthzMiddleware))
    return _Dispatcher(mw)


# The derivative tool names on the sweep stack.
_A_PLAIN = "remove_tool"
_A_PRIV = "update_manifest"  # tier-2, explicitly included
_B_BACKEND = "remove_tool_backendswap"  # BACKEND-kind combo branch
_C_INPROC = "remove_tool_argswrap"  # in-process (WRAPPER) combo branch
_D_PRESET = "d1_sweep_preset"  # preset baked over remove_tool
_MAIN_MODALITIES = [_A_PLAIN, _A_PRIV, _B_BACKEND, _C_INPROC, _D_PRESET]


async def _register_preset() -> None:
    """Bake a preset over the projected ``remove_tool`` so the resolver's
    ``PresetManager`` consultation is exercised (case d)."""
    await app.preset_manager.register(_D_PRESET, _A_PLAIN, {}, [], [], "")


@pytest.fixture(autouse=True)
def _tear_down_sweep_preset():
    """The process ``PresetManager`` outlives one ``app_context``, so the sweep
    preset would leak into the next test as a false name collision. Drop it after
    each test (store-free teardown)."""
    yield
    manager = app.preset_manager
    if manager.is_registered(_D_PRESET):
        asyncio.run(manager.remove(_D_PRESET))


# -- the sweep: DENY on every modality, ALLOW parity --------------------------


def test_d1_sweep_deny_without_identity_over_every_modality(monkeypatch: pytest.MonkeyPatch):
    """Access control ENABLED, no resolvable identity: every modality is DENIED
    with the ``PermissionDenied``-backed ``ToolError`` before ``call_next``."""
    _seed_ac(monkeypatch)

    async def run():
        async with app.app_context(_sweep_manifest()):
            await _register_preset()
            d = _main_dispatcher()
            # (a) plain, (a-priv) included tier-2, (b) backend combo, (c) in-process
            # combo, (d) preset — all denied on the MAIN MCP edge.
            for name in _MAIN_MODALITIES:
                await d.assert_denied(name)

    asyncio.run(run())


def _default_projected_names() -> list[str]:
    """Every DEFAULT-projected operation name, taken from the REAL projection over the
    live registry — so this exhaustive sweep can never silently drift from what
    actually projects (a new leaf op joins the sweep the moment it projects)."""
    from tai42_contract.manifest import ApiToolsConfig

    from tai42_skeleton.operations import operation_registry, project_operations

    class _Recorder:
        def tool(self, *, force: bool, name: str, tags: Any, annotations: Any):
            return lambda fn: fn

    class _RecorderApp:
        def __init__(self) -> None:
            self.tools = _Recorder()

    return project_operations(_RecorderApp(), ApiToolsConfig(enabled=True), registry=operation_registry)


def test_d1_sweep_deny_covers_every_default_projected_op(monkeypatch: pytest.MonkeyPatch):
    """Exhaustive fail-safe: EVERY default-projected op — not just a
    representative — is DENIED on the plain-op modality (AC enabled, no identity) at
    the REAL ``AuthzMiddleware`` edge. This is the mechanical proof that no projected
    tool is dispatchable without a check; a router/sub-mcp/preset gap fails HERE, not
    in production. Cheap: caller-side only, no backend."""
    _seed_ac(monkeypatch)

    async def run():
        async with app.app_context(_sweep_manifest()):
            projected = _default_projected_names()
            assert projected, "no operations projected — the sweep would be vacuous"
            d = _main_dispatcher()
            swept = 0
            for name in projected:
                await d.assert_denied(name)
                swept += 1
            # Non-vacuous: every projected op was swept, none skipped.
            assert swept == len(projected)

    asyncio.run(run())


def test_d1_sweep_deny_with_insufficient_scope_over_every_modality(monkeypatch: pytest.MonkeyPatch):
    """A resolvable caller whose scope does NOT cover the op's resource is denied
    identically (insufficient scope), on every modality."""
    _seed_ac(monkeypatch)

    async def run():
        async with app.app_context(_sweep_manifest()):
            await _register_preset()
            d = _main_dispatcher()
            tok = set_request_user_id("bob")  # bob holds no scopes
            try:
                for name in _MAIN_MODALITIES:
                    with pytest.raises(ToolError) as excinfo:
                        await d.dispatch(name)
                    assert isinstance(excinfo.value.__cause__, PermissionDenied)
                    assert "insufficient scope" in str(excinfo.value)
                    assert d.reached == 0
            finally:
                reset_request_user_id(tok)

    asyncio.run(run())


def test_d1_sweep_allow_with_sufficient_identity_over_every_modality(monkeypatch: pytest.MonkeyPatch):
    """ALLOW parity: a caller the whole edge decision admits (the swept ops are fenced, so
    that is the admin discriminator) reaches ``call_next`` on every modality."""
    _seed_ac(monkeypatch)

    async def run():
        async with app.app_context(_sweep_manifest()):
            await _register_preset()
            d = _main_dispatcher()
            tok = set_request_user_id("alice")  # alice is the admin discriminator
            try:
                for name in _MAIN_MODALITIES:
                    await d.assert_allowed(name)
            finally:
                reset_request_user_id(tok)

    asyncio.run(run())


def test_d1_sweep_allow_when_access_control_disabled(monkeypatch: pytest.MonkeyPatch):
    """With ``ACCESS_CONTROL_ENABLE=false`` the tool edge allows everything — the
    deny-on-no-identity rule applies only when access control is enabled."""
    _seed_ac(monkeypatch)
    check_mod = sys.modules["tai42_skeleton.authz.check"]
    monkeypatch.setattr(check_mod, "access_control_settings", lambda: AccessControlSettings(enable=False))

    async def run():
        async with app.app_context(_sweep_manifest()):
            await _register_preset()
            d = _main_dispatcher()
            # No identity bound, yet every modality passes straight through.
            for name in _MAIN_MODALITIES:
                await d.assert_allowed(name)

    asyncio.run(run())


# -- case (e): the sub-MCP mount ----------------------------------------------


def _sub_dispatcher(monkeypatch: pytest.MonkeyPatch, built_instances: list) -> None:
    """Record every sub-MCP ``FastMCP`` built so the test can reach the instance
    (and its installed middleware), not merely its ASGI http app."""
    import tai42_skeleton.app.sub_mcp_app as sub_mod

    real = sub_mod.FastMCP

    class _Recording(real):  # type: ignore[valid-type, misc]
        def __init__(self, *a, **k) -> None:
            super().__init__(*a, **k)
            built_instances.append(self)

    monkeypatch.setattr(sub_mod, "FastMCP", _Recording)


def test_d1_sweep_sub_mcp_mount_deny_and_allow(monkeypatch: pytest.MonkeyPatch):
    """Case (e): the projected op reached through a SUB-MCP mount is governed by an
    ``AuthzMiddleware`` installed on the SUB-app's own FastMCP — deny without
    identity, allow with a sufficient one. Proves the main server's middleware is
    not the only guard (a missing sub-app install would let this through)."""
    _seed_ac(monkeypatch)
    built: list = []
    _sub_dispatcher(monkeypatch, built)

    from tai42_skeleton.app.sub_mcp_app import SubMcpAppRouter

    async def run():
        async with app.app_context(_sweep_manifest()):
            router = cast("SubMcpAppRouter", app.sub_app.mcp_sub_app_router)
            async with router.lifespan(cast("Any", None)):
                await router.register_sub_mcp_app("d1svc", [_A_PLAIN], transport="http")
                await router._get_or_build_app("d1svc")

            assert built, "no sub-MCP FastMCP was built"
            sub_mw = next(m for inst in built for m in inst.middleware if isinstance(m, AuthzMiddleware))
            d = _Dispatcher(sub_mw)

            # Deny without identity — the SUB-app's middleware fires.
            await d.assert_denied(_A_PLAIN)

            # Allow with an identity the whole edge decision admits.
            tok = set_request_user_id("alice")
            try:
                await d.assert_allowed(_A_PLAIN)
            finally:
                reset_request_user_id(tok)

    asyncio.run(run())


# -- owned-key OWNER-attenuation parity on the MCP edge (from C6b) -------------


def test_d1_owned_key_attenuation_holds_on_the_mcp_dispatch_edge(monkeypatch: pytest.MonkeyPatch):
    """The MCP dispatch consumes the HTTP edge's owner-attenuated effective scopes
    (bound as a pair with the caller id), so an owned/delegated key attenuated
    BELOW an op's requirement is denied at the tool edge even though its OWN policy
    scopes would allow it — MCP never out-permits HTTP.

    On a FENCED op the fence keys on the CALLER, not the owner: an admin cannot delegate
    fence access by minting a broad owned key. The two legs therefore deny for two distinct
    reasons — insufficient scope, then the hard fence.

    (The full HTTP↔MCP owner-attenuation parity is pinned end-to-end through the
    real guard in ``tests/authz/test_check.py::test_owned_key_attenuation_parity_
    http_mcp``; this asserts the same attenuation is honored on the real
    ``AuthzMiddleware`` DISPATCH path for a projected op.)"""
    pg = _seed_ac(monkeypatch)
    # A delegated key whose OWN policy scopes include "admin", but whose
    # owner-attenuated effective scopes do NOT. Its owner is the admin discriminator, yet
    # the swept ops are FENCED, so it never clears the fence.
    pg.add_policy("delegated", scopes=["admin"], policy_data={OWNER_USER_ID_CLAIM: "alice"})

    async def run():
        async with app.app_context(_sweep_manifest()):
            d = _main_dispatcher()
            # The guard binds id, effective scopes and verified claims as one set.
            tok_id = set_request_user_id("delegated")
            tok_cl = set_request_identity_claims({OWNER_USER_ID_CLAIM: "alice"})
            # Attenuated below the requirement: the guard bound effective_scopes
            # that EXCLUDE "admin" (owner ∩ key). The scope term denies first.
            tok_sc = set_request_effective_scopes(("read",))
            try:
                with pytest.raises(ToolError) as excinfo:
                    await d.dispatch(_A_PLAIN)
                assert isinstance(excinfo.value.__cause__, PermissionDenied)
                assert "insufficient scope" in str(excinfo.value)
                assert d.reached == 0
            finally:
                reset_request_effective_scopes(tok_sc)

            # Scope-sufficient: the scope term passes, but an owned key is never the admin
            # principal, so the hard fence denies.
            tok_sc = set_request_effective_scopes(("admin",))
            try:
                with pytest.raises(ToolError) as excinfo:
                    await d.dispatch(_A_PLAIN)
                assert isinstance(excinfo.value.__cause__, PermissionDenied)
                assert "is not permitted" in str(excinfo.value)
                assert d.reached == 0
            finally:
                reset_request_effective_scopes(tok_sc)
                reset_request_identity_claims(tok_cl)
                reset_request_user_id(tok_id)

    asyncio.run(run())
