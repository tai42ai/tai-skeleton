"""D.1 LIVE-stack verification of the MCP projection.

Boots the app through the real ``app.app_context`` harness with an ``api_tools``
manifest that loads no management tool modules and enables projection, then
asserts the projected surface end-to-end (checklist items 1-6):

1. the projected tool surface is exactly the expected op surface — the 90
   default-projected ops (118 total - 25 tier-2 default-excluded - 3 tier-1
   hardcode-blocked);
2. ``destructiveHint`` is present on destructive ops (a DELETE, a mutating POST)
   and absent on reads (a GET);
3. a manifest that BOTH hand-binds a tool named after a projected op AND
   projects that op fails boot loudly (the duplicate-bind collision guard);
4. a tier-2 op (``/api/auth/*`` / ``update_manifest``) is absent by default but
   projects when named in ``api_tools.include``;
5. a tier-1 meta-executor (``run_tool`` + ``submit_run``) stays hardcode-absent
   even when explicitly listed in ``api_tools.include`` (loud startup log);
6. ``user_tools`` curation still works alongside ``api_tools``.

These boot the FULL product router set so every operation carries its route
template + method (the tier-2 ``/api/auth/*`` prefix classification and the
DELETE-forces-destructive rule both need the route attached), matching a
realistic production manifest shape.
"""

from __future__ import annotations

import asyncio
import logging
import pkgutil

import pytest
from tai_contract.manifest import ApiToolsConfig

import tai_skeleton.routers as _routers_pkg
from tai_skeleton.app.instance import app
from tai_skeleton.manifest import Manifest
from tai_skeleton.operations.projection import is_tier1, is_tier2, project_operations
from tai_skeleton.operations.registry import operation_registry

# Infra router modules that carry NO projectable operation (metrics/health/native
# helpers). Excluded from the boot list so the stack loads only the operation-bearing
# routers — importing the prometheus/metrics modules mutates process-global
# multiproc state, which would leak into the metrics-CLI tests.
_INFRA_ROUTERS = frozenset(
    {"_tool_call", "health", "metrics", "metrics_settings", "observability_support", "prometheus", "tool_runs_settings"}
)


def _all_router_modules() -> list[str]:
    """Every OPERATION-bearing module in the product ``routers`` package — the HTTP
    surface a realistic deployment mounts, so each op's route template + method
    attach before projection (the tier-2 ``/api/auth/*`` prefix classification and
    the DELETE-forces-destructive rule both need the route attached)."""
    return [
        info.name
        for info in pkgutil.iter_modules(_routers_pkg.__path__, _routers_pkg.__name__ + ".")
        if info.name.rsplit(".", 1)[-1] not in _INFRA_ROUTERS
    ]


def _manifest(**api_tools: object) -> Manifest:
    body: dict = {"enabled": True}
    body.update(api_tools)
    return Manifest.model_validate({"api_tools": body, "routers_modules": _all_router_modules()})


class _RecordingTools:
    """Captures ``app.tools.tool(...)`` calls the projection makes, so a test can
    measure the projected set without binding onto the shared FastMCP server."""

    def __init__(self) -> None:
        self.registered: dict[str, object] = {}

    def tool(self, *, force, name, tags, annotations):
        def decorator(func):
            self.registered[name] = annotations
            return func

        return decorator


class _RecordingApp:
    def __init__(self) -> None:
        self.tools = _RecordingTools()


# -- checklist 1: the projected surface is exactly the 90 default ops ----------


def test_d1_projected_surface_is_the_expected_op_count():
    async def run():
        async with app.app_context(_manifest()):
            reg = operation_registry
            ops = reg.all()
            total = len(ops)
            tier1 = sorted(op.name for op in ops if is_tier1(op))
            tier2 = sorted(op.name for op in ops if is_tier2(op) and not is_tier1(op))

            # The arithmetic the plan pins: 118 total - 25 tier-2 - 3 tier-1 = 90.
            assert total == 118, total
            # Tier-1 (never projectable): the two meta-executors PLUS ``get_me``, whose
            # params are the caller's own edge-derived identity (``caller_context=True``).
            assert tier1 == ["get_me", "run_tool", "submit_run"], tier1
            # The tier-2 set is the 18 non-``get_me`` api_keys ops (all under /api/auth/*,
            # including create_claim_link) + logout (/api/auth/logout) +
            # exchange_claim_token (authority_changing — a public credential door that
            # must never project) + import_backup + update_manifest + the three
            # marketplace mutators (install/uninstall/update, each authority_changing
            # because it runs arbitrary third-party code) — 25 in all. ``get_me`` is NOT
            # here: it is tier-1 hardcode-blocked, not tier-2 includable.
            assert set(tier2) == {
                "add_scope_url",
                "create_api_key",
                "create_claim_link",
                "delete_scope",
                "edit_api_key",
                "exchange_claim_token",
                "get_capabilities",
                "import_backup",
                "list_policy_versions",
                "list_public_routes",
                "list_roles",
                "list_routes",
                "list_scopes",
                "list_tokens_payload",
                "logout",
                "marketplace_install",
                "marketplace_uninstall",
                "marketplace_update",
                "pin_public_route",
                "remove_scope_url",
                "revoke_api_key",
                "rollback_policy",
                "unpin_public_route",
                "update_manifest",
                "validate_condition",
            }, tier2

            # The default-projected surface = 90, measured two ways.
            recorder = _RecordingApp()
            projected = project_operations(recorder, ApiToolsConfig(), registry=reg)
            assert len(projected) == 90, len(projected)
            assert total - len(tier2) - len(tier1) == 90

            # And the LIVE booted tool surface is exactly those 90 (no keep-set /
            # plugin / toolbox tools are loaded in this projection-only stack).
            live = await app.tools.get_tools()
            assert set(live) == set(projected)
            assert len(live) == 90

            # Tier-1 and default tier-2 never appear on the live surface.
            assert "run_tool" not in live
            assert "submit_run" not in live
            assert "update_manifest" not in live  # tier-2, not included
            assert "delete_scope" not in live  # tier-2 by /api/auth prefix

    asyncio.run(run())


# -- checklist 2: destructiveHint on destructive ops, absent on reads ----------


def test_d1_destructive_hint_present_on_mutations_absent_on_reads():
    async def run():
        async with app.app_context(_manifest()):
            live = await app.tools.get_tools()

            def hint(name: str) -> object:
                assert name in live, f"{name} not projected"
                ann = live[name].annotations
                return getattr(ann, "destructiveHint", None) if ann is not None else None

            # A DELETE op (destructive auto-forced by the adapter).
            assert hint("unregister_hook") is True
            assert hint("delete_topic_verifier") is True
            # A mutating POST op tagged destructive.
            assert hint("remove_tool") is True
            assert hint("notify_user") is True
            # A GET read carries no destructive hint.
            assert hint("list_system_kinds") in (None, False)
            assert hint("list_hooks") in (None, False)
            assert hint("list_channels") in (None, False)

    asyncio.run(run())


# -- checklist 3: the duplicate-bind collision guard fires at boot -------------


def test_d1_duplicate_bind_of_builtin_and_projection_fails_boot():
    """A manifest that hand-binds a tool named ``reload_config`` (via a
    ``tools[]`` module) while ``api_tools`` projects the SAME op name must fail
    boot loudly — never a running window with both surfaces. The tool binding
    raises on the duplicate name."""

    async def run():
        manifest = Manifest.model_validate(
            {
                "api_tools": {"enabled": True},
                "routers_modules": _all_router_modules(),
                "tools": [{"title": "collide", "module": "tests.operations._fixtures.collide_projected"}],
            }
        )
        async with app.app_context(manifest):
            pass

    with pytest.raises(Exception, match="already exists"):
        asyncio.run(run())


# -- checklist 4: tier-2 absent by default, includable ------------------------


def test_d1_tier2_absent_by_default_but_projects_when_included():
    async def run():
        # Default: the tier-2 ops are off the live surface.
        async with app.app_context(_manifest()):
            live = await app.tools.get_tools()
            assert "update_manifest" not in live  # authority_changing flag
            assert "add_scope_url" not in live  # /api/auth/* prefix

        # Explicitly included: they project as real live tools.
        async with app.app_context(_manifest(include=["update_manifest", "add_scope_url"])):
            live = await app.tools.get_tools()
            assert "update_manifest" in live
            assert "add_scope_url" in live

    asyncio.run(run())


# -- checklist 5: tier-1 hardcode-blocked even when included -------------------


def test_d1_tier1_blocked_even_when_explicitly_included(caplog):
    async def run():
        with caplog.at_level(logging.WARNING, logger="tai_skeleton.operations.projection"):
            async with app.app_context(_manifest(include=["run_tool", "submit_run"])):
                live = await app.tools.get_tools()
                # Neither meta-executor is projected despite the explicit include.
                assert "run_tool" not in live
                assert "submit_run" not in live

    asyncio.run(run())

    # The block is loud: a WARNING names each meta-executor kept off the surface.
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("run_tool" in m and "hardcode-blocked meta-executor" in m for m in warned)
    assert any("submit_run" in m and "hardcode-blocked meta-executor" in m for m in warned)


# -- checklist 5b: get_me (caller-context identity op) is tier-1, never projectable --


def test_d1_get_me_is_caller_context_tier1_never_projectable(caplog):
    """``get_me`` returns the caller's OWN capability projection from identity params
    the HTTP edge injects; as an MCP tool a caller would supply those params itself and
    read ANY principal's projection. It is ``caller_context=True`` (tier-1), so it must
    never reach the tool surface — not by default, and not even when explicitly
    included — with a loud, accurately-reasoned block log."""

    async def run():
        async with app.app_context(_manifest()):
            reg = operation_registry
            get_me_op = reg.get("get_me")
            # Classified tier-1 (never projectable), NOT tier-2 (includable).
            assert is_tier1(get_me_op)
            assert not (is_tier2(get_me_op) and not is_tier1(get_me_op))

            # Absent from the default live surface AND from the raw projected set.
            live = await app.tools.get_tools()
            assert "get_me" not in live
            projected = project_operations(_RecordingApp(), ApiToolsConfig(), registry=reg)
            assert "get_me" not in projected

        # Even when explicitly included it stays off the surface, with a loud log naming
        # it a caller-context identity op (not a meta-executor — the reason is accurate).
        with caplog.at_level(logging.WARNING, logger="tai_skeleton.operations.projection"):
            async with app.app_context(_manifest(include=["get_me"])):
                live = await app.tools.get_tools()
                assert "get_me" not in live

    asyncio.run(run())

    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("get_me" in m and "caller-context identity op" in m for m in warned)


# -- checklist 6: user_tools curation coexists with api_tools -----------------


def test_d1_user_tools_curation_coexists_with_api_tools():
    """``api_tools`` decides what is REGISTERED (projection); ``user_tools`` is the
    read-time flow-builder view filter, carried in the live manifest and exposed to
    the flow-builder surface. Both apply: the projected op is registered AND the
    curated ``user_tools`` subset is preserved."""

    async def run():
        manifest = Manifest.model_validate(
            {
                "api_tools": {"enabled": True},
                "routers_modules": _all_router_modules(),
                "user_tools": ["remove_tool", "list_hooks"],
            }
        )
        async with app.app_context(manifest):
            # api_tools projected the surface.
            live = await app.tools.get_tools()
            assert "remove_tool" in live
            assert "list_hooks" in live
            assert len(live) == 90

            # user_tools curation is preserved and surfaced to the flow builder
            # (the read-time view over the registered set).
            from tai_skeleton.operations.manifest import get_manifest

            got = await get_manifest()
            assert got["user_tools"] == ["list_hooks", "remove_tool"]  # sorted

    asyncio.run(run())
