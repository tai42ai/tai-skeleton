"""Tool projection: api_tools filtering, the two exclusion tiers, the
destructive gate/hint, the both-edges reload gate, and the exposure-tier
fail-safe."""

from __future__ import annotations

import asyncio

import pytest
from fastmcp.exceptions import ToolError
from tai_contract.manifest import ApiToolsConfig

from tai_skeleton.app.reload_gate import reload_gate
from tai_skeleton.operations import OperationRegistry, operation
from tai_skeleton.operations.errors import ConflictError
from tai_skeleton.operations.projection import is_tier1, is_tier2, project_operations


class _RecordingTools:
    def __init__(self) -> None:
        self.registered: dict[str, dict] = {}

    def tool(self, *, force, name, tags, annotations):
        def decorator(func):
            self.registered[name] = {
                "force": force,
                "tags": tags,
                "annotations": annotations,
                "func": func,
            }
            return func

        return decorator


class _FakeApp:
    def __init__(self) -> None:
        self.tools = _RecordingTools()


def _reg_with(*ops) -> OperationRegistry:
    reg = OperationRegistry()
    for name, kwargs in ops:

        @operation(name=name, summary="s", tags=["t"], registry=reg, **kwargs)
        async def _op(_name=name, **_):
            return {"op": _name}

    return reg


def _project(reg, config):
    app = _FakeApp()
    names = project_operations(app, config, registry=reg)
    return app, names


def test_disabled_projects_nothing():
    reg = _reg_with(("alpha", {}))
    app, names = _project(reg, ApiToolsConfig(enabled=False))
    assert names == []
    assert app.tools.registered == {}


def test_default_in_projects_normal_ops():
    reg = _reg_with(("alpha", {}), ("beta", {}))
    app, names = _project(reg, ApiToolsConfig())
    assert set(names) == {"alpha", "beta"}
    assert app.tools.registered["alpha"]["force"] is True


def test_exclude_suppresses_op():
    reg = _reg_with(("alpha", {}), ("beta", {}))
    _, names = _project(reg, ApiToolsConfig(exclude=["beta"]))
    assert names == ["alpha"]


def test_include_of_unknown_op_raises_loudly():
    reg = _reg_with(("alpha", {}))
    with pytest.raises(ValueError, match="not registered"):
        _project(reg, ApiToolsConfig(include=["ghost"]))


def test_tier1_meta_executor_never_projected_even_when_included():
    reg = _reg_with(("run_tool", {}), ("alpha", {"meta_executor": True}))
    _, names = _project(reg, ApiToolsConfig(include=["run_tool", "alpha"]))
    assert names == []
    assert is_tier1(reg.get("run_tool"))
    assert is_tier1(reg.get("alpha"))


def test_tier2_default_excluded_but_includable():
    reg = _reg_with(("update_manifest", {"authority_changing": True}))
    _, names = _project(reg, ApiToolsConfig())
    assert names == []
    _, names = _project(reg, ApiToolsConfig(include=["update_manifest"]))
    assert names == ["update_manifest"]


def test_tier2_by_auth_route_prefix():
    reg = OperationRegistry()

    @operation(name="mint_key", summary="s", tags=["t"], registry=reg)
    async def mint_key():
        return {}

    reg.get("mint_key").route_template = "/api/auth/keys"
    assert is_tier2(reg.get("mint_key"))
    _, names = _project(reg, ApiToolsConfig())
    assert names == []


def test_destructive_hint_annotation_and_expose_gate():
    reg = _reg_with(("wipe", {"destructive": True}))
    app, names = _project(reg, ApiToolsConfig())
    assert names == ["wipe"]
    assert app.tools.registered["wipe"]["annotations"].destructiveHint is True

    _, names = _project(reg, ApiToolsConfig(expose_destructive=False))
    assert names == []


def test_destructive_included_explicitly_overrides_expose_gate():
    reg = _reg_with(("wipe", {"destructive": True}))
    _, names = _project(reg, ApiToolsConfig(expose_destructive=False, include=["wipe"]))
    assert names == ["wipe"]


def test_projected_tool_reload_gate_raises_toolerror():
    reg = _reg_with(("reloady", {"reload_gated": True}))
    app, _ = _project(reg, ApiToolsConfig())
    wrapper = app.tools.registered["reloady"]["func"]

    async def run():
        reload_gate.bind_to_running_loop()
        async with reload_gate.lock:
            with pytest.raises(ToolError):
                await wrapper()

    asyncio.run(run())


def test_projected_tool_maps_operation_error_to_toolerror():
    reg = OperationRegistry()

    @operation(name="boom", summary="s", tags=["t"], errors=[ConflictError], registry=reg)
    async def boom():
        raise ConflictError("nope")

    app, _ = _project(reg, ApiToolsConfig())
    wrapper = app.tools.registered["boom"]["func"]
    with pytest.raises(ToolError, match="nope"):
        asyncio.run(wrapper())


def test_boot_projection_is_one_to_one_over_every_eligible_op():
    """Spec↔tools parity precursor (the live-stack version runs at D.1): over the
    FULL registry a real boot populates, every projection-eligible operation maps to
    exactly ONE projected tool — no op yields two tools, no eligible op is missing —
    and tier-1 (``run_tool`` + any meta-executor) never projects even when every op
    is explicitly included."""
    from tai_skeleton.app.instance import app
    from tai_skeleton.manifest import Manifest
    from tai_skeleton.operations.registry import operation_registry

    async def run():
        async with app.app_context(Manifest.model_validate({"api_tools": {"enabled": True}})):
            reg = operation_registry
            ops = reg.all()
            # Maximal config — include EVERY op + expose destructive — so the eligible
            # set is exactly "all ops that are not tier-1" (tier-2 becomes includable,
            # destructive is exposed).
            fake = _FakeApp()
            projected = project_operations(
                fake, ApiToolsConfig(include=sorted(reg.names()), expose_destructive=True), registry=reg
            )
            eligible = sorted(op.name for op in ops if not is_tier1(op))

            assert sorted(projected) == eligible  # no eligible op missing
            assert len(projected) == len(set(projected))  # no op yields two tools
            assert len(fake.tools.registered) == len(projected)  # one registration per projected op

            tier1 = {op.name for op in ops if is_tier1(op)}
            assert "run_tool" in tier1  # the named meta-executor
            assert "submit_run" in tier1  # the flagged meta-executor
            assert not (tier1 & set(projected))  # tier-1 blocked even when included

    asyncio.run(run())


def test_exposure_tier_fail_safe_predicates_match_projection():
    """Every op the tier predicates flag is off the default surface; a normal op
    is projected. A predicate-matching op that were absent from its tier would be
    projected default-in — this asserts the projection honors the predicates."""
    reg = _reg_with(
        ("normal", {}),
        ("meta", {"meta_executor": True}),
        ("authority", {"authority_changing": True}),
    )
    _, names = _project(reg, ApiToolsConfig())
    for op in reg.all():
        if is_tier1(op) or is_tier2(op):
            assert op.name not in names, f"{op.name} matched a tier predicate but was projected default-in"
        else:
            assert op.name in names
