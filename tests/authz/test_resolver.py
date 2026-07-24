"""The dispatch resolver: branch, preset, preset-over-branch, the arguments a preset bakes
into the call, the non-operation / cycle cases, and the refusal that keeps a rebuilding
operation surface from reading as "capability"."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from tai42_skeleton.app.reload_gate import reload_gate
from tai42_skeleton.authz.resolver import OperationSurfaceUnsettledError, resolve_dispatch
from tai42_skeleton.operations import OperationRegistry, operation


class _FakeToolRegistry:
    """Models ``_extend_tools`` via ``base_of``: a branch maps to its base."""

    def __init__(self, branches: dict[str, str]) -> None:
        self._branches = branches

    def base_of(self, name: str) -> str:
        return self._branches.get(name, name)


@dataclass
class _Body:
    base_tool: str
    fixed_kwargs: dict[str, Any] = field(default_factory=dict)


class _FakePresetManager:
    def __init__(self, specs: dict[str, _Body]) -> None:
        self._specs = specs

    def is_registered(self, name: str) -> bool:
        return name in self._specs

    def get_spec(self, name: str):
        return self._specs[name]


def _reg_with_op(name: str = "wipe") -> OperationRegistry:
    reg = OperationRegistry()

    @operation(name=name, summary="s", tags=["t"], registry=reg)
    async def _op(**_):
        return {}

    return reg


def test_base_name_resolves_directly():
    reg = _reg_with_op("wipe")
    resolved = resolve_dispatch(
        "wipe", {"a": 1}, tool_registry=_FakeToolRegistry({}), preset_manager=None, registry=reg
    )
    assert resolved is not None
    assert resolved.operation.name == "wipe"
    assert resolved.call_arguments == {"a": 1}


def test_branch_resolves_to_base_op():
    reg = _reg_with_op("wipe")
    tr = _FakeToolRegistry({"wipe_cache": "wipe"})
    resolved = resolve_dispatch("wipe_cache", {}, tool_registry=tr, preset_manager=None, registry=reg)
    assert resolved is not None
    assert resolved.operation.name == "wipe"


def test_preset_over_op_resolves():
    reg = _reg_with_op("wipe")
    pm = _FakePresetManager({"my_preset": _Body("wipe")})
    resolved = resolve_dispatch("my_preset", {}, tool_registry=_FakeToolRegistry({}), preset_manager=pm, registry=reg)
    assert resolved is not None
    assert resolved.operation.name == "wipe"


def test_preset_over_branch_resolves():
    reg = _reg_with_op("wipe")
    tr = _FakeToolRegistry({"wipe_cache": "wipe"})
    pm = _FakePresetManager({"p": _Body("wipe_cache")})
    resolved = resolve_dispatch("p", {}, tool_registry=tr, preset_manager=pm, registry=reg)
    assert resolved is not None
    assert resolved.operation.name == "wipe"


def test_a_presets_baked_kwargs_reach_the_call_arguments():
    # A baked value is a constant the caller cannot override, so the decision must see it.
    reg = _reg_with_op("wipe")
    pm = _FakePresetManager({"p": _Body("wipe", {"target": "prod"})})
    resolved = resolve_dispatch(
        "p", {"mark": "m"}, tool_registry=_FakeToolRegistry({}), preset_manager=pm, registry=reg
    )
    assert resolved is not None
    assert resolved.call_arguments == {"mark": "m", "target": "prod"}


def test_layered_presets_bake_along_the_whole_chain():
    reg = _reg_with_op("wipe")
    pm = _FakePresetManager({"outer": _Body("inner", {"mark": "m"}), "inner": _Body("wipe", {"target": "prod"})})
    resolved = resolve_dispatch("outer", {}, tool_registry=_FakeToolRegistry({}), preset_manager=pm, registry=reg)
    assert resolved is not None
    assert resolved.call_arguments == {"mark": "m", "target": "prod"}


def test_the_call_arguments_handed_in_are_never_mutated():
    reg = _reg_with_op("wipe")
    pm = _FakePresetManager({"p": _Body("wipe", {"target": "prod"})})
    arguments: dict[str, Any] = {}
    resolve_dispatch("p", arguments, tool_registry=_FakeToolRegistry({}), preset_manager=pm, registry=reg)
    assert arguments == {}


def test_non_operation_tool_resolves_to_none():
    reg = _reg_with_op("wipe")
    tr = _FakeToolRegistry({"toolbox_thing": "toolbox_base"})
    resolved = resolve_dispatch("toolbox_thing", {}, tool_registry=tr, preset_manager=None, registry=reg)
    assert resolved is None


def test_resolution_cycle_terminates():
    reg = _reg_with_op("wipe")
    # A preset whose base is itself (pathological) must not loop forever.
    pm = _FakePresetManager({"loop": _Body("loop")})
    resolved = resolve_dispatch("loop", {}, tool_registry=_FakeToolRegistry({}), preset_manager=pm, registry=reg)
    assert resolved is None


# -- the surface has to be settled for "not an operation" to mean anything ------


def test_an_empty_operation_registry_refuses_instead_of_reading_as_a_capability():
    # A started app always holds leaf operations, so an empty registry is only ever the
    # cleared half of a rebuild; answering from it would classify a fenced operation as a
    # capability tool and dispatch it undecided.
    with pytest.raises(OperationSurfaceUnsettledError, match="retry shortly"):
        resolve_dispatch("wipe", {}, tool_registry=None, preset_manager=None, registry=OperationRegistry())


def test_an_in_flight_rebuild_refuses_even_a_populated_registry():
    # The replay populates the registry as it goes, so only the rebuild mark says it is done.
    reg = _reg_with_op("wipe")

    with reg.rebuilding(), pytest.raises(OperationSurfaceUnsettledError, match="retry shortly"):
        resolve_dispatch("toolbox_thing", {}, tool_registry=None, preset_manager=None, registry=reg)

    # Finished: the settled surface classifies the same name as a capability again.
    assert resolve_dispatch("toolbox_thing", {}, tool_registry=None, preset_manager=None, registry=reg) is None


def test_a_record_held_mid_rebuild_refuses_rather_than_resolving_without_its_route():
    # The replay restores records BEFORE the routers re-attach their route templates, so a
    # record held mid-rebuild has no template to synthesize a resource path from.
    reg = _reg_with_op("wipe")
    assert reg.get("wipe").route_template is None

    with reg.rebuilding(), pytest.raises(OperationSurfaceUnsettledError, match="retry shortly"):
        resolve_dispatch("wipe", {}, tool_registry=None, preset_manager=None, registry=reg)

    # Settled again: the same name resolves to its record and the edge decision runs.
    resolved = resolve_dispatch("wipe", {}, tool_registry=None, preset_manager=None, registry=reg)
    assert resolved is not None
    assert resolved.operation.name == "wipe"


def test_a_rebuild_that_raises_still_releases_the_unsettled_mark():
    # A reload that fails mid-rebuild must not refuse every dispatch for the process lifetime.
    reg = _reg_with_op("wipe")

    with pytest.raises(RuntimeError), reg.rebuilding():
        raise RuntimeError("reload body failed")

    assert resolve_dispatch("toolbox_thing", {}, tool_registry=None, preset_manager=None, registry=reg) is None


def test_a_reload_gate_held_without_touching_the_operation_surface_does_not_refuse():
    # The reload gate has holders that never rebuild the operation registry (the periodic
    # failed-MCP re-probe), and a fire must not be dropped just because one is running.
    reg = _reg_with_op("wipe")

    async def run() -> None:
        async with reload_gate.lock:
            assert resolve_dispatch("toolbox_thing", {}, tool_registry=None, preset_manager=None, registry=reg) is None
            resolved = resolve_dispatch("wipe", {}, tool_registry=None, preset_manager=None, registry=reg)
            assert resolved is not None

    asyncio.run(run())
