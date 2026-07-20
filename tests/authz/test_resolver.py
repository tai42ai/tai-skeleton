"""The name→base-op resolver: branch, preset, preset-over-branch, and the
non-operation / cycle cases."""

from __future__ import annotations

from dataclasses import dataclass

from tai42_skeleton.authz.resolver import resolve_base_operation
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


class _FakePresetManager:
    def __init__(self, specs: dict[str, str]) -> None:
        self._specs = {n: _Body(base_tool=b) for n, b in specs.items()}

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
    op = resolve_base_operation("wipe", tool_registry=_FakeToolRegistry({}), preset_manager=None, registry=reg)
    assert op is not None
    assert op.name == "wipe"


def test_branch_resolves_to_base_op():
    reg = _reg_with_op("wipe")
    tr = _FakeToolRegistry({"wipe_cache": "wipe"})
    op = resolve_base_operation("wipe_cache", tool_registry=tr, preset_manager=None, registry=reg)
    assert op is not None
    assert op.name == "wipe"


def test_preset_over_op_resolves():
    reg = _reg_with_op("wipe")
    pm = _FakePresetManager({"my_preset": "wipe"})
    op = resolve_base_operation("my_preset", tool_registry=_FakeToolRegistry({}), preset_manager=pm, registry=reg)
    assert op is not None
    assert op.name == "wipe"


def test_preset_over_branch_resolves():
    reg = _reg_with_op("wipe")
    tr = _FakeToolRegistry({"wipe_cache": "wipe"})
    pm = _FakePresetManager({"p": "wipe_cache"})
    op = resolve_base_operation("p", tool_registry=tr, preset_manager=pm, registry=reg)
    assert op is not None
    assert op.name == "wipe"


def test_non_operation_tool_resolves_to_none():
    reg = _reg_with_op("wipe")
    tr = _FakeToolRegistry({"toolbox_thing": "toolbox_base"})
    op = resolve_base_operation("toolbox_thing", tool_registry=tr, preset_manager=None, registry=reg)
    assert op is None


def test_resolution_cycle_terminates():
    reg = _reg_with_op("wipe")
    # A preset whose base is itself (pathological) must not loop forever.
    pm = _FakePresetManager({"loop": "loop"})
    op = resolve_base_operation("loop", tool_registry=_FakeToolRegistry({}), preset_manager=pm, registry=reg)
    assert op is None
