"""tai_skeleton ships its own ``tools/`` and ``hooks/`` feature packages but
carries no ``tools.management`` submodule — the management tools are supplied by
a downstream plugin, not the skeleton. The feature packages must import; the
``tai_skeleton.tools.management`` module string must fail loudly — no compat
alias keeps it importable."""

import importlib

import pytest


def test_old_management_module_string_fails_loudly() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tai_skeleton.tools.management")


def test_tools_package_present() -> None:
    assert hasattr(importlib.import_module("tai_skeleton.tools"), "ToolRegistry")


def test_hooks_package_present() -> None:
    assert hasattr(importlib.import_module("tai_skeleton.hooks"), "get_hooks_manager")
