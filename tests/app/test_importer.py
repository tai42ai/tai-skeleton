"""Importer coverage: ``import_or_reload_package`` (empty/happy paths, the
raise-on-missing-package and raise-on-broken-submodule paths) and the
``_stable_cycle_fallback`` ordering.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from tai42_skeleton.app.importer import (
    _stable_cycle_fallback,
    import_or_reload_package,
)


def test_empty_name_returns_empty():
    assert import_or_reload_package("") == []


def test_happy_path_reimports_package_and_submodules():
    reloaded = import_or_reload_package("tests.app._fixtures.neutral")
    assert "tests.app._fixtures.neutral" in reloaded
    assert "tests.app._fixtures.neutral.leaf" in reloaded
    # The modules are live in ``sys.modules`` after the reload.
    assert sys.modules["tests.app._fixtures.neutral.leaf"].MARKER == "leaf"


def test_single_module_reload_has_no_submodules():
    # A plain module (no ``submodule_search_locations``) discovers only itself —
    # the false branch of the package-walk guard.
    reloaded = import_or_reload_package("tests.app._fixtures.neutral.leaf2")
    assert reloaded == ["tests.app._fixtures.neutral.leaf2"]
    assert sys.modules["tests.app._fixtures.neutral.leaf2"].VALUE == 2


def test_missing_top_level_package_raises():
    # A manifest-named package that cannot be found is corrupt configuration
    # and must abort loudly, naming the package.
    with pytest.raises(ImportError, match="Cannot find module totally_bogus_pkg_xyz"):
        import_or_reload_package("totally_bogus_pkg_xyz")


def test_failing_submodule_raises_with_module_name(monkeypatch):
    # Simulate one submodule failing to import (without a real import-time error
    # in a fixture, which would also fail collection): the importer raises,
    # naming the broken module and chaining the original error.
    real_import = importlib.import_module

    def flaky_import(name, *args, **kwargs):
        if name == "tests.app._fixtures.neutral.leaf2":
            raise ImportError("simulated import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", flaky_import)
    with pytest.raises(ImportError, match=r"Failed to import module tests\.app\._fixtures\.neutral\.leaf2") as ei:
        import_or_reload_package("tests.app._fixtures.neutral")
    assert isinstance(ei.value.__cause__, ImportError)


def test_side_effecting_init_runs_exactly_once():
    # The package __init__ has a side effect (appends to an out-of-tree log).
    # Discovery must enumerate submodules WITHOUT importing the package, so the
    # pop+reimport step is the ONLY import — the __init__ runs exactly once.
    # ``walk_packages`` would import the package to recurse, then reimport it,
    # running the __init__ twice (the double-register bug).
    from tests.app._fixtures import counter_probe

    counter_probe.INIT_CALLS.clear()
    reloaded = import_or_reload_package("tests.app._fixtures.side_effect_pkg")

    assert counter_probe.INIT_CALLS == ["side_effect_pkg"]
    # The submodule was discovered (without importing the package to find it).
    assert "tests.app._fixtures.side_effect_pkg.child" in reloaded


def test_stable_cycle_fallback_orders_by_depth_then_name():
    nodes = {"a.b.c", "a", "a.b", "z"}
    assert _stable_cycle_fallback(nodes) == ["a", "z", "a.b", "a.b.c"]
