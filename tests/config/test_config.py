"""Tests for the config provider seam.

Covers the factory's mode-to-module map and dynamic-import dispatch, the
string-literal (no static import) k8s entry, and the file provider's contract
conformance.
"""

import importlib
from collections.abc import Iterator

import pytest
from tai_contract.config.manager import ConfigManager

from tai_skeleton.config import (
    ConfigManagerFactory,
    FileConfigManager,
    build_config_manager,
)
from tai_skeleton.config import factory as factory_mod
from tai_skeleton.config.config_mode import ConfigMode, config_mode


@pytest.fixture(autouse=True)
def _file_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Default every test to the file config mode with a clean cached accessor."""
    monkeypatch.setenv("TAI_CONFIG_MODE", "file")
    config_mode.cache_clear()
    yield
    config_mode.cache_clear()


def test_file_mode_resolves_file_manager_via_build_factory() -> None:
    """``file`` mode dynamically imports the built-in provider and calls its
    ``build_config_manager()`` factory."""
    manager = ConfigManagerFactory.create()
    assert isinstance(manager, FileConfigManager)


def test_file_provider_module_exposes_build_config_manager() -> None:
    """The built-in file provider follows the ``build_config_manager()`` convention."""
    manager = build_config_manager()
    assert isinstance(manager, FileConfigManager)


def test_mode_module_map_holds_k8s_as_string_literal() -> None:
    """The k8s entry is a string module name, never a statically imported module."""
    k8s_entry = factory_mod._PROVIDER_MODULES["k8s"]
    assert isinstance(k8s_entry, str)
    assert k8s_entry.startswith("tai_config_k8s")


def test_factory_does_not_statically_import_k8s_plugin() -> None:
    """Importing the factory must not pull in the (not-installed) k8s plugin."""
    import sys

    assert "tai_config_k8s" not in sys.modules


def test_unknown_mode_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unmapped mode raises ValueError rather than silently defaulting."""
    monkeypatch.setitem(factory_mod._PROVIDER_MODULES, "file", "tai_skeleton.config.file_manager")
    monkeypatch.setattr(factory_mod, "config_mode", lambda: "vault")
    with pytest.raises(ValueError, match="Unknown config mode 'vault'"):
        ConfigManagerFactory.create()


def test_k8s_mode_raises_import_error_when_plugin_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting ``k8s`` without the ``tai-config-k8s`` plugin installed raises
    ImportError loudly rather than degrading to a default provider."""
    monkeypatch.setattr(factory_mod, "config_mode", lambda: "k8s")
    with pytest.raises(ImportError):
        ConfigManagerFactory.create()


def test_factory_dispatches_via_dynamic_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory imports the mapped module and calls its ``build_config_manager``."""
    sentinel = object()
    fake = importlib.import_module("tai_skeleton.config.file_manager")
    monkeypatch.setattr(fake, "build_config_manager", lambda: sentinel)
    assert ConfigManagerFactory.create() is sentinel


def test_file_manager_satisfies_contract() -> None:
    """``FileConfigManager`` is a concrete :class:`ConfigManager` (all abstracts implemented)."""
    assert issubclass(FileConfigManager, ConfigManager)
    manager = FileConfigManager()
    assert isinstance(manager, ConfigManager)


def test_config_mode_default_is_file() -> None:
    """With ``TAI_CONFIG_MODE=file`` the accessor returns the ``file`` string value."""
    assert config_mode() == ConfigMode.file.value == "file"
