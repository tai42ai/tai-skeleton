"""Configuration provider seam — factory + the built-in ``file`` provider.

This package implements :class:`~tai_contract.config.manager.ConfigManager` as a
pluggable-provider feature. It ships the selection seam (``ConfigMode`` +
``ConfigModeSettings`` + the ``ConfigManagerFactory`` mode-to-module map) and the
default :class:`FileConfigManager`. Other providers (k8s, future vault) ship as
separately-installed plugins exposing the same ``build_config_manager()``
convention; the factory loads the selected one by dynamic import.

Usage::

    from tai_skeleton.config import ConfigManagerFactory
    manager = ConfigManagerFactory.create()
"""

from tai_skeleton.config.config_mode import (
    ConfigMode,
    ConfigModeSettings,
    config_mode,
)
from tai_skeleton.config.factory import ConfigManagerFactory
from tai_skeleton.config.file_manager import FileConfigManager, build_config_manager

__all__ = [
    "ConfigManagerFactory",
    "ConfigMode",
    "ConfigModeSettings",
    "FileConfigManager",
    "build_config_manager",
    "config_mode",
]
