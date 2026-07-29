"""ConfigManagerFactory — selects the active config provider at startup.

Config is read *before* the manifest, so a provider cannot be loaded through the
manifest/handle mechanism every other feature uses. Instead the factory holds a
small ``mode -> provider-module`` map, **dynamically imports** the selected
module, and calls its ``build_config_manager()`` factory (a convention every
provider module exposes). The k8s entry is a **string literal**, not an import,
so the skeleton carries no static dependency on the k8s plugin — installing the
plugin makes its module importable at runtime; selecting ``k8s`` without it
raises loudly.
"""

from __future__ import annotations

import importlib

from tai42_contract.config.manager import ConfigManager

from tai42_skeleton.config.config_mode import config_mode

# mode -> provider module exposing ``build_config_manager() -> ConfigManager``.
# ``file`` is the built-in default shipped in the skeleton; ``k8s`` is the
# separately-installed ``tai42-config-k8s`` plugin, named here only as a string so
# there is no static skeleton -> plugin import.
_PROVIDER_MODULES: dict[str, str] = {
    "file": "tai42_skeleton.config.file_manager",
    "k8s": "tai42_config_k8s.manager",
}


class ConfigManagerFactory:
    """Resolves the active :class:`ConfigManager` for the current config mode."""

    @staticmethod
    def create() -> ConfigManager:
        """Build the :class:`ConfigManager` for the current config mode.

        Raises:
            ValueError: If the config mode has no registered provider module.
            ImportError: If the selected provider module (e.g. the k8s plugin)
                is not installed.
        """
        mode = config_mode()
        try:
            module_name = _PROVIDER_MODULES[mode]
        except KeyError:
            raise ValueError(f"Unknown config mode '{mode}'. Supported modes: {', '.join(_PROVIDER_MODULES)}") from None
        module = importlib.import_module(module_name)
        return module.build_config_manager()
