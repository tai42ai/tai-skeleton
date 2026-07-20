"""Configuration mode — the provider-selection seam.

Defines the ``ConfigMode`` enum and the ``ConfigModeSettings`` boot setting that
names which :class:`~tai_contract.config.manager.ConfigManager` provider backs
the deployment. The factory (:mod:`tai_skeleton.config.factory`) reads
``config_mode()`` and maps it to a provider module.

Environment variables
---------------------
- ``TAI_CONFIG_MODE`` -- ``file`` (default) or ``k8s``

Provider-specific settings (e.g. the k8s Secret / ConfigMap names) live with
their provider, not here — this module names only the seam.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import field_validator
from pydantic_settings import SettingsConfigDict
from tai_kit.settings import TaiBaseSettings, settings_cache


class ConfigMode(StrEnum):
    """Configuration source strategy."""

    file = "file"
    k8s = "k8s"


class ConfigModeSettings(TaiBaseSettings):
    """Reads ``TAI_CONFIG_MODE`` and validates it against :class:`ConfigMode`."""

    model_config = SettingsConfigDict(
        env_prefix="TAI_",
    )

    config_mode: ConfigMode = ConfigMode.file

    @field_validator("config_mode", mode="before")
    @classmethod
    def validate_config_mode(cls, v: object) -> str:
        valid = [m.value for m in ConfigMode]
        if not isinstance(v, str):
            raise ValueError(f"Invalid TAI_CONFIG_MODE='{v}'. Must be one of: {', '.join(valid)}")
        text = v.strip().lower()
        if text not in valid:
            raise ValueError(f"Invalid TAI_CONFIG_MODE='{v}'. Must be one of: {', '.join(valid)}")
        return text


@settings_cache
def config_mode() -> str:
    """Return the active configuration mode as a plain string (``'file'`` or ``'k8s'``)."""
    return ConfigModeSettings().config_mode.value
