from tai42_kit.settings import settings_cache

from tai42_skeleton.settings.settings import (
    AppArgsSettings,
    CoreSettings,
)


@settings_cache
def manifest_path() -> str | None:
    return CoreSettings().manifest_path


@settings_cache
def backend_provider() -> str:
    return (CoreSettings().backend or "").strip().lower()


@settings_cache
def template_provider() -> str:
    return (CoreSettings().template or "").strip().lower()


@settings_cache
def mcp_probe_timeout() -> float:
    return CoreSettings().mcp_probe_timeout


@settings_cache
def app_args_settings() -> AppArgsSettings:
    return AppArgsSettings()
