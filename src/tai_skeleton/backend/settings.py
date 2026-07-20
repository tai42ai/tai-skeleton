from pydantic_settings import SettingsConfigDict
from tai_kit.settings import TaiBaseSettings, settings_cache


class BackendSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BACKEND_",
    )
    manifest_key: str = "MANIFEST_KEY"
    task_timeout: int = 300
    tool_name_arg: str = "backend_tool_name"


@settings_cache
def base_backend_settings() -> BackendSettings:
    return BackendSettings()
