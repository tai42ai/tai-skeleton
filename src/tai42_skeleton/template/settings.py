"""Cache settings for :class:`ResourceManager`, co-located with the impl.

Composes the kit settings machinery (:class:`tai42_kit.settings.TaiBaseSettings`
plus the ``settings_cache`` reset registry).
"""

from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict
from tai42_kit.settings import TaiBaseSettings, settings_cache


class FileLoadingSettings(TaiBaseSettings):
    """Config for the document loaders behind :meth:`ResourceManager.load_file`."""

    model_config = SettingsConfigDict(env_prefix="FILE_LOADING_", frozen=True)

    # Hard cap (bytes) on the untrusted document bytes a loader decodes/parses
    # (HTML/EPUB via BeautifulSoup, PDF/XLSX/CSV/... via the path loaders). A
    # storage-id source bypasses ``fetch_url``'s own download cap, so this is the
    # bound before dispatch. Oversized -> loud raise, never a partial parse. Must
    # be positive.
    max_bytes: int = Field(default=25 * 1024 * 1024, gt=0)


class TemplateCacheSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TEMPLATE_CACHE_",
    )

    ttl: int | None = 60 * 5
    max_size: int | None = 256

    @field_validator("ttl", "max_size", mode="before")
    @classmethod
    def parse_empty_or_none(cls, v: Any) -> Any:
        if v == "":
            return None

        if isinstance(v, str) and v.lower() in ("none", "null", "undefined"):
            return None

        return v


@settings_cache
def template_cache_settings() -> TemplateCacheSettings:
    return TemplateCacheSettings()


@settings_cache
def file_loading_settings() -> FileLoadingSettings:
    return FileLoadingSettings()
