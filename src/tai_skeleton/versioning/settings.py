"""Pydantic-settings for the versioned-document store's Postgres connection.

The store is a platform primitive with its own ``VERSIONING_STORE_*`` namespace,
kept separate from the connector / access-control stores so each durable store
declares its own DSN. It targets the same ``tai`` database by default; a
deployment points every store at its Postgres via the shared env.
"""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict
from tai_kit.clients import PostgresConnectionSettings
from tai_kit.settings import settings_cache


class VersioningStorePgSettings(PostgresConnectionSettings):
    """``VERSIONING_STORE_*`` Postgres connection for ``versioned_documents`` +
    ``versioned_document_versions``. No baked-in credential — supply the password
    via ``VERSIONING_STORE_PG_PASSWORD``."""

    model_config = SettingsConfigDict(env_prefix="VERSIONING_STORE_")

    pg_db: str = "tai"


@settings_cache
def versioning_store_settings() -> VersioningStorePgSettings:
    return VersioningStorePgSettings()
