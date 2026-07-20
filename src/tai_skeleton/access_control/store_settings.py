"""Pydantic-settings for the access-control policy store's Postgres connection.

The policy store is a platform primitive with its own ``ACCESS_CONTROL_STORE_*``
namespace, kept separate from the connector / versioning stores so each durable
store declares its own DSN. It targets the same ``tai`` database by default; a
deployment points every store at its Postgres via the shared env.

This namespace is PURELY the PG connection config — it is NOT a store selector.
Postgres is the only policy store; there is nothing to select.
"""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict
from tai_kit.clients import PostgresConnectionSettings
from tai_kit.settings import settings_cache


class AccessControlStorePgSettings(PostgresConnectionSettings):
    """``ACCESS_CONTROL_STORE_*`` Postgres connection for
    ``access_control_policies`` + ``access_control_routes``. No baked-in
    credential — supply the password via ``ACCESS_CONTROL_STORE_PG_PASSWORD``."""

    model_config = SettingsConfigDict(env_prefix="ACCESS_CONTROL_STORE_")

    pg_db: str = "tai"


@settings_cache
def access_control_store_settings() -> AccessControlStorePgSettings:
    return AccessControlStorePgSettings()
