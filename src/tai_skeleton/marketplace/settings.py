"""Pydantic-settings for the marketplace client.

Two co-located groups: :class:`MarketplaceSettings` (``MARKETPLACE_*``) — the
registry endpoint plus the advisory-poll knobs — and
:class:`MarketplaceStorePgSettings` (``MARKETPLACE_STORE_*``) — the Postgres
connection for the ``marketplace_installs`` attribution table, kept in its own
namespace like the versioning / connector stores so each durable store declares
its own DSN. It targets the same ``tai`` database by default.

The advisory poll is the ONLY background outbound call this feature makes, and
it is a visible, documented setting: ``MARKETPLACE_ADVISORIES_POLL`` defaults to
on, the startup log names the polled URL, and one env var turns it off. Nothing
else is ever sent to the registry without an explicit operator request.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai_kit.clients import PostgresConnectionSettings
from tai_kit.settings import TaiBaseSettings, settings_cache


class MarketplaceSettings(TaiBaseSettings):
    """``MARKETPLACE_*`` — the registry endpoint and the advisory-poll knobs."""

    model_config = SettingsConfigDict(env_prefix="MARKETPLACE_")

    # Base URL of the marketplace registry's public API.
    url: str = "https://marketplace.tai42.ai"

    # Periodically re-fetch advisories for the installed plugins. A background
    # outbound call, so it is explicit and documented: default on, loud startup
    # log naming the URL, one env var to disable. Install-time advisory checks
    # are separate and unconditional.
    advisories_poll: bool = True

    # Seconds between advisory polls (and the freshness bound the advisories
    # route serves within). Must be positive.
    advisories_interval_s: int = Field(default=3600, gt=0)

    # Per-request timeout (seconds) for registry calls. Must be positive.
    request_timeout_s: float = Field(default=15, gt=0)


@settings_cache
def marketplace_settings() -> MarketplaceSettings:
    return MarketplaceSettings()


class MarketplaceStorePgSettings(PostgresConnectionSettings):
    """``MARKETPLACE_STORE_*`` Postgres connection for ``marketplace_installs``.
    No baked-in credential — supply the password via
    ``MARKETPLACE_STORE_PG_PASSWORD``."""

    model_config = SettingsConfigDict(env_prefix="MARKETPLACE_STORE_")

    pg_db: str = "tai"


@settings_cache
def marketplace_store_settings() -> MarketplaceStorePgSettings:
    return MarketplaceStorePgSettings()
