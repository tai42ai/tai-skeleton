"""Provider registry — the engine's in-memory map of supported third-parties.

The catalog endpoint reads the registry, so adding a provider needs no UI
changes. The skeleton ships no concrete provider: registration is
manifest-driven. A provider plugin module (named in the manifest) calls
``tai_app.connectors.register_connector(descriptor)`` on import, which forwards
to :func:`register_connector` here. Descriptors are validated when built so a
misconfigured provider fails deployment loudly rather than at first user click.

The descriptor models live in :mod:`tai_contract.connectors.providers`; this
module owns only the registry STATE (the code-built ``_REGISTRY`` and the
DB-sourced ``_CATALOG_CACHE``) and the registration / lookup functions.
"""

from __future__ import annotations

import logging

from tai_contract.connectors.errors import (  # noqa: F401  (re-exported)
    OperatorMisconfiguredError,
)
from tai_contract.connectors.providers import (  # noqa: F401  (re-exported)
    ConfigFieldSpec,
    McpServerDescriptor,
    OAuthEndpoints,
    ProviderDescriptor,
    SubServiceDescriptor,
)
from tai_kit.settings import register_settings_reset

logger = logging.getLogger(__name__)

# Code-side mirror of the connector_category seed rows in the init SQL.
# register_connector validates registry descriptors against it because
# registration runs at import time, before the DB is reachable; DB-sourced
# catalog rows are validated against the full category table at fetch time
# instead (store.catalog_store.fetch_catalog).
SEED_CATEGORY_IDS = (
    "communication",
    "productivity",
    "dev-tools",
    "data",
    "ai-ml",
    "other",
)


# -- Registry ----------------------------------------------------------------
# Provider plugin modules call register_connector at import with a descriptor.
# The single registry maps provider id -> descriptor.

_REGISTRY: dict[str, ProviderDescriptor] = {}

# No-auth providers loaded from the connector_catalog table by
# connectors.store.catalog_store.refresh_catalog(). Held in a sync-readable
# in-memory cache so get_provider/list_providers never do a per-call DB read.
# Populated by set_catalog() (called from the async refresh on startup).
_CATALOG_CACHE: dict[str, ProviderDescriptor] = {}


def register_connector(descriptor: ProviderDescriptor) -> None:
    if descriptor.id in _REGISTRY:
        raise ValueError(f"Provider {descriptor.id!r} already registered")
    # Registration runs at import time, before the DB is reachable, so the
    # category check goes against the code-side seed constants. DB-sourced rows
    # are checked against the full category table at fetch_catalog instead.
    if descriptor.category not in SEED_CATEGORY_IDS:
        raise ValueError(
            f"provider {descriptor.id!r} category {descriptor.category!r} is not "
            f"a seed category (expected one of {', '.join(SEED_CATEGORY_IDS)})"
        )
    _REGISTRY[descriptor.id] = descriptor
    logger.info("connectors: registered provider %s", descriptor.id)


def reset_registry() -> None:
    """Clear the code-built provider registry.

    Called by ``start()`` before it re-imports the manifest's connector plugin
    modules, which re-run their module-level ``register_connector(...)`` calls.
    Without this the duplicate guard in :func:`register_connector` would raise on
    every reload. Mirrors the ``_agents`` reset in the lifecycle mixin's
    ``start()``; the within-one-load duplicate guard is preserved.
    """
    _REGISTRY.clear()


def set_catalog(descriptors: list[ProviderDescriptor]) -> None:
    """Replace the in-memory no-auth catalog cache atomically.

    Raises on a catalog id that collides with a code-built provider — a catalog
    row must never shadow a registered provider (operator error, loud).
    """
    new_cache: dict[str, ProviderDescriptor] = {}
    for descriptor in descriptors:
        if descriptor.id in _REGISTRY:
            raise ValueError(f"catalog provider {descriptor.id!r} collides with a code-built provider")
        if descriptor.id in new_cache:
            raise ValueError(f"duplicate catalog provider id {descriptor.id!r}")
        new_cache[descriptor.id] = descriptor
    _CATALOG_CACHE.clear()
    _CATALOG_CACHE.update(new_cache)
    logger.info("connectors: catalog cache set with %d provider(s)", len(new_cache))


def get_provider(provider_id: str) -> ProviderDescriptor:
    descriptor = _REGISTRY.get(provider_id) or _CATALOG_CACHE.get(provider_id)
    if descriptor is None:
        raise KeyError(f"Unknown Connectors provider: {provider_id!r}")
    return descriptor


def list_providers() -> list[ProviderDescriptor]:
    return list(_REGISTRY.values()) + list(_CATALOG_CACHE.values())


@register_settings_reset
def _clear_caches() -> None:
    # Registered descriptors are owned by the plugin modules that registered
    # them, not by settings, so a settings reset only drops the DB-sourced
    # catalog cache (it repopulates on the next refresh). The code-built
    # ``_REGISTRY`` is left intact.
    _CATALOG_CACHE.clear()
