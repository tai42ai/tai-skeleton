"""Loader for the no-auth MCP market catalog.

The ``connector_catalog`` table holds one row per offered no-auth provider — a
no-auth :class:`ProviderDescriptor` serialized as JSON, plus the authoritative
``origin``/``category`` columns. It is a public template (no secrets).
:func:`refresh_catalog` fetches every enabled row, parses each into a
``ProviderDescriptor`` (loud on a malformed row — never silently skipped), and
publishes the set into the in-memory ``_CATALOG_CACHE`` via
:func:`providers.set_catalog`, so the SYNC ``get_provider`` / ``list_providers``
serve no-auth providers without a per-call DB read.

This module only reads the table; ops add a system MCP by inserting a row
(ops/SQL, visible on the next refresh) and the verified community add path
writes through ``store.catalog_write``. DDL is applied only by the API, the same
deploy-ordering assumption ``connector_connections`` already relies on.

Postgres is reached through the app-pooled ``PostgresClient`` so it shares one
pool with the token / sources stores (same DSN), closed centrally at shutdown.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel
from tai42_contract.connectors.providers import ProviderDescriptor
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.postgres import PostgresClient

from tai42_skeleton.connectors.providers.registry import set_catalog
from tai42_skeleton.connectors.settings import connector_store_settings

logger = logging.getLogger(__name__)


class ConnectorCategory(BaseModel):
    """One ``connector_category`` row — the UI grouping for providers."""

    id: str
    display_name: str
    sort_order: int


async def fetch_categories() -> list[ConnectorCategory]:
    """Read every ``connector_category`` row, ordered for display
    (``sort_order``, then id)."""
    async with (
        client_ctx(PostgresClient, connector_store_settings().pg) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        await cur.execute("SELECT id, display_name, sort_order FROM connector_category ORDER BY sort_order, id")
        rows = await cur.fetchall()
    return [
        ConnectorCategory(id=category_id, display_name=display_name, sort_order=sort_order)
        for category_id, display_name, sort_order in rows
    ]


async def fetch_catalog() -> list[ProviderDescriptor]:
    """Read every enabled catalog row and parse it into a ProviderDescriptor.

    The ``origin`` and ``category`` columns are the single source of truth and
    are injected into the descriptor; the stored jsonb must not embed them.
    Raises on a malformed row (bad descriptor, embedded origin/category,
    unknown category, community row without ``added_by``) — an operator error
    must surface loudly, not be skipped.
    """
    valid_category_ids = {category.id for category in await fetch_categories()}
    async with (
        client_ctx(PostgresClient, connector_store_settings().pg) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        await cur.execute(
            "SELECT provider_id, descriptor, origin, category, added_by "
            "FROM connector_catalog WHERE enabled ORDER BY provider_id"
        )
        rows = await cur.fetchall()

    descriptors: list[ProviderDescriptor] = []
    for provider_id, descriptor_json, origin, category, added_by in rows:
        if "origin" in descriptor_json or "category" in descriptor_json:
            raise ValueError(
                f"connector_catalog row {provider_id!r} descriptor must not embed "
                f"origin/category (the table columns are authoritative)"
            )
        if category not in valid_category_ids:
            raise ValueError(f"connector_catalog row {provider_id!r} has unknown category {category!r}")
        if origin == "community" and added_by is None:
            raise ValueError(f"connector_catalog row {provider_id!r} is community-origin but has no added_by")
        try:
            descriptor = ProviderDescriptor.model_validate({**descriptor_json, "origin": origin, "category": category})
        except Exception as exc:
            raise ValueError(f"connector_catalog row {provider_id!r} has an invalid descriptor: {exc}") from exc
        if descriptor.kind != "none":
            raise ValueError(f"connector_catalog row {provider_id!r} must have kind='none' (got {descriptor.kind!r})")
        if descriptor.id != provider_id:
            raise ValueError(
                f"connector_catalog row provider_id {provider_id!r} does not match descriptor id {descriptor.id!r}"
            )
        descriptors.append(descriptor)
    return descriptors


async def refresh_catalog() -> None:
    """Fetch the catalog and publish it into the in-memory provider cache.

    Wired to run at process startup (api + mcp). A collision with a code-built
    provider raises inside :func:`set_catalog`.
    """
    descriptors = await fetch_catalog()
    set_catalog(descriptors)
    logger.info("connectors: catalog refreshed (%d provider(s))", len(descriptors))
