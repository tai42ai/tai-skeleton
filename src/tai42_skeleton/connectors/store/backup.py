"""Store-level backup export/import for the connector Postgres tables.

Postgres is the durable source of truth for every connector table, so these
helpers operate at the SQL layer directly — NOT the network-gated service layer.
No provider is re-probed and no OAuth flow is replayed; a backup is a faithful
row-level copy of what the tables already hold.

Two independent section pairs back the two halves of the connector state:

  * :func:`export_connector_catalog` / :func:`import_connector_catalog` — the
    public, secret-free market: the ``connector_category`` grouping rows, the
    full ``connector_catalog`` (INCLUDING disabled rows, which
    :func:`catalog_store.fetch_catalog` drops), and the read-only
    ``connector_allowed_source`` discovery list. The stored ``descriptor`` JSONB
    is carried verbatim. Each row's ``created_at`` is exported and restored so
    the original creation time survives a round-trip (audit info for
    community-added rows); ``updated_at`` is the write time of the restore.
    Import upserts each row idempotently (``ON CONFLICT (pk) DO UPDATE``),
    categories before providers so the ``connector_catalog.category`` foreign key
    is satisfied, then reloads the in-memory provider cache via
    :func:`catalog_store.refresh_catalog` so the restored rows go live
    in-process.

  * :func:`export_connector_connections` / :func:`import_connector_connections`
    — the per-connection token records. Each entry carries the AES-GCM
    ciphertext verbatim (base64-encoded, NEVER decrypted); the ``encrypted_blob``
    bytes cross the backup as-is. Import re-inserts the ciphertext and its
    ``session_expires_at`` under the original ``connection_id``
    (``ON CONFLICT (connection_id) DO UPDATE``). ``cache_version`` and the
    timestamps are store-regenerated and omitted. After the durable Postgres
    write each restored ``connection_id``'s Redis cache key is invalidated
    (``DEL``) so the next ``get`` repopulates from Postgres:
    :meth:`RedisPgConnectorTokenStore.get` serves a warm cache HIT WITHOUT a
    read-side version check, so restoring into a running deployment with a warm
    cache would otherwise keep serving the pre-import (stale) token for an
    already-cached connection until its key expired. A failed invalidation raises
    loudly.

KEK constraint: the blobs are ciphertext under ``CONNECTORS_KEK``. A restore is
usable only if the SAME ``CONNECTORS_KEK`` is present in the target deployment;
under a different KEK the ciphertext is intact but undecryptable, and the record
load path (``connectors.store.persistence``) fails loudly on the first use of a
connection — the restore never silently produces a working-looking-but-dead
token.
"""

from __future__ import annotations

import base64
import uuid
from datetime import datetime
from typing import Any

from psycopg.errors import UniqueViolation
from tai42_contract.connectors.providers import ProviderDescriptor
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.postgres import Json, PostgresClient
from tai42_kit.clients.impl.redis import RedisClient

from tai42_skeleton.connectors.settings import connector_store_settings
from tai42_skeleton.connectors.store.catalog_store import refresh_catalog
from tai42_skeleton.connectors.store.redis_pg import _ALIAS_UNIQUE_CONSTRAINT, RedisPgConnectorTokenStore

# The report shape every importer returns, matching the backup section contract
# (``connectors.store.backup`` is a separate subsystem from ``backup.sections``,
# so it builds its own report rather than importing across the seam).
_SectionReport = dict[str, Any]


def _empty_report() -> _SectionReport:
    return {"created": 0, "updated": 0, "skipped": 0, "errors": []}


# -- connector_catalog (public, secret-free) ---------------------------------


async def export_connector_catalog() -> dict[str, Any]:
    """Export the categories, the full catalog, and the allowed-source list.

    Every row is read raw — the catalog INCLUDES disabled rows (unlike
    :func:`catalog_store.fetch_catalog`, which drops them) and the ``descriptor``
    JSONB is carried verbatim. No descriptor is parsed or re-validated here; the
    export is a faithful row copy.
    """
    async with (
        client_ctx(PostgresClient, connector_store_settings().pg) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        await cur.execute(
            "SELECT id, display_name, sort_order, created_at FROM connector_category ORDER BY sort_order, id"
        )
        categories = [
            {
                "id": category_id,
                "display_name": display_name,
                "sort_order": sort_order,
                "created_at": created_at.isoformat(),
            }
            for category_id, display_name, sort_order, created_at in await cur.fetchall()
        ]
        await cur.execute(
            "SELECT provider_id, descriptor, origin, category, source_url, added_by, enabled, created_at "
            "FROM connector_catalog ORDER BY provider_id"
        )
        providers = [
            {
                "provider_id": provider_id,
                "descriptor": descriptor,
                "origin": origin,
                "category": category,
                "source_url": source_url,
                "added_by": added_by,
                "enabled": enabled,
                "created_at": created_at.isoformat(),
            }
            for provider_id, descriptor, origin, category, source_url, added_by, enabled, created_at in await cur.fetchall()  # noqa: E501
        ]
        await cur.execute("SELECT id, url, enabled, created_at FROM connector_allowed_source ORDER BY id")
        sources = [
            {"id": source_id, "url": url, "enabled": enabled, "created_at": created_at.isoformat()}
            for source_id, url, enabled, created_at in await cur.fetchall()
        ]
    return {"categories": categories, "providers": providers, "sources": sources}


async def import_connector_catalog(payload: dict[str, Any]) -> _SectionReport:
    """Upsert the categories, catalog rows, and allowed sources idempotently.

    Categories are written before providers so the ``connector_catalog.category``
    foreign key is satisfied inside the same transaction. Each row is an
    ``ON CONFLICT (pk) DO UPDATE`` (idempotent), so a re-import over identical
    rows is a no-op change reported as ``updated``. The backed-up ``created_at``
    is written on the INSERT so the original creation time survives a restore
    into a fresh table; it is immutable, so the ``DO UPDATE`` branch leaves the
    existing value untouched. ``connector_catalog`` additionally refreshes its
    ``updated_at`` to the restore write time; the category and allowed-source
    tables have no such column.

    Every ENABLED catalog row is validated inside the transaction BEFORE the first
    INSERT — mirroring :func:`catalog_store.fetch_catalog`'s per-row checks (no
    embedded origin/category, a known category, a community row carries
    ``added_by``, a parseable no-auth ``ProviderDescriptor`` whose id matches the
    row) — so a malformed backup aborts with nothing committed rather than landing
    a poison row that then fails every connector-using worker at startup. Disabled
    rows are carried verbatim, exactly the rows ``fetch_catalog`` drops. After the
    commit the in-memory provider cache is reloaded via :func:`refresh_catalog`
    (the publish step, not the gate) so the restored rows serve in-process.
    """
    report = _empty_report()
    categories = payload.get("categories") or []
    providers = payload.get("providers") or []
    sources = payload.get("sources") or []

    async with (
        client_ctx(PostgresClient, connector_store_settings().pg) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        # Read the current keys first so each upsert is classified created vs
        # updated (the same read-existing-then-upsert pattern the other sections
        # use); this drives only the report counts, not the write itself.
        await cur.execute("SELECT id FROM connector_category")
        existing_categories = {row[0] for row in await cur.fetchall()}
        await cur.execute("SELECT provider_id FROM connector_catalog")
        existing_providers = {row[0] for row in await cur.fetchall()}
        await cur.execute("SELECT id FROM connector_allowed_source")
        existing_sources = {row[0] for row in await cur.fetchall()}

        # Gate every enabled row before any write: a bad backup must abort with
        # nothing committed, not commit a poison row that then breaks worker boot.
        # A category is valid if it already exists or arrives in this same payload
        # (imported categories are inserted before providers in this transaction).
        valid_category_ids = existing_categories | {category["id"] for category in categories}
        for provider in providers:
            if not provider["enabled"]:
                continue
            _validate_enabled_catalog_row(provider, valid_category_ids)

        for category in categories:
            await cur.execute(
                "INSERT INTO connector_category (id, display_name, sort_order, created_at) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE "
                "SET display_name = EXCLUDED.display_name, sort_order = EXCLUDED.sort_order",
                (
                    category["id"],
                    category["display_name"],
                    category["sort_order"],
                    datetime.fromisoformat(category["created_at"]),
                ),
            )
            _count(report, category["id"] in existing_categories)

        for provider in providers:
            await cur.execute(
                "INSERT INTO connector_catalog "
                "(provider_id, descriptor, origin, category, source_url, added_by, enabled, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (provider_id) DO UPDATE "
                "SET descriptor = EXCLUDED.descriptor, "
                "    origin = EXCLUDED.origin, "
                "    category = EXCLUDED.category, "
                "    source_url = EXCLUDED.source_url, "
                "    added_by = EXCLUDED.added_by, "
                "    enabled = EXCLUDED.enabled, "
                "    updated_at = now()",
                (
                    provider["provider_id"],
                    Json(provider["descriptor"]),
                    provider["origin"],
                    provider["category"],
                    provider["source_url"],
                    provider["added_by"],
                    provider["enabled"],
                    datetime.fromisoformat(provider["created_at"]),
                ),
            )
            _count(report, provider["provider_id"] in existing_providers)

        for source in sources:
            await cur.execute(
                "INSERT INTO connector_allowed_source (id, url, enabled, created_at) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE "
                "SET url = EXCLUDED.url, enabled = EXCLUDED.enabled",
                (source["id"], source["url"], source["enabled"], datetime.fromisoformat(source["created_at"])),
            )
            _count(report, source["id"] in existing_sources)

    # Publish the restored (enabled) rows into the in-memory provider cache. The
    # rows were already gated above, so this is the publish step, not the gate.
    await refresh_catalog()
    return report


def _validate_enabled_catalog_row(provider: dict[str, Any], valid_category_ids: set[str]) -> None:
    """Reject a malformed ENABLED catalog row before any durable write.

    Mirrors :func:`catalog_store.fetch_catalog`'s per-row checks so the two paths
    agree on what a valid enabled row is: the ``origin``/``category`` columns are
    authoritative (the descriptor jsonb must not embed them), the category must be
    known, a community-origin row must carry ``added_by``, and the descriptor must
    parse into a no-auth :class:`ProviderDescriptor` whose id matches the row.
    """
    provider_id = provider["provider_id"]
    descriptor_json = provider["descriptor"]
    origin = provider["origin"]
    category = provider["category"]
    added_by = provider["added_by"]
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


# -- connector_connections (encrypted token records, secret) -----------------


async def export_connector_connections() -> list[dict[str, Any]]:
    """Export every connection record with its ciphertext carried verbatim.

    The ``encrypted_blob`` bytes are base64-encoded AS-IS — never decrypted — so
    the KEK boundary is never crossed. ``cache_version`` and the timestamps are
    store-regenerated on restore and therefore omitted.
    """
    async with (
        client_ctx(PostgresClient, connector_store_settings().pg) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        await cur.execute(
            "SELECT connection_id, provider_id, alias, encrypted_blob, session_expires_at "
            "FROM connector_connections ORDER BY connection_id"
        )
        rows = await cur.fetchall()
    return [
        {
            "connection_id": str(connection_id),
            "provider_id": provider_id,
            "alias": alias,
            "session_expires_at": None if session_expires_at is None else session_expires_at.isoformat(),
            "encrypted_blob_b64": base64.b64encode(bytes(encrypted_blob)).decode("ascii"),
        }
        for connection_id, provider_id, alias, encrypted_blob, session_expires_at in rows
    ]


async def import_connector_connections(payload: list[dict[str, Any]]) -> _SectionReport:
    """Re-insert each connection's ciphertext under its original connection id.

    Each row runs inside its own savepoint (``conn.transaction()``) so a
    per-provider alias collision isolates to that row: an ``(provider_id, alias)``
    already held by a DIFFERENT ``connection_id`` trips the durable
    ``UNIQUE (provider_id, alias)`` constraint, which is caught and reported as a
    per-row error (never silently dropped) while the remaining rows still
    restore. Any other database error raises loudly and aborts the section.

    After the Postgres writes commit, each restored ``connection_id``'s Redis
    cache key is invalidated so a warm cache in a running deployment cannot keep
    serving the pre-import (stale) token — see :func:`_invalidate_connection_cache`.
    """
    report = _empty_report()
    restored_ids: list[str] = []
    async with (
        client_ctx(PostgresClient, connector_store_settings().pg) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        await cur.execute("SELECT connection_id FROM connector_connections")
        existing = {str(row[0]) for row in await cur.fetchall()}

        for entry in payload:
            connection_id = entry["connection_id"]
            conn_uuid = uuid.UUID(connection_id)
            blob = base64.b64decode(entry["encrypted_blob_b64"])
            raw_expiry = entry.get("session_expires_at")
            session_expires_at = None if raw_expiry is None else datetime.fromisoformat(raw_expiry)
            try:
                async with conn.transaction():
                    await cur.execute(
                        "INSERT INTO connector_connections "
                        "(connection_id, provider_id, alias, encrypted_blob, session_expires_at) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (connection_id) DO UPDATE "
                        "SET provider_id = EXCLUDED.provider_id, "
                        "    alias = EXCLUDED.alias, "
                        "    encrypted_blob = EXCLUDED.encrypted_blob, "
                        "    session_expires_at = EXCLUDED.session_expires_at, "
                        "    cache_version = connector_connections.cache_version + 1, "
                        "    updated_at = now()",
                        (conn_uuid, entry["provider_id"], entry["alias"], blob, session_expires_at),
                    )
            except UniqueViolation as exc:
                if getattr(exc.diag, "constraint_name", None) == _ALIAS_UNIQUE_CONSTRAINT:
                    report["errors"].append(
                        f"connection {connection_id!r}: alias {entry['alias']!r} is already in use "
                        f"for provider {entry['provider_id']!r} by a different connection"
                    )
                    report["skipped"] += 1
                    continue
                raise
            _count(report, connection_id in existing)
            # Invalidate on the canonical UUID the row is stored under (and that
            # get() is always handed), not the raw backup string — a non-canonical
            # id in a foreign backup would otherwise drop the wrong cache key.
            restored_ids.append(str(conn_uuid))

    # Postgres is now durable; drop the stale cache entries so the next read
    # repopulates the restored token from Postgres (raises loudly on failure).
    await _invalidate_connection_cache(restored_ids)
    return report


async def _invalidate_connection_cache(connection_ids: list[str]) -> None:
    """Drop each restored connection's Redis cache key after the durable Postgres
    write so the next ``get`` repopulates from Postgres.

    :meth:`RedisPgConnectorTokenStore.get` returns a cached blob on a cache HIT
    WITHOUT a read-side version check (the ``cache_version`` fence guards only the
    cache-MISS repopulate). A restore into a running deployment with a warm cache
    would therefore keep serving the pre-import (stale) token for any
    already-cached ``connection_id`` until its key expired. Deleting the key
    forces a version-fenced repopulate from the freshly restored row.

    The key is built through the store's own ``_rec_key`` helper so the cache
    keyspace stays single-sourced. A failed ``DEL`` raises loudly — the restore
    must not silently leave a stale token cached.
    """
    if not connection_ids:
        return
    store = RedisPgConnectorTokenStore()
    async with client_ctx(RedisClient, connector_store_settings().redis) as client:
        for connection_id in connection_ids:
            await client.delete(store._rec_key(connection_id))


def _count(report: _SectionReport, existed: bool) -> None:
    """Bump the created/updated tally for one upserted row."""
    if existed:
        report["updated"] += 1
    else:
        report["created"] += 1
