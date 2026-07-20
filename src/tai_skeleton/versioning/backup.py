"""Store-level backup export/import for the versioned-document Postgres tables.

Postgres is the durable source of truth for the versioned-document store, so these
helpers operate at the SQL layer directly — NOT through any typed view. The store
is body-opaque and ``kind``-discriminated, so ONE section covers EVERY kind at
once (presets under ``kind='preset'``, AC policies under ``kind='ac_policy'``,
authored agents, and any future kind); no per-kind backup code is ever needed.

The export is a faithful row-level copy of BOTH tables — ``versioned_documents``
(including soft-deleted ghosts, so the audit history survives) and
``versioned_document_versions`` (the full append-only version log). Each row's
synthetic ``id`` is carried verbatim so the ``document_id`` foreign key linking a
version to its document is preserved across the round-trip; the opaque ``body``
JSONB is carried as-is and never inspected. Import re-inserts each row under its
original id (``ON CONFLICT (id) DO UPDATE``), documents before versions so the FK
is satisfied, then advances both ``BIGSERIAL`` sequences past the restored ids so
a later insert cannot collide with a restored id.

Secret constraint: the bodies are opaque and at least one kind is secret-bearing —
a preset's ``fixed_kwargs`` can embed credentials and an AC-policy condition body
is sensitive — so the section is registered ``secret=True`` (default-OFF in the
export UI, treated like a secret), mirroring the connector-connections section.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from tai_kit.clients import client_ctx
from tai_kit.clients.impl.postgres import Json, PostgresClient

from tai_skeleton.versioning.settings import versioning_store_settings

# The report shape every importer returns, matching the backup section contract.
_SectionReport = dict[str, Any]


def _empty_report() -> _SectionReport:
    return {"created": 0, "updated": 0, "skipped": 0, "errors": []}


async def export_versioned_documents() -> dict[str, Any]:
    """Export every document row and every version row verbatim.

    Both tables are read whole — soft-deleted ghosts and their history included —
    and each row keeps its synthetic ``id`` so the version-to-document link
    survives the round-trip. The ``body`` JSONB is carried as-is (never parsed);
    ``created_at`` is serialized so the original timestamps restore intact.
    """
    async with (
        client_ctx(PostgresClient, versioning_store_settings()) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        await cur.execute(
            "SELECT id, kind, name, active_version, is_active, created_at FROM versioned_documents ORDER BY id"
        )
        documents = [
            {
                "id": doc_id,
                "kind": kind,
                "name": name,
                "active_version": active_version,
                "is_active": is_active,
                "created_at": created_at.isoformat(),
            }
            for doc_id, kind, name, active_version, is_active, created_at in await cur.fetchall()
        ]
        await cur.execute(
            "SELECT id, document_id, version, body, tags, created_at FROM versioned_document_versions ORDER BY id"
        )
        versions = [
            {
                "id": version_id,
                "document_id": document_id,
                "version": version,
                "body": body,
                "tags": list(tags or []),
                "created_at": created_at.isoformat(),
            }
            for version_id, document_id, version, body, tags, created_at in await cur.fetchall()
        ]
    return {"documents": documents, "versions": versions}


async def import_versioned_documents(payload: dict[str, Any]) -> _SectionReport:
    """Restore document + version rows under their original ids.

    Documents are written before versions so the ``document_id`` foreign key is
    satisfied. Every row is an ``ON CONFLICT (id) DO UPDATE`` (idempotent, so a
    re-import over identical rows is a no-op reported as ``updated``); the created
    count classifies each document by whether its id already existed. After the
    writes, both ``BIGSERIAL`` sequences are advanced past the largest restored id
    so a later insert cannot collide with a restored row.
    """
    report = _empty_report()
    documents = payload.get("documents") or []
    versions = payload.get("versions") or []

    async with (
        client_ctx(PostgresClient, versioning_store_settings()) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        # Classify created vs updated by pre-existing id (the same read-existing-
        # then-upsert pattern the connector sections use); drives the counts only.
        await cur.execute("SELECT id FROM versioned_documents")
        existing = {row[0] for row in await cur.fetchall()}

        for document in documents:
            await cur.execute(
                "INSERT INTO versioned_documents (id, kind, name, active_version, is_active, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE "
                "SET kind = EXCLUDED.kind, "
                "    name = EXCLUDED.name, "
                "    active_version = EXCLUDED.active_version, "
                "    is_active = EXCLUDED.is_active, "
                "    created_at = EXCLUDED.created_at",
                (
                    document["id"],
                    document["kind"],
                    document["name"],
                    document["active_version"],
                    document["is_active"],
                    datetime.fromisoformat(document["created_at"]),
                ),
            )
            if document["id"] in existing:
                report["updated"] += 1
            else:
                report["created"] += 1

        for version in versions:
            await cur.execute(
                "INSERT INTO versioned_document_versions (id, document_id, version, body, tags, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE "
                "SET document_id = EXCLUDED.document_id, "
                "    version = EXCLUDED.version, "
                "    body = EXCLUDED.body, "
                "    tags = EXCLUDED.tags, "
                "    created_at = EXCLUDED.created_at",
                (
                    version["id"],
                    version["document_id"],
                    version["version"],
                    Json(version["body"]),
                    list(version["tags"] or []),
                    datetime.fromisoformat(version["created_at"]),
                ),
            )

        # Advance the serial sequences past the restored ids so the next natural
        # insert does not collide with a preserved id. Guarded on non-empty so
        # ``setval`` never sees a NULL ``MAX(id)``.
        if documents:
            await cur.execute(
                "SELECT setval(pg_get_serial_sequence('versioned_documents', 'id'), "
                "(SELECT MAX(id) FROM versioned_documents))"
            )
        if versions:
            await cur.execute(
                "SELECT setval(pg_get_serial_sequence('versioned_document_versions', 'id'), "
                "(SELECT MAX(id) FROM versioned_document_versions))"
            )

    return report
