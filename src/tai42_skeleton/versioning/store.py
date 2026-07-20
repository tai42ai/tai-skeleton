"""Postgres-backed concrete :class:`~tai42_contract.versioning.VersionedStore`.

The generic versioning primitive: append-only version rows over an opaque JSONB
``body``, an active-version pointer, and rollback, discriminated by ``kind`` and
identified by ``(kind, name)``. Two tables (``versioned_documents`` +
``versioned_document_versions``, defined in the skeleton init SQL) hold the state;
Postgres is the sole store — there is no cache layer, because a document's body is
read only at registration/startup and then held in the consumer's closure, not on
a hot per-request path (a consumer that needs a hot read adds its own cache).

The store is body-opaque: it NEVER inspects ``body`` (a typed view owns the shape).
Every write runs inside one ``conn.transaction()`` so a partial failure rolls the
whole write back — no orphan version, no half-bumped pointer. The partial-unique
index ``versioned_documents_active_name_unique`` enforces one ACTIVE row per
``(kind, name)``; a duplicate active insert trips it and surfaces as
:class:`DocumentExistsError`. The FK ``ON DELETE CASCADE`` drops a document's
version rows with its active row on a hard :meth:`delete`.

Postgres is reached through the app-pooled ``PostgresClient`` so it shares one pool
per DSN with the other durable stores, closed centrally at shutdown.
"""

from __future__ import annotations

from typing import Any

from psycopg.errors import UniqueViolation
from tai42_contract.versioning import VersionedStore
from tai42_contract.versioning.errors import (
    DocumentExistsError,
    DocumentNotFoundError,
    DocumentVersionNotFoundError,
)
from tai42_contract.versioning.models import DocumentRecord, DocumentVersion
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.postgres import Json, PostgresClient

from tai42_skeleton.versioning.settings import versioning_store_settings

# Name of the partial-unique index enforcing one ACTIVE row per ``(kind, name)``
# (see the skeleton init SQL). A create whose insert trips THIS index is a live
# duplicate → ``DocumentExistsError``; any other unique violation re-raises.
_ACTIVE_NAME_UNIQUE_INDEX = "versioned_documents_active_name_unique"


def _is_active_name_violation(exc: UniqueViolation) -> bool:
    return getattr(exc.diag, "constraint_name", None) == _ACTIVE_NAME_UNIQUE_INDEX


class PostgresVersionedStore(VersionedStore):
    """Postgres implementation of the generic versioned-document store."""

    async def create(self, kind: str, name: str, body: dict[str, Any], tags: list[str] | None = None) -> DocumentRecord:
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
        ):
            try:
                async with conn.transaction(), conn.cursor() as cur:
                    await cur.execute(
                        "INSERT INTO versioned_documents (kind, name, active_version) "
                        "VALUES (%s, %s, 1) RETURNING id, created_at",
                        (kind, name),
                    )
                    doc_id, created_at = _require_row(await cur.fetchone())
                    await cur.execute(
                        "INSERT INTO versioned_document_versions (document_id, version, body, tags) "
                        "VALUES (%s, %s, %s, %s)",
                        (doc_id, 1, Json(body), tags or []),
                    )
            except UniqueViolation as exc:
                if _is_active_name_violation(exc):
                    raise DocumentExistsError(kind, name) from exc
                raise
            return DocumentRecord(
                kind=kind, name=name, active_version=1, is_active=True, created_at=created_at.isoformat()
            )

    async def save_version(
        self, kind: str, name: str, body: dict[str, Any], tags: list[str] | None = None
    ) -> DocumentVersion:
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
        ):
            async with conn.transaction(), conn.cursor() as cur:
                # ``FOR UPDATE`` row-locks the active document so two concurrent
                # saves serialize: the second waits here, then reads the MAX the
                # first already appended and numbers MAX+1 — no duplicate
                # ``(document_id, version)`` insert, no retry loop.
                await cur.execute(
                    "SELECT id FROM versioned_documents WHERE kind = %s AND name = %s AND is_active FOR UPDATE",
                    (kind, name),
                )
                row = await cur.fetchone()
                if row is None:
                    raise DocumentNotFoundError(kind, name)
                doc_id = row[0]
                # ``MAX(version) + 1``, NOT ``active_version + 1``: after a rollback
                # the active pointer trails MAX, so the next version must extend the
                # append log rather than collide with an existing version number.
                await cur.execute(
                    "SELECT MAX(version) FROM versioned_document_versions WHERE document_id = %s",
                    (doc_id,),
                )
                new_version = _require_row(await cur.fetchone())[0] + 1
                await cur.execute(
                    "INSERT INTO versioned_document_versions (document_id, version, body, tags) "
                    "VALUES (%s, %s, %s, %s) RETURNING created_at",
                    (doc_id, new_version, Json(body), tags or []),
                )
                created_at = _require_row(await cur.fetchone())[0]
                await cur.execute(
                    "UPDATE versioned_documents SET active_version = %s WHERE id = %s",
                    (new_version, doc_id),
                )
            return DocumentVersion(
                version=new_version,
                body=body,
                tags=list(tags or []),
                created_at=created_at.isoformat(),
                is_current=True,
            )

    async def list(self, kind: str) -> list[DocumentRecord]:
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT kind, name, active_version, is_active, created_at "
                "FROM versioned_documents WHERE kind = %s AND is_active ORDER BY name",
                (kind,),
            )
            rows = await cur.fetchall()
        return [
            DocumentRecord(
                kind=row_kind,
                name=row_name,
                active_version=active_version,
                is_active=is_active,
                created_at=created_at.isoformat(),
            )
            for row_kind, row_name, active_version, is_active, created_at in rows
        ]

    async def list_active_bodies(self, kind: str) -> dict[str, dict[str, Any]]:
        """Every active document body of ``kind``, keyed by name — the batched
        read that replaces a per-record ``get_active_body`` round-trip (the list
        route + rehydrate N+1). One JOIN on ``version = active_version``.

        Concrete-only (not on the ``VersionedStore`` protocol): callers reach it
        through the concretely-typed ``_versioned_store`` accessor."""
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT d.name, v.body FROM versioned_documents d "
                "JOIN versioned_document_versions v ON v.document_id = d.id AND v.version = d.active_version "
                "WHERE d.kind = %s AND d.is_active ORDER BY d.name",
                (kind,),
            )
            rows = await cur.fetchall()
        return dict(rows)

    async def get(self, kind: str, name: str) -> DocumentRecord:
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT active_version, created_at FROM versioned_documents "
                "WHERE kind = %s AND name = %s AND is_active",
                (kind, name),
            )
            row = await cur.fetchone()
        if row is None:
            raise DocumentNotFoundError(kind, name)
        active_version, created_at = row
        return DocumentRecord(
            kind=kind, name=name, active_version=active_version, is_active=True, created_at=created_at.isoformat()
        )

    async def get_active_body(self, kind: str, name: str) -> dict[str, Any]:
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT v.body FROM versioned_documents d "
                "JOIN versioned_document_versions v ON v.document_id = d.id AND v.version = d.active_version "
                "WHERE d.kind = %s AND d.name = %s AND d.is_active",
                (kind, name),
            )
            row = await cur.fetchone()
        if row is None:
            raise DocumentNotFoundError(kind, name)
        return row[0]

    async def list_versions(self, kind: str, name: str) -> list[DocumentVersion]:
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT id, active_version FROM versioned_documents WHERE kind = %s AND name = %s AND is_active",
                (kind, name),
            )
            doc = await cur.fetchone()
            if doc is None:
                raise DocumentNotFoundError(kind, name)
            doc_id, active_version = doc
            await cur.execute(
                "SELECT version, body, tags, created_at FROM versioned_document_versions "
                "WHERE document_id = %s ORDER BY version",
                (doc_id,),
            )
            rows = await cur.fetchall()
        return [
            DocumentVersion(
                version=version,
                body=body,
                tags=list(row_tags or []),
                created_at=created_at.isoformat(),
                is_current=version == active_version,
            )
            for version, body, row_tags, created_at in rows
        ]

    async def get_version(self, kind: str, name: str, version: int) -> DocumentVersion:
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT d.active_version, v.body, v.tags, v.created_at FROM versioned_documents d "
                "JOIN versioned_document_versions v ON v.document_id = d.id "
                "WHERE d.kind = %s AND d.name = %s AND d.is_active AND v.version = %s",
                (kind, name, version),
            )
            row = await cur.fetchone()
        if row is None:
            raise DocumentVersionNotFoundError(kind, name, version)
        active_version, body, row_tags, created_at = row
        return DocumentVersion(
            version=version,
            body=body,
            tags=list(row_tags or []),
            created_at=created_at.isoformat(),
            is_current=version == active_version,
        )

    async def set_version_tags(self, kind: str, name: str, version: int, tags: list[str]) -> None:
        """Replace the ``tags`` annotation on one version row — labels on an
        immutable version body, edited without touching the body.

        Concrete-only (not on the ``VersionedStore`` protocol): reached through the
        concretely-typed ``_versioned_store`` accessor. Resolves the active
        document, then UPDATEs the named version's ``tags`` column. Raises
        :class:`DocumentVersionNotFoundError` for an unknown document or version,
        mirroring :meth:`get_version`'s error style."""
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT id FROM versioned_documents WHERE kind = %s AND name = %s AND is_active",
                (kind, name),
            )
            doc = await cur.fetchone()
            if doc is None:
                raise DocumentVersionNotFoundError(kind, name, version)
            await cur.execute(
                "UPDATE versioned_document_versions SET tags = %s WHERE document_id = %s AND version = %s",
                (list(tags), doc[0], version),
            )
            if cur.rowcount == 0:
                raise DocumentVersionNotFoundError(kind, name, version)

    async def rollback(self, kind: str, name: str, version: int) -> DocumentRecord:
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
        ):
            async with conn.transaction(), conn.cursor() as cur:
                # ``FOR UPDATE`` row-locks the active document so a rollback never
                # races a concurrent save's pointer bump — the two serialize on the
                # same row rather than interleaving their ``active_version`` writes.
                await cur.execute(
                    "SELECT id, created_at FROM versioned_documents "
                    "WHERE kind = %s AND name = %s AND is_active FOR UPDATE",
                    (kind, name),
                )
                doc = await cur.fetchone()
                if doc is None:
                    raise DocumentVersionNotFoundError(kind, name, version)
                doc_id, created_at = doc
                await cur.execute(
                    "SELECT 1 FROM versioned_document_versions WHERE document_id = %s AND version = %s",
                    (doc_id, version),
                )
                if await cur.fetchone() is None:
                    raise DocumentVersionNotFoundError(kind, name, version)
                # Re-point the active pointer only — NO data copy.
                await cur.execute(
                    "UPDATE versioned_documents SET active_version = %s WHERE id = %s",
                    (version, doc_id),
                )
            return DocumentRecord(
                kind=kind, name=name, active_version=version, is_active=True, created_at=created_at.isoformat()
            )

    async def soft_delete(self, kind: str, name: str) -> None:
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "UPDATE versioned_documents SET is_active = FALSE WHERE kind = %s AND name = %s AND is_active",
                (kind, name),
            )
            if cur.rowcount == 0:
                raise DocumentNotFoundError(kind, name)

    async def delete(self, kind: str, name: str) -> None:
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            # HARD delete of the ACTIVE row only (structurally at most one per
            # name); the FK cascade drops its version rows with it, while any
            # soft-deleted ghost of the same name is left untouched.
            await cur.execute(
                "DELETE FROM versioned_documents WHERE kind = %s AND name = %s AND is_active",
                (kind, name),
            )
            if cur.rowcount == 0:
                raise DocumentNotFoundError(kind, name)

    async def rename(self, kind: str, name: str, new_name: str) -> DocumentRecord:
        async with (
            client_ctx(PostgresClient, versioning_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            # One UPDATE re-keys the ACTIVE row — atomic by itself, no explicit
            # transaction. The version rows FK on ``document_id``, so the full history,
            # per-version tags, and the ``active_version`` pointer move with the row
            # untouched — nothing is copied. The partial-unique active index makes the
            # collision check-and-claim atomic with the move: a race with a concurrent
            # ``create(new_name)`` resolves to one winner, the loser surfacing as
            # ``DocumentExistsError``.
            try:
                await cur.execute(
                    "UPDATE versioned_documents SET name = %s WHERE kind = %s AND name = %s AND is_active "
                    "RETURNING active_version, created_at",
                    (new_name, kind, name),
                )
            except UniqueViolation as exc:
                if _is_active_name_violation(exc):
                    raise DocumentExistsError(kind, new_name) from exc
                raise
            row = await cur.fetchone()
            if row is None:
                raise DocumentNotFoundError(kind, name)
            active_version, created_at = row
            return DocumentRecord(
                kind=kind,
                name=new_name,
                active_version=active_version,
                is_active=True,
                created_at=created_at.isoformat(),
            )


def _require_row(row: Any) -> Any:
    """Return ``row`` or fail loudly — a ``RETURNING``/aggregate read the store
    just issued must produce a row; a ``None`` here is a broken invariant, never a
    silent default."""
    if row is None:
        raise RuntimeError("expected a row from the preceding statement, got none")
    return row
