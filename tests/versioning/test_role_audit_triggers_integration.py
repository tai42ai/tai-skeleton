"""REAL-Postgres tests for the ``kind='role_audit'`` append-only triggers.

The role-edit audit trail rides the generic versioned store under
``kind='role_audit'``. The schema (``tai42_skeleton.init.sql``) installs three row
triggers that make those documents append-only IN THE DATABASE, so no code path or
bug can rewrite the security record. These tests prove the triggers on real
Postgres — there is no fake for DB-side triggers, so they need a live server.

Proven here:

* an APPEND (create → save_version) succeeds — a new version row plus the
  ``active_version`` pointer bump are the only legitimate audit writes and must NOT
  be blocked;
* a direct ``UPDATE`` of a role_audit version row RAISES (versions are immutable);
* a hard ``DELETE`` of the role_audit document RAISES (no erasing the trail);
* a soft-delete ``is_active`` flip RAISES;
* a ``rename`` RAISES;
* a NON-role_audit kind is entirely unaffected — normal edit / rename / soft-delete
  / hard-delete still work.

OPT-IN: set ``TAI42_SKELETON_REAL_PG=1`` and point ``VERSIONING_STORE_*`` at a live
Postgres. Without the opt-in the tests SKIP VISIBLY with a clear reason (never a
silent skip).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import LiteralString, cast

import psycopg
import pytest
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.postgres import PostgresClient
from tai42_kit.settings import reset_all_settings

from tai42_skeleton.sql.schema import load_ddl
from tai42_skeleton.versioning.settings import versioning_store_settings
from tai42_skeleton.versioning.store import PostgresVersionedStore

pytestmark = pytest.mark.integration

_OPT_IN_ENV = "TAI42_SKELETON_REAL_PG"

_AUDIT_KIND = "role_audit"


async def _exec(sql: LiteralString, params: tuple | None = None) -> None:
    # ``params=None`` (not an empty tuple) so psycopg skips placeholder parsing —
    # the full DDL carries literal ``%`` in the trigger RAISE messages, which an
    # empty-tuple call would misread as an incomplete placeholder (this mirrors the
    # production apply path in ``cli/native/db.py``, which passes no params).
    async with (
        client_ctx(PostgresClient, versioning_store_settings()) as pool,
        pool.connection() as conn,
    ):
        await conn.execute(sql, params)


async def _fetchval(sql: LiteralString, params: tuple | None = None) -> object:
    async with (
        client_ctx(PostgresClient, versioning_store_settings()) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        await cur.execute(sql, params)
        row = await cur.fetchone()
    return None if row is None else row[0]


async def _cleanup(tag: str) -> None:
    """Remove every row this test tagged. role_audit rows are undeletable while the
    triggers are live, so disable USER triggers on both tables for the delete (the FK
    cascade is a system trigger and stays on), then re-enable. Disabling triggers is
    the owner-level admin escape the WORM boundary deliberately leaves open."""
    like = f"%{tag}%"
    async with (
        client_ctx(PostgresClient, versioning_store_settings()) as pool,
        pool.connection() as conn,
    ):
        await conn.execute("ALTER TABLE versioned_document_versions DISABLE TRIGGER USER")
        await conn.execute("ALTER TABLE versioned_documents DISABLE TRIGGER USER")
        await conn.execute("DELETE FROM versioned_documents WHERE name LIKE %s", (like,))
        await conn.execute("ALTER TABLE versioned_documents ENABLE TRIGGER USER")
        await conn.execute("ALTER TABLE versioned_document_versions ENABLE TRIGGER USER")


@pytest.fixture
async def real_store() -> AsyncIterator[tuple[PostgresVersionedStore, str]]:
    if os.environ.get(_OPT_IN_ENV) not in ("1", "true", "True"):
        pytest.skip(
            f"real-Postgres role_audit trigger test is opt-in: set {_OPT_IN_ENV}=1 and point the "
            "VERSIONING_STORE_* env at a live Postgres to run it (DB-side triggers have no fake)"
        )
    # Rebuild cached settings so ``VERSIONING_STORE_*`` from the environment is read.
    reset_all_settings()
    # Apply the FULL shipped DDL (idempotent) so the test exercises the EXACT triggers
    # that ship — never a hand-copied duplicate that could drift.
    await _exec(cast(LiteralString, load_ddl()))
    tag = uuid.uuid4().hex
    try:
        yield PostgresVersionedStore(), tag
    finally:
        await _cleanup(tag)


async def _audit_doc_id(kind: str, name: str) -> int:
    doc_id = await _fetchval(
        "SELECT id FROM versioned_documents WHERE kind = %s AND name = %s AND is_active",
        (kind, name),
    )
    assert isinstance(doc_id, int)
    return doc_id


async def test_role_audit_append_succeeds(real_store) -> None:
    """create + save_version: a new version row and the active_version bump — the
    legitimate audit append — are NOT blocked by the guards."""
    store, tag = real_store
    name = f"role_{tag}"

    await store.create(_AUDIT_KIND, name, {"event": 1})
    version = await store.save_version(_AUDIT_KIND, name, {"event": 2})

    assert version.version == 2
    versions = await store.list_versions(_AUDIT_KIND, name)
    assert [v.version for v in versions] == [1, 2]
    assert versions[-1].is_current  # active_version bumped to 2


async def test_role_audit_version_update_raises(real_store) -> None:
    """A direct UPDATE of a role_audit version row RAISES — versions are immutable."""
    store, tag = real_store
    name = f"role_{tag}"
    await store.create(_AUDIT_KIND, name, {"event": 1})
    doc_id = await _audit_doc_id(_AUDIT_KIND, name)

    with pytest.raises(psycopg.errors.RaiseException):
        await _exec(
            "UPDATE versioned_document_versions SET body = %s WHERE document_id = %s AND version = 1",
            ('{"event": "tampered"}', doc_id),
        )
    # The original body is untouched.
    body = await store.get_active_body(_AUDIT_KIND, name)
    assert body == {"event": 1}


async def test_role_audit_version_delete_raises(real_store) -> None:
    """A direct DELETE of a role_audit version row RAISES — history is append-only."""
    store, tag = real_store
    name = f"role_{tag}"
    await store.create(_AUDIT_KIND, name, {"event": 1})
    doc_id = await _audit_doc_id(_AUDIT_KIND, name)

    with pytest.raises(psycopg.errors.RaiseException):
        await _exec(
            "DELETE FROM versioned_document_versions WHERE document_id = %s AND version = 1",
            (doc_id,),
        )


async def test_role_audit_doc_delete_raises(real_store) -> None:
    """A hard delete of the role_audit document RAISES — no erasing the trail (and the
    FK cascade never reaches the version rows)."""
    store, tag = real_store
    name = f"role_{tag}"
    await store.create(_AUDIT_KIND, name, {"event": 1})

    with pytest.raises(psycopg.errors.RaiseException):
        await store.delete(_AUDIT_KIND, name)
    # Still present.
    assert await store.get_active_body(_AUDIT_KIND, name) == {"event": 1}


async def test_role_audit_is_active_flip_raises(real_store) -> None:
    """A soft-delete is_active flip on a role_audit document RAISES."""
    store, tag = real_store
    name = f"role_{tag}"
    await store.create(_AUDIT_KIND, name, {"event": 1})

    with pytest.raises(psycopg.errors.RaiseException):
        await store.soft_delete(_AUDIT_KIND, name)


async def test_role_audit_rename_raises(real_store) -> None:
    """A rename of a role_audit document RAISES."""
    store, tag = real_store
    name = f"role_{tag}"
    await store.create(_AUDIT_KIND, name, {"event": 1})

    with pytest.raises(psycopg.errors.RaiseException):
        await store.rename(_AUDIT_KIND, name, f"renamed_{tag}")


async def test_non_role_audit_kind_unaffected(real_store) -> None:
    """A kind that is NOT role_audit keeps the store's full mutable surface: edit,
    version-tag update, rename, soft-delete, and hard-delete all still work."""
    store, tag = real_store
    kind = f"kind_{tag}"
    name = f"doc_{tag}"

    await store.create(kind, name, {"v": 1})
    await store.save_version(kind, name, {"v": 2})
    # A version-row UPDATE (tags) is allowed for a non-audit kind.
    await store.set_version_tags(kind, name, 1, ["reviewed"])
    v1 = await store.get_version(kind, name, 1)
    assert v1.tags == ["reviewed"]
    # Rename is allowed.
    renamed = f"doc2_{tag}"
    await store.rename(kind, name, renamed)
    # Soft-delete (is_active flip) is allowed.
    await store.soft_delete(kind, renamed)
    # Recreate + hard-delete (with FK cascade of version rows) is allowed.
    await store.create(kind, renamed, {"v": 1})
    await store.delete(kind, renamed)
