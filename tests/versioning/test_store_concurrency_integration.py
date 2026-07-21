"""A REAL two-transaction Postgres concurrency test for the ``for_update`` locked read.

Proves the non-JOIN two-statement lock in ``PostgresVersionedStore.get_active_body`` does
NOT raise a spurious ``DocumentNotFoundError`` when a second concurrent editor blocks on
the row lock and the first commits a NEW active version. A single ``JOIN ... FOR UPDATE``
trips an EvalPlanQual hazard under READ COMMITTED: the blocked scan re-evaluates the
version join under its ORIGINAL snapshot, where the just-committed version row is
invisible, and finds 0 rows for a document that plainly exists.

This needs real Postgres MVCC — there is no fake here. It is OPT-IN: set
``TAI42_SKELETON_REAL_PG=1`` and point ``VERSIONING_STORE_*`` at a live Postgres. Without
the opt-in the test SKIPS VISIBLY with a clear reason (never a silent skip).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator

import pytest
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.postgres import PostgresClient
from tai42_kit.settings import reset_all_settings

from tai42_skeleton.versioning.settings import versioning_store_settings
from tai42_skeleton.versioning.store import PostgresVersionedStore

pytestmark = pytest.mark.integration

_OPT_IN_ENV = "TAI42_SKELETON_REAL_PG"

# The two tables + the partial-unique active index, verbatim from the skeleton init SQL
# (idempotent ``IF NOT EXISTS`` so an already-migrated database is left untouched).
_SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS versioned_documents ("
    " id BIGSERIAL NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL,"
    " active_version INTEGER NOT NULL, is_active BOOLEAN NOT NULL DEFAULT TRUE,"
    " created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (id))",
    "CREATE UNIQUE INDEX IF NOT EXISTS versioned_documents_active_name_unique"
    " ON versioned_documents (kind, name) WHERE is_active",
    "CREATE TABLE IF NOT EXISTS versioned_document_versions ("
    " id BIGSERIAL NOT NULL,"
    " document_id BIGINT NOT NULL REFERENCES versioned_documents (id) ON DELETE CASCADE,"
    " version INTEGER NOT NULL, body JSONB NOT NULL, tags TEXT[] NOT NULL DEFAULT '{}',"
    " created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (id),"
    " CONSTRAINT versioned_document_versions_doc_version_unique UNIQUE (document_id, version))",
)


async def _exec(sql: str, params: tuple = ()) -> None:
    async with (
        client_ctx(PostgresClient, versioning_store_settings()) as pool,
        pool.connection() as conn,
    ):
        await conn.execute(sql, params)


@pytest.fixture
async def real_store() -> AsyncIterator[tuple[PostgresVersionedStore, str]]:
    if os.environ.get(_OPT_IN_ENV) not in ("1", "true", "True"):
        pytest.skip(
            f"real-Postgres concurrency test is opt-in: set {_OPT_IN_ENV}=1 and point the "
            "VERSIONING_STORE_* env at a live Postgres to run it (needs real MVCC — no fake)"
        )
    # Rebuild the cached settings so ``VERSIONING_STORE_*`` from the environment is read
    # (a stale cached settings object would target the wrong database).
    reset_all_settings()
    for statement in _SCHEMA_SQL:
        await _exec(statement)
    kind = f"role_tx1_it_{uuid.uuid4().hex}"
    try:
        yield PostgresVersionedStore(), kind
    finally:
        # The unique per-test kind confines the fixture to its own rows; the FK cascade
        # drops the version rows with the parent.
        await _exec("DELETE FROM versioned_documents WHERE kind = %s", (kind,))


async def test_concurrent_editor_reads_committed_body_no_spurious_404(real_store):
    store, kind = real_store
    name = "ops"
    await store.create(kind, name, {"gen": 1})

    lock_held = asyncio.Event()
    reader_started = asyncio.Event()

    async def first_editor() -> None:
        async with store.transaction() as tx:
            # Lock the parent row, append v2, and re-point active_version -> 2. The whole
            # transaction (lock + uncommitted v2) is held until the second editor's locked
            # read is blocking on it, then commits on context exit.
            await store.get_active_body(kind, name, tx=tx, for_update=True)
            await store.save_version(kind, name, {"gen": 2}, tx=tx)
            lock_held.set()
            await reader_started.wait()
            # Give the second editor's FOR UPDATE statement time to reach Postgres and
            # block on the lock BEFORE this transaction commits — the exact race that trips
            # the JOIN's EvalPlanQual hazard.
            await asyncio.sleep(0.5)

    async def second_editor() -> dict:
        await lock_held.wait()
        async with store.transaction() as tx:
            reader_started.set()
            # Blocks on the first editor's row lock; unblocks once it commits v2. The fix
            # must then read the COMMITTED v2 body rather than raise a spurious 404.
            return await store.get_active_body(kind, name, tx=tx, for_update=True)

    writer = asyncio.create_task(first_editor())
    body = await second_editor()
    await writer

    assert body == {"gen": 2}
