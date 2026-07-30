"""REAL-Postgres test for the ``marketplace_installs`` schema upgrade path.

``marketplace_installs`` gained the ``contract_version`` / ``skeleton_version``
diagnostics columns after its first deploy. They are declared INSIDE the
``CREATE TABLE IF NOT EXISTS marketplace_installs`` block, which is a no-op on any
database where the table already exists (every real deployment) — so on its own
that block never adds the columns to an existing install, and the store's
SELECT / INSERT (which name them) would fail with ``UndefinedColumn``. The shipped
DDL therefore also carries two idempotent
``ALTER TABLE marketplace_installs ADD COLUMN IF NOT EXISTS ...`` statements.

This test proves that upgrade on real Postgres — there is no fake for DDL column
evolution, so it needs a live server. It plants the OLD table shape (the CREATE
WITHOUT the two columns), runs the FULL shipped DDL, and asserts the columns now
exist and the store's ``list_installed`` SELECT succeeds against a pre-existing
row (its new columns reading back NULL).

OPT-IN: set ``TAI42_SKELETON_REAL_PG=1`` and point ``MARKETPLACE_STORE_*`` at a
live Postgres. Without the opt-in the test SKIPS VISIBLY with a clear reason
(never a silent skip).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import LiteralString, cast

import pytest
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.postgres import PostgresClient
from tai42_kit.settings import reset_all_settings

from tai42_skeleton.marketplace.settings import marketplace_store_settings
from tai42_skeleton.marketplace.store import MarketplaceInstallStore
from tai42_skeleton.sql.schema import load_ddl

pytestmark = pytest.mark.integration

_OPT_IN_ENV = "TAI42_SKELETON_REAL_PG"

# The pre-evolution shape: the CREATE without contract_version / skeleton_version.
# Planting this and then running the shipped DDL reproduces an existing deployment
# being upgraded in place.
_OLD_MARKETPLACE_INSTALLS_DDL = """
CREATE TABLE marketplace_installs (
    ref            TEXT         NOT NULL,
    version        TEXT         NOT NULL,
    source         TEXT         NOT NULL CHECK (source IN ('pypi', 'github')),
    repository_url TEXT,
    tag            TEXT,
    artifact_ref   TEXT,
    sha256         TEXT,
    spec           JSONB        NOT NULL,
    installed_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (ref)
);
"""


async def _exec(sql: LiteralString, params: tuple | None = None) -> None:
    # ``params=None`` (not an empty tuple) so psycopg skips placeholder parsing —
    # the full DDL carries literal ``%`` in the trigger RAISE messages, which an
    # empty-tuple call would misread as an incomplete placeholder (mirrors the
    # production apply path in ``cli/native/db.py``, which passes no params).
    async with (
        client_ctx(PostgresClient, marketplace_store_settings()) as pool,
        pool.connection() as conn,
    ):
        await conn.execute(sql, params)


async def _fetchval(sql: LiteralString, params: tuple | None = None) -> object:
    async with (
        client_ctx(PostgresClient, marketplace_store_settings()) as pool,
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        await cur.execute(sql, params)
        row = await cur.fetchone()
    return None if row is None else row[0]


async def _column_exists(column: str) -> bool:
    found = await _fetchval(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'marketplace_installs' AND column_name = %s",
        (column,),
    )
    return found is not None


@pytest.fixture
async def upgraded_db() -> AsyncIterator[str]:
    if os.environ.get(_OPT_IN_ENV) not in ("1", "true", "True"):
        pytest.skip(
            f"real-Postgres marketplace upgrade-path test is opt-in: set {_OPT_IN_ENV}=1 and point the "
            "MARKETPLACE_STORE_* env at a live Postgres to run it (DDL column evolution has no fake)"
        )
    # Rebuild cached settings so ``MARKETPLACE_STORE_*`` from the environment is read.
    reset_all_settings()
    tag = uuid.uuid4().hex
    # Plant the OLD table shape: drop any existing table, then create it WITHOUT the
    # two columns, so running the shipped DDL exercises the in-place upgrade.
    await _exec("DROP TABLE IF EXISTS marketplace_installs")
    await _exec(cast(LiteralString, _OLD_MARKETPLACE_INSTALLS_DDL))
    try:
        yield tag
    finally:
        # Leave the table in its shipped shape (the DDL under test already produced
        # it); just drop the row this test planted.
        await _exec("DELETE FROM marketplace_installs WHERE ref LIKE %s", (f"%{tag}%",))


async def test_full_ddl_adds_missing_columns_to_existing_table(upgraded_db) -> None:
    """Old table present, both columns absent → run the full shipped DDL → both
    columns exist. The CREATE TABLE IF NOT EXISTS is a no-op here; the two
    ADD COLUMN IF NOT EXISTS statements do the work."""
    assert await _column_exists("contract_version") is False
    assert await _column_exists("skeleton_version") is False

    await _exec(cast(LiteralString, load_ddl()))

    assert await _column_exists("contract_version") is True
    assert await _column_exists("skeleton_version") is True


async def test_list_installed_select_succeeds_after_upgrade(upgraded_db) -> None:
    """The store's ``list_installed`` SELECT names the new columns; it must succeed
    against a row written BEFORE the upgrade, reading the added columns back as
    NULL — proving the UndefinedColumn failure is gone."""
    tag = upgraded_db
    ref = f"tai42/{tag}"
    # A pre-upgrade row: inserted through the OLD column set (no version stamps).
    await _exec(
        cast(
            LiteralString,
            "INSERT INTO marketplace_installs (ref, version, source, spec) VALUES (%s, %s, %s, %s)",
        ),
        (ref, "1.0.0", "pypi", "{}"),
    )

    await _exec(cast(LiteralString, load_ddl()))

    records = await MarketplaceInstallStore().list_installed()
    row = next(r for r in records if r.ref == ref)
    assert row.version == "1.0.0"
    assert row.source == "pypi"
    # The columns the upgrade added read back NULL for a pre-upgrade row.
    assert row.contract_version is None
    assert row.skeleton_version is None


async def test_full_ddl_is_idempotent_on_already_upgraded_table(upgraded_db) -> None:
    """Running the shipped DDL twice is a no-op the second time — ADD COLUMN IF NOT
    EXISTS skips the present columns rather than erroring."""
    await _exec(cast(LiteralString, load_ddl()))
    await _exec(cast(LiteralString, load_ddl()))
    assert await _column_exists("contract_version") is True
    assert await _column_exists("skeleton_version") is True
