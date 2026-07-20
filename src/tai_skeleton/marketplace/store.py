"""Postgres-backed attribution store for marketplace-installed plugins.

One row per installed listing in ``marketplace_installs``: the listing ``ref``
(``namespace/name``), the exact installed ``version``, the artifact ``source``
(``pypi``/``github``), the github ``repository_url`` + ``tag`` the pin came from
(both ``None`` for a pypi source), the github ``artifact_ref`` + ``sha256`` the
verified install fetched from (both ``None`` for a pypi source, whose immutable
``package==version`` needs no fetch), and the full ``PluginSpec`` document that
version shipped (``spec``, JSONB). Storing the spec locally is what lets
uninstall unpatch the manifest without asking the registry, and lets
``installed``/``update``/``advisories`` answer from local truth; storing the
github artifact_ref + sha256 is what lets update-unwind reinstall the old pin
through the same download-and-verify path the forward install took.

Written only by the installer: :meth:`~MarketplaceInstallStore.record` serves
both install and update (the update flow replaces the row), :meth:`delete`
drops it on uninstall. There is no cache layer — the table is read on operator
requests and the advisory-poll cadence, never on a hot per-request path.

Postgres is reached through the app-pooled ``PostgresClient`` (module-level
``client_ctx`` import so tests can monkeypatch the seam), sharing one pool per
DSN with the other durable stores and closed centrally at shutdown.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.postgres import Json, PostgresClient

from tai_skeleton.marketplace.settings import marketplace_store_settings


class InstallRecord(BaseModel):
    """One installed-plugin attribution row.

    ``spec`` is the full ``PluginSpec`` document (as a plain dict) the installed
    version shipped; ``repository_url``, ``tag``, ``artifact_ref``, and ``sha256``
    are set only for a github source. Frozen: a record read from the store is an
    immutable snapshot.
    """

    model_config = ConfigDict(frozen=True)

    ref: str
    version: str
    source: str
    repository_url: str | None = None
    tag: str | None = None
    artifact_ref: str | None = None
    sha256: str | None = None
    spec: dict[str, Any]
    installed_at: datetime


class MarketplaceInstallStore:
    """CRUD over the ``marketplace_installs`` attribution table."""

    async def record(
        self,
        ref: str,
        version: str,
        source: str,
        repository_url: str | None,
        tag: str | None,
        artifact_ref: str | None,
        sha256: str | None,
        spec: dict[str, Any],
    ) -> None:
        """Insert the install row, or replace it when one already exists.

        Serves install (no prior row) and update (the update flow replaces the
        row's version/source/pin/spec and re-stamps ``installed_at``). The github
        ``artifact_ref`` + ``sha256`` are persisted so update-unwind can reinstall
        the old pin through the same verified fetch; both are ``None`` for pypi.
        """
        async with (
            client_ctx(PostgresClient, marketplace_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "INSERT INTO marketplace_installs "
                "(ref, version, source, repository_url, tag, artifact_ref, sha256, spec) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (ref) DO UPDATE SET "
                "version = EXCLUDED.version, source = EXCLUDED.source, "
                "repository_url = EXCLUDED.repository_url, tag = EXCLUDED.tag, "
                "artifact_ref = EXCLUDED.artifact_ref, sha256 = EXCLUDED.sha256, "
                "spec = EXCLUDED.spec, installed_at = now()",
                (ref, version, source, repository_url, tag, artifact_ref, sha256, Json(spec)),
            )

    async def get(self, ref: str) -> InstallRecord | None:
        """The install record for ``ref``, or ``None`` when it is not installed."""
        async with (
            client_ctx(PostgresClient, marketplace_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT ref, version, source, repository_url, tag, artifact_ref, sha256, spec, installed_at "
                "FROM marketplace_installs WHERE ref = %s",
                (ref,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    async def delete(self, ref: str) -> bool:
        """Drop the install row; ``True`` when a row existed, ``False`` otherwise."""
        async with (
            client_ctx(PostgresClient, marketplace_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("DELETE FROM marketplace_installs WHERE ref = %s", (ref,))
            return cur.rowcount > 0

    async def list_installed(self) -> list[InstallRecord]:
        """Every install record, ordered by ``ref``."""
        async with (
            client_ctx(PostgresClient, marketplace_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT ref, version, source, repository_url, tag, artifact_ref, sha256, spec, installed_at "
                "FROM marketplace_installs ORDER BY ref"
            )
            rows = await cur.fetchall()
        return [_row_to_record(row) for row in rows]


def _row_to_record(row: tuple[Any, ...]) -> InstallRecord:
    ref, version, source, repository_url, tag, artifact_ref, sha256, spec, installed_at = row
    return InstallRecord(
        ref=ref,
        version=version,
        source=source,
        repository_url=repository_url,
        tag=tag,
        artifact_ref=artifact_ref,
        sha256=sha256,
        spec=spec,
        installed_at=installed_at,
    )
