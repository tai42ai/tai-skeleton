"""The concrete :class:`~tai42_contract.tool_meta.ToolMetaStore` — two plain
Postgres tables behind the folder + overlay contract.

``tool_folders`` is a tree of real folder entities; ``tool_meta`` is a per-tool
overlay row keyed by tool name. Postgres is reached through the app-pooled
``PostgresClient`` (module-level ``client_ctx`` import so tests can monkeypatch the
seam), sharing one pool per DSN with the other durable stores. Each mutation runs
inside one ``conn.transaction()`` so a multi-statement change (the clean-slate
delete-then-re-key on :meth:`rename_tool`, the delete-dest before an upsert claim)
commits or rolls back as a unit.

Invariants the tables cannot express are enforced here:

* **cycle-freedom** — a folder may never become its own ancestor. Move/create
  walks the parent chain and raises :class:`FolderCycleError` on a loop.
* **empty-folder delete** — a folder is deletable only when it holds no subfolders
  AND no overlay rows, else :class:`FolderNotEmptyError`.
* **clean-slate name reclaim** — a dangling overlay row survives a plugin uninstall
  and could collide with a name a DIFFERENT tool later claims; :meth:`rename_tool`
  deletes any pre-existing destination row before re-keying, so a freed name never
  inherits a ghost's overlay.

Idempotency exception to the raise-loudly rule: :meth:`delete_meta` and
:meth:`rename_tool` are NO-OPS when the tool owns no row — both run on EVERY preset
delete/rename and most tools never own a row. Every other failure (folder absent,
not empty, cycle, sibling-name conflict) raises loudly.
"""

from __future__ import annotations

from typing import Any

from psycopg.errors import UniqueViolation
from tai42_contract.tool_meta import (
    FolderCycleError,
    FolderNameConflictError,
    FolderNotEmptyError,
    FolderNotFoundError,
    FolderRecord,
    ToolMetaRecord,
    ToolMetaStore,
)
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.postgres import PostgresClient

from tai42_skeleton.tool_meta.settings import tool_meta_store_settings


def _folder_record(row: tuple[Any, ...]) -> FolderRecord:
    folder_id, name, parent_id = row
    return FolderRecord(id=str(folder_id), name=name, parent_id=None if parent_id is None else str(parent_id))


def _meta_record(row: tuple[Any, ...]) -> ToolMetaRecord:
    tool_name, display_name, folder_id, tags, hidden = row
    return ToolMetaRecord(
        tool_name=tool_name,
        display_name=display_name,
        folder_id=None if folder_id is None else str(folder_id),
        tags=list(tags or []),
        hidden=hidden,
    )


class PostgresToolMetaStore(ToolMetaStore):
    """Postgres implementation of the tool-metadata overlay store."""

    # -- folders --------------------------------------------------------------

    async def create_folder(self, name: str, parent_id: str | None = None) -> FolderRecord:
        async with (
            client_ctx(PostgresClient, tool_meta_store_settings()) as pool,
            pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            if parent_id is not None:
                await self._require_folder(cur, parent_id)
            try:
                await cur.execute(
                    "INSERT INTO tool_folders (name, parent_id) VALUES (%s, %s) RETURNING id, name, parent_id",
                    (name, parent_id),
                )
            except UniqueViolation as exc:
                raise FolderNameConflictError(name, parent_id) from exc
            return _folder_record(_require_row(await cur.fetchone()))

    async def rename_folder(self, folder_id: str, name: str) -> FolderRecord:
        async with (
            client_ctx(PostgresClient, tool_meta_store_settings()) as pool,
            pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            _, parent_id = await self._require_folder(cur, folder_id)
            try:
                await cur.execute(
                    "UPDATE tool_folders SET name = %s WHERE id = %s RETURNING id, name, parent_id",
                    (name, folder_id),
                )
            except UniqueViolation as exc:
                raise FolderNameConflictError(name, parent_id) from exc
            return _folder_record(_require_row(await cur.fetchone()))

    async def move_folder(self, folder_id: str, parent_id: str | None) -> FolderRecord:
        async with (
            client_ctx(PostgresClient, tool_meta_store_settings()) as pool,
            pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            # Read the moving folder's name up front: on a destination-name clash the
            # UPDATE below aborts the transaction, so the conflict must be raised from
            # values already in hand — a SELECT after the failed UPDATE would itself
            # raise ``InFailedSqlTransaction`` and mask the real 409.
            name, _ = await self._require_folder(cur, folder_id)
            if parent_id is not None:
                await self._require_folder(cur, parent_id)
                await self._reject_cycle(cur, folder_id, parent_id)
            try:
                await cur.execute(
                    "UPDATE tool_folders SET parent_id = %s WHERE id = %s RETURNING id, name, parent_id",
                    (parent_id, folder_id),
                )
            except UniqueViolation as exc:
                # A destination sibling already holds this folder's name.
                raise FolderNameConflictError(name, parent_id) from exc
            return _folder_record(_require_row(await cur.fetchone()))

    async def delete_folder(self, folder_id: str) -> None:
        async with (
            client_ctx(PostgresClient, tool_meta_store_settings()) as pool,
            pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await self._require_folder(cur, folder_id)
            await cur.execute("SELECT 1 FROM tool_folders WHERE parent_id = %s LIMIT 1", (folder_id,))
            if await cur.fetchone() is not None:
                raise FolderNotEmptyError(folder_id)
            await cur.execute("SELECT 1 FROM tool_meta WHERE folder_id = %s LIMIT 1", (folder_id,))
            if await cur.fetchone() is not None:
                raise FolderNotEmptyError(folder_id)
            await cur.execute("DELETE FROM tool_folders WHERE id = %s", (folder_id,))

    async def list_folders(self) -> list[FolderRecord]:
        async with (
            client_ctx(PostgresClient, tool_meta_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT id, name, parent_id FROM tool_folders ORDER BY name")
            return [_folder_record(row) for row in await cur.fetchall()]

    # -- overlay --------------------------------------------------------------

    async def upsert_meta(
        self,
        tool_name: str,
        *,
        display_name: str | None,
        folder_id: str | None,
        tags: list[str],
        hidden: bool | None,
    ) -> ToolMetaRecord:
        async with (
            client_ctx(PostgresClient, tool_meta_store_settings()) as pool,
            pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            if folder_id is not None:
                await self._require_folder(cur, folder_id)
            # ``created_at`` is stamped on first insert and preserved on conflict — an
            # upsert re-writes only the mutable overlay columns.
            await cur.execute(
                "INSERT INTO tool_meta (tool_name, display_name, folder_id, tags, hidden) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (tool_name) DO UPDATE SET "
                "display_name = EXCLUDED.display_name, folder_id = EXCLUDED.folder_id, "
                "tags = EXCLUDED.tags, hidden = EXCLUDED.hidden "
                "RETURNING tool_name, display_name, folder_id, tags, hidden",
                (tool_name, display_name, folder_id, list(tags), hidden),
            )
            return _meta_record(_require_row(await cur.fetchone()))

    async def merge_meta(self, tool_name: str, *, patch: dict[str, Any]) -> ToolMetaRecord:
        # Atomic read-modify-write: the whole merge-patch runs in ONE transaction so
        # two concurrent patches to the same tool serialize on a row lock instead of
        # racing on separate connections (a get-then-upsert split loses an update).
        # ``INSERT ... ON CONFLICT DO NOTHING`` first materializes the row so there is
        # always something to lock — a concurrent creator either commits before us (we
        # then read and merge on top of its row) or blocks until we commit — and
        # ``SELECT ... FOR UPDATE`` then holds that row for the merge.
        async with (
            client_ctx(PostgresClient, tool_meta_store_settings()) as pool,
            pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await cur.execute(
                "INSERT INTO tool_meta (tool_name) VALUES (%s) ON CONFLICT (tool_name) DO NOTHING",
                (tool_name,),
            )
            await cur.execute(
                "SELECT display_name, folder_id, tags, hidden FROM tool_meta WHERE tool_name = %s FOR UPDATE",
                (tool_name,),
            )
            current = _require_row(await cur.fetchone())
            display_name = current[0]
            folder_id = None if current[1] is None else str(current[1])
            tags = list(current[2] or [])
            hidden = current[3]

            # Apply only the fields the caller sent; a present value writes (a present
            # ``None`` clears), ``tags`` replaces the whole set. ``display_name`` is
            # already normalized upstream (blank-refusal is the operation's job).
            if "display_name" in patch:
                display_name = patch["display_name"]
            if "folder_id" in patch:
                folder_id = patch["folder_id"]
            if "tags" in patch:
                tags = list(patch["tags"])
            if "hidden" in patch:
                hidden = patch["hidden"]

            if folder_id is not None:
                await self._require_folder(cur, folder_id)
            await cur.execute(
                "UPDATE tool_meta SET display_name = %s, folder_id = %s, tags = %s, hidden = %s "
                "WHERE tool_name = %s "
                "RETURNING tool_name, display_name, folder_id, tags, hidden",
                (display_name, folder_id, list(tags), hidden, tool_name),
            )
            return _meta_record(_require_row(await cur.fetchone()))

    async def get_meta(self, tool_name: str) -> ToolMetaRecord | None:
        async with (
            client_ctx(PostgresClient, tool_meta_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT tool_name, display_name, folder_id, tags, hidden FROM tool_meta WHERE tool_name = %s",
                (tool_name,),
            )
            row = await cur.fetchone()
        return None if row is None else _meta_record(row)

    async def list_meta(self) -> list[ToolMetaRecord]:
        async with (
            client_ctx(PostgresClient, tool_meta_store_settings()) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT tool_name, display_name, folder_id, tags, hidden FROM tool_meta ORDER BY tool_name"
            )
            return [_meta_record(row) for row in await cur.fetchall()]

    async def delete_meta(self, tool_name: str) -> None:
        # NO-OP when no row exists (most tools never own one, and this runs on every
        # preset delete) — a missing row is not an error here.
        async with (
            client_ctx(PostgresClient, tool_meta_store_settings()) as pool,
            pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await cur.execute("DELETE FROM tool_meta WHERE tool_name = %s", (tool_name,))

    async def rename_tool(self, old_name: str, new_name: str) -> None:
        # Clean slate: delete any pre-existing destination row FIRST (a freed name must
        # never inherit a ghost's overlay), then re-key — both in one transaction so
        # the re-key can never hit a PK violation. When ``old_name`` owns no row the
        # re-key moves nothing, but the destination ghost is still cleared.
        async with (
            client_ctx(PostgresClient, tool_meta_store_settings()) as pool,
            pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await cur.execute("DELETE FROM tool_meta WHERE tool_name = %s", (new_name,))
            await cur.execute("UPDATE tool_meta SET tool_name = %s WHERE tool_name = %s", (new_name, old_name))

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    async def _require_folder(cur: Any, folder_id: str) -> tuple[str, str | None]:
        """Assert ``folder_id`` exists, returning its ``(name, parent_id)``; raise
        :class:`FolderNotFoundError` otherwise. The name is returned so a caller can
        build a :class:`FolderNameConflictError` from values in hand, without a query
        in an already-aborted transaction."""
        await cur.execute("SELECT name, parent_id FROM tool_folders WHERE id = %s", (folder_id,))
        row = await cur.fetchone()
        if row is None:
            raise FolderNotFoundError(folder_id)
        return row[0], (None if row[1] is None else str(row[1]))

    @staticmethod
    async def _reject_cycle(cur: Any, folder_id: str, new_parent_id: str) -> None:
        """Refuse a move that would make ``folder_id`` its own ancestor. Walks the
        parent chain up from ``new_parent_id``: reaching ``folder_id`` is a cycle."""
        await cur.execute("SELECT id, parent_id FROM tool_folders")
        parents = {str(fid): (None if pid is None else str(pid)) for fid, pid in await cur.fetchall()}
        cursor: str | None = new_parent_id
        while cursor is not None:
            if cursor == folder_id:
                raise FolderCycleError(folder_id)
            cursor = parents.get(cursor)


def _require_row(row: tuple[Any, ...] | None) -> tuple[Any, ...]:
    """Return ``row`` or fail loudly — a ``RETURNING`` read the store issued must
    always produce a row; a ``None`` is a broken invariant, never a normal path."""
    if row is None:
        raise RuntimeError("expected a row from a RETURNING statement, got none")
    return row


def tool_meta_store() -> PostgresToolMetaStore:
    """Build the tool-metadata overlay store over its Postgres tables."""
    return PostgresToolMetaStore()
