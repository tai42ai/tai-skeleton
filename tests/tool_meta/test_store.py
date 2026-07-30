"""The tool-metadata overlay store over an in-memory fake Postgres.

Every case drives the REAL :class:`PostgresToolMetaStore` against
``FakeToolMetaPg`` (the ``pg``/``store`` fixtures in ``conftest``), so the store's
own invariants — sibling-name uniqueness, cycle-freedom, empty-folder deletes,
clean-slate tool rename, the merge-patch-agnostic full-row upsert — run through
their true SQL with no live database.
"""

from __future__ import annotations

import pytest
from tai42_contract.tool_meta import (
    FolderCycleError,
    FolderNameConflictError,
    FolderNotEmptyError,
    FolderNotFoundError,
)

from tai42_skeleton.tool_meta.store import PostgresToolMetaStore
from tests.tool_meta.conftest import FakeToolMetaPg

# -- folders: create / rename / move / delete / list -------------------------


async def test_create_root_and_nested_folder(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    root = await store.create_folder("root")
    assert root.parent_id is None
    child = await store.create_folder("child", root.id)
    assert child.parent_id == root.id
    # Both landed in the table.
    assert {f["name"] for f in pg.folders.values()} == {"root", "child"}


async def test_list_folders_ordered_by_name(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    await store.create_folder("zeta")
    await store.create_folder("alpha")
    await store.create_folder("mid")
    assert [f.name for f in await store.list_folders()] == ["alpha", "mid", "zeta"]


async def test_rename_folder_in_place(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    folder = await store.create_folder("old")
    renamed = await store.rename_folder(folder.id, "new")
    assert (renamed.id, renamed.name) == (folder.id, "new")
    assert pg.folders[folder.id]["name"] == "new"


async def test_move_folder_reparents(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    a = await store.create_folder("a")
    b = await store.create_folder("b")
    moved = await store.move_folder(b.id, a.id)
    assert moved.parent_id == a.id
    # Move back to root.
    rooted = await store.move_folder(b.id, None)
    assert rooted.parent_id is None


async def test_delete_folder_when_empty(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    folder = await store.create_folder("gone")
    await store.delete_folder(folder.id)
    assert folder.id not in pg.folders


# -- folder cycle prevention -------------------------------------------------


async def test_move_under_own_child_raises_cycle(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    parent = await store.create_folder("parent")
    child = await store.create_folder("child", parent.id)
    with pytest.raises(FolderCycleError):
        await store.move_folder(parent.id, child.id)


async def test_move_under_own_grandchild_raises_cycle(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    a = await store.create_folder("a")
    b = await store.create_folder("b", a.id)
    c = await store.create_folder("c", b.id)
    # a is c's ancestor; re-parenting a under c would close the loop.
    with pytest.raises(FolderCycleError):
        await store.move_folder(a.id, c.id)


async def test_move_folder_under_itself_raises_cycle(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    folder = await store.create_folder("self")
    with pytest.raises(FolderCycleError):
        await store.move_folder(folder.id, folder.id)


# -- folder empty-delete refusal ---------------------------------------------


async def test_delete_folder_with_subfolder_refused(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    parent = await store.create_folder("parent")
    await store.create_folder("child", parent.id)
    with pytest.raises(FolderNotEmptyError):
        await store.delete_folder(parent.id)
    assert parent.id in pg.folders  # not deleted


async def test_delete_folder_holding_overlay_row_refused(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    folder = await store.create_folder("filed")
    await store.upsert_meta("weather", display_name=None, folder_id=folder.id, tags=[], hidden=None)
    with pytest.raises(FolderNotEmptyError):
        await store.delete_folder(folder.id)


# -- sibling-name uniqueness -------------------------------------------------


async def test_create_duplicate_sibling_name_conflicts(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    root = await store.create_folder("root")
    await store.create_folder("dup", root.id)
    with pytest.raises(FolderNameConflictError):
        await store.create_folder("dup", root.id)


async def test_duplicate_root_names_conflict_nulls_not_distinct(
    pg: FakeToolMetaPg, store: PostgresToolMetaStore
) -> None:
    # Two roots share the NULL parent — NULLS-NOT-DISTINCT makes a duplicate root
    # name collide, exactly as the table constraint enforces.
    await store.create_folder("top")
    with pytest.raises(FolderNameConflictError):
        await store.create_folder("top")


async def test_rename_to_duplicate_sibling_conflicts(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    root = await store.create_folder("root")
    await store.create_folder("taken", root.id)
    other = await store.create_folder("free", root.id)
    with pytest.raises(FolderNameConflictError):
        await store.rename_folder(other.id, "taken")


async def test_move_into_duplicate_sibling_conflicts(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    a = await store.create_folder("a")
    b = await store.create_folder("b")
    await store.create_folder("shared", a.id)
    clash = await store.create_folder("shared", b.id)
    # Moving b's "shared" under a collides with a's existing "shared".
    with pytest.raises(FolderNameConflictError):
        await store.move_folder(clash.id, a.id)


# -- unknown folder / parent lookups -----------------------------------------


async def test_rename_unknown_folder_not_found(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    with pytest.raises(FolderNotFoundError):
        await store.rename_folder("no-such-id", "x")


async def test_move_unknown_folder_not_found(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    with pytest.raises(FolderNotFoundError):
        await store.move_folder("no-such-id", None)


async def test_delete_unknown_folder_not_found(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    with pytest.raises(FolderNotFoundError):
        await store.delete_folder("no-such-id")


async def test_create_under_unknown_parent_not_found(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    with pytest.raises(FolderNotFoundError):
        await store.create_folder("orphan", "no-such-parent")


async def test_move_to_unknown_parent_not_found(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    folder = await store.create_folder("f")
    with pytest.raises(FolderNotFoundError):
        await store.move_folder(folder.id, "no-such-parent")


# -- overlay upsert / get / list ---------------------------------------------


async def test_upsert_inserts_then_updates_on_conflict(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    first = await store.upsert_meta("weather", display_name="Weather", folder_id=None, tags=["a"], hidden=None)
    assert (first.tool_name, first.display_name, first.tags) == ("weather", "Weather", ["a"])
    second = await store.upsert_meta("weather", display_name="Forecast", folder_id=None, tags=["b"], hidden=True)
    assert second.display_name == "Forecast"
    assert second.tags == ["b"]
    assert second.hidden is True
    assert len(pg.meta) == 1  # replaced, not duplicated


async def test_get_meta_hit_and_miss(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    assert await store.get_meta("weather") is None
    await store.upsert_meta("weather", display_name="W", folder_id=None, tags=[], hidden=None)
    got = await store.get_meta("weather")
    assert got is not None
    assert got.tool_name == "weather"


async def test_list_meta_ordered_by_tool_name(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    await store.upsert_meta("zebra", display_name=None, folder_id=None, tags=[], hidden=None)
    await store.upsert_meta("apple", display_name=None, folder_id=None, tags=[], hidden=None)
    assert [r.tool_name for r in await store.list_meta()] == ["apple", "zebra"]


@pytest.mark.parametrize("hidden", [None, True, False])
async def test_hidden_tri_state_round_trips(
    pg: FakeToolMetaPg, store: PostgresToolMetaStore, hidden: bool | None
) -> None:
    await store.upsert_meta("t", display_name=None, folder_id=None, tags=[], hidden=hidden)
    got = await store.get_meta("t")
    assert got is not None
    assert got.hidden is hidden


async def test_tags_round_trip(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    await store.upsert_meta("t", display_name=None, folder_id=None, tags=["x", "y", "z"], hidden=None)
    got = await store.get_meta("t")
    assert got is not None
    assert got.tags == ["x", "y", "z"]


async def test_upsert_places_row_in_folder(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    folder = await store.create_folder("bucket")
    row = await store.upsert_meta("t", display_name=None, folder_id=folder.id, tags=[], hidden=None)
    assert row.folder_id == folder.id


async def test_upsert_unknown_folder_not_found(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    with pytest.raises(FolderNotFoundError):
        await store.upsert_meta("t", display_name=None, folder_id="no-such-folder", tags=[], hidden=None)
    assert "t" not in pg.meta  # nothing written


# -- overlay delete / rename -------------------------------------------------


async def test_delete_meta_removes_row(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    await store.upsert_meta("t", display_name=None, folder_id=None, tags=[], hidden=None)
    await store.delete_meta("t")
    assert await store.get_meta("t") is None


async def test_delete_meta_missing_row_is_noop(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    # No row, no error — this runs on every preset delete.
    await store.delete_meta("never-had-one")
    assert pg.meta == {}


async def test_rename_tool_rekeys_row(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    await store.upsert_meta("old", display_name="Label", folder_id=None, tags=["k"], hidden=True)
    await store.rename_tool("old", "new")
    assert await store.get_meta("old") is None
    moved = await store.get_meta("new")
    assert moved is not None
    assert (moved.display_name, moved.tags, moved.hidden) == ("Label", ["k"], True)


async def test_rename_tool_clean_slate_deletes_existing_destination(
    pg: FakeToolMetaPg, store: PostgresToolMetaStore
) -> None:
    await store.upsert_meta("old", display_name="Old", folder_id=None, tags=["from-old"], hidden=None)
    await store.upsert_meta("new", display_name="Ghost", folder_id=None, tags=["ghost"], hidden=True)
    await store.rename_tool("old", "new")
    # The old row's values win under ``new``; the pre-existing ghost was deleted first.
    dest = await store.get_meta("new")
    assert dest is not None
    assert (dest.display_name, dest.tags, dest.hidden) == ("Old", ["from-old"], None)
    assert len(pg.meta) == 1


async def test_rename_tool_missing_source_is_noop(pg: FakeToolMetaPg, store: PostgresToolMetaStore) -> None:
    # Neither name owns a row — a pure no-op (this runs on every preset rename and
    # most tools never own an overlay row); it must simply not error.
    await store.rename_tool("old", "new")
    assert await store.get_meta("old") is None
    assert await store.get_meta("new") is None
    assert pg.meta == {}
