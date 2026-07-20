"""Generic ``PostgresVersionedStore`` semantics over the fake Postgres.

Exercises the store as a body-opaque, ``(kind, name)``-identified primitive:
cross-kind isolation, the append-only ``MAX(version)+1`` numbering (including the
post-rollback case), rollback as a pointer re-point with no data copy,
transactional partial-failure rollback, the partial-unique active index (both
directions), and the hard-delete scoping that spares a soft-deleted ghost.
"""

from __future__ import annotations

import pytest
from tai_contract.versioning.errors import (
    DocumentExistsError,
    DocumentNotFoundError,
    DocumentVersionNotFoundError,
)

from tai_skeleton.versioning.store import PostgresVersionedStore


@pytest.fixture
def store() -> PostgresVersionedStore:
    return PostgresVersionedStore()


async def test_create_get_round_trip(pg, store):
    rec = await store.create("preset", "alpha", {"x": 1}, tags=["rel"])
    assert (rec.kind, rec.name, rec.active_version, rec.is_active) == ("preset", "alpha", 1, True)
    got = await store.get("preset", "alpha")
    assert got.active_version == 1
    assert await store.get_active_body("preset", "alpha") == {"x": 1}
    versions = await store.list_versions("preset", "alpha")
    assert [v.version for v in versions] == [1]
    assert versions[0].tags == ["rel"]
    assert versions[0].is_current is True


async def test_kind_name_isolation(pg, store):
    # A preset and a policy of the SAME name coexist — identity is (kind, name).
    await store.create("preset", "shared", {"who": "preset"})
    await store.create("ac_policy", "shared", {"who": "policy"})
    assert await store.get_active_body("preset", "shared") == {"who": "preset"}
    assert await store.get_active_body("ac_policy", "shared") == {"who": "policy"}
    assert [r.name for r in await store.list("preset")] == ["shared"]
    assert [r.name for r in await store.list("ac_policy")] == ["shared"]
    # A save to one kind never touches the other.
    await store.save_version("preset", "shared", {"who": "preset2"})
    assert await store.get_active_body("ac_policy", "shared") == {"who": "policy"}


async def test_duplicate_active_create_raises(pg, store):
    await store.create("preset", "dup", {"a": 1})
    with pytest.raises(DocumentExistsError) as exc:
        await store.create("preset", "dup", {"a": 2})
    assert (exc.value.kind, exc.value.name) == ("preset", "dup")


async def test_non_active_name_unique_violation_reraises(pg, store):
    # A unique violation that is NOT the active-name index must propagate RAW —
    # never be mis-mapped to DocumentExistsError (which is reserved for the live
    # duplicate the partial-unique index catches).
    from psycopg.errors import UniqueViolation

    pg.fault = ("INSERT INTO versioned_documents", UniqueViolation("some other constraint"))
    with pytest.raises(UniqueViolation):
        await store.create("preset", "other", {"a": 1})
    # The transaction rolled back — nothing persisted.
    assert pg.documents == []


async def test_partial_unique_index_both_directions(pg, store):
    # create -> soft_delete -> create the SAME (kind, name) SUCCEEDS: the partial
    # unique index only constrains ACTIVE rows, so the ghost does not block it.
    await store.create("preset", "recycle", {"v": 1})
    await store.soft_delete("preset", "recycle")
    rec = await store.create("preset", "recycle", {"v": 2})
    assert rec.active_version == 1
    assert await store.get_active_body("preset", "recycle") == {"v": 2}
    # The soft-deleted ghost's version history is untouched by the recreate.
    assert len([d for d in pg.documents if d["name"] == "recycle"]) == 2


async def test_save_version_monotonic(pg, store):
    await store.create("preset", "seq", {"n": 1})
    v2 = await store.save_version("preset", "seq", {"n": 2})
    v3 = await store.save_version("preset", "seq", {"n": 3})
    assert (v2.version, v3.version) == (2, 3)
    assert (await store.get("preset", "seq")).active_version == 3
    assert v3.is_current is True


async def test_rollback_repoints_without_copy(pg, store):
    await store.create("preset", "roll", {"n": 1})
    await store.save_version("preset", "roll", {"n": 2})
    await store.save_version("preset", "roll", {"n": 3})
    before = pg.versions.copy()
    rec = await store.rollback("preset", "roll", 1)
    assert rec.active_version == 1
    assert await store.get_active_body("preset", "roll") == {"n": 1}
    # No data copy: the version rows are byte-for-byte unchanged.
    assert pg.versions == before
    # is_current tracks the re-pointed active version.
    current = [v.version for v in await store.list_versions("preset", "roll") if v.is_current]
    assert current == [1]


async def test_save_version_numbering_is_max_plus_one_post_rollback(pg, store):
    await store.create("preset", "mp", {"n": 1})
    for n in range(2, 6):  # -> versions 2,3,4,5
        await store.save_version("preset", "mp", {"n": n})
    await store.rollback("preset", "mp", 2)
    assert (await store.get("preset", "mp")).active_version == 2
    new_version = await store.save_version("preset", "mp", {"n": 99})
    # MAX(5)+1, NOT active(2)+1 — no collision at version 3.
    assert new_version.version == 6
    assert (await store.get("preset", "mp")).active_version == 6


async def test_create_partial_failure_rolls_back(pg, store):
    # The version-1 insert fails AFTER the document row insert; the whole create
    # rolls back — no orphan document, no version.
    pg.fault = ("INSERT INTO versioned_document_versions", RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await store.create("preset", "half", {"x": 1})
    assert pg.documents == []
    assert pg.versions == []
    with pytest.raises(DocumentNotFoundError):
        await store.get("preset", "half")


async def test_save_version_locks_active_row_for_update(pg, store):
    # Concurrent version writes serialize on the active document row: the id select
    # in ``save_version`` must take a ``FOR UPDATE`` row lock so a second writer
    # waits, reads the first writer's appended MAX, and numbers MAX+1 rather than
    # colliding on the ``(document_id, version)`` unique constraint. (The in-memory
    # fake serializes all statements, so the guard is pinned by asserting the
    # emitted SQL carries the lock.)
    await store.create("preset", "lock", {"n": 1})
    await store.save_version("preset", "lock", {"n": 2})
    id_selects = [sql for sql in pg.executed if sql.startswith("SELECT id FROM versioned_documents")]
    assert id_selects
    assert all(sql.endswith("AND is_active FOR UPDATE") for sql in id_selects)


async def test_rollback_locks_active_row_for_update(pg, store):
    # A rollback re-points the active version, so it must take the same
    # ``FOR UPDATE`` lock as ``save_version`` to serialize against a concurrent save.
    await store.create("preset", "rlock", {"n": 1})
    await store.save_version("preset", "rlock", {"n": 2})
    await store.rollback("preset", "rlock", 1)
    id_selects = [sql for sql in pg.executed if sql.startswith("SELECT id, created_at FROM versioned_documents")]
    assert id_selects
    assert all(sql.endswith("AND is_active FOR UPDATE") for sql in id_selects)


async def test_save_version_partial_failure_rolls_back(pg, store):
    await store.create("preset", "atomic", {"n": 1})
    versions_before = pg.versions.copy()
    # The pointer bump fails AFTER the new version row is inserted; the whole save
    # rolls back — no orphan version, no half-bumped active_version.
    pg.fault = ("UPDATE versioned_documents SET active_version", RuntimeError("nope"))
    with pytest.raises(RuntimeError, match="nope"):
        await store.save_version("preset", "atomic", {"n": 2})
    assert pg.versions == versions_before
    assert (await store.get("preset", "atomic")).active_version == 1


async def test_list_returns_active_only(pg, store):
    await store.create("preset", "live", {})
    await store.create("preset", "gone", {})
    await store.soft_delete("preset", "gone")
    assert [r.name for r in await store.list("preset")] == ["live"]


async def test_list_active_bodies_batched_read(pg, store):
    # Active bodies of one kind, name-keyed, from a SINGLE JOIN read — the batched
    # replacement for a per-record get_active_body round-trip (the list/rehydrate
    # N+1). Only active rows of the requested kind, at their active version.
    await store.create("preset", "alpha", {"n": 1})
    await store.save_version("preset", "alpha", {"n": 2})  # active = v2
    await store.create("preset", "beta", {"m": 1})
    await store.create("other", "gamma", {"k": 1})  # different kind — excluded
    await store.create("preset", "ghost", {"g": 1})
    await store.soft_delete("preset", "ghost")  # inactive — excluded
    pg.executed.clear()

    bodies = await store.list_active_bodies("preset")
    assert bodies == {"alpha": {"n": 2}, "beta": {"m": 1}}

    # One batched JOIN read, never a per-row get_active_body.
    batched = [s for s in pg.executed if s.startswith("SELECT d.name, v.body FROM versioned_documents d")]
    per_row = [s for s in pg.executed if s.startswith("SELECT v.body FROM versioned_documents d")]
    assert len(batched) == 1
    assert per_row == []


async def test_get_version_and_missing(pg, store):
    await store.create("preset", "gv", {"n": 1})
    await store.save_version("preset", "gv", {"n": 2})
    v1 = await store.get_version("preset", "gv", 1)
    assert v1.body == {"n": 1}
    assert v1.is_current is False
    with pytest.raises(DocumentVersionNotFoundError) as exc:
        await store.get_version("preset", "gv", 99)
    assert exc.value.version == 99


async def test_soft_delete_keeps_history(pg, store):
    await store.create("preset", "sd", {"n": 1})
    await store.save_version("preset", "sd", {"n": 2})
    await store.soft_delete("preset", "sd")
    with pytest.raises(DocumentNotFoundError):
        await store.get("preset", "sd")
    # History rows survive (audit).
    assert len(pg.versions) == 2


async def test_hard_delete_scoping(pg, store):
    # A soft-deleted ghost of the SAME name plus a live active row, then hard
    # delete the active one: the ghost + its history survive; the active row and
    # its versions are gone.
    await store.create("preset", "scope", {"gen": 1})
    await store.save_version("preset", "scope", {"gen": 1.5})
    await store.soft_delete("preset", "scope")  # ghost with 2 versions
    ghost_doc_id = pg.documents[0]["id"]
    await store.create("preset", "scope", {"gen": 2})  # new active row
    active_doc_id = next(d for d in pg.documents if d["is_active"])["id"]

    await store.delete("preset", "scope")

    # The active row and its version rows are gone...
    assert active_doc_id not in {d["id"] for d in pg.documents}
    assert active_doc_id not in {v["document_id"] for v in pg.versions}
    # ...while the soft-deleted ghost and its full history remain intact.
    assert ghost_doc_id in {d["id"] for d in pg.documents}
    assert len([v for v in pg.versions if v["document_id"] == ghost_doc_id]) == 2


async def test_delete_no_active_raises(pg, store):
    await store.create("preset", "onlyghost", {})
    await store.soft_delete("preset", "onlyghost")
    with pytest.raises(DocumentNotFoundError):
        await store.delete("preset", "onlyghost")


async def test_missing_targets_raise_typed_errors(pg, store):
    with pytest.raises(DocumentNotFoundError):
        await store.get("preset", "nope")
    with pytest.raises(DocumentNotFoundError):
        await store.get_active_body("preset", "nope")
    with pytest.raises(DocumentNotFoundError):
        await store.save_version("preset", "nope", {})
    with pytest.raises(DocumentNotFoundError):
        await store.list_versions("preset", "nope")
    with pytest.raises(DocumentNotFoundError):
        await store.soft_delete("preset", "nope")
    with pytest.raises(DocumentVersionNotFoundError):
        await store.rollback("preset", "nope", 1)
    with pytest.raises(DocumentVersionNotFoundError):
        await store.get_version("preset", "nope", 1)


async def test_body_is_opaque(pg, store):
    # An arbitrary nested body round-trips untouched — the store never inspects it.
    body = {"deep": {"list": [1, {"k": "v"}], "flag": True}, "n": None}
    await store.create("misc", "opaque", body)
    assert await store.get_active_body("misc", "opaque") == body


# -- rename ------------------------------------------------------------------


async def test_rename_round_trip(pg, store):
    rec = await store.create("preset", "old", {"x": 1})
    renamed = await store.rename("preset", "old", "new")
    assert (renamed.kind, renamed.name, renamed.active_version, renamed.is_active) == ("preset", "new", 1, True)
    assert renamed.created_at == rec.created_at  # the row moved, its timestamp is preserved
    # Reachable under the new name, gone under the old.
    assert await store.get_active_body("preset", "new") == {"x": 1}
    assert [r.name for r in await store.list("preset")] == ["new"]
    with pytest.raises(DocumentNotFoundError):
        await store.get("preset", "old")


async def test_rename_preserves_history_tags_and_active_pointer(pg, store):
    # Full version history, per-version tags, and a post-rollback trailing active
    # pointer all move untouched — versions key on the document, not the name.
    await store.create("preset", "h", {"n": 1}, tags=["rel"])
    await store.save_version("preset", "h", {"n": 2}, tags=["beta"])
    await store.save_version("preset", "h", {"n": 3}, tags=["rc"])
    await store.rollback("preset", "h", 2)  # active trails MAX
    assert (await store.get("preset", "h")).active_version == 2

    await store.rename("preset", "h", "h2")

    versions = await store.list_versions("preset", "h2")
    assert [(v.version, v.tags, v.body) for v in versions] == [
        (1, ["rel"], {"n": 1}),
        (2, ["beta"], {"n": 2}),
        (3, ["rc"], {"n": 3}),
    ]
    # The active pointer is preserved (still v2, not reset), so is_current tracks it.
    assert [v.version for v in versions if v.is_current] == [2]
    assert (await store.get("preset", "h2")).active_version == 2
    # No version row was copied — the count is unchanged by the rename.
    assert len([v for v in pg.versions if v["document_id"] == pg.documents[0]["id"]]) == 3


async def test_rename_is_cross_kind_isolated(pg, store):
    # A same-named document of ANOTHER kind is untouched by the rename.
    await store.create("preset", "shared", {"who": "preset"})
    await store.create("ac_policy", "shared", {"who": "policy"})
    await store.rename("preset", "shared", "moved")
    assert await store.get_active_body("preset", "moved") == {"who": "preset"}
    assert await store.get_active_body("ac_policy", "shared") == {"who": "policy"}
    with pytest.raises(DocumentNotFoundError):
        await store.get("preset", "shared")


async def test_rename_absent_raises_not_found(pg, store):
    with pytest.raises(DocumentNotFoundError) as exc:
        await store.rename("preset", "ghost", "whatever")
    assert (exc.value.kind, exc.value.name) == ("preset", "ghost")


async def test_rename_onto_live_document_raises_exists(pg, store):
    await store.create("preset", "a", {"n": 1})
    await store.create("preset", "b", {"n": 2})
    with pytest.raises(DocumentExistsError) as exc:
        await store.rename("preset", "a", "b")
    assert (exc.value.kind, exc.value.name) == ("preset", "b")  # carries the TARGET name
    # Both rows are untouched — the failed rename committed nothing.
    assert await store.get_active_body("preset", "a") == {"n": 1}
    assert await store.get_active_body("preset", "b") == {"n": 2}


async def test_rename_onto_soft_deleted_ghost_name_succeeds(pg, store):
    # The partial-unique active index only constrains ACTIVE rows, so a soft-deleted
    # ghost named ``new`` does NOT block a rename onto ``new``.
    await store.create("preset", "new", {"ghost": True})
    await store.soft_delete("preset", "new")  # ghost occupies the name, inactive
    await store.create("preset", "src", {"live": True})
    renamed = await store.rename("preset", "src", "new")
    assert renamed.name == "new"
    assert await store.get_active_body("preset", "new") == {"live": True}
    # The ghost row + its history survive alongside the freshly-renamed active row.
    assert len([d for d in pg.documents if d["name"] == "new"]) == 2


async def test_rename_leaves_soft_deleted_ghost_of_old_name_untouched(pg, store):
    # A soft-deleted ghost of the OLD name is audit history — a later rename of a new
    # active row away from that name must not disturb the ghost.
    await store.create("preset", "old", {"gen": 1})
    await store.soft_delete("preset", "old")  # ghost of ``old``
    await store.create("preset", "old", {"gen": 2})  # new active row reusing the name
    ghost_id = next(d["id"] for d in pg.documents if d["name"] == "old" and not d["is_active"])

    await store.rename("preset", "old", "old2")

    ghost = next(d for d in pg.documents if d["id"] == ghost_id)
    assert ghost["name"] == "old"  # untouched
    assert ghost["is_active"] is False
    assert await store.get_active_body("preset", "old2") == {"gen": 2}


async def test_create_old_name_succeeds_after_rename(pg, store):
    # After a rename the old name is genuinely free — a fresh create claims it.
    await store.create("preset", "old", {"n": 1})
    await store.rename("preset", "old", "new")
    rec = await store.create("preset", "old", {"n": 2})
    assert rec.active_version == 1
    assert await store.get_active_body("preset", "old") == {"n": 2}
    assert await store.get_active_body("preset", "new") == {"n": 1}
