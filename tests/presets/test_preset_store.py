"""The concrete ``PresetStoreView`` over an in-memory generic store.

Pins the view's three jobs: delegation to ``kind="preset"``, generic→preset error
mapping (plus ``PresetNameConflictError``), and the carry-forward/sentinel body
reshaping. The generic store's own Postgres semantics are covered in
``tests/versioning``; here the store is a faithful in-memory stand-in so the view
logic is exercised in isolation.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from tai42_contract.agent.base import PresetSpec
from tai42_contract.presets.errors import (
    PresetExistsError,
    PresetNameConflictError,
    PresetNotFoundError,
    PresetVersionNotFoundError,
)
from tai42_contract.versioning import VersionedStore, VersionedStoreTransaction
from tai42_contract.versioning.errors import (
    DocumentExistsError,
    DocumentNotFoundError,
    DocumentVersionNotFoundError,
)
from tai42_contract.versioning.models import DocumentRecord, DocumentVersion

from tai42_skeleton.presets.store import PresetStoreView


class _MemTx:
    """The in-memory unit-of-work handle (an opaque token, like the real store's)."""


class _MemStore(VersionedStore):
    """Faithful in-memory ``VersionedStore`` for view tests (one active row per key)."""

    def __init__(self) -> None:
        self.docs: dict[tuple[str, str], dict[str, Any]] = {}

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[VersionedStoreTransaction]:
        # Apply-or-discard: snapshot the docs on enter, restore them on any exception so
        # every write done under the yielded handle commits or rolls back together.
        snapshot = copy.deepcopy(self.docs)
        try:
            yield _MemTx()
        except BaseException:
            self.docs = snapshot
            raise

    async def create(
        self, kind, name, body, tags=None, *, tx: VersionedStoreTransaction | None = None
    ) -> DocumentRecord:
        key = (kind, name)
        if key in self.docs:
            raise DocumentExistsError(kind, name)
        self.docs[key] = {"active": 1, "versions": {1: (dict(body), list(tags or []))}}
        return DocumentRecord(kind=kind, name=name, active_version=1, is_active=True, created_at="t")

    async def save_version(
        self, kind, name, body, tags=None, *, tx: VersionedStoreTransaction | None = None
    ) -> DocumentVersion:
        doc = self.docs.get((kind, name))
        if doc is None:
            raise DocumentNotFoundError(kind, name)
        new_version = max(doc["versions"]) + 1
        doc["versions"][new_version] = (dict(body), list(tags or []))
        doc["active"] = new_version
        return DocumentVersion(version=new_version, body=body, tags=list(tags or []), created_at="t", is_current=True)

    async def list(self, kind) -> list[DocumentRecord]:
        return [
            DocumentRecord(kind=k, name=n, active_version=d["active"], is_active=True, created_at="t")
            for (k, n), d in self.docs.items()
            if k == kind
        ]

    async def get(self, kind, name) -> DocumentRecord:
        doc = self._require(kind, name)
        return DocumentRecord(kind=kind, name=name, active_version=doc["active"], is_active=True, created_at="t")

    async def get_active_body(self, kind, name, *, tx=None, for_update=False) -> dict[str, Any]:
        if for_update and tx is None:
            raise ValueError("get_active_body(for_update=True) requires a tx")
        doc = self._require(kind, name)
        return dict(doc["versions"][doc["active"]][0])

    async def list_versions(self, kind, name) -> list[DocumentVersion]:
        doc = self._require(kind, name)
        return [
            DocumentVersion(version=v, body=body, tags=tags, created_at="t", is_current=v == doc["active"])
            for v, (body, tags) in sorted(doc["versions"].items())
        ]

    async def get_version(self, kind, name, version, *, tx=None) -> DocumentVersion:
        doc = self.docs.get((kind, name))
        if doc is None or version not in doc["versions"]:
            raise DocumentVersionNotFoundError(kind, name, version)
        body, tags = doc["versions"][version]
        return DocumentVersion(
            version=version, body=body, tags=tags, created_at="t", is_current=version == doc["active"]
        )

    async def rollback(self, kind, name, version) -> DocumentRecord:
        doc = self.docs.get((kind, name))
        if doc is None or version not in doc["versions"]:
            raise DocumentVersionNotFoundError(kind, name, version)
        doc["active"] = version
        return DocumentRecord(kind=kind, name=name, active_version=version, is_active=True, created_at="t")

    async def soft_delete(self, kind, name) -> None:
        self._require(kind, name)
        del self.docs[(kind, name)]

    async def delete(self, kind, name) -> None:
        self._require(kind, name)
        del self.docs[(kind, name)]

    async def rename(self, kind, name, new_name) -> DocumentRecord:
        if (kind, new_name) in self.docs:
            raise DocumentExistsError(kind, new_name)
        doc = self._require(kind, name)
        self.docs[(kind, new_name)] = doc
        del self.docs[(kind, name)]
        return DocumentRecord(kind=kind, name=new_name, active_version=doc["active"], is_active=True, created_at="t")

    def _require(self, kind, name) -> dict[str, Any]:
        doc = self.docs.get((kind, name))
        if doc is None:
            raise DocumentNotFoundError(kind, name)
        return doc


def _spec(name: str = "p", *, base_tool: str = "weather", description: str = "d", fixed_kwargs=None) -> PresetSpec:
    return PresetSpec(name=name, description=description, base_tool=base_tool, fixed_kwargs=fixed_kwargs or {"a": 1})


@pytest.fixture
def store() -> _MemStore:
    return _MemStore()


@pytest.fixture
def view(store: _MemStore) -> PresetStoreView:
    return PresetStoreView(store)


# -- delegation + body round-trip -------------------------------------------


async def test_create_persists_full_body(view, store):
    await view.create_preset(_spec(), extensions=[["chain"]], tags=["t1"])
    assert ("preset", "p") in store.docs
    body = await view.get_active_body("p")
    assert body.base_tool == "weather"
    assert body.description == "d"
    assert body.fixed_kwargs == {"a": 1}
    assert body.extensions == [["chain"]]
    assert body.tags == ["t1"]
    # The typed kwargs accessor projects just fixed_kwargs.
    assert await view.get_active_kwargs("p") == {"a": 1}


async def test_tags_round_trip_per_version(view):
    await view.create_preset(_spec(), extensions=[], tags=["release"])
    await view.save_version("p", tags=["beta"])
    assert (await view.get_version("p", 1)).body["tags"] == ["release"]
    assert (await view.get_version("p", 2)).body["tags"] == ["beta"]


async def test_list_presets_delegates(view):
    await view.create_preset(_spec("a"), extensions=[], tags=[])
    await view.create_preset(_spec("b"), extensions=[], tags=[])
    assert sorted(r.name for r in await view.list_presets()) == ["a", "b"]


# -- error mapping ----------------------------------------------------------


async def test_duplicate_maps_to_preset_exists(view):
    await view.create_preset(_spec(), extensions=[], tags=[])
    with pytest.raises(PresetExistsError):
        await view.create_preset(_spec(), extensions=[], tags=[])


async def test_missing_maps_to_preset_not_found(view):
    with pytest.raises(PresetNotFoundError):
        await view.get_preset("nope")
    with pytest.raises(PresetNotFoundError):
        await view.get_active_kwargs("nope")
    with pytest.raises(PresetNotFoundError):
        await view.get_active_body("nope")
    with pytest.raises(PresetNotFoundError):
        await view.list_versions("nope")
    with pytest.raises(PresetNotFoundError):
        await view.save_version("nope", fixed_kwargs={"x": 1})
    with pytest.raises(PresetNotFoundError):
        await view.soft_delete("nope")


async def test_missing_version_maps_to_preset_version_not_found(view):
    await view.create_preset(_spec(), extensions=[], tags=[])
    with pytest.raises(PresetVersionNotFoundError):
        await view.get_version("p", 99)
    with pytest.raises(PresetVersionNotFoundError):
        await view.rollback("p", 99)


async def test_rollback_delegates(view):
    await view.create_preset(_spec(fixed_kwargs={"a": 1}), extensions=[], tags=[])
    await view.save_version("p", fixed_kwargs={"a": 2})
    rec = await view.rollback("p", 1)
    assert rec.active_version == 1
    assert await view.get_active_kwargs("p") == {"a": 1}


# -- name-conflict guard -----------------------------------------------------


async def test_name_conflict_raises_before_store_write(store):
    async def conflicts(name: str) -> bool:
        return name == "p"

    view = PresetStoreView(store, name_conflicts=conflicts)
    with pytest.raises(PresetNameConflictError):
        await view.create_preset(_spec("p"), extensions=[], tags=[])
    # The guard runs before any store write — nothing persisted.
    assert store.docs == {}


async def test_no_conflict_allows_create(store):
    async def conflicts(name: str) -> bool:
        return False

    view = PresetStoreView(store, name_conflicts=conflicts)
    await view.create_preset(_spec("free"), extensions=[], tags=[])
    assert ("preset", "free") in store.docs


# -- rename ------------------------------------------------------------------


async def test_rename_preset_delegates(view, store):
    await view.create_preset(_spec("old"), extensions=[], tags=[])
    rec = await view.rename_preset("old", "new")
    assert rec.name == "new"
    assert ("preset", "new") in store.docs
    assert ("preset", "old") not in store.docs
    with pytest.raises(PresetNotFoundError):
        await view.get_preset("old")


async def test_rename_absent_maps_to_preset_not_found(view):
    with pytest.raises(PresetNotFoundError):
        await view.rename_preset("nope", "new")


async def test_rename_onto_existing_maps_to_preset_exists(view):
    await view.create_preset(_spec("a"), extensions=[], tags=[])
    await view.create_preset(_spec("b"), extensions=[], tags=[])
    with pytest.raises(PresetExistsError):
        await view.rename_preset("a", "b")


async def test_rename_name_conflict_raises_before_store_write(store):
    async def conflicts(name: str) -> bool:
        return name == "taken"

    view = PresetStoreView(store, name_conflicts=conflicts)
    await view.create_preset(_spec("src"), extensions=[], tags=[])
    with pytest.raises(PresetNameConflictError):
        await view.rename_preset("src", "taken")
    # The predicate gates the NEW name before any store write — the source is intact.
    assert ("preset", "src") in store.docs


# -- extension validation ----------------------------------------------------


async def test_create_rejects_empty_inner_combo(view):
    with pytest.raises(ValueError, match="empty combo"):
        await view.create_preset(_spec(), extensions=[["chain"], []], tags=[])
    with pytest.raises(ValueError, match="empty combo"):
        await view.create_preset(_spec(), extensions=[[]], tags=[])


async def test_create_allows_empty_outer_extensions(view):
    # Empty OUTER list is valid on create (no extensions).
    rec = await view.create_preset(_spec(), extensions=[], tags=[])
    assert rec.active_version == 1


async def test_save_version_rejects_empty_inner_combo(view):
    await view.create_preset(_spec(), extensions=[["chain"]], tags=[])
    with pytest.raises(ValueError, match="empty combo"):
        await view.save_version("p", extensions=[["batch"], []])


# -- carry-forward matrix ----------------------------------------------------


async def _seed(view) -> None:
    await view.create_preset(
        PresetSpec(name="p", description="orig", base_tool="weather", fixed_kwargs={"a": 1}),
        extensions=[["chain"]],
        tags=["cat"],
    )


async def test_save_version_omitted_carries_everything(view):
    await _seed(view)
    await view.save_version("p")  # nothing provided
    body = await view.get_active_body("p")
    assert body.base_tool == "weather"
    assert body.description == "orig"
    assert body.fixed_kwargs == {"a": 1}
    assert body.extensions == [["chain"]]
    assert body.tags == ["cat"]


async def test_save_version_overrides_only_provided_fields(view):
    await _seed(view)
    await view.save_version("p", fixed_kwargs={"b": 2})
    body = await view.get_active_body("p")
    # base_tool + description ALWAYS carried; extensions + tags carried (omitted).
    assert body.base_tool == "weather"
    assert body.description == "orig"
    assert body.fixed_kwargs == {"b": 2}
    assert body.extensions == [["chain"]]
    assert body.tags == ["cat"]


async def test_save_version_override_tags(view):
    await _seed(view)
    await view.save_version("p", tags=["new"])
    body = await view.get_active_body("p")
    assert body.tags == ["new"]
    assert body.fixed_kwargs == {"a": 1}  # carried
    assert body.extensions == [["chain"]]  # carried


async def test_save_version_override_extensions(view):
    await _seed(view)
    await view.save_version("p", extensions=[["batch"]])
    body = await view.get_active_body("p")
    assert body.extensions == [["batch"]]
    assert body.tags == ["cat"]  # carried
    assert body.fixed_kwargs == {"a": 1}  # carried


async def test_save_version_empty_list_clears_tags(view):
    await _seed(view)
    await view.save_version("p", tags=[])
    body = await view.get_active_body("p")
    assert body.tags == []  # explicitly cleared
    assert body.extensions == [["chain"]]  # carried, not cleared


async def test_save_version_empty_list_clears_extensions(view):
    await _seed(view)
    await view.save_version("p", extensions=[])
    body = await view.get_active_body("p")
    assert body.extensions == []  # explicitly cleared
    assert body.tags == ["cat"]  # carried, not cleared


async def test_save_version_empty_dict_clears_fixed_kwargs(view):
    await _seed(view)
    await view.save_version("p", fixed_kwargs={})
    body = await view.get_active_body("p")
    assert body.fixed_kwargs == {}  # explicit empty dict is a value, not "not provided"
    assert body.description == "orig"  # never dropped


# -- output_schema carry-forward ---------------------------------------------

_OUTPUT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"x": {"type": "string"}}}


async def test_create_persists_output_schema(view):
    await view.create_preset(_spec(), extensions=[], tags=[], output_schema=_OUTPUT_SCHEMA)
    body = await view.get_active_body("p")
    assert body.output_schema == _OUTPUT_SCHEMA


async def _seed_with_schema(view) -> None:
    await view.create_preset(
        PresetSpec(name="p", description="orig", base_tool="weather", fixed_kwargs={"a": 1}),
        extensions=[["chain"]],
        tags=["cat"],
        output_schema=_OUTPUT_SCHEMA,
    )


async def test_save_version_output_schema_carries_forward_when_omitted(view):
    await _seed_with_schema(view)
    # ``output_schema`` omitted → the CARRY_FORWARD sentinel keeps the active value.
    await view.save_version("p", tags=["new"])
    body = await view.get_active_body("p")
    assert body.output_schema == _OUTPUT_SCHEMA
    assert body.tags == ["new"]


async def test_save_version_output_schema_explicit_value_wins(view):
    await _seed_with_schema(view)
    new_schema = {"type": "object", "properties": {"y": {"type": "integer"}}}
    await view.save_version("p", output_schema=new_schema)
    body = await view.get_active_body("p")
    assert body.output_schema == new_schema
    assert body.fixed_kwargs == {"a": 1}  # carried


async def test_save_version_output_schema_explicit_none_clears(view):
    await _seed_with_schema(view)
    # An explicit ``None`` CLEARS the field (distinct from the carry-forward sentinel).
    await view.save_version("p", output_schema=None)
    body = await view.get_active_body("p")
    assert body.output_schema is None
    assert body.fixed_kwargs == {"a": 1}  # carried, not cleared
