"""The ``AcPolicyStore`` view over an in-memory generic store.

Pins the view's one piece of logic — create-or-append — in isolation: the FIRST
write for a ``user_id`` inserts version 1 (``create``), a later changed write
appends (``save_version``), an unchanged write is a no-op, and rollback/history
delegate straight through. The generic store's own Postgres semantics are covered
in ``tests/versioning``; here the store is a faithful in-memory stand-in.
"""

from __future__ import annotations

from typing import Any

import pytest
from tai_contract.versioning import VersionedStore
from tai_contract.versioning.errors import (
    DocumentExistsError,
    DocumentNotFoundError,
    DocumentVersionNotFoundError,
)
from tai_contract.versioning.models import DocumentRecord, DocumentVersion

from tai_skeleton.access_control.policy_store import AcPolicyStore


class _MemStore(VersionedStore):
    """Faithful in-memory ``VersionedStore`` (one active row per ``(kind, name)``)."""

    def __init__(self) -> None:
        self.docs: dict[tuple[str, str], dict[str, Any]] = {}

    async def create(self, kind, name, body, tags=None) -> DocumentRecord:
        key = (kind, name)
        if key in self.docs:
            raise DocumentExistsError(kind, name)
        self.docs[key] = {"active": 1, "versions": {1: (dict(body), list(tags or []))}}
        return DocumentRecord(kind=kind, name=name, active_version=1, is_active=True, created_at="t")

    async def save_version(self, kind, name, body, tags=None) -> DocumentVersion:
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

    async def get_active_body(self, kind, name) -> dict[str, Any]:
        doc = self._require(kind, name)
        return dict(doc["versions"][doc["active"]][0])

    async def list_versions(self, kind, name) -> list[DocumentVersion]:
        doc = self._require(kind, name)
        return [
            DocumentVersion(version=v, body=body, tags=tags, created_at="t", is_current=v == doc["active"])
            for v, (body, tags) in sorted(doc["versions"].items())
        ]

    async def get_version(self, kind, name, version) -> DocumentVersion:
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


def _policy(**over: Any) -> dict[str, Any]:
    base = {"scopes": ["s"], "policy_data": {}, "condition": None, "condition_id": None, "condition_kwargs": None}
    base.update(over)
    return base


@pytest.fixture
def store() -> _MemStore:
    return _MemStore()


@pytest.fixture
def view(store: _MemStore) -> AcPolicyStore:
    return AcPolicyStore(store)


async def test_first_write_creates_version_one(view: AcPolicyStore, store: _MemStore) -> None:
    # A uniform ``save_version`` would raise DocumentNotFoundError here; the view's
    # create-or-append inserts version 1 instead.
    wrote = await view.write("u1", _policy(condition="a"))
    assert wrote is True
    assert store.docs[("ac_policy", "u1")]["active"] == 1
    assert (await view.get_active_body("u1"))["condition"] == "a"


async def test_second_changed_write_appends(view: AcPolicyStore) -> None:
    await view.write("u1", _policy(condition="a"))
    wrote = await view.write("u1", _policy(condition="b"))
    assert wrote is True
    versions = await view.list_versions("u1")
    assert [v.version for v in versions] == [1, 2]
    assert [v.is_current for v in versions] == [False, True]
    assert (await view.get_active_body("u1"))["condition"] == "b"


async def test_unchanged_write_is_noop(view: AcPolicyStore) -> None:
    await view.write("u1", _policy(condition="a"))
    wrote = await view.write("u1", _policy(condition="a"))
    assert wrote is False
    assert [v.version for v in await view.list_versions("u1")] == [1]


async def test_rollback_repoints_active(view: AcPolicyStore) -> None:
    await view.write("u1", _policy(condition="a"))
    await view.write("u1", _policy(condition="b"))
    rec = await view.rollback("u1", 1)
    assert rec.active_version == 1
    assert (await view.get_active_body("u1"))["condition"] == "a"


async def test_missing_version_raises(view: AcPolicyStore) -> None:
    await view.write("u1", _policy())
    with pytest.raises(DocumentVersionNotFoundError):
        await view.get_version("u1", 99)


async def test_history_absent_raises_not_found(view: AcPolicyStore) -> None:
    with pytest.raises(DocumentNotFoundError):
        await view.list_versions("ghost")
