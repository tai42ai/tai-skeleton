"""The access-control policy version history view over the generic store.

A thin, skeleton-internal view on the generic
:class:`~tai_contract.versioning.VersionedStore` with ``kind="ac_policy"`` and
``name = user_id`` — the SECOND consumer of the shared versioning primitive
(mirroring :class:`~tai_skeleton.presets.store.PresetStoreView`). It adds NO new
contract type and NO new versioning code.

Its role is narrow and deliberate: it holds a policy's **append-only version
history** — the rollback SOURCE and audit trail — NOT the enforcement authority.
The Postgres access-control policy store
(:class:`~tai_skeleton.access_control.store.PostgresAccessControlStore`) holds the
enforced policy the auth gate reads; this view's active pointer FOLLOWS that store,
written through after each policy write.

The body is the same shape enforcement reads from the policy store:
``{scopes, policy_data, condition, condition_id, condition_kwargs}``.

The one piece of view logic is **create-or-append**: the generic ``save_version``
raises :class:`DocumentNotFoundError` when no document exists yet, so a uniform
``save_version`` would raise on every user's FIRST policy write. ``write``
therefore checks existence first — the first write for a ``user_id`` inserts
version 1 (``create``), every later write with a changed body appends
(``save_version``). A write whose body is byte-for-byte the active version is a
no-op (a description-only key edit re-writes the policy record verbatim; recording
an identical version would only pollute the audit trail).
"""

from __future__ import annotations

from typing import Any

from tai_contract.versioning import VersionedStore
from tai_contract.versioning.errors import DocumentNotFoundError
from tai_contract.versioning.models import DocumentRecord, DocumentVersion

_KIND = "ac_policy"


class AcPolicyStore:
    """Typed access-control-policy view delegating to a generic
    :class:`VersionedStore` under ``kind="ac_policy"``."""

    def __init__(self, store: VersionedStore) -> None:
        self._store = store

    async def write(self, user_id: str, body: dict[str, Any]) -> bool:
        """Persist ``body`` as this user's policy history, create-or-append.

        Returns ``True`` when a new version was written (a fresh ``create`` or an
        appended ``save_version``), ``False`` when ``body`` already equals the
        active version and nothing was appended. Any real store error propagates
        loudly — only the "no document yet" case is handled, by creating version 1.
        """
        try:
            active = await self._store.get_active_body(_KIND, user_id)
        except DocumentNotFoundError:
            await self._store.create(_KIND, user_id, body)
            return True
        if active == body:
            return False
        await self._store.save_version(_KIND, user_id, body)
        return True

    async def list_versions(self, user_id: str) -> list[DocumentVersion]:
        return await self._store.list_versions(_KIND, user_id)

    async def get_version(self, user_id: str, version: int) -> DocumentVersion:
        return await self._store.get_version(_KIND, user_id, version)

    async def get_active_body(self, user_id: str) -> dict[str, Any]:
        return await self._store.get_active_body(_KIND, user_id)

    async def rollback(self, user_id: str, version: int) -> DocumentRecord:
        return await self._store.rollback(_KIND, user_id, version)


def ac_policy_store() -> AcPolicyStore:
    """Build the active access-control-policy view over the generic store."""
    from tai_skeleton.versioning import versioned_store

    return AcPolicyStore(versioned_store())
