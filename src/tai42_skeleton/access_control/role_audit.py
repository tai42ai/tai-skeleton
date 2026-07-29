"""The append-only role-edit audit trail.

Every role mutation (create / edit / delete / rollback) appends a who/action/
before→after record. It rides the generic :class:`VersionedStore` under
``kind="role_audit"`` (name = the role name), each event an appended version — a
SEPARATE kind from the role itself, so a role's audit survives a hard delete of the
role. The WHEN is the version's own ``created_at``; the WHO + before→after are the event
body. An audit-write failure PROPAGATES loudly — a security surface never silently drops
an audit entry.
"""

from __future__ import annotations

from typing import Any

from tai42_contract.versioning import VersionedStore, VersionedStoreTransaction
from tai42_contract.versioning.errors import DocumentNotFoundError
from tai42_contract.versioning.models import DocumentVersion

_KIND = "role_audit"


class RoleAuditView:
    """Typed append-only audit view over the generic versioned store."""

    def __init__(self, store: VersionedStore) -> None:
        self._store = store

    async def record(
        self,
        role_name: str,
        action: str,
        actor: str | None,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        tx: VersionedStoreTransaction | None = None,
    ) -> None:
        """Append one audit event. Create-or-append: the first event for a role name
        inserts version 1, every later event appends. Raises loudly on any store fault.
        Runs the append within ``tx`` when one is supplied, so the audit commits or rolls
        back atomically with the role mutation it records."""
        event = {"action": action, "actor": actor, "before": before, "after": after}
        try:
            await self._store.get_active_body(_KIND, role_name, tx=tx)
        except DocumentNotFoundError:
            await self._store.create(_KIND, role_name, event, tx=tx)
            return
        await self._store.save_version(_KIND, role_name, event, tx=tx)

    async def list_events(self, role_name: str) -> list[DocumentVersion]:
        """Every audit event for ``role_name`` (each version carries the ``created_at``
        timestamp). Empty when the role has never been mutated."""
        try:
            return await self._store.list_versions(_KIND, role_name)
        except DocumentNotFoundError:
            return []


def role_audit() -> RoleAuditView:
    """Build the active audit view over the generic versioned store."""
    from tai42_skeleton.versioning import versioned_store

    return RoleAuditView(versioned_store())
