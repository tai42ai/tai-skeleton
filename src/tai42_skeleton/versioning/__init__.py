"""The versioned-document store package + its construction point.

:func:`versioned_store` builds the active concrete
:class:`~tai42_contract.versioning.VersionedStore`. The contract facet
``tai42_app.versioning.store`` forwards TO this builder, so this function is the
single construction point and must not call the facet back (that would loop).
"""

from __future__ import annotations

from tai42_contract.versioning import VersionedStore

from tai42_skeleton.versioning.settings import VersioningStorePgSettings
from tai42_skeleton.versioning.store import PostgresVersionedStore


def versioned_store_configured() -> bool:
    """Whether this deployment configures the versioned-document store at all.

    Resolved through the SAME pydantic-settings the store connects with
    (:class:`VersioningStorePgSettings` — the ``VERSIONING_STORE_*`` env AND the
    ``.env`` file), so a box that supplies its DSN only in ``.env`` — not the
    process environment — is still detected. The store carries no baked-in
    credential, so a supplied ``VERSIONING_STORE_PG_PASSWORD`` is the signal that a
    real store is wired up; without one, store-touching paths (list / delete /
    reconcile / versioned create) skip the Postgres open rather than fail to
    connect. Read fresh (not the cached settings singleton) so it always reflects
    the live env after a config reload."""
    return bool(VersioningStorePgSettings().pg_password.get_secret_value())


def versioned_store() -> PostgresVersionedStore:
    """Return the active generic versioned-document store.

    Typed as the concrete :class:`PostgresVersionedStore` (not the
    ``VersionedStore`` protocol) so the concrete-only batched
    ``list_active_bodies`` accessor resolves through the ``_versioned_store``
    reference; every protocol-typed surface accepts the concrete subtype."""
    return PostgresVersionedStore()


__all__ = ["PostgresVersionedStore", "VersionedStore", "versioned_store", "versioned_store_configured"]
