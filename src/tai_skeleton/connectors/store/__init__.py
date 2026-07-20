"""Connector token-store package + the engine's store accessor.

:func:`token_store` is the concrete builder of the active
:class:`ConnectorTokenStore` — the engine (resolver / persistence / connection
service) reaches it through here. The contract facet
``tai_app.connectors.token_store`` forwards TO this builder, so this function is
the single construction point and must not call the facet back (that would
loop).
"""

from __future__ import annotations

from tai_contract.connectors.store import ConnectorTokenStore


def token_store() -> ConnectorTokenStore:
    """Return the active connector token store, typed to the contract ABC."""
    from tai_skeleton.connectors.store.redis_pg import RedisPgConnectorTokenStore

    return RedisPgConnectorTokenStore()


__all__ = ["ConnectorTokenStore", "token_store"]
