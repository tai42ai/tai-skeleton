"""Connectors feature — the provider-agnostic OAuth engine.

Domain/wire models + ABCs live in ``tai42_contract.connectors``; this package owns
the runtime implementation (oauth/, store/, runtime/, service/, stdio/), reached
on demand by the app and the mcp adapter. Tokens are resolved at call time via
``runtime.resolver.resolve_managed_auth``. The mcp-resident token-injection glue
(:mod:`tai42_skeleton.connectors.token_injection`) + the ``_meta`` log redactor
(:mod:`tai42_skeleton.connectors.meta_log_redactor`) sit alongside the engine.
"""

from tai42_contract.connectors.errors import ConnectorError
from tai42_contract.connectors.models import (
    AuthHealthState,
    ConnectionRecord,
    ConnectorRef,
)
from tai42_contract.connectors.store import ConnectorTokenStore

from tai42_skeleton.connectors.runtime.resolver import (
    ConnectorAuthExpiredError,
    ConnectorConnectionError,
    ConnectorReconnectRequiredError,
    ConnectorRefreshFailingError,
    ManagedAuth,
    force_refresh,
    resolve_managed_auth,
)

__all__ = [
    "AuthHealthState",
    "ConnectionRecord",
    "ConnectorAuthExpiredError",
    "ConnectorConnectionError",
    "ConnectorError",
    "ConnectorReconnectRequiredError",
    "ConnectorRef",
    "ConnectorRefreshFailingError",
    "ConnectorTokenStore",
    "ManagedAuth",
    "force_refresh",
    "resolve_managed_auth",
]
