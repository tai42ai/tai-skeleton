"""The bound EXECUTION identity a background fire runs as.

A fire carries no HTTP request, so none of the request-scope caller facts are bound. The
fire path binds the synthetic :class:`CallerIdentity` built from the execution key's live
stored grants here instead, and the tool-dispatch seam gates on it: ``None`` (the default)
means the seam does nothing.

DELIBERATELY DISTINCT from the request-scope identity vars —
:func:`~tai42_skeleton.authz.identity.resolve_caller_identity` never reads this one — so a
background identity can never be mistaken for an authenticated caller, or the reverse.
Always release via :func:`reset_execution_identity` on the matching token in a ``finally``.

Release is per-CONTEXT, not global: a task created inside the block runs on a COPY and
keeps the identity for its own lifetime. That is load-bearing —
:func:`~tai42_skeleton.operations.tool_runs._spawn_supervisor` detaches such a task, and
the inherited identity is what keeps the tool it later runs authorized as the submitter.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

from tai42_skeleton.authz.identity import CallerIdentity

__all__ = [
    "get_execution_identity",
    "reset_execution_identity",
    "set_execution_identity",
]

_current_execution_identity: ContextVar[CallerIdentity | None] = ContextVar(
    "tai42_current_execution_identity", default=None
)


def get_execution_identity() -> CallerIdentity | None:
    """The execution identity bound to the current fire, ``None`` outside a background
    execution — the signal the tool-dispatch seam gates enforcement on."""
    return _current_execution_identity.get()


def set_execution_identity(identity: CallerIdentity | None) -> Token[CallerIdentity | None]:
    """Bind ``identity`` as the current fire's execution identity; pass the returned token
    to :func:`reset_execution_identity` to restore the previous value."""
    return _current_execution_identity.set(identity)


def reset_execution_identity(token: Token[CallerIdentity | None]) -> None:
    """Restore the execution identity to the value captured in ``token`` by the matching
    :func:`set_execution_identity` call."""
    _current_execution_identity.reset(token)
