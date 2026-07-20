"""Tool-edge authorization — one decision, consumed on the MCP surface.

The ``access_control/`` primitives are the single implementation of the
permission decision; this package is their SECOND consumer (the HTTP middleware
is the first, unchanged). :func:`check` is the one entry point; ``AuthzMiddleware``
installs it on every MCP-serving FastMCP instance.
"""

from tai_skeleton.authz.check import check, synthesize_path
from tai_skeleton.authz.identity import INTERNAL_PRINCIPAL, CallerIdentity, resolve_caller_identity
from tai_skeleton.authz.middleware import AuthzMiddleware
from tai_skeleton.authz.resolver import resolve_base_operation

__all__ = [
    "INTERNAL_PRINCIPAL",
    "AuthzMiddleware",
    "CallerIdentity",
    "check",
    "resolve_base_operation",
    "resolve_caller_identity",
    "synthesize_path",
]
