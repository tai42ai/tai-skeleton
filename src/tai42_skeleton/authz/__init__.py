"""Tool-edge authorization — one decision, consumed at the MCP surface and by
background executions at the shared tool-dispatch seam.

The ``access_control/`` primitives are the single implementation of the
permission decision; this package is their SECOND consumer (the HTTP middleware
is the first). :func:`check` is the one entry point; ``AuthzMiddleware``
installs it on every MCP-serving FastMCP instance.

A background fire consumes the same decision without a token: it binds the identity
of its execution key into the dedicated ``execution_identity`` contextvar, and the
shared tool-dispatch seam authorizes each call against it. That surface stays on
``authz.execution`` / ``authz.execution_identity`` rather than being republished
here — ``bind_execution_identity`` is the one way to bind, and keeping it beside the
contextvar's set/reset trio it wraps is what makes the paired-``finally`` discipline
the path of least resistance, mirroring how ``access_control`` keeps its
request-scope setters on their own module.

So the package root publishes only the tool-edge decision itself — :func:`check` and its
``synthesize_path`` helper, the two names a root consumer reaches through. Every other
authz surface stays on its submodule: the identity types on ``authz.identity``, the
middleware on ``authz.middleware``, the fire-time binding on ``authz.execution`` /
``authz.execution_identity`` (above), and ``authz.resolver`` — the name→operation
resolution both edges run before the decision, and the retriable refusal it raises while
the operation surface is mid-rebuild. Every consumer imports each from its own module,
and a re-export nothing reaches through would be a second sanctioned-looking import path
to keep in sync with the first.
"""

from tai42_skeleton.authz.check import check, synthesize_path

__all__ = [
    "check",
    "synthesize_path",
]
