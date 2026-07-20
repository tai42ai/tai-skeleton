"""The typed error vocabulary the operations layer raises.

An operation raises one of these instead of returning an HTTP response: each
class declares the HTTP status the route adapter maps it to, and the tool
projection surfaces the same message as a loud ``ToolError``. The classification
(an unknown resource is a ``404``, a rejected body is a ``422``) lives on the
operation as a typed raise, so the same decision serves the route, the CLI, and
the projected tool.
"""

from __future__ import annotations

from typing import ClassVar


class OperationError(Exception):
    """Base class for every declared operation failure.

    ``status`` is the HTTP status the route adapter answers with and the family
    the projection reports; subclasses set it. The base itself carries ``500`` so
    an operation raising the base (rather than a specific subclass) still maps to
    a defined status rather than an unhandled crash.
    """

    status: ClassVar[int] = 500

    def __init__(self, message: str, *, extra: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        # Additional fields merged into the ``{"error": …}`` body the adapter emits
        # (e.g. a stable ``code`` a UI keys a dedicated error state on). Empty by
        # default, so an error carries only ``{"error": …}`` unless a raiser opts in.
        self.extra: dict[str, object] = extra or {}


class ValidationRejected(OperationError):
    """The request was well-formed but failed the operation's own validation."""

    status: ClassVar[int] = 422


class BadRequestError(OperationError):
    """The request was malformed and could not be processed."""

    status: ClassVar[int] = 400


class NotFoundError(OperationError):
    """The addressed resource does not exist."""

    status: ClassVar[int] = 404


class PayloadTooLargeError(OperationError):
    """The request body (or query string) exceeded the configured byte cap.

    The ``413`` a door answers when a caller's payload is rejected on ACTUAL
    bytes before it is parsed — never truncated, always loud.
    """

    status: ClassVar[int] = 413


class PermissionDenied(OperationError):
    """The caller is authenticated but not authorized for this operation.

    The single denial type both edges share: the route adapter is never the
    raiser (the HTTP middleware owns the route edge), but the tool-edge
    ``AuthzMiddleware`` raises it and surfaces it as a ``ToolError`` while the
    route edge maps it to the same ``403`` the middleware already emits.
    """

    status: ClassVar[int] = 403


class ForbiddenError(OperationError):
    """The authenticated caller is not authorized for this operation by the
    operation's OWN rules — an ownership/administration gate the operation enforces
    on itself (e.g. a non-admin acting on a key it does not own, or a non-admin
    reaching an admin-only policy-administration door).

    Distinct from :class:`PermissionDenied`, which is the shared route-edge/tool-edge
    denial the access-control middleware and ``AuthzMiddleware`` raise: this ``403``
    is an in-operation business-authorization decision the operation itself makes and
    the adapter maps to a ``403`` response.
    """

    status: ClassVar[int] = 403


class ConflictError(OperationError):
    """The operation conflicts with the current state of the resource."""

    status: ClassVar[int] = 409


class NotSupportedError(OperationError):
    """The operation needs a capability this deployment does not provide.

    The ``501`` a fleet door answers when no backend plugin is registered, or a
    channel that cannot deliver a notification — a capability the deployment
    lacks, distinct from a transient outage (``503``) or an upstream failure
    (``502``).
    """

    status: ClassVar[int] = 501


class UpstreamError(OperationError):
    """An upstream dependency the operation delegates to failed.

    The ``502`` a door answers when a fleet op (or a channel send) reaches its
    dependency but the dependency fails — the failure is reported with the
    dependency's own message, never swallowed.
    """

    status: ClassVar[int] = 502


class UnavailableError(OperationError):
    """A dependency the operation needs is temporarily unavailable."""

    status: ClassVar[int] = 503


class OperationFailed(OperationError):
    """The operation was reached but failed while executing."""

    status: ClassVar[int] = 500


__all__ = [
    "BadRequestError",
    "ConflictError",
    "ForbiddenError",
    "NotFoundError",
    "NotSupportedError",
    "OperationError",
    "OperationFailed",
    "PayloadTooLargeError",
    "PermissionDenied",
    "UnavailableError",
    "UpstreamError",
    "ValidationRejected",
]
