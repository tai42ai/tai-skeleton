"""Storage operations — the deployment content store (``/api/storage*``).

A thin skin over the registered :class:`~tai42_contract.storage.Storage` provider.
Storage is dead by default (the skeleton ships no provider); a backend registers
one as a manifest-loaded plugin. Every operation reads the provider through the
concrete storage facet (``instance.app.storage.provider``) and reports honestly
when none is installed — a loud ``501`` (``NotSupportedError``) rather than a
fabricated default (except ``storage_info``, which answers ``present: false``).

Every id/path-carrying input must be a relative path with no ``..`` segment — an
absolute path (leading ``/``) or a ``..`` segment is rejected with a ``400``: the
flagship provider is filesystem-backed, where both are real traversal vectors. The
guard runs INSIDE the operation, so it defends the MCP tool edge and the CLI as
well as the HTTP route; every operation that passes a path to the provider also maps
the provider's boundary ``ValueError`` to a ``400`` so a provider-reported violation
is a loud ``400``, never a ``500``.

``list_resources`` is the op behind ``GET /api/storage/resources``; the write/delete ops
(``upload_resource``, ``delete_resource``, ``delete_dir``) mutate the store, so they
are ``destructive``.
"""

from __future__ import annotations

import base64

from pydantic import BaseModel
from tai42_contract.storage import Storage

from tai42_skeleton.app import instance
from tai42_skeleton.operations import BadRequestError, NotFoundError, NotSupportedError, operation

_NO_PROVIDER_MESSAGE = "storage needs a storage-provider plugin"
_UNSAFE_ID_MESSAGE = "must be a relative path with no '..' segment"


class StorageUpload(BaseModel):
    """A storage upload: exactly ONE of ``content_text`` (stored verbatim) or
    ``content_base64`` (decoded to bytes) supplies the content for ``id``. An
    existing id is overwritten — provider passthrough semantics."""

    id: str
    content_text: str | None = None
    content_base64: str | None = None


def _provider() -> Storage | None:
    """The registered storage provider, or ``None`` while dead by default."""
    return instance.app.storage.provider


def _require_provider() -> Storage:
    """The registered provider, or a loud ``501`` when none is installed."""
    provider = _provider()
    if provider is None:
        raise NotSupportedError(_NO_PROVIDER_MESSAGE)
    return provider


def _is_unsafe_path(value: str) -> bool:
    """Whether an id/path input is unsafe to pass to the provider — it is absolute
    (a leading ``/``) or carries a ``..`` segment. A safe input is a relative path
    with no ``..`` segment."""
    return value.startswith("/") or ".." in value.split("/")


def _reject_unsafe(kind: str, value: str) -> None:
    if _is_unsafe_path(value):
        raise BadRequestError(f"{kind} {value!r} {_UNSAFE_ID_MESSAGE}")


def _content_disposition(filename: str) -> str:
    """A ``Content-Disposition: attachment`` header with a well-formed quoted
    ``filename``. Per RFC 6266 / RFC 2616 quoted-string rules a literal ``"`` or
    ``\\`` must be backslash-escaped and control characters are not permitted, so the
    basename is sanitized before interpolation."""
    sanitized = "".join(ch for ch in filename if ch >= " " and ch != "\x7f")
    sanitized = sanitized.replace("\\", "\\\\").replace('"', '\\"')
    return f'attachment; filename="{sanitized}"'


@operation(summary="Get the storage provider identity", tags=["storage"])
async def storage_info() -> dict:
    """Report the registered provider's identity, or ``present: false`` when none is
    installed (a ``200``, so the UI renders the empty state without an error)."""
    provider = _provider()
    if provider is None:
        return {"present": False, "provider": None, "module": None}
    return {"present": True, "provider": type(provider).__name__, "module": type(provider).__module__}


@operation(
    summary="List storage resources",
    tags=["storage"],
    errors=[NotSupportedError],
)
async def list_resources() -> dict:
    """List the sorted resource ids from the active storage provider."""
    provider = _require_provider()
    return {"resources": sorted(await provider.list())}


@operation(
    summary="Stat a storage resource",
    tags=["storage"],
    errors=[BadRequestError, NotSupportedError],
)
async def stat_resource(resource_id: str) -> dict:
    """Return the resource's inferred content type."""
    _reject_unsafe("resource id", resource_id)
    provider = _require_provider()
    try:
        stat = await provider.stat(resource_id)
    except ValueError as exc:
        # A provider-reported boundary violation is a client error (400), never a 500.
        raise BadRequestError(str(exc)) from exc
    return {"id": resource_id, "content_type": stat.content_type}


@operation(
    summary="Upload a storage resource",
    tags=["storage"],
    destructive=True,
    errors=[BadRequestError, NotSupportedError],
    request_model=StorageUpload,
)
async def upload_resource(
    resource_id: str,
    content_text: str | None = None,
    content_base64: str | None = None,
) -> dict:
    """Store text OR base64-decoded bytes under ``resource_id`` (overwrite on reuse).

    The type/shape validation the tool schema cannot express — a non-empty ``id``,
    exactly one content field, and each field's type — is enforced here so the MCP
    tool edge carries it too; the HTTP route's extractor passes the raw body through
    to the same checks."""
    if not isinstance(resource_id, str) or not resource_id:
        raise BadRequestError("body must contain a non-empty string 'id'")
    _reject_unsafe("resource id", resource_id)

    if (content_text is None) == (content_base64 is None):
        raise BadRequestError("exactly one of 'content_text' or 'content_base64' is required")
    if content_text is not None and not isinstance(content_text, str):
        raise BadRequestError("'content_text' must be a string")
    if content_base64 is not None and not isinstance(content_base64, str):
        raise BadRequestError("'content_base64' must be a base64 string")

    provider = _require_provider()
    try:
        if content_text is not None:
            await provider.upload(resource_id, content_text)
        elif content_base64 is not None:
            try:
                # ``binascii.Error`` (bad padding / alphabet) subclasses ``ValueError``.
                data = base64.b64decode(content_base64, validate=True)
            except ValueError as exc:
                raise BadRequestError(f"'content_base64' is not valid base64: {exc}") from exc
            await provider.upload_bytes(resource_id, data)
    except ValueError as exc:
        # A provider-reported boundary/validation error (e.g. content a text-only
        # provider cannot store) is a client error, surfaced as 400 rather than 500.
        raise BadRequestError(str(exc)) from exc
    return {"id": resource_id, "stored": True}


@operation(
    summary="Delete a storage resource",
    tags=["storage"],
    errors=[BadRequestError, NotFoundError, NotSupportedError],
)
async def delete_resource(resource_id: str) -> dict:
    """Remove one object from the store."""
    _reject_unsafe("resource id", resource_id)
    provider = _require_provider()
    try:
        await provider.delete(resource_id)
    except FileNotFoundError as exc:
        raise NotFoundError(f"resource {resource_id!r} not found") from exc
    except ValueError as exc:
        # A provider-reported boundary violation is a client error (400), never a 500.
        raise BadRequestError(str(exc)) from exc
    return {"id": resource_id, "deleted": True}


@operation(
    summary="Delete a storage directory",
    tags=["storage"],
    errors=[BadRequestError, NotFoundError, NotSupportedError],
)
async def delete_dir(dir_path: str) -> dict:
    """Remove a directory subtree from the store."""
    _reject_unsafe("directory path", dir_path)
    provider = _require_provider()
    try:
        await provider.delete_dir(dir_path)
    except FileNotFoundError as exc:
        raise NotFoundError(f"directory {dir_path!r} not found") from exc
    except ValueError as exc:
        # ``assert_not_root`` raises ``ValueError`` for a root-resolving dir path
        # (``"."`` / ``"/"`` / ``"a/.."``) — a client error, surfaced as 400.
        raise BadRequestError(str(exc)) from exc
    return {"dir": dir_path, "deleted": True}
