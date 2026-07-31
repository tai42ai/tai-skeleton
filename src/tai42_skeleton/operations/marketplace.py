"""Marketplace operations — search, browse, and manage installed plugins.

Ten operations back the ``/api/marketplace/*`` surface: six reads (search the
registry, one listing's detail with its versions, the category vocabulary, the
item-kind vocabulary, the installed inventory with per-row compat/update
availability plus the boot's plugin-quarantine entries, and the advisory
snapshot) and four environment-mutating flows (install, uninstall, update,
upgrade-all) driven by
:class:`~tai42_skeleton.marketplace.installer.Installer`.

The internal :class:`~tai42_skeleton.marketplace.errors.MarketplaceError` family is
load-bearing inside the marketplace package (it carries the not-installed flag,
the pip argv/return code, the two errors of a failed unwind). At THIS boundary
each is translated to the shared operation-error vocabulary
(:func:`_to_operation_error`): a dead or garbled registry is an
:class:`UpstreamError` (502 — this surface proxies the registry, so an upstream
failure is never a this-server 500); an in-progress operation elsewhere in the
fleet is an :class:`UnavailableError` (503, retriable); an unknown listing or a
not-installed ref is a :class:`NotFoundError` (404); a version/collision/contract,
already-installed, or environment-shadowed-prefix conflict is a
:class:`ConflictError` (409); a pip failure,
a github artifact-integrity mismatch, a failed unwind, a manifest-compose fault,
an unknown-item-kind binding drift, or corrupt local
state is an :class:`OperationFailed` (500). A malformed ``ref`` is the caller's
own author error — a typed :class:`~tai42_skeleton.marketplace.errors.MalformedRefError`
the boundary maps to :class:`BadRequestError` (400).

Retry-After limitation: the 503 an in-progress operation answers carries no
``Retry-After`` header. The route adapter stamps static headers on the SUCCESS
response only; an error response is header-less by construction, and the typed
error's ``extra`` dict merges into the JSON body, not the headers. The retriable
signal therefore rides the message and the 503 status, not a header. (The
reload-gate 503 — a separate concern the adapter honors from ``reload_gated``
metadata — DOES carry ``Retry-After: 5``, since that response is built by the
reload gate itself, not the error path.)

install/uninstall/update/upgrade-all mutate the running environment by running
arbitrary third-party code, so each is ``destructive`` and ``authority_changing``
(off the default MCP tool surface — tier 2 — includable only by an explicit
``api_tools.include``) and ``reload_gated`` (the flow ends in a manifest reload,
so the adapter answers a retriable 503 while a reload is in flight).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from tai42_skeleton.marketplace import advisories
from tai42_skeleton.marketplace.client import RegistryClient
from tai42_skeleton.marketplace.compat import (
    CompatVerdict,
    UpdateTargets,
    dist_compat,
    running_contract_version,
    update_targets,
)
from tai42_skeleton.marketplace.errors import (
    ArtifactIntegrityError,
    ContractIncompatibleError,
    EnvironmentShadowError,
    InstallStateError,
    ListingNotFoundError,
    MalformedRefError,
    ManifestCollisionError,
    MarketplaceError,
    OperationInProgressError,
    PipFailedError,
    RegistryResponseError,
    RegistryUnreachableError,
    VersionRefusedError,
)
from tai42_skeleton.marketplace.installer import Installer
from tai42_skeleton.marketplace.settings import marketplace_settings
from tai42_skeleton.marketplace.store import MarketplaceInstallStore
from tai42_skeleton.operations import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    OperationError,
    OperationFailed,
    UnavailableError,
    UpstreamError,
    operation,
)
from tai42_skeleton.plugins.quarantine import quarantined_plugins

logger = logging.getLogger(__name__)

# Keep the tail of an oversized detail (a pip failure's captured output) for the
# error envelope; the full text is logged whole. The prefix marks a cut.
_ENVELOPE_DETAIL_CHARS = 4000


def _truncate(text: str) -> str:
    """The last ``_ENVELOPE_DETAIL_CHARS`` of ``text``, prefixed with a visible cut
    marker when anything was dropped."""
    if len(text) <= _ENVELOPE_DETAIL_CHARS:
        return text
    return f"... (truncated) {text[-_ENVELOPE_DETAIL_CHARS:]}"


def _to_operation_error(exc: MarketplaceError) -> OperationError:
    """Translate an internal marketplace failure to its honest operation error.

    The classification keys on the typed class (and, for a state conflict, its
    ``not_installed`` flag) — never on message text.
    """
    if isinstance(exc, MalformedRefError):
        # The caller's own author error — a ref that is not a well-formed
        # lowercase ``namespace/name``.
        return BadRequestError(str(exc))
    if isinstance(exc, RegistryUnreachableError | RegistryResponseError):
        # The registry is this surface's upstream; a dead/garbled upstream is a
        # 502, never a this-server 500 and never a promise that retry fixes it.
        return UpstreamError(str(exc))
    if isinstance(exc, ListingNotFoundError):
        return NotFoundError(str(exc))
    if isinstance(exc, OperationInProgressError):
        # Another worker (or this one) holds the fleet-wide marketplace lock —
        # retriable. See the module docstring's Retry-After limitation.
        return UnavailableError(str(exc))
    if isinstance(exc, InstallStateError):
        # A not-installed ref is a 404; every other state conflict is a 409.
        return NotFoundError(str(exc)) if exc.not_installed else ConflictError(str(exc))
    if isinstance(
        exc, VersionRefusedError | ManifestCollisionError | ContractIncompatibleError | EnvironmentShadowError
    ):
        # State conflicts the operator resolves, not by retrying as-is (an
        # environment-shadowed prefix version is re-pinned or the image rebuilt).
        return ConflictError(str(exc))
    if isinstance(exc, PipFailedError):
        # The full captured output stays in the log; the envelope carries the
        # summary plus a truncated tail so the operator sees the failing lines.
        logger.error("marketplace pip failure: %s\n%s", exc, exc.output)
        return OperationFailed(str(exc), extra={"pip_output": _truncate(exc.output)})
    if isinstance(exc, ArtifactIntegrityError):
        # A github artifact whose sha256 disagrees with the registry's ingest
        # digest — an install-integrity failure (a possibly re-pointed release
        # tag), a loud terminal 500 carrying the digests and the rejected URL.
        logger.error("marketplace artifact integrity failure: %s", exc)
        return OperationFailed(
            str(exc),
            extra={
                "artifact_ref": exc.artifact_ref,
                "expected_sha256": exc.expected_sha256,
                "actual_sha256": exc.actual_sha256,
            },
        )
    # PipUnavailableError / InstallUnwindError / ManifestComposeError /
    # LocalStateError / the base: the deployment environment failed the operation.
    return OperationFailed(_truncate(str(exc)))


class MarketplaceInstall(BaseModel):
    """Install a marketplace plugin by ref, optionally pinning a version."""

    ref: str
    version: str | None = None


class MarketplaceUninstall(BaseModel):
    """Uninstall a marketplace-installed plugin by ref."""

    ref: str


class MarketplaceUpdate(BaseModel):
    """Update an installed plugin to a newer (or named) version."""

    ref: str
    version: str | None = None


@operation(summary="Search the marketplace", tags=["marketplace"], errors=[UpstreamError])
async def marketplace_search(
    q: str | None = None,
    kind: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    namespace: str | None = None,
    tier: str | None = None,
    contract: str | None = None,
    sort: str | None = None,
    page: str | None = None,
    page_size: str | None = None,
) -> dict[str, Any]:
    """Proxy the registry's public search, forwarding the whitelisted facets.

    ``tags`` is multi-value end to end (the registry receives repeated ``tags``
    params); every other facet is single-valued. ``None`` facets are dropped. The
    registry's rows ride through unchanged, so display metadata (``display_name``,
    ``icon_url``) is transparently forwarded.
    """
    params: dict[str, str | list[str]] = {}
    if q is not None:
        params["q"] = q
    if kind is not None:
        params["kind"] = kind
    if category is not None:
        params["category"] = category
    if tags:
        params["tags"] = tags
    if namespace is not None:
        params["namespace"] = namespace
    if tier is not None:
        params["tier"] = tier
    if contract is not None:
        params["contract"] = contract
    if sort is not None:
        params["sort"] = sort
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    try:
        return await RegistryClient().search(params)
    except MarketplaceError as exc:
        raise _to_operation_error(exc) from exc


@operation(summary="Get a marketplace listing's detail", tags=["marketplace"], errors=[NotFoundError, UpstreamError])
async def marketplace_plugin_detail(ns: str, name: str) -> dict[str, Any]:
    """One listing's detail composed with its version rows in a single body, so
    the detail view (listing + the Versions card) is one request. The registry's
    display metadata (``display_name``/``homepage_url``/``license``/``readme_md``)
    survives the spread.
    """
    registry = RegistryClient()
    try:
        listing = await registry.plugin(ns, name)
        versions = await registry.versions(ns, name)
    except MarketplaceError as exc:
        raise _to_operation_error(exc) from exc
    return {**listing, "versions": versions}


@operation(summary="List marketplace categories", tags=["marketplace"], errors=[UpstreamError])
async def marketplace_categories() -> list[str]:
    """The registry's controlled category vocabulary — a plain array Studio
    renders as facet chips."""
    try:
        return await RegistryClient().categories()
    except MarketplaceError as exc:
        raise _to_operation_error(exc) from exc


@operation(summary="List marketplace item kinds", tags=["marketplace"], errors=[UpstreamError])
async def marketplace_kinds() -> list[str]:
    """The registry's controlled item-kind vocabulary — a plain array Studio
    renders as facet chips."""
    try:
        return await RegistryClient().kinds()
    except MarketplaceError as exc:
        raise _to_operation_error(exc) from exc


@operation(summary="List installed marketplace plugins", tags=["marketplace"], errors=[UpstreamError, OperationFailed])
async def marketplace_installed() -> dict[str, Any]:
    """The installed inventory + the boot-quarantined plugins, in one body:
    ``{"installed": [...], "quarantined": [{"name", "reason"}, ...]}``.

    Each installed row carries the update picture computed from the registry's
    version rows against the RUNNING contract: ``latest`` (newest published
    version, or ``null`` with none), ``update_available`` (a newer COMPATIBLE
    version exists — an incompatible newer version never advertises as an
    update), ``incompatible_newer`` (the newest such blocked version, so "an
    update exists but needs a newer core" is visible, or ``null``), and
    ``compat`` — the live ``{status, reason}`` verdict of the row's installed
    distribution against the running contract. A per-row not-found — the
    upstream listing vanished or was suspended — is row STATE, not a route
    failure: that row answers ``latest: null``, ``update_available: false``,
    ``missing_upstream: true``, so one dead listing never fails the whole
    inventory. A transport/garbled-upstream failure still surfaces as a 502,
    and a garbled LOCAL row (an unparsable stored version) as a 500 — serving
    rows without their update picture would be a silent degrade of the spec'd
    shape.

    ``quarantined`` mirrors the boot pass's plugin-quarantine registry — the
    plugins this worker SKIPPED (incompatible or import-broken) with the
    human-readable reason each.
    """
    registry = RegistryClient()
    contract = running_contract_version()
    rows: list[dict[str, Any]] = []
    for record in await MarketplaceInstallStore().list_installed():
        ns, _, name = record.ref.partition("/")
        missing_upstream = False
        targets: UpdateTargets | None = None
        try:
            versions = await registry.versions(ns, name)
            targets = update_targets(versions, installed_version=record.version, contract_version=contract)
        except ListingNotFoundError:
            missing_upstream = True
        except MarketplaceError as exc:
            raise _to_operation_error(exc) from exc
        package = record.spec.get("package")
        if isinstance(package, str):
            compat = dist_compat(package)
        else:
            # A row whose stored spec names no package distribution cannot be
            # verdicted — surfaced as unknown, never a silent "compatible".
            compat = CompatVerdict("unknown", f"the stored spec for {record.ref} names no package distribution")
        rows.append(
            {
                "ref": record.ref,
                "version": record.version,
                "source": record.source,
                "installed_at": record.installed_at.isoformat(),
                "latest": targets.latest if targets is not None else None,
                "update_available": targets.update_available if targets is not None else False,
                "incompatible_newer": targets.incompatible_newer if targets is not None else None,
                "missing_upstream": missing_upstream,
                "compat": compat.as_payload(),
            }
        )
    quarantined = [{"name": name, "reason": reason} for name, reason in sorted(quarantined_plugins().items())]
    return {"installed": rows, "quarantined": quarantined}


@operation(summary="Get advisories for installed plugins", tags=["marketplace"], errors=[UpstreamError])
async def marketplace_advisories() -> dict[str, Any]:
    """The advisory snapshot for the installed plugins, no older than the
    configured poll interval (a stale snapshot is refreshed on demand, and a
    refresh failure raises a loud 502 rather than serving stale data)."""
    try:
        state = await advisories.current(marketplace_settings().advisories_interval_s)
    except MarketplaceError as exc:
        raise _to_operation_error(exc) from exc
    return {"advisories": state.advisories, "fetched_at": state.fetched_at.isoformat()}


@operation(
    summary="Install a marketplace plugin",
    tags=["marketplace"],
    destructive=True,
    reload_gated=True,
    authority_changing=True,
    errors=[BadRequestError, NotFoundError, ConflictError, UpstreamError, UnavailableError, OperationFailed],
    request_model=MarketplaceInstall,
)
async def marketplace_install(ref: str, version: str | None = None) -> dict[str, Any]:
    """Resolve, pip install, patch the manifest, reload, and record attribution —
    aborting and unwinding on any failure (see :meth:`Installer.install`)."""
    try:
        return await Installer().install(ref, version)
    except MarketplaceError as exc:
        raise _to_operation_error(exc) from exc


@operation(
    summary="Uninstall a marketplace plugin",
    tags=["marketplace"],
    destructive=True,
    reload_gated=True,
    authority_changing=True,
    errors=[NotFoundError, UnavailableError, OperationFailed],
    request_model=MarketplaceUninstall,
)
async def marketplace_uninstall(ref: str) -> dict[str, Any]:
    """Unpatch the manifest, reload, pip uninstall, and drop attribution —
    convergent and registry-free (see :meth:`Installer.uninstall`)."""
    try:
        return await Installer().uninstall(ref)
    except MarketplaceError as exc:
        raise _to_operation_error(exc) from exc


@operation(
    summary="Update a marketplace plugin",
    tags=["marketplace"],
    destructive=True,
    reload_gated=True,
    authority_changing=True,
    errors=[BadRequestError, NotFoundError, ConflictError, UpstreamError, UnavailableError, OperationFailed],
    request_model=MarketplaceUpdate,
)
async def marketplace_update(ref: str, version: str | None = None) -> dict[str, Any]:
    """Resolve the target, pip upgrade, re-patch the manifest, reload, and upsert
    attribution — with the same pre-flights as install (see
    :meth:`Installer.update`)."""
    try:
        return await Installer().update(ref, version)
    except MarketplaceError as exc:
        raise _to_operation_error(exc) from exc


@operation(
    summary="Upgrade all installed marketplace plugins",
    tags=["marketplace"],
    destructive=True,
    reload_gated=True,
    authority_changing=True,
    errors=[UnavailableError],
)
async def marketplace_upgrade_all() -> dict[str, Any]:
    """Move every installed plugin onto its latest COMPATIBLE version in one
    lock-held batch, answering ``{"results": [{ref, outcome, detail}, ...]}``.

    Per-ref failures (including a ref with NO compatible version) are report
    entries, never a batch abort — the ONE operation-level failure is the
    fleet marketplace lock being held elsewhere (the retriable 503). See
    :meth:`Installer.upgrade_all` for the outcome vocabulary.
    """
    try:
        return {"results": await Installer().upgrade_all()}
    except MarketplaceError as exc:
        raise _to_operation_error(exc) from exc
