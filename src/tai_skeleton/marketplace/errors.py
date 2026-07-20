"""The typed failure hierarchy the marketplace client and installer raise.

Every failure family has one class under :class:`MarketplaceError`. These are
the internal, load-bearing errors: they carry the attributes the installer and
the operation boundary branch on (an HTTP status, a not-installed flag, a pip
argv + return code, the two errors of a failed unwind). The operation layer
catches them at its edge and translates each to the shared operation-error
vocabulary; nothing here leaks a raw third-party exception.
"""

from __future__ import annotations

from tai_skeleton.exceptions.exceptions import TaiMCPServerError


class MarketplaceError(TaiMCPServerError):
    """Base for every marketplace-client failure."""


class RegistryUnreachableError(MarketplaceError):
    """An upstream the registry directed us to could not be reached, errored, or
    was named unusably — the message carries the failing URL and the transport
    detail. The gateway case: this server reached out to the registry (by its base
    URL) or to a registry-named artifact host (by the artifact URL) and got no
    valid upstream response. Distinct from a this-server fault; maps to a 502."""


class RegistryResponseError(MarketplaceError):
    """The registry answered an unexpected status or a malformed envelope.

    ``status`` is the upstream HTTP status when the fault originates from an HTTP
    response, and ``None`` when the fault is the shape of registry-served data
    with no status to attribute it to (a malformed resolve spec, a pin field
    that fails validation). One constructor serves both: ``status`` defaults to
    ``None`` and the HTTP site passes the code.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status: int | None = status


class MalformedRefError(MarketplaceError):
    """The caller's ``ref`` is not a well-formed lowercase ``namespace/name`` —
    the caller's own author error, mapped at the boundary to a 400. Distinct from
    every registry/environment fault so the operation layer maps ONLY this to a
    bad-request response."""


class ListingNotFoundError(MarketplaceError):
    """Unknown listing ref or version — the message names the ref (and the
    version, when one was requested) exactly."""


class VersionRefusedError(MarketplaceError):
    """The target version is refused: a registry refusal of a killed or
    otherwise unpublished version, or a non-withdrawn critical advisory affects
    it — the message carries the reason."""


class ContractIncompatibleError(MarketplaceError):
    """The plugin's ``contract_range`` excludes the ``tai-contract`` version
    installed in this environment — the message names both. Nothing was
    installed."""


class ManifestCollisionError(MarketplaceError):
    """A provides item collides with an existing manifest entry — the message
    names every colliding field/module so the operator can resolve it."""


class ManifestBindingError(MarketplaceError):
    """A spec provides an item kind this repo's manifest bindings do not name —
    contract drift past the bindings, a server-side invariant fault (500), never
    the caller's request. Raised in place of a silently skipped item."""


class InstallStateError(MarketplaceError):
    """A state conflict: the ref is already installed, the ref is not installed,
    or the target version equals the installed one.

    ``not_installed`` is ``True`` only for the not-installed case, so the
    boundary maps that one to a not-found response and the others to a conflict.
    """

    def __init__(self, message: str, *, not_installed: bool = False) -> None:
        super().__init__(message)
        self.not_installed = not_installed


class OperationInProgressError(MarketplaceError):
    """Another marketplace operation holds the fleet-wide advisory lock (or the
    per-worker fast-path lock) — retriable; surfaced as a temporary-unavailable
    response the caller may retry."""


class PipUnavailableError(MarketplaceError):
    """The running environment cannot perform the install — the ``pip`` module is
    missing (a uv-synced venv can ship without it). The message names the exact
    fix."""


class PipFailedError(MarketplaceError):
    """A ``pip`` subprocess exited non-zero.

    Carries the credential-free ``argv``, the ``returncode``, and the captured
    combined ``output``. The message summarizes them; the full output stays on
    the attribute so the log can keep it whole while the boundary truncates for
    the response envelope.
    """

    def __init__(self, argv: list[str], returncode: int, output: str) -> None:
        super().__init__(f"pip {' '.join(argv)} exited with code {returncode}")
        self.argv = argv
        self.returncode = returncode
        self.output = output


class ManifestComposeError(MarketplaceError):
    """A composed manifest failed ``Manifest.model_validate`` — the registry
    spec plus the local manifest produced an invalid document. A server-side
    fault, never the caller's request."""


class InstallUnwindError(MarketplaceError):
    """A step failed AND the unwind of the already-applied steps also failed.

    Carries both the original ``step_error`` and the ``unwind_error`` raised
    while rolling back, and a composed message stating exactly what was left
    behind so the operator can finish the cleanup by hand.
    """

    def __init__(self, step_error: Exception, unwind_error: Exception) -> None:
        super().__init__(
            f"install step failed ({step_error}); the unwind then also failed "
            f"({unwind_error}) — the environment was left partially changed"
        )
        self.step_error = step_error
        self.unwind_error = unwind_error


class LocalStateError(MarketplaceError):
    """The local attribution-store data is corrupt (a row that cannot be
    reconstructed into an install record). A server-side fault, never the
    caller's request."""


class ArtifactIntegrityError(MarketplaceError):
    """A fetched github artifact's sha256 does not match the digest the registry
    captured at release ingest — the download is refused, not installed.

    This is its OWN type, distinct from :class:`RegistryResponseError`: the
    registry data was well-formed (a valid ``https://`` artifact_ref and a valid
    hex digest), but the FETCHED BYTES disagree with that digest — the likely
    cause is a release tag re-pointed to a different commit since ingest. Carries
    the expected and actual digests and the artifact URL so the operator can see
    exactly which download was rejected and why.
    """

    def __init__(self, *, expected_sha256: str, actual_sha256: str, artifact_ref: str) -> None:
        super().__init__(
            f"artifact sha256 mismatch for {artifact_ref}: expected {expected_sha256}, got {actual_sha256} "
            "— the release tag may have been re-pointed since registry ingest; refusing to install"
        )
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256
        self.artifact_ref = artifact_ref
