"""The pip subprocess seam: pre-flights, the runner, and pin-argv composition.

Install and uninstall shell out to ``sys.executable -m pip`` — the one mechanism
that installs into exactly the interpreter environment serving the app, working
identically in a pip-made venv, a uv-made venv, and a container. There is no uv
path and no fallback chain: one door, honest failure.

The pin handed to pip is composed LOCALLY from the registry resolve response's
validated fields (:func:`install_args`), never taken as an opaque
registry-provided string — a registry compromise must not become an
arbitrary-package install. A pypi source pins ``package==version``; a github
source installs a LOCAL tarball that :func:`fetch_verified_artifact` has already
downloaded and checksum-verified, so pip clones nothing and runs no code the
registry did not vouch for. Every one of those fields is registry data, so a
validation failure raises :class:`RegistryResponseError` (with ``status=None``,
a registry-data fault → a gateway error), never a bare ``ValueError`` the
operation layer would mis-blame on the caller.

The github integrity anchor
---------------------------
:func:`fetch_verified_artifact` downloads the registry-named artifact tarball
over the SSRF-guarded, size-capped :func:`tai42_kit.net.fetch_url`, recomputes its
sha256, and compares it to the digest the registry captured when it ingested the
release. A match yields a local tarball path pip installs directly; a MISMATCH
raises :class:`ArtifactIntegrityError` and installs nothing — a release tag
re-pointed to a different commit since ingest is refused, never silently cloned.

The pre-flight (:func:`ensure_pip_available`) is a defensive check run before any
state change — pip is a declared runtime dependency, but a uv-synced venv can
ship without it.

POSIX-only: :func:`run_pip` uses ``start_new_session`` + ``os.killpg`` to reap
pip's whole process group (including its PEP-517 build subprocesses) on
cancellation. This ecosystem targets darwin/linux.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import os
import re
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

from packaging.version import InvalidVersion, Version
from tai42_kit.net import fetch_url

from tai42_skeleton.marketplace.errors import (
    ArtifactIntegrityError,
    PipFailedError,
    PipUnavailableError,
    RegistryResponseError,
    RegistryUnreachableError,
)

# ``run_pip``-shaped runner: the argv tail after ``-m pip`` → the combined
# stdout/stderr output. Injected into the installer so tests fake the subprocess.
PipRunner = Callable[[list[str]], Awaitable[str]]

# PEP 508 distribution name — a letter/digit, then name characters, ending on a
# letter/digit. The pin is composed from registry data; this stops a poisoned
# listing from smuggling pip options or a path into the package position.
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")

# A hex sha256 digest, exactly 64 lowercase hex characters. The registry captures
# this at release ingest; a value of any other shape is garbled registry data.
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def ensure_pip_available() -> None:
    """Raise :class:`PipUnavailableError` when the ``pip`` module is not importable.

    A uv-synced venv ships without pip by default; the message names both fixes.
    """
    if importlib.util.find_spec("pip") is None:
        raise PipUnavailableError(
            "the environment serving this app has no 'pip' module; install it with "
            "'python -m ensurepip --upgrade', or 'uv pip install pip' in a uv-managed venv"
        )


async def run_pip(args: list[str]) -> str:
    """Run ``sys.executable -m pip <args>`` and return its combined output.

    No shell, ever. A non-zero exit raises :class:`PipFailedError` carrying the
    argv, the return code, and the full captured output. There is no timeout
    knob: a pip resolve may legitimately run minutes, and a hung pip surfaces via
    the operator's own request timeout and the install lock's retriable refusal
    for followers.

    The pip subprocess is spawned WITHOUT ``env=``, so it inherits the worker's
    environment and the operator's pip configuration (``PIP_INDEX_URL``, proxies,
    …) reaches pip unchanged.

    Spawned in a NEW session (``start_new_session=True``), so pip is a
    process-group leader: on ``CancelledError`` (a client disconnect mid-run
    cancels the awaiting task) the whole group is killed with
    ``os.killpg(..., SIGKILL)`` — reaping pip's PEP-517 build subprocesses, which
    asyncio does not reap on cancellation — and the child is awaited before the
    cancellation re-raises, so "task cancelled ⇒ no pip still mutating the venv"
    holds. A ``ProcessLookupError`` from the ``killpg`` means pip already exited
    in the gap and is suppressed, so it never replaces the in-flight
    ``CancelledError``.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pip",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = await proc.communicate()
    except asyncio.CancelledError:
        # pip already exited in the gap ⇒ its group is gone ⇒ the desired end
        # state, so the lookup miss must not replace the in-flight cancellation.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()
        raise
    output = stdout.decode(errors="replace")
    if proc.returncode != 0:
        raise PipFailedError(args, proc.returncode or -1, output)
    return output


def _validate_package(package: str) -> None:
    if not _PACKAGE_NAME_RE.fullmatch(package):
        raise RegistryResponseError(f"registry returned an invalid package name {package!r}", status=None)


def _validate_version(version: str) -> None:
    # ``version`` is a non-null string by construction: on the resolve path
    # ``_require`` rejects a null and the registry client's resolve boundary types
    # the field when present, and a stored pin rides a str-typed ``InstallRecord``
    # column. This check owns only the string's FORMAT: a non-PEP440 value is
    # garbled registry data → 502.
    try:
        Version(version)
    except InvalidVersion as exc:
        raise RegistryResponseError(f"registry returned an unusable version {version!r}: {exc}", status=None) from exc


async def fetch_verified_artifact(package: str, version: str, artifact_ref: str, sha256: str, dest_dir: Path) -> Path:
    """Download the registry-named artifact, verify its sha256, and write it out.

    ``artifact_ref`` is the registry's artifact URL (the github codeload
    tag-tarball) and ``sha256`` the digest the registry captured at release
    ingest. Both are registry data: ``artifact_ref`` MUST be ``https://`` and
    ``sha256`` MUST be 64 lowercase hex characters, and ``package`` / ``version``
    must be a valid PEP 508 name and PEP 440 version (they name the on-disk file,
    so a garbled value must never traverse the path). Any shape failure raises
    :class:`RegistryResponseError` (``status=None``).

    The bytes are fetched over the SSRF-guarded, size-capped
    :func:`tai42_kit.net.fetch_url` and their recomputed sha256 is compared,
    case-normalized, to the registry digest. A MATCH writes the verified bytes to
    ``dest_dir/<package>-<version>.tar.gz`` and returns that path for pip to
    install directly. A MISMATCH raises :class:`ArtifactIntegrityError` and writes
    nothing — the release tag may have been re-pointed to a different commit since
    ingest, and there is no fallback that would install it unverified.

    ANY failure of the fetch itself — an ``httpx.HTTPError`` (a deleted-tag 404,
    a 429, a transport error), an ``httpx.InvalidURL`` (an https-prefixed but
    unparseable URL), a :class:`tai42_kit.net.UrlGuardError` (an SSRF/size/redirect
    refusal), or a raw connect-time surprise (an ``OverflowError`` from an
    out-of-range port, wrapped in an ``ExceptionGroup`` by anyio) — is mapped to
    :class:`RegistryUnreachableError` (a 502 at the boundary). The mapping is by
    PROVENANCE, not by exception class: ``artifact_ref`` is untrusted registry
    data and the fetch is the only line that acts on it, so a failure there is an
    upstream/registry-data fault by construction, never an untyped 500. The
    digest verification and the local write sit outside that mapping, so an
    integrity mismatch and a this-server disk fault each surface as themselves.
    """
    _validate_package(package)
    _validate_version(version)
    if not artifact_ref.startswith("https://"):
        raise RegistryResponseError(
            f"registry returned a non-https artifact_ref {artifact_ref!r} for a github source", status=None
        )
    expected = sha256.lower()
    if not _SHA256_RE.fullmatch(expected):
        raise RegistryResponseError(f"registry returned a malformed artifact sha256 {sha256!r}", status=None)

    try:
        data, _mime = await fetch_url(artifact_ref)
    except Exception as exc:
        # TRUST BOUNDARY — typed by provenance, not by exception class.
        # ``artifact_ref`` is untrusted registry data, and this call is the only
        # line that acts on it over the network, so ANY failure here means "the
        # registry-named upstream did not yield the bytes": an upstream/
        # registry-data fault, mapped to a typed RegistryUnreachableError (a 502
        # at the boundary), never an untyped 500. Enumerating exception classes
        # is unsound here — fetch_url's guarded branch re-raises non-guard
        # failures raw, so its surface spans httpx.HTTPError, httpx.InvalidURL,
        # UrlGuardError, AND arbitrary connect-time errors (an OverflowError from
        # a registry-supplied out-of-range port, wrapped by anyio in a builtins
        # ExceptionGroup — itself an Exception subclass, so caught here whole).
        # The try scope is exactly this one call: the digest comparison, the
        # ArtifactIntegrityError, and the local write_bytes sit OUTSIDE it, so a
        # genuine this-server fault still surfaces as itself. CancelledError and
        # a BaseExceptionGroup carrying one are BaseExceptions and pass through
        # untouched, so cancellation is never swallowed into a 502.
        raise RegistryUnreachableError(
            f"failed to fetch the github artifact at {artifact_ref}: {type(exc).__name__}: {exc}"
        ) from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ArtifactIntegrityError(expected_sha256=expected, actual_sha256=actual, artifact_ref=artifact_ref)

    dest = dest_dir / f"{package}-{version}.tar.gz"
    dest.write_bytes(data)
    return dest


def install_args(
    package: str,
    version: str,
    source: str,
    verified_path: Path | None = None,
) -> list[str]:
    """Validate the registry-supplied pin fields, then build the ``pip install`` argv.

    ``package`` must be a PEP 508 name and ``version`` PEP 440-parseable. A pypi
    source pins ``package==version`` (PyPI enforces file immutability). A github
    source installs ``verified_path`` — the local tarball
    :func:`fetch_verified_artifact` already downloaded and checksum-verified — as a
    plain absolute path, so pip runs only the vouched-for bytes and never clones a
    mutable tag. A validation deviation raises :class:`RegistryResponseError`
    (``status=None``) — these are registry data, not caller input.
    """
    _validate_package(package)
    _validate_version(version)
    flags = ["install", "--no-input", "--disable-pip-version-check"]
    if source == "pypi":
        return [*flags, f"{package}=={version}"]
    if source == "github":
        if verified_path is None:
            raise RegistryResponseError(
                "a github install needs a verified artifact path, but none was supplied", status=None
            )
        return [*flags, str(verified_path)]
    raise RegistryResponseError(f"registry returned an unknown install source {source!r}", status=None)


def uninstall_args(package: str) -> list[str]:
    """Validate the distribution name, then build the ``pip uninstall`` argv."""
    _validate_package(package)
    return ["uninstall", "--yes", package]
