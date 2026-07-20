"""The pip seam: pin-argv composition, the verified github artifact fetch, the
pip pre-flight, and the runner's subprocess lifecycle including cancellation
reaping.

``run_pip`` is exercised against a fake process (``create_subprocess_exec`` is
patched), never a real pip: the zero/non-zero exit paths, the no-shell argv, the
``start_new_session`` spawn, and the process-group kill on cancellation (including
the already-exited kill race) are all asserted on the fake.
``fetch_verified_artifact`` is exercised against a fake ``fetch_url`` (patched),
never the network: the sha256 verify, the mismatch refusal, and the shape guards
are all asserted on the fake.
"""

from __future__ import annotations

import asyncio
import hashlib
import signal
from pathlib import Path

import httpx
import pytest
from tai42_kit.net import UrlGuardError

from tai42_skeleton.marketplace import pip as pip_module
from tai42_skeleton.marketplace.errors import (
    ArtifactIntegrityError,
    PipFailedError,
    PipUnavailableError,
    RegistryResponseError,
    RegistryUnreachableError,
)
from tai42_skeleton.marketplace.pip import (
    ensure_pip_available,
    fetch_verified_artifact,
    install_args,
    run_pip,
    uninstall_args,
)

# -- pin composition ---------------------------------------------------------


def test_pypi_install_args() -> None:
    args = install_args("tai42-toolbox", "1.2.3", "pypi")
    assert args == ["install", "--no-input", "--disable-pip-version-check", "tai42-toolbox==1.2.3"]


def test_github_install_args_is_the_verified_local_tarball(tmp_path: Path) -> None:
    # The github pin is the verified LOCAL tarball path — a plain absolute path,
    # never a ``git+url@tag`` clone.
    tarball = tmp_path / "tai42-toolbox-1.2.3.tar.gz"
    args = install_args("tai42-toolbox", "1.2.3", "github", tarball)
    assert args == ["install", "--no-input", "--disable-pip-version-check", str(tarball)]
    assert not args[-1].startswith("git+")


def test_github_install_args_without_verified_path_raises() -> None:
    # A github install with no verified artifact path is refused, never falling
    # through to some unverified pin.
    with pytest.raises(RegistryResponseError, match="verified artifact path"):
        install_args("tai42-toolbox", "1.2.3", "github")


def test_uninstall_args() -> None:
    assert uninstall_args("tai42-toolbox") == ["uninstall", "--yes", "tai42-toolbox"]


# -- rejection: registry-data faults, never ValueError -----------------------


def test_malformed_package_name_raises_response_error() -> None:
    with pytest.raises(RegistryResponseError, match="package name"):
        install_args("bad name; rm -rf", "1.0.0", "pypi")


def test_unparseable_version_raises_response_error() -> None:
    with pytest.raises(RegistryResponseError, match="version"):
        install_args("tai42-toolbox", "not-a-version", "pypi")


def test_unknown_source_raises_response_error() -> None:
    with pytest.raises(RegistryResponseError, match="unknown install source"):
        install_args("tai42-toolbox", "1.0.0", "svn")


# -- fetch_verified_artifact -------------------------------------------------

_ARTIFACT_URL = "https://codeload.github.com/tai42ai/toolbox/tar.gz/refs/tags/v1.2.3"


def _fake_fetch(monkeypatch: pytest.MonkeyPatch, data: bytes) -> list[str]:
    """Patch ``fetch_url`` to return ``data`` and record the URLs it was asked
    for, so a test can prove the SSRF-guarded downloader is the one used."""
    seen: list[str] = []

    async def fake(url: str):
        seen.append(url)
        return data, "application/gzip"

    monkeypatch.setattr(pip_module, "fetch_url", fake)
    return seen


async def test_fetch_verified_artifact_downloads_verifies_and_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = b"tarball-bytes"
    digest = hashlib.sha256(data).hexdigest()
    seen = _fake_fetch(monkeypatch, data)

    dest = await fetch_verified_artifact("tai42-toolbox", "1.2.3", _ARTIFACT_URL, digest, tmp_path)

    # The guarded fetch was used, for exactly the registry artifact URL.
    assert seen == [_ARTIFACT_URL]
    # The verified bytes are written to <package>-<version>.tar.gz and returned.
    assert dest == tmp_path / "tai42-toolbox-1.2.3.tar.gz"
    assert dest.read_bytes() == data


async def test_fetch_verified_artifact_accepts_uppercased_registry_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The registry digest is compared case-normalized: an upper-case hex digest
    # still matches the lower-case computed one.
    data = b"tarball-bytes"
    digest = hashlib.sha256(data).hexdigest().upper()
    _fake_fetch(monkeypatch, data)
    dest = await fetch_verified_artifact("tai42-toolbox", "1.2.3", _ARTIFACT_URL, digest, tmp_path)
    assert dest.read_bytes() == data


async def test_fetch_verified_artifact_mismatch_raises_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = b"malicious-bytes"
    wrong = "0" * 64  # not the digest of ``data``
    _fake_fetch(monkeypatch, data)

    with pytest.raises(ArtifactIntegrityError) as exc:
        await fetch_verified_artifact("tai42-toolbox", "1.2.3", _ARTIFACT_URL, wrong, tmp_path)
    assert exc.value.expected_sha256 == wrong
    assert exc.value.actual_sha256 == hashlib.sha256(data).hexdigest()
    assert exc.value.artifact_ref == _ARTIFACT_URL
    # Nothing was written — the mismatch refuses the install outright.
    assert list(tmp_path.iterdir()) == []


async def test_fetch_verified_artifact_any_fetch_failure_maps_to_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The fetch is a trust boundary typed by PROVENANCE: fetch_url's guarded
    # branch re-raises non-guard failures raw, so ANY exception class can come
    # out of it — and every one of them means the registry-named upstream did
    # not yield the bytes. Even an exception type nobody enumerated maps to the
    # typed RegistryUnreachableError (a 502), never an untyped 500, with no
    # fallback and nothing on disk.
    async def boom(url: str):
        raise RuntimeError("network down")

    monkeypatch.setattr(pip_module, "fetch_url", boom)
    with pytest.raises(RegistryUnreachableError, match="RuntimeError: network down"):
        await fetch_verified_artifact("tai42-toolbox", "1.2.3", _ARTIFACT_URL, "0" * 64, tmp_path)
    assert list(tmp_path.iterdir()) == []


async def test_fetch_verified_artifact_exception_group_maps_to_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Cycle-4 reproducer: a registry-supplied artifact_ref with an out-of-range
    # port ("https://host:99999/x") parses as an httpx.URL but fails at CONNECT
    # time with OverflowError('port must be 0-65535'), which anyio wraps in a
    # builtins ExceptionGroup — not an httpx.HTTPError, httpx.InvalidURL, or
    # UrlGuardError. ExceptionGroup subclasses Exception, so the provenance
    # catch takes the group whole and maps it to RegistryUnreachableError.
    async def boom(url: str):
        raise ExceptionGroup("unhandled errors in a TaskGroup", [OverflowError("port must be 0-65535")])

    monkeypatch.setattr(pip_module, "fetch_url", boom)
    with pytest.raises(RegistryUnreachableError, match="ExceptionGroup") as exc:
        await fetch_verified_artifact("tai42-toolbox", "1.2.3", "https://host:99999/x", "0" * 64, tmp_path)
    # The original group is chained for the server log.
    assert isinstance(exc.value.__cause__, ExceptionGroup)
    assert list(tmp_path.iterdir()) == []


async def test_fetch_verified_artifact_cancellation_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Cancellation (a client disconnect mid-download) is a BaseException and
    # must pass through the provenance catch untouched — never a 502.
    async def cancelled(url: str):
        raise asyncio.CancelledError

    monkeypatch.setattr(pip_module, "fetch_url", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await fetch_verified_artifact("tai42-toolbox", "1.2.3", _ARTIFACT_URL, "0" * 64, tmp_path)


async def test_fetch_verified_artifact_write_fault_is_not_masked_as_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The provenance catch is scoped to the fetch call ONLY: a this-server fault
    # writing the verified bytes (here, a dest_dir that does not exist) surfaces
    # as the raw OSError — a genuine server-side failure, never blamed on the
    # registry as a 502.
    data = b"tarball-bytes"
    digest = hashlib.sha256(data).hexdigest()
    _fake_fetch(monkeypatch, data)
    with pytest.raises(FileNotFoundError):
        await fetch_verified_artifact("tai42-toolbox", "1.2.3", _ARTIFACT_URL, digest, tmp_path / "absent")


async def test_fetch_verified_artifact_http_error_maps_to_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A transport/HTTP error from the artifact host (a deleted-tag 404, a 429, a
    # connection failure) is a typed RegistryUnreachableError (a 502 upstream),
    # never an untyped 500 — and nothing is written.
    async def boom(url: str):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(pip_module, "fetch_url", boom)
    with pytest.raises(RegistryUnreachableError, match=_ARTIFACT_URL):
        await fetch_verified_artifact("tai42-toolbox", "1.2.3", _ARTIFACT_URL, "0" * 64, tmp_path)
    assert list(tmp_path.iterdir()) == []


async def test_fetch_verified_artifact_guard_rejection_maps_to_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An SSRF/size/redirect refusal from the download guard is likewise a typed
    # RegistryUnreachableError, never leaking the bare UrlGuardError as a 500.
    async def blocked(url: str):
        raise UrlGuardError("SSRF guard: blocked non-public host")

    monkeypatch.setattr(pip_module, "fetch_url", blocked)
    with pytest.raises(RegistryUnreachableError, match=_ARTIFACT_URL):
        await fetch_verified_artifact("tai42-toolbox", "1.2.3", _ARTIFACT_URL, "0" * 64, tmp_path)
    assert list(tmp_path.iterdir()) == []


async def test_fetch_verified_artifact_unparseable_https_ref_maps_to_unreachable(tmp_path: Path) -> None:
    # A registry-served artifact_ref that IS https-prefixed and passes the sha256
    # shape guard but is NOT a parseable URL (an invalid port) makes the real
    # SSRF-guarded fetch_url raise httpx.InvalidURL — which is NOT an
    # httpx.HTTPError subclass. It must still surface as a typed
    # RegistryUnreachableError (a 502 upstream), never an untyped escape, with
    # nothing written. Driven through the real fetch_url so httpx genuinely raises
    # InvalidURL (no network is reached — the URL fails to parse).
    bad_ref = "https://[::1"
    with pytest.raises(RegistryUnreachableError, match="failed to fetch"):
        await fetch_verified_artifact("tai42-toolbox", "1.2.3", bad_ref, "0" * 64, tmp_path)
    assert list(tmp_path.iterdir()) == []


async def test_fetch_verified_artifact_non_https_ref_is_response_error(tmp_path: Path) -> None:
    with pytest.raises(RegistryResponseError, match="non-https artifact_ref"):
        await fetch_verified_artifact("tai42-toolbox", "1.2.3", "http://evil/x.tgz", "0" * 64, tmp_path)


async def test_fetch_verified_artifact_non_hex64_sha256_is_response_error(tmp_path: Path) -> None:
    with pytest.raises(RegistryResponseError, match="malformed artifact sha256"):
        await fetch_verified_artifact("tai42-toolbox", "1.2.3", _ARTIFACT_URL, "not-a-digest", tmp_path)


async def test_fetch_verified_artifact_bad_package_name_is_response_error(tmp_path: Path) -> None:
    # A garbled package name must not traverse the on-disk artifact path.
    with pytest.raises(RegistryResponseError, match="package name"):
        await fetch_verified_artifact("../evil", "1.2.3", _ARTIFACT_URL, "0" * 64, tmp_path)


# -- pre-flight --------------------------------------------------------------


def test_ensure_pip_available_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pip_module.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(PipUnavailableError, match="ensurepip"):
        ensure_pip_available()


# -- run_pip lifecycle -------------------------------------------------------


class _FakeProc:
    def __init__(self, returncode: int = 0, output: bytes = b"done", *, hang: bool = False) -> None:
        self.returncode = returncode
        self._output = output
        self._hang = hang
        self.pid = 4242
        self.waited = False

    async def communicate(self) -> tuple[bytes, None]:
        if self._hang:
            await asyncio.sleep(3600)  # cancelled mid-run
        return self._output, None

    async def wait(self) -> None:
        self.waited = True


def _patch_exec(monkeypatch: pytest.MonkeyPatch, proc: _FakeProc) -> dict:
    captured: dict = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(pip_module.asyncio, "create_subprocess_exec", fake_exec)
    return captured


async def test_run_pip_returns_output_and_spawns_no_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_exec(monkeypatch, _FakeProc(returncode=0, output=b"installed x"))
    out = await run_pip(["install", "x"])
    assert out == "installed x"
    # No shell: argv is (python, -m, pip, *args), and start_new_session is set.
    assert captured["args"][1:3] == ("-m", "pip")
    assert captured["args"][-2:] == ("install", "x")
    assert captured["kwargs"]["start_new_session"] is True


async def test_run_pip_nonzero_raises_pip_failed_carrying_argv_code_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_exec(monkeypatch, _FakeProc(returncode=2, output=b"resolution failed"))
    with pytest.raises(PipFailedError) as exc:
        await run_pip(["install", "x"])
    assert exc.value.argv == ["install", "x"]
    assert exc.value.returncode == 2
    assert exc.value.output == "resolution failed"


async def test_run_pip_cancel_kills_process_group_and_awaits_child(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(hang=True)
    _patch_exec(monkeypatch, proc)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(pip_module.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    task = asyncio.create_task(run_pip(["install", "x"]))
    await asyncio.sleep(0.05)  # let the fake enter communicate()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert killed == [(proc.pid, signal.SIGKILL)]
    assert proc.waited is True


async def test_run_pip_cancel_swallows_process_lookup_race(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc(hang=True)
    _patch_exec(monkeypatch, proc)

    def _already_gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(pip_module.os, "killpg", _already_gone)

    task = asyncio.create_task(run_pip(["install", "x"]))
    await asyncio.sleep(0.05)
    task.cancel()
    # The lookup miss (pip already exited) is swallowed; the in-flight
    # CancelledError — not ProcessLookupError — is what propagates.
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.waited is True
