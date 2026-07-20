"""The installer's step/unwind matrix over fully faked seams.

Every collaborator is a recording fake (registry, pip runner, attribution store,
config-mutation pipeline, fleet lock, config manager), so each install/uninstall/update path —
happy, refused, and every unwind — is driven deterministically. A shared event
log pins the fleet-lock ordering (acquired before any pre-flight, released after
the attribution write).
"""

from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from pyaml_env import parse_config
from tai_contract.plugins import PluginSpec
from tai_kit.utils.data import dump_manifest

from tai_skeleton.app.boot_rules import BackendNeedsBusError
from tai_skeleton.app.bus import FleetResult
from tai_skeleton.config.service import ApplyResult
from tai_skeleton.manifest import Manifest
from tai_skeleton.marketplace import installer as installer_module
from tai_skeleton.marketplace.errors import (
    ArtifactIntegrityError,
    ContractIncompatibleError,
    InstallStateError,
    InstallUnwindError,
    LocalStateError,
    MalformedRefError,
    ManifestCollisionError,
    ManifestComposeError,
    OperationInProgressError,
    PipFailedError,
    PipUnavailableError,
    RegistryResponseError,
    VersionRefusedError,
)
from tai_skeleton.marketplace.installer import Installer
from tai_skeleton.marketplace.store import InstallRecord
from tai_skeleton.operations._broadcast import FleetBroadcastError
from tests.marketplace._specs import make_resolved, make_spec


def _tool_provides(module: str) -> list[dict[str, str]]:
    return [{"kind": "tool", "name": "gen-uuid", "module": module, "description": "d"}]


class FakeRegistry:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.resolved: dict[str, Any] | None = None
        self.resolve_error: Exception | None = None
        self.resolve_calls: list[tuple[str, str, str | None]] = []

    async def resolve(self, ns: str, name: str, version: str | None = None) -> dict[str, Any]:
        self._events.append("registry:resolve")
        self.resolve_calls.append((ns, name, version))
        if self.resolve_error is not None:
            raise self.resolve_error
        assert self.resolved is not None
        return self.resolved


class FakePip:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.calls: list[list[str]] = []
        self.fail_on: set[int] = set()

    async def __call__(self, args: list[str]) -> str:
        idx = len(self.calls)
        self.calls.append(args)
        self._events.append("pip")
        if idx in self.fail_on:
            raise PipFailedError(args, 1, "pip boom")
        return "pip ok"


class FakeStore:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.rows: dict[str, InstallRecord] = {}
        self.record_calls: list[tuple] = []
        self.record_error: Exception | None = None

    async def get(self, ref: str) -> InstallRecord | None:
        self._events.append("store:get")
        return self.rows.get(ref)

    async def record(self, ref, version, source, repository_url, tag, artifact_ref, sha256, spec) -> None:
        self._events.append("store:record")
        self.record_calls.append((ref, version, source, repository_url, tag, artifact_ref, sha256, spec))
        if self.record_error is not None:
            raise self.record_error
        self.rows[ref] = InstallRecord(
            ref=ref,
            version=version,
            source=source,
            repository_url=repository_url,
            tag=tag,
            artifact_ref=artifact_ref,
            sha256=sha256,
            spec=spec,
            installed_at=datetime.now(UTC),
        )

    async def delete(self, ref: str) -> bool:
        self._events.append("store:delete")
        return self.rows.pop(ref, None) is not None

    async def list_installed(self) -> list[InstallRecord]:
        return list(self.rows.values())

    def preload(
        self,
        spec: PluginSpec,
        *,
        version: str,
        source: str = "pypi",
        repository_url=None,
        tag=None,
        artifact_ref=None,
        sha256=None,
    ) -> None:
        self.rows[spec.ref] = InstallRecord(
            ref=spec.ref,
            version=version,
            source=source,
            repository_url=repository_url,
            tag=tag,
            artifact_ref=artifact_ref,
            sha256=sha256,
            spec=spec.model_dump(mode="json"),
            installed_at=datetime.now(UTC),
        )


class FakeCM:
    """The read seam the installer drives for its pre-flights (collision check and
    the saved-manifest capture). Writes never go through here — they cross the
    :class:`FakeConfigService` pipeline, which mutates ``_manifest`` in place."""

    def __init__(self, events: list[str], manifest: dict[str, Any] | None = None) -> None:
        self._events = events
        self._manifest = manifest or {}

    def read_manifest(self) -> dict[str, Any]:
        self._events.append("cm:read")
        return copy.deepcopy(self._manifest)

    def read_manifest_preserved(self) -> dict[str, Any]:
        self._events.append("cm:read")
        return copy.deepcopy(self._manifest)


def _fleet_report() -> FleetResult:
    return FleetResult(op="reload_config", results=[])


def _unreachable_report() -> FleetResult:
    """The honest bus-unreachable report shape ConfigService now attaches when a
    broadcast raises after the persist committed — no origin list, only the error."""
    return FleetResult(op="reload_config", reachable=False, error="ResponseError: WRONGTYPE")


class FakeConfigService:
    """The manifest-mutation pipeline seam: records each apply and mutates the shared
    :class:`FakeCM`, standing in for :class:`ConfigService`.

    Like the real pipeline, ``apply_change`` validates the RESOLVED projection of the
    composed document BEFORE it persists — ``!ENV`` markers materialized through the
    same ``dump_manifest`` / ``parse_config`` round-trip — so a marker on a non-string
    manifest field validates against its resolved value, and a schema-invalid resolved
    compose raises a :class:`ValidationError` with nothing persisted. ``raise_on_validate``
    injects an invariant failure the pipeline would raise in that same gate (e.g. a
    backend-needs-bus refusal).

    ``fail_persist_on`` raises BEFORE the persist lands (a transaction/write failure
    — nothing persisted, no reload). ``fail_reload_on`` raises a
    :class:`FleetBroadcastError` AFTER the persist lands (the local reload failed once
    the change had committed and the broadcast went out). ``fail_broadcast_on`` models
    the third post-persist outcome: the persist committed and the local reload
    succeeded, but the FLEET BROADCAST then raised — ConfigService wraps that raw
    broadcast fault as a :class:`FleetBroadcastError` carrying the honest
    bus-unreachable report, so it too surfaces as a landed-but-propagation-failed
    change. Each apply appends ``cm:write`` then ``reload`` to the shared event log,
    mirroring the pipeline's persist-then-reload order, and increments ``calls``."""

    def __init__(self, events: list[str], cm: FakeCM) -> None:
        self._events = events
        self._cm = cm
        self.writes: list[dict[str, Any]] = []
        self.calls = 0
        self.fail_persist_on: set[int] = set()
        self.fail_reload_on: set[int] = set()
        self.fail_broadcast_on: set[int] = set()
        self.raise_on_validate: Exception | None = None

    async def apply_change(self, mutator: Any) -> ApplyResult:
        document = copy.deepcopy(self._cm._manifest)
        mutator(document)  # the structural provides patch (may raise, e.g. a collision)
        self._validate_resolved(document)  # the pipeline's pre-persist RESOLVED-projection gate
        return self._persist(document)

    def _validate_resolved(self, document: dict[str, Any]) -> None:
        # Materialize the ``!ENV`` markers exactly as ConfigService does, then validate
        # the schema — so a marker on a non-string field is checked against its resolved
        # value, never the literal marker string. ``raise_on_validate`` stands in for an
        # invariant the pipeline evaluates in this same gate (e.g. backend-needs-bus).
        if self.raise_on_validate is not None:
            raise self.raise_on_validate
        resolved = parse_config(data=dump_manifest(cast("Any", document))) or {}
        Manifest.model_validate(resolved)

    async def apply_replace(self, document: dict[str, Any]) -> ApplyResult:
        return self._persist(copy.deepcopy(document))

    def _persist(self, document: dict[str, Any]) -> ApplyResult:
        idx = self.calls
        self.calls += 1
        if idx in self.fail_persist_on:
            raise RuntimeError(f"manifest persist failed at call {idx}")
        self._events.append("cm:write")
        self._cm._manifest = copy.deepcopy(document)
        self.writes.append(copy.deepcopy(document))
        self._events.append("reload")
        if idx in self.fail_reload_on:
            raise FleetBroadcastError("reload_config", _fleet_report(), RuntimeError(f"reload failed at call {idx}"))
        if idx in self.fail_broadcast_on:
            raise FleetBroadcastError(
                "reload_config", _unreachable_report(), RuntimeError(f"broadcast failed at call {idx}")
            )
        return ApplyResult(fleet=_fleet_report(), local={"reloaded": True}, document=document)


class FakeFleetLock:
    def __init__(self, events: list[str], *, held: bool = False) -> None:
        self._events = events
        self.held = held

    def __call__(self):
        return self._cm()

    @asynccontextmanager
    async def _cm(self):
        if self.held:
            raise OperationInProgressError("another marketplace operation is in progress; retry shortly")
        self._events.append("lock:acquire")
        try:
            yield
        finally:
            self._events.append("lock:release")


class Harness:
    def __init__(self, **cm_kwargs) -> None:
        self.events: list[str] = []
        self.registry = FakeRegistry(self.events)
        self.pip = FakePip(self.events)
        self.store = FakeStore(self.events)
        self.fleet = FakeFleetLock(self.events)
        self.cm = FakeCM(self.events, **cm_kwargs)
        self.svc = FakeConfigService(self.events, self.cm)

    def installer(self) -> Installer:
        # The fakes duck-type the real collaborators; cast past the typed params.
        return Installer(
            registry=cast(Any, self.registry),
            pip_runner=self.pip,
            store=cast(Any, self.store),
            config_service=cast(Any, self.svc),
            fleet_lock=self.fleet,
            config_manager=self.cm,
        )


# -- install: happy ----------------------------------------------------------


async def test_install_happy_step_order_manifest_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    spec = make_spec(provides=[{"kind": "tool", "name": "gen-uuid", "module": "pkg.tools.uuid", "description": "d"}])
    h.registry.resolved = make_resolved(spec, version="1.0.0")

    result = await h.installer().install("tai42/toolbox")  # no version -> resolved pin

    # Fleet lock wraps the whole operation; resolve/pip/write/reload/record order.
    assert h.events[0] == "lock:acquire"
    assert h.events[-1] == "lock:release"
    order = [e for e in h.events if e in ("registry:resolve", "pip", "cm:write", "reload", "store:record")]
    assert order == ["registry:resolve", "pip", "cm:write", "reload", "store:record"]
    # The persisted manifest carries the patched tool entry.
    assert h.svc.writes[-1]["tools"] == [{"title": "pkg.tools.uuid", "module": "pkg.tools.uuid"}]
    # The response reports the RESOLVED version even though the caller passed None.
    assert result["version"] == "1.0.0"
    assert result["package"] == "tai-toolbox"
    # The reload field carries the local reload result plus the standard fleet fan-out
    # summary (single-worker harness => the local-only note).
    assert result["reload"]["reloaded"] is True
    assert result["reload"]["fanout"] == {
        "mode": "local-only",
        "note": "no worker bus configured; only this worker reloaded",
    }


async def test_install_config_item_contributes_a_note(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    spec = make_spec(provides=[{"kind": "config", "name": "k8s", "module": "pkg.config.k8s", "description": "d"}])
    h.registry.resolved = make_resolved(spec)
    result = await h.installer().install("tai42/toolbox")
    assert any("TAI_CONFIG_MODE" in note for note in result["notes"])


async def test_install_version_pinning_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    spec = make_spec(version="1.2.3")
    h.registry.resolved = make_resolved(spec, version="1.2.3")
    await h.installer().install("tai42/toolbox", "1.2.3")
    assert h.registry.resolve_calls == [("tai42", "toolbox", "1.2.3")]
    assert h.pip.calls[0][-1] == "tai-toolbox==1.2.3"


_GH_ARTIFACT = "https://codeload.github.com/tai42ai/toolbox/tar.gz/refs/tags/v2.0.0"
_GH_ARTIFACT_OLD = "https://codeload.github.com/tai42ai/toolbox/tar.gz/refs/tags/v1.0.0"


def _fake_verified_fetch(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch the installer's ``fetch_verified_artifact`` to record its args and
    return the local tarball path it would have written, without a real download.
    Lets a test assert the installer fetched the registry artifact_ref + sha256
    and pip-installed the LOCAL tarball."""
    calls: list[dict[str, Any]] = []

    async def fake(package, version, artifact_ref, sha256, dest_dir):
        path = dest_dir / f"{package}-{version}.tar.gz"
        calls.append(
            {"package": package, "version": version, "artifact_ref": artifact_ref, "sha256": sha256, "path": path}
        )
        return path

    monkeypatch.setattr(installer_module, "fetch_verified_artifact", fake)
    return calls


async def test_install_github_fetches_verifies_and_installs_local_tarball(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    calls = _fake_verified_fetch(monkeypatch)
    h = Harness()
    spec = make_spec(version="2.0.0")
    h.registry.resolved = make_resolved(
        spec,
        source="github",
        version="2.0.0",
        repository_url="https://github.com/tai42ai/toolbox",
        tag="v2.0.0",
        artifact_ref=_GH_ARTIFACT,
        sha256="a" * 64,
    )
    await h.installer().install("tai42/toolbox")

    # The verified fetch was driven with the registry artifact_ref + sha256.
    assert calls[0]["artifact_ref"] == _GH_ARTIFACT
    assert calls[0]["sha256"] == "a" * 64
    # pip installed the verified LOCAL tarball path, never a git+url clone.
    assert h.pip.calls[0][-1] == str(calls[0]["path"])
    assert not h.pip.calls[0][-1].startswith("git+")
    # The attribution row keeps the github provenance AND the verified pin, so a
    # later update-unwind can re-fetch and re-verify the old artifact.
    row = h.store.record_calls[-1]
    assert row[3:7] == ("https://github.com/tai42ai/toolbox", "v2.0.0", _GH_ARTIFACT, "a" * 64)


async def test_install_github_integrity_mismatch_never_calls_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    # A sha256 mismatch during the verified fetch aborts the install: no pip, no
    # manifest write, no attribution — and no git fallback.
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")

    async def mismatch(package, version, artifact_ref, sha256, dest_dir):
        raise ArtifactIntegrityError(expected_sha256="a" * 64, actual_sha256="b" * 64, artifact_ref=artifact_ref)

    monkeypatch.setattr(installer_module, "fetch_verified_artifact", mismatch)
    h = Harness()
    spec = make_spec(version="2.0.0")
    h.registry.resolved = make_resolved(
        spec, source="github", version="2.0.0", repository_url="https://github.com/tai42ai/toolbox", tag="v2.0.0"
    )
    with pytest.raises(ArtifactIntegrityError):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)
    assert h.svc.writes == []
    assert h.store.record_calls == []


async def test_install_github_fetch_failure_never_calls_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    # A download failure likewise aborts with no pip and no git fallback.
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")

    async def boom(package, version, artifact_ref, sha256, dest_dir):
        raise RuntimeError("network down")

    monkeypatch.setattr(installer_module, "fetch_verified_artifact", boom)
    h = Harness()
    spec = make_spec(version="2.0.0")
    h.registry.resolved = make_resolved(
        spec, source="github", version="2.0.0", repository_url="https://github.com/tai42ai/toolbox", tag="v2.0.0"
    )
    with pytest.raises(RuntimeError, match="network down"):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


# -- install: pre-flight refusals (pip never called) -------------------------


def _assert_no_pip(h: Harness) -> None:
    assert h.pip.calls == []


async def test_install_already_installed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness()
    spec = make_spec()
    h.store.preload(spec, version="1.0.0")
    with pytest.raises(InstallStateError, match="already installed"):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


async def test_install_no_published_version_maps_to_not_found() -> None:
    from tai_skeleton.marketplace.errors import ListingNotFoundError

    h = Harness()
    h.registry.resolve_error = ListingNotFoundError("marketplace listing not found: tai42/toolbox")
    with pytest.raises(ListingNotFoundError):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


async def test_install_killed_version_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness()
    h.registry.resolve_error = VersionRefusedError("version is killed")
    with pytest.raises(VersionRefusedError):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


async def test_install_critical_advisory_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    spec = make_spec()
    h.registry.resolved = make_resolved(
        spec, advisories=[{"severity": "critical", "withdrawn_at": None, "summary": "RCE"}]
    )
    with pytest.raises(VersionRefusedError, match="critical advisory"):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


async def test_install_noncritical_advisory_installs_and_rides_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    spec = make_spec()
    adv = [{"severity": "medium", "withdrawn_at": None, "summary": "minor"}]
    h.registry.resolved = make_resolved(spec, advisories=adv)
    result = await h.installer().install("tai42/toolbox")
    assert h.pip.calls  # installed
    assert result["advisories"] == adv


async def test_install_invalid_spec_is_registry_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness()
    spec = make_spec()
    resolved = make_resolved(spec)
    resolved["spec"] = {"not": "a valid spec"}
    h.registry.resolved = resolved
    with pytest.raises(RegistryResponseError, match="invalid plugin spec"):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


async def test_install_contract_incompatible_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "2.0.0")
    h = Harness()
    spec = make_spec()
    h.registry.resolved = make_resolved(spec, contract_range=">=0.1,<1.0")
    with pytest.raises(ContractIncompatibleError) as exc:
        await h.installer().install("tai42/toolbox")
    assert ">=0.1,<1.0" in str(exc.value)
    assert "2.0.0" in str(exc.value)
    _assert_no_pip(h)


async def test_install_dev_versioned_contract_inside_range_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    # A dev/pre-release installed contract inside the range must NOT be refused
    # (the SpecifierSet is evaluated with prereleases=True).
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.5.0.dev3")
    h = Harness()
    spec = make_spec()
    h.registry.resolved = make_resolved(spec, contract_range=">=0.1,<1.0")
    await h.installer().install("tai42/toolbox")
    assert h.pip.calls  # proceeded to install


async def test_install_manifest_collision_preflight_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness(manifest={"tools": [{"title": "x", "module": "pkg.tools.uuid"}]})
    spec = make_spec(provides=[{"kind": "tool", "name": "gen-uuid", "module": "pkg.tools.uuid", "description": "d"}])
    h.registry.resolved = make_resolved(spec)
    with pytest.raises(ManifestCollisionError):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


async def test_install_bad_ref_raises_malformed_ref_error() -> None:
    h = Harness()
    # A malformed ref is a typed MalformedRefError (mapped to a 400 at the
    # boundary), distinct from a server-side invariant fault.
    with pytest.raises(MalformedRefError, match="namespace/name"):
        await h.installer().install("noslash")


# -- install: unwind matrix --------------------------------------------------


async def test_install_pip_failure_leaves_manifest_and_store_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    h.pip.fail_on = {0}
    spec = make_spec()
    h.registry.resolved = make_resolved(spec)
    with pytest.raises(PipFailedError):
        await h.installer().install("tai42/toolbox")
    assert h.svc.writes == []
    assert h.store.record_calls == []


async def test_install_manifest_persist_failure_unwinds_with_pip_uninstall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    # The manifest persist fails BEFORE the change lands (nothing persisted), so the
    # pipeline aborts and the installer needs no manifest restore — just the pip
    # uninstall of the freshly-installed package.
    h.svc.fail_persist_on = {0}
    spec = make_spec()
    h.registry.resolved = make_resolved(spec)
    with pytest.raises(RuntimeError, match="manifest persist failed"):
        await h.installer().install("tai42/toolbox")
    # pip install then the unwind pip uninstall; nothing persisted, so no restore.
    assert h.pip.calls[0][0] == "install"
    assert h.pip.calls[1] == ["uninstall", "--yes", "tai-toolbox"]
    assert h.svc.writes == []


async def test_install_reload_failure_restores_manifest_and_uninstalls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    # The apply persists but its local reload fails afterwards (FleetBroadcastError):
    # the change landed, so the unwind must restore the manifest through the pipeline.
    h.svc.fail_reload_on = {0}
    spec = make_spec()
    h.registry.resolved = make_resolved(spec)
    with pytest.raises(FleetBroadcastError, match="reload failed"):
        await h.installer().install("tai42/toolbox")
    # Manifest patched then restored (the saved pre-patch dict = {}).
    assert h.svc.writes[-1] == {}
    # Two applies (forward + converge-back), pip install then uninstall.
    assert h.svc.calls == 2
    assert h.pip.calls[-1] == ["uninstall", "--yes", "tai-toolbox"]


async def test_install_broadcast_failure_after_persist_restores_manifest_and_uninstalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    # The apply persists and the local reload succeeds, but the FLEET BROADCAST then
    # raises. ConfigService wraps that raw broadcast fault as a FleetBroadcastError with
    # the bus-unreachable report — so the change LANDED and the installer's persist
    # detector (persisted=True on a FleetBroadcastError) MUST restore the manifest. This
    # is the exact half-write the fix closes: without the wrap the raw error would read
    # persisted=False, skip the restore, yet still pip-uninstall the package.
    h.svc.fail_broadcast_on = {0}
    spec = make_spec()
    h.registry.resolved = make_resolved(spec)
    with pytest.raises(FleetBroadcastError, match="broadcast failed"):
        await h.installer().install("tai42/toolbox")
    # Manifest patched then RESTORED to the saved pre-patch dict ({}): no half-write.
    assert h.svc.writes[-1] == {}
    # Two applies (forward patch + converge-back restore), and the freshly-installed
    # package is pip-uninstalled — the manifest never references a pip-removed package.
    assert h.svc.calls == 2
    assert h.pip.calls[-1] == ["uninstall", "--yes", "tai-toolbox"]


async def test_install_store_failure_full_unwind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    h.store.record_error = RuntimeError("pg down")
    spec = make_spec()
    h.registry.resolved = make_resolved(spec)
    with pytest.raises(RuntimeError, match="pg down"):
        await h.installer().install("tai42/toolbox")
    assert h.svc.writes[-1] == {}  # restored
    assert h.svc.calls == 2  # restore apply (converge-back)
    assert h.pip.calls[-1] == ["uninstall", "--yes", "tai-toolbox"]


async def test_install_unwind_substep_failure_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    h.svc.fail_reload_on = {0}  # forward apply persists then its reload fails
    h.pip.fail_on = {1}  # the unwind pip uninstall also fails
    spec = make_spec()
    h.registry.resolved = make_resolved(spec)
    with pytest.raises(InstallUnwindError) as exc:
        await h.installer().install("tai42/toolbox")
    # FleetBroadcastError is a RuntimeError; the restore apply succeeded, so the
    # escalating sub-step is the failing pip uninstall.
    assert isinstance(exc.value.step_error, FleetBroadcastError)
    assert isinstance(exc.value.unwind_error, PipFailedError)


async def test_install_compose_failure_is_manifest_compose_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A composed manifest whose RESOLVED projection fails the schema is a registry-spec
    # + local-manifest fault: the pipeline raises inside the transaction (nothing
    # persists) and the installer maps that ValidationError to the typed compose error
    # (a 500), then unwinds the pip install. Here a pre-existing api_tools entry carries
    # a non-bool literal, so the resolved compose fails Manifest.model_validate.
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness(manifest={"api_tools": {"expose_destructive": "not-a-bool"}})
    spec = make_spec()
    h.registry.resolved = make_resolved(spec)
    with pytest.raises(ManifestComposeError):
        await h.installer().install("tai42/toolbox")
    # Nothing persisted; the freshly-installed package is unwound.
    assert h.svc.writes == []
    assert h.pip.calls[0][0] == "install"
    assert h.pip.calls[-1] == ["uninstall", "--yes", "tai-toolbox"]


async def test_install_env_marker_on_non_string_field_validates_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    # A composed manifest that sets a NON-STRING field via !ENV must validate against
    # the RESOLVED value, not the literal marker string. The pipeline materializes the
    # marker before Manifest.model_validate, so the install succeeds and the preserved
    # !ENV marker persists verbatim — the resolved bool never bakes in. Validating the
    # PRESERVED document instead would reject "!ENV ${EXPOSE_DESTRUCTIVE}" as an invalid
    # bool and 500 the install even though the resolved manifest is valid.
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    monkeypatch.setenv("EXPOSE_DESTRUCTIVE", "true")
    marker = "!ENV ${EXPOSE_DESTRUCTIVE}"
    h = Harness(manifest={"api_tools": {"expose_destructive": marker}})
    spec = make_spec(provides=[{"kind": "tool", "name": "gen-uuid", "module": "pkg.tools.uuid", "description": "d"}])
    h.registry.resolved = make_resolved(spec, version="1.0.0")

    result = await h.installer().install("tai42/toolbox")

    # The resolved expose_destructive=true is a valid bool, so the install succeeded; the
    # persisted manifest carries both the patched tool AND the untouched !ENV marker.
    persisted = h.svc.writes[-1]
    assert persisted["tools"] == [{"title": "pkg.tools.uuid", "module": "pkg.tools.uuid"}]
    assert persisted["api_tools"]["expose_destructive"] == marker
    # One apply persisted + reloaded + broadcast, and the attribution row was written.
    assert h.svc.calls == 1
    assert result["reload"]["reloaded"] is True
    assert h.store.record_calls[-1][0] == "tai42/toolbox"


async def test_install_backend_needs_bus_maps_to_manifest_compose_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A plugin that registers a backend on a busless deployment fails the pipeline's
    # backend-needs-bus invariant. The marketplace surface translates only its own error
    # family, so the installer maps that refusal to the typed compose error — a loud,
    # attributed 500 carrying the "Set TAI_BUS_REDIS_URL" message — rather than letting
    # the RuntimeError escape untyped. Nothing persists; the pip install unwinds.
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    h.svc.raise_on_validate = BackendNeedsBusError(
        "Refusing a config that registers a task backend ('pkg.backend') with no worker bus. Set TAI_BUS_REDIS_URL."
    )
    spec = make_spec(provides=[{"kind": "backend", "name": "worker", "module": "pkg.backend", "description": "d"}])
    h.registry.resolved = make_resolved(spec)
    with pytest.raises(ManifestComposeError, match="TAI_BUS_REDIS_URL"):
        await h.installer().install("tai42/toolbox")
    assert h.svc.writes == []
    assert h.pip.calls[-1] == ["uninstall", "--yes", "tai-toolbox"]


# -- uninstall ---------------------------------------------------------------


async def test_uninstall_happy_path() -> None:
    h = Harness(manifest={"tools": [{"title": "pkg.tools.uuid", "module": "pkg.tools.uuid"}]})
    spec = make_spec(provides=[{"kind": "tool", "name": "gen-uuid", "module": "pkg.tools.uuid", "description": "d"}])
    h.store.preload(spec, version="1.0.0")
    result = await h.installer().uninstall("tai42/toolbox")
    order = [e for e in h.events if e in ("cm:write", "reload", "pip", "store:delete")]
    assert order == ["cm:write", "reload", "pip", "store:delete"]
    assert h.pip.calls[-1] == ["uninstall", "--yes", "tai-toolbox"]
    assert result["uninstalled"] is True
    assert "tai42/toolbox" not in h.store.rows
    # Never touches the registry.
    assert h.registry.resolve_calls == []


async def test_uninstall_unknown_ref_is_not_installed() -> None:
    h = Harness()
    with pytest.raises(InstallStateError) as exc:
        await h.installer().uninstall("tai42/gone")
    assert exc.value.not_installed is True


async def test_uninstall_reloads_to_convergence_when_manifest_already_clean() -> None:
    # A row exists but the manifest has no matching entry (a re-run after a partial
    # uninstall whose deregister reload failed): the spec still targets manifest
    # fields, so the pipeline apply is RE-ATTEMPTED — an idempotent re-strip that
    # re-persists the (already clean) manifest and reloads so the still-live tools
    # are deregistered before the package is pip-uninstalled and the row is dropped.
    h = Harness(manifest={})
    spec = make_spec(provides=[{"kind": "tool", "name": "gen-uuid", "module": "pkg.tools.uuid", "description": "d"}])
    h.store.preload(spec, version="1.0.0")
    result = await h.installer().uninstall("tai42/toolbox")
    # The apply re-persists the (unchanged) empty manifest and reloads through the pipeline.
    assert h.svc.writes == [{}]
    assert h.svc.calls == 1
    assert result["reload"]["reloaded"] is True
    # Convergence order: the apply (persist + deregister reload) BEFORE pip uninstall + row delete.
    order = [e for e in h.events if e in ("cm:write", "reload", "pip", "store:delete")]
    assert order == ["cm:write", "reload", "pip", "store:delete"]
    assert h.pip.calls[-1] == ["uninstall", "--yes", "tai-toolbox"]


async def test_uninstall_env_selected_only_plugin_takes_skip_path() -> None:
    h = Harness(manifest={})
    spec = make_spec(provides=[{"kind": "config", "name": "k8s", "module": "pkg.config.k8s", "description": "d"}])
    h.store.preload(spec, version="1.0.0")
    result = await h.installer().uninstall("tai42/toolbox")
    assert result["reload"] is None
    # The config provider gets a loud note about TAI_CONFIG_MODE.
    assert any("TAI_CONFIG_MODE" in note for note in result["notes"])


async def test_uninstall_corrupt_local_row_is_local_state_error() -> None:
    h = Harness()
    # A stored spec that no longer validates is corrupt LOCAL state -> 500, not 400.
    h.store.rows["tai42/toolbox"] = InstallRecord(
        ref="tai42/toolbox",
        version="1.0.0",
        source="pypi",
        repository_url=None,
        tag=None,
        spec={"garbage": True},
        installed_at=datetime.now(UTC),
    )
    with pytest.raises(LocalStateError):
        await h.installer().uninstall("tai42/toolbox")


# -- update ------------------------------------------------------------------


async def test_update_happy_replaces_row_in_one_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    old = make_spec(version="1.0.0", provides=_tool_provides("pkg.old"))
    new = make_spec(version="2.0.0", provides=_tool_provides("pkg.new"))
    h = Harness(manifest={"tools": [{"title": "pkg.old", "module": "pkg.old"}]})
    h.store.preload(old, version="1.0.0")
    h.registry.resolved = make_resolved(new, version="2.0.0")

    result = await h.installer().update("tai42/toolbox")

    assert result["version"] == "2.0.0"
    # One RMW: old removed, new applied.
    assert h.svc.writes[-1]["tools"] == [{"title": "pkg.new", "module": "pkg.new"}]
    assert h.store.record_calls[-1][1] == "2.0.0"


async def test_update_same_version_is_state_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    spec = make_spec(version="1.0.0")
    h = Harness()
    h.store.preload(spec, version="1.0.0")
    h.registry.resolved = make_resolved(spec, version="1.0.0")
    with pytest.raises(InstallStateError, match=r"already at 1\.0\.0"):
        await h.installer().update("tai42/toolbox")


async def test_update_unknown_ref_is_not_installed() -> None:
    h = Harness()
    with pytest.raises(InstallStateError) as exc:
        await h.installer().update("tai42/gone")
    assert exc.value.not_installed is True


async def test_update_bad_ref_raises_malformed_ref_error() -> None:
    # A malformed ref is parsed BEFORE the store read, so it is the caller's typed
    # MalformedRefError (a 400) — matching install — never a phantom 404 from a
    # not-installed lookup on an unparseable ref.
    h = Harness()
    with pytest.raises(MalformedRefError, match="namespace/name"):
        await h.installer().update("noslash")


async def test_update_pipless_fails_before_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_pip() -> None:
        raise PipUnavailableError("no pip")

    monkeypatch.setattr(installer_module, "ensure_pip_available", _no_pip)
    spec = make_spec(version="1.0.0")
    h = Harness()
    h.store.preload(spec, version="1.0.0")
    with pytest.raises(PipUnavailableError):
        await h.installer().update("tai42/toolbox")
    # step 0 pre-flight fires before resolve counts a download.
    assert h.registry.resolve_calls == []


async def test_update_unwind_reinstalls_old_github_pin_through_verified_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    calls = _fake_verified_fetch(monkeypatch)
    old = make_spec(version="1.0.0")
    new = make_spec(version="2.0.0")
    h = Harness(manifest={"tools": [{"title": "pkg.tools.gen_uuid", "module": "pkg.tools.gen_uuid"}]})
    # The stored row is a github pin carrying the OLD artifact_ref + sha256, so the
    # unwind re-fetches and re-verifies exactly that old artifact.
    h.store.preload(
        old,
        version="1.0.0",
        source="github",
        repository_url="https://github.com/tai42ai/toolbox",
        tag="v1.0.0",
        artifact_ref=_GH_ARTIFACT_OLD,
        sha256="1" * 64,
    )
    h.registry.resolved = make_resolved(
        new,
        source="github",
        version="2.0.0",
        repository_url="https://github.com/tai42ai/toolbox",
        tag="v2.0.0",
        artifact_ref=_GH_ARTIFACT,
        sha256="2" * 64,
    )
    h.svc.fail_reload_on = {0}  # the update apply persists then its reload fails -> unwind

    with pytest.raises(FleetBroadcastError, match="reload failed"):
        await h.installer().update("tai42/toolbox")

    # Verified fetch: first the NEW pin (from resolve), then the OLD pin (from the
    # stored row's artifact_ref + sha256), never a mutable git+url clone.
    assert (calls[0]["artifact_ref"], calls[0]["sha256"]) == (_GH_ARTIFACT, "2" * 64)
    assert (calls[1]["artifact_ref"], calls[1]["sha256"]) == (_GH_ARTIFACT_OLD, "1" * 64)
    # pip installed each verified LOCAL tarball, in that order.
    assert h.pip.calls[0][-1] == str(calls[0]["path"])
    assert h.pip.calls[1][-1] == str(calls[1]["path"])
    assert not h.pip.calls[1][-1].startswith("git+")
    # Ordering: old-pin reinstall -> manifest restore apply (persist + reload-back).
    # The old wheel is back BEFORE the restore's reload loads the old manifest.
    tail = [e for e in h.events if e in ("cm:write", "pip", "reload")]
    # ... write(new), reload(fail), pip(old), write(saved), reload(back)
    assert tail[-3:] == ["pip", "cm:write", "reload"]


# -- update: collision pre-flight --------------------------------------------


async def test_update_same_module_rename_does_not_self_collide(monkeypatch: pytest.MonkeyPatch) -> None:
    # The new spec re-provides the SAME module the old spec already wrote to the
    # manifest (a version bump that keeps the module path). The pre-flight removes
    # the OLD spec's entries in memory BEFORE the collision check, so the shared
    # module must NOT be read as a self-collision — the update proceeds to pip.
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    old = make_spec(version="1.0.0", provides=_tool_provides("pkg.same"))
    new = make_spec(version="2.0.0", provides=_tool_provides("pkg.same"))
    h = Harness(manifest={"tools": [{"title": "pkg.same", "module": "pkg.same"}]})
    h.store.preload(old, version="1.0.0")
    h.registry.resolved = make_resolved(new, version="2.0.0")

    result = await h.installer().update("tai42/toolbox")

    assert result["version"] == "2.0.0"
    assert h.pip.calls  # proceeded past the (non-)collision to the pip upgrade
    # One RMW keeps the single shared-module entry (removed then re-applied).
    assert h.svc.writes[-1]["tools"] == [{"title": "pkg.same", "module": "pkg.same"}]


async def test_update_genuine_collision_refuses_before_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    # The new spec's module collides with a FOREIGN manifest entry (one that does
    # not belong to the old spec, so removing the old entries does not clear it).
    # That is a real update-side collision: a 409 raised BEFORE any pip call.
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    old = make_spec(version="1.0.0", provides=_tool_provides("pkg.old"))
    new = make_spec(version="2.0.0", provides=_tool_provides("pkg.foreign"))
    h = Harness(
        manifest={
            "tools": [
                {"title": "pkg.old", "module": "pkg.old"},
                {"title": "other-plugin", "module": "pkg.foreign"},
            ]
        }
    )
    h.store.preload(old, version="1.0.0")
    h.registry.resolved = make_resolved(new, version="2.0.0")
    with pytest.raises(ManifestCollisionError, match=r"pkg\.foreign"):
        await h.installer().update("tai42/toolbox")
    _assert_no_pip(h)


# -- update: unwind escalation -----------------------------------------------


async def test_update_unwind_reload_back_failure_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    # The update apply (call 0) persists then its reload fails, triggering the unwind;
    # the unwind's own restore apply (call 1) then fails its reload too. A failed
    # unwind sub-step escalates to InstallUnwindError carrying both the original step
    # error and the unwind error.
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    old = make_spec(version="1.0.0", provides=_tool_provides("pkg.old"))
    new = make_spec(version="2.0.0", provides=_tool_provides("pkg.new"))
    h = Harness(manifest={"tools": [{"title": "pkg.old", "module": "pkg.old"}]})
    h.store.preload(old, version="1.0.0")
    h.registry.resolved = make_resolved(new, version="2.0.0")
    h.svc.fail_reload_on = {0, 1}  # forward update apply AND the unwind restore apply both fail their reload

    with pytest.raises(InstallUnwindError) as exc:
        await h.installer().update("tai42/toolbox")
    assert isinstance(exc.value.step_error, FleetBroadcastError)
    assert isinstance(exc.value.unwind_error, FleetBroadcastError)
    # The old pin WAS reinstalled before the failing restore apply (unwind reached it).
    assert h.pip.calls[-1][0] == "install"


# -- install: unwind reload-back escalation ----------------------------------


async def test_install_unwind_reload_back_failure_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    # The forward apply (call 0) persists then its reload fails, triggering the
    # unwind; the unwind's restore apply (call 1) then fails its reload too. That
    # failed sub-step escalates to InstallUnwindError before the pip uninstall is
    # ever attempted.
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    h.svc.fail_reload_on = {0, 1}  # forward apply AND the unwind restore apply both fail their reload
    spec = make_spec()
    h.registry.resolved = make_resolved(spec)
    with pytest.raises(InstallUnwindError) as exc:
        await h.installer().install("tai42/toolbox")
    assert isinstance(exc.value.step_error, FleetBroadcastError)
    assert isinstance(exc.value.unwind_error, FleetBroadcastError)
    # The manifest restore apply failed, so the pip uninstall was never reached.
    assert [c[0] for c in h.pip.calls] == ["install"]


# -- resolve-response guards (garbled/compromised registry data) -------------


@pytest.mark.parametrize(
    ("key", "absent"),
    [
        ("source", True),  # a required field entirely absent
        ("version", False),  # present-but-null
        ("contract_range", False),  # present-but-null
    ],
)
async def test_install_resolve_missing_or_null_required_field_is_registry_response_error(
    key: str, absent: bool
) -> None:
    # A required resolve field that is absent OR present-but-null is garbled upstream
    # data (a 502 at the boundary), never a caller error, and ``_require`` catches it
    # before any pip call. Null matters specially: the client's resolve boundary
    # type-checks a field only when it is present and non-null (a null is legitimate
    # for the github-only optional pins), so a null ``version`` / ``contract_range``
    # would otherwise reach ``Version(None)`` / ``SpecifierSet(None)`` — a ``TypeError``,
    # not a typed parse error — and escape the operation boundary as an untyped 500.
    h = Harness()
    spec = make_spec()
    resolved = make_resolved(spec)
    if absent:
        del resolved[key]
    else:
        resolved[key] = None
    h.registry.resolved = resolved
    with pytest.raises(RegistryResponseError, match=key):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


async def test_install_resolve_advisories_not_list_is_registry_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness()
    spec = make_spec()
    resolved = make_resolved(spec)
    resolved["advisories"] = {"not": "a list"}
    h.registry.resolved = resolved
    with pytest.raises(RegistryResponseError, match="advisories"):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


async def test_install_unknown_source_is_registry_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness()
    spec = make_spec()
    h.registry.resolved = make_resolved(spec, source="svn")
    with pytest.raises(RegistryResponseError, match="unknown install source"):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


async def test_install_github_missing_provenance_is_registry_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A github source with no repository_url/tag cannot be pinned — garbled
    # upstream data, refused before pip.
    h = Harness()
    spec = make_spec()
    h.registry.resolved = make_resolved(spec, source="github", repository_url=None, tag=None)
    with pytest.raises(RegistryResponseError, match="repository_url or tag"):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


async def test_install_github_missing_artifact_ref_is_registry_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A github source with provenance but no artifact_ref/sha256 cannot be
    # verified — garbled upstream data, refused before any fetch or pip.
    h = Harness()
    spec = make_spec()
    resolved = make_resolved(spec, source="github", repository_url="https://github.com/tai42ai/toolbox", tag="v1.0.0")
    resolved["artifact_ref"] = None
    resolved["sha256"] = None
    h.registry.resolved = resolved
    with pytest.raises(RegistryResponseError, match="artifact_ref or sha256"):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


async def test_install_malformed_contract_range_is_registry_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ``contract_range`` string that does not parse as a specifier set is
    # garbled registry data → RegistryResponseError (a 502), never a caller 400.
    # A non-STRING value never reaches this parse — the registry client's resolve
    # boundary types the field. The refusal is before pip runs.
    h = Harness()
    spec = make_spec()
    h.registry.resolved = make_resolved(spec, contract_range="not a specifier!!")
    with pytest.raises(RegistryResponseError, match="unusable contract_range"):
        await h.installer().install("tai42/toolbox")
    _assert_no_pip(h)


# -- ref normalization -------------------------------------------------------


async def test_install_mixed_case_ref_is_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ref with the right shape (one slash, two non-empty halves) but an uppercase
    # half is still malformed — refs are lowercase 'namespace/name'.
    h = Harness()
    with pytest.raises(MalformedRefError, match="lowercase"):
        await h.installer().install("Tai42/Toolbox")
    _assert_no_pip(h)


# -- concurrency -------------------------------------------------------------


async def test_operation_in_progress_refuses_second_same_worker_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer_module.importlib.metadata, "version", lambda name: "0.1.0")
    h = Harness()
    spec = make_spec()
    h.registry.resolved = make_resolved(spec)
    # The per-worker fast path: while the operation lock is held, a second
    # same-worker call is refused immediately by _guard, before any work.
    async with installer_module._operation_lock:
        assert installer_module._operation_lock.locked() is True
        with pytest.raises(OperationInProgressError):
            await h.installer().install("tai42/toolbox")
    assert installer_module._operation_lock.locked() is False


async def test_fleet_lock_held_elsewhere_touches_nothing() -> None:
    h = Harness()
    h.fleet.held = True
    spec = make_spec()
    h.registry.resolved = make_resolved(spec)
    with pytest.raises(OperationInProgressError):
        await h.installer().install("tai42/toolbox")
    # No store/registry/pip/manifest call was made.
    assert h.registry.resolve_calls == []
    assert h.pip.calls == []
    assert h.store.record_calls == []
    assert h.svc.writes == []
    assert "store:get" not in h.events


# -- _fleet_lock internals ---------------------------------------------------


class _RecordingCursor:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql: str, params=None) -> None:
        self._log.append(" ".join(sql.split()))

    async def fetchone(self):
        return (True,)


class _RecordingConn:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.autocommit_before: list[str] = []

    async def set_autocommit(self, value: bool) -> None:
        # Snapshot the statements seen so far to prove autocommit precedes the try-lock.
        self._log.append(f"autocommit={value}")

    def cursor(self):
        return _RecordingCursor(self._log)


class _RecordingPool:
    def __init__(self, log: list[str], closed: list[bool]) -> None:
        self._log = log
        self._closed = closed

    @asynccontextmanager
    async def connection(self):
        yield _RecordingConn(self._log)


def _patch_fleet_client(monkeypatch: pytest.MonkeyPatch, log: list[str], closed: list[bool]):
    @asynccontextmanager
    async def fake_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        assert fresh is True  # the one-shot dedicated client
        assert kwargs.get("min_size") == 1
        assert kwargs.get("max_size") == 1
        try:
            yield _RecordingPool(log, closed)
        finally:
            closed.append(True)  # the fresh path closes on ANY exit

    monkeypatch.setattr(installer_module, "client_ctx", fake_ctx)

    class _Settings:
        def client_kwargs(self):
            return {}

    monkeypatch.setattr(installer_module, "marketplace_store_settings", lambda: _Settings())


async def test_fleet_lock_autocommit_before_trylock_and_unlock_in_finally(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []
    closed: list[bool] = []
    _patch_fleet_client(monkeypatch, log, closed)
    async with installer_module._fleet_lock():
        pass
    # autocommit set BEFORE the try-lock; unlock issued in the finally; client closed.
    assert log[0] == "autocommit=True"
    assert "pg_try_advisory_lock" in log[1]
    assert any("pg_advisory_unlock" in s for s in log)
    assert closed == [True]


async def test_fleet_lock_closes_on_body_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []
    closed: list[bool] = []
    _patch_fleet_client(monkeypatch, log, closed)
    with pytest.raises(RuntimeError, match="boom"):
        async with installer_module._fleet_lock():
            raise RuntimeError("boom")
    assert closed == [True]  # the lock connection never outlives the context


async def test_fleet_lock_closes_on_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []
    closed: list[bool] = []
    _patch_fleet_client(monkeypatch, log, closed)
    with pytest.raises(asyncio.CancelledError):
        async with installer_module._fleet_lock():
            raise asyncio.CancelledError
    assert closed == [True]


async def test_fleet_lock_refuses_when_lock_held_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []
    closed: list[bool] = []

    class _FalseCursor(_RecordingCursor):
        async def fetchone(self):
            return (False,)  # pg_try_advisory_lock returned false

    @asynccontextmanager
    async def fake_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        class _Conn(_RecordingConn):
            def cursor(self):
                return _FalseCursor(log)

        class _Pool:
            @asynccontextmanager
            async def connection(self):
                yield _Conn(log)

        try:
            yield _Pool()
        finally:
            closed.append(True)

    monkeypatch.setattr(installer_module, "client_ctx", fake_ctx)

    class _Settings:
        def client_kwargs(self):
            return {}

    monkeypatch.setattr(installer_module, "marketplace_store_settings", lambda: _Settings())

    with pytest.raises(OperationInProgressError):
        async with installer_module._fleet_lock():
            pass
    assert closed == [True]
