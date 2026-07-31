"""The marketplace routes as thin adapters over their operations.

Each handler is called directly with a built request; the operation-layer
constructions (``RegistryClient``/``Installer``/``MarketplaceInstallStore``/the
advisory module) are monkeypatched in the ``operations.marketplace`` namespace,
which is the per-request construction seam. The tests pin the envelope, the
search param forwarding (repeated ``tags`` via ``getlist``), the detail compose,
the installed-inventory per-row ``missing_upstream`` handling, and the error→
status mapping (502/404/409/500 and the retriable 503s).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.requests import Request

import tai42_skeleton.operations.marketplace as mkt_ops
import tai42_skeleton.routers.marketplace as router
from tai42_skeleton.app.reload_gate import reload_gate
from tai42_skeleton.marketplace.advisories import AdvisoryState
from tai42_skeleton.marketplace.compat import CompatVerdict
from tai42_skeleton.marketplace.errors import (
    ArtifactIntegrityError,
    ContractIncompatibleError,
    EnvironmentShadowError,
    InstallStateError,
    InstallUnwindError,
    ListingNotFoundError,
    LocalStateError,
    MalformedRefError,
    ManifestCollisionError,
    ManifestComposeError,
    OperationInProgressError,
    PipFailedError,
    PipUnavailableError,
    RegistryUnreachableError,
    VersionRefusedError,
)
from tai42_skeleton.marketplace.store import InstallRecord
from tai42_skeleton.plugins.quarantine import quarantine_plugin, reset_quarantine


def _get(path: str = "/", query: bytes = b"", **path_params: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "query_string": query,
        "headers": [],
        "path_params": path_params,
    }
    return Request(scope)


def _post(body: Any) -> Request:
    async def _json() -> Any:
        return body

    return cast(Request, SimpleNamespace(json=_json, path_params={}, query_params={}, headers={}))


def _data(resp) -> Any:
    return json.loads(bytes(resp.body))


def _record(ref: str, version: str = "1.0.0") -> InstallRecord:
    return InstallRecord(
        ref=ref,
        version=version,
        source="pypi",
        repository_url=None,
        tag=None,
        spec={},
        installed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class _FakeRegistry:
    def __init__(self, **behaviour: Any) -> None:
        self._b = behaviour
        self.search_params: Any = None

    async def search(self, params):
        self.search_params = params
        return self._b.get("search", {"items": []})

    async def plugin(self, ns, name):
        result = self._b["plugin"](ns, name) if callable(self._b.get("plugin")) else self._b.get("plugin", {})
        if isinstance(result, Exception):
            raise result
        return result

    async def versions(self, ns, name):
        result = self._b["versions"](ns, name) if callable(self._b.get("versions")) else self._b.get("versions", [])
        if isinstance(result, Exception):
            raise result
        return result

    async def categories(self):
        return self._b.get("categories", [])

    async def kinds(self):
        return self._b.get("kinds", [])


def _use_registry(monkeypatch: pytest.MonkeyPatch, registry: _FakeRegistry) -> None:
    monkeypatch.setattr(mkt_ops, "RegistryClient", lambda *a, **k: registry)


def _use_store(monkeypatch: pytest.MonkeyPatch, rows: list[InstallRecord]) -> None:
    class _Store:
        async def list_installed(self):
            return rows

    monkeypatch.setattr(mkt_ops, "MarketplaceInstallStore", lambda: _Store())


def _use_installer(monkeypatch: pytest.MonkeyPatch, *, install: Callable[[str, str | None], Awaitable[Any]]) -> None:
    class _Installer:
        async def install(self, ref, version=None):
            return await install(ref, version)

        async def uninstall(self, ref):
            return await install(ref, None)

        async def update(self, ref, version=None):
            return await install(ref, version)

        async def upgrade_all(self):
            return await install("--all--", None)

    monkeypatch.setattr(mkt_ops, "Installer", lambda *a, **k: _Installer())


# -- search ------------------------------------------------------------------


async def test_search_forwards_params_and_wraps_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _FakeRegistry(search={"items": [{"ref": "tai42/toolbox"}]})
    _use_registry(monkeypatch, registry)
    resp = await router.marketplace_search(_get(query=b"q=uuid&tags=a&tags=b&tier=official&page_size=5"))
    assert resp.status_code == 200
    assert _data(resp) == {"data": {"items": [{"ref": "tai42/toolbox"}]}}
    # Repeated tags survive multi-value; single facets forwarded; unknown dropped.
    assert registry.search_params["tags"] == ["a", "b"]
    assert registry.search_params["q"] == "uuid"
    assert registry.search_params["tier"] == "official"
    assert registry.search_params["page_size"] == "5"


async def test_search_maps_upstream_failure_to_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom(_FakeRegistry):
        async def search(self, params):
            raise RegistryUnreachableError("registry down at https://reg")

    _use_registry(monkeypatch, _Boom())
    resp = await router.marketplace_search(_get())
    assert resp.status_code == 502
    assert "registry down" in _data(resp)["error"]


# -- detail ------------------------------------------------------------------


async def test_detail_composes_versions_into_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _FakeRegistry(plugin={"ref": "tai42/toolbox", "display_name": "T"}, versions=[{"version": "1.0.0"}])
    _use_registry(monkeypatch, registry)
    resp = await router.marketplace_plugin_detail(_get(ns="tai42", name="toolbox"))
    body = _data(resp)["data"]
    assert body["display_name"] == "T"
    assert body["versions"] == [{"version": "1.0.0"}]


async def test_detail_unknown_listing_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _FakeRegistry(plugin=ListingNotFoundError("marketplace listing not found: tai42/nope"))
    _use_registry(monkeypatch, registry)
    resp = await router.marketplace_plugin_detail(_get(ns="tai42", name="nope"))
    assert resp.status_code == 404


# -- categories --------------------------------------------------------------


async def test_categories_proxies_the_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_registry(monkeypatch, _FakeRegistry(categories=["dev", "data"]))
    resp = await router.marketplace_categories(_get())
    assert _data(resp) == {"data": ["dev", "data"]}


# -- kinds -------------------------------------------------------------------


async def test_kinds_proxies_the_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_registry(monkeypatch, _FakeRegistry(kinds=["tool", "agent"]))
    resp = await router.marketplace_kinds(_get())
    assert _data(resp) == {"data": ["tool", "agent"]}


# -- installed ---------------------------------------------------------------


def _published(version: str, contract_range: str | None = ">=0.3,<0.4") -> dict[str, Any]:
    return {"version": version, "status": "published", "contract_range": contract_range}


@pytest.fixture(autouse=True)
def _pin_contract(monkeypatch: pytest.MonkeyPatch):
    # The installed listing computes against the RUNNING contract; pin it so the
    # tests are independent of the environment's tai42-contract version. The
    # process-global quarantine registry is reset around each test too — boot
    # passes other tests run would otherwise leak entries into ``quarantined``.
    monkeypatch.setattr(mkt_ops, "running_contract_version", lambda: "0.3.0")
    reset_quarantine()
    yield
    reset_quarantine()


async def test_installed_happy_computes_update_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, [_record("tai42/toolbox", "1.0.0")])
    _use_registry(monkeypatch, _FakeRegistry(versions=[_published("2.0.0"), _published("1.0.0")]))
    resp = await router.marketplace_installed(_get())
    body = _data(resp)["data"]
    row = body["installed"][0]
    assert row["latest"] == "2.0.0"
    assert row["update_available"] is True
    assert row["incompatible_newer"] is None
    assert row["missing_upstream"] is False
    # A row whose stored spec names no package cannot be verdicted — surfaced
    # as unknown, never a silent "compatible".
    assert row["compat"]["status"] == "unknown"
    assert body["quarantined"] == []


async def test_installed_incompatible_newer_never_advertises_as_update(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2.0.0 needs a different contract: it must NOT drive update_available, but
    # it must be named in incompatible_newer — visible, never silently dropped.
    _use_store(monkeypatch, [_record("tai42/toolbox", "1.0.0")])
    _use_registry(
        monkeypatch,
        _FakeRegistry(versions=[_published("2.0.0", ">=0.4,<0.5"), _published("1.5.0"), _published("1.0.0")]),
    )
    resp = await router.marketplace_installed(_get())
    row = _data(resp)["data"]["installed"][0]
    assert row["latest"] == "2.0.0"
    assert row["update_available"] is True  # 1.5.0 is the compatible target
    assert row["incompatible_newer"] == "2.0.0"


async def test_installed_row_with_vanished_upstream_is_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    def _versions(ns, name):
        if name == "gone":
            return ListingNotFoundError("marketplace listing not found: tai42/gone")
        return [_published("2.0.0")]

    _use_store(monkeypatch, [_record("tai42/gone", "1.0.0"), _record("tai42/live", "1.0.0")])
    _use_registry(monkeypatch, _FakeRegistry(versions=_versions))
    resp = await router.marketplace_installed(_get())
    rows = {r["ref"]: r for r in _data(resp)["data"]["installed"]}
    assert rows["tai42/gone"]["missing_upstream"] is True
    assert rows["tai42/gone"]["latest"] is None
    assert rows["tai42/live"]["latest"] == "2.0.0"  # other rows intact


async def test_installed_zero_published_versions_is_not_a_500(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, [_record("tai42/toolbox", "1.0.0")])
    _use_registry(monkeypatch, _FakeRegistry(versions=[{"version": "2.0.0", "status": "yanked"}]))
    resp = await router.marketplace_installed(_get())
    row = _data(resp)["data"]["installed"][0]
    assert row["latest"] is None
    assert row["update_available"] is False
    assert row["missing_upstream"] is False  # the listing is there, just unpublished


async def test_installed_transport_failure_still_502(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, [_record("tai42/toolbox", "1.0.0")])
    _use_registry(monkeypatch, _FakeRegistry(versions=RegistryUnreachableError("down")))
    resp = await router.marketplace_installed(_get())
    assert resp.status_code == 502


async def test_installed_non_pep440_published_version_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    # A registry-supplied published version that is not PEP 440 is garbled
    # upstream data — a 502, never a raw this-server 500 from the version compare.
    _use_store(monkeypatch, [_record("tai42/toolbox", "1.0.0")])
    _use_registry(monkeypatch, _FakeRegistry(versions=[_published("not-a-version")]))
    resp = await router.marketplace_installed(_get())
    assert resp.status_code == 502
    assert "error" in _data(resp)


async def test_installed_non_string_published_version_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ``version`` that is not even a string (an int here) makes ``Version()``
    # raise TypeError — NOT InvalidVersion — so the compare would otherwise
    # escape as a raw this-server 500. Garbled upstream data is a 502.
    _use_store(monkeypatch, [_record("tai42/toolbox", "1.0.0")])
    _use_registry(monkeypatch, _FakeRegistry(versions=[_published(123)]))  # pyright: ignore[reportArgumentType]
    resp = await router.marketplace_installed(_get())
    assert resp.status_code == 502
    assert "error" in _data(resp)


async def test_installed_malformed_contract_range_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, [_record("tai42/toolbox", "1.0.0")])
    _use_registry(monkeypatch, _FakeRegistry(versions=[_published("2.0.0", "not a specifier !!")]))
    resp = await router.marketplace_installed(_get())
    assert resp.status_code == 502
    assert "error" in _data(resp)


async def test_installed_garbled_local_version_is_500(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unparsable STORED version is corrupt local state — a 500, never blamed
    # on the registry as a 502.
    _use_store(monkeypatch, [_record("tai42/toolbox", "not-a-version")])
    _use_registry(monkeypatch, _FakeRegistry(versions=[_published("2.0.0")]))
    resp = await router.marketplace_installed(_get())
    assert resp.status_code == 500
    assert "error" in _data(resp)


async def test_installed_serves_per_row_compat_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    record = InstallRecord(
        ref="tai42/toolbox",
        version="1.0.0",
        source="pypi",
        spec={"package": "acme-toolbox"},
        installed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    _use_store(monkeypatch, [record])
    _use_registry(monkeypatch, _FakeRegistry(versions=[_published("1.0.0")]))
    seen: list[str] = []

    def _dist_compat(dist_name: str):
        seen.append(dist_name)
        return CompatVerdict("incompatible", "acme-toolbox requires tai42-contract <0.3, but 0.3.0 is running")

    monkeypatch.setattr(mkt_ops, "dist_compat", _dist_compat)
    resp = await router.marketplace_installed(_get())
    row = _data(resp)["data"]["installed"][0]
    assert seen == ["acme-toolbox"]  # verdicted by the stored spec's package dist
    assert row["compat"]["status"] == "incompatible"
    assert "requires tai42-contract" in row["compat"]["reason"]


async def test_installed_surfaces_the_boot_quarantine(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_store(monkeypatch, [])
    _use_registry(monkeypatch, _FakeRegistry())
    reset_quarantine()
    try:
        quarantine_plugin("acme_plugin", "tools module failed to import: boom")
        resp = await router.marketplace_installed(_get())
    finally:
        reset_quarantine()
    body = _data(resp)["data"]
    assert body["installed"] == []
    assert body["quarantined"] == [{"name": "acme_plugin", "reason": "tools module failed to import: boom"}]


# -- advisories --------------------------------------------------------------


async def test_advisories_serves_the_cache_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched = datetime(2026, 6, 1, tzinfo=UTC)

    class _Adv:
        async def current(self, max_age_s):
            return AdvisoryState(advisories=[{"summary": "x"}], fetched_at=fetched)

    monkeypatch.setattr(mkt_ops, "advisories", _Adv())
    resp = await router.marketplace_advisories(_get())
    body = _data(resp)["data"]
    assert body["advisories"] == [{"summary": "x"}]
    assert body["fetched_at"] == fetched.isoformat()


# -- install: gates, envelope, error mapping ---------------------------------


async def test_install_happy_wraps_data(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _ok(ref, version):
        return {"ref": ref, "version": "1.0.0", "uninstalled": False}

    _use_installer(monkeypatch, install=_ok)
    resp = await router.marketplace_install(_post({"ref": "tai42/toolbox"}))
    assert resp.status_code == 200
    assert _data(resp)["data"]["ref"] == "tai42/toolbox"


async def test_install_malformed_body_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    # A body missing the required ``ref`` fails the request-model parse (422).
    resp = await router.marketplace_install(_post({"version": "1.0.0"}))
    assert resp.status_code == 422


async def test_install_bad_ref_is_400(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _bad(ref, version):
        raise MalformedRefError("ref must be 'namespace/name', got 'noslash'")

    _use_installer(monkeypatch, install=_bad)
    resp = await router.marketplace_install(_post({"ref": "noslash"}))
    assert resp.status_code == 400


async def test_install_reload_gate_locked_is_503_reloading(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _never(ref, version):  # pragma: no cover - the gate rejects first
        raise AssertionError("the operation must not run while the gate is locked")

    _use_installer(monkeypatch, install=_never)
    async with reload_gate.lock:
        resp = await router.marketplace_install(_post({"ref": "tai42/toolbox"}))
    assert resp.status_code == 503
    body = _data(resp)
    assert body["reloading"] is True
    assert resp.headers["Retry-After"] == "5"


async def test_install_operation_in_progress_is_retriable_503(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _busy(ref, version):
        raise OperationInProgressError("another marketplace operation is in progress; retry shortly")

    _use_installer(monkeypatch, install=_busy)
    resp = await router.marketplace_install(_post({"ref": "tai42/toolbox"}))
    assert resp.status_code == 503
    assert "in progress" in _data(resp)["error"]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (VersionRefusedError("killed"), 409),
        (ManifestCollisionError("collides"), 409),
        (ContractIncompatibleError("range"), 409),
        (EnvironmentShadowError("env 1.0 shadows prefix 2.0"), 409),
        (InstallStateError("already installed"), 409),
        (InstallStateError("not installed", not_installed=True), 404),
        (ListingNotFoundError("not found"), 404),
        (RegistryUnreachableError("down"), 502),
        (PipFailedError(["install", "x"], 1, "boom"), 500),
        (
            ArtifactIntegrityError(expected_sha256="a" * 64, actual_sha256="b" * 64, artifact_ref="https://x/y.tgz"),
            500,
        ),
        # The _to_operation_error else-branch: an environment/invariant fault the
        # operation could not complete maps to a terminal 500.
        (ManifestComposeError("composed manifest invalid"), 500),
        (LocalStateError("stored spec corrupt"), 500),
        (InstallUnwindError(RuntimeError("step"), RuntimeError("unwind")), 500),
        (PipUnavailableError("no pip"), 500),
    ],
)
async def test_install_error_families_map_to_status(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status: int
) -> None:
    async def _raise(ref, version):
        raise error

    _use_installer(monkeypatch, install=_raise)
    resp = await router.marketplace_install(_post({"ref": "tai42/toolbox"}))
    assert resp.status_code == status
    assert "error" in _data(resp)


async def test_install_artifact_integrity_carries_digests_in_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    # The integrity failure is a terminal 500 whose envelope carries the digests
    # and the rejected artifact URL for the operator.
    async def _raise(ref, version):
        raise ArtifactIntegrityError(
            expected_sha256="a" * 64, actual_sha256="b" * 64, artifact_ref="https://codeload/x.tgz"
        )

    _use_installer(monkeypatch, install=_raise)
    resp = await router.marketplace_install(_post({"ref": "tai42/toolbox"}))
    assert resp.status_code == 500
    body = _data(resp)
    assert body["expected_sha256"] == "a" * 64
    assert body["actual_sha256"] == "b" * 64
    assert body["artifact_ref"] == "https://codeload/x.tgz"


async def test_uninstall_and_update_wrap_data(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _ok(ref, version):
        return {"ref": ref, "ok": True}

    _use_installer(monkeypatch, install=_ok)
    r1 = await router.marketplace_uninstall(_post({"ref": "tai42/toolbox"}))
    r2 = await router.marketplace_update(_post({"ref": "tai42/toolbox", "version": "2.0.0"}))
    assert _data(r1)["data"]["ok"] is True
    assert _data(r2)["data"]["ok"] is True


# -- upgrade-all --------------------------------------------------------------


async def test_upgrade_all_wraps_the_per_ref_report(monkeypatch: pytest.MonkeyPatch) -> None:
    report = [
        {"ref": "tai42/a", "outcome": "upgraded", "detail": "1.0.0 -> 1.2.0"},
        {"ref": "tai42/b", "outcome": "no-compatible-version", "detail": "no published version supports 0.3.0"},
    ]

    async def _run(ref, version):
        return report

    _use_installer(monkeypatch, install=_run)
    resp = await router.marketplace_upgrade_all(_post({}))
    assert resp.status_code == 200
    assert _data(resp)["data"] == {"results": report}


async def test_upgrade_all_busy_fleet_lock_is_retriable_503(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _busy(ref, version):
        raise OperationInProgressError("another marketplace operation is in progress; retry shortly")

    _use_installer(monkeypatch, install=_busy)
    resp = await router.marketplace_upgrade_all(_post({}))
    assert resp.status_code == 503
    assert "in progress" in _data(resp)["error"]


async def test_upgrade_all_reload_gate_locked_is_503_reloading(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _never(ref, version):  # pragma: no cover - the gate rejects first
        raise AssertionError("the operation must not run while the gate is locked")

    _use_installer(monkeypatch, install=_never)
    async with reload_gate.lock:
        resp = await router.marketplace_upgrade_all(_post({}))
    assert resp.status_code == 503
    assert _data(resp)["reloading"] is True
