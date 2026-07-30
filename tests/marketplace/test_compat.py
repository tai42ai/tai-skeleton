"""Contract-compat verdicts and the update-target computation.

``compat.metadata`` (the module's importlib.metadata seam) is monkeypatched with
a stub namespace per test, so every verdict is driven by controlled dist
metadata — no dependence on the environment's installed distributions.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from tai42_skeleton.marketplace import compat
from tai42_skeleton.marketplace.compat import (
    CompatVerdict,
    dist_compat,
    module_compat,
    update_targets,
)
from tai42_skeleton.marketplace.errors import LocalStateError, RegistryResponseError


def _stub_metadata(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contract_version: str = "0.3.0",
    requires: dict[str, list[str] | None] | None = None,
    dist_map: dict[str, list[str]] | None = None,
) -> None:
    """Install a fake ``compat.metadata``: ``version()`` answers only for
    tai42-contract, ``requires()`` from the given map (a missing key raises the
    real PackageNotFoundError), ``packages_distributions()`` from ``dist_map``."""
    requires_map = requires or {}

    def _version(dist: str) -> str:
        assert dist == "tai42-contract"
        return contract_version

    def _requires(dist: str) -> list[str] | None:
        if dist not in requires_map:
            raise PackageNotFoundError(dist)
        return requires_map[dist]

    stub = SimpleNamespace(
        version=_version,
        requires=_requires,
        packages_distributions=lambda: dict(dist_map or {}),
        PackageNotFoundError=PackageNotFoundError,
    )
    monkeypatch.setattr(compat, "metadata", stub)


# -- dist_compat --------------------------------------------------------------


def test_declared_and_satisfied_is_compatible(monkeypatch) -> None:
    _stub_metadata(monkeypatch, requires={"acme-plugin": ["pydantic>=2", "tai42-contract>=0.3,<0.4"]})
    assert dist_compat("acme-plugin") == CompatVerdict("compatible")


def test_declared_and_excluded_is_incompatible_naming_both_versions(monkeypatch) -> None:
    _stub_metadata(monkeypatch, requires={"acme-plugin": ["tai42-contract>=0.2,<0.3"]})
    verdict = dist_compat("acme-plugin")
    assert verdict.status == "incompatible"
    assert verdict.reason is not None
    assert "acme-plugin" in verdict.reason  # the dist
    assert "<0.3" in verdict.reason  # its declared range...
    assert ">=0.2" in verdict.reason  # ...both halves
    assert "0.3.0 is running" in verdict.reason  # the running contract
    assert "tai plugins update" in verdict.reason  # the remedy


def test_no_declared_range_is_unknown(monkeypatch) -> None:
    _stub_metadata(monkeypatch, requires={"acme-plugin": ["pydantic>=2"]})
    verdict = dist_compat("acme-plugin")
    assert verdict.status == "unknown"
    assert verdict.reason is not None
    assert "declares no tai42-contract requirement" in verdict.reason


def test_missing_dist_is_unknown(monkeypatch) -> None:
    _stub_metadata(monkeypatch, requires={})
    assert dist_compat("ghost-plugin").status == "unknown"


def test_extra_gated_declaration_is_not_a_runtime_range(monkeypatch) -> None:
    # A ``; extra == "dev"`` line is not part of the runtime dependency set, so
    # it must not produce a (possibly refusing) verdict.
    _stub_metadata(monkeypatch, requires={"acme-plugin": ['tai42-contract<0.1; extra == "dev"']})
    assert dist_compat("acme-plugin").status == "unknown"


def test_malformed_foreign_requirement_line_is_skipped(monkeypatch) -> None:
    # A stranger's garbled Requires-Dist line must not poison the verdict: the
    # tai42-contract line still decides.
    _stub_metadata(monkeypatch, requires={"acme-plugin": ["¡not a requirement¡", "tai42-contract>=0.3,<0.4"]})
    assert dist_compat("acme-plugin") == CompatVerdict("compatible")


def test_prerelease_running_contract_inside_the_range_passes(monkeypatch) -> None:
    # Same prereleases=True semantics as the installer's install-time check: a
    # dev-versioned editable contract checkout is never spuriously refused.
    _stub_metadata(monkeypatch, contract_version="0.3.1.dev3", requires={"acme-plugin": ["tai42-contract>=0.3,<0.4"]})
    assert dist_compat("acme-plugin") == CompatVerdict("compatible")


def test_multiple_runtime_declarations_and_together(monkeypatch) -> None:
    _stub_metadata(monkeypatch, requires={"acme-plugin": ["tai42-contract>=0.3", "tai42-contract<0.3.0"]})
    assert dist_compat("acme-plugin").status == "incompatible"


# -- module_compat ------------------------------------------------------------


def test_module_with_no_dist_mapping_is_unknown(monkeypatch) -> None:
    _stub_metadata(monkeypatch, dist_map={})
    verdict = module_compat("acme_pkg.tools")
    assert verdict.status == "unknown"
    assert verdict.reason is not None
    assert "maps to no installed distribution" in verdict.reason


def test_module_maps_through_its_top_level_package(monkeypatch) -> None:
    _stub_metadata(
        monkeypatch,
        requires={"acme-plugin": ["tai42-contract>=0.2,<0.3"]},
        dist_map={"acme_pkg": ["acme-plugin"]},
    )
    assert module_compat("acme_pkg.sub.module").status == "incompatible"


def test_module_any_incompatible_owning_dist_decides(monkeypatch) -> None:
    _stub_metadata(
        monkeypatch,
        requires={"acme-a": ["tai42-contract>=0.3,<0.4"], "acme-b": ["tai42-contract<0.2"]},
        dist_map={"acme_pkg": ["acme-a", "acme-b"]},
    )
    verdict = module_compat("acme_pkg")
    assert verdict.status == "incompatible"
    assert verdict.reason is not None
    assert "acme-b" in verdict.reason


def test_module_uses_the_passed_snapshot_map(monkeypatch) -> None:
    # A dist_map snapshot bypasses packages_distributions() entirely — the one
    # boot pass reuses its snapshot instead of rescanning per module.
    def _boom():  # pragma: no cover - the snapshot must win
        raise AssertionError("packages_distributions must not be re-read when a snapshot is passed")

    _stub_metadata(monkeypatch, requires={"acme-plugin": ["tai42-contract>=0.3,<0.4"]})
    assert compat.metadata is not None
    monkeypatch.setattr(compat.metadata, "packages_distributions", _boom)
    verdict = module_compat("acme_pkg", {"acme_pkg": ["acme-plugin"]})
    assert verdict == CompatVerdict("compatible")


# -- update_targets -----------------------------------------------------------


def _row(version: str, status: str = "published", contract_range: str | None = ">=0.3,<0.4") -> dict:
    return {"version": version, "status": status, "contract_range": contract_range}


def test_update_targets_full_picture() -> None:
    targets = update_targets(
        [
            _row("2.0.0", contract_range=">=0.4,<0.5"),  # newer but blocked
            _row("1.5.0"),  # the compatible target
            _row("1.6.0", status="yanked"),  # unpublished — never counts
            _row("1.0.0"),
        ],
        installed_version="1.0.0",
        contract_version="0.3.0",
    )
    assert targets.latest == "2.0.0"
    assert targets.latest_compatible == "1.5.0"
    assert targets.update_available is True
    assert targets.incompatible_newer == "2.0.0"


def test_update_targets_up_to_date_and_no_rows() -> None:
    targets = update_targets([_row("1.0.0")], installed_version="1.0.0", contract_version="0.3.0")
    assert targets.update_available is False
    assert targets.incompatible_newer is None
    empty = update_targets([], installed_version="1.0.0", contract_version="0.3.0")
    assert (empty.latest, empty.latest_compatible, empty.update_available) == (None, None, False)


def test_update_targets_null_contract_range_counts_compatible() -> None:
    # Same unknown-means-compatible semantics as the dist verdict: a version
    # published without a declared range must keep updating.
    targets = update_targets([_row("2.0.0", contract_range=None)], installed_version="1.0.0", contract_version="0.3.0")
    assert targets.latest_compatible == "2.0.0"
    assert targets.update_available is True


@pytest.mark.parametrize(
    "rows",
    [
        ["not-a-row"],  # non-object element
        [_row("not-a-version")],  # non-PEP440 published version
        [{"status": "published", "contract_range": ">=0.3"}],  # missing version
        [_row("1.0.0", contract_range="!! garbage")],  # malformed range
        [{"version": "1.0.0", "status": "published", "contract_range": 42}],  # non-string range
    ],
)
def test_update_targets_garbled_registry_rows_raise_the_typed_fault(rows) -> None:
    with pytest.raises(RegistryResponseError):
        update_targets(rows, installed_version="1.0.0", contract_version="0.3.0")


def test_update_targets_garbled_local_version_is_local_state() -> None:
    # The stored installed version is LOCAL truth; its garbling is never blamed
    # on the registry.
    with pytest.raises(LocalStateError, match="not PEP 440"):
        update_targets([_row("1.0.0")], installed_version="not-a-version", contract_version="0.3.0")
