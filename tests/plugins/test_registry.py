"""Studio-plugin registry: manifest validation, integrity hashing, traversal
defense, vendor hashing, per-plugin quarantine on load faults, and the
startup/reload rebuild pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tai42_skeleton.plugins.registry as reg
from tai42_skeleton.marketplace import compat
from tai42_skeleton.plugins.quarantine import quarantined_plugins, reset_quarantine
from tai42_skeleton.plugins.registry import (
    Contributions,
    StudioPluginError,
    StudioPluginManifest,
    build_registry,
    resolve_under,
)


@pytest.fixture(autouse=True)
def _clean_quarantine():
    reset_quarantine()
    yield
    reset_quarantine()


def _write_plugin(
    root: Path,
    *,
    name: str = "acme_plugin",
    entry: str = "index-a1b2c3.js",
    content: str = "export const x = 1;\n",
    extra_chunks: dict[str, str] | None = None,
) -> Path:
    """Create a valid ``studio/`` dist under ``root`` and return the studio dir.
    ``extra_chunks`` maps additional integrity-listed chunk filenames to their
    contents (e.g. lazy chunks the entry imports)."""
    studio = root / "studio"
    studio.mkdir(parents=True)
    (studio / entry).write_text(content, encoding="utf-8")
    integrity = {entry: reg._hash_file(studio / entry)}
    for chunk_name, chunk_content in (extra_chunks or {}).items():
        (studio / chunk_name).write_text(chunk_content, encoding="utf-8")
        integrity[chunk_name] = reg._hash_file(studio / chunk_name)
    manifest = {
        "name": name,
        "version": "0.1.0",
        "api_version": 1,
        "entry": entry,
        "integrity": integrity,
        "contributions": {"tool_panels": {"acme_demo": "panel"}, "pages": ["home"], "settings_tabs": []},
    }
    (studio / "studio-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return studio


# -- Manifest validation -----------------------------------------------------


def test_manifest_rejects_bad_name():
    with pytest.raises(ValueError, match="not a valid package name"):
        StudioPluginManifest.model_validate(
            {
                "name": "bad-name!",
                "version": "1",
                "api_version": 1,
                "entry": "e.js",
                "integrity": {"e.js": "sha384-" + "A" * 64},
                "contributions": {},
            }
        )


def test_manifest_rejects_bad_hash():
    with pytest.raises(ValueError, match="sha384"):
        StudioPluginManifest.model_validate(
            {
                "name": "ok",
                "version": "1",
                "api_version": 1,
                "entry": "e.js",
                "integrity": {"e.js": "md5-nope"},
                "contributions": {},
            }
        )


def test_manifest_rejects_traversal_filename():
    with pytest.raises(ValueError, match="not a valid asset path"):
        StudioPluginManifest.model_validate(
            {
                "name": "ok",
                "version": "1",
                "api_version": 1,
                "entry": "e.js",
                "integrity": {"../secret.js": "sha384-" + "A" * 64},
                "contributions": {},
            }
        )


def test_manifest_rejects_traversal_entry():
    with pytest.raises(ValueError, match="not a valid asset path"):
        StudioPluginManifest.model_validate(
            {
                "name": "ok",
                "version": "1",
                "api_version": 1,
                "entry": "../e.js",
                "integrity": {"e.js": "sha384-" + "A" * 64},
                "contributions": {},
            }
        )


def test_manifest_rejects_empty_integrity():
    with pytest.raises(ValueError, match="integrity map must not be empty"):
        StudioPluginManifest.model_validate(
            {
                "name": "ok",
                "version": "1",
                "api_version": 1,
                "entry": "e.js",
                "integrity": {},
                "contributions": {},
            }
        )


def test_manifest_rejects_unknown_field():
    with pytest.raises(ValueError, match="Extra inputs"):
        StudioPluginManifest.model_validate(
            {
                "name": "ok",
                "version": "1",
                "api_version": 1,
                "entry": "e.js",
                "integrity": {"e.js": "sha384-" + "A" * 64},
                "contributions": {},
                "surprise": 1,
            }
        )


# -- nav_entries contribution ------------------------------------------------


def test_contributions_nav_entries_defaults_empty():
    contrib = Contributions()
    assert contrib.nav_entries == []


def test_contributions_nav_entries_must_link_to_a_page():
    contrib = Contributions(pages=["home", "settings"], nav_entries=["home"])
    assert contrib.nav_entries == ["home"]


def test_contributions_rejects_nav_entry_without_page():
    with pytest.raises(ValueError, match="do not appear in the plugin's pages"):
        Contributions(pages=["home"], nav_entries=["missing"])


def test_manifest_carries_nav_entries():
    manifest = StudioPluginManifest.model_validate(
        {
            "name": "ok",
            "version": "1",
            "api_version": 1,
            "entry": "e.js",
            "integrity": {"e.js": "sha384-" + "A" * 64},
            "contributions": {"pages": ["home"], "nav_entries": ["home"]},
        }
    )
    # The field rides along in the model dump the ``/api/plugins`` route returns.
    assert manifest.model_dump()["contributions"]["nav_entries"] == ["home"]


# -- Traversal primitive -----------------------------------------------------


def test_resolve_under_rejects_dotdot(tmp_path):
    (tmp_path / "studio").mkdir()
    with pytest.raises(StudioPluginError, match="escapes"):
        resolve_under(tmp_path / "studio", "../etc/passwd")


def test_resolve_under_rejects_absolute(tmp_path):
    (tmp_path / "studio").mkdir()
    with pytest.raises(StudioPluginError, match="escapes"):
        resolve_under(tmp_path / "studio", "/etc/passwd")


def test_resolve_under_rejects_symlink_escape(tmp_path):
    studio = tmp_path / "studio"
    studio.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("s")
    (studio / "link").symlink_to(outside / "secret")
    with pytest.raises(StudioPluginError, match="escapes"):
        resolve_under(studio, "link")


def test_resolve_under_allows_in_root(tmp_path):
    studio = tmp_path / "studio"
    studio.mkdir()
    (studio / "ok.js").write_text("1")
    assert resolve_under(studio, "ok.js") == (studio / "ok.js").resolve()


# -- build_registry (disk load via monkeypatched _studio_root) ---------------


def test_build_registry_happy(tmp_path, monkeypatch):
    studio = _write_plugin(tmp_path)
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin"], None)
    assert "acme_plugin" in registry.plugins
    plugin = registry.plugins["acme_plugin"]
    # Integrity keys are FULLY-RESOLVED absolute served URLs.
    assert any(url.startswith("/api/plugins/acme_plugin/studio/") for url in plugin.integrity_by_url)


def test_build_registry_missing_manifest_quarantines(tmp_path, monkeypatch):
    studio = tmp_path / "studio"
    studio.mkdir()
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin"], None)
    assert registry.plugins == {}  # an empty registry is valid
    assert "missing studio-manifest.json" in quarantined_plugins()["acme_plugin"]


def test_build_registry_integrity_mismatch_quarantines(tmp_path, monkeypatch):
    studio = _write_plugin(tmp_path)
    # Corrupt the entry file AFTER the manifest recorded its hash.
    entry = next(p for p in studio.iterdir() if p.suffix == ".js")
    entry.write_text("export const x = 2; // mutated\n", encoding="utf-8")
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin"], None)
    assert registry.plugins == {}
    assert "sha384 mismatch" in quarantined_plugins()["acme_plugin"]


def test_build_registry_name_package_mismatch_quarantines(tmp_path, monkeypatch):
    # manifest.name != package: the shell builds the bundle URL from the name, so
    # a mismatch 404s on every browser load — quarantined at load time.
    studio = _write_plugin(tmp_path, name="other_name")
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin"], None)
    assert registry.plugins == {}
    assert "manifest name" in quarantined_plugins()["acme_plugin"]


def test_build_registry_duplicate_package_loads_once(tmp_path, monkeypatch, caplog):
    # A duplicate listing is manifest hygiene, not a plugin fault: the plugin
    # loads ONCE and the duplication is logged loudly, never quarantined away.
    studio = _write_plugin(tmp_path)
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin", "acme_plugin"], None)
    assert list(registry.plugins) == ["acme_plugin"]
    assert quarantined_plugins() == {}
    assert any("listed more than once" in rec.message for rec in caplog.records)


# -- the real dist-root resolution (unstubbed _studio_root) -------------------


def test_studio_root_rejects_invalid_package_name():
    with pytest.raises(StudioPluginError, match="is invalid"):
        reg._studio_root("not a package!")


def test_studio_root_rejects_unimportable_package():
    with pytest.raises(StudioPluginError, match="is not importable"):
        reg._studio_root("totally_bogus_studio_pkg")


def test_studio_root_rejects_package_without_studio_dir():
    # A real importable package that ships no ``studio/`` dist directory.
    with pytest.raises(StudioPluginError, match=r"has no ``studio/`` dist directory"):
        reg._studio_root("json")


def test_studio_root_resolves_a_real_package_dist(tmp_path, monkeypatch):
    import sys

    pkg = tmp_path / "studio_fixture_pkg"
    (pkg / "studio").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert reg._studio_root("studio_fixture_pkg") == pkg / "studio"
    finally:
        sys.modules.pop("studio_fixture_pkg", None)


# -- manifest-file faults quarantine at load ---------------------------------


def test_build_registry_unreadable_manifest_json_quarantines(tmp_path, monkeypatch):
    studio = tmp_path / "studio"
    studio.mkdir()
    (studio / "studio-manifest.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin"], None)
    assert registry.plugins == {}
    assert "not readable JSON" in quarantined_plugins()["acme_plugin"]


def test_build_registry_schema_invalid_manifest_quarantines(tmp_path, monkeypatch):
    studio = tmp_path / "studio"
    studio.mkdir()
    (studio / "studio-manifest.json").write_text(json.dumps({"name": "acme_plugin"}), encoding="utf-8")
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin"], None)
    assert registry.plugins == {}
    assert "manifest is invalid" in quarantined_plugins()["acme_plugin"]


def test_build_registry_entry_without_integrity_hash_quarantines(tmp_path, monkeypatch):
    studio = _write_plugin(tmp_path)
    manifest_path = studio / "studio-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Keep the integrity map non-empty (the model requires that) but drop the
    # entry's own hash.
    manifest["integrity"] = {"other-chunk.js": manifest["integrity"][manifest["entry"]]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin"], None)
    assert registry.plugins == {}
    assert "has no integrity hash" in quarantined_plugins()["acme_plugin"]


def test_build_registry_integrity_listing_a_missing_file_quarantines(tmp_path, monkeypatch):
    studio = _write_plugin(tmp_path)
    manifest_path = studio / "studio-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["ghost-chunk.js"] = manifest["integrity"][manifest["entry"]]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin"], None)
    assert registry.plugins == {}
    assert "no such file exists" in quarantined_plugins()["acme_plugin"]


def test_build_registry_missing_studio_dist_quarantines(monkeypatch):
    def _raise(package):
        raise StudioPluginError(f"studio plugin package {package!r} has no ``studio/`` dist directory")

    monkeypatch.setattr(reg, "_studio_root", _raise)
    registry = build_registry(["acme_plugin"], None)
    assert registry.plugins == {}
    assert "no ``studio/`` dist" in quarantined_plugins()["acme_plugin"]


def test_build_registry_incompatible_plugin_never_loads(tmp_path, monkeypatch):
    # A contract-incompatible studio plugin is quarantined on the VERDICT alone —
    # its dist is never even opened (loading it is what misbehaves).
    def _boom(package):  # pragma: no cover - the compat gate must reject first
        raise AssertionError("an incompatible plugin's dist must not be read")

    monkeypatch.setattr(reg, "_studio_root", _boom)
    monkeypatch.setattr(
        compat,
        "module_compat",
        lambda module, dist_map=None: compat.CompatVerdict("incompatible", "needs tai42-contract <0.2, 0.3.0 running"),
    )
    registry = build_registry(["acme_plugin"], None)
    assert registry.plugins == {}
    assert "needs tai42-contract <0.2" in quarantined_plugins()["acme_plugin"]


def test_build_registry_one_broken_plugin_keeps_siblings(tmp_path, monkeypatch):
    # Partial failure: the broken plugin quarantines, the healthy sibling serves.
    good = _write_plugin(tmp_path, name="good_plugin")

    def _root(package):
        if package == "good_plugin":
            return good
        raise StudioPluginError(f"studio plugin package {package!r} is not importable")

    monkeypatch.setattr(reg, "_studio_root", _root)
    registry = build_registry(["good_plugin", "bad_plugin"], None)
    assert list(registry.plugins) == ["good_plugin"]
    assert "not importable" in quarantined_plugins()["bad_plugin"]


# -- Host-only specifier gate ------------------------------------------------


def test_build_registry_normal_sdk_import_loads(tmp_path, monkeypatch):
    studio = _write_plugin(tmp_path, content='import {run} from "@tai42/studio-sdk";\nrun();\n')
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin"], None)
    assert "acme_plugin" in registry.plugins


@pytest.mark.parametrize("specifier", ["@tai42/studio-sdk/host", "@tai42/studio-sdk/testing"])
def test_build_registry_rejects_host_only_specifier(tmp_path, monkeypatch, specifier):
    studio = _write_plugin(tmp_path, content=f'import {{registry}} from "{specifier}";\n')
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin"], None)
    # The offending bundle never serves: quarantined and excluded, so the asset
    # route (integrity-listed files only) can never hand it to a browser.
    assert registry.plugins == {}
    assert "host-only Studio SDK" in quarantined_plugins()["acme_plugin"]


def test_build_registry_scans_non_entry_chunk(tmp_path, monkeypatch):
    # The entry is clean, but a SECOND integrity-listed chunk carries the host-only
    # specifier. The byte-scan covers every listed file, not just the entry, so the
    # load must still reject the plugin (into quarantine).
    studio = _write_plugin(
        tmp_path,
        content='import {run} from "@tai42/studio-sdk";\n',
        extra_chunks={"chunk-d4e5f6.js": 'import {registry} from "@tai42/studio-sdk/host";\n'},
    )
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)
    registry = build_registry(["acme_plugin"], None)
    assert registry.plugins == {}
    assert "host-only Studio SDK" in quarantined_plugins()["acme_plugin"]


# -- Vendor hashing ----------------------------------------------------------


def test_vendor_hashing_missing_asset_is_loud(tmp_path):
    (tmp_path / "vendor").mkdir()  # empty — react.js absent
    with pytest.raises(StudioPluginError, match="shared-vendor asset"):
        build_registry([], str(tmp_path))


def test_vendor_hashing_happy(tmp_path):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    for rel in reg.VENDOR_MODULES.values():
        (tmp_path / rel).write_text("export {};\n", encoding="utf-8")
    registry = build_registry([], str(tmp_path))
    assert set(registry.vendor_integrity_by_url) == {f"/{rel}" for rel in reg.VENDOR_MODULES.values()}
    assert all(v.startswith("sha384-") for v in registry.vendor_integrity_by_url.values())


# -- Rebuild pass (startup/reload) -------------------------------------------


async def test_rebuild_pass_reflects_reload(tmp_path, monkeypatch):
    studio = _write_plugin(tmp_path)
    monkeypatch.setattr(reg, "_studio_root", lambda package: studio)

    # Fake the live-manifest seam + settings the handler reads at call time.
    live = {"studio_plugins": []}

    class _FakeAdmin:
        @property
        def live_manifest(self):
            return live

    class _FakeApp:
        admin = _FakeAdmin()

    monkeypatch.setattr("tai42_contract.app.tai42_app", _FakeApp())
    monkeypatch.setattr(
        "tai42_skeleton.plugins.settings.plugins_settings", lambda: type("S", (), {"studio_dist_path": None})()
    )

    await reg.rebuild_studio_plugin_registry()
    assert reg.current_registry().plugins == {}

    # A reload adds the plugin -> the rebuilt registry reflects it WITHOUT restart.
    live["studio_plugins"] = ["acme_plugin"]
    await reg.rebuild_studio_plugin_registry()
    assert "acme_plugin" in reg.current_registry().plugins
