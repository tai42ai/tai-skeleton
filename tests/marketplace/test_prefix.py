"""The plugin prefix: settings, the sys.path activation (env-wins ordering + loud
failures), the write pre-flight, and the RECORD-based prefix uninstall.

Every seam here is exercised against a real temp directory — a hand-built
``.dist-info`` + ``RECORD`` for the uninstall path, real ``sys.path`` mutation for
the activation path — so the file operations the feature relies on are asserted for
real, never mocked away.
"""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

import pytest

from tai42_skeleton.marketplace import prefix as prefix_module
from tai42_skeleton.marketplace.errors import PluginPrefixError
from tai42_skeleton.marketplace.prefix import (
    PluginPrefixSettings,
    activate_prefix,
    configured_prefix,
    ensure_prefix_writable,
    environment_distribution_version,
    prefix_has_distribution,
    prefix_site_dirs,
    uninstall_from_prefix,
)

# -- settings ----------------------------------------------------------------


def test_settings_reads_tai_plugins_prefix_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_PLUGINS_PREFIX", "/opt/plugins")
    assert PluginPrefixSettings().prefix == "/opt/plugins"


def test_settings_unset_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_PLUGINS_PREFIX", raising=False)
    assert PluginPrefixSettings().prefix is None


def test_configured_prefix_reads_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prefix_module, "plugin_prefix_settings", lambda: PluginPrefixSettings(prefix="/x"))
    assert configured_prefix() == "/x"


# -- site dirs ---------------------------------------------------------------


def test_prefix_site_dirs_are_under_the_prefix() -> None:
    dirs = prefix_site_dirs("/opt/plugins")
    assert dirs
    assert all(d.startswith("/opt/plugins") for d in dirs)
    assert all(d.endswith("site-packages") for d in dirs)


# -- activation: sys.path ordering + loud failures ---------------------------


def test_activate_prefix_noop_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prefix_module, "configured_prefix", lambda: None)
    before = list(sys.path)
    activate_prefix()
    assert sys.path == before


def test_activate_prefix_appends_after_environment_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prefix = str(tmp_path / "pfx")
    monkeypatch.setattr(prefix_module, "configured_prefix", lambda: prefix)
    site = prefix_site_dirs(prefix)[0]
    env_site = sysconfig.get_paths()["purelib"]
    orig = list(sys.path)
    try:
        activate_prefix()
        # The prefix dir is on the path, created on disk, and sits at the END so the
        # environment shadows it for any package present in both.
        assert site in sys.path
        assert Path(site).is_dir()
        assert sys.path[-1] == site
        if env_site in sys.path:
            assert sys.path.index(site) > sys.path.index(env_site)
        # A second activation (a reload) adds no duplicate entry.
        activate_prefix()
        assert sys.path.count(site) == 1
    finally:
        sys.path[:] = orig


def test_activate_prefix_tolerates_an_existing_empty_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prefix = str(tmp_path / "pfx")
    Path(prefix_site_dirs(prefix)[0]).mkdir(parents=True)
    monkeypatch.setattr(prefix_module, "configured_prefix", lambda: prefix)
    orig = list(sys.path)
    try:
        activate_prefix()
        assert prefix_site_dirs(prefix)[0] in sys.path
    finally:
        sys.path[:] = orig


def test_activate_prefix_missing_and_uncreatable_raises_loudly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A prefix under an ancestor that is a FILE cannot be created — a set-but-broken
    # prefix fails loudly at boot rather than silently activating an empty path.
    a_file = tmp_path / "not-a-dir"
    a_file.write_text("x", encoding="utf-8")
    prefix = str(a_file / "sub")
    monkeypatch.setattr(prefix_module, "configured_prefix", lambda: prefix)
    orig = list(sys.path)
    try:
        with pytest.raises(OSError, match=r"Not a directory"):
            activate_prefix()
    finally:
        sys.path[:] = orig


# -- install write pre-flight ------------------------------------------------


def test_ensure_prefix_writable_creates_site_dirs(tmp_path: Path) -> None:
    prefix = str(tmp_path / "pfx")
    ensure_prefix_writable(prefix)
    assert Path(prefix_site_dirs(prefix)[0]).is_dir()
    # The write probe left nothing behind.
    assert not (Path(prefix) / ".tai42-prefix-write-probe").exists()


def test_ensure_prefix_writable_raises_on_unwritable(tmp_path: Path) -> None:
    a_file = tmp_path / "not-a-dir"
    a_file.write_text("x", encoding="utf-8")
    prefix = str(a_file / "sub")
    with pytest.raises(PluginPrefixError, match="not writable"):
        ensure_prefix_writable(prefix)


# -- prefix uninstall: RECORD-based removal ----------------------------------


def _install_fake_dist(prefix: str, package: str, version: str, *, record_paths: list[str] | None = None) -> Path:
    """Write a minimal installed distribution (a package dir + ``.dist-info`` with
    ``METADATA`` and ``RECORD``) into the prefix's site directory, the way a
    ``pip install --prefix`` would lay one out, and return that site directory."""
    site = Path(prefix_site_dirs(prefix)[0])
    site.mkdir(parents=True, exist_ok=True)
    pkg_dir = site / package
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    info = site / f"{package}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n", encoding="utf-8")
    lines = record_paths or [
        f"{package}/__init__.py,,",
        f"{package}-{version}.dist-info/METADATA,,",
        f"{package}-{version}.dist-info/RECORD,,",
    ]
    (info / "RECORD").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return site


# -- env vs prefix distribution resolution -----------------------------------


def test_prefix_has_distribution_true_for_prefix_dist_and_false_for_absent(tmp_path: Path) -> None:
    prefix = str(tmp_path / "pfx")
    _install_fake_dist(prefix, "foo", "1.0")
    assert prefix_has_distribution("foo", prefix) is True
    assert prefix_has_distribution("bar", prefix) is False


def test_prefix_has_distribution_ignores_environment_only_copy(tmp_path: Path) -> None:
    # A distribution present only in the ENVIRONMENT (never written to the prefix) is
    # not reported as a prefix distribution — the check scans the prefix site dirs only.
    prefix = str(tmp_path / "pfx")
    Path(prefix_site_dirs(prefix)[0]).mkdir(parents=True)
    assert prefix_has_distribution("pytest", prefix) is False


def test_environment_distribution_version_reads_env_not_prefix(tmp_path: Path) -> None:
    # A real environment package reports its installed version; a package that lives
    # ONLY under the prefix is invisible to the environment read.
    prefix = str(tmp_path / "pfx")
    _install_fake_dist(prefix, "foo", "9.9")
    assert environment_distribution_version("pytest", prefix) is not None
    assert environment_distribution_version("foo", prefix) is None


def test_uninstall_from_prefix_removes_named_dist_and_keeps_dependencies(tmp_path: Path) -> None:
    prefix = str(tmp_path / "pfx")
    site = _install_fake_dist(prefix, "foo", "1.0")
    _install_fake_dist(prefix, "bar", "2.0")  # a dependency foo pulled in

    uninstall_from_prefix("foo", prefix)

    # The named distribution's own files are gone (package tree + dist-info).
    assert not (site / "foo").exists()
    assert not (site / "foo-1.0.dist-info").exists()
    # The dependency stays — only the named distribution is removed, exactly like
    # ``pip uninstall`` on the environment path.
    assert (site / "bar" / "__init__.py").exists()
    assert (site / "bar-2.0.dist-info").exists()


def test_uninstall_from_prefix_absent_package_raises_and_touches_nothing(tmp_path: Path) -> None:
    prefix = str(tmp_path / "pfx")
    Path(prefix_site_dirs(prefix)[0]).mkdir(parents=True)
    with pytest.raises(PluginPrefixError, match="not installed in the plugin prefix"):
        uninstall_from_prefix("foo", prefix)


def test_uninstall_from_prefix_record_escape_raises_and_deletes_nothing_outside(tmp_path: Path) -> None:
    prefix = str(tmp_path / "pfx")
    outside = tmp_path / "evil.txt"
    outside.write_text("keep me", encoding="utf-8")
    site = Path(prefix_site_dirs(prefix)[0])
    # Enough ``..`` segments to climb out of the prefix entirely.
    depth = len(site.relative_to(tmp_path).parts)
    escape = "/".join([".."] * depth) + "/evil.txt"
    _install_fake_dist(prefix, "foo", "1.0", record_paths=[escape + ",,", "foo-1.0.dist-info/RECORD,,"])

    with pytest.raises(PluginPrefixError, match="escapes the plugin prefix"):
        uninstall_from_prefix("foo", prefix)
    # A RECORD path outside the prefix is refused before deletion — the outside file
    # is untouched.
    assert outside.read_text(encoding="utf-8") == "keep me"
