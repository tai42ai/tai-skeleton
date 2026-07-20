"""Offline ``manifest validate`` and ``config lint`` — no server required.

Both load a manifest file and run it through the in-repo :class:`Manifest` model;
a valid file is accepted, a broken one is rejected loudly with the model's error and
a non-zero exit. ``config lint`` additionally checks that required registered
settings resolve. No client, database, or Redis is touched.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from tai_skeleton.cli import app as app_module

_VALID_MANIFEST = """
tools:
  - title: toolbox
    module: tai_skeleton.examples
    include: []
"""

_BROKEN_MANIFEST = """
tools:
  - module: 123
"""


def test_manifest_validate_accepts_valid(tmp_path) -> None:
    path = tmp_path / "manifest.yml"
    path.write_text(_VALID_MANIFEST, encoding="utf-8")

    result = CliRunner().invoke(app_module.app, ["manifest", "validate", str(path)])
    assert result.exit_code == 0, result.output
    assert "is valid" in result.output


def test_manifest_validate_rejects_broken(tmp_path) -> None:
    path = tmp_path / "manifest.yml"
    path.write_text(_BROKEN_MANIFEST, encoding="utf-8")

    result = CliRunner().invoke(app_module.app, ["manifest", "validate", str(path)])
    assert result.exit_code != 0
    assert "invalid manifest" in result.output


def test_manifest_validate_missing_file() -> None:
    result = CliRunner().invoke(app_module.app, ["manifest", "validate", "/no/such/manifest.yml"])
    assert result.exit_code != 0


def test_config_lint_accepts_valid(tmp_path) -> None:
    path = tmp_path / "manifest.yml"
    path.write_text(_VALID_MANIFEST, encoding="utf-8")

    result = CliRunner().invoke(app_module.app, ["config", "lint", str(path)])
    assert result.exit_code == 0, result.output
    assert "is valid" in result.output
    assert "required settings resolve" in result.output


def test_config_lint_rejects_broken_manifest(tmp_path) -> None:
    path = tmp_path / "manifest.yml"
    path.write_text(_BROKEN_MANIFEST, encoding="utf-8")

    result = CliRunner().invoke(app_module.app, ["config", "lint", str(path)])
    assert result.exit_code != 0
    assert "invalid manifest" in result.output


def test_config_lint_without_manifest_still_checks_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)
    result = CliRunner().invoke(app_module.app, ["config", "lint"])
    assert result.exit_code == 0, result.output
    assert "required settings resolve" in result.output


def test_config_lint_flags_unresolved_required_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    # A required setting whose env var is absent must fail the lint loudly.
    from tai_kit.settings import SettingsClassInfo, SettingsFieldInfo

    field = SettingsFieldInfo(
        name="token",
        env_var="TAI_LINT_TEST_REQUIRED",
        type="str",
        default=None,
        required=True,
        secret=False,
        description="",
        nested_group=None,
    )
    group = SettingsClassInfo(name="LintProbe", module="probe", qualname="probe.LintProbe", fields=[field])

    import tai_skeleton.cli.commands.config as config_cmd

    monkeypatch.setattr(config_cmd, "registered_settings", lambda: [group])
    monkeypatch.setattr(config_cmd, "load_api_routes", list)
    monkeypatch.delenv("TAI_LINT_TEST_REQUIRED", raising=False)

    result = CliRunner().invoke(app_module.app, ["config", "lint"])
    assert result.exit_code != 0
    assert "required settings do not resolve" in result.output
