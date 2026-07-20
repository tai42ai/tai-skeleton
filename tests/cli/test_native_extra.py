"""Edge paths of the native (offline) commands: version, openapi, completion.

Each covers the branch a normal healthy environment never takes — a dependency
that is not installed, an invalid emitted spec, and a shell click cannot complete.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from tai_skeleton.cli import app as app_module
from tai_skeleton.cli.native import completion as completion_mod
from tai_skeleton.cli.native import version as version_mod


def test_version_marks_absent_package_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(version_mod, "_PACKAGES", ["tai-skeleton", "definitely-not-a-real-package-xyz"])
    records = version_mod._versions()
    by_name = {row["package"]: row["version"] for row in records}
    assert by_name["definitely-not-a-real-package-xyz"] == "not installed"
    assert by_name["tai-skeleton"] != "not installed"


def test_openapi_check_rejects_invalid_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    import tai_skeleton.cli.openapi as builder

    # A 3.1-tagged document missing the required ``paths`` object is detected as
    # OpenAPI yet fails schema validation, so the command's ``--check`` surfaces it.
    monkeypatch.setattr(
        builder, "build_openapi_spec", lambda: {"openapi": "3.1.0", "info": {"title": "t", "version": "1"}}
    )
    result = CliRunner().invoke(app_module.app, ["openapi", "--check"])
    assert result.exit_code != 0
    assert "invalid" in result.output.lower()


def test_completion_install_reports_missing_completion_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(completion_mod, "get_completion_class", lambda shell: None)
    result = CliRunner().invoke(app_module.app, ["completion", "install", "bash"])
    assert result.exit_code != 0
    assert "no completion support" in result.output
