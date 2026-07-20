"""``tai version`` — skeleton + key dependency versions."""

from __future__ import annotations

import json

from click.testing import CliRunner

from tai_skeleton.cli import app as app_module


def test_version_lists_skeleton_and_deps() -> None:
    result = CliRunner().invoke(app_module.app, ["version"])
    assert result.exit_code == 0, result.output
    assert "tai-skeleton" in result.output
    assert "tai-contract" in result.output
    assert "tai-kit" in result.output


def test_version_json_parses() -> None:
    result = CliRunner().invoke(app_module.app, ["--json", "version"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    names = {row["package"]: row["version"] for row in data}
    assert names["tai-skeleton"] != "not installed"
