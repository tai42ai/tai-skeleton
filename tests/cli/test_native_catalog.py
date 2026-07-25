"""``tai catalog`` — the packaged ecosystem catalog.

Covers the offline read of ``data/ecosystem.yml``, the package->repo join (the
repo lives in one place, joined at render time), the loud error when an entry's
package is missing from that map, and the ``--json`` / table rendering.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from tai42_skeleton.cli import app as app_module
from tai42_skeleton.cli.native import catalog

_VALID_KINDS = {
    "tool",
    "agent",
    "accounts",
    "extension",
    "connector",
    "channel",
    "identity",
    "backend",
    "storage",
    "monitoring",
    "webhook-verifier",
    "config",
}


def test_load_catalog_joins_repo_for_every_entry() -> None:
    records = catalog.load_catalog()
    assert records, "the packaged catalog is empty"
    for record in records:
        assert record["repo"], f"{record['name']} has no derived repo"
        assert record["kind"] in _VALID_KINDS, f"{record['name']} has an unknown kind {record['kind']!r}"
        for field in ("name", "group", "package", "module", "description"):
            assert record[field], f"{record['name']} is missing {field}"


def test_load_catalog_raises_on_unmapped_package(monkeypatch: pytest.MonkeyPatch) -> None:
    document = """
entries:
  - name: orphan
    kind: tool
    group: g
    package: tai-unmapped
    module: m
    description: d
packages:
  tai42-skeleton: tai42-skeleton
"""
    fake_resource = SimpleNamespace(read_text=lambda encoding="utf-8": document)
    monkeypatch.setattr(
        catalog.importlib.resources,
        "files",
        lambda package: SimpleNamespace(joinpath=lambda *parts: fake_resource),
    )
    with pytest.raises(RuntimeError, match="tai-unmapped"):
        catalog.load_catalog()


def test_catalog_json_parses() -> None:
    result = CliRunner().invoke(app_module.app, ["--json", "catalog"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert {"name", "kind", "repo", "package", "module"} <= set(data[0])


def test_catalog_json_trailing_flag_parses() -> None:
    # The habitual trailing form ``tai catalog --json`` must parse the same as the
    # flag-first ``tai --json catalog``.
    result = CliRunner().invoke(app_module.app, ["catalog", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert {"name", "kind", "repo"} <= set(data[0])


def test_catalog_command_wraps_packaging_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A packaging bug surfaced by load_catalog renders as the clean CLI error line,
    # not a raw RuntimeError traceback.
    def boom() -> list[dict[str, object]]:
        raise RuntimeError("entry 'orphan' names package 'x' missing from the map")

    monkeypatch.setattr(catalog, "load_catalog", boom)
    result = CliRunner().invoke(app_module.app, ["catalog"])
    assert result.exit_code != 0
    assert "missing from the map" in result.output
    assert "Traceback" not in result.output


def test_catalog_table_renders_with_repo_column() -> None:
    result = CliRunner().invoke(app_module.app, ["catalog"])
    assert result.exit_code == 0, result.output
    assert "repo" in result.output
    assert "tai42-skeleton" in result.output
