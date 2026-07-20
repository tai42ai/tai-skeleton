"""The unified ``tai`` app: every group is registered, the launchers are
re-homed as subcommands, the root callback builds the shared context, and a
raised :class:`ApiError` surfaces as a clean CLI error.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from tai_skeleton.cli import app as app_module
from tai_skeleton.cli.client import ApiError
from tai_skeleton.cli.context import AppContext

# Every command group / command that must render under ``tai --help``.
_REMOTE_GROUPS = {
    "tools",
    "presets",
    "agents",
    "extensions",
    "connectors",
    "hooks",
    "manifest",
    "mcp",
    "sub-mcp",
    "templates",
    "config",
    "keys",
    "scopes",
    "backup",
    "schedules",
    "obs",
    "traces",
    "interactions",
}
_NATIVE = {"db", "completion", "doctor", "catalog", "openapi", "version"}
_LAUNCHERS = {"serve", "backend", "metrics"}


def test_help_renders_every_registered_group() -> None:
    result = CliRunner().invoke(app_module.app, ["--help"])

    assert result.exit_code == 0, result.output
    for name in _REMOTE_GROUPS | _NATIVE | _LAUNCHERS:
        assert name in result.output, f"missing group in --help: {name}"


def test_compiled_group_exposes_every_command() -> None:
    commands = set(app_module.app.commands)
    assert commands.issuperset(_REMOTE_GROUPS)
    assert commands.issuperset(_NATIVE)
    assert commands.issuperset(_LAUNCHERS)


def test_rehomed_launchers_are_the_original_commands() -> None:
    # The re-homed launcher subcommands are the existing click launcher
    # commands, only renamed to their ``tai`` subcommand names.
    from tai_skeleton.cli import backend, mcp_app, metrics

    assert app_module.app.commands["serve"] is mcp_app.cli
    assert app_module.app.commands["backend"] is backend.main
    assert app_module.app.commands["metrics"] is metrics.main


def test_serve_help_renders() -> None:
    result = CliRunner().invoke(app_module.app, ["serve", "--help"])

    assert result.exit_code == 0, result.output
    assert "--transport" in result.output


@pytest.mark.parametrize("command", ["catalog", "version"])
def test_native_command_runs_offline(command: str) -> None:
    # ``catalog`` and ``version`` read only packaged data / installed metadata, so
    # they succeed on a bare install with no server, DB, or network.
    result = CliRunner().invoke(app_module.app, [command])

    assert result.exit_code == 0, result.output


def test_doctor_command_is_registered() -> None:
    result = CliRunner().invoke(app_module.app, ["doctor", "--help"])

    assert result.exit_code == 0, result.output


def test_callback_populates_app_context(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from tai_skeleton.cli import context

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv(context.SERVER_URL_ENV, "http://probe-host")

    group = app_module._build_app()
    captured: dict[str, object] = {}

    @click.command("probe")
    @click.pass_context
    def probe(ctx: click.Context) -> None:
        captured["obj"] = ctx.obj

    group.add_command(probe, "probe")
    result = CliRunner().invoke(group, ["--json", "probe"])

    assert result.exit_code == 0, result.output
    obj = captured["obj"]
    assert isinstance(obj, AppContext)
    assert obj.json_output is True
    assert obj.server_url == "http://probe-host"


def test_trailing_json_flag_on_remote_leaf_renders_json(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``tai tools list --json`` (flag AFTER the leaf command) must render JSON just
    # like the flag-first ``tai --json tools list`` form.
    import json

    import httpx

    from tai_skeleton.cli.client import ApiClient

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": ["alpha", "beta"]}))
    monkeypatch.setattr(
        AppContext,
        "client",
        lambda self: ApiClient(self.server_url, self.api_key, transport=transport),
    )
    monkeypatch.setenv("TAI_API_KEY", "test-key")
    monkeypatch.setenv("TAI_SERVER_URL", "http://testserver")

    result = CliRunner().invoke(app_module.app, ["tools", "list", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == ["alpha", "beta"]


def test_trailing_no_json_flag_forces_table(monkeypatch: pytest.MonkeyPatch) -> None:
    # The trailing ``--no-json`` overrides a flag-first ``--json`` back to the table.
    import httpx

    from tai_skeleton.cli.client import ApiClient

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": ["alpha"]}))
    monkeypatch.setattr(
        AppContext,
        "client",
        lambda self: ApiClient(self.server_url, self.api_key, transport=transport),
    )
    monkeypatch.setenv("TAI_API_KEY", "test-key")
    monkeypatch.setenv("TAI_SERVER_URL", "http://testserver")

    result = CliRunner().invoke(app_module.app, ["--json", "tools", "list", "--no-json"])

    assert result.exit_code == 0, result.output
    # A table has a header/separator, not a JSON array.
    assert "name" in result.output
    assert not result.output.lstrip().startswith("[")


def test_api_error_renders_as_clean_cli_error() -> None:
    group = app_module._build_app()

    @click.command("boom")
    def boom() -> None:
        raise ApiError("server refused the request", status_code=409)

    group.add_command(boom, "boom")
    result = CliRunner().invoke(group, ["boom"])

    assert result.exit_code != 0
    assert "server refused the request" in result.output
