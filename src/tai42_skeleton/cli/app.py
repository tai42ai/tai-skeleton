"""The unified ``tai`` command.

One Typer app exposes every operator function: the remote command groups (thin
clients over the skeleton's ``/api/*`` routes), the CLI-native local commands,
and the re-homed runtime launchers (``serve``/``backend``/``metrics``). The
console entry point ``app`` is the compiled click group, which lets the existing
click launcher commands mount as subcommands alongside the Typer groups.

Registration lives here so each downstream command module only fills its own
Typer app (or command function) without editing this file: a remote group adds
commands to ``cli/commands/<domain>.py``'s ``app``; a native command fills its
``cli/native/<name>.py`` module. The seams the command bodies build on are the
:class:`~tai42_skeleton.cli.context.AppContext` on the Typer context (for a
configured client), :class:`~tai42_skeleton.cli.client.ApiClient`, and the
:mod:`tai42_skeleton.cli.render` helpers.
"""

from typing import Any, cast

import click
import typer
from dotenv import load_dotenv
from typer.core import TyperGroup, TyperOption
from typer.main import get_command

from tai42_skeleton.cli import backend, mcp_app, metrics
from tai42_skeleton.cli.client import ApiError
from tai42_skeleton.cli.commands import (
    agents,
    auth,
    backup,
    channels,
    checkpoints,
    config,
    connectors,
    conversations,
    extensions,
    fleet,
    hooks,
    interactions,
    keys,
    manifest,
    mcp,
    notifications,
    obs,
    plugins,
    presets,
    resources,
    roles,
    schedules,
    scopes,
    storage,
    sub_mcp,
    system,
    templates,
    tools,
    traces,
)
from tai42_skeleton.cli.context import AppContext
from tai42_skeleton.cli.native import catalog, completion, db, doctor, openapi, version
from tai42_skeleton.config.config_mode import config_mode


class TaiCLIGroup(TyperGroup):
    """Root group that bridges standard-``click`` outcomes into Typer's runner.

    Typer bundles its own vendored copy of click, so Typer's runner only
    recognises its vendored exception types. The re-homed launcher commands (and
    the native command modules) raise standard-``click`` exceptions, and remote
    commands raise :class:`ApiError`. This override translates all of those into
    Typer's own control-flow exceptions so they render cleanly — a server/usage
    message on stderr with a non-zero exit — instead of a traceback."""

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except ApiError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        except click.exceptions.Exit as exc:
            raise typer.Exit(exc.exit_code) from exc
        except click.exceptions.Abort as exc:
            raise typer.Abort() from exc
        except click.ClickException as exc:
            typer.echo(f"Error: {exc.format_message()}", err=True)
            raise typer.Exit(exc.exit_code) from exc


def _apply_json_flag(ctx: click.Context, _param: click.Parameter, value: bool | None) -> bool | None:
    """Merge a per-subcommand ``--json/--no-json`` into the shared context.

    The root callback owns the flag-first form (``tai --json <cmd>``); this lets the
    same flag ride AFTER the subcommand (``tai <cmd> --json``). ``value`` is ``None``
    unless the flag was actually passed, so the root callback's setting stands when
    the trailing flag is absent."""
    if value is not None and isinstance(ctx.obj, AppContext):
        ctx.obj.json_output = value
    return value


def _json_flag_option() -> TyperOption:
    """A fresh trailing ``--json/--no-json`` flag that merges into the context.

    Built as a :class:`TyperOption` so its parse handling matches the Typer context
    the compiled commands run under (a raw ``click.Option`` reads a context attribute
    Typer's vendored context does not carry)."""
    return TyperOption(
        param_decls=["--json/--no-json"],
        is_flag=True,
        default=None,
        expose_value=False,
        callback=_apply_json_flag,
        help="Emit raw JSON instead of human tables.",
    )


def _inject_json_flag(group: click.Group) -> None:
    """Add the trailing ``--json/--no-json`` flag to every leaf subcommand.

    Recurses into nested groups so a leaf command reached as ``tai <group> <cmd>``
    accepts the flag in its own right (``tai tools list --json``), matching the root
    callback's flag-first form. The flag is only meaningful once, so groups keep the
    root form and only leaf commands carry the trailing one.

    A group is detected by its ``commands`` mapping rather than by ``isinstance`` —
    Typer's compiled sub-groups are its own vendored ``Group`` type, not a
    :class:`click.Group` subclass."""
    for command in group.commands.values():
        if getattr(command, "commands", None) is not None:
            _inject_json_flag(cast(click.Group, command))
        else:
            command.params.append(cast(click.Parameter, _json_flag_option()))


cli_app = typer.Typer(
    name="tai",
    cls=TaiCLIGroup,
    help="Operate a tai42-skeleton server from the terminal.",
    no_args_is_help=True,
    add_completion=False,
)


@cli_app.callback()
def main(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Emit raw JSON instead of human tables.",
    ),
    server: str | None = typer.Option(
        None,
        "--server",
        metavar="URL",
        help="Server base URL. Resolved: this flag -> TAI_SERVER_URL -> config.toml -> local default.",
    ),
    api_key_stdin: bool = typer.Option(
        False,
        "--api-key-stdin",
        help=(
            "Read the API key as one line from stdin. Resolved: this flag -> TAI_API_KEY -> "
            "config.toml -> interactive prompt. There is no --api-key VALUE flag (a value leaks "
            "via ps and shell history)."
        ),
    ),
) -> None:
    # Bootstrap a local ``.env`` once for the whole CLI (skipped under k8s config,
    # where env comes from the platform), so every subcommand — the remote client
    # and the runtime launchers alike — sees it.
    if config_mode() != "k8s":
        load_dotenv()
    ctx.obj = AppContext(json_output=json_output, server_override=server, api_key_stdin=api_key_stdin)


# Remote command groups — thin clients over the ``/api/*`` routes.
_REMOTE_GROUPS: list[tuple[typer.Typer, str]] = [
    (tools.app, "tools"),
    (presets.app, "presets"),
    (agents.app, "agents"),
    (extensions.app, "extensions"),
    (connectors.app, "connectors"),
    (conversations.app, "conversations"),
    (hooks.app, "hooks"),
    (channels.app, "channels"),
    (checkpoints.app, "checkpoints"),
    (notifications.app, "notifications"),
    (storage.app, "storage"),
    (resources.app, "resources"),
    (fleet.app, "fleet"),
    (manifest.app, "manifest"),
    (mcp.app, "mcp"),
    (sub_mcp.app, "sub-mcp"),
    (templates.app, "templates"),
    (config.app, "config"),
    (keys.app, "keys"),
    (scopes.app, "scopes"),
    (roles.app, "roles"),
    (auth.app, "auth"),
    (backup.app, "backup"),
    (schedules.app, "schedules"),
    (obs.app, "obs"),
    (plugins.app, "plugins"),
    (traces.app, "traces"),
    (interactions.app, "interactions"),
    (system.app, "system"),
]
for group_app, group_name in _REMOTE_GROUPS:
    cli_app.add_typer(group_app, name=group_name)

# CLI-native groups.
cli_app.add_typer(db.app, name="db")
cli_app.add_typer(completion.app, name="completion")

# CLI-native single commands (bodies filled by the local-ops / OpenAPI work).
cli_app.command(name="doctor")(doctor.doctor)
cli_app.command(name="catalog")(catalog.catalog)
cli_app.command(name="openapi")(openapi.openapi)
cli_app.command(name="version")(version.version)


def _mount_launcher(group: click.Group, launcher: click.Command, name: str) -> None:
    """Mount a re-homed click launcher command under its ``tai`` subcommand name.

    The command's own ``name`` is the canonical subcommand name now, so it is set
    here (the rich help lists commands by that name, not by the registration key)."""
    launcher.name = name
    group.add_command(launcher)


def _build_app() -> click.Group:
    """Compile the Typer app to a click group and mount the re-homed launcher
    commands, which are click commands rather than Typer commands."""
    command = cast(click.Group, get_command(cli_app))
    # Give every Typer-backed leaf command the trailing ``--json`` form before the
    # launcher commands mount; the launchers have their own flags and no JSON output.
    _inject_json_flag(command)
    _mount_launcher(command, mcp_app.cli, "serve")
    _mount_launcher(command, backend.main, "backend")
    _mount_launcher(command, metrics.main, "metrics")
    return command


app = _build_app()


if __name__ == "__main__":
    app()
