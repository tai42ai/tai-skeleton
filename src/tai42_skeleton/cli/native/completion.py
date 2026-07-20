"""``tai completion`` — install shell completion for the tai CLI."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import typer
from click.shell_completion import get_completion_class

app = typer.Typer(
    name="completion",
    help="Install shell completion for the tai CLI.",
    no_args_is_help=True,
)

# The program name the generated script completes, and the env var click reads to
# switch into completion mode — derived from that name (``_<PROG>_COMPLETE``).
_PROG_NAME = "tai"
_COMPLETE_VAR = "_TAI_COMPLETE"


class Shell(StrEnum):
    bash = "bash"
    zsh = "zsh"
    fish = "fish"


@app.command("install")
def install(shell: Annotated[Shell, typer.Argument(help="Shell to emit a completion script for.")]) -> None:
    """Emit a shell completion script for the tai CLI.

    Pipe it into the shell's completion directory, e.g.
    ``tai completion install bash > /etc/bash_completion.d/tai``.
    """
    completion_class = get_completion_class(shell.value)
    if completion_class is None:
        raise typer.BadParameter(f"shell '{shell.value}' has no completion support in this click build.")

    # Import here: the root app imports this module at load, and the compiled
    # command is only available after that import completes.
    from tai42_skeleton.cli.app import app as tai42_command

    completion = completion_class(tai42_command, {}, _PROG_NAME, _COMPLETE_VAR)
    typer.echo(completion.source())
