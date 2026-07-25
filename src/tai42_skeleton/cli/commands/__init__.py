"""Remote command groups — thin clients over the skeleton's ``/api/*`` routes.

Each module exposes a ``typer.Typer`` app named ``app`` that the root CLI
registers as a subcommand group. The command bodies are added per domain.
"""
