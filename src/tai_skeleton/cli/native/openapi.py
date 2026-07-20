"""``tai openapi`` — emit the OpenAPI 3.1 specification for the API."""

from __future__ import annotations

import json
from pathlib import Path

import typer


def openapi(
    out: str | None = typer.Option(
        None,
        "--out",
        metavar="PATH",
        help="Write the spec to PATH instead of stdout.",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help="Validate the spec against the OpenAPI 3.1 schema and exit non-zero if invalid; write nothing.",
    ),
) -> None:
    """Emit the OpenAPI 3.1 specification for the ``/api/*`` surface.

    Builds the spec offline from the app's registered routes — no database,
    Redis, or live config services required. ``--check`` validates the spec
    against the OpenAPI 3.1 schema (for CI / pre-deploy) and writes nothing.
    """
    from tai_skeleton.cli.openapi import build_openapi_spec

    spec = build_openapi_spec()

    if check:
        from openapi_spec_validator import validate
        from openapi_spec_validator.validation.exceptions import OpenAPIValidationError

        try:
            validate(spec)
        except OpenAPIValidationError as exc:
            raise typer.BadParameter(f"emitted OpenAPI spec is invalid: {exc.message}") from exc
        typer.echo("OpenAPI spec is valid.", err=True)
        return

    document = json.dumps(spec, indent=2, sort_keys=True)
    if out is None:
        typer.echo(document)
    else:
        target = Path(out)
        target.write_text(document + "\n", encoding="utf-8")
        typer.echo(f"Wrote OpenAPI spec to {target}", err=True)
