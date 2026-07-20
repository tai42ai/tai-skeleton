"""``tai config`` — read and update server configuration.

``env`` and ``mode`` and ``settings-schema`` are thin wrappers over the
``/api/config/*`` routes; ``lint`` is OFFLINE (no server) — it validates a manifest
file's shape against the :class:`Manifest` model and checks that every required
registered setting resolves, for CI / pre-deploy gating.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer
from tai42_kit.settings import registered_settings

from tai42_skeleton.app.route_registry import load_api_routes
from tai42_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_records,
    emit_result,
    validate_manifest_file,
)

app = typer.Typer(
    name="config",
    help="Read and update server configuration.",
    no_args_is_help=True,
)

env_app = typer.Typer(name="env", help="Read and update the stored env overrides.", no_args_is_help=True)
app.add_typer(env_app, name="env")


@env_app.command("get")
@covers(("GET", "/api/config/env"))
def get_env(ctx: typer.Context) -> None:
    """Read the stored env config and the operator's secret-key marks.

    Example: ``tai config env get``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/config/env")
    emit_result(ctx_obj, data)


@env_app.command("set")
@covers(("POST", "/api/config/env"))
def set_env(
    ctx: typer.Context,
    assignment: Annotated[list[str], typer.Argument(help="One or more KEY=VALUE env overrides to merge.")],
) -> None:
    """Merge KEY=VALUE env overrides and hot-reload the process config.

    Example: ``tai config env set LOG_LEVEL=debug FEATURE_X=1``
    """
    ctx_obj = app_context(ctx)
    overrides: dict[str, str] = {}
    for pair in assignment:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"expected KEY=VALUE, got {pair!r}")
        overrides[key] = value
    with ctx_obj.client() as client:
        data = client.post("/api/config/env", json=overrides)
    emit_result(ctx_obj, data)


@app.command("reload")
@covers(("POST", "/api/config/reload"))
def reload_config(
    ctx: typer.Context,
    target: Annotated[
        list[str] | None, typer.Option("--target", help="A worker to restrict the reload to (repeatable).")
    ] = None,
) -> None:
    """Soft-restart this process from its manifest, fanning out to every worker.

    Refreshes env, resets settings caches, and re-initializes from the manifest
    in-process (no pod restart), then propagates to the fleet; ``--target`` restricts
    the fan-out to named workers.

    Example: ``tai config reload``
    """
    ctx_obj = app_context(ctx)
    targets = list(target) if target else None
    with ctx_obj.client() as client:
        data = client.post("/api/config/reload", json={"targets": targets})
    emit_result(ctx_obj, data)


@app.command("mode")
@covers(("GET", "/api/config/mode"))
def config_mode(ctx: typer.Context) -> None:
    """Read the active config backend mode (file / k8s).

    Example: ``tai config mode``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/config/mode")
    emit_result(ctx_obj, data)


@app.command("settings-schema")
@covers(("GET", "/api/config/settings-schema"))
def settings_schema(ctx: typer.Context) -> None:
    """List the registered settings groups with their resolved field values.

    Example: ``tai config settings-schema --json``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/config/settings-schema")
    emit_records(ctx_obj, data, ["name", "module"], items_key="groups")


@app.command("lint")
def lint(
    file: Annotated[
        str | None, typer.Argument(help="Manifest file to validate; defaults to $TAI_MANIFEST_PATH when set.")
    ] = None,
) -> None:
    """Lint config OFFLINE: a manifest file's shape plus required-settings resolution.

    Validates the manifest file against the ``Manifest`` model (a broken manifest
    fails loudly with the model error) and checks that every required registered
    setting resolves from the environment. No server is started.

    Example: ``tai config lint config/manifest.yml``
    """
    manifest_path = file or os.environ.get("TAI_MANIFEST_PATH")
    if manifest_path is not None:
        if not os.path.isfile(manifest_path):
            raise typer.BadParameter(f"manifest file not found: {manifest_path}", param_hint="FILE")
        validate_manifest_file(manifest_path)
        typer.echo(f"Manifest {manifest_path} is valid.", err=True)
    else:
        typer.echo("No manifest file given and TAI_MANIFEST_PATH is unset; skipping manifest shape check.", err=True)

    unresolved = _unresolved_required_settings()
    if unresolved:
        detail = ", ".join(unresolved)
        raise typer.BadParameter(f"required settings do not resolve: {detail}", param_hint="settings")
    typer.echo("All required settings resolve.", err=True)


def _unresolved_required_settings() -> list[str]:
    """Every ``group.field`` whose setting is required but resolves to no value.

    Mirrors the ``/api/config/settings-schema`` surface: the settings modules are
    imported offline (the same offline import primitive the OpenAPI emitter uses),
    then each required field with an ``env_var`` must be present in the process
    environment (which already includes any bootstrapped ``.env``).
    """
    # Importing the API surface offline registers the skeleton's settings groups —
    # the same set the settings-schema route reports — without booting a server.
    load_api_routes()

    unresolved: list[str] = []
    for group in registered_settings():
        for field in group.fields:
            if field.required and field.env_var and field.env_var not in os.environ:
                unresolved.append(f"{group.name}.{field.name} (${field.env_var})")
    return unresolved
