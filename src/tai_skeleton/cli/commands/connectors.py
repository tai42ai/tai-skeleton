"""``tai connectors`` — manage connector providers and connections.

Thin wrappers over the ``/api/connectors/*`` routes. The browser OAuth callback
(``/api/connectors/oauth/complete``) is not an operator command and is intentionally
not exposed.
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_records,
    emit_result,
    parse_json_object,
)

app = typer.Typer(
    name="connectors",
    help="Manage connector providers and connections.",
    no_args_is_help=True,
)

_SUB_SERVICE_HELP = "An enabled sub-service (repeatable)."
_RETURN_URL_HELP = "Where to return after the flow completes."


@app.command("providers")
@covers(("GET", "/api/connectors/providers"))
def providers(ctx: typer.Context) -> None:
    """List the available connector providers.

    Example: ``tai connectors providers``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/connectors/providers")
    emit_records(ctx_obj, data, ["id", "display_name", "kind", "category"])


@app.command("connections")
@covers(("GET", "/api/connectors/connections"))
def connections(ctx: typer.Context) -> None:
    """List the installed connections (no secrets).

    Example: ``tai connectors connections``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/connectors/connections")
    emit_records(ctx_obj, data, ["connection_id", "provider_id", "alias", "auth_health_state"], items_key="items")


@app.command("get")
@covers(("GET", "/api/connectors/connections/{connection_id}"))
def get_connection(ctx: typer.Context, connection_id: Annotated[str, typer.Argument(help="Connection id.")]) -> None:
    """Get one connection.

    Example: ``tai connectors get conn_123``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/connectors/connections/{connection_id}")
    emit_result(ctx_obj, data)


@app.command("connect")
@covers(("POST", "/api/connectors/connections/start"))
def connect(
    ctx: typer.Context,
    provider: Annotated[str, typer.Argument(help="Provider id.")],
    alias: Annotated[str, typer.Option("--alias", help="A unique alias for the new connection.")],
    sub_service: Annotated[list[str], typer.Option("--sub-service", help=_SUB_SERVICE_HELP)],
    config_values: Annotated[
        str | None, typer.Option("--config", help="Provider config values as a JSON object.")
    ] = None,
    return_url: Annotated[str, typer.Option("--return-url", help=_RETURN_URL_HELP)] = "/connectors",
) -> None:
    """Start a connection flow; prints the authorize URL (or the created connection).

    Example: ``tai connectors connect google --alias work --sub-service gmail``
    """
    ctx_obj = app_context(ctx)
    body: dict = {
        "provider_id": provider,
        "alias": alias,
        "enabled_sub_services": list(sub_service),
        "config_values": parse_json_object(config_values, param_hint="--config") if config_values else {},
        "return_url": return_url,
    }
    with ctx_obj.client() as client:
        data = client.post("/api/connectors/connections/start", json=body)
    emit_result(ctx_obj, data)


@app.command("disconnect")
@covers(("DELETE", "/api/connectors/connections/{connection_id}"))
def disconnect(ctx: typer.Context, connection_id: Annotated[str, typer.Argument(help="Connection id.")]) -> None:
    """Disconnect (delete) a connection.

    Example: ``tai connectors disconnect conn_123``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/connectors/connections/{connection_id}")
    emit_result(ctx_obj, data)


@app.command("reconnect")
@covers(("POST", "/api/connectors/connections/{connection_id}/reconnect"))
def reconnect(
    ctx: typer.Context,
    connection_id: Annotated[str, typer.Argument(help="Connection id.")],
    sub_service: Annotated[list[str], typer.Option("--sub-service", help=_SUB_SERVICE_HELP)],
    return_url: Annotated[str, typer.Option("--return-url", help=_RETURN_URL_HELP)] = "/connectors",
) -> None:
    """Start a reconnect (re-authorize) flow for an existing connection.

    Example: ``tai connectors reconnect conn_123 --sub-service gmail``
    """
    ctx_obj = app_context(ctx)
    body = {"enabled_sub_services": list(sub_service), "return_url": return_url}
    with ctx_obj.client() as client:
        data = client.post(f"/api/connectors/connections/{connection_id}/reconnect", json=body)
    emit_result(ctx_obj, data)


@app.command("sub-services")
@covers(("PATCH", "/api/connectors/connections/{connection_id}/sub-services"))
def patch_sub_services(
    ctx: typer.Context,
    connection_id: Annotated[str, typer.Argument(help="Connection id.")],
    sub_service: Annotated[
        list[str], typer.Option("--sub-service", help="The desired enabled sub-service (repeatable).")
    ],
    return_url: Annotated[
        str, typer.Option("--return-url", help="Where to return if consent is required.")
    ] = "/connectors",
) -> None:
    """Change a connection's enabled sub-services.

    Example: ``tai connectors sub-services conn_123 --sub-service gmail --sub-service calendar``
    """
    ctx_obj = app_context(ctx)
    body = {"enabled_sub_services": list(sub_service), "return_url": return_url}
    with ctx_obj.client() as client:
        data = client.patch(f"/api/connectors/connections/{connection_id}/sub-services", json=body)
    emit_result(ctx_obj, data)
