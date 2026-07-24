"""``tai conversations`` — manage the conversation routing table.

Thin wrappers over the authed ``/api/conversations*`` management routes. The channel
door and the authed message door are not operator routes and are not exposed here.
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai42_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_records,
    emit_result,
)

app = typer.Typer(
    name="conversations",
    help="Manage conversation routes.",
    no_args_is_help=True,
)


@app.command("list")
@covers(("GET", "/api/conversations"))
def list_routes(ctx: typer.Context) -> None:
    """List conversation routes (each row's callback_secret is withheld).

    Example: ``tai conversations list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/conversations")
    emit_records(
        ctx_obj,
        data,
        ["route_name", "door", "agent_name", "execution_key", "channel", "our_identity"],
        items_key="items",
    )


@app.command("get")
@covers(("GET", "/api/conversations/{route_name}"))
def get_route(ctx: typer.Context, route_name: Annotated[str, typer.Argument(help="Route name (slug).")]) -> None:
    """Show one conversation route by name (its callback_secret is withheld).

    Example: ``tai conversations get support-line``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/conversations/{route_name}")
    emit_result(ctx_obj, data)


@app.command("create")
@covers(("POST", "/api/conversations/{route_name}"))
def create_route(
    ctx: typer.Context,
    route_name: Annotated[str, typer.Argument(help="Route name — a slug [a-z0-9-]+ (the thread-key handle).")],
    door: Annotated[str, typer.Option("--door", help="Inbound door: 'api' or 'channel'.")],
    agent: Annotated[str, typer.Option("--agent", help="The agent the turn runs (must exist).")],
    execution_key: Annotated[
        str,
        typer.Option(
            "--execution-key",
            help="The api-key user id the turn runs AS; you must own it (or be admin), tokenless-evaluable.",
        ),
    ],
    channel: Annotated[
        str | None, typer.Option("--channel", help="door=channel: the channel registry name (e.g. twilio).")
    ] = None,
    our_identity: Annotated[
        str | None, typer.Option("--identity", help="door=channel: the medium address we are texted at.")
    ] = None,
    callback_url: Annotated[
        str | None, typer.Option("--callback-url", help="door=api: the https answer-delivery URL.")
    ] = None,
) -> None:
    """Create or replace a conversation route.

    An UPSERT — a name that already exists is REPLACED, rebinding its ``execution_key``
    along with everything else (``created`` is ``false`` for a replace). A ``door=api``
    route's ``callback_secret`` is minted server-side and shown ONCE in the result; it
    signs the delivery callback and is never re-readable. There is no check that you can
    run the agent — the execution key's live grants bound the turn.

    Example: ``tai conversations create support-line --door channel --agent triage \\
    --execution-key svc --channel twilio --identity +15550001111``
    """
    ctx_obj = app_context(ctx)
    body: dict = {"door": door, "agent_name": agent, "execution_key": execution_key}
    if channel is not None:
        body["channel"] = channel
    if our_identity is not None:
        body["our_identity"] = our_identity
    if callback_url is not None:
        body["callback_url"] = callback_url
    with ctx_obj.client() as client:
        data = client.post(f"/api/conversations/{route_name}", json=body)
    emit_result(ctx_obj, data)


@app.command("delete")
@covers(("DELETE", "/api/conversations/{route_name}"))
def delete_route(ctx: typer.Context, route_name: Annotated[str, typer.Argument(help="Route name (slug).")]) -> None:
    """Delete a conversation route by name.

    Example: ``tai conversations delete support-line``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/conversations/{route_name}")
    emit_result(ctx_obj, data)


@app.command("get-message")
@covers(("GET", "/api/conversations/{route_name}/messages/{message_id}"))
def get_message(
    ctx: typer.Context,
    route_name: Annotated[str, typer.Argument(help="Route name (slug).")],
    message_id: Annotated[str, typer.Argument(help="Answer record message id (uuid4).")],
) -> None:
    """Read one conversation answer record (caller-scoped: your own records, or any as
    admin; channel records are admin-only).

    Example: ``tai conversations get-message support-line 4f1c...``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/conversations/{route_name}/messages/{message_id}")
    emit_result(ctx_obj, data)


@app.command("failed")
@covers(("GET", "/api/conversations/messages/failed"))
def list_failed(ctx: typer.Context) -> None:
    """List answer records whose delivery ended failed (admin only).

    Example: ``tai conversations failed``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/conversations/messages/failed")
    emit_records(
        ctx_obj,
        data,
        ["message_id", "route_name", "door", "client_address", "answer_status", "attempts"],
        items_key="items",
    )
