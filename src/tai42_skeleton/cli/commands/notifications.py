"""``tai notifications`` — read the internal notifications feed and send a
notification.

Thin wrappers over the authed ``/api/notifications`` routes: ``list`` reads the
deployment's internal notifications feed — channel-less sends plus any
audience-addressed notification, recorded even when a channel also delivers it —
newest-first (the feed is a bounded ring buffer written by the sink); ``notify``
sends a human a one-way, fire-and-forget message on a named channel or (channel
omitted) into the sink.
"""

from __future__ import annotations

from typing import Annotated

import typer
from pydantic import TypeAdapter, ValidationError
from tai42_contract.channels import ChannelTemplate
from tai42_contract.interactions.models import MediaItem

from tai42_skeleton.cli.commands._common import app_context, covers, emit_records, emit_result, parse_json_value

_MEDIA_ADAPTER = TypeAdapter(list[MediaItem])

app = typer.Typer(
    name="notifications",
    help="Read and send internal notifications.",
    no_args_is_help=True,
)


@app.command("list")
@covers(("GET", "/api/notifications"))
def list_notifications(ctx: typer.Context) -> None:
    """List the internal notifications, newest-first.

    Example: ``tai notifications list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/notifications")
    emit_records(ctx_obj, data, ["message", "recipient", "created_at"], items_key="notifications")


@app.command("notify")
@covers(("POST", "/api/notifications"))
def notify(
    ctx: typer.Context,
    message: Annotated[str, typer.Argument(help="The notification text shown to the human.")],
    channel: Annotated[
        str | None,
        typer.Option("--channel", help="Named channel to send on; omit to record to the internal sink."),
    ] = None,
    recipient: Annotated[
        str | None,
        typer.Option("--recipient", help="Optional per-call address (chat id, phone number, ...)."),
    ] = None,
    media: Annotated[
        str | None,
        typer.Option(
            "--media",
            help='JSON array of display-media items sent WITH the message, e.g. \'[{"kind":"image","url":"https://…"}]\'.',
        ),
    ] = None,
    template: Annotated[
        str | None,
        typer.Option(
            "--template",
            help='JSON object for an out-of-window template send, e.g. \'{"name":"order_update","language":"en_US"}\'.',
        ),
    ] = None,
) -> None:
    """Send a human a one-way, fire-and-forget notification.

    ``--media`` and ``--template`` are JSON strings (the nested shapes have no flat-flag
    form); each is validated into its contract model before the request, so a malformed
    value raises loudly here rather than at the server.

    Example: ``tai notifications notify "Deploy finished" --channel telegram``
    """
    ctx_obj = app_context(ctx)
    body: dict[str, object] = {"message": message}
    if channel is not None:
        body["channel"] = channel
    if recipient is not None:
        body["recipient"] = recipient
    if media is not None:
        parsed_media = parse_json_value(media, param_hint="--media")
        try:
            items = _MEDIA_ADAPTER.validate_python(parsed_media)
        except ValidationError as exc:
            raise typer.BadParameter(f"invalid media item(s): {exc}", param_hint="--media") from exc
        body["media"] = [item.model_dump(mode="json") for item in items]
    if template is not None:
        parsed_template = parse_json_value(template, param_hint="--template")
        try:
            model = ChannelTemplate.model_validate(parsed_template)
        except ValidationError as exc:
            raise typer.BadParameter(f"invalid template: {exc}", param_hint="--template") from exc
        body["template"] = model.model_dump(mode="json")
    with ctx_obj.client() as client:
        data = client.post("/api/notifications", json=body)
    emit_result(ctx_obj, data)
