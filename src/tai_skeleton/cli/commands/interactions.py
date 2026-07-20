"""``tai interactions`` — stream and answer pending interactions.

The interactions surface is stream/answer only — there is no GET list route. The
pending backlog arrives as the SSE stream's first frames, so ``list`` connects to
the stream, prints the backlog frames up to the ``backlog_done`` marker, and exits;
``stream`` tails the inbox live; ``answer`` posts a human answer.
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai_skeleton.cli.commands._common import (
    app_context,
    covers,
    emit_result,
    parse_json_value,
    stream_frames,
)

app = typer.Typer(
    name="interactions",
    help="Stream and answer pending interactions.",
    no_args_is_help=True,
)


@app.command("list")
@covers(("GET", "/api/interactions/stream"))
def list_interactions(ctx: typer.Context) -> None:
    """Print the pending-interaction backlog (the stream's initial frames), then exit.

    Example: ``tai interactions list``
    """
    ctx_obj = app_context(ctx)
    stream_frames(ctx_obj, "GET", "/api/interactions/stream", until_empty=True)


@app.command("stream")
@covers(("GET", "/api/interactions/stream"))
def stream_interactions(ctx: typer.Context) -> None:
    """Tail the interactions inbox live (backlog then add/answered/removed events).

    Example: ``tai interactions stream``
    """
    ctx_obj = app_context(ctx)
    stream_frames(ctx_obj, "GET", "/api/interactions/stream")


@app.command("answer")
@covers(("POST", "/api/interactions/{interaction_id}/answer"))
def answer_interaction(
    ctx: typer.Context,
    interaction_id: Annotated[str, typer.Argument(help="Interaction id.")],
    answer: Annotated[str, typer.Option("--answer", help="The answer value as JSON (a string, bool, object, ...).")],
) -> None:
    """Answer a pending interaction. The value is validated server-side.

    Example: ``tai interactions answer i_123 --answer '"yes"'``
    """
    ctx_obj = app_context(ctx)
    value = parse_json_value(answer, param_hint="--answer")
    with ctx_obj.client() as client:
        data = client.post(f"/api/interactions/{interaction_id}/answer", json={"answer": value})
    emit_result(ctx_obj, data)
