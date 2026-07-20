"""``tai hooks`` — register and inspect webhook hooks and topic verifiers.

Thin wrappers over the authed ``/api/hooks*`` management routes. The public
``/universal_webhook/{topic}`` ingress door is not an ``/api/*`` operator route
and is not exposed here.
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
    name="hooks",
    help="Register and inspect webhook hooks.",
    no_args_is_help=True,
)


@app.command("list")
@covers(("GET", "/api/hooks"))
def list_hooks(
    ctx: typer.Context,
    topic: Annotated[str | None, typer.Option("--topic", help="Filter to one topic.")] = None,
) -> None:
    """List registered hooks (and the per-topic verifier bindings under ``--json``).

    Example: ``tai hooks list --topic github``
    """
    ctx_obj = app_context(ctx)
    params = {"topic": topic} if topic else None
    with ctx_obj.client() as client:
        data = client.get("/api/hooks", params=params)
    emit_records(ctx_obj, data, ["name", "topic", "tool"], items_key="items")


@app.command("verifiers")
@covers(("GET", "/api/hooks/verifiers"))
def list_verifiers(ctx: typer.Context) -> None:
    """List the registered webhook-verifier names (the bind catalog).

    Example: ``tai hooks verifiers``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/hooks/verifiers")
    emit_result(ctx_obj, data)


@app.command("register")
@covers(("POST", "/api/hooks"))
def register_hook(
    ctx: typer.Context,
    params_json: Annotated[str, typer.Option("--params", help="The full HookParams as a JSON object.")],
) -> None:
    """Register a hook from a HookParams JSON body.

    Example: ``tai hooks register --params '{"name":"h1","topic":"github","tool":"notify"}'``
    """
    ctx_obj = app_context(ctx)
    body = parse_json_object(params_json, param_hint="--params")
    with ctx_obj.client() as client:
        data = client.post("/api/hooks", json=body)
    emit_result(ctx_obj, data)


@app.command("delete")
@covers(("DELETE", "/api/hooks/{name}"))
def delete_hook(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Hook name.")]) -> None:
    """Unregister a hook by name.

    Example: ``tai hooks delete h1``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/hooks/{name}")
    emit_result(ctx_obj, data)


@app.command("set-verifier")
@covers(("PUT", "/api/hooks/topics/{topic}/verifier"))
def set_topic_verifier(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Hook topic.")],
    verifier: Annotated[str, typer.Option("--verifier", help="Registered webhook-verifier name.")],
    config_json: Annotated[str | None, typer.Option("--config", help="Verifier config as a JSON object.")] = None,
) -> None:
    """Bind a webhook verifier to a topic so its deliveries are signature-verified.

    Example: ``tai hooks set-verifier github --verifier github_hmac``
    """
    ctx_obj = app_context(ctx)
    body: dict = {"verifier": verifier}
    if config_json is not None:
        body["config"] = parse_json_object(config_json, param_hint="--config")
    with ctx_obj.client() as client:
        data = client.put(f"/api/hooks/topics/{topic}/verifier", json=body)
    emit_result(ctx_obj, data)


@app.command("delete-verifier")
@covers(("DELETE", "/api/hooks/topics/{topic}/verifier"))
def delete_topic_verifier(ctx: typer.Context, topic: Annotated[str, typer.Argument(help="Hook topic.")]) -> None:
    """Remove a topic's verifier binding.

    Example: ``tai hooks delete-verifier github``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/hooks/topics/{topic}/verifier")
    emit_result(ctx_obj, data)
