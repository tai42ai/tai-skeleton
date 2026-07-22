"""``tai hooks`` — register and inspect webhook hooks and topic verifiers.

Thin wrappers over the authed ``/api/hooks*`` management routes. The public
``/universal_webhook/{topic}`` ingress door is not an ``/api/*`` operator route
and is not exposed here.
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai42_skeleton.cli.commands._common import (
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


@app.command("trigger-links")
@covers(("GET", "/api/hooks/trigger-links"))
def list_trigger_links(ctx: typer.Context) -> None:
    """List trigger links (name, topic, expiry, hash prefix; never a raw token).

    Example: ``tai hooks trigger-links``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/hooks/trigger-links")
    emit_records(ctx_obj, data, ["name", "topic", "expires_at", "token_hash_prefix"], items_key="items")


@app.command("create-trigger-link")
@covers(("POST", "/api/hooks/trigger-links"))
def create_trigger_link(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="The hook topic the link fires.")],
    name: Annotated[
        str | None,
        typer.Option("--name", help="Link name (the revocation handle); a unique name is generated when omitted."),
    ] = None,
    ttl: Annotated[int | None, typer.Option("--ttl", help="Link lifetime in seconds (a timed link).")] = None,
    permanent: Annotated[bool, typer.Option("--permanent", help="Mint a link that never expires.")] = False,
    params_json: Annotated[
        str | None, typer.Option("--params", help="Per-link tool_kwargs as a JSON object, merged last at each fire.")
    ] = None,
) -> None:
    """Mint a trigger link — a PUBLIC URL that fires the topic's hooks.

    Exactly ONE of ``--ttl SECONDS`` or ``--permanent`` is required (expiry is an
    explicit choice — there is no default). The token is shown ONCE, in the printed
    absolute URL; the link is MULTI-use and revocable by name (``tai hooks
    delete-trigger-link NAME``). Regenerate = revoke + create.

    Example: ``tai hooks create-trigger-link orders --ttl 3600 --params '{"priority":"high"}'``
    """
    ctx_obj = app_context(ctx)
    if ttl is None and not permanent:
        raise typer.BadParameter("exactly one of --ttl or --permanent is required", param_hint="--ttl/--permanent")
    if ttl is not None and permanent:
        raise typer.BadParameter("--ttl and --permanent are mutually exclusive", param_hint="--ttl/--permanent")

    body: dict = {"topic": topic, "ttl_seconds": None if permanent else ttl}
    if name is not None:
        body["name"] = name
    if params_json is not None:
        body["tool_kwargs"] = parse_json_object(params_json, param_hint="--params")

    with ctx_obj.client() as client:
        data = client.post("/api/hooks/trigger-links", json=body)

    # The server returns a PATH (it does not know its public origin); compose the
    # absolute URL from the configured server base, stripping a trailing slash so the
    # join never doubles it.
    absolute_url = f"{ctx_obj.server_url.rstrip('/')}{data['trigger_path']}"
    emit_result(
        ctx_obj,
        {
            "name": data["name"],
            "topic": data["topic"],
            "url": absolute_url,
            "expires_at": data["expires_at"],
        },
    )


@app.command("delete-trigger-link")
@covers(("DELETE", "/api/hooks/trigger-links/{name}"))
def delete_trigger_link(ctx: typer.Context, name: Annotated[str, typer.Argument(help="Trigger link name.")]) -> None:
    """Revoke a trigger link by name (immediate and durable — a restored backup
    cannot re-arm it).

    Example: ``tai hooks delete-trigger-link my-wall-qr``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/hooks/trigger-links/{name}")
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
