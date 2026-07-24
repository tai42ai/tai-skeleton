"""``tai keys`` — provision API keys and manage their access-control policy.

Thin wrappers over the authed ``/api/auth/api-keys*``, ``/api/auth/tokens-payload``
and ``/api/auth/validate-condition`` routes. The raw ``sk-…`` key is returned ONCE
by ``create`` — capture it then.
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
    name="keys",
    help="Manage API keys and their access-control conditions.",
    no_args_is_help=True,
)

_CONDITION_KWARGS_HELP = "Condition kwargs as a JSON object."


@app.command("list")
@covers(("GET", "/api/auth/tokens-payload"))
def list_keys(ctx: typer.Context) -> None:
    """List every provisioned key's identity and policy (never key material).

    Example: ``tai keys list``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/auth/tokens-payload")
    emit_records(ctx_obj, data, ["user_id", "description", "scopes"])


@app.command("create")
@covers(("POST", "/api/auth/api-keys"))
def create_key(
    ctx: typer.Context,
    user: Annotated[str, typer.Option("--user", help="The key's user id.")],
    description: Annotated[str, typer.Option("--description", help="Human description (required identity field).")],
    scope: Annotated[list[str] | None, typer.Option("--scope", help="A scope to grant (repeatable).")] = None,
    condition: Annotated[str | None, typer.Option("--condition", help="An inline jq authorization condition.")] = None,
    condition_id: Annotated[str | None, typer.Option("--condition-id", help="A stored jq condition id.")] = None,
    condition_kwargs: Annotated[str | None, typer.Option("--condition-kwargs", help=_CONDITION_KWARGS_HELP)] = None,
    policy_data: Annotated[
        str | None, typer.Option("--policy-data", help="Extra policy data as a JSON object.")
    ] = None,
) -> None:
    """Provision an API key; the raw ``sk-…`` value is printed ONCE.

    Example: ``tai keys create --user alice --description 'CI key' --scope read``
    """
    ctx_obj = app_context(ctx)
    body: dict = {"user_id": user, "description": description, "scopes": list(scope or [])}
    if condition is not None:
        body["condition"] = condition
    if condition_id is not None:
        body["condition_id"] = condition_id
    if condition_kwargs is not None:
        body["condition_kwargs"] = parse_json_object(condition_kwargs, param_hint="--condition-kwargs")
    if policy_data is not None:
        body["policy_data"] = parse_json_object(policy_data, param_hint="--policy-data")
    with ctx_obj.client() as client:
        data = client.post("/api/auth/api-keys", json=body)
    emit_result(ctx_obj, data)


@app.command("edit")
@covers(("PUT", "/api/auth/api-keys/{user_id}"))
def edit_key(
    ctx: typer.Context,
    user: Annotated[str, typer.Argument(help="The key's user id.")],
    description: Annotated[str | None, typer.Option("--description", help="New description.")] = None,
    scope: Annotated[
        list[str] | None, typer.Option("--scope", help="Replacement scope (repeatable); replaces the set.")
    ] = None,
    condition: Annotated[str | None, typer.Option("--condition", help="New condition; pass '' to clear.")] = None,
    condition_id: Annotated[
        str | None, typer.Option("--condition-id", help="New condition id; pass '' to clear.")
    ] = None,
    condition_kwargs: Annotated[
        str | None, typer.Option("--condition-kwargs", help="Condition kwargs JSON; '{}' clears.")
    ] = None,
    policy_data: Annotated[str | None, typer.Option("--policy-data", help="Policy data JSON; '{}' clears.")] = None,
) -> None:
    """Partially edit a key's description/scopes/policy in place (no rotation).

    Only the flags you pass are written; omitted fields are preserved. Pass an
    empty value to clear an optional condition gate.

    De-scoping this key (or its owner) also NARROWS what every hook and trigger link
    bound to it as its ``execution_key`` may call at its next fire — see ``tai hooks
    list`` for which records bind it.

    Example: ``tai keys edit alice --scope read --scope write``
    """
    ctx_obj = app_context(ctx)
    updates: dict = {}
    if description is not None:
        updates["description"] = description
    if scope:
        updates["scopes"] = list(scope)
    if condition is not None:
        updates["condition"] = condition
    if condition_id is not None:
        updates["condition_id"] = condition_id
    if condition_kwargs is not None:
        updates["condition_kwargs"] = parse_json_object(condition_kwargs, param_hint="--condition-kwargs")
    if policy_data is not None:
        updates["policy_data"] = parse_json_object(policy_data, param_hint="--policy-data")
    if not updates:
        raise typer.BadParameter("provide at least one field to edit")
    with ctx_obj.client() as client:
        data = client.put(f"/api/auth/api-keys/{user}", json=updates)
    emit_result(ctx_obj, data)


@app.command("delete")
@covers(("DELETE", "/api/auth/api-keys/{user_id}"))
def delete_key(ctx: typer.Context, user: Annotated[str, typer.Argument(help="The key's user id.")]) -> None:
    """Revoke a key (immediate: the next request fails to auth).

    Revoking also STOPS every hook and trigger link bound to this key as its
    ``execution_key`` — their next fire is refused. Run ``tai hooks list`` and
    ``tai hooks trigger-links`` first to see which records bind it.

    Example: ``tai keys delete alice``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.delete(f"/api/auth/api-keys/{user}")
    emit_result(ctx_obj, data)


@app.command("claim-link")
@covers(("POST", "/api/auth/claim-links"))
def claim_link(
    ctx: typer.Context,
    raw_key: Annotated[
        str | None,
        typer.Argument(
            help="The raw API key to share (omit to enter it at a hidden prompt so it never lands in shell history)."
        ),
    ] = None,
    ttl: Annotated[
        int | None, typer.Option("--ttl", help="Link lifetime in seconds (capped by the server ceiling).")
    ] = None,
) -> None:
    """Mint a one-time claim link that carries a key you hold to another device.

    The link's token rides the URL FRAGMENT (``/login#claim=<token>``) and is
    SINGLE-USE — the first exchange burns it. Prints the claim path and expiry; compose
    the absolute URL (or a QR) from your own origin.

    Example: ``tai keys claim-link sk-abc123 --ttl 300``
    """
    ctx_obj = app_context(ctx)
    key = raw_key if raw_key is not None else typer.prompt("API key to share", hide_input=True)
    body: dict = {"api_key": key}
    if ttl is not None:
        body["ttl_seconds"] = ttl
    with ctx_obj.client() as client:
        data = client.post("/api/auth/claim-links", json=body)
    emit_result(ctx_obj, data)


@app.command("validate-condition")
@covers(("POST", "/api/auth/validate-condition"))
def validate_condition(
    ctx: typer.Context,
    condition: Annotated[str | None, typer.Option("--condition", help="An inline jq condition to compile.")] = None,
    condition_id: Annotated[str | None, typer.Option("--condition-id", help="A stored jq condition id.")] = None,
    condition_kwargs: Annotated[str | None, typer.Option("--condition-kwargs", help=_CONDITION_KWARGS_HELP)] = None,
    sample_context: Annotated[
        str | None, typer.Option("--sample-context", help="A JqAuthContext-shaped sample to evaluate against, as JSON.")
    ] = None,
) -> None:
    """Compile (and optionally sample-evaluate) a jq policy condition without saving.

    Example: ``tai keys validate-condition --condition '.method == "GET"'``
    """
    ctx_obj = app_context(ctx)
    body: dict = {}
    if condition is not None:
        body["condition"] = condition
    if condition_id is not None:
        body["condition_id"] = condition_id
    if condition_kwargs is not None:
        body["condition_kwargs"] = parse_json_object(condition_kwargs, param_hint="--condition-kwargs")
    if sample_context is not None:
        body["sample_context"] = parse_json_object(sample_context, param_hint="--sample-context")
    with ctx_obj.client() as client:
        data = client.post("/api/auth/validate-condition", json=body)
    emit_result(ctx_obj, data)


@app.command("policy-versions")
@covers(("GET", "/api/auth/api-keys/{user_id}/policy/versions"))
def policy_versions(ctx: typer.Context, user: Annotated[str, typer.Argument(help="The key's user id.")]) -> None:
    """List a user's append-only policy version history.

    Example: ``tai keys policy-versions alice``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get(f"/api/auth/api-keys/{user}/policy/versions")
    emit_records(ctx_obj, data, ["version", "is_current", "created_at"])


@app.command("policy-rollback")
@covers(("POST", "/api/auth/api-keys/{user_id}/policy/rollback"))
def policy_rollback(
    ctx: typer.Context,
    user: Annotated[str, typer.Argument(help="The key's user id.")],
    version: Annotated[int, typer.Argument(help="Target policy version to enforce.")],
) -> None:
    """Roll a user's enforced policy back to a prior version.

    Example: ``tai keys policy-rollback alice 2``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.post(f"/api/auth/api-keys/{user}/policy/rollback", json={"version": version})
    emit_result(ctx_obj, data)
