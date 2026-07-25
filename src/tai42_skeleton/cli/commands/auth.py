"""``tai auth`` — identity and capability introspection.

Thin wrappers over the authed ``/api/auth/me`` route and the PUBLIC
``/api/login/claim`` claim-exchange door (``claim`` runs credential-free).
"""

from __future__ import annotations

from typing import Annotated

import typer

from tai42_skeleton.cli.commands._common import app_context, covers, emit_result

app = typer.Typer(
    name="auth",
    help="Identity and capability introspection.",
    no_args_is_help=True,
)

_CLAIM_FRAGMENT_MARKER = "#claim="


def _extract_claim_token(value: str) -> str:
    """The claim token from either a bare token or a full claim URL/fragment.

    The token rides the URL FRAGMENT (``…/login#claim=<token>``), so pasting the whole
    link just works: take the tail after ``#claim=``. A bare token (no marker) is
    returned as-is. The token is NEVER matched by its mint prefix — the server validates
    it, and a prefix check here would add nothing."""
    if _CLAIM_FRAGMENT_MARKER in value:
        value = value.rsplit(_CLAIM_FRAGMENT_MARKER, 1)[1]
    return value.strip()


@app.command("whoami")
@covers(("GET", "/api/auth/me"))
def whoami(ctx: typer.Context) -> None:
    """Print the caller's derived capability projection.

    Example: ``tai auth whoami``
    """
    ctx_obj = app_context(ctx)
    with ctx_obj.client() as client:
        data = client.get("/api/auth/me")
    emit_result(ctx_obj, data)


@app.command("claim")
@covers(("POST", "/api/login/claim"))
def claim(
    ctx: typer.Context,
    token: Annotated[str, typer.Argument(help="A claim token or a full claim URL/fragment (either works).")],
) -> None:
    """Exchange a one-time claim link for its API key — runs WITHOUT a credential.

    Pass the bare token or the whole claim URL; the token is taken from the ``#claim=``
    fragment. The exchanged API key is printed ONCE — capture it now (there is no second
    exchange; the link is single-use). A used/unknown/expired token answers the same
    ``unknown or already used claim token``.

    Example: ``tai auth claim 'https://host/login#claim=<token>'``
    """
    ctx_obj = app_context(ctx)
    claim_token = _extract_claim_token(token)
    # The caller has no key yet — this is the whole point — so the exchange runs over the
    # no-credential client path; a stale/wrong credential is never sent to the public door.
    with ctx_obj.client(anonymous=True) as client:
        data = client.post("/api/login/claim", json={"token": claim_token})
    emit_result(ctx_obj, data)
