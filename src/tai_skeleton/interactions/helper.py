"""The author-facing ``ask_user`` surface — the ``AskUser`` contract impl.

Engine-agnostic: it reads no flow/engine context and depends on nothing the
engine threads. Each call generates its own ``interaction_id`` and an optional
caller ``group_id`` (uuid4 when absent), persists the question to Redis, then
blocks on a per-interaction reply channel until the answer returns or the
timeout budget elapses (loud ``InteractionTimeoutError`` — never a silent default).

The ``external`` answer format acts on an EXTERNAL surface (sign, approve, pay):
the caller blocks exactly as for any other format while the external system
delivers the answer through a public callback door. ``link`` supplies that
surface — a template carrying ``{callback_url}`` or a callable that builds the
external resource from the callback URL and returns its final URL.
"""

from __future__ import annotations

import asyncio
import logging
import math
import secrets
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel
from tai_contract.app import tai_app
from tai_contract.channels import Channel, ChannelDelivery, ChannelDeliveryError
from tai_contract.interactions import AnswerFormat, InteractionRequest, MediaItem
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.redis import RedisClient

from tai_skeleton.access_control.user import clamp_write_audience
from tai_skeleton.interactions.settings import InteractionsSettings, interactions_settings
from tai_skeleton.interactions.store import InteractionStore

logger = logging.getLogger(__name__)

_CALLBACK_PLACEHOLDER = "{callback_url}"


class InteractionTimeoutError(Exception):
    """Raised when ``ask_user`` gets no answer within its timeout budget."""


class InteractionLimitError(Exception):
    """Raised when a new ``ask_user`` call is refused because too many questions
    are already open (the ``max_concurrent`` guard)."""


def _normalize_schema(schema: type[BaseModel] | dict[str, Any]) -> dict:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    if isinstance(schema, dict):
        return schema
    raise ValueError("schema must be a pydantic model or a JSON-schema dict")


def _build_payload(
    answer_format: AnswerFormat,
    options: list[str] | None,
    schema: type[BaseModel] | dict[str, Any] | None,
    url: str | None = None,
    verifier: dict[str, Any] | None = None,
) -> dict | None:
    if answer_format is AnswerFormat.SELECT:
        if not options:
            raise ValueError("answer_format 'select' requires options")
        return {"options": options}
    if answer_format is AnswerFormat.FORM:
        if schema is None:
            raise ValueError("answer_format 'form' requires a schema")
        return {"schema": _normalize_schema(schema)}
    if answer_format is AnswerFormat.EXTERNAL:
        # The URL exists only after the link is resolved, so this branch is called
        # after that step; schema (optional here) validates the callback payload.
        # A ``verifier`` (``{"name", "config"}``) rides the payload server-side so
        # the callback route can authenticate the signed server-to-server answer;
        # the client-facing serialization strips it (see ``routers.interactions``).
        payload: dict[str, Any] = {"url": url, **({"schema": _normalize_schema(schema)} if schema is not None else {})}
        if verifier is not None:
            payload["verifier"] = verifier
        return payload
    return None


async def _resolve_link(link: str | Callable[[str], Awaitable[str]], callback_url: str) -> str:
    """Turn the ``link`` argument into the final external URL the human visits."""
    if isinstance(link, str):
        if _CALLBACK_PLACEHOLDER not in link:
            raise ValueError(f"template link must contain {_CALLBACK_PLACEHOLDER}")
        # ``replace`` not ``format``: other braces in a real URL must survive.
        return link.replace(_CALLBACK_PLACEHOLDER, callback_url)
    # Callable flavor: it creates the external resource and returns its URL. An
    # exception from the builder propagates unchanged — nothing is persisted yet.
    final = await link(callback_url)
    if not isinstance(final, str) or not final.startswith(("http://", "https://")):
        raise ValueError(f"link builder must return an http(s) URL, got {final!r}")
    return final


def _validate_verifier(verifier: Any) -> None:
    """Reject a malformed or unknown ``verifier`` at ask-time, before any state is
    written. It must be a dict carrying a non-empty ``name`` that resolves against
    the registered webhook verifiers. A non-dict (or a typo'd/unregistered name)
    would otherwise slip through as an unrecognised binding at the callback door
    and silently degrade the question to an open, unverified one — so this is a
    hard guard (raise), never a soft ignore."""
    name = verifier.get("name") if isinstance(verifier, dict) else None
    if not isinstance(name, str) or not name:
        raise ValueError("verifier must be a dict with a non-empty 'name'")
    try:
        tai_app.webhook_verifiers.get(name)
    except Exception as exc:
        raise ValueError(f"unknown webhook verifier: {name!r}") from exc


def _validate_channel(channel: Any) -> Channel:
    """Reject a malformed or unknown ``channel`` at ask-time, before any state
    is written. It must be a non-empty string naming a registered channel — an
    unknown name would otherwise persist a question no deliverer can ever push
    to a human, leaving the caller blocked until timeout. A hard guard (raise),
    never a soft ignore. Returns the resolved channel object; delivery reuses
    this exact validated instance, so a registry change between validation and
    delivery can never surface as a post-persist lookup failure."""
    if not isinstance(channel, str) or not channel:
        raise ValueError("channel must be a non-empty string")
    try:
        return tai_app.channels.get(channel)
    except KeyError as exc:
        raise ValueError(f"unknown channel: {channel!r}") from exc


async def _prune(settings: InteractionsSettings, store: InteractionStore, interaction_id: str, group_id: str) -> bool:
    """Prune an abandoned question on its OWN connection — never the cancelled
    BLPOP connection, which is not safely reusable for a WATCH/MULTI. Returns
    ``prune_pending``'s result: ``True`` when it pruned, ``False`` when there
    was nothing to prune (already answered, or already gone)."""
    async with client_ctx(RedisClient, settings.redis) as conn:
        return await store.prune_pending(conn, interaction_id, group_id)


async def ask_user(
    question: str,
    *,
    answer_format: str = "text",
    options: list[str] | None = None,
    schema: type[BaseModel] | dict[str, Any] | None = None,
    group_id: str | None = None,
    timeout: float | None = None,
    link: str | Callable[[str], Awaitable[str]] | None = None,
    verifier: dict[str, Any] | None = None,
    channel: str | None = None,
    recipient: str | None = None,
    sensitive: bool = False,
    audience: str | None = None,
    media: list[MediaItem | dict[str, Any]] | None = None,
) -> Any:
    """Ask a human ``question`` and block until the answer returns.

    Returns the typed answer per ``answer_format`` (text->str, confirm->bool,
    select->chosen value, form->validated dict, external->the callback payload).
    Raises ``InteractionTimeoutError`` on expiry, ``InteractionLimitError`` when
    the ``max_concurrent`` guard trips, ``ValueError`` for a bad format/argument
    combination or a blank ``audience``, ``CrossIdentityAudienceError`` when a
    RESTRICTED caller addresses another identity (a loud cross-identity authorization
    denial), and ``RuntimeError`` when an external question is asked without
    ``INTERACTIONS_PUBLIC_BASE_URL``. Invalid ``media`` raises
    ``pydantic.ValidationError`` when the ``InteractionRequest`` is built, before
    any state is written.

    ``link`` is required for ``answer_format="external"`` (unless a ``channel``
    delivers the question) and forbidden otherwise.

    ``verifier`` (``{"name", "config"}``) binds a registered webhook verifier to
    the external callback so the signed server-to-server answer is authenticated
    before it is recorded; it is only valid with ``answer_format="external"`` (a
    verifier is meaningless without the external callback route). It is stashed
    server-side in the ``format_payload`` and stripped from the client frame.

    ``sensitive`` marks the answer body as not-to-be-persisted: the caller still
    receives the full answer, but the durable answered record keeps only the
    status (no response body). Use it for credentials or personal data.

    ``channel`` names a registered channel that delivers the question to a human
    on an external medium; ``None`` keeps the default Studio-inbox-only surface.
    A set channel forces the ticket + callback-URL mint for EVERY answer format
    (the channel bridges the reply back through the public callback door),
    forbids ``link`` and ``verifier`` (the channel owns delivery, and its
    forward is unsigned), and rejects ``answer_format="form"`` (a multi-field
    form has no single-reply mapping on a chat/SMS medium). An unknown name
    raises ``ValueError`` before any state is written. A failed delivery prunes
    the question and re-raises (``ChannelDeliveryError`` for a delivery
    failure, including a deliver call that does not return within the ask's
    timeout budget) — unless the reply already landed first, in which case the
    recorded answer is returned.

    ``recipient`` is an OPTIONAL per-call address (chat id, phone number, ...)
    carried to the named channel, which validates it against its operator
    allowlist — an unlisted address makes the delivery fail loudly; omitted,
    the channel sends to its operator-configured default recipient. Nothing is
    resolved or validated here beyond presence and non-emptiness (a set value
    must be a non-blank string): the plugin owns the allowlist. ``recipient``
    is forbidden when ``channel`` is ``None`` (an address is meaningless
    without a channel to send on).

    ``audience`` is the identity (a user_id) the question is addressed to:
    a restricted identity sees and answers ONLY questions addressed to it, while an
    unrestricted operator sees and may answer everything. Leave it unset for an
    operator/broadcast question. It is the isolation axis — a WHO, distinct from
    ``recipient`` (a channel delivery address — a WHERE) — and the two may be set
    together (address the question to identity A AND deliver it over a channel).

    ``media`` is optional display-only content rendered WITH the question in the
    Studio inbox — a list of ``MediaItem`` (or their dict form) each
    ``{"kind": "image"|"link", "url", "caption"?}``. An ``image`` url is an
    absolute ``https`` URL or a ``data:image/*`` URI; a ``link`` url is an
    absolute ``http(s)`` URL; ``caption`` is the image alt text / link label. At
    most eight items, within a per-question total URI budget. It never becomes
    part of the answer — the human still answers via ``answer_format`` — and it is
    NOT forwarded to channel deliveries: a channel receives the question text
    only, while the inbox is where the media renders.
    """
    try:
        fmt = AnswerFormat(answer_format)
    except ValueError as exc:
        raise ValueError(f"unknown answer_format: {answer_format!r}") from exc

    is_external = fmt is AnswerFormat.EXTERNAL
    # ``audience`` (the addressed identity) is validated loud and up front — a
    # blank/whitespace value can never address a real identity — mirroring the
    # ``notify_user`` guard so both surfaces reject it identically before any
    # state is written.
    if audience is not None and (not isinstance(audience, str) or not audience.strip()):
        raise ValueError("audience must be a non-empty identity")
    # Write-side isolation clamp — before any state is written. A restricted caller
    # may address only its own slice: an unset audience is scoped to its own identity,
    # and any other identity is rejected loudly (cross-identity inject/exfil). An
    # unrestricted caller is unchanged.
    audience = clamp_write_audience(audience)
    # Channel validation, loud and up front — before any state is written
    # (mirrors the verifier guard, and mirrors the ``link`` guard's shape).
    # The resolved object is kept for the delivery below.
    channel_obj: Channel | None = None
    if channel is not None:
        channel_obj = _validate_channel(channel)
        if link is not None:
            # The channel owns the delivery surface for every format.
            raise ValueError("link is forbidden when a channel is set (the channel owns delivery)")
        if verifier is not None:
            # A channel's forward to the callback door is unsigned, so a bound
            # verifier would 401 every reply — the question could never be
            # answered. Reject loudly, never persist an unanswerable question.
            raise ValueError("verifier is forbidden when a channel is set (the channel forward is unsigned)")
        if fmt is AnswerFormat.FORM:
            # A multi-field form has no single-reply mapping on a chat/SMS
            # medium; the channel-supported set is text|confirm|select|external.
            raise ValueError("answer_format 'form' is not supported over a channel")
        if options is not None and fmt is not AnswerFormat.SELECT:
            raise ValueError("options are only valid with answer_format 'select'")
        if recipient is not None and (not isinstance(recipient, str) or not recipient.strip()):
            # Rejected up-front as a clean ValueError — never a post-persist
            # pydantic error from the delivery frame's own recipient validator.
            raise ValueError("recipient must be a non-empty address")
    elif recipient is not None:
        # An address is meaningless without a channel to send on; the named
        # channel is what carries (and allowlist-validates) the recipient.
        raise ValueError("recipient requires a channel (an address is meaningless without one)")
    # Combo validation, loud and up front. For external, the schema is normalized
    # here too so a bad schema fails BEFORE the link builder does external work.
    if is_external:
        if link is None and channel is None:
            raise ValueError("answer_format 'external' requires a link (or a channel)")
        if options is not None:
            raise ValueError("answer_format 'external' does not accept options")
        if schema is not None:
            _normalize_schema(schema)
        if verifier is not None:
            _validate_verifier(verifier)
    else:
        if link is not None:
            raise ValueError("link is only valid with answer_format 'external'")
        # A verifier authenticates the external server-to-server callback; on a
        # human-answerable format it would emit ``server_verified`` and make the
        # UI render a non-actionable card no human can ever answer. Reject it
        # loudly, mirroring the ``link`` guard — a hard guard, not a soft ignore.
        if verifier is not None:
            raise ValueError("verifier is only valid with answer_format 'external'")

    settings = interactions_settings()
    budget = settings.answer_timeout_seconds if timeout is None else timeout
    if budget <= 0:
        # Redis BLPOP treats 0 as "block forever" — the opposite of no-wait —
        # so a non-positive budget can never mean anything sane here.
        raise ValueError(f"timeout must be positive, got {budget!r}")
    created_at = datetime.now(UTC)
    timeout_at = created_at + timedelta(seconds=budget)

    interaction_id = str(uuid.uuid4())
    group = group_id or str(uuid.uuid4())
    store = InteractionStore(settings.key_prefix)
    reply_to = store.reply_key(interaction_id)

    ticket: str | None = None
    ticket_ttl: int | None = None
    callback_url: str | None = None
    if is_external or channel is not None:
        # A channel bridges the human's reply back through the public callback
        # door, so a set channel forces the ticket + callback-URL mint for EVERY
        # answer format. The settings validator guarantees a set public_base_url
        # is https:// (or localhost http://); only absence is checked here.
        if settings.public_base_url is None:
            raise RuntimeError(
                "external answer_format (and channel delivery) requires INTERACTIONS_PUBLIC_BASE_URL to be set"
            )
        ticket = secrets.token_urlsafe(32)
        # TTL = the question's timeout budget, ceiled so the ticket always
        # outlives the waiter's deadline (floor 1s); the ticket expires on this
        # TTL and is never deleted on claim.
        ticket_ttl = max(1, math.ceil(budget))
        callback_url = f"{settings.public_base_url.rstrip('/')}/api/interactions/callback/{ticket}"

    if is_external:
        assert callback_url is not None  # is_external forces the mint above
        if channel is not None:
            # Channel-delivered external ask: the channel presents the tappable
            # URL and that URL IS the callback door (GET confirm page / POST
            # answer sink) — no link builder runs.
            final_url = callback_url
        else:
            # ``link`` is non-None here (validated above); resolve BEFORE any
            # persist so a failed builder leaves zero state.
            final_url = await _resolve_link(link, callback_url)  # type: ignore[arg-type]
        format_payload = _build_payload(fmt, options, schema, url=final_url, verifier=verifier)
    else:
        format_payload = _build_payload(fmt, options, schema)

    request = InteractionRequest(
        interaction_id=interaction_id,
        group_id=group,
        question=question,
        answer_format=fmt,
        format_payload=format_payload,
        reply_to=reply_to,
        created_at=created_at,
        timeout_at=timeout_at,
        sensitive=sensitive,
        channel=channel,
        audience=audience,
        # dicts here are coerced to ``MediaItem`` by the model's own validation —
        # the single source of media validation, which raises before any persist.
        media=media,  # type: ignore[arg-type]
    )

    async with client_ctx(RedisClient, settings.redis) as r:
        # Concurrency guard (all formats). ``reserve_open_slot`` prunes stale open
        # members, refuses at the cap, and reserves this question's open-index
        # member in ONE atomic step — so a concurrent burst admits exactly
        # ``max_concurrent`` callers and refuses the rest, with no check-then-act
        # overshoot. A reserved slot means ``add`` must skip re-adding the member.
        if settings.max_concurrent is not None:
            reserved = await store.reserve_open_slot(r, request, settings.max_concurrent)
            if not reserved:
                raise InteractionLimitError(
                    f"ask_user refused: already at the max_concurrent limit ({settings.max_concurrent})"
                )
            await store.add(
                r,
                request,
                settings.idle_ttl_seconds,
                ticket=ticket,
                ticket_ttl=ticket_ttl,
                open_member_reserved=True,
            )
        else:
            await store.add(r, request, settings.idle_ttl_seconds, ticket=ticket, ticket_ttl=ticket_ttl)

    # Deliver through the channel AFTER the question is persisted (the callback
    # ticket must be claimable before any human can act on it) and BEFORE the
    # blocking wait. A failed delivery normally means the human never received
    # the question, so the persisted state must not linger open/claimable:
    # prune, then re-raise loudly. The ONE exception: ``prune_pending``
    # returning False means the answer was already recorded (a fast reply beat
    # the failure) — a recorded answer is never discarded, so fall through to
    # the blocking wait, which returns it immediately.
    if channel is not None:
        assert channel_obj is not None  # resolved with the up-front validation
        assert callback_url is not None  # a set channel forces the mint above
        try:
            try:
                # ``deliver`` is one prompt send attempt; the whole answer
                # budget is a generous ceiling for it. A plugin that consumes
                # the budget is hung, and an unbounded await here would block
                # the caller forever with the question persisted — so the send
                # is bounded, and a timeout is a typed delivery failure.
                await asyncio.wait_for(
                    channel_obj.deliver(
                        ChannelDelivery(
                            interaction_id=interaction_id,
                            recipient=recipient,
                            question=question,
                            answer_format=fmt.value,
                            options=options,
                            callback_url=callback_url,
                            timeout_at=timeout_at,
                        )
                    ),
                    timeout=budget,
                )
            except TimeoutError as exc:
                raise ChannelDeliveryError(
                    f"channel {channel!r} delivery timed out after {budget}s (interaction {interaction_id})"
                ) from exc
        except BaseException as exc:
            pruned = await _prune(settings, store, interaction_id, group)
            if pruned or not isinstance(exc, Exception):
                # Pruned → nothing was answered; propagate the failure loudly.
                # A non-Exception (asyncio.CancelledError mid-send, SystemExit)
                # ALWAYS propagates — cancellation is never swallowed, even
                # when an answer was recorded.
                raise
            logger.warning(
                "channel %r delivery failed for interaction %s after the answer"
                " was already recorded; returning the recorded answer",
                channel,
                interaction_id,
                exc_info=exc,
            )

    # Block for the answer on a dedicated connection: a human-scale wait holds
    # its connection for the whole budget, so pinning one from the shared pool
    # would starve other concurrent ask_user calls once the pool is drained.
    try:
        # Strip the socket read timeout on this connection only: the BLPOP blocks
        # legitimately for the whole budget, so a blanket 5s read timeout would
        # kill it. The store wraps the BLPOP in an outer wait_for (budget + grace)
        # instead, so a black-holed redis still fails loudly.
        reply_redis = settings.redis.model_copy(update={"socket_timeout": None})
        async with client_ctx(RedisClient, reply_redis, fresh=True) as reply_conn:
            response = await store.wait_for_reply(reply_conn, reply_to, budget, settings.blocking_grace_seconds)
    except asyncio.CancelledError:
        # Prune on cancel so an abandoned question does not inflate the group
        # count / open index. The status gate makes the cancelled-after-answer
        # race a no-op. A cleanup failure propagates (chained on the
        # CancelledError context), never swallowed.
        await _prune(settings, store, interaction_id, group)
        raise
    if response is None:
        # Timeout: prune first, else the abandoned question inflates the group
        # count until the idle TTL and stays claimable by a late callback.
        await _prune(settings, store, interaction_id, group)
        raise InteractionTimeoutError(
            f"ask_user timed out after {budget}s with no answer (interaction {interaction_id})"
        )
    return response.answer
