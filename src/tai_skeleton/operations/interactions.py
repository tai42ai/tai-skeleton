"""The human answer operation for the ask_user interactions surface.

``answer_interaction`` is the authenticated human answer door
(``POST /api/interactions/{interaction_id}/answer``): the value is validated
server-side against the stored question's ``answer_format`` before the blocked
caller is woken; an invalid answer is rejected loudly and the caller stays
blocked. An EXTERNAL question is answered through its callback URL, never here.

The answer-validation helpers (``_validate_answer``, ``_schema_mismatch``, …),
the reply-TTL clamp, and the serializer-guarded claim live here because the
router's still-handler callback door shares the exact same rules — it imports
them from this module (the store claim, the typed-format validation, the reply
TTL). The router's HTTP-edge extractor reads/parses the request body (the byte
cap → 413, invalid JSON / missing ``answer`` → 400) and hands this operation the
already-parsed ``answer`` value.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import jsonschema
from pydantic import BaseModel
from pydantic_core import PydanticSerializationError
from tai_contract.interactions import (
    AnswerFormat,
    InteractionRequest,
    InteractionResponse,
)
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.redis import RedisClient

from tai_skeleton.access_control.user import request_identity
from tai_skeleton.interactions.settings import interactions_settings
from tai_skeleton.interactions.store import InteractionStore
from tai_skeleton.operations import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PayloadTooLargeError,
    operation,
)

# The synthetic label recorded when access control is disabled and no caller
# identity exists. A namespaced ``system:`` sentinel (mirroring the
# ``external-callback`` label the callback door records) cannot collide with a
# looked-up user id.
_NO_AUTH_ANSWERED_BY = "system:no-auth"


class _AnswerInvalid(Exception):
    """Raised when a human-door answer fails its stored-format validation."""


class InteractionAnswer(BaseModel):
    """An answer to a pending interaction — the ``answer`` value validated at
    runtime against the interaction's own answer schema."""

    answer: Any


def _reply_ttl(request: InteractionRequest) -> int:
    """Short TTL for the reply key ≈ the remaining timeout budget, so a late
    answer to a timed-out question expires instead of resurrecting it."""
    remaining = int((request.timeout_at - datetime.now(UTC)).total_seconds())
    return max(1, remaining)


def _schema_error_message(exc: Exception) -> str:
    """The 400 message for a schema mismatch — ``jsonschema`` errors carry a
    ``.message``; other validator failures fall back to ``str``."""
    return f"answer does not match schema: {getattr(exc, 'message', str(exc))}"


# Failures of validating/parsing untrusted input convert to a loud 400; any
# exception outside these sets is a server bug and propagates as a 500.
# RecursionError covers recursive schemas / deeply-nested answers blowing up the
# validator.
_SCHEMA_VALIDATION_ERRORS = (jsonschema.ValidationError, jsonschema.SchemaError, RecursionError)


def _schema_mismatch(answer: Any, schema: dict) -> str | None:
    """Validate ``answer`` against ``schema``; return the 400 message on a
    validation failure, ``None`` when the answer conforms."""
    try:
        jsonschema.validate(answer, schema)
    except _SCHEMA_VALIDATION_ERRORS as exc:
        return _schema_error_message(exc)
    return None


def _validate_answer(request: InteractionRequest, answer: Any) -> Any:
    """Validate ``answer`` against the stored format; raise ``_AnswerInvalid``
    (mapped to 400) on mismatch. Returns the validated value."""
    fmt = request.answer_format
    if fmt is AnswerFormat.TEXT:
        if not isinstance(answer, str):
            raise _AnswerInvalid("answer must be a string")
        return answer
    if fmt is AnswerFormat.CONFIRM:
        if not isinstance(answer, bool):
            raise _AnswerInvalid("answer must be a boolean")
        return answer
    if fmt is AnswerFormat.SELECT:
        options = (request.format_payload or {}).get("options", [])
        if answer not in options:
            raise _AnswerInvalid(f"answer must be one of {options}")
        return answer
    if fmt is AnswerFormat.FORM:
        if not isinstance(answer, dict):
            raise _AnswerInvalid("answer must be an object")
        schema = (request.format_payload or {}).get("schema")
        if not isinstance(schema, dict):
            raise _AnswerInvalid("question schema is invalid: missing or non-object schema")
        message = _schema_mismatch(answer, schema)
        if message is not None:
            raise _AnswerInvalid(message)
        return answer
    # EXTERNAL is rejected by the answer door before validation runs; any other
    # member reaching here is a server bug, never a client error.
    raise RuntimeError(f"unhandled answer_format: {fmt}")


async def _claim_or_serialization_error(
    store: InteractionStore,
    r: Any,
    response: InteractionResponse,
    group_id: str,
    reply_ttl: int,
    *,
    ticket: str | None = None,
    ticket_ttl: int | None = None,
) -> bool | None:
    """Call ``record_answer``, converting a serializer blowup on an untrusted
    answer into a loud-400 signal. A pathological answer (e.g. a deeply-nested JSON
    object that parsed fine but exceeds the serializer's depth) raises when the
    response is serialized — which happens at the top of ``record_answer`` before
    any Redis write, so catching it here leaves no partial state. Returns the
    claim result (``True``/``False``), or ``None`` to signal "serialization
    failed → answer the caller with a 400"."""
    try:
        return await store.record_answer(r, response, group_id, reply_ttl, ticket=ticket, ticket_ttl=ticket_ttl)
    except (PydanticSerializationError, RecursionError):
        return None


@operation(
    name="answer_interaction",
    summary="Answer a pending interaction",
    tags=["interactions"],
    destructive=True,
    errors=[BadRequestError, ConflictError, ForbiddenError, NotFoundError, PayloadTooLargeError],
    request_model=InteractionAnswer,
)
async def answer_interaction(interaction_id: str, answer: Any) -> dict:
    """Answer a pending interaction through the authenticated human door.

    Audience gate (after the existence/format/status guards): a question's
    ``audience`` identity OR any unrestricted caller (the operator can always unblock
    a stuck question) may answer; every OTHER restricted caller is a loud ``403``.
    An unaddressed question is answerable by any unrestricted caller and by no
    restricted caller. This gate is sound ONLY because a restricted caller can never
    obtain a question's callback ticket — the ticket is delivered exclusively over the
    configured channel (never on any read/stream frame), so the unauthenticated
    callback door stays the sole ticket-bearing surface and no filtered stream leaks
    it."""
    settings = interactions_settings()
    store = InteractionStore(settings.key_prefix)
    user_id, restricted = request_identity()

    async with client_ctx(RedisClient, settings.redis) as r:
        state = await store.get_state(r, interaction_id)
        if state is None:
            raise NotFoundError("Interaction not found")
        if state.request.answer_format is AnswerFormat.EXTERNAL:
            raise BadRequestError("external interactions are answered via their callback URL")
        if state.status == "answered":
            raise ConflictError("Interaction already answered")
        # A restricted caller may answer ONLY a question addressed to its identity;
        # an unrestricted caller may answer anything.
        if restricted is not None:
            if state.request.audience is None:
                raise ForbiddenError("restricted identities may answer only interactions addressed to them")
            if state.request.audience != restricted:
                raise ForbiddenError("interaction is addressed to another identity")
        try:
            validated = _validate_answer(state.request, answer)
        except _AnswerInvalid as exc:
            raise BadRequestError(str(exc)) from exc
        response = InteractionResponse(
            interaction_id=interaction_id,
            answer=validated,
            # The authenticated caller; with access control off
            # (ACCESS_CONTROL_ENABLE=false) no identity exists, so the reserved
            # no-auth sentinel is recorded.
            answered_by=user_id or _NO_AUTH_ANSWERED_BY,
            answered_at=datetime.now(UTC),
        )
        claimed = await _claim_or_serialization_error(store, r, response, state.group_id, _reply_ttl(state.request))
        if claimed is None:
            raise BadRequestError("answer payload could not be serialized")
        if not claimed:
            raise ConflictError("Interaction already answered")

    return {"interaction_id": interaction_id, "status": "answered"}
