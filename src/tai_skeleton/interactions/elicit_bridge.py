"""Bridge FastMCP elicitation onto the interactions ``ask_user`` channel.

A tool's ``ctx.elicit()`` normally resolves only when the calling client
supports elicitation; an in-process caller (agent, webhook-triggered run,
scheduled backend) has no client to answer, so the call dead-ends. This bridge
routes such an elicit through the SAME human-in-the-loop waiter/store as
``ask_user`` (the interactions system), so a human answers on any interactions
channel and the typed result flows back to the caller.

The skeleton's in-process Context-injection layer (the ``run_tool`` path) drives
this: the in-process bridge context (see
:mod:`tai_skeleton.tools.context_bridge`) routes ``ctx.elicit`` through
:func:`resolve_elicit`, which DERIVES the schema from the ``response_type`` (a
Python type) via ``parse_elicit_response_type`` and carries it as the form
answer-schema.

Accept-or-raise: a validated form answer maps to ``AcceptedElicitation``; a
timeout / no-answer RAISES (the no-answerable-path invariant). The bridge never
rounds a decline or cancel through ``ask_user`` — an unanswerable elicit is a
loud error, never a silent default.
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp.server.elicitation import (
    AcceptedElicitation,
    handle_elicit_accept,
    parse_elicit_response_type,
)

from tai_skeleton.interactions.helper import ask_user

logger = logging.getLogger(__name__)


async def answer_elicit_via_ask_user(message: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Ask a human ``message`` with ``schema`` as the form answer-schema and
    return the validated answer dict. Schema fidelity is the point: the schema
    is carried through intact so the caller gets exactly the shape it asked for.
    A timeout / no-answer raises out of ``ask_user`` (accept-or-raise); nothing
    is swallowed."""
    return await ask_user(message, answer_format="form", schema=schema)


async def resolve_elicit(
    message: str,
    response_type: Any = None,
    *,
    response_title: str | None = None,
    response_description: str | None = None,
) -> AcceptedElicitation[Any]:
    """DERIVE the elicit schema from a Python ``response_type``, ask a
    human through ``ask_user``, then map the validated answer back to the typed
    ``AcceptedElicitation``. No decline/cancel round-trip — a no-answer raises."""
    config = parse_elicit_response_type(
        response_type,
        response_title=response_title,
        response_description=response_description,
    )
    answer = await answer_elicit_via_ask_user(message, config.schema)
    return handle_elicit_accept(config, answer)
