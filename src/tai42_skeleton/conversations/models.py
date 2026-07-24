"""The host-internal answer/delivery record — transient runtime state for one accepted
message. Not a contract type; the wire shapes live in :mod:`tai42_contract.conversations`.

``delivery_status`` and ``answer_status`` are ORTHOGONAL: ``answer_status`` is the nature
of the turn's outcome, fixed when the turn completes; ``delivery_status`` is where that
outcome sits in the send machine.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator
from tai42_contract.conversations import AnswerStatus, ConversationAnswer, ConversationDoor


class DeliveryStatus(StrEnum):
    """Where a record sits between intake and a terminal outcome.

    ``accepted`` is pre-turn intake and carries no answer; ``pending_delivery`` is
    persisted-but-unsent (what a re-drive resumes); ``provisional`` is sent and awaiting an
    out-of-band receipt or grace expiry; ``delivered``/``failed``/``shed`` are terminal and
    are the only states carrying the retention TTL. ``shed`` ran no turn and never sends.
    """

    ACCEPTED = "accepted"
    PENDING_DELIVERY = "pending_delivery"
    PROVISIONAL = "provisional"
    DELIVERED = "delivered"
    FAILED = "failed"
    SHED = "shed"


#: The states nothing drives further; the retention TTL is applied on reaching one.
TERMINAL_STATUSES = frozenset({DeliveryStatus.DELIVERED, DeliveryStatus.FAILED, DeliveryStatus.SHED})

#: The states carrying no produced answer; every other state carries one.
ANSWERLESS_STATUSES = frozenset({DeliveryStatus.ACCEPTED, DeliveryStatus.SHED})


class ConversationRecord(BaseModel):
    """One accepted message's durable record — its admission, the answer its turn produced
    and its delivery state. Frozen: a store read is a snapshot, and a transition is a fresh
    write through the record store's atomic seam.
    """

    model_config = ConfigDict(frozen=True)

    message_id: str = Field(min_length=1)
    route_name: str = Field(min_length=1)
    door: ConversationDoor
    thread_id: str = Field(min_length=1)
    client_address: str = Field(min_length=1)

    # door=channel delivery target: the channel to notify and the identity to send FROM.
    channel: str | None = None
    our_identity: str | None = None
    # door=channel intake: the provider's id this record was deduped under. ``None`` for an
    # api-door record; never blank, which would share one marker with every other blank id.
    provider_message_id: str | None = Field(default=None, min_length=1)
    # door=api delivery target. The signing secret is NOT stored here — the executor reads
    # it live from the route row at send.
    callback_url: str | None = None

    # The api-door caller the turn was invoked by. ``None`` for a channel record, which is
    # then admin-only to read.
    caller_principal: str | None = None

    # ``None`` exactly while the record carries no answer, set on every other status.
    answer_status: AnswerStatus | None = None
    # Client-facing text; for an ``error`` turn the internal detail lives in ``error``.
    answer: str | None = None
    error: str | None = None

    delivery_status: DeliveryStatus = DeliveryStatus.PENDING_DELIVERY
    # Provider-assigned ids of this record's sends, correlated by out-of-band receipts.
    outbound_message_ids: list[str] = Field(default_factory=list)
    attempts: int = 0

    created_at: float
    updated_at: float

    @model_validator(mode="after")
    def _outcome_matches_status(self) -> ConversationRecord:
        """A record carries a turn outcome exactly when its status says it has one, so
        nothing reaching the delivery machine can be missing the answer it must send."""
        answerless = self.delivery_status in ANSWERLESS_STATUSES
        if answerless != (self.answer_status is None):
            raise ValueError(
                f"delivery_status {self.delivery_status.value!r} and answer_status "
                f"{self.answer_status!r} disagree on whether this record carries an answer"
            )
        if self.answer_status is not None and not (self.answer or "").strip():
            raise ValueError("a record carrying an answer_status must carry non-blank answer text")
        return self

    def answer_payload(self) -> ConversationAnswer:
        """The :class:`ConversationAnswer` this record delivers — the one shape both the
        signed callback body and the sync-wait payload carry. Raises on an answerless
        record."""
        if self.answer_status is None or self.answer is None:
            raise RuntimeError(
                f"conversation record {self.message_id!r} is {self.delivery_status.value} and carries no answer"
            )
        return ConversationAnswer(
            message_id=self.message_id,
            thread_id=self.thread_id,
            status=self.answer_status,
            answer=self.answer,
        )

    def view(self) -> dict[str, object]:
        """The record as an ADMIN read door returns it. Includes ``error``, the turn's raw
        internal detail, so it is only for a caller with authority over the route's key."""
        return self.model_dump(mode="json")

    def caller_view(self) -> dict[str, object]:
        """The record as the CALLER-scoped read door returns it: the message, its outcome
        and where delivery stands. An allow-list, so a newly added field stays withheld
        until deliberately published here. ``error`` and the delivery bookkeeping are
        withheld — the turn ran as the ROUTE's key, not the caller's.
        """
        return self.model_dump(
            mode="json",
            include={
                "message_id",
                "route_name",
                "door",
                "thread_id",
                "client_address",
                "caller_principal",
                "answer_status",
                "answer",
                "delivery_status",
                "created_at",
                "updated_at",
            },
        )


__all__ = ["ANSWERLESS_STATUSES", "TERMINAL_STATUSES", "ConversationRecord", "DeliveryStatus"]
