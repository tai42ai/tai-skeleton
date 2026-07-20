"""The skeleton interactions impl conforms to the ``tai42_contract.interactions``
surface: the ``ask_user`` helper satisfies the ``AskUser`` protocol with the
exact call signature, and the models round-trip through JSON.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

from tai42_contract.interactions import (
    AnswerFormat,
    AskUser,
    InteractionRequest,
    InteractionResponse,
    InteractionState,
)

from tai42_skeleton import interactions as _skeleton_interactions
from tai42_skeleton.interactions import ask_user

# Re-exported through the skeleton namespace; reference via the module so the
# re-export identity assertion below stays meaningful.
SkeletonInteractionRequest = _skeleton_interactions.InteractionRequest


def test_ask_user_satisfies_protocol():
    assert isinstance(ask_user, AskUser)


def test_ask_user_signature_matches_protocol():
    impl = inspect.signature(ask_user)
    proto = inspect.signature(AskUser.__call__)
    proto_params = [p for name, p in proto.parameters.items() if name != "self"]
    assert list(impl.parameters.values()) == proto_params


def test_skeleton_reexports_contract_models():
    # The skeleton must not redefine the models — it re-exports the contract's.
    assert SkeletonInteractionRequest is InteractionRequest


def test_request_round_trip():
    request = InteractionRequest(
        interaction_id="i1",
        group_id="g1",
        question="pick one",
        answer_format=AnswerFormat.SELECT,
        format_payload={"options": ["a", "b"]},
        reply_to="interactions:reply:i1",
        created_at=datetime.now(UTC),
        timeout_at=datetime.now(UTC),
    )
    assert InteractionRequest.model_validate_json(request.model_dump_json()) == request


def test_state_round_trip_with_response():
    request = InteractionRequest(
        interaction_id="i1",
        group_id="g1",
        question="ok?",
        answer_format=AnswerFormat.CONFIRM,
        reply_to="interactions:reply:i1",
        created_at=datetime.now(UTC),
        timeout_at=datetime.now(UTC),
    )
    response = InteractionResponse(
        interaction_id="i1",
        answer=True,
        answered_by="tester",
        answered_at=datetime.now(UTC),
    )
    state = InteractionState(status="answered", group_id="g1", request=request, response=response)
    assert InteractionState.model_validate_json(state.model_dump_json()) == state
