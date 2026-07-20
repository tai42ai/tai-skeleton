"""Interactions feature — the ``ask_user`` core capability.

``ask_user`` (in ``helper``) is the engine-agnostic author surface satisfying the
``tai42_contract.interactions.AskUser`` protocol; the request/response/state models
and the ``AnswerFormat`` enum are re-exported from the contract; ``store`` holds
the Redis key shapes + operations shared with the API; ``settings`` the
``INTERACTIONS_*`` config.
"""

from tai42_contract.interactions import (
    AnswerFormat,
    AskUser,
    InteractionRequest,
    InteractionResponse,
    InteractionState,
)

from tai42_skeleton.interactions.helper import (
    InteractionLimitError,
    InteractionTimeoutError,
    ask_user,
)
from tai42_skeleton.interactions.settings import (
    InteractionsSettings,
    interactions_settings,
)
from tai42_skeleton.interactions.store import InteractionStore

__all__ = [
    "AnswerFormat",
    "AskUser",
    "InteractionLimitError",
    "InteractionRequest",
    "InteractionResponse",
    "InteractionState",
    "InteractionStore",
    "InteractionTimeoutError",
    "InteractionsSettings",
    "ask_user",
    "interactions_settings",
]
