"""Bridge FastMCP sampling (``ctx.sample()``) onto the platform's own LLM.

A tool calling ``ctx.sample()`` asks the CALLER's LLM. When the caller advertises
sampling that path is used natively; when it does not (an in-process agent /
webhook / scheduled backend) the call would dead-end. This bridge falls back to
the platform's own configured LLM (the
generic ``tai42_kit`` LLM helper). The fallback is EXPLICIT and LOGGED — never a
silent substitution.

Scope: a single-shot text completion (system prompt + messages -> text). Tool
loops and structured ``result_type`` sampling are a caller-LLM capability the
platform fallback does not reproduce, so they raise loudly rather than being
silently degraded to a plain completion.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from mcp.types import SamplingMessage, TextContent

from tai42_skeleton.tools.sampling_settings import sampling_settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def platform_llm() -> BaseChatModel:
    """The platform's own configured default chat model — the sampling fallback.
    Built from the provider + LLM settings, the same construction the platform's
    LLM middleware uses."""
    from tai42_kit.llm.models import get_llm
    from tai42_kit.llm.settings import llm_provider_settings, llm_settings

    return get_llm(llm_provider_settings().llm, **llm_settings().model_dump(exclude_none=True))


def _to_langchain_messages(
    messages: str | Sequence[str | SamplingMessage],
    system_prompt: str | None,
) -> list[BaseMessage]:
    lc: list[BaseMessage] = []
    if system_prompt:
        lc.append(SystemMessage(content=system_prompt))
    items: Sequence[str | SamplingMessage] = [messages] if isinstance(messages, str) else messages
    for item in items:
        if isinstance(item, str):
            lc.append(HumanMessage(content=item))
            continue
        # SamplingMessage: text content only (the platform fallback is a text
        # completion). A non-text content block cannot be forwarded to the
        # generic chat model, so it raises rather than being dropped.
        content = item.content
        if not isinstance(content, TextContent):
            raise ValueError(
                f"sampling fallback supports text content only; got {type(content).__name__} "
                "in a SamplingMessage — the platform LLM fallback cannot forward it."
            )
        lc.append(AIMessage(content=content.text) if item.role == "assistant" else HumanMessage(content=content.text))
    return lc


async def platform_sample(
    messages: str | Sequence[str | SamplingMessage],
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: Any = None,
    result_type: Any = None,
) -> str:
    """Run a single-shot completion on the platform LLM and return the text.

    Logged as an explicit fallback so it is never a silent substitution for the
    caller's model. ``tools`` / ``result_type`` are caller-LLM capabilities the
    fallback does not reproduce, so they raise rather than silently degrade.

    ``max_tokens`` is bounded by ``TAI_SAMPLING_MAX_TOKENS_PER_CALL``: a caller
    passing none gets the cap as the default; a caller asking for more is refused
    loudly (never silently clamped). Any delegate with no invocation scope of its
    own (a single downstream sample per call) inherits this token cap here — it
    has no per-invocation call budget."""
    if tools:
        raise NotImplementedError(
            "sampling fallback to the platform LLM does not support a tool loop; "
            "only a caller that advertises sampling can run tools through ctx.sample()."
        )
    if result_type is not None:
        raise NotImplementedError(
            "sampling fallback to the platform LLM does not support structured result_type; "
            "only a caller that advertises sampling can return a structured sample."
        )

    cap = sampling_settings().max_tokens_per_call
    if max_tokens is None:
        max_tokens = cap
    elif max_tokens > cap:
        raise ValueError(
            f"sampling fallback max_tokens {max_tokens} exceeds the platform cap {cap} "
            "(TAI_SAMPLING_MAX_TOKENS_PER_CALL)"
        )

    model = platform_llm()
    bind_kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        bind_kwargs["temperature"] = temperature
    bound = model.bind(**bind_kwargs)

    logger.info("ctx.sample(): caller advertised no sampling capability — falling back to the platform LLM")
    result = await bound.ainvoke(_to_langchain_messages(messages, system_prompt))
    text = result.content
    if not isinstance(text, str):
        # A multimodal/structured content list from the model cannot stand in for
        # the plain-text sample the caller expects; surface it rather than coerce.
        raise TypeError(f"platform LLM returned non-text content ({type(text).__name__}) for a text sample")
    return text
