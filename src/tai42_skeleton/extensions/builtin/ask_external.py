"""The ``ask_external`` builtin tool extension.

A tool extension extends a single tool (plugins extend the platform).
``ask_external`` is a TRANSFORMER: it wraps a tool that builds an external
resource from a ``callback_url`` and returns that resource's URL, presenting a
composed signature that drives the human-in-the-loop ``ask_user`` external flow.
The wrapped tool's own inputs stay; the injected control params ``question`` /
``answer_schema`` / ``timeout`` are added, and ``callback_url`` is hidden (the
platform supplies it).

The ``verifier`` that authenticates the signed server-to-server callback is
AUTHOR-BOUND: it is supplied as extension config on the manifest ``extensions``
entry (``{"name": "ask_external", "config": {"verifier": ...}}``) and closed over
at build time — it is NEVER an LLM-facing tool param, so a calling agent can
neither drop nor forge the callback authentication.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, cast

from makefun import create_function
from tai42_contract.app import tai42_app
from tai42_contract.extensions import ExtensionKind

from tai42_skeleton.exceptions.exceptions import TaiValidationError
from tai42_skeleton.interactions import ask_user

# Injected onto the composed signature — a tool param sharing one of these names
# would be silently shadowed, so a collision is rejected at wrap time. ``verifier``
# is deliberately absent: it is author-bound via extension config, never a param.
_CONTROL_PARAMS = ("question", "answer_schema", "timeout")

# The keys ``ask_external`` accepts in its author-supplied extension config.
_CONFIG_KEYS = frozenset({"verifier"})


@tai42_app.extensions.extension(kind=ExtensionKind.TRANSFORMER, name="ask_external")
def ask_external(
    func: Callable[..., Any], name: str, description: str, config: dict[str, Any] | None = None
) -> Callable[..., Any]:
    """Wrap ``func`` into an external-question tool.

    ``func`` must accept a ``callback_url`` parameter and return the external URL
    the human visits (the helper enforces the http(s)-str check). The composed
    tool presents ``func``'s params minus ``callback_url``, plus ``question`` /
    ``answer_schema`` / ``timeout``; calling it opens the external interaction and
    blocks until the callback delivers the answer.

    ``config`` is the author-supplied extension config: its optional ``verifier``
    binds a webhook verifier to the callback so the signed answer is authenticated.
    It is closed over here, never surfaced as an LLM-facing param."""
    config = config or {}
    unknown = sorted(set(config) - _CONFIG_KEYS)
    if unknown:
        raise TaiValidationError(f"ask_external config has unknown key(s): {', '.join(unknown)}")
    verifier = cast("dict[str, Any] | None", config.get("verifier"))

    sig = inspect.signature(func)
    params = sig.parameters
    if "callback_url" not in params:
        raise TaiValidationError("tool wrapped by ask_external must accept callback_url")
    for control in _CONTROL_PARAMS:
        if control in params:
            raise TaiValidationError(
                f"tool wrapped by ask_external must not define '{control}'; it collides with an injected control param"
            )
    # Every param is forwarded by keyword: the tool's own params ride the composed
    # signature (re-presented keyword-only) into ``**tool_kwargs``, and
    # ``callback_url`` is passed by name in ``build``. A positional-only param,
    # ``*args``, or ``**kwargs`` cannot survive that round-trip and would crash at
    # call time — including a positional-only ``callback_url`` — so reject them
    # loudly at wrap time (checking ``callback_url`` too, not just the tool params).
    unsupported = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    for p in params.values():
        if p.kind in unsupported:
            raise TaiValidationError(
                f"tool wrapped by ask_external cannot use parameter '{p.name}' of kind {p.kind.description}; "
                "its params must be positional-or-keyword or keyword-only"
            )

    async def wrapper(
        *,
        question: str,
        answer_schema: dict | None = None,
        timeout: float | None = None,
        **tool_kwargs,
    ):
        async def build(callback_url: str) -> str:
            # The wrapped tool may be sync or async; await only when it returns a
            # coroutine (mirroring the monitor extension).
            result = func(callback_url=callback_url, **tool_kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return await ask_user(
            question,
            answer_format="external",
            schema=answer_schema,
            timeout=timeout,
            link=build,
            verifier=verifier,
        )

    # Composed signature: original params (minus callback_url) re-presented as
    # keyword-only so makefun forwards them by name into ``**tool_kwargs``, plus
    # the three control params. ``verifier`` is NOT presented — it is author-bound
    # via config and closed over above. A concrete presented signature satisfies
    # the transformer schema rule.
    composed = [p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in params.values() if p.name != "callback_url"]
    composed.extend(
        [
            inspect.Parameter("question", inspect.Parameter.KEYWORD_ONLY, annotation=str),
            inspect.Parameter("answer_schema", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=dict | None),
            inspect.Parameter("timeout", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=float | None),
        ]
    )
    composed_sig = inspect.Signature(parameters=composed, return_annotation=Any)

    new_name = f"{name}_{ask_external.__name__}"
    return create_function(
        func_signature=composed_sig,
        # makefun's ``func_impl`` is annotated ``Callable[[Any], Any]`` but it
        # drives the separate presented signature above; our keyword-only impl is
        # valid at runtime.
        func_impl=cast("Callable[[Any], Any]", wrapper),
        func_name=new_name,
        qualname=new_name,
        module_name=func.__module__,
        doc=description,
    )
