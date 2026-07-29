import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from tai42_contract.app import tai42_app
from tai42_contract.hooks.models import HookParams
from tai42_contract.monitoring import MonitoringLevel, SpanKind
from tai42_kit.utils.data import run_jq_first
from tai42_kit.utils.data.jq_util import get_compiled_jq

from tai42_skeleton.authz.execution import bind_execution_identity
from tai42_skeleton.hooks.settings import HooksSettings
from tai42_skeleton.monitoring import get_monitoring
from tai42_skeleton.operations.errors import PermissionDenied

logger = logging.getLogger(__name__)


class BaseHooksManager(ABC):
    def __init__(self, settings: HooksSettings):
        self.settings = settings
        # One semaphore for the manager's lifetime, bounding TOTAL in-flight hook
        # executions across ALL events at ``settings.max_workers`` — concurrent
        # events share the same bound rather than each fanning out its own.
        self._run_semaphore = asyncio.Semaphore(settings.max_workers)

    @staticmethod
    def validate_jq_fields(params: HookParams) -> None:
        """Reject inline jq that does not compile, at registration time.

        A broken condition/expr would otherwise surface only as a hook that
        never fires (indistinguishable from a false condition). Template-id
        fields render per event and cannot be compiled here — their failures
        surface loudly at fire time instead."""
        for field in ("condition", "expr"):
            raw = getattr(params, field)
            if not raw:
                continue
            try:
                get_compiled_jq(raw)
            except Exception as exc:
                raise ValueError(f"hook {params.name!r}: {field} is not valid jq: {exc}") from exc

    async def _check_condition(self, hook: HookParams, payload: dict[str, Any]) -> bool:
        writer = get_monitoring().writer
        with writer.start_span(name="hook_check_condition", kind=SpanKind.CHAIN):
            raw_condition = await tai42_app.storage.resource_manager.render_by_id_or_content(
                content=hook.condition,
                template_id=hook.condition_id,
                kwargs=hook.condition_kwargs,
            )
            writer.update_current_span(metadata={"hook_name": hook.name, "raw_condition": raw_condition})

            if not raw_condition:
                return True

            try:
                result = await run_jq_first(raw_condition, payload)
            except Exception as e:
                # A genuine jq EVALUATION error at fire time must surface loudly
                # (the registration-time validator's docstring promises this), not
                # be swallowed as a skipped hook -- a skip is indistinguishable
                # from a condition that cleanly evaluated to false. A condition
                # that evaluates without error to a falsy value still skips below.
                writer.update_current_span(level=MonitoringLevel.ERROR, status_message=str(e))
                raise
            return bool(result)

    @staticmethod
    async def _run_hook(hook: HookParams, payload: dict[str, Any], tool_kwargs_override: dict[str, Any] | None = None):
        """Fire one hook's tool AS the hook's bound execution key.

        The bind must stay HERE, inside the per-hook coroutine: a contextvar set inside a
        task is invisible to its siblings, which is what gives each fanned-out hook its
        own key rather than a sibling's or the server's unbounded authority."""
        writer = get_monitoring().writer
        with writer.start_span(name="hook_run_tool", kind=SpanKind.CHAIN):
            writer.update_current_span(
                metadata={
                    "tool": hook.tool,
                    "tool_kwargs": hook.tool_kwargs,
                    "tool_kwargs_override": tool_kwargs_override,
                }
            )
            if not hook.execution_key:
                # No bound key means no authority to fire under; the server's own is not a
                # substitute. Refuse before any work.
                raise PermissionDenied(f"hook {hook.name!r} binds no execution key; refusing to fire")

            rendered_expr = await tai42_app.storage.resource_manager.render_by_id_or_content(
                content=hook.expr,
                template_id=hook.expr_id,
                kwargs=hook.expr_kwargs,
            )
            event_input = (await run_jq_first(rendered_expr, payload)) if rendered_expr else {}
            # Shallow top-level merge, strongest last: expr input, then the per-link
            # override, then the hook author's static ``tool_kwargs``. The author's pinned
            # keys must stay unoverridable — they are the hook's only lock against a link
            # minted by someone with no relation to the topic.
            tool_input = {**event_input, **(tool_kwargs_override or {}), **(hook.tool_kwargs or {})}
            async with bind_execution_identity(hook.execution_key, bound_fingerprint=hook.execution_key_fingerprint):
                await tai42_app.tools.run_tool(hook.tool, tool_input)

    async def _run_hook_with_limit(
        self, hook: HookParams, payload: dict[str, Any], tool_kwargs_override: dict[str, Any] | None = None
    ):
        async with self._run_semaphore:
            await self._run_hook(hook, payload, tool_kwargs_override)

    @abstractmethod
    async def register(self, params: HookParams) -> bool: ...

    @abstractmethod
    async def unregister(self, name: str) -> bool: ...

    @abstractmethod
    async def list_hooks(self) -> dict[str, HookParams]: ...

    @abstractmethod
    async def list_hooks_by_topic(self, topic: str) -> dict[str, HookParams]: ...

    # -- Per-topic webhook-verifier bindings ---------------------------------
    #
    # A binding is the ``{"verifier": <name>, "config": {...}}`` shape validated
    # against ``TopicVerifierBinding`` naming a registered webhook verifier and its
    # per-topic config. The config carries a ``secret_env`` (an env var NAME),
    # never a secret value. Both manager backends store the same shape behind these
    # four methods.

    @abstractmethod
    async def set_topic_verifier(self, topic: str, binding: dict[str, Any]) -> None: ...

    @abstractmethod
    async def get_topic_verifier(self, topic: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def delete_topic_verifier(self, topic: str) -> bool: ...

    @abstractmethod
    async def all_topic_verifiers(self) -> dict[str, dict[str, Any]]: ...

    async def on_event(
        self, topic: str, payload: dict[str, Any], *, tool_kwargs_override: dict[str, Any] | None = None
    ):
        """Fan the event out to every hook on ``topic``.

        ``tool_kwargs_override`` (kw-only) is merged into every fired hook's tool input
        ABOVE the rendered ``expr`` input but BELOW the hook's static ``tool_kwargs``, so
        it supplies only the keys the hook's author left unpinned.

        Each hook runs in its own task and binds its own execution identity there, so a
        hook is never fired under a sibling's key; a denied fire is that hook's error
        outcome and leaves the rest of the fan-out untouched."""
        writer = get_monitoring().writer
        with (
            writer.start_span(name="hook_on_event", kind=SpanKind.CHAIN),
            writer.trace_attributes(tags=[f"hook:{topic}"], metadata={"topic": topic}, name=f"event: {topic}"),
        ):
            hooks_map = await self.list_hooks_by_topic(topic)
            if not hooks_map:
                return

            valid_hooks = [hook for hook in hooks_map.values() if await self._check_condition(hook, payload)]
            if not valid_hooks:
                return

            # Every execution goes through the manager-wide semaphore, so this
            # event's fan-out shares the global in-flight bound with every
            # concurrently firing event.
            tasks = [self._run_hook_with_limit(h, payload, tool_kwargs_override) for h in valid_hooks]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for hook, result in zip(valid_hooks, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error(
                        "hook %r on topic %r failed: %s",
                        hook.name,
                        topic,
                        result,
                        exc_info=result,
                    )
