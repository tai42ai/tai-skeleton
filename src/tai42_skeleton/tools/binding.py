"""Tool-binding engine — the impl body behind the ``app.tools`` facet.

Owns every tool/toolkit/remote-MCP binding path: manifest gating, extension
stacking, FastMCP registration, lookup, and direct invocation. State that the
lifecycle swaps on every ``start()`` (manifest, registries, the FastMCP server)
is read through the owning app so a reload is always visible here.
"""

import asyncio
import inspect
import logging
import sys
import types
from collections import OrderedDict
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from fastmcp.server.dependencies import without_injected_parameters
from fastmcp.tools.base import Tool
from fastmcp.tools.function_parsing import ParsedFunction
from fastmcp.tools.function_tool import FunctionTool
from fastmcp.tools.tool_transform import TransformedTool
from fastmcp.utilities.types import Audio, File, Image, get_cached_typeadapter
from langchain_core.tools import StructuredTool, tool
from makefun import create_function
from pydantic_core import to_jsonable_python
from tai42_contract.extensions import ExtensionKind
from tai42_contract.manifest import ExtensionElement, TaiMCPConfig

from tai42_skeleton.agent.binding import _UNSET
from tai42_skeleton.exceptions.exceptions import TaiValidationError
from tai42_skeleton.extensions.registry import extension_config, extension_name, factory_accepts_config
from tai42_skeleton.tools.adapters.lc_tool_to_func import lc_tool_to_func
from tai42_skeleton.tools.context_bridge import bridge_context

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from tai42_skeleton.app.server import TaiMCP
    from tai42_skeleton.authz.identity import CallerIdentity
    from tai42_skeleton.extensions import ExtensionRegistry
    from tai42_skeleton.manifest import Manifest
    from tai42_skeleton.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Namespace for the synthetic modules that own remote-MCP wrapper functions, so
# a manifest title can never shadow (or be shadowed by) a real importable module
# in ``sys.modules``.
MCP_VIRTUAL_MODULE_PREFIX = "tai42_mcp_virtual."

# Model providers cap a LangChain client tool's function name at 64 characters;
# a client-facing tool name is truncated to this length, and a post-truncation
# collision is a hard error.
CLIENT_TOOL_NAME_MAX_LEN = 64

_VAR_PARAM_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


class UnknownToolError(RuntimeError):
    """Raised when a tool name is not registered on the live server.

    Carries the missing ``tool_name`` so a caller can build a typed 404/501
    without matching the message text. Subclasses ``RuntimeError`` so an existing
    ``except RuntimeError`` handler still catches it."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"No such tool: {tool_name}.")
        self.tool_name = tool_name


def is_unknown_tool_error(exc: BaseException, tool_name: str) -> bool:
    """Whether ``exc`` is the run-tool seam's unknown-tool error for ``tool_name``.

    Recognizes the typed :class:`UnknownToolError` and, defensively, a plain
    ``RuntimeError("No such tool: {name}.")`` carrying the same message, so a
    caller can tell an unknown-tool failure apart from any other error whichever
    shape the binding raises."""
    if isinstance(exc, UnknownToolError):
        return exc.tool_name == tool_name
    return isinstance(exc, RuntimeError) and str(exc) == f"No such tool: {tool_name}."


def _derive_input_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """JSON schema of ``func``'s PRESENTED signature, derived via
    ``get_cached_typeadapter`` (the adapter fastmcp builds tools on), which
    follows ``__signature__`` / ``__wrapped__``. This is the RAW
    presented-signature schema — deliberately WITHOUT fastmcp's exposure-time
    post-processing (injected Context/``Depends`` stripping, title removal):
    the wrapper identity check compares the full declared input contract. Never
    inspects the raw implementation body — extensions legitimately implement
    with ``*args/**kwargs`` behind a concrete makefun-presented signature, and
    reading the impl would wrongly reject them."""
    return get_cached_typeadapter(func).json_schema()


def _drop_empty_required(schema: dict[str, Any]) -> None:
    """Normalize an empty ``required`` list to no key at all: removing a
    default-less reserved param can empty ``required``, and a baseline with no
    required params carries no ``required`` key — they must compare equal."""
    if schema.get("required") == []:
        del schema["required"]


def _derive_output_schema(func: Callable[..., Any]) -> dict[str, Any] | None:
    """The tool OUTPUT schema FastMCP would derive from ``func``'s return type
    (``None`` when the return is unannotated). A non-object return comes back as
    a WRAPPED ``{... "x-fastmcp-wrap-result": true}`` object schema. Derived from
    the FUNCTION (not a registered ``Tool``) because branch tools are bound
    BEFORE the base, so the base is not yet a registered tool at propagation
    time.

    A body presenting a bare ``**kwargs`` (e.g. a synthesized agent run tool,
    whose typed contract lives on its explicit ``parameters`` rather than its
    signature) has no schema FastMCP can parse from the function, so there is no
    output schema to derive or propagate — return ``None``."""
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(func).parameters.values()):
        return None
    return ParsedFunction.from_function(func).output_schema


def _declares_own_output_schema(func: Callable[..., Any]) -> bool:
    """Whether ``func`` genuinely declares its OWN object output schema — a real
    object return, not the auto-wrapped placeholder for a non-object/absent
    return. A branch that declares its own KEEPS it; only a branch that declares
    none may inherit the base tool's output schema."""
    schema = _derive_output_schema(func)
    return schema is not None and not schema.get("x-fastmcp-wrap-result")


def _baked_partial(tool_obj: TransformedTool) -> Callable[..., Any]:
    """A typed partial of a transformed tool's underlying function with its hidden
    baked args applied — the callable an extension branch wraps.

    The remaining parameters keep the underlying function's real typed signature;
    a call that passes a baked key is rejected (the branch may not re-open a baked
    constant), matching the bound tool's own contract. A preset only ever bakes
    ``ArgTransform(hide=True, default=<value>)``, so a non-hidden transform arg has
    no branch-composable meaning and raises loudly rather than mis-binding."""
    parent = tool_obj.parent_tool
    if not isinstance(parent, FunctionTool):
        raise TypeError(f"transformed tool {tool_obj.name!r} has no callable base to branch")
    base_fn = parent.fn

    baked: dict[str, Any] = {}
    for key, transform in tool_obj.transform_args.items():
        if transform.hide is not True:
            raise TypeError(
                f"cannot branch-bind transform arg {key!r} of {tool_obj.name!r}: only hidden baked args are supported"
            )
        baked[key] = transform.default

    signature = inspect.signature(base_fn)
    remaining = [p for p in signature.parameters.values() if p.name not in baked]
    presented = signature.replace(parameters=remaining)

    def _reject_baked(kwargs: dict[str, Any]) -> None:
        clashing = [key for key in baked if key in kwargs]
        if clashing:
            raise TypeError(f"{tool_obj.name!r} does not accept baked argument(s): {', '.join(sorted(clashing))}")

    def _forward_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        # Bind positionals through the PRESENTED signature (which excludes the
        # baked params), so a positional call maps each value to the remaining
        # param it actually names — never onto a hidden baked slot. The baked
        # constants are then merged in by name.
        _reject_baked(kwargs)
        bound = presented.bind(*args, **kwargs)
        return {**baked, **bound.arguments}

    if inspect.iscoroutinefunction(base_fn):

        async def _impl_async(*args, **kwargs):
            return await base_fn(**_forward_args(args, kwargs))

        impl: Callable[..., Any] = _impl_async
    else:

        def _impl_sync(*args, **kwargs):
            return base_fn(**_forward_args(args, kwargs))

        impl = _impl_sync

    # makefun's ``doc`` is annotated ``str`` but accepts ``None`` (its own default),
    # falling back to the impl's docstring.
    return create_function(
        presented, cast(Callable[[Any], Any], impl), func_name=tool_obj.name, doc=cast(str, base_fn.__doc__)
    )


# A preset's makefun-compiled partial per resolved tool object, so a repeated live resolve
# yields the SAME callable and fastmcp's identity-keyed ``without_injected_parameters`` LRU
# still hits. Keyed on ``id`` (a fastmcp ``Tool`` is unhashable): safe only because the cached
# partial strongly references its source tool, pinning that id for the entry's life.
_BAKED_PARTIAL_CACHE_MAX = 2048
_baked_partial_cache: "OrderedDict[int, Callable[..., Any]]" = OrderedDict()


def _cached_baked_partial(tool_obj: TransformedTool) -> Callable[..., Any]:
    key = id(tool_obj)
    cached = _baked_partial_cache.get(key)
    if cached is not None:
        _baked_partial_cache.move_to_end(key)
        return cached
    partial = _baked_partial(tool_obj)
    _baked_partial_cache[key] = partial
    if len(_baked_partial_cache) > _BAKED_PARTIAL_CACHE_MAX:
        _baked_partial_cache.popitem(last=False)
    return partial


def _tool_result_value(result: Any) -> Any:
    """Reduce a ``ToolResult`` (a transformed tool's ``run`` output) to the same
    raw, JSON-able value the callable run path returns.

    ``structured_content`` is the structured form of the tool's return; for a
    non-object return FastMCP WRAPS it as ``{"result": <value>}`` and flags the
    wrap on ``_meta.fastmcp.wrap_result`` — unwrap that so a scalar/string preset
    returns its bare value, exactly as a direct call of the base tool would. With
    no structured content, fall back to the text blocks; a media return
    (Image/Audio/File) carries no structured and no text, so serialize its
    remaining content blocks to their JSON wire dicts — the same media shape the
    direct-run path preserves. Only a genuinely empty result reduces to ``None``."""
    structured = result.structured_content
    meta = result.meta or {}
    if isinstance(structured, dict) and meta.get("fastmcp", {}).get("wrap_result"):
        return to_jsonable_python(structured["result"])
    if structured is not None:
        return to_jsonable_python(structured)
    texts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
    if texts:
        return texts[0] if len(texts) == 1 else texts
    non_text = [to_jsonable_python(block) for block in result.content if getattr(block, "type", None) != "text"]
    if not non_text:
        return None
    return non_text[0] if len(non_text) == 1 else non_text


def _serialize_result(result: Any) -> Any:
    """Reduce a direct tool-run return to a JSON-native value.

    A live fastmcp media object (``Image`` / ``Audio`` / ``File``) is NOT
    serializable by ``to_jsonable_python`` (it raises) — fastmcp's own
    media-to-MCP-content conversion lives only in ``Tool.run``, which the direct
    run path bypasses. Convert it to its MCP content first: ``Image`` / ``Audio``
    become the media wire dict ``{"type": "image"|"audio", "data": <b64>,
    "mimeType": <mime>}``; a ``File`` becomes an ``EmbeddedResource``
    (``{"type": "resource", ...}``) — JSON-native, but NOT the media wire shape,
    so the direct-run UI renders it as JSON rather than as media. Any other
    result serializes directly via ``to_jsonable_python``."""
    if isinstance(result, Image):
        result = result.to_image_content()
    elif isinstance(result, Audio):
        result = result.to_audio_content()
    elif isinstance(result, File):
        result = result.to_resource_content()
    return to_jsonable_python(result)


def _named_call_arguments(
    signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """A client-tool call's arguments keyed by PARAMETER NAME — the shape the
    execution-identity decision reads them in.

    Positionals are bound through ``signature``, a ``**kwargs`` catch-all is flattened
    back in, ``*args`` is dropped, and the :data:`_UNSET` sentinel is stripped — so the
    decision sees exactly the set-fields-only argument set ``run_tool`` authorizes."""
    bound = signature.bind_partial(*args, **kwargs)
    arguments = dict(bound.arguments)
    for name, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            arguments.update(arguments.pop(name, {}))
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            arguments.pop(name, None)
    return {name: value for name, value in arguments.items() if value is not _UNSET}


@lru_cache(maxsize=2048)
def _validation_wrapper(resolved_fn: Callable[..., Any], offload: bool) -> Callable[..., Any]:
    """A stable makefun wrapper presenting ``resolved_fn``'s signature, whose body
    resolves and invokes ``resolved_fn`` (offloading a sync call to a worker thread
    when ``offload`` is set).

    Keyed on ``(resolved_fn, offload)`` and cached module-wide. ``resolved_fn`` is
    fastmcp's process-cached ``without_injected_parameters`` wrapper — a stable
    object per tool — so ``run_tool`` reuses one wrapper across calls instead of
    compiling a fresh function each time. That keeps fastmcp's process-global
    ``get_cached_typeadapter`` LRU hitting on the same wrapper rather than
    thrashing it with a per-call throwaway."""

    async def safe_impl(**kwargs):
        result = await asyncio.to_thread(resolved_fn, **kwargs) if offload else resolved_fn(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    return create_function(
        inspect.signature(resolved_fn),
        # makefun's ``func_impl`` is annotated ``Callable[[Any], Any]`` but it
        # accepts any callable (it drives the separate signature above); our
        # **kwargs impl is valid at runtime.
        cast(Callable[[Any], Any], safe_impl),
        func_name=resolved_fn.__name__,
        # makefun's ``doc`` is annotated ``str`` but accepts ``None`` (its own
        # default), falling back to the impl's docstring.
        doc=cast(str, resolved_fn.__doc__),
    )


class ToolBinding:
    """Binds tools onto the app's live FastMCP server.

    Holds no lifecycle state of its own — manifest/registries/server are
    properties over the owning app, which rebuilds them on every start/reload.
    """

    def __init__(self, app: "TaiMCP") -> None:
        self._app = app

    # -- live app state (swapped by the lifecycle on every start) -------------

    @property
    def _fast_mcp(self) -> "FastMCP":
        return self._app._fast_mcp

    @property
    def _manifest(self) -> "Manifest | None":
        return self._app._manifest

    @property
    def _tool_registry(self) -> "ToolRegistry":
        return self._app._tool_registry

    @property
    def _extension_registry(self) -> "ExtensionRegistry":
        return self._app._extension_registry

    @property
    def _mcp_bound_tools(self) -> dict[str, set[str]]:
        return self._app._mcp_bound_tools

    def _require_manifest(self) -> "Manifest":
        manifest = self._manifest
        if manifest is None:
            raise RuntimeError("TaiMCP is not started — call start()/app_context first.")
        return manifest

    # -- execution-identity seam ----------------------------------------------

    @staticmethod
    def _bound_execution_identity() -> "CallerIdentity | None":
        """The execution identity bound to the current fire, or ``None`` outside one.

        Imported inside the function: ``authz`` reaches back into this module."""
        from tai42_skeleton.authz.execution_identity import get_execution_identity

        return get_execution_identity()

    async def _authorize_execution_dispatch(
        self, identity: "CallerIdentity", tool_name: str, call_arguments: dict[str, Any]
    ) -> None:
        """Authorize one tool dispatch against the bound execution ``identity``; a denial
        raises ``PermissionDenied`` out of the dispatch.

        Call ONLY with a non-``None`` identity. The registries handed to the decision are
        the live ones, so a reload is reflected immediately; an unsettled surface is
        retried once behind the reload gate, since a background fire has no client to
        retry it and would otherwise be lost."""
        from tai42_skeleton.app.reload_gate import reload_gate
        from tai42_skeleton.authz.execution import authorize_execution_tool_call
        from tai42_skeleton.authz.resolver import OperationSurfaceUnsettledError

        async def _authorize() -> None:
            await authorize_execution_tool_call(
                identity,
                tool_name,
                call_arguments,
                tool_registry=self._tool_registry,
                preset_manager=self._app.preset_manager,
            )

        try:
            await _authorize()
        except OperationSurfaceUnsettledError:
            async with reload_gate.lock:
                pass
            await _authorize()

    # -- lookup / invocation ---------------------------------------------------

    async def get_tools(self) -> dict[str, Tool]:
        return {t.name: t for t in await self._fast_mcp.list_tools()}

    async def get_tool(self, key: str) -> Tool:
        mcp_tool = await self._fast_mcp.get_tool(key)
        if mcp_tool is None:
            # FastMCP returns None for an unregistered name; the contract promises
            # a Tool, so fail loud with the name rather than leaking None.
            raise UnknownToolError(key)
        return mcp_tool

    async def _resolve_run_target(self, key: str) -> Tool:
        """Resolve ``key`` to its bound tool for a RUN, waiting out an in-flight reload.

        A reload rebuilds the tool registry non-atomically (the FastMCP server is torn
        down and re-registered under the reload gate), so a run that resolves mid-rebuild
        can miss a tool that is bound both before and after — e.g. a backend worker job
        dispatched while the boot self-resync reload is still rebuilding. This is the
        run-surface half of the retriable ``reloading`` contract the HTTP/MCP edges
        answer with a 503/``ToolError``: on a miss, wait for any reload holding the gate
        to release, then re-resolve once. A genuinely unknown tool (no reload in flight)
        still fails fast — the gate is uncontended and the second lookup raises again."""
        from tai42_skeleton.app.reload_gate import reload_gate

        try:
            return await self.get_tool(key)
        except UnknownToolError:
            async with reload_gate.lock:
                pass
            return await self.get_tool(key)

    def remove_tool(self, name: str) -> None:
        return self._fast_mcp.local_provider.remove_tool(name)

    def tool_title(self, func) -> str:
        manifest = self._require_manifest()
        module = func.__module__
        if module.startswith(MCP_VIRTUAL_MODULE_PREFIX):
            # Remote-MCP wrappers live in synthetic modules; their title is the
            # manifest title the namespace was minted from.
            return module[len(MCP_VIRTUAL_MODULE_PREFIX) :]
        return manifest.find_title(module)

    async def run_tool(self, key: str, arguments: dict[str, Any], *, offload_sync: bool = False) -> Any:
        """Validate ``arguments`` against ``key``'s signature and invoke it.

        With ``offload_sync`` set AND the resolved tool function being a plain
        (non-coroutine) callable, the sync call runs on a worker thread via
        ``asyncio.to_thread`` instead of inline on the event loop — a blocking
        sync tool then cannot starve the loop (or a co-running supervisor's
        liveness refresh). ``asyncio.to_thread`` copies the current contextvars
        into the thread, so ``without_injected_parameters``' ctx/``Depends``
        resolution still sees the active context. Async tools and the default
        (``offload_sync=False``) path run inline on the event loop.

        A dispatch under a bound execution identity is authorized against it first, on the
        arguments actually about to be fired (:meth:`_authorize_execution_dispatch`)."""
        # An agent run tool's re-dispatch (e.g. a chain TRANSFORMER re-invoking it by
        # name) materializes the _UNSET sentinel for optionals the caller never
        # supplied; strip it here so the set-fields-only contract holds and the tool's
        # own defaults apply instead of a sentinel failing validation. No external
        # caller can produce _UNSET, so this is a no-op for ordinary arguments.
        arguments = {name: value for name, value in arguments.items() if value is not _UNSET}
        # Execution-identity seam: decided at INVOCATION, before the tool is resolved, on
        # the exact arguments this call fires. With no identity bound it is a contextvar
        # read and the path below is untouched.
        execution_identity = self._bound_execution_identity()
        if execution_identity is not None:
            await self._authorize_execution_dispatch(execution_identity, key, arguments)
        mcp_tool = await self._resolve_run_target(key)
        if isinstance(mcp_tool, TransformedTool):
            # A prebuilt transformed tool (e.g. a preset) has no plain callable to
            # validate against — its ``fn`` takes one opaque ``**kwargs`` and its
            # typed contract lives on the object. Run it through the tool's OWN
            # schema-validating ``run``, which applies the baked hidden constants
            # and REJECTS a caller that passes a baked key. Bridge the in-process
            # Context the same way the callable path does.
            with bridge_context(self._app.fastmcp):
                result = await mcp_tool.run(arguments)
            return _tool_result_value(result)
        if not isinstance(mcp_tool, FunctionTool):
            raise RuntimeError(f"Tool {key!r} has no callable body and cannot be run directly.")

        # ``without_injected_parameters`` strips the fastmcp Context / ``Depends``
        # params from the signature AND, when called, resolves and injects them
        # before invoking ``fn``. Validation runs against that stripped signature,
        # and invocation goes through this same wrapper — calling the raw ``fn``
        # would bypass injection (a ctx tool raises TypeError; a Depends tool
        # receives the unresolved sentinel object).
        fn = mcp_tool.fn
        resolved_fn = without_injected_parameters(fn)
        # A coroutine tool never blocks the loop, so it is never offloaded; only a
        # sync callable is, and only when the caller opted in. ``iscoroutinefunction``
        # tracks the wrapped async-ness through ``without_injected_parameters``.
        offload = offload_sync and not inspect.iscoroutinefunction(resolved_fn)

        # A stable per-(resolved_fn, offload) wrapper: fastmcp caches
        # ``resolved_fn``, so this reuses one makefun wrapper across calls and
        # ``get_cached_typeadapter`` below hits its process-global LRU instead of
        # thrashing it with a per-call throwaway.
        safe_wrapper = _validation_wrapper(resolved_fn, offload)

        async def run_validated_adapter(args):
            type_adapter = get_cached_typeadapter(safe_wrapper)
            result = type_adapter.validate_python(args)
            if inspect.isawaitable(result):
                return await result
            return result

        # In-process Context bridge: this invocation has no connected client, so
        # an injected ``ctx.elicit()`` routes to the interactions ``ask_user``
        # waiter and ``ctx.sample()`` falls back to the platform LLM. The bridge
        # no-ops when a live request context is already active, so a capable
        # client still resolves in-client. ``asyncio.to_thread`` copies
        # contextvars into the offload thread, so the pushed context reaches the
        # sync path too.
        with bridge_context(self._app.fastmcp):
            result = await run_validated_adapter(arguments)

        return _serialize_result(result)

    async def get_client_tools(self, names: list[str] | None = None) -> list[StructuredTool]:
        tools = await self.get_tools()
        for name in names or []:
            if name not in tools:
                raise UnknownToolError(name)

        selected = {name: t for name, t in tools.items() if not names or name in names}
        truncated: dict[str, str] = {}
        for name in selected:
            key = name[:CLIENT_TOOL_NAME_MAX_LEN]
            if key in truncated:
                raise ValueError(
                    f"Tool names {truncated[key]!r} and {name!r} collide after "
                    f"{CLIENT_TOOL_NAME_MAX_LEN}-char truncation."
                )
            truncated[key] = name

        client_tools: list[StructuredTool] = []
        for name, t in selected.items():
            # An opaque-signature tool (an agent run tool's ``**arguments`` body)
            # cannot have its input schema inferred from the signature — that
            # advertises NO fields — so pass its explicit ``.parameters`` schema and
            # its own description (langchain does not fall back to the impl's
            # docstring once a dict ``args_schema`` is given, and the ``**arguments``
            # body carries none). A normal tool gets neither: langchain infers the
            # schema from the presented signature and reads the description from the
            # runnable's docstring.
            explicit_schema = self._client_args_schema(t)
            extra_kwargs: dict[str, Any] = (
                {"args_schema": explicit_schema, "description": t.description} if explicit_schema is not None else {}
            )
            # langchain's overloaded ``tool`` is typed to return ``BaseTool | Callable``
            # and to want a ``Runnable`` for ``runnable``; this call shape (name + a
            # plain callable) always builds a ``StructuredTool`` at runtime.
            client_tools.append(
                cast(
                    StructuredTool,
                    tool(
                        name_or_callable=name[:CLIENT_TOOL_NAME_MAX_LEN],
                        runnable=cast(Any, self._client_runnable(t)),
                        **extra_kwargs,
                    ),
                )
            )
        return client_tools

    def _client_args_schema(self, tool_obj: Tool) -> dict[str, Any] | None:
        """The explicit input JSON schema a client tool must advertise when the
        tool's fn signature cannot be round-tripped through langchain's inferred
        args model — otherwise ``None`` (langchain infers the schema from the
        presented signature).

        Two signature shapes need the explicit ``.parameters`` instead of
        inference:

        * a bare ``*args``/``**kwargs`` passthrough — inference advertises NO
          fields, hiding every input from the LLM;
        * a synthesized agent run tool, whose concrete per-field signature carries
          the non-JSON-serializable :data:`_UNSET` sentinel as the default of every
          optional parameter (so the body can forward set fields only). Building a
          langchain args model over that signature would materialize and then fail
          to serialize the sentinel, so the tool advertises its explicit
          ``.parameters`` (the agent's exact ``ToolInput`` schema) and is invoked
          through a permissive runnable (:meth:`_client_runnable`).

        A tool with an ordinary concrete signature returns ``None`` and keeps the
        signature-inference path (injected Context/``Depends`` stripping included)
        unchanged."""
        if not isinstance(tool_obj, FunctionTool):
            return None
        resolved = without_injected_parameters(tool_obj.fn)
        params = list(inspect.signature(resolved).parameters.values())
        all_var = bool(params) and all(p.kind in _VAR_PARAM_KINDS for p in params)
        has_unset_default = any(p.default is _UNSET for p in params)
        if all_var or has_unset_default:
            return tool_obj.parameters
        return None

    def _client_runnable(self, tool_obj: Tool) -> Callable[..., Any]:
        """The callable langchain builds a client tool over, for an in-process
        agent to invoke.

        Resolves the tool's typed callable via ``_branch_base_callable``
        (``FunctionTool`` → ``fn``; a preset ``TransformedTool`` → its baked
        partial; anything else raises), then strips the fastmcp-injected
        Context/``Depends`` params with ``without_injected_parameters`` so the
        schema langchain infers advertises only the real user args — never a
        ``Context`` the LLM would be asked to supply. The returned closure carries
        that stripped ``__signature__``/``__name__`` (langchain infers the args
        schema via ``inspect.signature``, which honors ``__signature__``) and runs
        the call under ``bridge_context`` so an injected ``ctx.elicit()`` /
        ``ctx.sample()`` resolves through the platform bridges, exactly as
        ``run_tool`` does.

        When the tool advertises an explicit ``args_schema`` (:meth:`_client_args_schema`
        returns a schema — an all-VAR passthrough or a synthesized agent run tool
        with :data:`_UNSET` sentinel defaults), the runnable is left PERMISSIVE (its
        native ``*args``/``**kwargs`` signature): the LLM-facing schema rides on the
        explicit ``args_schema``, and forwarding only the caller-supplied kwargs
        keeps ``from_tool_input``'s set-fields-only contract intact — never
        materializing (and failing to serialize) the sentinel defaults.

        The closure gates on the execution identity exactly as ``run_tool`` does, under
        the tool's FULL registered name (the client-facing truncation is only a label).
        With no identity bound the call forwards straight through to the snapshot's
        callable.

        Under a fire the body is re-resolved from the name LIVE, so the registration
        decided about is the one that runs — a client-tool snapshot outlives an agent
        turn, and a preset re-based or deleted mid-turn would otherwise run its stale
        baked body; a vanished registration fails loudly as an unknown tool."""
        base_callable = self._branch_base_callable(tool_obj)
        resolved = without_injected_parameters(base_callable)
        resolved_sig = inspect.signature(resolved)

        async def runnable(*args, **kwargs):
            target, target_sig = resolved, resolved_sig
            execution_identity = self._bound_execution_identity()
            if execution_identity is not None:
                target = without_injected_parameters(
                    self._branch_base_callable(await self._resolve_run_target(tool_obj.name))
                )
                target_sig = inspect.signature(target)
                await self._authorize_execution_dispatch(
                    execution_identity, tool_obj.name, _named_call_arguments(target_sig, args, kwargs)
                )
            with bridge_context(self._app.fastmcp):
                result = target(*args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

        runnable.__name__ = resolved.__name__
        # langchain reads the runnable's docstring for the client tool's
        # description (it raises without one), so carry the resolved callable's.
        runnable.__doc__ = resolved.__doc__

        if self._client_args_schema(tool_obj) is not None:
            # Explicit args_schema advertised: keep the permissive signature so
            # langchain forwards only the supplied kwargs to the concrete body.
            return runnable

        runnable.__signature__ = resolved_sig  # type: ignore[attr-defined]
        # langchain infers the args schema via pydantic, which reads BOTH the
        # signature AND ``get_type_hints`` (i.e. ``__annotations__``). Carry the
        # presented signature's annotations so every advertised param has a type;
        # a bare ``**kwargs`` closure otherwise raises a ``KeyError`` at inference.
        annotations = {
            pname: p.annotation
            for pname, p in resolved_sig.parameters.items()
            if p.annotation is not inspect.Parameter.empty
        }
        if resolved_sig.return_annotation is not inspect.Signature.empty:
            annotations["return"] = resolved_sig.return_annotation
        runnable.__annotations__ = annotations
        return runnable

    # -- registration ----------------------------------------------------------

    def tool(self, *args, force=False, **kwargs) -> Any:
        func_to_register = None
        decorator_args = args

        # A prebuilt FastMCP ``Tool`` object (e.g. a preset's baked transform) is
        # registered DIRECTLY, preserving its typed schema — it is not callable, so
        # detect it explicitly alongside the callable/decorator forms.
        if args and (callable(args[0]) or isinstance(args[0], Tool)):
            func_to_register = args[0]
            decorator_args = args[1:]

        def decorator(func):
            if not self._manifest:
                return func

            name = kwargs.get("name") or (func.name if isinstance(func, Tool) else func.__name__)
            module = "" if isinstance(func, Tool) else func.__module__
            bind_func = self.bind_tool_func(*decorator_args, **kwargs)
            return bind_func(func) if force or self._manifest.should_include_tool(name, module) else func

        if func_to_register is not None:
            return decorator(func_to_register)

        return decorator

    def toolkit(self, *args, **kwargs):
        func_to_register = None
        decorator_args = args

        if args and callable(args[0]):
            func_to_register = args[0]
            decorator_args = args[1:]

        def decorator(func):
            if not self._manifest:
                return func

            for t in func().get_tools():
                toolkit_name = kwargs.get("name", func.__name__)
                name = self.normalized_name(toolkit_name, t.name)

                if not self._manifest.should_include_tool(name=name, module=func.__module__):
                    continue

                adapted_tool = lc_tool_to_func(t, name=name, module=func.__module__)

                tool_kwargs = kwargs.copy()
                tool_kwargs["name"] = name
                self.bind_tool_func(*decorator_args, **tool_kwargs)(adapted_tool)

            return func

        if func_to_register:
            return decorator(func_to_register)

        return decorator

    def mcp_tools(self, config: TaiMCPConfig, tools) -> None:
        manifest = self._require_manifest()
        virtual_module = MCP_VIRTUAL_MODULE_PREFIX + config.title
        if virtual_module not in sys.modules:
            sys.modules[virtual_module] = types.ModuleType(virtual_module)

        # Reset so a reload cleanly re-tracks what this MCP binds now.
        self._mcp_bound_tools[config.title] = set()
        self._app._mcp_preset_conflicts[config.title] = set()

        # Resolve the schema-depth bound ONCE, up front and outside the per-tool skip
        # guard below, so a malformed TAI_MCP_SCHEMA_MAX_DEPTH surfaces loudly as a
        # config error instead of being caught per tool and mis-logged as every tool
        # advertising an unusable schema.
        from tai42_skeleton.settings.mcp_settings import mcp_dispatch_settings

        schema_max_depth = mcp_dispatch_settings().schema_max_depth

        for t in tools:
            name = self.normalized_name(config.title, t.name)
            if not manifest.should_include_mcp_tool(name=name, title=config.title):
                continue

            if self._app.preset_manager.is_registered(name):
                # A registered preset already owns this name (the sanctioned
                # create-over-a-requested-unbound-MCP-name flow). Binding the
                # returning server's tool would clobber it — a hard error under
                # ``on_duplicate="error"`` — so skip it and record the conflict for
                # the reload result to surface.
                self._app._mcp_preset_conflicts[config.title].add(name)
                logger.warning("mcp %r tool %r skipped: a registered preset owns the name", config.title, name)
                continue

            from tai42_skeleton.tools.adapters.mcp_tool_to_func import mcp_tool_to_func

            try:
                adapted_tool = mcp_tool_to_func(
                    config=config, tool=t, name=name, module=virtual_module, schema_max_depth=schema_max_depth
                )
            except Exception:
                # A single tool advertising an unusable schema (empty anyOf/oneOf,
                # over-depth nesting) is SKIPPED with a loud log, never allowed to
                # take down the whole binding pass / startup — every other tool this
                # server advertises still binds. ``bind_tool_func`` stays OUTSIDE
                # this guard so a genuine registration bug still raises loudly.
                logger.error(
                    "MCP server %r advertised tool %r with an unusable schema; "
                    "skipping this tool (every other tool still binds)",
                    config.title,
                    t.name,
                    exc_info=True,
                )
                continue
            self.bind_tool_func(owner=config.title)(adapted_tool)

    def register_tool_info(self, name: str, combos: Sequence[Sequence[ExtensionElement]] | None = None) -> None:
        self._tool_registry.register_tool(name, combos)

    def unregister_tool_info(self, name: str) -> None:
        self._tool_registry.unregister_tool(name)

    def unregister_tool_base(self, tool_name: str) -> list[str]:
        return self._tool_registry.unregister_tool_base(tool_name)

    def base_of(self, name: str) -> str:
        return self._tool_registry.base_of(name)

    def is_branch(self, name: str) -> bool:
        return self._tool_registry.is_branch(name)

    def mcp_bound_names(self, title: str) -> frozenset[str]:
        """The tool names currently bound by the MCP server ``title`` — a read-only
        snapshot of the per-title bound-tool map (empty for an unknown title)."""
        return frozenset(self._mcp_bound_tools.get(title, set()))

    def available_extensions(self) -> list[dict[str, str]]:
        return self._extension_registry.available_extensions()

    def _enforce_extension_schema(
        self,
        extension: str,
        kind: ExtensionKind,
        extension_func: Callable[..., Any],
        prev_func: Callable[..., Any],
        curr_func: Callable[..., Any],
        tool: str,
    ) -> None:
        """Enforce the branch's input-schema rule by kind PROPERTY, never by
        member identity. WRAPPER (``preserves_schema``) must present the layer's
        input schema unchanged; TRANSFORMER (``declares_schema``) must present
        its own concrete schema; BACKEND (neither) has no schema rule — its
        single-strategy cardinality is enforced by ``ExtensionRegistry.validate``,
        and there is no in-place path to guard (every kind branches)."""
        if kind.preserves_schema:
            self._enforce_wrapper_schema(extension, extension_func, prev_func, curr_func, tool)
        elif kind.declares_schema:
            self._enforce_transformer_schema(extension, curr_func, tool)

    def _enforce_wrapper_schema(
        self,
        extension: str,
        extension_func: Callable[..., Any],
        prev_func: Callable[..., Any],
        curr_func: Callable[..., Any],
        tool: str,
    ) -> None:
        baseline = _derive_input_schema(prev_func)
        branch = _derive_input_schema(curr_func)

        # A wrapper may add control kwargs for itself (e.g. cache's ``exp``),
        # declared as a ``reserved_params`` attribute on the factory (it survives
        # registration — the registry decorator returns the factory unchanged).
        reserved = getattr(extension_func, "reserved_params", frozenset())
        baseline_props = baseline.get("properties", {})
        branch_props = branch.get("properties", {})
        for name in reserved:
            if name in baseline_props:
                raise TaiValidationError(
                    f"wrapper tool extension '{extension}' declares reserved param '{name}' that already exists "
                    f"on the input schema fed to this layer of tool '{tool}'; excluding it would mask a real change"
                )
            # Subtract from BOTH properties and required — a reserved param with
            # no default lands in ``required`` too, and dropping it only from
            # ``properties`` would leave a dangling ``required`` entry.
            branch_props.pop(name, None)
            if "required" in branch:
                branch["required"] = [p for p in branch["required"] if p != name]

        _drop_empty_required(baseline)
        _drop_empty_required(branch)

        if branch != baseline:
            raise TaiValidationError(
                f"wrapper tool extension '{extension}' changed the schema of tool '{tool}'; "
                "wrapper-kind extensions must preserve the input schema exactly"
            )

    def _enforce_transformer_schema(self, extension: str, curr_func: Callable[..., Any], tool: str) -> None:
        params = list(inspect.signature(curr_func).parameters.values())
        # A concrete makefun signature (batch/chain/ask_external) has named
        # params; a bare passthrough presents only ``*args``/``**kwargs``. A
        # zero-arg ``()`` signature is concrete (no VAR params), so it passes.
        if params and all(p.kind in _VAR_PARAM_KINDS for p in params):
            raise TaiValidationError(
                f"transformer tool extension '{extension}' presents a bare (*args, **kwargs) signature for "
                f"tool '{tool}'; transformer-kind extensions must present their own concrete input schema"
            )

    def bind_tool_func(self, *args, owner: str | None = None, **kwargs):
        def bind(func):
            # ``func`` is either a callable (the ordinary decorator path) or a
            # prebuilt ``Tool`` object (a preset's baked transform). A Tool object
            # is registered DIRECTLY for its bare name (preserving its typed
            # schema), and its BRANCHES wrap a reconstructed typed callable — a
            # transformed tool exposes no plain callable of its own.
            is_obj = isinstance(func, Tool)
            branch_base = self._branch_base_callable(func) if is_obj else func
            # The docstring an extension inherits from the layer it wraps: the
            # branch base callable's, NOT the ``Tool`` object's (a Tool object's
            # ``__doc__`` is the fastmcp CLASS docstring, not the tool
            # description). A wrapper that leaves the docstring at this inherited
            # value has not authored its own, so the running description survives.
            base_doc = branch_base.__doc__ if is_obj else func.__doc__

            orig_name = kwargs.pop("name", None) or (func.name if is_obj else func.__name__)
            # Pop so ``description`` reaches ``bind_tool`` exactly once (as the
            # carried value), never a second time through ``**kwargs``.
            orig_desc = kwargs.pop("description", None) or (func.description if is_obj else inspect.getdoc(func))

            # The base tool's declared OUTPUT schema, derived from the branch base
            # callable (the base is bound AFTER its branches, so it is not yet a
            # registered Tool to read from). A shape-preserving branch that
            # declares no output schema of its own inherits this, so the
            # structured-output contract survives the wrap.
            base_output_schema = _derive_output_schema(branch_base)

            # curr_name -> (func/tool, description, stack_preserves_output_shape).
            # A branch preserves the base's output shape only when EVERY extension
            # in its stack does (one TRANSFORMER anywhere reshapes the result).
            extend_tools: dict[str, tuple[Callable[..., Any] | Tool, str | None, bool]] = {}
            for extensions in self._tool_registry.tool_extensions_iterator(orig_name):
                self._extension_registry.validate(extensions)

                curr_func, curr_name, curr_desc = branch_base, orig_name, orig_desc
                stack_preserves_output = True
                # The relocating extension already applied in this stack, if any:
                # extensions apply left-to-right, so a later element wraps (sits
                # OUTSIDE) everything applied before it, and a relocating layer
                # ships exactly the callable it received to the worker.
                relocating_name: str | None = None
                for extension in extensions:
                    # A combo element is an extension name or a ``{"name",
                    # "config"}`` mapping binding author config. The registry keys
                    # on the name; the config is threaded to the factory so an
                    # extension closes over author-bound values (e.g.
                    # ``ask_external``'s verifier) never exposed as a tool param.
                    ext_name = extension_name(extension)
                    ext_config = extension_config(extension)
                    extension_func = self._extension_registry.get_extension(extension)
                    kind = self._extension_registry.get_kind(extension)
                    # A locality-requiring extension's wrapper only works in the
                    # process running the tool body. Applied AFTER a relocating
                    # extension it would wrap the worker-submitting stub in this
                    # process — the wrapper stays behind and silently never
                    # applies — so the combo is rejected loudly at bind time.
                    if relocating_name is not None and self._extension_registry.requires_body_locality(extension):
                        raise TaiValidationError(
                            f"tool '{orig_name}': extension '{ext_name}' requires body locality but is "
                            f"stacked outside the execution-relocating extension '{relocating_name}'; "
                            f"a locality-requiring extension must bind INSIDE the relocating one — "
                            f"place '{ext_name}' before '{relocating_name}' in the combo so its wrapper "
                            f"travels with the tool body to the worker"
                        )
                    if kind.relocates_execution:
                        relocating_name = ext_name
                    # Capture the layer's INPUT function: wrapper schema
                    # enforcement compares against what this extension received,
                    # not the original tool, so a transformer->wrapper stack is
                    # judged against the transformer's composed schema.
                    prev_func = curr_func
                    # Author config is threaded (by keyword) only to a factory that
                    # declares the config parameter; a config-agnostic factory keeps
                    # its three-argument signature. Binding config to a factory that
                    # does not accept it would silently drop the author's intent, so it
                    # raises instead.
                    if factory_accepts_config(extension_func):
                        curr_func = extension_func(curr_func, curr_name, curr_desc, config=ext_config)
                    else:
                        if ext_config:
                            raise ValueError(f"extension '{ext_name}' does not accept config")
                        curr_func = extension_func(curr_func, curr_name, curr_desc)

                    if curr_func.__name__ == orig_name:
                        raise ValueError(
                            f"Extension '{ext_name}' returned the same name '{orig_name}' as the original tool. "
                            "Extensions must return a new name to create a branch."
                        )

                    self._enforce_extension_schema(ext_name, kind, extension_func, prev_func, curr_func, orig_name)

                    stack_preserves_output = stack_preserves_output and kind.preserves_output_shape
                    curr_name = curr_func.__name__
                    # Carry the running description forward so each stacked
                    # extension composes on the previous one's output, not the
                    # original. Adopt the extension's docstring only when it set a
                    # NEW non-None one; a wrapper that left the docstring unchanged
                    # — or has none at all (``__doc__`` is ``None``, e.g. no
                    # ``functools.wraps``) — keeps the running description rather
                    # than dropping it.
                    if curr_func.__doc__ is not None and curr_func.__doc__ != base_doc:
                        curr_desc = curr_func.__doc__
                    extend_tools[curr_name] = (curr_func, curr_desc, stack_preserves_output)

            if orig_name not in extend_tools:
                # The base tool auto-derives its own output schema on
                # registration, so it never needs propagation (preserves=False).
                extend_tools[orig_name] = (func, orig_desc, False)

            for curr_name, (tool_func, tool_desc, preserves_output) in extend_tools.items():
                bind_kwargs = dict(kwargs)
                # Shape-aware output-schema propagation: carry the base's
                # output schema onto a branch ONLY when the branch preserves the
                # output shape (WRAPPER/BACKEND stack) AND declares none of its
                # own AND the caller did not pin one for every branch. A
                # shape-changing TRANSFORMER branch is excluded — it declares its
                # own output schema or none, never the base's.
                if (
                    preserves_output
                    and not isinstance(tool_func, Tool)
                    and "output_schema" not in bind_kwargs
                    and base_output_schema is not None
                    and not _declares_own_output_schema(tool_func)
                ):
                    bind_kwargs["output_schema"] = base_output_schema
                self.bind_tool(tool_func, curr_name, orig_name, *args, description=tool_desc, **bind_kwargs)
                if owner is not None:
                    self._mcp_bound_tools.setdefault(owner, set()).add(curr_name)

            return func

        return bind

    def bind_tool(self, func, curr_name, orig_name, *args, description: str | None = None, **kwargs):
        self._tool_registry.register_extend_tool(orig_name, curr_name)

        if isinstance(func, Tool):
            # A prebuilt Tool object is registered DIRECTLY so its typed schema
            # (hidden baked args, remaining real arg types/descriptions) survives
            # unchanged; wrapping it in a function would flatten that to one opaque
            # blob. Its own name/description/tags ride on the object.
            return self._fast_mcp.add_tool(func)
        if not callable(func):
            raise TypeError(
                f"cannot bind {type(func).__name__} as tool {curr_name!r}: expected a callable or a FastMCP Tool object"
            )

        return self._fast_mcp.tool(
            *args,
            name=curr_name,
            description=description if description is not None else inspect.getdoc(func),
            **kwargs,
        )(func)

    def _branch_base_callable(self, tool_obj: Tool) -> Callable[..., Any]:
        """The callable an extension branch wraps when the bound base is a prebuilt
        Tool object.

        A ``FunctionTool`` exposes its real ``fn`` directly. A ``TransformedTool``
        (a preset's baked tool) has no plain callable — its ``fn`` takes one opaque
        ``**kwargs`` and returns a ``ToolResult`` — so reconstruct a typed partial
        of the UNDERLYING function with the hidden baked args applied: it presents
        the remaining typed signature and returns the raw value, so a
        schema-preserving wrapper composes on it exactly as on a native tool.
        Anything else has no branchable body and raises loudly."""
        if isinstance(tool_obj, FunctionTool):
            return tool_obj.fn
        if isinstance(tool_obj, TransformedTool):
            return _cached_baked_partial(tool_obj)
        raise TypeError(
            f"cannot branch-bind {type(tool_obj).__name__} {tool_obj.name!r}: "
            "expected a FunctionTool or TransformedTool"
        )

    @staticmethod
    def normalized_name(prefix: str, name: str) -> str:
        name = name.lower().replace("-", "_")
        prefix = prefix.lower().replace("-", "_")
        # Prefix unless already prefixed. The check must respect the ``_``
        # separator: ``slacker`` under prefix ``slack`` is NOT already prefixed
        # (a bare ``startswith(prefix)`` would wrongly treat it as such), so it
        # becomes ``slack_slacker``.
        already_prefixed = name == prefix or name.startswith(prefix + "_")
        return name if not prefix or already_prefixed else f"{prefix}_{name}"
