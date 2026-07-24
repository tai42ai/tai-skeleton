"""Agent registration engine — the impl body behind the ``app.agents`` facet.

Owns the live agent instances and the synthesized per-agent ``run`` tool. The
lifecycle calls :meth:`reset` on every start/reload so a dropped agent doesn't
linger; the importer then re-fires each agents-module's decorator.
"""

import copy
import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, cast

from fastmcp.tools.function_tool import FunctionTool
from makefun import create_function
from pydantic import BaseModel
from pydantic_core import PydanticUndefined, core_schema
from tai42_contract.agent import Agent

from tai42_skeleton.agent.thread_reservation import run_kwargs_from_tool_input
from tai42_skeleton.exceptions.exceptions import TaiValidationError

if TYPE_CHECKING:
    from tai42_skeleton.app.server import TaiMCP

logger = logging.getLogger(__name__)

_AgentT = TypeVar("_AgentT", bound=Agent)


class _Unset:
    """Marker for an agent run-tool parameter the caller did not supply.

    The synthesized run tool presents a concrete keyword-only parameter per
    ``ToolInput`` field so a wrapper/transformer extension can compose over a real
    signature. An optional field's parameter defaults to the module-level
    :data:`_UNSET` instance; the body forwards only the parameters whose value is
    NOT that sentinel, preserving ``from_tool_input``'s set-fields-only contract (a
    materialized model default would forward every field). ``__get_pydantic_core_schema__``
    maps the sentinel to an ``any`` schema so pydantic derives the parameter as
    optional and omits the non-serializable sentinel default from the JSON schema.
    """

    __get_pydantic_core_schema__ = classmethod(lambda cls, source, handler: core_schema.any_schema())


_UNSET = _Unset()

# The output schema advertised for every synthesized agent run tool. ``result``
# is intentionally unconstrained (an agent returns free-form text OR a structured
# dict); ``x-fastmcp-wrap-result`` tells FastMCP to wrap the return as
# ``{"result": <value>}`` in ``structuredContent`` and the client to unwrap it, so
# a scalar (text) return surfaces in ``result.data`` instead of only ``content``.
_AGENT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"result": {"title": "Result"}},
    "x-fastmcp-wrap-result": True,
}


def _run_tool_signature(tool_input: type[BaseModel]) -> inspect.Signature:
    """A concrete keyword-only signature mirroring ``tool_input``'s fields.

    Each field becomes a keyword-only parameter annotated ``Annotated[<field
    annotation>, <a copy of the field's FieldInfo with its default cleared>]`` —
    the copy-and-clear keeps every schema-affecting attribute (description,
    constraints, alias, title, examples) while removing the model default so it
    cannot clash with the parameter default. A required field gets no parameter
    default; an optional field defaults to :data:`_UNSET`, the runtime marker the
    body strips before validation. Extensions derive their branch schema from this
    signature, so it carries the agent's real input contract."""
    params: list[inspect.Parameter] = []
    for field_name, field_info in tool_input.model_fields.items():
        stripped = copy.deepcopy(field_info)
        stripped.default = PydanticUndefined
        stripped.default_factory = None
        annotation = Annotated[field_info.annotation, stripped]
        if field_info.is_required():
            params.append(inspect.Parameter(field_name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation))
        else:
            params.append(
                inspect.Parameter(field_name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation, default=_UNSET)
            )
    return inspect.Signature(params, return_annotation=Any)


class AgentBinding:
    """Registers agents, keeps their live instances, and binds their run tools."""

    def __init__(self, app: "TaiMCP") -> None:
        self._app = app
        self._agents: dict[str, Agent] = {}

    def reset(self) -> None:
        """Drop every registered agent (start/reload re-imports their modules)."""
        self._agents = {}

    def agent(self, name: str) -> Callable[[type[_AgentT]], type[_AgentT]]:
        """Register an :class:`Agent` subclass under ``name`` and synthesize its
        JSON ``run`` tool.

        Fires when an ``agents:``-listed module imports. Gates the agent via the
        manifest ``agents:`` section (NOT the tools namespace), instantiates and
        stores it for in-process ``get_agent(name).astream(...)``, and
        auto-registers a ``run`` tool whose signature == the agent's ``ToolInput``
        and whose body drives ``run`` to its final value.

        Enforces the ``preset_bakeable_fields`` invariant at registration: every
        declared name must be a real ``ToolInput`` field. A stray name could never
        pass the preset route's unknown-field check, so it would be a silently dead
        declaration — this raises loudly instead.

        Returns the class unchanged, so the decorated symbol keeps its concrete
        subclass type.
        """

        def decorator(agent_cls: type[_AgentT]) -> type[_AgentT]:
            manifest = self._app._manifest
            if not manifest:
                return agent_cls
            if not manifest.should_include_agent(name, agent_cls.__module__):
                return agent_cls

            # The registry is reset on every start/reload, so a name already
            # present within one boot is a genuine collision (two modules
            # registering the same agent) — fail loud rather than last-write-win.
            if name in self._agents:
                raise TaiValidationError(f"Agent '{name}' is already registered.")

            # Every ``preset_bakeable_fields`` name must be a real ``ToolInput``
            # field: a stray name could never pass the preset route's unknown-field
            # check, so it would be a silently dead declaration. Reject it loudly
            # here rather than let it lie dormant.
            stray = set(agent_cls.preset_bakeable_fields) - set(agent_cls.ToolInput.model_fields)
            if stray:
                raise RuntimeError(
                    f"Agent '{name}' declares preset_bakeable_fields not on its ToolInput: {sorted(stray)}."
                )

            instance: Agent = agent_cls()
            self._agents[name] = instance
            self._register_agent_tool(name, agent_cls.__module__)
            return agent_cls

        return decorator

    def get_agent(self, name: str) -> Agent:
        agent = self._agents.get(name)
        if agent is None:
            raise RuntimeError(f"No such agent: {name}.")
        return agent

    def all_agents(self) -> dict[str, Agent]:
        """Every registered agent keyed by registration name.

        A shallow copy, so a caller iterating the result cannot mutate the live
        registry. The API agents surface reads this to list agents alongside the
        per-agent ``run`` tool the same registration synthesized.
        """
        return dict(self._agents)

    def _register_agent_tool(self, name: str, module: str) -> None:
        """Build + bind the JSON ``run`` tool for a registered agent.

        The tool advertises the agent's ``ToolInput`` schema (live ``tools=`` is
        deliberately absent — API-only via ``astream``). The body sees EXACTLY the
        caller-supplied argument keys — the optional parameters the caller omitted
        arrive as the :data:`_UNSET` sentinel and are stripped, so
        ``model_fields_set`` after ``model_validate`` reflects only what was passed
        — then maps to ``run`` kwargs through the shared reservation seam (the
        agent's ``from_tool_input``, which forwards set fields only, plus the
        ``bridge:`` thread refusal) and returns ``run``'s final value (drained per
        the terminal rule inside the agent's own ``run``). A non-JSON ``ToolInput``
        field fails loudly here at ``model_json_schema()`` time.

        The body presents a concrete per-field signature (:func:`_run_tool_signature`)
        so a wrapper/transformer extension can compose its branch schema over a real
        signature; the tool binds through the shared extension-capable path
        (``bind_tool_func``), which mints a branch per attached extension combo. The
        base itself is registered as a prebuilt ``FunctionTool`` whose advertised
        ``parameters`` is set explicitly to the model's JSON schema — exact by
        construction — while the concrete signature exists FOR the branch composition,
        not for base-schema derivation.
        """
        agent = self._agents[name]
        tool_input = agent.ToolInput

        signature = _run_tool_signature(tool_input)

        async def run_impl(**arguments: Any) -> Any:
            supplied = {key: value for key, value in arguments.items() if value is not _UNSET}
            validated = tool_input.model_validate(supplied)
            run_kwargs = run_kwargs_from_tool_input(agent, validated)
            return await self.get_agent(name).run(**run_kwargs)

        # makefun's ``func_impl`` is typed ``Callable[[Any], Any]`` but it accepts
        # any callable (it drives the separate ``signature`` above); the ``**arguments``
        # impl is valid at runtime.
        impl = create_function(
            signature,
            cast(Callable[[Any], Any], run_impl),
            func_name=name,
            module_name=module,
            doc=agent.tool_description,
        )
        run_tool = FunctionTool(
            name=name,
            description=agent.tool_description,
            parameters=tool_input.model_json_schema(),
            fn=impl,
            # The body returns ``Any`` — free-form text (no ``response_format``) or a
            # structured dict — so FastMCP derives no output schema and a scalar
            # (text) return would reach the caller only as unstructured ``content``,
            # never ``structuredContent``/``result.data`` (a dict return already
            # populates it). Advertise a permissive wrap schema so EVERY agent run
            # surfaces its result in ``result.data`` like any typed tool: the server
            # wraps the return as ``{"result": <value>}`` and the client unwraps it
            # back (a string stays a string, a dict stays a dict).
            output_schema=_AGENT_RESULT_SCHEMA,
        )
        # Bind through the shared extension-capable path: the base registers as this
        # prebuilt ``FunctionTool`` (advertised schema exact by construction) and one
        # branch is minted per attached extension combo, composed over the concrete
        # signature. The optional parameters default to the non-serializable
        # ``_UNSET`` sentinel, so any JSON-schema derivation over this signature emits
        # a benign ``PydanticJsonSchemaWarning`` (excluding the sentinel default) — a
        # narrow ``filterwarnings`` ignore in ``pyproject.toml`` keeps the intended
        # omission from tripping the warnings-as-error test policy.
        self._app._tool_binding.bind_tool_func(name=name)(run_tool)
