from collections.abc import Callable
from inspect import Parameter, Signature
from typing import Any, cast

from langchain_core.tools import BaseTool
from makefun import create_function
from pydantic import BaseModel


def build_signature(
    input_model: type[BaseModel] | None,
    return_annotation: Any = Any,  # Default to Any; customize if you have output schemas
) -> Signature:
    if input_model is None:
        params = [Parameter("input", Parameter.POSITIONAL_OR_KEYWORD, annotation=str)]
    else:
        # Iterate ``model_fields`` directly and key each parameter by FIELD name:
        # the JSON-schema ``properties`` are keyed by ALIAS, so indexing
        # ``model_fields`` by a property key breaks for a ``Field(alias=...)``
        # field. Callers pass by field name; ``_tool_input`` maps to the alias.
        params = [
            Parameter(
                name=field_name,
                kind=Parameter.KEYWORD_ONLY,
                # call_default_factory so a ``default_factory`` field (e.g.
                # ``tool_names: list = Field(default_factory=list)``) synthesizes
                # with its real default ([]), not None — passing None back into
                # the model would fail validation for a non-Optional field.
                default=Parameter.empty if field.is_required() else field.get_default(call_default_factory=True),
                annotation=field.annotation,
            )
            for field_name, field in input_model.model_fields.items()
        ]

    return Signature(parameters=params, return_annotation=return_annotation)


def lc_tool_to_func(
    lc_tool: BaseTool,
    name: str | None = None,
    description: str | None = None,
    async_mode: bool = True,
    module: str | None = None,
    output_schema: dict[str, Any] | None = None,  # If you have JSON schema for output
) -> Callable:
    func_name = (name or lc_tool.name).lower()
    func_desc = description or lc_tool.description

    # ``args_schema`` is a Pydantic-model type, a raw JSON-schema dict, or None;
    # this adapter supports only the model form, so reject a dict schema loudly.
    raw_schema = lc_tool.args_schema
    if raw_schema is not None and not (isinstance(raw_schema, type) and issubclass(raw_schema, BaseModel)):
        raise TypeError(
            f"Tool {lc_tool.name!r} uses a non-model args_schema; only Pydantic-model schemas are supported."
        )
    input_model: type[BaseModel] | None = raw_schema

    # If output_schema is provided, convert to type; else Any
    if output_schema:
        from tai_kit.utils.data.json_schema_util import json_schema_to_pydantic_model

        return_annotation = json_schema_to_pydantic_model(output_schema, "OutputModel")
    else:
        return_annotation = Any

    sig = build_signature(input_model, return_annotation)

    def _tool_input(kwargs: dict[str, Any]) -> Any:
        if input_model is None:
            # Schemaless: pass the raw 'input' string (the synthesized signature
            # always supplies it).
            return kwargs["input"]
        # Map each supplied FIELD-name kwarg to its ALIAS before validating, so a
        # ``Field(alias=...)`` field constructs correctly (validate-by-name is not
        # assumed). Dump by alias, excluding only fields still at their default —
        # NOT ``exclude_none``, which would drop an explicitly-passed ``None``
        # indistinguishably from an unset field, silently rewriting the call.
        data = {}
        for field_name, field in input_model.model_fields.items():
            if field_name in kwargs:
                data[field.alias or field_name] = kwargs[field_name]
        return input_model.model_validate(data).model_dump(by_alias=True, exclude_defaults=True)

    async def _impl_async(**kwargs: Any) -> Any:
        return await lc_tool.arun(_tool_input(kwargs))

    def _impl_sync(**kwargs: Any) -> Any:
        return lc_tool.run(_tool_input(kwargs))

    impl: Callable[..., Any] = _impl_async if async_mode else _impl_sync

    module = module or __name__
    wrapper_fn = create_function(
        func_signature=sig,
        # makefun's ``func_impl`` is annotated ``Callable[[Any], Any]`` but accepts
        # any callable, driven by ``func_signature``.
        func_impl=cast(Callable[[Any], Any], impl),
        func_name=func_name,
        qualname=func_name,
        doc=func_desc,
        module_name=module,
    )

    return wrapper_fn
