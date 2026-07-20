"""Backend feature tests.

Conformance: the seam re-exports the contract ``Backend`` ABC, an incomplete
subclass cannot instantiate, and a complete subclass satisfies the contract.
Behavior: ``prepare_backend_kwargs`` injects the tool name, and
``callback_execution`` gates on the rendered condition before running a tool.
"""

from __future__ import annotations

import pytest
from tai42_contract.app import tai42_app
from tai42_contract.backend import Backend as ContractBackend

from tai42_skeleton import backend as _skeleton_backend
from tai42_skeleton.backend import (
    CallbackSchema,
    callback_execution,
    prepare_backend_kwargs,
)

# The seam re-exports the contract ``Backend`` ABC through ``tai42_skeleton``;
# reference it via the module so the identity check below stays meaningful.
Backend = _skeleton_backend.Backend

# --- conformance ----------------------------------------------------------


def test_seam_re_exports_contract_abc() -> None:
    assert Backend is ContractBackend


def test_incomplete_backend_cannot_instantiate() -> None:
    # ``launch`` is the sole abstract member of the Backend ABC — a subclass that
    # omits it cannot instantiate.
    class Partial(Backend):
        pass

    with pytest.raises(TypeError):
        Partial()  # pyright: ignore[reportAbstractUsage]


def test_complete_backend_satisfies_contract() -> None:
    # The task backend carries task execution only — ``launch`` is the whole ABC;
    # fleet fan-out is the app's worker bus, not a backend surface.
    class Dummy(Backend):
        async def launch(self, args) -> None:
            return None

    backend = Dummy()
    assert isinstance(backend, Backend)
    assert isinstance(backend, ContractBackend)


# --- behavior -------------------------------------------------------------


class _FakeResourceManager:
    async def render_by_id_or_content(self, content=None, template_id=None, kwargs=None):
        # The impl render mixins forward inline content through unchanged.
        return content


class _FakeStorage:
    resource_manager = _FakeResourceManager()


class _FakeTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def run_tool(self, key, arguments):
        self.calls.append((key, arguments))
        return {"ran": key, "arguments": arguments}


class _FakeApp:
    def __init__(self) -> None:
        self.storage = _FakeStorage()
        self.tools = _FakeTools()


@pytest.fixture
def app():
    fake = _FakeApp()
    previous = object.__getattribute__(tai42_app, "_impl")
    tai42_app.bind(fake)
    try:
        yield fake
    finally:
        tai42_app.bind(previous)


async def test_prepare_backend_kwargs_injects_tool_name() -> None:
    def func(a, b):  # no FastMCP Context arg → kwargs pass through
        ...

    out = await prepare_backend_kwargs(func, "tool_name", "my_tool", {"a": 1, "b": 2})
    assert out == {"a": 1, "b": 2, "tool_name": "my_tool"}


async def test_callback_runs_tool_when_condition_passes(app) -> None:
    callback = CallbackSchema(condition=".ok", expr=".value", tool="follow_up")
    result = await callback_execution({"ok": True, "value": 42}, callback, app)
    assert result == {"ran": "follow_up", "arguments": 42}
    assert app.tools.calls == [("follow_up", 42)]


async def test_callback_skips_when_condition_fails(app) -> None:
    callback = CallbackSchema(condition=".ok", expr=".value", tool="follow_up")
    result = await callback_execution({"ok": False, "value": 42}, callback, app)
    assert result is None
    assert app.tools.calls == []


async def test_callback_returns_expr_when_no_tool(app) -> None:
    callback = CallbackSchema(condition=".ok", expr=".value", tool="")
    result = await callback_execution({"ok": True, "value": 42}, callback, app)
    assert result == 42
    assert app.tools.calls == []


async def test_callback_no_expr_returns_empty_mapping(app) -> None:
    # No expr must not crash — ``get_compiled_jq("")`` would raise, so an absent
    # expr yields an empty mapping instead of evaluating jq.
    callback = CallbackSchema(condition=".ok", tool="")
    result = await callback_execution({"ok": True, "value": 42}, callback, app)
    assert result == {}
    assert app.tools.calls == []


async def test_callback_no_expr_runs_tool_with_empty_args(app) -> None:
    callback = CallbackSchema(condition=".ok", tool="follow_up")
    result = await callback_execution({"ok": True, "value": 42}, callback, app)
    assert result == {"ran": "follow_up", "arguments": {}}
    assert app.tools.calls == [("follow_up", {})]
