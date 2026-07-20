"""Apply-site enforcement of the tool-extension kind contract.

Every case drives the REAL ``ToolBinding.bind_tool_func`` path through
``app.app_context`` with a fixture manifest: the kind rules
(``preserves_schema`` / ``declares_schema``) are enforced when a tool module is
imported at start. Fixtures live in :mod:`tests.app._fixtures.ext_kinds` and
extend the base tool ``shout(text: str)`` in
:mod:`tests.app._fixtures.tools_b`.
"""

from __future__ import annotations

import asyncio

import pytest

from tai42_skeleton.app.instance import app
from tai42_skeleton.exceptions.exceptions import TaiValidationError
from tai42_skeleton.manifest import Manifest


@pytest.fixture(autouse=True)
def _clean_server():
    """Clear the singleton FastMCP server's tools around each test — it outlives
    one ``app_context``, so a tool a prior test bound would collide with this
    test's bind under ``on_duplicate="error"``."""

    async def _clear() -> None:
        provider = app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    yield
    asyncio.run(_clear())


def _manifest(tool: str, *extensions: str) -> Manifest:
    # Selection (``include``) and attachment (``extensions``) are separate: the
    # variadic ``extensions`` form ONE combo attached to ``tool``, so a stacked
    # call like ``_manifest("shout", "concretetf", "argswrap")`` layers both onto
    # a single ``shout_concretetf_argswrap`` branch.
    return Manifest.model_validate(
        {
            "extensions_modules": ["tests.app._fixtures.ext_kinds"],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.app._fixtures.tools_b",
                    "include": [tool],
                    "extensions": {tool: [list(extensions)]},
                }
            ],
        }
    )


# --- wrapper: preserves_schema ----------------------------------------------


def test_wrapper_preserving_schema_binds_and_runs():
    async def run():
        async with app.app_context(_manifest("shout", "argswrap")):
            tools = await app.tools.get_tools()
            # Branch bound under the new name; original still bound.
            assert {"shout", "shout_argswrap"} <= set(tools)
            # The *args/**kwargs impl body ran behind the preserved signature.
            assert await app.tools.run_tool("shout_argswrap", {"text": "hi"}) == "HI"

    asyncio.run(run())


def test_wrapper_changing_schema_rejected():
    async def run():
        async with app.app_context(_manifest("shout", "renamep")):
            pass

    with pytest.raises(TaiValidationError, match=r"'renamep' changed the schema of tool 'shout'"):
        asyncio.run(run())


def test_wrapper_after_transformer_baseline_is_layer_input():
    # ``{shout: [[concretetf, argswrap]]}``: the wrapper ``argswrap`` must preserve the
    # TRANSFORMER's composed schema, not the original tool's — comparing against
    # the original would wrongly reject this valid stack.
    async def run():
        async with app.app_context(_manifest("shout", "concretetf", "argswrap")):
            tools = await app.tools.get_tools()
            assert {"shout", "shout_concretetf", "shout_concretetf_argswrap"} <= set(tools)

    asyncio.run(run())


def test_wrapper_default_less_reserved_param_tolerated():
    # ``exp`` is reserved and default-less, so it lands in both ``properties``
    # and ``required``; both must be subtracted before the equality check.
    async def run():
        async with app.app_context(_manifest("shout", "cachereq")):
            assert "shout_cachereq" in await app.tools.get_tools()

    asyncio.run(run())


def test_wrapper_reserved_list_param_tolerated():
    async def run():
        async with app.app_context(_manifest("shout", "proxylike")):
            assert "shout_proxylike" in await app.tools.get_tools()

    asyncio.run(run())


def test_wrapper_reserved_only_param_empties_required():
    # ``ping`` has no required params; the wrapper's sole added param is a
    # default-less reserved ``exp``, so subtracting it empties the branch's
    # ``required`` list — the normalization must make it compare equal to the
    # no-required baseline.
    async def run():
        async with app.app_context(_manifest("ping", "onlyreserved")):
            assert "ping_onlyreserved" in await app.tools.get_tools()

    asyncio.run(run())


def test_wrapper_reserved_name_colliding_with_input_rejected():
    async def run():
        async with app.app_context(_manifest("shout", "collidewrap")):
            pass

    with pytest.raises(TaiValidationError, match=r"reserved param 'text' that already exists"):
        asyncio.run(run())


def test_wrapper_reserved_does_not_mask_real_drift():
    # ``exp`` is tolerated, but the rename ``text`` -> ``txt`` is real drift the
    # subtraction must not hide.
    async def run():
        async with app.app_context(_manifest("shout", "driftreserved")):
            pass

    with pytest.raises(TaiValidationError, match=r"'driftreserved' changed the schema of tool 'shout'"):
        asyncio.run(run())


# --- transformer: declares_schema -------------------------------------------


def test_transformer_concrete_signature_binds():
    async def run():
        async with app.app_context(_manifest("shout", "concretetf")):
            assert "shout_concretetf" in await app.tools.get_tools()

    asyncio.run(run())


def test_transformer_bare_signature_rejected():
    async def run():
        async with app.app_context(_manifest("shout", "baretf")):
            pass

    with pytest.raises(TaiValidationError, match=r"'baretf' presents a bare \(\*args, \*\*kwargs\)"):
        asyncio.run(run())


def test_transformer_ignores_reserved_params():
    # ``reserved_params`` on a non-wrapper kind is ignored — the concrete
    # transformer binds clean regardless.
    async def run():
        async with app.app_context(_manifest("shout", "reservedtf")):
            assert "shout_reservedtf" in await app.tools.get_tools()

    asyncio.run(run())


# --- stacking order ----------------------------------------------------------


def test_stacking_order_pinned():
    # The ``[marka, markb]`` combo applies ``marka`` first then ``markb`` -> ``b(a(tool))``
    # with ``markb`` outermost; each prefix binds its own branch tool.
    async def run():
        async with app.app_context(_manifest("shout", "marka", "markb")):
            tools = await app.tools.get_tools()
            assert {"shout", "shout_marka", "shout_marka_markb"} <= set(tools)
            # marka appended first, then markb -> the ``b`` marker is outermost.
            assert await app.tools.run_tool("shout_marka_markb", {"text": "hi"}) == "hi|a|b"

    asyncio.run(run())


# --- backend: no schema rule -------------------------------------------------


def test_two_backends_on_one_tool_rejected():
    async def run():
        async with app.app_context(_manifest("shout", "backendx", "backendy")):
            pass

    with pytest.raises(TaiValidationError, match=r"Only one 'backend' extension is allowed"):
        asyncio.run(run())


def test_single_backend_altering_schema_binds_clean():
    # A backend branch carries NO schema rule, so a schema-altering single
    # backend binds without a TaiValidationError.
    async def run():
        async with app.app_context(_manifest("shout", "backendswap")):
            assert "shout_backendswap" in await app.tools.get_tools()

    asyncio.run(run())
