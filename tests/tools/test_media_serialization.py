"""Direct-run ``run_tool`` media serialization.

A tool returning a live fastmcp ``Image`` / ``Audio`` is converted to the MCP
media wire dict (``{"type", "data", "mimeType"}``) BEFORE ``to_jsonable_python``,
which would otherwise raise on the live object; a ``File`` falls through to a
JSON ``EmbeddedResource`` (NOT the media wire shape); a plain text result is
unchanged. The result is JSON-native (survives the route's ``JSONResponse``).

Driven through the real ``instance.app.tools.run_tool`` end to end (the singleton
FastMCP server, cleared around each test), so ``to_jsonable_python`` runs for
real."""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from typing import Any

import pytest

from tai42_skeleton.app import instance
from tai42_skeleton.manifest import Manifest

_MEDIA_MODULE = "tests.app._fixtures.media_tools"
_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
_WAV_B64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVE").decode()
_FILE_B64 = base64.b64encode(b"arbitrary blob").decode()


def _manifest() -> dict:
    return {
        "tools": [
            {
                "title": "media",
                "module": _MEDIA_MODULE,
                "include": ["make_image", "make_audio", "make_file", "make_text"],
            }
        ]
    }


async def _clear_server() -> None:
    provider = instance.app._fast_mcp.local_provider
    for tool in list(await provider.list_tools()):
        provider.remove_tool(tool.name)


@pytest.fixture(autouse=True)
def _clean_server():
    """Clear the singleton FastMCP server's tools around each test — it outlives
    one ``app_context``, so a tool bound in a prior test would otherwise linger."""
    asyncio.run(_clear_server())
    yield
    asyncio.run(_clear_server())


@asynccontextmanager
async def _running():
    async with instance.app.app_context(Manifest.model_validate(_manifest())):
        yield


def _run(name: str) -> Any:
    async def go() -> Any:
        # Clear first: the shared singleton server outlives one ``app_context``,
        # and a test calling ``_run`` more than once re-enters ``app_context`` and
        # re-binds the same media tools — a collision under ``on_duplicate="error"``.
        await _clear_server()
        async with _running():
            return await instance.app.tools.run_tool(name, {})

    return asyncio.run(go())


def test_image_return_serializes_to_media_wire_shape():
    result = _run("make_image")
    assert result["type"] == "image"
    assert result["data"] == _PNG_B64
    assert result["mimeType"] == "image/png"


def test_audio_return_serializes_to_media_wire_shape():
    result = _run("make_audio")
    assert result["type"] == "audio"
    assert result["data"] == _WAV_B64
    assert result["mimeType"] == "audio/wav"


def test_file_return_falls_through_to_json_resource():
    result = _run("make_file")
    # A File is NOT media-rendered — it serializes as an EmbeddedResource, never
    # the image/audio media wire shape.
    assert result["type"] == "resource"
    assert result["type"] not in {"image", "audio"}
    assert result["resource"]["blob"] == _FILE_B64


def test_text_return_is_unchanged():
    assert _run("make_text") == "just text"


def test_preset_over_media_tool_preserves_media():
    # A preset (a TransformedTool) over a media-returning base has no structured
    # and no text content — its result-value reduction must serialize the media
    # content block to its wire dict, not silently drop it to None.
    async def go() -> Any:
        await _clear_server()
        async with _running():
            await instance.app.preset_manager.register("img_preset", "make_image", {}, [], [], "d")
            return await instance.app.tools.run_tool("img_preset", {})

    result = asyncio.run(go())
    assert result["type"] == "image"
    assert result["data"] == _PNG_B64
    assert result["mimeType"] == "image/png"


def test_media_result_is_json_native():
    # The route serializes via ``JSONResponse({"data": result})`` — the media wire
    # dict must survive a plain ``json.dumps`` (no live fastmcp object leaks).
    for name in ("make_image", "make_audio", "make_file"):
        round_tripped = json.loads(json.dumps(_run(name)))
        assert round_tripped["type"] in {"image", "audio", "resource"}
