"""Op-level oracles for ``get_resource_by_id``.

``get_resource_by_id`` backs the ``POST /api/resources/get`` route and carries the
route's typed error surface: it maps ``FileNotFoundError`` / ``ValueError`` /
``UnsafeTemplatePathError`` to ``NotFoundError`` (404) and ``BadRequestError`` (400).
A READ door: not destructive, projects normally.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastmcp.utilities.types import Image
from tai_contract.manifest import ApiToolsConfig

import tai_skeleton.operations.resources as resources_ops
from tai_skeleton.operations import BadRequestError, NotFoundError, OperationRegistry, operation_metadata_of
from tai_skeleton.operations.projection import project_operations
from tai_skeleton.operations.resources import get_resource_by_id
from tai_skeleton.template.media import MediaBlock


class _ResourceManager:
    def __init__(
        self,
        *,
        loaded: str | MediaBlock = "",
        rendered: str = "",
        raise_exc: Exception | None = None,
    ) -> None:
        self.load_calls: list[str] = []
        self.render_calls: list[tuple[str | None, dict | None]] = []
        self._loaded = loaded
        self._rendered = rendered
        self._raise = raise_exc

    async def load_file(self, source: str) -> str | MediaBlock:
        self.load_calls.append(source)
        if self._raise is not None:
            raise self._raise
        return self._loaded

    async def render_by_id_or_content(
        self, content: str | None = None, template_id: str | None = None, kwargs: dict[str, Any] | None = None
    ) -> str:
        self.render_calls.append((content, kwargs))
        return self._rendered


def _bind(monkeypatch: pytest.MonkeyPatch, manager: _ResourceManager) -> None:
    fake_app = SimpleNamespace(storage=SimpleNamespace(resource_manager=manager))
    monkeypatch.setattr(resources_ops, "tai_app", fake_app)


async def test_returns_loaded_text_unrendered(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _ResourceManager(loaded="raw {{ body }}")
    _bind(monkeypatch, manager)

    result = await get_resource_by_id("doc.txt")

    assert result == "raw {{ body }}"
    assert manager.load_calls == ["doc.txt"]
    assert manager.render_calls == []  # no kwargs -> no render


async def test_returns_media_block(monkeypatch: pytest.MonkeyPatch) -> None:
    block = Image(data=b"\x89PNG\r\n", format="png")
    manager = _ResourceManager(loaded=block)
    _bind(monkeypatch, manager)

    assert await get_resource_by_id("logo.png") is block


async def test_renders_text_with_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _ResourceManager(loaded="Hello {{ name }}", rendered="Hello Ada")
    _bind(monkeypatch, manager)

    result = await get_resource_by_id("greeting.j2", {"name": "Ada"})

    assert result == "Hello Ada"
    assert manager.render_calls == [("Hello {{ name }}", {"name": "Ada"})]


async def test_render_of_media_is_bad_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # The op raises the typed 400 for a render ``ValueError``.
    manager = _ResourceManager(loaded=Image(data=b"\x89PNG", format="png"))
    _bind(monkeypatch, manager)

    with pytest.raises(BadRequestError, match="Cannot render media"):
        await get_resource_by_id("logo.png", {"x": 1})


async def test_missing_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    # The op maps a ``FileNotFoundError`` to the route 404.
    manager = _ResourceManager(raise_exc=FileNotFoundError("greeting.j2"))
    _bind(monkeypatch, manager)

    with pytest.raises(NotFoundError, match="not found"):
        await get_resource_by_id("greeting.j2")


async def test_rejects_traversal_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``get_resource_by_id`` funnels through ``ResourceManager.load``'s empty-scheme
    # branch, so the read-side guard fires on a traversal id before any storage read;
    # the op maps the guard's ``UnsafeTemplatePathError`` to the route 400.
    from tai_skeleton.storage import StorageRegistry
    from tai_skeleton.template import ResourceManager
    from tests.template.test_resource_manager import _InMemoryStorage

    registry = StorageRegistry()
    registry.register_storage(_InMemoryStorage)
    real_manager = ResourceManager(registry.provider)
    monkeypatch.setattr(
        resources_ops, "tai_app", SimpleNamespace(storage=SimpleNamespace(resource_manager=real_manager))
    )

    with pytest.raises(BadRequestError):
        await get_resource_by_id("../x")


async def test_broken_jinja_is_bad_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from jinja2 import TemplateSyntaxError

    manager = _ResourceManager(loaded="{{ oops")
    _bind(monkeypatch, manager)

    async def _broken(content=None, template_id=None, kwargs=None) -> str:
        raise TemplateSyntaxError("unexpected end of template", 1)

    monkeypatch.setattr(manager, "render_by_id_or_content", _broken)
    with pytest.raises(BadRequestError, match="template error"):
        await get_resource_by_id("t.j2", {})


def test_projects_as_a_read_tool() -> None:
    reg = OperationRegistry()
    reg.register(operation_metadata_of(get_resource_by_id))

    class _Rec:
        def __init__(self) -> None:
            self.registered: dict[str, dict] = {}

        def tool(self, *, force, name, tags, annotations):
            self.registered[name] = {"annotations": annotations}
            return lambda fn: fn

    app = SimpleNamespace(tools=_Rec())
    names = project_operations(app, ApiToolsConfig(), registry=reg)

    assert names == ["get_resource_by_id"]  # default-in read door
    assert app.tools.registered["get_resource_by_id"]["annotations"] is None  # not destructive
