"""Op-level oracles for the templates operations.

``upload_template`` takes a ``path`` argument and returns ``{"path", "uploaded"}``;
the traversal guard raises the typed ``BadRequestError`` (the route's ``400``). The
other ops are pinned directly (the route oracles pin the same behavior through the
adapter). Projection carries ``destructiveHint`` for the mutating ops.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tai42_contract.manifest import ApiToolsConfig

import tai42_skeleton.operations.templates as templates_ops
from tai42_skeleton.operations import BadRequestError, NotFoundError, OperationRegistry, operation_metadata_of
from tai42_skeleton.operations.projection import project_operations
from tai42_skeleton.operations.templates import (
    clear_templates_cache,
    delete_template,
    get_template,
    list_templates,
    render_template,
    upload_template,
)
from tai42_skeleton.template import TemplateNotFoundError


class _ResourceManager:
    def __init__(self, *, listed: list[str] | None = None) -> None:
        self.uploaded: dict[str, str] = {}
        self.deleted: list[str] = []
        self.cleared = False
        self._listed = listed or []

    async def list_resources(self) -> list[str]:
        return self._listed

    async def fetch_template(self, template_id: str) -> str:
        return f"content of {template_id}"

    async def get_template_schema(self, content=None, template_id=None) -> dict:
        return {"vars": ["name"]}

    async def upload_template(self, path: str, content: str) -> None:
        self.uploaded[path] = content

    async def delete_template(self, path: str) -> None:
        self.deleted.append(path)

    async def render_by_id_or_content(self, content=None, template_id=None, kwargs=None) -> str:
        return f"rendered:{template_id or content}:{kwargs}"

    def clear_cache(self) -> None:
        self.cleared = True


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> _ResourceManager:
    tm = _ResourceManager()
    fake_app = SimpleNamespace(storage=SimpleNamespace(resource_manager=tm))
    monkeypatch.setattr(templates_ops, "tai42_app", fake_app)
    return tm


# -- upload_template --


async def test_upload_template_delegates_and_returns_path_uploaded(manager: _ResourceManager) -> None:
    # Arg ``path``; returns ``{"path", "uploaded"}``.
    result = await upload_template("greeting.j2", "Hi {{ name }}")

    assert result == {"path": "greeting.j2", "uploaded": True}
    assert manager.uploaded == {"greeting.j2": "Hi {{ name }}"}


@pytest.mark.parametrize("bad", ["/abs.j2", "../escape.j2", "a/../../etc", "back\\slash"])
async def test_upload_template_rejects_traversal_path(manager: _ResourceManager, bad: str) -> None:
    # A traversal path raises the route's typed ``BadRequestError`` (400) — never
    # reaching the store.
    with pytest.raises(BadRequestError):
        await upload_template(bad, "content")
    assert manager.uploaded == {}


async def test_upload_template_rejects_non_string_content(manager: _ResourceManager) -> None:
    with pytest.raises(BadRequestError, match="content must be a string"):
        await upload_template("ok.j2", 123)  # type: ignore[arg-type]
    assert manager.uploaded == {}


# -- get / delete / render / list / clear characterization ---------------------


async def test_get_template_returns_content_and_schema(manager: _ResourceManager) -> None:
    result = await get_template("a.j2")
    assert result == {"template": "content of a.j2", "schema": {"vars": ["name"]}}


async def test_get_template_empty_id_is_field_specific_400(manager: _ResourceManager) -> None:
    # A blank id is a field-specific 400 naming ``template_id`` (ahead of the path
    # guard, whose generic message names ``path``).
    with pytest.raises(BadRequestError, match="template_id must be a non-empty string"):
        await get_template("")


async def test_get_template_missing_is_not_found(manager: _ResourceManager, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _missing(template_id: str) -> str:
        raise TemplateNotFoundError(f"Template '{template_id}' not found.")

    monkeypatch.setattr(manager, "fetch_template", _missing)
    with pytest.raises(NotFoundError, match="not found"):
        await get_template("gone.j2")


async def test_delete_template_delegates(manager: _ResourceManager) -> None:
    result = await delete_template("x.j2")
    assert result == {"path": "x.j2", "deleted": True}
    assert manager.deleted == ["x.j2"]


@pytest.mark.parametrize("bad", ["/abs.j2", "../escape.j2", "back\\slash"])
async def test_delete_template_rejects_traversal(manager: _ResourceManager, bad: str) -> None:
    with pytest.raises(BadRequestError):
        await delete_template(bad)
    assert manager.deleted == []


async def test_render_template_requires_a_source(manager: _ResourceManager) -> None:
    with pytest.raises(BadRequestError, match="one of"):
        await render_template()


async def test_render_template_rejects_both(manager: _ResourceManager) -> None:
    with pytest.raises(BadRequestError, match="not both"):
        await render_template(content="hi", template_id="a.j2")


async def test_render_template_by_id(manager: _ResourceManager) -> None:
    result = await render_template(template_id="a.j2", kwargs={"name": "Z"})
    assert "rendered:a.j2" in result["rendered"]


async def test_render_template_rejects_non_dict_kwargs(manager: _ResourceManager) -> None:
    with pytest.raises(BadRequestError, match="'kwargs' must be a JSON object"):
        await render_template(template_id="a.j2", kwargs=["not", "a", "dict"])  # type: ignore[arg-type]


async def test_render_template_rejects_non_string_content(manager: _ResourceManager) -> None:
    with pytest.raises(BadRequestError, match="'content' must be a string"):
        await render_template(content=123)  # type: ignore[arg-type]


async def test_render_template_missing_is_not_found(manager: _ResourceManager, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _missing(content=None, template_id=None, kwargs=None) -> str:
        raise TemplateNotFoundError("Template 'gone.j2' not found.")

    monkeypatch.setattr(manager, "render_by_id_or_content", _missing)
    with pytest.raises(NotFoundError):
        await render_template(template_id="gone.j2")


async def test_list_templates_returns_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    tm = _ResourceManager(listed=["a.j2", "b.j2"])
    monkeypatch.setattr(templates_ops, "tai42_app", SimpleNamespace(storage=SimpleNamespace(resource_manager=tm)))
    assert await list_templates() == ["a.j2", "b.j2"]


async def test_clear_templates_cache(manager: _ResourceManager) -> None:
    assert await clear_templates_cache() == {"cleared": True}
    assert manager.cleared is True


# -- projection: the mutating ops carry destructiveHint ------------------------


def test_upload_and_delete_project_with_destructive_hint() -> None:
    reg = OperationRegistry()
    for op in (upload_template, delete_template, get_template, list_templates):
        reg.register(operation_metadata_of(op))

    class _Rec:
        def __init__(self) -> None:
            self.registered: dict[str, dict] = {}

        def tool(self, *, force, name, tags, annotations):
            self.registered[name] = {"annotations": annotations}
            return lambda fn: fn

    app = SimpleNamespace(tools=_Rec())
    names = project_operations(app, ApiToolsConfig(expose_destructive=True), registry=reg)

    assert {"upload_template", "delete_template", "get_template", "list_templates"} <= set(names)
    assert app.tools.registered["upload_template"]["annotations"].destructiveHint is True
    assert app.tools.registered["delete_template"]["annotations"].destructiveHint is True
    assert app.tools.registered["get_template"]["annotations"] is None  # read, not destructive
