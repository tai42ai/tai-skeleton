"""ResourceManager tests: dead by default (raises loudly with no provider) and
renders through a registered in-memory Storage provider."""

from __future__ import annotations

import pytest
from tai42_contract.storage import Storage

from tai42_skeleton.storage import StorageRegistry
from tai42_skeleton.template import ResourceManager


class _InMemoryStorage(Storage):
    def __init__(self, items: dict[str, str] | None = None) -> None:
        self.items: dict[str, str] = dict(items or {})

    async def load(self, path: str) -> str:
        try:
            return self.items[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc

    async def list(self) -> list[str]:
        return sorted(self.items)

    async def upload(self, path: str, content: str) -> None:
        self.items[path] = content

    async def delete(self, path: str) -> None:
        self.items.pop(path, None)

    async def delete_dir(self, path: str) -> None:
        prefix = path.rstrip("/") + "/"
        for key in [k for k in self.items if k.startswith(prefix)]:
            del self.items[key]


async def test_dead_by_default_raises_loudly() -> None:
    """An unconfigured manager (no provider) raises a clear error on use —
    never a silent no-op."""
    manager = ResourceManager(StorageRegistry().provider)  # provider is None

    with pytest.raises(RuntimeError, match="no Storage provider registered"):
        await manager.fetch_template("greeting.j2")

    with pytest.raises(RuntimeError, match="no Storage provider registered"):
        await manager.list_resources()


async def test_render_by_id_with_registered_provider() -> None:
    """A registered in-memory provider serves templates the manager renders."""
    registry = StorageRegistry()

    @registry.register_storage
    class Provider(_InMemoryStorage):
        def __init__(self) -> None:
            super().__init__({"greeting.j2": "Hello {{ name }}!"})

    manager = ResourceManager(registry.provider)

    rendered = await manager.render_by_id("greeting.j2", {"name": "World"})
    assert rendered == "Hello World!"


async def test_render_by_content_needs_no_provider() -> None:
    """Inline content renders without touching storage, even unconfigured."""
    manager = ResourceManager(StorageRegistry().provider)

    rendered = await manager.render_by_id_or_content(content="Hi {{ who }}", kwargs={"who": "there"})
    assert rendered == "Hi there"


async def test_upload_then_render() -> None:
    """A template uploaded through the manager is rendered back by id."""
    registry = StorageRegistry()
    registry.register_storage(_InMemoryStorage)

    manager = ResourceManager(registry.provider)

    await manager.upload_template("msg.j2", "Count: {{ n }}")
    rendered = await manager.render_by_id("msg.j2", {"n": 3})
    assert rendered == "Count: 3"


# --- read-seam containment guard (empty-scheme branch of load) --------------


async def test_load_rejects_traversal_id() -> None:
    """The empty-scheme branch guards the bare storage id against a root escape
    BEFORE it reaches the provider — the single seam every unguarded read funnels
    through."""
    from tai42_skeleton.template.path_guard import UnsafeTemplatePathError

    registry = StorageRegistry()
    registry.register_storage(_InMemoryStorage)
    manager = ResourceManager(registry.provider)

    with pytest.raises(UnsafeTemplatePathError):
        await manager.load("../x")


async def test_load_accepts_clean_id() -> None:
    registry = StorageRegistry()

    @registry.register_storage
    class Provider(_InMemoryStorage):
        def __init__(self) -> None:
            super().__init__({"clean/name.j2": "hello"})

    manager = ResourceManager(registry.provider)
    data, _mime = await manager.load("clean/name.j2", with_mime=False)
    assert data == b"hello"


async def test_normalize_media_rejects_traversal_id() -> None:
    from tai42_skeleton.template.path_guard import UnsafeTemplatePathError

    registry = StorageRegistry()
    registry.register_storage(_InMemoryStorage)
    manager = ResourceManager(registry.provider)

    with pytest.raises(UnsafeTemplatePathError):
        await manager.normalize_media("../x")


# --- render/fetch/schema by-id containment guard (_load_by_id seam) ----------


async def test_render_and_fetch_by_id_reject_traversal() -> None:
    """Every by-id read — fetch/render/schema/undeclared-vars — funnels through
    ``_load_by_id``, which runs the id through ``safe_template_path`` BEFORE the
    provider. A ``../x`` id is refused with ``UnsafeTemplatePathError`` at this
    app-layer seam, not passed into storage (where it would surface as a
    ``FileNotFoundError``/``TemplateNotFoundError``)."""
    from tai42_skeleton.template.path_guard import UnsafeTemplatePathError

    registry = StorageRegistry()

    @registry.register_storage
    class Provider(_InMemoryStorage):
        def __init__(self) -> None:
            super().__init__({"greeting.j2": "Hello {{ name }}!"})

    manager = ResourceManager(registry.provider)

    with pytest.raises(UnsafeTemplatePathError):
        await manager.fetch_template("../x")
    with pytest.raises(UnsafeTemplatePathError):
        await manager.render_by_id("../x")
    with pytest.raises(UnsafeTemplatePathError):
        await manager._fetch_and_compile("../x")
    with pytest.raises(UnsafeTemplatePathError):
        await manager.get_template_schema(template_id="../x")
    with pytest.raises(UnsafeTemplatePathError):
        await manager.find_undeclared_variables(template_id="../x")

    # A clean id still fetches and renders through the same seam.
    assert await manager.fetch_template("greeting.j2") == "Hello {{ name }}!"
    assert await manager.render_by_id("greeting.j2", {"name": "World"}) == "Hello World!"
