"""Extended ResourceManager coverage: cache-enabled vs disabled construction,
the cache controls, the dispatch/guard branches of the render + schema + var
helpers, the {% include %} sync-loader bridge, and the cache-clearing delete
paths. All storage is an in-memory ``Storage`` double; no real backend."""

from __future__ import annotations

import logging
from typing import Protocol, cast

import pytest
from jinja2.exceptions import SecurityError
from tai42_contract.storage import Storage

from tai42_skeleton.storage import StorageRegistry
from tai42_skeleton.template import ResourceManager, TemplateNotFoundError
from tai42_skeleton.template import resource_manager as rm_mod
from tai42_skeleton.template.settings import TemplateCacheSettings


class _SupportsCacheInvalidate(Protocol):
    def cache_invalidate(self, *args: object) -> bool: ...


class _SupportsCacheContains(Protocol):
    def cache_contains(self, *args: object) -> bool: ...


class _InMemoryStorage(Storage):
    def __init__(self, items: dict[str, str] | None = None) -> None:
        self.items: dict[str, str] = dict(items or {})
        self.deleted: list[str] = []
        self.deleted_dirs: list[str] = []

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
        self.deleted.append(path)
        self.items.pop(path, None)

    async def delete_dir(self, path: str) -> None:
        self.deleted_dirs.append(path)
        prefix = path.rstrip("/") + "/"
        for key in [k for k in self.items if k.startswith(prefix)]:
            del self.items[key]


def _manager(items: dict[str, str] | None = None) -> tuple[ResourceManager, _InMemoryStorage]:
    registry = StorageRegistry()
    store = _InMemoryStorage(items)
    registry.register_storage(lambda: store)  # type: ignore[arg-type]
    return ResourceManager(registry.provider), store


# --- construction: cache enabled vs disabled --------------------------------


def test_cache_enabled_exposes_cache_info_and_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (positive ttl + size) wraps fetch in an alru cache: info reports
    hits/misses and clear empties it."""
    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: TemplateCacheSettings(ttl=300, max_size=8))
    manager, _ = _manager({"a.j2": "{{ x }}"})

    assert manager.get_cache_info() is not None

    async def run() -> None:
        await manager.render_by_id("a.j2", {"x": 1})  # miss -> populate
        await manager.render_by_id("a.j2", {"x": 2})  # hit
        info = manager.get_cache_info()
        assert info is not None
        assert info.hits == 1
        assert info.misses == 1
        manager.clear_cache()
        cleared = manager.get_cache_info()
        assert cleared is not None
        assert cleared.currsize == 0

    import asyncio

    asyncio.run(run())


async def test_inline_template_is_compiled_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inline (content) template is compiled once and memoized by content, so
    a hook condition/expr rendered on every fire is not re-parsed each time."""
    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: TemplateCacheSettings(ttl=300, max_size=8))
    manager, _ = _manager()

    first = manager._compile_inline("Hi {{ x }}")
    second = manager._compile_inline("Hi {{ x }}")
    assert first is second  # same content -> one compiled template, reused

    # Rendering still works through the memoized inline template.
    assert await manager.render_by_id_or_content(content="Hi {{ x }}", kwargs={"x": 1}) == "Hi 1"


@pytest.mark.parametrize(
    "settings",
    [
        TemplateCacheSettings(ttl=0, max_size=8),
        TemplateCacheSettings(ttl=300, max_size=0),
    ],
)
def test_cache_disabled_has_no_cache_controls(monkeypatch: pytest.MonkeyPatch, settings: TemplateCacheSettings) -> None:
    """ttl=0 or max_size=0 bypasses the cache wrapper entirely, so the cache
    controls are inert (no info, clear is a no-op) but rendering still works."""
    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: settings)
    manager, _ = _manager({"a.j2": "{{ x }}"})

    assert manager.get_cache_info() is None
    manager.clear_cache()  # no-op, must not raise

    import asyncio

    assert asyncio.run(manager.render_by_id("a.j2", {"x": 5})) == "5"


# --- sandboxed engine (SSTI is blocked) -------------------------------------


async def test_sandbox_blocks_dunder_traversal_chain() -> None:
    """The shared engine is a ``SandboxedEnvironment``: a full dunder traversal +
    call chain (the classic SSTI-to-RCE reach for subprocess primitives) raises
    ``SecurityError`` at render, not host code execution."""
    manager, _ = _manager()
    with pytest.raises(SecurityError):
        await manager.render_by_id_or_content(content="{{ ''.__class__.__mro__[1].__subclasses__() }}")


async def test_sandbox_leaves_normal_rendering_intact() -> None:
    """Ordinary variable/filter rendering is unaffected by the sandbox."""
    manager, _ = _manager()
    assert await manager.render_by_id_or_content(content="{{ name | upper }}", kwargs={"name": "ada"}) == "ADA"


# --- render_by_id_or_content dispatch + guards ------------------------------


async def test_render_by_id_or_content_rejects_both() -> None:
    manager, _ = _manager()
    with pytest.raises(ValueError, match="not both"):
        await manager.render_by_id_or_content(content="x", template_id="y")


async def test_render_by_id_or_content_rejects_both_with_empty_content() -> None:
    """The mutual-exclusion guard uses ``content is not None``: an empty inline
    ``content`` alongside a ``template_id`` still trips 'not both' (it is not
    treated as 'no content')."""
    manager, _ = _manager()
    with pytest.raises(ValueError, match="not both"):
        await manager.render_by_id_or_content(content="", template_id="x")


async def test_render_by_id_or_content_empty_content_renders_blank() -> None:
    """An explicit empty inline template renders as '' (content is honoured, not
    skipped as falsy)."""
    manager, _ = _manager()
    assert await manager.render_by_id_or_content(content="") == ""


async def test_render_by_id_or_content_via_template_id() -> None:
    manager, _ = _manager({"g.j2": "Hi {{ name }}"})
    assert await manager.render_by_id_or_content(template_id="g.j2", kwargs={"name": "Z"}) == "Hi Z"


async def test_render_by_id_or_content_empty_allowed_returns_blank() -> None:
    manager, _ = _manager()
    assert await manager.render_by_id_or_content() == ""


async def test_render_by_id_or_content_empty_disallowed_raises() -> None:
    manager, _ = _manager()
    with pytest.raises(ValueError, match="either a template or a template_id"):
        await manager.render_by_id_or_content(allow_empty=False)


# --- fetch + compile error path ---------------------------------------------


async def test_fetch_template_returns_content() -> None:
    manager, _ = _manager({"t.j2": "body"})
    assert await manager.fetch_template("t.j2") == "body"


async def test_fetch_and_compile_empty_content_renders_empty() -> None:
    """A present-but-empty stored template is valid: it renders to empty, never
    mistaken for a missing template."""
    manager, _ = _manager({"empty.j2": ""})
    assert await manager.render_by_id("empty.j2") == ""


async def test_fetch_and_compile_missing_template_raises() -> None:
    """A genuinely missing template (storage raises FileNotFoundError) surfaces
    as a typed ``TemplateNotFoundError`` (mapped to 404 by the HTTP surface)."""
    manager, _ = _manager({"present.j2": "x"})
    with pytest.raises(TemplateNotFoundError, match="not found"):
        await manager.render_by_id("absent.j2")


# --- {% include %} sync-loader bridge ---------------------------------------


async def test_include_resolves_via_sync_loader() -> None:
    manager, _ = _manager({"child.j2": "CHILD", "parent.j2": 'P[{% include "child.j2" %}]'})
    assert await manager.render_by_id("parent.j2") == "P[CHILD]"


async def test_include_client_cleanup_failure_is_logged_not_fatal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A pooled-client cleanup failure after a template load is logged loudly but
    must not replace the load's result — the include still renders."""

    async def boom() -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(rm_mod, "shutdown_all_clients", boom)
    manager, _ = _manager({"child.j2": "CHILD", "parent.j2": 'P[{% include "child.j2" %}]'})
    with caplog.at_level(logging.ERROR):
        assert await manager.render_by_id("parent.j2") == "P[CHILD]"
    # The cleanup failure is logged loudly, not silently swallowed.
    assert "Error closing pooled clients" in caplog.text


async def test_include_missing_with_ignore_missing_renders_blank() -> None:
    """A genuinely missing include (FileNotFoundError -> None) keeps
    ``ignore missing`` working rather than masking other failures."""
    manager, _ = _manager({"m.j2": 'X[{% include "nope.j2" ignore missing %}]'})
    assert await manager.render_by_id("m.j2") == "X[]"


# --- schema inference -------------------------------------------------------


async def test_get_template_schema_from_content() -> None:
    manager, _ = _manager()
    schema = await manager.get_template_schema(content="{{ a }}")
    assert schema["type"] == "object"
    assert "a" in schema["properties"]


async def test_get_template_schema_from_template_id() -> None:
    manager, _ = _manager({"s.j2": "{{ a }}"})
    schema = await manager.get_template_schema(template_id="s.j2")
    assert "a" in schema["properties"]


async def test_get_template_schema_rejects_both() -> None:
    manager, _ = _manager()
    with pytest.raises(ValueError, match="not both"):
        await manager.get_template_schema(content="x", template_id="y")


async def test_get_template_schema_rejects_neither() -> None:
    manager, _ = _manager()
    with pytest.raises(ValueError, match="either a template or a template_id"):
        await manager.get_template_schema()


async def test_get_template_schema_empty_template_is_empty_schema() -> None:
    """A present-but-empty template infers an empty (variable-less) schema rather
    than raising not-found."""
    manager, _ = _manager({"s.j2": ""})
    schema = await manager.get_template_schema(template_id="s.j2")
    assert schema.get("properties", {}) == {}


async def test_get_template_schema_missing_template_id_raises() -> None:
    manager, _ = _manager({"present.j2": "{{ a }}"})
    with pytest.raises(TemplateNotFoundError, match="not found"):
        await manager.get_template_schema(template_id="absent.j2")


# --- undeclared variables ---------------------------------------------------


async def test_find_undeclared_variables_from_content() -> None:
    manager, _ = _manager()
    assert await manager.find_undeclared_variables(content="{{ a }}{{ b }}") == {"a", "b"}


async def test_find_undeclared_variables_from_template_id() -> None:
    manager, _ = _manager({"v.j2": "{{ a }}"})
    assert await manager.find_undeclared_variables(template_id="v.j2") == {"a"}


async def test_find_undeclared_variables_rejects_both() -> None:
    manager, _ = _manager()
    with pytest.raises(ValueError, match="not both"):
        await manager.find_undeclared_variables(content="x", template_id="y")


async def test_find_undeclared_variables_rejects_neither() -> None:
    manager, _ = _manager()
    with pytest.raises(ValueError, match="either a template or a template_id"):
        await manager.find_undeclared_variables()


async def test_find_undeclared_variables_empty_template_is_empty_set() -> None:
    """A present-but-empty template has no undeclared variables rather than
    raising not-found."""
    manager, _ = _manager({"v.j2": ""})
    assert await manager.find_undeclared_variables(template_id="v.j2") == set()


async def test_find_undeclared_variables_missing_template_id_raises() -> None:
    manager, _ = _manager({"present.j2": "{{ a }}"})
    with pytest.raises(TemplateNotFoundError, match="not found"):
        await manager.find_undeclared_variables(template_id="absent.j2")


# --- upload invalidates the compiled-template cache -------------------------


async def test_upload_template_evicts_stale_compiled_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-uploading to the same id must not keep serving the old compiled body:
    the single key is evicted on upload, so the next render reflects new content
    (cache enabled)."""
    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: TemplateCacheSettings(ttl=300, max_size=8))
    manager, _ = _manager({"m.j2": "v1 {{ x }}"})
    assert await manager.render_by_id("m.j2", {"x": 1}) == "v1 1"  # populate cache

    await manager.upload_template("m.j2", "v2 {{ x }}")
    # Uploading evicts the compiled entry, so the next render reflects v2.
    assert await manager.render_by_id("m.j2", {"x": 1}) == "v2 1"


# --- delete invalidates the compiled-template cache (targeted) --------------


async def test_delete_template_delegates_and_evicts_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: TemplateCacheSettings(ttl=300, max_size=8))
    manager, store = _manager({"d.j2": "{{ x }}"})
    await manager.render_by_id("d.j2", {"x": 1})  # populate cache
    populated = manager.get_cache_info()
    assert populated is not None
    assert populated.currsize == 1

    await manager.delete_template("d.j2")
    assert store.deleted == ["d.j2"]
    evicted = manager.get_cache_info()
    assert evicted is not None
    assert evicted.currsize == 0


async def test_delete_template_evicts_only_deleted_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting one template evicts only that compiled entry; others stay cached
    (no whole-cache flush)."""
    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: TemplateCacheSettings(ttl=300, max_size=8))
    manager, _ = _manager({"a.j2": "{{ x }}", "b.j2": "{{ y }}"})
    await manager.render_by_id("a.j2", {"x": 1})
    await manager.render_by_id("b.j2", {"y": 2})
    both = manager.get_cache_info()
    assert both is not None
    assert both.currsize == 2

    await manager.delete_template("a.j2")
    survivor = manager.get_cache_info()
    assert survivor is not None
    assert survivor.currsize == 1
    # The surviving entry is served from cache (no re-fetch needed).
    assert await manager.render_by_id("b.j2", {"y": 9}) == "9"


async def test_delete_template_dir_evicts_prefix_even_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dir delete is non-atomic: entries under the prefix must be evicted in the
    ``finally`` even when the provider's delete_dir raises."""
    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: TemplateCacheSettings(ttl=300, max_size=8))
    manager, store = _manager({"dir/a.j2": "{{ x }}"})
    await manager.render_by_id("dir/a.j2", {"x": 1})
    populated = manager.get_cache_info()
    assert populated is not None
    assert populated.currsize == 1

    async def boom(path: str) -> None:
        raise RuntimeError("partial delete")

    monkeypatch.setattr(store, "delete_dir", boom)
    with pytest.raises(RuntimeError, match="partial delete"):
        await manager.delete_template_dir("dir")
    evicted = manager.get_cache_info()
    assert evicted is not None
    assert evicted.currsize == 0


async def test_delete_template_dir_evicts_only_under_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dir delete evicts only compiled templates under that prefix; templates
    outside the dir stay cached."""
    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: TemplateCacheSettings(ttl=300, max_size=8))
    manager, _ = _manager({"dir/a.j2": "{{ x }}", "other.j2": "{{ y }}"})
    await manager.render_by_id("dir/a.j2", {"x": 1})
    await manager.render_by_id("other.j2", {"y": 2})
    both = manager.get_cache_info()
    assert both is not None
    assert both.currsize == 2

    await manager.delete_template_dir("dir")
    survivor = manager.get_cache_info()
    assert survivor is not None
    assert survivor.currsize == 1
    assert await manager.render_by_id("other.j2", {"y": 7}) == "7"


async def test_delete_template_dir_success_delegates() -> None:
    manager, store = _manager({"dir/a.j2": "A", "dir/b.j2": "B"})
    await manager.delete_template_dir("dir")
    assert store.deleted_dirs == ["dir"]
    assert store.items == {}


async def test_delete_template_dir_with_caching_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """With caching off there is no compiled-template cache to invalidate; the
    dir delete still delegates and must not raise."""
    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: TemplateCacheSettings(ttl=0, max_size=8))
    manager, store = _manager({"dir/a.j2": "A"})
    assert manager.get_cache_info() is None  # caching disabled

    await manager.delete_template_dir("dir")
    assert store.deleted_dirs == ["dir"]
    assert store.items == {}


# --- tracking set pruned against the cache's own evictions ------------------


async def test_cached_ids_pruned_against_lru_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under a bounded ``max_size`` the compile-time sweep drops tracking-set ids
    the LRU has evicted, so the set never outgrows the live cache."""
    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: TemplateCacheSettings(ttl=300, max_size=2))
    manager, _ = _manager({"a.j2": "{{ x }}", "b.j2": "{{ y }}", "c.j2": "{{ z }}"})

    await manager.render_by_id("a.j2", {"x": 1})
    await manager.render_by_id("b.j2", {"y": 1})
    await manager.render_by_id("c.j2", {"z": 1})  # evicts a.j2 (LRU), sweeps the set

    ids = manager._cached_template_ids
    assert len(ids) <= 2
    contains = cast(_SupportsCacheContains, manager._get_compiled_template).cache_contains
    # Every tracked id is still genuinely in the cache — no stale ids linger.
    assert all(contains(tid) for tid in ids)


async def test_cached_ids_pruned_against_ttl_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``max_size=None`` + a ttl, ids the cache expires on its own are still
    swept from the tracking set on the next compile (the unbounded+ttl mode)."""
    import asyncio

    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: TemplateCacheSettings(ttl=1, max_size=None))
    manager, _ = _manager({"a.j2": "{{ x }}", "b.j2": "{{ y }}", "c.j2": "{{ z }}"})

    await manager.render_by_id("a.j2", {"x": 1})
    await manager.render_by_id("b.j2", {"y": 1})
    assert manager._cached_template_ids == {"a.j2", "b.j2"}

    # Let the ttl call_later fire (evicting a.j2/b.j2 from the underlying cache),
    # then a fresh compile runs the amortized sweep.
    await asyncio.sleep(1.3)
    await manager.render_by_id("c.j2", {"z": 1})

    ids = manager._cached_template_ids
    assert "a.j2" not in ids
    assert "b.j2" not in ids
    assert "c.j2" in ids
    contains = cast(_SupportsCacheContains, manager._get_compiled_template).cache_contains
    assert all(contains(tid) for tid in ids)


async def test_delete_template_dir_prunes_ids_already_evicted_by_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The id registry stays bounded by what is actually cached: an entry the
    cache dropped on its own (simulating TTL/LRU) is pruned during a dir delete
    even though it lies outside the deleted prefix."""
    monkeypatch.setattr(rm_mod, "template_cache_settings", lambda: TemplateCacheSettings(ttl=300, max_size=8))
    manager, _ = _manager({"keep.j2": "{{ x }}", "gone.j2": "{{ y }}"})
    await manager.render_by_id("keep.j2", {"x": 1})
    await manager.render_by_id("gone.j2", {"y": 2})
    assert manager._cached_template_ids == {"keep.j2", "gone.j2"}

    # Drop "gone.j2" straight from the underlying cache (as TTL/LRU would),
    # leaving it stale in the tracking registry.
    # With caching on, ``_get_compiled_template`` is the alru cache wrapper, which
    # exposes ``cache_invalidate``; its static type is a union with the bare
    # (uncached) method, so narrow to the cache-bearing shape here.
    cached = cast(_SupportsCacheInvalidate, manager._get_compiled_template)
    assert cached.cache_invalidate("gone.j2") is True
    assert manager._cached_template_ids == {"keep.j2", "gone.j2"}

    # A dir delete that matches neither id still prunes the stale "gone.j2".
    await manager.delete_template_dir("unrelated")
    assert manager._cached_template_ids == {"keep.j2"}
