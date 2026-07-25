"""Op-level oracles for the storage operations.

``list_resources`` reads the storage provider and returns ``{"resources": [...]}``
with the resource ids SORTED (a deterministic order). The mutating ops project with
``destructiveHint``; ``list_resources`` (a read) does not.
"""

from __future__ import annotations

import mimetypes
from types import SimpleNamespace

import pytest
from tai42_contract.manifest import ApiToolsConfig
from tai42_contract.storage import ObjectStat, Storage

from tai42_skeleton.app import instance
from tai42_skeleton.operations import NotSupportedError, OperationRegistry, operation_metadata_of
from tai42_skeleton.operations.projection import project_operations
from tai42_skeleton.operations.storage import (
    delete_resource,
    list_resources,
    storage_info,
    upload_resource,
)


class _FakeStorage(Storage):
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str | None]] = {}

    async def load(self, path: str) -> str:
        return self.objects[path][0].decode("utf-8")

    async def list(self) -> list[str]:
        return list(self.objects)

    async def upload(self, path: str, content: str) -> None:
        content_type, _ = mimetypes.guess_type(path)
        self.objects[path] = (content.encode("utf-8"), content_type)

    async def delete(self, path: str) -> None:
        if path not in self.objects:
            raise FileNotFoundError(path)
        del self.objects[path]

    async def delete_dir(self, path: str) -> None:
        raise FileNotFoundError(path)

    async def load_bytes(self, path: str) -> bytes:
        return self.objects[path][0]

    async def upload_bytes(self, path: str, data: bytes, content_type: str | None = None) -> None:
        self.objects[path] = (data, content_type)

    async def stat(self, path: str) -> ObjectStat:
        return ObjectStat(content_type=self.objects.get(path, (b"", None))[1])


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch):
    def _install(provider: Storage | None) -> Storage | None:
        monkeypatch.setattr(instance.app._storage_registry, "_provider", provider)
        return provider

    return _install


# -- list_resources ------------------


async def test_list_resources_returns_sorted_resources_object(install) -> None:
    provider = install(_FakeStorage())
    provider.objects = {"b.j2": (b"", None), "a.j2": (b"", None)}
    # Returns ``{"resources": [...]}`` with the ids sorted.
    assert await list_resources() == {"resources": ["a.j2", "b.j2"]}


async def test_list_resources_501_without_provider(install) -> None:
    install(None)
    with pytest.raises(NotSupportedError):
        await list_resources()


# -- op characterization -------------------------------------------------------


async def test_storage_info_absent(install) -> None:
    install(None)
    assert await storage_info() == {"present": False, "provider": None, "module": None}


async def test_upload_resource_requires_exactly_one_content(install) -> None:
    install(_FakeStorage())
    from tai42_skeleton.operations import BadRequestError

    with pytest.raises(BadRequestError, match="exactly one"):
        await upload_resource("a", content_text="t", content_base64="eA==")
    with pytest.raises(BadRequestError, match="exactly one"):
        await upload_resource("a")


async def test_upload_resource_rejects_traversal(install) -> None:
    install(_FakeStorage())
    from tai42_skeleton.operations import BadRequestError

    with pytest.raises(BadRequestError, match=r"\.\."):
        await upload_resource("a/../x", content_text="t")


async def test_upload_resource_rejects_non_string_content(install) -> None:
    # The HTTP extractor passes the raw body through; the op guards non-string
    # content that the MCP tool schema (str | None) would already reject.
    install(_FakeStorage())
    from tai42_skeleton.operations import BadRequestError

    with pytest.raises(BadRequestError, match="'content_text' must be a string"):
        await upload_resource("a", content_text=123)  # type: ignore[arg-type]
    with pytest.raises(BadRequestError, match="'content_base64' must be a base64 string"):
        await upload_resource("a", content_base64=123)  # type: ignore[arg-type]


async def test_delete_resource_missing_is_not_found(install) -> None:
    install(_FakeStorage())
    from tai42_skeleton.operations import NotFoundError

    with pytest.raises(NotFoundError):
        await delete_resource("nope.txt")


# -- projection: the mutating ops carry destructiveHint ------------------------


def test_projection_destructive_hints() -> None:
    # ``upload_resource`` declares ``destructive`` on the decorator; the DELETE ops
    # are destructive via the adapter's DELETE auto-force at route registration (their
    # ``x-destructive`` is pinned by the OpenAPI spec, not asserted here in isolation).
    reg = OperationRegistry()
    for op in (upload_resource, list_resources, storage_info):
        reg.register(operation_metadata_of(op))

    class _Rec:
        def __init__(self) -> None:
            self.registered: dict[str, dict] = {}

        def tool(self, *, force, name, tags, annotations):
            self.registered[name] = {"annotations": annotations}
            return lambda fn: fn

    app = SimpleNamespace(tools=_Rec())
    names = project_operations(app, ApiToolsConfig(expose_destructive=True), registry=reg)

    assert {"upload_resource", "list_resources", "storage_info"} == set(names)
    assert app.tools.registered["upload_resource"]["annotations"].destructiveHint is True
    assert app.tools.registered["list_resources"]["annotations"] is None  # read
    assert app.tools.registered["storage_info"]["annotations"] is None  # read
