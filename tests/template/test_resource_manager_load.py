"""ResourceManager unified content faces: ``load`` (scheme-disambiguated resolver,
no silent fallback), ``load_file`` (text vs MediaBlock), and ``normalize_media``
(image-only ContentPart). Storage is an in-memory binary-native double."""

from __future__ import annotations

import base64

import pytest
from fastmcp.utilities.types import Image
from tai_contract.storage import ObjectStat, Storage

from tai_skeleton.storage import StorageRegistry
from tai_skeleton.template import ResourceManager
from tai_skeleton.template import resource_manager as rm_mod
from tai_skeleton.template.media import MediaBlock

# A 1x1 transparent PNG — real bytes so Magika classifies it as an image.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class _BinStorage(Storage):
    """A binary-native in-memory store (path -> bytes)."""

    def __init__(self, items: dict[str, bytes] | None = None) -> None:
        self.items: dict[str, bytes] = dict(items or {})

    async def load(self, path: str) -> str:
        try:
            return self.items[path].decode("utf-8")
        except KeyError as exc:
            raise FileNotFoundError(path) from exc

    async def list(self) -> list[str]:
        return sorted(self.items)

    async def upload(self, path: str, content: str) -> None:
        self.items[path] = content.encode("utf-8")

    async def delete(self, path: str) -> None:
        self.items.pop(path, None)

    async def delete_dir(self, path: str) -> None:
        prefix = path.rstrip("/") + "/"
        for key in [k for k in self.items if k.startswith(prefix)]:
            del self.items[key]

    async def load_bytes(self, path: str) -> bytes:
        try:
            return self.items[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc

    async def stat(self, path: str) -> ObjectStat:
        import mimetypes

        content_type, _ = mimetypes.guess_type(path)
        return ObjectStat(content_type=content_type)


def _manager(items: dict[str, bytes] | None = None) -> ResourceManager:
    registry = StorageRegistry()
    store = _BinStorage(items)
    registry.register_storage(lambda: store)  # type: ignore[arg-type]
    return ResourceManager(registry.provider)


# --- load: scheme disambiguation --------------------------------------------


async def test_load_http_routes_to_fetch_url(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(url: str) -> tuple[bytes, str | None]:
        assert url == "https://example.com/a.bin"
        return b"remote-bytes", "application/octet-stream"

    monkeypatch.setattr(rm_mod, "fetch_url", fake_fetch)
    manager = _manager()
    assert await manager.load("https://example.com/a.bin") == (b"remote-bytes", "application/octet-stream")


async def test_load_data_uri_is_decoded() -> None:
    manager = _manager()
    data, mime = await manager.load("data:text/plain;base64,aGVsbG8=")
    assert data == b"hello"
    assert mime == "text/plain"


async def test_load_storage_id_reads_bytes_and_mime() -> None:
    manager = _manager({"docs/note.txt": b"body"})
    data, mime = await manager.load("docs/note.txt")
    assert data == b"body"
    assert mime == "text/plain"


async def test_load_missing_storage_id_raises_filenotfound_not_swallowed() -> None:
    manager = _manager({"present.txt": b"x"})
    with pytest.raises(FileNotFoundError, match=r"absent\.txt"):
        await manager.load("absent.txt")


@pytest.mark.parametrize("source", ["file:///etc/passwd", "ftp://host/x", "weird/id://escape"])
async def test_load_rejects_ambiguous_source(source: str) -> None:
    manager = _manager()
    with pytest.raises(ValueError, match="Cannot resolve source"):
        await manager.load(source)


# --- load_file: text vs MediaBlock ------------------------------------------


async def test_load_file_returns_text_for_document() -> None:
    manager = _manager({"note.txt": b"the quick brown fox\n"})
    result = await manager.load_file("note.txt")
    assert result == "the quick brown fox\n"


async def test_load_file_returns_media_block_for_image() -> None:
    manager = _manager({"logo.png": _PNG})
    result = await manager.load_file("logo.png")
    assert isinstance(result, Image)


# --- normalize_media: image-only ContentPart --------------------------------


async def test_normalize_media_passes_through_public_image_url() -> None:
    manager = _manager()
    part = await manager.normalize_media("https://example.com/pic.png")
    assert part == {"type": "image_url", "image_url": {"url": "https://example.com/pic.png"}}


async def test_normalize_media_url_indeterminable_suffix_passes_through() -> None:
    manager = _manager()
    part = await manager.normalize_media("https://example.com/pic")
    assert part["image_url"]["url"] == "https://example.com/pic"


async def test_normalize_media_non_image_url_raises() -> None:
    manager = _manager()
    with pytest.raises(ValueError, match="requires an image"):
        await manager.normalize_media("https://example.com/notes.txt")


async def test_normalize_media_passes_through_image_data_uri() -> None:
    manager = _manager()
    uri = "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii")
    part = await manager.normalize_media(uri)
    assert part["image_url"]["url"] == uri


async def test_normalize_media_non_image_data_uri_raises() -> None:
    manager = _manager()
    with pytest.raises(ValueError, match="requires an image"):
        await manager.normalize_media("data:text/plain;base64,aGk=")


async def test_normalize_media_storage_id_emits_base64_data_uri() -> None:
    manager = _manager({"logo.png": _PNG})
    part = await manager.normalize_media("logo.png")
    expected = "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii")
    assert part["image_url"]["url"] == expected


async def test_normalize_media_non_image_storage_id_raises() -> None:
    manager = _manager({"notes.txt": b"hi"})
    with pytest.raises(ValueError, match="requires an image"):
        await manager.normalize_media("notes.txt")


async def test_normalize_media_raw_bytes_emits_base64_data_uri() -> None:
    manager = _manager()
    part = await manager.normalize_media(_PNG)
    url: str = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == _PNG


async def test_normalize_media_non_image_bytes_raises() -> None:
    manager = _manager()
    with pytest.raises(ValueError, match="requires an image"):
        await manager.normalize_media(b"just some plain text bytes, not an image")


def test_media_block_alias_covers_fastmcp_union() -> None:
    # ``MediaBlock`` is the fastmcp media union used for isinstance narrowing.
    assert isinstance(Image(data=_PNG, format="png"), MediaBlock)
