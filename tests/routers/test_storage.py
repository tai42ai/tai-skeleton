"""Storage router: identity + CRUD over a fake in-memory ``Storage`` provider,
plus the honest 501 when none is registered.

Handlers are driven directly (the router-test pattern). The provider is installed
by setting the process app's storage-registry provider to a stateful fake; the
absent case sets it to ``None`` so every door but ``GET /api/storage`` answers a
loud 501.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from types import SimpleNamespace
from typing import cast

import pytest
from starlette.requests import Request
from tai42_contract.storage import ObjectStat, Storage, assert_not_root

from tai42_skeleton.app import instance
from tai42_skeleton.routers import storage as router


class _FakeStorage(Storage):
    """An in-memory content store: ``id -> (bytes, content_type)``. Binary-native
    (overrides the byte/stat surface) so content type and downloads are exercised
    faithfully. ``delete``/``delete_dir`` raise ``FileNotFoundError`` for a missing
    target, pinning the router's 404 mapping."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str | None]] = {}

    async def load(self, path: str) -> str:
        return self._get(path)[0].decode("utf-8")

    async def list(self) -> list[str]:
        return list(self.objects)

    async def upload(self, path: str, content: str) -> None:
        content_type, _ = mimetypes.guess_type(path)
        self.objects[path] = (content.encode("utf-8"), content_type)

    async def delete(self, path: str) -> None:
        self._get(path)
        del self.objects[path]

    async def delete_dir(self, path: str) -> None:
        assert_not_root(path)
        prefix = path.rstrip("/") + "/"
        matched = [key for key in self.objects if key == path or key.startswith(prefix)]
        if not matched:
            raise FileNotFoundError(path)
        for key in matched:
            del self.objects[key]

    async def load_bytes(self, path: str) -> bytes:
        return self._get(path)[0]

    async def upload_bytes(self, path: str, data: bytes, content_type: str | None = None) -> None:
        self.objects[path] = (data, content_type)

    async def stat(self, path: str) -> ObjectStat:
        if path in self.objects:
            return ObjectStat(content_type=self.objects[path][1])
        content_type, _ = mimetypes.guess_type(path)
        return ObjectStat(content_type=content_type)

    def _get(self, path: str) -> tuple[bytes, str | None]:
        if path not in self.objects:
            raise FileNotFoundError(path)
        return self.objects[path]


@pytest.fixture
def install(monkeypatch):
    def _install(provider: Storage | None) -> Storage | None:
        monkeypatch.setattr(instance.app._storage_registry, "_provider", provider)
        return provider

    return _install


def _req(**path_params) -> Request:
    return cast(Request, SimpleNamespace(path_params=path_params))


def _body_req(body: bytes) -> Request:
    scope = {"type": "http", "method": "POST", "path": "/api/storage/resources", "headers": [], "query_string": b""}
    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _json(resp) -> dict:
    return json.loads(bytes(resp.body))


# -- identity ----------------------------------------------------------------


async def test_info_present(install):
    fake = install(_FakeStorage())
    resp = await router.storage_info(_req())
    assert resp.status_code == 200
    data = _json(resp)["data"]
    assert data["present"] is True
    assert data["provider"] == "_FakeStorage"
    assert data["module"] == type(fake).__module__


async def test_info_absent_is_200_present_false(install):
    install(None)
    resp = await router.storage_info(_req())
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"present": False, "provider": None, "module": None}}


# -- 501 when no provider ----------------------------------------------------


async def test_list_501_when_absent(install):
    install(None)
    resp = await router.list_resources(_req())
    assert resp.status_code == 501
    assert _json(resp) == {"error": "storage needs a storage-provider plugin"}


async def test_stat_501_when_absent(install):
    install(None)
    resp = await router.stat_resource(_req(resource_id="x"))
    assert resp.status_code == 501


async def test_download_501_when_absent(install):
    install(None)
    resp = await router.download_resource(_req(resource_id="x"))
    assert resp.status_code == 501


async def test_upload_501_when_absent(install):
    install(None)
    resp = await router.upload_resource(_body_req(b'{"id": "x", "content_text": "hi"}'))
    assert resp.status_code == 501


async def test_delete_501_when_absent(install):
    install(None)
    resp = await router.delete_resource(_req(resource_id="x"))
    assert resp.status_code == 501


async def test_delete_dir_501_when_absent(install):
    install(None)
    resp = await router.delete_dir(_req(dir_path="d"))
    assert resp.status_code == 501


# -- list / stat -------------------------------------------------------------


async def test_list_sorted(install):
    fake = cast(_FakeStorage, install(_FakeStorage()))
    fake.objects = {"z.txt": (b"", "text/plain"), "a/b.txt": (b"", "text/plain")}
    resp = await router.list_resources(_req())
    assert _json(resp) == {"data": {"resources": ["a/b.txt", "z.txt"]}}


async def test_stat_returns_content_type(install):
    fake = cast(_FakeStorage, install(_FakeStorage()))
    fake.objects = {"logo.png": (b"x", "image/png")}
    resp = await router.stat_resource(_req(resource_id="logo.png"))
    assert _json(resp) == {"data": {"id": "logo.png", "content_type": "image/png"}}


async def test_stat_content_type_null(install):
    fake = cast(_FakeStorage, install(_FakeStorage()))
    fake.objects = {"data.unknownext": (b"x", None)}
    resp = await router.stat_resource(_req(resource_id="data.unknownext"))
    assert _json(resp) == {"data": {"id": "data.unknownext", "content_type": None}}


# -- download ----------------------------------------------------------------


async def test_download_content_type_and_disposition(install):
    fake = cast(_FakeStorage, install(_FakeStorage()))
    fake.objects = {"docs/report.pdf": (b"%PDF-bytes", "application/pdf")}
    resp = await router.download_resource(_req(resource_id="docs/report.pdf"))
    assert resp.status_code == 200
    assert bytes(resp.body) == b"%PDF-bytes"
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.headers["content-disposition"] == 'attachment; filename="report.pdf"'


async def test_download_octet_stream_fallback(install):
    fake = cast(_FakeStorage, install(_FakeStorage()))
    fake.objects = {"blob": (b"\x00\x01", None)}
    resp = await router.download_resource(_req(resource_id="blob"))
    assert resp.headers["content-type"] == "application/octet-stream"


async def test_download_404_missing(install):
    install(_FakeStorage())
    resp = await router.download_resource(_req(resource_id="nope.txt"))
    assert resp.status_code == 404
    assert "nope.txt" in _json(resp)["error"]


# -- upload ------------------------------------------------------------------


async def test_upload_text(install):
    fake = cast(_FakeStorage, install(_FakeStorage()))
    resp = await router.upload_resource(_body_req(b'{"id": "notes/todo.txt", "content_text": "buy milk"}'))
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"id": "notes/todo.txt", "stored": True}}
    assert fake.objects["notes/todo.txt"][0] == b"buy milk"


async def test_upload_base64(install):
    fake = cast(_FakeStorage, install(_FakeStorage()))
    encoded = base64.b64encode(b"\x00\x01\x02").decode("ascii")
    resp = await router.upload_resource(_body_req(json.dumps({"id": "b.bin", "content_base64": encoded}).encode()))
    assert resp.status_code == 200
    assert fake.objects["b.bin"][0] == b"\x00\x01\x02"


async def test_upload_overwrite(install):
    fake = cast(_FakeStorage, install(_FakeStorage()))
    fake.objects = {"a.txt": (b"old", "text/plain")}
    await router.upload_resource(_body_req(b'{"id": "a.txt", "content_text": "new"}'))
    assert fake.objects["a.txt"][0] == b"new"


async def test_upload_requires_exactly_one_content_field(install):
    install(_FakeStorage())
    both = await router.upload_resource(_body_req(b'{"id": "a", "content_text": "t", "content_base64": "eA=="}'))
    assert both.status_code == 400
    both_error = _json(both)["error"]
    assert "content_text" in both_error
    assert "content_base64" in both_error
    neither = await router.upload_resource(_body_req(b'{"id": "a"}'))
    assert neither.status_code == 400


async def test_upload_invalid_base64_400(install):
    install(_FakeStorage())
    resp = await router.upload_resource(_body_req(b'{"id": "a", "content_base64": "not!base64!"}'))
    assert resp.status_code == 400
    assert "base64" in _json(resp)["error"]


async def test_upload_missing_id_400(install):
    install(_FakeStorage())
    resp = await router.upload_resource(_body_req(b'{"content_text": "t"}'))
    assert resp.status_code == 400
    assert "id" in _json(resp)["error"]


async def test_upload_invalid_json_400(install):
    install(_FakeStorage())
    resp = await router.upload_resource(_body_req(b"not json"))
    assert resp.status_code == 400
    assert "invalid JSON body" in _json(resp)["error"]


async def test_upload_non_object_body_400(install):
    install(_FakeStorage())
    resp = await router.upload_resource(_body_req(b'"a string"'))
    assert resp.status_code == 400
    assert "must be a JSON object" in _json(resp)["error"]


# -- delete ------------------------------------------------------------------


async def test_delete_resource(install):
    fake = cast(_FakeStorage, install(_FakeStorage()))
    fake.objects = {"a.txt": (b"x", None)}
    resp = await router.delete_resource(_req(resource_id="a.txt"))
    assert _json(resp) == {"data": {"id": "a.txt", "deleted": True}}
    assert "a.txt" not in fake.objects


async def test_delete_404_missing(install):
    install(_FakeStorage())
    resp = await router.delete_resource(_req(resource_id="nope.txt"))
    assert resp.status_code == 404


async def test_delete_dir(install):
    fake = cast(_FakeStorage, install(_FakeStorage()))
    fake.objects = {"d/a.txt": (b"x", None), "d/b.txt": (b"y", None), "other.txt": (b"z", None)}
    resp = await router.delete_dir(_req(dir_path="d"))
    assert _json(resp) == {"data": {"dir": "d", "deleted": True}}
    assert set(fake.objects) == {"other.txt"}


async def test_delete_dir_404_missing(install):
    install(_FakeStorage())
    resp = await router.delete_dir(_req(dir_path="nope"))
    assert resp.status_code == 404


async def test_delete_dir_root_is_400(install):
    install(_FakeStorage())
    resp = await router.delete_dir(_req(dir_path="."))
    assert resp.status_code == 400
    assert "root" in _json(resp)["error"].lower()


# -- traversal rejection (400 on all five id/path-carrying inputs) ------------


async def test_stat_rejects_dotdot(install):
    install(_FakeStorage())
    resp = await router.stat_resource(_req(resource_id="a/../secret"))
    assert resp.status_code == 400
    assert ".." in _json(resp)["error"]


async def test_download_rejects_dotdot(install):
    install(_FakeStorage())
    resp = await router.download_resource(_req(resource_id="a/../secret"))
    assert resp.status_code == 400


async def test_delete_rejects_dotdot(install):
    install(_FakeStorage())
    resp = await router.delete_resource(_req(resource_id="a/../secret"))
    assert resp.status_code == 400


async def test_delete_dir_rejects_dotdot(install):
    install(_FakeStorage())
    resp = await router.delete_dir(_req(dir_path="a/.."))
    assert resp.status_code == 400


async def test_upload_rejects_dotdot_id(install):
    install(_FakeStorage())
    resp = await router.upload_resource(_body_req(b'{"id": "a/../x", "content_text": "t"}'))
    assert resp.status_code == 400
    assert ".." in _json(resp)["error"]


# -- absolute-path rejection (400 on all five id/path-carrying inputs) ---------
#
# A leading ``/`` reaches the handler via the ``{...:path}`` converter capturing a
# leading slash from a ``//``-prefixed request, so an absolute path must be rejected
# as loudly as a ``..`` segment before it reaches the filesystem-backed provider.


async def test_stat_rejects_absolute(install):
    install(_FakeStorage())
    resp = await router.stat_resource(_req(resource_id="/etc/passwd"))
    assert resp.status_code == 400
    assert "relative path" in _json(resp)["error"]


async def test_download_rejects_absolute(install):
    install(_FakeStorage())
    resp = await router.download_resource(_req(resource_id="/etc/passwd"))
    assert resp.status_code == 400
    assert "relative path" in _json(resp)["error"]


async def test_delete_rejects_absolute(install):
    install(_FakeStorage())
    resp = await router.delete_resource(_req(resource_id="/etc/passwd"))
    assert resp.status_code == 400
    assert "relative path" in _json(resp)["error"]


async def test_delete_dir_rejects_absolute(install):
    install(_FakeStorage())
    resp = await router.delete_dir(_req(dir_path="/etc"))
    assert resp.status_code == 400
    assert "relative path" in _json(resp)["error"]


async def test_upload_rejects_absolute_id(install):
    install(_FakeStorage())
    resp = await router.upload_resource(_body_req(b'{"id": "/etc/passwd", "content_text": "t"}'))
    assert resp.status_code == 400
    assert "relative path" in _json(resp)["error"]


# -- provider boundary ValueError maps to 400 (never an uncaught 500) ----------


async def test_stat_provider_valueerror_is_400(install, monkeypatch):
    fake = cast(_FakeStorage, install(_FakeStorage()))

    async def boom(path: str):
        raise ValueError("path escapes the storage root")

    monkeypatch.setattr(fake, "stat", boom)
    resp = await router.stat_resource(_req(resource_id="ok.txt"))
    assert resp.status_code == 400
    assert "escapes" in _json(resp)["error"]


async def test_download_provider_valueerror_is_400(install, monkeypatch):
    fake = cast(_FakeStorage, install(_FakeStorage()))

    async def boom(path: str):
        raise ValueError("path escapes the storage root")

    monkeypatch.setattr(fake, "load_bytes", boom)
    resp = await router.download_resource(_req(resource_id="ok.txt"))
    assert resp.status_code == 400
    assert "escapes" in _json(resp)["error"]


async def test_delete_provider_valueerror_is_400(install, monkeypatch):
    fake = cast(_FakeStorage, install(_FakeStorage()))

    async def boom(path: str):
        raise ValueError("path escapes the storage root")

    monkeypatch.setattr(fake, "delete", boom)
    resp = await router.delete_resource(_req(resource_id="ok.txt"))
    assert resp.status_code == 400
    assert "escapes" in _json(resp)["error"]


async def test_upload_provider_valueerror_is_400(install, monkeypatch):
    fake = cast(_FakeStorage, install(_FakeStorage()))

    async def boom(path: str, data: bytes, content_type: str | None = None):
        raise ValueError("content is not storable by this provider")

    monkeypatch.setattr(fake, "upload_bytes", boom)
    encoded = base64.b64encode(b"\x00\x01\x02").decode("ascii")
    resp = await router.upload_resource(_body_req(json.dumps({"id": "b.bin", "content_base64": encoded}).encode()))
    assert resp.status_code == 400
    assert "not storable" in _json(resp)["error"]


# -- Content-Disposition filename is a well-formed quoted-string ---------------


async def test_download_disposition_escapes_quote_in_basename(install):
    fake = cast(_FakeStorage, install(_FakeStorage()))
    fake.objects = {'weird".txt': (b"x", "text/plain")}
    resp = await router.download_resource(_req(resource_id='weird".txt'))
    assert resp.status_code == 200
    # The literal ``"`` in the basename is backslash-escaped, so the quoted-string is
    # well-formed rather than prematurely terminated.
    assert resp.headers["content-disposition"] == 'attachment; filename="weird\\".txt"'
