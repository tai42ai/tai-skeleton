"""The storage router's auth boundary, pinned with access control ENABLED.

Every ``/api/storage*`` door reads or mutates the deployment's content store, so
all are AUTHED (no public door). Each asserts an unauthenticated request is denied
before the handler runs.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.storage as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/storage", router.storage_info, methods=["GET"]),
    Route("/api/storage/resources", router.list_resources, methods=["GET"]),
    Route("/api/storage/resources", router.upload_resource, methods=["POST"]),
    Route("/api/storage/resources/{resource_id:path}/stat", router.stat_resource, methods=["GET"]),
    Route("/api/storage/resources/{resource_id:path}/content", router.download_resource, methods=["GET"]),
    Route("/api/storage/resources/{resource_id:path}", router.delete_resource, methods=["DELETE"]),
    Route("/api/storage/dirs/{dir_path:path}", router.delete_dir, methods=["DELETE"]),
]
_STANCES = {
    r"/api/storage": AUTHED,
    r"/api/storage/resources": AUTHED,
    r"/api/storage/resources/[^/]+/stat": AUTHED,
    r"/api/storage/resources/[^/]+/content": AUTHED,
    r"/api/storage/resources/[^/]+": AUTHED,
    r"/api/storage/dirs/[^/]+": AUTHED,
}


def test_info_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/storage").status_code in (401, 403)


def test_list_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/storage/resources").status_code in (401, 403)


def test_upload_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/storage/resources", json={"id": "x", "content_text": "t"}).status_code in (401, 403)


def test_stat_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/storage/resources/x/stat").status_code in (401, 403)


def test_download_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/storage/resources/x/content").status_code in (401, 403)


def test_delete_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.delete("/api/storage/resources/x").status_code in (401, 403)


def test_delete_dir_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.delete("/api/storage/dirs/x").status_code in (401, 403)
