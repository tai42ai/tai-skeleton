"""The templates router's auth boundary, pinned with access control ENABLED.

Every ``/api/*template*`` door reads, writes, deletes, or renders stored template
content (arbitrary-file primitives), so all are AUTHED. Each asserts an
unauthenticated request is denied before the handler runs.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.templates as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/templates", router.list_templates, methods=["GET"]),
    Route("/api/template", router.get_template, methods=["POST"]),
    Route("/api/upload-template", router.upload_template, methods=["POST"]),
    Route("/api/delete-template", router.delete_template, methods=["POST"]),
    Route("/api/render-template", router.render_template, methods=["POST"]),
    Route("/api/clear-templates-cache", router.clear_templates_cache, methods=["POST"]),
]
_STANCES = {
    r"/api/templates": AUTHED,
    r"/api/template": AUTHED,
    r"/api/upload-template": AUTHED,
    r"/api/delete-template": AUTHED,
    r"/api/render-template": AUTHED,
    r"/api/clear-templates-cache": AUTHED,
}


def test_list_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/templates").status_code in (401, 403)


def test_get_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/template", json={"template_id": "a.j2"}).status_code in (401, 403)


def test_upload_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/upload-template", json={"path": "x.j2", "content": "y"}).status_code in (401, 403)


def test_delete_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/delete-template", json={"path": "x.j2"}).status_code in (401, 403)


def test_render_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/render-template", json={"content": "hi"}).status_code in (401, 403)


def test_clear_cache_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/clear-templates-cache").status_code in (401, 403)
