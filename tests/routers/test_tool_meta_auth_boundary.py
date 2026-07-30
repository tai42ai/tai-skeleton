"""The tool-metadata auth boundary, pinned with access control ENABLED.

Every ``/api/tool-meta*`` door reads or mutates the organizational overlay, so all
are AUTHED — there is no public door on this surface. Each asserts an
unauthenticated request is denied before the handler runs, mirroring the presets
boundary test's shared harness.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.tool_meta as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/tool-meta", router.list_tool_meta, methods=["GET"]),
    Route("/api/tool-meta/tools/{tool_name}", router.upsert_tool_meta, methods=["PATCH"]),
    Route("/api/tool-meta/tools/{tool_name}", router.delete_tool_meta, methods=["DELETE"]),
    Route("/api/tool-meta/folders", router.create_folder, methods=["POST"]),
    Route("/api/tool-meta/folders/{folder_id}/rename", router.rename_folder, methods=["POST"]),
    Route("/api/tool-meta/folders/{folder_id}/move", router.move_folder, methods=["POST"]),
    Route("/api/tool-meta/folders/{folder_id}", router.delete_folder, methods=["DELETE"]),
]
_STANCES = {
    r"/api/tool-meta": AUTHED,
    r"/api/tool-meta/tools/[^/]+": AUTHED,
    r"/api/tool-meta/folders": AUTHED,
    r"/api/tool-meta/folders/[^/]+": AUTHED,
    r"/api/tool-meta/folders/[^/]+/rename": AUTHED,
    r"/api/tool-meta/folders/[^/]+/move": AUTHED,
}


def test_list_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/tool-meta").status_code in (401, 403)


def test_upsert_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.patch("/api/tool-meta/tools/weather", json={"display_name": "W"}).status_code in (401, 403)


def test_delete_tool_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.delete("/api/tool-meta/tools/weather").status_code in (401, 403)


def test_create_folder_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/tool-meta/folders", json={"name": "root"}).status_code in (401, 403)


def test_rename_folder_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/tool-meta/folders/f1/rename", json={"name": "x"}).status_code in (401, 403)


def test_move_folder_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/tool-meta/folders/f1/move", json={"parent_id": None}).status_code in (401, 403)


def test_delete_folder_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.delete("/api/tool-meta/folders/f1").status_code in (401, 403)
