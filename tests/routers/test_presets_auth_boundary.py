"""The presets auth boundary, pinned with access control ENABLED.

Every ``/api/presets*`` door reads or mutates presets whose ``fixed_kwargs`` can
carry credentials, so all are AUTHED (no public door on this surface). Each
asserts an unauthenticated request is denied before the handler runs, mirroring
the tools boundary test's shared harness.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.presets as router
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/presets", router.list_presets, methods=["GET"]),
    Route("/api/presets", router.create_preset, methods=["POST"]),
    Route("/api/presets/{name}", router.get_preset, methods=["GET"]),
    Route("/api/presets/{name}", router.delete_preset, methods=["DELETE"]),
    Route("/api/presets/{name}/versions", router.list_versions, methods=["GET"]),
    Route("/api/presets/{name}/versions", router.save_version, methods=["POST"]),
    Route("/api/presets/{name}/versions/{version}", router.get_version, methods=["GET"]),
    Route("/api/presets/{name}/rollback", router.rollback_preset, methods=["POST"]),
    Route("/api/presets/{name}/rename", router.rename_preset, methods=["POST"]),
    Route("/api/presets/validate", router.validate_preset, methods=["POST"]),
    Route("/api/presets/{name}/referees", router.preset_referees, methods=["GET"]),
    Route("/api/presets/{name}/versions/{version}/tags", router.set_preset_version_tags, methods=["PUT"]),
]
_STANCES = {
    r"/api/presets": AUTHED,
    r"/api/presets/validate": AUTHED,
    r"/api/presets/[^/]+": AUTHED,
    r"/api/presets/[^/]+/referees": AUTHED,
    r"/api/presets/[^/]+/versions": AUTHED,
    r"/api/presets/[^/]+/versions/[^/]+": AUTHED,
    r"/api/presets/[^/]+/versions/[^/]+/tags": AUTHED,
    r"/api/presets/[^/]+/rollback": AUTHED,
    r"/api/presets/[^/]+/rename": AUTHED,
}


def test_list_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/presets").status_code in (401, 403)


def test_create_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/presets", json={"name": "p"}).status_code in (401, 403)


def test_get_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/presets/p").status_code in (401, 403)


def test_delete_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.delete("/api/presets/p").status_code in (401, 403)


def test_list_versions_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/presets/p/versions").status_code in (401, 403)


def test_save_version_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/presets/p/versions", json={"tags": []}).status_code in (401, 403)


def test_get_version_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/presets/p/versions/1").status_code in (401, 403)


def test_rollback_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/presets/p/rollback", json={"version": 1}).status_code in (401, 403)


def test_rename_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/presets/p/rename", json={"new_name": "q"}).status_code in (401, 403)


def test_validate_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/presets/validate", json={"name": "p"}).status_code in (401, 403)


def test_referees_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/presets/p/referees").status_code in (401, 403)


def test_set_version_tags_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.put("/api/presets/p/versions/1/tags", json={"tags": []}).status_code in (401, 403)
