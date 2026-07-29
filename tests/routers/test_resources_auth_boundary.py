"""The resources router's auth boundary, pinned with access control ENABLED.

``/api/resources/get`` reads arbitrary stored resource content (an
arbitrary-file-read primitive), so BOTH methods are AUTHED; an unauthenticated
request is denied before the handler runs. The GET fetch door is ``action="read"``
and the POST render door is ``action="write"``, so a ``resources`` READ grant opens
the GET but not the POST — the Layer-2 per-tag level gate below pins that.
"""

from __future__ import annotations

from starlette.routing import Route

import tai42_skeleton.routers.resources as router
from tai42_skeleton.access_control.role_gate import DenialCause, grant_map_admits, resolve_route_meta
from tests.routers._auth_boundary import AUTHED, boundary_client

_ROUTES = [
    Route("/api/resources/get", router.fetch_resource, methods=["GET"]),
    Route("/api/resources/get", router.get_resource_by_id, methods=["POST"]),
]
_STANCES = {
    r"/api/resources/get": AUTHED,
}


def test_post_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.post("/api/resources/get", json={"resource_id": "x"}).status_code in (401, 403)


def test_get_rejected_without_auth(monkeypatch):
    client = boundary_client(monkeypatch, _ROUTES, _STANCES)
    assert client.get("/api/resources/get", params={"resource_id": "x"}).status_code in (401, 403)


def test_read_grant_opens_get_but_not_post_render():
    # The GET fetch door and the POST render door on the same path carry distinct
    # action-classes, so the per-tag level gate opens each differently.
    get_meta = resolve_route_meta("/api/resources/get", "GET")
    post_meta = resolve_route_meta("/api/resources/get", "POST")
    assert get_meta is not None
    assert post_meta is not None
    assert get_meta.action == "read"
    assert post_meta.action == "write"

    read_grant = {"resources": "read"}
    # A resources READ grantee fetches via GET ...
    assert grant_map_admits(get_meta, "GET", read_grant) == (True, None)
    # ... but is denied the POST render (a write action the read level does not satisfy).
    allowed, cause = grant_map_admits(post_meta, "POST", read_grant)
    assert allowed is False
    assert cause is DenialCause.LEVEL_MISS


def test_write_grant_opens_both_doors():
    get_meta = resolve_route_meta("/api/resources/get", "GET")
    post_meta = resolve_route_meta("/api/resources/get", "POST")
    assert get_meta is not None
    assert post_meta is not None

    write_grant = {"resources": "write"}
    assert grant_map_admits(get_meta, "GET", write_grant) == (True, None)
    assert grant_map_admits(post_meta, "POST", write_grant) == (True, None)
