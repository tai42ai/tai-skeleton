"""Templates router: list/get/upload/delete/render/clear-cache + the write-path
traversal guard."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest
from jinja2 import TemplateSyntaxError
from starlette.requests import Request

import tai42_skeleton.operations.templates as templates_ops
import tai42_skeleton.routers.templates as router
import tai42_skeleton.template.path_guard as path_guard
from tai42_skeleton.template import TemplateNotFoundError
from tai42_skeleton.template.path_guard import UnsafeTemplatePathError


def _req(body=None) -> Request:
    async def _json():
        if isinstance(body, Exception):
            raise body
        return body

    return cast(Request, SimpleNamespace(json=_json, path_params={}, query_params={}))


def _data(resp):
    return json.loads(bytes(resp.body))


class _FakeResourceManager:
    def __init__(self):
        self.uploaded = {}
        self.deleted = []
        self.cleared = False

    async def list_resources(self):
        return ["a.j2", "b.j2"]

    async def fetch_template(self, template_id):
        return f"content of {template_id}"

    async def get_template_schema(self, content=None, template_id=None):
        return {"vars": ["name"]}

    async def upload_template(self, path, content):
        self.uploaded[path] = content

    async def delete_template(self, path):
        self.deleted.append(path)

    async def render_by_id_or_content(self, content=None, template_id=None, kwargs=None):
        return f"rendered:{template_id or content}:{kwargs}"

    def clear_cache(self):
        self.cleared = True


@pytest.fixture
def manager(monkeypatch):
    tm = _FakeResourceManager()
    fake_app = SimpleNamespace(storage=SimpleNamespace(resource_manager=tm))
    # The handlers are thin adapters over operations.templates; the business logic
    # reads ``tai42_app`` from the OP module, so the fake is installed there (the
    # same module-local ``tai42_app`` symbol the handlers resolve at call time).
    monkeypatch.setattr(templates_ops, "tai42_app", fake_app)
    return tm


async def test_list(manager):
    resp = await router.list_templates(_req())
    assert _data(resp)["data"] == ["a.j2", "b.j2"]


async def test_get(manager):
    resp = await router.get_template(_req({"template_id": "a.j2"}))
    body = _data(resp)["data"]
    assert body["template"] == "content of a.j2"
    assert body["schema"] == {"vars": ["name"]}


async def test_get_missing_id_400(manager):
    resp = await router.get_template(_req({}))
    assert resp.status_code == 400


async def test_get_invalid_json_400(manager):
    resp = await router.get_template(_req(ValueError("bad json")))
    assert resp.status_code == 400
    assert "invalid JSON body" in _data(resp)["error"]


async def test_get_non_object_body_400(manager):
    resp = await router.get_template(_req("a string"))
    assert resp.status_code == 400
    assert "must be a JSON object" in _data(resp)["error"]


async def test_upload(manager):
    resp = await router.upload_template(_req({"path": "dir/x.j2", "content": "hi"}))
    assert resp.status_code == 200
    assert manager.uploaded == {"dir/x.j2": "hi"}


@pytest.mark.parametrize("bad", ["/abs.j2", "../escape.j2", "a/../../etc", "back\\slash"])
async def test_upload_traversal_rejected(manager, bad):
    resp = await router.upload_template(_req({"path": bad, "content": "x"}))
    assert resp.status_code == 400
    assert manager.uploaded == {}


async def test_delete_traversal_rejected(manager):
    resp = await router.delete_template(_req({"path": "../secret"}))
    assert resp.status_code == 400
    assert manager.deleted == []


async def test_delete(manager):
    resp = await router.delete_template(_req({"path": "x.j2"}))
    assert resp.status_code == 200
    assert manager.deleted == ["x.j2"]


async def test_delete_absent_path_is_idempotent_200(manager, monkeypatch):
    # Deleting a path that is already absent is idempotent: the store treats a
    # missing template as a no-op (it does NOT raise), so the route returns 200,
    # not 404.
    seen: list[str] = []

    async def _noop_delete(path):
        # A store that is idempotent for an absent path: succeeds without raising.
        seen.append(path)

    monkeypatch.setattr(manager, "delete_template", _noop_delete)
    resp = await router.delete_template(_req({"path": "never-existed.j2"}))
    assert resp.status_code == 200
    assert _data(resp)["data"] == {"path": "never-existed.j2", "deleted": True}
    assert seen == ["never-existed.j2"]


@pytest.mark.parametrize("bad", ["/abs.j2", "../escape.j2", "a/../../etc", "back\\slash"])
async def test_get_template_id_traversal_rejected(manager, bad):
    resp = await router.get_template(_req({"template_id": bad}))
    assert resp.status_code == 400


@pytest.mark.parametrize("bad", ["/abs.j2", "../escape.j2", "a/../../etc", "back\\slash"])
async def test_render_template_id_traversal_rejected(manager, bad):
    resp = await router.render_template(_req({"template_id": bad}))
    assert resp.status_code == 400


async def test_render_legit_template_id_passes(manager):
    resp = await router.render_template(_req({"template_id": "dir/ok.j2"}))
    assert resp.status_code == 200


def test_safe_path_symlink_escape_rejected(monkeypatch, tmp_path):
    """With ``_TEMPLATE_ROOT`` monkeypatched to a REAL directory (as a
    filesystem-backed store's own root would be), a key through a component that
    symlinks OUTSIDE the root resolves out via realpath and is refused. This
    exercises realpath's symlink handling against a real root — production uses a
    virtual (non-existent) anchor, so its guard is lexical only and this branch
    never fires there; the store layer owns symlink defense for a real root."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.j2").write_text("secret", encoding="utf-8")
    (root / "link").symlink_to(outside)  # a component under-root that escapes
    monkeypatch.setattr(path_guard, "_TEMPLATE_ROOT", str(root))

    with pytest.raises(UnsafeTemplatePathError):
        path_guard.safe_template_path("link/secret.j2")


def test_safe_path_symlink_inside_root_passes(monkeypatch, tmp_path):
    """With ``_TEMPLATE_ROOT`` monkeypatched to a REAL directory, a symlink that
    stays INSIDE the root resolves under-root and is allowed — the guard rejects
    escapes, not every symlink."""
    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    (root / "real" / "t.j2").write_text("hi", encoding="utf-8")
    (root / "link").symlink_to(root / "real")
    monkeypatch.setattr(path_guard, "_TEMPLATE_ROOT", str(root))

    assert path_guard.safe_template_path("link/t.j2") == "link/t.j2"


async def test_render_requires_source(manager):
    resp = await router.render_template(_req({}))
    assert resp.status_code == 400


@pytest.mark.parametrize("bad", [123, {"nested": "object"}, ["a", "b"], True])
async def test_render_non_string_content_rejected(manager, bad):
    """A non-string inline ``content`` is refused with a clean 400 before it can
    reach Jinja and surface as a 500."""
    resp = await router.render_template(_req({"content": bad}))
    assert resp.status_code == 400


async def test_render(manager):
    resp = await router.render_template(_req({"template_id": "a.j2", "kwargs": {"name": "Z"}}))
    assert "rendered:a.j2" in _data(resp)["data"]["rendered"]


async def test_clear_cache(manager):
    resp = await router.clear_templates_cache(_req())
    assert _data(resp)["data"] == {"cleared": True}
    assert manager.cleared is True


# --- error mapping: 404 on missing, 400 on author error, 500 on infra --------


async def test_get_missing_template_404(manager, monkeypatch):
    async def _missing(template_id):
        raise TemplateNotFoundError(f"Template '{template_id}' not found.")

    monkeypatch.setattr(manager, "fetch_template", _missing)
    resp = await router.get_template(_req({"template_id": "gone.j2"}))
    assert resp.status_code == 404
    assert "not found" in _data(resp)["error"]


async def test_get_schema_call_missing_404(manager, monkeypatch):
    # fetch_template succeeds, the schema call is what discovers the template is
    # gone -> still a 404 (the second raise site is covered).
    async def _missing_schema(content=None, template_id=None):
        raise TemplateNotFoundError(f"Template '{template_id}' not found.")

    monkeypatch.setattr(manager, "get_template_schema", _missing_schema)
    resp = await router.get_template(_req({"template_id": "gone.j2"}))
    assert resp.status_code == 404


async def test_get_bad_jinja_400(manager, monkeypatch):
    async def _broken(content=None, template_id=None):
        raise TemplateSyntaxError("unexpected end of template", 1)

    monkeypatch.setattr(manager, "get_template_schema", _broken)
    resp = await router.get_template(_req({"template_id": "a.j2"}))
    assert resp.status_code == 400
    assert "template error" in _data(resp)["error"]


async def test_render_missing_template_404(manager, monkeypatch):
    async def _missing(content=None, template_id=None, kwargs=None):
        raise TemplateNotFoundError(f"Template '{template_id}' not found.")

    monkeypatch.setattr(manager, "render_by_id_or_content", _missing)
    resp = await router.render_template(_req({"template_id": "gone.j2"}))
    assert resp.status_code == 404


async def test_render_bad_jinja_400(manager, monkeypatch):
    async def _broken(content=None, template_id=None, kwargs=None):
        raise TemplateSyntaxError("bad syntax", 1)

    monkeypatch.setattr(manager, "render_by_id_or_content", _broken)
    resp = await router.render_template(_req({"content": "{{ oops"}))
    assert resp.status_code == 400
    assert "template error" in _data(resp)["error"]


async def test_render_both_content_and_id_400(manager):
    resp = await router.render_template(_req({"content": "hi", "template_id": "a.j2"}))
    assert resp.status_code == 400
    assert "not both" in _data(resp)["error"]


async def test_render_infra_error_propagates_as_500(manager, monkeypatch):
    # A genuine storage/infra failure is NOT an author error: it must propagate
    # (surfacing as a 500), never be masked as a 400/404.
    async def _boom(content=None, template_id=None, kwargs=None):
        raise RuntimeError("redis down")

    monkeypatch.setattr(manager, "render_by_id_or_content", _boom)
    with pytest.raises(RuntimeError, match="redis down"):
        await router.render_template(_req({"template_id": "a.j2"}))
