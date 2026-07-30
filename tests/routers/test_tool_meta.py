"""The tool-metadata routes as thin adapters over the REAL ops + store.

Each handler is driven directly with a built request, live over the true
:class:`PostgresToolMetaStore` running against the stateful fake Postgres (the
``pg`` fixture, whose ``FakeToolMetaPg`` is reused from the store suite). So the
HTTP-edge merge-patch extraction, the blank-display-name refusal, the tri-state
``hidden``, and the folder error→status mapping are exercised end-to-end, not
mocked. The app singleton is bound at import so the ``@tai42_app`` route
decorators land on a live app.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from starlette.requests import Request
from tai42_contract.app import tai42_app

import tai42_skeleton.tool_meta.store as store_module
from tai42_skeleton.app import instance
from tai42_skeleton.routers import tool_meta as router
from tests.tool_meta.conftest import FakeToolMetaPg, make_pg_ctx

tai42_app.bind(instance.build_app())


# -- request / response helpers ----------------------------------------------


def _run(coro_factory: Callable[[], Coroutine[Any, Any, None]]) -> None:
    asyncio.run(coro_factory())


def _request(method: str, path: str, *, body: Any = None, **path_params: str) -> Request:
    payload = b"" if body is None else json.dumps(body).encode()
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "path_params": path_params,
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(scope, receive)


def _data(resp) -> Any:
    return json.loads(bytes(resp.body))["data"]


def _err(resp) -> str:
    return json.loads(bytes(resp.body))["error"]


# -- fixtures ----------------------------------------------------------------


@pytest.fixture
def pg(monkeypatch: pytest.MonkeyPatch) -> FakeToolMetaPg:
    fake = FakeToolMetaPg()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(fake))
    return fake


# -- edge helpers ------------------------------------------------------------


async def _create_folder(name: str, parent_id: str | None = None):
    body: dict[str, Any] = {"name": name}
    if parent_id is not None:
        body["parent_id"] = parent_id
    return await router.create_folder(_request("POST", "/api/tool-meta/folders", body=body))


async def _patch_tool(tool_name: str, body: dict[str, Any]):
    return await router.upsert_tool_meta(
        _request("PATCH", f"/api/tool-meta/tools/{tool_name}", body=body, tool_name=tool_name)
    )


# -- list --------------------------------------------------------------------


def test_list_returns_folders_and_meta(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        folder = _data(await _create_folder("root"))
        await _patch_tool("weather", {"display_name": "Weather"})

        listing = _data(await router.list_tool_meta(_request("GET", "/api/tool-meta")))
        assert [f["name"] for f in listing["folders"]] == ["root"]
        assert listing["folders"][0]["id"] == folder["id"]
        assert [m["tool_name"] for m in listing["meta"]] == ["weather"]
        assert listing["meta"][0]["display_name"] == "Weather"

    _run(run)


# -- overlay merge-patch -----------------------------------------------------


def test_patch_only_display_name_leaves_other_fields(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        # Seed a full row, then patch just the display name.
        await _patch_tool("weather", {"display_name": "Old", "tags": ["a"], "hidden": True})
        row = _data(await _patch_tool("weather", {"display_name": "New"}))
        assert row["display_name"] == "New"
        # Untouched fields survive the merge-patch.
        assert row["tags"] == ["a"]
        assert row["hidden"] is True

    _run(run)


def test_patch_present_null_clears_field(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        await _patch_tool("weather", {"display_name": "Named"})
        row = _data(await _patch_tool("weather", {"display_name": None}))
        assert row["display_name"] is None

    _run(run)


def test_patch_tags_array_replaces_set(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        await _patch_tool("weather", {"tags": ["a", "b"]})
        row = _data(await _patch_tool("weather", {"tags": ["c"]}))
        assert row["tags"] == ["c"]

    _run(run)


@pytest.mark.parametrize("hidden", [None, True, False])
def test_patch_hidden_tri_state(pg: FakeToolMetaPg, hidden: bool | None) -> None:
    async def run() -> None:
        row = _data(await _patch_tool("weather", {"hidden": hidden}))
        assert row["hidden"] is hidden

    _run(run)


def test_patch_absent_field_unchanged_across_successive_patches(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        await _patch_tool("weather", {"display_name": "Keep", "tags": ["x"], "hidden": False})
        # A patch touching only tags leaves display_name and hidden intact.
        await _patch_tool("weather", {"tags": ["y"]})
        # A patch touching only hidden leaves display_name and tags intact.
        row = _data(await _patch_tool("weather", {"hidden": True}))
        assert row["display_name"] == "Keep"
        assert row["tags"] == ["y"]
        assert row["hidden"] is True

    _run(run)


def test_patch_folder_id_places_and_clears(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        folder = _data(await _create_folder("bucket"))
        placed = _data(await _patch_tool("weather", {"folder_id": folder["id"]}))
        assert placed["folder_id"] == folder["id"]
        cleared = _data(await _patch_tool("weather", {"folder_id": None}))
        assert cleared["folder_id"] is None

    _run(run)


def test_patch_unknown_folder_id_is_400(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        resp = await _patch_tool("weather", {"folder_id": "no-such-folder"})
        assert resp.status_code == 400
        assert "does not exist" in _err(resp)

    _run(run)


# -- blank display_name ------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_display_name_is_400(pg: FakeToolMetaPg, blank: str) -> None:
    async def run() -> None:
        resp = await _patch_tool("weather", {"display_name": blank})
        assert resp.status_code == 400
        assert "display_name" in _err(resp)

    _run(run)


def test_accepted_display_name_is_stored_stripped(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        row = _data(await _patch_tool("weather", {"display_name": "  Weather  "}))
        assert row["display_name"] == "Weather"

    _run(run)


# -- overlay delete ----------------------------------------------------------


def test_delete_overlay_is_idempotent(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        await _patch_tool("weather", {"display_name": "W"})
        first = await router.delete_tool_meta(_request("DELETE", "/api/tool-meta/tools/weather", tool_name="weather"))
        assert first.status_code == 200
        assert _data(first)["deleted"] is True
        # Deleting again — no row now — is still a clean 200.
        second = await router.delete_tool_meta(_request("DELETE", "/api/tool-meta/tools/weather", tool_name="weather"))
        assert second.status_code == 200
        assert await instance.app.tool_meta.store.get_meta("weather") is None

    _run(run)


# -- folders: happy paths ----------------------------------------------------


def test_folder_create_rename_move_delete_happy_path(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        root = _data(await _create_folder("root"))
        assert root["parent_id"] is None

        child = _data(await _create_folder("child", root["id"]))
        assert child["parent_id"] == root["id"]

        renamed = _data(
            await router.rename_folder(
                _request(
                    "POST",
                    f"/api/tool-meta/folders/{child['id']}/rename",
                    body={"name": "renamed"},
                    folder_id=child["id"],
                )
            )
        )
        assert renamed["name"] == "renamed"

        moved = _data(
            await router.move_folder(
                _request(
                    "POST",
                    f"/api/tool-meta/folders/{child['id']}/move",
                    body={"parent_id": None},
                    folder_id=child["id"],
                )
            )
        )
        assert moved["parent_id"] is None

        deleted = await router.delete_folder(
            _request("DELETE", f"/api/tool-meta/folders/{child['id']}", folder_id=child["id"])
        )
        assert deleted.status_code == 200
        assert _data(deleted)["deleted"] is True

    _run(run)


def test_folder_move_cycle_is_400(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        parent = _data(await _create_folder("parent"))
        child = _data(await _create_folder("child", parent["id"]))
        resp = await router.move_folder(
            _request(
                "POST",
                f"/api/tool-meta/folders/{parent['id']}/move",
                body={"parent_id": child["id"]},
                folder_id=parent["id"],
            )
        )
        assert resp.status_code == 400

    _run(run)


def test_folder_delete_non_empty_is_409(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        parent = _data(await _create_folder("parent"))
        await _create_folder("child", parent["id"])
        resp = await router.delete_folder(
            _request("DELETE", f"/api/tool-meta/folders/{parent['id']}", folder_id=parent["id"])
        )
        assert resp.status_code == 409

    _run(run)


def test_folder_rename_unknown_is_404(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        resp = await router.rename_folder(
            _request(
                "POST",
                "/api/tool-meta/folders/no-such/rename",
                body={"name": "x"},
                folder_id="no-such",
            )
        )
        assert resp.status_code == 404

    _run(run)


def test_folder_delete_unknown_is_404(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        resp = await router.delete_folder(_request("DELETE", "/api/tool-meta/folders/no-such", folder_id="no-such"))
        assert resp.status_code == 404

    _run(run)


def test_folder_duplicate_sibling_is_409(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        await _create_folder("dup")
        resp = await _create_folder("dup")
        assert resp.status_code == 409

    _run(run)


def test_folder_create_unknown_parent_is_400(pg: FakeToolMetaPg) -> None:
    async def run() -> None:
        resp = await _create_folder("orphan", "no-such-parent")
        assert resp.status_code == 400

    _run(run)
