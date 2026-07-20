"""Op-level oracles for the tool-extension operations' defensive branches that the
route-level suite (``tests/routers/test_tool_extensions.py``) cannot reach without
bypassing the boot-time duplicate-name collision: the multi-owner reject and the
other-config mcp mapper. The happy paths and every 400/404/409/503 the route emits
are pinned by the route suite over the real reload path."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tai_contract.app import tai_app
from tai_kit.settings import reset_all_settings

from tai_skeleton.app import instance
from tai_skeleton.manifest import Manifest
from tai_skeleton.operations import BadRequestError
from tai_skeleton.operations import tool_extensions as ops
from tests._fakes.bus import FakeBus


def _fake_manifest(*, tools: list, mcp: list) -> Manifest:
    return cast(Manifest, SimpleNamespace(tools=tools, mcp=mcp))


def test_other_mappers_reports_mcp_config_carrying_the_name():
    # Owner is a tools config; an mcp config OTHER than the owner also maps the
    # name — it is reported so the route can 409 rather than orphan those combos.
    manifest = _fake_manifest(
        tools=[SimpleNamespace(module="mod", extensions={})],
        mcp=[SimpleNamespace(title="srv", extensions={"shout": [["marka"]]})],
    )
    others = ops._other_mappers(manifest, "shout", ("tools", "mod"))
    assert others == ["mcp:srv"]


def test_other_mappers_excludes_the_owning_mcp_config():
    manifest = _fake_manifest(
        tools=[],
        mcp=[SimpleNamespace(title="srv", extensions={"shout": [["marka"]]})],
    )
    assert ops._other_mappers(manifest, "shout", ("mcp", "srv")) == []


async def test_set_tool_extensions_multiple_owners_is_bad_request(monkeypatch):
    monkeypatch.setattr(ops, "_live_manifest", lambda: SimpleNamespace())
    monkeypatch.setattr(
        ops,
        "_owning_configs",
        lambda _manifest, _name: [("tools", "a"), ("mcp", "b")],
    )
    with pytest.raises(BadRequestError, match="provided by multiple configs"):
        await ops.set_tool_extensions("shout", [["marka"]])


class _MutateStore:
    """A config manager whose ``mutate_manifest`` runs the guarded mutator on a copy of
    the stored document and persists only if it returns without raising — the seam the
    set_tool_extensions pipeline drives (a raise inside leaves the store untouched)."""

    def __init__(self, *, manifest: dict) -> None:
        self.manifest = manifest
        self.persisted: list[dict] = []

    def mutate_manifest(self, mutator: Any) -> dict:
        document = deepcopy(self.manifest)
        mutator(document)  # a raise here propagates before any persist
        self.manifest = document
        self.persisted.append(deepcopy(document))
        return document

    def read_manifest_preserved(self) -> dict:
        return deepcopy(self.manifest)


class _ReloadAdmin:
    def reload_config(self) -> dict:
        return {"status": "ok", "env_keys": 0}


async def test_set_tool_extensions_backend_without_bus_maps_to_400(monkeypatch):
    # The owner-resolution / conflict / registry checks pass, so the write reaches
    # ConfigService; the persisted manifest registers a task backend with no bus, so the
    # pipeline raises the RuntimeError ``BackendNeedsBusError`` at MUTATE time. The op
    # must map it to a loud 400 naming TAI_BUS_REDIS_URL, not let it escape as a 500.
    monkeypatch.delenv("TAI_BUS_REDIS_URL", raising=False)
    reset_all_settings()
    try:
        monkeypatch.setattr(ops, "_live_manifest", lambda: SimpleNamespace())
        monkeypatch.setattr(ops, "_owning_configs", lambda _manifest, _name: [("tools", "mod")])
        monkeypatch.setattr(ops, "_other_mappers", lambda _manifest, _name, _owner: [])
        monkeypatch.setattr(ops, "_validate_combos_against_registry", lambda _combos: None)

        store = _MutateStore(manifest={"backend_module": "myapp.backend", "tools": [{"title": "T", "module": "mod"}]})
        impl = SimpleNamespace(
            config=SimpleNamespace(config_manager=store),
            admin=_ReloadAdmin(),
            backends=SimpleNamespace(backend=None),
        )
        monkeypatch.setattr(tai_app, "_impl", impl)
        monkeypatch.setattr(instance.app, "_bus", FakeBus())

        with pytest.raises(BadRequestError, match="TAI_BUS_REDIS_URL"):
            await ops.set_tool_extensions("shout", [["marka"]])

        assert store.persisted == []  # rejected in validation, before any persist
    finally:
        reset_all_settings()
