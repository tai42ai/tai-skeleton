"""Config router: env read, env write + reload, the active config mode, and the
settings-schema surface with per-field current-value overlay."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import SecretStr
from starlette.requests import Request
from tai42_contract.app import tai42_app
from tai42_kit.settings import SettingsClassInfo, SettingsFieldInfo, TaiBaseSettings

from tai42_skeleton.app import instance
from tai42_skeleton.operations import config as config_ops
from tai42_skeleton.routers import config as router
from tests._fakes.bus import FakeBus


class _SecretDemoSettings(TaiBaseSettings):
    """Registered at import time — carries a ``SecretStr`` field so the schema
    route can be checked to report the field as secret AND round-trip its real
    value (the wire is unmasked)."""

    demo_secret: SecretStr | None = None


def _field(
    name: str,
    env_var: str,
    *,
    default: object = None,
    type_: str = "string",
    nested_group: str | None = None,
) -> SettingsFieldInfo:
    return SettingsFieldInfo(
        name=name,
        env_var=env_var,
        type=type_,
        default=default,
        required=False,
        secret=False,
        description=None,
        nested_group=nested_group,
    )


@pytest.fixture(autouse=True)
def _clear_marks_cache():
    # The secret-marks accessor is an ``@settings_cache`` singleton keyed off the
    # process env; clear it around each test so ``TAI_ENV_SECRET_KEYS`` set by
    # one test never bleeds into another.
    config_ops.env_secret_marks_settings.cache_clear()
    yield
    config_ops.env_secret_marks_settings.cache_clear()


def _req() -> Request:
    return cast(Request, SimpleNamespace(path_params={}))


def _body_req(body: bytes) -> Request:
    scope = {"type": "http", "method": "POST", "path": "/api/config/env", "headers": [], "query_string": b""}
    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _json(resp) -> dict:
    return json.loads(bytes(resp.body))


class _FakeConfigManager:
    def __init__(self, env):
        self._env = env
        self.written: list[dict] = []

    def read_env(self):
        return self._env

    def read_manifest_preserved(self):
        # No backend registered, so the env-change invariant has nothing to reject.
        return {}

    def write_env(self, config):
        self.written.append(config)
        self._env = {**self._env, **config}


class _FakeAdmin:
    def __init__(self, manager):
        self._manager = manager
        self.reloads = 0

    def reload_config(self):
        self.reloads += 1
        return {"status": "ok", "env_keys": len(self._manager._env)}


@pytest.fixture
def install(monkeypatch):
    def _install(env=None):
        manager = _FakeConfigManager(env if env is not None else {})
        admin = _FakeAdmin(manager)
        # No worker bus: the reload stays local-only (the fan-out itself is
        # covered by the dedicated propagation test).
        impl = SimpleNamespace(
            config=SimpleNamespace(config_manager=manager),
            admin=admin,
            backends=SimpleNamespace(backend=None),
        )
        monkeypatch.setattr(tai42_app, "_impl", impl)
        bus = FakeBus(origin="serve-x")
        monkeypatch.setattr(instance.app, "_bus", bus)
        return SimpleNamespace(manager=manager, admin=admin, bus=bus)

    return _install


# -- GET /api/config/env -----------------------------------------------------


async def test_read_env(install, monkeypatch):
    install({"API_KEY": "abc", "DEBUG": "1"})
    monkeypatch.setenv("TAI_ENV_SECRET_KEYS", "API_KEY")
    config_ops.env_secret_marks_settings.cache_clear()
    resp = await router.read_env(_req())
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"env": {"API_KEY": "abc", "DEBUG": "1"}, "secret_keys": ["API_KEY"]}}


async def test_read_env_missing_file_yields_empty_env(monkeypatch):
    class _Missing:
        def read_env(self):
            raise FileNotFoundError

    impl = SimpleNamespace(config=SimpleNamespace(config_manager=_Missing()), admin=None)
    monkeypatch.setattr(tai42_app, "_impl", impl)
    monkeypatch.delenv("TAI_ENV_SECRET_KEYS", raising=False)
    resp = await router.read_env(_req())
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"env": {}, "secret_keys": []}}


# -- GET /api/config/settings-schema -----------------------------------------


async def test_settings_schema_shape(install, monkeypatch):
    install({})
    info = SettingsClassInfo(
        name="Demo",
        module="mod",
        qualname="mod.Demo",
        fields=[_field("a", "A_VAR", default="x")],
    )
    monkeypatch.setattr(config_ops, "registered_settings", lambda: [info])
    monkeypatch.delenv("A_VAR", raising=False)
    resp = await router.read_settings_schema(_req())
    assert resp.status_code == 200
    data = _json(resp)["data"]
    assert list(data.keys()) == ["groups"]
    group = data["groups"][0]
    assert group["name"] == "Demo"
    assert group["module"] == "mod"
    assert group["qualname"] == "mod.Demo"
    field = group["fields"][0]
    for key in (
        "name",
        "env_var",
        "type",
        "default",
        "required",
        "secret",
        "description",
        "nested_group",
        "value",
    ):
        assert key in field
    assert field["value"] == "x"


async def test_settings_schema_value_overlay(install, monkeypatch):
    # PROC_WINS is in BOTH the store and process env — process must win.
    install({"STORE_ONLY": "from_store", "PROC_WINS": "store_value"})
    monkeypatch.setenv("PROC_WINS", "proc_value")
    monkeypatch.delenv("STORE_ONLY", raising=False)
    monkeypatch.delenv("DEFAULT_ONLY", raising=False)
    info = SettingsClassInfo(
        name="Demo",
        module="mod",
        qualname="mod.Demo",
        fields=[
            _field("proc", "PROC_WINS"),
            _field("store", "STORE_ONLY"),
            _field("dflt", "DEFAULT_ONLY", default="the_default"),
            _field("nested", "", type_="object", nested_group="Other"),
        ],
    )
    monkeypatch.setattr(config_ops, "registered_settings", lambda: [info])
    resp = await router.read_settings_schema(_req())
    fields = {f["name"]: f for f in _json(resp)["data"]["groups"][0]["fields"]}
    assert fields["proc"]["value"] == "proc_value"  # process env wins over store
    assert fields["store"]["value"] == "from_store"  # store-only
    assert fields["dflt"]["value"] == "the_default"  # neither -> default
    assert fields["nested"]["value"] is None  # nested reference: non-editable


async def test_settings_schema_secret_value_unmasked(install, monkeypatch):
    # Drive the REAL registry so the registered secret-bearing class appears.
    install({})
    monkeypatch.setenv("DEMO_SECRET", "supersecret")
    resp = await router.read_settings_schema(_req())
    groups = {g["name"]: g for g in _json(resp)["data"]["groups"]}
    assert "_SecretDemoSettings" in groups
    field = next(f for f in groups["_SecretDemoSettings"]["fields"] if f["name"] == "demo_secret")
    assert field["secret"] is True
    # The wire carries the REAL value — a display-side mask must NOT reach here.
    assert field["value"] == "supersecret"


async def test_settings_schema_missing_env_is_empty_not_500(monkeypatch):
    class _Missing:
        def read_env(self):
            raise FileNotFoundError

    impl = SimpleNamespace(config=SimpleNamespace(config_manager=_Missing()), admin=None)
    monkeypatch.setattr(tai42_app, "_impl", impl)
    monkeypatch.delenv("DEFAULT_X", raising=False)
    info = SettingsClassInfo(
        name="Demo",
        module="mod",
        qualname="mod.Demo",
        fields=[_field("dflt", "DEFAULT_X", default="d")],
    )
    monkeypatch.setattr(config_ops, "registered_settings", lambda: [info])
    resp = await router.read_settings_schema(_req())
    assert resp.status_code == 200  # missing .env -> empty overrides, not a 500
    fields = {f["name"]: f for f in _json(resp)["data"]["groups"][0]["fields"]}
    assert fields["dflt"]["value"] == "d"


async def test_secret_marks_roundtrip_and_group(install, monkeypatch):
    monkeypatch.setenv("TAI_ENV_SECRET_KEYS", "API_KEY, DB_URL ,")
    config_ops.env_secret_marks_settings.cache_clear()
    assert config_ops.env_secret_marks_settings().secret_keys == ["API_KEY", "DB_URL"]
    install({})
    resp = await router.read_settings_schema(_req())
    names = [g["name"] for g in _json(resp)["data"]["groups"]]
    assert "EnvSecretMarksSettings" in names


# -- POST /api/config/env ----------------------------------------------------


async def test_write_env_happy(install):
    ctx = install({"OLD": "keep"})
    resp = await router.write_env(_body_req(b'{"NEW": "val"}'))
    assert resp.status_code == 200
    assert _json(resp) == {
        "data": {
            "status": "ok",
            "env_keys": 2,
            "fanout": {"mode": "local-only", "note": "no worker bus configured; only this worker reloaded"},
        }
    }
    assert ctx.manager.written == [{"NEW": "val"}]
    assert ctx.admin.reloads == 1


async def test_write_env_non_string_value_400(install):
    ctx = install({})
    resp = await router.write_env(_body_req(b'{"PORT": 8080}'))
    assert resp.status_code == 400
    assert "strings" in _json(resp)["error"]
    assert ctx.manager.written == []
    assert ctx.admin.reloads == 0


async def test_write_env_not_object_400(install):
    install({})
    resp = await router.write_env(_body_req(b'["a", "b"]'))
    assert resp.status_code == 400


async def test_write_env_bad_json_400(install):
    install({})
    resp = await router.write_env(_body_req(b"nope"))
    assert resp.status_code == 400
    assert "invalid JSON" in _json(resp)["error"]


async def test_write_env_manager_value_error_maps_to_400(install):
    """A malformed key rejected by the config manager (``ValueError``) becomes a
    400, not an uncaught 500."""
    ctx = install({})

    def _raise(config):
        raise ValueError("invalid env key 'BAD KEY': must match [A-Za-z_][A-Za-z0-9_]*")

    ctx.manager.write_env = _raise
    resp = await router.write_env(_body_req(b'{"BAD KEY": "val"}'))
    assert resp.status_code == 400
    assert "invalid env key" in _json(resp)["error"]
    assert ctx.admin.reloads == 0


# -- GET /api/config/mode ----------------------------------------------------


async def test_read_mode(install, monkeypatch):
    install({})
    monkeypatch.setattr(config_ops, "config_mode", lambda: "k8s")
    resp = await router.read_mode(_req())
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"config_mode": "k8s"}}
