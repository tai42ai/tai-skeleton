"""The backup/restore router — the section round-trips plus the export/import
envelope and error-reporting rules.

Handlers are driven directly (the router-test pattern): the bound app impl is
swapped for a stand-in exposing the facets each section touches
(``backup``/``config``/``storage``/``sub_app``), and the two subsystems a section
reaches WITHOUT the ``tai42_app`` handle — access_control's redis provisioning and
the hooks manager — are faked at their own module seams.

Each feasible core section is round-tripped (fresh state -> export -> wipe ->
import -> state equal), and the router's envelope rules are pinned: unknown
export section -> 400, version gate -> 400, a Content-Disposition download, an
absent subsystem recorded as a section error (not a 500), an unselected section
left untouched, an unknown/failed section reported with ``ok: false``, and a
plugin-registered section flowing through both doors.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM
from tai42_contract.app import tai42_app
from tai42_contract.hooks import HookParams

from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.app import instance
from tai42_skeleton.authz import execution as execution_module
from tai42_skeleton.backup.registry import BackupRegistry
from tai42_skeleton.backup.sections import _empty_report, register_core_sections
from tai42_skeleton.routers.backup import export_backup, import_backup, list_sections
from tai42_skeleton.template import TemplateNotFoundError
from tests._fakes.bus import FakeBus

# -- request builders --------------------------------------------------------


def _get_req() -> Request:
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def _post_req(payload: dict) -> Request:
    body = json.dumps(payload).encode()
    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _json(resp) -> dict:
    return json.loads(bytes(resp.body))


# -- subsystem fakes ---------------------------------------------------------


class _FakeConfigManager:
    """In-memory stand-in for the file config manager's env/manifest seam."""

    def __init__(self, env=None, manifest=None, manifest_preserved=None):
        self._env = env
        self._manifest = manifest
        # The preserved-tag view exported by the manifest section; defaults to the
        # plain manifest when a test does not exercise ``!ENV`` markers.
        self._manifest_preserved = manifest_preserved

    def read_env(self) -> dict[str, str]:
        if self._env is None:
            raise FileNotFoundError("no env file")
        return dict(self._env)

    def write_env(self, config: dict[str, str]) -> None:
        base = dict(self._env or {})
        base.update({k: v for k, v in config.items() if v})
        self._env = base

    def read_manifest(self) -> dict:
        if self._manifest is None:
            raise FileNotFoundError("no manifest")
        return dict(self._manifest)

    def read_manifest_preserved(self) -> dict:
        if self._manifest is None:
            raise FileNotFoundError("no manifest")
        preserved = self._manifest if self._manifest_preserved is None else self._manifest_preserved
        return dict(preserved)

    def replace_manifest(self, document: dict) -> dict:
        # The transactional whole-document replace the manifest section imports
        # through :meth:`ConfigService.apply_replace`.
        self._manifest = dict(document)
        return dict(self._manifest)


class _FakeResourceManager:
    def __init__(self, templates=None):
        self._templates = dict(templates or {})

    async def list_resources(self) -> list[str]:
        return list(self._templates)

    async def fetch_template(self, template_id: str) -> str:
        return self._templates[template_id]

    async def upload_template(self, path: str, content: str) -> None:
        self._templates[path] = content

    async def render_by_id_or_content(self, *, content, template_id, kwargs) -> str:
        """The condition render, resolving a ``template_id`` against the SAME store
        ``upload_template`` writes into — so a condition template that has not been
        restored yet raises exactly as the real manager does."""
        if template_id is None:
            return content or ""
        if template_id not in self._templates:
            raise TemplateNotFoundError(template_id)
        return self._templates[template_id]


class _FakeRouteConfig:
    def __init__(self, tools, transport="http"):
        self.tools = tools
        self.transport = transport

    def model_dump(self) -> dict:
        return {"tools": self.tools, "transport": self.transport}


class _FakeSubAppRouter:
    def __init__(self, routes=None):
        self.routes = dict(routes or {})

    async def register_sub_mcp_app(self, slug, tools, transport="http"):
        self.routes[slug] = _FakeRouteConfig(tools, transport)


def _registry() -> BackupRegistry:
    registry = BackupRegistry()
    register_core_sections(registry)
    return registry


@pytest.fixture
def execution_gate_off(monkeypatch):
    """Access control OFF for the token-free-evaluable assertion each imported hook
    record runs, so the webhooks round-trips below read the section's own envelope
    behavior; the execution-key rules have their own coverage."""
    monkeypatch.setattr(execution_module, "access_control_settings", lambda: AccessControlSettings(enable=False))


def _install(monkeypatch, *, registry=None, **facets) -> SimpleNamespace:
    # The manifest/env sections import through ``ConfigService.from_app()``, which
    # resolves the admin reload seam off ``tai42_app`` and the worker bus off the app
    # singleton — wire a local-reload admin and a recording bus so a restore runs the
    # real pipeline (validate + reload + broadcast) against the faked config manager.
    facets.setdefault("admin", SimpleNamespace(reload_config=lambda: {"status": "ok", "reloaded": True}))
    impl = SimpleNamespace(backup=registry or _registry(), **facets)
    monkeypatch.setattr(tai42_app, "_impl", impl)
    monkeypatch.setattr(instance.app, "_bus", FakeBus(origin="serve-test"))
    return impl


# -- GET /api/backup/sections ------------------------------------------------


async def test_sections_lists_core_sections_with_secret_flags(monkeypatch):
    _install(monkeypatch)
    resp = await list_sections(_get_req())
    assert resp.status_code == 200
    sections = _json(resp)["data"]
    by_name = {s["name"]: s["secret"] for s in sections}
    # Exactly the registered core sections.
    assert set(by_name) == {
        "manifest",
        "env",
        "access_control",
        "sub_mcp",
        "webhooks",
        "conversations",
        "templates",
        "schedules",
        "connector_catalog",
        "connector_connections",
        "versioned_documents",
    }
    assert by_name["manifest"] is False
    assert by_name["env"] is True
    assert by_name["access_control"] is True
    assert by_name["templates"] is False
    # Secret sections carry the flag so the export credential gate applies.
    assert by_name["schedules"] is True
    assert by_name["connector_connections"] is True
    # The versioned-document store covers secret-bearing bodies (preset kwargs,
    # AC-policy conditions), so its one kind-agnostic section is flagged secret.
    assert by_name["versioned_documents"] is True
    # The connector catalog is a public template — no secrets — so it is not flagged.
    assert by_name["connector_catalog"] is False
    # The routing rows' export equals the grantable route-list read (each row's
    # callback_secret is excluded and re-minted on import), so it is not flagged.
    assert by_name["conversations"] is False


# -- manifest / env round-trips (sync sections) ------------------------------


async def test_manifest_roundtrip(monkeypatch):
    manifest = {"mcp": [{"title": "x", "config": {"url": "http://x"}}]}
    cm = _FakeConfigManager(manifest=manifest)
    _install(monkeypatch, config=SimpleNamespace(config_manager=cm))

    doc = _json(await export_backup(_post_req({"sections": ["manifest"]})))
    assert doc["sections"]["manifest"] == manifest

    cm._manifest = {}  # wipe
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["manifest"]})))["data"]
    assert data["ok"] is True
    section = data["sections"]["manifest"]
    # The section keeps its existing keys and GAINS the per-origin fleet report; the
    # restore applies immediately through the pipeline (validate + reload + broadcast).
    assert {k: section[k] for k in ("created", "updated", "skipped", "errors")} == {
        "created": 0,
        "updated": 1,
        "skipped": 0,
        "errors": [],
    }
    assert section["fanout"]["mode"] == "local-only"
    assert cm._manifest == manifest


async def test_manifest_import_invalid_document_reports_error_and_leaves_store_untouched(monkeypatch):
    # A restore now validates through the pipeline BEFORE persisting: an invalid
    # document raises inside ``apply_replace`` (nothing persisted), and the router
    # records it as this section's error rather than corrupting the stored manifest.
    original = {"mcp": [{"title": "keep", "config": {"url": "http://keep"}}]}
    cm = _FakeConfigManager(manifest=dict(original))
    _install(monkeypatch, config=SimpleNamespace(config_manager=cm))
    document = {"version": 1, "sections": {"manifest": {"tools": "not-a-list"}}}
    data = _json(await import_backup(_post_req({"document": document, "sections": ["manifest"]})))["data"]
    assert data["ok"] is False
    assert data["sections"]["manifest"]["errors"]  # the validation failure is surfaced
    assert cm._manifest == original  # store untouched


async def test_manifest_export_carries_preserved_env_marker_not_secret(monkeypatch):
    # The non-secret manifest section must export the PRESERVED-tag view: an
    # ``!ENV`` placeholder travels as its literal marker string, never the resolved
    # secret, so no live secret leaks into a section the UI presents as non-secret.
    cm = _FakeConfigManager(
        manifest={"auth": {"token": "super-secret-value"}},
        manifest_preserved={"auth": {"token": "!ENV ${SOME_VAR}"}},
    )
    _install(monkeypatch, config=SimpleNamespace(config_manager=cm))

    doc = _json(await export_backup(_post_req({"sections": ["manifest"]})))
    section = doc["sections"]["manifest"]
    assert section == {"auth": {"token": "!ENV ${SOME_VAR}"}}
    # JSON-safe and the resolved secret is absent from the exported section.
    assert "super-secret-value" not in json.dumps(section)


async def test_env_roundtrip_and_created_updated_counts(monkeypatch):
    cm = _FakeConfigManager(env={"A": "1", "B": "2"})
    _install(monkeypatch, config=SimpleNamespace(config_manager=cm))

    doc = _json(await export_backup(_post_req({"sections": ["env"]})))
    assert doc["sections"]["env"] == {"A": "1", "B": "2"}

    cm._env = {"A": "old"}  # A pre-exists (update), B is new (create)
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["env"]})))["data"]
    assert data["ok"] is True
    section = data["sections"]["env"]
    assert {k: section[k] for k in ("created", "updated", "skipped", "errors")} == {
        "created": 1,
        "updated": 1,
        "skipped": 0,
        "errors": [],
    }
    # The env restore applied immediately through the pipeline and carries the fleet report.
    assert section["fanout"]["mode"] == "local-only"
    assert cm._env == {"A": "1", "B": "2"}


async def test_secret_section_absent_from_export_unless_requested(monkeypatch):
    cm = _FakeConfigManager(env={"A": "1"}, manifest={"mcp": []})
    _install(monkeypatch, config=SimpleNamespace(config_manager=cm))
    doc = _json(await export_backup(_post_req({"sections": ["manifest"]})))
    assert "manifest" in doc["sections"]
    assert "env" not in doc["sections"]


# -- access_control round-trip (redis provisioning faked) --------------------


async def test_access_control_roundtrip_mints_new_keys(monkeypatch):
    from tai42_identity_redis import redis_api_key_provider as provider_module

    from tai42_skeleton.access_control import management
    from tai42_skeleton.access_control import store as store_module
    from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

    # The POLICY store is Postgres; the identity record + live context are Redis.
    source_pg = FakeAccessControlPg()
    source_redis = FakeRedis()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(source_pg))
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(source_redis))
    monkeypatch.setattr(provider_module, "client_ctx", make_client_ctx(source_redis))
    await management.add_url_to_scope("scope-a", "/api/x")
    await management.add_user_api_key("user1", "first key", ["scope-a"])

    _install(monkeypatch)
    doc = _json(await export_backup(_post_req({"sections": ["access_control"]})))
    section = doc["sections"]["access_control"]
    assert section["scopes"] == {"/api/x": "scope-a"}
    assert any(token["user_id"] == "user1" for token in section["tokens"])

    # Restore into a fresh (wiped) store + Redis.
    target_pg = FakeAccessControlPg()
    target_redis = FakeRedis()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(target_pg))
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(target_redis))
    monkeypatch.setattr(provider_module, "client_ctx", make_client_ctx(target_redis))
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["access_control"]})))["data"]
    assert data["ok"] is True
    report = data["sections"]["access_control"]
    minted = report["new_api_keys"]
    assert len(minted) == 1
    assert minted[0]["user_id"] == "user1"
    assert minted[0]["description"] == "first key"
    assert minted[0]["api_key"].startswith("sk-")

    assert await management.get_all_existing_scopes() == {"/api/x": "scope-a"}


# -- sub_mcp round-trip ------------------------------------------------------


async def test_sub_mcp_roundtrip(monkeypatch):
    # Backup export/import go through the DURABLE store (the source of truth), with
    # the service binding restored registrations into the local router too.
    from tai42_contract.sub_mcp import RouteConfig

    from tai42_skeleton.sub_mcp import store as sub_mcp_store

    seeded = sub_mcp_store.InMemorySubMcpStore()
    await seeded.save_route("slug1", RouteConfig(tools=["tool_a"], transport="http"))
    monkeypatch.setattr(sub_mcp_store, "_IN_MEMORY_STORE", seeded)
    router = _FakeSubAppRouter()
    _install(monkeypatch, sub_app=SimpleNamespace(mcp_sub_app_router=router))

    doc = _json(await export_backup(_post_req({"sections": ["sub_mcp"]})))
    assert doc["sections"]["sub_mcp"] == {"slug1": {"tools": ["tool_a"], "transport": "http"}}

    # Restore into a FRESH (empty) store + router: the slug is not present, so the
    # import is a create, and the service persists it to the store AND binds it locally.
    fresh = sub_mcp_store.InMemorySubMcpStore()
    monkeypatch.setattr(sub_mcp_store, "_IN_MEMORY_STORE", fresh)
    router.routes = {}
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["sub_mcp"]})))["data"]
    assert data["ok"] is True
    assert data["sections"]["sub_mcp"] == {"created": 1, "updated": 0, "skipped": 0, "errors": []}
    assert router.routes["slug1"].tools == ["tool_a"]
    restored = await fresh.get_route("slug1")
    assert restored is not None
    assert restored.tools == ["tool_a"]


# -- webhooks round-trip (hooks manager faked) -------------------------------


async def test_webhooks_roundtrip(monkeypatch, execution_gate_off):
    from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
    from tai42_skeleton.hooks.settings import HooksSettings

    source = InMemoryHooksManager(HooksSettings())
    monkeypatch.setattr("tai42_skeleton.hooks.cache.get_hooks_manager", lambda: source)
    await source.register(
        HookParams(name="h1", topic="t1", tool="mytool", execution_key="k-fire", execution_key_fingerprint="fp-fire")
    )

    _install(monkeypatch)
    doc = _json(await export_backup(_post_req({"sections": ["webhooks"]})))
    # The webhooks section is now an ENVELOPE; an in-memory deployment holds no
    # trigger links, so the trigger halves are truthfully empty.
    section = doc["sections"]["webhooks"]
    assert [h["name"] for h in section["hooks"]] == ["h1"]
    assert section["trigger_links"] == []
    assert section["tombstones"] == []

    target = InMemoryHooksManager(HooksSettings())
    monkeypatch.setattr("tai42_skeleton.hooks.cache.get_hooks_manager", lambda: target)
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["webhooks"]})))["data"]
    assert data["ok"] is True
    assert data["sections"]["webhooks"] == {"created": 1, "updated": 0, "skipped": 0, "errors": []}
    assert set((await target.list_hooks()).keys()) == {"h1"}


async def test_import_restores_condition_templates_before_the_hooks_that_render_them(monkeypatch):
    """A hook bound to a key whose policy carries a ``condition_id`` restores from a
    document that also carries the template — the replay order is what makes the
    execution-key scan able to render it."""
    from types import SimpleNamespace as _NS

    from tai42_skeleton.access_control import policy as policy_module
    from tai42_skeleton.access_control import store as store_module
    from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
    from tai42_skeleton.hooks.settings import HooksSettings
    from tests.access_control.conftest import FakeAccessControlPg, make_pg_ctx
    from tests.access_control.conftest import FakeRedis as FakeAccessControlRedis
    from tests.access_control.conftest import make_client_ctx as make_access_control_client_ctx

    pg = FakeAccessControlPg()
    # The key's condition lives in the template store, exactly as a ``condition_id``
    # policy does on a real host.
    pg.add_policy("svc", ["a"], condition_id="policies/svc.j2", policy_data={KEY_FINGERPRINT_CLAIM: "fp-svc"})
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(policy_module, "client_ctx", make_access_control_client_ctx(FakeAccessControlRedis()))
    monkeypatch.setattr(execution_module, "access_control_settings", lambda: AccessControlSettings(enable=True))

    target = InMemoryHooksManager(HooksSettings())
    monkeypatch.setattr("tai42_skeleton.hooks.cache.get_hooks_manager", lambda: target)
    manager = _FakeResourceManager()
    _install(monkeypatch, storage=_NS(resource_manager=manager))

    document = {
        "version": 1,
        "sections": {
            "templates": {"policies/svc.j2": ".context.used < 10"},
            "webhooks": {
                "hooks": [
                    {
                        "name": "h1",
                        "topic": "t1",
                        "tool": "mytool",
                        "execution_key": "svc",
                        "execution_key_fingerprint": "fp-svc",
                    }
                ],
                "topic_verifiers": {},
                "trigger_links": [],
                "tombstones": [],
            },
        },
    }
    data = _json(await import_backup(_post_req({"document": document, "sections": ["webhooks", "templates"]})))["data"]
    assert data["ok"] is True
    assert data["sections"]["webhooks"] == {"created": 1, "updated": 0, "skipped": 0, "errors": []}
    assert set((await target.list_hooks()).keys()) == {"h1"}


# -- templates round-trip ----------------------------------------------------


async def test_templates_roundtrip(monkeypatch):
    manager = _FakeResourceManager({"greeting.j2": "hello {{ name }}"})
    _install(monkeypatch, storage=SimpleNamespace(resource_manager=manager))

    doc = _json(await export_backup(_post_req({"sections": ["templates"]})))
    assert doc["sections"]["templates"] == {"greeting.j2": "hello {{ name }}"}

    manager._templates = {}  # wipe
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["templates"]})))["data"]
    assert data["ok"] is True
    assert data["sections"]["templates"] == {"created": 1, "updated": 0, "skipped": 0, "errors": []}
    assert manager._templates == {"greeting.j2": "hello {{ name }}"}


async def test_templates_import_skips_traversal_key(monkeypatch):
    # An untrusted backup carrying a traversal key must be rejected per-key: the
    # unsafe key is skipped with a report error + log, the safe key still uploads,
    # and one bad key never aborts the restore.
    manager = _FakeResourceManager()
    _install(monkeypatch, storage=SimpleNamespace(resource_manager=manager))

    doc = {"version": 1, "sections": {"templates": {"../../evil": "x", "ok/name.j2": "y"}}}
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["templates"]})))["data"]
    # A per-key skip surfaces in the report (so ``ok`` is False), but the restore is
    # NOT aborted — the safe key still imports.
    assert data["ok"] is False
    report = data["sections"]["templates"]
    assert report["created"] == 1
    assert report["skipped"] == 1
    assert len(report["errors"]) == 1
    assert "../../evil" in report["errors"][0]
    # Exactly the safe key was uploaded; the traversal key never reached the store.
    assert manager._templates == {"ok/name.j2": "y"}


# -- absent/failing subsystem is a section error, not a 500 ------------------


async def test_export_absent_subsystem_records_section_error_not_500(monkeypatch):
    # A section whose exporter raises (its backing subsystem is absent) is
    # recorded per-section into the document's ``errors`` and omitted from
    # ``sections`` — the export still returns a download, never a 500.
    registry = _registry()

    def _absent_exporter():
        raise RuntimeError("backing subsystem is not installed")

    registry.register_section("absent", _absent_exporter, lambda _payload: _empty_report())
    _install(monkeypatch, registry=registry)

    resp = await export_backup(_post_req({"sections": ["absent"]}))
    assert resp.status_code == 200
    assert resp.headers["Content-Disposition"].startswith("attachment; filename=")
    doc = _json(resp)
    assert "absent" not in doc["sections"]
    assert "backing subsystem is not installed" in doc["errors"]["absent"]


# -- access_control pattern + validation -------------------------------------


async def test_access_control_pattern_scoped_route_round_trips(monkeypatch):
    from tai42_identity_redis import redis_api_key_provider as provider_module

    from tai42_skeleton.access_control import management
    from tai42_skeleton.access_control import store as store_module
    from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(FakeAccessControlPg()))
    source = FakeRedis()
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(source))
    # The export enumerates tokens through the provider, so point it at the fake too.
    monkeypatch.setattr(provider_module, "client_ctx", make_client_ctx(source))
    await management.add_url_to_scope("scope-a", "/orders/{id}", pattern=r"/orders/\d+")

    _install(monkeypatch)
    doc = _json(await export_backup(_post_req({"sections": ["access_control"]})))
    section = doc["sections"]["access_control"]
    assert section["scopes"] == {"/orders/{id}": "scope-a"}
    assert section["patterns"] == {"/orders/{id}": r"/orders/\d+"}

    # Restore into a fresh (wiped) store — the dynamic pattern comes back so a
    # pattern-scoped route re-authorizes exactly as before the backup.
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(FakeAccessControlPg()))
    target = FakeRedis()
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(target))
    monkeypatch.setattr(provider_module, "client_ctx", make_client_ctx(target))
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["access_control"]})))["data"]
    assert data["ok"] is True
    assert await management.get_all_existing_scopes() == {"/orders/{id}": "scope-a"}
    assert await management.get_all_existing_patterns() == {"/orders/{id}": r"/orders/\d+"}


async def test_access_control_public_route_round_trips(monkeypatch):
    from tai42_identity_redis import redis_api_key_provider as provider_module

    from tai42_skeleton.access_control import management
    from tai42_skeleton.access_control import store as store_module
    from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(FakeAccessControlPg()))
    source = FakeRedis()
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(source))
    # The export enumerates tokens through the provider, so point it at the fake too.
    monkeypatch.setattr(provider_module, "client_ctx", make_client_ctx(source))
    from tai42_skeleton.access_control.settings import access_control_settings

    marker = access_control_settings().public_resource_id
    # An operator-set explicit public mapping (value ``public_resource_id``).
    await management.add_url_to_scope(marker, "/api/public")

    _install(monkeypatch)
    doc = _json(await export_backup(_post_req({"sections": ["access_control"]})))
    section = doc["sections"]["access_control"]
    # The public route is carried in the backup, not dropped by the non-public filter.
    assert section["scopes"]["/api/public"] == marker

    # Restore into a fresh (wiped) store — the public route comes back public,
    # never silently reverting to protected.
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(FakeAccessControlPg()))
    target = FakeRedis()
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(target))
    monkeypatch.setattr(provider_module, "client_ctx", make_client_ctx(target))
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["access_control"]})))["data"]
    assert data["ok"] is True
    assert await management.get_all_route_mappings() == {"/api/public": marker}


async def test_access_control_import_marker_mapping_lands_via_pin_route_public(monkeypatch):
    # A restored marker-valued mapping restores through the dedicated ``pin_route_public``
    # writer, never ``add_url_to_scope`` — the marker is a column value, not a scope.
    from tai42_skeleton.access_control import management
    from tai42_skeleton.access_control import store as store_module
    from tai42_skeleton.access_control.settings import access_control_settings
    from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

    pg = FakeAccessControlPg()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(FakeRedis()))
    marker = access_control_settings().public_resource_id

    pinned: list[tuple[str, str | None]] = []
    added: list[tuple[str, str, str | None]] = []
    orig_pin = management.pin_route_public
    orig_add = management.add_url_to_scope

    async def spy_pin(url, pattern=None):
        pinned.append((url, pattern))
        await orig_pin(url, pattern)

    async def spy_add(scope_id, url, pattern=None):
        added.append((scope_id, url, pattern))
        await orig_add(scope_id, url, pattern)

    monkeypatch.setattr(management, "pin_route_public", spy_pin)
    monkeypatch.setattr(management, "add_url_to_scope", spy_add)
    _install(monkeypatch)

    document = {
        "version": 1,
        "sections": {
            "access_control": {
                "scopes": {"/pub": marker, "/prot": "scope-a"},
                "patterns": {"/pub": r"/pub/\d+"},
                "tokens": [],
            }
        },
    }
    data = _json(await import_backup(_post_req({"document": document, "sections": ["access_control"]})))["data"]
    assert data["ok"] is True
    # The public route went through pin_route_public; the protected one through add_url_to_scope.
    # The public route's dynamic pattern is forwarded into pin_route_public, not dropped.
    assert pinned == [("/pub", r"/pub/\d+")]
    assert added == [("scope-a", "/prot", None)]
    # The public route restored as a marker-valued row (with its pattern), never routed
    # through the scope setter.
    assert pg.route("/pub")["scope_id"] == marker
    assert await management.get_all_existing_patterns() == {"/pub": r"/pub/\d+"}


async def test_access_control_import_rejects_blank_user_id(monkeypatch):
    from tai42_skeleton.access_control import management
    from tests.access_control.conftest import FakeRedis, make_client_ctx

    target = FakeRedis()
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(target))
    _install(monkeypatch)

    document = {
        "version": 1,
        "sections": {
            "access_control": {
                "scopes": {},
                "patterns": {},
                "tokens": [{"user_id": "", "description": "blank"}],
            }
        },
    }
    data = _json(await import_backup(_post_req({"document": document, "sections": ["access_control"]})))["data"]
    # A corrupt token with an empty user id is a loud per-section rejection, not a
    # silent record written under a blank id.
    assert data["ok"] is False
    report = data["sections"]["access_control"]
    assert report["skipped"] == 1
    assert report["new_api_keys"] == []
    assert any("user_id" in err for err in report["errors"])


# -- envelope + error-reporting rules ----------------------------------------


async def test_export_unknown_section_is_400(monkeypatch):
    _install(monkeypatch)
    resp = await export_backup(_post_req({"sections": ["not_a_section"]}))
    assert resp.status_code == 400
    assert "unknown section" in _json(resp)["error"]


async def test_export_sets_attachment_content_disposition(monkeypatch):
    cm = _FakeConfigManager(manifest={"mcp": []})
    _install(monkeypatch, config=SimpleNamespace(config_manager=cm))
    resp = await export_backup(_post_req({"sections": ["manifest"]}))
    assert resp.headers["Content-Disposition"].startswith('attachment; filename="tai-backup-')
    assert resp.headers["Content-Disposition"].endswith('.json"')


async def test_import_rejects_wrong_document_version(monkeypatch):
    _install(monkeypatch)
    document = {"version": 2, "sections": {"manifest": {}}}
    resp = await import_backup(_post_req({"document": document, "sections": ["manifest"]}))
    assert resp.status_code == 400
    assert "version" in _json(resp)["error"]


async def test_import_ignores_unselected_section(monkeypatch):
    cm = _FakeConfigManager(env={"A": "1"}, manifest={"mcp": []})
    _install(monkeypatch, config=SimpleNamespace(config_manager=cm))
    imported = {"mcp": [{"title": "y", "config": {"url": "http://y"}}]}
    document = {"version": 1, "sections": {"manifest": imported, "env": {"Z": "9"}}}
    data = _json(await import_backup(_post_req({"document": document, "sections": ["manifest"]})))["data"]
    assert data["ok"] is True
    assert set(data["sections"]) == {"manifest"}  # env absent from the report
    assert cm._manifest == imported
    assert cm._env == {"A": "1"}  # env untouched


async def test_import_unknown_section_name_reports_error_not_400(monkeypatch):
    _install(monkeypatch)
    document = {"version": 1, "sections": {"ghost": {"anything": True}}}
    resp = await import_backup(_post_req({"document": document, "sections": ["ghost"]}))
    assert resp.status_code == 200
    data = _json(resp)["data"]
    assert data["ok"] is False
    assert "unknown section" in data["sections"]["ghost"]["errors"][0]


async def test_import_failing_importer_reports_error_and_not_ok(monkeypatch):
    registry = _registry()

    def _boom(_payload):
        raise RuntimeError("importer exploded")

    registry.register_section("boomer", lambda: {"x": 1}, _boom)
    _install(monkeypatch, registry=registry)

    document = {"version": 1, "sections": {"boomer": {"x": 1}}}
    data = _json(await import_backup(_post_req({"document": document, "sections": ["boomer"]})))["data"]
    assert data["ok"] is False
    assert data["sections"]["boomer"]["errors"] == ["importer exploded"]


async def test_import_replays_sections_in_registration_order(monkeypatch):
    # Registration order is a declared DEPENDENCY order, so the caller's list is a SET:
    # how it was typed must not change the stored state or the report.
    registry = _registry()
    replayed: list[str] = []

    def _record(name):
        def _importer(_payload):
            replayed.append(name)
            return _empty_report()

        return _importer

    for name in ("first", "second", "third"):
        registry.register_section(name, dict, _record(name))
    _install(monkeypatch, registry=registry)

    document = {"version": 1, "sections": {"first": {}, "second": {}, "third": {}}}
    data = _json(await import_backup(_post_req({"document": document, "sections": ["third", "first", "second"]})))[
        "data"
    ]
    assert data["ok"] is True
    assert replayed == ["first", "second", "third"]


async def test_import_still_reports_an_unregistered_section_named_out_of_order(monkeypatch):
    # Reordering must not lose a name this host does not register: it still gets its own
    # report entry rather than being dropped by the ordering pass.
    registry = _registry()
    registry.register_section("known", dict, lambda _payload: _empty_report())
    _install(monkeypatch, registry=registry)

    document = {"version": 1, "sections": {"ghost": {}, "known": {}}}
    data = _json(await import_backup(_post_req({"document": document, "sections": ["ghost", "known"]})))["data"]
    assert data["ok"] is False
    assert "unknown section" in data["sections"]["ghost"]["errors"][0]
    assert data["sections"]["known"]["errors"] == []


# -- a plugin-registered section flows through BOTH doors ---------------------


async def test_plugin_section_round_trips_through_both_doors(monkeypatch):
    store: dict = {"value": "seeded"}
    registry = _registry()

    def _exporter():
        return dict(store)

    def _importer(payload):
        store.clear()
        store.update(payload)
        return {"created": 0, "updated": 1, "skipped": 0, "errors": []}

    registry.register_section("demo_plugin", _exporter, _importer)
    _install(monkeypatch, registry=registry)

    # The plugin section appears in the live registry listing.
    names = {s["name"] for s in _json(await list_sections(_get_req()))["data"]}
    assert "demo_plugin" in names

    doc = _json(await export_backup(_post_req({"sections": ["demo_plugin"]})))
    assert doc["sections"]["demo_plugin"] == {"value": "seeded"}

    store["value"] = "changed"  # wipe/mutate
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["demo_plugin"]})))["data"]
    assert data["ok"] is True
    assert data["sections"]["demo_plugin"] == {"created": 0, "updated": 1, "skipped": 0, "errors": []}
    assert store == {"value": "seeded"}


# -- env absent, existing-row updates, per-token failure ---------------------


async def test_export_env_absent_returns_empty(monkeypatch):
    cm = _FakeConfigManager(env=None, manifest={"mcp": []})
    _install(monkeypatch, config=SimpleNamespace(config_manager=cm))
    doc = _json(await export_backup(_post_req({"sections": ["env"]})))
    # No env file yet is a normal empty state — the section exports {}.
    assert doc["sections"]["env"] == {}


async def test_import_env_no_existing_counts_all_created(monkeypatch):
    cm = _FakeConfigManager(env=None)  # read_env raises FileNotFoundError
    _install(monkeypatch, config=SimpleNamespace(config_manager=cm))
    document = {"version": 1, "sections": {"env": {"A": "1", "B": "2"}}}
    data = _json(await import_backup(_post_req({"document": document, "sections": ["env"]})))["data"]
    assert data["ok"] is True
    # With no existing env, every key is a create (none updated).
    section = data["sections"]["env"]
    assert {k: section[k] for k in ("created", "updated", "skipped", "errors")} == {
        "created": 2,
        "updated": 0,
        "skipped": 0,
        "errors": [],
    }
    assert section["fanout"]["mode"] == "local-only"
    assert cm._env == {"A": "1", "B": "2"}


async def test_import_sub_mcp_existing_slug_updates(monkeypatch):
    # The slug already exists in the DURABLE store, so the import is an update.
    from tai42_contract.sub_mcp import RouteConfig

    from tai42_skeleton.sub_mcp import store as sub_mcp_store

    seeded = sub_mcp_store.InMemorySubMcpStore()
    await seeded.save_route("slug1", RouteConfig(tools=["old"], transport="http"))
    monkeypatch.setattr(sub_mcp_store, "_IN_MEMORY_STORE", seeded)
    router = _FakeSubAppRouter(routes={"slug1": _FakeRouteConfig(["old"], "http")})
    _install(monkeypatch, sub_app=SimpleNamespace(mcp_sub_app_router=router))
    document = {"version": 1, "sections": {"sub_mcp": {"slug1": {"tools": ["new"], "transport": "http"}}}}
    data = _json(await import_backup(_post_req({"document": document, "sections": ["sub_mcp"]})))["data"]
    assert data["ok"] is True
    assert data["sections"]["sub_mcp"] == {"created": 0, "updated": 1, "skipped": 0, "errors": []}
    updated = await seeded.get_route("slug1")
    assert updated is not None
    assert updated.tools == ["new"]
    assert router.routes["slug1"].tools == ["new"]


async def test_import_sub_mcp_malformed_entry_is_skipped_not_fatal(monkeypatch):
    from tai42_skeleton.sub_mcp import store as sub_mcp_store

    monkeypatch.setattr(sub_mcp_store, "_IN_MEMORY_STORE", sub_mcp_store.InMemorySubMcpStore())
    router = _FakeSubAppRouter(routes={})
    _install(monkeypatch, sub_app=SimpleNamespace(mcp_sub_app_router=router))
    # ``bad`` is missing its ``tools`` key (malformed entry); ``bad/slug`` is a
    # well-formed entry whose SLUG is invalid (the service rejects it before its
    # store write); ``good`` is fully valid. Each per-slug failure is recorded, and
    # the restore still processes ``good``.
    document = {
        "version": 1,
        "sections": {
            "sub_mcp": {
                "bad": {"transport": "http"},
                "bad/slug": {"tools": ["t"], "transport": "http"},
                "good": {"tools": ["t"], "transport": "http"},
            }
        },
    }
    data = _json(await import_backup(_post_req({"document": document, "sections": ["sub_mcp"]})))["data"]
    # Per-slug errors make the overall import ``ok=False``, but they do NOT abort the
    # restore — ``good`` is still created and persisted.
    assert data["ok"] is False
    section = data["sections"]["sub_mcp"]
    assert section["created"] == 1
    assert section["skipped"] == 2
    assert any("bad" in err and "malformed" in err for err in section["errors"])
    assert any("bad/slug" in err and "must match" in err for err in section["errors"])
    assert "good" in router.routes
    assert "bad" not in router.routes
    # The invalid slug never reached the durable store.
    assert await sub_mcp_store._IN_MEMORY_STORE.get_route("bad/slug") is None
    assert await sub_mcp_store._IN_MEMORY_STORE.get_route("good") is not None


async def test_import_webhooks_existing_name_updates(monkeypatch, execution_gate_off):
    from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
    from tai42_skeleton.hooks.settings import HooksSettings

    manager = InMemoryHooksManager(HooksSettings())
    await manager.register(
        HookParams(name="h1", topic="t1", tool="mytool", execution_key="k-fire", execution_key_fingerprint="fp-fire")
    )
    monkeypatch.setattr("tai42_skeleton.hooks.cache.get_hooks_manager", lambda: manager)
    _install(monkeypatch)

    document = {
        "version": 1,
        "sections": {
            "webhooks": [
                HookParams(
                    name="h1", topic="t2", tool="mytool", execution_key="k-fire", execution_key_fingerprint="fp-fire"
                ).model_dump(mode="json")
            ]
        },
    }
    data = _json(await import_backup(_post_req({"document": document, "sections": ["webhooks"]})))["data"]
    assert data["ok"] is True
    # ``h1`` already existed, so the re-register is an update.
    assert data["sections"]["webhooks"] == {"created": 0, "updated": 1, "skipped": 0, "errors": []}


async def test_import_webhooks_keyless_record_makes_the_import_report_failure(monkeypatch, execution_gate_off):
    # A keyless hook is refused PER RECORD; any per-record error forces ok=false, and
    # the record beside it still imports.
    from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
    from tai42_skeleton.hooks.settings import HooksSettings

    manager = InMemoryHooksManager(HooksSettings())
    monkeypatch.setattr("tai42_skeleton.hooks.cache.get_hooks_manager", lambda: manager)
    _install(monkeypatch)

    document = {
        "version": 1,
        "sections": {
            "webhooks": [
                {"name": "keyless", "topic": "t1", "tool": "mytool"},
                HookParams(
                    name="bound", topic="t1", tool="mytool", execution_key="k-fire", execution_key_fingerprint="fp-fire"
                ).model_dump(mode="json"),
            ]
        },
    }
    data = _json(await import_backup(_post_req({"document": document, "sections": ["webhooks"]})))["data"]
    assert data["ok"] is False
    section = data["sections"]["webhooks"]
    assert section["created"] == 1
    assert section["skipped"] == 1
    assert any("keyless" in err and "execution_key" in err for err in section["errors"])
    assert set(await manager.list_hooks()) == {"bound"}


async def test_import_webhooks_uncompilable_jq_record_is_per_record_not_a_torn_import(monkeypatch, execution_gate_off):
    # A non-compiling inline jq is refused PER RECORD (reported, ok=false) and the
    # records on EITHER side still import — an abort would leave earlier hooks written.
    from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
    from tai42_skeleton.hooks.settings import HooksSettings

    manager = InMemoryHooksManager(HooksSettings())
    monkeypatch.setattr("tai42_skeleton.hooks.cache.get_hooks_manager", lambda: manager)
    _install(monkeypatch)

    document = {
        "version": 1,
        "sections": {
            "webhooks": [
                HookParams(
                    name="first", topic="t1", tool="mytool", execution_key="k-fire", execution_key_fingerprint="fp-fire"
                ).model_dump(mode="json"),
                HookParams(
                    name="broken",
                    topic="t1",
                    tool="mytool",
                    execution_key="k-fire",
                    execution_key_fingerprint="fp-fire",
                    condition=".foo | (",
                ).model_dump(mode="json"),
                HookParams(
                    name="last", topic="t1", tool="mytool", execution_key="k-fire", execution_key_fingerprint="fp-fire"
                ).model_dump(mode="json"),
            ]
        },
    }
    data = _json(await import_backup(_post_req({"document": document, "sections": ["webhooks"]})))["data"]
    assert data["ok"] is False
    section = data["sections"]["webhooks"]
    assert section["created"] == 2
    assert section["skipped"] == 1
    assert any("broken" in err and "not valid jq" in err for err in section["errors"])
    assert set(await manager.list_hooks()) == {"first", "last"}


async def test_import_templates_existing_path_updates(monkeypatch):
    manager = _FakeResourceManager({"greeting.j2": "old"})
    _install(monkeypatch, storage=SimpleNamespace(resource_manager=manager))
    document = {"version": 1, "sections": {"templates": {"greeting.j2": "new"}}}
    data = _json(await import_backup(_post_req({"document": document, "sections": ["templates"]})))["data"]
    assert data["ok"] is True
    # The path already existed, so the upload is an update.
    assert data["sections"]["templates"] == {"created": 0, "updated": 1, "skipped": 0, "errors": []}
    assert manager._templates == {"greeting.j2": "new"}


async def test_import_access_control_scope_failure_is_per_token_skip(monkeypatch):
    from tai42_skeleton.access_control import management
    from tai42_skeleton.access_control import store as store_module
    from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(FakeAccessControlPg()))
    target = FakeRedis()
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(target))
    _install(monkeypatch)

    # A token referencing a scope that was never provisioned: ``add_user_api_key``
    # raises ValueError, surfaced as a loud per-token skip — the section stays not-ok
    # and no key is minted.
    document = {
        "version": 1,
        "sections": {
            "access_control": {
                "scopes": {},
                "patterns": {},
                "tokens": [{"user_id": "u1", "description": "d", "scopes": ["ghost-scope"]}],
            }
        },
    }
    data = _json(await import_backup(_post_req({"document": document, "sections": ["access_control"]})))["data"]
    assert data["ok"] is False
    report = data["sections"]["access_control"]
    assert report["skipped"] == 1
    assert report["created"] == 0
    assert report["new_api_keys"] == []
    assert any("u1" in err for err in report["errors"])


async def test_versioned_documents_section_round_trips_through_router(monkeypatch):
    from contextlib import asynccontextmanager

    from tai42_kit.clients.impl.postgres import PostgresClient

    import tai42_skeleton.versioning.backup as versioning_backup
    from tests.versioning.test_backup import _FakeVersioningBackupPg, _seed

    def _ctx_for(fake):
        @asynccontextmanager
        async def _ctx(client_cls, settings=None, **kwargs):
            assert client_cls is PostgresClient
            yield fake

        return _ctx

    source = _FakeVersioningBackupPg()
    _seed(source)
    monkeypatch.setattr(versioning_backup, "client_ctx", _ctx_for(source))
    _install(monkeypatch)

    # Export the whole kind-agnostic store through the section exporter.
    doc = _json(await export_backup(_post_req({"sections": ["versioned_documents"]})))
    section = doc["sections"]["versioned_documents"]
    assert {d["name"] for d in section["documents"]} == {"wv", "guard"}
    assert len(section["versions"]) == 6

    # Restore the exported payload into a fresh store through the section importer.
    target = _FakeVersioningBackupPg()
    monkeypatch.setattr(versioning_backup, "client_ctx", _ctx_for(target))
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["versioned_documents"]})))["data"]
    assert data["ok"] is True
    assert data["sections"]["versioned_documents"]["created"] == 3
    assert {d["name"] for d in target.documents} == {"wv", "guard"}
