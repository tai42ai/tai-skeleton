"""Role management: the admin-only CRUD/version ops, validate-before-persist, the
un-lockout guards, LIVE grant propagation + the version-keyed cache, keys-inherit-owner,
the audit trail, and the admin discriminator never routing through condition_id.
"""

from __future__ import annotations

import pytest
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.models import AccessPolicy

import tai42_skeleton.operations.roles as roles_ops
import tai42_skeleton.versioning as versioning_module
from tai42_skeleton.access_control import management
from tai42_skeleton.access_control import role_grants as role_grants_module
from tai42_skeleton.access_control.role_gate import DenialCause, reset_route_index, resolve_route_meta
from tai42_skeleton.access_control.role_grants import resolve_role_grants, role_level_decision
from tai42_skeleton.access_control.roles import (
    ROLE_POINTER_KEY,
    apply_role,
    grantable_feature_tags,
    role_store,
    seed_default_roles,
)
from tai42_skeleton.operations.errors import BadRequestError, ConflictError, ForbiddenError, NotFoundError

from .conftest import FakeAccessControlPg, FakeRedis, make_client_ctx
from .test_policy_store import _MemStore


@pytest.fixture
def mem() -> _MemStore:
    return _MemStore()


@pytest.fixture(autouse=True)
def _wire(monkeypatch, mem: _MemStore) -> None:
    monkeypatch.setattr(versioning_module, "versioned_store", lambda: mem)
    monkeypatch.setattr(versioning_module, "versioned_store_configured", lambda: True)
    # A fresh grant cache + route index per test so a prior test's cache never leaks.
    role_grants_module.reset_role_grants_cache()
    reset_route_index()


@pytest.fixture
def redis_mgmt(monkeypatch) -> FakeRedis:
    fake = FakeRedis(strings={})
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(fake))
    return fake


def _admin_caller(monkeypatch) -> None:
    from tai42_skeleton.operations._authority import Caller

    caller = Caller(caller_id="root", policy=AccessPolicy(scopes=["*"]), is_admin=True, owner_claim=None)

    async def _resolve():
        return caller

    monkeypatch.setattr(roles_ops, "resolve_caller", _resolve)


def _nonadmin_caller(monkeypatch) -> None:
    from tai42_skeleton.operations._authority import Caller

    caller = Caller(caller_id="ed", policy=AccessPolicy(scopes=["*"]), is_admin=False, owner_claim=None)

    async def _resolve():
        return caller

    monkeypatch.setattr(roles_ops, "resolve_caller", _resolve)


def _grantable_tag() -> str:
    return sorted(grantable_feature_tags())[0]


# -- store CRUD round-trip ---------------------------------------------------


async def test_management_crud_roundtrip(mem, pg: FakeAccessControlPg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    tag = _grantable_tag()

    created = await roles_ops.create_role("ops", "ops role", "editor", {tag: "read"})
    assert created["name"] == "ops"
    assert created["grants"] == {tag: "read"}
    assert created["condition"] is not None
    assert created["allow_all"] is False

    names = {r["name"] for r in await role_store().list_roles()}
    assert names == {"admin", "editor", "viewer", "ops"}

    edited = await roles_ops.update_role("ops", {tag: "write"}, "edited")
    assert edited["grants"] == {tag: "write"}
    assert edited["description"] == "edited"

    history = await roles_ops.list_role_versions("ops")
    assert len(history["versions"]) == 2  # create + edit

    rolled = await roles_ops.rollback_role("ops", 1)
    assert rolled["grants"] == {tag: "read"}  # back to the create version

    deleted = await roles_ops.delete_role("ops")
    assert deleted == {"name": "ops", "deleted": True}
    assert "ops" not in {r["name"] for r in await role_store().list_roles()}


async def test_create_duplicate_conflicts(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    with pytest.raises(ConflictError):
        await roles_ops.create_role("editor", "dup", "editor", {})


async def test_edit_and_delete_unknown_404(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    with pytest.raises(NotFoundError):
        await roles_ops.update_role("nope", {}, None)
    with pytest.raises(NotFoundError):
        await roles_ops.delete_role("nope")


# -- validate-before-persist -------------------------------------------------


async def test_create_rejects_nonexistent_or_fenced_tag(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    with pytest.raises(BadRequestError):
        await roles_ops.create_role("ops", "x", "editor", {"does-not-exist": "read"})


async def test_create_rejects_bad_base_tier(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    with pytest.raises(BadRequestError):
        await roles_ops.create_role("ops", "x", "admin", {})  # admin base is reserved
    with pytest.raises(BadRequestError):
        await roles_ops.create_role("ops", "x", "nonsense", {})


# -- reserved admin / block-downgrade ----------------------------------------


async def test_reserved_admin_cannot_be_created_edited_deleted(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    with pytest.raises(ForbiddenError):
        await roles_ops.create_role("admin", "x", "editor", {})
    with pytest.raises(ForbiddenError):
        await roles_ops.update_role("admin", {_grantable_tag(): "read"}, None)
    with pytest.raises(ForbiddenError):
        await roles_ops.delete_role("admin")
    with pytest.raises(ForbiddenError):
        await roles_ops.rollback_role("admin", 1)


async def test_non_admin_caller_denied_every_mutation(mem, pg, redis_mgmt, monkeypatch):
    _nonadmin_caller(monkeypatch)
    await seed_default_roles()
    with pytest.raises(ForbiddenError):
        await roles_ops.create_role("ops", "x", "editor", {})
    with pytest.raises(ForbiddenError):
        await roles_ops.update_role("editor", {}, None)
    with pytest.raises(ForbiddenError):
        await roles_ops.delete_role("editor")


# -- delete-of-assigned rejected --------------------------------------------


async def test_delete_of_assigned_role_rejected(mem, pg: FakeAccessControlPg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    await roles_ops.create_role("ops", "x", "editor", {})
    await apply_role("bob", "ops")  # bob now holds the LIVE pointer to "ops"
    with pytest.raises(ConflictError, match="assigned"):
        await roles_ops.delete_role("ops")


async def test_delete_role_counts_assignments_before_opening_the_tx(mem, pg, redis_mgmt, monkeypatch):
    # The assigned-role guard reads a DIFFERENT store (access_control_store) that cannot ride
    # the versioned-store tx, so it MUST run before the transaction opens — inside it would
    # pin a SECOND pooled connection for the tx's whole duration (a pool-exhaustion deadlock
    # risk under concurrent deletes). Pin the ordering: the count precedes the tx.
    _admin_caller(monkeypatch)
    await seed_default_roles()
    await roles_ops.create_role("ops", "x", "editor", {})  # unassigned, so the delete proceeds

    events: list[str] = []
    orig_count = roles_ops.access_control_store().count_policies_with_role

    async def _spy_count(role_name, pointer_key):
        events.append("count")
        return await orig_count(role_name, pointer_key)

    class _Proxy:
        count_policies_with_role = staticmethod(_spy_count)

    monkeypatch.setattr(roles_ops, "access_control_store", lambda: _Proxy())

    orig_tx = mem.transaction
    monkeypatch.setattr(mem, "transaction", lambda: (events.append("tx-open"), orig_tx())[1])

    await roles_ops.delete_role("ops")
    # The cross-store count ran BEFORE the versioned-store transaction was opened.
    assert events == ["count", "tx-open"]
    assert "ops" not in {r["name"] for r in await role_store().list_roles()}


# -- multiple admins ---------------------------------------------------------


async def test_multiple_admins_and_admin_pointer_absent(mem, pg: FakeAccessControlPg, redis_mgmt, monkeypatch):
    await seed_default_roles()
    await apply_role("alice", "admin")
    await apply_role("bob", "admin")
    for who in ("alice", "bob"):
        body = pg.policy_body(who)
        assert body["scopes"] == ["*"]
        assert body["condition"] is None
        assert body["condition_id"] is None  # admin is never routed through condition_id
        assert ROLE_POINTER_KEY not in (body.get("policy_data") or {})  # admin carries no pointer


# -- the role pointer is a separate field, never condition_id ----------------


def test_role_pointer_is_not_condition_id():
    assert ROLE_POINTER_KEY == "role"
    assert ROLE_POINTER_KEY != "condition_id"


async def test_apply_role_refuses_conditionless_non_admin_role(mem, pg, redis_mgmt, monkeypatch):
    # Fail-closed guard on the admin discriminator: a non-allow_all role with no base-tier
    # condition would assign a condition-free ["*"] policy that is_admin_policy misreads as
    # full admin. apply_role refuses it loudly rather than silently escalating.
    await mem.create(
        "role",
        "broken",
        {"name": "broken", "description": "x", "scopes": ["*"], "grants": {}, "condition": None, "allow_all": False},
    )
    with pytest.raises(ValueError, match="admin discriminator misreads"):
        await apply_role("bob", "broken")


async def test_editor_pointer_lives_in_policy_data_not_condition_id(mem, pg, redis_mgmt, monkeypatch):
    await seed_default_roles()
    await apply_role("bob", "editor")
    body = pg.policy_body("bob")
    assert body["policy_data"][ROLE_POINTER_KEY] == "editor"
    assert body["condition_id"] is None  # the pointer never populates condition_id


# -- LIVE propagation + the version-keyed grant cache ------------------------


async def test_live_edit_propagates_via_version_keyed_cache(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    tag = _grantable_tag()
    await roles_ops.create_role("ops", "x", "editor", {tag: "read"})

    grants_v1 = await resolve_role_grants("ops", 1)
    assert grants_v1[tag] == "read"

    await roles_ops.update_role("ops", {tag: "write"}, None)

    # A new version key sees the edited grants; the old version key still serves the
    # cached old grants (version-keyed cache bust).
    grants_v2 = await resolve_role_grants("ops", 2)
    assert grants_v2[tag] == "write"
    assert (await resolve_role_grants("ops", 1))[tag] == "read"


async def test_missing_role_denies_fail_closed(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    # A policy pointing at a role that does not exist is denied on a grantable route.
    policy = AccessPolicy(scopes=["*"], condition="true", policy_data={ROLE_POINTER_KEY: "ghost"})
    meta = _a_grantable_read_route()
    allowed, cause = await role_level_decision(policy, None, meta.path, "GET", 1)
    assert allowed is False
    assert cause is DenialCause.LEVEL_MISS


async def test_role_level_decision_allows_a_method_less_non_http_scope(mem, pg, redis_mgmt):
    # A websocket/MCP scope legitimately carries NO method. The per-tag HTTP-route fence
    # governs HTTP routes only — every fenced/secret route IS an HTTP route that always
    # arrives WITH its method — so a method-less scope is not an HTTP-route request and the
    # fence cannot apply: the scope + jq base govern. This is a deliberate non-HTTP branch,
    # not an allow-by-omission (a real registered fenced route can never reach here without
    # a method). Passing a real fenced PATH with method=None still allows, proving the
    # branch keys on the absent method, not the path.
    await seed_default_roles()
    editor = AccessPolicy(scopes=["*"], condition="editorbase", policy_data={ROLE_POINTER_KEY: "editor"})
    allowed, cause = await role_level_decision(editor, None, "/api/marketplace/install", None, 1)
    assert allowed is True
    assert cause is None


async def test_role_level_decision_does_not_act_on_a_path_with_no_registered_route(mem, pg, redis_mgmt):
    # Caller-REQUESTED paths (SPA shell, probes) carry no registered route, and the scope
    # layer plus jq base govern there. Not an allow-by-omission: a registered route always
    # resolves back to itself (the boot audit refuses to start otherwise), and only a
    # SYNTHESIZED path can miss the route it should have hit — which is why the tool edge
    # pins the operation's own route instead of resolving one here.
    await seed_default_roles()
    editor = AccessPolicy(scopes=["*"], condition="editorbase", policy_data={ROLE_POINTER_KEY: "editor"})
    assert resolve_route_meta("/studio/settings", "POST") is None
    allowed, cause = await role_level_decision(editor, None, "/studio/settings", "POST", 1)
    assert allowed is True
    assert cause is None


# -- keys inherit the owner's role grant map ---------------------------------


def _a_grantable_read_route():
    from tai42_skeleton.app.route_registry import load_all_routes

    for meta in load_all_routes():
        if meta.authed and meta.action == "read" and "GET" in meta.methods and "{" not in meta.path:
            return meta
    raise AssertionError("no concrete grantable GET route found")


def _a_grantable_write_route():
    from tai42_skeleton.app.route_registry import load_all_routes

    for meta in load_all_routes():
        if meta.authed and meta.action == "write" and "{" not in meta.path:
            return meta
    raise AssertionError("no concrete grantable write route found")


async def test_owned_key_inherits_owner_role(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    read_route = _a_grantable_read_route()
    # A viewer owner: read on every grantable tag. An owned key (no role of its own)
    # inherits the owner's viewer grant map.
    owner = AccessPolicy(scopes=["*"], condition="viewerbase", policy_data={ROLE_POINTER_KEY: "viewer"})
    key = AccessPolicy(scopes=["*"], policy_data={OWNER_USER_ID_CLAIM: "owner1"})

    allowed, _ = await role_level_decision(key, owner, read_route.path, "GET", 1)
    assert allowed is True  # inherits the owner's viewer read grant

    # A fenced route stays admin-only even for the owned key.
    allowed, cause = await role_level_decision(key, owner, "/api/marketplace/install", "POST", 1)
    assert allowed is False
    assert cause is DenialCause.HARD_FENCE


async def test_owned_key_of_admin_owner_is_hard_fenced(mem, pg, redis_mgmt, monkeypatch):
    await seed_default_roles()
    # The fence keys on the CALLER's own admin verdict, and an owner-claim-bearing policy is
    # never the admin principal: an admin cannot delegate fence access via an owned key.
    owner = AccessPolicy(scopes=["*"], policy_data={})
    key = AccessPolicy(scopes=["*"], policy_data={OWNER_USER_ID_CLAIM: "owner1"})
    allowed, cause = await role_level_decision(key, owner, "/api/marketplace/install", "POST", 1)
    assert allowed is False
    assert cause is DenialCause.HARD_FENCE


async def test_admin_owned_scoped_key_cannot_reach_backup_import(mem, pg, redis_mgmt):
    await seed_default_roles()
    # The common delegated-key shape (admin-owned, scoped) is hard-fenced at
    # ``/api/backup/import``, so the restore writers are reached only by an admin.
    owner = AccessPolicy(scopes=["*"], policy_data={})
    key = AccessPolicy(scopes=["backup"], policy_data={OWNER_USER_ID_CLAIM: "owner1"})
    allowed, cause = await role_level_decision(key, owner, "/api/backup/import", "POST", 1)
    assert allowed is False
    assert cause is DenialCause.HARD_FENCE


# -- defaults reproduce today's reach (structural) ---------------------------


async def test_default_grant_maps_cover_all_grantable_tags(mem, pg, redis_mgmt, monkeypatch):
    await seed_default_roles()
    grantable = grantable_feature_tags()
    roles = {r["name"]: r for r in await role_store().list_roles()}
    assert roles["editor"]["grants"] == dict.fromkeys(grantable, "write")
    assert roles["viewer"]["grants"] == dict.fromkeys(grantable, "read")
    assert roles["admin"]["allow_all"] is True
    assert roles["admin"]["grants"] == {}


async def test_new_tag_is_fail_closed_none(mem, pg, redis_mgmt, monkeypatch):
    await seed_default_roles()
    # A brand-new tag is absent from every non-admin grant map → level none → the per-tag
    # check denies it (the grant_map_admits absent-tag rule).
    from tai42_skeleton.access_control.role_gate import grant_map_admits

    meta = _a_grantable_read_route()
    allowed, cause = grant_map_admits(meta, "GET", {"a-brand-new-tag": "write"})
    assert allowed is False
    assert cause is DenialCause.LEVEL_MISS


# -- audit trail -------------------------------------------------------------


async def test_audit_records_actor_and_before_after(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    tag = _grantable_tag()
    await roles_ops.create_role("ops", "x", "editor", {tag: "read"})
    await roles_ops.update_role("ops", {tag: "write"}, None)

    history = await roles_ops.list_role_versions("ops")
    events = history["audit"]
    assert [e["body"]["action"] for e in events] == ["create", "edit"]
    assert all(e["body"]["actor"] == "root" for e in events)
    edit_event = events[1]["body"]
    assert edit_event["before"]["grants"] == {tag: "read"}
    assert edit_event["after"]["grants"] == {tag: "write"}


# -- distinguishable denials -------------------------------------------------


async def test_denial_causes_are_internally_distinguishable(mem, pg, redis_mgmt, monkeypatch):
    await seed_default_roles()
    editor = AccessPolicy(scopes=["*"], condition="editorbase", policy_data={ROLE_POINTER_KEY: "editor"})
    # A fenced route → HARD_FENCE.
    _, fence_cause = await role_level_decision(editor, None, "/api/marketplace/install", "POST", 1)
    assert fence_cause is DenialCause.HARD_FENCE
    # A grantable route on a tag the role does not hold → LEVEL_MISS.
    from tai42_skeleton.access_control.role_gate import grant_map_admits

    meta = _a_grantable_read_route()
    _, level_cause = grant_map_admits(meta, "GET", {})  # empty map → constant deny
    assert level_cause is DenialCause.LEVEL_MISS
    assert DenialCause.HARD_FENCE is not DenialCause.LEVEL_MISS is not DenialCause.SCOPE_MISS


# -- admin discriminator guard: condition_id is never the escape hatch --------


async def test_apply_role_refuses_conditionless_role_even_with_condition_id_set(mem, pg, redis_mgmt):
    # apply_role hardcodes the WRITTEN condition_id to None, so a role body carrying
    # condition=None but a stored condition_id must STILL be refused: the policy it would
    # write is a condition-free ["*"] the admin discriminator misreads as full admin. The
    # guard fires on the value actually written, not the role's stored condition_id.
    await mem.create(
        "role",
        "sneaky",
        {
            "name": "sneaky",
            "description": "x",
            "scopes": ["*"],
            "grants": {},
            "condition": None,
            "condition_id": "cid",
            "allow_all": False,
        },
    )
    with pytest.raises(ValueError, match="admin discriminator misreads"):
        await apply_role("bob", "sneaky")
    assert pg.policy("bob") is None  # nothing minted


# -- the admin control plane is HARD-FENCED to editor/viewer -----------------

_ADMIN_CONTROL_ROUTES = [
    ("/api/auth/roles", "GET"),
    ("/api/auth/roles", "POST"),
    ("/api/auth/roles/x", "PUT"),
    ("/api/auth/roles/x", "DELETE"),
    ("/api/auth/roles/x/versions", "GET"),
    ("/api/auth/roles/x/rollback", "POST"),
    ("/api/auth/api-keys/u/policy/versions", "GET"),
    ("/api/auth/api-keys/u/policy/rollback", "POST"),
]


async def test_admin_control_routes_hard_fenced_to_editor_and_viewer(mem, pg, redis_mgmt):
    # The role/policy-administration routes are fenced/secret, so no per-tag level opens
    # them — every non-admin (editor + viewer) is a HARD_FENCE deny even though both carry an
    # access-control grant. The action-class fence, not the shared tag, is the boundary.
    from tai42_skeleton.access_control.role_gate import grant_map_admits, resolve_route_meta

    await seed_default_roles()
    roles = {r["name"]: r for r in await role_store().list_roles()}
    for path, method in _ADMIN_CONTROL_ROUTES:
        meta = resolve_route_meta(path, method)
        assert meta is not None, f"{method} {path} did not resolve"
        assert meta.action in ("fenced", "secret")
        for role in ("editor", "viewer"):
            allowed, cause = grant_map_admits(meta, method, roles[role]["grants"])
            assert allowed is False
            assert cause is DenialCause.HARD_FENCE


async def test_access_control_tag_stays_grantable_for_self_service(mem, pg, redis_mgmt):
    # The access-control tag remains grantable via the self-service key/scope routes (own-key
    # management), so editor/viewer keep it in their seeded grant maps and a self-service key
    # route is NOT hard-fenced — the fence is scoped to the admin routes by action class.
    from tai42_skeleton.access_control.role_gate import grant_map_admits, resolve_route_meta

    assert "access-control" in grantable_feature_tags()
    await seed_default_roles()
    roles = {r["name"]: r for r in await role_store().list_roles()}
    assert roles["editor"]["grants"]["access-control"] == "write"
    meta = resolve_route_meta("/api/auth/api-keys", "POST")
    assert meta is not None
    allowed, cause = grant_map_admits(meta, "POST", roles["editor"]["grants"])
    assert allowed is True
    assert cause is None


async def test_access_control_tag_still_grantable_via_a_non_admin_editor_read(mem, pg, redis_mgmt):
    # The access-control tag is grantable via its self-service reads (GET /scopes, GET /me,
    # ...), so a non-admin editor reaches a self-service read through its seeded
    # access-control: write grant. (The scope + public-route MUTATIONS are action=write but
    # stay denied to editor/viewer by the base-tier /api/auth jq ceiling, not the fence.)
    from tai42_skeleton.access_control.role_gate import grant_map_admits, resolve_route_meta

    assert "access-control" in grantable_feature_tags()
    await seed_default_roles()
    roles = {r["name"]: r for r in await role_store().list_roles()}
    meta = resolve_route_meta("/api/auth/scopes", "GET")  # list_scopes — a self-service read
    assert meta is not None
    assert meta.action == "read"
    allowed, cause = grant_map_admits(meta, "GET", roles["editor"]["grants"])
    assert allowed is True
    assert cause is None


def test_scope_and_public_route_mutations_are_grantable_writes() -> None:
    # The scope + public-route mutations are grantable ``write`` routes: no action-class
    # fence. A seeded editor/viewer is still denied them, but by the base-tier /api/auth jq
    # ceiling (pinned in the editor/viewer jq matrices), not the fence. Pin the write class +
    # authed=True here so a re-fence regression is caught.
    from tai42_skeleton.app.route_registry import load_all_routes

    write_mutations = {
        ("/api/auth/scopes", "POST"),
        ("/api/auth/scopes/urls", "DELETE"),
        ("/api/auth/scopes/{scope_id}", "DELETE"),
        ("/api/auth/public-routes", "POST"),
        ("/api/auth/public-routes", "DELETE"),
    }
    by_key = {(meta.path, method): meta for meta in load_all_routes() for method in meta.methods}
    for key in write_mutations:
        meta = by_key.get(key)
        assert meta is not None, f"{key} not registered"
        assert meta.action == "write", f"{key} is {meta.action!r}, expected write"
        assert meta.authed is True, f"{key} must be authed=True"


# -- description-only edit keeps the grant map -------------------------------


async def test_description_only_edit_keeps_grants(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    tag = _grantable_tag()
    await roles_ops.create_role("ops", "x", "editor", {tag: "write"})
    edited = await roles_ops.update_role("ops", None, "new desc")  # grants=None → keep
    assert edited["grants"] == {tag: "write"}  # the grant map survived
    assert edited["description"] == "new desc"


async def test_edit_with_grants_replaces_them(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    tag = _grantable_tag()
    await roles_ops.create_role("ops", "x", "editor", {tag: "write"})
    edited = await roles_ops.update_role("ops", {tag: "read"}, None)
    assert edited["grants"] == {tag: "read"}


async def test_update_role_validates_before_persist(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    await roles_ops.create_role("ops", "x", "editor", {})
    with pytest.raises(BadRequestError):
        await roles_ops.update_role("ops", {"does-not-exist": "read"}, None)
    # The rejected edit never persisted a version.
    history = await roles_ops.list_role_versions("ops")
    assert len(history["versions"]) == 1  # only the create


# -- level_satisfies / grant_map_admits read<->write matrix ------------------


def test_level_satisfies_matrix():
    from tai42_skeleton.access_control.role_gate import level_satisfies

    assert level_satisfies("write", "read") is True
    assert level_satisfies("write", "write") is True
    assert level_satisfies("read", "read") is True
    assert level_satisfies("read", "write") is False
    assert level_satisfies("none", "read") is False
    assert level_satisfies("none", "write") is False


def test_grant_map_admits_read_write_matrix():
    from tai42_skeleton.access_control.role_gate import grant_map_admits

    read_meta = _a_grantable_read_route()
    write_meta = _a_grantable_write_route()
    rtag = read_meta.tags[0]
    wtag = write_meta.tags[0]
    rmethod = next(m for m in read_meta.methods if m in ("GET", "HEAD", "OPTIONS"))
    wmethod = next(m for m in write_meta.methods if m in ("POST", "PUT", "PATCH", "DELETE"))
    # a read-level grant admits a read route but is DENIED on a write route
    assert grant_map_admits(read_meta, rmethod, {rtag: "read"}) == (True, None)
    assert grant_map_admits(write_meta, wmethod, {wtag: "read"}) == (False, DenialCause.LEVEL_MISS)
    # a write-level grant admits BOTH a read and a write route
    assert grant_map_admits(read_meta, rmethod, {rtag: "write"}) == (True, None)
    assert grant_map_admits(write_meta, wmethod, {wtag: "write"}) == (True, None)
    # none / absent denies all
    assert grant_map_admits(read_meta, rmethod, {rtag: "none"}) == (False, DenialCause.LEVEL_MISS)
    assert grant_map_admits(read_meta, rmethod, {}) == (False, DenialCause.LEVEL_MISS)


# -- templated route resolution (single-segment + greedy :path) --------------


def test_resolve_templated_routes():
    from tai42_skeleton.access_control.role_gate import resolve_route_meta

    single = resolve_route_meta("/api/auth/roles/myrole/versions", "GET")
    assert single is not None
    assert single.path == "/api/auth/roles/{name}/versions"
    assert single.action == "secret"

    greedy = resolve_route_meta("/api/storage/resources/a/b/c/content", "GET")
    assert greedy is not None
    assert greedy.path == "/api/storage/resources/{resource_id:path}/content"


# -- delete + rollback audit before->after -----------------------------------


async def test_rollback_and_delete_audit_before_after(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    tag = _grantable_tag()
    await roles_ops.create_role("ops", "x", "editor", {tag: "read"})  # v1
    await roles_ops.update_role("ops", {tag: "write"}, None)  # v2
    await roles_ops.rollback_role("ops", 1)  # roll back to v1

    history = await roles_ops.list_role_versions("ops")
    events = [e["body"] for e in history["audit"]]
    assert [e["action"] for e in events] == ["create", "edit", "rollback"]
    rb = events[2]
    assert rb["actor"] == "root"
    assert rb["before"]["grants"] == {tag: "write"}  # the active body BEFORE the rollback
    assert rb["after"]["grants"] == {tag: "read"}  # the version rolled back to

    await roles_ops.delete_role("ops")
    # The role is gone, but its audit trail survives the hard delete (a separate kind), so
    # read it through the audit view directly.
    from tai42_skeleton.access_control.role_audit import role_audit

    delete_event = (await role_audit().list_events("ops"))[-1].body
    assert delete_event["action"] == "delete"
    assert delete_event["actor"] == "root"
    assert delete_event["before"]["grants"] == {tag: "read"}  # the body before delete
    assert delete_event["after"] is None


# -- the list-roles op is admin-only (op-level backstop) ---------------------


async def test_list_roles_op_requires_admin(mem, pg, redis_mgmt, monkeypatch):
    import tai42_skeleton.operations.api_keys as api_keys_ops
    from tai42_skeleton.operations._authority import Caller

    caller = Caller(caller_id="ed", policy=AccessPolicy(scopes=["*"]), is_admin=False, owner_claim=None)

    async def _resolve():
        return caller

    monkeypatch.setattr(api_keys_ops, "resolve_caller", _resolve)
    with pytest.raises(ForbiddenError):
        await api_keys_ops.list_roles()


# -- atomic role mutation + audit (the unit of work) -------------------------


async def test_role_change_and_audit_commit_together(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    await roles_ops.create_role("ops", "x", "editor", {})
    # Both the role and its audit event landed.
    assert ("role", "ops") in mem.docs
    history = await roles_ops.list_role_versions("ops")
    assert [e["body"]["action"] for e in history["audit"]] == ["create"]


async def test_audit_failure_rolls_back_the_role_mutation(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    from tai42_skeleton.access_control import role_audit as role_audit_module

    async def _boom(self, *args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(role_audit_module.RoleAuditView, "record", _boom)
    with pytest.raises(RuntimeError, match="audit down"):
        await roles_ops.create_role("ops", "x", "editor", {})
    # The audit append failed INSIDE the transaction, so the role write rolled back too:
    # no live role, no audit row — never a privilege change without its audit record.
    assert ("role", "ops") not in mem.docs
    assert ("role_audit", "ops") not in mem.docs


async def test_version_bump_only_after_commit(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    before = int(redis_mgmt._strings.get("ac:policy_version", "0"))
    await roles_ops.create_role("ops", "x", "editor", {})
    after = int(redis_mgmt._strings["ac:policy_version"])
    assert after == before + 1  # a successful commit bumps once

    # A mutation whose audit append fails rolls back and does NOT bump (the bump is
    # strictly after the commit, never before it).
    from tai42_skeleton.access_control import role_audit as role_audit_module

    async def _boom(self, *args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(role_audit_module.RoleAuditView, "record", _boom)
    with pytest.raises(RuntimeError, match="audit down"):
        await roles_ops.create_role("ops2", "x", "editor", {})
    assert int(redis_mgmt._strings["ac:policy_version"]) == after  # no bump on rollback


# -- the read-modify-write's locking read rides the transaction --------------


async def test_update_role_before_read_is_locked_inside_the_tx(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    tag = _grantable_tag()
    await roles_ops.create_role("ops", "x", "editor", {tag: "read"})

    calls: list[tuple[str, str, object, bool]] = []
    orig = mem.get_active_body

    async def _spy(kind, name, *, tx=None, for_update=False):
        calls.append((kind, name, tx, for_update))
        return await orig(kind, name, tx=tx, for_update=for_update)

    monkeypatch.setattr(mem, "get_active_body", _spy)

    edited = await roles_ops.update_role("ops", {tag: "write"}, None)
    assert edited["grants"] == {tag: "write"}

    # The role before-read rode the transaction (tx not None) AND took the row lock
    # (for_update=True): the read-modify-write's locking read happens INSIDE the tx, on the
    # tx connection, before the new body is computed — never an unlocked out-of-tx read.
    role_reads = [(tx, for_update) for kind, name, tx, for_update in calls if kind == "role" and name == "ops"]
    assert role_reads, "update_role never read the active role body"
    assert all(tx is not None for tx, _ in role_reads)
    assert any(for_update for _, for_update in role_reads)
    # Every get_active_body during the mutation rode the tx — none opened a second pooled
    # connection while the transaction held one (the pool-exhaustion guard).
    assert all(tx is not None for _kind, _name, tx, _fu in calls)


async def test_second_edit_reads_first_edits_committed_body_no_lost_update(mem, pg, redis_mgmt, monkeypatch):
    _admin_caller(monkeypatch)
    await seed_default_roles()
    tag = _grantable_tag()
    await roles_ops.create_role("ops", "x", "editor", {tag: "read"})  # v1
    await roles_ops.update_role("ops", {tag: "write"}, None)  # v2: before = v1 grants (read)
    await roles_ops.update_role("ops", None, "desc2")  # v3: description-only, keeps grants

    history = await roles_ops.list_role_versions("ops")
    audit = [e["body"] for e in history["audit"]]
    assert [e["action"] for e in audit] == ["create", "edit", "edit"]
    # The third edit's ``before`` is the SECOND edit's committed body (grants=write), proving
    # the before-read is the live current body captured inside the tx — not a stale snapshot
    # that would silently drop the second edit's grant change (a lost update).
    assert audit[2]["before"]["grants"] == {tag: "write"}
    assert audit[2]["after"]["grants"] == {tag: "write"}  # description-only edit kept the grants
