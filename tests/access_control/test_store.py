"""Behavior of :class:`PostgresAccessControlStore` against a stateful fake Postgres.

Covers every policy-store op — scope/route CRUD, the scope-strip cascade, the
public-marker semantics (public excluded from scope enumeration, ``remove_scope``
of the marker rejected, backup round-trips public mappings), the
cascade return shapes, policy body read/restore, and the OIDC-subject case
(identifiers with ``:``/``@``/unicode round-trip, since Postgres identities are
parameterized column values, not key-name fragments). Concurrency is proven by
transaction isolation (a fault mid-cascade rolls the whole mutation back), not a
watched set.
"""

from __future__ import annotations

import pytest
from psycopg.errors import UniqueViolation

from tai42_skeleton.access_control.settings import access_control_settings
from tai42_skeleton.access_control.store import access_control_store

from .conftest import FakeAccessControlPg

STORE = access_control_store
PUBLIC = access_control_settings().public_resource_id


# -- route / scope CRUD ------------------------------------------------------


async def test_add_url_to_scope_and_enumerate(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("scope-a", "/a")
    assert await STORE().get_all_existing_scopes() == {"/a": "scope-a"}
    assert pg.route("/a")["scope_id"] == "scope-a"


async def test_public_route_excluded_from_scope_enumeration(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("scope-a", "/a")
    await STORE().add_url_to_scope(PUBLIC, "/open")
    # The public marker names no scope, so a public route is filtered out of the
    # scope enumeration but present in the full route mapping a backup needs.
    assert await STORE().get_all_existing_scopes() == {"/a": "scope-a"}
    assert await STORE().get_all_route_mappings() == {"/a": "scope-a", "/open": PUBLIC}


async def test_add_url_normalizes_trailing_slash(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("scope-a", "/a/")
    assert pg.route("/a") is not None
    assert await STORE().fetch_route("/a") == "scope-a"


async def test_add_url_with_pattern_registers_dynamic_pattern(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("scope-a", "/orders/{id}", pattern=r"/orders/\d+")
    assert await STORE().get_all_existing_patterns() == {"/orders/{id}": r"/orders/\d+"}
    assert await STORE().fetch_dynamic_patterns() == {r"/orders/\d+": "/orders/{id}"}


async def test_repoint_url_replaces_scope_and_pattern(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("scope-a", "/x", pattern=r"/x/\d+")
    await STORE().add_url_to_scope("scope-b", "/x")
    # The upsert replaces both the scope and the (now dropped) pattern — no orphan.
    assert pg.route("/x")["scope_id"] == "scope-b"
    assert pg.route("/x")["pattern"] is None
    assert await STORE().get_all_existing_scopes() == {"/x": "scope-b"}
    assert await STORE().get_all_existing_patterns() == {}


async def test_fetch_route_unknown_is_none(pg: FakeAccessControlPg) -> None:
    assert await STORE().fetch_route("/nope") is None


# -- remove_url_from_scope ---------------------------------------------------


async def test_remove_unmapped_url_reports_not_existed(pg: FakeAccessControlPg) -> None:
    existed, affected = await STORE().remove_url_from_scope("/ghost")
    assert existed is False
    assert affected == []


async def test_remove_url_cascades_scope_out_of_token_policies(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("scope-a", "/a")
    pg.add_policy("u1", scopes=["scope-a", "keep"])
    existed, affected = await STORE().remove_url_from_scope("/a")
    assert existed is True
    # The scope lost its last url, so it is stripped from every token that held it,
    # and the committed body is returned for the caller's version record.
    assert affected == [
        (
            "u1",
            {"scopes": ["keep"], "policy_data": {}, "condition": None, "condition_id": None, "condition_kwargs": None},
        )
    ]
    assert pg.policy("u1")["scopes"] == ["keep"]


async def test_remove_url_keeps_scope_with_remaining_urls(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("scope-a", "/a")
    await STORE().add_url_to_scope("scope-a", "/b")
    pg.add_policy("u1", scopes=["scope-a"])
    existed, affected = await STORE().remove_url_from_scope("/a")
    assert existed is True
    # The scope still has /b, so no cascade fires and the token keeps the scope.
    assert affected == []
    assert pg.policy("u1")["scopes"] == ["scope-a"]


async def test_remove_public_url_skips_cascade(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope(PUBLIC, "/open")
    pg.add_policy("u1", scopes=[PUBLIC])
    existed, affected = await STORE().remove_url_from_scope("/open")
    assert existed is True
    assert affected == []
    # A public url owns no scope, so no token policy is touched.
    assert pg.policy("u1")["scopes"] == [PUBLIC]


# -- public route pins -------------------------------------------------------


async def test_pin_route_public_writes_marker(pg: FakeAccessControlPg) -> None:
    await STORE().pin_route_public("/open")
    assert pg.route("/open")["scope_id"] == PUBLIC
    # No scope-membership row for the marker — it is not a scope.
    assert await STORE().get_all_existing_scopes() == {}
    assert await STORE().get_public_route_pins() == ["/open"]


async def test_pin_route_public_repoints_off_scope_and_clears_pattern(pg: FakeAccessControlPg) -> None:
    # A previously scope-mapped url (with a dynamic pattern) is re-pointed OFF its
    # scope to the marker, and its prior pattern is cleared by the same upsert.
    await STORE().add_url_to_scope("scope-a", "/x", pattern=r"/x/\d+")
    await STORE().pin_route_public("/x")
    assert pg.route("/x")["scope_id"] == PUBLIC
    assert pg.route("/x")["pattern"] is None
    assert await STORE().get_all_existing_scopes() == {}
    assert await STORE().get_public_route_pins() == ["/x"]


async def test_pin_route_public_with_pattern_registers_it(pg: FakeAccessControlPg) -> None:
    await STORE().pin_route_public("/orders/{id}", pattern=r"/orders/\d+")
    assert pg.route("/orders/{id}")["scope_id"] == PUBLIC
    assert await STORE().fetch_dynamic_patterns() == {r"/orders/\d+": "/orders/{id}"}


async def test_pin_route_public_normalizes_trailing_slash(pg: FakeAccessControlPg) -> None:
    await STORE().pin_route_public("/open/")
    assert pg.route("/open") is not None
    assert await STORE().get_public_route_pins() == ["/open"]


async def test_pin_route_public_rejects_reserved_management_prefix(pg: FakeAccessControlPg) -> None:
    # The control plane must not be pinnable public. The sole public-pin writer rejects
    # a reserved-prefix url loudly and writes nothing — the invariant holds for every
    # caller (router, backup restore), not just the HTTP door.
    with pytest.raises(ValueError, match="reserved"):
        await STORE().pin_route_public("/api/auth/api-keys")
    with pytest.raises(ValueError, match="reserved"):
        await STORE().pin_route_public("/api/auth")
    assert pg.routes == []
    # A sibling prefix that only shares a string boundary is NOT reserved.
    await STORE().pin_route_public("/api/authorized")
    assert await STORE().get_public_route_pins() == ["/api/authorized"]


async def test_get_public_route_pins_returns_only_marker_urls_sorted(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("scope-a", "/a")
    await STORE().pin_route_public("/z")
    await STORE().pin_route_public("/m")
    assert await STORE().get_public_route_pins() == ["/m", "/z"]
    # Public pins are excluded from scope enumeration but present in the full mapping.
    assert await STORE().get_all_existing_scopes() == {"/a": "scope-a"}
    assert await STORE().get_all_route_mappings() == {"/a": "scope-a", "/m": PUBLIC, "/z": PUBLIC}


async def test_get_public_route_pins_empty(pg: FakeAccessControlPg) -> None:
    assert await STORE().get_public_route_pins() == []


async def test_unpin_public_route_removes_route_and_pattern(pg: FakeAccessControlPg) -> None:
    await STORE().pin_route_public("/open", pattern=r"/open/\d+")
    assert await STORE().unpin_public_route("/open") is True
    assert pg.route("/open") is None
    assert await STORE().get_public_route_pins() == []
    assert await STORE().fetch_dynamic_patterns() == {}


async def test_unpin_scope_mapped_url_is_false_and_leaves_scope_row(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("scope-a", "/a")
    assert await STORE().unpin_public_route("/a") is False
    # A scope-mapped url is not unpinnable through this path; its row is untouched.
    assert pg.route("/a")["scope_id"] == "scope-a"


async def test_unpin_absent_url_is_false(pg: FakeAccessControlPg) -> None:
    assert await STORE().unpin_public_route("/ghost") is False


async def test_remove_url_from_scope_on_marker_deletes_route_only(pg: FakeAccessControlPg) -> None:
    # The generic DELETE door on a marker-valued url is a single guarded delete:
    # ``(True, [])`` (the mapping existed, no cascade), the route row gone, and every
    # scope row / policy untouched.
    await STORE().pin_route_public("/open")
    pg.add_policy("u1", scopes=[PUBLIC])
    assert await STORE().remove_url_from_scope("/open") == (True, [])
    assert pg.route("/open") is None
    assert pg.policy("u1")["scopes"] == [PUBLIC]


# -- remove_scope ------------------------------------------------------------


async def test_remove_scope_rejects_public_marker(pg: FakeAccessControlPg) -> None:
    with pytest.raises(ValueError, match="public marker"):
        await STORE().remove_scope(PUBLIC)


async def test_remove_scope_strips_policies_and_returns_count(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("scope-a", "/a")
    await STORE().add_url_to_scope("scope-a", "/b")
    pg.add_policy("u1", scopes=["scope-a", "other"])
    deleted, affected = await STORE().remove_scope("scope-a")
    # 2 route rows deleted + 1 policy stripped.
    assert deleted == 3
    assert affected == [
        (
            "u1",
            {"scopes": ["other"], "policy_data": {}, "condition": None, "condition_id": None, "condition_kwargs": None},
        )
    ]
    assert pg.policy("u1")["scopes"] == ["other"]
    assert pg.scope_urls("scope-a") == set()


async def test_remove_unknown_scope_returns_zero(pg: FakeAccessControlPg) -> None:
    deleted, affected = await STORE().remove_scope("nope")
    assert deleted == 0
    assert affected == []


async def test_remove_scope_inconsistent_state_reports_mutation(pg: FakeAccessControlPg) -> None:
    # A scope referenced by a token but with NO url mapping still strips the
    # reference, so the count is non-zero and the caller treats it as found.
    pg.add_policy("u1", scopes=["scope-a"])
    deleted, affected = await STORE().remove_scope("scope-a")
    assert deleted == 1
    assert affected == [
        ("u1", {"scopes": [], "policy_data": {}, "condition": None, "condition_id": None, "condition_kwargs": None})
    ]


# -- policy body read / write ------------------------------------------------


async def test_get_policy_body_reads_stored_policy(pg: FakeAccessControlPg) -> None:
    pg.add_policy("u1", scopes=["admin"], policy_data={"plan_limit": 100}, condition=".x")
    assert await STORE().get_policy_body("u1") == {
        "scopes": ["admin"],
        "policy_data": {"plan_limit": 100},
        "condition": ".x",
        "condition_id": None,
        "condition_kwargs": None,
    }


async def test_get_policy_body_unknown_user_is_none(pg: FakeAccessControlPg) -> None:
    assert await STORE().get_policy_body("missing") is None


async def test_policy_exists(pg: FakeAccessControlPg) -> None:
    pg.add_policy("u1")
    assert await STORE().policy_exists("u1") is True
    assert await STORE().policy_exists("u2") is False


async def test_create_policy_writes_row(pg: FakeAccessControlPg) -> None:
    pg.add_route("/s", "s")  # the granted scope must have a live route
    body = await STORE().create_policy("u1", ["s"], {"k": 1}, ".c", None, {"a": 2})
    assert body == {
        "scopes": ["s"],
        "policy_data": {"k": 1},
        "condition": ".c",
        "condition_id": None,
        "condition_kwargs": {"a": 2},
    }
    assert await STORE().get_policy_body("u1") == body


async def test_create_policy_duplicate_user_raises(pg: FakeAccessControlPg) -> None:
    await STORE().create_policy("u1", [])
    with pytest.raises(UniqueViolation):
        await STORE().create_policy("u1", [])


async def test_create_policy_rejects_unrouted_scope(pg: FakeAccessControlPg) -> None:
    # A minted key cannot be granted a scope with no live route — the grant-side
    # lock-and-validate rejects it (mirroring the edit path) and writes no row.
    with pytest.raises(ValueError, match="does not exist"):
        await STORE().create_policy("u1", ["ghost"])
    assert await STORE().policy_exists("u1") is False


async def test_wildcard_scope_exempt_from_route_validation(pg: FakeAccessControlPg) -> None:
    # The universal ``"*"`` is the read-side wildcard, not a concrete route scope, so
    # it is exempt from route validation on both the create and update write paths.
    await STORE().create_policy("u1", ["*"])
    assert await STORE().policy_exists("u1") is True
    assert await STORE().update_policy_fields("u1", {"scopes": ["*"]}) is not None


async def test_wildcard_mixed_with_unrouted_scope_still_raises(pg: FakeAccessControlPg) -> None:
    # ``"*"`` is exempt, but every OTHER scope in the list is still validated: a bogus
    # concrete scope alongside the wildcard is rejected and no row is written.
    with pytest.raises(ValueError, match="does not exist"):
        await STORE().create_policy("u1", ["*", "ghost"])
    assert await STORE().policy_exists("u1") is False


async def test_delete_policy(pg: FakeAccessControlPg) -> None:
    pg.add_policy("u1")
    assert await STORE().delete_policy("u1") is True
    assert await STORE().delete_policy("u1") is False


# -- update_policy_fields (the policy half of edit) --------------------------


async def test_update_scopes_only_preserves_other_fields(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("s2", "/b")
    pg.add_policy("u1", scopes=["s1"], policy_data={"k": 1}, condition=".c")
    body = await STORE().update_policy_fields("u1", {"scopes": ["s2"]})
    assert body == {
        "scopes": ["s2"],
        "policy_data": {"k": 1},
        "condition": ".c",
        "condition_id": None,
        "condition_kwargs": None,
    }


async def test_update_explicit_clear(pg: FakeAccessControlPg) -> None:
    pg.add_policy("u1", scopes=["s1"], policy_data={"k": 1}, condition=".c")
    body = await STORE().update_policy_fields("u1", {"policy_data": None, "condition": None})
    assert body is not None
    assert body["policy_data"] == {}
    assert body["condition"] is None
    assert body["scopes"] == ["s1"]


async def test_update_unknown_user_returns_none(pg: FakeAccessControlPg) -> None:
    assert await STORE().update_policy_fields("missing", {"condition": ".c"}) is None


async def test_update_rejects_unknown_scope(pg: FakeAccessControlPg) -> None:
    pg.add_policy("u1", scopes=[])
    with pytest.raises(ValueError, match="does not exist"):
        await STORE().update_policy_fields("u1", {"scopes": ["ghost"]})


# -- restore_policy_body -----------------------------------------------------


async def test_restore_policy_body_writes_body(pg: FakeAccessControlPg) -> None:
    pg.add_policy("u1", scopes=["old"])
    body = {
        "scopes": ["new"],
        "policy_data": {"k": 1},
        "condition": ".c",
        "condition_id": None,
        "condition_kwargs": None,
    }
    restored = await STORE().restore_policy_body("u1", body)
    assert restored == body
    assert pg.policy("u1")["scopes"] == ["new"]


async def test_restore_policy_body_unknown_user_returns_none(pg: FakeAccessControlPg) -> None:
    assert await STORE().restore_policy_body("missing", {"scopes": []}) is None


async def test_restore_does_not_revalidate_scopes(pg: FakeAccessControlPg) -> None:
    # A historical body restores verbatim even if a scope's route was removed after
    # the version was saved (unlike an edit, which re-validates).
    pg.add_policy("u1", scopes=[])
    restored = await STORE().restore_policy_body("u1", {"scopes": ["route-less-scope"]})
    assert restored is not None
    assert pg.policy("u1")["scopes"] == ["route-less-scope"]


# -- OIDC-subject identifiers (charset allowlist dropped for Postgres) --------


@pytest.mark.parametrize("user_id", ["auth0|abc:123", "user@example.com", "naïve-Ünïcode"])
async def test_oidc_subject_round_trips(pg: FakeAccessControlPg, user_id: str) -> None:
    # Postgres identities are parameterized column values, so a subject containing
    # ``:``/``@``/unicode works natively — no charset guard rejects it.
    pg.add_route("/s", "s")  # the granted scope must have a live route
    await STORE().create_policy(user_id, ["s"])
    assert await STORE().policy_exists(user_id) is True
    body = await STORE().get_policy_body(user_id)
    assert body is not None
    assert body["scopes"] == ["s"]


async def test_scope_id_with_metacharacters_round_trips(pg: FakeAccessControlPg) -> None:
    await STORE().add_url_to_scope("team:read@x", "/a")
    assert await STORE().get_all_existing_scopes() == {"/a": "team:read@x"}
    assert await STORE().fetch_route("/a") == "team:read@x"


# -- transaction isolation / atomicity (no watched set) ----------------------


async def test_remove_scope_cascade_is_atomic_on_fault(pg: FakeAccessControlPg) -> None:
    # A fault mid-cascade rolls the WHOLE mutation back: the route delete that ran
    # first is undone, so the scope survives intact rather than being half-removed.
    await STORE().add_url_to_scope("scope-a", "/a")
    pg.add_policy("u1", scopes=["scope-a"])
    pg.fault = ("UPDATE access_control_policies SET scopes = array_remove", RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await STORE().remove_scope("scope-a")
    assert pg.route("/a") is not None
    assert pg.policy("u1")["scopes"] == ["scope-a"]


async def test_grant_after_remove_is_fail_closed(pg: FakeAccessControlPg) -> None:
    # No reverse index: after a scope is removed, re-adding a route for the same
    # scope id leaves it referenced by NO policy (derived membership), so nobody is
    # authorized — fail-closed, never a corrupt index.
    await STORE().add_url_to_scope("scope-a", "/a")
    pg.add_policy("u1", scopes=["scope-a"])
    await STORE().remove_scope("scope-a")
    await STORE().add_url_to_scope("scope-a", "/a-again")
    assert await STORE().get_all_existing_scopes() == {"/a-again": "scope-a"}
    assert pg.policy("u1")["scopes"] == []
