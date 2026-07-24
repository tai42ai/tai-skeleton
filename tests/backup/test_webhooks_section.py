"""The ``webhooks`` backup section: the hooks + trigger-link envelope, tombstones
winning over restore, per-item vs whole-section refusals, the token-free-evaluable rule
at the import door, and the in-memory seam — driven through the extended ``FakeRedis``
(string ops + clock + trigger scripts)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM
from tai42_contract.app import tai42_app
from tai42_contract.hooks import HookParams

from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control.settings import AccessControlSettings
from tai42_skeleton.authz import execution as execution_module
from tai42_skeleton.backup import sections
from tai42_skeleton.hooks import trigger_links
from tai42_skeleton.hooks.cache import get_hooks_manager as _cache_get_manager  # noqa: F401
from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai42_skeleton.hooks.managers.redis_hooks_manager import RedisHooksManager
from tai42_skeleton.hooks.settings import HooksSettings
from tai42_skeleton.hooks.trigger_links import TriggerLinkError, create_trigger_link, resolve_trigger_token
from tests.access_control.conftest import FakeAccessControlPg, make_pg_ctx
from tests.access_control.conftest import FakeRedis as FakeAccessControlRedis
from tests.access_control.conftest import make_client_ctx as make_access_control_client_ctx
from tests.hooks.conftest import FakeRedis, make_client_ctx


@pytest.fixture(autouse=True)
def execution_gate_off(monkeypatch):
    """Access control OFF, so the import's token-free-evaluable assertion short-circuits
    and these oracles stay on the section's own envelope. ``policy_store`` turns it on."""
    monkeypatch.setattr(execution_module, "access_control_settings", lambda: AccessControlSettings(enable=False))


@pytest.fixture
def store(monkeypatch):
    """A redis-backed hooks + trigger store shared by the section and the trigger
    module, over one fake redis."""
    import tai42_skeleton.hooks.cache as cache
    import tai42_skeleton.hooks.managers.redis_hooks_manager as rhm

    fake = FakeRedis()
    manager = RedisHooksManager(HooksSettings())
    ctx = make_client_ctx(fake)
    monkeypatch.setattr(trigger_links, "get_hooks_manager", lambda: manager)
    monkeypatch.setattr(cache, "get_hooks_manager", lambda: manager)
    monkeypatch.setattr(trigger_links, "client_ctx", ctx)
    monkeypatch.setattr(rhm, "client_ctx", ctx)
    return type("Store", (), {"manager": manager, "redis": fake, "settings": manager.settings})()


@pytest.fixture
def in_memory_store(monkeypatch):
    import tai42_skeleton.hooks.cache as cache

    manager = InMemoryHooksManager(HooksSettings())
    monkeypatch.setattr(trigger_links, "get_hooks_manager", lambda: manager)
    monkeypatch.setattr(cache, "get_hooks_manager", lambda: manager)
    return type("Store", (), {"manager": manager})()


class _CountingRenderer:
    """Condition renderer that records every render so a test can count them; inline
    ``content`` renders to itself."""

    def __init__(self) -> None:
        self.rendered: list[str] = []

    async def render_by_id_or_content(self, *, content, template_id, kwargs) -> str:
        self.rendered.append(content or "")
        return content or ""


@pytest.fixture
def policy_store(monkeypatch):
    """The execution gate ON over a fake policy store and condition renderer.

    Exposes ``add_policy``, ``rendered`` and ``executed``. Every execution key a record
    names must be seeded — an unseeded key is refused by the gate's existence half."""
    from types import SimpleNamespace

    pg = FakeAccessControlPg()
    renderer = _CountingRenderer()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(policy_module, "client_ctx", make_access_control_client_ctx(FakeAccessControlRedis()))
    monkeypatch.setattr(execution_module, "access_control_settings", lambda: AccessControlSettings(enable=True))
    with tai42_app.bound(SimpleNamespace(storage=SimpleNamespace(resource_manager=renderer))):
        yield SimpleNamespace(add_policy=pg.add_policy, rendered=renderer.rendered, executed=pg.executed)


def _wipe(store) -> None:
    store.redis._strings.clear()
    store.redis._hashes.clear()


# -- envelope round-trip ------------------------------------------------------


async def test_envelope_roundtrip_timed_and_permanent(store) -> None:
    await store.manager.register(
        HookParams(name="h1", topic="t", tool="notify", execution_key="k-fire", execution_key_fingerprint="fp")
    )
    timed = await create_trigger_link(
        topic="t",
        name="timed",
        ttl_seconds=3600,
        tool_kwargs={"k": 1},
        execution_key="k-fire",
        execution_key_fingerprint="fp",
        require_api_key=False,
        created_by="a",
    )
    perm = await create_trigger_link(
        topic="t",
        name="perm",
        ttl_seconds=None,
        tool_kwargs=None,
        execution_key="k-fire",
        execution_key_fingerprint="fp",
        require_api_key=False,
        created_by="a",
    )

    doc = await sections._export_webhooks()
    assert [h["name"] for h in doc["hooks"]] == ["h1"]
    assert {link["name"] for link in doc["trigger_links"]} == {"timed", "perm"}
    assert doc["tombstones"] == []

    _wipe(store)
    report = await sections._import_webhooks(doc)
    assert report["errors"] == []
    assert report["created"] == 3  # one hook + two links
    # Both original URLs resolve again, still bound to the key their records named.
    assert (await resolve_trigger_token(timed["token"])).topic == "t"
    restored = await resolve_trigger_token(perm["token"])
    assert (restored.topic, restored.execution_key) == ("t", "k-fire")


async def test_tool_kwargs_survive_roundtrip_and_merge(store) -> None:
    link = await create_trigger_link(
        topic="t",
        name="k",
        ttl_seconds=None,
        tool_kwargs={"flow": {"x": 1}},
        execution_key="k-fire",
        execution_key_fingerprint="fp",
        require_api_key=False,
        created_by=None,
    )
    doc = await sections._export_webhooks()
    _wipe(store)
    await sections._import_webhooks(doc)
    assert (await resolve_trigger_token(link["token"])).tool_kwargs == {"flow": {"x": 1}}


# -- tombstone durability -----------------------------------------------------


async def test_import_pre_revocation_export_stays_dead_via_local_tombstone(store) -> None:
    link = await create_trigger_link(
        topic="t",
        name="r",
        ttl_seconds=None,
        tool_kwargs=None,
        execution_key="k-fire",
        execution_key_fingerprint="fp",
        require_api_key=False,
        created_by=None,
    )
    pre = await sections._export_webhooks()  # exported while live (no tombstone yet)
    await trigger_links.revoke_trigger_link("r")  # local tombstone now guards it
    report = await sections._import_webhooks(pre)
    assert report["skipped"] >= 1  # the tombstoned record is refused
    with pytest.raises(TriggerLinkError):
        await resolve_trigger_token(link["token"])


async def test_exported_tombstone_restores_and_gates(store) -> None:
    link = await create_trigger_link(
        topic="t",
        name="r",
        ttl_seconds=None,
        tool_kwargs=None,
        execution_key="k-fire",
        execution_key_fingerprint="fp",
        require_api_key=False,
        created_by=None,
    )
    pre = await sections._export_webhooks()  # pre-revocation (record live, no tombstone)
    await trigger_links.revoke_trigger_link("r")
    post = await sections._export_webhooks()  # post-revocation (tombstone present)
    assert len(post["tombstones"]) == 1

    _wipe(store)
    await sections._import_webhooks(post)  # restores the tombstone, no live record
    with pytest.raises(TriggerLinkError):
        await resolve_trigger_token(link["token"])
    # A follow-up import of the PRE-revocation export stays dead too (tombstone gates).
    report = await sections._import_webhooks(pre)
    assert report["skipped"] >= 1
    with pytest.raises(TriggerLinkError):
        await resolve_trigger_token(link["token"])


# -- expiry / collisions ------------------------------------------------------


async def test_expired_at_import_skipped_and_logged(store, caplog) -> None:
    doc = {
        "hooks": [],
        "trigger_links": [
            {
                "name": "exp",
                "token_hash": "a" * 64,
                "record": {
                    "name": "exp",
                    "topic": "t",
                    "execution_key": "k-fire",
                    "execution_key_fingerprint": "fp",
                    "require_api_key": False,
                    "tool_kwargs": None,
                    "created_by": None,
                    "created_at": "2000-01-01T00:00:00+00:00",
                    "expires_at": "2000-01-02T00:00:00+00:00",
                },
            }
        ],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert report["skipped"] == 1
    assert report["created"] == 0


async def test_live_name_collision_counts_updated(store) -> None:
    await create_trigger_link(
        topic="t",
        name="shared",
        ttl_seconds=None,
        tool_kwargs=None,
        execution_key="k-fire",
        execution_key_fingerprint="fp",
        require_api_key=False,
        created_by=None,
    )
    doc = {
        "hooks": [],
        "trigger_links": [
            {
                "name": "shared",
                "token_hash": "b" * 64,
                "record": {
                    "name": "shared",
                    "topic": "t",
                    "execution_key": "k-fire",
                    "execution_key_fingerprint": "fp",
                    "require_api_key": False,
                    "tool_kwargs": None,
                    "created_by": None,
                    "created_at": "2026-07-21T00:00:00+00:00",
                    "expires_at": None,
                },
            }
        ],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert report["updated"] == 1


async def test_one_hash_two_names_whole_section_refusal_zero_written(store) -> None:
    await store.manager.register(
        HookParams(name="h", topic="t", tool="notify", execution_key="k-fire", execution_key_fingerprint="fp")
    )
    rec = {
        "name": "",
        "topic": "t",
        "execution_key": "k-fire",
        "execution_key_fingerprint": "fp",
        "require_api_key": False,
        "tool_kwargs": None,
        "created_by": None,
        "created_at": "2026-07-21T00:00:00+00:00",
        "expires_at": None,
    }
    doc = {
        "hooks": [
            {
                "name": "new-hook",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-fire",
                "execution_key_fingerprint": "fp",
            }
        ],
        "trigger_links": [
            {"name": "A", "token_hash": "c" * 64, "record": {**rec, "name": "A"}},
            {"name": "B", "token_hash": "c" * 64, "record": {**rec, "name": "B"}},
        ],
        "topic_verifiers": {},
        "tombstones": [],
    }
    with pytest.raises(ValueError, match="two names"):  # whole-section refusal (zero keys written)
        await sections._import_webhooks(doc)
    # Zero keys written — even the hooks portion. The pre-existing hook "h" is
    # untouched; "new-hook" was never registered.
    hooks = await store.manager.list_hooks()
    assert set(hooks) == {"h"}


async def test_same_name_twice_last_wins(store) -> None:
    rec = {
        "topic": "t",
        "execution_key": "k-fire",
        "execution_key_fingerprint": "fp",
        "require_api_key": False,
        "tool_kwargs": None,
        "created_by": None,
        "created_at": "2026-07-21T00:00:00+00:00",
        "expires_at": None,
    }
    h1, h2 = "d" * 64, "e" * 64
    doc = {
        "hooks": [],
        "trigger_links": [
            {"name": "A", "token_hash": h1, "record": {**rec, "name": "A"}},
            {"name": "A", "token_hash": h2, "record": {**rec, "name": "A"}},
        ],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert report["errors"] == []
    # H2 is the live record; H1's record was displaced (deleted, NO tombstone).
    assert store.settings.trigger_record_key(h2) in store.redis._strings
    assert store.settings.trigger_record_key(h1) not in store.redis._strings
    assert trigger_links._as_str(store.redis._get_str(store.settings.trigger_name_key("A"))) == h2
    assert store.redis._get_str(store.settings.trigger_tomb_key(h1)) is None


async def test_live_index_dup_hash_whole_section_refusal(store) -> None:
    # Live A→H in the store; the payload carries B→H → whole-section refusal.
    link = await create_trigger_link(
        topic="t",
        name="A",
        ttl_seconds=None,
        tool_kwargs=None,
        execution_key="k-fire",
        execution_key_fingerprint="fp",
        require_api_key=False,
        created_by=None,
    )
    live = (await trigger_links.export_trigger_links())["trigger_links"][0]
    doc = {
        "hooks": [],
        "trigger_links": [{"name": "B", "token_hash": live["token_hash"], "record": {**live["record"], "name": "B"}}],
        "topic_verifiers": {},
        "tombstones": [],
    }
    with pytest.raises(ValueError, match="already live under"):
        await sections._import_webhooks(doc)
    # A still resolves — nothing applied.
    assert (await resolve_trigger_token(link["token"])).topic == "t"


async def test_orphan_index_dup_hash_whole_section_refusal_zero_written(store) -> None:
    # An ORPHAN in the store: name key A → hash H with NO record key (a corrupt
    # hand-edited backup). A normal create can't produce this, so seed raw state.
    # Import binds a NEW name B to that same H → whole-section refusal, zero written:
    # writing B→H would let a later revoke of A destroy B's live record.
    orphan_hash = "c" * 64
    store.redis._set_str(store.settings.trigger_name_key("A"), orphan_hash)
    await store.manager.register(
        HookParams(name="h", topic="t", tool="notify", execution_key="k-fire", execution_key_fingerprint="fp")
    )
    doc = {
        "hooks": [
            {
                "name": "new-hook",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-fire",
                "execution_key_fingerprint": "fp",
            }
        ],
        "trigger_links": [
            {
                "name": "B",
                "token_hash": orphan_hash,
                "record": {
                    "name": "B",
                    "topic": "t",
                    "execution_key": "k-fire",
                    "execution_key_fingerprint": "fp",
                    "require_api_key": False,
                    "tool_kwargs": None,
                    "created_by": None,
                    "created_at": "2026-07-21T00:00:00+00:00",
                    "expires_at": None,
                },
            }
        ],
        "topic_verifiers": {},
        "tombstones": [],
    }
    with pytest.raises(ValueError, match="already live under"):
        await sections._import_webhooks(doc)
    # Zero keys written: B never bound, its record never written, hooks untouched.
    assert store.redis._get_str(store.settings.trigger_name_key("B")) is None
    assert store.settings.trigger_record_key(orphan_hash) not in store.redis._strings
    assert set(await store.manager.list_hooks()) == {"h"}


# -- per-item malformations ---------------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        {
            "name": "g",
            "topic": "t",
            "execution_key": "k-fire",
            "execution_key_fingerprint": "fp",
            "created_at": "2026-07-21T00:00:00+00:00",
            "expires_at": "garbage",
        },
        {
            "name": "g",
            "topic": "",
            "execution_key": "k-fire",
            "execution_key_fingerprint": "fp",
            "created_at": "2026-07-21T00:00:00+00:00",
            "expires_at": None,
        },
        # A link record naming no execution key has no bounded authority to fire under.
        {"name": "g", "topic": "t", "execution_key": "", "created_at": "2026-07-21T00:00:00+00:00", "expires_at": None},
        {"name": "g", "topic": "t", "created_at": "2026-07-21T00:00:00+00:00", "expires_at": None},
        {
            "name": "g",
            "topic": "t",
            "execution_key": "k-fire",
            "execution_key_fingerprint": "fp",
            "tool_kwargs": [1],
            "created_at": "2026-07-21T00:00:00+00:00",
            "expires_at": None,
        },
    ],
)
async def test_per_item_malformation_error_and_skipped_rest_proceeds(store, record) -> None:
    doc = {
        "hooks": [
            {"name": "h", "topic": "t", "tool": "notify", "execution_key": "k-fire", "execution_key_fingerprint": "fp"}
        ],
        "trigger_links": [{"name": "g", "token_hash": "f" * 64, "record": record}],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert report["errors"]  # the malformed item is surfaced
    assert report["skipped"] == 1
    assert report["created"] == 1  # the hook still restored
    assert set(await store.manager.list_hooks()) == {"h"}


@pytest.mark.parametrize(
    "keyless",
    [
        {"name": "keyless", "topic": "t", "tool": "notify"},
        {"name": "keyless", "topic": "t", "tool": "notify", "execution_key": ""},
    ],
)
async def test_import_rejects_a_keyless_hook_per_record_rest_restored(store, keyless) -> None:
    # A hook record naming no execution key has no bounded authority to fire under, and
    # the server's own is not a substitute: refused PER RECORD into ``errors``, never
    # written, never aborting the rest.
    doc = {
        "hooks": [
            keyless,
            {
                "name": "bound",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-fire",
                "execution_key_fingerprint": "fp",
            },
        ],
        "trigger_links": [],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)

    assert len(report["errors"]) == 1
    assert "keyless" in report["errors"][0]
    assert "execution_key" in report["errors"][0]
    assert report["skipped"] == 1
    assert report["created"] == 1
    # The keyless record was never stored; the record after it still was.
    assert set(await store.manager.list_hooks()) == {"bound"}


async def test_non_string_name_or_hash_per_item_error_valid_entries_restored(store) -> None:
    # A JSON-valid but ill-typed triple (non-str name, non-str token_hash) must
    # become a per-item report error + skipped — never an uncaught TypeError that
    # aborts the whole import after the hook and earlier records were written.
    good = {
        "name": "good",
        "topic": "t",
        "execution_key": "k-fire",
        "execution_key_fingerprint": "fp",
        "require_api_key": False,
        "tool_kwargs": None,
        "created_by": None,
        "created_at": "2026-07-21T00:00:00+00:00",
        "expires_at": None,
    }
    doc = {
        "hooks": [
            {"name": "h", "topic": "t", "tool": "notify", "execution_key": "k-fire", "execution_key_fingerprint": "fp"}
        ],
        "trigger_links": [
            {"name": 123, "token_hash": "a" * 64, "record": {**good, "name": "x"}},
            {"name": "y", "token_hash": 456, "record": {**good, "name": "y"}},
            {"name": "good", "token_hash": "b" * 64, "record": good},
        ],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert len(report["errors"]) == 2  # both ill-typed rows surfaced per-item
    assert report["skipped"] == 2
    assert report["created"] == 2  # the hook + the one valid trigger link
    # The hook and the valid trigger link were restored despite the two bad rows.
    assert set(await store.manager.list_hooks()) == {"h"}
    assert store.settings.trigger_record_key("b" * 64) in store.redis._strings
    assert trigger_links._as_str(store.redis._get_str(store.settings.trigger_name_key("good"))) == "b" * 64


async def test_non_string_tombstone_per_item_error_section_proceeds(store) -> None:
    # A JSON-valid but ill-typed tombstone entry (non-str) must become a per-item
    # report error + skipped — never an uncaught TypeError that aborts the whole
    # webhooks section after the hooks and earlier tombstones were written.
    good = {
        "name": "good",
        "topic": "t",
        "execution_key": "k-fire",
        "execution_key_fingerprint": "fp",
        "require_api_key": False,
        "tool_kwargs": None,
        "created_by": None,
        "created_at": "2026-07-21T00:00:00+00:00",
        "expires_at": None,
    }
    doc = {
        "hooks": [
            {"name": "h", "topic": "t", "tool": "notify", "execution_key": "k-fire", "execution_key_fingerprint": "fp"}
        ],
        "trigger_links": [{"name": "good", "token_hash": "b" * 64, "record": good}],
        "topic_verifiers": {},
        "tombstones": [123, "a" * 64],
    }
    report = await sections._import_webhooks(doc)
    # The bad tombstone is surfaced per-item and skipped; the section does NOT abort.
    assert len(report["errors"]) == 1
    assert report["skipped"] == 1
    assert report["created"] == 2  # the hook + the one valid trigger link
    # The hook, the valid trigger link, and the valid tombstone were all applied.
    assert set(await store.manager.list_hooks()) == {"h"}
    assert store.settings.trigger_record_key("b" * 64) in store.redis._strings
    assert store.redis._get_str(store.settings.trigger_tomb_key("a" * 64)) is not None


async def test_hook_with_non_compiling_jq_per_record_rest_restored(store) -> None:
    # An inline jq that does not compile is a bad RECORD: refused per hook, never an
    # abort that leaves earlier hooks written while the router reports nothing created.
    doc = {
        "hooks": [
            {
                "name": "broken",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-fire",
                "execution_key_fingerprint": "fp",
                "expr": ".foo |",
            },
            {
                "name": "sound",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-fire",
                "execution_key_fingerprint": "fp",
            },
        ],
        "trigger_links": [],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)

    assert len(report["errors"]) == 1
    assert "broken" in report["errors"][0]
    assert "not valid jq" in report["errors"][0]
    assert report["skipped"] == 1
    assert report["created"] == 1
    assert set(await store.manager.list_hooks()) == {"sound"}


# -- the token-free-evaluable rule at the import door -------------------------

# Evaluable / unevaluable by a tokenless fire: under the reduced claim set any identity
# claim beyond ``.identity.owner_user_id`` is absent.
_EVALUABLE = '.sub != "banned"'
_UNEVALUABLE = '.identity.description == "ops"'


def _link_record(name: str, execution_key: str, topic: str = "t") -> dict:
    return {
        "name": name,
        "topic": topic,
        "execution_key": execution_key,
        "execution_key_fingerprint": "fp",
        "require_api_key": False,
        "tool_kwargs": None,
        "created_by": None,
        "created_at": "2026-07-21T00:00:00+00:00",
        "expires_at": None,
    }


async def test_import_refuses_records_bound_to_a_key_no_fire_can_evaluate(store, policy_store) -> None:
    # BOTH import writers assert token-free-evaluability: a hook and a trigger link
    # naming an unevaluable key are each refused per record into ``errors``.
    policy_store.add_policy("k-blind", condition=_UNEVALUABLE, policy_data={KEY_FINGERPRINT_CLAIM: "fp"})
    policy_store.add_policy("k-fire", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "fp"})
    doc = {
        "hooks": [
            {
                "name": "blind",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-blind",
                "execution_key_fingerprint": "fp",
            },
            {
                "name": "bound",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-fire",
                "execution_key_fingerprint": "fp",
            },
        ],
        "trigger_links": [
            {"name": "blind", "token_hash": "a" * 64, "record": _link_record("blind", "k-blind")},
            {"name": "bound", "token_hash": "b" * 64, "record": _link_record("bound", "k-fire")},
        ],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)

    assert len(report["errors"]) == 2
    assert all("k-blind" in error and "unusable at a fire" in error for error in report["errors"])
    assert report["skipped"] == 2
    assert report["created"] == 2  # the hook and the link bound to the evaluable key
    assert set(await store.manager.list_hooks()) == {"bound"}
    assert store.settings.trigger_record_key("a" * 64) not in store.redis._strings
    assert store.settings.trigger_record_key("b" * 64) in store.redis._strings


async def test_import_refuses_records_bound_to_a_key_with_no_policy_row(store, policy_store) -> None:
    # The EXISTENCE half of the gate: a key with no policy row passes the evaluable half
    # vacuously, so both writers must refuse it per record or store a dead record.
    policy_store.add_policy("k-fire", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "fp"})
    doc = {
        "hooks": [
            {
                "name": "ghost",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-ghost",
                "execution_key_fingerprint": "fp",
            },
            {
                "name": "bound",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-fire",
                "execution_key_fingerprint": "fp",
            },
        ],
        "trigger_links": [
            {"name": "ghost", "token_hash": "a" * 64, "record": _link_record("ghost", "k-ghost")},
            {"name": "bound", "token_hash": "b" * 64, "record": _link_record("bound", "k-fire")},
        ],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)

    assert len(report["errors"]) == 2
    assert all("k-ghost" in error and "has no policy" in error for error in report["errors"])
    assert report["skipped"] == 2
    assert report["created"] == 2
    assert set(await store.manager.list_hooks()) == {"bound"}
    assert store.settings.trigger_record_key("a" * 64) not in store.redis._strings


async def test_import_refuses_a_record_whose_bound_fingerprint_no_longer_matches(store, policy_store) -> None:
    # A remint writes a fresh per-mint fingerprint: a record bound to the OLD one is
    # refused per record, so a stale record is never revived as the reminted key.
    policy_store.add_policy("svc", scopes=["hooks"], policy_data={KEY_FINGERPRINT_CLAIM: "F2"})
    stale_link = _link_record("stale", "svc")
    stale_link["execution_key_fingerprint"] = "F1"
    fresh_link = _link_record("fresh", "svc")
    fresh_link["execution_key_fingerprint"] = "F2"
    doc = {
        "hooks": [
            {
                "name": "stale",
                "topic": "t",
                "tool": "notify",
                "execution_key": "svc",
                "execution_key_fingerprint": "F1",
            },
            {
                "name": "fresh",
                "topic": "t",
                "tool": "notify",
                "execution_key": "svc",
                "execution_key_fingerprint": "F2",
            },
        ],
        "trigger_links": [
            {"name": "stale", "token_hash": "a" * 64, "record": stale_link},
            {"name": "fresh", "token_hash": "b" * 64, "record": fresh_link},
        ],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)

    assert len(report["errors"]) == 2
    assert all("no longer matches the bound key identity" in error for error in report["errors"])
    assert report["skipped"] == 2
    assert report["created"] == 2  # the hook and the link carrying the current fingerprint
    assert set(await store.manager.list_hooks()) == {"fresh"}
    assert store.settings.trigger_record_key("a" * 64) not in store.redis._strings
    assert store.settings.trigger_record_key("b" * 64) in store.redis._strings


async def test_a_corrupt_stored_policy_fails_the_section_instead_of_blaming_the_record(store, policy_store) -> None:
    # A policy-store integrity fault is not a bad record: it propagates as the section's
    # own failure, not a per-hook rejection blaming an intact backup.
    policy_store.add_policy("k-corrupt", policy_data=[1])
    doc = {
        "hooks": [
            {
                "name": "h",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-corrupt",
                "execution_key_fingerprint": "fp",
            }
        ],
        "trigger_links": [],
        "topic_verifiers": {},
        "tombstones": [],
    }
    with pytest.raises(ValidationError):
        await sections._import_webhooks(doc)

    assert await store.manager.list_hooks() == {}


async def test_one_execution_key_is_read_once_for_the_whole_import(store, policy_store) -> None:
    # Each DISTINCT execution key is asserted once, so the policy read and condition
    # render do not repeat per record.
    policy_store.add_policy("k-fire", condition=_EVALUABLE, policy_data={KEY_FINGERPRINT_CLAIM: "fp"})
    doc = {
        "hooks": [
            {
                "name": "h1",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-fire",
                "execution_key_fingerprint": "fp",
            },
            {
                "name": "h2",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-fire",
                "execution_key_fingerprint": "fp",
            },
        ],
        "trigger_links": [
            {"name": "l1", "token_hash": "a" * 64, "record": _link_record("l1", "k-fire")},
            {"name": "l2", "token_hash": "b" * 64, "record": _link_record("l2", "k-fire")},
        ],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)

    assert report["created"] == 4
    assert report["errors"] == []
    assert policy_store.rendered == [_EVALUABLE]


async def test_a_refused_key_is_read_once_and_refuses_every_record_naming_it(store, policy_store) -> None:
    # Caching the verdict must not merge the refusals: every record naming the unusable
    # key still gets its own error, off one read and one render.
    policy_store.add_policy("k-bad", condition=_UNEVALUABLE, policy_data={KEY_FINGERPRINT_CLAIM: "fp"})
    doc = {
        "hooks": [
            {"name": "h1", "topic": "t", "tool": "notify", "execution_key": "k-bad", "execution_key_fingerprint": "fp"},
            {"name": "h2", "topic": "t", "tool": "notify", "execution_key": "k-bad", "execution_key_fingerprint": "fp"},
        ],
        "trigger_links": [
            {"name": "l1", "token_hash": "a" * 64, "record": _link_record("l1", "k-bad")},
        ],
        "topic_verifiers": {},
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)

    assert report["created"] == 0
    assert len(report["errors"]) == 3
    assert all("unusable at a fire" in error for error in report["errors"])
    assert policy_store.rendered == [_UNEVALUABLE]


async def test_keys_of_one_owner_read_the_owner_row_once_for_the_batch(store, policy_store) -> None:
    # The batch holds ONE enforcer, so the OWNER row two distinct execution keys share
    # is fetched once for the whole restore instead of once per key.
    owner_claim = {"owner_user_id": "acct", KEY_FINGERPRINT_CLAIM: "fp"}
    policy_store.add_policy("acct", scopes=["hooks"], condition=_EVALUABLE)
    policy_store.add_policy("k-one", scopes=["hooks"], policy_data=owner_claim)
    policy_store.add_policy("k-two", scopes=["hooks"], policy_data=owner_claim)
    doc = {
        "hooks": [
            {"name": "h1", "topic": "t", "tool": "notify", "execution_key": "k-one", "execution_key_fingerprint": "fp"},
            {"name": "h2", "topic": "t", "tool": "notify", "execution_key": "k-two", "execution_key_fingerprint": "fp"},
        ],
        "topic_verifiers": {},
        "trigger_links": [],
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)

    assert report["created"] == 2
    assert report["errors"] == []
    selects = [sql for sql in policy_store.executed if sql.startswith("SELECT") and "access_control_policies" in sql]
    assert len(selects) == 3  # k-one, k-two, and the owner ONCE


async def test_a_tombstoned_record_skips_benignly_before_its_key_is_read(store, policy_store) -> None:
    # The tombstone check runs BEFORE the key assertion: a tombstoned record is a benign
    # skip, not an import error over something nothing can revive.
    policy_store.add_policy("k-blind", condition=_UNEVALUABLE, policy_data={KEY_FINGERPRINT_CLAIM: "fp"})
    doc = {
        "hooks": [],
        "trigger_links": [{"name": "dead", "token_hash": "a" * 64, "record": _link_record("dead", "k-blind")}],
        "topic_verifiers": {},
        "tombstones": ["a" * 64],
    }
    report = await sections._import_webhooks(doc)

    assert report["errors"] == []
    assert report["skipped"] == 1
    assert report["created"] == 0
    assert policy_store.rendered == []
    assert store.settings.trigger_record_key("a" * 64) not in store.redis._strings


# -- old shape + malformed envelope ------------------------------------------


async def test_old_list_shape_imports_clean(store) -> None:
    payload = [
        {"name": "h1", "topic": "t", "tool": "notify", "execution_key": "k-fire", "execution_key_fingerprint": "fp"}
    ]
    report = await sections._import_webhooks(payload)
    assert report["created"] == 1
    assert set(await store.manager.list_hooks()) == {"h1"}


@pytest.mark.parametrize("missing", ["hooks", "topic_verifiers", "trigger_links", "tombstones"])
async def test_envelope_missing_key_raises(store, missing) -> None:
    doc = {"hooks": [], "topic_verifiers": {}, "trigger_links": [], "tombstones": []}
    del doc[missing]
    with pytest.raises(ValueError, match="missing the required"):
        await sections._import_webhooks(doc)


@pytest.mark.parametrize(("key", "value"), [("hooks", {}), ("tombstones", "x")])
async def test_envelope_non_list_key_whole_section_refusal_zero_written(store, key, value) -> None:
    # A hand-edited envelope with a NON-LIST value for a required key is a loud
    # whole-section refusal naming the key, BEFORE any write — never a silent
    # degrade (an empty dict iterating to nothing) or an ungraceful failure deeper in.
    await store.manager.register(
        HookParams(name="pre", topic="t", tool="notify", execution_key="k-fire", execution_key_fingerprint="fp")
    )
    doc = {"hooks": [], "topic_verifiers": {}, "trigger_links": [], "tombstones": []}
    doc[key] = value
    with pytest.raises(ValueError, match=f"{key!r} must be a list"):
        await sections._import_webhooks(doc)
    # Zero keys written — the pre-existing hook is untouched.
    assert set(await store.manager.list_hooks()) == {"pre"}


async def test_envelope_non_mapping_topic_verifiers_whole_section_refusal_zero_written(store) -> None:
    # The bindings ride as a MAPPING; a hand-edited list is a loud whole-section
    # refusal before any write, exactly as a non-list required key is.
    await store.manager.register(
        HookParams(name="pre", topic="t", tool="notify", execution_key="k-fire", execution_key_fingerprint="fp")
    )
    doc = {"hooks": [], "topic_verifiers": [], "trigger_links": [], "tombstones": []}
    with pytest.raises(ValueError, match="'topic_verifiers' must be a mapping"):
        await sections._import_webhooks(doc)
    assert set(await store.manager.list_hooks()) == {"pre"}


# -- topic verifier bindings (the topic's ingress lock) -----------------------


async def test_verifier_binding_round_trips_so_a_verified_topic_stays_verified(store) -> None:
    # A binding is the topic's ingress lock: dropping it on restore would bring the
    # topic's hooks back on a door anyone may ring unsigned.
    await store.manager.set_topic_verifier("payments", {"verifier": "github", "config": {"secret_env": "GH_SECRET"}})
    await store.manager.register(
        HookParams(name="pay", topic="payments", tool="notify", execution_key="k-fire", execution_key_fingerprint="fp")
    )

    doc = await sections._export_webhooks()
    assert doc["topic_verifiers"] == {"payments": {"verifier": "github", "config": {"secret_env": "GH_SECRET"}}}

    _wipe(store)
    report = await sections._import_webhooks(doc)
    assert report["errors"] == []
    assert report["created"] == 2  # the binding + the hook
    assert await store.manager.get_topic_verifier("payments") == {
        "verifier": "github",
        "config": {"secret_env": "GH_SECRET"},
    }


async def test_replacing_a_live_binding_counts_updated(store) -> None:
    await store.manager.set_topic_verifier("payments", {"verifier": "github", "config": {}})
    doc = {
        "hooks": [],
        "topic_verifiers": {"payments": {"verifier": "hmac", "config": {}}},
        "trigger_links": [],
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert report["errors"] == []
    assert (report["updated"], report["created"]) == (1, 0)
    assert (await store.manager.get_topic_verifier("payments"))["verifier"] == "hmac"


@pytest.mark.parametrize("binding", [{"config": {}}, {"verifier": "", "config": {}}, "hmac", {"verifier": 1}])
async def test_malformed_binding_per_topic_error_rest_restored(store, binding) -> None:
    # A hand-edited binding is a loud per-topic rejection in the report — never a
    # malformed binding stored, and never an abort of the records around it.
    doc = {
        "hooks": [
            {"name": "h", "topic": "t", "tool": "notify", "execution_key": "k-fire", "execution_key_fingerprint": "fp"}
        ],
        "topic_verifiers": {"broken": binding, "sound": {"verifier": "hmac", "config": {}}},
        "trigger_links": [],
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert len(report["errors"]) == 1
    assert "broken" in report["errors"][0]
    assert report["skipped"] == 1
    assert report["created"] == 2  # the sound binding + the hook
    assert await store.manager.get_topic_verifier("broken") is None
    assert set(await store.manager.list_hooks()) == {"h"}


async def test_an_offset_less_expires_at_is_a_per_record_error_not_a_torn_section(store) -> None:
    # An ``expires_at`` that parses but carries no UTC offset is a record-shaped fault:
    # ``errors`` + ``skipped``, links either side still restore, no mid-write abort.
    doc = {
        "hooks": [],
        "topic_verifiers": {},
        "trigger_links": [
            {"name": "good", "token_hash": "a" * 64, "record": _link_record("good", "k-fire")},
            {
                "name": "naive",
                "token_hash": "b" * 64,
                "record": {**_link_record("naive", "k-fire"), "expires_at": "2030-01-01T00:00:00"},
            },
            {"name": "after", "token_hash": "c" * 64, "record": _link_record("after", "k-fire")},
        ],
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)

    assert len(report["errors"]) == 1
    assert "carries no timezone offset" in report["errors"][0]
    assert (report["created"], report["skipped"]) == (2, 1)
    assert store.settings.trigger_record_key("b" * 64) not in store.redis._strings
    assert store.settings.trigger_record_key("c" * 64) in store.redis._strings


async def test_records_on_a_topic_whose_lock_failed_are_refused(store) -> None:
    # Ingress locks restore BEFORE the records they gate, and a lock that failed to land
    # leaves its topic's door open: every record on that topic is refused, others stand.
    doc = {
        "hooks": [
            {
                "name": "pay",
                "topic": "payments",
                "tool": "notify",
                "execution_key": "k-fire",
                "execution_key_fingerprint": "fp",
            },
            {
                "name": "other",
                "topic": "t",
                "tool": "notify",
                "execution_key": "k-fire",
                "execution_key_fingerprint": "fp",
            },
        ],
        "topic_verifiers": {"payments": {"verifier": "", "config": {}}},
        "trigger_links": [
            {"name": "paylink", "token_hash": "a" * 64, "record": _link_record("paylink", "k-fire", topic="payments")},
            {"name": "otherlink", "token_hash": "b" * 64, "record": _link_record("otherlink", "k-fire")},
        ],
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)

    assert await store.manager.all_topic_verifiers() == {}
    assert set(await store.manager.list_hooks()) == {"other"}
    assert store.settings.trigger_record_key("a" * 64) not in store.redis._strings
    assert store.settings.trigger_record_key("b" * 64) in store.redis._strings
    # The binding's own failure, plus one per record it could not gate.
    assert len(report["errors"]) == 3
    assert report["skipped"] == 3
    assert report["created"] == 2


@pytest.mark.parametrize("topic", ["", 123])
async def test_ill_typed_topic_key_per_item_error_section_proceeds(store, topic) -> None:
    # A blank or ill-typed topic key names no door; it is a per-item report error,
    # never a binding written under a topic nothing can deliver to.
    doc = {
        "hooks": [],
        "topic_verifiers": {topic: {"verifier": "hmac", "config": {}}},
        "trigger_links": [],
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert len(report["errors"]) == 1
    assert report["skipped"] == 1
    assert report["created"] == 0


async def test_binding_naming_an_unregistered_verifier_is_restored_not_dropped(store) -> None:
    # An unknown verifier NAME still restores: the ingress door resolves it live and
    # denies what it cannot resolve, whereas refusing here would restore a PUBLIC topic.
    doc = {
        "hooks": [],
        "topic_verifiers": {"payments": {"verifier": "not-installed-here", "config": {}}},
        "trigger_links": [],
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert report["errors"] == []
    assert report["created"] == 1
    assert (await store.manager.get_topic_verifier("payments"))["verifier"] == "not-installed-here"


# -- restore into a now-verified topic (fire-time enforcement) -------------


async def test_restore_into_verified_topic_created_but_resolves_404(store) -> None:
    link = await create_trigger_link(
        topic="secure",
        name="v",
        ttl_seconds=None,
        tool_kwargs=None,
        execution_key="k-fire",
        execution_key_fingerprint="fp",
        require_api_key=False,
        created_by=None,
    )
    doc = await sections._export_webhooks()
    _wipe(store)
    # The topic gains a verifier binding after the export.
    await store.manager.set_topic_verifier("secure", {"verifier": "hmac", "config": {}})
    report = await sections._import_webhooks(doc)
    assert report["created"] == 1  # restore does NOT re-run the create-time verifier check
    with pytest.raises(TriggerLinkError):  # but the door enforces it (uniform 404)
        await resolve_trigger_token(link["token"])


# -- in-memory seam -----------------------------------------------------------


async def test_in_memory_export_truthfully_empty_hooks_unchanged(in_memory_store) -> None:
    await in_memory_store.manager.register(
        HookParams(name="h1", topic="t", tool="notify", execution_key="k-fire", execution_key_fingerprint="fp")
    )
    doc = await sections._export_webhooks()
    assert [h["name"] for h in doc["hooks"]] == ["h1"]
    assert doc["trigger_links"] == []
    assert doc["tombstones"] == []


async def test_in_memory_import_refuses_trigger_portion_hooks_restore(in_memory_store, caplog) -> None:
    doc = {
        "hooks": [
            {"name": "h1", "topic": "t", "tool": "notify", "execution_key": "k-fire", "execution_key_fingerprint": "fp"}
        ],
        "trigger_links": [
            {
                "name": "x",
                "token_hash": "a" * 64,
                "record": {
                    "name": "x",
                    "topic": "t",
                    "execution_key": "k-fire",
                    "execution_key_fingerprint": "fp",
                    "require_api_key": False,
                    "tool_kwargs": None,
                    "created_by": None,
                    "created_at": "2026-07-21T00:00:00+00:00",
                    "expires_at": None,
                },
            }
        ],
        "topic_verifiers": {},
        "tombstones": ["b" * 64],
    }
    report = await sections._import_webhooks(doc)
    # The hooks portion restores; the trigger + tombstone portions refuse loudly.
    assert set(await in_memory_store.manager.list_hooks()) == {"h1"}
    assert len(report["errors"]) == 2  # one for the record, one for the tombstone
    assert report["created"] == 1
