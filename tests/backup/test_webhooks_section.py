"""The ``webhooks`` backup section: the hooks + trigger-link envelope, tombstones
winning over restore, per-item vs whole-section refusals, and the in-memory seam —
driven through the extended ``FakeRedis`` (string ops + clock + trigger scripts)."""

from __future__ import annotations

import pytest
from tai42_contract.hooks import HookParams

from tai42_skeleton.backup import sections
from tai42_skeleton.hooks import trigger_links
from tai42_skeleton.hooks.cache import get_hooks_manager as _cache_get_manager  # noqa: F401
from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai42_skeleton.hooks.managers.redis_hooks_manager import RedisHooksManager
from tai42_skeleton.hooks.settings import HooksSettings
from tai42_skeleton.hooks.trigger_links import TriggerLinkError, create_trigger_link, resolve_trigger_token
from tests.hooks.conftest import FakeRedis, make_client_ctx


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


def _wipe(store) -> None:
    store.redis._strings.clear()
    store.redis._hashes.clear()


# -- envelope round-trip ------------------------------------------------------


async def test_envelope_roundtrip_timed_and_permanent(store) -> None:
    await store.manager.register(HookParams(name="h1", topic="t", tool="notify"))
    timed = await create_trigger_link(topic="t", name="timed", ttl_seconds=3600, tool_kwargs={"k": 1}, created_by="a")
    perm = await create_trigger_link(topic="t", name="perm", ttl_seconds=None, tool_kwargs=None, created_by="a")

    doc = await sections._export_webhooks()
    assert [h["name"] for h in doc["hooks"]] == ["h1"]
    assert {link["name"] for link in doc["trigger_links"]} == {"timed", "perm"}
    assert doc["tombstones"] == []

    _wipe(store)
    report = await sections._import_webhooks(doc)
    assert report["errors"] == []
    assert report["created"] == 3  # one hook + two links
    # Both original URLs resolve again.
    assert (await resolve_trigger_token(timed["token"]))[0] == "t"
    assert (await resolve_trigger_token(perm["token"]))[0] == "t"


async def test_tool_kwargs_survive_roundtrip_and_merge(store) -> None:
    link = await create_trigger_link(
        topic="t", name="k", ttl_seconds=None, tool_kwargs={"flow": {"x": 1}}, created_by=None
    )
    doc = await sections._export_webhooks()
    _wipe(store)
    await sections._import_webhooks(doc)
    _topic, kwargs = await resolve_trigger_token(link["token"])
    assert kwargs == {"flow": {"x": 1}}


# -- tombstone durability -----------------------------------------------------


async def test_import_pre_revocation_export_stays_dead_via_local_tombstone(store) -> None:
    link = await create_trigger_link(topic="t", name="r", ttl_seconds=None, tool_kwargs=None, created_by=None)
    pre = await sections._export_webhooks()  # exported while live (no tombstone yet)
    await trigger_links.revoke_trigger_link("r")  # local tombstone now guards it
    report = await sections._import_webhooks(pre)
    assert report["skipped"] >= 1  # the tombstoned record is refused
    with pytest.raises(TriggerLinkError):
        await resolve_trigger_token(link["token"])


async def test_exported_tombstone_restores_and_gates(store) -> None:
    link = await create_trigger_link(topic="t", name="r", ttl_seconds=None, tool_kwargs=None, created_by=None)
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
                    "tool_kwargs": None,
                    "created_by": None,
                    "created_at": "2000-01-01T00:00:00+00:00",
                    "expires_at": "2000-01-02T00:00:00+00:00",
                },
            }
        ],
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert report["skipped"] == 1
    assert report["created"] == 0


async def test_live_name_collision_counts_updated(store) -> None:
    await create_trigger_link(topic="t", name="shared", ttl_seconds=None, tool_kwargs=None, created_by=None)
    doc = {
        "hooks": [],
        "trigger_links": [
            {
                "name": "shared",
                "token_hash": "b" * 64,
                "record": {
                    "name": "shared",
                    "topic": "t",
                    "tool_kwargs": None,
                    "created_by": None,
                    "created_at": "2026-07-21T00:00:00+00:00",
                    "expires_at": None,
                },
            }
        ],
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert report["updated"] == 1


async def test_one_hash_two_names_whole_section_refusal_zero_written(store) -> None:
    await store.manager.register(HookParams(name="h", topic="t", tool="notify"))
    rec = {
        "name": "",
        "topic": "t",
        "tool_kwargs": None,
        "created_by": None,
        "created_at": "2026-07-21T00:00:00+00:00",
        "expires_at": None,
    }
    doc = {
        "hooks": [{"name": "new-hook", "topic": "t", "tool": "notify"}],
        "trigger_links": [
            {"name": "A", "token_hash": "c" * 64, "record": {**rec, "name": "A"}},
            {"name": "B", "token_hash": "c" * 64, "record": {**rec, "name": "B"}},
        ],
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
    link = await create_trigger_link(topic="t", name="A", ttl_seconds=None, tool_kwargs=None, created_by=None)
    live = (await trigger_links.export_trigger_links())["trigger_links"][0]
    doc = {
        "hooks": [],
        "trigger_links": [{"name": "B", "token_hash": live["token_hash"], "record": {**live["record"], "name": "B"}}],
        "tombstones": [],
    }
    with pytest.raises(ValueError, match="already live under"):
        await sections._import_webhooks(doc)
    # A still resolves — nothing applied.
    assert (await resolve_trigger_token(link["token"]))[0] == "t"


async def test_orphan_index_dup_hash_whole_section_refusal_zero_written(store) -> None:
    # An ORPHAN in the store: name key A → hash H with NO record key (a corrupt
    # hand-edited backup). A normal create can't produce this, so seed raw state.
    # Import binds a NEW name B to that same H → whole-section refusal, zero written.
    # Without the fix the orphan-skipping live index misses H, the refusal never
    # fires, B→H is written, and later revoking A would destroy B's live record.
    orphan_hash = "c" * 64
    store.redis._set_str(store.settings.trigger_name_key("A"), orphan_hash)
    await store.manager.register(HookParams(name="h", topic="t", tool="notify"))
    doc = {
        "hooks": [{"name": "new-hook", "topic": "t", "tool": "notify"}],
        "trigger_links": [
            {
                "name": "B",
                "token_hash": orphan_hash,
                "record": {
                    "name": "B",
                    "topic": "t",
                    "tool_kwargs": None,
                    "created_by": None,
                    "created_at": "2026-07-21T00:00:00+00:00",
                    "expires_at": None,
                },
            }
        ],
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
        {"name": "g", "topic": "t", "created_at": "2026-07-21T00:00:00+00:00", "expires_at": "garbage"},
        {"name": "g", "topic": "", "created_at": "2026-07-21T00:00:00+00:00", "expires_at": None},
        {"name": "g", "topic": "t", "tool_kwargs": [1], "created_at": "2026-07-21T00:00:00+00:00", "expires_at": None},
    ],
)
async def test_per_item_malformation_error_and_skipped_rest_proceeds(store, record) -> None:
    doc = {
        "hooks": [{"name": "h", "topic": "t", "tool": "notify"}],
        "trigger_links": [{"name": "g", "token_hash": "f" * 64, "record": record}],
        "tombstones": [],
    }
    report = await sections._import_webhooks(doc)
    assert report["errors"]  # the malformed item is surfaced
    assert report["skipped"] == 1
    assert report["created"] == 1  # the hook still restored
    assert set(await store.manager.list_hooks()) == {"h"}


async def test_non_string_name_or_hash_per_item_error_valid_entries_restored(store) -> None:
    # A JSON-valid but ill-typed triple (non-str name, non-str token_hash) must
    # become a per-item report error + skipped — never an uncaught TypeError that
    # aborts the whole import after the hook and earlier records were written.
    good = {
        "name": "good",
        "topic": "t",
        "tool_kwargs": None,
        "created_by": None,
        "created_at": "2026-07-21T00:00:00+00:00",
        "expires_at": None,
    }
    doc = {
        "hooks": [{"name": "h", "topic": "t", "tool": "notify"}],
        "trigger_links": [
            {"name": 123, "token_hash": "a" * 64, "record": {**good, "name": "x"}},
            {"name": "y", "token_hash": 456, "record": {**good, "name": "y"}},
            {"name": "good", "token_hash": "b" * 64, "record": good},
        ],
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
        "tool_kwargs": None,
        "created_by": None,
        "created_at": "2026-07-21T00:00:00+00:00",
        "expires_at": None,
    }
    doc = {
        "hooks": [{"name": "h", "topic": "t", "tool": "notify"}],
        "trigger_links": [{"name": "good", "token_hash": "b" * 64, "record": good}],
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


# -- old shape + malformed envelope ------------------------------------------


async def test_old_list_shape_imports_clean(store) -> None:
    payload = [{"name": "h1", "topic": "t", "tool": "notify"}]
    report = await sections._import_webhooks(payload)
    assert report["created"] == 1
    assert set(await store.manager.list_hooks()) == {"h1"}


@pytest.mark.parametrize("missing", ["hooks", "trigger_links", "tombstones"])
async def test_envelope_missing_key_raises(store, missing) -> None:
    doc = {"hooks": [], "trigger_links": [], "tombstones": []}
    del doc[missing]
    with pytest.raises(ValueError, match="missing the required"):
        await sections._import_webhooks(doc)


@pytest.mark.parametrize(("key", "value"), [("hooks", {}), ("tombstones", "x")])
async def test_envelope_non_list_key_whole_section_refusal_zero_written(store, key, value) -> None:
    # A hand-edited envelope with a NON-LIST value for a required key is a loud
    # whole-section refusal naming the key, BEFORE any write — never a silent
    # degrade (an empty dict iterating to nothing) or an ungraceful failure deeper in.
    await store.manager.register(HookParams(name="pre", topic="t", tool="notify"))
    doc = {"hooks": [], "trigger_links": [], "tombstones": []}
    doc[key] = value
    with pytest.raises(ValueError, match=f"{key!r} must be a list"):
        await sections._import_webhooks(doc)
    # Zero keys written — the pre-existing hook is untouched.
    assert set(await store.manager.list_hooks()) == {"pre"}


# -- restore into a now-verified topic (fire-time enforcement) -------------


async def test_restore_into_verified_topic_created_but_resolves_404(store) -> None:
    link = await create_trigger_link(topic="secure", name="v", ttl_seconds=None, tool_kwargs=None, created_by=None)
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
    await in_memory_store.manager.register(HookParams(name="h1", topic="t", tool="notify"))
    doc = await sections._export_webhooks()
    assert [h["name"] for h in doc["hooks"]] == ["h1"]
    assert doc["trigger_links"] == []
    assert doc["tombstones"] == []


async def test_in_memory_import_refuses_trigger_portion_hooks_restore(in_memory_store, caplog) -> None:
    doc = {
        "hooks": [{"name": "h1", "topic": "t", "tool": "notify"}],
        "trigger_links": [
            {
                "name": "x",
                "token_hash": "a" * 64,
                "record": {
                    "name": "x",
                    "topic": "t",
                    "tool_kwargs": None,
                    "created_by": None,
                    "created_at": "2026-07-21T00:00:00+00:00",
                    "expires_at": None,
                },
            }
        ],
        "tombstones": ["b" * 64],
    }
    report = await sections._import_webhooks(doc)
    # The hooks portion restores; the trigger + tombstone portions refuse loudly.
    assert set(await in_memory_store.manager.list_hooks()) == {"h1"}
    assert len(report["errors"]) == 2  # one for the record, one for the tombstone
    assert report["created"] == 1
