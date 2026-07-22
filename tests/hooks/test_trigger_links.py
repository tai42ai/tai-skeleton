"""The trigger-link store: create/list/revoke/resolve, the backup seams, and the
in-memory refusal — driven through the extended ``FakeRedis`` (string ops +
injectable clock + the three atomic trigger scripts)."""

from __future__ import annotations

import json
import re
from datetime import datetime
from types import SimpleNamespace

import pytest
from tai42_kit.utils.data.string_util import hash_api_key

from tai42_skeleton.hooks import trigger_links
from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai42_skeleton.hooks.managers.redis_hooks_manager import RedisHooksManager
from tai42_skeleton.hooks.settings import HooksSettings
from tai42_skeleton.hooks.trigger_links import (
    TriggerLinkError,
    bound_hashes_by_name,
    create_trigger_link,
    export_trigger_links,
    list_trigger_links,
    resolve_trigger_token,
    restore_tombstone,
    restore_trigger_link,
    revoke_trigger_link,
)


@pytest.fixture
def store(monkeypatch, fake_redis, make_ctx):
    """A redis-backed trigger-link store over the fake: ``get_hooks_manager`` returns
    a ``RedisHooksManager`` (never in-memory) and both the module's and the manager's
    ``client_ctx`` yield the SAME fake, so verifier reads and store writes share it."""
    import tai42_skeleton.hooks.managers.redis_hooks_manager as rhm

    manager = RedisHooksManager(HooksSettings())
    monkeypatch.setattr(trigger_links, "get_hooks_manager", lambda: manager)
    monkeypatch.setattr(trigger_links, "client_ctx", make_ctx(fake_redis))
    monkeypatch.setattr(rhm, "client_ctx", make_ctx(fake_redis))
    return SimpleNamespace(manager=manager, redis=fake_redis, settings=manager.settings)


@pytest.fixture
def in_memory_store(monkeypatch):
    manager = InMemoryHooksManager(HooksSettings())
    monkeypatch.setattr(trigger_links, "get_hooks_manager", lambda: manager)
    return SimpleNamespace(manager=manager)


def _rec_keys(store) -> list[str]:
    return [k for k in store.redis._strings if k.startswith(store.settings.trigger_record_key_prefix)]


def _name_keys(store) -> list[str]:
    return [k for k in store.redis._strings if k.startswith(store.settings.trigger_name_key_prefix)]


def _tomb_keys(store) -> list[str]:
    return [k for k in store.redis._strings if k.startswith(store.settings.trigger_tomb_key_prefix)]


# -- FakeRedis self-tests ----------------------------


async def test_fake_set_get_delete_exists(fake_redis) -> None:
    assert await fake_redis.set("k", "v") is True
    assert await fake_redis.get("k") == "v"
    assert await fake_redis.exists("k") == 1
    assert await fake_redis.delete("k") == 1
    assert await fake_redis.get("k") is None
    assert await fake_redis.exists("k") == 0


async def test_fake_ex_expiry_uses_injectable_clock(fake_redis) -> None:
    await fake_redis.set("k", "v", ex=10)
    fake_redis.advance(9)
    assert await fake_redis.get("k") == "v"
    fake_redis.advance(1)
    assert await fake_redis.get("k") is None


async def test_fake_mget_and_paging_scan(fake_redis) -> None:
    for i in range(25):
        await fake_redis.set(f"p:{i:02d}", str(i))
    assert await fake_redis.mget(["p:00", "missing", "p:24"]) == ["0", None, "24"]
    seen: list[str] = []
    cursor = 0
    pages = 0
    while True:
        cursor, batch = await fake_redis.scan(cursor, match="p:*", count=100)
        seen.extend(batch)
        pages += 1
        if cursor == 0:
            break
    assert len(seen) == 25
    assert pages >= 3  # deliberately multi-page, so a first-page-only bug fails


# -- create round-trips -------------------------------------------------------


async def test_create_timed_roundtrip_with_tool_kwargs(store) -> None:
    result = await create_trigger_link(
        topic="orders", name="link1", ttl_seconds=3600, tool_kwargs={"a": 1}, created_by="alice"
    )
    assert result["name"] == "link1"
    assert result["topic"] == "orders"
    assert result["trigger_path"] == f"/trigger/{result['token']}"
    assert result["expires_at"] is not None
    topic, kwargs = await resolve_trigger_token(result["token"])
    assert (topic, kwargs) == ("orders", {"a": 1})


async def test_create_permanent_roundtrip_without_tool_kwargs(store) -> None:
    result = await create_trigger_link(topic="orders", name="perm", ttl_seconds=None, tool_kwargs=None, created_by=None)
    assert result["expires_at"] is None
    topic, kwargs = await resolve_trigger_token(result["token"])
    assert (topic, kwargs) == ("orders", None)


async def test_create_empty_tool_kwargs_stored_verbatim(store) -> None:
    result = await create_trigger_link(topic="t", name="e", ttl_seconds=None, tool_kwargs={}, created_by=None)
    _topic, kwargs = await resolve_trigger_token(result["token"])
    assert kwargs == {}


async def test_create_non_dict_tool_kwargs_400(store) -> None:
    with pytest.raises(TriggerLinkError) as ei:
        await create_trigger_link(topic="t", name="x", ttl_seconds=None, tool_kwargs=[1, 2], created_by=None)  # type: ignore[arg-type]
    assert ei.value.status == 400


async def test_expires_at_minus_created_at_equals_ttl(store) -> None:
    result = await create_trigger_link(topic="t", name="ttlpin", ttl_seconds=3600, tool_kwargs=None, created_by=None)
    listing = await list_trigger_links()
    (record,) = [r for r in listing["items"] if r["name"] == "ttlpin"]
    delta = datetime.fromisoformat(result["expires_at"]) - datetime.fromisoformat(record["created_at"])
    assert delta.total_seconds() == 3600


@pytest.mark.parametrize("bad_ttl", [0, -1, -3600])
async def test_ttl_zero_or_negative_400(store, bad_ttl) -> None:
    with pytest.raises(TriggerLinkError) as ei:
        await create_trigger_link(topic="t", name="x", ttl_seconds=bad_ttl, tool_kwargs=None, created_by=None)
    assert ei.value.status == 400


async def test_physical_ttl_bound_boundary(store) -> None:
    ok = await create_trigger_link(topic="t", name="onbound", ttl_seconds=10**10, tool_kwargs=None, created_by=None)
    assert ok["name"] == "onbound"
    for over in (10**10 + 1, 10**16):
        with pytest.raises(TriggerLinkError) as ei:
            await create_trigger_link(
                topic="t", name=f"over{over}", ttl_seconds=over, tool_kwargs=None, created_by=None
            )
        assert ei.value.status == 400


# -- name rules ---------------------------------------------------------------

_INVALID_NAMES = ["foo/bar", "a" * 65, "", ".", "..", "-", "--", ".-", "bad name", "bad$char", "a\tb", "abc\n"]


@pytest.mark.parametrize("name", _INVALID_NAMES)
async def test_invalid_name_400(store, name) -> None:
    with pytest.raises(TriggerLinkError) as ei:
        await create_trigger_link(topic="t", name=name, ttl_seconds=None, tool_kwargs=None, created_by=None)
    assert ei.value.status == 400
    assert _name_keys(store) == []  # nothing written on a rejected name


async def test_64_char_name_accepted(store) -> None:
    name = "a" * 64
    result = await create_trigger_link(topic="t", name=name, ttl_seconds=None, tool_kwargs=None, created_by=None)
    assert result["name"] == name


async def test_nameless_create_uses_default_name(store) -> None:
    result = await create_trigger_link(topic="t", name=None, ttl_seconds=None, tool_kwargs=None, created_by=None)
    assert re.fullmatch(r"trg-link-[0-9a-f]{8}", result["name"])
    listing = await list_trigger_links()
    assert result["name"] in {r["name"] for r in listing["items"]}
    await revoke_trigger_link(result["name"])  # round-trips through revoke


# -- generated-name collision ---------------------------------------


async def test_generated_name_first_collision_then_fresh_succeeds(store, monkeypatch) -> None:
    await create_trigger_link(topic="t", name="taken-gen", ttl_seconds=None, tool_kwargs=None, created_by=None)
    names = iter(["taken-gen", "fresh-gen"])
    monkeypatch.setattr(trigger_links, "_default_name", lambda: next(names))
    result = await create_trigger_link(topic="t", name=None, ttl_seconds=None, tool_kwargs=None, created_by=None)
    assert result["name"] == "fresh-gen"


async def test_generated_name_double_collision_raises(store, monkeypatch) -> None:
    await create_trigger_link(topic="t", name="always", ttl_seconds=None, tool_kwargs=None, created_by=None)
    monkeypatch.setattr(trigger_links, "_default_name", lambda: "always")
    with pytest.raises(TriggerLinkError) as ei:
        await create_trigger_link(topic="t", name=None, ttl_seconds=None, tool_kwargs=None, created_by=None)
    assert ei.value.status == 409


# -- duplicate explicit name (409) + one-eval structural pin -----------------


async def test_duplicate_explicit_name_409_loser_record_not_written(store) -> None:
    await create_trigger_link(topic="t", name="dup", ttl_seconds=None, tool_kwargs=None, created_by=None)
    before = set(_rec_keys(store))
    with pytest.raises(TriggerLinkError) as ei:
        await create_trigger_link(topic="t", name="dup", ttl_seconds=None, tool_kwargs=None, created_by=None)
    assert ei.value.status == 409
    # The loser minted a fresh token but its record key was never written.
    assert set(_rec_keys(store)) == before


async def test_create_revoke_restore_are_each_one_eval(store, monkeypatch) -> None:
    calls: list[str] = []
    original = store.redis.eval

    async def _counting(script, numkeys, *args):
        calls.append(script)
        return await original(script, numkeys, *args)

    monkeypatch.setattr(store.redis, "eval", _counting)
    await create_trigger_link(topic="t", name="one", ttl_seconds=None, tool_kwargs=None, created_by=None)
    assert sum("trigger:create:atomic" in c for c in calls) == 1
    await revoke_trigger_link("one")
    assert sum("trigger:revoke:atomic" in c for c in calls) == 1
    await restore_trigger_link(
        name="two",
        token_hash="a" * 64,
        record={
            "name": "two",
            "topic": "t",
            "tool_kwargs": None,
            "created_by": None,
            "created_at": "2030-01-01T00:00:00",
            "expires_at": None,
        },
    )
    assert sum("trigger:restore:atomic" in c for c in calls) == 1


@pytest.mark.parametrize(
    ("name", "token_hash", "match"),
    [
        (123, "a" * 64, "name must be a string"),
        ("ok", 456, "token_hash must be a string"),
    ],
)
async def test_restore_non_string_name_or_hash_400(store, name, token_hash, match) -> None:
    with pytest.raises(TriggerLinkError) as ei:
        await restore_trigger_link(
            name=name,
            token_hash=token_hash,
            record={
                "name": "ok",
                "topic": "t",
                "tool_kwargs": None,
                "created_by": None,
                "created_at": "2030-01-01T00:00:00",
                "expires_at": None,
            },
        )
    assert ei.value.status == 400
    assert match in ei.value.message
    assert _name_keys(store) == []  # nothing written on a rejected triple


# -- verifier binding ----------------------------------------------------


async def test_create_refused_on_verifier_bound_topic_400(store) -> None:
    await store.manager.set_topic_verifier("secure", {"verifier": "hmac", "config": {}})
    with pytest.raises(TriggerLinkError) as ei:
        await create_trigger_link(topic="secure", name="x", ttl_seconds=None, tool_kwargs=None, created_by=None)
    assert ei.value.status == 400
    assert _name_keys(store) == []


async def test_resolve_on_late_bound_verifier_404_and_logs(store, caplog) -> None:
    result = await create_trigger_link(topic="late", name="l", ttl_seconds=None, tool_kwargs=None, created_by=None)
    await store.manager.set_topic_verifier("late", {"verifier": "hmac", "config": {}})
    with caplog.at_level("INFO"), pytest.raises(TriggerLinkError) as ei:
        await resolve_trigger_token(result["token"])
    assert ei.value.status == 404
    assert "verifier-bound" in caplog.text


# -- resolve misses -----------------------------------------------------------


async def test_resolve_after_revoke_404(store) -> None:
    result = await create_trigger_link(topic="t", name="r", ttl_seconds=None, tool_kwargs=None, created_by=None)
    await revoke_trigger_link("r")
    with pytest.raises(TriggerLinkError) as ei:
        await resolve_trigger_token(result["token"])
    assert ei.value.status == 404


async def test_resolve_after_ttl_expiry_404_and_same_name_recreatable(store) -> None:
    result = await create_trigger_link(topic="t", name="timed", ttl_seconds=100, tool_kwargs=None, created_by=None)
    store.redis.advance(101)
    with pytest.raises(TriggerLinkError) as ei:
        await resolve_trigger_token(result["token"])
    assert ei.value.status == 404
    # The name key expired WITH the record (both-keys-EX), so the name is free again.
    again = await create_trigger_link(topic="t", name="timed", ttl_seconds=None, tool_kwargs=None, created_by=None)
    topic, _ = await resolve_trigger_token(again["token"])
    assert topic == "t"


async def test_five_miss_causes_are_byte_equal(store, monkeypatch, in_memory_store_factory) -> None:
    bodies: list[str] = []

    # unknown
    with pytest.raises(TriggerLinkError) as ei:
        await resolve_trigger_token("trg-never-existed")
    bodies.append(ei.value.message)

    # expired
    exp = await create_trigger_link(topic="t", name="m-exp", ttl_seconds=50, tool_kwargs=None, created_by=None)
    store.redis.advance(51)
    with pytest.raises(TriggerLinkError) as ei:
        await resolve_trigger_token(exp["token"])
    bodies.append(ei.value.message)

    # revoked
    rev = await create_trigger_link(topic="t", name="m-rev", ttl_seconds=None, tool_kwargs=None, created_by=None)
    await revoke_trigger_link("m-rev")
    with pytest.raises(TriggerLinkError) as ei:
        await resolve_trigger_token(rev["token"])
    bodies.append(ei.value.message)

    # verifier-bound
    vb = await create_trigger_link(topic="vb", name="m-vb", ttl_seconds=None, tool_kwargs=None, created_by=None)
    await store.manager.set_topic_verifier("vb", {"verifier": "h", "config": {}})
    with pytest.raises(TriggerLinkError) as ei:
        await resolve_trigger_token(vb["token"])
    bodies.append(ei.value.message)

    # in-memory
    in_memory_store_factory()
    with pytest.raises(TriggerLinkError) as ei:
        await resolve_trigger_token("trg-anything")
    bodies.append(ei.value.message)

    assert len(set(bodies)) == 1


# -- fail-closed 500s ---------------------------------------------------------


async def test_erroring_verifier_at_create_propagates_nothing_written(store, monkeypatch) -> None:
    async def _boom(topic):
        raise RuntimeError("verifier store down")

    monkeypatch.setattr(store.manager, "get_topic_verifier", _boom)
    with pytest.raises(RuntimeError):
        await create_trigger_link(topic="t", name="x", ttl_seconds=None, tool_kwargs=None, created_by=None)
    assert _name_keys(store) == []
    assert _rec_keys(store) == []


async def test_erroring_verifier_at_resolve_propagates(store, monkeypatch) -> None:
    result = await create_trigger_link(topic="t", name="x", ttl_seconds=None, tool_kwargs=None, created_by=None)

    async def _boom(topic):
        raise RuntimeError("verifier store down")

    monkeypatch.setattr(store.manager, "get_topic_verifier", _boom)
    with pytest.raises(RuntimeError):
        await resolve_trigger_token(result["token"])


async def test_corrupt_stored_record_at_resolve_raises(store) -> None:
    token = "trg-corrupt-token"
    token_hash = hash_api_key(token)
    store.redis._set_str(store.settings.trigger_record_key(token_hash), "{not json")
    with pytest.raises(json.JSONDecodeError):
        await resolve_trigger_token(token)


# -- list ---------------------------------------------------------------------


async def test_list_returns_records_and_prefix_no_token_multipage(store) -> None:
    tokens = {}
    for i in range(25):
        r = await create_trigger_link(
            topic="t", name=f"n{i:02d}", ttl_seconds=None, tool_kwargs={"i": i}, created_by="bob"
        )
        tokens[r["name"]] = r["token"]
    listing = await list_trigger_links()
    assert listing["total"] == 25
    for record in listing["items"]:
        assert "token" not in record
        assert set(record) >= {
            "name",
            "topic",
            "tool_kwargs",
            "created_by",
            "created_at",
            "expires_at",
            "token_hash_prefix",
        }
        assert len(record["token_hash_prefix"]) == 12
        assert record["token_hash_prefix"] == hash_api_key(tokens[record["name"]])[:12]


async def test_list_skips_permanent_orphan_warns_but_nil_race_silent(store, caplog) -> None:
    # A live link.
    await create_trigger_link(topic="t", name="alive", ttl_seconds=None, tool_kwargs=None, created_by=None)
    # A PERMANENT orphan: a name key with a hash whose record is absent.
    store.redis._set_str(store.settings.trigger_name_key("orphan"), "a" * 64)
    with caplog.at_level("WARNING"):
        listing = await list_trigger_links()
    assert {r["name"] for r in listing["items"]} == {"alive"}
    assert "orphan" in caplog.text  # the permanent orphan is logged by name


async def test_list_nil_name_race_logs_nothing(store, caplog) -> None:
    await create_trigger_link(topic="t", name="alive", ttl_seconds=None, tool_kwargs=None, created_by=None)
    # An expiring name key that vanishes before MGET is a pure TTL race — no warning.
    store.redis._set_str(store.settings.trigger_name_key("racing"), "b" * 64, ex=5)
    store.redis.advance(6)
    with caplog.at_level("WARNING"):
        listing = await list_trigger_links()
    assert {r["name"] for r in listing["items"]} == {"alive"}
    assert "racing" not in caplog.text


# -- tombstones + revoke ------------------------------------------------------


async def test_revoke_writes_tombstone_and_resolve_of_tombstoned_hash_404(store) -> None:
    result = await create_trigger_link(topic="t", name="rev", ttl_seconds=None, tool_kwargs=None, created_by=None)
    token_hash = hash_api_key(result["token"])
    await revoke_trigger_link("rev")
    assert store.settings.trigger_tomb_key(token_hash) in store.redis._strings
    # Belt-and-braces: even with a live record key present, a tombstone → uniform 404.
    store.redis._set_str(store.settings.trigger_record_key(token_hash), json.dumps({"topic": "t"}))
    with pytest.raises(TriggerLinkError) as ei:
        await resolve_trigger_token(result["token"])
    assert ei.value.status == 404
    with pytest.raises(TriggerLinkError) as unknown:
        await resolve_trigger_token("trg-unknown")
    assert ei.value.message == unknown.value.message


async def test_revoke_unknown_name_404(store) -> None:
    with pytest.raises(TriggerLinkError) as ei:
        await revoke_trigger_link("nope")
    assert ei.value.status == 404


async def test_no_orphan_revoke_vs_recreate_both_orderings(store) -> None:
    a = await create_trigger_link(topic="t", name="x", ttl_seconds=None, tool_kwargs=None, created_by=None)
    await revoke_trigger_link("x")
    b = await create_trigger_link(topic="t", name="x", ttl_seconds=None, tool_kwargs=None, created_by=None)
    # Exactly one live name key + one live record key for the surviving link.
    assert len(_name_keys(store)) == 1
    assert len(_rec_keys(store)) == 1
    topic, _ = await resolve_trigger_token(b["token"])
    assert topic == "t"
    # The revoked token is dead.
    with pytest.raises(TriggerLinkError):
        await resolve_trigger_token(a["token"])


# -- restore refusals ---------------------------------------------------------


def _valid_record(name="rn", topic="t"):
    return {
        "name": name,
        "topic": topic,
        "tool_kwargs": None,
        "created_by": None,
        "created_at": "2026-07-21T00:00:00+00:00",
        "expires_at": None,
    }


async def test_restore_refuses_name_mismatch(store) -> None:
    with pytest.raises(TriggerLinkError):
        await restore_trigger_link(name="idx", token_hash="a" * 64, record=_valid_record(name="other"))


@pytest.mark.parametrize("bad_name", ["foo/bar", "-lead", "", "a" * 65])
async def test_restore_refuses_pattern_violating_name(store, bad_name) -> None:
    with pytest.raises(TriggerLinkError):
        await restore_trigger_link(name=bad_name, token_hash="a" * 64, record=_valid_record(name=bad_name))


@pytest.mark.parametrize("bad_hash", ["nothex", "A" * 64, "a" * 63, "a" * 65])
async def test_restore_refuses_non_hex_hash(store, bad_hash) -> None:
    with pytest.raises(TriggerLinkError):
        await restore_trigger_link(name="rn", token_hash=bad_hash, record=_valid_record())


async def test_restore_refuses_unparseable_expires_at(store) -> None:
    record = _valid_record()
    record["expires_at"] = "not-a-timestamp"
    with pytest.raises(TriggerLinkError):
        await restore_trigger_link(name="rn", token_hash="a" * 64, record=record)


@pytest.mark.parametrize("mutate", [{"topic": ""}, {"tool_kwargs": [1, 2]}])
async def test_restore_refuses_malformed_body(store, mutate) -> None:
    record = _valid_record()
    record.update(mutate)
    with pytest.raises(TriggerLinkError):
        await restore_trigger_link(name="rn", token_hash="a" * 64, record=record)


async def test_restore_refuses_hash_with_trailing_newline(store) -> None:
    with pytest.raises(TriggerLinkError) as ei:
        await restore_trigger_link(name="rn", token_hash="a" * 64 + "\n", record=_valid_record(name="rn"))
    assert ei.value.status == 400


async def test_restore_refuses_unknown_field(store) -> None:
    record = _valid_record(name="rn")
    record["unexpected_field"] = "x"
    with pytest.raises(TriggerLinkError) as ei:
        await restore_trigger_link(name="rn", token_hash="a" * 64, record=record)
    assert ei.value.status == 400


async def test_restore_refuses_hash_live_under_different_name(store) -> None:
    result = await create_trigger_link(topic="t", name="livename", ttl_seconds=None, tool_kwargs=None, created_by=None)
    token_hash = hash_api_key(result["token"])
    with pytest.raises(TriggerLinkError) as ei:
        await restore_trigger_link(name="othername", token_hash=token_hash, record=_valid_record(name="othername"))
    assert ei.value.status == 400


# -- restore / revoke ordering + no-orphan -----------------------------------


async def test_restore_after_revoke_refused_tombstoned(store) -> None:
    result = await create_trigger_link(topic="t", name="tw", ttl_seconds=None, tool_kwargs=None, created_by=None)
    exported = (await export_trigger_links())["trigger_links"][0]
    await revoke_trigger_link("tw")
    outcome = await restore_trigger_link(
        name=exported["name"], token_hash=exported["token_hash"], record=exported["record"]
    )
    assert outcome == "skipped_tombstoned"
    # No live pair slipped in behind the tombstone.
    with pytest.raises(TriggerLinkError):
        await resolve_trigger_token(result["token"])


async def test_restore_over_live_different_hash_deletes_displaced_no_tombstone(store) -> None:
    first = await create_trigger_link(topic="t", name="shared", ttl_seconds=None, tool_kwargs=None, created_by=None)
    first_hash = hash_api_key(first["token"])
    # A second exported record under the SAME name but a different hash.
    other_hash = "c" * 64
    outcome = await restore_trigger_link(name="shared", token_hash=other_hash, record=_valid_record(name="shared"))
    assert outcome == "updated"
    # The displaced hash's record key is gone, and NO tombstone was written for it.
    assert store.settings.trigger_record_key(first_hash) not in store.redis._strings
    assert _tomb_keys(store) == []
    with pytest.raises(TriggerLinkError):
        await resolve_trigger_token(first["token"])


async def test_restore_over_same_hash_updates_still_resolves_no_tombstone(store) -> None:
    result = await create_trigger_link(topic="t", name="same", ttl_seconds=None, tool_kwargs={"k": 1}, created_by=None)
    exported = (await export_trigger_links())["trigger_links"][0]
    outcome = await restore_trigger_link(
        name=exported["name"], token_hash=exported["token_hash"], record=exported["record"]
    )
    assert outcome == "updated"
    assert _tomb_keys(store) == []
    topic, kwargs = await resolve_trigger_token(result["token"])
    assert (topic, kwargs) == ("t", {"k": 1})


async def test_restore_expired_record_skipped(store) -> None:
    record = _valid_record()
    record["expires_at"] = "2000-01-01T00:00:00+00:00"
    outcome = await restore_trigger_link(name="rn", token_hash="a" * 64, record=record)
    assert outcome == "skipped_expired"
    assert _rec_keys(store) == []


async def test_restore_sub_second_remaining_lives_with_ex_one(store, monkeypatch) -> None:
    # A record whose deadline is 0.4s away restores with EX 1 (ceil), never EX 0.
    from tai42_skeleton.hooks import trigger_links as tl

    class _Now:
        @staticmethod
        def fromisoformat(value):
            return datetime.fromisoformat(value)

    deadline = "2026-07-21T00:00:00.400000+00:00"
    monkeypatch.setattr(
        tl,
        "datetime",
        SimpleNamespace(
            fromisoformat=datetime.fromisoformat,
            now=lambda tz=None: datetime.fromisoformat("2026-07-21T00:00:00+00:00"),
        ),
    )
    record = _valid_record()
    record["expires_at"] = deadline
    outcome = await restore_trigger_link(name="rn", token_hash="d" * 64, record=record)
    assert outcome in ("created", "updated")
    # The record lives (EX 1, not EX 0 which errors); the key is present at t=0.
    assert store.settings.trigger_record_key("d" * 64) in store.redis._strings


async def test_restore_tombstone_idempotent(store) -> None:
    h = "e" * 64
    await restore_tombstone(h)
    await restore_tombstone(h)
    assert store.settings.trigger_tomb_key(h) in store.redis._strings


@pytest.mark.parametrize("token_hash", [123])
async def test_restore_tombstone_non_string_hash_400(store, token_hash) -> None:
    # A non-str token_hash is a typed refusal, never a raw TypeError out of the regex.
    with pytest.raises(TriggerLinkError, match="token_hash must be a string"):
        await restore_tombstone(token_hash)


# -- _scan_all rehash dedupe --------------------------------------------------


async def test_scan_all_dedupes_rehash_duplicate_keys() -> None:
    class _DupScan:
        def __init__(self) -> None:
            self._calls = 0

        async def scan(self, cursor, match=None, count=None):
            self._calls += 1
            if self._calls == 1:
                return 1, ["k:a", "k:a", "k:b"]
            return 0, ["k:a"]

    result = await trigger_links._scan_all(_DupScan(), "k:*")
    assert result == ["k:a", "k:b"]


# -- export -------------------------------------------------------------------


async def test_export_carries_hashes_and_tombstones_no_token_multipage(store) -> None:
    tokens = []
    for i in range(15):
        r = await create_trigger_link(topic="t", name=f"e{i:02d}", ttl_seconds=None, tool_kwargs=None, created_by=None)
        tokens.append(r["token"])
    await revoke_trigger_link("e00")  # one tombstone
    exported = await export_trigger_links()
    assert len(exported["trigger_links"]) == 14  # the revoked one is gone from the index
    assert len(exported["tombstones"]) == 1
    dumped = json.dumps(exported)
    for token in tokens:
        assert token not in dumped
    for item in exported["trigger_links"]:
        assert len(item["token_hash"]) == 64


# -- bound-hashes index (orphans included) ------------------------------------


async def test_bound_hashes_by_name_includes_orphans(store) -> None:
    # A live link plus a PERMANENT orphan (name key → hash with no record). The
    # bound-hashes index the import conflict check reads MUST surface both, unlike the
    # orphan-skipping export.
    live = await create_trigger_link(topic="t", name="alive", ttl_seconds=None, tool_kwargs=None, created_by=None)
    live_hash = hash_api_key(live["token"])
    orphan_hash = "a" * 64
    store.redis._set_str(store.settings.trigger_name_key("orphan"), orphan_hash)

    bindings = await bound_hashes_by_name()
    assert bindings == {"alive": live_hash, "orphan": orphan_hash}
    # The orphan-skipping export omits the orphan — the exact gap this index closes.
    exported = await export_trigger_links()
    assert {e["name"] for e in exported["trigger_links"]} == {"alive"}


# -- in-memory refusal --------------------------------------------------------


async def test_in_memory_crud_501_resolve_404(in_memory_store) -> None:
    for coro in (
        create_trigger_link(topic="t", name="x", ttl_seconds=None, tool_kwargs=None, created_by=None),
        list_trigger_links(),
        revoke_trigger_link("x"),
        restore_tombstone("a" * 64),
        restore_trigger_link(name="x", token_hash="a" * 64, record=_valid_record(name="x")),
    ):
        with pytest.raises(TriggerLinkError) as ei:
            await coro
        assert ei.value.status == 501
    with pytest.raises(TriggerLinkError) as ei:
        await resolve_trigger_token("trg-anything")
    assert ei.value.status == 404


async def test_in_memory_export_truthfully_empty(in_memory_store) -> None:
    assert await export_trigger_links() == {"trigger_links": [], "tombstones": []}


# -- log doctrine --------------------------------------------------------


async def test_log_doctrine_no_raw_token_and_correlation(store, caplog) -> None:
    with caplog.at_level("INFO"):
        result = await create_trigger_link(
            topic="topicX", name="logname", ttl_seconds=1200, tool_kwargs=None, created_by="carol"
        )
        listing = await list_trigger_links()
        await resolve_trigger_token(result["token"])
        await revoke_trigger_link("logname")
    text = caplog.text
    # create line carries caller + name + topic + ttl.
    for token_part in ("carol", "logname", "topicX", "1200"):
        assert token_part in text
    # resolve outcome line carries the hash prefix matching the list's token_hash_prefix.
    (record,) = [r for r in listing["items"] if r["name"] == "logname"]
    assert record["token_hash_prefix"] in text
    # revoke line carries name + hash prefix.
    assert "revoked name=logname" in text
    # NEVER the raw token.
    assert result["token"] not in text


# -- fixtures for the five-way test ------------------------------------------


@pytest.fixture
def in_memory_store_factory(monkeypatch):
    def _install():
        manager = InMemoryHooksManager(HooksSettings())
        monkeypatch.setattr(trigger_links, "get_hooks_manager", lambda: manager)
        return manager

    return _install
