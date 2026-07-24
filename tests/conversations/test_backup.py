"""The ``conversations`` backup section: the export excludes each row's
``callback_secret``; the import re-mints one per ``api`` row and surfaces it in
``new_callback_secrets``, while asserting the bound execution key is still live.

The live-key assertion runs for real: a scan that REFUSES a named key stands in for a
key revoked since the backup was taken."""

from __future__ import annotations

import pytest
from tai42_contract.conversations import ConversationRoute

from tai42_skeleton.authz.execution import ExecutionKeyAuthorityError
from tai42_skeleton.authz.token_free import TokenFreeConditionError
from tai42_skeleton.conversations import backup
from tai42_skeleton.conversations.managers.base_conversations_manager import BaseConversationsManager
from tai42_skeleton.conversations.settings import ConversationsSettings


class _DictManager(BaseConversationsManager):
    def __init__(self) -> None:
        super().__init__(ConversationsSettings())
        self.rows: dict[str, ConversationRoute] = {}

    async def put_route(self, route: ConversationRoute) -> bool:
        created = route.route_name not in self.rows
        self.rows[route.route_name] = route
        return created

    async def get_route(self, route_name):
        return self.rows.get(route_name)

    async def delete_route(self, route_name):
        return self.rows.pop(route_name, None) is not None

    async def list_routes(self):
        return dict(self.rows)


class _NoopScan:
    """A scan every key clears — the live-key assertion is pinned by the refusing scans
    below, so the rows here are about the export/import shape."""

    async def assert_usable(self, execution_key, *, bound_fingerprint):
        return None


class _RefusingScan:
    """A scan that refuses ONE named key, the way the live gate refuses a key that was
    revoked and re-minted (its live fingerprint no longer matches the row's) — every
    other key clears, so a refusal is shown to be per row."""

    refused = "revoked"
    error: Exception = ExecutionKeyAuthorityError(
        "access denied: execution key 'revoked' has no policy", principal="revoked", defect="has no policy"
    )

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    async def assert_usable(self, execution_key, *, bound_fingerprint):
        self.seen.append((execution_key, bound_fingerprint))
        if execution_key == self.refused:
            raise self.error


class _ConditionRefusingScan(_RefusingScan):
    """The scan's other per-row refusal: a key whose policy condition no background turn
    could evaluate."""

    error = TokenFreeConditionError("condition reads an identity claim beyond a tokenless fire")


@pytest.fixture
def wired(monkeypatch):
    manager = _DictManager()
    monkeypatch.setattr(backup, "get_conversations_manager", lambda: manager)
    monkeypatch.setattr(backup, "ExecutionKeyScan", _NoopScan)
    return manager


def _api_route(name: str, secret: str, execution_key: str = "svc") -> ConversationRoute:
    return ConversationRoute(
        route_name=name,
        door="api",
        agent_name="triage",
        execution_key=execution_key,
        callback_url="https://example.com/cb",
        execution_key_fingerprint="fp-1",
        callback_secret=secret,
    )


def _channel_route(name: str) -> ConversationRoute:
    return ConversationRoute(
        route_name=name,
        door="channel",
        agent_name="triage",
        execution_key="svc",
        channel="twilio",
        our_identity="+15550001111",
        execution_key_fingerprint="fp-1",
    )


async def test_export_excludes_the_callback_secret(wired):
    await wired.put_route(_api_route("support", "live-secret"))
    payload = await backup.export_conversation_routes()
    assert payload["routes"][0]["route_name"] == "support"
    assert "callback_secret" not in payload["routes"][0]
    # The fingerprint IS exported (import re-asserts the live key against it).
    assert payload["routes"][0]["execution_key_fingerprint"] == "fp-1"


async def test_import_remints_secret_and_surfaces_it(wired):
    # The export shape carries no secret.
    row = _api_route("support", "old").model_dump(mode="json")
    del row["callback_secret"]
    payload = {"routes": [row]}

    report = await backup.import_conversation_routes(payload)
    assert report["created"] == 1
    assert len(report["new_callback_secrets"]) == 1
    entry = report["new_callback_secrets"][0]
    assert entry["route_name"] == "support"
    # The re-minted secret is now the stored one.
    assert wired.rows["support"].callback_secret == entry["callback_secret"]
    assert entry["callback_secret"]


async def test_import_channel_row_carries_no_secret(wired):
    payload = {"routes": [_channel_route("line").model_dump(mode="json")]}
    report = await backup.import_conversation_routes(payload)
    assert report["created"] == 1
    assert report["new_callback_secrets"] == []
    assert wired.rows["line"].callback_secret is None


async def test_round_trip_preserves_every_field_but_the_secret(monkeypatch):
    source = _DictManager()
    await source.put_route(_api_route("support", "live-secret"))
    monkeypatch.setattr(backup, "ExecutionKeyScan", _NoopScan)
    monkeypatch.setattr(backup, "get_conversations_manager", lambda: source)
    payload = await backup.export_conversation_routes()

    # Restore into a fresh store.
    restored = _DictManager()
    monkeypatch.setattr(backup, "get_conversations_manager", lambda: restored)
    report = await backup.import_conversation_routes(payload)

    assert report["created"] == 1
    row = restored.rows["support"]
    assert row.agent_name == "triage"
    assert row.execution_key == "svc"
    assert row.callback_url == "https://example.com/cb"
    # The secret is a fresh mint, not the pre-export one.
    assert row.callback_secret
    assert row.callback_secret != "live-secret"


async def test_import_rejects_a_malformed_envelope(wired):
    with pytest.raises(ValueError, match="missing the required 'routes'"):
        await backup.import_conversation_routes({})
    with pytest.raises(ValueError, match="'routes' must be a list"):
        await backup.import_conversation_routes({"routes": {}})


async def test_import_rejects_a_row_missing_its_fingerprint(wired):
    bad = {
        "route_name": "support",
        "door": "api",
        "agent_name": "triage",
        "execution_key": "svc",
        "callback_url": "https://example.com/cb",
        # execution_key_fingerprint missing
    }
    report = await backup.import_conversation_routes({"routes": [bad]})
    assert report["skipped"] == 1
    assert report["errors"]
    assert "support" not in wired.rows


async def test_export_empty_without_a_backend(monkeypatch):
    from tai42_skeleton.conversations.managers.in_memory_conversations_manager import InMemoryConversationsManager

    monkeypatch.setattr(
        backup, "get_conversations_manager", lambda: InMemoryConversationsManager(ConversationsSettings())
    )
    assert await backup.export_conversation_routes() == {"routes": []}


async def test_a_backend_less_deployment_restores_its_own_empty_backup(monkeypatch):
    # A backend-less host exports empty and restores as a clean no-op, not a section
    # failure on every full restore.
    from tai42_skeleton.conversations.managers.in_memory_conversations_manager import InMemoryConversationsManager

    monkeypatch.setattr(
        backup, "get_conversations_manager", lambda: InMemoryConversationsManager(ConversationsSettings())
    )
    report = await backup.import_conversation_routes({"routes": []})
    assert report["created"] == 0
    assert report["skipped"] == 0
    assert report["errors"] == []


async def test_a_backend_less_deployment_refuses_to_restore_actual_rows(monkeypatch):
    from tai42_skeleton.conversations.managers.in_memory_conversations_manager import InMemoryConversationsManager

    monkeypatch.setattr(
        backup, "get_conversations_manager", lambda: InMemoryConversationsManager(ConversationsSettings())
    )
    row = _channel_route("line").model_dump(mode="json")
    with pytest.raises(RuntimeError, match="require the redis conversations backend"):
        await backup.import_conversation_routes({"routes": [row]})


# -- the restore-time live-key assertion --------------------------------------


def _use_scan(monkeypatch, scan) -> None:
    monkeypatch.setattr(backup, "ExecutionKeyScan", lambda: scan)


async def test_import_skips_a_row_whose_bound_execution_key_is_no_longer_live(wired, monkeypatch):
    # The row's key was reminted since the backup, so the scan refuses it: that row is
    # rejected on its own while its live-key sibling still restores.
    scan = _RefusingScan()
    _use_scan(monkeypatch, scan)
    dead = _api_route("billing", "old", execution_key=_RefusingScan.refused).model_dump(mode="json")
    live = _api_route("support", "old").model_dump(mode="json")
    for row in (dead, live):
        del row["callback_secret"]

    report = await backup.import_conversation_routes({"routes": [dead, live]})

    assert report["skipped"] == 1
    assert report["created"] == 1
    assert any("billing" in error and "has no policy" in error for error in report["errors"])
    assert "billing" not in wired.rows  # the refused row was never written
    assert "support" in wired.rows
    # Only the restored row got a secret, and each row was asserted against the
    # fingerprint the backup pinned for it.
    assert [entry["route_name"] for entry in report["new_callback_secrets"]] == ["support"]
    assert scan.seen == [(_RefusingScan.refused, "fp-1"), ("svc", "fp-1")]


async def test_import_skips_a_row_whose_key_condition_no_background_turn_can_evaluate(wired, monkeypatch):
    scan = _ConditionRefusingScan()
    _use_scan(monkeypatch, scan)
    row = _api_route("billing", "old", execution_key=_ConditionRefusingScan.refused).model_dump(mode="json")
    del row["callback_secret"]

    report = await backup.import_conversation_routes({"routes": [row]})

    assert report["skipped"] == 1
    assert report["created"] == 0
    assert any("identity claim" in error for error in report["errors"])
    assert wired.rows == {}


async def test_import_propagates_a_scan_failure_that_is_not_the_rows_own_defect(wired, monkeypatch):
    # A store read that blows up is the SECTION's failure, never a per-row rejection
    # that quietly drops the route.
    class _BrokenScan:
        async def assert_usable(self, execution_key, *, bound_fingerprint):
            raise RuntimeError("the policy store is unreachable")

    _use_scan(monkeypatch, _BrokenScan())
    row = _api_route("support", "old").model_dump(mode="json")
    del row["callback_secret"]

    with pytest.raises(RuntimeError, match="policy store is unreachable"):
        await backup.import_conversation_routes({"routes": [row]})
    assert wired.rows == {}


async def test_import_skips_a_row_claiming_a_live_channel_identity(wired):
    # A backup taken before a rename restores the OLD name alongside the live row; two
    # rows on one identity make every inbound message to it unresolvable.
    await wired.put_route(_channel_route("sms-support"))
    renamed = _channel_route("sms-line").model_dump(mode="json")

    report = await backup.import_conversation_routes({"routes": [renamed]})

    assert report["created"] == 0
    assert report["skipped"] == 1
    assert "already routed by 'sms-support'" in report["errors"][0]
    assert set(wired.rows) == {"sms-support"}


async def test_import_skips_the_second_of_two_colliding_rows_in_one_payload(wired):
    first = _channel_route("line-a").model_dump(mode="json")
    # The same identity in a different spelling: the claim is matched canonically.
    second = _channel_route("line-b").model_dump(mode="json") | {"our_identity": "  +15550001111 "}

    report = await backup.import_conversation_routes({"routes": [first, second]})

    assert report["created"] == 1
    assert report["skipped"] == 1
    assert "already routed by 'line-a'" in report["errors"][0]
    assert set(wired.rows) == {"line-a"}


async def test_import_replaces_a_row_on_the_identity_it_already_holds(wired):
    # A row keeps its own claim across a restore, and its old pair is freed for another row.
    await wired.put_route(_channel_route("line-a"))
    moved = _channel_route("line-a").model_dump(mode="json") | {"our_identity": "+15559999999"}
    taking_over = _channel_route("line-b").model_dump(mode="json")

    report = await backup.import_conversation_routes({"routes": [moved, taking_over]})

    assert report["errors"] == []
    assert report["skipped"] == 0
    assert wired.rows["line-a"].our_identity == "+15559999999"
    assert wired.rows["line-b"].our_identity == "+15550001111"
