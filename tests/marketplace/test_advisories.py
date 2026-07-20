"""The advisory cache: refresh matching, on-demand freshness, and the poll.

The store and the registry client are faked at the module seams; the module-level
cache and poll-task globals are reset around each test. The poll's loop-ownership
contract (start on the serving loop, restart marshalled from a foreign thread) is
driven against a real background-thread loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from tai_skeleton.marketplace import advisories
from tai_skeleton.marketplace.advisories import AdvisoryState, current, refresh
from tai_skeleton.marketplace.errors import (
    ListingNotFoundError,
    RegistryResponseError,
    RegistryUnreachableError,
)
from tai_skeleton.marketplace.settings import marketplace_settings
from tai_skeleton.marketplace.store import InstallRecord


def _record(ref: str, version: str) -> InstallRecord:
    return InstallRecord(
        ref=ref, version=version, source="pypi", repository_url=None, tag=None, spec={}, installed_at=datetime.now(UTC)
    )


class _FakeStore:
    def __init__(self, rows: list[InstallRecord]) -> None:
        self._rows = rows
        self.calls = 0

    async def list_installed(self) -> list[InstallRecord]:
        self.calls += 1
        return self._rows


class _FakeRegistry:
    def __init__(self, per_listing: dict[str, object]) -> None:
        self._per_listing = per_listing
        self.calls: list[str] = []

    async def advisories(self, *, listing: str | None = None, since: str | None = None):
        assert listing is not None
        self.calls.append(listing)
        result = self._per_listing[listing]
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture(autouse=True)
def _reset_module():
    advisories._state = None
    advisories._poll_task = None
    advisories._serving_loop = None
    yield
    advisories._state = None
    advisories._poll_task = None
    advisories._serving_loop = None


def _wire(monkeypatch, store: _FakeStore, registry: _FakeRegistry | None = None) -> None:
    monkeypatch.setattr(advisories, "MarketplaceInstallStore", lambda: store)
    if registry is not None:
        monkeypatch.setattr(advisories, "RegistryClient", lambda: registry)


# -- refresh -----------------------------------------------------------------


async def test_refresh_no_installs_makes_no_registry_call(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _FakeStore([])
    registry = _FakeRegistry({})
    _wire(monkeypatch, store, registry)
    state = await refresh()
    assert state.advisories == []
    assert registry.calls == []  # nothing to ask about


async def test_refresh_keeps_only_non_withdrawn_affecting_advisories(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _FakeStore([_record("tai42/toolbox", "1.0.0")])
    rows = [
        {"affected_versions": "<2.0", "withdrawn_at": None, "severity": "high", "summary": "affects 1.0"},
        {"affected_versions": ">=2.0", "withdrawn_at": None, "severity": "high", "summary": "not 1.0"},
        {"affected_versions": "<2.0", "withdrawn_at": "2026-01-01", "severity": "high", "summary": "withdrawn"},
    ]
    registry = _FakeRegistry({"tai42/toolbox": rows})
    _wire(monkeypatch, store, registry)
    state = await refresh()
    assert [a["summary"] for a in state.advisories] == ["affects 1.0"]


async def test_refresh_matches_a_prerelease_installed_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # An installed prerelease must match a range via contains(prereleases=True).
    store = _FakeStore([_record("tai42/beta", "0.2.0rc1")])
    rows = [{"affected_versions": "<0.3", "withdrawn_at": None, "severity": "critical", "summary": "pre affected"}]
    registry = _FakeRegistry({"tai42/beta": rows})
    _wire(monkeypatch, store, registry)
    state = await refresh()
    assert [a["summary"] for a in state.advisories] == ["pre affected"]


async def test_refresh_skips_a_vanished_listing_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = _FakeStore([_record("tai42/gone", "1.0.0"), _record("tai42/live", "1.0.0")])
    live_rows = [{"affected_versions": "<2.0", "withdrawn_at": None, "severity": "low", "summary": "live adv"}]
    registry = _FakeRegistry(
        {"tai42/gone": ListingNotFoundError("marketplace listing not found: tai42/gone"), "tai42/live": live_rows}
    )
    _wire(monkeypatch, store, registry)
    with caplog.at_level(logging.WARNING, logger="tai_skeleton.marketplace.advisories"):
        state = await refresh()
    # The vanished ref is skipped (WARNING names it); the other ref's advisory lands.
    assert [a["summary"] for a in state.advisories] == ["live adv"]
    assert any("tai42/gone" in r.getMessage() for r in caplog.records)


async def test_refresh_transport_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _FakeStore([_record("tai42/toolbox", "1.0.0")])
    registry = _FakeRegistry({"tai42/toolbox": RegistryUnreachableError("down")})
    _wire(monkeypatch, store, registry)
    with pytest.raises(RegistryUnreachableError):
        await refresh()


async def test_refresh_skips_a_single_malformed_advisory_row_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # One advisory row with a non-PEP440 affected_versions must not abort the whole
    # refresh — it is skipped with a warning; the well-formed rows still land.
    store = _FakeStore([_record("tai42/toolbox", "1.0.0")])
    rows = [
        {"affected_versions": "not a specifier", "withdrawn_at": None, "severity": "high", "summary": "malformed"},
        {"affected_versions": "<2.0", "withdrawn_at": None, "severity": "high", "summary": "good"},
    ]
    registry = _FakeRegistry({"tai42/toolbox": rows})
    _wire(monkeypatch, store, registry)
    with caplog.at_level(logging.WARNING, logger="tai_skeleton.marketplace.advisories"):
        state = await refresh()
    assert [a["summary"] for a in state.advisories] == ["good"]
    assert any("malformed affected_versions" in r.getMessage() for r in caplog.records)


# -- _affects ----------------------------------------------------------------


def test_affects_list_typed_affected_versions_raises_response_error() -> None:
    # A JSON array affected_versions constructs a SpecifierSet without raising and
    # only fails INSIDE .contains() with an AttributeError — it must surface as a
    # typed RegistryResponseError (garbled registry data), never an untyped 500.
    with pytest.raises(RegistryResponseError, match="invalid affected_versions"):
        advisories._affects(["==1.0"], "1.0.0")


def test_affects_dict_typed_affected_versions_raises_response_error() -> None:
    # A JSON object affected_versions likewise builds a SpecifierSet then fails in
    # .contains() with an AttributeError — a typed RegistryResponseError, not a 500.
    with pytest.raises(RegistryResponseError, match="invalid affected_versions"):
        advisories._affects({"x": 1}, "1.0.0")


# -- current -----------------------------------------------------------------


async def test_current_serves_young_cache_without_a_registry_call(monkeypatch: pytest.MonkeyPatch) -> None:
    advisories._state = AdvisoryState(advisories=[{"summary": "cached"}], fetched_at=datetime.now(UTC))
    store = _FakeStore([])  # would be hit if refresh ran
    _wire(monkeypatch, store)
    state = await current(3600)
    assert state.advisories == [{"summary": "cached"}]
    assert store.calls == 0  # served from cache, no refresh


async def test_current_refreshes_a_stale_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    advisories._state = AdvisoryState(
        advisories=[{"summary": "old"}], fetched_at=datetime.now(UTC) - timedelta(seconds=10_000)
    )
    store = _FakeStore([])
    registry = _FakeRegistry({})
    _wire(monkeypatch, store, registry)
    state = await current(3600)
    assert state.advisories == []  # refreshed (no installs -> empty)
    assert store.calls == 1


# -- start_poll --------------------------------------------------------------


async def test_start_poll_disabled_starts_nothing_and_logs_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("MARKETPLACE_ADVISORIES_POLL", "false")
    marketplace_settings.cache_clear()
    try:
        with caplog.at_level(logging.INFO, logger="tai_skeleton.marketplace.advisories"):
            advisories.start_poll()
        assert advisories._poll_task is None
        assert caplog.records == []
    finally:
        marketplace_settings.cache_clear()


async def test_start_poll_enabled_logs_url_and_interval_and_remembers_loop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("MARKETPLACE_ADVISORIES_POLL", "true")
    monkeypatch.setenv("MARKETPLACE_URL", "https://reg.example")
    monkeypatch.setenv("MARKETPLACE_ADVISORIES_INTERVAL_S", "1800")
    marketplace_settings.cache_clear()
    try:
        with caplog.at_level(logging.INFO, logger="tai_skeleton.marketplace.advisories"):
            advisories.start_poll()
        assert advisories._serving_loop is asyncio.get_running_loop()
        assert advisories._poll_task is not None
        line = " ".join(r.getMessage() for r in caplog.records)
        assert "https://reg.example" in line
        assert "1800" in line
    finally:
        await advisories.stop_poll()
        marketplace_settings.cache_clear()


# -- restart_poll_from_reload ------------------------------------------------


def test_restart_from_reload_is_noop_when_never_served() -> None:
    # No serving loop was ever remembered: there is no poll to re-pace, so the
    # reload hook is a quiet no-op (never a hang, never a raise that would fail an
    # unrelated app's reload — the hook is registered process-wide).
    advisories._serving_loop = None
    advisories.restart_poll_from_reload()
    assert advisories._poll_task is None


def test_restart_from_reload_is_noop_when_serving_loop_not_running() -> None:
    # A serving loop remembered from a prior (now torn-down) context is not
    # running; marshalling onto it would block forever, so the restart is skipped.
    stale = asyncio.new_event_loop()
    try:
        advisories._serving_loop = stale
        advisories.restart_poll_from_reload()  # must return promptly, not hang
        assert advisories._poll_task is None
    finally:
        stale.close()


def test_restart_from_foreign_thread_marshals_onto_serving_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKETPLACE_ADVISORIES_POLL", "true")
    marketplace_settings.cache_clear()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def _start() -> None:
        advisories.start_poll()

    try:
        asyncio.run_coroutine_threadsafe(_start(), loop).result(timeout=2)
        old_task = advisories._poll_task
        assert old_task is not None

        # Called from THIS (foreign) thread — it marshals the restart onto the
        # serving loop (fire-and-forget), so wait for the replacement to land.
        advisories.restart_poll_from_reload()
        deadline = time.monotonic() + 2
        while advisories._poll_task is old_task and time.monotonic() < deadline:
            time.sleep(0.01)
        new_task = advisories._poll_task
        assert new_task is not None
        assert new_task is not old_task
        assert new_task.get_loop() is loop  # owned by the serving loop
        assert old_task.cancelled() or old_task.done()
    finally:
        asyncio.run_coroutine_threadsafe(advisories.stop_poll(), loop).result(timeout=2)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()
        marketplace_settings.cache_clear()


# -- _poll_loop + stop_poll --------------------------------------------------


async def test_poll_loop_survives_one_failing_refresh(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = {"n": 0}

    async def _fake_sleep(_seconds: float) -> None:
        # One-shot: allow one refresh, then stop the loop.
        calls["n"] += 1
        if calls["n"] >= 2:
            raise asyncio.CancelledError

    async def _boom() -> AdvisoryState:
        raise RegistryUnreachableError("poll down")

    monkeypatch.setattr(advisories.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(advisories, "refresh", _boom)
    with (
        caplog.at_level(logging.WARNING, logger="tai_skeleton.marketplace.advisories"),
        pytest.raises(asyncio.CancelledError),
    ):
        await advisories._poll_loop()
    # The failing poll logged a WARNING and the loop continued to the next sleep.
    assert any("poll failed" in r.getMessage() for r in caplog.records)


async def test_poll_loop_warns_on_high_or_critical_advisory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A fresh snapshot carrying a high/critical advisory logs each at WARNING naming
    # the listing, severity, and summary; lower severities stay silent.
    calls = {"n": 0}

    async def _fake_sleep(_seconds: float) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:  # one refresh, then stop the loop
            raise asyncio.CancelledError

    async def _fresh() -> AdvisoryState:
        return AdvisoryState(
            advisories=[
                {"listing": "tai42/toolbox", "severity": "critical", "summary": "RCE"},
                {"listing": "tai42/other", "severity": "low", "summary": "cosmetic"},
            ],
            fetched_at=datetime.now(UTC),
        )

    monkeypatch.setattr(advisories.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(advisories, "refresh", _fresh)
    with (
        caplog.at_level(logging.WARNING, logger="tai_skeleton.marketplace.advisories"),
        pytest.raises(asyncio.CancelledError),
    ):
        await advisories._poll_loop()
    warnings = [r.getMessage() for r in caplog.records]
    assert any("tai42/toolbox" in m and "critical" in m and "RCE" in m for m in warnings)
    # The low-severity advisory is not warned about.
    assert not any("cosmetic" in m for m in warnings)


async def test_stop_poll_cancels_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKETPLACE_ADVISORIES_POLL", "true")
    marketplace_settings.cache_clear()
    try:
        advisories.start_poll()
        task = advisories._poll_task
        assert task is not None
        await advisories.stop_poll()
        assert task.cancelled()
        assert advisories._poll_task is None
    finally:
        marketplace_settings.cache_clear()
