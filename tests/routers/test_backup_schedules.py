"""The opaque ``schedules`` backup section, driven through the backup router.

The section carries whatever the scheduling backend's ``backend_export_schedules``
/ ``backend_import_schedules`` tools return, without parsing schedule internals.
The ``tai42_app.tools`` facet is faked: an unbound backend has neither tool, so
``run_tool`` raises the binding's real unknown-tool ``RuntimeError`` and the
router records it per-section; a bound backend round-trips the document opaquely.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from starlette.requests import Request
from tai42_contract.app import tai42_app

from tai42_skeleton.backup.registry import BackupRegistry
from tai42_skeleton.backup.sections import register_core_sections
from tai42_skeleton.routers.backup import export_backup, import_backup


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


class _FakeTools:
    """A tool registry where ``registered`` names run and return their queued
    result; any other name raises the binding's unknown-tool ``RuntimeError``."""

    def __init__(self, registered: dict[str, object] | None = None) -> None:
        self._registered = registered or {}
        self.run_calls: list[tuple[str, dict]] = []

    async def run_tool(self, key, arguments):
        if key not in self._registered:
            raise RuntimeError(f"No such tool: {key}.")
        self.run_calls.append((key, arguments))
        return self._registered[key]


def _install(monkeypatch, tools: _FakeTools) -> None:
    registry = BackupRegistry()
    register_core_sections(registry)
    monkeypatch.setattr(tai42_app, "_impl", SimpleNamespace(backup=registry, tools=tools))


# -- no backend: absence is a per-section error, not a crash -----------------


async def test_export_without_backend_records_section_error(monkeypatch):
    _install(monkeypatch, _FakeTools())  # no scheduling tools registered
    resp = await export_backup(_post_req({"sections": ["schedules"]}))
    assert resp.status_code == 200  # still a download, never a 500
    doc = _json(resp)
    assert "schedules" not in doc["sections"]  # nothing exported
    assert "backend_export_schedules" in doc["errors"]["schedules"]


async def test_import_without_backend_reports_error_and_not_ok(monkeypatch):
    _install(monkeypatch, _FakeTools())
    document = {"version": 1, "sections": {"schedules": [{"name": "nightly"}]}}
    data = _json(await import_backup(_post_req({"document": document, "sections": ["schedules"]})))["data"]
    assert data["ok"] is False
    assert "backend_import_schedules" in data["sections"]["schedules"]["errors"][0]


# -- bound backend: the document round-trips opaquely ------------------------


async def test_schedules_round_trip_through_bound_backend(monkeypatch):
    backend_doc = [
        {"name": "nightly", "cron": "0 0 * * *", "tool": "cleanup", "arguments": {"scope": "all"}},
        {"name": "hourly", "cron": "0 * * * *", "tool": "sync", "arguments": {}},
    ]
    import_report = {"created": 2, "updated": 0, "skipped": 0, "errors": []}
    tools = _FakeTools(
        {
            "backend_export_schedules": backend_doc,
            "backend_import_schedules": import_report,
        }
    )
    _install(monkeypatch, tools)

    # Export carries the backend's list verbatim.
    doc = _json(await export_backup(_post_req({"sections": ["schedules"]})))
    assert doc["sections"]["schedules"] == backend_doc
    assert doc["errors"] == {}

    # Import hands the whole document back to the backend under ``schedules``.
    data = _json(await import_backup(_post_req({"document": doc, "sections": ["schedules"]})))["data"]
    assert data["ok"] is True
    assert data["sections"]["schedules"] == import_report
    assert ("backend_import_schedules", {"schedules": backend_doc}) in tools.run_calls
