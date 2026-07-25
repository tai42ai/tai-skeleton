"""``tai plugins`` commands against a fake server.

Each command's request shaping (path, query params, JSON body), the envelope
rendering, and the typed-error surfacing are exercised through the shared remote
harness (an httpx ``MockTransport``), so no request leaves the process.
"""

from __future__ import annotations

import json

import httpx

from tests.cli.remote_harness import data_response, error_response, run_cli


def _capture():
    """A handler that records the last request and answers a canned data envelope;
    returns ``(handler, captured)`` where ``captured`` is filled per call."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = request.url.query.decode()
        captured["body"] = json.loads(request.content) if request.content else None
        return data_response(captured.get("_payload", {"items": []}))

    return handler, captured


# -- search ------------------------------------------------------------------


def test_search_positional_becomes_q_with_facets(monkeypatch) -> None:
    handler, captured = _capture()
    result = run_cli(monkeypatch, handler, ["plugins", "search", "uuid", "--kind", "tool"])
    assert result.exit_code == 0, result.output
    assert captured["path"] == "/api/marketplace/search"
    params = httpx.QueryParams(captured["query"])
    assert params.get("q") == "uuid"
    assert params.get("kind") == "tool"


def test_search_no_positional_sends_no_q(monkeypatch) -> None:
    handler, captured = _capture()
    result = run_cli(monkeypatch, handler, ["plugins", "search", "--tier", "official"])
    assert result.exit_code == 0, result.output
    params = httpx.QueryParams(captured["query"])
    assert "q" not in params
    assert params.get("tier") == "official"


def test_search_repeated_tag_becomes_repeated_tags_param(monkeypatch) -> None:
    handler, captured = _capture()
    result = run_cli(monkeypatch, handler, ["plugins", "search", "--tag", "a", "--tag", "b"])
    assert result.exit_code == 0, result.output
    # Never comma-joined: each --tag is its own repeated tags query param.
    assert captured["query"].count("tags=") == 2
    assert httpx.QueryParams(captured["query"]).get_list("tags") == ["a", "b"]


# -- info --------------------------------------------------------------------


def test_info_splits_ref_into_the_path(monkeypatch) -> None:
    handler, captured = _capture()
    captured["_payload"] = {"ref": "tai42/toolbox"}
    result = run_cli(monkeypatch, handler, ["plugins", "info", "tai42/toolbox"])
    assert result.exit_code == 0, result.output
    assert captured["path"] == "/api/marketplace/plugins/tai42/toolbox"


def test_info_malformed_ref_is_a_usage_error(monkeypatch) -> None:
    handler, _ = _capture()
    result = run_cli(monkeypatch, handler, ["plugins", "info", "noslash"])
    assert result.exit_code != 0
    assert "namespace/name" in result.output


# -- categories / installed --------------------------------------------------


def test_categories_issues_the_get(monkeypatch) -> None:
    handler, captured = _capture()
    captured["_payload"] = ["dev", "data"]
    result = run_cli(monkeypatch, handler, ["plugins", "categories"])
    assert result.exit_code == 0, result.output
    assert captured["path"] == "/api/marketplace/categories"


def test_installed_json_passthrough(monkeypatch) -> None:
    handler, captured = _capture()
    rows = [{"ref": "tai42/toolbox", "version": "1.0.0", "latest": "2.0.0", "update_available": True}]
    captured["_payload"] = rows
    result = run_cli(monkeypatch, handler, ["plugins", "installed"], json_output=True)
    assert result.exit_code == 0, result.output
    assert captured["path"] == "/api/marketplace/installed"
    assert json.loads(result.output) == rows


# -- install / uninstall / update bodies -------------------------------------


def test_install_posts_ref_and_version(monkeypatch) -> None:
    handler, captured = _capture()
    captured["_payload"] = {"ref": "tai42/toolbox"}
    result = run_cli(monkeypatch, handler, ["plugins", "install", "tai42/toolbox", "--version", "1.2.3"])
    assert result.exit_code == 0, result.output
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/marketplace/install"
    assert captured["body"] == {"ref": "tai42/toolbox", "version": "1.2.3"}


def test_uninstall_posts_ref(monkeypatch) -> None:
    handler, captured = _capture()
    captured["_payload"] = {"ref": "tai42/toolbox", "uninstalled": True}
    result = run_cli(monkeypatch, handler, ["plugins", "uninstall", "tai42/toolbox"])
    assert result.exit_code == 0, result.output
    assert captured["path"] == "/api/marketplace/uninstall"
    assert captured["body"] == {"ref": "tai42/toolbox"}


def test_update_posts_ref_without_version_when_omitted(monkeypatch) -> None:
    handler, captured = _capture()
    captured["_payload"] = {"ref": "tai42/toolbox"}
    result = run_cli(monkeypatch, handler, ["plugins", "update", "tai42/toolbox"])
    assert result.exit_code == 0, result.output
    assert captured["path"] == "/api/marketplace/update"
    assert captured["body"] == {"ref": "tai42/toolbox"}


# -- error surfacing ---------------------------------------------------------


def test_conflict_error_envelope_surfaces_nonzero(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response("tai42/toolbox is already installed", 409)

    result = run_cli(monkeypatch, handler, ["plugins", "install", "tai42/toolbox"])
    assert result.exit_code != 0
    assert "already installed" in result.output


def test_advisories_issues_the_get(monkeypatch) -> None:
    handler, captured = _capture()
    captured["_payload"] = {"advisories": [], "fetched_at": "2026-06-01T00:00:00+00:00"}
    result = run_cli(monkeypatch, handler, ["plugins", "advisories"])
    assert result.exit_code == 0, result.output
    assert captured["path"] == "/api/marketplace/advisories"
