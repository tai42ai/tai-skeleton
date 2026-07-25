"""Further remote command-group coverage against the fake ``/api/*`` server.

Complements :mod:`tests.cli.test_remote_commands`: the same fake-transport harness
drives the commands each group's read/write/delete verbs still leave unexercised,
asserting the request each shapes (method, path, params, body) and the result it
renders (table rows, the ``{data}`` envelope under ``--json``, or a surfaced
typed error).
"""

from __future__ import annotations

import json

import httpx
import pytest

from tests.cli.remote_harness import data_response, run_cli

# -- keys --------------------------------------------------------------------


def test_keys_list_renders_identity_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/auth/tokens-payload"
        return data_response([{"user_id": "alice", "description": "ci", "scopes": ["read"]}])

    result = run_cli(monkeypatch, handler, ["keys", "list"])
    assert result.exit_code == 0, result.output
    assert "alice" in result.output
    assert "read" in result.output


def test_keys_create_includes_all_optional_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["user_id"] == "bob"
        assert body["condition"] == '.method == "GET"'
        assert body["condition_id"] == "cond1"
        assert body["condition_kwargs"] == {"role": "admin"}
        assert body["policy_data"] == {"team": "ops"}
        return data_response("sk-secret")

    result = run_cli(
        monkeypatch,
        handler,
        [
            "keys",
            "create",
            "--user",
            "bob",
            "--description",
            "ci",
            "--condition",
            '.method == "GET"',
            "--condition-id",
            "cond1",
            "--condition-kwargs",
            '{"role":"admin"}',
            "--policy-data",
            '{"team":"ops"}',
        ],
    )
    assert result.exit_code == 0, result.output
    assert "sk-secret" in result.output


def test_keys_create_rejects_non_object_condition_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response("sk")

    result = run_cli(
        monkeypatch,
        handler,
        ["keys", "create", "--user", "bob", "--description", "ci", "--condition-kwargs", "[1,2]"],
    )
    assert result.exit_code != 0
    assert "JSON object" in result.output


def test_keys_edit_sends_only_provided_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert request.url.path == "/api/auth/api-keys/alice"
        assert json.loads(request.content) == {"description": "new", "scopes": ["read", "write"]}
        return data_response({"user_id": "alice", "updated": True})

    result = run_cli(
        monkeypatch,
        handler,
        ["keys", "edit", "alice", "--description", "new", "--scope", "read", "--scope", "write"],
    )
    assert result.exit_code == 0, result.output


def test_keys_edit_writes_every_optional_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/auth/api-keys/alice"
        assert json.loads(request.content) == {
            "condition": ".ok",
            "condition_id": "cond1",
            "condition_kwargs": {"role": "admin"},
            "policy_data": {"team": "ops"},
        }
        return data_response({"updated": True})

    result = run_cli(
        monkeypatch,
        handler,
        [
            "keys",
            "edit",
            "alice",
            "--condition",
            ".ok",
            "--condition-id",
            "cond1",
            "--condition-kwargs",
            '{"role":"admin"}',
            "--policy-data",
            '{"team":"ops"}',
        ],
    )
    assert result.exit_code == 0, result.output


def test_keys_validate_condition_by_stored_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"condition_id": "cond1"}
        return data_response({"valid": True})

    result = run_cli(monkeypatch, handler, ["keys", "validate-condition", "--condition-id", "cond1"])
    assert result.exit_code == 0, result.output


def test_keys_delete_revokes(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/auth/api-keys/alice"
        return data_response({"deleted": True})

    result = run_cli(monkeypatch, handler, ["keys", "delete", "alice"])
    assert result.exit_code == 0, result.output


def test_keys_validate_condition_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/auth/validate-condition"
        body = json.loads(request.content)
        assert body["condition"] == ".ok"
        assert body["condition_kwargs"] == {"n": 1}
        assert body["sample_context"] == {"method": "GET"}
        return data_response({"valid": True})

    result = run_cli(
        monkeypatch,
        handler,
        [
            "keys",
            "validate-condition",
            "--condition",
            ".ok",
            "--condition-kwargs",
            '{"n":1}',
            "--sample-context",
            '{"method":"GET"}',
        ],
    )
    assert result.exit_code == 0, result.output
    assert "valid" in result.output


def test_keys_policy_versions_renders_history(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/auth/api-keys/alice/policy/versions"
        return data_response([{"version": 2, "is_current": True, "created_at": "t"}])

    result = run_cli(monkeypatch, handler, ["keys", "policy-versions", "alice"])
    assert result.exit_code == 0, result.output
    assert "2" in result.output


def test_keys_policy_rollback_posts_version(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/auth/api-keys/alice/policy/rollback"
        assert json.loads(request.content) == {"version": 3}
        return data_response({"version": 3})

    result = run_cli(monkeypatch, handler, ["keys", "policy-rollback", "alice", "3"])
    assert result.exit_code == 0, result.output


# -- presets -----------------------------------------------------------------


def test_presets_list_sends_no_tier_param(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/presets"
        # The list is versioned-only (store-backed); no tier param is sent.
        assert "tier" not in request.url.params
        return data_response([{"name": "greet", "base_tool": "echo", "active_version": 1}])

    result = run_cli(monkeypatch, handler, ["presets", "list"])
    assert result.exit_code == 0, result.output
    assert "greet" in result.output


def test_presets_get_renders_record(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/presets/greet"
        return data_response({"name": "greet", "base_tool": "echo"})

    result = run_cli(monkeypatch, handler, ["presets", "get", "greet"])
    assert result.exit_code == 0, result.output
    assert "echo" in result.output


def test_presets_create_parses_extension_combos(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["extensions"] == [["chain", "batch"], ["chain"]]
        return data_response({"name": "greet", "persisted": True})

    result = run_cli(
        monkeypatch,
        handler,
        ["presets", "create", "greet", "--base-tool", "echo", "--extensions", '[["chain","batch"],["chain"]]'],
    )
    assert result.exit_code == 0, result.output


def test_presets_create_carries_element_config_losslessly(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = {"type": "object"}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["extensions"] == [
            ["chain", {"name": "output_schema", "config": {"schema": schema}}],
        ]
        return data_response({"name": "greet", "persisted": True})

    result = run_cli(
        monkeypatch,
        handler,
        [
            "presets",
            "create",
            "greet",
            "--base-tool",
            "echo",
            "--extensions",
            '[["chain",{"name":"output_schema","config":{"schema":{"type":"object"}}}]]',
        ],
    )
    assert result.exit_code == 0, result.output


def test_presets_create_rejects_malformed_extensions_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(
        monkeypatch, handler, ["presets", "create", "greet", "--base-tool", "echo", "--extensions", "not json"]
    )
    assert result.exit_code != 0
    assert "valid JSON" in result.output


def test_presets_create_rejects_flat_combo_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    # A flat array of names is not a list OF COMBOS — each combo must itself be an array.
    result = run_cli(
        monkeypatch, handler, ["presets", "create", "greet", "--base-tool", "echo", "--extensions", '["chain"]']
    )
    assert result.exit_code != 0
    assert "non-empty JSON array" in result.output


def test_presets_create_rejects_empty_inner_combo(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty inner combo ``[[]]`` names no extension -> parse_extension_combos
    rejects it via the ``not combo`` sub-condition (distinct from the not-a-list one)."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(
        monkeypatch, handler, ["presets", "create", "greet", "--base-tool", "echo", "--extensions", "[[]]"]
    )
    assert result.exit_code != 0
    assert "non-empty JSON array" in result.output


def test_presets_create_rejects_non_list_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed JSON value that is not a top-level array (an object) hits
    parse_extension_combos's own not-a-list branch, distinct from the per-combo one."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["presets", "create", "greet", "--base-tool", "echo", "--extensions", "{}"])
    assert result.exit_code != 0
    assert "array of extension combos" in result.output


def test_presets_create_rejects_config_free_object_element(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(
        monkeypatch,
        handler,
        ["presets", "create", "greet", "--base-tool", "echo", "--extensions", '[[{"name":"output_schema"}]]'],
    )
    assert result.exit_code != 0
    assert "'config' mapping" in result.output


def test_presets_create_rejects_empty_string_element(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare "" element is not a valid extension name (parse_extension_element str branch)."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(
        monkeypatch, handler, ["presets", "create", "greet", "--base-tool", "echo", "--extensions", '[[""]]']
    )
    assert result.exit_code != 0
    assert "an extension name must be a non-empty string" in result.output


def test_presets_create_rejects_non_string_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """An object element with a non-string 'name' (parse_extension_element dict/name branch)."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(
        monkeypatch,
        handler,
        ["presets", "create", "greet", "--base-tool", "echo", "--extensions", '[[{"name":5,"config":{}}]]'],
    )
    assert result.exit_code != 0
    assert "extension element must have" in result.output


def test_presets_create_rejects_object_with_extra_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """An object element with keys beyond name/config (parse_extension_element extra branch)."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(
        monkeypatch,
        handler,
        ["presets", "create", "greet", "--base-tool", "echo", "--extensions", '[[{"name":"chain","config":{},"x":1}]]'],
    )
    assert result.exit_code != 0
    assert "unexpected" in result.output


def test_presets_create_rejects_non_element(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-string, non-object element (parse_extension_element fall-through branch)."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(
        monkeypatch, handler, ["presets", "create", "greet", "--base-tool", "echo", "--extensions", "[[5]]"]
    )
    assert result.exit_code != 0
    assert "combo element must be an extension" in result.output


def test_presets_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/presets/greet"
        return data_response({"deleted": True})

    result = run_cli(monkeypatch, handler, ["presets", "delete", "greet"])
    assert result.exit_code == 0, result.output


def test_presets_versions_lists_history(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/presets/greet/versions"
        return data_response([{"version": 1, "created_at": "t1"}, {"version": 2, "created_at": "t2"}])

    result = run_cli(monkeypatch, handler, ["presets", "versions", "greet"])
    assert result.exit_code == 0, result.output
    assert "t2" in result.output


def test_presets_get_version(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/presets/greet/versions/3"
        return data_response({"version": 3, "fixed_kwargs": {"n": 2}})

    result = run_cli(monkeypatch, handler, ["presets", "get-version", "greet", "3"])
    assert result.exit_code == 0, result.output
    assert "3" in result.output


def test_presets_save_version_sends_kwargs_and_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/presets/greet/versions"
        body = json.loads(request.content)
        assert body["fixed_kwargs"] == {"n": 2}
        assert body["extensions"] == []
        return data_response({"version": 4})

    result = run_cli(
        monkeypatch, handler, ["presets", "save-version", "greet", "--kwargs", '{"n":2}', "--extensions", "[]"]
    )
    assert result.exit_code == 0, result.output


def test_presets_save_version_replaces_tag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/presets/greet/versions"
        assert json.loads(request.content) == {"tags": ["a", "b"]}
        return data_response({"version": 5})

    result = run_cli(monkeypatch, handler, ["presets", "save-version", "greet", "--tag", "a", "--tag", "b"])
    assert result.exit_code == 0, result.output


def test_presets_save_version_rejects_tag_and_clear_together(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["presets", "save-version", "greet", "--tag", "x", "--clear-tags"])
    assert result.exit_code != 0
    assert "not both" in result.output


def test_presets_rollback_posts_version(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/presets/greet/rollback"
        assert json.loads(request.content) == {"version": 2}
        return data_response({"active_version": 2})

    result = run_cli(monkeypatch, handler, ["presets", "rollback", "greet", "2"])
    assert result.exit_code == 0, result.output


# -- scopes ------------------------------------------------------------------


def test_scopes_list_renders_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/auth/scopes"
        return data_response({"/api/tools": "read"})

    result = run_cli(monkeypatch, handler, ["scopes", "list"])
    assert result.exit_code == 0, result.output
    assert "/api/tools" in result.output


def test_scopes_add_sends_optional_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/auth/scopes"
        assert json.loads(request.content) == {"scope_id": "read", "url": "/api/tools", "pattern": "^/api/t"}
        return data_response({"scope_id": "read"})

    result = run_cli(monkeypatch, handler, ["scopes", "add", "read", "/api/tools", "--pattern", "^/api/t"])
    assert result.exit_code == 0, result.output


def test_scopes_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/auth/scopes/read"
        return data_response({"deleted": True})

    result = run_cli(monkeypatch, handler, ["scopes", "delete", "read"])
    assert result.exit_code == 0, result.output


# -- schedules ---------------------------------------------------------------


def test_schedules_list_happy_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/schedules"
        return data_response({"schedules": [{"name": "report"}]})

    result = run_cli(monkeypatch, handler, ["schedules", "list"])
    assert result.exit_code == 0, result.output
    assert "report" in result.output


def test_schedules_server_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/schedules/server-datetime"
        return data_response({"now": "2026-07-08T00:00:00Z"})

    result = run_cli(monkeypatch, handler, ["schedules", "server-datetime"])
    assert result.exit_code == 0, result.output
    assert "2026-07-08" in result.output


def test_schedules_add_merges_tool_and_schedule_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/schedules"
        body = json.loads(request.content)
        assert body["tool_name"] == "report"
        assert body["tool_kwargs"] == {"n": 5}
        assert body["schedule_kwargs"] == {"cron": "0 9 * * *"}
        return data_response({"name": "report"})

    result = run_cli(
        monkeypatch,
        handler,
        ["schedules", "add", "report", "--tool-kw", "n=5", "--schedule-kw", "cron=0 9 * * *"],
    )
    assert result.exit_code == 0, result.output


def test_schedules_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/schedules/report"
        return data_response({"deleted": True})

    result = run_cli(monkeypatch, handler, ["schedules", "delete", "report"])
    assert result.exit_code == 0, result.output


# -- templates ---------------------------------------------------------------


def test_templates_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/templates"
        return data_response({"templates": ["prompts/greeting.md"]})

    result = run_cli(monkeypatch, handler, ["templates", "list"])
    assert result.exit_code == 0, result.output
    assert "greeting" in result.output


def test_templates_get_posts_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/template"
        assert json.loads(request.content) == {"template_id": "prompts/greeting.md"}
        return data_response({"content": "Hi {{ name }}"})

    result = run_cli(monkeypatch, handler, ["templates", "get", "prompts/greeting.md"])
    assert result.exit_code == 0, result.output


def test_templates_upload_sends_file_content(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    local = tmp_path / "greeting.md"
    local.write_text("Hello {{ name }}", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/upload-template"
        assert json.loads(request.content) == {"path": "prompts/greeting.md", "content": "Hello {{ name }}"}
        return data_response({"written": True})

    result = run_cli(monkeypatch, handler, ["templates", "upload", "prompts/greeting.md", "--file", str(local)])
    assert result.exit_code == 0, result.output


def test_templates_delete_posts_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/delete-template"
        assert json.loads(request.content) == {"path": "prompts/greeting.md"}
        return data_response({"deleted": True})

    result = run_cli(monkeypatch, handler, ["templates", "delete", "prompts/greeting.md"])
    assert result.exit_code == 0, result.output


def test_templates_render_by_inline_content(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["content"] == "Hi {{ name }}"
        assert "template_id" not in body
        assert body["kwargs"] == {"name": "Ada"}
        return data_response({"rendered": "Hi Ada"})

    result = run_cli(monkeypatch, handler, ["templates", "render", "--content", "Hi {{ name }}", "--kw", 'name="Ada"'])
    assert result.exit_code == 0, result.output


def test_templates_clear_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/clear-templates-cache"
        return data_response({"cleared": True})

    result = run_cli(monkeypatch, handler, ["templates", "clear-cache"])
    assert result.exit_code == 0, result.output


# -- backup ------------------------------------------------------------------


def test_backup_sections_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/backup/sections"
        return data_response([{"name": "access_control", "secret": False}])

    result = run_cli(monkeypatch, handler, ["backup", "sections"])
    assert result.exit_code == 0, result.output
    assert "access_control" in result.output


def test_backup_import_posts_document(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    document = {"version": 1, "sections": {"access_control": {}}}
    backup_file = tmp_path / "backup.json"
    backup_file.write_text(json.dumps(document), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/backup/import"
        assert json.loads(request.content) == {"document": document, "sections": ["access_control"]}
        return data_response({"imported": ["access_control"]})

    result = run_cli(monkeypatch, handler, ["backup", "import", str(backup_file), "--section", "access_control"])
    assert result.exit_code == 0, result.output


def test_backup_import_rejects_malformed_json_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    backup_file = tmp_path / "backup.json"
    backup_file.write_text("{not json", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["backup", "import", str(backup_file), "--section", "access_control"])
    assert result.exit_code != 0
    assert "valid JSON" in result.output


# -- connectors --------------------------------------------------------------


def test_connectors_providers_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/connectors/providers"
        return data_response([{"id": "google", "display_name": "Google", "kind": "oauth", "category": "email"}])

    result = run_cli(monkeypatch, handler, ["connectors", "providers"])
    assert result.exit_code == 0, result.output
    assert "google" in result.output


def test_connectors_connections_uses_items_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/connectors/connections"
        return data_response(
            {"items": [{"connection_id": "c1", "provider_id": "google", "alias": "work", "auth_health_state": "ok"}]}
        )

    result = run_cli(monkeypatch, handler, ["connectors", "connections"])
    assert result.exit_code == 0, result.output
    assert "c1" in result.output


def test_connectors_get_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/connectors/connections/c1"
        return data_response({"connection_id": "c1", "alias": "work"})

    result = run_cli(monkeypatch, handler, ["connectors", "get", "c1"])
    assert result.exit_code == 0, result.output
    assert "work" in result.output


def test_connectors_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/connectors/connections/c1"
        return data_response({"deleted": True})

    result = run_cli(monkeypatch, handler, ["connectors", "disconnect", "c1"])
    assert result.exit_code == 0, result.output


def test_connectors_reconnect_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/connectors/connections/c1/reconnect"
        assert json.loads(request.content) == {"enabled_sub_services": ["gmail"], "return_url": "/connectors"}
        return data_response({"authorize_url": "https://x"})

    result = run_cli(monkeypatch, handler, ["connectors", "reconnect", "c1", "--sub-service", "gmail"])
    assert result.exit_code == 0, result.output


def test_connectors_sub_services_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/api/connectors/connections/c1/sub-services"
        assert json.loads(request.content) == {
            "enabled_sub_services": ["gmail", "calendar"],
            "return_url": "/connectors",
        }
        return data_response({"enabled_sub_services": ["gmail", "calendar"]})

    result = run_cli(
        monkeypatch,
        handler,
        ["connectors", "sub-services", "c1", "--sub-service", "gmail", "--sub-service", "calendar"],
    )
    assert result.exit_code == 0, result.output


# -- mcp ---------------------------------------------------------------------


def test_mcp_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/mcp-status"
        return data_response({"bindings": []})

    result = run_cli(monkeypatch, handler, ["mcp", "status"])
    assert result.exit_code == 0, result.output


def test_mcp_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/mcp-config/schema"
        return data_response({"type": "object"})

    result = run_cli(monkeypatch, handler, ["mcp", "schema"])
    assert result.exit_code == 0, result.output


def test_mcp_set_from_object_with_mcp_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcp": [{"title": "srv"}]}), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"mcp": [{"title": "srv"}]}
        return data_response({"reloaded": True})

    result = run_cli(monkeypatch, handler, ["mcp", "set", "--file", str(config)])
    assert result.exit_code == 0, result.output


def test_mcp_set_rejects_malformed_json(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text("{not json", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["mcp", "set", "--file", str(config)])
    assert result.exit_code != 0
    assert "valid JSON" in result.output


def test_mcp_set_rejects_wrong_shape(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"other": 1}), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["mcp", "set", "--file", str(config)])
    assert result.exit_code != 0
    assert "JSON list" in result.output


def test_mcp_reload_by_title(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/mcp-status/my-server/reload"
        return data_response({"reloaded": True})

    result = run_cli(monkeypatch, handler, ["mcp", "reload", "my-server"])
    assert result.exit_code == 0, result.output


# -- sub-mcp -----------------------------------------------------------------


def test_sub_mcp_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/sub-mcp"
        return data_response({"apps": [{"slug": "billing"}]})

    result = run_cli(monkeypatch, handler, ["sub-mcp", "list"])
    assert result.exit_code == 0, result.output
    assert "billing" in result.output


def test_sub_mcp_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/sub-mcp/billing"
        return data_response({"deleted": True})

    result = run_cli(monkeypatch, handler, ["sub-mcp", "delete", "billing"])
    assert result.exit_code == 0, result.output


# -- tools -------------------------------------------------------------------


def test_tools_tags_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tools/tags"
        return data_response([{"name": "add", "tags": ["math"]}])

    result = run_cli(monkeypatch, handler, ["tools", "tags"])
    assert result.exit_code == 0, result.output
    assert "add" in result.output
    assert "math" in result.output


def test_tools_schema_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tools/add/schema"
        return data_response({"input": {"a": "int"}})

    result = run_cli(monkeypatch, handler, ["tools", "schema", "add"])
    assert result.exit_code == 0, result.output


def test_tools_schemas_all(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tools-schema"
        return data_response({"add": {"input": {}}})

    result = run_cli(monkeypatch, handler, ["tools", "schemas"], json_output=True)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"add": {"input": {}}}


def test_tools_extensions_shows_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tools/add/extensions"
        return data_response({"applied": [["chain"]], "catalog": ["chain", "batch"]})

    result = run_cli(monkeypatch, handler, ["tools", "extensions", "add"])
    assert result.exit_code == 0, result.output


def test_tools_runs_submit_posts_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/tool-runs"
        assert json.loads(request.content) == {"tool_name": "slow", "arguments": {"n": 100}}
        return data_response({"run_id": "r1", "status": "running"})

    result = run_cli(monkeypatch, handler, ["tools", "runs", "submit", "slow", "--kw", "n=100"])
    assert result.exit_code == 0, result.output
    assert "r1" in result.output


def test_tools_runs_get(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tool-runs/r1"
        return data_response({"run_id": "r1", "status": "succeeded"})

    result = run_cli(monkeypatch, handler, ["tools", "runs", "get", "r1"])
    assert result.exit_code == 0, result.output
    assert "succeeded" in result.output


# -- config ------------------------------------------------------------------


def test_config_env_get(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/config/env"
        return data_response({"LOG_LEVEL": "info"})

    result = run_cli(monkeypatch, handler, ["config", "env", "get"])
    assert result.exit_code == 0, result.output
    assert "LOG_LEVEL" in result.output


def test_config_env_set_rejects_bare_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["config", "env", "set", "NOEQUALS"])
    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output


def test_config_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/config/mode"
        return data_response({"mode": "file"})

    result = run_cli(monkeypatch, handler, ["config", "mode"])
    assert result.exit_code == 0, result.output
    assert "file" in result.output


def test_config_settings_schema_uses_groups_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/config/settings-schema"
        return data_response({"groups": [{"name": "redis", "module": "tai42_skeleton.settings"}]})

    result = run_cli(monkeypatch, handler, ["config", "settings-schema"])
    assert result.exit_code == 0, result.output
    assert "redis" in result.output


def test_config_lint_missing_file_is_usage_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    missing = tmp_path / "nope.yml"
    result = run_cli(monkeypatch, lambda r: data_response({}), ["config", "lint", str(missing)])
    assert result.exit_code != 0
    assert "not found" in result.output


# -- traces ------------------------------------------------------------------


def test_traces_list_passes_all_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/observability/runs"
        params = request.url.params
        assert params.get("from") == "30d"
        assert params.get("to") == "now"
        assert params.get("status") == "error"
        assert params.get("sort") == "cost"
        assert params.get("dir") == "desc"
        assert params.get("page") == "2"
        assert params.get("pageSize") == "50"
        return data_response({"items": [{"traceId": "t9", "status": "error"}], "page": 2})

    result = run_cli(
        monkeypatch,
        handler,
        [
            "traces",
            "list",
            "--from",
            "30d",
            "--to",
            "now",
            "--status",
            "error",
            "--sort",
            "cost",
            "--dir",
            "desc",
            "--page",
            "2",
            "--page-size",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "t9" in result.output


def test_traces_get_full_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/observability/runs/trace_abc/trace"
        return data_response({"traceId": "trace_abc", "spans": []})

    result = run_cli(monkeypatch, handler, ["traces", "get", "trace_abc"])
    assert result.exit_code == 0, result.output
    assert "trace_abc" in result.output


def test_traces_get_export_downloads_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/observability/runs/trace_abc/trace/export"
        return httpx.Response(200, json={"traceId": "trace_abc"}, headers={"content-type": "application/json"})

    result = run_cli(monkeypatch, handler, ["traces", "get", "trace_abc", "--export"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"traceId": "trace_abc"}


# -- hooks -------------------------------------------------------------------


def test_hooks_list_filters_by_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/hooks"
        assert request.url.params.get("topic") == "github"
        return data_response({"items": [{"name": "h1", "topic": "github", "tool": "notify"}]})

    result = run_cli(monkeypatch, handler, ["hooks", "list", "--topic", "github"])
    assert result.exit_code == 0, result.output
    assert "h1" in result.output


def test_hooks_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/hooks/h1"
        return data_response({"deleted": True})

    result = run_cli(monkeypatch, handler, ["hooks", "delete", "h1"])
    assert result.exit_code == 0, result.output


def test_hooks_set_verifier_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert request.url.path == "/api/hooks/topics/github/verifier"
        assert json.loads(request.content) == {"verifier": "github_hmac", "config": {"secret_env": "GH"}}
        return data_response({"bound": True})

    result = run_cli(
        monkeypatch,
        handler,
        ["hooks", "set-verifier", "github", "--verifier", "github_hmac", "--config", '{"secret_env":"GH"}'],
    )
    assert result.exit_code == 0, result.output


def test_hooks_delete_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/hooks/topics/github/verifier"
        return data_response({"removed": True})

    result = run_cli(monkeypatch, handler, ["hooks", "delete-verifier", "github"])
    assert result.exit_code == 0, result.output


# -- agents / obs ------------------------------------------------------------


def test_agents_spec_runnable_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/agents/spec-runnable"
        return data_response({"items": [{"name": "r", "tool_name": "r_run", "spec_runnable": True}], "total": 1})

    result = run_cli(monkeypatch, handler, ["agents", "spec-runnable"])
    assert result.exit_code == 0, result.output
    assert "r_run" in result.output


def test_obs_metrics_passes_to_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/observability/metrics"
        assert request.url.params.get("to") == "now"
        return data_response({"summary": {"runs": 3}})

    result = run_cli(monkeypatch, handler, ["obs", "metrics", "--to", "now"])
    assert result.exit_code == 0, result.output


# -- system ------------------------------------------------------------------


def test_system_kinds_renders_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/system/kinds"
        return data_response(
            [
                {"kind": "monitoring", "state": "default", "plugin": None, "detail": "noop"},
                {"kind": "storage", "state": "off", "plugin": None, "detail": "dead by default"},
            ]
        )

    result = run_cli(monkeypatch, handler, ["system", "kinds"])
    assert result.exit_code == 0, result.output
    assert "monitoring" in result.output
    assert "storage" in result.output


def test_system_kinds_json_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"kind": "config", "state": "default", "plugin": None, "detail": "file"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return data_response(rows)

    result = run_cli(monkeypatch, handler, ["system", "kinds"], json_output=True)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == rows
