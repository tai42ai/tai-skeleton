"""Remote command groups exercised against a fake ``/api/*`` server.

For each group: a happy path renders the route's result, at least one error status
surfaces as a non-zero exit carrying the server message, ``--json`` output parses,
and the download routes stream their raw (unenveloped) body.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tests.cli.remote_harness import data_response, error_response, run_cli


def test_tools_list_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/tools"
        assert request.headers["x-api-key"] == "test-key"
        return data_response(["alpha", "beta"])

    result = run_cli(monkeypatch, handler, ["tools", "list"])
    assert result.exit_code == 0, result.output
    assert "alpha" in result.output
    assert "beta" in result.output


def test_tools_list_json_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return data_response(["alpha", "beta"])

    result = run_cli(monkeypatch, handler, ["tools", "list"], json_output=True)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == ["alpha", "beta"]


def test_tools_run_posts_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/run-tool"
        assert json.loads(request.content) == {"tool": "add", "kwargs": {"a": 1, "b": 2}}
        return data_response({"sum": 3})

    result = run_cli(monkeypatch, handler, ["tools", "run", "add", "--kw", "a=1", "--kw", "b=2"], json_output=True)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"sum": 3}


def test_tools_schema_not_found_surfaces_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response("Tool 'nope' not registered", 404)

    result = run_cli(monkeypatch, handler, ["tools", "schema", "nope"])
    assert result.exit_code != 0
    assert "Tool 'nope' not registered" in result.output


def test_tools_apply_sends_full_combo_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tools/my_tool/extensions"
        assert json.loads(request.content) == {"combos": [["chain", "batch"], ["chain"]]}
        return data_response({"ok": True})

    result = run_cli(
        monkeypatch, handler, ["tools", "apply", "my_tool", "--combo", '["chain","batch"]', "--combo", '["chain"]']
    )
    assert result.exit_code == 0, result.output


def test_tools_apply_carries_element_config_losslessly(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = {"type": "object"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"combos": [[{"name": "output_schema", "config": {"schema": schema}}]]}
        return data_response({"ok": True})

    result = run_cli(
        monkeypatch,
        handler,
        ["tools", "apply", "my_tool", "--combo", '[{"name":"output_schema","config":{"schema":{"type":"object"}}}]'],
    )
    assert result.exit_code == 0, result.output


def test_tools_apply_mixes_bare_and_configured_elements(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "combos": [["chain", {"name": "output_schema", "config": {"schema": {"type": "string"}}}]]
        }
        return data_response({"ok": True})

    result = run_cli(
        monkeypatch,
        handler,
        [
            "tools",
            "apply",
            "my_tool",
            "--combo",
            '["chain",{"name":"output_schema","config":{"schema":{"type":"string"}}}]',
        ],
    )
    assert result.exit_code == 0, result.output


def test_tools_apply_no_combo_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"combos": []}
        return data_response({"ok": True})

    result = run_cli(monkeypatch, handler, ["tools", "apply", "my_tool"])
    assert result.exit_code == 0, result.output


def test_tools_apply_rejects_malformed_combo_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["tools", "apply", "my_tool", "--combo", "not json"])
    assert result.exit_code != 0
    assert "valid JSON" in result.output


def test_tools_apply_rejects_config_free_object_element(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["tools", "apply", "my_tool", "--combo", '[{"name":"output_schema"}]'])
    assert result.exit_code != 0
    assert "'config' mapping" in result.output


def test_tools_apply_rejects_empty_combo(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["tools", "apply", "my_tool", "--combo", "[]"])
    assert result.exit_code != 0
    assert "non-empty" in result.output


def test_tools_apply_rejects_empty_string_element(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare "" is not a valid extension name (parse_extension_element str branch)."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["tools", "apply", "my_tool", "--combo", '[""]'])
    assert result.exit_code != 0
    assert "an extension name must be a non-empty string" in result.output


def test_tools_apply_rejects_object_without_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """An object element missing 'name' (parse_extension_element dict/name branch)."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["tools", "apply", "my_tool", "--combo", '[{"config":{}}]'])
    assert result.exit_code != 0
    # (substring stays on one wrapped panel line regardless of the option prefix width)
    assert "extension element must have" in result.output


def test_tools_apply_rejects_object_with_extra_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """An object element with keys beyond name/config (parse_extension_element extra branch)."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(
        monkeypatch, handler, ["tools", "apply", "my_tool", "--combo", '[{"name":"chain","config":{},"extra":1}]']
    )
    assert result.exit_code != 0
    assert "unexpected keys" in result.output


def test_tools_apply_rejects_non_element(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-string, non-object element (parse_extension_element fall-through branch)."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["tools", "apply", "my_tool", "--combo", "[5]"])
    assert result.exit_code != 0
    assert "must be an extension name or a" in result.output


def test_tools_runs_list_passes_tool_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tool-runs"
        assert request.url.params.get("tool_name") == "slow"
        return data_response([{"run_id": "r1", "tool_name": "slow", "status": "running", "started_at": "t"}])

    result = run_cli(monkeypatch, handler, ["tools", "runs", "list", "slow"])
    assert result.exit_code == 0, result.output
    assert "r1" in result.output


def test_presets_create_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/presets"
        body = json.loads(request.content)
        assert body["name"] == "greet"
        assert body["base_tool"] == "echo"
        assert body["fixed_kwargs"] == {"prefix": "hi"}
        # Every create is versioned; the body carries no ``versioned`` key.
        assert "versioned" not in body
        return data_response({"name": "greet", "persisted": True})

    result = run_cli(
        monkeypatch,
        handler,
        ["presets", "create", "greet", "--base-tool", "echo", "--kwargs", '{"prefix":"hi"}'],
    )
    assert result.exit_code == 0, result.output


def test_presets_save_version_requires_a_field(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["presets", "save-version", "greet"])
    assert result.exit_code != 0
    assert "at least one" in result.output


def test_presets_save_version_clear_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"tags": []}
        return data_response({"version": 2})

    result = run_cli(monkeypatch, handler, ["presets", "save-version", "greet", "--clear-tags"])
    assert result.exit_code == 0, result.output


def test_agents_list_renders_items(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/agents"
        return data_response({"items": [{"name": "r", "tool_name": "r_run", "spec_runnable": True}], "total": 1})

    result = run_cli(monkeypatch, handler, ["agents", "list"])
    assert result.exit_code == 0, result.output
    assert "r_run" in result.output


def test_extensions_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/extensions"
        return data_response([{"name": "chain", "kind": "wrapper"}])

    result = run_cli(monkeypatch, handler, ["extensions", "list"])
    assert result.exit_code == 0, result.output
    assert "chain" in result.output


def test_connectors_connect_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/connectors/connections/start"
        body = json.loads(request.content)
        assert body["provider_id"] == "google"
        assert body["enabled_sub_services"] == ["gmail"]
        return data_response({"flow_id": "f1", "authorize_url": "https://x"})

    result = run_cli(
        monkeypatch, handler, ["connectors", "connect", "google", "--alias", "work", "--sub-service", "gmail"]
    )
    assert result.exit_code == 0, result.output


def test_connectors_get_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response("connection not found", 404)

    result = run_cli(monkeypatch, handler, ["connectors", "get", "missing"])
    assert result.exit_code != 0
    assert "connection not found" in result.output


def test_hooks_register_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/hooks"
        assert json.loads(request.content) == {"name": "h1", "topic": "gh", "tool": "notify"}
        return data_response({"registered": True, "name": "h1"})

    result = run_cli(
        monkeypatch, handler, ["hooks", "register", "--params", '{"name":"h1","topic":"gh","tool":"notify"}']
    )
    assert result.exit_code == 0, result.output


def test_hooks_verifiers_lists_names(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/hooks/verifiers"
        return data_response(["github_hmac", "shared_secret"])

    result = run_cli(monkeypatch, handler, ["hooks", "verifiers"], json_output=True)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == ["github_hmac", "shared_secret"]


def _trigger_link_reply(topic: str = "orders") -> dict:
    return {
        "name": "trg-link-deadbeef",
        "trigger_path": "/trigger/SECRET",
        "token": "SECRET",
        "topic": topic,
        "expires_at": None,
    }


def test_hooks_trigger_links_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/hooks/trigger-links"
        return data_response(
            {"items": [{"name": "l1", "topic": "orders", "expires_at": None, "token_hash_prefix": "abc"}], "total": 1}
        )

    result = run_cli(monkeypatch, handler, ["hooks", "trigger-links"])
    assert result.exit_code == 0, result.output
    assert "l1" in result.output


def test_hooks_create_trigger_link_timed_composes_absolute_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/hooks/trigger-links"
        assert json.loads(request.content) == {"topic": "orders", "ttl_seconds": 3600}
        return data_response(_trigger_link_reply())

    result = run_cli(
        monkeypatch, handler, ["hooks", "create-trigger-link", "orders", "--ttl", "3600"], json_output=True
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["url"] == "http://testserver/trigger/SECRET"


def test_hooks_create_trigger_link_permanent_null_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"topic": "orders", "ttl_seconds": None}
        return data_response(_trigger_link_reply())

    result = run_cli(monkeypatch, handler, ["hooks", "create-trigger-link", "orders", "--permanent"])
    assert result.exit_code == 0, result.output


def test_hooks_create_trigger_link_params_land_in_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tool_kwargs"] == {"flow_graph_kwargs": {"x": 1}}
        assert body["name"] == "mylink"
        return data_response(_trigger_link_reply())

    result = run_cli(
        monkeypatch,
        handler,
        [
            "hooks",
            "create-trigger-link",
            "orders",
            "--permanent",
            "--name",
            "mylink",
            "--params",
            '{"flow_graph_kwargs":{"x":1}}',
        ],
    )
    assert result.exit_code == 0, result.output


def test_hooks_create_trigger_link_requires_neither_flag_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response(_trigger_link_reply())

    result = run_cli(monkeypatch, handler, ["hooks", "create-trigger-link", "orders"])
    # Neither flag → a loud usage error (no silent default), never a request.
    assert result.exit_code != 0


def test_hooks_create_trigger_link_both_flags_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response(_trigger_link_reply())

    result = run_cli(monkeypatch, handler, ["hooks", "create-trigger-link", "orders", "--ttl", "60", "--permanent"])
    # Both flags → a loud usage error, never a request.
    assert result.exit_code != 0


def test_hooks_create_trigger_link_trailing_slash_base_no_double_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return data_response(_trigger_link_reply())

    result = run_cli(
        monkeypatch,
        handler,
        ["--server", "http://testserver/", "hooks", "create-trigger-link", "orders", "--permanent"],
        json_output=True,
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["url"] == "http://testserver/trigger/SECRET"


def test_hooks_delete_trigger_link(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/hooks/trigger-links/mylink"
        return data_response({"removed": True, "name": "mylink"})

    result = run_cli(monkeypatch, handler, ["hooks", "delete-trigger-link", "mylink"])
    assert result.exit_code == 0, result.output


def test_scopes_routes_renders_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/auth/routes"
        return data_response([{"path": "/api/tools", "methods": ["GET"], "mapped": None}])

    result = run_cli(monkeypatch, handler, ["scopes", "routes"])
    assert result.exit_code == 0, result.output
    assert "/api/tools" in result.output


def test_scopes_public_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/auth/public-routes"
        return data_response(["/universal_webhook/orders"])

    result = run_cli(monkeypatch, handler, ["scopes", "public-list"], json_output=True)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == ["/universal_webhook/orders"]


def test_scopes_public_pin_sends_url_and_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/auth/public-routes"
        assert json.loads(request.content) == {"url": "/open", "pattern": r"/open/\d+"}
        return data_response({"url": "/open"})

    result = run_cli(monkeypatch, handler, ["scopes", "public-pin", "/open", "--pattern", r"/open/\d+"])
    assert result.exit_code == 0, result.output


def test_scopes_public_unpin_sends_delete_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/auth/public-routes"
        assert json.loads(request.content) == {"url": "/open"}
        return data_response({"url": "/open"})

    result = run_cli(monkeypatch, handler, ["scopes", "public-unpin", "/open"])
    assert result.exit_code == 0, result.output


def test_scopes_public_unpin_404_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response("url is not pinned public: '/open'", 404)

    result = run_cli(monkeypatch, handler, ["scopes", "public-unpin", "/open"])
    assert result.exit_code != 0
    assert "not pinned public" in result.output


def test_manifest_show(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/manifest"
        return data_response({"mcp": [], "user_tools": ["a"]})

    result = run_cli(monkeypatch, handler, ["manifest", "show"], json_output=True)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"mcp": [], "user_tools": ["a"]}


def test_manifest_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/plugins"
        return data_response([{"name": "studio-x"}])

    result = run_cli(monkeypatch, handler, ["manifest", "plugins"])
    assert result.exit_code == 0, result.output
    assert "studio-x" in result.output


def test_manifest_replace_posts_text_verbatim_with_markers_intact(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # The command posts the manifest TEXT verbatim — ``!ENV`` markers are NEVER
    # resolved client-side (a first-party client resolving a secret before a
    # persist-through replace would bake it to disk); the server owns resolution.
    raw = "backend_module: !ENV ${TAI_BACKEND}\nmcp: []\n"
    manifest_file = tmp_path / "manifest.yml"
    manifest_file.write_text(raw, encoding="utf-8")
    monkeypatch.setenv("TAI_BACKEND", "myapp.backend")  # would resolve IF the client parsed it

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/manifest/replace"
        body = json.loads(request.content)
        # The exact file text, marker intact — no resolved value, no ``targets``.
        assert body == {"manifest_text": raw}
        assert "myapp.backend" not in request.content.decode()
        return data_response({"status": "ok"})

    result = run_cli(monkeypatch, handler, ["manifest", "replace", "--file", str(manifest_file)])
    assert result.exit_code == 0, result.output


def test_mcp_set_from_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps([{"title": "srv"}]), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/mcp-config"
        assert json.loads(request.content) == {"mcp": [{"title": "srv"}]}
        return data_response({"reloaded": True})

    result = run_cli(monkeypatch, handler, ["mcp", "set", "--file", str(config)])
    assert result.exit_code == 0, result.output


def test_sub_mcp_register(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"slug": "billing", "tools": ["invoice"]}
        return data_response({"slug": "billing", "tools": ["invoice"]})

    result = run_cli(monkeypatch, handler, ["sub-mcp", "register", "billing", "--tool", "invoice"])
    assert result.exit_code == 0, result.output


def test_templates_render_exclusive_args(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["templates", "render"])
    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_templates_render_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["template_id"] == "greet"
        assert body["kwargs"] == {"name": "Ada"}
        return data_response({"rendered": "hi Ada"})

    result = run_cli(monkeypatch, handler, ["templates", "render", "--template-id", "greet", "--kw", 'name="Ada"'])
    assert result.exit_code == 0, result.output


def test_config_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/config/env"
        assert json.loads(request.content) == {"LOG_LEVEL": "debug"}
        return data_response({"reloaded": True})

    result = run_cli(monkeypatch, handler, ["config", "env", "set", "LOG_LEVEL=debug"])
    assert result.exit_code == 0, result.output


def test_keys_create_returns_raw_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/auth/api-keys"
        body = json.loads(request.content)
        assert body["user_id"] == "alice"
        assert body["scopes"] == ["read"]
        return data_response("sk-secret")

    result = run_cli(
        monkeypatch, handler, ["keys", "create", "--user", "alice", "--description", "ci", "--scope", "read"]
    )
    assert result.exit_code == 0, result.output
    assert "sk-secret" in result.output


def test_keys_edit_requires_a_field(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return data_response({})

    result = run_cli(monkeypatch, handler, ["keys", "edit", "alice"])
    assert result.exit_code != 0


def test_scopes_remove_url_sends_delete_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/auth/scopes/urls"
        assert json.loads(request.content) == {"url": "/api/tools"}
        return data_response({"url": "/api/tools"})

    result = run_cli(monkeypatch, handler, ["scopes", "remove-url", "/api/tools"])
    assert result.exit_code == 0, result.output


def test_schedules_list_501_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response("no installed backend exposes scheduling tools", 501)

    result = run_cli(monkeypatch, handler, ["schedules", "list"])
    assert result.exit_code != 0
    assert "scheduling tools" in result.output


def test_obs_metrics_passes_params(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/observability/metrics"
        assert request.url.params.get("from") == "7d"
        assert request.url.params.get("granularity") == "day"
        return data_response({"summary": {}})

    result = run_cli(monkeypatch, handler, ["obs", "metrics", "--from", "7d", "--granularity", "day"])
    assert result.exit_code == 0, result.output


def test_traces_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/observability/runs"
        return data_response({"items": [{"traceId": "t1", "status": "ok"}], "page": 1, "nextPage": None})

    result = run_cli(monkeypatch, handler, ["traces", "list"])
    assert result.exit_code == 0, result.output
    assert "t1" in result.output


def test_backup_export_streams_raw_document(monkeypatch: pytest.MonkeyPatch) -> None:
    document = {"version": 1, "created_at": "t", "sections": {"access_control": {}}, "errors": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/backup/export"
        assert json.loads(request.content) == {"sections": ["access_control"]}
        # A download route: the body is the RAW document, not a {"data": ...} envelope.
        return httpx.Response(200, json=document, headers={"content-disposition": 'attachment; filename="b.json"'})

    result = run_cli(monkeypatch, handler, ["backup", "export", "--section", "access_control"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == document


def test_backup_export_error_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response("unknown section(s): bogus", 400)

    result = run_cli(monkeypatch, handler, ["backup", "export", "--section", "bogus"])
    assert result.exit_code != 0
    assert "unknown section" in result.output


def test_traces_export_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/observability/runs/export"
        assert request.url.params.get("format") == "csv"
        return httpx.Response(200, text="traceId,status\nt1,ok\n", headers={"content-type": "text/csv"})

    result = run_cli(monkeypatch, handler, ["traces", "list", "--export", "--format", "csv"])
    assert result.exit_code == 0, result.output
    assert "traceId,status" in result.output


def test_unauthenticated_surfaces_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response("missing key", 401)

    result = run_cli(monkeypatch, handler, ["tools", "list"])
    assert result.exit_code != 0
    assert "not authenticated" in result.output


# -- channels ------------------------------------------------------------------


def test_channels_list_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/channels"
        assert request.headers["x-api-key"] == "test-key"
        return data_response({"channels": ["slack", "telegram"]})

    result = run_cli(monkeypatch, handler, ["channels", "list"])
    assert result.exit_code == 0, result.output
    assert "slack" in result.output
    assert "telegram" in result.output


def test_channels_list_json_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return data_response({"channels": ["telegram"]})

    result = run_cli(monkeypatch, handler, ["channels", "list"], json_output=True)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"channels": ["telegram"]}


def test_channels_list_error_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response("boom", 500)

    result = run_cli(monkeypatch, handler, ["channels", "list"])
    assert result.exit_code != 0
    assert "boom" in result.output


# -- notifications -------------------------------------------------------------


def test_notifications_list_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/notifications"
        assert request.headers["x-api-key"] == "test-key"
        return data_response(
            {
                "notifications": [
                    {"id": "b", "message": "deploy done", "recipient": "ops", "created_at": "2026-07-11T00:00:01Z"},
                    {"id": "a", "message": "deploy started", "recipient": None, "created_at": "2026-07-11T00:00:00Z"},
                ]
            }
        )

    result = run_cli(monkeypatch, handler, ["notifications", "list"])
    assert result.exit_code == 0, result.output
    # Newest-first, as the feed returns it.
    assert result.output.index("deploy done") < result.output.index("deploy started")
    assert "ops" in result.output


def test_notifications_list_empty_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return data_response({"notifications": []})

    result = run_cli(monkeypatch, handler, ["notifications", "list"])
    assert result.exit_code == 0, result.output


def test_notifications_list_json_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return data_response({"notifications": [{"message": "hi", "recipient": None, "created_at": "t"}]})

    result = run_cli(monkeypatch, handler, ["notifications", "list"], json_output=True)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"notifications": [{"message": "hi", "recipient": None, "created_at": "t"}]}


def test_notifications_list_error_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response("boom", 500)

    result = run_cli(monkeypatch, handler, ["notifications", "list"])
    assert result.exit_code != 0
    assert "boom" in result.output


# -- auth whoami -------------------------------------------------------------


def test_auth_whoami_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/auth/me"
        assert request.headers["x-api-key"] == "test-key"
        return data_response(
            {
                "user_id": "u1",
                "owner_user_id": None,
                "admin": False,
                "scopes": ["read"],
                "routes": [],
                "route_patterns": [],
                "sub_mcp": [],
                "tools": [],
                "agents": [],
                "mintable": True,
            }
        )

    result = run_cli(monkeypatch, handler, ["auth", "whoami"])
    assert result.exit_code == 0, result.output
    assert "u1" in result.output


def test_auth_whoami_json_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return data_response({"user_id": "u1", "admin": True, "scopes": ["*"]})

    result = run_cli(monkeypatch, handler, ["auth", "whoami"], json_output=True)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["admin"] is True


def test_auth_whoami_error_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response("Unauthorized", 401)

    result = run_cli(monkeypatch, handler, ["auth", "whoami"])
    assert result.exit_code != 0
    assert "Unauthorized" in result.output


# -- keys claim-link ---------------------------------------------------------


def test_keys_claim_link_posts_body_and_prints_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/auth/claim-links"
        assert request.headers["x-api-key"] == "test-key"
        # The --ttl flag maps to the wire's ``ttl_seconds`` field.
        assert json.loads(request.content) == {"api_key": "sk-secret", "ttl_seconds": 300}
        return data_response(
            {"claim_path": "/login#claim=clm-xyz", "token": "clm-xyz", "expires_at": "2026-07-16T00:00:00+00:00"}
        )

    result = run_cli(monkeypatch, handler, ["keys", "claim-link", "sk-secret", "--ttl", "300"])
    assert result.exit_code == 0, result.output
    assert "/login#claim=clm-xyz" in result.output


def test_keys_claim_link_reads_key_from_hidden_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    # With the positional key omitted the command reads it from a HIDDEN prompt so the
    # secret never lands in shell history; the prompted key rides the POST body.
    from tai42_skeleton.cli.commands import keys as keys_module

    prompt_call: dict = {}

    def fake_prompt(text: str, **kwargs: object) -> str:
        prompt_call["text"] = text
        prompt_call["kwargs"] = kwargs
        return "sk-prompted"

    monkeypatch.setattr(keys_module.typer, "prompt", fake_prompt)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/auth/claim-links"
        assert json.loads(request.content) == {"api_key": "sk-prompted"}
        return data_response(
            {"claim_path": "/login#claim=clm-xyz", "token": "clm-xyz", "expires_at": "2026-07-16T00:00:00+00:00"}
        )

    result = run_cli(monkeypatch, handler, ["keys", "claim-link"])
    assert result.exit_code == 0, result.output
    assert "/login#claim=clm-xyz" in result.output
    # The key was read at a HIDDEN prompt (hide_input=True) and never echoed anywhere.
    assert prompt_call["kwargs"].get("hide_input") is True
    assert "sk-prompted" not in result.output


def test_keys_claim_link_omits_ttl_when_flag_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without --ttl the body carries only ``api_key`` — ``ttl_seconds`` is ABSENT, not null.
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/auth/claim-links"
        assert json.loads(request.content) == {"api_key": "sk-x"}
        return data_response(
            {"claim_path": "/login#claim=clm-xyz", "token": "clm-xyz", "expires_at": "2026-07-16T00:00:00+00:00"}
        )

    result = run_cli(monkeypatch, handler, ["keys", "claim-link", "sk-x"])
    assert result.exit_code == 0, result.output
    assert "/login#claim=clm-xyz" in result.output


# -- auth claim (public, credential-free) ------------------------------------


def test_auth_claim_sends_no_credential_and_extracts_token_from_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/login/claim"
        # The public exchange carries NO credential header.
        assert "x-api-key" not in request.headers
        assert "authorization" not in request.headers
        # A full pasted claim URL is reduced to its bare fragment token.
        assert json.loads(request.content) == {"token": "clm-abc123"}
        return data_response({"token": "sk-live", "user_id": "u1"})

    result = run_cli(monkeypatch, handler, ["auth", "claim", "https://host/login#claim=clm-abc123"])
    assert result.exit_code == 0, result.output
    assert "sk-live" in result.output


def test_auth_claim_accepts_a_bare_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"token": "clm-bare"}
        return data_response({"token": "sk-live", "user_id": "u1"})

    result = run_cli(monkeypatch, handler, ["auth", "claim", "clm-bare"])
    assert result.exit_code == 0, result.output


def test_auth_claim_404_renders_uniform_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response("unknown or already used claim token", 404)

    result = run_cli(monkeypatch, handler, ["auth", "claim", "clm-nope"])
    assert result.exit_code != 0
    assert "unknown or already used claim token" in result.output
