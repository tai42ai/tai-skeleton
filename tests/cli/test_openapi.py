"""OpenAPI 3.1 emitter tests — the coverage gate, offline emission, and the
``tai openapi`` command.

The coverage gate is the contract enforcer: every registered ``/api/*`` route
must appear in the emitted spec, meet the self-describe minimum bar, carry the
reload-gate ``503`` when it is gated, and the whole document must validate
against the OpenAPI 3.1 schema. A route that fails to self-describe fails here.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from click.testing import CliRunner
from openapi_spec_validator import validate
from pydantic import BaseModel

from tai42_skeleton.app.reload_gate import REJECT_MESSAGE
from tai42_skeleton.app.route_registry import RouteMetadata, load_api_routes
from tai42_skeleton.cli import app as app_module
from tai42_skeleton.cli.openapi import _assign_component, _openapi_path, _register_model, build_openapi_spec


@pytest.fixture(scope="module")
def spec() -> dict:
    return build_openapi_spec()


@pytest.fixture(scope="module")
def api_routes() -> list[RouteMetadata]:
    return load_api_routes()


def _operation(spec: dict, meta: RouteMetadata, method: str) -> dict:
    # Normalize the Starlette path to its OpenAPI form exactly as the emitter does
    # (dropping the ``:path`` converter from any param), so a ``{name:path}`` route
    # is matched however it is named.
    oapath = _openapi_path(meta.path)
    assert oapath in spec["paths"], f"route {meta.path} missing from spec"
    op = spec["paths"][oapath].get(method.lower())
    assert op is not None, f"{method} {meta.path} missing from spec"
    return op


# -- Coverage gate -----------------------------------------------------------


def test_every_api_route_appears_in_the_spec(spec: dict, api_routes: list[RouteMetadata]) -> None:
    assert api_routes, "no /api routes enumerated — the registry is empty"
    for meta in api_routes:
        for method in meta.methods:
            _operation(spec, meta, method)


def test_every_route_meets_the_self_describe_bar(api_routes: list[RouteMetadata]) -> None:
    for meta in api_routes:
        assert meta.summary, f"{meta.path} missing a summary"
        assert meta.tags, f"{meta.path} missing tags"
        assert isinstance(meta.authed, bool), f"{meta.path} has a non-bool authed"
        # request_model is REQUIRED for every authed route that reads a body; a
        # public external door (authed=False) may accept an opaque provider body.
        if meta.authed and meta.reads_body:
            assert meta.request_model is not None, f"{meta.path} reads a body but declares no request_model"
        # response_model is always present as an attribute; None IS the accepted
        # opaque marker, so its mere presence is the bar (never AttributeError).
        assert meta.response_model is None or isinstance(meta.response_model, type)


def test_gated_routes_declare_the_retriable_503(spec: dict, api_routes: list[RouteMetadata]) -> None:
    gated = [m for m in api_routes if m.reload_gated]
    assert gated, "no reload-gated routes derived — the reload-gate derivation is broken"
    for meta in gated:
        for method in meta.methods:
            op = _operation(spec, meta, method)
            assert "503" in op["responses"], f"gated route {method} {meta.path} lacks the 503 response"
            response = op["responses"]["503"]
            assert response["headers"]["Retry-After"]["schema"]["type"] == "integer"
            ref = response["content"]["application/json"]["schema"]["$ref"]
            assert ref.endswith("/ReloadingError")


# The reload-gated routes, hand-maintained as ground truth. ``reload_gated`` is
# DECLARED per route (an operation's metadata, or a native handler's explicit
# declaration). Pinning the declared set to this list turns any change to the
# gated surface into a test failure, forcing the author to confirm the 503
# coverage.
_EXPECTED_RELOAD_GATED: set[tuple[str, str]] = {
    ("POST", "/api/agents/authored/{name}/runs"),
    ("POST", "/api/agents/{name}/runs"),
    ("POST", "/api/backup/import"),
    ("POST", "/api/config/env"),
    ("POST", "/api/config/reload"),
    ("POST", "/api/manifest/replace"),
    ("POST", "/api/marketplace/install"),
    ("POST", "/api/marketplace/uninstall"),
    ("POST", "/api/marketplace/update"),
    ("POST", "/api/mcp-config"),
    ("POST", "/api/mcp-status/reload-failed"),
    ("POST", "/api/mcp-status/{title}/deregister"),
    ("POST", "/api/mcp-status/{title}/reload"),
    ("POST", "/api/presets"),
    ("DELETE", "/api/presets/{name}"),
    ("POST", "/api/presets/{name}/rename"),
    ("POST", "/api/presets/{name}/rollback"),
    ("POST", "/api/presets/{name}/versions"),
    ("POST", "/api/run-tool"),
    ("POST", "/api/schedules"),
    ("DELETE", "/api/schedules/{schedule_name}"),
    ("POST", "/api/sub-mcp"),
    ("DELETE", "/api/sub-mcp/{slug}"),
    ("POST", "/api/tool-runs"),
    ("POST", "/api/tools/reload"),
    ("POST", "/api/tools/remove"),
    ("POST", "/api/tools/{name}/extensions"),
}

# The body-reading routes, hand-maintained as ground truth (same coverage intent as
# the gated set: ``reads_body`` is declared per route, so pinning the full set trips
# on any change to the body-reading surface).
_EXPECTED_READS_BODY: set[tuple[str, str]] = {
    ("POST", "/api/agents/authored/{name}/runs"),
    ("POST", "/api/agents/{name}/runs"),
    ("POST", "/api/auth/api-keys"),
    ("PUT", "/api/auth/api-keys/{user_id}"),
    ("POST", "/api/auth/api-keys/{user_id}/policy/rollback"),
    ("POST", "/api/auth/claim-links"),
    ("POST", "/api/auth/roles"),
    ("PUT", "/api/auth/roles/{name}"),
    ("POST", "/api/auth/roles/{name}/rollback"),
    ("POST", "/api/auth/scopes"),
    ("DELETE", "/api/auth/scopes/urls"),
    ("POST", "/api/auth/public-routes"),
    ("DELETE", "/api/auth/public-routes"),
    ("POST", "/api/auth/validate-condition"),
    ("POST", "/api/fleet/reload-config"),
    ("POST", "/api/backup/export"),
    ("POST", "/api/backup/import"),
    ("POST", "/api/config/env"),
    ("POST", "/api/config/reload"),
    ("POST", "/api/connectors/connections/start"),
    ("POST", "/api/connectors/connections/{connection_id}/reconnect"),
    ("PATCH", "/api/connectors/connections/{connection_id}/sub-services"),
    ("POST", "/api/connectors/oauth/complete"),
    ("POST", "/api/delete-template"),
    ("POST", "/api/hooks"),
    ("PUT", "/api/hooks/topics/{topic}/verifier"),
    ("POST", "/api/interactions/{interaction_id}/answer"),
    ("POST", "/api/login/claim"),
    ("POST", "/api/manifest/replace"),
    ("POST", "/api/marketplace/install"),
    ("POST", "/api/marketplace/uninstall"),
    ("POST", "/api/marketplace/update"),
    ("POST", "/api/mcp-config"),
    ("POST", "/api/mcp-status/reload-failed"),
    ("POST", "/api/mcp-status/{title}/deregister"),
    ("POST", "/api/mcp-status/{title}/reload"),
    ("POST", "/api/notifications"),
    ("POST", "/api/presets"),
    ("POST", "/api/presets/validate"),
    ("POST", "/api/presets/{name}/rename"),
    ("POST", "/api/presets/{name}/rollback"),
    ("POST", "/api/presets/{name}/versions"),
    ("PUT", "/api/presets/{name}/versions/{version}/tags"),
    ("POST", "/api/render-template"),
    ("POST", "/api/resources/get"),
    ("POST", "/api/run-tool"),
    ("POST", "/api/schedules"),
    ("POST", "/api/storage/resources"),
    ("POST", "/api/sub-mcp"),
    ("POST", "/api/template"),
    ("POST", "/api/tool-runs"),
    ("POST", "/api/tools/reload"),
    ("POST", "/api/tools/remove"),
    ("POST", "/api/tools/{name}/extensions"),
    ("POST", "/api/upload-template"),
}


def test_declared_reload_gated_set_matches_ground_truth(api_routes: list[RouteMetadata]) -> None:
    declared = {(method, meta.path) for meta in api_routes if meta.reload_gated for method in meta.methods}
    assert declared == _EXPECTED_RELOAD_GATED


def test_declared_reads_body_set_matches_ground_truth(api_routes: list[RouteMetadata]) -> None:
    declared = {(method, meta.path) for meta in api_routes if meta.reads_body for method in meta.methods}
    assert declared == _EXPECTED_READS_BODY


def test_tool_runs_submission_documents_the_202_accepted(spec: dict) -> None:
    responses = spec["paths"]["/api/tool-runs"]["post"]["responses"]
    assert "202" in responses, "the detached tool-run submission returns 202, not 200"
    assert "200" not in responses
    assert responses["202"]["content"]["application/json"]["schema"]["required"] == ["data"]


def test_callback_documents_html_get_and_json_post(spec: dict) -> None:
    # The callback door is one registration with two methods that answer different
    # media types: GET serves the browser confirm page (HTML) while POST is the
    # programmatic answer door returning the ``{"data": ...}`` JSON envelope.
    callback = spec["paths"]["/api/interactions/callback/{ticket}"]
    get_content = callback["get"]["responses"]["200"]["content"]
    assert list(get_content) == ["text/html"]
    post_content = callback["post"]["responses"]["200"]["content"]
    assert list(post_content) == ["application/json"]
    assert post_content["application/json"]["schema"]["required"] == ["data"]


def test_callback_documents_its_error_statuses(spec: dict, api_routes: list[RouteMetadata]) -> None:
    # The callback door declares the full set it answers: 400 (malformed JSON body),
    # 401 (failed verification), 404 (unknown/expired ticket), 413 (oversized
    # body/query), 500 (verifier error). Pinned as ground truth so a change to the
    # declared set trips here.
    (callback,) = [m for m in api_routes if m.path == "/api/interactions/callback/{ticket}"]
    assert set(callback.error_statuses) == {400, 401, 404, 413, 500}
    responses = spec["paths"]["/api/interactions/callback/{ticket}"]["post"]["responses"]
    for status in ("400", "401", "404", "413", "500"):
        assert status in responses, f"callback POST is missing the {status} response"


def test_observability_routes_document_the_501(api_routes: list[RouteMetadata]) -> None:
    # Every observability route answers 501 when monitoring reads are unsupported
    # (MonitoringReadNotSupportedError), so each declares it.
    observability = [m for m in api_routes if m.path.startswith("/api/observability/")]
    assert observability, "no observability routes enumerated"
    for meta in observability:
        assert 501 in meta.error_statuses, f"{meta.path} lost the 501"


def test_delete_template_declares_only_its_typed_errors(api_routes: list[RouteMetadata]) -> None:
    # delete-template's operation declares BadRequestError only, so the route
    # documents {400, 401} and no spurious 500.
    (meta,) = [m for m in api_routes if m.path == "/api/delete-template"]
    assert set(meta.error_statuses) == {400, 401}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("DELETE", "/api/auth/scopes/urls"),
        ("DELETE", "/api/auth/public-routes"),
    ],
)
def test_scope_url_delete_doors_document_the_400(
    spec: dict, api_routes: list[RouteMetadata], method: str, path: str
) -> None:
    # Both delete doors validate their JSON body at the request edge (a blank/missing
    # ``url`` is a 400) before the operation runs (a url that was never mapped is a 404), so
    # each documents {400, 401, 404} — the 400 the extractor answers must be in the spec,
    # not just the runtime.
    (meta,) = [m for m in api_routes if m.path == path and method in m.methods]
    assert set(meta.error_statuses) == {400, 401, 404}
    responses = spec["paths"][path][method.lower()]["responses"]
    assert "400" in responses, f"{method} {path} is missing the 400 response"


def test_runs_export_documents_both_csv_and_json_download(spec: dict) -> None:
    # The runs export serves either a CSV body or a JSON download from one GET, so
    # its 200 lists both content types rather than being pinned to CSV alone.
    content = spec["paths"]["/api/observability/runs/export"]["get"]["responses"]["200"]["content"]
    assert "text/csv" in content
    assert "application/octet-stream" in content


def test_register_model_rejects_a_reserved_envelope_name() -> None:
    class Error(BaseModel):
        detail: str

    with pytest.raises(ValueError, match="reserved"):
        _register_model(Error, {})


def test_register_model_rejects_a_conflicting_same_name_schema() -> None:
    components: dict = {}
    _assign_component(components, "Widget", {"type": "object", "properties": {"a": {"type": "integer"}}})
    with pytest.raises(ValueError, match="collision"):
        _assign_component(components, "Widget", {"type": "object", "properties": {"b": {"type": "string"}}})


def test_register_model_allows_idempotent_reregistration() -> None:
    class Gadget(BaseModel):
        a: int

    components: dict = {}
    assert _register_model(Gadget, components) == "Gadget"
    # The same model reached from a second route registers identically — no raise.
    assert _register_model(Gadget, components) == "Gadget"


def test_reloading_error_schema_matches_the_gate_body(spec: dict) -> None:
    schema = spec["components"]["schemas"]["ReloadingError"]
    assert schema["properties"]["error"]["const"] == REJECT_MESSAGE
    assert schema["properties"]["reloading"]["const"] is True


def test_authed_routes_require_the_api_key_security(spec: dict, api_routes: list[RouteMetadata]) -> None:
    for meta in api_routes:
        for method in meta.methods:
            op = _operation(spec, meta, method)
            if meta.authed:
                assert op["security"] == [{"ApiKeyAuth": []}], f"{meta.path} missing api-key security"
                assert "401" in op["responses"]
            else:
                assert "security" not in op


def test_body_routes_declare_a_request_body(spec: dict, api_routes: list[RouteMetadata]) -> None:
    for meta in api_routes:
        if meta.request_model is None:
            continue
        for method in meta.methods:
            op = _operation(spec, meta, method)
            ref = op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
            assert ref.endswith("/" + meta.request_model.__name__)
            assert meta.request_model.__name__ in spec["components"]["schemas"]


def test_spec_validates_against_openapi_31(spec: dict) -> None:
    validate(spec)
    assert spec["openapi"] == "3.1.0"


# -- Offline emission --------------------------------------------------------


def test_emission_touches_no_db_or_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the spec must not open a client — patch the pooled-client seam to
    raise, then prove emission still succeeds and validates."""
    import tai42_kit.clients as clients

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("spec emission must not open a database/Redis client")

    monkeypatch.setattr(clients, "client_ctx", _forbidden)
    spec = build_openapi_spec()
    validate(spec)


def test_emission_runs_in_a_bare_process_without_db_or_redis_env() -> None:
    """A fresh interpreter with all DB/Redis/manifest env stripped emits a valid
    spec — the docs pipeline's usage, with no environment booted."""
    import os

    stripped = {
        k: v
        for k, v in os.environ.items()
        if not any(token in k.upper() for token in ("REDIS", "POSTGRES", "DATABASE", "TAI_"))
    }
    code = (
        "import json;"
        "from tai42_skeleton.cli.openapi import build_openapi_spec;"
        "from openapi_spec_validator import validate;"
        "s = build_openapi_spec();"
        "validate(s);"
        "print('OK', len(s['paths']))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        env=stripped,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK ")


# -- ``tai openapi`` command -------------------------------------------------


def test_openapi_command_prints_valid_spec_to_stdout() -> None:
    result = CliRunner().invoke(app_module.app, ["openapi"])
    assert result.exit_code == 0, result.output
    validate(json.loads(result.output))


def test_openapi_command_writes_to_out_path(tmp_path) -> None:
    target = tmp_path / "openapi.json"
    result = CliRunner().invoke(app_module.app, ["openapi", "--out", str(target)])
    assert result.exit_code == 0, result.output
    validate(json.loads(target.read_text()))


def test_openapi_check_succeeds_on_a_valid_spec() -> None:
    result = CliRunner().invoke(app_module.app, ["openapi", "--check"])
    assert result.exit_code == 0, result.output
