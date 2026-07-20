"""Resolution precedence for the server URL and the API key.

Server URL: ``--server`` flag -> ``TAI_SERVER_URL`` -> ``config.toml`` ->
the local default (port from the serve defaults). API key: ``--api-key-stdin``
-> ``TAI_API_KEY`` -> ``config.toml`` -> interactive prompt.
"""

from __future__ import annotations

import io

import pytest

from tai_skeleton.cli import context
from tai_skeleton.settings.cache import app_args_settings


@pytest.fixture
def config_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point config resolution at a writable temp ``config.toml`` via
    ``XDG_CONFIG_HOME`` and return a writer for its contents."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    def write(body: str) -> None:
        path = tmp_path / "tai" / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    return write


# --- server URL -----------------------------------------------------------


def test_server_flag_wins(monkeypatch: pytest.MonkeyPatch, config_file) -> None:
    monkeypatch.setenv(context.SERVER_URL_ENV, "http://from-env")
    config_file('server_url = "http://from-config"\n')

    assert context.resolve_server_url("http://from-flag") == "http://from-flag"


def test_server_env_beats_config_and_default(monkeypatch: pytest.MonkeyPatch, config_file) -> None:
    monkeypatch.setenv(context.SERVER_URL_ENV, "http://from-env")
    config_file('server_url = "http://from-config"\n')

    assert context.resolve_server_url(None) == "http://from-env"


def test_server_config_beats_default(monkeypatch: pytest.MonkeyPatch, config_file) -> None:
    monkeypatch.delenv(context.SERVER_URL_ENV, raising=False)
    config_file('server_url = "http://from-config"\n')

    assert context.resolve_server_url(None) == "http://from-config"


def test_server_default_uses_serve_default_port(monkeypatch: pytest.MonkeyPatch, config_file) -> None:
    monkeypatch.delenv(context.SERVER_URL_ENV, raising=False)
    # No config file written -> falls through to the local default.
    expected = f"http://127.0.0.1:{app_args_settings().port}"
    assert context.resolve_server_url(None) == expected
    assert context.default_server_url() == expected


def test_missing_config_file_is_empty(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert context.load_config() == {}


def test_malformed_config_raises_clean_error(config_file) -> None:
    # A broken config file surfaces as a typer ``BadParameter`` usage error naming the
    # path (typer renders it as its vendored Rich usage-error panel, exit code 2), never
    # a raw tomllib traceback.
    config_file("[unterminated\n")
    with pytest.raises(context.typer.BadParameter, match="malformed config file"):
        context.load_config()


# --- API key --------------------------------------------------------------


def test_api_key_stdin_wins(monkeypatch: pytest.MonkeyPatch, config_file) -> None:
    monkeypatch.setenv(context.API_KEY_ENV, "env-key")
    config_file('api_key = "config-key"\n')
    monkeypatch.setattr(context.sys, "stdin", io.StringIO("stdin-key\n"))

    assert context.resolve_api_key(from_stdin=True) == "stdin-key"


def test_api_key_stdin_empty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context.sys, "stdin", io.StringIO("\n"))

    with pytest.raises(context.typer.BadParameter):
        context.resolve_api_key(from_stdin=True)


def test_api_key_env_beats_config_and_prompt(monkeypatch: pytest.MonkeyPatch, config_file) -> None:
    monkeypatch.setenv(context.API_KEY_ENV, "env-key")
    config_file('api_key = "config-key"\n')

    assert context.resolve_api_key(from_stdin=False) == "env-key"


def test_api_key_config_beats_prompt(monkeypatch: pytest.MonkeyPatch, config_file) -> None:
    monkeypatch.delenv(context.API_KEY_ENV, raising=False)
    config_file('api_key = "config-key"\n')

    assert context.resolve_api_key(from_stdin=False) == "config-key"


def test_api_key_prompt_is_last_resort(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv(context.API_KEY_ENV, raising=False)
    monkeypatch.setattr(context.typer, "prompt", lambda *args, **kwargs: "prompted-key")

    assert context.resolve_api_key(from_stdin=False) == "prompted-key"


# --- AppContext -----------------------------------------------------------


def test_app_context_resolves_lazily_and_caches(monkeypatch: pytest.MonkeyPatch, config_file) -> None:
    monkeypatch.setenv(context.SERVER_URL_ENV, "http://from-env")
    monkeypatch.setenv(context.API_KEY_ENV, "env-key")

    ctx = context.AppContext(json_output=True, server_override=None, api_key_stdin=False)
    assert ctx.json_output is True
    assert ctx.server_url == "http://from-env"
    assert ctx.api_key == "env-key"

    # A later env change does not disturb the already-cached resolution.
    monkeypatch.setenv(context.SERVER_URL_ENV, "http://changed")
    assert ctx.server_url == "http://from-env"


def test_app_context_client_uses_resolved_values(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv(context.SERVER_URL_ENV, "http://client-host")
    monkeypatch.setenv(context.API_KEY_ENV, "client-key")

    ctx = context.AppContext(json_output=False, server_override="http://override", api_key_stdin=False)
    client = ctx.client()
    try:
        assert str(client._client.base_url) == "http://override"
        assert client._client.headers["x-api-key"] == "client-key"
    finally:
        client.close()
