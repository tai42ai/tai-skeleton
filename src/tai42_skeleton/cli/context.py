"""Shared invocation context for the ``tai`` CLI.

The root callback builds an :class:`AppContext` and stashes it on the Typer
context; every remote command reads it to obtain a configured
:class:`~tai42_skeleton.cli.client.ApiClient`. Resolution of the server URL and the
API key is lazy — a local command such as ``tai serve`` never triggers the
interactive key prompt.

Server URL resolution order: ``--server`` flag → ``TAI_SERVER_URL`` env →
``config.toml`` → the local default. The local default port is NOT hard-coded:
it comes from the serve defaults (``app_args_settings().port``), the single
source of truth shared with ``tai serve``, so a default ``tai serve`` and the
default ``--server`` URL always agree.

API key resolution order: ``--api-key-stdin`` (read one line from stdin) →
``TAI_API_KEY`` env → ``config.toml`` → an interactive prompt as a last resort.
There is deliberately no ``--api-key VALUE`` flag: a value on the command line
leaks through ``ps``/``/proc`` and shell history.
"""

import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import typer

from tai42_skeleton.cli.client import ApiClient
from tai42_skeleton.settings.cache import app_args_settings

SERVER_URL_ENV = "TAI_SERVER_URL"
API_KEY_ENV = "TAI_API_KEY"


def config_path() -> Path:
    """The CLI config file location: ``$XDG_CONFIG_HOME/tai/config.toml`` when
    ``XDG_CONFIG_HOME`` is set, else ``~/.config/tai/config.toml``."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "tai" / "config.toml"


def load_config() -> dict[str, Any]:
    """Parse the CLI config file, or return an empty mapping when it is absent.

    A malformed config file raises loudly rather than being silently ignored.
    """
    path = config_path()
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        try:
            return tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise typer.BadParameter(f"malformed config file {path}: {exc}", param_hint="config.toml") from exc


def default_server_url() -> str:
    """The local server URL built from the serve defaults' port."""
    return f"http://127.0.0.1:{app_args_settings().port}"


def resolve_server_url(override: str | None) -> str:
    if override:
        return override
    from_env = os.environ.get(SERVER_URL_ENV)
    if from_env:
        return from_env
    from_config = load_config().get("server_url")
    if isinstance(from_config, str) and from_config:
        return from_config
    return default_server_url()


def resolve_api_key(*, from_stdin: bool) -> str:
    if from_stdin:
        key = sys.stdin.readline().strip()
        if not key:
            raise typer.BadParameter("no API key was provided on stdin.", param_hint="--api-key-stdin")
        return key
    from_env = os.environ.get(API_KEY_ENV)
    if from_env:
        return from_env
    from_config = load_config().get("api_key")
    if isinstance(from_config, str) and from_config:
        return from_config
    return typer.prompt("API key", hide_input=True)


class AppContext:
    """Per-invocation CLI state: the global flags plus lazy access to the
    resolved server URL, API key, and a configured :class:`ApiClient`."""

    def __init__(self, *, json_output: bool, server_override: str | None, api_key_stdin: bool) -> None:
        self.json_output = json_output
        self._server_override = server_override
        self._api_key_stdin = api_key_stdin
        self._server_url: str | None = None
        self._api_key: str | None = None

    @property
    def server_url(self) -> str:
        if self._server_url is None:
            self._server_url = resolve_server_url(self._server_override)
        return self._server_url

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            self._api_key = resolve_api_key(from_stdin=self._api_key_stdin)
        return self._api_key

    def client(self, *, anonymous: bool = False) -> ApiClient:
        """A configured client for the resolved server + API key.

        With ``anonymous=True`` the client carries NO credential — the api-key
        resolution (and its interactive prompt) is never triggered — for the single
        public door the CLI calls (``tai auth claim``, which has no key yet)."""
        return ApiClient(self.server_url, None if anonymous else self.api_key)
