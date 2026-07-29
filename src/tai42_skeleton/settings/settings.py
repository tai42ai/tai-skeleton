from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai42_kit.settings import TaiBaseSettings


class CoreSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TAI_MCP_",
    )
    # TAI_MANIFEST_PATH is the single manifest-location env var: it feeds the
    # CLI's --manifest-path default here and is what the config file manager
    # reads directly.
    manifest_path: str | None = Field(default=None, validation_alias="TAI_MANIFEST_PATH")
    backend: str | None = None
    template: str | None = None
    # Max seconds to spend on a single MCP server viability check (connect +
    # list_tools) during startup or reload. A server that exceeds this is
    # skipped and recorded instead of blocking the whole server.
    mcp_probe_timeout: float = 15.0
    # The failed-MCP re-probe backoff bounds. The lifespan-owned re-probe task
    # sleeps ``initial`` seconds between passes, doubling up to ``max`` after a
    # pass where every probed server stayed down, and resetting to ``initial``
    # whenever a server recovers or a new one fails.
    mcp_reprobe_initial_seconds: float = Field(default=30.0, gt=0)
    mcp_reprobe_max_seconds: float = Field(default=600.0, gt=0)


class AppArgsSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_ARGS_",
    )

    transport: str = "http"
    host: str = "127.0.0.1"
    port: int = 8000
    uds: str | None = None

    # uvicorn's graceful-shutdown budget (seconds): on SIGTERM uvicorn force-
    # completes in-flight requests within this window instead of waiting
    # indefinitely, so the lifespan teardown always runs. A shipped
    # ``--timeout-graceful-shutdown`` CLI extra-arg overrides this default. Must
    # be positive.
    timeout_graceful_shutdown: int = Field(default=10, gt=0)
