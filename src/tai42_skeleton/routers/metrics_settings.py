import os
import tempfile

from pydantic import field_validator
from tai42_kit.settings import TaiBaseSettings, settings_cache


class MetricsSettings(TaiBaseSettings):
    backend_metrics_host: str = "127.0.0.1"
    backend_metrics_port: int = 8012
    # A fixed, CWD-independent absolute path so every process in the run family
    # (mcp_app + backend worker + metrics server) resolves the SAME multiproc dir
    # regardless of the directory each was launched from. Honors ``TMPDIR`` (host
    # env, not per-process CWD). Overridable via ``PROMETHEUS_MULTIPROC_DIR``; an
    # override MUST be absolute (validated below).
    prometheus_multiproc_dir: str = os.path.join(tempfile.gettempdir(), "tai42_prometheus")

    @field_validator("prometheus_multiproc_dir")
    @classmethod
    def _multiproc_dir_must_be_absolute(cls, value: str) -> str:
        # A relative dir resolves against each process's CWD, splitting the run
        # family into disjoint dirs and silently emptying the scrape. Refuse it
        # loudly rather than normalize it (normalizing against this process's CWD
        # would re-introduce the split under a different guise).
        if not os.path.isabs(value):
            raise ValueError(
                f"prometheus_multiproc_dir must be an absolute path, got {value!r}; "
                "a relative dir resolves against each process's working directory and "
                "splits the shared multiproc dir per process."
            )
        return value


@settings_cache
def metrics_settings() -> MetricsSettings:
    return MetricsSettings()


def activate_multiproc_env() -> str:
    """Publish the multiproc dir to ``PROMETHEUS_MULTIPROC_DIR`` in ``os.environ``.

    ``prometheus_client`` freezes its value backend (mmap vs in-process mutex) the
    first time it is imported, choosing the multiprocess mmap backend only when
    this env var is already set. Writer entry points call this BEFORE anything
    imports ``prometheus_client`` so tool counters are recorded to the shared dir;
    spawned uvicorn workers and the in-process stdio/debug writers inherit it. The
    value is the validated (absolute) setting, so operators and forked workers all
    agree on one dir.
    """
    metrics_dir = metrics_settings().prometheus_multiproc_dir
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = metrics_dir
    return metrics_dir
