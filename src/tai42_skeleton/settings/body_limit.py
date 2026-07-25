"""``TAI_BODY_LIMIT_*`` config for the app-level request body-size cap.

The cap is an app middleware applied to every route (a backstop behind the public
doors' own smaller readers): a request body larger than the cap is answered with
413, loudly, never truncated. The bound is on ACTUAL bytes read, never a
client-declared ``Content-Length``. Read at call time.
"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai42_kit.settings import TaiBaseSettings, settings_cache


class BodyLimitSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAI_BODY_LIMIT_")

    # Cap (actual bytes, never client Content-Length) on any request body.
    # Oversized -> 413, loudly - never truncated. Must be positive.
    max_body_bytes: int = Field(default=10 * 1024 * 1024, gt=0)


@settings_cache
def body_limit_settings() -> BodyLimitSettings:
    return BodyLimitSettings()
