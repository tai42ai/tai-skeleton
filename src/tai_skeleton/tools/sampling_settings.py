"""``TAI_SAMPLING_*`` config for the platform-LLM sampling fallback budgets.

Co-located with the sampling bridge: a per-call token ceiling (the default
applied when a caller passes no ``max_tokens``; an over-ask is refused loudly,
never silently clamped) and a per-invocation ceiling on how many ``ctx.sample()``
calls one in-process tool invocation may make through the bridge. Read at call
time through the ``settings_cache`` accessor.
"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai_kit.settings import TaiBaseSettings, settings_cache


class SamplingSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAI_SAMPLING_")

    # Per-sample token ceiling on the platform-LLM fallback: applied as the
    # default when the caller passes no max_tokens; a caller asking for more is
    # refused loudly (never silently clamped). The default is deliberately
    # generous (model-appropriate long-generation headroom) so legitimate long
    # generations do not trip the refuse; operators tune it down. Must be positive.
    max_tokens_per_call: int = Field(default=32768, gt=0)

    # How many ctx.sample() calls one in-process tool invocation may make through
    # the bridge before it is refused loudly. Must be positive.
    max_calls_per_invocation: int = Field(default=20, gt=0)


@settings_cache
def sampling_settings() -> SamplingSettings:
    return SamplingSettings()
