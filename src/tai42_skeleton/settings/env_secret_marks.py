"""Operator-marked secret env keys — a registered settings group.

The env store holds arbitrary operator-set keys the platform does not own (no
settings class declares them), so nothing knows those keys are sensitive. This
group carries the operator's own "treat these env keys as secret" marks: a
comma-separated list of env key NAMES under ``TAI_ENV_SECRET_KEYS``. It is a
``TaiBaseSettings`` subclass so the marks (a) surface in the settings-schema
view like any other group and (b) live in the env store, so config backups carry
them. Masking driven by these marks is display-side (Studio); the marks
themselves are plain data.
"""

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import NoDecode
from tai42_kit.settings import TaiBaseSettings, settings_cache


class EnvSecretMarksSettings(TaiBaseSettings):
    # Names of env keys the operator marked secret. ``NoDecode`` disables
    # pydantic-settings' JSON decode for this complex field so the raw
    # comma-separated env string reaches the ``mode="before"`` validator, which
    # splits it into a list.
    secret_keys: Annotated[list[str], NoDecode] = Field(default_factory=list, validation_alias="TAI_ENV_SECRET_KEYS")

    @field_validator("secret_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept a comma-separated string (env values are strings), trimming
        whitespace and dropping empty segments; pass non-strings through."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@settings_cache
def env_secret_marks_settings() -> EnvSecretMarksSettings:
    return EnvSecretMarksSettings()
