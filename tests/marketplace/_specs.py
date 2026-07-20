"""Builders for a valid :class:`PluginSpec` and a resolve response.

The installer, the manifest patch, and the stores all consume a validated
``PluginSpec`` and the registry's resolve-response dict; these helpers produce
minimal-but-valid instances so a test states only the fields it cares about.
"""

from __future__ import annotations

from typing import Any

from tai42_contract.plugins import PluginSpec

# The tai42-contract version installed in this environment satisfies this range, so
# a resolve built with it passes the installer's contract check by default.
DEFAULT_CONTRACT_RANGE = ">=0.1,<1.0"


def tool_item(module: str = "tai42_toolbox.tools.gen_uuid", name: str = "gen-uuid") -> dict[str, Any]:
    """A ``tool``-kind provides item."""
    return {"kind": "tool", "name": name, "module": module, "description": "Generate a UUID"}


def make_spec(
    *,
    namespace: str = "tai42",
    name: str = "toolbox",
    package: str = "tai42-toolbox",
    version: str = "1.0.0",
    provides: list[dict[str, Any]] | None = None,
    contract: str = DEFAULT_CONTRACT_RANGE,
) -> PluginSpec:
    """A valid :class:`PluginSpec` with one tool item unless ``provides`` is given."""
    return PluginSpec.model_validate(
        {
            "spec_version": 1,
            "namespace": namespace,
            "name": name,
            "package": package,
            "version": version,
            "description": "A test plugin",
            "license": "Apache-2.0",
            "contract": contract,
            "categories": ["dev"],
            "provides": provides if provides is not None else [tool_item()],
        }
    )


def make_resolved(
    spec: PluginSpec,
    *,
    source: str = "pypi",
    version: str | None = None,
    repository_url: str | None = None,
    tag: str | None = None,
    artifact_ref: str | None = None,
    sha256: str | None = None,
    advisories: list[dict[str, Any]] | None = None,
    contract_range: str = DEFAULT_CONTRACT_RANGE,
) -> dict[str, Any]:
    """The registry resolve-response dict the installer consumes.

    For a github source, ``artifact_ref`` defaults to a plausible codeload
    tag-tarball URL and ``sha256`` to a valid hex digest so the resolve passes the
    installer's github-provenance presence check; a test asserting the verified
    fetch fakes the download and can override either.
    """
    pinned = version if version is not None else spec.version
    default_artifact = f"https://codeload.github.com/tai42ai/{spec.name}/tar.gz/refs/tags/v{pinned}"
    return {
        "ref": spec.ref,
        "package": spec.package,
        "source": source,
        "repository_url": repository_url,
        "version": pinned,
        "tag": tag,
        "artifact_ref": artifact_ref if artifact_ref is not None else default_artifact,
        "sha256": sha256 if sha256 is not None else "0" * 64,
        "contract_range": contract_range,
        "spec": spec.model_dump(mode="json"),
        "advisories": advisories if advisories is not None else [],
    }
