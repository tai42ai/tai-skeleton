"""The pure manifest-patch functions, one behaviour per test, across every kind.

``collisions``/``apply_provides``/``remove_provides`` are side-effect-free over a
plain manifest dict, so each patch shape (config row, module list, package list,
scalar slot, env-selected no-op) is checked in isolation, plus the convergence and
collision semantics and the final ``Manifest.model_validate`` acceptance.
"""

from __future__ import annotations

import pytest

from tai_skeleton.manifest import Manifest
from tai_skeleton.marketplace import manifest_patch
from tai_skeleton.marketplace.errors import ManifestBindingError, ManifestCollisionError
from tai_skeleton.marketplace.manifest_patch import apply_provides, collisions, remove_provides
from tests.marketplace._specs import make_spec


def _item(kind: str, name: str, module: str) -> dict:
    return {"kind": kind, "name": name, "module": module, "description": "d"}


# -- apply per kind ----------------------------------------------------------


def test_apply_tool_adds_a_config_row_titled_by_module() -> None:
    spec = make_spec(provides=[_item("tool", "gen-uuid", "pkg.tools.uuid")])
    manifest: dict = {}
    apply_provides(manifest, spec)
    assert manifest["tools"] == [{"title": "pkg.tools.uuid", "module": "pkg.tools.uuid"}]


def test_apply_agent_adds_a_config_row() -> None:
    spec = make_spec(provides=[_item("agent", "helper", "pkg.agents.helper")])
    manifest: dict = {}
    apply_provides(manifest, spec)
    assert manifest["agents"] == [{"title": "pkg.agents.helper", "module": "pkg.agents.helper"}]


def test_apply_tools_sharing_a_module_coalesce_to_one_entry() -> None:
    spec = make_spec(
        provides=[
            _item("tool", "gen-uuid", "pkg.tools.multi"),
            _item("tool", "gen-ulid", "pkg.tools.multi"),
        ]
    )
    manifest: dict = {}
    apply_provides(manifest, spec)
    assert manifest["tools"] == [{"title": "pkg.tools.multi", "module": "pkg.tools.multi"}]


def test_apply_extension_appends_to_module_list() -> None:
    spec = make_spec(provides=[_item("extension", "ext", "pkg.ext")])
    manifest: dict = {}
    apply_provides(manifest, spec)
    assert manifest["extensions_modules"] == ["pkg.ext"]


def test_apply_channel_appends_to_channel_modules() -> None:
    spec = make_spec(provides=[_item("channel", "chan", "pkg.chan")])
    manifest: dict = {}
    apply_provides(manifest, spec)
    assert manifest["channel_modules"] == ["pkg.chan"]


def test_apply_connector_and_identity_target_lifecycle_modules() -> None:
    spec = make_spec(
        provides=[
            _item("connector", "conn", "pkg.conn"),
            _item("identity", "idp", "pkg.idp"),
        ]
    )
    manifest: dict = {}
    apply_provides(manifest, spec)
    assert manifest["lifecycle_modules"] == ["pkg.conn", "pkg.idp"]


def test_apply_webhook_verifier_appends_to_its_module_list() -> None:
    spec = make_spec(provides=[_item("webhook-verifier", "wv", "pkg.wv")])
    manifest: dict = {}
    apply_provides(manifest, spec)
    assert manifest["webhook_verifier_modules"] == ["pkg.wv"]


def test_apply_studio_plugin_appends_the_package_name_not_the_module() -> None:
    spec = make_spec(package="tai-studio-ext", provides=[_item("studio-plugin", "sp", "pkg.studio.entry")])
    manifest: dict = {}
    apply_provides(manifest, spec)
    # package_list stores the DISTRIBUTION name, never the item's module.
    assert manifest["studio_plugins"] == ["tai-studio-ext"]


def test_apply_scalar_backend_sets_the_slot() -> None:
    spec = make_spec(provides=[_item("backend", "be", "pkg.backend")])
    manifest: dict = {}
    apply_provides(manifest, spec)
    assert manifest["backend_module"] == "pkg.backend"


@pytest.mark.parametrize(("kind", "field"), [("storage", "storage_module"), ("monitoring", "monitoring_module")])
def test_apply_scalar_storage_and_monitoring(kind: str, field: str) -> None:
    spec = make_spec(provides=[_item(kind, "x", "pkg.mod")])
    manifest: dict = {}
    apply_provides(manifest, spec)
    assert manifest[field] == "pkg.mod"


# -- collisions per field shape ----------------------------------------------


def test_config_row_collides_on_existing_module() -> None:
    spec = make_spec(provides=[_item("tool", "gen-uuid", "pkg.tools.uuid")])
    manifest = {"tools": [{"title": "other", "module": "pkg.tools.uuid"}]}
    found = collisions(manifest, spec)
    assert found == ["tools entry with module 'pkg.tools.uuid' already exists"]


def test_config_row_collides_on_existing_title() -> None:
    spec = make_spec(provides=[_item("tool", "gen-uuid", "pkg.tools.uuid")])
    # An existing entry whose TITLE equals the incoming module is a collision too.
    manifest = {"tools": [{"title": "pkg.tools.uuid", "module": "something.else"}]}
    assert collisions(manifest, spec) == ["tools entry with module 'pkg.tools.uuid' already exists"]


def test_module_list_collides_on_exact_string() -> None:
    spec = make_spec(provides=[_item("extension", "ext", "pkg.ext")])
    manifest = {"extensions_modules": ["pkg.ext"]}
    assert collisions(manifest, spec) == ["extensions_modules already contains 'pkg.ext'"]


def test_scalar_collides_when_slot_is_truthy() -> None:
    spec = make_spec(provides=[_item("backend", "be", "pkg.backend")])
    manifest = {"backend_module": "existing.backend"}
    found = collisions(manifest, spec)
    assert found == ["backend_module is already set to 'existing.backend' (cannot install 'pkg.backend')"]


def test_empty_scalar_slot_is_not_a_collision() -> None:
    # The example manifest uses "" for an unset scalar slot; "" is falsy, so no clash.
    spec = make_spec(provides=[_item("backend", "be", "pkg.backend")])
    assert collisions({"backend_module": ""}, spec) == []


def test_scalar_self_conflict_two_distinct_modules_one_slot_is_a_collision() -> None:
    # One spec providing two distinct backend modules for the single backend slot
    # is a self-conflict: a last-write-wins apply would silently drop the first.
    spec = make_spec(
        provides=[
            _item("backend", "be-a", "pkg.backend.a"),
            _item("backend", "be-b", "pkg.backend.b"),
        ]
    )
    found = collisions({}, spec)
    assert found == ["backend_module is a single-module slot but this plugin provides 'pkg.backend.a', 'pkg.backend.b'"]


def test_scalar_self_conflict_blocks_apply() -> None:
    spec = make_spec(
        provides=[
            _item("backend", "be-a", "pkg.backend.a"),
            _item("backend", "be-b", "pkg.backend.b"),
        ]
    )
    manifest: dict = {}
    with pytest.raises(ManifestCollisionError, match="single-module slot"):
        apply_provides(manifest, spec)
    # Nothing was written — no silent last-write-wins drop.
    assert "backend_module" not in manifest


def test_apply_raises_listing_every_collision() -> None:
    spec = make_spec(
        provides=[
            _item("tool", "gen-uuid", "pkg.tools.uuid"),
            _item("extension", "ext", "pkg.ext"),
        ]
    )
    manifest = {"tools": [{"title": "t", "module": "pkg.tools.uuid"}], "extensions_modules": ["pkg.ext"]}
    with pytest.raises(ManifestCollisionError) as exc:
        apply_provides(manifest, spec)
    assert "pkg.tools.uuid" in str(exc.value)
    assert "pkg.ext" in str(exc.value)


# -- remove semantics --------------------------------------------------------


def test_remove_drops_exactly_the_specs_references() -> None:
    spec = make_spec(provides=[_item("tool", "gen-uuid", "pkg.tools.uuid")])
    manifest = {"tools": [{"title": "pkg.tools.uuid", "module": "pkg.tools.uuid"}, {"title": "keep", "module": "keep"}]}
    changed = remove_provides(manifest, spec)
    assert changed is True
    assert manifest["tools"] == [{"title": "keep", "module": "keep"}]


def test_remove_is_convergent_second_run_returns_false() -> None:
    spec = make_spec(provides=[_item("extension", "ext", "pkg.ext")])
    manifest = {"extensions_modules": ["pkg.ext"]}
    assert remove_provides(manifest, spec) is True
    assert remove_provides(manifest, spec) is False


def test_remove_drops_a_tools_entry_by_module_even_when_title_was_edited() -> None:
    spec = make_spec(provides=[_item("tool", "gen-uuid", "pkg.tools.uuid")])
    # An operator renamed the title; removal still matches on module (a leftover
    # entry would brick the next boot after the pip uninstall).
    manifest = {"tools": [{"title": "operator-renamed", "module": "pkg.tools.uuid"}]}
    assert remove_provides(manifest, spec) is True
    assert manifest["tools"] == []


def test_remove_leaves_a_foreign_scalar_value_alone() -> None:
    spec = make_spec(provides=[_item("backend", "be", "pkg.backend")])
    # The operator replaced the slot with a different module; removal must not clear it.
    manifest = {"backend_module": "operator.replacement"}
    assert remove_provides(manifest, spec) is False
    assert manifest["backend_module"] == "operator.replacement"


def test_remove_clears_a_scalar_still_holding_the_specs_module() -> None:
    spec = make_spec(provides=[_item("backend", "be", "pkg.backend")])
    manifest = {"backend_module": "pkg.backend"}
    assert remove_provides(manifest, spec) is True
    assert manifest["backend_module"] is None


# -- env-selected config: documented no-op across all three ------------------


def test_config_kind_is_a_no_op_everywhere() -> None:
    spec = make_spec(provides=[_item("config", "k8s", "pkg.config.k8s")])
    manifest: dict = {}
    assert collisions(manifest, spec) == []
    apply_provides(manifest, spec)
    assert manifest == {}  # nothing applied — pip install/uninstall is the registration
    assert remove_provides(manifest, spec) is False


# -- unknown kind ------------------------------------------------------------


def test_unknown_kind_raises_binding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A kind with no binding is contract drift; the mapping is patched to drop the
    # tool binding so the loud raise is exercised rather than silently skipped. The
    # typed error maps to a server-side 500 at the operation boundary, never a 400.
    spec = make_spec(provides=[_item("tool", "gen-uuid", "pkg.tools.uuid")])
    monkeypatch.setattr(manifest_patch, "KIND_MANIFEST_BINDINGS", {})
    with pytest.raises(ManifestBindingError, match="no manifest binding"):
        collisions({}, spec)


# -- the patched dict validates ----------------------------------------------


def test_patched_manifest_validates() -> None:
    spec = make_spec(
        provides=[
            _item("tool", "gen-uuid", "pkg.tools.uuid"),
            _item("extension", "ext", "pkg.ext"),
            _item("backend", "be", "pkg.backend"),
        ]
    )
    manifest: dict = {}
    apply_provides(manifest, spec)
    # A malformed compose would raise here; the patched dict must be a valid Manifest.
    Manifest.model_validate(manifest)
