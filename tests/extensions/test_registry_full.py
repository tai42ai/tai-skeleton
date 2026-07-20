"""Extra ``ExtensionRegistry`` coverage: the passing ``validation`` branch (no
missing requested extension), ``get_kind``/``missing_extensions``, and the
``kind_iterator`` taxonomy source.
"""

from __future__ import annotations

import pytest
from tai_contract.extensions import ExtensionKind

from tai_skeleton.exceptions.exceptions import TaiValidationError
from tai_skeleton.extensions import ExtensionRegistry
from tai_skeleton.extensions.registry import extension_config, extension_name


def _noop(func, name, desc, config=None):
    return func


def test_register_extension_duplicate_name_raises():
    reg = ExtensionRegistry(requested_extensions=frozenset())
    reg.register_extension(_noop, ExtensionKind.WRAPPER, "dup")
    with pytest.raises(TaiValidationError, match="already registered"):
        reg.register_extension(_noop, ExtensionKind.WRAPPER, "dup")


def test_get_extension_unknown_raises():
    reg = ExtensionRegistry(requested_extensions=frozenset())
    with pytest.raises(TaiValidationError, match="not registered"):
        reg.get_extension("ghost")


def test_get_kind_unknown_raises():
    reg = ExtensionRegistry(requested_extensions=frozenset())
    with pytest.raises(TaiValidationError, match="not registered"):
        reg.get_kind("ghost")


def test_requires_body_locality_stored_and_defaults_false():
    reg = ExtensionRegistry(requested_extensions=frozenset())
    reg.register_extension(_noop, ExtensionKind.WRAPPER, "plain")
    reg.register_extension(_noop, ExtensionKind.WRAPPER, "local", requires_body_locality=True)
    assert reg.requires_body_locality("plain") is False
    assert reg.requires_body_locality("local") is True
    # The dict combo-element form keys on its name, like the other lookups.
    assert reg.requires_body_locality({"name": "local", "config": {}}) is True


def test_requires_body_locality_unknown_raises():
    reg = ExtensionRegistry(requested_extensions=frozenset())
    with pytest.raises(TaiValidationError, match="not registered"):
        reg.requires_body_locality("ghost")


def test_extension_decorator_threads_requires_body_locality():
    reg = ExtensionRegistry(requested_extensions=frozenset())

    @reg.extension(kind=ExtensionKind.WRAPPER, requires_body_locality=True)
    def localdeco(func, name, desc, config=None):
        return func

    assert reg.requires_body_locality("localdeco") is True


def test_validate_unknown_name_raises():
    reg = ExtensionRegistry(requested_extensions=frozenset())
    with pytest.raises(TaiValidationError, match="ghost"):
        reg.validate(["ghost"])


def test_validation_passes_when_no_missing():
    # The requested extension is registered, so ``missing_extensions`` is empty
    # and ``validation`` returns without raising (the ``if missing`` false path).
    reg = ExtensionRegistry(requested_extensions=frozenset({"monitor"}))
    reg.register_extension(_noop, ExtensionKind.WRAPPER, "monitor")
    assert reg.missing_extensions() == set()
    reg.validation()  # no raise


def test_validation_passes_with_empty_request():
    reg = ExtensionRegistry(requested_extensions=frozenset())
    reg.validation()  # nothing requested -> nothing missing


def test_missing_extensions_reports_unregistered():
    reg = ExtensionRegistry(requested_extensions=frozenset({"monitor", "ghost"}))
    reg.register_extension(_noop, ExtensionKind.WRAPPER, "monitor")
    assert reg.missing_extensions() == {"ghost"}


def test_kind_iterator_yields_every_kind():
    assert set(ExtensionRegistry.kind_iterator()) == set(ExtensionKind)


def test_validate_passes_for_single_backend():
    # A single BACKEND (the single-only kind) is legal — exercises the
    # ``validate`` loop with the cardinality check satisfied.
    reg = ExtensionRegistry(requested_extensions=frozenset())
    reg.register_extension(_noop, ExtensionKind.BACKEND, "async_task")
    reg.validate(["async_task"])


# -- combo elements: bare name vs {"name", "config"} mapping ------------------


def test_element_helpers_extract_name_and_config():
    assert extension_name("monitor") == "monitor"
    assert extension_config("monitor") == {}
    element = {"name": "ask_external", "config": {"verifier": {"name": "github"}}}
    assert extension_name(element) == "ask_external"
    assert extension_config(element) == {"verifier": {"name": "github"}}


def test_registry_lookups_key_on_dict_element_name():
    # ``validate`` / ``get_extension`` / ``get_kind`` all resolve a combo element's
    # NAME, so a ``{"name", "config"}`` mapping keys the registry identically to
    # the bare name.
    reg = ExtensionRegistry(requested_extensions=frozenset())
    reg.register_extension(_noop, ExtensionKind.TRANSFORMER, "ask_external")
    element = {"name": "ask_external", "config": {"verifier": {"name": "github"}}}
    reg.validate([element])  # no raise
    assert reg.get_extension(element) is _noop
    assert reg.get_kind(element) is ExtensionKind.TRANSFORMER


def test_validate_rejects_unknown_dict_element_name():
    reg = ExtensionRegistry(requested_extensions=frozenset())
    with pytest.raises(TaiValidationError, match="ghost"):
        reg.validate([{"name": "ghost", "config": {}}])
