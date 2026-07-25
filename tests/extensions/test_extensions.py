"""Extension subsystem tests.

Taxonomy: the three kinds are ``WRAPPER`` / ``TRANSFORMER`` / ``BACKEND`` and
carry the ``multiple`` cardinality flag (WRAPPER/TRANSFORMER stackable, BACKEND
single).
Registry: a Wrapper / Transformer / Backend can each be registered and read back;
the ``multiple`` cardinality rule rejects a duplicate of a single-only (BACKEND)
kind and allows duplicates of a stackable kind.
"""

from __future__ import annotations

import pytest
from tai42_contract.extensions import ExtensionKind

from tai42_skeleton.exceptions.exceptions import TaiValidationError
from tai42_skeleton.extensions import ExtensionRegistry


def _noop(func, name, desc):
    return func


# --- taxonomy -------------------------------------------------------------


def test_kinds_are_wrapper_transformer_backend():
    assert {kind.name for kind in ExtensionKind} == {"WRAPPER", "TRANSFORMER", "BACKEND"}


def test_cardinality_flags():
    # Stackable kinds allow many; BACKEND is the single-only kind.
    assert ExtensionKind.WRAPPER.multiple is True
    assert ExtensionKind.TRANSFORMER.multiple is True
    assert ExtensionKind.BACKEND.multiple is False


def test_dead_helpers_dropped():
    for attr in ("tag", "is_backend", "is_decorator", "is_flow"):
        assert not hasattr(ExtensionKind.BACKEND, attr)


def test_old_member_names_gone():
    for old in ("DECORATOR", "FLOW"):
        with pytest.raises(KeyError):
            ExtensionKind[old]


# --- registry -------------------------------------------------------------


def _registry():
    reg = ExtensionRegistry(requested_extensions=frozenset())
    reg.register_extension(_noop, ExtensionKind.WRAPPER, "monitor")
    reg.register_extension(_noop, ExtensionKind.TRANSFORMER, "chain")
    reg.register_extension(_noop, ExtensionKind.BACKEND, "async_task")
    return reg


def test_register_and_read_back_each_kind():
    reg = _registry()
    assert reg.get_kind("monitor") is ExtensionKind.WRAPPER
    assert reg.get_kind("chain") is ExtensionKind.TRANSFORMER
    assert reg.get_kind("async_task") is ExtensionKind.BACKEND
    assert reg.get_extension("monitor") is _noop


def test_available_extensions_exposes_kind():
    reg = _registry()
    # ``kind`` serializes the enum VALUE (lowercase), the ONLY form that round-
    # trips back through ``ExtensionKind(...)`` — the member name ("BACKEND")
    # would not.
    available = reg.available_extensions()
    assert available == [
        {"name": "async_task", "kind": "backend"},
        {"name": "chain", "kind": "transformer"},
        {"name": "monitor", "kind": "wrapper"},
    ]
    # Every serialized kind reconstructs the original ExtensionKind.
    assert [ExtensionKind(entry["kind"]) for entry in available] == [
        ExtensionKind.BACKEND,
        ExtensionKind.TRANSFORMER,
        ExtensionKind.WRAPPER,
    ]


def test_validate_rejects_duplicate_backend():
    reg = _registry()
    reg.register_extension(_noop, ExtensionKind.BACKEND, "sync_task")
    with pytest.raises(TaiValidationError):
        reg.validate(["async_task", "sync_task"])


def test_validate_allows_stacked_wrappers_and_transformers():
    reg = _registry()
    reg.register_extension(_noop, ExtensionKind.WRAPPER, "cache")
    reg.register_extension(_noop, ExtensionKind.TRANSFORMER, "batch")
    # Two wrappers + two transformers + one backend is legal.
    reg.validate(["monitor", "cache", "chain", "batch", "async_task"])


def test_validation_raises_on_missing_requested():
    reg = ExtensionRegistry(requested_extensions=frozenset({"monitor", "ghost"}))
    reg.register_extension(_noop, ExtensionKind.WRAPPER, "monitor")
    with pytest.raises(TaiValidationError):
        reg.validation()
