"""The ``@operation`` decorator, the registry, and its duplicate-name guard."""

from __future__ import annotations

import pytest

from tai_skeleton.operations import OperationRegistry, operation
from tai_skeleton.operations.decorator import operation_metadata_of
from tai_skeleton.operations.errors import NotFoundError


def test_decorator_defaults_name_to_function_and_records_metadata():
    reg = OperationRegistry()

    @operation(summary="Do a thing", tags=["x"], registry=reg)
    async def do_thing(a: int) -> int:
        """Docstring."""
        return a

    meta = operation_metadata_of(do_thing)
    assert meta.name == "do_thing"
    assert meta.summary == "Do a thing"
    assert meta.tags == ("x",)
    assert meta.func is do_thing
    assert reg.get("do_thing") is meta
    assert reg.has("do_thing")


def test_decorator_carries_all_declared_metadata():
    reg = OperationRegistry()

    @operation(
        name="renamed",
        summary="s",
        tags=["t"],
        destructive=True,
        reload_gated=True,
        meta_executor=True,
        authority_changing=True,
        errors=[NotFoundError],
        registry=reg,
    )
    async def fn() -> None:
        return None

    meta = reg.get("renamed")
    assert meta.destructive is True
    assert meta.reload_gated is True
    assert meta.meta_executor is True
    assert meta.authority_changing is True
    assert meta.error_classes == (NotFoundError,)


def test_duplicate_operation_name_raises():
    reg = OperationRegistry()

    @operation(summary="first", tags=["x"], registry=reg)
    async def dup() -> None:
        return None

    with pytest.raises(ValueError, match="already registered"):

        @operation(name="dup", summary="second", tags=["x"], registry=reg)
        async def dup_again() -> None:
            return None


def test_registry_all_is_sorted_and_names_is_frozenset():
    reg = OperationRegistry()

    @operation(name="b", summary="s", tags=["x"], registry=reg)
    async def b() -> None:
        return None

    @operation(name="a", summary="s", tags=["x"], registry=reg)
    async def a() -> None:
        return None

    assert [m.name for m in reg.all()] == ["a", "b"]
    assert reg.names() == frozenset({"a", "b"})


def test_registry_get_unknown_raises_keyerror():
    reg = OperationRegistry()
    with pytest.raises(KeyError, match="not registered"):
        reg.get("nope")


def test_registry_clear_empties_it():
    reg = OperationRegistry()

    @operation(summary="s", tags=["x"], registry=reg)
    async def op() -> None:
        return None

    reg.clear()
    assert reg.names() == frozenset()


def test_operation_metadata_of_non_operation_raises():
    def plain() -> None:
        return None

    with pytest.raises(TypeError, match="not an @operation"):
        operation_metadata_of(plain)


def test_reregister_replays_a_stable_snapshot_without_reimporting_leaves():
    """A reload repopulates the cleared registry from the in-memory snapshot rather
    than re-importing the leaf modules — so the reload does no ``sys.modules`` churn
    and the re-added records keep their object identity, which is what lets a router's
    ``register_operation_route`` re-attach its template to the very record the
    projection reads."""
    import sys

    from tai_skeleton.operations import operation_leaf_modules, reregister_operations
    from tai_skeleton.operations.registry import operation_registry

    # Prime the snapshot: the first call re-imports the leaves; every later call replays.
    reregister_operations()
    surface = operation_registry.names()
    assert surface, "the skeleton leaf operations must register"

    records = {name: operation_registry.get(name) for name in surface}
    leaf_module_ids = {name: id(sys.modules[name]) for name in operation_leaf_modules() if name in sys.modules}
    assert leaf_module_ids, "the leaf modules must be imported to compare identity"

    # Simulate a reload: clear the registry, then repopulate.
    operation_registry.clear()
    reloaded = reregister_operations()

    # No leaf module was popped/re-imported — the replay path touches sys.modules for none.
    assert reloaded == []
    for name, module_id in leaf_module_ids.items():
        assert id(sys.modules[name]) == module_id

    # The full surface is back, each entry the SAME OperationMetadata object as before —
    # so the record the router decorates on re-import is the record now in the registry.
    assert operation_registry.names() == surface
    for name, record in records.items():
        assert operation_registry.get(name) is record
