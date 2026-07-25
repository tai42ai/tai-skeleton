"""The isolation layer's acting principal: the bound EXECUTION identity inside a background
fire, the request-scoped caller outside one.

``restricted_identity`` and ``request_identity`` resolve the principal the tool-dispatch
seam authorized the call against, so a fire is confined to the KEY it runs as. The execution
identity takes PRECEDENCE, not fallback: a Starlette ``BackgroundTask`` runs inside the
triggering request's contextvar context, so those vars are still the ringer's while the fire
runs. The claims bound below are exactly those ``build_execution_identity`` builds.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.context import reset_request_user_id, set_request_user_id

from tai42_skeleton.access_control.request_scopes import (
    reset_request_identity_claims,
    set_request_identity_claims,
)
from tai42_skeleton.access_control.user import (
    CrossIdentityAudienceError,
    clamp_write_audience,
    request_identity,
    restricted_identity,
)
from tai42_skeleton.authz.execution_identity import reset_execution_identity, set_execution_identity
from tai42_skeleton.authz.identity import CallerIdentity


@contextmanager
def _fire(identity: CallerIdentity) -> Iterator[None]:
    """Bind ``identity`` as the current fire's execution identity, released on exit."""
    token = set_execution_identity(identity)
    try:
        yield
    finally:
        reset_execution_identity(token)


@contextmanager
def _triggering_caller(own_id: str, owner: str | None) -> Iterator[None]:
    """Bind a request-scope caller — whose live contextvars a ``BackgroundTask`` fire
    inherits. ``owner`` is its owner claim; ``None`` means ownerless and unrestricted."""
    claims = {OWNER_USER_ID_CLAIM: owner} if owner is not None else {"src": "stub"}
    claims_token = set_request_identity_claims(claims)
    uid_token = set_request_user_id(own_id)
    try:
        yield
    finally:
        reset_request_user_id(uid_token)
        reset_request_identity_claims(claims_token)


def _owned_key(key: str = "k-fire", owner: str = "alice") -> CallerIdentity:
    return CallerIdentity(user_id=key, effective_scopes=("notifications",), claims={OWNER_USER_ID_CLAIM: owner})


def _ownerless_key(key: str = "k-fire") -> CallerIdentity:
    return CallerIdentity(user_id=key, effective_scopes=("notifications",), claims={})


# -- a fire is isolated to the key it runs as ---------------------------------


def test_owned_execution_key_is_restricted_to_itself() -> None:
    # The owner claim MARKS the fire restricted, but it is confined to the KEY's own id.
    with _fire(_owned_key(owner="alice")):
        assert restricted_identity() == "k-fire"
        assert request_identity() == ("k-fire", "k-fire")


def test_ownerless_execution_key_is_unrestricted_and_attributed_to_itself() -> None:
    # Unrestricted, but its writes are still attributed to the KEY.
    with _fire(_ownerless_key()):
        assert restricted_identity() is None
        assert request_identity() == ("k-fire", None)


def test_gate_off_execution_identity_is_unrestricted() -> None:
    # With the gate off the identity carries the key alone, so there is nothing to restrict.
    with _fire(CallerIdentity(user_id="k-fire")):
        assert restricted_identity() is None
        assert request_identity() == ("k-fire", None)


# -- precedence over an inherited request context (the BackgroundTask door) ----


def test_owned_fire_isolates_as_the_key_not_the_triggering_caller() -> None:
    # A foreign caller rang the trigger door, so its request-scope vars are still bound in
    # the inherited context; a fallback would write the fire's records into that caller's island.
    with _triggering_caller("k-bob", owner="bob"), _fire(_owned_key(owner="alice")):
        assert restricted_identity() == "k-fire"
        assert request_identity() == ("k-fire", "k-fire")


def test_ownerless_fire_is_unrestricted_even_under_a_restricted_triggering_caller() -> None:
    # Precedence, not fallback: the inherited context is a RESTRICTED caller's, yet the
    # ownerless fire stays unrestricted and attributed to the key.
    with _triggering_caller("k-bob", owner="bob"), _fire(_ownerless_key()):
        assert restricted_identity() is None
        assert request_identity() == ("k-fire", None)


def test_owned_fire_isolates_as_the_key_under_an_unrestricted_triggering_caller() -> None:
    # The mirror: an unrestricted ringer must not open the owned fire's island either.
    with _triggering_caller("k-bob", owner=None), _fire(_owned_key(owner="alice")):
        assert restricted_identity() == "k-fire"
        assert request_identity() == ("k-fire", "k-fire")


# -- the write clamp under a fire ---------------------------------------------


def test_write_clamp_scopes_an_owned_fire_to_its_own_slice() -> None:
    with _triggering_caller("k-bob", owner="bob"), _fire(_owned_key(owner="alice")):
        # An unset audience is scoped to the KEY's own slice.
        assert clamp_write_audience(None) == "k-fire"
        assert clamp_write_audience("k-fire") == "k-fire"
        # Every other identity, owner and ringer included, is a cross-identity inject.
        for foreign in ("alice", "k-bob", "victim"):
            with pytest.raises(CrossIdentityAudienceError):
                clamp_write_audience(foreign)


def test_write_clamp_leaves_an_ownerless_fire_unclamped() -> None:
    with _fire(_ownerless_key()):
        assert clamp_write_audience("victim") == "victim"
        assert clamp_write_audience(None) is None


# -- the invariant breach, on both resolution paths ---------------------------


def test_owner_claim_with_no_key_id_raises() -> None:
    # An owner claim with no ``user_id`` is constructible, and confining it to ``None``
    # would silently open the full view to a restricted fire.
    with _fire(CallerIdentity(claims={OWNER_USER_ID_CLAIM: "alice"})), pytest.raises(RuntimeError):
        restricted_identity()


def test_owner_claim_with_no_key_id_raises_even_under_a_bound_triggering_caller() -> None:
    # Not papered over by the inherited context: falling through to the ringer's id would
    # confine the fire to a stranger instead of raising.
    with (
        _triggering_caller("k-bob", owner="bob"),
        _fire(CallerIdentity(claims={OWNER_USER_ID_CLAIM: "alice"})),
        pytest.raises(RuntimeError),
    ):
        restricted_identity()


# -- regression: nothing changes outside a fire -------------------------------


def test_no_execution_identity_resolves_the_request_caller() -> None:
    # With no fire bound the request-scope caller is the acting principal, unchanged.
    with _triggering_caller("k-bob", owner="bob"):
        assert restricted_identity() == "k-bob"
        assert request_identity() == ("k-bob", "k-bob")
        assert clamp_write_audience(None) == "k-bob"
        with pytest.raises(CrossIdentityAudienceError):
            clamp_write_audience("victim")

    with _triggering_caller("k-bob", owner=None):
        assert restricted_identity() is None
        assert request_identity() == ("k-bob", None)
        assert clamp_write_audience("victim") == "victim"


def test_no_principal_bound_at_all_is_unrestricted() -> None:
    assert restricted_identity() is None
    assert request_identity() == (None, None)
    assert clamp_write_audience("victim") == "victim"


def test_a_released_fire_leaves_the_request_caller_intact() -> None:
    # The precedence lasts exactly as long as the binding.
    with _triggering_caller("k-bob", owner="bob"):
        with _fire(_ownerless_key()):
            assert restricted_identity() is None
        assert restricted_identity() == "k-bob"
