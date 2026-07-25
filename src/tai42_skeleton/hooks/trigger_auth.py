"""How a fire door authenticates the caller that rings it — the trigger-auth axis.

* ``/universal_webhook/{topic}``: keyed by TOPIC, no token — ``"public"``, or
  ``"verifier"`` once the topic carries a webhook verifier binding.
* ``/trigger/{token}``: keyed by the LINK record — ``"token"``, or ``"token+api_key"``
  when the link was minted with ``require_api_key`` (an authenticated principal on top,
  plus the ordinary ``hooks``-tag level pass for a role-governed one).

Binding a verifier to a topic takes that topic's links out of service — their door
answers the uniform 404 while it stands — without touching a link record; the link axis
reports ``"out-of-service"``. A link's door auth binds that door alone, never the topic.

The value is DERIVED at every read, never stored: the verifier binding is independently
mutable and re-read live at each fire, so a stored copy would be staleable.
"""

from __future__ import annotations

from typing import Literal

from tai42_skeleton.access_control.settings import access_control_settings

# Closed vocabulary: a misspelt comparison is a type error, not a dead branch.
TriggerAuth = Literal["public", "verifier", "token", "token+api_key", "out-of-service"]


def webhook_trigger_auth(*, verifier_bound: bool) -> TriggerAuth:
    """The axis value of a topic's webhook ingress door, from whether that topic
    currently has a verifier binding."""
    return "verifier" if verifier_bound else "public"


def link_trigger_auth(*, require_api_key: bool, verifier_bound: bool) -> TriggerAuth:
    """The axis value of a trigger link's door, from the record's stored
    ``require_api_key`` and whether its topic currently carries a verifier binding.

    A verifier binding wins (the door admits nobody). The api-key requirement is reported
    only where the door can enforce it: with access control disabled the axis says
    ``"token"`` while the record keeps its stored requirement."""
    if verifier_bound:
        return "out-of-service"
    if require_api_key and access_control_settings().enable:
        return "token+api_key"
    return "token"
