"""The ``CONVERSATIONS_*`` settings: the four keyspace helpers, the configurable
bounds and their defaults, and the backend-selection property."""

from __future__ import annotations

from tai42_skeleton.conversations.settings import ConversationsSettings


def test_default_bounds_match_the_design():
    s = ConversationsSettings()
    assert s.max_concurrent_turns == 10
    assert s.thread_queue_depth == 20
    assert s.per_address_turns_per_hour == 20
    assert s.sync_wait_max_seconds == 120
    assert s.delivery_max_attempts == 8
    assert s.delivery_grace_seconds == 3600
    assert s.inbound_dedupe_ttl_seconds >= 48 * 3600
    assert s.answer_retention_ttl_seconds == 30 * 86400
    assert s.max_message_chars == {"twilio": 1600, "telegram": 4096, "slack": 40000, "whatsapp-cloud": 4096}


def test_keyspace_helpers_are_distinct_greppable_segments():
    s = ConversationsSettings()
    # The FOUR keyspaces, each its own segment under the shared prefix; the provider /
    # outbound id sits LAST so a ``:`` in it cannot bleed across segments.
    assert s.dedupe_key("twilio", "SM:1") == "conversations:dedupe:twilio:SM:1"
    assert s.record_key("m-1") == "conversations:record:m-1"
    assert s.outbound_index_key("twilio", "OB:2") == "conversations:outbound:twilio:OB:2"
    assert s.route_key("support-line") == "conversations:route:support-line"
    assert s.route_names_key == "conversations:route_names"
    assert s.route_key_prefix == "conversations:route:"


def test_a_blank_provider_supplied_segment_is_refused():
    import pytest

    s = ConversationsSettings()
    # Every blank id builds the SAME key, so two messages would share one idempotency
    # marker; refused loudly instead.
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="provider_message_id must be a non-blank string"):
            s.dedupe_key("twilio", blank)
        with pytest.raises(ValueError, match="channel must be a non-blank string"):
            s.dedupe_key(blank, "SM1")
        with pytest.raises(ValueError, match="outbound_message_id must be a non-blank string"):
            s.outbound_index_key("twilio", blank)
        with pytest.raises(ValueError, match="channel must be a non-blank string"):
            s.outbound_index_key(blank, "OB1")


def test_a_colon_in_the_channel_segment_is_refused():
    import pytest

    s = ConversationsSettings()
    # A ``:`` in the channel would move the boundary — ('a', 'b:c') and ('a:b', 'c')
    # would collide. The trailing provider id is unconstrained (nothing follows it).
    for builder in (s.dedupe_key, s.outbound_index_key):
        with pytest.raises(ValueError, match="channel must not contain ':'"):
            builder("a:b", "c")


def test_the_delivery_backoff_schedule_spans_an_hour():
    from tai42_skeleton.conversations.delivery import _backoff_seconds

    # A receiver down for an hour is still retried before its answer fails. The waits
    # sit BETWEEN attempts, so there are ``delivery_max_attempts - 1`` of them.
    s = ConversationsSettings()
    waits = [_backoff_seconds(s, attempt) for attempt in range(1, s.delivery_max_attempts)]
    assert waits == [60, 120, 240, 480, 900, 900, 900]
    assert sum(waits) == 3600


def test_in_memory_is_true_without_a_redis_url():
    # No CONVERSATIONS_REDIS_URL configured -> no durable backend.
    assert ConversationsSettings().in_memory is True


def test_bounds_reject_non_positive(monkeypatch):
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("CONVERSATIONS_THREAD_QUEUE_DEPTH", "0")
    with pytest.raises(ValidationError):
        ConversationsSettings()


def test_a_non_positive_split_cap_is_refused_at_startup(monkeypatch):
    import pytest
    from pydantic import ValidationError

    # A cap no send can be split against is a config error at startup, not at send time
    # where it would strand the record it fails.
    monkeypatch.setenv("CONVERSATIONS_MAX_MESSAGE_CHARS", '{"twilio": 0}')
    with pytest.raises(ValidationError, match="must be positive"):
        ConversationsSettings()
