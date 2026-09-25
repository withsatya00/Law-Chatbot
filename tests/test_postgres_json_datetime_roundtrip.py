"""Regression tests for BUG-102 (QA session 2026-09-24): `load_payload` must
reconstruct `created_at`/`updated_at` as real `datetime` objects, not leave
them as the ISO strings `dump_payload` serializes them to. Three independent
call sites broke on this exact round-trip before the fix:
  - app/drafting/engine.py::history() -- GET/POST /draft-history 500'd
    (fixed directly in an earlier session, "BUG-02")
  - app/api/drafting.py::list_draft_versions -- GET /draft/{id}/versions
    500'd (found this QA session, "BUG-102")
  - ConversationMemoryRepository.upsert_by_session -- every write after the
    first silently failed with an asyncpg type error, logged as
    `long_term_memory_write_failed` (found under BUG-101's concurrency test)
"""

from datetime import UTC, datetime

from app.repositories.postgres_json import dump_payload, load_payload


def test_load_payload_restores_created_at_and_updated_at_as_datetime():
    now = datetime(2026, 9, 24, 6, 30, 16, 59660, tzinfo=UTC)
    document = {"_id": "draft-1", "draft_type": "cheque_bounce_notice", "created_at": now, "updated_at": now}
    round_tripped = load_payload(dump_payload(document))
    assert isinstance(round_tripped["created_at"], datetime)
    assert isinstance(round_tripped["updated_at"], datetime)
    assert round_tripped["created_at"] == now
    assert round_tripped["updated_at"] == now


def test_load_payload_restored_datetime_supports_isoformat():
    """The exact failure mode this bug produced: calling `.isoformat()` on
    what should be a `datetime` -- `AttributeError: 'str' object has no
    attribute 'isoformat'` -- must no longer be possible after a round-trip."""
    now = datetime.now(UTC)
    round_tripped = load_payload(dump_payload({"created_at": now}))
    assert round_tripped["created_at"].isoformat()  # would raise AttributeError pre-fix


def test_load_payload_leaves_non_datetime_string_fields_untouched():
    document = {"draft_type": "cheque_bounce_notice", "language": "english", "status": "preview_ready"}
    round_tripped = load_payload(dump_payload(document))
    assert round_tripped == document


def test_load_payload_leaves_non_iso_created_at_string_unparsed_rather_than_raising():
    """Defensive: a malformed/legacy value must not crash the read path --
    left as-is, same as before this fix, rather than a new failure mode."""
    round_tripped = load_payload({"created_at": "not-a-real-timestamp"})
    assert round_tripped["created_at"] == "not-a-real-timestamp"


def test_load_payload_still_accepts_a_plain_dict_not_just_a_json_string():
    """`load_payload` is called with either a JSONB-decoded dict (asyncpg
    sometimes hands back a dict directly) or a raw JSON string -- both paths
    must still work."""
    now = datetime.now(UTC)
    from_dict = load_payload({"created_at": now.isoformat()})
    assert isinstance(from_dict["created_at"], datetime)
