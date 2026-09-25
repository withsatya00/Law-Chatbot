"""Regression tests for security finding C2 (DELETE /me/data was a no-op
for chat history).

Root cause: `chats`/`query_logs` were stamped with the client-supplied
`ChatRequest.user_id` rather than the server-verified owner, and
`intent_events`/`feedback` never carried a `user_id` field at all -- so
`erase_user_data`'s `delete_many({"user_id": <real authenticated id>})`
matched nothing, for every account, every time. Fixed by (1) stamping the
real, JWT-verified owner at write time (`ChatService`, `app/services/
chat_support/analytics.py`) and (2) resolving deletion by session
membership as well as by `user_id` (`MongoRepository.delete_by_owner`,
`ConversationMemoryRepository.list_session_ids_by_owner`) so historical
rows written before fix (1), or while a since-claimed session was still
anonymous, are still reachable.

These tests seed real documents into the real, locally-running Mongo this
project's test suite already targets, call `erase_user_data` directly, and
independently re-query each collection afterward -- write, then verify via
a separate read, per this task's persistence-verification requirement.

All tests in this module run on ONE shared event loop (the `loop` fixture)
rather than one `asyncio.run` per test: motor binds its client to the loop
it first does real I/O on, and `mongodb._client` is a process-wide
singleton, so a second, independently-created loop reusing that same
client fails with 'Event loop is closed' (see the identical note in
`tests/test_legal_benchmark.py`). Closing/reconnecting the shared client
between tests was tried and rejected -- it leaked into unrelated test
files run later in the same session (`mongodb`/`redis_client` are
process-wide), breaking tests that assume the connection just stays up.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar
from uuid import uuid4

import pytest

from app.database.mongodb import mongodb
from app.models.collections import CONVERSATION_MEMORY
from app.repositories.analytics import IntentEventRepository, QueryLogRepository
from app.repositories.chat_history import ChatRepository, FeedbackRepository
from app.services.user_data import erase_user_data

T = TypeVar("T")


@pytest.fixture(scope="module")
def loop():
    ev_loop = asyncio.new_event_loop()
    # Force a real connection regardless of what an earlier, unrelated unit
    # test left `mongodb._client` monkeypatched to (see the identical note
    # in `tests/test_session_ownership_hardening.py`'s fixture) -- `connect()`
    # alone is a no-op once it is non-`None`.
    mongodb._client = None
    ev_loop.run_until_complete(mongodb.connect())
    yield ev_loop
    # `mongodb` is a process-wide singleton shared with every other test
    # file in this session. Closing this loop below without first resetting
    # `mongodb._client` would leave it bound to a now-dead loop -- the next
    # test file to touch real Mongo would then fail with 'Event loop is
    # closed' despite having nothing to do with this module (confirmed live
    # this way: it broke unrelated `test_draft_export.py`/
    # `test_draft_lifecycle.py` runs that merely happened to run
    # afterward). Resetting it here, while `ev_loop` is still alive, means
    # whoever touches `mongodb` next just reconnects fresh, transparently.
    ev_loop.run_until_complete(mongodb.close())
    ev_loop.close()


def run[T](loop, coro: Coroutine[Any, Any, T]) -> T:
    return loop.run_until_complete(coro)


def _uid() -> str:
    return f"c2-user-{uuid4()}"


def _sid() -> str:
    return f"c2-session-{uuid4()}"


def test_correctly_stamped_chat_messages_are_deleted_and_verifiably_gone(loop) -> None:
    """The going-forward case: a message written the way the fixed
    `ChatService` now writes it (real `user_id`) must actually be reachable
    by `erase_user_data`."""
    user_id = _uid()
    session_id = _sid()
    message_id = str(uuid4())
    chats = ChatRepository()

    run(loop, chats.insert({
        "message_id": message_id, "session_id": session_id, "conversation_id": None,
        "user_id": user_id, "question": "q", "answer": "a", "intent": {},
        "conversation_intent": "General Legal Query", "entities": {}, "sources": [],
    }))

    result = run(loop, erase_user_data(user_id))

    assert result["deleted"]["chat_messages"] >= 1
    # Independent re-fetch, not just the trusted return value.
    assert run(loop, chats.collection.count_documents({"user_id": user_id})) == 0
    assert run(loop, chats.find_by_id(message_id)) is None


def test_legacy_message_with_no_user_id_is_still_reachable_via_session_ownership(loop) -> None:
    """Reproduces the EXACT reported failure: a `chats` row whose `user_id`
    field is absent/None (as every pre-fix write, and every write from a
    since-claimed anonymous session, actually looks) must still be deleted,
    by resolving it through the session it belongs to instead."""
    user_id = _uid()
    session_id = _sid()
    message_id = str(uuid4())
    chats = ChatRepository()

    run(loop, mongodb.db[CONVERSATION_MEMORY].insert_one({
        "_id": session_id, "owner_user_id": user_id, "summary": "", "recent_messages": [],
    }))
    run(loop, chats.insert({
        "message_id": message_id, "session_id": session_id, "conversation_id": None,
        "user_id": None, "question": "q", "answer": "a", "intent": {},
        "conversation_intent": "General Legal Query", "entities": {}, "sources": [],
    }))

    result = run(loop, erase_user_data(user_id))

    assert result["deleted"]["chat_messages"] >= 1, "a legacy user_id=None message must still be erased"
    assert run(loop, chats.find_by_id(message_id)) is None
    assert run(loop, mongodb.db[CONVERSATION_MEMORY].find_one({"_id": session_id})) is None


def test_query_logs_are_deleted_by_owner(loop) -> None:
    user_id = _uid()
    session_id = _sid()
    message_id = str(uuid4())
    query_log = QueryLogRepository()

    run(loop, query_log.insert({
        "_id": message_id, "message_id": message_id, "session_id": session_id,
        "user_id": user_id, "question": "q", "answer": "a",
    }))

    result = run(loop, erase_user_data(user_id))

    assert result["deleted"]["query_logs"] >= 1
    assert run(loop, query_log.find_by_id(message_id)) is None


def test_intent_events_never_had_a_user_id_field_and_are_now_deleted_via_session(loop) -> None:
    """`intent_events` documents never carried `user_id` at all before this
    fix (see `ChatService._log_intent_event`) -- session membership is the
    ONLY way these are reachable for a real account, old or new."""
    user_id = _uid()
    session_id = _sid()
    event_id = str(uuid4())
    intent_events = IntentEventRepository()

    run(loop, mongodb.db[CONVERSATION_MEMORY].insert_one({
        "_id": session_id, "owner_user_id": user_id, "summary": "", "recent_messages": [],
    }))
    run(loop, intent_events.insert({
        "_id": event_id, "session_id": session_id, "question": "q", "primary_intent": "General Legal Query",
    }))

    result = run(loop, erase_user_data(user_id))

    assert result["deleted"]["intent_events"] >= 1
    assert run(loop, intent_events.find_by_id(event_id)) is None


def test_feedback_has_no_user_id_field_and_is_deleted_via_session_ownership(loop) -> None:
    """`FeedbackRequest`/`POST /feedback` never carry an authenticated
    identity at all -- feedback is only ever reachable through the session
    it was left on, which is exactly what `delete_by_owner` resolves via
    `ConversationMemoryRepository.list_session_ids_by_owner`."""
    user_id = _uid()
    session_id = _sid()
    feedback_id = str(uuid4())
    feedback = FeedbackRepository()

    run(loop, mongodb.db[CONVERSATION_MEMORY].insert_one({
        "_id": session_id, "owner_user_id": user_id, "summary": "", "recent_messages": [],
    }))
    run(loop, feedback.insert({
        "_id": feedback_id, "session_id": session_id, "message_id": None, "rating": 5, "comment": "great",
    }))

    result = run(loop, erase_user_data(user_id))

    assert result["deleted"]["feedback"] >= 1
    assert run(loop, feedback.find_by_id(feedback_id)) is None


def test_conversation_memory_sessions_are_cleared_by_account_erasure(loop) -> None:
    """`erase_user_data` previously never touched `conversation_memory` at
    all for an account-wide erase -- only `DELETE /session` cleared it, one
    session at a time. A user who asks to delete everything must not still
    have their confirmed facts sitting in an unlisted session."""
    user_id = _uid()
    session_id = _sid()

    run(loop, mongodb.db[CONVERSATION_MEMORY].insert_one({
        "_id": session_id, "owner_user_id": user_id,
        "confirmed_facts": {"city": "Jaipur"}, "summary": "", "recent_messages": [],
    }))

    result = run(loop, erase_user_data(user_id))

    assert result["deleted"]["sessions"] >= 1
    assert run(loop, mongodb.db[CONVERSATION_MEMORY].find_one({"_id": session_id})) is None


def test_erasure_report_reflects_true_deleted_counts_not_a_fake_success(loop) -> None:
    """Rule: never return fake success -- an account with genuinely nothing
    to delete must report zero, not a lie in either direction."""
    user_id = _uid()

    result = run(loop, erase_user_data(user_id))

    assert result["status"] == "deleted"
    assert result["deleted"]["chat_messages"] == 0
    assert result["deleted"]["sessions"] == 0
