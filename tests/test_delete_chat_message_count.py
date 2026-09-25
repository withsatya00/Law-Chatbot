"""Regression test for correctness finding C7 (DELETE /chat under-reported
its own deleted-message count).

Root cause: each stored `chats` row holds ONE user question AND ONE
assistant answer for a single turn, but `GET /history`'s own
`list_session_messages` caller already unpacks each row into two separate
`{"role": "user"|"assistant"}` messages -- `DELETE /chat` returned the raw
row count instead, understating "messages deleted" by roughly half relative
to that same API's own counting convention.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar
from uuid import uuid4

import pytest

from app.database.mongodb import mongodb
from app.repositories.chat_history import ChatRepository

T = TypeVar("T")


@pytest.fixture(scope="module")
def loop():
    ev_loop = asyncio.new_event_loop()
    mongodb._client = None
    ev_loop.run_until_complete(mongodb.connect())
    yield ev_loop
    ev_loop.run_until_complete(mongodb.close())
    ev_loop.close()


def run[T](loop, coro: Coroutine[Any, Any, T]) -> T:
    return loop.run_until_complete(coro)


def _sid() -> str:
    return f"c7-session-{uuid4()}"


def test_counts_both_sides_of_two_complete_turns(loop) -> None:
    session_id = _sid()
    chats = ChatRepository()
    for i in range(2):
        run(loop, chats.insert({
            "message_id": str(uuid4()), "session_id": session_id, "conversation_id": None,
            "user_id": None, "question": f"question {i}", "answer": f"answer {i}",
            "intent": {}, "conversation_intent": "General Legal Query", "entities": {}, "sources": [],
        }))

    deleted = run(loop, chats.delete_by_session_counting_messages(session_id))

    assert deleted == 4  # 2 turns x (1 question + 1 answer)
    # Independent re-fetch: the rows are actually gone, not just reported as such.
    remaining = run(loop, chats.collection.count_documents({"session_id": session_id}))
    assert remaining == 0


def test_does_not_overcount_a_turn_with_no_answer_yet(loop) -> None:
    """A failed/aborted turn (the request timed out before an answer was
    produced) has a question but no answer -- must count as 1, not 2."""
    session_id = _sid()
    chats = ChatRepository()
    run(loop, chats.insert({
        "message_id": str(uuid4()), "session_id": session_id, "conversation_id": None,
        "user_id": None, "question": "an unanswered question", "answer": "",
        "intent": {}, "conversation_intent": "General Legal Query", "entities": {}, "sources": [],
    }))

    deleted = run(loop, chats.delete_by_session_counting_messages(session_id))

    assert deleted == 1


def test_zero_messages_for_a_session_with_no_chat_history(loop) -> None:
    deleted = run(loop, ChatRepository().delete_by_session_counting_messages(_sid()))
    assert deleted == 0
