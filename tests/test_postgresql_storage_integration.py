"""Opt-in tests against the real local PostgreSQL container."""

import os
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("POSTGRESQL_LIVE_TESTS") != "1",
    reason="set POSTGRESQL_LIVE_TESTS=1 to run real PostgreSQL tests",
)


@pytest.mark.asyncio
async def test_chat_memory_and_draft_crud() -> None:
    from app.database.postgresql import postgresql
    from app.repositories.chat_history import ChatRepository
    from app.repositories.conversation_memory import ConversationMemoryRepository
    from app.repositories.drafts import DraftRepository, DraftVersionRepository

    session_id = f"pg-test-{uuid4()}"
    user_id = f"user-{uuid4()}"
    drafts = DraftRepository()
    versions = DraftVersionRepository()
    chats = ChatRepository()
    memory = ConversationMemoryRepository()
    await postgresql.connect()
    try:
        assert await postgresql.ping()
        chat_id = await chats.insert({
            "session_id": session_id, "user_id": user_id,
            "question": "मेरी सहायता करें", "answer": "ज़रूर",
        })
        messages = await chats.list_session_messages(session_id)
        assert messages[0]["_id"] == chat_id
        assert messages[0]["question"] == "मेरी सहायता करें"
        assert (await chats.search_for_user(user_id, "सहायता"))[0]["_id"] == chat_id

        await memory.upsert_by_session(session_id, {"owner_user_id": user_id, "summary": "test"})
        assert (await memory.find_by_session(session_id))["summary"] == "test"
        assert session_id in await memory.list_session_ids_by_owner(user_id)

        draft_id = await drafts.insert({
            "session_id": session_id, "user_id": user_id,
            "draft_type": "notice", "full_text": "version one", "version": 1,
        })
        await versions.insert({"draft_id": draft_id, "version_number": 1, "full_text": "version one"})
        assert await drafts.compare_and_swap(draft_id, 1, {"full_text": "version two"})
        assert not await drafts.compare_and_swap(draft_id, 1, {"full_text": "stale"})
        assert (await drafts.find_by_id(draft_id))["version"] == 2
        assert (await versions.latest_for_draft(draft_id))["version_number"] == 1
    finally:
        draft_rows = await drafts.list_for_session(session_id)
        await versions.delete_for_drafts([str(row["_id"]) for row in draft_rows])
        await drafts.delete_by_session(session_id)
        await chats.delete_by_session(session_id)
        await memory.delete_by_session(session_id)
        await postgresql.close()
