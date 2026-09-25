import re
from typing import Any

from app.core.config import settings
from app.database.postgresql import postgresql
from app.models.collections import CHATS, FEEDBACK
from app.repositories.base import MongoRepository
from app.repositories.postgres_json import delete_where, insert_payload, load_payload


class ChatRepository(MongoRepository):
    collection_name = CHATS

    async def insert(self, document: dict[str, Any]) -> str:
        if settings.postgresql_enabled:
            return await insert_payload("ai_chat_turns", document)
        return await super().insert(document)

    async def list_session_messages(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        if settings.postgresql_enabled:
            async with postgresql.acquire() as connection:
                rows = await connection.fetch(
                    "SELECT payload FROM ai_chat_turns WHERE session_id=$1 ORDER BY created_at LIMIT $2",
                    session_id, limit,
                )
            return [load_payload(row["payload"]) for row in rows]
        cursor = self.collection.find({"session_id": session_id}).sort("created_at", 1).limit(limit)
        return [message async for message in cursor]

    async def delete_by_session_counting_messages(self, session_id: str) -> int:
        """Security/correctness finding C7: `delete_by_session` (the base
        method) returns the number of stored TURN documents, each of which
        holds one user question AND one assistant answer -- exactly how
        `list_session_messages`'s own caller (`GET /history`, `app/api/
        history.py`) already unpacks each row into two separate `{"role":
        "user"|"assistant", ...}` messages. Reporting the raw document count
        from `DELETE /chat` understated "messages deleted" by roughly half
        relative to that same API's own counting convention. Counts only
        the non-empty side of each turn (a failed/aborted turn can have a
        question with no answer yet) rather than a blind `* 2`, so the
        count stays accurate for a turn that never completed.
        """
        if settings.postgresql_enabled:
            messages = await self.list_session_messages(session_id, limit=1_000_000)
            deleted = sum(bool(row.get(key)) for row in messages for key in ("question", "answer"))
            await self.delete_by_session(session_id)
            return deleted
        deleted = 0
        cursor = self.collection.find({"session_id": session_id}, {"question": 1, "answer": 1})
        async for document in cursor:
            deleted += bool(document.get("question"))
            deleted += bool(document.get("answer"))
        await self.collection.delete_many({"session_id": session_id})
        return deleted

    async def delete_by_session(self, session_id: str) -> int:
        if settings.postgresql_enabled:
            return await delete_where("ai_chat_turns", "session_id=$1", session_id)
        return await super().delete_by_session(session_id)

    async def delete_by_user(self, user_id: str) -> int:
        if settings.postgresql_enabled:
            return await delete_where("ai_chat_turns", "user_id=$1", user_id)
        return await super().delete_by_user(user_id)

    async def delete_by_owner(self, user_id: str, session_ids: list[str] | None = None) -> int:
        if settings.postgresql_enabled:
            if session_ids:
                return await delete_where(
                    "ai_chat_turns", "user_id=$1 OR session_id=ANY($2::text[])", user_id, session_ids
                )
            return await self.delete_by_user(user_id)
        return await super().delete_by_owner(user_id, session_ids)

    async def search_for_user(self, user_id: str, query: str, limit: int = 30) -> list[dict[str, Any]]:
        if settings.postgresql_enabled:
            pg_pattern = f"%{query}%"
            async with postgresql.acquire() as connection:
                rows = await connection.fetch(
                    "SELECT payload FROM ai_chat_turns WHERE user_id=$1 AND "
                    "(payload->>'question' ILIKE $2 OR payload->>'answer' ILIKE $2) "
                    "ORDER BY created_at DESC LIMIT $3",
                    user_id, pg_pattern, limit,
                )
            return [load_payload(row["payload"]) for row in rows]
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        cursor = self.collection.find(
            {"user_id": user_id, "$or": [{"question": pattern}, {"answer": pattern}]},
            {"question": 1, "answer": 1, "session_id": 1, "created_at": 1},
        ).sort("created_at", -1).limit(limit)
        return [item async for item in cursor]


class FeedbackRepository(MongoRepository):
    collection_name = FEEDBACK
