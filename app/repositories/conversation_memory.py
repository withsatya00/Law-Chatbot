from datetime import UTC, datetime
from typing import Any

from app.core.config import settings
from app.database.postgresql import postgresql
from app.models.collections import CONVERSATION_MEMORY
from app.repositories.base import MongoRepository
from app.repositories.postgres_json import dump_payload, load_payload


class ConversationMemoryRepository(MongoRepository):
    """Durable, long-term counterpart to the Redis-backed short-term session memory.

    Keyed directly by ``session_id`` so a session's memory survives a Redis TTL
    expiry or cache flush without needing a secondary lookup index.
    """

    collection_name = CONVERSATION_MEMORY

    async def find_by_session(self, session_id: str) -> dict[str, Any] | None:
        if settings.postgresql_enabled:
            async with postgresql.acquire() as connection:
                value = await connection.fetchval(
                    "SELECT payload FROM ai_conversation_states WHERE session_id=$1", session_id
                )
            return load_payload(value) if value is not None else None
        return await self.collection.find_one({"_id": session_id})

    async def upsert_by_session(self, session_id: str, payload: dict[str, Any]) -> None:
        if settings.postgresql_enabled:
            now = datetime.now(UTC)
            current = await self.find_by_session(session_id)
            document = {**(current or {}), **payload, "_id": session_id, "updated_at": now}
            document.setdefault("created_at", now)
            user_id = document.get("owner_user_id") or document.get("user_id")
            async with postgresql.acquire() as connection:
                await connection.execute(
                    "INSERT INTO ai_conversation_states (id,session_id,user_id,payload,created_at,updated_at) "
                    "VALUES ($1,$1,$2,$3::jsonb,$4,$5) ON CONFLICT (session_id) DO UPDATE SET "
                    "user_id=EXCLUDED.user_id,payload=EXCLUDED.payload,updated_at=EXCLUDED.updated_at",
                    session_id, user_id, dump_payload(document), document["created_at"], now,
                )
            return
        await self.collection.update_one(
            {"_id": session_id},
            {
                "$set": {**payload, "updated_at": datetime.now(UTC)},
                "$setOnInsert": {"created_at": datetime.now(UTC)},
            },
            upsert=True,
        )

    async def delete_by_session(self, session_id: str) -> int:
        """Overridden because this collection keys the session on `_id`, not on
        a `session_id` field. Returns the deleted count like the base method:
        `DELETE /session` reports one number per collection, and returning
        `None` here would have put a null in that report."""
        if settings.postgresql_enabled:
            async with postgresql.acquire() as connection:
                tag = await connection.execute(
                    "DELETE FROM ai_conversation_states WHERE session_id=$1", session_id
                )
            return int(tag.rsplit(" ", 1)[-1])
        result = await self.collection.delete_one({"_id": session_id})
        return int(result.deleted_count)

    async def list_session_ids_by_owner(self, owner_user_id: str) -> list[str]:
        """Every session this account has ever claimed -- security finding
        C2: `erase_user_data` needs this to actually clear an account's
        conversation memory/facts (previously not cleared at all by that
        route) and to resolve which `chats`/`feedback`/analytics rows are
        this account's even when they predate having their own `user_id`
        field (see `MongoRepository.delete_by_owner`).
        """
        if settings.postgresql_enabled:
            async with postgresql.acquire() as connection:
                rows = await connection.fetch(
                    "SELECT session_id FROM ai_conversation_states WHERE user_id=$1", owner_user_id
                )
            return [str(row["session_id"]) for row in rows]
        cursor = self.collection.find({"owner_user_id": owner_user_id}, {"_id": 1})
        return [str(doc["_id"]) async for doc in cursor]

    async def find_most_recent_by_owner(self, owner_user_id: str) -> dict[str, Any] | None:
        """The session an authenticated user was last actively talking in --
        used by `/voice/chat` (see `app/api/voice_router.py`) to resume the
        SAME conversation by default when a voice client omits `session_id`,
        instead of `ChatService._run_shared_preflight`'s normal
        `request.session_id or str(uuid4())` (a fresh, memory-less session
        every single call) -- which is fine for a text UI that persists
        `session_id` client-side across a chat window, but silently
        defeats "conversation" for a voice client that calls this endpoint
        once per utterance without its own session bookkeeping.
        """
        if settings.postgresql_enabled:
            async with postgresql.acquire() as connection:
                value = await connection.fetchval(
                    "SELECT payload FROM ai_conversation_states WHERE user_id=$1 "
                    "ORDER BY updated_at DESC LIMIT 1", owner_user_id,
                )
            return load_payload(value) if value is not None else None
        return await self.collection.find_one({"owner_user_id": owner_user_id}, sort=[("updated_at", -1)])
