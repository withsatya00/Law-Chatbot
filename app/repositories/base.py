from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorCollection

from app.database.mongodb import mongodb


class MongoRepository:
    collection_name: str

    @property
    def collection(self) -> AsyncIOMotorCollection[Any]:
        return mongodb.db[self.collection_name]

    async def insert(self, document: dict[str, Any]) -> str:
        document.setdefault("_id", str(uuid4()))
        document.setdefault("created_at", datetime.now(UTC))
        document.setdefault("updated_at", datetime.now(UTC))
        await self.collection.insert_one(document)
        return str(document["_id"])

    async def find_by_id(self, item_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"_id": item_id})

    async def update_by_id(self, item_id: str, updates: dict[str, Any]) -> bool:
        updates["updated_at"] = datetime.now(UTC)
        result = await self.collection.update_one({"_id": item_id}, {"$set": updates})
        return result.modified_count > 0

    async def delete_by_session(self, session_id: str) -> int:
        """Removes every document this collection holds for one session.

        Part 58 "Answer Quality Audit" issue 25: `DELETE /session` used to
        clear only the conversation-memory document, leaving the same chat
        text behind in `chats`, `query_logs` and `intent_events` -- so a user
        who deleted their session still had their address, phone number and
        transaction details retained. Defined on the base repository so
        "erase everything for this session" is one obvious call per
        collection rather than a bespoke query at the call site.
        """
        result = await self.collection.delete_many({"session_id": session_id})
        return result.deleted_count

    async def delete_by_user(self, user_id: str) -> int:
        """Removes every document this collection holds for one account.

        Phase 1 item 7 ("delete-user-data"): a user can already delete one
        session, but an account accumulates rows across many sessions, and
        there was no way to clear all of them. Paired with
        `delete_by_session` so the erase-my-data endpoint is one call per
        collection either way.
        """
        result = await self.collection.delete_many({"user_id": user_id})
        return result.deleted_count

    async def delete_by_id(self, item_id: str) -> bool:
        result = await self.collection.delete_one({"_id": item_id})
        return result.deleted_count > 0

    async def delete_by_owner(self, user_id: str, session_ids: list[str] | None = None) -> int:
        """Security finding C2: `delete_by_user` alone under-reports (and
        under-deletes) for any collection whose documents can predate a
        `user_id` field being stamped at write time -- an anonymous turn
        later claimed by an account, or a document written before a given
        collection carried `user_id` at all (e.g. `intent_events`,
        `feedback`, neither of which had it until this fix). Matches on
        `user_id` directly OR membership in `session_ids` (the caller's own
        sessions, resolved once from `conversation_memory.owner_user_id`),
        so account-wide erasure reaches records a bare `user_id` filter
        would silently miss.
        """
        query: dict[str, Any] = {"user_id": user_id}
        if session_ids:
            query = {"$or": [{"user_id": user_id}, {"session_id": {"$in": session_ids}}]}
        result = await self.collection.delete_many(query)
        return result.deleted_count
