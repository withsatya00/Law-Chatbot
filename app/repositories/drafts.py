from typing import Any

from app.core.config import settings
from app.database.postgresql import postgresql
from app.models.collections import DRAFT_VERSIONS, LEGAL_DRAFTS
from app.repositories.base import MongoRepository
from app.repositories.postgres_json import (
    delete_where,
    dump_payload,
    find_payload,
    insert_payload,
    load_payload,
    update_payload,
)


class DraftRepository(MongoRepository):
    collection_name = LEGAL_DRAFTS

    async def insert(self, document: dict[str, Any]) -> str:
        if settings.postgresql_enabled:
            return await insert_payload("ai_legal_drafts", document)
        return await super().insert(document)

    async def find_by_id(self, item_id: str) -> dict[str, Any] | None:
        if settings.postgresql_enabled:
            return await find_payload("ai_legal_drafts", item_id)
        return await super().find_by_id(item_id)

    async def update_by_id(self, item_id: str, updates: dict[str, Any]) -> bool:
        if settings.postgresql_enabled:
            return await update_payload("ai_legal_drafts", item_id, updates)
        return await super().update_by_id(item_id, updates)

    async def delete_by_id(self, item_id: str) -> bool:
        if settings.postgresql_enabled:
            return bool(await delete_where("ai_legal_drafts", "id=$1", item_id))
        return await super().delete_by_id(item_id)

    async def delete_by_session(self, session_id: str) -> int:
        if settings.postgresql_enabled:
            return await delete_where("ai_legal_drafts", "session_id=$1", session_id)
        return await super().delete_by_session(session_id)

    async def delete_by_user(self, user_id: str) -> int:
        if settings.postgresql_enabled:
            return await delete_where("ai_legal_drafts", "user_id=$1", user_id)
        return await super().delete_by_user(user_id)

    async def list_for_session(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        if settings.postgresql_enabled:
            async with postgresql.acquire() as connection:
                rows = await connection.fetch(
                    "SELECT payload FROM ai_legal_drafts WHERE session_id=$1 ORDER BY created_at DESC LIMIT $2",
                    session_id, limit,
                )
            return [load_payload(row["payload"]) for row in rows]
        cursor = self.collection.find({"session_id": session_id}).sort("created_at", -1).limit(limit)
        return [draft async for draft in cursor]

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        if settings.postgresql_enabled:
            async with postgresql.acquire() as connection:
                rows = await connection.fetch(
                    "SELECT payload FROM ai_legal_drafts WHERE user_id=$1 ORDER BY created_at DESC LIMIT $2",
                    user_id, limit,
                )
            return [load_payload(row["payload"]) for row in rows]
        cursor = self.collection.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        return [draft async for draft in cursor]

    async def compare_and_swap(self, draft_id: str, expected_version: int, updates: dict[str, Any]) -> bool:
        """Optimistic-concurrency update: applies `updates` (plus a bumped
        `version`) ONLY if the document's `version` is still exactly
        `expected_version` -- BUG-015 (`update_by_id`'s plain, unconditional
        `update_one($set)` let two overlapping `LegalDraftEngine.regenerate`
        calls each read a pre-edit snapshot and each write their own full
        merged snapshot, so whichever committed last silently discarded the
        other's field change while both reported success).

        Returns `True` if the write applied, `False` if `expected_version` was
        already stale (someone else's write landed first) -- the caller is
        expected to re-read, re-merge onto the fresh document, and retry, or
        surface a conflict rather than pretend the edit succeeded.

        Backward compatible with drafts persisted before `version` existed:
        `expected_version == 1` also matches a document with no `version`
        field at all (rather than requiring a one-off migration script), so
        the very next edit to a pre-existing draft self-heals it onto the
        new scheme -- every edit after that is a normal integer compare.
        """
        from datetime import UTC, datetime

        if settings.postgresql_enabled:
            now = datetime.now(UTC)
            async with postgresql.acquire() as connection, connection.transaction():
                value = await connection.fetchval(
                    "SELECT payload FROM ai_legal_drafts WHERE id=$1 FOR UPDATE", draft_id
                )
                if value is None:
                    return False
                document = load_payload(value)
                if int(document.get("version", 1)) != expected_version:
                    return False
                document.update(updates)
                document.update({"version": expected_version + 1, "updated_at": now})
                await connection.execute(
                    "UPDATE ai_legal_drafts SET session_id=$2,user_id=$3,payload=$4::jsonb,updated_at=$5 WHERE id=$1",
                    draft_id, document.get("session_id"), document.get("user_id"), dump_payload(document), now,
                )
                return True

        version_filter: dict[str, Any] = {"version": expected_version}
        if expected_version == 1:
            version_filter = {"$or": [{"version": 1}, {"version": {"$exists": False}}]}
        payload = {**updates, "version": expected_version + 1, "updated_at": datetime.now(UTC)}
        result = await self.collection.update_one(
            {"_id": draft_id, **version_filter}, {"$set": payload}
        )
        return result.modified_count > 0


class DraftVersionRepository(MongoRepository):
    collection_name = DRAFT_VERSIONS

    async def insert(self, document: dict[str, Any]) -> str:
        if settings.postgresql_enabled:
            return await insert_payload("ai_legal_draft_versions", document)
        return await super().insert(document)

    async def find_by_id(self, item_id: str) -> dict[str, Any] | None:
        if settings.postgresql_enabled:
            return await find_payload("ai_legal_draft_versions", item_id)
        return await super().find_by_id(item_id)

    async def delete_for_drafts(self, draft_ids: list[str]) -> int:
        if settings.postgresql_enabled:
            return await delete_where(
                "ai_legal_draft_versions", "payload->>'draft_id'=ANY($1::text[])", draft_ids
            )
        result = await self.collection.delete_many({"draft_id": {"$in": draft_ids}})
        return int(result.deleted_count)

    async def latest_for_draft(self, draft_id: str) -> dict[str, Any] | None:
        if settings.postgresql_enabled:
            async with postgresql.acquire() as connection:
                value = await connection.fetchval(
                    "SELECT payload FROM ai_legal_draft_versions WHERE payload->>'draft_id'=$1 "
                    "ORDER BY (payload->>'version_number')::integer DESC LIMIT 1", draft_id,
                )
            return load_payload(value) if value is not None else None
        return await self.collection.find_one(
            {"draft_id": draft_id},
            sort=[("version_number", -1)],
        )

    async def get_version(self, draft_id: str, version_number: int) -> dict[str, Any] | None:
        """A specific historical version of `draft_id`, for rollback."""
        if settings.postgresql_enabled:
            async with postgresql.acquire() as connection:
                value = await connection.fetchval(
                    "SELECT payload FROM ai_legal_draft_versions WHERE payload->>'draft_id'=$1 "
                    "AND (payload->>'version_number')::integer=$2", draft_id, version_number,
                )
            return load_payload(value) if value is not None else None
        return await self.collection.find_one({"draft_id": draft_id, "version_number": version_number})

    async def list_for_draft(self, draft_id: str) -> list[dict[str, Any]]:
        """Every version of `draft_id`, oldest first -- the full audit trail
        for a "show version history" chat command / the `/draft/{id}/versions`
        endpoint."""
        if settings.postgresql_enabled:
            async with postgresql.acquire() as connection:
                rows = await connection.fetch(
                    "SELECT payload FROM ai_legal_draft_versions WHERE payload->>'draft_id'=$1 "
                    "ORDER BY (payload->>'version_number')::integer", draft_id,
                )
            return [load_payload(row["payload"]) for row in rows]
        cursor = self.collection.find({"draft_id": draft_id}).sort("version_number", 1)
        return [version async for version in cursor]
