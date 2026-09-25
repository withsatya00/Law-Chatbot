from datetime import UTC, datetime
from typing import Any

from app.models.collections import (
    AUDIT_LOGS,
    BACKGROUND_JOBS,
    DOWNLOAD_ARTIFACTS,
    EVALUATION_RUNS,
    FORM_WORKFLOWS,
    LEGAL_SOURCES,
    OPERATIONAL_EVENTS,
    USER_PREFERENCES,
)
from app.repositories.base import MongoRepository


class LegalSourceRepository(MongoRepository):
    collection_name = LEGAL_SOURCES

    async def list_sources(self, query: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.collection.find(query).sort("updated_at", -1).limit(limit)
        return [item async for item in cursor]


class OperationalEventRepository(MongoRepository):
    collection_name = OPERATIONAL_EVENTS


class AuditLogRepository(MongoRepository):
    collection_name = AUDIT_LOGS

    async def find_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Most recent audit entries first.

        Audit visibility is now reachable from chat as well as from
        `GET /admin/phase3/audit-logs`; both read through here so the two can
        never disagree about what was recorded.
        """
        cursor = self.collection.find({}).sort("created_at", -1).limit(limit)
        return [item async for item in cursor]


class BackgroundJobRepository(MongoRepository):
    collection_name = BACKGROUND_JOBS

    async def list_for_owner(self, owner_user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.collection.find({"owner_user_id": owner_user_id}).sort("created_at", -1).limit(limit)
        return [item async for item in cursor]

    async def claim_next(self, worker_id: str) -> dict[str, Any] | None:
        from pymongo import ReturnDocument

        claimed: dict[str, Any] | None = await self.collection.find_one_and_update(
            {"status": "queued"},
            {"$set": {"status": "processing", "worker_id": worker_id, "started_at": datetime.now(UTC), "progress": 5}, "$inc": {"attempts": 1}},
            sort=[("created_at", 1)],
            return_document=ReturnDocument.AFTER,
        )
        return claimed


class UserPreferenceRepository(MongoRepository):
    collection_name = USER_PREFERENCES

    async def get_for_owner(self, owner_user_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"owner_user_id": owner_user_id})

    async def upsert_for_owner(self, owner_user_id: str, values: dict[str, Any]) -> None:
        now = datetime.now(UTC)
        await self.collection.update_one(
            {"owner_user_id": owner_user_id},
            {"$set": {**values, "updated_at": now}, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )


class FormWorkflowRepository(MongoRepository):
    collection_name = FORM_WORKFLOWS

    async def list_for_owner(self, owner_user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.collection.find({"owner_user_id": owner_user_id}).sort("updated_at", -1).limit(limit)
        return [item async for item in cursor]


class EvaluationRunRepository(MongoRepository):
    collection_name = EVALUATION_RUNS

    async def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Security/correctness finding G6: `POST /evaluations/run` was the
        only way to ever see a run's result -- once that response was gone,
        nothing let an admin list or re-open a historical run, even though
        every run was already durably recorded in this collection."""
        cursor = self.collection.find().sort("created_at", -1).limit(limit)
        return [item async for item in cursor]


class DownloadArtifactRepository(MongoRepository):
    collection_name = DOWNLOAD_ARTIFACTS

    async def list_for_owner(self, owner_user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.collection.find({"owner_user_id": owner_user_id}).sort("created_at", -1).limit(limit)
        return [item async for item in cursor]
