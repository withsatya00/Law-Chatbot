from typing import Any

from app.models.collections import CASES
from app.repositories.base import MongoRepository


class CaseRepository(MongoRepository):
    collection_name = CASES

    async def list_for_owner(self, owner_user_id: str, status: str | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"owner_user_id": owner_user_id}
        if status:
            query["status"] = status
        cursor = self.collection.find(query).sort("updated_at", -1)
        return [case async for case in cursor]

    async def delete_by_id(self, case_id: str) -> bool:
        result = await self.collection.delete_one({"_id": case_id})
        return result.deleted_count > 0

    async def upcoming_hearings(self, owner_user_id: str, within_days: int) -> list[dict[str, Any]]:
        """Cases with a `next_hearing_date` between now and `within_days`
        from now -- the query surface a future Court Calendar view would
        read from, kept here rather than in that not-yet-built feature so
        the underlying data access isn't duplicated when it arrives.
        """
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        horizon = now + timedelta(days=within_days)
        query = {
            "owner_user_id": owner_user_id,
            "next_hearing_date": {"$gte": now, "$lte": horizon},
            "status": {"$ne": "closed"},
        }
        cursor = self.collection.find(query).sort("next_hearing_date", 1)
        return [case async for case in cursor]
