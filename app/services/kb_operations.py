"""Phase-6 operational audit for corpus review, OCR, language and source health."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.config import settings
from app.database.mongodb import mongodb
from app.models.collections import (
    KB_AUTOMATION_JOBS,
    KB_AUTOMATION_STATE,
    OPERATIONAL_EVENTS,
    UPLOADED_DOCUMENTS,
)


class KBOperationsService:
    def __init__(self, db: Any = None) -> None:
        self.db = db if db is not None else mongodb.db
        self.jobs = self.db[KB_AUTOMATION_JOBS]
        self.state = self.db[KB_AUTOMATION_STATE]
        self.documents = self.db[UPLOADED_DOCUMENTS]
        self.events = self.db[OPERATIONAL_EVENTS]

    async def status(self, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        cutoff = now - timedelta(hours=max(1, settings.kb_review_sla_hours))
        overdue = [item async for item in self.jobs.find({
            "status": "quarantined", "last_error": {"$exists": False},
            "updated_at": {"$lte": cutoff},
        }, {"_id": 1, "candidate": 1, "updated_at": 1}).sort("updated_at", 1).limit(200)]
        relationship_pending = await self.jobs.count_documents({
            "status": "quarantined",
            "candidate.change_type": {"$in": ["amendment", "repeal", "commencement"]},
            "relationship_reviewed_at": {"$exists": False},
        })
        manual_access = await self.jobs.count_documents({"status": "manual_access_required"})
        low_ocr = await self.documents.count_documents({
            "quality_report.ocr_quality_score": {"$lt": settings.min_ocr_quality_score},
            "index_status": {"$ne": "deleted"},
        })
        unhealthy_adapters = [item async for item in self.state.find({
            "$or": [{"layout_changed": True}, {"last_error": {"$nin": [None, ""]}}],
        }, {"_id": 1, "last_error": 1, "layout_changed": 1, "last_failure_at": 1})]
        language_rows = await self.jobs.aggregate([
            {"$match": {"status": "published"}},
            {"$group": {"_id": {
                "jurisdiction": "$candidate.jurisdiction_code",
                "language": "$candidate.language",
            }, "documents": {"$sum": 1}}},
            {"$sort": {"_id.jurisdiction": 1, "_id.language": 1}},
        ]).to_list(length=None)
        return {
            "generated_at": now.isoformat(),
            "review_sla_hours": settings.kb_review_sla_hours,
            "review_overdue_count": len(overdue),
            "review_overdue": overdue,
            "relationship_review_pending": relationship_pending,
            "manual_access_required": manual_access,
            "low_ocr_documents": low_ocr,
            "unhealthy_adapters": unhealthy_adapters,
            "published_language_coverage": [{
                "jurisdiction_code": row["_id"].get("jurisdiction"),
                "language": row["_id"].get("language") or "unknown",
                "documents": row["documents"],
            } for row in language_rows],
        }

    async def audit_and_alert(self) -> dict[str, Any]:
        report = await self.status()
        conditions = {
            "kb_review_sla_breached": report["review_overdue_count"],
            "kb_relationship_review_backlog": report["relationship_review_pending"],
            "kb_manual_access_backlog": report["manual_access_required"],
            "kb_ocr_quality_backlog": report["low_ocr_documents"],
            "kb_adapter_health_degraded": len(report["unhealthy_adapters"]),
        }
        emitted: list[str] = []
        now = datetime.now(UTC)
        for event_type, count in conditions.items():
            if not count:
                continue
            alert_key = f"open:{event_type}"
            result = await self.events.update_one(
                {"alert_key": alert_key, "resolved_at": {"$exists": False}},
                {"$setOnInsert": {
                    "event_type": event_type, "alert_key": alert_key,
                    "details": {"count": count}, "created_at": now,
                }, "$set": {"last_seen_at": now, "details.count": count}},
                upsert=True,
            )
            if result.upserted_id:
                emitted.append(event_type)
        return {**report, "alerts_emitted": emitted}
