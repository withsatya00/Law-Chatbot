from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.kb_operations import KBOperationsService


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def sort(self, *_args):
        return self

    def limit(self, count):
        self.rows = self.rows[:count]
        return self

    def to_list(self, length=None):
        async def result():
            return self.rows[:length] if length else self.rows
        return result()

    def __aiter__(self):
        async def iterate():
            for row in self.rows:
                yield row
        return iterate()


def test_operations_report_surfaces_review_ocr_language_and_adapter_health() -> None:
    now = datetime(2026, 9, 7, tzinfo=UTC)
    jobs = AsyncMock()
    jobs.find = lambda *args: _Cursor([{
        "_id": "old-review", "updated_at": now - timedelta(days=5),
        "candidate": {"title": "Act"},
    }])
    jobs.count_documents = AsyncMock(side_effect=[2, 1])
    jobs.aggregate = lambda pipeline: _Cursor([{
        "_id": {"jurisdiction": "UP", "language": "hindi"}, "documents": 4,
    }])
    state = AsyncMock()
    state.find = lambda *args: _Cursor([{"_id": "broken", "last_error": "TimeoutError"}])
    documents = AsyncMock()
    documents.count_documents = AsyncMock(return_value=3)
    service = KBOperationsService(db={
        "kb_automation_jobs": jobs, "kb_automation_state": state,
        "uploaded_documents": documents, "operational_events": AsyncMock(),
    })

    report = asyncio.run(service.status(now))

    assert report["review_overdue_count"] == 1
    assert report["relationship_review_pending"] == 2
    assert report["manual_access_required"] == 1
    assert report["low_ocr_documents"] == 3
    assert report["published_language_coverage"][0]["language"] == "hindi"
    assert report["unhealthy_adapters"][0]["_id"] == "broken"


def test_alert_audit_is_deduplicated_by_open_alert_key() -> None:
    service = KBOperationsService(db={
        "kb_automation_jobs": AsyncMock(), "kb_automation_state": AsyncMock(),
        "uploaded_documents": AsyncMock(), "operational_events": AsyncMock(),
    })
    service.status = AsyncMock(return_value={
        "review_overdue_count": 2, "relationship_review_pending": 0,
        "manual_access_required": 0, "low_ocr_documents": 0, "unhealthy_adapters": [],
    })
    service.events.update_one = AsyncMock(
        return_value=SimpleNamespace(upserted_id="new-alert"),
    )

    report = asyncio.run(service.audit_and_alert())

    assert report["alerts_emitted"] == ["kb_review_sla_breached"]
    query = service.events.update_one.await_args.args[0]
    assert query["alert_key"] == "open:kb_review_sla_breached"


def test_operations_routes_are_admin_gated() -> None:
    from app.api.admin_phase3 import router
    from app.api.deps import require_admin

    assert any(dependency.dependency is require_admin for dependency in router.dependencies)
    paths = {route.path for route in router.routes}
    assert "/admin/phase3/kb-operations/status" in paths
    assert "/admin/phase3/kb-operations/audit" in paths
