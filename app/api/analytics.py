from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.deps import require_admin
from app.cache.response_cache import response_cache
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/admin/analytics", tags=["admin-analytics"], dependencies=[Depends(require_admin)])


@router.get("/dashboard")
async def dashboard(days: int = Query(30, ge=1, le=365)) -> dict[str, Any]:
    """Admin-facing knowledge-improvement analytics: FAQs, knowledge gaps, failed
    queries, low-confidence responses, frequently-missing documents, low-quality
    retrieval, and rule-based recommendations -- all derived read-only from
    `QUERY_LOGS`. Never writes back to the knowledge base; adding documents still
    requires an administrator to upload them and trigger `/admin/reindex`.
    """
    return await AnalyticsService().dashboard(days=days)


@router.get("/cache")
async def cache_stats() -> dict[str, Any]:
    """Response-cache performance: hits, misses, hit ratio, average lookup
    latency, and estimated tokens/cost saved by skipping the LLM on a hit."""
    return await response_cache.stats()


@router.get("/unanswered-queue")
async def unanswered_queue(
    days: int = Query(30, ge=1, le=365),
    status: str = Query("pending", pattern="^(pending|reviewed|resolved|all)$"),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
) -> dict[str, Any]:
    """The PRD "Continuous Learning System" queue: individual unanswered
    questions (no knowledge source matched) an admin can work through one at
    a time -- `dashboard()`'s `knowledge_gaps`/`missing_documents` cover the
    same underlying data but grouped by topic/Act for a trends view, with no
    single entry left to act on. `status` defaults to "pending" so the
    queue's default view is exactly the backlog still needing attention.
    """
    service = AnalyticsService()
    items = await service.unanswered_queue(days=days, status=status, limit=limit, skip=skip)
    # Correctness finding K1: `count` is the TOTAL number of matching
    # records across every page, not `len(items)` (which is capped at
    # `limit` and would silently misreport the backlog size once it exceeds
    # one page).
    total = await service.count_unanswered(days=days, status=status)
    return {"items": items, "count": total}


class ReviewRequest(BaseModel):
    status: str
    notes: str | None = None


@router.post("/unanswered-queue/{message_id}/review")
async def review_unanswered(message_id: str, request: ReviewRequest) -> dict[str, Any]:
    """Marks one queue entry "reviewed" (looked at, no KB change needed yet)
    or "resolved" (a document was uploaded/indexed to cover it) -- the
    "Admin Review" step of the PRD's Question -> Queue -> Review -> New
    Document -> KB Update flow. Never re-indexes anything itself: adding the
    actual document still goes through the existing `/upload`/`/admin`
    ingestion endpoints, same boundary `dashboard()`'s own docstring already
    draws for `knowledge_gaps`.
    """
    updated = await AnalyticsService().mark_reviewed(message_id, request.status, request.notes)
    if not updated:
        raise HTTPException(status_code=404, detail="No unanswered-queue entry found with that message_id.")
    return {"message_id": message_id, "status": request.status}
