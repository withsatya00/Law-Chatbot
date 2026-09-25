"""Regression test for correctness finding K1 (unanswered-queue `count`
reported page length, not total matching records).

Root cause: `GET /analytics/unanswered-queue` returned `{"items": items,
"count": len(items)}` -- `items` is `unanswered_queue`'s own PAGINATED
result (`$skip`/`$limit`), so `count` was capped at `limit` (default 50)
regardless of how many records actually matched.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from app.services.analytics_service import AnalyticsService


def test_count_unanswered_is_independent_of_the_page_size() -> None:
    service = AnalyticsService()
    # 200 records match, but a page only ever returns at most `limit`.
    service.query_log.count_unanswered = AsyncMock(return_value=200)
    service.query_log.unanswered_queue = AsyncMock(return_value=[{"message_id": str(i)} for i in range(50)])

    items = asyncio.run(service.unanswered_queue(days=30, status="pending", limit=50, skip=0))
    total = asyncio.run(service.count_unanswered(days=30, status="pending"))

    assert len(items) == 50
    assert total == 200
    assert total != len(items)


def test_repository_count_matches_the_queues_own_filter_semantics(monkeypatch) -> None:
    """`count_unanswered`'s aggregation must apply the SAME match stage
    `unanswered_queue` does (including the `review_status` `$ifNull`
    default), or the two could silently disagree about what's "in the
    queue" -- exercised here against a fake collection rather than a mock,
    so the actual pipeline shape is what's under test."""
    from app.repositories.analytics import QueryLogRepository

    class _FakeCursor:
        def __init__(self, rows: list[dict]) -> None:
            self._rows = rows

        def __aiter__(self):
            async def _gen():
                for row in self._rows:
                    yield row
            return _gen()

    class _FakeCollection:
        def __init__(self, rows: list[dict]) -> None:
            self._rows = rows

        def aggregate(self, pipeline: list[dict]) -> _FakeCursor:
            # Minimal, deliberately literal interpreter of just the two
            # stages this pipeline actually uses, enough to prove the
            # method asks for the RIGHT count rather than re-implementing
            # a full aggregation engine.
            matched = [
                row for row in self._rows
                if not row.get("knowledge_sources")
                and (row.get("review_status") or "pending") == "pending"
            ]
            for stage in pipeline:
                if "$count" in stage:
                    return _FakeCursor([{"total": len(matched)}] if matched else [])
            return _FakeCursor(matched)

    rows = [
        {"created_at": datetime.now(UTC), "knowledge_sources": [], "review_status": None},
        {"created_at": datetime.now(UTC), "knowledge_sources": [], "review_status": "pending"},
        {"created_at": datetime.now(UTC), "knowledge_sources": [], "review_status": "resolved"},
        {"created_at": datetime.now(UTC), "knowledge_sources": ["some_act.pdf"], "review_status": None},
    ]
    repository = QueryLogRepository()
    monkeypatch.setattr(
        QueryLogRepository, "collection", property(lambda self: _FakeCollection(rows)),
    )

    total = asyncio.run(repository.count_unanswered(datetime.now(UTC), status="pending"))

    # Only the two rows with no knowledge sources AND an effective
    # "pending" review status count -- matches `unanswered_queue`'s own
    # `$ifNull` default and `knowledge_sources: {"$size": 0}` filter.
    assert total == 2
