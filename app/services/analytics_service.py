from datetime import UTC, datetime, timedelta
from typing import Any

from app.repositories.analytics import (
    IntentEventRepository,
    IntentFeedbackRepository,
    QueryLogRepository,
)
from app.utils.pii import mask_pii

# Phase 1 item 7: fields in an analytics payload that hold free user text.
# `ChatService` already redacts these on the way IN, but that only protects
# records written since; every row logged before it, and anything a future
# aggregation adds, would still reach the admin dashboard in the clear.
# Masking again on the way OUT makes the dashboard safe regardless of when
# a record was written, and costs nothing on already-masked text (the
# patterns simply find no identifiers).
_FREE_TEXT_ANALYTICS_FIELDS = frozenset({
    "question", "sample_question", "answer", "feedback_comment", "notes",
    "review_notes", "corrected_text", "original_question", "_id",
})


def _mask_analytics(value: Any) -> Any:
    """Recursively redact the free-text fields of an analytics payload.

    `_id` is included because several aggregations `$group` BY the question
    text, which puts the raw question into the group key.
    """
    if isinstance(value, dict):
        return {
            key: mask_pii(item) if key in _FREE_TEXT_ANALYTICS_FIELDS and isinstance(item, str)
            else _mask_analytics(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_mask_analytics(item) for item in value]
    return value


class AnalyticsService:
    def __init__(self) -> None:
        self.query_log = QueryLogRepository()
        self.intent_events = IntentEventRepository()
        self.intent_feedback = IntentFeedbackRepository()

    async def dashboard(self, days: int = 30) -> dict[str, Any]:
        since = datetime.now(UTC) - timedelta(days=days)
        overview = await self.query_log.overview_counts(since)
        knowledge_gaps = await self.query_log.knowledge_gaps(since)
        missing_documents = await self.query_log.missing_documents(since)
        low_quality_retrieval = await self.query_log.low_quality_retrieval(since)
        rating_summary = await self.query_log.rating_summary(since)
        result = {
            "window_days": days,
            "overview": overview,
            "frequent_questions": await self.query_log.frequent_questions(since),
            "top_intents": await self.query_log.top_intents(since),
            "knowledge_gaps": knowledge_gaps,
            "missing_documents": missing_documents,
            "low_confidence_responses": await self.query_log.low_confidence_responses(since),
            "failed_queries": await self.query_log.failed_queries(since),
            "low_quality_retrieval": low_quality_retrieval,
            "rating_summary": rating_summary,
            "intent_analytics": await self._intent_analytics(since),
            "language_failure_rates": await self.query_log.language_failure_rates(since),
            "feedback_trends": await self.query_log.feedback_trends(since),
        }
        result["recommendations"] = self._build_recommendations(
            overview, knowledge_gaps, missing_documents, low_quality_retrieval, rating_summary
        )
        masked: dict[str, Any] = _mask_analytics(result)
        return masked

    async def _intent_analytics(self, since: datetime) -> dict[str, Any]:
        """Phase 1 "Intent Analytics": most-common/low-confidence
        conversation intents, correction rate, multi-intent frequency,
        workflow-chain usage, LLM-vs-deterministic classification split,
        and the explicit "wrong intent" feedback rate -- all derived from
        `intent_events`/`intent_feedback` (see `ChatService._log_intent_event`
        and the `/feedback/intent` endpoint), separate from `QUERY_LOGS`'s
        RAG-answer-quality analytics above.
        """
        total_events = await self.intent_events.collection.count_documents({"created_at": {"$gte": since}})
        return {
            "most_common_intents": await self.intent_events.top_intents(since),
            "low_confidence_intents": await self.intent_events.low_confidence_intents(since),
            "correction_rate": await self.intent_events.correction_rate(since),
            "multi_intent_frequency": await self.intent_events.multi_intent_frequency(since),
            "workflow_chain_usage": await self.intent_events.workflow_chain_usage(since),
            "classifier_source_breakdown": await self.intent_events.classifier_source_breakdown(since),
            "wrong_intent_feedback": await self.intent_feedback.wrong_intent_rate(since, total_events),
            "top_corrected_intents": await self.intent_feedback.top_corrected_intents(since),
        }

    async def unanswered_queue(self, days: int, status: str, limit: int, skip: int) -> list[dict[str, Any]]:
        since = datetime.now(UTC) - timedelta(days=days)
        masked: list[dict[str, Any]] = _mask_analytics(
            await self.query_log.unanswered_queue(since, status=status, limit=limit, skip=skip)
        )
        return masked

    async def count_unanswered(self, days: int, status: str) -> int:
        """Correctness finding K1: the total number of matching records,
        independent of `unanswered_queue`'s own page size -- see
        `QueryLogRepository.count_unanswered`'s docstring."""
        since = datetime.now(UTC) - timedelta(days=days)
        return await self.query_log.count_unanswered(since, status=status)

    async def mark_reviewed(self, message_id: str, status: str, notes: str | None) -> bool:
        return await self.query_log.mark_reviewed(message_id, status, notes)

    def _build_recommendations(
        self,
        overview: dict[str, int],
        knowledge_gaps: list[dict[str, Any]],
        missing_documents: list[dict[str, Any]],
        low_quality_retrieval: list[dict[str, Any]],
        rating_summary: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        recommendations: list[dict[str, str]] = []

        for gap in knowledge_gaps[:5]:
            topic = gap["_id"] or "an unclassified topic"
            recommendations.append(
                {
                    "type": "coverage_gap",
                    "priority": "high" if gap["count"] >= 5 else "medium",
                    "message": (
                        f"{gap['count']} queries about '{topic}' returned no matching source. "
                        "Consider adding source material for this topic."
                    ),
                }
            )

        for doc in missing_documents[:5]:
            recommendations.append(
                {
                    "type": "missing_document",
                    "priority": "high" if doc["count"] >= 3 else "medium",
                    "message": (
                        f"Users asked about '{doc['_id']}' {doc['count']} time(s), but it isn't in the "
                        "knowledge base. Consider indexing it."
                    ),
                }
            )

        if low_quality_retrieval:
            recommendations.append(
                {
                    "type": "retrieval_quality",
                    "priority": "medium",
                    "message": (
                        f"{len(low_quality_retrieval)} responses in this window matched only weakly to "
                        "indexed content (top similarity score below 0.3). Review chunking or embedding "
                        "quality for these topics."
                    ),
                }
            )

        total_rated = sum(entry["count"] for entry in rating_summary)
        negative = sum(entry["count"] for entry in rating_summary if (entry["_id"] or 0) <= 2)
        if total_rated and negative / total_rated > 0.2:
            recommendations.append(
                {
                    "type": "feedback",
                    "priority": "high",
                    "message": f"{negative}/{total_rated} rated responses in this window scored 2 or below. Review recent low-rated answers.",
                }
            )

        total_queries = overview["total_queries"]
        if total_queries and overview["no_knowledge_source"] / total_queries > 0.15:
            share = round(overview["no_knowledge_source"] * 100 / total_queries)
            recommendations.append(
                {
                    "type": "coverage",
                    "priority": "high",
                    "message": (
                        f"{overview['no_knowledge_source']} of {total_queries} queries ({share}%) found no "
                        "knowledge source at all. Knowledge base coverage may need expansion."
                    ),
                }
            )

        return recommendations
