from datetime import UTC, datetime
from typing import Any

from app.models.collections import INTENT_EVENTS, INTENT_FEEDBACK, QUERY_LOGS
from app.repositories.base import MongoRepository

LOW_CONFIDENCE_THRESHOLD = 0.35
LOW_RETRIEVAL_SCORE_THRESHOLD = 0.3
NEGATIVE_RATING_THRESHOLD = 2


class QueryLogRepository(MongoRepository):
    collection_name = QUERY_LOGS

    async def attach_feedback(
        self, message_id: str, rating: int, comment: str | None, category: str | None = None
    ) -> bool:
        values = {"rating": rating, "feedback_comment": comment, "feedback_at": datetime.now(UTC)}
        if category:
            values["feedback_category"] = category
        return await self.update_by_id(
            message_id, values
        )

    async def overview_counts(self, since: datetime) -> dict[str, int]:
        window = {"created_at": {"$gte": since}}
        return {
            "total_queries": await self.collection.count_documents(window),
            "low_confidence": await self.collection.count_documents(
                {**window, "confidence": {"$lt": LOW_CONFIDENCE_THRESHOLD}}
            ),
            "no_knowledge_source": await self.collection.count_documents({**window, "knowledge_sources": {"$size": 0}}),
            "rated": await self.collection.count_documents({**window, "rating": {"$ne": None}}),
            "negative_feedback": await self.collection.count_documents(
                {**window, "rating": {"$lte": NEGATIVE_RATING_THRESHOLD}}
            ),
        }

    async def frequent_questions(self, since: datetime, limit: int = 15) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}}},
            {
                "$group": {
                    "_id": {"$toLower": {"$trim": {"input": "$question"}}},
                    "count": {"$sum": 1},
                    "sample_question": {"$first": "$question"},
                    "avg_confidence": {"$avg": "$confidence"},
                }
            },
            {"$match": {"count": {"$gt": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def top_intents(self, since: datetime, limit: int = 15) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}}},
            {"$group": {"_id": "$intent", "count": {"$sum": 1}, "avg_confidence": {"$avg": "$confidence"}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def knowledge_gaps(self, since: datetime, limit: int = 15) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}, "knowledge_sources": {"$size": 0}}},
            {"$group": {"_id": "$intent", "count": {"$sum": 1}, "sample_questions": {"$push": "$question"}}},
            {"$project": {"count": 1, "sample_questions": {"$slice": ["$sample_questions", 3]}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def missing_documents(self, since: datetime, limit: int = 15) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {
                "$match": {
                    "created_at": {"$gte": since},
                    "knowledge_sources": {"$size": 0},
                    "entities.act_name": {"$exists": True, "$ne": []},
                }
            },
            {"$unwind": "$entities.act_name"},
            {"$group": {"_id": "$entities.act_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def low_confidence_responses(
        self, since: datetime, threshold: float = LOW_CONFIDENCE_THRESHOLD, limit: int = 20
    ) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}, "confidence": {"$lt": threshold}}},
            {"$sort": {"confidence": 1}},
            {"$limit": limit},
            {
                "$project": {
                    "question": 1,
                    "confidence": 1,
                    "confidence_label": 1,
                    "intent": 1,
                    "language": 1,
                    "knowledge_sources": 1,
                    "created_at": 1,
                }
            },
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def failed_queries(self, since: datetime, limit: int = 20) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {
                "$match": {
                    "created_at": {"$gte": since},
                    "$or": [
                        {"knowledge_sources": {"$size": 0}},
                        {"rating": {"$lte": NEGATIVE_RATING_THRESHOLD}},
                    ],
                }
            },
            {"$sort": {"created_at": -1}},
            {"$limit": limit},
            {
                "$project": {
                    "question": 1,
                    "confidence": 1,
                    "rating": 1,
                    "feedback_comment": 1,
                    "knowledge_sources": 1,
                    "intent": 1,
                    "created_at": 1,
                }
            },
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def low_quality_retrieval(
        self, since: datetime, score_threshold: float = LOW_RETRIEVAL_SCORE_THRESHOLD, limit: int = 20
    ) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}, "retrieved_chunks.0": {"$exists": True}}},
            {"$addFields": {"top_score": {"$max": "$retrieved_chunks.score"}}},
            {"$match": {"top_score": {"$lt": score_threshold}}},
            {"$sort": {"top_score": 1}},
            {"$limit": limit},
            {"$project": {"question": 1, "top_score": 1, "intent": 1, "knowledge_sources": 1, "created_at": 1}},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def unanswered_queue(
        self, since: datetime, status: str = "pending", limit: int = 50, skip: int = 0
    ) -> list[dict[str, Any]]:
        """The PRD's "Continuous Learning System" queue: individual, actionable
        unanswered-question entries -- distinct from `knowledge_gaps`/
        `missing_documents` above, which GROUP entries by intent/Act for a
        trends view and can't be acted on one at a time (no per-entry id
        survives a `$group`). `review_status` defaults to "pending" via
        `$ifNull` for every entry logged before this field existed, so
        nothing already in `QUERY_LOGS` is silently excluded from the queue
        just because it predates review tracking.
        """
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}, "knowledge_sources": {"$size": 0}}},
            {"$addFields": {"review_status": {"$ifNull": ["$review_status", "pending"]}}},
            {"$match": {"review_status": status}} if status != "all" else {"$match": {}},
            {"$sort": {"created_at": -1}},
            {"$skip": skip},
            {"$limit": limit},
            {
                "$project": {
                    "message_id": 1,
                    "question": 1,
                    "intent": 1,
                    "language": 1,
                    "confidence": 1,
                    "created_at": 1,
                    "review_status": 1,
                    "review_notes": 1,
                }
            },
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def count_unanswered(self, since: datetime, status: str = "pending") -> int:
        """Correctness finding K1: `unanswered_queue` above is paginated
        (`$skip`/`$limit`), so `len(...)` of its result is at most `limit`
        -- never the true number of matching records once the backlog
        exceeds one page. This mirrors that method's own match stage
        exactly (`$ifNull` review_status default included) so the two never
        silently disagree about which rows count as "in the queue".
        """
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}, "knowledge_sources": {"$size": 0}}},
            {"$addFields": {"review_status": {"$ifNull": ["$review_status", "pending"]}}},
            {"$match": {"review_status": status}} if status != "all" else {"$match": {}},
            {"$count": "total"},
        ]
        result = [doc async for doc in self.collection.aggregate(pipeline)]
        return result[0]["total"] if result else 0

    async def mark_reviewed(self, message_id: str, status: str, notes: str | None) -> bool:
        return await self.update_by_id(
            message_id, {"review_status": status, "review_notes": notes, "reviewed_at": datetime.now(UTC)}
        )

    async def rating_summary(self, since: datetime) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}, "rating": {"$ne": None}}},
            {"$group": {"_id": "$rating", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def language_failure_rates(self, since: datetime) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}}},
            {"$group": {
                "_id": {"$ifNull": ["$language", "unknown"]},
                "total": {"$sum": 1},
                "failures": {"$sum": {"$cond": [{"$or": [
                    {"$lt": ["$confidence", LOW_CONFIDENCE_THRESHOLD]},
                    {"$eq": [{"$size": {"$ifNull": ["$knowledge_sources", []]}}, 0]},
                    {"$lte": [{"$ifNull": ["$rating", 5]}, NEGATIVE_RATING_THRESHOLD]},
                ]}, 1, 0]}},
            }},
            {"$project": {
                "language": "$_id", "total": 1, "failures": 1,
                "failure_rate": {"$cond": [{"$gt": ["$total", 0]}, {"$divide": ["$failures", "$total"]}, 0]},
            }},
            {"$sort": {"failure_rate": -1}},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def feedback_trends(self, since: datetime) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}, "rating": {"$ne": None}}},
            {"$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                    "rating": "$rating",
                    "category": {"$ifNull": ["$feedback_category", "unspecified"]},
                },
                "count": {"$sum": 1},
            }},
            {"$sort": {"_id.date": 1, "_id.rating": 1}},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]


class IntentEventRepository(MongoRepository):
    """Phase 1 "Intent Analytics": one document per conversation-intent
    classification decision (`ChatService._log_intent_event`, called
    alongside the existing session-scoped `ConversationMemoryStore.
    append_intent_event` at every classification site) -- unbounded and
    cross-session, unlike `intent_history` (capped at 50 events, scoped to
    one session's memory document), so these aggregations can actually
    answer "most common intent this month" instead of just "this
    conversation so far."
    """

    collection_name = INTENT_EVENTS

    async def top_intents(self, since: datetime, limit: int = 15) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}}},
            {"$group": {"_id": "$primary_intent", "count": {"$sum": 1}, "avg_confidence": {"$avg": "$confidence"}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def low_confidence_intents(
        self, since: datetime, threshold: float = LOW_CONFIDENCE_THRESHOLD, limit: int = 20
    ) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}, "confidence": {"$lt": threshold}}},
            {"$sort": {"confidence": 1}},
            {"$limit": limit},
            {"$project": {"question": 1, "primary_intent": 1, "confidence": 1, "classifier_source": 1, "created_at": 1}},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def correction_rate(self, since: datetime) -> dict[str, Any]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}}},
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "corrections": {"$sum": {"$cond": ["$is_correction", 1, 0]}},
                }
            },
        ]
        results = [doc async for doc in self.collection.aggregate(pipeline)]
        if not results:
            return {"total": 0, "corrections": 0, "rate": 0.0}
        total, corrections = results[0]["total"], results[0]["corrections"]
        return {"total": total, "corrections": corrections, "rate": round(corrections / total, 4) if total else 0.0}

    async def multi_intent_frequency(self, since: datetime) -> dict[str, Any]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}}},
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "multi_intent": {
                        "$sum": {"$cond": [{"$gte": [{"$size": {"$ifNull": ["$detected_intents", []]}}, 2]}, 1, 0]}
                    },
                }
            },
        ]
        results = [doc async for doc in self.collection.aggregate(pipeline)]
        if not results:
            return {"total": 0, "multi_intent": 0, "rate": 0.0}
        total, multi = results[0]["total"], results[0]["multi_intent"]
        return {"total": total, "multi_intent": multi, "rate": round(multi / total, 4) if total else 0.0}

    async def workflow_chain_usage(self, since: datetime) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}, "workflow_chain": {"$exists": True, "$ne": []}}},
            {"$group": {"_id": "$workflow_chain", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]

    async def classifier_source_breakdown(self, since: datetime) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}}},
            {"$group": {"_id": "$classifier_source", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]


class IntentFeedbackRepository(MongoRepository):
    """Phase 1 "Intent Feedback": explicit user corrections of intent
    classification ("Wrong intent", "I wanted legal research") -- kept in
    its own collection, deliberately separate from both `intent_events`
    (silent, automatic classification telemetry) and any legal-facts store
    (`app/memory/entity_memory.py`), since this is metadata ABOUT the
    conversation's routing, not a legal fact extracted FROM it.
    """

    collection_name = INTENT_FEEDBACK

    async def wrong_intent_rate(self, since: datetime, total_intent_events: int) -> dict[str, Any]:
        count = await self.collection.count_documents({"created_at": {"$gte": since}})
        return {
            "wrong_intent_feedback_count": count,
            "rate": round(count / total_intent_events, 4) if total_intent_events else 0.0,
        }

    async def top_corrected_intents(self, since: datetime, limit: int = 15) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"created_at": {"$gte": since}}},
            {
                "$group": {
                    "_id": {"original": "$original_intent", "corrected": "$corrected_intent"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return [doc async for doc in self.collection.aggregate(pipeline)]
