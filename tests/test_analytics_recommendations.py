import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from app.services.analytics_service import AnalyticsService


def _service() -> AnalyticsService:
    return AnalyticsService.__new__(AnalyticsService)


def test_coverage_gap_recommendation_priority_scales_with_count() -> None:
    service = _service()
    knowledge_gaps = [
        {"_id": "Cyber Law", "count": 6, "sample_questions": ["q1"]},
        {"_id": "Tenancy", "count": 2, "sample_questions": ["q2"]},
    ]
    recommendations = service._build_recommendations(
        overview={"total_queries": 10, "no_knowledge_source": 0, "low_confidence": 0, "rated": 0, "negative_feedback": 0},
        knowledge_gaps=knowledge_gaps,
        missing_documents=[],
        low_quality_retrieval=[],
        rating_summary=[],
    )
    gap_recs = [r for r in recommendations if r["type"] == "coverage_gap"]
    assert gap_recs[0]["priority"] == "high"
    assert "Cyber Law" in gap_recs[0]["message"]
    assert gap_recs[1]["priority"] == "medium"


def test_missing_document_recommendation() -> None:
    service = _service()
    recommendations = service._build_recommendations(
        overview={"total_queries": 10, "no_knowledge_source": 0, "low_confidence": 0, "rated": 0, "negative_feedback": 0},
        knowledge_gaps=[],
        missing_documents=[{"_id": "Motor Vehicles Act", "count": 4}],
        low_quality_retrieval=[],
        rating_summary=[],
    )
    assert len(recommendations) == 1
    assert recommendations[0]["type"] == "missing_document"
    assert "Motor Vehicles Act" in recommendations[0]["message"]
    assert recommendations[0]["priority"] == "high"


def test_negative_feedback_recommendation_only_above_threshold() -> None:
    service = _service()
    overview = {"total_queries": 10, "no_knowledge_source": 0, "low_confidence": 0, "rated": 10, "negative_feedback": 3}

    below_threshold = service._build_recommendations(
        overview=overview,
        knowledge_gaps=[],
        missing_documents=[],
        low_quality_retrieval=[],
        rating_summary=[{"_id": 1, "count": 1}, {"_id": 5, "count": 9}],
    )
    assert not any(r["type"] == "feedback" for r in below_threshold)

    above_threshold = service._build_recommendations(
        overview=overview,
        knowledge_gaps=[],
        missing_documents=[],
        low_quality_retrieval=[],
        rating_summary=[{"_id": 1, "count": 3}, {"_id": 5, "count": 7}],
    )
    feedback_recs = [r for r in above_threshold if r["type"] == "feedback"]
    assert len(feedback_recs) == 1
    assert "3/10" in feedback_recs[0]["message"]


def test_coverage_recommendation_only_above_15_percent_missing() -> None:
    service = _service()

    below_threshold = service._build_recommendations(
        overview={"total_queries": 100, "no_knowledge_source": 10, "low_confidence": 0, "rated": 0, "negative_feedback": 0},
        knowledge_gaps=[],
        missing_documents=[],
        low_quality_retrieval=[],
        rating_summary=[],
    )
    assert not any(r["type"] == "coverage" for r in below_threshold)

    above_threshold = service._build_recommendations(
        overview={"total_queries": 100, "no_knowledge_source": 20, "low_confidence": 0, "rated": 0, "negative_feedback": 0},
        knowledge_gaps=[],
        missing_documents=[],
        low_quality_retrieval=[],
        rating_summary=[],
    )
    coverage_recs = [r for r in above_threshold if r["type"] == "coverage"]
    assert len(coverage_recs) == 1
    assert "20%" in coverage_recs[0]["message"]


def test_no_recommendations_when_everything_healthy() -> None:
    service = _service()
    recommendations = service._build_recommendations(
        overview={"total_queries": 50, "no_knowledge_source": 1, "low_confidence": 0, "rated": 20, "negative_feedback": 0},
        knowledge_gaps=[],
        missing_documents=[],
        low_quality_retrieval=[],
        rating_summary=[{"_id": 5, "count": 20}],
    )
    assert recommendations == []


# Phase 1 "Intent Analytics"


def test_intent_analytics_aggregates_all_required_fields() -> None:
    service = _service()
    service.intent_events = MagicMock()
    service.intent_events.collection.count_documents = AsyncMock(return_value=42)
    service.intent_events.top_intents = AsyncMock(return_value=[{"_id": "Document Analysis", "count": 10}])
    service.intent_events.low_confidence_intents = AsyncMock(return_value=[])
    service.intent_events.correction_rate = AsyncMock(return_value={"total": 42, "corrections": 4, "rate": 0.0952})
    service.intent_events.multi_intent_frequency = AsyncMock(return_value={"total": 42, "multi_intent": 6, "rate": 0.1429})
    service.intent_events.workflow_chain_usage = AsyncMock(
        return_value=[{"_id": ["Document Analysis", "Draft Generation"], "count": 3}]
    )
    service.intent_events.classifier_source_breakdown = AsyncMock(
        return_value=[{"_id": "deterministic", "count": 30}, {"_id": "llm", "count": 12}]
    )
    service.intent_feedback = MagicMock()
    service.intent_feedback.wrong_intent_rate = AsyncMock(return_value={"wrong_intent_feedback_count": 2, "rate": 0.0476})
    service.intent_feedback.top_corrected_intents = AsyncMock(return_value=[])

    result = asyncio.run(service._intent_analytics(datetime.now(UTC)))

    assert result["most_common_intents"][0]["_id"] == "Document Analysis"
    assert result["correction_rate"]["corrections"] == 4
    assert result["multi_intent_frequency"]["multi_intent"] == 6
    assert result["workflow_chain_usage"][0]["_id"] == ["Document Analysis", "Draft Generation"]
    assert result["classifier_source_breakdown"] == [{"_id": "deterministic", "count": 30}, {"_id": "llm", "count": 12}]
    assert result["wrong_intent_feedback"]["wrong_intent_feedback_count"] == 2
    # `total_intent_events` (second positional arg) must come from the same
    # window's `intent_events` count, not an unrelated/default value.
    assert service.intent_feedback.wrong_intent_rate.await_args.args[1] == 42
