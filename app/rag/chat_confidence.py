"""Chat-response confidence scoring and the `ChatResponse` confidence-field
breakdown.

Extracted out of `app.services.chat_service.ChatService` (Phase 1 of the
god-object split). Deliberately a sibling of `app.rag.confidence` (the
three-dimension retrieval/grounding/model math), not an addition to it --
`_ConfidenceFields`/`confidence_fields` glue those numbers into a
`ChatResponse`'s kwargs, and `confidence_score`/`confidence_label`/
`confidence_reason` are an older, unrelated blended-scalar scheme, both
chat-orchestration-specific concerns rather than the pure math
`app.rag.confidence` owns.
"""
from typing import Any, TypedDict

from app.rag.confidence import combine, retrieval_confidence, source_grounding_confidence
from app.schemas.common import RetrievedChunk, SourceCitation


class _ConfidenceFields(TypedDict):
    """The exact `ChatResponse` keys `_confidence_fields` fills.

    A `TypedDict` rather than a plain dict so `**` into the response model is
    type-checked: a renamed field is then a type error here instead of a
    silently-ignored extra key at runtime.
    """

    retrieval_confidence: float
    source_grounding_confidence: float
    model_confidence: float
    overall_confidence: float


def _confidence_fields(
    sources: list[SourceCitation],
    ranked: list[RetrievedChunk],
    overall: float,
) -> "_ConfidenceFields":
    """The confidence dimensions for one response, as response kwargs.

    Derived at each response site from the citations and chunks already in
    scope there, rather than threaded through three method signatures.

    `overall` is the scalar the existing pipeline already computed, and is
    passed through unchanged -- every Phase 1 rule that caps or zeroes it (the
    quality gate, the language-mismatch cap, the no-verified-context
    short-circuit) keeps deciding the headline number. The dimensions explain
    that number; they do not recompute or override it.

    Grounding is inferred from whether identifiable citations survived rather
    than re-running validation: an answer that reached a response site with
    citations attached has already passed `validate_grounding`, and re-deriving
    it here could disagree with the number the caller acted on.
    """
    breakdown = combine(
        retrieval=retrieval_confidence([chunk.score for chunk in ranked]),
        source_grounding=source_grounding_confidence(
            citations=sources,
            grounded=bool(sources),
            verified_sources=sum(
                1 for citation in sources if citation.verification_status == "verified"
            ),
        ),
        # No separate model signal is available at the response sites, so the
        # already-computed overall stands in for it. It only carries 20% weight
        # and is capped by grounding below, so this cannot inflate the result.
        model=overall,
    )
    return _ConfidenceFields(
        retrieval_confidence=breakdown.retrieval,
        source_grounding_confidence=breakdown.source_grounding,
        model_confidence=breakdown.model,
        overall_confidence=min(overall, breakdown.source_grounding),
    )


def confidence_score(
    chunks: list[RetrievedChunk], intent_confidence: float, entity_confidence: float
) -> float:
    retrieval = sum(chunk.score for chunk in chunks[:3]) / max(min(len(chunks), 3), 1)
    return round(max(0.0, min(1.0, 0.55 * retrieval + 0.3 * intent_confidence + 0.15 * entity_confidence)), 2)


def confidence_label(value: float) -> str:
    # Calibrated against the actual `confidence_score` distribution, not
    # round numbers: most well-formed conceptual questions fall outside
    # `IntentDetector`'s ~12 keyword buckets (fixed 0.45 fallback) and
    # `EntityExtractor` floors at 0.55 with zero entities found, so a
    # genuinely good retrieval match commonly blends to ~0.55-0.65, not
    # 0.9+. A naive >=0.7 threshold would make most correct, grounded
    # answers structurally incapable of ever showing "High."
    if value >= 0.6:
        return "High"
    if value >= 0.35:
        return "Medium"
    return "Low"


def confidence_reason(ranked: list[Any], subject_matter_intent: str) -> str:
    top = ranked[0]
    act = top.metadata.get("act_name") or "the retrieved legal source"
    count = len(ranked)
    reason = f"Based on {act} and {count} matching provision{'s' if count != 1 else ''} from the retrieved legal material."
    if subject_matter_intent == "General Legal Query":
        reason += " The specific sub-topic wasn't one of the classifier's known categories, but the retrieval match itself is grounded in verified text."
    return reason
