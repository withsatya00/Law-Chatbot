"""Direct unit tests for `app.rag.chat_confidence` (Phase 1 god-object split
out of `ChatService`). `test_retrieval_confidence.py` already covers
`_confidence_fields` via its re-export from `app.services.chat_service`;
these cover the older, unrelated blended-scalar scheme
(`confidence_score`/`confidence_label`/`confidence_reason`) directly.
"""
from app.rag.chat_confidence import confidence_label, confidence_reason, confidence_score
from app.schemas.common import RetrievedChunk


def _chunk(score: float, **metadata: object) -> RetrievedChunk:
    return RetrievedChunk(chunk_id="c", text="text", score=score, metadata=metadata)


def test_confidence_score_blends_retrieval_intent_and_entity() -> None:
    chunks = [_chunk(0.8), _chunk(0.6), _chunk(0.4)]
    score = confidence_score(chunks, intent_confidence=1.0, entity_confidence=1.0)
    # retrieval = avg(0.8, 0.6, 0.4) = 0.6; 0.55*0.6 + 0.3*1.0 + 0.15*1.0 = 0.78
    assert score == 0.78


def test_confidence_score_clamped_to_zero_one() -> None:
    assert confidence_score([], intent_confidence=0.0, entity_confidence=0.0) == 0.0
    assert confidence_score([_chunk(1.0)], intent_confidence=1.0, entity_confidence=1.0) <= 1.0


def test_confidence_label_thresholds() -> None:
    assert confidence_label(0.6) == "High"
    assert confidence_label(0.59) == "Medium"
    assert confidence_label(0.35) == "Medium"
    assert confidence_label(0.34) == "Low"


def test_confidence_reason_cites_act_and_count() -> None:
    ranked = [_chunk(0.9, act_name="Negotiable Instruments Act"), _chunk(0.5, act_name="Negotiable Instruments Act")]
    reason = confidence_reason(ranked, subject_matter_intent="Cheque Bounce")
    assert "Negotiable Instruments Act" in reason
    assert "2 matching provisions" in reason


def test_confidence_reason_adds_disclaimer_for_general_legal_query() -> None:
    ranked = [_chunk(0.5)]
    reason = confidence_reason(ranked, subject_matter_intent="General Legal Query")
    assert "classifier's known categories" in reason
