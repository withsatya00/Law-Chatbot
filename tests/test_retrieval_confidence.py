"""Phase 2 Milestone F: retrieval quality and structured confidence.

The rule this module defends: **no verified evidence, no high-confidence legal
answer.** A fluent answer over thin sources and a hedged answer over excellent
sources both land mid-scale on a single blended number, and a reader cannot
tell which they have. Splitting the dimensions makes the binding constraint
visible, and making grounding a hard ceiling stops strong retrieval plus a
confident classifier from papering over an answer nobody can check.
"""

import pytest

from app.rag.confidence import (
    HIGH_CONFIDENCE_FLOOR,
    combine,
    model_confidence,
    retrieval_confidence,
    source_grounding_confidence,
)
from app.schemas.common import SourceCitation
from app.services.chat_service import _applicable_law_from, _confidence_fields


def _citation(**overrides: object) -> SourceCitation:
    base: dict[str, object] = {"source_document": "bns.pdf", "act_name": "Bharatiya Nyaya Sanhita", "section": "318"}
    base.update(overrides)
    return SourceCitation(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dimensions in isolation
# ---------------------------------------------------------------------------


def test_retrieval_confidence_reflects_the_top_chunks() -> None:
    assert retrieval_confidence([]) == 0.0
    assert retrieval_confidence([0.0, 0.0]) == 0.0
    assert retrieval_confidence([0.9, 0.85, 0.8]) > retrieval_confidence([0.3, 0.25, 0.2])


def test_retrieval_confidence_ignores_a_weak_tail() -> None:
    """A long tail of poor chunks says nothing about whether the question was
    answerable, and averaging over it punishes a broad recall pass."""
    strong = retrieval_confidence([0.9, 0.88, 0.86])
    with_tail = retrieval_confidence([0.9, 0.88, 0.86, 0.01, 0.01, 0.01])
    assert strong == with_tail


def test_grounding_is_zero_when_the_answer_is_not_grounded() -> None:
    """There is no partial credit. The question this answers is "can a reader
    check this?", and that is either yes or no."""
    assert source_grounding_confidence(citations=[_citation()], grounded=False) == 0.0


def test_grounding_is_zero_without_citations() -> None:
    assert source_grounding_confidence(citations=[], grounded=True) == 0.0


def test_identifiable_citations_ground_better_than_opaque_ones() -> None:
    identifiable = source_grounding_confidence(citations=[_citation()], grounded=True)
    opaque = source_grounding_confidence(
        citations=[SourceCitation(source_document="internal_upload_42.pdf")], grounded=True
    )
    assert identifiable > opaque


def test_page_evidence_raises_grounding_confidence() -> None:
    """A page reference is the strongest checkability available: a reader can
    open that page."""
    without = source_grounding_confidence(citations=[_citation()], grounded=True)
    with_page = source_grounding_confidence(
        citations=[_citation(page_number=42, page_start=42, page_end=42)], grounded=True
    )
    assert with_page > without


def test_verified_sources_raise_grounding_confidence() -> None:
    plain = source_grounding_confidence(citations=[_citation()], grounded=True)
    verified = source_grounding_confidence(
        citations=[_citation(verification_status="verified")], grounded=True, verified_sources=1
    )
    assert verified > plain


def test_a_quality_gate_intervention_halves_model_confidence() -> None:
    passed = model_confidence(intent_confidence=0.9, entity_confidence=0.9, quality_passed=True)
    failed = model_confidence(intent_confidence=0.9, entity_confidence=0.9, quality_passed=False)
    assert failed == pytest.approx(passed / 2, abs=0.01)


# ---------------------------------------------------------------------------
# Grounding is a ceiling, not a term
# ---------------------------------------------------------------------------


def test_ungrounded_answers_can_never_be_confident() -> None:
    """The failure this whole module exists to prevent: excellent retrieval and
    a certain classifier producing a confident number for an answer that rests
    on nothing."""
    result = combine(retrieval=1.0, source_grounding=0.0, model=1.0)
    assert result.overall == 0.0
    assert result.is_high is False
    assert "not supported by identifiable retrieved sources" in result.reason


def test_overall_never_exceeds_grounding() -> None:
    for grounding in (0.0, 0.2, 0.5, 0.8, 1.0):
        result = combine(retrieval=1.0, source_grounding=grounding, model=1.0)
        assert result.overall <= grounding + 1e-9


def test_strong_grounding_and_retrieval_reaches_high_confidence() -> None:
    result = combine(retrieval=0.9, source_grounding=0.9, model=0.9)
    assert result.is_high is True
    assert result.overall >= HIGH_CONFIDENCE_FLOOR


def test_the_reason_names_the_binding_constraint() -> None:
    """A reason that just says "low confidence" tells a reader nothing they
    cannot read off the number."""
    assert "retrieval" in combine(retrieval=0.2, source_grounding=0.9, model=0.9).reason.lower()
    assert "grounding" in combine(retrieval=0.9, source_grounding=0.4, model=0.9).reason.lower()
    assert "classification" in combine(retrieval=0.9, source_grounding=0.95, model=0.1).reason.lower()


def test_the_breakdown_serializes_every_dimension() -> None:
    payload = combine(retrieval=0.8, source_grounding=0.7, model=0.6).as_dict()
    assert set(payload) == {
        "retrieval_confidence", "source_grounding_confidence",
        "model_confidence", "overall_confidence", "confidence_reason",
    }


# ---------------------------------------------------------------------------
# Response wiring
# ---------------------------------------------------------------------------


def test_confidence_fields_never_raise_the_pipelines_own_number() -> None:
    """Phase 1's rules -- the quality gate, the language-mismatch cap, the
    no-verified-context short-circuit -- keep deciding the headline number.
    The dimensions explain it; they must not override it upward."""
    fields = _confidence_fields([_citation(verification_status="verified")], [], 0.20)
    assert fields["overall_confidence"] <= 0.20


def test_confidence_fields_are_zero_without_sources() -> None:
    fields = _confidence_fields([], [], 0.95)
    assert fields["source_grounding_confidence"] == 0.0
    assert fields["overall_confidence"] == 0.0


def test_a_cache_hit_reports_no_retrieval_this_turn_but_keeps_grounding() -> None:
    """A cache hit skips retrieval by design. Reporting retrieval as 0 is
    honest; reporting grounding as 0 would wrongly mark a cached, well-sourced
    answer as uncheckable."""
    fields = _confidence_fields([_citation()], [], 0.8)
    assert fields["retrieval_confidence"] == 0.0
    assert fields["source_grounding_confidence"] > 0.0


def test_the_chat_response_defaults_every_dimension_to_zero() -> None:
    from app.schemas.chat import ChatResponse
    from app.schemas.common import LawyerRecommendation

    response = ChatResponse(
        answer="a", sources=[], confidence=0.5,
        lawyer_recommendation=LawyerRecommendation(category="c", confidence=0.5, reason="r"),
        detected_language="english", detected_intent="General", latency_ms=1.0,
    )
    assert response.retrieval_confidence == 0.0
    assert response.source_grounding_confidence == 0.0
    assert response.model_confidence == 0.0
    assert response.overall_confidence == 0.0
    # Structured fields default empty, never invented.
    assert response.applicable_law == []
    assert response.risks == []
    assert response.next_steps == []


def test_existing_response_fields_are_unchanged() -> None:
    """Backward compatibility: Phase 1 clients read these names."""
    from app.schemas.chat import ChatResponse

    for field in ("answer", "sources", "confidence", "confidence_label", "confidence_reason", "disclaimer"):
        assert field in ChatResponse.model_fields


# ---------------------------------------------------------------------------
# Unsupported citations are rejected
# ---------------------------------------------------------------------------


def test_applicable_law_comes_from_citation_metadata_not_the_answer_text() -> None:
    """A model naming "Section 420 IPC" in prose is not evidence that any
    retrieved source says so. Echoing it into a structured field would launder
    an unsupported citation into something that looks authoritative."""
    assert _applicable_law_from([]) == []


def test_applicable_law_lists_only_identified_provisions() -> None:
    entries = _applicable_law_from([
        _citation(act_name="Bharatiya Nyaya Sanhita", section="318"),
        SourceCitation(source_document="unnamed_upload.pdf"),
    ])
    assert entries == ["Bharatiya Nyaya Sanhita — Section 318"]


def test_applicable_law_deduplicates() -> None:
    entries = _applicable_law_from([_citation(), _citation()])
    assert len(entries) == 1


def test_applicable_law_handles_an_act_without_a_section() -> None:
    entries = _applicable_law_from([SourceCitation(source_document="rti.pdf", act_name="Right to Information Act")])
    assert entries == ["Right to Information Act"]


def test_an_adversarial_fake_citation_earns_no_grounding() -> None:
    """A source document naming a case that does not exist still cannot ground
    an answer unless it identifies a provision -- and even then, grounding is
    capped by whether validation passed at all."""
    fake = SourceCitation(source_document="Sharma_v_Union_of_India_2029.pdf")
    assert source_grounding_confidence(citations=[fake], grounded=False) == 0.0
    assert _applicable_law_from([fake]) == []
