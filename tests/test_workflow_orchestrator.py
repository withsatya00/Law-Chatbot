"""Phase 1 "Multi-Intent Workflow Orchestration": pure-function tests for
the chain allowlist and the conservative fact-to-draft-field mapper. Both
functions have no I/O, so these are plain unit tests -- the cross-step
execution itself (document ownership, `DraftConversationEngine` seeding)
is covered in `test_chat_service_routing.py`.
"""

from app.drafting.templates import get_template
from app.schemas.common import LawyerRecommendation
from app.schemas.document import DocumentAnalysisResponse
from app.services.workflow_orchestrator import detect_chain, map_facts_to_draft_fields


def _analysis(**overrides: object) -> DocumentAnalysisResponse:
    base = {
        "executive_summary": "The tenant vacated without paying the last two months' rent.",
        "legal_summary": "",
        "important_clauses": [],
        "important_dates": [],
        "important_names": [],
        "important_sections": [],
        "key_risks": [],
        "action_items": [],
        "missing_information": [],
        "structured_data": {},
        "sources": [],
        "recommended_lawyer": LawyerRecommendation(category="General", confidence=0.5, reason="n/a"),
        "confidence": 0.7,
    }
    base.update(overrides)
    return DocumentAnalysisResponse(**base)


# ---------------------------------------------------------------------------
# detect_chain -- allowlist only, never an arbitrary sequence
# ---------------------------------------------------------------------------


def test_document_analysis_then_draft_generation_is_detected() -> None:
    chain = detect_chain(("Document Analysis", "Draft Generation"))
    assert chain == ("Document Analysis", "Draft Generation")


def test_chain_order_is_canonical_regardless_of_classifier_primary_order() -> None:
    # Classifier ranked "Draft Generation" as primary (first in the tuple),
    # but a draft can't be grounded in facts that haven't been analyzed
    # yet -- the returned order must still be analysis-first.
    chain = detect_chain(("Draft Generation", "Document Analysis"))
    assert chain == ("Document Analysis", "Draft Generation")


def test_document_analysis_then_legal_research_resolves_the_real_rag_intent() -> None:
    chain = detect_chain(("Document Analysis", "Legal Explanation"))
    assert chain == ("Document Analysis", "Legal Explanation")


def test_legal_research_then_draft_generation_is_detected() -> None:
    chain = detect_chain(("Legal Advice", "Draft Generation"))
    assert chain == ("Legal Advice", "Draft Generation")


def test_translation_then_response_modification_is_detected() -> None:
    chain = detect_chain(("Translation", "Response Modification"))
    assert chain == ("Translation", "Response Modification")


def test_single_intent_never_forms_a_chain() -> None:
    assert detect_chain(("Document Analysis",)) is None
    assert detect_chain(()) is None


def test_unlisted_intent_pair_is_not_a_chain() -> None:
    # Real pair the classifier could plausibly produce, but not on the
    # allowlist -- must never be treated as a workflow.
    assert detect_chain(("Lawyer Recommendation", "Summarization")) is None


def test_arbitrary_three_intent_llm_output_only_matches_an_allowlisted_pair() -> None:
    # Guards against ever executing something beyond the fixed allowlist
    # even when the classifier's own multi-intent output has more than two
    # candidates -- only a genuinely allowlisted pair inside it counts.
    chain = detect_chain(("Document Analysis", "Draft Generation", "Lawyer Recommendation"))
    assert chain == ("Document Analysis", "Draft Generation")


# ---------------------------------------------------------------------------
# map_facts_to_draft_fields -- conservative, no fabrication
# ---------------------------------------------------------------------------


def test_summary_is_mapped_to_the_narrative_field_when_present() -> None:
    template = get_template("legal_notice")
    mapped = map_facts_to_draft_fields(template, _analysis())
    assert mapped["facts"] == "The tenant vacated without paying the last two months' rent."


def test_no_summary_means_no_narrative_field_mapped() -> None:
    template = get_template("legal_notice")
    mapped = map_facts_to_draft_fields(template, _analysis(executive_summary=""))
    assert "facts" not in mapped


def test_single_amount_is_mapped_when_template_has_an_amount_field() -> None:
    template = get_template("recovery_notice")
    amount_keys = [key for key in template.field_keys() if "amount" in key]
    assert amount_keys, "recovery_notice must have an amount-ish field for this test to be meaningful"
    mapped = map_facts_to_draft_fields(
        template, _analysis(structured_data={"entities": {"amount": ["50,000"]}})
    )
    assert mapped[amount_keys[0]] == "50,000"


def test_multiple_amounts_are_never_mapped_ambiguously() -> None:
    template = get_template("recovery_notice")
    mapped = map_facts_to_draft_fields(
        template, _analysis(structured_data={"entities": {"amount": ["50,000", "75,000"]}})
    )
    assert not any("amount" in key for key in mapped)


def test_person_names_are_never_mapped_no_role_classification_exists() -> None:
    # Two names with no way to tell applicant from respondent -- must be
    # left for the user to answer directly, never guessed.
    template = get_template("legal_notice")
    mapped = map_facts_to_draft_fields(
        template, _analysis(important_names=["Ramesh Kumar", "Suresh Sharma"])
    )
    assert "applicant_name" not in mapped
    assert "respondent_name" not in mapped
