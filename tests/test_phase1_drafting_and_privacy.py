"""Phase 1 items 4, 6, 7 and 8: the drafting state manager, the fact-only
draft guard, privacy controls, and import-time reliability.

The drafting-state tests are written against the exact phrasings the phase
specifies as equivalent, because "these must work equivalently" is the
requirement -- a test that only exercises the canonical "continue draft"
would pass while every natural phrasing of it failed, which is precisely the
state this phase found.
"""

import asyncio
import importlib
import sys
from unittest.mock import AsyncMock

import pytest

from app.core.optional_deps import (
    MissingOptionalDependencyError,
    is_available,
    load,
    missing_optional_dependencies,
)
from app.drafting.conversation import (
    _LIST_DRAFTS_PATTERN,
    _NEW_DRAFT_PATTERN,
    _SWITCH_DRAFT_PATTERN,
    DraftConversationEngine,
    _is_resume_draft_request,
    _switched_template_name,
)
from app.drafting.fact_audit import audit_draft
from app.schemas.chat import ChatRequest
from app.schemas.common import LawyerRecommendation
from app.services.chat_service import _BARE_REFUSAL_PATTERN, _CANCEL_DRAFT_PATTERN

_RECOMMENDATION = LawyerRecommendation(category="Criminal Law", confidence=0.6, reason="Test fixture.")


def _service_with_draft(memory: dict | None = None):
    from app.services.chat_service import ChatService

    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])
    base = {
        "messages": [], "summary": "", "current_intent": None, "legal_category": None,
        "draft_mode": True, "draft_stage": "collecting", "draft_template_id": "bank_fraud_complaint",
        "draft_fields": {}, "draft_id": None, "draft_language": "english",
    }
    base.update(memory or {})
    service._memory_state = base
    service.memory.append = AsyncMock(return_value=base)
    service.memory.load = AsyncMock(return_value=base)
    service.memory.check_access = AsyncMock(return_value=base)
    service.memory.update = AsyncMock(return_value={})
    service.memory.summarize_if_needed = AsyncMock(return_value={})
    service.history.insert = AsyncMock(return_value="history-id")
    service.query_log.insert = AsyncMock(return_value="query-log-id")
    service.intent_events.insert = AsyncMock(return_value="intent-event-id")
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    service.recommendations.recommend = AsyncMock(return_value=_RECOMMENDATION)
    service.retriever.retrieve = AsyncMock(return_value=("q", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service._generate_related_questions = AsyncMock(return_value=[])
    return service


def _collecting_memory(**overrides):
    memory = {
        "draft_mode": True,
        "draft_stage": "collecting",
        "draft_template_id": "bank_fraud_complaint",
        "draft_fields": {"applicant_name": "Rahul Sharma"},
        "draft_id": None,
        "draft_language": "english",
    }
    memory.update(overrides)
    return memory


# ---------------------------------------------------------------------------
# Item 4 -- natural draft commands, not exact strings
# ---------------------------------------------------------------------------


# The exact phrasings the phase lists as needing to work equivalently.
_RESUME_PHRASINGS = [
    "continue draft",
    "mujhe draft continue karna hai",
    "resume my complaint",
    "wahi draft khol do",
    "draft chalu karo",
]


@pytest.mark.parametrize("message", _RESUME_PHRASINGS)
def test_every_specified_resume_phrasing_is_recognized(message: str) -> None:
    assert _is_resume_draft_request(message), message


@pytest.mark.parametrize("message", _RESUME_PHRASINGS)
def test_every_resume_phrasing_actually_resumes_a_paused_draft(message: str) -> None:
    # Recognising the phrase is only half of it -- it has to bring the paused
    # draft back with its collected fields intact.
    engine = DraftConversationEngine()
    memory = _collecting_memory(draft_paused=True)
    result = asyncio.run(engine.handle_turn("s", message, "english", memory))
    assert result is not None, f"{message!r} did not re-engage the draft"
    assert memory["draft_paused"] is False
    assert memory["draft_fields"]["applicant_name"] == "Rahul Sharma"
    assert memory["draft_template_id"] == "bank_fraud_complaint"


@pytest.mark.parametrize(
    "message",
    [
        "मुझे ड्राफ्ट जारी रखना है", "मसौदा जारी रखें", "draft aage badhao",
        "pehle wala draft dikhao", "open my application", "draft resume karo",
    ],
)
def test_additional_natural_resume_phrasings(message: str) -> None:
    assert _is_resume_draft_request(message), message


@pytest.mark.parametrize(
    "message",
    ["Rahul Sharma", "24, Shanti Vihar, Gomti Nagar, Lucknow", "What is an FIR?", "जमानत क्या है", "cancel draft"],
)
def test_field_values_and_questions_are_not_mistaken_for_resume(message: str) -> None:
    assert not _is_resume_draft_request(message), message


@pytest.mark.parametrize(
    "message",
    ["cancel draft", "cancel kr do", "cancel karo", "draft hata do", "hata do", "cancel kr do draft ko",
     "ड्राफ्ट रद्द करें", "मसौदा हटा दें"],
)
def test_every_specified_cancel_phrasing_is_recognized(message: str) -> None:
    assert _CANCEL_DRAFT_PATTERN.search(message), message


@pytest.mark.parametrize("message", ["no", "nahi", "nhi", "नहीं", "Nope"])
def test_every_specified_refusal_is_recognized(message: str) -> None:
    assert _BARE_REFUSAL_PATTERN.match(message), message


@pytest.mark.parametrize(
    "message",
    ["show my drafts", "my drafts", "list drafts", "mere drafts dikhao", "मेरे ड्राफ्ट दिखाओ"],
)
def test_show_my_drafts_phrasings(message: str) -> None:
    assert _LIST_DRAFTS_PATTERN.match(message), message


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("switch to cyber complaint", "cyber complaint"),
        ("switch to cyber complaint draft", "cyber complaint"),
        ("switch to police complaint", "police complaint"),
        ("cyber complaint pe switch karo", "cyber complaint"),
    ],
)
def test_switch_to_a_named_template_without_the_word_draft(message: str, expected: str) -> None:
    match = _SWITCH_DRAFT_PATTERN.match(message)
    assert match is not None, message
    assert _switched_template_name(match) == expected


def test_switch_actually_restores_the_parked_draft_it_names() -> None:
    engine = DraftConversationEngine()
    memory = _collecting_memory(
        parked_drafts=[{
            "draft_stage": "collecting",
            "draft_template_id": "cyber_crime_complaint",
            "draft_fields": {"applicant_name": "Amol"},
            "draft_id": None,
            "draft_language": "english",
        }]
    )
    result = asyncio.run(engine.handle_turn("s", "switch to cyber complaint", "english", memory))
    assert result is not None
    assert memory["draft_template_id"] == "cyber_crime_complaint"
    assert memory["draft_fields"]["applicant_name"] == "Amol"
    # The one that was active is parked, not lost.
    assert any(p["draft_template_id"] == "bank_fraud_complaint" for p in memory["parked_drafts"])


@pytest.mark.parametrize(
    "message", ["start another draft", "start a new draft", "naya draft banao", "ek aur draft banao"],
)
def test_start_another_draft_parks_the_current_one_and_shows_the_menu(message: str) -> None:
    engine = DraftConversationEngine()
    memory = _collecting_memory()
    result = asyncio.run(engine.handle_turn("s", message, "english", memory))
    assert result is not None
    assert result.info.stage == "selecting"
    # The half-filled draft is recoverable, not discarded.
    assert [p["draft_template_id"] for p in memory["parked_drafts"]] == ["bank_fraud_complaint"]
    assert _NEW_DRAFT_PATTERN.match(message)
    # ...and it is not answered with "I don't have a template for that".
    assert "don't have a template" not in result.reply_text.lower()


def test_an_unrelated_legal_question_is_answered_and_the_draft_stays_paused() -> None:
    # The phase's own wording: "An active draft must never trap the user. An
    # unrelated legal question should be answered normally and the draft
    # should remain safely paused."
    service = _service_with_draft(_collecting_memory())
    response = asyncio.run(service.answer(ChatRequest(question="What is the difference between bail and "
                                                              "anticipatory bail?")))
    memory = service._memory_state
    # Answered through the normal chat path, not by re-showing the field list.
    assert "Applicant Address" not in response.answer
    # ...and the draft survives, paused, with its fields.
    assert memory["draft_mode"] is True
    assert memory["draft_paused"] is True
    assert memory["draft_fields"]["applicant_name"] == "Rahul Sharma"


def test_a_paused_draft_then_resumes_on_a_natural_phrasing() -> None:
    engine = DraftConversationEngine()
    memory = _collecting_memory()
    engine.pause(memory)
    # An unrelated question falls through to normal chat...
    assert asyncio.run(engine.handle_turn("s", "What is anticipatory bail?", "english", memory)) is None
    # ...and a natural resume phrasing brings it back.
    assert asyncio.run(engine.handle_turn("s", "resume my complaint", "english", memory)) is not None
    assert memory["draft_paused"] is False


# ---------------------------------------------------------------------------
# Item 6 -- fact-only draft guard
# ---------------------------------------------------------------------------


_USER_FIELDS = {
    "applicant_name": "राहुल शर्मा",
    "facts": (
        "दिनांक 25 अगस्त 2026 को मुझे एक व्यक्ति ने फोन करके ऑनलाइन निवेश में अधिक लाभ का झांसा दिया। "
        "मैंने ₹35,000 ट्रांसफर किए। मुझे संदेह है कि मेरे साथ धोखाधड़ी हुई है।"
    ),
    "expected_relief": "मेरी शिकायत दर्ज की जाए और जांच की जाए",
}


def test_audit_flags_a_conclusion_the_user_never_asserted() -> None:
    audit = audit_draft(
        {"Facts of the Case": "उक्त व्यक्ति द्वारा सुनियोजित तरीके से ऑनलाइन धोखाधड़ी की गई है।"},
        _USER_FIELDS,
        category="Complaint",
    )
    assert any(finding.category == "unsupported_conclusion" for finding in audit.findings)


def test_audit_flags_an_escalation_threat_the_user_never_asked_for() -> None:
    audit = audit_draft(
        {"Prayer": "यदि कार्यवाही नहीं हुई तो मैं उच्चाधिकारियों के समक्ष जाने के लिए बाध्य होऊंगा।"},
        _USER_FIELDS,
        category="Complaint",
    )
    assert any(finding.category == "unrequested_escalation" for finding in audit.findings)


def test_a_notice_may_state_a_consequence_without_being_flagged() -> None:
    # A notice's whole purpose is to state what follows from non-compliance.
    audit = audit_draft(
        {"Prayer": "If the amount is not paid within 15 days, legal proceedings will be initiated."},
        {"facts": "The cheque for 50000 was dishonoured.", "expected_relief": "Recover the amount"},
        category="Notice",
    )
    assert not any(finding.category == "unrequested_escalation" for finding in audit.findings)


def test_audit_flags_a_figure_the_user_never_supplied() -> None:
    audit = audit_draft(
        {"Consequences": "प्रार्थी को ₹45,000 की प्रत्यक्ष आर्थिक हानि हुई है।"},
        _USER_FIELDS,
        category="Complaint",
    )
    findings = [f for f in audit.findings if f.category == "unsupported_figure"]
    assert findings
    assert "45,000" in findings[0].detail


def test_a_correctly_copied_figure_is_not_flagged() -> None:
    audit = audit_draft(
        {"Consequences": "प्रार्थी को ₹35,000 की प्रत्यक्ष आर्थिक हानि हुई है।"},
        _USER_FIELDS,
        category="Complaint",
    )
    assert not any(finding.category == "unsupported_figure" for finding in audit.findings)


def test_a_statutory_citation_is_not_mistaken_for_an_invented_figure() -> None:
    audit = audit_draft(
        {"Legal Position": "The facts stated prima facie appear to attract Section 318 of the Bharatiya "
                           "Nyaya Sanhita, 2023."},
        {"facts": "I was cheated out of money by a caller."},
        category="Complaint",
    )
    assert not any(finding.category == "unsupported_figure" for finding in audit.findings)


def test_audit_flags_a_legal_position_stated_without_any_hedge() -> None:
    audit = audit_draft(
        {"Legal Position": "The incident falls under Section 318 of the Bharatiya Nyaya Sanhita, 2023."},
        {"facts": "I was cheated by a caller."},
        category="Complaint",
    )
    assert any(finding.category == "unhedged_legal_conclusion" for finding in audit.findings)


@pytest.mark.parametrize(
    "hedged",
    [
        "The facts stated prima facie appear to attract Section 318 of the Bharatiya Nyaya Sanhita, 2023.",
        "Section 318 of the Bharatiya Nyaya Sanhita, 2023 may be applicable, subject to verification.",
        "If established on investigation, the matter would fall under Section 318.",
    ],
)
def test_a_properly_hedged_legal_position_passes(hedged: str) -> None:
    audit = audit_draft({"Legal Position": hedged}, {"facts": "I was cheated by a caller."}, category="Complaint")
    assert not any(finding.category == "unhedged_legal_conclusion" for finding in audit.findings)


def test_a_faithful_draft_produces_no_findings_at_all() -> None:
    audit = audit_draft(
        {
            "Introduction": "प्रार्थी राहुल शर्मा इस शिकायत के माध्यम से एक घटना की सूचना दे रहे हैं।",
            "Facts of the Case": "प्रार्थी का कथन है कि उसने ₹35,000 की राशि स्थानांतरित की।",
            "Legal Position": "उपरोक्त तथ्य प्रथम दृष्टया धोखाधड़ी से संबंधित प्रावधानों को आकर्षित करते प्रतीत होते हैं।",
            "Prayer": "प्रार्थना है कि शिकायत दर्ज कर जांच की जाए।",
        },
        _USER_FIELDS,
        category="Complaint",
    )
    assert audit.ok, audit.as_dicts()


def test_audit_findings_reach_the_chat_preview_and_the_api_response() -> None:
    from app.schemas.drafting import DraftGenerateResponse, DraftTurnInfo

    # Both carrier shapes exist and default to empty, so a clean draft says
    # nothing rather than showing an empty warnings box.
    assert DraftGenerateResponse(template_id="t", template_name="T", status="complete").audit_findings == []
    assert DraftTurnInfo(stage="preview").audit_findings == []


# ---------------------------------------------------------------------------
# Item 7 -- privacy controls
# ---------------------------------------------------------------------------


def test_error_payloads_never_carry_personal_data() -> None:
    from app.core.exceptions import _error_payload

    payload = _error_payload(
        "validation_error",
        "The value '9876543210' is not a valid police station.",
        {"field": "facts", "value": "I live at 24, Shanti Vihar, Gomti Nagar, Lucknow and my PAN is ABCDE1234F"},
    )
    serialized = str(payload)
    assert "9876543210" not in serialized
    assert "Shanti Vihar" not in serialized
    assert "ABCDE1234F" not in serialized
    # ...and the error stays useful.
    assert payload["error"]["code"] == "validation_error"


def test_analytics_dashboard_masks_free_text_on_the_way_out() -> None:
    from app.services.analytics_service import _mask_analytics

    masked = _mask_analytics({
        "frequent_questions": [
            {"_id": "my number is 9876543210", "sample_question": "my number is 9876543210", "count": 3}
        ],
        "unanswered": [{"question": "PAN ABCDE1234F issue", "answer": "contact me at a@b.com"}],
        "overview": {"total_queries": 10},
    })
    serialized = str(masked)
    assert "9876543210" not in serialized
    assert "ABCDE1234F" not in serialized
    assert "a@b.com" not in serialized
    # Non-text analytics values are untouched, or the dashboard breaks.
    assert masked["overview"]["total_queries"] == 10
    assert masked["frequent_questions"][0]["count"] == 3


def test_deletion_controls_exist_for_draft_chat_and_account() -> None:
    # Asserted against the routers rather than `app.routes`: this build wraps
    # included routers in an opaque `_IncludedRouter`, so the flattened app
    # route list does not expose the paths.
    from app.api import drafting as drafting_api
    from app.api import history as history_api

    def _routes(router):
        return {(route.path, method) for route in router.routes for method in getattr(route, "methods", [])}

    assert ("/draft/{draft_id}", "DELETE") in _routes(drafting_api.router), "delete-draft control is missing"
    assert ("/session", "DELETE") in _routes(history_api.router), "delete-chat control is missing"
    assert ("/me/data", "DELETE") in _routes(history_api.router), "delete-user-data control is missing"


def test_delete_user_data_refuses_an_unauthenticated_caller() -> None:
    # The route acts on the CALLER's own id and takes no user_id parameter, so
    # it cannot be pointed at someone else's data -- but it must still refuse
    # when there is no caller identity at all.
    from app.api.history import delete_my_data
    from app.core.exceptions import UnauthorizedError

    with pytest.raises(UnauthorizedError):
        asyncio.run(delete_my_data(user_id=None))


def test_every_pii_bearing_repository_can_erase_by_session_and_by_user() -> None:
    from app.repositories.analytics import IntentEventRepository, QueryLogRepository
    from app.repositories.chat_history import ChatRepository, FeedbackRepository
    from app.repositories.drafts import DraftRepository

    for repository in (ChatRepository, QueryLogRepository, IntentEventRepository, FeedbackRepository, DraftRepository):
        assert hasattr(repository, "delete_by_session"), repository.__name__
        assert hasattr(repository, "delete_by_user"), repository.__name__


# ---------------------------------------------------------------------------
# Item 8 -- optional dependencies must not break chat or drafting
# ---------------------------------------------------------------------------


def test_chat_and_drafting_import_without_touching_document_libraries() -> None:
    # The regression this guards: `chat_service` -> `DocumentService` ->
    # `rag.pipeline` -> `rag.loader` -> `rag.ocr` used to import pypdf,
    # python-docx, odfpy, striprtf, BeautifulSoup, Pillow and pytesseract at
    # module scope, so a host missing any one of them could not start plain
    # legal chat -- a feature that never opens a document.
    import app.drafting.conversation
    import app.rag.loader
    import app.rag.ocr
    import app.services.chat_service  # noqa: F401

    for module in ("pypdf", "docx", "odf", "striprtf", "pytesseract", "PIL", "pdf2image", "weasyprint"):
        source = sys.modules["app.rag.loader"].__dict__
        assert module not in source, f"{module} is bound at import time in rag.loader"


def test_a_missing_optional_dependency_names_the_feature_and_the_fix() -> None:
    with pytest.raises(MissingOptionalDependencyError) as excinfo:
        load("a_library_that_does_not_exist", feature="PDF export")
    message = str(excinfo.value)
    assert "PDF export" in message
    assert "pip install" in message


def test_a_missing_optional_dependency_is_reported_not_raised_by_the_probe() -> None:
    assert is_available("json") is True
    assert is_available("a_library_that_does_not_exist") is False
    # The health-style summary never raises, whatever is installed.
    assert isinstance(missing_optional_dependencies(), dict)


def test_ocr_reports_a_missing_toolchain_as_a_handled_error() -> None:
    # Callers already handle `OcrUnavailableError` by degrading gracefully;
    # a missing library must arrive as that, not as a bare ImportError.
    from app.rag.ocr import OcrEngine, OcrUnavailableError

    engine = OcrEngine()
    original = sys.modules.pop("pytesseract", None)
    try:
        import app.core.optional_deps as deps

        deps._cache.pop("pytesseract", None)
        deps._failures["pytesseract"] = ImportError("simulated missing pytesseract")
        with pytest.raises(OcrUnavailableError):
            engine._tesseract()
    finally:
        deps._failures.pop("pytesseract", None)
        if original is not None:
            sys.modules["pytesseract"] = original
        importlib.invalidate_caches()


def test_audit_does_not_block_a_conclusion_phrased_as_a_condition() -> None:
    # "यदि यह सिद्ध होता है कि ... तो ... आ सकता है" is the hedge the audit asks
    # for; flagging it blocked export of a properly qualified Legal Position.
    audit = audit_draft(
        {"Legal Position": "यदि यह सिद्ध होता है कि राशि रोकी गई है, तो यह अनुबंध के उल्लंघन की श्रेणी में आ सकता है।"},
        _USER_FIELDS,
        category="Notice",
    )
    assert not any(f.category == "unsupported_conclusion" for f in audit.findings)
    unhedged = audit_draft({"Facts": "यह सिद्ध होता है कि उसने धोखा दिया।"}, _USER_FIELDS, category="Notice")
    assert any(f.category == "unsupported_conclusion" for f in unhedged.findings)


def test_audit_flags_invented_hardship_repeated_requests_and_deadline() -> None:
    audit = audit_draft(
        {
            "Facts": "बार-बार अनुरोध करने के बावजूद भुगतान नहीं हुआ और मुझे मानसिक तनाव हुआ।",
            "Demand": "इस नोटिस की प्राप्ति के 15 दिनों के भीतर भुगतान करें।",
        },
        _USER_FIELDS,
        category="Notice",
    )
    categories = {f.category for f in audit.findings}
    assert {"invented_detail:hardship", "invented_detail:repeated_requests", "unsupported_deadline"} <= categories


def test_place_and_demand_relief_are_derived_from_facts_the_user_gave() -> None:
    from app.drafting.field_extraction import infer_derivable_fields
    from app.drafting.templates import get_template

    template = get_template("demand_notice")
    got = infer_derivable_fields(
        template, {"applicant_address": "21, Andheri West, Mumbai - 400053", "dues_amount": "45,000"}
    )
    assert got["place"] == "Mumbai"
    assert "45,000" in got["expected_relief"]
    # Never overwrites, never guesses without evidence.
    assert infer_derivable_fields(template, {"place": "Pune", "applicant_address": "Mumbai"}) == {}


def test_strip_invented_details_removes_embellishments_and_neutralises_deadlines() -> None:
    from app.drafting.fact_audit import strip_invented_details

    fields = {"facts": "Deposit Rs 45,000 not returned. Keys handed over on 1 August 2026."}
    sections = {
        "Facts": (
            "The applicant paid Rs 45,000 as deposit. The flat was in good condition when the keys were handed over. "
            "Despite repeated requests nothing was paid. The applicant suffers financial inconvenience."
        ),
        "Demand": "Pay within 15 days of receipt. The applicant handed over keys on 1 August 2026.",
        "Hindi": "इस नोटिस की प्राप्ति के 15 दिनों के भीतर भुगतान करें।",
    }
    out = strip_invented_details(sections, fields)
    assert out["Facts"] == "The applicant paid Rs 45,000 as deposit."
    assert "15" not in out["Demand"] and "a reasonable time" in out["Demand"]
    assert "1 August 2026" in out["Demand"]
    assert "उचित समय" in out["Hindi"] and "15" not in out["Hindi"]
    # Something the user DID say is never stripped.
    kept = strip_invented_details({"Facts": "Pay within 15 days."}, {"facts": "give them 15 days"})
    assert "15 days" in kept["Facts"]
