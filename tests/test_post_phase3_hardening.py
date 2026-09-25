"""Post-Phase-3 hardening: regressions reproduced from an observed UAT session.

Every test here drives the SAME path `POST /chat` uses -- a real
`ChatService.answer()` with only I/O collaborators (memory store, history,
query log, cache, retriever, draft persistence) replaced. Routing, the draft
conversation engine, field extraction and the ChatOps orchestrator all run
for real, because those are exactly the layers the observed defects live in.

Session memory is a single mutable dict shared by the stubbed store, so a
multi-turn conversation behaves as it does in production.

All personal details below are synthetic.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.drafting.templates import get_template
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService


def _run(coro):
    return asyncio.run(coro)


def _service(memory: dict[str, Any]) -> ChatService:
    """A real `ChatService` whose I/O is stubbed and whose memory is `memory`.

    Retrieval is allowed (some assertions are precisely "this fell through to
    RAG"), but it returns nothing so no network or database is touched.
    """
    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])

    async def _load(session_id: str) -> dict[str, Any]:
        return memory

    async def _append(session_id: str, role: str, content: str) -> dict[str, Any]:
        memory.setdefault("messages", []).append({"role": role, "content": content})
        return memory

    async def _update(session_id: str, **values: Any) -> dict[str, Any]:
        memory.update(values)
        return memory

    service.memory.load = _load
    service.memory.append = _append
    service.memory.update = _update
    service.memory.summarize_if_needed = AsyncMock(return_value=memory)
    service.memory.append_intent_event = AsyncMock(return_value=memory)
    service.history.insert = AsyncMock(return_value="history-id")
    service.query_log.insert = AsyncMock(return_value="query-log-id")
    service.intent_events.insert = AsyncMock(return_value="intent-event-id")
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    # `retrieve` returns `(rewritten_query, chunks)`; an empty corpus is a
    # legitimate outcome and is what a "this fell through to RAG" assertion
    # observes.
    service.retriever.retrieve = AsyncMock(return_value=("", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    # The optional model-assisted extraction pass is not what these tests are
    # about, and leaving it live would make them depend on a provider.
    service.draft_conversation.extractor._extract_with_llm = AsyncMock(return_value={})
    return service


def _ask(service: ChatService, question: str):
    return _run(service.answer(ChatRequest(question=question, session_id="s-hardening")))


def _draft_fields(memory: dict[str, Any]) -> dict[str, str]:
    return dict(memory.get("draft_fields") or {})


# Synthetic labelled answers, in the shape the assistant itself asks for.
_HINDI_LABELLED = (
    "आवेदक का पता: 24, नेहरू नगर, गाज़ियाबाद, उत्तर प्रदेश – 201001 "
    "मोबाइल नंबर: 9812345670 "
    "आवेदक का नाम: अमित कुमार शर्मा "
    "स्थान: गाज़ियाबाद, उत्तर प्रदेश "
    "किराए की संपत्ति का पता: मकान नंबर 112, गली नंबर 4, शास्त्री नगर, गाज़ियाबाद – 201002 "
    "प्रतिवादी का पता: 78, सेक्टर 18, नोएडा, उत्तर प्रदेश – 201301 "
    "प्रतिवादी का नाम: राजेश वर्मा"
)

_ENGLISH_LABELLED = (
    "Applicant Name: Rahul Sharma "
    "Applicant Address: 123, Shastri Nagar, Ghaziabad, Uttar Pradesh - 201002 "
    "Mobile Number: 9812345671 "
    "Place: Ghaziabad, Uttar Pradesh "
    "Respondent / Opposite Party Name: ABC Electronics Pvt. Ltd. "
    "Respondent Address: 45, Nehru Place, New Delhi - 110019"
)


# ---------------------------------------------------------------------------
# A2 / A3 -- labelled multilingual extraction
# ---------------------------------------------------------------------------


def test_english_labelled_fields_are_extracted_separately() -> None:
    """Each labelled value must stop at the next label, not run through it."""
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")
    _ask(service, _ENGLISH_LABELLED)

    fields = _draft_fields(memory)
    assert fields.get("applicant_name") == "Rahul Sharma"
    assert fields.get("applicant_mobile") == "9812345671"
    assert fields.get("place") == "Ghaziabad, Uttar Pradesh"
    assert fields.get("applicant_address") == "123, Shastri Nagar, Ghaziabad, Uttar Pradesh - 201002"
    assert fields.get("respondent_address") == "45, Nehru Place, New Delhi - 110019"


def test_a_value_containing_a_full_stop_is_not_truncated_at_it() -> None:
    """"ABC Electronics Pvt. Ltd." must survive intact."""
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")
    _ask(service, _ENGLISH_LABELLED)
    assert _draft_fields(memory).get("respondent_name") == "ABC Electronics Pvt. Ltd."


def test_hindi_labels_are_recognised_as_field_labels() -> None:
    """The assistant asks in Hindi; the answer must be parsed in Hindi."""
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mere landlord ko security deposit wapas karne ke liye rent notice draft karo")
    _ask(service, _HINDI_LABELLED)

    fields = _draft_fields(memory)
    assert fields.get("applicant_name") == "अमित कुमार शर्मा"
    assert fields.get("respondent_name") == "राजेश वर्मा"
    assert fields.get("place") == "गाज़ियाबाद, उत्तर प्रदेश"
    assert fields.get("applicant_mobile") == "9812345670"


def test_no_single_field_swallows_the_whole_labelled_message() -> None:
    """The observed failure: Recipient and Sender each held the entire message."""
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mere landlord ko security deposit wapas karne ke liye rent notice draft karo")
    _ask(service, _HINDI_LABELLED)

    for key, value in _draft_fields(memory).items():
        assert len(value) < len(_HINDI_LABELLED) * 0.6, f"{key} swallowed the whole message"


def test_a_hindi_address_containing_commas_is_kept_whole() -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mere landlord ko security deposit wapas karne ke liye rent notice draft karo")
    _ask(service, _HINDI_LABELLED)

    address = _draft_fields(memory).get("applicant_address", "")
    assert address.startswith("24, नेहरू नगर")
    assert "201001" in address
    assert "मोबाइल" not in address, "the address ran into the next label"


def test_a_unicode_colon_variant_is_accepted_as_a_label_separator() -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")
    _ask(service, "Applicant Name： Rahul Sharma\nPlace： Ghaziabad")

    fields = _draft_fields(memory)
    assert fields.get("applicant_name") == "Rahul Sharma"
    assert fields.get("place") == "Ghaziabad"


def test_an_extracted_value_is_only_stored_against_a_real_template_field() -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")
    _ask(service, _ENGLISH_LABELLED)

    template = get_template("consumer_complaint")
    assert template is not None
    assert set(_draft_fields(memory)) <= template.field_keys()


# ---------------------------------------------------------------------------
# A1 -- draft/template isolation
# ---------------------------------------------------------------------------


def test_consumer_complaint_facts_do_not_leak_into_a_later_rent_notice() -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")
    _ask(
        service,
        "Facts of the Case: The smartphone I bought stopped working within a week and "
        "the seller refused to replace it. "
        "Expected Relief / What you want: Full refund with compensation.",
    )
    complaint_fields = _draft_fields(memory)
    assert complaint_fields.get("facts"), "precondition: the complaint captured its own facts"

    _ask(service, "Ab mujhe rent security deposit recovery notice draft karna hai")

    notice_fields = _draft_fields(memory)
    assert memory.get("draft_template_id") == "rent_notice"
    assert notice_fields.get("facts") != complaint_fields.get("facts")
    assert notice_fields.get("expected_relief") != complaint_fields.get("expected_relief")


def test_starting_a_second_template_inherits_no_party_or_address_fields() -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")
    _ask(service, _ENGLISH_LABELLED)
    assert _draft_fields(memory).get("applicant_name") == "Rahul Sharma"

    _ask(service, "Ab ek rent security deposit recovery notice draft karo")

    inherited = _draft_fields(memory)
    for key in ("applicant_name", "applicant_address", "respondent_name", "respondent_address", "place"):
        assert not inherited.get(key), f"{key} leaked from the previous draft"


def test_the_conversation_language_preference_survives_a_template_change() -> None:
    """Legal facts must not carry over; the user's chosen language may.

    The language is named on the first draft only. Starting a second document
    without naming it again must not silently drop back to English.
    """
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe hindi me consumer complaint banani hai")
    assert memory.get("draft_language") == "hindi"

    _ask(service, "Ab rent security deposit recovery notice draft karo")
    assert memory.get("draft_template_id") == "rent_notice"
    assert memory.get("draft_language") == "hindi"


def test_two_drafts_in_one_session_keep_separate_field_bags() -> None:
    """Parking one draft and starting another must not merge their facts."""
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")
    _ask(service, "Applicant Name: Rahul Sharma")
    _ask(service, "Ab ek rent security deposit recovery notice draft karo")
    _ask(service, "Applicant Name: Amit Kumar Sharma")

    active = _draft_fields(memory)
    assert memory.get("draft_template_id") == "rent_notice"
    assert active.get("applicant_name") == "Amit Kumar Sharma"

    parked = memory.get("parked_drafts") or []
    assert [entry["draft_template_id"] for entry in parked] == ["consumer_complaint"]
    assert parked[0]["draft_fields"].get("applicant_name") == "Rahul Sharma"


def test_a_parked_draft_keeps_its_own_fields_when_resumed() -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")
    _ask(service, "Applicant Name: Rahul Sharma")
    _ask(service, "Ab ek rent security deposit recovery notice draft karo")
    _ask(service, "Applicant Name: Amit Kumar Sharma")

    _ask(service, "switch to consumer complaint draft")

    assert memory.get("draft_template_id") == "consumer_complaint"
    assert _draft_fields(memory).get("applicant_name") == "Rahul Sharma"


def test_an_authenticated_owners_draft_is_refused_to_another_account() -> None:
    """The one ownership rule both `/draft/*` and the chat workflows call."""
    from app.core.exceptions import ForbiddenError
    from app.services.draft_management import ensure_draft_access

    draft = {"_id": "d1", "user_id": "user-A"}
    ensure_draft_access(draft, authenticated_user_id="user-A", session_id="any-session")
    with pytest.raises(ForbiddenError):
        ensure_draft_access(draft, authenticated_user_id="user-B", session_id="any-session")


def test_an_anonymous_session_draft_is_scoped_to_that_session() -> None:
    from app.core.exceptions import ForbiddenError
    from app.services.draft_management import ensure_draft_access

    draft = {"_id": "d2", "session_id": "session-A"}
    ensure_draft_access(draft, authenticated_user_id=None, session_id="session-A")
    with pytest.raises(ForbiddenError):
        ensure_draft_access(draft, authenticated_user_id=None, session_id="session-B")


# ---------------------------------------------------------------------------
# A5 -- resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", ["continue draft", "continue kro draft ko"])
def test_resuming_returns_to_the_same_draft_with_its_facts(phrase: str) -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")
    _ask(service, "Applicant Name: Rahul Sharma")
    assert _draft_fields(memory).get("applicant_name") == "Rahul Sharma"

    # An ordinary legal question parks the draft.
    _ask(service, "Anticipatory bail kya hoti hai?")
    response = _ask(service, phrase)

    assert memory.get("draft_template_id") == "consumer_complaint"
    assert _draft_fields(memory).get("applicant_name") == "Rahul Sharma"
    assert "I don't have a template" not in response.answer


# ---------------------------------------------------------------------------
# A4 -- editing the active draft
# ---------------------------------------------------------------------------


def _preview_memory() -> dict[str, Any]:
    return {
        "messages": [],
        "draft_mode": True,
        "draft_stage": "preview",
        "draft_template_id": "rent_notice",
        "draft_id": "draft-123",
        "draft_language": "hinglish",
        "draft_fields": {
            "applicant_name": "Amit Kumar Sharma",
            "respondent_name": "Rajesh Verma",
            "respondent_address": "78, Sector 18, Noida",
        },
    }


def test_a_hinglish_edit_command_edits_the_active_draft() -> None:
    memory = _preview_memory()
    service = _service(memory)
    regenerate = AsyncMock(return_value=_fake_draft())
    service.draft_conversation.draft_engine.regenerate = regenerate

    response = _ask(service, "Recipient ka address Noida se Ghaziabad kar do")

    assert regenerate.await_count == 1, "the edit never reached the draft engine"
    _draft_id, updates = regenerate.await_args.args[0], regenerate.await_args.args[1]
    assert updates == {"respondent_address": "Ghaziabad"}
    assert "Knowledge Base" not in response.answer


def _fake_draft():
    from app.schemas.drafting import DraftGenerateResponse

    return DraftGenerateResponse(
        status="complete",
        draft_id="draft-123",
        template_id="rent_notice",
        template_name="Rent / Security Deposit Recovery Notice",
        language="hinglish",
        sections={"Recipient": "Rajesh Verma, Ghaziabad"},
        full_text="Rajesh Verma, Ghaziabad",
        missing_fields=[],
    )


# ---------------------------------------------------------------------------
# A6 / A7 / A8 -- an active draft must not swallow an explicit other request
# ---------------------------------------------------------------------------


def test_saved_drafts_listing_is_reachable_while_a_draft_is_active() -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")

    response = _ask(service, "Meri save ki hui drafts dikhao")

    assert response.conversation_intent == "Workflow"
    assert "saved_drafts" in {response.intent, response.active_workflow} or response.workflow_status == "completed"


def test_notarization_preparation_is_not_read_as_a_new_affidavit_draft() -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")

    response = _ask(service, "Mere affidavit ko notarization ke liye prepare karo")

    assert memory.get("draft_template_id") != "affidavit", "a new affidavit draft was started"
    assert response.conversation_intent == "Workflow"
    assert "notarization_prepare" in {response.intent, response.active_workflow}


def test_a_verification_request_without_a_token_asks_for_one() -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")

    response = _ask(service, "Is document ka verification token check karo")

    assert response.conversation_intent == "Workflow"
    assert "notarization_verify" in {response.intent, response.active_workflow}
    assert response.missing_field or "token" in response.answer.lower()


def test_an_ordinary_legal_question_still_parks_the_draft_and_reaches_rag() -> None:
    """Precedence must not swing the other way: a real question is answered."""
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe consumer complaint banani hai")

    response = _ask(service, "Anticipatory bail kya hoti hai?")

    assert response.conversation_intent != "Workflow"
    assert memory.get("draft_template_id") == "consumer_complaint"


@pytest.mark.parametrize(
    "message,expected_workflow",
    [
        ("Mere affidavit ko notarization ke liye prepare karo", "notarization_prepare"),
        ("Meri notarization request ka status batao", "notarization_status"),
        ("Verification token check karna hai", "notarization_verify"),
    ],
)
def test_the_notarization_intents_are_distinguished(message: str, expected_workflow: str) -> None:
    from app.chatops.registry import best_match

    match = best_match(message, "english")
    assert match is not None, f"nothing matched: {message!r}"
    assert match[0].name == expected_workflow


def test_asking_to_draft_an_affidavit_is_not_a_notarization_request() -> None:
    """The two must stay apart in BOTH directions."""
    memory: dict[str, Any] = {}
    service = _service(memory)
    _ask(service, "Mujhe ek general affidavit draft karna hai")
    assert memory.get("draft_template_id") == "affidavit"
    assert memory.get("draft_mode") is True


def test_verification_without_a_token_asks_for_the_token_or_qr() -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    response = _ask(service, "Verification token check karna hai")

    assert response.missing_field == "verification_token"
    lowered = response.answer.lower()
    assert "qr" in lowered or "verification code" in lowered


def test_no_chat_turn_claims_a_document_was_notarized() -> None:
    memory: dict[str, Any] = {}
    service = _service(memory)
    response = _ask(service, "Mere affidavit ko notarization ke liye prepare karo")
    lowered = response.answer.lower()
    for claim in ("has been notarized", "notarized successfully", "legally verified", "we have notarized"):
        assert claim not in lowered
