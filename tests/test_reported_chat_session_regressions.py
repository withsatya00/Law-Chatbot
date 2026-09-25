"""Regressions from one reported chat session.

Every case below is taken verbatim from a transcript a user shared, where six
separate turns went wrong. The failures were independent of each other, so
they are grouped here by the turn that exposed them rather than spread across
the existing per-module test files, which is where the fixes themselves live.

The session, in order:

1. "mujhe batao ki tum meri kin legal problems me madad kar sakte ho?" --
   a question about the assistant, answered "no verified document related to
   this question is available in the Knowledge Base".
2. "Iska detailed advocate-style answer do." -- a pronoun-only follow-up to
   the previous answer, also answered with the knowledge-base refusal.
3. "Document ready kar do" -- accepting the assistant's own offer to prepare
   a legal notice, answered with the same refusal.
4. Nine labelled fields pasted in one message -- the request failed outright
   ("Sorry, I couldn't reach the assistant just now") and every field was
   lost; the next message asked for all nine again.
5. A PDF uploaded, then "iss document ko smjhao" -- answered by re-printing
   the drafting flow's pending-field list instead of explaining the document.

A second, separate reported session (2026-09-23) surfaced two more
independent bugs once the GK fallback flag was turned on:

6. "write a pyton code for adding two number" -- a plain coding request with
   no legal content at all launched Draft Mode ("describe your problem")
   because `DraftIntentDetector.detect`'s no-template-evidence fallback
   assumed any drafting verb ("write") with no question phrasing meant a
   legal document was wanted. Every unrelated follow-up in the rest of that
   same session (a pizza craving, a recipe request) then kept getting
   swallowed by the drafting state machine instead of answered normally,
   because it never escaped Draft Mode in the first place.
7. "pizaa khane ki recipie btao" (typo'd "pizza"/"recipe") -- evaded
   `_OFF_DOMAIN_PATTERN`'s literal word match entirely (unlike the
   correctly-spelled "mujhe pizza khana hai" one turn earlier, which WAS
   caught), fell through to the ordinary pipeline, and got a real recipe
   answered and mislabeled "General Legal Knowledge" by the GK fallback.
"""

import asyncio
import re
from unittest.mock import AsyncMock

import pytest

from app.core.constants import capability_overview
from app.drafting.field_extraction import DraftFieldExtractor
from app.drafting.intent import DraftIntentDetector
from app.drafting.templates import get_template, list_templates
from app.drafting.validation import DraftFieldValidator
from app.intent.classifier import ConversationIntentClassifier
from app.llm.base import LLMResponse
from app.llm.deadline import current_deadline
from app.schemas.chat import ChatRequest
from app.services.chat_service import _DRAFT_INTERRUPT_INTENTS, ChatService

classifier = ConversationIntentClassifier()
draft_detector = DraftIntentDetector()


def test_explicit_lost_document_affidavit_command_outranks_prior_police_complaint_fact() -> None:
    message = (
        "मेरा नाम अमित कुमार है। मेरा मूल स्नातक प्रमाणपत्र 10 सितंबर 2026 को "
        "लखनऊ के चारबाग क्षेत्र में खो गया। प्रमाणपत्र संख्या LU-2018-45821 है। "
        "मैंने ऑनलाइन पुलिस शिकायत दर्ज की है। Lost Document Affidavit बनाओ।"
    )

    match = draft_detector.detect(message)

    assert match.matched
    assert match.draft_id == "lost_document_affidavit"


def test_lost_document_affidavit_has_native_labels_and_document_specific_material_fields() -> None:
    template = get_template("lost_document_affidavit")
    assert template is not None
    assert template.hindi_name == "दस्तावेज़ खोने का शपथपत्र"
    required = {field.key for field in template.required_fields}
    assert {
        "document_identifier", "loss_date", "loss_place", "affidavit_purpose"
    }.issubset(required)
    assert "police_station" not in required
    assert template.applicable_sections_hint == []


def test_lost_document_affidavit_extracts_material_hindi_facts_without_llm() -> None:
    template = get_template("lost_document_affidavit")
    assert template is not None
    message = (
        "मेरा नाम अमित कुमार है। मेरा मूल स्नातक प्रमाणपत्र 10 सितंबर 2026 को "
        "लखनऊ के चारबाग क्षेत्र में खो गया। प्रमाणपत्र संख्या LU-2018-45821 है।"
    )

    fields = DraftFieldExtractor()._extract_with_regex(message, template.field_keys())

    assert fields["applicant_name"] == "अमित कुमार"
    assert fields["lost_document_name"] == "मूल स्नातक प्रमाणपत्र"
    assert fields["document_identifier"] == "LU-2018-45821"
    assert fields["loss_date"] == "10 सितंबर 2026"
    assert fields["loss_place"] == "लखनऊ के चारबाग क्षेत्र"


def _memory(*, prior_reply: bool = True, uploaded_document: bool = False) -> dict:
    """Memory in the shape `ChatService` hands the classifier: the current
    user turn is already appended as the last message."""
    messages: list[dict[str, str]] = []
    if prior_reply:
        messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    memory: dict = {"messages": [*messages, {"role": "user", "content": "current"}]}
    if uploaded_document:
        memory["last_uploaded_document_id"] = "doc-1"
    return memory


# --- Turn 1: "what can you help me with?" -----------------------------------

@pytest.mark.parametrize(
    "message",
    [
        # The reported message, verbatim.
        "mujhe batao ki tum meri kin legal problems me madad kar sakte ho?",
        "what can you do?",
        "what can you help me with?",
        "tum kya kar sakte ho?",
        "aap kaise madad kar sakte hain?",
        "आप क्या कर सकते हैं?",
        "what are your capabilities?",
    ],
)
def test_questions_about_the_assistant_are_not_sent_to_retrieval(message: str) -> None:
    assert classifier.classify(message, _memory()).intent == "Capability Question"


@pytest.mark.parametrize(
    "message",
    [
        # About the USER's own ability, not the assistant's.
        "kya main FIR file kar sakta hoon?",
        # Second person, but asking for legal content rather than capabilities.
        "kya aap mujhe FIR ka process bata sakte hain?",
        # A real legal question that happens to address the assistant.
        "My landlord won't return my deposit. What are my legal options?",
        "मेरे पास क्या legal options हैं?",
    ],
)
def test_real_legal_questions_are_not_mistaken_for_capability_questions(message: str) -> None:
    assert classifier.classify(message, _memory()).intent != "Capability Question"


def test_capability_question_interrupts_an_active_draft() -> None:
    # It can never be a field value: it takes an interrogative, a
    # second-person reference and an ability word all at once.
    assert "Capability Question" in _DRAFT_INTERRUPT_INTENTS


@pytest.mark.parametrize("language", ["english", "hindi", "hinglish", "tamil", None])
def test_capability_overview_is_fixed_text_naming_the_real_template_count(language: str | None) -> None:
    overview = capability_overview(language, len(list_templates()))
    assert str(len(list_templates())) in overview
    # Never left with an unformatted placeholder, in any language.
    assert "{count}" not in overview
    # An unsupported language falls back to English, never to Devanagari.
    if language == "tamil":
        assert overview == capability_overview("english", len(list_templates()))


# --- Turn 2: a pronoun-only follow-up ---------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        # The reported message, verbatim.
        "Iska detailed advocate-style answer do.",
        "iski detail me explain karo",
        "isme aur kya hota hai",
        "uska matlab detail me batao",
    ],
)
def test_hinglish_deictic_followups_stay_on_the_previous_topic(message: str) -> None:
    # Only "isko" was recognised before, so these fell into the "General Legal
    # Information" catch-all and were sent to retrieval with no topic in them.
    assert classifier.classify(message, _memory()).intent != "General Legal Information"


def test_deictic_followup_needs_a_prior_reply() -> None:
    # Nothing to point back at on a session's first message.
    assert classifier.classify(
        "Iska detailed advocate-style answer do.", _memory(prior_reply=False)
    ).intent != "Follow-up Question"


def test_deictic_followup_keeps_previous_topic_when_rewriter_is_unavailable() -> None:
    service = ChatService.__new__(ChatService)
    service.llm = type("FailingLLM", (), {})()
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="", model="test", provider="gemini", error="timeout", error_kind="timeout"
        )
    )
    memory = {
        "messages": [
            {"role": "user", "content": "Mera landlord bina notice ghar khali karwa raha hai."},
            {"role": "assistant", "content": "Previous legal answer."},
            {"role": "user", "content": "Iska detailed advocate-style answer do."},
        ]
    }

    resolved = asyncio.run(
        service._resolve_followup_question("Iska detailed advocate-style answer do.", memory)
    )

    assert "landlord bina notice" in resolved
    assert "Follow-up instruction" in resolved


# --- Turn 3: accepting an offer to prepare a document -----------------------

@pytest.mark.parametrize(
    "message",
    ["Document ready kar do", "document ready karo", "notice taiyar kar do", "ye ready kr do"],
)
def test_ready_kar_do_is_a_drafting_request(message: str) -> None:
    assert draft_detector.detect(message).matched
    assert classifier.classify(message, _memory()).intent == "Draft Generation"


@pytest.mark.parametrize(
    "message",
    [
        # A statement/question ABOUT readiness is not a command to draft: the
        # new pattern requires the imperative tail ("kar do"/"karo"), which
        # none of these have.
        "Is the document ready?",
        "ye document ready hai",
        "notice taiyar hai ya nahi",
    ],
)
def test_asking_whether_something_is_ready_is_not_a_drafting_request(message: str) -> None:
    assert not draft_detector.detect(message).matched


# --- Turn 4: nine labelled fields pasted in one message ---------------------

_NINE_FIELDS_MESSAGE = (
    "Applicant Address: 123, Shastri Nagar, Ghaziabad, Uttar Pradesh - 201002 "
    "Alternate Mobile Number: 9876543210 "
    "Applicant Name: Rahul Sharma "
    "Expected Relief / What you want: Stolen mobile phone ko trace/recover karke mujhe wapas "
    "dilaya jaye aur zaroori kanooni karwai ki jaye. "
    "Facts of the Case (in your own words): Mera mobile phone 1 September 2026 ko shaam lagbhag "
    "7:30 baje market mein kahin gir gaya/chori ho gaya. Maine phone ko aas-paas kaafi search kiya "
    "lekin phone nahi mila. Maine apne primary number par call bhi kiya, lekin phone receive nahi hua. "
    "IMEI Number: 352099123456789 "
    "Location Phone Was Stolen From: Ghanta Ghar Market, Ghaziabad "
    "Place: Ghaziabad, Uttar Pradesh "
    "Police Station: Kotwali Nagar Police Station, Ghaziabad"
)


def _extract_without_llm(message: str, template) -> dict[str, str]:
    """The deterministic half of `DraftFieldExtractor.extract` -- regex first,
    then labelled prefixes, exactly as `extract()` combines them. Built
    without `__init__` so no provider is constructed: these fields must be
    recoverable with no LLM available at all."""
    extractor = DraftFieldExtractor.__new__(DraftFieldExtractor)
    extracted = extractor._extract_with_regex(message, template.field_keys())
    for key, value in extractor._extract_by_label_prefix(message, template).items():
        if value and not extracted.get(key):
            extracted[key] = value
    return extracted


def test_all_nine_labelled_fields_are_extracted_from_one_message() -> None:
    template = get_template("mobile_theft_complaint")
    assert template is not None
    extracted = _extract_without_llm(_NINE_FIELDS_MESSAGE, template)

    issues = DraftFieldValidator().validate(template, dict(extracted), "hinglish", [])
    assert issues == []
    missing = template.required_field_keys() - {key for key, value in extracted.items() if value}
    assert missing == set()


def test_complete_labelled_form_skips_optional_llm_extraction() -> None:
    """The exact live timeout began with an unnecessary ~84s extraction
    call even though deterministic label parsing already had all nine fields.
    """
    template = get_template("mobile_theft_complaint")
    assert template is not None
    extractor = DraftFieldExtractor()
    extractor.llm.chat = AsyncMock(side_effect=AssertionError("LLM must not be called"))

    extracted = asyncio.run(extractor.extract(_NINE_FIELDS_MESSAGE, template))

    assert template.required_field_keys() <= {
        key for key, value in extracted.items() if value
    }
    extractor.llm.chat.assert_not_awaited()


def test_chat_budget_must_leave_response_headroom() -> None:
    from app.core.config import Settings

    with pytest.raises(RuntimeError, match="at least 5 seconds below"):
        Settings(chat_request_budget_seconds=180, client_request_timeout_seconds=180)


def test_chat_answer_binds_one_request_wide_deadline() -> None:
    class _Response:
        request_id = ""

        def model_copy(self, update=None):
            self.request_id = (update or {}).get("request_id", "")
            return self

    service = ChatService.__new__(ChatService)

    async def _inside(*_args, **_kwargs):
        bound = current_deadline()
        assert bound is not None
        assert bound.label == "POST /chat"
        return _Response()

    service._answer_within_deadline = _inside
    response = asyncio.run(service.answer(ChatRequest(question="hello")))
    assert isinstance(response, _Response)
    assert current_deadline() is None


def test_a_short_field_stops_at_the_next_label_instead_of_running_into_it() -> None:
    # The reported failure: with no comma between "Rahul Sharma" and the next
    # field's label, the name swallowed the whole rest of the message.
    template = get_template("mobile_theft_complaint")
    extracted = _extract_without_llm(_NINE_FIELDS_MESSAGE, template)
    assert extracted["applicant_name"] == "Rahul Sharma"
    assert extracted["imei_number"] == "352099123456789"


def test_a_short_field_keeps_commas_inside_its_own_value() -> None:
    # The mirror-image failure: stopping at the first comma truncated
    # "Kotwali Nagar Police Station, Ghaziabad" to just the station name.
    template = get_template("mobile_theft_complaint")
    extracted = _extract_without_llm(_NINE_FIELDS_MESSAGE, template)
    assert extracted["police_station"] == "Kotwali Nagar Police Station, Ghaziabad"
    assert extracted["incident_location"] == "Ghanta Ghar Market, Ghaziabad"
    assert extracted["place"] == "Ghaziabad, Uttar Pradesh"


def test_a_conversational_single_label_message_still_stops_at_the_comma() -> None:
    # One labelled field means the user is talking, not filling in a form --
    # the original comma-stopping read is still the right one there.
    template = get_template("mobile_theft_complaint")
    extracted = _extract_without_llm(
        "Police Station: Hazratganj, and it happened yesterday evening", template
    )
    assert extracted["police_station"] == "Hazratganj"


def test_narrative_field_keeps_the_users_whole_account() -> None:
    template = get_template("mobile_theft_complaint")
    extracted = _extract_without_llm(_NINE_FIELDS_MESSAGE, template)
    facts = extracted["facts"]
    # Every sentence of the account survives, and it stops before the next label.
    assert "gir gaya/chori ho gaya" in facts
    assert "phone receive nahi hua" in facts
    assert "IMEI Number" not in facts


def test_draft_generation_has_a_wall_clock_budget_below_the_client_timeout() -> None:
    # The nine-field turn died as a transport error because nothing bounded
    # the total time across the first drafting call and its expansion passes.
    from app.core.config import settings

    assert settings.draft_generation_budget_seconds < 180


def test_collected_fields_are_checkpointed_before_generation() -> None:
    # The fields must already be durable when the (slow, retried) generation
    # call starts, so a failure there cannot discard them. The actual
    # `self.draft_engine.generate` call lives in `_finalize_generation_reply`
    # now (split out of `_process_collecting_message` so the discovery
    # flow's pre-generation summary confirmation -- `_continue_confirm_
    # summary` -- can call the exact same generation path once the user
    # confirms, without duplicating it) -- so the property under test is
    # checked at each of its two call sites instead of inside one method's
    # source directly.
    import inspect

    from app.drafting.conversation import DraftConversationEngine

    generation_body = inspect.getsource(DraftConversationEngine._finalize_generation_reply)
    assert "self.draft_engine.generate" in generation_body

    for caller in (
        DraftConversationEngine._process_collecting_message,
        DraftConversationEngine._continue_confirm_summary,
    ):
        body = inspect.getsource(caller)
        # `rindex` (last occurrence), not `index` (first) -- both methods
        # explain the property they implement in a comment before the real
        # code does it (e.g. "...before `_finalize_generation_reply` is
        # ever called" precedes the actual checkpoint call in
        # `_process_collecting_message`'s discovery-gate branch), so the
        # FIRST textual mention of either name is prose, not code. The last
        # occurrence of each is always the real call.
        checkpoint_at = body.rindex("_checkpoint_collected_fields")
        finalize_at = body.rindex("_finalize_generation_reply")
        assert checkpoint_at < finalize_at, f"{caller.__name__} must checkpoint before generating"


# --- Turn 5: "explain this document" while a draft is in progress -----------

@pytest.mark.parametrize(
    "message",
    [
        # The reported message, verbatim.
        "iss document ko smjhao",
        "is document ko samjhao",
        "ye pdf samjha do",
        "isko smjhao",
    ],
)
def test_explain_this_document_reaches_document_analysis(message: str) -> None:
    match = classifier.classify(message, _memory(uploaded_document=True))
    assert match.intent == "Document Analysis"


def test_document_analysis_interrupts_an_active_draft() -> None:
    # Which is what stops it being swallowed as a field answer and answered
    # with the pending-field list, as it was in the reported session.
    assert "Document Analysis" in _DRAFT_INTERRUPT_INTENTS


def test_samjhao_alone_is_not_enough_without_a_document_noun() -> None:
    # "samjhao" is a plain "explain it to me" -- on its own it must keep
    # falling through to ordinary routing, not claim to analyse a document.
    match = classifier.classify("bail ka matlab samjhao", _memory())
    assert match.intent != "Document Analysis"


def test_every_conversation_intent_has_a_suggested_actions_entry() -> None:
    # A new intent with no entry here silently renders no action chips.
    from app.intent.classifier import CONVERSATION_INTENT_ACTIONS, CONVERSATION_INTENTS

    assert set(CONVERSATION_INTENTS) <= set(CONVERSATION_INTENT_ACTIONS)


def test_capability_question_is_protected_from_llm_reclassification() -> None:
    # Downgrading it to a RAG intent reintroduces the knowledge-base refusal
    # this whole branch exists to prevent.
    import inspect

    source = inspect.getsource(ConversationIntentClassifier.classify_advanced)
    protected_block = source[source.index("protected_intents"):source.index("if deterministic.intent in")]
    assert re.search(r'"Capability Question"', protected_block)


# --- Turn 6: a non-legal drafting verb must not launch Draft Mode -----------

@pytest.mark.parametrize(
    "message",
    [
        "write a pyton code for adding two number",  # the reported message, verbatim (typo included)
        "write a python code for adding two numbers",
        "write me a poem about the rain",
        "prepare a recipe for pizza",
    ],
)
def test_non_legal_drafting_verb_does_not_start_a_draft(message: str) -> None:
    assert draft_detector.detect(message).matched is False


@pytest.mark.parametrize(
    "message",
    [
        "create draft",
        "Create a draft",
        "generate draft please",
        "make a draft",
        "I need a draft",
        "draft chahiye",
        "please draft a complaint for me",
        "write a legal notice",
    ],
)
def test_genuinely_vague_legal_drafting_requests_still_start_a_draft(message: str) -> None:
    # The fix for the case above must not regress the deliberately-vague
    # "I don't know the document name" discovery flow these messages start.
    assert draft_detector.detect(message).matched is True


# --- Turn 7: a typo must not evade off-domain detection ---------------------

@pytest.mark.parametrize(
    "message",
    [
        "pizaa khane ki recipie btao",  # the reported message, verbatim
        "write a pyton code for adding two number",
    ],
)
def test_typo_d_off_domain_request_is_still_recognized_as_out_of_domain(message: str) -> None:
    match = classifier.classify(message, _memory())
    assert match.intent == "Out of Domain"


def test_off_domain_typo_correction_does_not_break_correct_spelling() -> None:
    # The fix must not regress the already-working, correctly-spelled case.
    match = classifier.classify("mujhe pizza khana hai", _memory())
    assert match.intent == "Out of Domain"
