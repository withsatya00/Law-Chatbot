"""Regressions from a second reported chat session: four turns, four defects.

The transcript, verbatim:

1. "um mere liye kya-kya kar sakte ho?" -- a "t" missing from "tum". Routed
   "General Legal Information" -> RAG -> "no verified document related to this
   question is available in the Knowledge Base", while the correctly spelled
   "Tum mere liye kya-kya kar sakte ho?" was answered from the fixed capability
   overview.
2. "Mujhe simple Hindi mein jawab diya karo" -- classified "Legal Advice" and
   answered from retrieval, instead of recording the language preference.
3. "Actually Hinglish mein jawab do" -- classified "General Legal Information"
   and answered from retrieval, same defect, different phrasing.
4. A no-verified-context refusal displayed with BNS and GST sources, an
   applicable-law list and a source-currency note, none of which supported it
   -- the answer says nothing supports it.

The fixes live in `app/language/typo_tolerance.py` (one bounded normalizer for
routing and retrieval), `app/language/detector.py`
(`is_language_preference_command`), `app/intent/classifier.py` (control intents
resolved before the legal-intent fallback) and
`app/services/safe_decline.py` (one response invariant for the refusal).

Everything here is deterministic: no test in this module calls a real LLM, and
`_service_with_mocks` makes `retriever.retrieve`/`reranker.rerank` raise if a
turn that must not retrieve ever does.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.constants import no_verified_context_message
from app.intent.classifier import ConversationIntentClassifier
from app.language.detector import is_language_preference_command
from app.language.typo_tolerance import (
    clarification_question,
    normalize_for_routing,
)
from app.llm.base import LLMResponse
from app.schemas.chat import ChatRequest
from app.schemas.common import RetrievedChunk
from app.services.chat_service import ChatService
from app.services.safe_decline import sanitize_payload
from streamlit_app.answer_apparatus import source_lines, suppresses_apparatus

classifier = ConversationIntentClassifier()


def _memory(*, prior_reply: bool = True) -> dict:
    messages: list[dict[str, str]] = []
    if prior_reply:
        messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    return {"messages": [*messages, {"role": "user", "content": "current"}]}


def _service_with_mocks(messages: list[dict[str, str]] | None = None) -> ChatService:
    """Same shape as `tests/test_chat_service_routing.py`'s helper, restated
    here rather than imported because the test directory is not an importable
    package. `retriever.retrieve` and `reranker.rerank` raise on any call, so a
    turn that is supposed to skip retrieval fails loudly if it regresses."""
    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])
    base = {"messages": messages or [], "summary": "", "current_intent": None, "legal_category": None}
    service.memory.append = AsyncMock(return_value=dict(base))
    service.memory.load = AsyncMock(return_value=dict(base))
    service.memory.update = AsyncMock(return_value={})
    service.memory.summarize_if_needed = AsyncMock(return_value={})
    service.history.insert = AsyncMock(return_value="history-id")
    service.query_log.insert = AsyncMock(return_value="query-log-id")
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    service.retriever.retrieve = AsyncMock(side_effect=AssertionError("retrieval should have been skipped"))
    service.reranker.rerank = AsyncMock(side_effect=AssertionError("reranking should have been skipped"))
    service.llm.chat = AsyncMock(side_effect=AssertionError("the answer LLM should not have been called"))
    return service


# ---------------------------------------------------------------------------
# 1. The four reported messages route correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("um mere liye kya-kya kar sakte ho?", "Capability Question"),
        ("Mujhe simple Hindi mein jawab diya karo", "Language Preference"),
        ("Actually Hinglish mein jawab do", "Language Preference"),
    ],
)
def test_the_reported_messages_route_to_conversation_control(message: str, expected: str) -> None:
    assert classifier.classify(message, _memory()).intent == expected


def test_the_misspelled_capability_question_matches_the_correctly_spelled_one() -> None:
    """Defect 1, stated as an equality: one dropped character must not change
    the route."""
    misspelled = classifier.classify("um mere liye kya-kya kar sakte ho?", _memory())
    correct = classifier.classify("Tum mere liye kya-kya kar sakte ho?", _memory())
    assert misspelled.intent == correct.intent == "Capability Question"


@pytest.mark.parametrize(
    "message,expected_language",
    [
        ("Mujhe simple Hindi mein jawab diya karo", "hindi"),
        ("Hindi me simple answer dena", "hindi"),
        ("Actually Hinglish mein jawab do", "hinglish"),
        ("Ab English me continue karo", "english"),
        ("कृपया सरल हिंदी में जवाब दें", "hindi"),
        ("Please reply in Hinglish only", "hinglish"),
    ],
)
def test_natural_language_switch_phrasings_are_recognised(message: str, expected_language: str) -> None:
    assert is_language_preference_command(message) == expected_language
    match = classifier.classify(message, _memory())
    assert match.intent == "Language Preference"
    assert match.resolved_language_preference == expected_language


@pytest.mark.parametrize(
    "message",
    [
        # Mentions a language; asks for nothing about the reply language.
        "I found a lawyer who speaks Hindi",
        "Hindi mein FIR kaise file karein?",
        "My client only understands Hindi, what should the notice say?",
        "Tamil me draft karo",
    ],
)
def test_merely_naming_a_language_is_not_a_preference_command(message: str) -> None:
    assert is_language_preference_command(message) is None
    assert classifier.classify(message, _memory()).intent != "Language Preference"


# ---------------------------------------------------------------------------
# 2. The preference persists to the following turn
# ---------------------------------------------------------------------------


def test_a_language_preference_is_persisted_as_the_requested_language() -> None:
    """Not as the language the SENTENCE was detected as: "Mujhe simple Hindi
    mein jawab diya karo" is Latin-script Hinglish, so a turn that stored its
    detected language would store "hinglish" -- the opposite of what was
    asked for."""
    service = _service_with_mocks()
    response = asyncio.run(service.answer(ChatRequest(question="Mujhe simple Hindi mein jawab diya karo")))

    assert response.conversation_intent == "Language Preference"
    assert response.detected_language == "hindi"
    stored = [call.kwargs for call in service.memory.update.await_args_list]
    assert any(call.get("language_preference") == "hindi" for call in stored), stored


def test_the_stored_preference_drives_the_next_turn() -> None:
    """The following turn inherits it: `ChatService` resolves the turn's
    language from `memory["language_preference"]` when the message carries no
    language signal of its own."""
    service = _service_with_mocks()
    service.memory.append = AsyncMock(
        return_value={
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            "summary": "",
            "current_intent": None,
            "legal_category": None,
            "language_preference": "hindi",
        }
    )
    # "Namaste" is a greeting, which is answered by the general-conversation
    # LLM in production -- this test is about the inherited language, so the
    # helper's "the answer LLM must not be called" guard is lifted here only.
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Namaste!", model="test", provider="test"))
    response = asyncio.run(service.answer(ChatRequest(question="Namaste")))
    assert response.detected_language == "hindi"


def test_the_acknowledgement_is_in_the_requested_language() -> None:
    for question, marker in (
        ("Mujhe simple Hindi mein jawab diya karo", "हिंदी"),
        ("Actually Hinglish mein jawab do", "Hinglish"),
    ):
        service = _service_with_mocks()
        response = asyncio.run(service.answer(ChatRequest(question=question)))
        assert marker in response.answer, response.answer


# ---------------------------------------------------------------------------
# 3. Control turns reach neither retrieval, reranking nor the answer LLM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "um mere liye kya-kya kar sakte ho?",
        "tum mre liye kya kya kr skte ho",
        "aap kon kon se legal kaam kr skte ho",
        "what can yu help me with?",
        "Mujhe simple Hindi mein jawab diya karo",
        "Actually Hinglish mein jawab do",
    ],
)
def test_control_turns_never_reach_retrieval_or_the_answer_llm(question: str) -> None:
    """`_service_with_mocks` raises from `retriever.retrieve`,
    `reranker.rerank` and `llm.chat`, so any of the three being reached fails
    this test rather than passing quietly."""
    service = _service_with_mocks()
    response = asyncio.run(service.answer(ChatRequest(question=question)))

    assert response.sources == []
    assert response.retrieved_chunks == []
    assert response.currency_notice == ""
    service.retriever.retrieve.assert_not_awaited()
    service.reranker.rerank.assert_not_awaited()
    service.llm.chat.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. A misspelled capability question behaves like a correctly spelled one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Tum mere liye kya-kya kar sakte ho?",
        "um mere liye kya-kya kar sakte ho?",
        "tum mre liye kya kya kr skte ho",
        "aap kon kon se legal kaam kr skte ho",
        "what can yu help me with?",
    ],
)
def test_typo_capability_questions_are_capability_questions(message: str) -> None:
    assert classifier.classify(message, _memory()).intent == "Capability Question"


def test_the_capability_overview_is_identical_however_it_was_spelled() -> None:
    answers = []
    for question in ("Tum mere liye kya-kya kar sakte ho?", "um mere liye kya-kya kar sakte ho?"):
        service = _service_with_mocks()
        answers.append(asyncio.run(service.answer(ChatRequest(question=question))).answer)
    assert answers[0] == answers[1]


@pytest.mark.parametrize(
    "message",
    [
        # About the USER's own ability.
        "Kya main FIR file kar sakta hoon?",
        # Asks for legal content, not for a list of features.
        "Kya aap FIR ka process bata sakte hain?",
        # A tenancy question that happens to address the assistant -- the
        # conditional clause is what makes it one.
        "Aap kya kar sakte hain agar landlord notice na de?",
    ],
)
def test_real_legal_questions_are_not_capability_questions(message: str) -> None:
    assert classifier.classify(message, _memory()).intent != "Capability Question"


# ---------------------------------------------------------------------------
# 5. Legal typos improve retrieval without touching numbers or entities
# ---------------------------------------------------------------------------


def test_cheque_bounce_typos_reach_the_section_138_vocabulary() -> None:
    result = normalize_for_routing("chek bouns hone pr secshun 138 me kya hota hai?")
    normalized = result.normalized_for_routing.lower()
    assert "cheque" in normalized
    assert "bounce" in normalized
    assert "section" in normalized
    # The section NUMBER is never touched, and the original spelling is still
    # searched alongside the corrected one.
    assert "138" in result.retrieval_query
    assert result.original in result.retrieval_query
    assert "chek bouns" in result.retrieval_query


def test_the_retrieval_query_keeps_the_original_wording() -> None:
    """`retrieval_query` is original + aliases, never a replacement -- so an
    exact match on what the user actually typed can still win."""
    for question in ("chek bouns hone pr secshun 138 me kya hota hai?", "notry ke liye affidavit tyar kro"):
        result = normalize_for_routing(question)
        assert result.retrieval_query.startswith(result.original)


@pytest.mark.parametrize(
    "message,expected_fragment",
    [
        ("FIR kya htoi hai?", "hota"),
        ("notry ke liye affidavit tyar kro", "notary ke liye affidavit tayar karo"),
    ],
)
def test_hinglish_shorthand_is_expanded_for_routing(message: str, expected_fragment: str) -> None:
    assert expected_fragment in normalize_for_routing(message).normalized_for_routing


def test_a_notarization_typo_still_starts_the_notarization_workflow() -> None:
    from app.chatops.registry import best_match

    match = best_match("notry ke liye affidavit tyar kro", "hinglish")
    assert match is not None, "the misspelled notarization request matched no workflow"
    assert match[0].name.startswith("notarization")


def test_a_legal_typo_is_still_a_legal_question_not_a_control_turn() -> None:
    for message in ("chek bouns hone pr secshun 138 me kya hota hai?", "FIR kya htoi hai?"):
        intent = classifier.classify(message, _memory()).intent
        assert intent not in {"Capability Question", "Language Preference", "Typo Clarification"}


# ---------------------------------------------------------------------------
# 6. An ambiguous correction asks instead of guessing
# ---------------------------------------------------------------------------


def test_an_ambiguous_control_typo_is_reported_not_guessed() -> None:
    result = normalize_for_routing("aap kya sakto ho")
    assert result.corrections == ()
    # Left exactly as typed; the tie is reported instead.
    assert result.normalized_for_routing == "aap kya sakto ho"
    (token, candidates), = result.control_ambiguities
    assert token == "sakto"
    assert set(candidates) == {"sakta", "sakte", "sakti"}


def test_an_ambiguous_control_typo_produces_a_clarification_question() -> None:
    question = clarification_question(normalize_for_routing("aap kya sakto ho"))
    assert question is not None
    assert "sakto" in question
    # The alternatives offered are exactly the tied candidates -- never a
    # preferred guess dressed up as a question.
    assert "sakta" in question and "sakte" in question


def test_an_ambiguous_control_typo_asks_rather_than_retrieving() -> None:
    service = _service_with_mocks()
    response = asyncio.run(service.answer(ChatRequest(question="aap kya sakto ho")))
    assert response.conversation_intent == "Typo Clarification"
    assert "sakto" in response.answer
    service.retriever.retrieve.assert_not_awaited()


def test_a_clear_correction_is_applied_and_no_clarification_is_asked() -> None:
    result = normalize_for_routing("what can yu help me with?")
    assert result.corrections == (("yu", "you"),)
    assert clarification_question(result) is None


# ---------------------------------------------------------------------------
# 7. Names, numbers, dates, amounts, sections and quoted text are untouched
# ---------------------------------------------------------------------------

_PASTED_FIELDS = (
    "Applicant Name: Rahul Sharma "
    "Applicant Address: 123, Shastri Nagar, Ghaziabad, Uttar Pradesh - 201002 "
    "Alternate Mobile Number: 9876543210 "
    "IMEI Number: 352099123456789 "
    "Invoice: INV-2026/114 dated 01/09/2026 for Rs. 45,500 "
    "Police Station: Kotwali Nagar Police Station, Ghaziabad"
)


@pytest.mark.parametrize(
    "text",
    [
        _PASTED_FIELDS,
        "Section 138 of the Negotiable Instruments Act, 1881",
        "Invoice INV-2026/114 for Rs. 45,500 dated 01/09/2026",
        "My name is Rahul Sharma and I live at 12/B Shastri Nagar",
        'He said "chek bouns ho gya" in the meeting',
        "Contact me at rahul.sharma@example.com or 9876543210",
        "The file is notice_final_v2.pdf",
    ],
)
def test_protected_content_is_returned_byte_for_byte(text: str) -> None:
    result = normalize_for_routing(text)
    assert result.normalized_for_routing == text
    assert result.retrieval_query == text
    assert result.corrections == ()


def test_the_original_message_is_always_preserved() -> None:
    """Whatever else it does, the normalizer never mutates the text a caller
    keeps for chat history and audit."""
    for text in ("um mere liye kya-kya kar sakte ho?", _PASTED_FIELDS, "chek bouns hone pr secshun 138"):
        assert normalize_for_routing(text).original == text


def test_quoted_text_inside_a_correctable_sentence_is_left_alone() -> None:
    result = normalize_for_routing('bta do ki "kr skte" ka matlab kya hai')
    assert '"kr skte"' in result.normalized_for_routing
    assert result.normalized_for_routing.startswith("bata do")


# ---------------------------------------------------------------------------
# 8. A safe decline carries no sources, currency note or retrieved data
# ---------------------------------------------------------------------------


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        text="Section 318 of the Bharatiya Nyaya Sanhita defines cheating.",
        score=0.9,
        metadata={
            "source_document": "Bharatiya Nyaya Sanhita",
            "act_name": "Bharatiya Nyaya Sanhita",
            "section_number": "318",
        },
    )


def _service_with_retrieval() -> ChatService:
    service = _service_with_mocks()
    chunk = _chunk()
    service.retriever.retrieve = AsyncMock(return_value=("q", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    return service


def test_a_safe_decline_from_the_normal_path_carries_no_apparatus() -> None:
    """Retrieval succeeded and the LLM itself emitted the refusal. Every field
    that would suggest the refusal is supported must be empty."""
    service = _service_with_retrieval()
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content=no_verified_context_message("hindi"), model="test", provider="test")
    )
    response = asyncio.run(service.answer(ChatRequest(question="GST par kya niyam hai?")))

    assert response.no_verified_context is True
    assert response.confidence == 0.0
    assert response.sources == []
    assert response.evidence_pages == []
    assert response.applicable_law == []
    assert response.retrieved_sections == []
    assert response.retrieved_chunks == []
    assert response.currency_notice == ""


def test_a_safe_decline_is_not_stored_in_history_with_citations() -> None:
    """Sanitization happens before persistence, so a misleading citation is
    never written down either."""
    service = _service_with_retrieval()
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content=no_verified_context_message("hindi"), model="test", provider="test")
    )
    asyncio.run(service.answer(ChatRequest(question="GST par kya niyam hai?")))

    (record,), _ = service.history.insert.await_args
    assert record["sources"] == []


def test_a_streaming_safe_decline_carries_no_apparatus() -> None:
    service = _service_with_retrieval()
    refusal = no_verified_context_message("hindi")

    async def _stream(messages, temperature: float = 0.1):
        yield refusal

    service.llm.stream = _stream
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    events = asyncio.run(
        _collect(service.answer_stream(ChatRequest(question="GST par kya niyam hai?", stream=True)))
    )
    done = [event for event in events if event["event"] == "done"][-1]["data"]

    assert done["no_verified_context"] is True
    assert done["confidence"] == 0.0
    assert done["sources"] == []
    assert done["evidence_pages"] == []
    assert done["applicable_law"] == []
    assert done["retrieved_sections"] == []
    assert done["retrieved_chunks"] == []
    assert done["currency_notice"] == ""


async def _collect(agen) -> list[dict]:
    return [event async for event in agen]


# ---------------------------------------------------------------------------
# 9. A legacy cached refusal is sanitized on the way out
# ---------------------------------------------------------------------------


def test_a_legacy_cached_refusal_payload_is_sanitized() -> None:
    """Entries written before the invariant existed still carry the citations
    they were stored with."""
    legacy = {
        "answer": no_verified_context_message("hindi"),
        "sources": [{"source_document": "Bharatiya Nyaya Sanhita", "label": "BNS s.318"}],
        "retrieved_sections": ["318"],
        "currency_notice": "This Act was amended in 2023.",
        "applicable_law": ["Bharatiya Nyaya Sanhita, Section 318"],
        "confidence": 0.82,
    }
    cleaned = sanitize_payload(legacy)

    assert cleaned["sources"] == []
    assert cleaned["retrieved_sections"] == []
    assert cleaned["applicable_law"] == []
    assert cleaned["currency_notice"] == ""
    assert cleaned["confidence"] == 0.0
    assert cleaned["no_verified_context"] is True
    # The original dict is not mutated in place.
    assert legacy["sources"]


def test_a_cached_refusal_is_served_without_its_stored_citations() -> None:
    service = _service_with_mocks()
    service.response_cache.lookup = AsyncMock(
        return_value=(
            {
                "response": {
                    "answer": no_verified_context_message("hindi"),
                    "sources": [
                        {"source_document": "Bharatiya Nyaya Sanhita", "label": "BNS s.318", "section": "318"}
                    ],
                    "retrieved_sections": ["318"],
                    "confidence": 0.9,
                }
            },
            "exact",
        )
    )
    response = asyncio.run(service.answer(ChatRequest(question="GST par kya niyam hai?")))

    assert response.no_verified_context is True
    assert response.sources == []
    assert response.retrieved_sections == []
    assert response.currency_notice == ""


def test_sanitization_leaves_a_real_cached_answer_alone() -> None:
    payload = {
        "answer": "Section 138 of the Negotiable Instruments Act applies.",
        "sources": [{"source_document": "NI Act"}],
        "confidence": 0.8,
    }
    assert sanitize_payload(payload) is payload


# ---------------------------------------------------------------------------
# 10. A genuinely grounded answer keeps its real citations
# ---------------------------------------------------------------------------


def test_a_grounded_answer_keeps_its_citations() -> None:
    service = _service_with_retrieval()
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content=(
                "Cheating is defined in the Bharatiya Nyaya Sanhita. File a written complaint at the "
                "police station with the payment records and the messages you received."
            ),
            model="test",
            provider="test",
        )
    )
    response = asyncio.run(service.answer(ChatRequest(question="Mujhe online cheat kiya gaya hai, kya karun?")))

    assert response.no_verified_context is False
    assert response.confidence > 0.0
    assert len(response.sources) == 1
    assert response.retrieved_chunks


# ---------------------------------------------------------------------------
# The UI band, as defence in depth
# ---------------------------------------------------------------------------


def test_the_ui_hides_the_apparatus_for_a_legacy_refusal_payload() -> None:
    """The API is authoritative, but a payload replayed from history may
    predate the invariant."""
    legacy = {
        "answer": no_verified_context_message("hindi"),
        "sources": [{"source_document": "Bharatiya Nyaya Sanhita", "label": "BNS s.318"}],
        "applicable_law": ["Bharatiya Nyaya Sanhita, Section 318"],
        "evidence_pages": [{"source_document": "Bharatiya Nyaya Sanhita", "page_number": 4}],
    }
    assert suppresses_apparatus(legacy) is True
    assert source_lines(legacy) == []


def test_the_ui_honours_the_explicit_flag() -> None:
    assert suppresses_apparatus({"answer": "anything", "no_verified_context": True}) is True


def test_the_ui_still_renders_sources_for_a_real_answer() -> None:
    grounded = {
        "answer": "Section 138 of the Negotiable Instruments Act applies.",
        "sources": [{"source_document": "NI Act", "label": "NI Act s.138"}],
        "applicable_law": ["Negotiable Instruments Act, Section 138"],
    }
    assert suppresses_apparatus(grounded) is False
    assert source_lines(grounded)


# ---------------------------------------------------------------------------
# 11. The two post-release regressions, pinned in both directions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Correctly spelled everyday words that sit one edit from a legal term.
        "Contact me on this number",
        "contact the police station",
        "The latter option is cheaper",
        "What is your plan for the hearing?",
        "I will stare at the notice later",
    ],
)
def test_a_correctly_spelled_common_word_is_never_pulled_to_a_legal_term(text: str) -> None:
    """"Contact" must not become "Contract" merely because "contract" is one
    edit away and is on the legal allowlist."""
    result = normalize_for_routing(text)
    assert result.normalized_for_routing == text
    assert result.corrections == ()


def test_the_legal_terms_those_words_shadow_are_still_reachable() -> None:
    """The negative case above must not have disabled the corrections it sits
    next to."""
    assert "contract" in normalize_for_routing("mera contrct toot gaya").normalized_for_routing
    assert "notary" in normalize_for_routing("notry ke pass jana hai").normalized_for_routing


@pytest.mark.parametrize(
    "message",
    [
        "notry ke liye affidavit tyar kro",
        "notarizaton ke liye document prepare karo",
    ],
)
def test_allowlisted_notarization_misspellings_reach_a_notarization_workflow(message: str) -> None:
    from app.chatops.registry import best_match

    match = best_match(message, "hinglish")
    assert match is not None, message
    assert match[0].name.startswith("notar")


@pytest.mark.parametrize(
    "message",
    [
        # Names a notary as a fact; asks for no notarization capability.
        "Notary ka matlab kya hota hai?",
        "chek bouns hone pr secshun 138 me kya hota hai?",
    ],
)
def test_normalized_matching_does_not_invent_a_notarization_workflow(message: str) -> None:
    """The normalized form is scored, never substituted -- an ordinary legal
    question must still fall through to retrieval."""
    from app.chatops.registry import best_match

    match = best_match(message, "hinglish")
    assert match is None or not match[0].name.startswith("notarization_prepare")


def test_the_british_spelling_normalizes_to_the_workflow_vocabulary() -> None:
    """"notarisation" is a spelling, not a typo; the matchers are keyed on the
    "-z-" form, so it is normalized rather than left to miss."""
    result = normalize_for_routing("affidavit ko notarisation ke liye bhejo")
    assert "notarization" in result.normalized_for_routing
    assert result.original == "affidavit ko notarisation ke liye bhejo"
