"""Regression tests for the Part 58 "Answer Quality Audit" findings.

Each test names the audit issue it locks down. The audit was produced from a
single real multilingual session, so several of these use that session's exact
messages -- the point is that those specific inputs, which really did produce
the wrong behaviour, now produce the right one.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.constants import no_verified_context_message
from app.core.fallback_guidance import (
    actionable_guidance,
    detect_category,
    generic_next_steps_guidance,
)
from app.drafting.conversation import DraftConversationEngine
from app.drafting.intent import DraftIntentDetector, looks_informational
from app.intent.detector import parse_section_lookup, parse_section_lookup_act
from app.language.detector import LanguageDetector
from app.rag.query_rewriter import SmartQueryRewriter
from app.rag.statute_currency import annotate_answer, currency_directive
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import LawyerRecommendation, RetrievedChunk
from app.services.chat_service import (
    _BARE_REFUSAL_PATTERN,
    _CANCEL_DRAFT_PATTERN,
    _replace_source_ordinals,
    _source_label,
)
from app.utils.pii import mask_entities, mask_pii

_RECOMMENDATION = LawyerRecommendation(category="Criminal Law", confidence=0.6, reason="Test fixture.")


def _service_with_mocks(messages: list[dict[str, str]] | None = None, memory: dict | None = None):
    """Same shape as `tests/test_chat_service_routing._service_with_mocks`,
    duplicated rather than imported so these tests stay readable on their own
    and can seed extra memory keys (draft state) the routing suite doesn't."""
    from app.services.chat_service import ChatService

    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])
    base = {"messages": messages or [], "summary": "", "current_intent": None, "legal_category": None}
    base.update(memory or {})
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
    return service


# ---------------------------------------------------------------------------
# Issue 1 -- core KB coverage: the same provision must be reachable topically,
# by its current number, and in every language, not only via the IPC crosswalk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "BNS 318 kya hai?",
        "BNS 318",
        "BNSS 35 kya hota hai",
        "IPC 420 ka matlab kya hai",
        "धारा 173 क्या है",
    ],
)
def test_act_prefixed_citation_with_explanatory_filler_is_a_section_lookup(question: str) -> None:
    # "BNS 318 kya hai?" named the Act instead of the word "section", so it
    # matched none of the whole-message citation shapes and was never
    # classified SECTION_LOOKUP -- which skipped the exact-section-number
    # rescue in retrieval and produced "no verified document" for a section
    # the KB indexes.
    assert parse_section_lookup(question) is not None


def test_ipc_citation_with_filler_still_resolves_its_act() -> None:
    assert parse_section_lookup("IPC 420 ka matlab kya hai") == "318"
    assert parse_section_lookup_act("IPC 420 ka matlab kya hai") == "BNS"


def test_bns_section_number_resolves_its_act_for_retrieval() -> None:
    assert parse_section_lookup_act("BNS 318 kya hai?") == "BNS"
    assert parse_section_lookup_act("BNSS 35 kya hota hai") == "BNSS"


def test_a_section_number_mentioned_in_passing_is_not_a_citation_lookup() -> None:
    assert parse_section_lookup("cheque bounce case under section 138") is None
    assert parse_section_lookup("I paid 500 rupees to the shop") is None


@pytest.mark.parametrize(
    "question",
    [
        "What is the punishment for cheating under BNS?",
        "धोखाधड़ी के लिए BNS में क्या सजा है?",
        "BNS 318 kya hai?",
    ],
)
def test_cheating_questions_expand_to_the_provisions_own_vocabulary(question: str) -> None:
    # All three asked about the same provision the IPC-420 question already
    # answered correctly, and all three returned "no verified document" --
    # the corpus indexes it under its statutory heading ("cheating and
    # dishonestly inducing delivery of property"), which none of these
    # phrasings shares enough words with to clear the relevance gate.
    expanded = " ".join(SmartQueryRewriter().expand_queries(question)).lower()
    assert "318" in expanded
    assert "cheating" in expanded


def test_arrest_question_retrieves_the_safeguards_not_just_the_power() -> None:
    # Issue 4: the answer explained only that a cognizable offence permits
    # arrest. The safeguards can only reach the answer if they reach the
    # context first.
    expanded = " ".join(
        SmartQueryRewriter().expand_queries("पुलिस बिना वारंट के कब गिरफ्तार कर सकती है?")
    ).lower()
    assert "recorded in writing" in expanded
    assert "notice of appearance" in expanded


def test_threat_question_expands_to_the_relevant_offences() -> None:
    expanded = " ".join(
        SmartQueryRewriter().expand_queries("एक व्यक्ति मुझे लगातार फोन करके परेशान और धमकी दे रहा है।")
    ).lower()
    assert "criminal intimidation" in expanded


# ---------------------------------------------------------------------------
# Issue 2 -- repealed-code citations must not be presented as current law
# ---------------------------------------------------------------------------


def test_bare_crpc_section_in_an_fir_answer_gets_a_currency_note() -> None:
    answer = (
        "Agar kisi cognizable offence ki jaankari milti hai, toh Section 154 ke under FIR register karna "
        "mandatory hai."
    )
    annotated = annotate_answer(answer, "hinglish")
    assert "173" in annotated
    assert "BNSS" in annotated
    assert answer in annotated  # additive only -- the answer itself is untouched


def test_an_answer_that_already_cites_the_current_provision_is_left_alone() -> None:
    answer = "Under Section 173 of the BNSS, an FIR can be registered through electronic communication."
    assert annotate_answer(answer, "english") == answer


def test_a_same_numbered_section_of_an_unrelated_act_is_not_annotated() -> None:
    # "Section 125" is CrPC maintenance, but it is also a section of plenty of
    # other Acts -- the topic-keyword guard is what keeps this from firing.
    answer = "You may pay the Section 125 charges under the Electricity Act."
    assert annotate_answer(answer, "english") == answer
    answer2 = "The consumer can file under Section 35 of the Consumer Protection Act, 2019."
    assert annotate_answer(answer2, "english") == answer2


def test_currency_directive_is_empty_for_an_ordinary_question() -> None:
    assert currency_directive("What is a rent agreement?", "A rent agreement records the terms of tenancy.") == ""


def test_currency_directive_names_the_mapping_when_context_cites_the_old_code() -> None:
    directive = currency_directive("What is an FIR", "Section 154 CrPC information in cognizable cases")
    assert "BNSS Section 173" in directive


# ---------------------------------------------------------------------------
# Issue 3 & 14 -- language routing
# ---------------------------------------------------------------------------


def test_gujarati_ending_in_a_danda_is_not_detected_as_hindi() -> None:
    # Issue 14: "।" (U+0964) is shared Indic punctuation that happens to live
    # in the Devanagari block, so a wholly Gujarati sentence ending in one was
    # routed down the Devanagari branch and answered in Hindi.
    assert LanguageDetector().detect("મારી સાથે છેતરપિંડી થઈ છે. ફરિયાદ તૈયાર કરો।") == "gujarati"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("माझ्या बँक खात्यातून २५,००० काढण्यात आले आहेत.", "marathi"),
        ("धोखाधड़ी के लिए BNS में क्या सजा है?", "hindi"),
        ("আমি অনলাইনে একটি মোবাইল কিনেছিলাম।", "bengali"),
        ("What is the punishment for cheating under BNS?", "english"),
    ],
)
def test_other_scripts_still_detect_correctly(text: str, expected: str) -> None:
    assert LanguageDetector().detect(text) == expected


def test_system_prompt_pins_the_reply_language_after_the_examples() -> None:
    # Issue 3: the second "What is an FIR" came back with a Hinglish heading
    # ("FIR kya hoti hai?") that is verbatim the system prompt's own worked
    # example -- the model reproduced the example instead of answering. The
    # examples now say what they are, and the language rule is restated after
    # them so it is the last instruction the model reads.
    from app.llm.prompts import prompt_registry

    prompt = prompt_registry.load("system_prompt")
    examples_at = prompt.index("Three worked examples of SHAPE ONLY")
    assert "never reuse their wording" in prompt[examples_at:]
    assert prompt.rindex("write the entire reply in {language}") > examples_at


# ---------------------------------------------------------------------------
# Issue 4 & 6 & 12 -- no unqualified legal conclusions
# ---------------------------------------------------------------------------


def test_drafting_prompt_forbids_conclusive_offence_classification() -> None:
    from app.llm.prompts import prompt_registry

    prompt = prompt_registry.load("legal_drafting_prompt")
    assert "prima facie" in prompt
    assert "IS a cognizable offence" in prompt


def test_drafting_prompt_forbids_strengthening_the_users_own_certainty() -> None:
    from app.llm.prompts import prompt_registry

    prompt = prompt_registry.load("legal_drafting_prompt")
    assert "सुनियोजित" in prompt          # "premeditated", added to a suspicion
    assert "स्पष्ट संकेत" in prompt        # "clear indications", never stated by the user
    assert "पूर्ण उदासीनता" in prompt      # "complete indifference", a conclusion about the seller


def test_system_prompt_requires_safeguards_alongside_a_coercive_power() -> None:
    from app.llm.prompts import prompt_registry

    prompt = prompt_registry.load("system_prompt")
    assert "Coercive powers must never be stated without their conditions" in prompt


# ---------------------------------------------------------------------------
# Issue 7 -- no unsolicited escalation threat in a complaint
# ---------------------------------------------------------------------------


def test_complaint_guidance_forbids_an_escalation_clause_the_user_never_asked_for() -> None:
    from app.drafting.engine import _narrative_guidance

    complaint = _narrative_guidance("Complaint")
    assert "Do not add a deadline, a warning" in complaint
    assert "escalate to higher authorities" in complaint


def test_notice_guidance_keeps_its_consequence_clause() -> None:
    from app.drafting.engine import _narrative_guidance

    assert "consequence of non-compliance" in _narrative_guidance("Notice")


# ---------------------------------------------------------------------------
# Issue 8 -- the document's own language reaches the drafting prompt
# ---------------------------------------------------------------------------


def test_field_labels_are_localized_in_the_drafting_prompt() -> None:
    from app.drafting.engine import LegalDraftEngine
    from app.drafting.templates import get_template

    template = get_template("police_complaint")
    rendered = LegalDraftEngine()._render_fields(
        template, {"applicant_name": "राहुल शर्मा", "police_station": "गोमती नगर"}, "hindi"
    )
    assert "आवेदक का नाम" in rendered
    assert "पुलिस थाना" in rendered


def test_english_drafts_are_unaffected_by_label_localization() -> None:
    from app.drafting.engine import LegalDraftEngine
    from app.drafting.templates import get_template

    template = get_template("police_complaint")
    rendered = LegalDraftEngine()._render_fields(template, {"applicant_name": "Rahul Sharma"}, "english")
    assert rendered == "Applicant Name: Rahul Sharma"


# ---------------------------------------------------------------------------
# Issue 9 -- the preview hint must describe the flow that actually exists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["hindi", "marathi", "gujarati", "bengali", "tamil", "urdu"])
def test_preview_hints_point_at_download_not_the_dead_approve_command(language: str) -> None:
    from app.drafting.wrapper_messages import msg

    for key in ("draft_edit_hint", "draft_ready_for_review"):
        text = msg(key, language, "fallback {template}", template="X")
        assert '"approve"' not in text, f"{key}/{language} still tells the user to say approve"


def test_preview_hint_survives_for_an_untranslated_language() -> None:
    from app.drafting.wrapper_messages import msg

    assert msg("draft_edit_hint", "english", "ENGLISH DEFAULT") == "ENGLISH DEFAULT"


# ---------------------------------------------------------------------------
# Issue 10 -- a city is not an address
# ---------------------------------------------------------------------------


def test_drafting_prompt_forbids_promoting_a_city_into_an_address() -> None:
    from app.llm.prompts import prompt_registry

    prompt = prompt_registry.load("legal_drafting_prompt")
    assert "Address fields are not interchangeable" in prompt
    assert "never present a bare city name as the applicant's postal address" in prompt


# ---------------------------------------------------------------------------
# Issue 11 -- no invented background facts
# ---------------------------------------------------------------------------


def test_drafting_prompt_forbids_invented_background_detail() -> None:
    from app.llm.prompts import prompt_registry

    prompt = prompt_registry.load("legal_drafting_prompt")
    assert "regular financial transactions" in prompt
    assert "requires urgent legal action" in prompt


# ---------------------------------------------------------------------------
# Issue 13 -- consumer complaint jurisdiction has to be established
# ---------------------------------------------------------------------------


def test_consumer_complaint_template_requires_a_jurisdiction_paragraph() -> None:
    from app.drafting.templates import get_template

    notes = get_template("consumer_complaint").drafting_notes
    assert "jurisdiction" in notes.lower()
    for basis in ("residence", "cause of action", "opposite party", "value"):
        assert basis in notes.lower(), f"jurisdiction basis {basis!r} is not required by the template"


# ---------------------------------------------------------------------------
# Issue 15 -- an ambiguous drafting request must ask, not guess
# ---------------------------------------------------------------------------


def test_generic_fraud_complaint_request_asks_instead_of_picking_bank_fraud() -> None:
    # The user said only "I've been cheated, prepare a complaint" -- no bank,
    # no transaction. Starting a Bank Fraud Complaint then demanded a bank
    # name and a UTR number they had never mentioned.
    match = DraftIntentDetector().detect("મારી સાથે છેતરપિંડી થઈ છે. ફરિયાદ તૈયાર કરો।")
    assert match.matched
    assert match.ambiguous
    assert match.draft_id is None
    assert "bank_fraud_complaint" in match.candidates
    assert "online_fraud_complaint" in match.candidates


def test_kashmiri_police_notice_question_is_not_hijacked_by_drafting() -> None:
    question = (
        "می ہس اکھ پولیس نوٹِس ملیو، یہِ حاضر گژھُن چھُ لکھنہ آمُت۔ "
        "کیانہٕ کرُن چھُ فیصلہ کرنہ برونٹھ کُن کیا تفصیلات تصدیق کرُن ضروری؟"
    )

    assert looks_informational(question)
    assert not DraftIntentDetector().detect(question).matched


def test_a_clearly_named_document_is_still_matched_outright() -> None:
    detector = DraftIntentDetector()
    for message, expected in [
        ("माझ्या बँक खात्यातून पैसे काढले. सायबर क्राईम तक्रार तयार करा.", "cyber_crime_complaint"),
        ("Generate a police complaint", "police_complaint"),
        ("RTI application banana hai.", "rti_application"),
        ("Consumer complaint draft.", "consumer_complaint"),
    ]:
        match = detector.detect(message)
        assert match.draft_id == expected, f"{message!r} -> {match.draft_id!r}"


def test_ambiguous_request_shows_the_shortlist_before_the_full_catalogue() -> None:
    engine = DraftConversationEngine()
    memory: dict = {}
    result = asyncio.run(
        engine.handle_turn("s-amb", "મારી સાથે છેતરપિંડી થઈ છે. ફરિયાદ તૈયાર કરો।", "gujarati", memory)
    )
    assert result is not None
    assert result.info.stage == "selecting"
    shortlist_at = result.reply_text.index("Bank Fraud Complaint")
    full_list_at = result.reply_text.index("Account Closure Application")
    assert shortlist_at < full_list_at


# ---------------------------------------------------------------------------
# Issues 16 & 23 -- an unanswerable urgent question must still be actionable
# ---------------------------------------------------------------------------


def test_threat_question_is_classified_as_needing_safety_guidance() -> None:
    category = detect_category("एक व्यक्ति मुझे लगातार फोन करके परेशान और धमकी दे रहा है। पुलिस शिकायत कैसे लिखूं?")
    assert category is not None
    assert category.draft_id == "threat_complaint"


def test_guidance_names_official_channels_evidence_and_escalation() -> None:
    guidance = actionable_guidance("someone keeps threatening me", "english", "Threat Complaint", "Threat Complaint")
    assert "112" in guidance
    assert "Superintendent of Police" in guidance
    assert "Preserve the evidence" in guidance
    assert "qualified advocate" in guidance
    assert "Threat Complaint" in guidance


def test_guidance_declares_its_source_and_is_not_offered_as_legal_analysis() -> None:
    guidance = actionable_guidance("my account was debited without my consent", "english")
    assert "Government of India" in guidance
    assert "not an interpretation of the law" in guidance


def test_no_guidance_is_attached_to_an_ordinary_informational_question() -> None:
    assert actionable_guidance("What is the punishment for cheating under BNS?", "english") == ""


# ---------------------------------------------------------------------------
# Regression tests for qa-40q-multilingual-20260921 BUG-01/BUG-02: the
# strict-RAG refusal itself was correctly localized, but the actionable-
# guidance block appended after it silently fell back to English for any
# language whose coverage didn't match across `_STEPS`/`_HEADINGS`/
# `_CHANNEL_LABELS`/`_SOURCE_NOTES`/`_DRAFT_OFFERS` (`actionable_guidance`)
# or `_HEADINGS`/`_GENERIC_NEXT_STEPS` (`generic_next_steps_guidance`) --
# producing a native-script refusal followed by an English or mixed-script
# guidance block in the SAME reply.
# ---------------------------------------------------------------------------


def test_actionable_guidance_draft_offer_stays_in_the_user_language() -> None:
    # "மிரட்ட"/"தொல்லை" (threaten/harass) trigger `threat_harassment` in
    # Tamil. `_DRAFT_OFFERS` previously had no Tamil entry even though
    # `_STEPS`/`_HEADINGS`/`_CHANNEL_LABELS`/`_SOURCE_NOTES` all did, so this
    # exact call used to end with an English "I can also prepare a ... draft
    # for you" sentence tacked onto an otherwise-Tamil block.
    guidance = actionable_guidance(
        "ஒருவர் தொடர்ந்து என்னை மிரட்டுகிறார்", "tamil", "மிரட்டல் புகார்", "Threat Complaint",
    )
    assert guidance
    assert "I can also prepare" not in guidance
    assert "வரைவையும்" in guidance


@pytest.mark.parametrize(
    "language,heading_fragment",
    [
        ("tamil", "இதற்கிடையில்"),
        ("telugu", "ఈలోగా"),
        ("kannada", "ಈ ಮಧ್ಯೆ"),
        ("malayalam", "അതിനിടെ"),
        ("punjabi", "ਇਸ ਦੌਰਾਨ"),
        ("odia", "ଏହି ସମୟରେ"),
        ("urdu", "اس دوران"),
        ("marathi", "दरम्यान"),
        ("gujarati", "દરમિયાન"),
        ("bengali", "ইতিমধ্যে"),
    ],
)
def test_generic_next_steps_guidance_body_matches_the_localized_heading(
    language: str, heading_fragment: str
) -> None:
    # Previously `_HEADINGS` covered these 10 languages but `_GENERIC_NEXT_
    # STEPS` covered only english/hindi/hinglish, so an ordinary informational
    # question ("BNS Section 12 kya hai"-shaped) that missed the Knowledge
    # Base got a native-script heading followed by English numbered steps --
    # a same-block script mix, not a clean fallback.
    guidance = generic_next_steps_guidance(language)
    assert heading_fragment in guidance
    assert "Name the exact Act and section number" not in guidance
    assert "qualified advocate can confirm" not in guidance


def test_unanswerable_urgent_question_keeps_the_refusal_and_adds_next_steps() -> None:
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("threat", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.recommendations.recommend = AsyncMock(return_value=_RECOMMENDATION)
    service.llm.chat = AsyncMock(
        side_effect=AssertionError("the strict-RAG guardrail must not call the LLM")
    )
    response = asyncio.run(
        service.answer(ChatRequest(question="Someone keeps calling and threatening me. How do I complain?"))
    )
    # The guardrail itself is untouched: the refusal is still the first thing
    # the user reads, verbatim.
    assert response.answer.startswith(no_verified_context_message("english"))
    assert response.confidence == 0.0
    assert "1930" in response.answer or "112" in response.answer


# ---------------------------------------------------------------------------
# Issues 17, 18, 21, 22 -- cancelling a draft must actually cancel it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "cancel kr do draft ko",
        "draft cancel kar do",
        "cancel draft",
        "ड्राफ्ट रद्द करें",
        "मसौदा हटा दें",
        "मसुदा रद्द करा",
        "ડ્રાફ્ટ રદ કરો",
        "delete this draft",
    ],
)
def test_cancel_phrasings_are_recognized(message: str) -> None:
    assert _CANCEL_DRAFT_PATTERN.search(message)


@pytest.mark.parametrize(
    "message",
    [
        "Facts: the accused deleted my documents from the drive",
        "Gomti Nagar Police Station, Lucknow",
        "I want to draft a police complaint",
    ],
)
def test_ordinary_messages_are_not_mistaken_for_cancellation(message: str) -> None:
    assert not _CANCEL_DRAFT_PATTERN.search(message)


def test_rehne_do_meaning_preserve_is_not_mistaken_for_cancellation() -> None:
    # BUG-012 (QA pass, 2026-09-11, CRITICAL): "rehne do" is genuinely
    # ambiguous in Hindi/Hinglish -- "leave it, forget it" (dismissal) OR
    # "let it remain/stay as it is" (the OPPOSITE: preservation). Live
    # repro: an explanation request that explicitly asked NOT to modify the
    # draft ("...lekin draft Hindi mein hi rehne do, badlo mat.") was
    # misread as a cancel command purely because "draft" and "rehne"
    # appeared within the pattern's 20-character proximity window, and the
    # whole draft (10 turns of accumulated edits) was reported as deleted.
    # "rehne do" alone, with nothing else in the message, is still
    # recognised (see `test_cancel_phrasings_are_recognized`-adjacent bare
    # dismissal branch) -- only the loose "draft" + "rehne" proximity match
    # was removed.
    message = (
        "Ye draft mujhe Hinglish mein samjha do, lekin draft Hindi mein hi rehne do, badlo mat."
    )
    assert not _CANCEL_DRAFT_PATTERN.search(message), message
    # A second, shorter phrasing of the same preservation sense.
    assert not _CANCEL_DRAFT_PATTERN.search("Draft Hindi mein hi rehne do."), "Draft Hindi mein hi rehne do."


def test_bare_rehne_do_alone_is_still_recognized_as_cancellation() -> None:
    # The narrow, low-ambiguity case BUG-012's fix deliberately preserved:
    # a message that is NOTHING but the dismissal phrase.
    assert _CANCEL_DRAFT_PATTERN.search("rehne do")
    assert _CANCEL_DRAFT_PATTERN.search("Rehne do.")


def test_cancel_is_never_scored_as_a_request_to_start_a_new_document() -> None:
    # Issue 22: "cancel kr do draft ko" contains both "draft" and the Hinglish
    # imperative "kr do", so the drafting detector read it as an ambiguous new
    # drafting request and answered with the full 56-template menu.
    assert not DraftIntentDetector().detect("cancel kr do draft ko").matched


def test_cancel_clears_parked_drafts_and_language_too() -> None:
    # Issue 18: the system said "I've discarded that draft" and then, in the
    # same reply, "your Bank Fraud Complaint draft is still saved".
    engine = DraftConversationEngine()
    memory: dict = {
        "draft_mode": True,
        "draft_stage": "collecting",
        "draft_template_id": "bank_fraud_complaint",
        "draft_fields": {"applicant_name": "A"},
        "draft_id": "d1",
        "draft_language": "hindi",
        "draft_paused": True,
        "parked_drafts": [{"draft_template_id": "police_complaint", "draft_stage": "collecting", "draft_fields": {}}],
    }
    engine.reset(memory)
    assert memory["draft_mode"] is False
    assert memory["draft_fields"] == {}
    assert memory["parked_drafts"] == []
    assert memory["draft_paused"] is False
    assert memory["draft_language"] is None
    assert engine.describe_pending(memory) is None


def test_cancel_reply_is_written_in_the_drafts_own_language() -> None:
    service = _service_with_mocks(
        memory={
            "draft_mode": True,
            "draft_stage": "collecting",
            "draft_template_id": "bank_fraud_complaint",
            "draft_fields": {},
            "draft_language": "hindi",
        }
    )
    service.recommendations.recommend = AsyncMock(return_value=_RECOMMENDATION)
    response = asyncio.run(service.answer(ChatRequest(question="cancel kr do draft ko")))
    assert "मसौदा" in response.answer
    persisted = dict(service.memory.update.await_args_list[0].kwargs)
    assert persisted["draft_mode"] is False
    assert persisted["parked_drafts"] == []
    assert persisted["draft_paused"] is False


# ---------------------------------------------------------------------------
# Issues 19, 20, 21 -- the draft state machine must not trap the user
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "जमानत और अग्रिम जमानत में क्या अंतर है?",
        "जमानत क्या होती है?",
        "What is the difference between bail and anticipatory bail?",
        "FIR kaise file karte hain?",
    ],
)
def test_a_real_question_reads_as_a_question(question: str) -> None:
    # Issue 19: "... में क्या अंतर है?" put a noun between "क्या" and "है",
    # which none of the informational patterns allowed for -- so a genuine
    # legal question asked mid-draft was swallowed as field input and the
    # pending-field list was shown again.
    assert looks_informational(question)


@pytest.mark.parametrize(
    "field_value",
    [
        "Rahul Sharma",
        "Gomti Nagar Police Station, Lucknow",
        "24, Shanti Vihar, Gomti Nagar, Lucknow, Uttar Pradesh - 226010",
        (
            "दिनांक 25 अगस्त 2026 को मुझे एक व्यक्ति ने फोन करके ऑनलाइन निवेश में अधिक लाभ देने का झांसा दिया "
            "और मैंने ₹35,000 ट्रांसफर कर दिए, अब उसका नंबर बंद है।"
        ),
    ],
)
def test_a_field_answer_does_not_read_as_a_question(field_value: str) -> None:
    assert not looks_informational(field_value)


@pytest.mark.parametrize("message", ["nhi", "no", "नहीं", "Nahi", "no thanks", "नको"])
def test_bare_refusals_are_recognized(message: str) -> None:
    assert _BARE_REFUSAL_PATTERN.match(message)


@pytest.mark.parametrize("message", ["Nohar, Rajasthan", "9876543210", "No objection certificate"])
def test_a_field_value_is_not_mistaken_for_a_refusal(message: str) -> None:
    assert not _BARE_REFUSAL_PATTERN.match(message)


def test_refusal_pauses_the_draft_instead_of_repeating_the_field_list() -> None:
    # Issue 20: the user said "nhi" and got back the exact nine-field list
    # they were saying no to.
    service = _service_with_mocks(
        memory={
            "draft_mode": True,
            "draft_stage": "collecting",
            "draft_template_id": "bank_fraud_complaint",
            "draft_fields": {},
            "draft_language": "hindi",
        }
    )
    service.recommendations.recommend = AsyncMock(return_value=_RECOMMENDATION)
    response = asyncio.run(service.answer(ChatRequest(question="nhi")))
    assert "Applicant Address" not in response.answer
    assert "continue draft" in response.answer
    assert "cancel draft" in response.answer


def test_a_paused_draft_releases_the_next_message() -> None:
    # Issue 21: an unrelated question, a refusal and a cancel all led back
    # into field collection. Once paused, the engine hands the turn back.
    engine = DraftConversationEngine()
    memory: dict = {
        "draft_mode": True,
        "draft_stage": "collecting",
        "draft_template_id": "bank_fraud_complaint",
        "draft_fields": {},
        "draft_language": "hindi",
    }
    engine.pause(memory)
    assert memory["draft_paused"] is True
    assert asyncio.run(engine.handle_turn("s", "जमानत और अग्रिम जमानत में क्या अंतर है?", "hindi", memory)) is None
    # ...and comes back on request, with the collected fields intact.
    resumed = asyncio.run(engine.handle_turn("s", "continue draft", "hindi", memory))
    assert resumed is not None
    assert memory["draft_paused"] is False


def test_a_paused_draft_still_answers_to_a_named_document() -> None:
    engine = DraftConversationEngine()
    memory: dict = {
        "draft_mode": True,
        "draft_stage": "collecting",
        "draft_template_id": "bank_fraud_complaint",
        "draft_fields": {},
        "draft_language": "english",
        "draft_paused": True,
    }
    result = asyncio.run(engine.handle_turn("s", "Generate a police complaint", "english", memory))
    assert result is not None
    assert memory["draft_template_id"] == "police_complaint"


def test_pause_is_a_no_op_while_the_user_is_still_choosing_a_template() -> None:
    # At "selecting" the engine has just asked "which document?" -- the next
    # message is the answer to its own question, so pausing there would drop
    # it into ordinary chat.
    engine = DraftConversationEngine()
    memory: dict = {"draft_mode": True, "draft_stage": "selecting", "draft_fields": {}}
    engine.pause(memory)
    assert not memory.get("draft_paused")


# ---------------------------------------------------------------------------
# Issue 24 -- citations the reader can actually resolve
# ---------------------------------------------------------------------------


def test_a_leaked_source_ordinal_is_replaced_by_the_real_citation() -> None:
    chunks = [
        RetrievedChunk(chunk_id="a", text="...", score=0.7, metadata={"source_document": "faq.pdf"}),
        RetrievedChunk(
            chunk_id="b",
            text="...",
            score=0.7,
            metadata={
                "source_document": "bns_2023.pdf",
                "act_name": "Bharatiya Nyaya Sanhita, 2023",
                "section_number": "318",
            },
        ),
    ]
    answer = "Source 2 ke mutabiq, cheating ab Section 318 ke under aata hai."
    rewritten = _replace_source_ordinals(answer, chunks)
    assert "Source 2" not in rewritten
    assert "Bharatiya Nyaya Sanhita, 2023, Section 318" in rewritten


def test_an_out_of_range_ordinal_degrades_instead_of_citing_the_wrong_act() -> None:
    chunks = [RetrievedChunk(chunk_id="a", text="...", score=0.7, metadata={"source_document": "faq.pdf"})]
    rewritten = _replace_source_ordinals("As per Source 9, ...", chunks)
    assert "Source 9" not in rewritten
    assert "retrieved source material" in rewritten


def test_source_label_names_the_act_and_section_when_metadata_has_them() -> None:
    chunk = RetrievedChunk(
        chunk_id="a",
        text="...",
        score=0.7,
        metadata={
            "source_document": "bnss_2023.pdf",
            "act_name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
            "section_number": "173",
        },
    )
    assert _source_label(chunk) == "Bharatiya Nagarik Suraksha Sanhita, 2023, Section 173 (bnss_2023.pdf)"


def test_source_label_falls_back_to_the_document_rather_than_inventing_an_act() -> None:
    chunk = RetrievedChunk(chunk_id="a", text="...", score=0.7, metadata={"source_document": "bprd_handbook.pdf"})
    assert _source_label(chunk) == "bprd_handbook.pdf"


def test_answers_without_a_source_reference_are_untouched() -> None:
    answer = "An FIR sets the criminal law in motion."
    assert _replace_source_ordinals(answer, []) == answer


# ---------------------------------------------------------------------------
# Issue 25 -- personal data must not accumulate in logs and analytics
# ---------------------------------------------------------------------------


def test_mask_pii_redacts_the_identifiers_a_drafting_turn_carries() -> None:
    masked = mask_pii(
        "Applicant Address: 24, Shanti Vihar, Gomti Nagar, Lucknow, Uttar Pradesh - 226010 "
        "Mobile Number: 9876543210 Email: rahul@example.com PAN ABCDE1234F"
    )
    assert "Shanti Vihar" not in masked
    assert "9876543210" not in masked
    assert "rahul@example.com" not in masked
    assert "ABCDE1234F" not in masked


def test_mask_pii_redacts_transaction_and_account_references() -> None:
    masked = mask_pii("Transaction ID / UTR: HDFC260828458721, IFSC HDFC0001234")
    assert "HDFC260828458721" not in masked
    assert "HDFC0001234" not in masked


def test_mask_pii_leaves_legal_citations_and_amounts_alone() -> None:
    for text in [
        "What is the punishment for cheating under BNS Section 318?",
        "Under Section 173 of the BNSS an FIR can be registered.",
        "Section 35, Consumer Protection Act, 2019, applies here.",
        "I paid Rs 18,500 on 15 August 2026.",
        "IPC 420 ab BNS ki kaunsi dhara hai?",
    ]:
        assert mask_pii(text) == text, text


def test_mask_entities_redacts_string_and_list_values() -> None:
    masked = mask_entities({"phone": "9876543210", "docs": ["rahul@example.com", "receipt"], "count": 2})
    assert "9876543210" not in masked["phone"]
    assert "rahul@example.com" not in masked["docs"][0]
    assert masked["docs"][1] == "receipt"
    assert masked["count"] == 2


def test_query_log_stores_redacted_text() -> None:
    # Exercised through `_log_query` directly rather than a full turn: the
    # point is that the analytics record is redacted at the one place it is
    # written, and a full turn would drag in retrieval/embeddings that have
    # nothing to do with this.
    service = _service_with_mocks()
    request = ChatRequest(
        question="My address is 24, Shanti Vihar, Gomti Nagar, Lucknow and my number is 9876543210"
    )
    response = ChatResponse(
        message_id="m1",
        session_id="s1",
        answer="Noted. Your number 9876543210 has been recorded in the draft.",
        sources=[],
        confidence=0.8,
        confidence_label="High",
        confidence_reason="test",
        lawyer_recommendation=_RECOMMENDATION,
        detected_language="english",
        detected_intent="General Legal Query",
        conversation_intent="Draft Generation",
        extracted_entities={"phone": "9876543210", "address": "24, Shanti Vihar, Gomti Nagar, Lucknow"},
        latency_ms=12.0,
    )
    asyncio.run(service._log_query(request, "s1", "m1", response, None))
    logged = service.query_log.insert.await_args.args[0]
    assert "9876543210" not in logged["question"]
    assert "Shanti Vihar" not in logged["question"]
    assert "9876543210" not in logged["answer"]
    assert "9876543210" not in logged["entities"]["phone"]


def test_intent_analytics_events_store_redacted_text() -> None:
    service = _service_with_mocks()
    asyncio.run(
        service._log_intent_event(
            "s1",
            {
                "question": "Transfer 9876543210 par kiya tha",
                "primary_intent": "Draft Generation",
                "detected_intents": ["Draft Generation"],
                "confidence": 0.9,
                "reason": "test",
                "classifier_source": "rules",
                "is_correction": False,
                "corrected_text": None,
            },
        )
    )
    stored = service.intent_events.insert.await_args.args[0]
    assert "9876543210" not in stored["question"]


def test_session_deletion_erases_analytics_records_too() -> None:
    from app.repositories.analytics import IntentEventRepository, QueryLogRepository
    from app.repositories.chat_history import ChatRepository

    # Issue 25: `DELETE /session` cleared only conversation memory, so the
    # same address/phone/transaction text stayed behind in these three.
    for repository in (ChatRepository, QueryLogRepository, IntentEventRepository):
        assert hasattr(repository, "delete_by_session")


def test_refusal_prefaced_by_an_empathy_sentence_is_still_recognized() -> None:
    from app.core.constants import is_no_verified_context

    refusal = no_verified_context_message("hinglish")
    assert is_no_verified_context(f"Yeh sunkar bahut bura laga ki aapka phone chori ho gaya. {refusal}")
    assert not is_no_verified_context("A long real answer. " * 60 + refusal)


def test_unrelated_cited_act_is_pruned_when_the_answer_never_mentions_it() -> None:
    from types import SimpleNamespace as NS

    from app.services.safe_decline import prune_unreferenced_sources

    answer = "Under the Consumer Protection Act, 2019 you may file before the District Commission."
    sources = [NS(act_name="THE CONSUMER PROTECTION ACT"), NS(act_name="THE BOMBAY GAS SUPPLY ACT"), NS(act_name=None)]
    kept, _ = prune_unreferenced_sources(answer, sources, [])
    assert [s.act_name for s in kept] == ["THE CONSUMER PROTECTION ACT", None]
    # Nothing referenced -> nothing pruned (never leave an answer sourceless).
    kept2, _ = prune_unreferenced_sources("Under Section 10 of some other law you may proceed.", sources, [])
    assert len(kept2) == 3
    # A reply that names no provision and references no source (a clarifying
    # question) rests on none of them -- confirmed live with a Marathi reply
    # carrying five unrelated Maharashtra Acts.
    kept3, ranked3 = prune_unreferenced_sources("Do you have the rent agreement? When did you vacate?", sources, [1])
    assert kept3 == [] and ranked3 == []


@pytest.mark.parametrize("language", ["nepali", "maithili", "sanskrit", "konkani", "dogri", "assamese", "sindhi"])
def test_generic_guidance_is_fully_native_for_the_added_languages(language: str) -> None:
    text = generic_next_steps_guidance(language)
    assert "In the meantime" not in text and "Name the exact Act" not in text
    assert "Knowledge Base" in text  # the product term stays untranslated, like every other language


@pytest.mark.parametrize("language", ["bodo", "manipuri", "santali", "kashmiri"])
def test_languages_without_reviewed_guidance_fall_back_to_english_consistently(language: str) -> None:
    text = generic_next_steps_guidance(language)
    assert "In the meantime" in text and "Name the exact Act" in text


def test_urgent_guidance_never_mixes_a_native_heading_with_english_steps() -> None:
    text = actionable_guidance("someone keeps threatening me", "nepali", "Threat Complaint", "Threat Complaint")
    assert "In the meantime" in text  # whole block English, heading included
    assert "यसै बीचमा" not in text


def test_a_pasted_excerpt_is_remembered_and_follow_ups_are_answered_from_it() -> None:
    import json as _json

    from app.llm.base import LLMResponse
    from app.schemas.chat import ChatRequest as _Req

    excerpt = (
        "Tenant: Meera. Landlord: Dev. Clause 7: Deposit shall be refunded within 30 days after return of keys, "
        "subject only to documented unpaid utility charges."
    )
    service = _service_with_mocks()
    memory: dict = {}
    first = _Req(question=f"Read only this synthetic agreement excerpt:\n\u201c{excerpt}\u201d\nQuestion: refund period?")
    assert asyncio.run(service._maybe_answer_from_inline_document(first, "s", "english", memory, 0.0)) is None
    assert memory["inline_document"]["text"].startswith("Tenant: Meera")

    service.llm.chat = AsyncMock(return_value=LLMResponse(
        content=_json.dumps({"found_in_document": True, "answer": "No. Clause 7 allows only documented unpaid utility charges.",
                             "supporting_quote": "subject only to documented unpaid utility charges"}),
        model="t", provider="t",
    ))
    follow = _Req(question="Kya Clause 7 painting charges deduct karne ki permission deta hai?")
    response = asyncio.run(service._maybe_answer_from_inline_document(follow, "s", "hinglish", memory, 0.0))
    assert response is not None and "utility charges" in response.answer

    service.llm.chat = AsyncMock(return_value=LLMResponse(
        content=_json.dumps({"found_in_document": False, "answer": "", "supporting_quote": ""}), model="t", provider="t",
    ))
    page = _Req(question="Is excerpt ka exact PDF page number, government URL aur notarization status batao.")
    response = asyncio.run(service._maybe_answer_from_inline_document(page, "s", "hinglish", memory, 0.0))
    assert response is not None and "excerpt mein yeh baat nahi hai" in response.answer

    # No pasted excerpt in the session -> untouched normal routing.
    assert asyncio.run(service._maybe_answer_from_inline_document(follow, "s", "hinglish", {}, 0.0)) is None
