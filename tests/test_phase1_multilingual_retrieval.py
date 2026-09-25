"""Phase 1 items 1-3: multilingual retrieval, source-first citations, and the
legal-answer quality gate.

The multilingual tests are written as PARITY tests: the same legal question is
asked in English and in each supported language, and the two must reach the
same place. That is the actual product requirement -- not "Hindi works" but
"Hindi works the same as English" -- and it is the shape that catches a
regression in either direction.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.constants import no_verified_context_message
from app.intent.detector import IntentDetector
from app.language.detector import LanguageDetector
from app.rag.answer_quality import Severity, evaluate
from app.rag.multilingual import has_legal_concept, legal_english_variants, legal_intent_hint
from app.rag.query_rewriter import SmartQueryRewriter
from app.schemas.chat import ChatRequest
from app.schemas.common import (
    LawyerRecommendation,
    RetrievedChunk,
    SourceCitation,
    build_citation_label,
)

_RECOMMENDATION = LawyerRecommendation(category="Criminal Law", confidence=0.6, reason="Test fixture.")

# The same five questions, in every language this phase names. Grouped by
# concept so a parity failure names which concept broke, in which language.
_PARALLEL_QUESTIONS: dict[str, dict[str, str]] = {
    "cheating": {
        "english": "What is the punishment for cheating?",
        "hindi": "धोखाधड़ी के लिए क्या सजा है?",
        "hinglish": "Dhokhadhadi ki saza kya hai?",
        "marathi": "फसवणुकीसाठी काय शिक्षा आहे?",
        "gujarati": "છેતરપિંડી માટે શું સજા છે?",
        "tamil": "ஏமாற்றுதலுக்கு என்ன தண்டனை?",
        "telugu": "మోసానికి శిక్ష ఏమిటి?",
        "bengali": "প্রতারণার শাস্তি কী?",
        "urdu": "دھوکہ دہی کی سزا کیا ہے؟",
    },
    "bail": {
        "english": "How do I apply for bail?",
        "hindi": "जमानत के लिए आवेदन कैसे करें?",
        "hinglish": "Jamanat ke liye kaise apply karein?",
        "marathi": "जामीन कसा मिळवावा?",
        "gujarati": "જામીન કેવી રીતે મળે?",
        "tamil": "பிணை எப்படி பெறுவது?",
        "telugu": "బెయిల్ ఎలా పొందాలి?",
        "bengali": "জামিন কীভাবে পাব?",
        "urdu": "ضمانت کیسے ملتی ہے؟",
    },
    "fir": {
        "english": "How do I file a police complaint?",
        "hindi": "पुलिस शिकायत कैसे दर्ज करें?",
        "hinglish": "Police complaint kaise darj karein?",
        "marathi": "पोलीस तक्रार कशी नोंदवावी?",
        "gujarati": "પોલીસ ફરિયાદ કેવી રીતે નોંધાવવી?",
        "tamil": "காவல் புகார் எப்படி பதிவு செய்வது?",
        "telugu": "పోలీసు ఫిర్యాదు ఎలా చేయాలి?",
        "bengali": "পুলিশ অভিযোগ কীভাবে করব?",
        "urdu": "پولیس شکایت کیسے درج کریں؟",
    },
    "threat": {
        "english": "Someone is threatening me, what can I do?",
        "hindi": "कोई मुझे धमकी दे रहा है, मैं क्या करूं?",
        "hinglish": "Koi mujhe dhamki de raha hai, kya karun?",
        "marathi": "कोणी मला धमकी देत आहे, काय करावे?",
        "gujarati": "કોઈ મને ધમકી આપે છે, શું કરવું?",
        "tamil": "யாரோ என்னை மிரட்டுகிறார், என்ன செய்வது?",
        "telugu": "ఎవరో నన్ను బెదిరిస్తున్నారు, ఏం చేయాలి?",
        "bengali": "কেউ আমাকে হুমকি দিচ্ছে, কী করব?",
        "urdu": "کوئی مجھے دھمکی دے رہا ہے، کیا کروں؟",
    },
    "consumer": {
        "english": "The seller refused a refund for a defective product.",
        "hindi": "विक्रेता ने खराब उत्पाद का रिफंड देने से मना कर दिया।",
        "hinglish": "Seller ne defective product ka refund nahi diya.",
        "marathi": "विक्रेत्याने सदोष वस्तूचा परतावा नाकारला. ग्राहक तक्रार करायची आहे.",
        "gujarati": "વિક્રેતાએ ખામીયુક્ત વસ્તુનું રિફંડ આપવાની ના પાડી. ગ્રાહક ફરિયાદ કરવી છે.",
        "tamil": "விற்பனையாளர் பணத்தைத் திருப்பித் தர மறுத்தார். நுகர்வோர் புகார் அளிக்க வேண்டும்.",
        "telugu": "విక్రేత రీఫండ్ ఇవ్వడానికి నిరాకరించారు. వినియోగదారు ఫిర్యాదు చేయాలి.",
        "bengali": "বিক্রেতা ত্রুটিপূর্ণ পণ্যের টাকা ফেরত দিতে অস্বীকার করেছে। ভোক্তা অভিযোগ করব।",
        "urdu": "بیچنے والے نے رقم واپس کرنے سے انکار کیا۔ صارف شکایت درج کرنی ہے۔",
    },
}
_NON_ENGLISH = ("hindi", "hinglish", "marathi", "gujarati", "tamil", "telugu", "bengali", "urdu")


def _service_with_mocks(memory: dict | None = None):
    from app.services.chat_service import ChatService

    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])
    base = {"messages": [], "summary": "", "current_intent": None, "legal_category": None}
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
    service.recommendations.recommend = AsyncMock(return_value=_RECOMMENDATION)
    service._generate_related_questions = AsyncMock(return_value=[])
    return service


# ---------------------------------------------------------------------------
# Item 1 -- the same question in any supported language reaches the same
# English legal vocabulary, so it can retrieve the same verified material
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("concept", sorted(_PARALLEL_QUESTIONS))
@pytest.mark.parametrize("language", _NON_ENGLISH)
def test_every_language_reaches_english_legal_search_vocabulary(concept: str, language: str) -> None:
    question = _PARALLEL_QUESTIONS[concept][language]
    variants = legal_english_variants(question)
    assert variants, f"{language} {concept!r} produced no English search variant: {question}"
    assert all(variant.isascii() for variant in variants)


@pytest.mark.parametrize("concept", sorted(_PARALLEL_QUESTIONS))
@pytest.mark.parametrize("language", _NON_ENGLISH)
def test_expanded_query_carries_the_same_statutory_terms_as_english(concept: str, language: str) -> None:
    # The point of the bridge: BM25 (which has no multilingual capability at
    # all) and the downstream lexical relevance gate must see the SAME English
    # statutory vocabulary whichever language the question arrived in.
    rewriter = SmartQueryRewriter()
    marker = {
        "cheating": "cheating",
        "bail": "bail",
        "fir": "first information report",
        "threat": "criminal intimidation",
        "consumer": "consumer",
    }[concept]
    expanded = " ".join(rewriter.expand_queries(_PARALLEL_QUESTIONS[concept][language])).lower()
    assert marker in expanded, f"{language} {concept!r} expansion missing {marker!r}"


@pytest.mark.parametrize("concept", sorted(_PARALLEL_QUESTIONS))
@pytest.mark.parametrize("language", _NON_ENGLISH)
def test_intent_matches_the_english_equivalent(concept: str, language: str) -> None:
    detector = IntentDetector()
    english = asyncio.run(detector.detect(_PARALLEL_QUESTIONS[concept]["english"]))
    other = asyncio.run(detector.detect(_PARALLEL_QUESTIONS[concept][language]))
    assert other.legal_category == english.legal_category, (
        f"{language} {concept!r}: {other.intent}/{other.legal_category} != "
        f"{english.intent}/{english.legal_category}"
    )


@pytest.mark.parametrize("concept", sorted(_PARALLEL_QUESTIONS))
@pytest.mark.parametrize("language", _NON_ENGLISH)
def test_language_is_detected_and_preserved(concept: str, language: str) -> None:
    # "Answer in the user's original language" starts with detecting it. The
    # romanized languages are detected as hinglish by design (the product
    # replies in romanized script for those), so only native-script languages
    # are asserted exactly.
    if language == "hinglish":
        return
    detected = LanguageDetector().detect(_PARALLEL_QUESTIONS[concept][language])
    assert detected == language, f"{concept!r} in {language} detected as {detected}"


def test_ordinary_text_produces_no_spurious_legal_variants() -> None:
    # The bridge must not fire on names, addresses or field values, or every
    # drafting turn would drag unrelated statutory vocabulary into retrieval.
    for text in ["Rahul Sharma", "24, Shanti Vihar, Gomti Nagar, Lucknow", "9876543210", "HDFC260828458721"]:
        assert legal_english_variants(text) == [], text
        assert not has_legal_concept(text)


# ---------------------------------------------------------------------------
# Regression tests for qa-40q-multilingual-20260921 BUG-03: a Gurmukhi
# (Punjabi) "police complaint" question got the bare "no verified document"
# refusal while the identical topic succeeded in Hinglish, English and
# Devanagari Hindi. Root cause was two-fold: (1) the Punjabi FIR/police-
# complaint concept term only covered the bare phrase "ਪੁਲਿਸ ਸ਼ਿਕਾਇਤ", not the
# natural phrasing with the "ਨੂੰ" postposition inserted between the two
# words, so the concept bridge never fired for a realistic sentence; (2)
# `LegalRetriever`'s `FIR_CONCEPT_RE`/`RAPE_CONCEPT_RE`/`THEFT_CONCEPT_RE`
# checks (retriever.py) only matched literal English/Hinglish trigger words
# against the RAW query, never against the rewritten query that carries the
# concept bridge's English expansion -- so even a correctly-bridged native-
# script query never triggered the section-floor guarantee.
# ---------------------------------------------------------------------------


def test_punjabi_police_complaint_with_postposition_reaches_the_fir_concept() -> None:
    # The exact phrasing from the QA transcript: "ਪੁਲਿਸ ਨੂੰ ਸ਼ਿਕਾਇਤ ਦਿੰਦਿਆਂ" (while
    # giving a complaint TO the police) -- not the bare "ਪੁਲਿਸ ਸ਼ਿਕਾਇਤ" substring.
    question = "ਮੇਰਾ ਮੋਬਾਈਲ ਚੋਰੀ ਹੋ ਗਿਆ ਹੈ। ਪੁਲਿਸ ਨੂੰ ਸ਼ਿਕਾਇਤ ਦਿੰਦਿਆਂ ਕਿਹੜੀ ਜਾਣਕਾਰੀ ਦੇਣੀ ਚਾਹੀਦੀ ਹੈ?"
    variants = legal_english_variants(question)
    assert variants, f"Punjabi FIR question produced no English search variant: {question}"
    assert any("police complaint" in variant or "FIR" in variant for variant in variants)


def test_fir_concept_regex_matches_the_rewritten_query_not_only_the_raw_one() -> None:
    # `retriever.py` must check its concept regexes against the REWRITTEN
    # query (which carries the English concept-bridge expansion), not only
    # the raw native-script query -- otherwise the section-floor guarantee
    # this regex exists for never fires for a non-Latin-script question.
    from app.rag.query_rewriter import FIR_CONCEPT_RE, SmartQueryRewriter

    question = "ਪੁਲਿਸ ਨੂੰ ਸ਼ਿਕਾਇਤ ਦਿੰਦਿਆਂ ਕਿਹੜੀ ਜਾਣਕਾਰੀ ਦੇਣੀ ਚਾਹੀਦੀ ਹੈ?"
    assert not FIR_CONCEPT_RE.search(question), "raw Gurmukhi text should not itself match the English-only regex"
    rewriter = SmartQueryRewriter()
    rewritten = rewriter.combine(rewriter.expand_queries(question))
    assert FIR_CONCEPT_RE.search(rewritten), "the rewritten query must carry the FIR concept-bridge expansion"


def test_concepts_without_a_distinctive_intent_do_not_guess_one() -> None:
    # "punishment"/"section"/"rights" say what KIND of question it is, not
    # what it is about -- guessing an intent from them would be worse than the
    # honest General Legal Query default.
    assert legal_intent_hint("सजा क्या है") is None
    assert legal_intent_hint("धारा क्या कहती है") is None


def test_a_non_english_question_is_not_rejected_for_lacking_english_tokens() -> None:
    # The whole point of item 1: "never return 'No verified document' merely
    # because the query language differs from the KB language". This chunk
    # shares no token at all with the Hinglish question, and would fail the
    # English lexical-overlap gate, but the cross-lingual allowance accepts it.
    from app.services.chat_service import ChatService

    service = ChatService()
    chunk = RetrievedChunk(
        chunk_id="c",
        text="Whoever cheats and thereby dishonestly induces the person deceived to deliver any property...",
        score=0.30,
        metadata={"source_document": "bns.pdf", "act_name": "Bharatiya Nyaya Sanhita", "section_number": "318"},
    )
    assert service._is_relevant_chunk("dhokhadhadi ki saza kya hai", "", chunk, language="hinglish")
    # ...and the same allowance is NOT extended to English, where literal
    # overlap is a meaningful signal and dropping it would weaken the gate.
    assert not service._is_relevant_chunk("what is the weather today", "", chunk, language="english")


# ---------------------------------------------------------------------------
# Item 2 -- source-first citations
# ---------------------------------------------------------------------------


def test_citation_label_names_act_section_document_and_url() -> None:
    citation = SourceCitation(
        act_name="Bharatiya Nyaya Sanhita, 2023",
        section="318",
        source_document="bns_2023.pdf",
        url="https://www.indiacode.nic.in/example",
    )
    assert citation.label == (
        "Bharatiya Nyaya Sanhita, 2023 — Section 318 (bns_2023.pdf) · https://www.indiacode.nic.in/example"
    )
    assert citation.is_identifiable


def test_citation_label_never_invents_an_act_it_does_not_have() -> None:
    citation = SourceCitation(source_document="internal_faq.pdf")
    assert citation.label == "internal_faq.pdf"
    assert not citation.is_identifiable


def test_citation_label_is_never_a_bare_index() -> None:
    for label in (
        build_citation_label(act_name="Consumer Protection Act, 2019", section="35"),
        build_citation_label(article="21", source_document="constitution.pdf"),
        build_citation_label(source_document="x.pdf"),
        build_citation_label(),
    ):
        assert "Source" not in label
        assert label.strip()


def test_an_answer_backed_only_by_unidentifiable_sources_is_not_returned() -> None:
    # Item 2: "if retrieval has no reliable source, do not make a legal claim."
    verdict = evaluate(
        "Under the applicable provision, an FIR must be registered for any cognizable offence reported.",
        question="What is an FIR",
        language="english",
        sources=[SourceCitation(source_document="scratch_notes.pdf")],
        ranked_chunks=[RetrievedChunk(chunk_id="c", text="FIR cognizable offence registered", score=0.7)],
    )
    assert verdict.severity is Severity.REJECT
    assert verdict.failed_check == "grounded"


def test_answer_cannot_cite_a_section_absent_from_its_sources() -> None:
    verdict = evaluate(
        "Bharatiya Nyaya Sanhita Section 318 applies to the defective-phone sale.",
        question="The seller will not refund my defective phone.",
        language="english",
        sources=[SourceCitation(
            act_name="Bharatiya Nyaya Sanhita", section="37", source_document="bns.pdf"
        )],
        ranked_chunks=[
            RetrievedChunk(
                chunk_id="c",
                text="Section 37 concerns private defence.",
                score=0.9,
                metadata={"act_name": "Bharatiya Nyaya Sanhita", "section_number": "37"},
            )
        ],
    )
    assert verdict.severity is Severity.REJECT
    assert verdict.failed_check == "section_grounding"


# ---------------------------------------------------------------------------
# Item 3 -- the legal-answer quality gate
# ---------------------------------------------------------------------------


_GOOD_SOURCES = [
    SourceCitation(act_name="Bharatiya Nagarik Suraksha Sanhita, 2023", section="173", source_document="bnss.pdf")
]
_GOOD_CHUNKS = [
    RetrievedChunk(
        chunk_id="c1",
        text=(
            "Information in cognizable cases. Every information relating to the commission of a cognizable "
            "offence shall be registered as a first information report by the officer in charge of a police "
            "station, and a copy shall be given free of cost to the informant."
        ),
        score=0.72,
        metadata={"act_name": "BNSS", "section_number": "173", "source_document": "bnss.pdf"},
    )
]
_GOOD_ANSWER = (
    "Under Section 173 of the BNSS, information relating to a cognizable offence must be registered as a "
    "first information report by the officer in charge of the police station, and the informant is entitled "
    "to a free copy."
)


def test_a_good_grounded_answer_passes_every_check() -> None:
    verdict = evaluate(
        _GOOD_ANSWER, question="What is an FIR", language="english",
        sources=_GOOD_SOURCES, ranked_chunks=_GOOD_CHUNKS,
    )
    assert verdict.ok
    assert "grounded" in verdict.checks_run
    assert "language_match" in verdict.checks_run


@pytest.mark.parametrize(
    ("label", "answer"),
    [
        ("gemini outage", "The Gemini API is unreachable at the moment, please try again shortly."),
        ("ollama down", "Local LLM is unavailable. Please start Ollama and retry the request now."),
        ("http status", "The upstream call failed with status_code: 502 while fetching the completion."),
        ("traceback", "Traceback (most recent call last): httpx.ConnectError: connection refused"),
        ("credentials", "Set the api_key environment variable before calling this endpoint again please."),
        ("internal host", "Could not reach the model server at localhost:11434, please retry the request."),
    ],
)
def test_provider_and_network_errors_never_reach_the_user(label: str, answer: str) -> None:
    verdict = evaluate(
        answer, question="What is an FIR", language="english",
        sources=_GOOD_SOURCES, ranked_chunks=_GOOD_CHUNKS,
    )
    assert verdict.severity is Severity.REJECT, label
    assert verdict.failed_check == "no_provider_error", label
    # The reason is logged and returned to the client, so it must not echo the
    # matched text -- that is how a host name or key fragment would escape.
    assert "localhost" not in (verdict.reason or "")
    assert "api_key" not in (verdict.reason or "")


@pytest.mark.parametrize(
    "answer",
    [
        "According to Source 2, an FIR must be registered whenever a cognizable offence is reported to police.",
        "Per {context}, an FIR must be registered whenever a cognizable offence is reported to the police.",
        "As an AI language model I can tell you an FIR is registered for any cognizable offence reported.",
        "<context> An FIR must be registered whenever a cognizable offence is reported to the police.",
    ],
)
def test_internal_prompt_scaffolding_never_reaches_the_user(answer: str) -> None:
    verdict = evaluate(
        answer, question="What is an FIR", language="english",
        sources=_GOOD_SOURCES, ranked_chunks=_GOOD_CHUNKS,
    )
    assert verdict.severity is Severity.REJECT
    assert verdict.failed_check == "no_internal_artefacts"


def test_an_answer_about_a_different_topic_is_rejected() -> None:
    verdict = evaluate(
        "Mortgage of immovable property requires a registered instrument executed before two attesting "
        "witnesses under the conveyancing statute.",
        question="What is an FIR", language="english",
        sources=_GOOD_SOURCES, ranked_chunks=_GOOD_CHUNKS,
    )
    assert verdict.severity is Severity.REJECT
    assert verdict.failed_check == "on_topic"


def test_an_empty_or_stub_answer_is_rejected() -> None:
    for answer in ("", "   ", "OK.", "Yes."):
        verdict = evaluate(
            answer, question="What is an FIR", language="english",
            sources=_GOOD_SOURCES, ranked_chunks=_GOOD_CHUNKS,
        )
        assert verdict.severity is Severity.REJECT
        assert verdict.failed_check == "non_empty"


def test_a_terse_but_real_answer_survives() -> None:
    verdict = evaluate(
        "An FIR records a cognizable offence at the police station.",
        question="What is an FIR", language="english",
        sources=_GOOD_SOURCES, ranked_chunks=_GOOD_CHUNKS,
    )
    assert verdict.ok


@pytest.mark.parametrize(
    ("asked", "answer"),
    [
        ("hindi", _GOOD_ANSWER),
        ("english", "बीएनएसएस की धारा 173 के अनुसार संज्ञेय अपराध की सूचना पर प्राथमिकी दर्ज करना अनिवार्य है।"),
    ],
)
def test_a_wrong_language_answer_is_flagged_for_retry_not_discarded(asked: str, answer: str) -> None:
    # Item 3 lists language as a validation, but a language mismatch is a
    # PRESENTATION failure on what may be legally correct content -- so it is
    # a retry, not the outright rejection an ungrounded answer gets.
    verdict = evaluate(
        answer, question="q", language=asked, sources=_GOOD_SOURCES, ranked_chunks=_GOOD_CHUNKS,
    )
    assert verdict.severity is Severity.RETRY_LANGUAGE
    assert verdict.failed_check == "language_match"


def test_a_hindi_answer_full_of_english_statute_names_is_not_flagged() -> None:
    # A correct Hindi legal answer is unavoidably full of Latin script:
    # "Bharatiya Nagarik Suraksha Sanhita", "Section 173", "FIR". The
    # threshold has to tolerate that or it rejects every good answer.
    verdict = evaluate(
        "BNSS (Bharatiya Nagarik Suraksha Sanhita, 2023) की Section 173 के अनुसार, cognizable offence की "
        "सूचना मिलने पर police station के अधिकारी को FIR दर्ज करनी होती है।",
        question="FIR क्या है", language="hindi", sources=_GOOD_SOURCES, ranked_chunks=_GOOD_CHUNKS,
    )
    assert verdict.ok


def test_a_hinglish_answer_is_not_flagged_as_wrong_language() -> None:
    verdict = evaluate(
        "BNSS ki Section 173 ke under, cognizable offence ki information milne par police station ke officer "
        "ko FIR register karni hoti hai, aur informant ko free copy milti hai.",
        question="FIR kya hoti hai", language="hinglish", sources=_GOOD_SOURCES, ranked_chunks=_GOOD_CHUNKS,
    )
    assert verdict.ok


# A second, unrelated section genuinely retrieved alongside Section 173 --
# real rape/arrest-procedure text, sharing no vocabulary with an FIR-refusal
# claim, for `_section_attribution_mismatch`'s own tests below.
_UNRELATED_SECTION_64_CHUNK = RetrievedChunk(
    chunk_id="c2",
    text=(
        "Discharge of person apprehended. Power, on escape, to pursue and retake. Arrest to be made "
        "strictly according to the provisions of this Sanhita or any other law for the time being in "
        "force providing for arrest."
    ),
    score=0.5,
    metadata={"act_name": "BNSS", "section_number": "64", "source_document": "bnss.pdf"},
)


def test_claim_misattributed_to_a_genuinely_retrieved_but_unrelated_section_is_flagged_for_retry() -> None:
    """Live repro: an otherwise-good FIR-procedure answer attached the right
    to escalate to the Superintendent of Police (when an officer refuses to
    register an FIR) to "Section 64" -- a section genuinely retrieved
    alongside 173, just about arrest/escape procedure, sharing nothing with
    the FIR-refusal claim. `section_grounding` alone cannot catch this (64
    genuinely appears in `ranked_chunks`), so this needs the narrower,
    per-claim check.
    """
    answer = (
        "Agar police FIR likhne se mana kar de, to aap Superintendent of Police ko likhit complaint bhej "
        "sakte hain. Ye process Bharatiya Nagarik Suraksha Sanhita ki Section 64 mein di gayi hai, jo "
        "police dwara FIR darj karne se mana karne par aggrieved person ko ye adhikar deti hai."
    )
    verdict = evaluate(
        answer, question="agar police FIR na likhe to kya karu", language="hinglish",
        sources=_GOOD_SOURCES, ranked_chunks=[*_GOOD_CHUNKS, _UNRELATED_SECTION_64_CHUNK],
    )
    assert verdict.severity is Severity.RETRY_SECTION_ATTRIBUTION
    assert verdict.failed_check == "section_attribution"
    assert "64" in (verdict.reason or "")


def test_claim_correctly_attributed_to_its_own_section_is_not_flagged() -> None:
    """The same two retrieved sections as above, but the claim about
    Section 64 genuinely matches Section 64's own retrieved text (arrest
    procedure) -- must not be flagged just because a second, unrelated
    section is also present in context.
    """
    answer = (
        "FIR darj hone ke baad, cognizable offence ki information police station ke officer ko di jaati hai. "
        "Arrest ki procedure Sanhita ke provisions ke anusar strictly follow ki jaati hai, jaisa ki Section "
        "64 mein bataya gaya hai, jisme escape hone par pursue aur retake karne ka adhikar bhi shamil hai."
    )
    verdict = evaluate(
        answer, question="arrest ki procedure kya hai", language="hinglish",
        sources=_GOOD_SOURCES, ranked_chunks=[*_GOOD_CHUNKS, _UNRELATED_SECTION_64_CHUNK],
    )
    assert verdict.failed_check != "section_attribution"


def test_the_topical_check_is_skipped_for_non_latin_answers() -> None:
    # A Tamil answer over English source text shares no vocabulary by
    # construction. Treating that as off-topic would reject every non-English
    # answer this product exists to produce.
    verdict = evaluate(
        "காவல் நிலையத்தில் அறியக்கூடிய குற்றம் குறித்த தகவல் பெறப்பட்டால் முதல் தகவல் அறிக்கை "
        "பதிவு செய்யப்பட வேண்டும் என்பது சட்டப்படி கட்டாயமாகும்.",
        question="FIR என்றால் என்ன?", language="tamil",
        sources=_GOOD_SOURCES, ranked_chunks=_GOOD_CHUNKS,
    )
    assert verdict.ok


def test_a_rejected_answer_is_replaced_by_the_safe_fallback_end_to_end() -> None:
    from app.llm.base import LLMResponse

    service = _service_with_mocks()
    chunk = _GOOD_CHUNKS[0]
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="The Gemini API is unreachable right now, please try again in a few minutes.",
            model="test", provider="test",
        )
    )
    response = asyncio.run(service.answer(ChatRequest(question="what is fir")))
    assert response.answer.startswith(no_verified_context_message("english"))
    assert response.confidence == 0.0
    assert "Gemini" not in response.answer


def test_a_language_mismatch_triggers_one_retry_and_keeps_the_retried_answer() -> None:
    from app.llm.base import LLMResponse

    service = _service_with_mocks()
    chunk = _GOOD_CHUNKS[0]
    service.retriever.retrieve = AsyncMock(return_value=("fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    hindi_answer = (
        "BNSS की Section 173 के अनुसार, cognizable offence की सूचना मिलने पर police station के अधिकारी "
        "को FIR दर्ज करनी होती है और informant को निःशुल्क प्रति दी जाती है।"
    )
    service.llm.chat = AsyncMock(side_effect=[
        LLMResponse(content=_GOOD_ANSWER, model="t", provider="t"),   # wrong language
        LLMResponse(content=hindi_answer, model="t", provider="t"),   # retry lands
    ])
    response = asyncio.run(service.answer(ChatRequest(question="FIR क्या है?", language="hindi")))
    assert "धारा" in response.answer or "अनुसार" in response.answer
    assert service.llm.chat.await_count == 2


def test_a_language_retry_that_also_misses_keeps_the_answer_but_caps_confidence() -> None:
    from app.llm.base import LLMResponse
    from app.services.chat_service import _LANGUAGE_MISMATCH_CONFIDENCE_CAP

    service = _service_with_mocks()
    chunk = _GOOD_CHUNKS[0]
    service.retriever.retrieve = AsyncMock(return_value=("fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(return_value=LLMResponse(content=_GOOD_ANSWER, model="t", provider="t"))
    response = asyncio.run(service.answer(ChatRequest(question="FIR क्या है?", language="hindi")))
    # The legally-correct content is kept rather than thrown away...
    assert "Section 173" in response.answer
    # ...but it is not presented as a high-confidence answer, and the cap is
    # below the cache threshold so it is never stored in this state.
    assert response.confidence <= _LANGUAGE_MISMATCH_CONFIDENCE_CAP
    assert service.llm.chat.await_count == 2
