"""Regression tests for the draft-quality incident of Sept 2026.

The reported symptom was "draft ki quality bahut gir gayi hai": a user asked
for a police complaint in Hindi and received a one-page document whose entire
Facts section was a third-person English sentence, whose Legal Position was a
single flat "this matter is governed by <Act>", and whose Prayer was two
half-sentences welded together at a full stop.

Reconstructing it from the transcript, five independent defects had to line
up, and each one is pinned here:

1. A transient LLM failure was invisible to the drafting engine, because
   providers report failure by RETURNING an `LLMResponse` with `error` set
   rather than by raising. The engine only looked at `content`, found no
   "## " headings in "Gemini API is currently unavailable", and silently
   produced the deterministic skeleton.
2. The user was never told. The skeleton was introduced with the same
   "Here is your draft" wrapper as a real one.
3. The word target for sparse facts (900) was below the three-page minimum
   the product promises, and was applied without regard to script.
4. The LLM extraction pass summarised, translated, and re-voiced the user's
   narrative before drafting ever started.
5. "Kal" was rejected as an invalid date, so the incident date was dropped.
"""

from datetime import date
from unittest.mock import AsyncMock

import pytest

from app.core.fallback_guidance import actionable_guidance, detect_category
from app.drafting.engine import (
    _WORDS_PER_BODY_PAGE_INDIC,
    _WORDS_PER_BODY_PAGE_LATIN,
    MINIMUM_BODY_PAGES,
    LegalDraftEngine,
    _is_complete_sentence,
    _numbered_pleading_paragraphs,
    _split_sentences,
    _target_word_count,
    estimated_page_count,
)
from app.drafting.field_extraction import is_faithful_narrative
from app.drafting.localized_dates import resolve_relative_date
from app.drafting.templates import get_template
from app.drafting.validation import DraftFieldValidator
from app.llm.base import ChatMessage, LLMResponse
from app.llm.resilient import ResilientLLMProvider
from app.schemas.drafting import DraftPreviewRequest

# The transcript's actual values.
_TRANSCRIPT_FIELDS = {
    "applicant_name": "Rahul Sharma",
    "applicant_address": "24, Shastri Nagar, Ghaziabad, Uttar Pradesh",
    "applicant_mobile": "9876543210",
    "police_station": "Kavinagar Police Station, Ghaziabad",
    "incident_location": "Shastri Nagar, Ghaziabad, Uttar Pradesh",
    "facts": (
        "28 अगस्त 2026 को मुझे एक व्यक्ति का फोन आया जिसने खुद को एक ऑनलाइन शॉपिंग कंपनी का "
        "प्रतिनिधि बताया। उसने कहा कि मेरे नाम पर ₹15,000 का कैशबैक लंबित है। "
        "मैंने QR कोड स्कैन किया, जिसके बाद मेरे बैंक खाते से ₹15,000 डेबिट हो गए।"
    ),
    "expected_relief": "आरोपी के विरुद्ध उचित कानूनी कार्रवाई की जाए तथा मेरे साथ हुई धोखाधड़ी की शिकायत दर्ज की जाए।",
    "place": "Ghaziabad, Uttar Pradesh",
}


# ---------------------------------------------------------------------------
# 1. A provider error must never masquerade as a generated draft.
# ---------------------------------------------------------------------------


def test_provider_error_response_is_detected_and_reported_not_silently_swallowed() -> None:
    """The core defect. A provider returning `error` must mark the draft
    degraded, not quietly hand back a skeleton labelled like a full draft."""
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="Gemini API is currently unavailable. Please try again.",
            model="test",
            provider="test",
            error="Gemini API is currently unavailable. Please try again.",
        )
    )
    request = DraftPreviewRequest(draft_id="police_complaint", language="hindi", fields=_TRANSCRIPT_FIELDS)
    response = pytest.importorskip("asyncio").run(engine.preview(request))

    assert response.generation_mode == "deterministic"
    assert response.generated_by_llm is False
    # The reason travels with it, so the chat layer can explain what happened.
    assert response.generation_error is not None
    assert "unavailable" in response.generation_error.lower()


def test_successful_generation_is_not_marked_degraded() -> None:
    engine = LegalDraftEngine()
    from app.drafting.templates.base import structure_sections_for

    headings = structure_sections_for("Complaint")
    body = " ".join(["शब्द"] * 120)
    content = "\n".join(f"## {heading}\n{body}" for heading in headings)
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content=content, model="test", provider="test"))

    request = DraftPreviewRequest(draft_id="police_complaint", language="hindi", fields=_TRANSCRIPT_FIELDS)
    response = pytest.importorskip("asyncio").run(engine.preview(request))

    assert response.generation_mode == "llm"
    assert response.generation_error is None


class _ScriptedProvider:
    """Minimal `LLMProvider` stand-in returning a queued list of responses."""

    provider_name = "scripted"
    model = "scripted-model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> LLMResponse:
        self.calls += 1
        return self._responses.pop(0) if self._responses else self._responses_exhausted()

    def _responses_exhausted(self) -> LLMResponse:
        raise AssertionError("provider called more times than the test scripted")

    async def stream(self, messages, temperature: float = 0.1):  # pragma: no cover - unused
        yield ""

    async def health(self) -> bool:  # pragma: no cover - unused
        return True


def test_resilient_provider_retries_a_transient_failure_before_giving_up() -> None:
    asyncio = pytest.importorskip("asyncio")
    primary = _ScriptedProvider([
        LLMResponse(content="", model="m", provider="p", error="Service unavailable, try again"),
        LLMResponse(content="recovered", model="m", provider="p"),
    ])
    provider = ResilientLLMProvider(primary, [], max_attempts=3, initial_backoff_seconds=0)

    response = asyncio.run(provider.chat([ChatMessage(role="user", content="hi")]))

    assert response.content == "recovered"
    assert response.error is None
    assert primary.calls == 2


def test_resilient_provider_fails_over_to_the_next_provider() -> None:
    asyncio = pytest.importorskip("asyncio")
    primary = _ScriptedProvider([LLMResponse(content="", model="m", provider="p", error="quota exceeded")] * 3)
    backup = _ScriptedProvider([LLMResponse(content="from backup", model="m2", provider="p2")])
    provider = ResilientLLMProvider(primary, [backup], max_attempts=3, initial_backoff_seconds=0)

    response = asyncio.run(provider.chat([ChatMessage(role="user", content="hi")]))

    assert response.content == "from backup"
    assert backup.calls == 1


def test_resilient_provider_returns_the_real_error_when_everything_fails() -> None:
    """It must never fabricate content to paper over a total outage."""
    asyncio = pytest.importorskip("asyncio")
    primary = _ScriptedProvider([LLMResponse(content="", model="m", provider="p", error="connection refused")] * 3)
    provider = ResilientLLMProvider(primary, [], max_attempts=3, initial_backoff_seconds=0)

    response = asyncio.run(provider.chat([ChatMessage(role="user", content="hi")]))

    assert response.error == "connection refused"


def test_resilient_provider_does_not_retry_a_permanently_failing_error() -> None:
    """A rejected API key fails identically every time; retrying only adds
    latency to an outcome already decided."""
    asyncio = pytest.importorskip("asyncio")
    primary = _ScriptedProvider([LLMResponse(content="", model="m", provider="p", error="API key was rejected")])
    provider = ResilientLLMProvider(primary, [], max_attempts=3, initial_backoff_seconds=0)

    asyncio.run(provider.chat([ChatMessage(role="user", content="hi")]))

    assert primary.calls == 1


# ---------------------------------------------------------------------------
# 2/3. Length, structure and the three-page minimum.
# ---------------------------------------------------------------------------


def test_word_target_floor_is_three_pages_and_script_aware() -> None:
    assert _target_word_count({"facts": "x"}, "english") == _WORDS_PER_BODY_PAGE_LATIN * MINIMUM_BODY_PAGES
    assert _target_word_count({"facts": "x"}, "hindi") == _WORDS_PER_BODY_PAGE_INDIC * MINIMUM_BODY_PAGES
    # Never below the floor, however sparse the input.
    assert _target_word_count({}, "hindi") >= _WORDS_PER_BODY_PAGE_INDIC * MINIMUM_BODY_PAGES


def test_estimated_page_count_distinguishes_scripts() -> None:
    assert estimated_page_count({"a": "complaint " * 400}) < estimated_page_count({"a": "शिकायत " * 400})


def test_short_llm_complaint_gets_fact_neutral_scaffolding_before_shipping() -> None:
    """A valid but repeatedly short provider reply must not bypass the page floor."""
    engine = LegalDraftEngine()
    from app.drafting.templates.base import structure_sections_for

    headings = structure_sections_for("Complaint")
    # Mirrors the observed 475-word consumer complaint closely enough to
    # exercise the real regression without pretending that an almost-empty
    # user account can safely be inflated into three pages.
    short_body = "Supplied fact only. " * 12
    content = "\n".join(f"## {heading}\n{short_body}" for heading in headings)
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content=content, model="test", provider="test"))
    request = DraftPreviewRequest(draft_id="police_complaint", language="english", fields=_TRANSCRIPT_FIELDS)

    response = pytest.importorskip("asyncio").run(engine.preview(request))

    assert response.generation_mode == "llm"
    assert response.estimated_page_count >= MINIMUM_BODY_PAGES
    assert "Nothing in this draft is intended to enlarge that account" in response.sections["Introduction"]


def test_degraded_draft_is_still_a_properly_structured_legal_document() -> None:
    """Even the LLM-unavailable path must produce a real pleading, not a form.

    This is the document the transcript's user actually received, and every
    assertion here is something that output lacked.
    """
    engine = LegalDraftEngine()
    template = get_template("police_complaint")
    assert template is not None
    request = DraftPreviewRequest(draft_id="police_complaint", language="hindi", fields=_TRANSCRIPT_FIELDS)
    sections = engine._deterministic_sections(template, request)

    # Facts are numbered pleading paragraphs, not a label dump.
    facts = sections["Facts of the Case"]
    assert "1. " in facts and "2. " in facts

    # The Introduction is a real opening submission, not a bare salutation.
    assert len(sections["Introduction"].split()) > 15

    # The legal position is HEDGED -- never asserted as settled law.
    legal = sections["Legal Position"]
    assert "प्रथम दृष्टया" in legal
    assert "शासित है" not in legal


def test_every_complaint_has_a_formal_verification_before_signature() -> None:
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(
        return_value=LLMResponse(content="", model="test", provider="test", error="offline")
    )
    request = DraftPreviewRequest(draft_id="police_complaint", language="english", fields=_TRANSCRIPT_FIELDS)

    response = pytest.importorskip("asyncio").run(engine.preview(request))

    headings = list(response.sections)
    assert headings.index("Verification") < headings.index("Signature Block")
    assert "true and correct" in response.sections["Verification"].lower()


def test_prayer_does_not_weld_two_sentences_together() -> None:
    """The transcript's Prayer read '...दर्ज की जाए। करने का अनुरोध किया जाता
    है।' -- the inflecting template wrapped around an already-complete
    sentence. A complete-sentence relief must be INTRODUCED instead."""
    engine = LegalDraftEngine()
    prayer = engine._prayer_block(_TRANSCRIPT_FIELDS, "hindi", non_compliance=None)

    assert "। करने का अनुरोध" not in prayer
    # It is introduced, and the user's own wording survives intact.
    assert "प्रार्थी सादर प्रार्थना करता/करती है" in prayer
    assert _TRANSCRIPT_FIELDS["expected_relief"] in prayer


def test_prayer_still_inflects_a_bare_noun_phrase_relief() -> None:
    """The inflecting template is correct for a short noun phrase, so it must
    not be discarded -- only bypassed when the relief is a full sentence."""
    engine = LegalDraftEngine()
    prayer = engine._prayer_block({"expected_relief": "refund of the amount"}, "english", non_compliance=None)

    assert "You are therefore called upon to refund of the amount." in prayer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("आरोपी के विरुद्ध कार्रवाई की जाए।", True),
        ("recovery of Rs. 15,000", False),
        ("", False),
    ],
)
def test_complete_sentence_detection(text: str, expected: bool) -> None:
    assert _is_complete_sentence(text) is expected


def test_sentence_splitting_respects_legal_abbreviations() -> None:
    """'Rs. 30000' must not become two pleading paragraphs."""
    assert _split_sentences("Goods worth Rs. 30000 were not delivered despite full payment.") == [
        "Goods worth Rs. 30000 were not delivered despite full payment."
    ]
    assert _split_sentences("I paid Rs. 5,000 to M/s. Raj Traders. They never delivered.") == [
        "I paid Rs. 5,000 to M/s. Raj Traders.",
        "They never delivered.",
    ]
    # A date's internal full stops are not sentence ends either.
    assert len(_split_sentences("On 15.08.2026 the amount was debited.")) == 1


def test_numbered_pleading_paragraphs_preserve_every_word() -> None:
    facts = "मुझे फोन आया। मैंने QR स्कैन किया। पैसे कट गए।"
    rendered = _numbered_pleading_paragraphs(facts, "hindi")
    for fragment in ("मुझे फोन आया", "मैंने QR स्कैन किया", "पैसे कट गए"):
        assert fragment in rendered
    assert rendered.count("\n\n") == 2


# ---------------------------------------------------------------------------
# 4. Narrative fidelity: the model must not rewrite the user's account.
# ---------------------------------------------------------------------------


def test_summarised_translated_third_person_extraction_is_rejected() -> None:
    """The exact string from the incident."""
    source = _TRANSCRIPT_FIELDS["facts"]
    corrupted = "A fraud of ₹15,000 occurred via PhonePe yesterday, but the user does not have the UTR number."
    assert is_faithful_narrative(corrupted, source) is False


def test_faithful_extraction_is_accepted() -> None:
    source = _TRANSCRIPT_FIELDS["facts"]
    assert is_faithful_narrative(source, source) is True


def test_translation_out_of_the_users_script_is_rejected() -> None:
    source = "मेरे खाते से ₹15,000 काट लिए गए और मैंने बैंक को सूचित किया।"
    translated = "Rs 15,000 was deducted from my account and I informed the bank about it immediately."
    assert is_faithful_narrative(translated, source) is False


def test_an_english_conversation_is_not_penalised() -> None:
    """The guard must not fire when the user genuinely wrote English."""
    source = "I received a call from someone claiming to be a company representative and 15000 was debited."
    kept = "I received a call from someone claiming to be a company representative and 15000 was debited."
    assert is_faithful_narrative(kept, source) is True


# ---------------------------------------------------------------------------
# 5. Relative dates.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,offset_days",
    [("Kal", -1), ("kal", -1), ("yesterday", -1), ("कल", -1), ("परसों", -2), ("நேற்று", -1), ("आज", 0)],
)
def test_relative_dates_resolve_instead_of_being_rejected(expression: str, offset_days: int) -> None:
    from datetime import timedelta

    reference = date(2026, 9, 1)
    assert resolve_relative_date(expression, reference) == reference + timedelta(days=offset_days)


def test_absolute_dates_are_left_to_the_normal_parsers() -> None:
    assert resolve_relative_date("15 July 2026") is None
    assert resolve_relative_date("not a date at all") is None


def test_validator_resolves_a_relative_date_and_reports_the_assumption() -> None:
    """'Kal' used to be rejected, dropping the incident date entirely."""
    validator = DraftFieldValidator()
    template = get_template("police_complaint")
    assert template is not None
    fields = {"incident_date": "Kal"}
    resolved: list[tuple[str, str, str]] = []

    issues = validator.validate(template, fields, "hindi", resolved)

    assert not [issue for issue in issues if issue[0] == "incident_date"]
    assert resolved and resolved[0][0] == "incident_date"
    # The field was rewritten in place to an absolute date.
    assert fields["incident_date"] != "Kal"
    assert "/" in fields["incident_date"]


# ---------------------------------------------------------------------------
# 6. The "no verified document" fallback must still help, in every language.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,language",
    [
        ("मेरे बैंक खाते से बिना अनुमति ₹40,000 निकल गए हैं। मुझे तुरंत क्या करना चाहिए?", "hindi"),
        ("माझ्या बँक खात्यातून परवानगीशिवाय ₹३०,००० काढले गेले आहेत. मी तातडीने काय करावे?", "marathi"),
        ("Kal PhonePe se ₹15,000 ka fraud hua, ab kya karun?", "hinglish"),
        ("என் வங்கிக் கணக்கிலிருந்து அனுமதி இன்றி பணம் எடுக்கப்பட்டது", "tamil"),
        ("నా ఖాతా నుండి అనుమతి లేకుండా డబ్బు తీసివేయబడింది", "telugu"),
        ("আমার অ্যাকাউন্ট থেকে অনুমতি ছাড়া টাকা কেটে নেওয়া হয়েছে", "bengali"),
        ("મારા ખાતામાંથી પરવાનગી વગર પૈસા ઉપડી ગયા", "gujarati"),
    ],
)
def test_financial_fraud_is_detected_in_every_supported_language(question: str, language: str) -> None:
    """Both transcript questions returned a bare refusal with no guidance,
    while the identical question in English was matched."""
    category = detect_category(question)
    assert category is not None, f"{language} question was not recognised"
    assert category.key == "cyber_financial_fraud"


def test_guidance_is_written_in_the_users_own_language() -> None:
    """Marathi previously fell back to English step text."""
    guidance = actionable_guidance("माझ्या बँक खात्यातून परवानगीशिवाय पैसे काढले गेले", "marathi")
    assert guidance
    assert "1930" in guidance
    assert "पोलीस ठाण्यात" in guidance or "सायबर" in guidance
    # No English step text leaked through.
    assert "Report the transaction to your bank" not in guidance


def test_unrelated_questions_get_no_safety_guidance() -> None:
    """Broadening detection must not make every question an emergency."""
    assert detect_category("What is the procedure to register a company in India?") is None
    assert detect_category("Explain anticipatory bail in simple language") is None
    assert actionable_guidance("Explain anticipatory bail", "english") == ""


# ---------------------------------------------------------------------------
# 7. Honest reporting of partial language coverage.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "language",
    ["hindi", "tamil", "telugu", "kannada", "bengali", "marathi", "gujarati", "punjabi",
     "malayalam", "odia", "urdu", "english", "hinglish"],
)
def test_headline_languages_are_fully_localized(language: str) -> None:
    """The languages this product actually advertises must localize every
    layer -- headings, title and fallback boilerplate -- not just the body."""
    from app.drafting.heading_translations import translated_heading
    from app.drafting.language_support import supported_level

    assert supported_level(language) == "full"
    if language not in ("english", "hinglish"):
        assert translated_heading("Facts of the Case", language) != "Facts of the Case"


@pytest.mark.parametrize("language", ["assamese", "nepali", "bodo", "santali", "manipuri"])
def test_partially_covered_languages_are_flagged_not_silently_mixed(language: str) -> None:
    """These produce a translated body under English headings. That is
    acceptable; doing it without telling the user is not."""
    from app.drafting.language_support import supported_level

    assert supported_level(language) == "body_only"
