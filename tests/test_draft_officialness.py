"""Regressions from two real generated drafts shared side by side.

The English **Mobile Theft Complaint** came out as a three-page document with
a proper opening submission, a hedged four-paragraph Legal Position, an
enumerated Prayer with six procedural requests and a Verification clause.

The Hindi **धन वसूली नोटिस** (Recovery Notice), from the same engine, came out
as a two-page skeleton in which:

* the subject line read "₹₹85,000 की धनवापसी/वसूली हेतु मांग" -- two rupee
  signs, on the first line of the document;
* the entire प्रार्थना (Prayer) section -- the operative paragraph, the one
  that says what is being demanded -- read "उपलब्ध नहीं कराया गया।"
  ("not provided");
* the sender's own name and address were headed "परिवादी का विवरण"
  ("Complainant's details"), which is the wrong legal term for the author of
  a demand notice sent to a private party;
* the document referred to itself as "यह आवेदन" ("this application") three
  separate times, and described its facts as subject to "verification and
  investigation by the competent authority" -- complaint vocabulary in a
  notice served on an individual;
* Legal Position was one sentence and Consequences was one line.

The cause was a single condition in the deterministic builder::

    if template.category == "Complaint" and language in {"english", "hinglish"}:

Both halves excluded the Hindi notice. See `app/drafting/officialese.py`.

The second half of the report -- "agar user jis language me bole, usi
language me generate hona chahiye" -- is covered by the language tests at the
bottom.
"""

import pytest

from app.drafting.engine import LegalDraftEngine
from app.drafting.officialese import (
    SCAFFOLDED_CATEGORIES,
    apply_official_scaffolding,
    supports_scaffolding,
)
from app.drafting.templates import get_template
from app.drafting.validation import DraftFieldValidator, normalize_amount
from app.language.detector import extract_requested_language
from app.schemas.drafting import DraftGenerateRequest

RECOVERY_FIELDS = {
    "applicant_name": "Rahul Sharma",
    "applicant_address": "24, Shastri Nagar, Ghaziabad, Uttar Pradesh - 201002",
    "applicant_mobile": "9876543210",
    "respondent_name": "Amit Verma",
    "respondent_address": "18, Nehru Colony, Indirapuram, Ghaziabad, Uttar Pradesh - 201014",
    "principal_amount": "₹85,000",
    "facts": (
        "Maine dinank 15 January 2026 ko prativadi ko 85,000 udhaar diye the. "
        "Prativadi ne ukt rashi 15 April 2026 tak wapas karne ka aashwasan diya tha. "
        "Nirdharit samay beet jane ke baad bhi prativadi ne rashi wapas nahi ki."
    ),
    "place": "Ghaziabad, Uttar Pradesh",
}


def _deterministic_notice(language: str, fields: dict[str, str] | None = None) -> dict[str, str]:
    """The notice as the NON-LLM path builds it -- the path a provider outage
    actually lands the user on, and the one that produced the reported PDF."""
    template = get_template("recovery_notice")
    assert template is not None
    values = dict(fields if fields is not None else RECOVERY_FIELDS)
    DraftFieldValidator().validate(template, values, language, [])
    engine = LegalDraftEngine()
    request = DraftGenerateRequest(
        draft_id="recovery_notice", language=language, fields=values, session_id="test"
    )
    return engine._deterministic_sections(template, request)


# --- The currency bug -------------------------------------------------------

@pytest.mark.parametrize(
    "typed,expected",
    [
        ("₹85,000", "85,000"),
        ("Rs. 85,000", "85,000"),
        ("Rs 85000", "85000"),
        ("INR 85,000", "85,000"),
        ("रु. 85,000", "85,000"),
        ("₨85,000", "85,000"),
        # Already bare -- untouched.
        ("85,000", "85,000"),
        # Never strips the digits themselves.
        ("85", "85"),
    ],
)
def test_amount_is_stored_without_its_currency_mark(typed: str, expected: str) -> None:
    assert normalize_amount(typed) == expected


def test_subject_line_renders_exactly_one_rupee_sign() -> None:
    # The reported first line was "₹₹85,000 की धनवापसी/वसूली हेतु मांग": the
    # localized subject template supplies its own ₹, and the stored value
    # carried a second one because the user typed it after a field label.
    subject = _deterministic_notice("hindi")["Subject"]
    assert "₹₹" not in subject
    assert subject.count("₹") == 1
    assert "85,000" in subject


# --- The placeholder in the operative paragraph -----------------------------

@pytest.mark.parametrize("language", ["hindi", "english"])
def test_prayer_is_never_a_not_provided_placeholder(language: str) -> None:
    fields = {key: value for key, value in RECOVERY_FIELDS.items()}
    fields.pop("expected_relief", None)  # exactly the reported case
    prayer = _deterministic_notice(language, fields)["Prayer"]
    assert prayer.strip()
    assert "उपलब्ध नहीं कराया गया" not in prayer
    assert "Not provided" not in prayer


def test_prayer_with_no_relief_states_the_actual_money_demand() -> None:
    # A notice that names a sum is, by construction, demanding that sum --
    # so say so, rather than leaving the Prayer blank or placeholdered.
    prayer = _deterministic_notice("hindi")["Prayer"]
    assert "85,000" in prayer
    assert "₹₹" not in prayer


def test_prayer_without_an_amount_still_reads_as_a_prayer() -> None:
    fields = {key: value for key, value in RECOVERY_FIELDS.items() if key != "principal_amount"}
    prayer = _deterministic_notice("english", fields)["Prayer"]
    assert "Not provided" not in prayer
    assert "relief" in prayer.lower()


# --- The wrong legal term for the document's own author ---------------------

def test_notice_heads_the_author_block_as_sender_not_complainant() -> None:
    sections = _deterministic_notice("hindi")
    assert "Sender Details" in sections
    assert "Complainant Details" not in sections
    assert "Rahul Sharma" in sections["Sender Details"]


def test_complaint_still_heads_the_author_block_as_complainant() -> None:
    # The rename is scoped to Notice; a complaint's author really is a
    # complainant and must not be renamed with it.
    template = get_template("mobile_theft_complaint")
    engine = LegalDraftEngine()
    request = DraftGenerateRequest(
        draft_id="mobile_theft_complaint", language="english",
        fields={"applicant_name": "Neeraj Singh", "facts": "Phone stolen.", "place": "Ghaziabad"},
        session_id="test",
    )
    sections = engine._deterministic_sections(template, request)
    assert "Complainant Details" in sections
    assert "Sender Details" not in sections


# --- Complaint vocabulary inside a notice -----------------------------------

def test_notice_never_calls_itself_an_application_in_hindi() -> None:
    sections = _deterministic_notice("hindi")
    body = "\n".join(sections.values())
    # "यह आवेदन"/"इस आवेदन" -- the document describing itself as an
    # application. Appeared three times in the reported draft.
    assert "यह आवेदन" not in body
    assert "इस आवेदन" not in body


def test_notice_does_not_claim_investigation_by_a_competent_authority() -> None:
    # A demand notice is served on a private individual. Nobody is
    # investigating it, and saying so is a plain factual error in the draft.
    body = "\n".join(_deterministic_notice("hindi").values())
    assert "अन्वेषण के अधीन" not in body


def test_notice_consequences_refer_to_the_sender_not_an_applicant() -> None:
    body = _deterministic_notice("hindi")["Consequences"]
    assert "प्रेषक" in body


# --- The scaffolding itself -------------------------------------------------

def test_hindi_notice_is_no_longer_a_thin_skeleton() -> None:
    sections = _deterministic_notice("hindi")
    words = len("\n".join(sections.values()).split())
    # The reported draft was ~300 words across two sparse pages. This is a
    # floor, not a target: the point is that a non-English notice is no
    # longer structurally thinner than the English complaint beside it.
    assert words > 450, f"only {words} words"
    for heading in ("Introduction", "Facts of the Case", "Legal Position", "Consequences", "Prayer"):
        assert len(sections[heading].split()) >= 20, f"{heading} is still a stub"


def test_scaffolding_covers_notices_and_applications_not_only_complaints() -> None:
    # The original gate was `category == "Complaint"`, which is why the
    # recovery notice received nothing at all.
    assert {"Complaint", "Notice", "Application"} <= SCAFFOLDED_CATEGORIES


@pytest.mark.parametrize("language", ["english", "hinglish", "hindi"])
def test_scaffolded_languages_get_the_formal_paragraphs(language: str) -> None:
    assert supports_scaffolding(language, "Notice")


@pytest.mark.parametrize("language", ["tamil", "bengali", "odia"])
def test_unauthored_languages_are_left_alone_rather_than_mixed_with_english(language: str) -> None:
    # Deliberate: a Tamil document padded with English paragraphs is worse
    # than a concise Tamil one. Adding a language here is purely additive.
    assert not supports_scaffolding(language, "Notice")
    sections = {"Introduction": "வணக்கம்", "Prayer": "வேண்டுகோள்"}
    before = dict(sections)
    assert apply_official_scaffolding(sections, category="Notice", language=language) == before


def test_scaffolding_never_invents_a_facts_section_out_of_nothing() -> None:
    # The Facts scaffolding reads as a continuation ("The foregoing numbered
    # statements ..."). With no facts to continue, it must not open the
    # section at all.
    sections: dict[str, str] = {}
    apply_official_scaffolding(sections, category="Complaint", language="english")
    assert "Facts of the Case" not in sections


def test_affidavits_are_deliberately_not_scaffolded() -> None:
    # An affidavit's force comes from its sworn verification clause; added
    # procedural narration weakens rather than helps it.
    assert not supports_scaffolding("english", "Affidavit")


# --- "usi language me generate hona chahiye" --------------------------------

@pytest.mark.parametrize(
    "message,expected",
    [
        ("Hindi me police complaint banao", "hindi"),
        ("hindi mein recovery notice ready kar do", "hindi"),
        ("Tamil me police complaint draft karo", "tamil"),
        ("Please draft a legal notice in Bengali", "bengali"),
        ("Punjabi me notice bana do", "punjabi"),
        ("gujarati ma notice banavo", "gujarati"),
        ("kannada me complaint banao", "kannada"),
        ("Urdu mein legal notice likh do", "urdu"),
        ("Malayalam il complaint undakku", "malayalam"),
        # Fused locatives -- the language name and its postposition are one
        # word, so there is no separate cue token to find. Devanagari had
        # none of these at all, which is why a request written entirely in
        # Marathi resolved to no language.
        ("मराठीत तक्रार तयार करा", "marathi"),
        ("ગુજરાતીમાં નોટિસ બનાવો", "gujarati"),
        ("മലയാളത്തിൽ പരാതി തയ്യാറാക്കുക", "malayalam"),
        ("Odia re notice tiari kara", "odia"),
    ],
)
def test_named_language_is_recognised(message: str, expected: str) -> None:
    assert extract_requested_language(message) == expected


def test_merely_mentioning_a_language_is_not_a_request_to_draft_in_it() -> None:
    assert extract_requested_language("I found a lawyer who speaks Tamil") is None


@pytest.mark.parametrize(
    "message",
    [
        # The bare romanised imperative. Previously matched only when
        # followed by "hai" ("banana hai") or in the two-word "bana do", so
        # the most natural phrasing of all started no draft whatsoever --
        # "Hindi me police complaint banao" produced no draft and no
        # language, while "Tamil me police complaint draft karo" worked only
        # because it happened to contain the English word "draft".
        "police complaint banao",
        "Hindi me police complaint banao",
        "legal notice likho",
        "complaint banaiye",
    ],
)
def test_bare_hinglish_imperative_starts_a_draft(message: str) -> None:
    from app.drafting.intent import DraftIntentDetector

    assert DraftIntentDetector().detect(message).matched


def test_requested_language_reaches_the_draft_state() -> None:
    """The end of the pipeline the user actually cares about: the language
    named in the opening message is what the document gets generated in."""
    import asyncio

    from app.drafting.conversation import DraftConversationEngine

    async def start(message: str) -> dict:
        engine = DraftConversationEngine()
        # No network: this asserts language routing, not extraction quality.
        engine.extractor.extract = lambda text, template, language="english": asyncio.sleep(0, result={})
        memory: dict = {}
        await engine.handle_turn("s1", message, "english", memory)
        return memory

    assert asyncio.run(start("Hindi me police complaint banao"))["draft_language"] == "hindi"
    assert asyncio.run(start("मराठीत पोलीस तक्रार तयार करा"))["draft_language"] == "marathi"
    # Nothing named -> the ambient conversation language, unchanged.
    assert asyncio.run(start("police complaint banao"))["draft_language"] == "english"
