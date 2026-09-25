import asyncio

import pytest

from app.entity_extraction.extractor import EntityExtractor
from app.intent.detector import IntentDetector, parse_section_lookup, parse_section_lookup_act
from app.language.detector import LanguageDetector, extract_requested_language


def test_salary_query_maps_to_labour_law() -> None:
    result = asyncio.run(IntentDetector().detect("Meri salary 3 months se nahi mili"))
    assert result.intent == "Salary Issue"
    assert result.legal_category == "Labour Law"


@pytest.mark.parametrize(
    ("query", "expected_act"),
    [
        ("bns 318", "BNS"),
        ("bnss 318", "BNSS"),
        # IPC has no chunks of its own in this corpus (repealed) -- a
        # crosswalk-verified IPC number resolves to the Act its BNS mapping
        # actually belongs to, not the literal (unindexed) "IPC".
        ("ipc 420", "BNS"),
        ("section 420 ipc", "BNS"),
        # Hindi/Hinglish explanatory phrasing should still count as a
        # section lookup, not be rejected as a generic legal topic.
        ("section 30 kya hota hai", None),
        ("section 30 ka matlab kya hai", None),
        # No Act named at all -- must stay unresolved, not guessed.
        ("section 302", None),
        ("section 318", None),
        # An IPC section with no verified crosswalk entry is an explicit,
        # disclosed gap (same policy as `parse_section_lookup`'s own number
        # crosswalk) -- returns None rather than guessing "BNS".
        ("ipc 999", None),
        # Security finding C6: the FULL Act name must resolve identically to
        # its abbreviation -- live QA repro was exactly this shape ("Section
        # 302 of the Bharatiya Nyaya Sanhita" answered from BNSS instead).
        ("what is section 302 of the bharatiya nyaya sanhita?", "BNS"),
        ("section 302 of bharatiya nagarik suraksha sanhita", "BNSS"),
        ("explain section 63 under the bharatiya sakshya adhiniyam", "BSA"),
        ("section 420 of the indian penal code", "BNS"),
        ("section 154 of the code of criminal procedure", "CrPC"),
        ("bharatiya nyaya sanhita 318", "BNS"),
    ],
)
def test_parse_section_lookup_act_resolves_named_act_or_none(query: str, expected_act: str | None) -> None:
    assert parse_section_lookup_act(query) == expected_act


def test_parse_section_lookup_recognizes_hinglish_section_explanations() -> None:
    assert parse_section_lookup("section 30 kya hota hai") == "30"
    assert parse_section_lookup("section 30 ka matlab kya hai") == "30"
    assert parse_section_lookup("are section 30 kya hota hai") == "30"


def test_parse_section_lookup_recognizes_act_possessive_before_section() -> None:
    """Live repro: "BNS ki Section 100" (an entirely ordinary Hinglish
    possessive -- "BNS's Section 100") returned "no verified document" even
    though BNS Section 100 (culpable homicide) genuinely exists and answers
    correctly when asked topically -- the possessive sits BETWEEN the Act
    abbreviation and "section", which neither `_ACT_PREFIXED_SECTION_RE` nor
    `parse_section_lookup`'s trailing-filler fallback could see past.
    """
    assert parse_section_lookup("bns ki section 100") == "100"
    assert parse_section_lookup_act("bns ki section 100") == "BNS"
    assert parse_section_lookup("BNS ka section 318") == "318"
    assert parse_section_lookup_act("BNS ka section 318") == "BNS"


# Part 52 "Multilingual Draft Engine" -- explicit-language-request parsing
# used to decide which language a legal draft is generated/revised in,
# independent of whatever language the surrounding message is written in.
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Generate a legal notice in Hindi", "hindi"),
        ("Generate a legal notice in English", "english"),
        ("Generate a legal notice in Tamil", "tamil"),
        ("Draft complaint Tamil me banao", "tamil"),
        ("Police complaint Tamil la draft pannunga", "tamil"),
        ("Kannada me draft karo", "kannada"),
        ("Bangla me prepare karo", "bengali"),
        ("Draft it in English", "english"),
        ("Hindi me kar do", "hindi"),
        ("English me kar do", "english"),
        ("Please translate into english", "english"),
        # Regression for qa-40q-multilingual-20260921 BUG-08 Part B: "to" was
        # missing from `_LANGUAGE_PREFIX_CUE_WORDS` (only "in"/"into" were
        # covered), so this exact QA phrasing resolved to no requested
        # language and the draft-translate hook never fired.
        ("Convert this notice to English. Keep the facts the same.", "english"),
        ("Translate this to Hindi", "hindi"),
    ],
)
def test_extract_requested_language_recognizes_explicit_requests(text: str, expected: str) -> None:
    assert extract_requested_language(text) == expected


def test_extract_requested_language_returns_none_without_an_explicit_cue() -> None:
    # Merely mentioning a language must not count as a request to draft IN
    # that language -- only "in X" / "X me/mein/la/language" cue patterns do.
    assert extract_requested_language("I found a lawyer who speaks Tamil") is None
    assert extract_requested_language("Legal notice bana do") is None


def test_hinglish_detection() -> None:
    assert LanguageDetector().detect("Mera landlord deposit nahi de raha hai") == "hinglish"


def test_hinglish_detection_without_old_confirmation_words() -> None:
    """Part 39's flagship example: sentences that are unambiguously Hinglish
    but contain none of the original 4 "confirmation" words (nahi/hai/kya/
    mera) must still be detected as Hinglish, not silently fall through to
    `langdetect` and come back English.
    """
    assert LanguageDetector().detect("Teacher ne mujhe mara.") == "hinglish"
    assert LanguageDetector().detect("Mujhe kisne mara tha?") == "hinglish"


def test_language_resolve_inherits_previous_language_for_short_followups() -> None:
    """A short, script-less follow-up ("Why?", "Fir?") carries too little
    signal to trust a fresh per-message detection -- it should inherit the
    conversation's established language instead of risking a random switch.
    """
    detector = LanguageDetector()
    assert detector.resolve("Why?", "hinglish") == "hinglish"
    assert detector.resolve("Fir?", "hindi") == "hindi"
    # A message with its own script signal still overrides.
    assert detector.resolve("कैसे?", "english") == "hindi"


# Part 47.1 "Fix Tested Answer Quality Issues" -- regression tests for the
# real conversation transcript that surfaced these language-detection bugs.
def test_english_question_switches_language_even_after_hindi_turn() -> None:
    # "What is bail?" is a complete, unambiguous English question (14 chars,
    # under the short-text inherit threshold) -- it must switch the session
    # language, not silently inherit "hindi" from the previous turn.
    assert LanguageDetector().resolve("What is bail?", "hindi") == "english"


def test_hinglish_with_ko_kro_is_not_misread_as_english() -> None:
    # "ko"/"kro" were missing from the Hinglish token list -- these two
    # genuinely Hinglish messages have no other strong signal and were
    # falling through to `langdetect`, which read their majority-English
    # vocabulary as plain English.
    assert LanguageDetector().resolve("fir ko 30 words me explain kro", None) == "hinglish"
    assert LanguageDetector().resolve("Bail ko simple language me samjhao, example ke bina.", "english") == "hinglish"


def test_the_alone_does_not_force_hinglish_on_plain_english() -> None:
    # "the" was listed as a STRONG Hinglish signal (a romanized-Hindi
    # past-tense marker) but collides with the single most common word in
    # English -- any English sentence containing "the" was misdetected as
    # Hinglish. Confirmed live regression: this exact question came back in
    # Hindi despite being straightforward English.
    assert LanguageDetector().detect("What is the difference between FIR and NCR?") == "english"
    assert LanguageDetector().detect("Can police refuse to register an FIR?") == "english"


def test_mixed_multi_turn_conversation_follows_each_intentional_language_switch() -> None:
    # Part 48 item 4: a user who deliberately switches language mid-
    # conversation must be followed each time, not stuck on whichever
    # language happened to be detected first.
    detector = LanguageDetector()
    language = None
    turns = [
        # Romanized (not Devanagari) Hindi is classified as "hinglish" by
        # this system's own definitions -- "hindi" is reserved for text
        # actually written in Devanagari script.
        ("FIR kya hoti hai?", "hinglish"),
        ("What is bail?", "english"),
        ("bail kaise milti hai?", "hinglish"),
        ("Thanks, that's clear.", "english"),
        ("accha, ek aur sawaal hai.", "hinglish"),
    ]
    for text, expected in turns:
        language = detector.resolve(text, language)
        assert language == expected, f"{text!r} -> {language!r}, expected {expected!r}"


def test_short_ambiguous_followups_still_inherit_previous_language() -> None:
    # Regression guard: the two fixes above must not stop genuinely
    # ambiguous short follow-ups from inheriting, per the existing spec
    # examples.
    detector = LanguageDetector()
    assert detector.resolve("Why?", "hinglish") == "hinglish"
    assert detector.resolve("Kitna time?", "hinglish") == "hinglish"


def test_entity_extraction_sections_and_amounts() -> None:
    result = asyncio.run(EntityExtractor().extract("Section 138 notice for Rs. 50,000 dated 12/01/2025"))
    assert "section_number" in result.entities
    assert "amount" in result.entities
    assert "date" in result.entities


def test_entity_extraction_does_not_misread_ordinary_words_as_identifiers() -> None:
    # Regression guard: the keyword prefixes ("sec"/"case"/"fir"/"complaint")
    # used to be matched under a blanket re.IGNORECASE that also loosened the
    # value-capturing groups (meant to require digits/uppercase), so ordinary
    # lowercase text right after the keyword -- e.g. "sec" inside "security",
    # or free text after "case" -- was wrongly captured as a section/case
    # number. Real identifiers (uppercase/digits) must still be found.
    result = asyncio.run(
        EntityExtractor().extract(
            "My landlord is not returning my security deposit of Rs 50000. "
            "This case is very serious and I need help."
        )
    )
    assert "urity" not in result.entities.get("section_number", [])
    assert not any(v.isalpha() and v.islower() for v in result.entities.get("section_number", []))
    assert not any(v.isalpha() and v.islower() for v in result.entities.get("case_number", []))

    real_ids = asyncio.run(EntityExtractor().extract("Case No: CRL/1234/2025 under Section 138 NI Act"))
    assert "CRL/1234/2025" in real_ids.entities.get("case_number", [])
    assert "138" in real_ids.entities.get("section_number", [])


# ---------------------------------------------------------------------------
# Eighth Schedule coverage: the ten languages added alongside the original
# twelve. Three distinct identification mechanisms are exercised here, because
# script alone is only sufficient for two of them:
#   - own script      -> Manipuri (Meetei Mayek), Santali (Ol Chiki)
#   - shared script,
#     letter-based    -> Sindhi / Kashmiri vs Urdu (all Perso-Arabic)
#   - shared script,
#     marker words    -> Sanskrit / Maithili / Konkani / Dogri / Bodo / Nepali
#                        / Marathi vs Hindi (all Devanagari)
# ---------------------------------------------------------------------------

EIGHTH_SCHEDULE_SAMPLES = [
    ("hindi", "मुझे धोखाधड़ी के संबंध में पुलिस शिकायत तैयार करवानी है।"),
    ("marathi", "मला पोलीस तक्रार तयार करायची आहे, कृपया मदत करा."),
    ("nepali", "मलाई प्रहरी उजुरी तयार गर्नुपर्ने छ, कुनै सहयोग गर्नुहोस्।"),
    ("sanskrit", "अस्मिन् विषये किमपि प्रमाणितं पत्रं न उपलभ्यते इति।"),
    ("maithili", "हमरा पुलिस शिकायत तैयार करू, कोनो सहयोग अछि की नहि?"),
    ("konkani", "म्हाका पोलिस कागाळ तयार करचें आसा, तुका खंय जाय?"),
    ("dogri", "मिंजो पुलिस शिकायत तैयार करनी ऐह, तुस कन्ने गल्ल करो।"),
    ("bodo", "आं पुलिस बिबान खालामनाय थानाय, नोंथाङ आरो मोननो हायाखै।"),
    ("manipuri", "ꯃꯦꮋꯣꯩ ꮋꯤꯍꯩꮙꯣ ꮆꯩꮊꯩ ꮚꯩꯣꮛꯩ"),
    ("santali", "ᱚᱚᱲᱠ ᱠᱩᱠᱞᱲ ᱥᱱᱟ ᱜᱱᱲᱟᱩ ᱤᱱᱚᱟᱳ"),
    ("urdu", "مجھے پولیس میں شکایت درج کرانی ہے۔"),
    ("sindhi", "مون کي پوليس ھ شڪايت داخل ڪرڈي آهي."),
    ("kashmiri", "مےْ چھُ پولیس منْز شکایت درج کرٕنۍ۔"),
]


@pytest.mark.parametrize("expected,text", EIGHTH_SCHEDULE_SAMPLES)
def test_eighth_schedule_languages_are_detected(expected: str, text: str) -> None:
    assert LanguageDetector().detect(text) == expected


def test_devanagari_without_marker_words_still_defaults_to_hindi() -> None:
    """The marker tables must not steal ordinary Hindi. Anything with no marker
    hit falls through to langdetect and then Hindi -- the behaviour that
    existed before the other seven Devanagari languages were added."""
    assert LanguageDetector().detect(
        "पुलिस स्टेशन का पता बताइए"
    ) == "hindi"


def test_every_supported_language_has_a_no_verified_context_translation() -> None:
    """The strict-RAG refusal must never fall back to a script the user cannot
    read -- that was the original bug (hardcoded Devanagari for everyone)."""
    from app.core.constants import NO_VERIFIED_CONTEXT_MESSAGES, SUPPORTED_LANGUAGES

    assert not SUPPORTED_LANGUAGES - set(NO_VERIFIED_CONTEXT_MESSAGES)


def test_every_template_has_a_name_in_every_eighth_schedule_language() -> None:
    """Guards the table against drift: adding a 56th template must not leave
    ten languages silently falling back to the English name."""
    from app.drafting.eighth_schedule_names import _BY_LANGUAGE
    from app.drafting.templates import list_templates

    draft_ids = {template.draft_id for template in list_templates()}
    for language, names in _BY_LANGUAGE.items():
        assert set(names) == draft_ids, f"{language} is out of sync with the template list"


EIGHTH_SCHEDULE_DRAFT_REQUESTS = [
    ("police_complaint", "मलाई प्रहरी उजुरी तयार गर्नुपर्ने छ"),
    ("police_complaint", "म्हाका पोलिस कागाळ तयार करचें आसा"),
    ("police_complaint", "आं पुलिस बिबान बानाय नांगौ"),
    ("police_complaint", "पुलिस शिकायत तैयार करू"),
    ("police_complaint", "مون کي پوليس شڪايت تيار ڪر"),
]


@pytest.mark.parametrize("expected_draft_id,text", EIGHTH_SCHEDULE_DRAFT_REQUESTS)
def test_drafting_requests_in_eighth_schedule_languages_pick_the_right_template(
    expected_draft_id: str, text: str
) -> None:
    """Both halves of recognition have to work for these languages: a native
    drafting VERB (`_STRONG_DRAFT_PHRASE_PATTERNS`) and a native document NAME
    (`eighth_schedule_names`). Missing either one sent the request to ordinary
    RAG chat instead of starting a draft."""
    from app.drafting.intent import DraftIntentDetector

    assert DraftIntentDetector().detect(text).draft_id == expected_draft_id
