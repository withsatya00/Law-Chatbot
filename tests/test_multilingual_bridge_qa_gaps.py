"""Regression tests for the cross-language retrieval gaps found in
qa-40q-multilingual-20260921 / qa-40q-regression-20260921-133853.

Every question below is the EXACT sentence that came back "no verified
document" although the knowledge base holds the governing provision (verified
against MongoDB: NI Act s.138, Indian Contract Act s.10, RTI Act s.6/19,
BNSS s.173, RBI limiting-liability circular, Payment of Wages Act s.15). The
failure was in the concept bridge, not the corpus: none of these sentences
produced an English search variant, so BM25, the reranker's topic bonuses and
the relevance gate all saw zero English legal vocabulary.
"""

import asyncio

import pytest

from app.intent.detector import IntentDetector
from app.rag import multilingual
from app.rag.multilingual import has_legal_concept, legal_english_variants, legal_intent_hint
from app.rag.query_rewriter import SmartQueryRewriter

# (label, exact QA sentence, phrase the English expansion must carry). The
# phrase is one the reranker/gate already keys its topic handling on, so a hit
# here means the downstream bonus fires too, not merely that some text appeared.
_QA_GAPS = [
    (
        "kannada FIR refusal (ZWNJ inside the abbreviation)",
        "ಪೊಲೀಸರು ಸಂಜ್ಞೇಯ ಅಪರಾಧದ ಮಾಹಿತಿ ಪಡೆದರೂ ಎಫ್‌ಐಆರ್ ದಾಖಲಿಸಲು ನಿರಾಕರಿಸಿದರೆ ಮುಂದಿನ ಕಾನೂನು ಕ್ರಮಗಳೇನು?",
        "first information report",
    ),
    (
        "gujarati cheque returned, no fixed 'bounce' phrase",
        "મારો ચેક “Funds Insufficient” કારણસર પરત આવ્યો છે. કાનૂની નોટિસની સમયમર્યાદા કઈ તારીખથી ગણાય છે?",
        "cheque bounce",
    ),
    (
        "maithili unpaid salary ('तनखा', not 'तनख्वाह')",
        "हमर मालिक दू महिनाक तनखा नहि देलनि अछि। उचित उपाय चिन्हित करबाक लेल कोन-कोन कागजात आ तथ्य आवश्यक अछि?",
        "unpaid salary",
    ),
    (
        "malayalam unauthorised bank transaction",
        "എന്റെ ബാങ്ക് അക്കൗണ്ടിൽ നിന്ന് ഞാൻ അനുമതി നൽകാത്ത ഒരു ഇടപാട് നടന്നു. ഞാൻ ഉടൻ എന്ത് ചെയ്യണം?",
        "unauthorised",
    ),
    (
        "nepali valid contract",
        "भारतमा वैध करार बन्नका लागि मुख्य सर्तहरू के हुन्? सरल उदाहरणसहित बुझाउनुहोस्।",
        "valid contract",
    ),
    (
        "odia RTI",
        "ସୂଚନା ଅଧିକାର ଆବେଦନର ଉତ୍ତର ନମିଳିଲେ କ’ଣ କରାଯାଇପାରେ?",
        "right to information",
    ),
    (
        "sanskrit legal notice ('विधिसूचना')",
        "विधिसूचना (Legal Notice) किम् अस्ति? भारते प्रत्येकं दीवानीवादात् पूर्वं विधिसूचना अनिवार्या अस्ति वा?",
        "legal notice",
    ),
    (
        "kashmiri police notice to appear (Arabic-script vowel marks)",
        "می ہس اکھ پولیس نوٹِس ملیو، یہِ حاضر گژھُن چھُ لکھنہ آمُت۔ کیانہٕ کرُن چھُ فیصلہ کرنہ برون",
        "notice of appearance",
    ),
    (
        "bodo police complaint ('पुलिस अभियोग')",
        "आंनि मोटरसाइकेलखौ गुबैनाय जायो। पुलिस अभियोगआव आं मबोरॉ मोनथिखौ होगोन?",
        "police complaint",
    ),
]


@pytest.mark.parametrize(("label", "question", "marker"), _QA_GAPS, ids=[gap[0] for gap in _QA_GAPS])
def test_qa_gap_question_now_reaches_the_english_statutory_vocabulary(label: str, question: str, marker: str) -> None:
    variants = legal_english_variants(question)
    assert variants, f"{label}: still no English search variant"
    assert all(variant.isascii() for variant in variants)
    expanded = " ".join(SmartQueryRewriter().expand_queries(question)).lower()
    assert marker in expanded, f"{label}: expansion is missing {marker!r}"


# ---------------------------------------------------------------------------
# Normalisation: the same word typed three ways must be one word.
# ---------------------------------------------------------------------------


def test_zero_width_joiner_and_non_joiner_do_not_hide_a_term() -> None:
    plain = "ಎಫ್ಐಆರ್ ಹೇಗೆ ದಾಖಲಿಸುವುದು"
    with_zwnj = "ಎಫ್‌ಐಆರ್ ಹೇಗೆ ದಾಖಲಿಸುವುದು"
    assert legal_english_variants(plain) == legal_english_variants(with_zwnj) != []


def test_precomposed_and_decomposed_nukta_letters_match_the_same_term() -> None:
    precomposed = "ज़मानत कैसे मिलती है"  # ज़मानत
    decomposed = "ज़मानत कैसे मिलती है"
    assert legal_english_variants(precomposed) == legal_english_variants(decomposed) != []


def test_arabic_script_vowel_marks_do_not_hide_a_term() -> None:
    assert legal_english_variants("پولیس نوٹس") == legal_english_variants("پولیس نوٹِس") != []


# ---------------------------------------------------------------------------
# The paired-concept rule must not turn a bare noun into a legal concept.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "मैं यह दस्तावेज़ चेक करना चाहता हूँ",  # "I want to check this document" -- 'चेक' as 'check'
        "मेरा बैंक कल बंद रहेगा",  # bank, but nothing unauthorised
        "पुलिस स्टेशन का पता बताइए",  # police, but no notice
        "the bank is open on Monday",
        "police officers were present",
        "Rahul Sharma",
        "24, Shanti Vihar, Gomti Nagar, Lucknow",
        "HDFC260828458721",
    ],
)
def test_a_lone_half_of_a_pair_does_not_trigger_the_pair(text: str) -> None:
    variants = legal_english_variants(text)
    assert multilingual._CHEQUE_EXPANSION not in variants
    assert multilingual._UNAUTHORISED_TRANSACTION_EXPANSION not in variants
    assert multilingual._POLICE_NOTICE_EXPANSION not in variants


def test_ascii_pair_terms_match_whole_words_only() -> None:
    # "atm" must not match inside "batman", nor "return" inside "returnable
    # goods policy" style words the group does not list.
    assert multilingual._UNAUTHORISED_TRANSACTION_EXPANSION not in legal_english_variants(
        "batman fraudulent"
    )


def test_a_court_summons_is_not_bridged_to_a_wages_search() -> None:
    # "तलब" means salary in Nepali but "summoned" in Hindi/Urdu; it was
    # deliberately left out of the wages concept.
    assert "unpaid salary wages employment dues labour law appointment letter" not in legal_english_variants(
        "अदालत ने मुझे तलब किया है"
    )


# ---------------------------------------------------------------------------
# Intent: the same question must land in the same legal category as English.
# ---------------------------------------------------------------------------


def test_unauthorised_transaction_and_police_notice_get_a_legal_category_hint() -> None:
    assert legal_intent_hint(_QA_GAPS[3][1]) == ("Cyber Crime", "Cyber Law")
    assert legal_intent_hint(_QA_GAPS[7][1]) == ("Arrest and Custody", "Criminal Law")


def test_detector_resolves_a_gap_question_to_a_category_rather_than_the_default() -> None:
    detector = IntentDetector()
    response = asyncio.run(detector.detect(_QA_GAPS[3][1]))
    assert response.legal_category == "Cyber Law"


# ---------------------------------------------------------------------------
# Structural invariant. `_CONCEPT_INTENT` is keyed by the expansion string, and
# that string is spelled out in three places (the concept table, the romanised
# table, the intent map). A one-character edit in one of them silently drops
# the intent hint for that concept -- so assert they still agree.
# ---------------------------------------------------------------------------


def test_every_intent_map_key_and_romanised_key_is_a_real_concept_expansion() -> None:
    expansions = {expansion for expansion, _ in multilingual._CONCEPTS}
    expansions |= {expansion for expansion, _, _ in multilingual._PAIRED_CONCEPTS}
    assert set(multilingual._CONCEPT_INTENT) <= expansions
    assert set(multilingual._ROMANIZED_WORD_TERMS) <= expansions


def test_nothing_the_qa_gaps_added_makes_ordinary_text_a_legal_concept() -> None:
    for text in ["Rahul Sharma", "9876543210", "Lucknow"]:
        assert not has_legal_concept(text)
