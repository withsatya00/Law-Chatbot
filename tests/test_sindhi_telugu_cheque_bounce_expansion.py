"""Regression test for the Sindhi/Telugu cheque-bounce concept-bridge gap
(QA session 2026-09-24, from `QA_REPORT_100Q_RETEST_20260924.md` section 7
item 4). The retest report's own investigation claimed Sanskrit/Sindhi/
Telugu "genuinely need their own root-cause investigation" for the
cheque-bounce grounding failure -- checking against the corpus showed that
was only true for Sanskrit's ABSENCE of a bug (its query already matched
existing terms fine); Sindhi and Telugu had a real, narrow, confirmed gap.

`app/rag/multilingual.py`'s cheque/bounce concept bridge already listed
"چیک" (Urdu spelling of "cheque") and "చెక్కు" ("checku", one Telugu
transliteration), but the QA report's actual probe text used different,
equally real spellings that were simply never in either list:

* Sindhi test probe "چيڪ باؤنس ٿيڻ تي ڀارت ۾ ڪهڙي سزا آهي؟" spells "cheque"
  as "چيڪ", ending in "ڪ" (the Sindhi-specific swash kaf,
  `app/language/detector.py`'s `_SINDHI_ONLY_LETTERS`) -- a different
  Unicode string from Urdu's "چیک", not a duplicate.
* Telugu test probe "చెక్ బౌన్స్ అయితే భారతదేశంలో శిక్ష ఏమిటి?" spells
  "cheque" as "చెక్" ("chek"), distinct from "చెక్కు" ("checku") already
  listed.

Confirmed directly against `legal_english_variants`/`legal_intent_hint`
before the fix: both probes returned no cheque-bounce expansion and no
"Cheque Bounce" intent hint at all (only the generic, topic-agnostic
"punishment" pattern fired) -- so a cheque-bounce question that used either
real spelling had no lexical/BM25 boost toward the Negotiable Instruments
Act at all, relying entirely on the multilingual embedding leg.
"""

from app.rag.multilingual import legal_english_variants, legal_intent_hint

_SANSKRIT_CHEQUE_BOUNCE = "चेक-बाउंस-प्रसंगे भारते का शिक्षा भवति?"
_SINDHI_CHEQUE_BOUNCE = "چيڪ باؤنس ٿيڻ تي ڀارت ۾ ڪهڙي سزا آهي؟"
_TELUGU_CHEQUE_BOUNCE = "చెక్ బౌన్స్ అయితే భారతదేశంలో శిక్ష ఏమిటి?"

_EXPECTED_HINT = ("Cheque Bounce", "Banking and Criminal Law")


def test_sanskrit_cheque_bounce_probe_already_grounds_via_existing_terms() -> None:
    # Not a fix target -- documents that Sanskrit's failure (if reproduced
    # live) was never a missing-vocabulary problem, unlike Sindhi/Telugu
    # below. The probe's "चेक-बाउंस" already matches the plain Devanagari
    # "चेक"/"बाउंस" terms this module has always carried.
    assert legal_intent_hint(_SANSKRIT_CHEQUE_BOUNCE) == _EXPECTED_HINT


def test_sindhi_cheque_bounce_probe_gets_the_cheque_bounce_expansion() -> None:
    variants = legal_english_variants(_SINDHI_CHEQUE_BOUNCE)
    assert any("negotiable instruments act" in variant.lower() for variant in variants)
    assert legal_intent_hint(_SINDHI_CHEQUE_BOUNCE) == _EXPECTED_HINT


def test_telugu_short_cheque_spelling_gets_the_cheque_bounce_expansion() -> None:
    variants = legal_english_variants(_TELUGU_CHEQUE_BOUNCE)
    assert any("negotiable instruments act" in variant.lower() for variant in variants)
    assert legal_intent_hint(_TELUGU_CHEQUE_BOUNCE) == _EXPECTED_HINT


def test_telugu_long_cheque_spelling_is_unaffected() -> None:
    # "చెక్కు బౌన్స్" ("checku bounce") already worked before this fix --
    # the new short-form terms must not have displaced it.
    variants = legal_english_variants("నా చెక్కు బౌన్స్ అయ్యింది, ఇప్పుడు ఏమి చేయాలి?")
    assert any("negotiable instruments act" in variant.lower() for variant in variants)
