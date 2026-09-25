"""Regression test for BUG-110 (QA session 2026-09-24, from
`QA_REPORT_100Q_RETEST_20260924.md` section 7 item 2): romanized Manipuri
input had zero script signal and zero lexical signal anywhere in
`app/language/detector.py`, so it fell through `LanguageDetector.detect()`
all the way to `langdetect` -- which has no Manipuri model and defaults an
unrecognized code to English. The live-reproduced probe from the QA report,
"check bounce oirabadi India da kari punishment oi?", was detected as
English, and its Manipuri word "kari" ("what") then landed inside
`app/language/typo_tolerance.py`'s Hindi-verb-conjugation fuzzy allowlist
(edit-distance 1 from "kar"/"karo"/"karti"), producing a nonsensical Hindi
spell-check prompt for a Manipuri speaker.

Two independent, reusable fixes:
1. `detector.py` gained `_ROMANIZED_MANIPURI_TERMS`, a curated marker-word
   table for romanized Manipuri -- the same precision-first approach already
   used for the Devanagari-script Eighth Schedule languages
   (`_DEVANAGARI_MARKER_WORDS`) and for Hinglish (`_STRONG_HINGLISH_TERMS`).
2. `ConversationIntentClassifier.classify()`/`classify_advanced()` now take
   the detected `language` and only offer the Hindi-verb typo clarification
   for english/hindi/hinglish -- the only languages the correction allowlist
   actually has vocabulary for. Any other detected language falls through to
   the ordinary "General Legal Information" path, which reaches retrieval /
   the honest GK-fallback disclaimer instead of a language-mismatched
   spell-check prompt.
"""

from app.intent.classifier import ConversationIntentClassifier
from app.language.detector import LanguageDetector

_MANIPURI_PROBE = "check bounce oirabadi India da kari punishment oi?"


def _memory() -> dict:
    return {"messages": []}


def test_romanized_manipuri_is_detected_as_manipuri_not_english() -> None:
    detector = LanguageDetector()
    assert detector.detect(_MANIPURI_PROBE) == "manipuri"


def test_romanized_manipuri_probe_does_not_trigger_hindi_typo_clarification() -> None:
    classifier = ConversationIntentClassifier()
    match = classifier.classify(_MANIPURI_PROBE, _memory(), "manipuri")
    assert match.intent != "Typo Clarification"
    assert match.clarification_message is None


def test_same_ambiguous_token_still_clarifies_when_language_is_english_or_hindi() -> None:
    # Proves the language gate is conditional, not a blanket disable: the
    # exact same message that is silently routed past clarification for
    # "manipuri" (see the test above) must still ask when the detected
    # language is one the correction allowlist actually has vocabulary for.
    # "kari" alone is genuinely ambiguous between "kar"/"karo"/"karti" under
    # that allowlist -- the QA report's own reproduction confirmed it.
    classifier = ConversationIntentClassifier()
    for language in ("english", "hindi", "hinglish"):
        match = classifier.classify(_MANIPURI_PROBE, _memory(), language)
        assert match.intent == "Typo Clarification", language
        assert match.clarification_message is not None


def test_hinglish_kya_stays_gated_open_too() -> None:
    classifier = ConversationIntentClassifier()
    # A Hinglish-detected control-word ambiguity must still be allowed to
    # ask -- only genuinely different-language detections should be gated.
    result = classifier.classify("um mere liye kya-kya kar sakte ho?", _memory(), "hinglish")
    assert result.intent == "Capability Question"
