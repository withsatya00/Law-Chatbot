"""Regression test for the "Hindi retrieval investigation" fix (QA session
2026-09-24, part of BUG-106): the Devanagari cheating/fraud expansion
pattern in `SmartQueryRewriter.expansions` used to include "सजा"
("saza" -- the generic Hindi word for "punishment/sentence") alongside the
two terms actually specific to cheating/fraud
("धोखाधड़ी"/"dhokhadhadi" and
"चीट"/"cheat"). Since "saza" appears in most Hindi
criminal-law questions regardless of topic ("...ki saza kya hai?"), this
spuriously injected an unrelated "BNS Section 318 cheating..." expansion
into completely unrelated questions, diluting the correct answer's
reranking score. Live-reproduced root cause of the QA report's Hindi
cheque-bounce grounding failure.
"""

from app.rag.query_rewriter import SmartQueryRewriter

_CHEQUE_BOUNCE_HINDI = "चेक बाउंस होने पर भारत में क्या सजा है?"
_FRAUD_HINDI = "धोखाधड़ी की सजा क्या है?"
_CHEAT_HINDI = "चीट करने पर क्या सजा है?"


def test_unrelated_saza_question_no_longer_gets_spurious_cheating_expansion():
    expanded = " ".join(SmartQueryRewriter().expand_queries(_CHEQUE_BOUNCE_HINDI))
    assert "318" not in expanded
    assert "cheating" not in expanded.lower()
    # the genuinely relevant expansion must still be present
    assert "negotiable instruments act" in expanded.lower()


def test_genuine_fraud_question_still_gets_the_cheating_expansion():
    expanded = " ".join(SmartQueryRewriter().expand_queries(_FRAUD_HINDI))
    assert "318" in expanded
    assert "cheating" in expanded.lower()


def test_genuine_cheat_question_still_gets_the_cheating_expansion():
    expanded = " ".join(SmartQueryRewriter().expand_queries(_CHEAT_HINDI))
    assert "318" in expanded
    assert "cheating" in expanded.lower()
