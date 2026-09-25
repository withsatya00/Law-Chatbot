"""Regression test for BUG-108 (QA session 2026-09-24): "bonuce" (the "u"/"n"
transposed spelling of "bounce") is edit-distance 2 from "bounce" under plain
Levenshtein, one past `_fuzzy_candidates`' edit-distance-1 cutoff, so it fell
through both the exact alias table and the fuzzy matcher untouched. Live-
reproduced: "wat is teh punishment for chek bonuce in indai" left "chek"
corrected to "cheque" but "bonuce" uncorrected, so the retrieval query never
contained the contiguous phrase "cheque bounce" needed to trigger
`SmartQueryRewriter`'s cheque-bounce concept expansion -- the same question,
correctly spelled, grounds correctly; this one fell back to ungrounded
general knowledge.
"""

from app.language.typo_tolerance import normalize_for_routing
from app.rag.query_rewriter import SmartQueryRewriter


def test_bonuce_typo_is_corrected_in_the_retrieval_query():
    result = normalize_for_routing("wat is teh punishment for chek bonuce in indai")
    assert "cheque bounce" in result.retrieval_query.lower()


def test_bonuce_typo_now_triggers_the_cheque_bounce_concept_expansion_end_to_end():
    retrieval_query = normalize_for_routing(
        "wat is teh punishment for chek bonuce in indai"
    ).retrieval_query
    expanded = " ".join(SmartQueryRewriter().expand_queries(retrieval_query)).lower()
    assert "negotiable instruments act" in expanded
    assert "section 138" in expanded


def test_original_text_is_never_mutated():
    """Contract from the module's own docstring: the ORIGINAL text must
    survive untouched for chat history/audit -- only `retrieval_query`/
    `normalized_for_routing` carry corrections."""
    original = "wat is teh punishment for chek bonuce in indai"
    result = normalize_for_routing(original)
    assert original in result.retrieval_query
