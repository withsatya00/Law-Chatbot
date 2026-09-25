"""Direct unit tests for `app.rag.relevance` (Phase 1 god-object split out
of `ChatService`). These exercise the free functions directly with plain
`RetrievedChunk` fixtures -- no `ChatService` instance needed, which wasn't
possible before this extraction.
"""
from app.rag.relevance import (
    filter_relevant_context,
    is_cheque_notice_timing_query,
    is_ni_notice_period_evidence,
    is_ni_section_138,
    is_relevant_chunk,
    most_relevant_chunk,
    reorder_by_relevance,
    significant_terms,
)
from app.schemas.common import RetrievedChunk


def _chunk(text: str, score: float = 0.5, **metadata: object) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=text[:20], text=text, score=score, metadata=metadata)


def test_significant_terms_drops_stopwords_and_short_tokens() -> None:
    terms = significant_terms("What is a Zero FIR and how does it work?")
    assert "zero" in terms
    assert "fir" in terms
    assert "what" not in terms
    assert "is" not in terms
    assert "a" not in terms


def test_significant_terms_allows_short_numeric_when_requested() -> None:
    assert "2" in significant_terms("Section 2", allow_short_numeric=True)
    assert "2" not in significant_terms("Section 2", allow_short_numeric=False)


def test_is_cheque_notice_timing_query_requires_all_three_signals() -> None:
    assert is_cheque_notice_timing_query("cheque bounce notice kitne din mein bhejna hai")
    assert not is_cheque_notice_timing_query("cheque bounce case kya hota hai")  # no timing word
    assert not is_cheque_notice_timing_query("legal notice kab tak")  # no cheque bounce


def test_is_ni_section_138_matches_only_the_right_act_and_section() -> None:
    ni_138 = _chunk("text", section_number="138", act_name="Negotiable Instruments Act")
    other_138 = _chunk("text", section_number="138", act_name="GST Act")
    ni_other_section = _chunk("text", section_number="139", act_name="Negotiable Instruments Act")
    assert is_ni_section_138(ni_138)
    assert not is_ni_section_138(other_138)
    assert not is_ni_section_138(ni_other_section)


def test_is_ni_notice_period_evidence_requires_both_periods_and_drawer_and_notice() -> None:
    full = _chunk(
        "within thirty days of receipt the drawer shall pay, failing which notice within fifteen days applies",
        section_number="138", act_name="Negotiable Instruments Act",
    )
    partial = _chunk("within thirty days only", section_number="138", act_name="Negotiable Instruments Act")
    assert is_ni_notice_period_evidence(full)
    assert not is_ni_notice_period_evidence(partial)


def test_significant_terms_drops_common_question_words() -> None:
    """Live repro: "which section talks about culpable homiside" shared
    nothing topical with an unrelated Maharashtra GST Act chunk, but both
    texts happened to contain "which" and "about" -- ordinary English
    function/question words with no topic-discriminating power, which were
    missing from the stopword list entirely. That made the generic fallback
    in `is_relevant_chunk` accept genuinely unrelated State Acts as
    "sources" for a plain central-law question.
    """
    terms = significant_terms("Which section talks about culpable homicide?")
    assert "culpable" in terms
    assert "homicide" in terms
    for word in ("which", "talks", "about"):
        assert word not in terms


def test_is_relevant_chunk_rejects_unrelated_act_sharing_only_function_words() -> None:
    """The same live repro at the full `is_relevant_chunk` gate: a chunk
    from a clearly unrelated Act must not pass just because it also
    contains "which"/"about" -- the same shape as the real Maharashtra GST
    Act chunk that leaked into a BNS culpable-homicide answer's sources.
    """
    query = "which section talks about culpable homicide"
    unrelated = _chunk(
        "Section 135 of the Maharashtra Goods and Services Tax Act, which deals with which officer is "
        "empowered, is about registration and applies to any dealer.",
        score=0.27, act_name="THE MAHARASHTRA GOODS AND SERVICES TAX ACT", section_number="135",
    )
    assert not is_relevant_chunk(query, query, unrelated, intent="General Legal Query", language="english")


def test_is_relevant_chunk_accepts_high_score_unconditionally() -> None:
    chunk = _chunk("totally unrelated text")
    chunk.score = 0.9
    assert is_relevant_chunk("what is bail", "what is bail", chunk)


def test_is_relevant_chunk_section_lookup_exact_match() -> None:
    chunk = _chunk("some statute text", section_number="420")
    chunk.score = 0.05
    assert is_relevant_chunk("section 420", "section 420", chunk, intent="SECTION_LOOKUP")


def test_is_relevant_chunk_topic_gate_rejects_off_topic_low_score_chunk() -> None:
    chunk = _chunk("this is about income tax filing procedures")
    chunk.score = 0.05
    assert not is_relevant_chunk("what is bail", "what is bail", chunk)


def test_filter_relevant_context_keeps_only_accepted_chunks() -> None:
    on_topic = _chunk("bail provisions under CrPC")
    on_topic.score = 0.05
    off_topic = _chunk("income tax return filing")
    off_topic.score = 0.05
    result = filter_relevant_context("what is bail", "what is bail", [on_topic, off_topic])
    assert on_topic in result
    assert off_topic not in result


def test_most_relevant_chunk_prefers_lexical_overlap_over_rerank_order() -> None:
    top_ranked_wrong_topic = _chunk("legal notice template for landlord eviction")
    lower_ranked_right_topic = _chunk("bail is release from custody pending trial")
    result = most_relevant_chunk("what is bail", [top_ranked_wrong_topic, lower_ranked_right_topic])
    assert result is lower_ranked_right_topic


def test_most_relevant_chunk_falls_back_to_first_when_no_overlap_at_all() -> None:
    chunk_a = _chunk("vehicle theft provisions")
    chunk_b = _chunk("consumer protection provisions")
    result = most_relevant_chunk("mera bike chori ho gaya", [chunk_a, chunk_b])
    assert result is chunk_a


def test_reorder_by_relevance_promotes_the_better_match_to_front() -> None:
    wrong_first = _chunk("legal notice template for landlord eviction")
    right_second = _chunk("bail is release from custody pending trial")
    result = reorder_by_relevance("what is bail", [wrong_first, right_second])
    assert result[0] is right_second


def test_reorder_by_relevance_is_a_noop_on_empty_list() -> None:
    assert reorder_by_relevance("anything", []) == []
