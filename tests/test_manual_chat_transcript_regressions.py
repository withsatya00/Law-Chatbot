"""Focused regressions copied from the September 2026 manual chat transcript."""

import asyncio
from unittest.mock import AsyncMock

from app.cache.response_cache import ANSWER_CACHE_POLICY_VERSION, ResponseCache
from app.cache.semantic_cache import RETRIEVAL_CACHE_POLICY_VERSION, SemanticCache
from app.drafting.edit_commands import EditCommandInterpreter
from app.drafting.templates import get_template
from app.rag.metadata import MetadataExtractor
from app.rag.retriever import _named_section_citation
from app.schemas.common import RetrievedChunk, SourceCitation
from app.services.chat_service import (
    ChatService,
    _applicable_law_from,
    _correct_ni_notice_attribution,
    _normalize_ni_138_heading,
)


def test_natural_ni_act_section_orders_are_act_scoped() -> None:
    assert _named_section_citation("Section 138 Negotiable Instruments Act explain karo") == (
        "138",
        "Negotiable Instruments Act",
    )
    assert _named_section_citation(
        "Cheque bounce hone par Negotiable Instruments Act Section 138 notice kitne time me bhejna hai?"
    ) == ("138", "Negotiable Instruments Act")


def test_act_before_section_does_not_swallow_the_topic_prefix() -> None:
    parsed = _named_section_citation(
        "Cheque bounce hone par Negotiable Instruments Act Section 138 notice kitne time me bhejna hai?"
    )
    assert parsed is not None
    assert parsed[1] == "Negotiable Instruments Act"


# QA retest 2026-09-24 (T072, from `QA_REPORT_100Q_RETEST_20260924.md`
# section 7 item 6): the SECTION-before-ACT pattern's `act` capture group is
# non-greedy but had no case-sensitive anchor, so under the pattern's own
# `re.IGNORECASE` a lowercase topical phrase between the section number and
# the real Act name ("Section 138 cheque dishonour Negotiable Instruments
# Act") got swallowed into the "Act name" too, producing the garbled
# "cheque dishonour Negotiable Instruments Act" -- which matched no real
# `act_name`/`source_document` metadata, so both the `act_name`-filtered
# search and `find_named_section`'s own `requested_tokens <= identity_tokens`
# check rejected the real Negotiable Instruments Act chunks. Live-reproduced:
# this exact query retrieved an unrelated Maharashtra Court Fees Act
# "Section 138" ahead of the real one before the fix.
def test_section_before_act_skips_lowercase_topic_words_without_swallowing_them() -> None:
    assert _named_section_citation(
        "Section 138 cheque dishonour Negotiable Instruments Act"
    ) == ("138", "Negotiable Instruments Act")


def test_section_before_act_still_works_with_no_filler_words() -> None:
    # The original, already-covered shape must stay byte-for-byte unchanged:
    # zero filler words is tried first (non-greedy), not skipped over.
    assert _named_section_citation("Section 63 Bharatiya Sakshya Adhiniyam") == (
        "63", "Bharatiya Sakshya Adhiniyam",
    )


def test_section_before_act_filler_skip_is_bounded_not_unlimited() -> None:
    # The filler skip is capped at 4 lowercase words, not unlimited -- within
    # the cap it still finds the Act name; past it, it correctly gives up
    # rather than reaching arbitrarily far into the sentence.
    assert _named_section_citation(
        "Section 5 talks about something unrelated Registration Act"
    ) == ("5", "Registration Act")
    assert _named_section_citation(
        "Section 5 talks about something completely unrelated Registration Act"
    ) is None


def test_exact_constitution_article_survives_relevance_gate() -> None:
    service = ChatService()
    article = RetrievedChunk(
        chunk_id="constitution-21",
        text="21. Protection of life and personal liberty.",
        score=0.01,
        metadata={"source_document": "Constitution.pdf", "article_number": "21"},
    )
    assert service._is_relevant_chunk(
        "Article 21 mein privacy right explain karo",
        "article 21 privacy right",
        article,
        intent="General Legal Query",
        language="hinglish",
    )


def test_legacy_constitution_article_metadata_is_still_accepted() -> None:
    service = ChatService()
    legacy = RetrievedChunk(
        chunk_id="legacy-constitution-21",
        text="21. Protection of life and personal liberty.",
        score=0.01,
        metadata={"source_document": "Document.pdf", "section_number": "21"},
    )
    assert service._is_relevant_chunk(
        "Article 21 mein privacy right explain karo",
        "article 21 privacy right",
        legacy,
        intent="General Legal Query",
        language="hinglish",
    )


def test_wrong_article_number_is_not_accepted_by_exact_article_rule() -> None:
    service = ChatService()
    wrong = RetrievedChunk(
        chunk_id="constitution-14",
        text="14. Equality before law.",
        score=0.01,
        metadata={"source_document": "Constitution.pdf", "article_number": "14"},
    )
    assert not service._is_relevant_chunk(
        "Article 21 mein privacy right explain karo",
        "article 21 privacy right",
        wrong,
        intent="General Legal Query",
        language="english",
    )


def test_completed_police_draft_accepts_direct_hinglish_name_edit() -> None:
    template = get_template("police_complaint")
    assert template is not None
    command = EditCommandInterpreter().interpret(
        "Pichli detail mein naam Rahul Kumar kar do", template, "hinglish"
    )
    assert command.action == "replace_field"
    assert command.target_field == "applicant_name"
    assert command.new_value == "Rahul Kumar"


def test_plain_name_mention_is_not_mistaken_for_an_edit() -> None:
    template = get_template("police_complaint")
    assert template is not None
    command = EditCommandInterpreter().interpret("Mera naam Rahul Kumar hai", template, "hinglish")
    assert command.action == "unknown"


def test_recognized_preview_edit_is_not_routed_to_rag() -> None:
    service = ChatService()
    memory = {
        "draft_mode": True,
        "draft_stage": "preview",
        "draft_template_id": "police_complaint",
        "draft_language": "hinglish",
    }
    intent = type("Intent", (), {"intent": "General Legal Information", "confidence": 0.5})()
    assert not service._is_draft_interruption(
        intent, memory, "Pichli detail mein naam Rahul Kumar kar do"
    )


def test_language_switch_acknowledgements_are_real_unicode() -> None:
    from app.core.constants import language_preference_acknowledgement

    hindi = language_preference_acknowledgement("hindi")
    assert "हिंदी" in hindi
    assert not any(marker in hindi for marker in ("à¤", "â€", "Ã", "ðŸ"))


def test_constitution_bare_heading_is_indexed_as_article_not_section() -> None:
    extractor = MetadataExtractor()
    document = asyncio.run(
        extractor.extract("भारत का संविधान\nTHE CONSTITUTION OF INDIA", {"source_document": "constitution.pdf"})
    )
    chunk = asyncio.run(
        extractor.extract("21. Protection of life and personal liberty.—No person shall be deprived...", document)
    )
    assert chunk["instrument_type"] == "constitution"
    assert chunk["article_number"] == "21"
    assert "section_number" not in chunk


def test_article_is_not_rendered_as_section_in_applicable_law() -> None:
    citation = SourceCitation(
        source_document="Document.pdf",
        act_name="The Constitution of India",
        article="14",
    )
    assert _applicable_law_from([citation]) == ["The Constitution of India — Article 14"]


def _ni_chunk(section: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"ni-{section}",
        text=text,
        score=0.9,
        metadata={
            "act_name": "Negotiable Instruments Act",
            "source_document": "Negotiable_Instruments_Act.pdf",
            "section_number": section,
        },
    )


def test_cheque_notice_timing_context_keeps_only_ni_section_138() -> None:
    service = ChatService()
    correct = _ni_chunk(
        "138",
        "The payee shall give notice to the drawer within thirty days; the drawer shall pay within fifteen days.",
    )
    wrong_same_act = _ni_chunk("139", "It shall be presumed that the holder received the cheque for a debt.")
    wrong_act = RetrievedChunk(
        chunk_id="gst-138",
        text="Section 138 of the Central Goods and Services Tax Act.",
        score=0.99,
        metadata={
            "act_name": "Central Goods and Services Tax Act",
            "source_document": "CGST.pdf",
            "section_number": "138",
        },
    )
    accepted = service._filter_relevant_context(
        "Cheque bounce notice kitne time mein bhejna hota hai?",
        "cheque dishonour notice deadline section 138",
        [wrong_act, wrong_same_act, correct],
    )
    assert accepted == [correct]


def test_section_139_notice_attribution_is_corrected_only_with_source_support() -> None:
    answer = (
        "Notice 30 days ke andar bhejna hota hai. "
        "Yeh requirement Negotiable Instruments Act ke Section 139 ke under di gayi hai."
    )
    corrected = _correct_ni_notice_attribution(
        answer,
        "Cheque bounce notice kitne time mein bhejna hota hai?",
        [
            _ni_chunk(
                "138",
                "Demand notice to the drawer must be made within thirty days; payment is due within fifteen days.",
            )
        ],
    )
    assert "Section 138 proviso (b)" in corrected
    assert "Section 139" not in corrected


def test_cache_policy_versions_stop_replaying_pre_fix_answers() -> None:
    cache = ResponseCache()
    cache._generation = AsyncMock(return_value=0)
    bucket, entry = asyncio.run(cache._keys("hinglish", "Cheque Bounce", "cheque bounce notice"))
    assert ANSWER_CACHE_POLICY_VERSION in bucket
    assert ANSWER_CACHE_POLICY_VERSION in entry
    assert RETRIEVAL_CACHE_POLICY_VERSION in asyncio.run(SemanticCache("retrieval-cache").key("Article 21"))


def test_ni_section_138_continuation_repairs_stale_section_139_metadata() -> None:
    chunk = _ni_chunk(
        "139",
        "The payee gives notice to the drawer within thirty days, and the drawer must pay within fifteen days.",
    )
    _normalize_ni_138_heading(chunk)
    assert chunk.metadata["section_number"] == "138"
    assert chunk.metadata["source_type"] == "statute"
