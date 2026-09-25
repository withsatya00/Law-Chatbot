import pytest

from app.rag.relevance import filter_relevant_context
from app.schemas.common import RetrievedChunk


@pytest.mark.parametrize("score", [0.2, 0.4, 0.9])
@pytest.mark.parametrize("language", ["English", "Hinglish", "Hindi"])
@pytest.mark.parametrize(
    ("query", "rewrite", "text"),
    [
        ("Punishment for rape under BNS?", "", "64. Punishment for rape."),
        ("BNS culpable homicide?", "", "100. Culpable homicide."),
        ("BNS theft punishment?", "", "Whoever dishonestly takes property commits theft."),
        ("BNS cheating punishment?", "", "Cheating and dishonestly inducing delivery of property."),
        ("बलात्कार की सजा?", "BNS Section 64 punishment for rape", "Punishment for rape."),
    ],
)
def test_bns_offence_gate_excludes_generic_penalty_sources(
    score: float, language: str, query: str, rewrite: str, text: str,
) -> None:
    noise = RetrievedChunk(
        chunk_id="noise", score=score,
        text="Punishment for failure to pay tax: imprisonment and fine under this section.",
        metadata={"act_name": "Maharashtra Goods and Services Tax Act", "section_number": "64"},
    )
    relevant = RetrievedChunk(
        chunk_id="relevant", score=score, text=text,
        metadata={"act_name": "Bharatiya Nyaya Sanhita"},
    )

    accepted = filter_relevant_context(query, rewrite, [noise, relevant], language=language)

    assert [chunk.chunk_id for chunk in accepted] == ["relevant"]


def test_metadata_topic_tags_do_not_substitute_for_passage_evidence() -> None:
    noise = RetrievedChunk(
        chunk_id="noise", score=0.9, text="The local authority may impose a fine.",
        metadata={"title": "BNS rape punishment", "section_number": "64"},
    )
    assert filter_relevant_context(
        "BNS Section 64 punishment for rape", "", [noise], intent="SECTION_LOOKUP",
    ) == []


def test_relevant_procedure_source_is_kept_alongside_penal_source() -> None:
    procedure = RetrievedChunk(
        chunk_id="procedure", score=0.7,
        text="Medical examination of the victim of rape during investigation.",
        metadata={"act_name": "Bharatiya Nagarik Suraksha Sanhita"},
    )
    assert filter_relevant_context("Rape under BNS: investigation procedure?", "", [procedure]) == [procedure]


def test_bare_bns_section_query_without_a_known_topic_keeps_exact_match() -> None:
    chunk = RetrievedChunk(
        chunk_id="section", score=0.2, text="Definitions.",
        metadata={"act_name": "Bharatiya Nyaya Sanhita", "section_number": "2"},
    )
    assert filter_relevant_context("BNS Section 2", "", [chunk], intent="SECTION_LOOKUP") == [chunk]


def test_english_topic_patterns_do_not_reject_hindi_source_text() -> None:
    chunk = RetrievedChunk(
        chunk_id="hindi", score=0.8, text="बलात्कार के लिए दंड।",
        metadata={"act_name": "Bharatiya Nyaya Sanhita", "section_number": "64"},
    )
    assert filter_relevant_context("BNS punishment for rape?", "", [chunk]) == [chunk]
