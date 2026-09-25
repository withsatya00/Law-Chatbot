import pytest

from app.rag.citation import citation_from_metadata


@pytest.mark.parametrize(
    "act_name",
    [
        "The Bharatiya Nyaya Sanhita, 2023",
        "The Bharatiya Nagarik Suraksha Sanhita, 2023",
        "The Bharatiya Sakshya Adhiniyam, 2023",
        "Consumer Protection Act, 2019",
    ],
)
@pytest.mark.parametrize("section", ["35", "61", "63", "64", "100"])
def test_section_number_does_not_reassign_source_act(act_name: str, section: str) -> None:
    metadata = {"act_name": act_name, "section_number": section}

    citation = citation_from_metadata(metadata, source_document="source.pdf")

    assert citation.act_name == act_name
    assert citation.section == section
    assert citation.source_document == "source.pdf"
    assert metadata == {"act_name": act_name, "section_number": section}


@pytest.mark.parametrize(
    ("raw_name", "display_name"),
    [
        ("THE BHARA TIY A NAGARIK SURAKSHA SANHITA", "THE BHARATIYA NAGARIK SURAKSHA SANHITA"),
        ("THE BHARA TIY A NY A Y A SANHITA", "THE BHARATIYA NYAYA SANHITA"),
    ],
)
@pytest.mark.parametrize("section", ["35", "61", "63", "64", "100"])
def test_cosmetic_normalization_preserves_act_identity(
    raw_name: str, display_name: str, section: str,
) -> None:
    metadata = {"act_name": raw_name, "section_number": section}

    citation = citation_from_metadata(metadata, source_document="source.pdf")

    assert citation.act_name == display_name
    assert citation.section == section
    assert citation.source_document == "source.pdf"
    assert metadata == {"act_name": raw_name, "section_number": section}
