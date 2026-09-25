import asyncio

from app.rag.chunker import SectionAwareChunker
from app.rag.types import LoadedDocument


def test_section_aware_chunker_preserves_document_metadata() -> None:
    document = LoadedDocument(
        document_id="doc-1",
        filename="act.txt",
        text="Section 1 Short title\nText\n\nSection 2 Definitions\nMore text",
        metadata={"source_document": "act.txt"},
    )
    chunks = asyncio.run(SectionAwareChunker(max_chars=120, overlap_chars=10).chunk(document))
    assert chunks
    assert chunks[0].metadata["source_document"] == "act.txt"
    assert chunks[0].document_id == "doc-1"


def test_large_section_chunks_keep_parent_and_part_identity() -> None:
    clauses = [f"Clause {index} gives a distinct legal requirement with supporting context." for index in range(30)]
    document = LoadedDocument(
        document_id="long-act",
        filename="long-act.txt",
        text="Section 12 Duties\n" + " ".join(clauses),
        metadata={"source_document": "long-act.txt"},
    )

    chunks = asyncio.run(SectionAwareChunker(max_chars=260, overlap_chars=40).chunk(document))

    assert len(chunks) > 2
    assert len({chunk.metadata["parent_section_id"] for chunk in chunks}) == 1
    assert [chunk.metadata["section_part_index"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.metadata["section_part_count"] == len(chunks) for chunk in chunks)
    assert all(chunk.metadata["section_heading"] == "Section 12 Duties" for chunk in chunks)
    combined = " ".join(chunk.text for chunk in chunks)
    assert all(clause in combined for clause in clauses)


def test_section_with_no_title_still_forms_its_own_boundary() -> None:
    # Real text: BNSS_2023_Official_Gazette.pdf prints section 173 (FIR
    # registration) with no title/dash at all -- straight from "173." into
    # "(1)". Confirmed live: without this boundary shape, sections 172/173/174
    # (and 181, 223, 397, ...) across the whole document got packed into
    # shared buffers, and the packed buffer's OWN section_number ended up
    # "64" -- a bare cross-reference mentioned mid-sentence, not any of the
    # sections actually present -- because nothing split them apart to begin
    # with.
    text = (
        "172. Some earlier section text continues here about arrest procedure and related matters "
        "that go on for a while so this paragraph is not trivially short.\n"
        "173. (1) Every information relating to the commission of a cognizable offence, irrespective "
        "of the area where the offence is committed, may be given orally or by electronic communication "
        "to an officer in charge of a police station, and if given orally, it shall be reduced to writing."
    )
    document = LoadedDocument(
        document_id="bnss-doc", filename="BNSS_2023_Official_Gazette.pdf", text=text,
        metadata={"source_document": "BNSS_2023_Official_Gazette.pdf"},
    )

    chunks = asyncio.run(SectionAwareChunker(max_chars=2400, overlap_chars=250).chunk(document))

    assert len(chunks) == 2
    assert chunks[0].text.startswith("172.")
    assert chunks[1].text.startswith("173. (1)")
