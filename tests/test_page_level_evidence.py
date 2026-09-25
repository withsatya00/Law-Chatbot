"""Phase 2 Milestone C: page-level evidence.

The rule this module defends: **a page number shown to a user must be the page
the text is actually on, or absent.** There is no third option -- a wrong page
reference is worse than a missing one, because it looks checkable and a reader
will act on it.

Consequences tested here:
  * an empty or unreadable page keeps its slot, so it cannot shift the
    numbering of every page after it;
  * a format with no pages (TXT/DOCX/HTML) yields `page_number=None`, never 1;
  * a chunk whose text cannot be located verbatim claims no page at all;
  * the 2079 chunks indexed before Phase 2 still serialize cleanly.
"""

import asyncio
from pathlib import Path

import pytest

# `app.drafting.export` registers the GTK DLL directories on Windows; without
# it `import weasyprint` fails with error 0x7e regardless of PATH (see
# WINDOWS_SETUP.md section 3). Must be imported before weasyprint is probed.
import app.drafting.export  # noqa: F401
from app.rag.chunker import SectionAwareChunker
from app.rag.loader import DocumentLoader
from app.rag.types import LoadedDocument, LoadedPage
from app.schemas.common import RetrievedChunk, SourceCitation

pytest.importorskip("weasyprint")


def _document(pages: list[LoadedPage], *, ocr: bool = False) -> LoadedDocument:
    separator = "\n\n" if ocr else "\n"
    return LoadedDocument(
        document_id="doc-1",
        filename="probe.pdf",
        text=separator.join(page.text for page in pages),
        pages=pages,
        metadata={"source_document": "probe.pdf", "document_type": "pdf", "ocr_applied": ocr},
    )


def _chunks(document: LoadedDocument) -> list:
    return asyncio.run(SectionAwareChunker().chunk(document))


# ---------------------------------------------------------------------------
# Loader: real multi-page PDF
# ---------------------------------------------------------------------------


def _write_pdf(path: Path, page_bodies: list[str]) -> Path:
    from weasyprint import HTML

    html = "".join(
        f"<div style='page-break-after: always; font-size:20px'>{body}</div>"
        for body in page_bodies
    )
    HTML(string=html).write_pdf(str(path))
    return path


def test_a_multi_page_pdf_keeps_each_page_number(tmp_path: Path) -> None:
    pdf = _write_pdf(
        tmp_path / "multi.pdf",
        ["Alpha clause on the first page.", "Beta clause on the second page.",
         "Gamma clause on the third page."],
    )
    document = asyncio.run(DocumentLoader().load(pdf))

    assert [page.page_number for page in document.pages] == [1, 2, 3]
    assert "Alpha" in document.pages[0].text
    assert "Beta" in document.pages[1].text
    assert "Gamma" in document.pages[2].text
    assert document.metadata["page_count"] == 3
    assert document.metadata["extraction_method"] == "embedded_text"


def test_loader_joined_text_is_unchanged_by_page_capture(tmp_path: Path) -> None:
    """Page capture is additive. If the joined text changed, every existing
    chunk boundary and stored embedding would silently shift."""
    # Long enough that `_looks_scanned` is False, so the realistic embedded-text
    # path is exercised rather than the OCR fallback (which joins with "\n\n").
    body = "This page carries enough embedded text to be treated as a text layer. " * 6
    pdf = _write_pdf(tmp_path / "join.pdf", [f"First. {body}", f"Second. {body}"])
    document = asyncio.run(DocumentLoader().load(pdf))
    assert document.metadata["ocr_applied"] is False
    assert document.text == "\n".join(page.text for page in document.pages)


def test_a_non_paginated_format_reports_no_pages(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("Section 1. A plain text file has no pages.", encoding="utf-8")
    document = asyncio.run(DocumentLoader().load(target))

    assert document.pages == []
    assert document.is_paginated is False
    assert document.metadata["page_count"] is None
    chunk = _chunks(document)[0]
    assert "page_number" not in chunk.metadata


# ---------------------------------------------------------------------------
# Numbering integrity
# ---------------------------------------------------------------------------


def test_an_empty_page_keeps_its_slot_and_does_not_shift_later_pages() -> None:
    """Page 2 yields nothing. Page 3's content must still be attributed to
    page 3, not promoted to page 2."""
    document = _document([
        LoadedPage(page_number=1, text="Section 1. First provision.", extraction_method="embedded_text"),
        LoadedPage(page_number=2, text="", extraction_method="none"),
        LoadedPage(page_number=3, text="Section 3. Third provision.", extraction_method="embedded_text"),
    ])
    by_text = {chunk.text: chunk.metadata for chunk in _chunks(document)}
    third = next(meta for text, meta in by_text.items() if "Third provision" in text)
    assert third["page_number"] == 3


def test_ocr_failure_on_one_page_does_not_renumber_the_rest() -> None:
    """Mirrors `OcrEngine.extract_pdf_pages_individually`, which yields "" for a
    page whose OCR raised and keeps its position."""
    document = _document(
        [
            LoadedPage(page_number=1, text="Section 1. Readable scan.", extraction_method="ocr"),
            LoadedPage(page_number=2, text="", extraction_method="none"),
            LoadedPage(page_number=3, text="Section 3. Also readable.", extraction_method="ocr"),
        ],
        ocr=True,
    )
    pages = {
        chunk.metadata.get("page_number")
        for chunk in _chunks(document)
        if "Also readable" in chunk.text
    }
    assert pages == {3}


def test_ocr_pages_are_marked_as_ocr_extracted() -> None:
    document = _document(
        [LoadedPage(page_number=1, text="Section 1. Scanned provision.", extraction_method="ocr")],
        ocr=True,
    )
    assert _chunks(document)[0].metadata["extraction_method"] == "ocr"


def test_page_numbering_is_one_based_not_zero_based() -> None:
    document = _document([
        LoadedPage(page_number=1, text="Section 1. Opening provision.", extraction_method="embedded_text"),
    ])
    assert _chunks(document)[0].metadata["page_number"] == 1


# ---------------------------------------------------------------------------
# Chunks spanning page boundaries
# ---------------------------------------------------------------------------


def test_a_chunk_spanning_two_pages_reports_a_range() -> None:
    document = _document([
        LoadedPage(page_number=7, text="Section 12. A provision that begins here and", extraction_method="embedded_text"),
        LoadedPage(page_number=8, text="continues onto the following page before ending.", extraction_method="embedded_text"),
    ])
    spanning = [chunk for chunk in _chunks(document) if "continues onto" in chunk.text]
    assert spanning, "expected the boundary-spanning text in some chunk"
    meta = spanning[0].metadata
    assert meta["page_start"] == 7
    assert meta["page_end"] == 8
    assert meta["page_range"] == "7-8"


def test_a_single_page_chunk_reports_no_range() -> None:
    document = _document([
        LoadedPage(page_number=4, text="Section 9. Wholly contained on one page.", extraction_method="embedded_text"),
    ])
    meta = _chunks(document)[0].metadata
    assert meta["page_start"] == meta["page_end"] == 4
    assert "page_range" not in meta


# ---------------------------------------------------------------------------
# Never invent a page
# ---------------------------------------------------------------------------


def test_no_page_is_claimed_when_the_chunk_text_cannot_be_located() -> None:
    """A `LoadedDocument` whose `pages` disagree with its `text` (a malformed
    or externally-built document) must produce no page metadata rather than a
    guess."""
    document = LoadedDocument(
        document_id="doc-x",
        filename="mismatch.pdf",
        text="Section 1. Text that appears nowhere in the page list.",
        pages=[LoadedPage(page_number=1, text="Completely different content.", extraction_method="embedded_text")],
        metadata={"source_document": "mismatch.pdf", "document_type": "pdf"},
    )
    assert "page_number" not in _chunks(document)[0].metadata


def test_pages_are_never_fabricated_for_a_document_without_them() -> None:
    document = LoadedDocument(
        document_id="doc-y", filename="plain.docx",
        text="Section 1. Body.\n\nSection 2. More body.",
        metadata={"source_document": "plain.docx", "document_type": "docx"},
    )
    for chunk in _chunks(document):
        assert "page_number" not in chunk.metadata
        assert "page_start" not in chunk.metadata


# ---------------------------------------------------------------------------
# Citation serialization and backward compatibility
# ---------------------------------------------------------------------------


def test_a_citation_renders_a_single_page_reference() -> None:
    citation = SourceCitation(
        source_document="bns.pdf", act_name="Bharatiya Nyaya Sanhita", section="318",
        page_number=14, page_start=14, page_end=14, extraction_method="embedded_text",
    )
    assert citation.page_reference == "page 14"
    assert "page 14" in citation.label


def test_a_citation_renders_a_page_range() -> None:
    citation = SourceCitation(source_document="bnss.pdf", page_start=7, page_end=8)
    assert citation.page_reference == "pages 7-8"
    assert "pages 7-8" in citation.label


def test_a_citation_without_page_evidence_says_so_by_omission() -> None:
    """The 2079 chunks indexed before Phase 2 have no page data. They must
    serialize cleanly and claim nothing."""
    citation = SourceCitation(source_document="legacy.pdf", act_name="Some Act", section="12")
    assert citation.page_reference is None
    assert citation.page_number is None
    assert "page" not in citation.label.lower()
    assert citation.model_dump()["page_number"] is None


def test_citation_page_fields_survive_a_round_trip() -> None:
    original = SourceCitation(
        source_document="bns.pdf", page_number=3, page_start=3, page_end=4,
        extraction_method="ocr",
    )
    restored = SourceCitation(**original.model_dump())
    assert restored.page_start == 3
    assert restored.page_end == 4
    assert restored.extraction_method == "ocr"
    assert restored.page_reference == "pages 3-4"


def test_the_citation_engine_carries_page_evidence_from_chunk_metadata() -> None:
    from app.rag.citation import LegalCitationEngine

    chunks = [
        RetrievedChunk(
            chunk_id="c1", text="Section 318. Cheating.", score=0.9,
            metadata={
                "source_document": "bns.pdf", "act_name": "Bharatiya Nyaya Sanhita",
                "section_number": "318", "page_number": 42, "page_start": 42,
                "page_end": 42, "extraction_method": "embedded_text",
            },
        ),
        RetrievedChunk(
            chunk_id="c2", text="Older chunk with no page data.", score=0.7,
            metadata={"source_document": "legacy.pdf", "section_number": "9"},
        ),
    ]
    citations = LegalCitationEngine().citations_from_chunks(chunks)
    by_document = {citation.source_document: citation for citation in citations}

    assert by_document["bns.pdf"].page_number == 42
    assert by_document["bns.pdf"].page_reference == "page 42"
    # The pre-Phase-2 chunk must not inherit the other chunk's page.
    assert by_document["legacy.pdf"].page_number is None
    assert by_document["legacy.pdf"].page_reference is None


# ---------------------------------------------------------------------------
# Ownership is unaffected by page capture
# ---------------------------------------------------------------------------


def test_page_capture_preserves_owner_metadata_on_private_uploads() -> None:
    """1505 indexed chunks carry `owner_user_id`. Page metadata is additive and
    must not displace the field that keeps one user's upload out of another
    user's retrieval."""
    document = LoadedDocument(
        document_id="doc-private", filename="my_lease.pdf",
        text="Section 1. Tenancy terms.",
        pages=[LoadedPage(page_number=5, text="Section 1. Tenancy terms.", extraction_method="embedded_text")],
        metadata={
            "source_document": "my_lease.pdf", "document_type": "pdf",
            "owner_user_id": "user-1", "visibility": "private",
        },
    )
    meta = _chunks(document)[0].metadata
    assert meta["owner_user_id"] == "user-1"
    assert meta["visibility"] == "private"
    assert meta["page_number"] == 5
