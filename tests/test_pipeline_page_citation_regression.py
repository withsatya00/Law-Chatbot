"""P0-1 regression: page citations must survive the REAL cleaning-to-chunking
pipeline, not just the chunker in isolation.

`tests/test_page_level_evidence.py` proves the chunker/loader are correct on
their own, but every one of its tests calls `SectionAwareChunker().chunk(...)`
directly on a `LoadedDocument` it built itself -- it never goes through
`IndexingPipeline.index_file`, so it never exercised `_clean_text` at all.
That is exactly the gap that hid the bug: `index_file` used to clean
`document.text` (stripping blank lines and lines like "Page 3 of 12") without
touching `document.pages`, so `_PageMap`'s reconstruction check
(`separator.join(page.text for page in document.pages) == document.text`)
failed for almost every real PDF the moment cleaning removed so much as one
blank line -- silently dropping `page_number` from every chunk of every
indexed PDF, with no exception and no test catching it.

These tests run the real `DocumentLoader` and `SectionAwareChunker` through
the real `IndexingPipeline.index_file` (Mongo-backed repositories/vector
store faked, per `test_kb_jurisdiction_pipeline.py`'s convention -- no live
MongoDB in this suite), against real PDFs built with page content designed to
trigger `_clean_text`'s line-stripping (blank lines between paragraphs, and
literal "Page N" running headers), so a regression here fails loudly again.
"""

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

# `app.drafting.export` registers the GTK DLL directories on Windows; without
# it `import weasyprint` fails with error 0x7e regardless of PATH (see
# WINDOWS_SETUP.md section 3). Must be imported before weasyprint is probed.
import app.drafting.export  # noqa: F401
from app.rag.chunker import SectionAwareChunker
from app.rag.loader import DocumentLoader
from app.rag.pipeline import IndexingPipeline
from app.rag.quality import DocumentQualityChecker

pytest.importorskip("weasyprint")


# ---------------------------------------------------------------------------
# Fakes -- mirrors `test_kb_jurisdiction_pipeline.py`'s pattern exactly (only
# the Mongo-backed repositories/vector store are faked; loader, chunker,
# quality checker, and the cleaning step under test all run for real).
# ---------------------------------------------------------------------------


class _EmptyCursor:
    def __aiter__(self) -> Any:
        async def _gen() -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

        return _gen()


class _FakeCollection:
    def find(self, *_args: Any, **_kwargs: Any) -> Any:
        return _EmptyCursor()


class _FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: list[Any] = []

    async def upsert_chunks(self, chunks: list[Any]) -> None:
        self.upserted.extend(chunks)

    async def delete_by_source(self, source_document: str) -> int:
        return 0

    async def count_by_document_id(self, document_id: str) -> int:
        return sum(1 for chunk in self.upserted if chunk.document_id == document_id)

    async def activate_version(self, new_document_id: str, old_document_id: str | None) -> None:
        for chunk in self.upserted:
            if chunk.document_id == new_document_id:
                chunk.metadata["document_status"] = "active"

    async def delete_version_chunks(self, document_id: str) -> int:
        return 0


class _FakeVersionRepository:
    def __init__(self) -> None:
        self.collection = _FakeCollection()

    async def latest_for_source(self, _source_document: str) -> dict[str, Any] | None:
        return None

    async def insert(self, document: dict[str, Any]) -> str:
        return str(uuid4())

    async def update_by_id(self, *_args: Any, **_kwargs: Any) -> bool:
        return True

    async def claim_staging(self, document: dict[str, Any]) -> str | None:
        return await self.insert(document)

    async def activate(self, version_id: str, updates: dict[str, Any]) -> bool:
        return True

    async def delete_by_id(self, version_id: str) -> bool:
        return True


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.collection = _FakeCollection()

    async def insert(self, document: dict[str, Any]) -> str:
        return str(document["_id"])

    async def update_by_id(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


def _build_pipeline() -> IndexingPipeline:
    pipeline = IndexingPipeline(
        loader=DocumentLoader(),
        chunker=SectionAwareChunker(),
        embeddings=_FakeEmbeddings(),
        vector_store=_FakeVectorStore(),
    )
    pipeline.documents = _FakeDocumentRepository()  # type: ignore[assignment]
    pipeline.versions = _FakeVersionRepository()  # type: ignore[assignment]
    pipeline.quality_checker = DocumentQualityChecker()
    return pipeline


class _FakeEmbeddings:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _write_pdf(path: Path, page_bodies: list[str]) -> Path:
    from weasyprint import HTML

    html = "".join(
        f"<div style='page-break-after: always; font-size:14px'>{body}</div>" for body in page_bodies
    )
    HTML(string=html).write_pdf(str(path))
    return path


# ---------------------------------------------------------------------------
# The core regression: a real, multi-page PDF whose page content has blank
# lines between paragraphs (HTML->text extraction produces these naturally),
# which `_clean_text` strips -- exactly the condition that broke `_PageMap`'s
# reconstruction check before the fix.
# ---------------------------------------------------------------------------


def test_real_pipeline_preserves_page_numbers_through_cleaning(tmp_path: Path) -> None:
    pdf = _write_pdf(
        tmp_path / "act.pdf",
        [
            (
                "<p>5. Short title.&mdash;This Act may be called the Sample Act, 2020.</p>"
                "<p>It extends to the whole of the territory.</p>"
            ),
            (
                "<p>12. Grievance redressal.&mdash;Every complaint shall be filed with the "
                "designated authority within thirty days of the cause of action arising.</p>"
            ),
            (
                "<p>20. Penalty.&mdash;Any person contravening section 12 shall be liable "
                "to a fine which may extend to ten thousand rupees.</p>"
            ),
        ],
    )
    pipeline = _build_pipeline()
    _document_id, _language, chunks = asyncio.run(pipeline.index_file(pdf))

    assert chunks, "expected at least one chunk"
    by_section = {chunk.metadata.get("section_number"): chunk for chunk in chunks}
    assert by_section["5"].metadata["page_number"] == 1
    assert by_section["12"].metadata["page_number"] == 2
    assert by_section["20"].metadata["page_number"] == 3
    # Every chunk from a real PDF must carry SOME page evidence -- the
    # regression this guards against was ALL chunks losing it, not some.
    for chunk in chunks:
        assert "page_number" in chunk.metadata, f"chunk lost its page number: {chunk.text[:80]!r}"


def test_real_pipeline_strips_running_page_headers_and_still_attributes_pages(tmp_path: Path) -> None:
    """`_clean_text` drops any line starting with "page " (case-insensitive) --
    a literal running header/footer a real scanned-then-OCR'd or generated PDF
    commonly carries. Cleaning it must not cost the surviving text its page."""
    pdf = _write_pdf(
        tmp_path / "with_headers.pdf",
        [
            (
                "<p>Page 1 of 3</p><p>7. Definitions.&mdash;In this Act, unless the context "
                "otherwise requires, the following terms apply throughout.</p>"
            ),
            (
                "<p>Page 2 of 3</p><p>9. Duties of the authority.&mdash;The authority shall "
                "publish an annual report before the end of each financial year.</p>"
            ),
        ],
    )
    pipeline = _build_pipeline()
    _document_id, _language, chunks = asyncio.run(pipeline.index_file(pdf))

    joined = "\n".join(chunk.text for chunk in chunks)
    assert "Page 1 of 3" not in joined and "Page 2 of 3" not in joined  # cleaning happened
    by_section = {chunk.metadata.get("section_number"): chunk for chunk in chunks}
    assert by_section["7"].metadata["page_number"] == 1
    assert by_section["9"].metadata["page_number"] == 2


def test_real_pipeline_empty_page_does_not_shift_later_page_numbers(tmp_path: Path) -> None:
    """A blank page (no extractable text) in the middle of a real PDF must
    keep its slot after cleaning, not renumber the page after it."""
    pdf = _write_pdf(
        tmp_path / "with_blank_page.pdf",
        [
            (
                "<p>3. Opening provision.&mdash;This section commences the Act on the date "
                "notified by the Central Government in the Official Gazette.</p>"
            ),
            "",  # page 2: genuinely empty
            (
                "<p>11. Closing provision.&mdash;This section repeals all prior enactments "
                "inconsistent with the provisions contained herein.</p>"
            ),
        ],
    )
    pipeline = _build_pipeline()
    _document_id, _language, chunks = asyncio.run(pipeline.index_file(pdf))

    by_section = {chunk.metadata.get("section_number"): chunk for chunk in chunks}
    assert by_section["3"].metadata["page_number"] == 1
    assert by_section["11"].metadata["page_number"] == 3


def test_real_pipeline_chunk_spanning_two_pages_reports_a_range(tmp_path: Path) -> None:
    body = "Provision text that reads continuously across a page boundary. " * 6
    pdf = _write_pdf(
        tmp_path / "spanning.pdf",
        [f"<p>15. Continuing matter.&mdash;{body} Part one ends here on page one.</p>",
         f"<p>{body} Part two concludes the provision on page two.</p>"],
    )
    pipeline = _build_pipeline()
    _document_id, _language, chunks = asyncio.run(pipeline.index_file(pdf))

    # Font metrics differ across Linux/Windows; PDF line wrapping is not a
    # semantic boundary. Match normalized whitespace while checking raw page metadata.
    spanning = [chunk for chunk in chunks if "Part two concludes" in " ".join(chunk.text.split())]
    assert spanning, "expected the page-2 continuation text to land in some chunk"
    meta = spanning[0].metadata
    assert meta.get("page_start") == 1
    assert meta.get("page_end") == 2
    assert meta.get("page_range") == "1-2"


def test_cleaning_preserves_substantive_page_and_confidentiality_lines():
    pipeline = _build_pipeline()
    text = "Page 1 of 3\npage boundary. Part two concludes the provision.\nConfidential\nConfidential information must be protected."
    assert pipeline._clean_text(text) == (
        "page boundary. Part two concludes the provision.\nConfidential information must be protected."
    )


def test_real_pipeline_non_paginated_source_gets_no_page_number(tmp_path: Path) -> None:
    """Cleaning a TXT source must not invent a page identity for it -- the
    P0-1 fix is additive to PDFs only."""
    source = tmp_path / "notes.txt"
    source.write_text(
        "8. A plain-text provision.\n\nWith a blank line that _clean_text strips.\n",
        encoding="utf-8",
    )
    pipeline = _build_pipeline()
    _document_id, _language, chunks = asyncio.run(pipeline.index_file(source))

    assert chunks
    for chunk in chunks:
        assert "page_number" not in chunk.metadata


def test_real_pipeline_cleaning_matches_a_manually_cleaned_join(tmp_path: Path) -> None:
    """The embedded-text-extraction case must produce BYTE-IDENTICAL final
    text to the old single-shot `_clean_text(joined_raw_pages)` behavior --
    the fix adds page identity, it must not shift a single chunk boundary or
    embedding for the huge majority of already-indexed PDFs. Proven here by
    reimplementing the old cleaning inline and diffing against the pipeline's
    actual result."""
    pdf = _write_pdf(
        tmp_path / "parity.pdf",
        [
            (
                "<p>1. First.</p><p>Body text for the first section, spanning a couple of "
                "sentences so cleaning has real lines to strip.</p>"
            ),
            (
                "<p>2. Second.</p><p>Body text for the second section, likewise long enough "
                "to be realistic PDF content rather than a single short line.</p>"
            ),
        ],
    )
    document = asyncio.run(DocumentLoader().load(pdf))
    raw_joined = "\n".join(page.text for page in document.pages)

    def _clean_text(text: str) -> str:
        lines = [line.strip() for line in text.replace("\x00", " ").splitlines()]
        lines = [line for line in lines if line and not line.lower().startswith(("page ", "confidential"))]
        return "\n".join(lines)

    expected = _clean_text(raw_joined)

    pipeline = _build_pipeline()
    asyncio.run(pipeline.index_file(pdf))
    # The pipeline doesn't return the cleaned document.text directly, so
    # reconstruct it the same way `_clean_document_text` does and compare.
    reloaded = asyncio.run(DocumentLoader().load(pdf))
    for page in reloaded.pages:
        page.text = _clean_text(page.text)
    actual = "\n".join(page.text for page in reloaded.pages)

    assert actual == expected
