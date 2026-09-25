"""RTF/ODT upload support: `DocumentLoader` gains a suffix-dispatch branch for
each format alongside the existing PDF/DOCX/TXT/HTML/JSON/CSV branches (see
`app/rag/loader.py`). These exercise the loader directly against real files
rather than through the full indexing pipeline (covered separately by the
existing `test_kb_ingestion_service.py`/`test_document_service_ownership.py`
suites), since text extraction itself is the new behavior.
"""

import asyncio

from odf.opendocument import OpenDocumentText
from odf.text import P

from app.rag.loader import DocumentLoader


def test_load_rtf_extracts_plain_text(tmp_path) -> None:
    path = tmp_path / "notice.rtf"
    path.write_text(r"{\rtf1\ansi\deff0 Hello World, this is a legal notice.}", encoding="utf-8")

    document = asyncio.run(DocumentLoader().load(path))

    assert "Hello World, this is a legal notice." in document.text
    assert document.metadata["document_type"] == "rtf"


def test_load_odt_extracts_plain_text(tmp_path) -> None:
    path = tmp_path / "notice.odt"
    odt = OpenDocumentText()
    odt.text.addElement(P(text="This is an ODT legal notice paragraph."))
    odt.save(str(path))

    document = asyncio.run(DocumentLoader().load(path))

    assert "This is an ODT legal notice paragraph." in document.text
    assert document.metadata["document_type"] == "odt"
