import ast
import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.core.exceptions import BadRequestError
from app.rag.loader import DocumentLoader
from app.schemas.document_extraction import ExtractedPage, ExtractionBlock
from app.services.document_extraction import DocumentExtractionService


def test_private_image_uses_handwriting_and_preserves_frames(monkeypatch, tmp_path):
    async def extract(self, path):
        return [
            ExtractedPage(page_number=1, blocks=[]),
            ExtractedPage(page_number=2, blocks=[ExtractionBlock(kind="paragraph", text="Deposit 45000")]),
        ]

    monkeypatch.setattr(DocumentExtractionService, "extract_path", extract)
    document = asyncio.run(DocumentLoader(handwriting=True).load(tmp_path / "receipt.tiff"))
    assert [page.page_number for page in document.pages] == [1, 2]
    assert "Deposit 45000" in document.text
    assert document.metadata["ocr_applied"] is True


def test_private_image_does_not_index_empty_ocr(monkeypatch, tmp_path):
    async def extract(self, path):
        return [ExtractedPage(page_number=1, blocks=[])]

    monkeypatch.setattr(DocumentExtractionService, "extract_path", extract)
    with pytest.raises(BadRequestError, match="No readable text"):
        asyncio.run(DocumentLoader(handwriting=True).load(tmp_path / "empty.png"))


def test_private_scanned_pdf_uses_handwriting(monkeypatch, tmp_path):
    from pypdf import PdfWriter

    from app.core.config import settings

    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(path)

    async def extract(self, path):
        return [ExtractedPage(page_number=1, blocks=[ExtractionBlock(kind="paragraph", text="Handwritten clause")])]

    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr(DocumentExtractionService, "extract_path", extract)
    document = asyncio.run(DocumentLoader(handwriting=True).load(path))
    assert document.text == "Handwritten clause"
    assert document.pages[0].extraction_method == "ocr"


def test_scanned_pdf_with_ocr_disabled_is_flagged_degraded(monkeypatch, tmp_path):
    """A blank/scanned-looking PDF loaded with OCR turned off must not be
    silently indexed as if nothing were wrong -- see `DocumentLoader._load_pdf`."""
    from pypdf import PdfWriter

    from app.core.config import settings

    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(path)

    monkeypatch.setattr(settings, "ocr_enabled", False)
    document = asyncio.run(DocumentLoader().load(path))
    assert document.metadata["ocr_applied"] is False
    assert document.metadata["ocr_degraded"] is True
    assert document.metadata["ocr_degraded_reason"] == "ocr_disabled"


def test_scanned_pdf_with_ocr_engine_unavailable_is_flagged_degraded(monkeypatch, tmp_path):
    from pypdf import PdfWriter

    from app.core.config import settings
    from app.rag import loader as loader_module
    from app.rag.ocr import OcrUnavailableError

    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(path)

    async def raise_unavailable(path):
        raise OcrUnavailableError("tesseract not installed")

    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr(loader_module.ocr_engine, "extract_pdf_pages_individually", raise_unavailable)
    document = asyncio.run(DocumentLoader().load(path))
    assert document.metadata["ocr_degraded"] is True
    assert document.metadata["ocr_degraded_reason"] == "ocr_unavailable"


def test_low_confidence_ocr_retries_via_gemini_and_uses_it_if_better(monkeypatch, tmp_path):
    """Tesseract only knows `settings.ocr_language` ("eng+hin"). A scan in
    another script still gets "recognized" -- confidently wrong -- so low
    mean word-confidence is the only signal available that the wrong
    alphabet was used. When that happens, `DocumentLoader._load_pdf` should
    retry via the any-script Gemini path and prefer it if it did better."""
    from pypdf import PdfWriter

    from app.core.config import settings
    from app.rag import loader as loader_module

    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(path)

    async def low_confidence_ocr(path):
        return [(1, "gibberish misread by the wrong language model", 12.0)]

    async def extract(self, path):
        return [ExtractedPage(
            page_number=1,
            blocks=[ExtractionBlock(kind="paragraph", text="Correctly transcribed Tamil-script legal notice text")],
        )]

    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr(settings, "ocr_low_confidence_gemini_fallback", True)
    monkeypatch.setattr(settings, "ocr_min_confidence", 45.0)
    monkeypatch.setattr(loader_module.ocr_engine, "extract_pdf_pages_individually", low_confidence_ocr)
    monkeypatch.setattr(DocumentExtractionService, "extract_path", extract)

    document = asyncio.run(DocumentLoader().load(path))
    assert "Correctly transcribed Tamil-script legal notice text" in document.text
    assert document.metadata["ocr_degraded"] is False
    assert document.metadata["ocr_degraded_reason"] is None


def test_low_confidence_ocr_keeps_tesseract_text_when_gemini_unavailable(monkeypatch, tmp_path):
    from pypdf import PdfWriter

    from app.core.config import settings
    from app.core.exceptions import BadRequestError as InnerBadRequestError
    from app.rag import loader as loader_module

    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(path)

    async def low_confidence_ocr(path):
        return [(1, "gibberish misread by the wrong language model", 12.0)]

    async def extract(self, path):
        raise InnerBadRequestError("Gemini not configured")

    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr(settings, "ocr_low_confidence_gemini_fallback", True)
    monkeypatch.setattr(settings, "ocr_min_confidence", 45.0)
    monkeypatch.setattr(loader_module.ocr_engine, "extract_pdf_pages_individually", low_confidence_ocr)
    monkeypatch.setattr(DocumentExtractionService, "extract_path", extract)

    document = asyncio.run(DocumentLoader().load(path))
    assert "gibberish misread" in document.text
    assert document.metadata["ocr_degraded"] is True
    assert document.metadata["ocr_degraded_reason"] == "ocr_low_confidence"


def test_high_confidence_ocr_never_triggers_gemini_fallback(monkeypatch, tmp_path):
    from pypdf import PdfWriter

    from app.core.config import settings
    from app.rag import loader as loader_module

    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(path)

    async def high_confidence_ocr(path):
        return [(1, "cleanly recognized text", 92.0)]

    extract = Mock(side_effect=AssertionError("Gemini fallback must not run for high-confidence OCR"))

    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr(settings, "ocr_low_confidence_gemini_fallback", True)
    monkeypatch.setattr(settings, "ocr_min_confidence", 45.0)
    monkeypatch.setattr(loader_module.ocr_engine, "extract_pdf_pages_individually", high_confidence_ocr)
    monkeypatch.setattr(DocumentExtractionService, "extract_path", extract)

    document = asyncio.run(DocumentLoader().load(path))
    assert "cleanly recognized text" in document.text
    assert document.metadata["ocr_degraded"] is False
    extract.assert_not_called()


def test_normal_text_pdf_is_not_flagged_degraded(monkeypatch, tmp_path):
    import app.drafting.export  # noqa: F401  (registers GTK DLL dirs on Windows for weasyprint)

    pytest.importorskip("weasyprint")
    from weasyprint import HTML

    body = "This page carries enough embedded text to be treated as a text layer. " * 6
    path = tmp_path / "letter.pdf"
    HTML(string=f"<div>{body}</div>").write_pdf(str(path))

    document = asyncio.run(DocumentLoader().load(path))
    assert document.metadata["ocr_degraded"] is False
    assert document.metadata["ocr_degraded_reason"] is None


@pytest.mark.parametrize("uploads", [[False], [True, False]])
def test_failed_attachment_never_sends_chat_request(uploads):
    # Execute the actual handler without Streamlit's module-level app startup.
    tree = ast.parse(Path("streamlit_app/app.py").read_text(encoding="utf-8"))
    handler = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_handle_turn")
    client = Mock()
    client.current_role.return_value = "user"
    ui = SimpleNamespace(chat_message=lambda role: nullcontext(), markdown=Mock())
    namespace = {
        "Any": object, "st": ui, "auth_client": client, "SESSION": {},
        "upload_target": lambda text, role: "private",
        "_split_jurisdiction_metadata": lambda text, target: (text, None),
        "_upload_attachments": lambda *args: [
            {"name": f"scan{index}.png", "uploaded": ok} for index, ok in enumerate(uploads)
        ],
        "_render_attachment": Mock(),
    }
    exec(compile(ast.Module(body=[handler], type_ignores=[]), "handler", "exec"), namespace)  # noqa: S102
    chat = {"session_id": "test-session", "messages": [], "title": ""}
    namespace["_handle_turn"](chat, "", [object()], "Auto")
    client.request.assert_not_called()
    assert "cannot review" in chat["messages"][-1]["content"]


def test_default_attachment_question_routes_to_document_summary():
    from app.chatops.intents import DOCUMENT_SUMMARY

    assert DOCUMENT_SUMMARY.search("Summarize the uploaded document: Screenshot 2026-09-16 134403.png. Explain its key points.")
