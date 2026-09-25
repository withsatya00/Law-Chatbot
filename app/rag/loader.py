import asyncio
import csv
import json
import re
from pathlib import Path
from uuid import uuid4

import structlog

from app.core.config import settings
from app.core.exceptions import AppError, BadRequestError
from app.core.optional_deps import load_attr
from app.rag.ocr import IMAGE_SUFFIXES, OcrUnavailableError, ocr_engine
from app.rag.types import LoadedDocument, LoadedPage

log = structlog.get_logger(__name__)

# Below this average extractable-character count per page, a PDF is treated as
# scanned/image-only and routed through OCR instead of trusting the (near-empty) text layer.
SCANNED_PAGE_CHAR_THRESHOLD = 20

# Every page of an official Indian Gazette PDF (the source format for BNS/
# BNSS/BSA and now the Income-tax Act, 2025) repeats a running header/footer
# identifying the PUBLICATION itself, in one of two alternating forms
# depending on left/right page: "Sec. 1]  THE GAZETTE OF INDIA EXTRAORDINARY
# 3131" (odd pages, page number oddly doubled by pypdf's extraction) or "10
# THE GAZETTE OF INDIA EXTRAORDINARY [Part II— 10" (even pages). Left
# in the extracted text, this gets interleaved into the token stream once
# per page and confuses downstream section-heading extraction two ways:
# `SECTION_HEADING_RE` anchors at line-start, so "Sec. 1]" sitting at a
# chunk boundary gets misread as the chunk's own section number "1" --
# confirmed live across hundreds of chunks of a 572-page gazette PDF, all
# wrongly stamped `section_number: "1"` -- and the interleaving can also
# split what would otherwise be one contiguous section's text across a
# page boundary. Stripped once, globally, right after page-text extraction
# (never touching non-Gazette-sourced documents, since the anchor phrase
# "THE GAZETTE OF INDIA EXTRAORDINARY" is specific enough not to appear in
# ordinary operative statute text).
_GAZETTE_RUNNING_HEADER_RE = re.compile(
    r"(?:Sec\.\s*\d+\]\s*)?\d{0,4}\s*THE GAZETTE OF INDIA EXTRAORDINARY\s*(?:\[Part\s+II[—-]\s*\d+)?\s*\d{0,4}",
    re.IGNORECASE,
)


def _strip_gazette_running_headers(text: str) -> str:
    return _GAZETTE_RUNNING_HEADER_RE.sub("", text)


class DocumentLoader:
    def __init__(self, *, handwriting: bool = False) -> None:
        self.handwriting = handwriting

    async def _load_handwriting(self, path: Path) -> list[LoadedPage]:
        from app.services.document_extraction import DocumentExtractionService, page_text

        try:
            async with asyncio.timeout(settings.handwriting_ocr_document_timeout_seconds):
                extracted = await DocumentExtractionService().extract_path(path)
        except AppError:
            raise
        except TimeoutError as exc:
            raise BadRequestError("Extraction timed out. Please split the document.") from exc
        except Exception as exc:
            raise BadRequestError("Cannot read this image or scan. Please upload a clearer, valid file.") from exc
        pages = [
            LoadedPage(page_number=page.page_number or index, text=page_text(page), extraction_method="ocr")
            for index, page in enumerate(extracted, 1)
        ]
        if not any(page.text.strip() for page in pages):
            raise BadRequestError("No readable text was found. Please upload a clearer scan.")
        return pages

    async def load(self, path: Path) -> LoadedDocument:
        suffix = path.suffix.lower()
        ocr_applied = False
        ocr_degraded_reason: str | None = None
        pages: list[LoadedPage] = []
        if suffix == ".pdf":
            text, ocr_applied, pages, ocr_degraded_reason = await self._load_pdf(path)
        elif suffix == ".docx":
            text = self._load_docx(path)
        elif suffix in {".txt", ".md", ".markdown"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix in {".html", ".htm"}:
            beautiful_soup = load_attr("bs4", "BeautifulSoup", feature="HTML document loading")
            text = beautiful_soup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser").get_text("\n")
        elif suffix == ".json":
            text = json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
        elif suffix == ".csv":
            text = self._load_csv(path)
        elif suffix == ".rtf":
            text = self._load_rtf(path)
        elif suffix == ".odt":
            text = self._load_odt(path)
        elif suffix in IMAGE_SUFFIXES:
            if self.handwriting:
                pages = await self._load_handwriting(path)
                text = "\n\n".join(page.text for page in pages)
            else:
                text = await self._load_image(path)
            ocr_applied = True
        else:
            raise BadRequestError(f"Unsupported file type: {suffix}")
        return LoadedDocument(
            document_id=str(uuid4()),
            filename=path.name,
            text=text,
            pages=pages,
            metadata={
                "source_document": path.name,
                "document_type": suffix.lstrip("."),
                "ocr_applied": ocr_applied,
                # How this document's text was obtained, recorded so a citation
                # can say whether "page 14" came from the PDF's own text layer
                # or from OCR of a scan -- which is exactly the difference
                # between a reliable page reference and a best-effort one.
                "extraction_method": (
                    "ocr" if ocr_applied else ("embedded_text" if pages or text.strip() else "none")
                ),
                "page_count": len(pages) or None,
                # Set whenever a scanned-looking PDF did NOT end up with usable
                # OCR text -- OCR disabled, the engine unavailable (Tesseract/
                # Poppler missing), or OCR ran but produced no more text than
                # the (near-empty) embedded layer. Previously this was only a
                # server-side log line (`ocr_unavailable_for_scanned_pdf`);
                # the caller silently got back near-blank text with no signal
                # that the scan itself was the problem. Surfaced so
                # `document_service.upload_and_index` can warn the user
                # instead of the document just answering questions badly.
                "ocr_degraded": ocr_degraded_reason is not None,
                "ocr_degraded_reason": ocr_degraded_reason,
            },
        )

    async def _load_pdf(self, path: Path) -> tuple[str, bool, list[LoadedPage], str | None]:
        """`(joined_text, ocr_applied, pages, ocr_degraded_reason)`.

        `joined_text` is produced exactly as before -- pages joined with a
        single newline -- so every existing chunk boundary, section split and
        stored embedding stays byte-identical. The per-page list is additive;
        it is what lets a chunk say which page it came from.

        A page contributing no text keeps its slot with `extraction_method=
        "none"`. `page_number` is the page's own 1-based number in the file, so
        an empty or unreadable page can never shift the numbering of the pages
        after it.

        `ocr_degraded_reason` is `None` whenever the returned text is as good
        as this document can give -- either it never looked scanned, or OCR
        ran and actually helped. It is set the moment a scanned-looking PDF
        falls back to its (near-empty) embedded text layer for any reason.
        """
        reader = load_attr("pypdf", "PdfReader", feature="PDF text extraction")(str(path))
        page_texts = [_strip_gazette_running_headers(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(page_texts)
        embedded_pages = [
            LoadedPage(
                page_number=index + 1,
                text=page_text,
                extraction_method="embedded_text" if page_text.strip() else "none",
            )
            for index, page_text in enumerate(page_texts)
        ]
        if not self._looks_scanned(page_texts):
            return text, False, embedded_pages, None
        if not settings.ocr_enabled:
            return text, False, embedded_pages, "ocr_disabled"
        if self.handwriting:
            pages = await self._load_handwriting(path)
            return "\n\n".join(page.text for page in pages), True, pages, None
        try:
            ocr_pages = await ocr_engine.extract_pdf_pages_individually(path)
        except OcrUnavailableError as exc:
            log.warning("ocr_unavailable_for_scanned_pdf", filename=path.name, error=str(exc))
            return text, False, embedded_pages, "ocr_unavailable"

        ocr_page_texts = [_strip_gazette_running_headers(page_text) for _, page_text, _ in ocr_pages]
        ocr_text = "\n\n".join(ocr_page_texts)
        ocr_result_pages = [
            LoadedPage(
                page_number=page_number,
                text=page_text,
                extraction_method="ocr" if page_text.strip() else "none",
            )
            for (page_number, _, _), page_text in zip(ocr_pages, ocr_page_texts, strict=True)
        ]
        if len(ocr_text.strip()) <= len(text.strip()):
            return text, False, embedded_pages, "ocr_no_improvement"

        confidences = [confidence for _, _, confidence in ocr_pages if confidence is not None]
        mean_confidence = sum(confidences) / len(confidences) if confidences else None
        if (
            settings.ocr_low_confidence_gemini_fallback
            and mean_confidence is not None
            and mean_confidence < settings.ocr_min_confidence
        ):
            log.info(
                "ocr_low_confidence_trying_gemini_fallback",
                filename=path.name, mean_confidence=mean_confidence,
            )
            fallback_pages = await self._try_gemini_fallback(path)
            if fallback_pages is not None:
                fallback_text = "\n\n".join(page.text for page in fallback_pages)
                if len(fallback_text.strip()) > len(ocr_text.strip()):
                    return fallback_text, True, fallback_pages, None
            # Tesseract's language pack (`settings.ocr_language`, fixed at
            # "eng+hin") doesn't match this scan's script and the Gemini
            # retry either isn't configured or didn't improve on it -- the
            # text below is real Tesseract output, but low-confidence enough
            # that it may be a wrong-alphabet misread rather than a genuine
            # transcription. Kept (never discarded -- something is better
            # than nothing for retrieval) but flagged for the caller.
            return ocr_text, True, ocr_result_pages, "ocr_low_confidence"
        return ocr_text, True, ocr_result_pages, None

    async def _try_gemini_fallback(self, path: Path) -> list[LoadedPage] | None:
        """Best-effort retry of a low-Tesseract-confidence scan through the
        Gemini vision OCR path, which -- unlike Tesseract here -- reads any
        script rather than only `settings.ocr_language`. `None` means the
        retry didn't help or wasn't available; the caller keeps whatever
        Tesseract already produced, since this is strictly a bonus attempt
        and must never turn a usable (if imperfect) OCR result into nothing.
        """
        try:
            return await self._load_handwriting(path)
        except AppError as exc:
            log.info("ocr_gemini_fallback_unavailable", filename=path.name, error=str(exc))
            return None

    async def _load_image(self, path: Path) -> str:
        try:
            text = await ocr_engine.extract_from_image(path)
        except OcrUnavailableError as exc:
            raise BadRequestError(f"OCR is required for image uploads but is unavailable: {exc}") from exc
        if not text.strip():
            raise BadRequestError("OCR could not extract any text from the uploaded image.")
        return text

    def _looks_scanned(self, page_texts: list[str]) -> bool:
        if not page_texts:
            return True
        average_chars = sum(len(text.strip()) for text in page_texts) / len(page_texts)
        return average_chars < SCANNED_PAGE_CHAR_THRESHOLD

    def _load_docx(self, path: Path) -> str:
        document = load_attr("docx", "Document", feature="DOCX reading")(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    def _load_csv(self, path: Path) -> str:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            return "\n".join(" | ".join(row) for row in csv.reader(handle))

    def _load_rtf(self, path: Path) -> str:
        rtf_to_text = load_attr("striprtf.striprtf", "rtf_to_text", feature="RTF reading")
        return str(rtf_to_text(path.read_text(encoding="utf-8", errors="ignore")))

    def _load_odt(self, path: Path) -> str:
        load_odt = load_attr("odf.opendocument", "load", feature="ODT reading")
        extract_odt_text = load_attr("odf.teletype", "extractText", feature="ODT reading")
        return str(extract_odt_text(load_odt(str(path)).text))
