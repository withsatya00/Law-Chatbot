import asyncio
from pathlib import Path
from typing import Any

import structlog

from app.core.config import settings
from app.core.optional_deps import MissingOptionalDependencyError, load

log = structlog.get_logger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


class OcrUnavailableError(RuntimeError):
    """Raised when the OS-level OCR toolchain (Tesseract / Poppler) is missing."""


class OcrEngine:
    """Extracts text from scanned/image-only documents using Tesseract.

    All calls are synchronous, CPU-bound C-library work under the hood, so every
    public method runs on a worker thread via ``asyncio.to_thread``.
    """

    def __init__(self, language: str | None = None) -> None:
        self.language = language or settings.ocr_language

    async def extract_from_image(self, path: Path) -> str:
        return await asyncio.to_thread(self._extract_image_sync, path)

    async def extract_from_pdf_pages(self, path: Path, max_pages: int = 50) -> str:
        return await asyncio.to_thread(self._extract_pdf_sync, path, max_pages)

    def _pillow(self) -> Any:
        """Phase 1 item 8: Pillow and pytesseract are bound here, at OCR time,
        rather than at module import.

        This module sits inside `ChatService`'s import graph (chat_service ->
        DocumentService -> rag.pipeline -> rag.loader -> rag.ocr), so importing
        them at module scope meant a host without the OCR toolchain -- a very
        common deployment, since pytesseract additionally needs an OS-level
        Tesseract install -- could not start plain legal chat or drafting at
        all, features that never touch an image. `MissingOptionalDependency
        Error` is translated to `OcrUnavailableError` so every existing
        `except OcrUnavailableError` handler (which already degrades
        gracefully) keeps working unchanged.
        """
        try:
            return load("PIL.Image", feature="OCR of scanned documents")
        except MissingOptionalDependencyError as exc:
            raise OcrUnavailableError(str(exc)) from exc

    def _tesseract(self) -> Any:
        try:
            return load("pytesseract", feature="OCR of scanned documents")
        except MissingOptionalDependencyError as exc:
            raise OcrUnavailableError(str(exc)) from exc

    def _extract_image_sync(self, path: Path) -> str:
        image_module = self._pillow()
        pytesseract = self._tesseract()
        try:
            with image_module.open(path) as image:
                return str(pytesseract.image_to_string(image, lang=self.language))
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrUnavailableError(
                "Tesseract OCR engine is not installed on this host (install tesseract-ocr)."
            ) from exc

    def _extract_pdf_sync(self, path: Path, max_pages: int) -> str:
        pytesseract = self._tesseract()
        try:
            convert_from_path = load("pdf2image", feature="OCR of scanned PDFs").convert_from_path
        except MissingOptionalDependencyError as exc:
            raise OcrUnavailableError(str(exc)) from exc
        try:
            pages = convert_from_path(str(path), dpi=300, first_page=1, last_page=max_pages)
        except Exception as exc:
            raise OcrUnavailableError(
                "Poppler is not installed on this host (install poppler-utils) or the PDF could not be rasterized."
            ) from exc
        page_texts: list[str] = []
        for page_image in pages:
            try:
                page_texts.append(pytesseract.image_to_string(page_image, lang=self.language))
            except pytesseract.TesseractNotFoundError as exc:
                raise OcrUnavailableError(
                    "Tesseract OCR engine is not installed on this host (install tesseract-ocr)."
                ) from exc
        return "\n\n".join(page_texts)

    async def extract_pdf_pages_individually(
        self, path: Path, max_pages: int = 50
    ) -> list[tuple[int, str, float | None]]:
        """`[(page_number, text, mean_word_confidence), ...]`, 1-based, one
        entry per rasterized page.

        Separate from `extract_from_pdf_pages` (which returns one joined string)
        because page identity is the whole point here: a citation that says
        "page 14" has to mean page 14 of the file.

        A page whose OCR fails yields `("", None)` and KEEPS its slot. Dropping
        it would renumber every later page by one, quietly moving every
        citation after the failure -- a wrong page number is worse than a
        missing one, because it looks checkable.

        `mean_word_confidence` is Tesseract's own 0-100 confidence scale,
        averaged over the page's recognized words (`None` for a page with no
        recognized words, including a genuinely blank one). It never changes
        `text` -- confidence scoring is a second, independent pass over the
        same rasterized image so a scoring failure can never alter the text
        actually used for indexing.
        """
        return await asyncio.to_thread(self._extract_pdf_pages_sync, path, max_pages)

    def _mean_word_confidence(self, pytesseract: Any, image: Any) -> float | None:
        try:
            data = pytesseract.image_to_data(image, lang=self.language, output_type=pytesseract.Output.DICT)
        except Exception as exc:  # noqa: BLE001 - confidence is advisory; never let it break extraction
            log.warning("ocr_confidence_scoring_failed", error=str(exc))
            return None
        confidences: list[float] = []
        for raw in data.get("conf", []):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            # Tesseract marks non-text regions (images, whitespace blocks) with
            # conf=-1 -- excluded, or a blank/mostly-graphic page would count
            # as confidently empty instead of not counting at all.
            if value >= 0:
                confidences.append(value)
        return sum(confidences) / len(confidences) if confidences else None

    def _extract_pdf_pages_sync(self, path: Path, max_pages: int) -> list[tuple[int, str, float | None]]:
        pytesseract = self._tesseract()
        try:
            convert_from_path = load("pdf2image", feature="OCR of scanned PDFs").convert_from_path
        except MissingOptionalDependencyError as exc:
            raise OcrUnavailableError(str(exc)) from exc
        try:
            pages = convert_from_path(str(path), dpi=300, first_page=1, last_page=max_pages)
        except Exception as exc:
            raise OcrUnavailableError(
                "Poppler is not installed on this host (install poppler-utils) or the PDF could not be rasterized."
            ) from exc

        results: list[tuple[int, str, float | None]] = []
        for offset, page_image in enumerate(pages):
            page_number = offset + 1
            try:
                text = pytesseract.image_to_string(page_image, lang=self.language)
            except pytesseract.TesseractNotFoundError as exc:
                # The engine is absent, not this page -- every remaining page
                # would fail identically, so report it rather than emitting a
                # document of empty pages.
                raise OcrUnavailableError(
                    "Tesseract OCR engine is not installed on this host (install tesseract-ocr)."
                ) from exc
            except Exception as exc:  # noqa: BLE001 - one unreadable page must not lose the rest
                log.warning("ocr_page_failed", page_number=page_number, error=str(exc))
                results.append((page_number, "", None))
                continue
            results.append((page_number, text, self._mean_word_confidence(pytesseract, page_image)))
        return results


ocr_engine = OcrEngine()
