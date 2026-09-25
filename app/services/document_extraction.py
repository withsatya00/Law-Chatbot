"""Ephemeral, layout-aware transcription and editable export of private uploads."""

import asyncio
import base64
import io
import tempfile
import threading
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph
from fastapi import UploadFile
from PIL import Image
from pypdf import PdfReader

from app.core.config import settings
from app.core.exceptions import AppError, BadRequestError
from app.schemas.document_extraction import (
    DocumentExtractionResponse,
    ExtractedPage,
    ExtractionBlock,
)
from app.services.handwriting_ocr import HandwritingOCR, image_bytes
from app.utils.malware_scan import ClamAVScanner, MalwareScanner
from app.utils.upload_storage import write_upload

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
EXTRACTION_TYPES = IMAGE_TYPES | {".pdf", ".docx", ".txt"}
# PDFium's native library is not thread-safe, even across separate documents.
_PDF_RENDER_LOCK = threading.Lock()


def page_text(page: ExtractedPage) -> str:
    return "\n\n".join(
        "\n".join("\t".join(row) for row in block.rows) if block.kind == "table" else block.text
        for block in page.blocks
    )


def export_docx(pages: list[ExtractedPage]) -> bytes:
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(0.7)
    section.left_margin = section.right_margin = Inches(0.7)
    style = document.styles["Normal"]
    style.font.name, style.font.size = "Calibri", Pt(11)
    style.paragraph_format.space_after = Pt(6)
    for index, page in enumerate(pages):
        if index and page.page_number is not None:
            document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        for block in page.blocks:
            if block.kind == "table":
                columns = max(len(row) for row in block.rows)
                table = document.add_table(rows=0, cols=columns)
                table.style = "Table Grid"
                for row in block.rows:
                    cells = table.add_row().cells
                    for column, value in enumerate(row):
                        cells[column].text = value
            else:
                paragraph = (
                    document.add_heading(block.text, level=block.level)
                    if block.kind == "heading" else document.add_paragraph(block.text)
                )
                # Retain source list markers rather than generating duplicate numbers.
                if block.kind == "list_item":
                    paragraph.paragraph_format.left_indent = Inches(0.2)
                if any("\u0600" <= char <= "\u06ff" for char in block.text):
                    bidi = OxmlElement("w:bidi")
                    paragraph._p.get_or_add_pPr().append(bidi)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class DocumentExtractionService:
    def __init__(self, ocr: HandwritingOCR | None = None, scanner: MalwareScanner | None = None) -> None:
        self.ocr = ocr or HandwritingOCR()
        self.scanner = scanner or ClamAVScanner()

    async def extract_upload(self, upload: UploadFile) -> DocumentExtractionResponse:
        filename = Path((upload.filename or "document").replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in EXTRACTION_TYPES:
            raise BadRequestError("Use PDF, DOCX, TXT, PNG, JPEG, TIFF, BMP or WebP for text extraction.")
        root = Path(settings.upload_storage_dir)
        root.mkdir(parents=True, exist_ok=True)
        # No extraction is stored in the KB, caches or another user's session.
        with tempfile.TemporaryDirectory(prefix="extract-", dir=root) as directory:
            path = Path(directory) / ("source" + suffix)
            await write_upload(upload, path)
            try:
                await self.scanner.scan(path)
            except ValueError as exc:
                raise BadRequestError("Upload was rejected by the file scanner.") from exc
            try:
                async with asyncio.timeout(settings.handwriting_ocr_document_timeout_seconds):
                    pages = await self.extract_path(path)
            except TimeoutError as exc:
                raise BadRequestError("Extraction timed out. Split the document into smaller files.") from exc
            except AppError:
                raise
            except Exception as exc:
                # Parser/decoder errors should not expose file paths or provider internals.
                raise BadRequestError("Cannot read this document. Upload an unencrypted, valid file.") from exc
        text = "\n\n\f\n\n".join(page_text(page) for page in pages)
        if not text.strip():
            raise BadRequestError("No readable text was found. Please upload a clearer scan.")
        docx = await asyncio.to_thread(export_docx, pages)
        return DocumentExtractionResponse(
            filename=filename, pages=pages, text=text, docx_base64=base64.b64encode(docx).decode("ascii"),
            extraction_method="embedded_text_and_images" if suffix == ".docx" else (
                "embedded_text" if suffix == ".txt" else "vision_ocr"
            ),
            warnings=[
                "Review names, dates, amounts and [illegible] markers against the original before use.",
                "Headings, paragraphs, lists, tables and page order are reconstructed; exact fonts, spacing and handwriting are not reproduced.",
            ],
        )

    def _check_count(self, count: int) -> None:
        if count > settings.handwriting_ocr_max_pages:
            raise BadRequestError(
                f"Extraction supports up to {settings.handwriting_ocr_max_pages} pages/images per file. Split this document first."
            )

    async def extract_path(self, path: Path) -> list[ExtractedPage]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            count = await asyncio.to_thread(self._pdf_count, path)
            self._check_count(count)
            pages = []
            # Render ALL pages, including pages with a printed text layer plus handwriting.
            # One page at a time bounds memory and preserves physical page identity.
            for number in range(1, count + 1):
                png = await asyncio.to_thread(self._pdf_page, path, number)
                pages.append(await self.ocr.extract(png, number))
            return pages
        if suffix in IMAGE_TYPES:
            frames = await asyncio.to_thread(self._image_frames, path)
            return [await self.ocr.extract(frame, number) for number, frame in enumerate(frames, 1)]
        if suffix == ".docx":
            return await self._docx(path)
        if suffix == ".txt":
            text = await asyncio.to_thread(path.read_text, encoding="utf-8-sig")
            return [ExtractedPage(blocks=[ExtractionBlock(kind="paragraph", text=text)])]
        raise BadRequestError("Unsupported extraction format.")

    @staticmethod
    def _pdf_count(path: Path) -> int:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise BadRequestError("Upload an unlocked PDF for extraction.")
        return len(reader.pages)

    @staticmethod
    def _pdf_page(path: Path, number: int) -> bytes:
        import pypdfium2 as pdfium

        with _PDF_RENDER_LOCK, pdfium.PdfDocument(str(path)) as document:
            page = document[number - 1]
            scale = min(200 / 72, 2400 / max(page.get_size()))
            bitmap = page.render(scale=scale)
            try:
                with bitmap.to_pil() as image:
                    return image_bytes(image)
            finally:
                bitmap.close()
                page.close()

    def _image_frames(self, path: Path) -> list[bytes]:
        with Image.open(path) as image:
            count = getattr(image, "n_frames", 1)
            self._check_count(count)
            frames = []
            for number in range(count):
                image.seek(number)
                frames.append(image_bytes(image))
            return frames

    async def _docx(self, path: Path) -> list[ExtractedPage]:
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 100 * 1024 * 1024:
                raise BadRequestError("Expanded DOCX is too large; split this document.")
        document = await asyncio.to_thread(Document, str(path))
        blocks = []
        warnings = ["DOCX page boundaries, text boxes and floating-image positions are not reconstructed."]
        image_count = 0

        async def paragraph_blocks(paragraph: Paragraph) -> list[ExtractionBlock]:
            nonlocal image_count
            result = []
            if paragraph.text:
                style = paragraph.style.name if paragraph.style else ""
                heading = style.startswith("Heading ") and style[-1:].isdigit()
                result.append(ExtractionBlock(
                    kind="heading" if heading else "paragraph", text=paragraph.text,
                    level=min(int(style[-1]), 6) if heading else 1,
                ))
            for blip in paragraph._p.iter(qn("a:blip")):
                relationship = blip.get(qn("r:embed"))
                if not relationship:
                    warnings.append("Externally linked image was not fetched.")
                    continue
                image_count += 1
                self._check_count(image_count)
                blob = document.part.related_parts[relationship].blob
                with Image.open(io.BytesIO(blob)) as image:
                    png = await asyncio.to_thread(image_bytes, image)
                extracted = await self.ocr.extract(png, None)
                result.extend(extracted.blocks)
                warnings.extend(extracted.warnings)
            return result

        for child in document.element.body:
            if child.tag == qn("w:p"):
                blocks.extend(await paragraph_blocks(Paragraph(child, document)))
            elif child.tag == qn("w:tbl"):
                table = Table(child, document)
                rows = []
                for row in table.rows:
                    values = []
                    for cell in row.cells:
                        parts = []
                        for paragraph in cell.paragraphs:
                            parts.extend(await paragraph_blocks(paragraph))
                        values.append(page_text(ExtractedPage(blocks=parts)))
                    rows.append(values)
                blocks.append(ExtractionBlock(kind="table", rows=rows))
        return [ExtractedPage(blocks=blocks, warnings=list(dict.fromkeys(warnings)))]
