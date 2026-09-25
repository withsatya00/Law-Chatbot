"""OCR-capable manual ingestion for scanned (image-only) official PDFs.

`OfficialSourceSyncService.sync` cannot take a scanned Gazette/Ministry PDF:
its identity check reads the text layer, which is empty, so the source is held
on every run. This service is the supported manual path for those sources
(registry entries with `requires_ocr=True`, e.g. MeitY's IT (Amendment) Act,
2008 scan).

What it guarantees, and what it deliberately does not:
  - Provenance comes only from the official-source registry: the recorded
    `source_url`, `source_type` and `document_key` are the registry's, never
    caller-supplied, and the file's SHA-256 is recorded. A local file is
    accepted only in place of the download.
  - Identity is checked against the OCR text (registry `identity_tokens`); a
    document that does not identify itself is held and never indexed.
  - Every page's OCR confidence and character count is recorded, both in a
    manifest file and on the document/chunk metadata (`ocr_provenance`), so a
    reviewer can jump to the weak pages. Chunk-level `page_number` /
    `extraction_method` come from the normal loader.
  - The result is ALWAYS `verification_status="unverified"` /
    `review_status="needs_review"`: applicability is left `unknown`, and the
    ingest aborts if metadata normalisation ever yields anything else. OCR text
    can mis-read a section number, so only a human comparing it to the page
    images may verify it (admin review route). Nothing here calls
    machine-verification or the approval path.
  - The evidence pass (identity + confidence) and the indexing pass are two
    separate OCR runs of the same bytes: `IndexingPipeline` re-reads the file
    through the standard loader, which may use its own Gemini fallback for low
    confidence pages. The manifest therefore describes the evidence pass, and
    the reviewer must still check the indexed text itself.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import UploadFile
from pypdf import PdfReader

from app.core.config import settings
from app.rag.kb_jurisdiction import (
    PROVENANCE_MANUAL,
    REVIEW_NEEDS_REVIEW,
    VERIFICATION_UNVERIFIED,
    document_metadata_fields,
    normalize_jurisdiction,
)
from app.rag.loader import SCANNED_PAGE_CHAR_THRESHOLD
from app.rag.ocr import OcrUnavailableError, ocr_engine
from app.services.kb_official_source_sync import OfficialSource

log = structlog.get_logger(__name__)

INGESTION_ROUTE = "manual_ocr_scanned_official_pdf"
MAX_SCANNED_PAGES = 200
DEFAULT_EVIDENCE_DIR = Path("storage/kb_audit/scanned_ingestion")

STATUS_INDEXED_NEEDS_REVIEW = "indexed_needs_review"
STATUS_DRY_RUN_OK = "dry_run_ok"
STATUS_HELD = "held"
STATUS_REJECTED = "rejected"


@dataclass(frozen=True)
class OcrPageEvidence:
    page_number: int
    chars: int
    confidence: float | None  # Tesseract 0-100 mean word confidence; None = no recognised words

    def as_dict(self) -> dict[str, Any]:
        return {"page": self.page_number, "chars": self.chars, "confidence": self.confidence}


class ScannedOfficialPdfIngestion:
    def __init__(
        self, ingestion: Any = None, ocr: Any = None, propagate: Any = None,
        evidence_dir: Path | None = None,
    ) -> None:
        self._ingestion = ingestion
        self.ocr = ocr or ocr_engine
        self._propagate = propagate
        self.evidence_dir = evidence_dir or DEFAULT_EVIDENCE_DIR

    @property
    def ingestion(self) -> Any:
        if self._ingestion is None:
            # Lazy: constructing it wires up the malware scanner and repositories.
            from app.services.kb_ingestion_service import KnowledgeBaseIngestionService

            self._ingestion = KnowledgeBaseIngestionService()
        return self._ingestion

    async def ingest(
        self, source: OfficialSource, body: bytes, *, acquisition: str, dry_run: bool = False,
    ) -> dict[str, Any]:
        """`acquisition` says how `body` was obtained ("downloaded from the
        registry URL" or "operator-supplied local file"); it is recorded, not
        trusted."""
        result: dict[str, Any] = {
            "law": source.key, "url": source.url, "status": STATUS_REJECTED,
            "human_review_required": True,
        }
        if not source.requires_ocr:
            result["reason"] = "Source is text-extractable; use scripts/sync_official_kb_sources.py instead."
            return result
        if not body.startswith(b"%PDF"):
            result["reason"] = "Not a PDF."
            return result
        sha256 = hashlib.sha256(body).hexdigest()
        result["official_sha256"] = sha256
        try:
            reader = PdfReader(io.BytesIO(body))
            page_texts = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:  # noqa: BLE001 - unreadable PDF is a rejection, not a crash
            result["reason"] = f"PDF could not be parsed: {type(exc).__name__}."
            return result
        page_count = len(page_texts)
        text_layer_chars = sum(len(text.strip()) for text in page_texts)
        if not page_count or page_count > MAX_SCANNED_PAGES:
            result["reason"] = f"Page count {page_count} is outside 1-{MAX_SCANNED_PAGES}."
            return result
        if text_layer_chars / page_count >= SCANNED_PAGE_CHAR_THRESHOLD:
            result["reason"] = "PDF has a usable text layer; use scripts/sync_official_kb_sources.py instead."
            return result

        try:
            pages = await self._ocr_pages(body, page_count)
        except OcrUnavailableError as exc:
            result.update(status=STATUS_HELD, reason=f"OCR unavailable: {exc}")
            return result

        ocr_text = re.sub(r"[^a-z0-9]+", "", "\n".join(text for _, text, _ in pages).lower())
        evidence = self._evidence(source, sha256, acquisition, page_count, text_layer_chars, pages)
        matched = [token for token in source.identity_tokens if token in ocr_text]
        evidence["identity_tokens_matched"] = matched
        result["ocr"] = {
            key: evidence[key]
            for key in ("page_count", "mean_confidence", "min_page_confidence", "low_confidence_pages")
        }
        if not ocr_text or len(matched) != len(source.identity_tokens):
            result.update(
                status=STATUS_HELD,
                reason="OCR identity check failed: the scanned document did not identify itself as "
                f"{source.key} (matched {len(matched)}/{len(source.identity_tokens)} identity tokens).",
            )
            result["manifest"] = self._write_manifest(source, sha256, evidence, result)
            return result

        metadata = self._metadata(source, evidence)
        evidence["review_reasons"] = metadata["review_reasons"]
        if dry_run:
            result.update(status=STATUS_DRY_RUN_OK, review_status=metadata["review_status"])
            result["manifest"] = self._write_manifest(source, sha256, evidence, result)
            return result

        upload = UploadFile(file=io.BytesIO(body), filename=source.filename)
        response = await self.ingestion.ingest(upload, jurisdiction_metadata=metadata)
        result.update(
            generated_filename=response.generated_filename, document_id=response.document_id,
            chunks_indexed=response.chunks_indexed, review_status=response.review_status,
        )
        if response.status != "indexed":
            result.update(status=response.status, reason=getattr(response, "reason", None))
            result["manifest"] = self._write_manifest(source, sha256, evidence, result)
            return result
        if response.review_status != REVIEW_NEEDS_REVIEW:
            # Cannot happen through `_metadata`; if it ever does, surface it loudly.
            raise RuntimeError(f"Scanned source indexed as {response.review_status!r}, expected needs_review.")
        result["status"] = STATUS_INDEXED_NEEDS_REVIEW
        manifest = self._write_manifest(source, sha256, evidence, result)
        result["manifest"] = manifest
        evidence["manifest"] = manifest
        await self._record_on_kb(response, evidence)
        return result

    async def _ocr_pages(self, body: bytes, page_count: int) -> list[tuple[int, str, float | None]]:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(body)
            path = Path(handle.name)
        try:
            return await self.ocr.extract_pdf_pages_individually(path, max_pages=page_count)
        finally:
            path.unlink(missing_ok=True)

    def _evidence(
        self, source: OfficialSource, sha256: str, acquisition: str, page_count: int,
        text_layer_chars: int, pages: list[tuple[int, str, float | None]],
    ) -> dict[str, Any]:
        page_evidence = [OcrPageEvidence(number, len(text.strip()), confidence) for number, text, confidence in pages]
        confidences = [page.confidence for page in page_evidence if page.confidence is not None]
        threshold = settings.ocr_min_confidence
        low = [
            page.page_number for page in page_evidence
            if page.confidence is None or page.confidence < threshold
        ]
        return {
            "law": source.key, "document_key": source.document_key, "source_url": source.url,
            "source_type": source.source_type, "source_sha256": sha256, "acquisition": acquisition,
            "ingestion_route": INGESTION_ROUTE, "ocr_engine": "tesseract", "ocr_language": settings.ocr_language,
            "text_layer_chars": text_layer_chars, "page_count": page_count,
            "pages_recognised": len(pages),
            "mean_confidence": round(sum(confidences) / len(confidences), 2) if confidences else None,
            "min_page_confidence": round(min(confidences), 2) if confidences else None,
            "confidence_threshold": threshold, "low_confidence_pages": low,
            "pages": [page.as_dict() for page in page_evidence],
            "recorded_at": datetime.now(UTC).isoformat(), "human_review_required": True,
        }

    def _metadata(self, source: OfficialSource, evidence: dict[str, Any]) -> dict[str, Any]:
        """Registry-derived metadata, hard-wired to unverified/needs_review."""
        normalized = normalize_jurisdiction({
            "document_key": source.document_key,
            "issuing_level": "central",
            "applicability": "unknown",
            "jurisdiction_source_type": source.source_type,
            "source_url": source.url,
            "version_label": source.version_label,
            "verification_status": VERIFICATION_UNVERIFIED,
        }, provenance=PROVENANCE_MANUAL)
        metadata = document_metadata_fields(normalized)
        if metadata["review_status"] != REVIEW_NEEDS_REVIEW or metadata["verification_status"] != VERIFICATION_UNVERIFIED:
            raise RuntimeError("Scanned-source metadata must be unverified/needs_review.")
        metadata["review_reasons"] = [
            *metadata["review_reasons"],
            (
                "Scanned image-only source: the indexed text is OCR output and a human reviewer must check "
                "it (especially section numbers) against the page images before verifying."
            ),
        ]
        if evidence["low_confidence_pages"]:
            metadata["review_reasons"].append(
                f"OCR confidence below {evidence['confidence_threshold']} on pages "
                f"{evidence['low_confidence_pages']}.",
            )
        return metadata

    async def _record_on_kb(self, response: Any, evidence: dict[str, Any]) -> None:
        """Metadata-only write of the OCR evidence onto the document and its
        chunks (same helper the admin jurisdiction-correction route uses); it
        cannot alter `review_status` because it does not set it."""
        propagate = self._propagate
        if propagate is None:
            from app.database.mongodb import mongodb
            from app.models.collections import EMBEDDINGS_METADATA, UPLOADED_DOCUMENTS
            from app.rag.kb_jurisdiction import propagate_jurisdiction_metadata

            async def propagate(document_id: str, source_document: str, fields: dict[str, Any]) -> int:
                return await propagate_jurisdiction_metadata(
                    mongodb.db[UPLOADED_DOCUMENTS], mongodb.db[EMBEDDINGS_METADATA],
                    document_id=document_id, source_document=source_document, fields=fields,
                )
        await propagate(
            response.document_id, response.generated_filename,
            {"ingestion_route": INGESTION_ROUTE, "ocr_provenance": evidence},
        )

    def _write_manifest(self, source: OfficialSource, sha256: str, evidence: dict[str, Any], result: dict[str, Any]) -> str:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        path = self.evidence_dir / f"{source.key.lower()}_{sha256[:12]}.json"
        path.write_text(
            json.dumps({**evidence, "outcome": {k: v for k, v in result.items() if k != "manifest"}},
                       indent=2, default=str),
            encoding="utf-8",
        )
        return str(path)
