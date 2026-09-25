"""Admin Knowledge Base Ingestion Pipeline.

Separate from `DocumentService.upload_and_index` (`app/services/document_service.py`),
which is the general per-user chat-attachment upload -- those documents are
deliberately owner-scoped (Part 45/46) and never touched here. This service is
for an admin adding documents to the shared, globally-visible Knowledge Base:

    Admin Upload -> Validation -> Duplicate Detection -> Rename -> Indexing
        -> Knowledge Base Transfer -> Upload Cleanup

Reuses the existing `IndexingPipeline.index_file` (unmodified) for the actual
extraction/chunking/embedding work -- this module only adds what it doesn't
already do: content-fingerprint near-duplicate detection (on top of its
existing exact-hash check), content-derived renaming, and staging-directory
lifecycle management.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import UploadFile
from pypdf import PdfReader

from app.cache.response_cache import response_cache
from app.core.config import settings
from app.core.constants import ALLOWED_UPLOAD_EXTENSIONS
from app.core.exceptions import BadRequestError, NotFoundError
from app.database.mongodb import mongodb
from app.models.collections import KB_AUTOMATION_JOBS
from app.rag.kb_jurisdiction import (
    PROVENANCE_AUTOMATED_OFFICIAL,
    PROVENANCE_MANUAL,
    JurisdictionMetadataError,
    document_metadata_fields,
    normalize_jurisdiction,
    propagate_jurisdiction_metadata,
)
from app.rag.loader import DocumentLoader
from app.rag.pipeline import IndexingPipeline
from app.rag.quality import DocumentQualityChecker
from app.repositories.documents import DocumentRepository, EmbeddingMetadataRepository
from app.repositories.kb_staging import KnowledgeBaseStagingRepository
from app.repositories.versioning import DocumentVersionRepository
from app.schemas.admin import KnowledgeBaseUploadResponse
from app.utils.malware_scan import ClamAVScanner, MalwareScanner
from app.utils.upload_storage import write_upload

log = structlog.get_logger(__name__)

MAX_FINGERPRINT_WORDS = 500
FINGERPRINT_SIMILARITY_THRESHOLD = 0.95
MAX_FILENAME_LENGTH = 80
_STOPWORDS = {
    "this", "that", "with", "from", "shall", "under", "such", "have", "been",
    "which", "their", "these", "those", "into", "will", "than", "also",
}

# (pattern, label). First match wins. Only used to populate the response's
# `document_type` field -- pure regex, no LLM, matches rule 8's priority order.
DOCUMENT_TYPE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"cyber\s*crime|cyber\s*fraud|online\s*fraud|hacking", re.IGNORECASE), "Cyber Crime Complaint"),
    (re.compile(r"vehicle\s+theft|car\s+theft|bike\s+theft|stolen\s+vehicle", re.IGNORECASE), "Vehicle Theft Complaint"),
    (re.compile(r"right\s+to\s+information|\brti\b", re.IGNORECASE), "RTI Application"),
    (re.compile(r"first\s+information\s+report|\bfir\b", re.IGNORECASE), "FIR Complaint"),
    (re.compile(r"affidavit", re.IGNORECASE), "Affidavit"),
    (re.compile(r"legal\s+notice", re.IGNORECASE), "Legal Notice"),
    (re.compile(r"power\s+of\s+attorney", re.IGNORECASE), "Power Of Attorney"),
    (re.compile(r"petition", re.IGNORECASE), "Petition"),
    (re.compile(r"\bact\b|\bsanhita\b|\badhiniyam\b", re.IGNORECASE), "Act"),
    (re.compile(r"manual|guideline", re.IGNORECASE), "Guidelines"),
    (re.compile(r"agreement|contract", re.IGNORECASE), "Agreement"),
    (re.compile(r"complaint", re.IGNORECASE), "Complaint"),
]


@dataclass(frozen=True)
class StagedUpload:
    """Outcome of `stage`. `claimed` is False when identical content already
    had an active ingestion job, in which case the caller must NOT enqueue --
    the row returned is the terminal duplicate record for this attempt, not a
    second job for the same bytes."""

    staging_id: str
    staged_path: Path
    original_filename: str
    status: str
    claimed: bool
    reason: str | None = None


class KnowledgeBaseIngestionService:
    def __init__(self, scanner: MalwareScanner | None = None) -> None:
        self.loader = DocumentLoader()
        self.quality = DocumentQualityChecker()
        self.pipeline = IndexingPipeline()
        self.versions = DocumentVersionRepository()
        self.documents = DocumentRepository()
        self.chunks = EmbeddingMetadataRepository()
        self.staging = KnowledgeBaseStagingRepository()
        # Every other route that writes an untrusted upload to disk scans it
        # first -- `DocumentService.upload_and_index` (the end-user path) and
        # `KnowledgeBaseAutomationService` (the automated fetch pipeline, see
        # `app.utils.malware_scan`'s own docstring) both do. This admin manual
        # upload route (`/admin/knowledge-base/upload` -> `stage`/`ingest` ->
        # `_stage_upload`) was the one path with no scan at all -- an admin
        # session is still a session that can be compromised, and content
        # ingested here is served globally to every user of the shared
        # Knowledge Base, a strictly larger blast radius than a private
        # per-user chat attachment.
        self.scanner = scanner or ClamAVScanner()

    async def ingest(
        self, file: UploadFile, jurisdiction_metadata: dict[str, Any] | None = None,
    ) -> KnowledgeBaseUploadResponse:
        """Synchronous end-to-end ingest (validate, write, dedup, index, transfer).
        Used directly by callers that want to wait for the outcome; the admin
        upload route instead uses `stage` + the background queue's `process`
        so the request returns immediately (background indexing queue).

        `jurisdiction_metadata` is the already-validated flat dict from
        `kb_jurisdiction.document_metadata_fields` -- the caller (the admin
        route) is responsible for calling `normalize_jurisdiction` first, so
        this service never has to trust raw admin/user input directly.
        """
        staged_path, original_filename = await self._stage_upload(file)
        return await self._process(staged_path, original_filename, jurisdiction_metadata=jurisdiction_metadata)

    async def stage(
        self, file: UploadFile, jurisdiction_metadata: dict[str, Any] | None = None,
    ) -> StagedUpload:
        """Fast path: validate + write the upload to `kb_staging_dir`, hash it,
        and claim the single active ingestion slot for that content.

        The SHA-256 is computed *before* the ledger row is created, so a row
        never exists with `content_hash: None` -- that null was what made two
        submissions of identical bytes indistinguishable until indexing had
        already started on both. `claim_active` then makes "one active job per
        content" a Mongo-enforced invariant (sparse unique index on
        `active_key`), which holds across processes; a process-local
        `asyncio.Lock` would not have. A losing submission is not thrown away:
        its bytes are archived and a terminal `duplicate` row records the
        attempt, so history is preserved without a second active job.

        `jurisdiction_metadata` (already validated by the caller -- see
        `ingest` above) rides along on the ledger row so the background queue's
        `process()` -- which only ever receives `(staging_id, staged_path,
        original_filename)` -- can look it back up by `staging_id` rather than
        needing a second, parallel queue.
        """
        staged_path, original_filename = await self._stage_upload(file)
        content_hash = self.quality.hash_file(staged_path)
        staging_id = await self.staging.claim_active(
            self._ledger_fields(
                original_filename=original_filename,
                status="pending",
                content_hash=content_hash,
                current_path=staged_path,
                destination_path=settings.knowledge_base_dir,
                reason="Queued for background indexing.",
                jurisdiction_metadata=jurisdiction_metadata,
            )
        )
        if staging_id is None:
            active = await self.staging.find_active_by_content_hash(content_hash)
            active_id = str(active.get("_id")) if active else "an in-flight ingestion job"
            reason = f"Identical content already has an active ingestion job ({active_id})."
            archive_path = self._move(staged_path, settings.archive_dir, original_filename)
            duplicate_id = await self.staging.insert(
                self._ledger_fields(
                    original_filename=original_filename,
                    status="duplicate",
                    content_hash=content_hash,
                    current_path=archive_path,
                    destination_path=settings.archive_dir,
                    reason=reason,
                    ingestion_source="concurrent_submission",
                )
            )
            log.info("kb_stage_concurrent_duplicate", content_hash=content_hash, record_id=duplicate_id)
            return StagedUpload(duplicate_id, archive_path, original_filename, "duplicate", False, reason)
        return StagedUpload(staging_id, staged_path, original_filename, "pending", True, None)

    async def process(self, staging_id: str, staged_path: Path, original_filename: str) -> KnowledgeBaseUploadResponse:
        """Background-queue entry point: runs dedup/index/transfer for a file
        already staged by `stage`, updating that same ledger row in place."""
        return await self._process(staged_path, original_filename, staging_id)

    async def backfill_from_uploads(
        self, source_dir: Path, queue: Any = None, *, authorized: bool = False
    ) -> dict[str, Any]:
        """Explicit, admin-authorized quarantine of documents already sitting
        in `source_dir`.

        PRIVACY: `upload_storage_dir` holds users' own chat attachments. This
        used to copy them into the shared Knowledge Base and index them, and
        it ran automatically on every API start -- so a private upload could
        become globally retrievable with nobody deciding that it should. It
        now (a) refuses to run without `authorized=True` from an admin-only
        route, and (b) never indexes anything: every candidate is copied into
        `kb_review_dir/pending` with status `needs_review`, and only an
        admin's explicit `approve_needs_review` can promote one. The original
        in `source_dir` is never moved or deleted.
        """
        if not authorized:
            raise BadRequestError(
                "Backfilling from uploads promotes private user documents and requires explicit admin "
                "authorization (confirm=true)."
            )

        review_pending_dir = self.review_pending_dir()
        review_pending_dir.mkdir(parents=True, exist_ok=True)

        candidates = (
            sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
            if source_dir.exists()
            else []
        )

        queued_for_review = 0
        skipped = 0
        for source_path in candidates:
            content_hash = self.quality.hash_file(source_path)
            if await self.staging.find_by_content_hash(content_hash) is not None:
                skipped += 1
                continue
            review_path = self._reserve_path(review_pending_dir, source_path.name)
            shutil.copy2(source_path, review_path)
            await self.staging.insert(
                self._ledger_fields(
                    original_filename=source_path.name,
                    status="needs_review",
                    content_hash=content_hash,
                    current_path=review_path,
                    destination_path=None,
                    reason=(
                        "Unverified provenance: copied from the private uploads directory. An admin must "
                        "approve this document before it can enter the shared Knowledge Base."
                    ),
                    ingestion_source="upload_backfill",
                )
            )
            queued_for_review += 1

        log.info(
            "kb_uploads_backfill_quarantined",
            source_dir=str(source_dir),
            files_scanned=len(candidates),
            needs_review=queued_for_review,
            skipped_known=skipped,
        )
        return {
            "files_processed": len(candidates),
            "needs_review_count": queued_for_review,
            "skipped_known_count": skipped,
            "indexed_count": 0,
            "duplicate_count": 0,
            "failed_count": 0,
        }

    # ---------------------------------------------------------------- review

    async def update_jurisdiction_metadata(self, document_id: str, raw_jurisdiction: dict[str, Any]) -> dict[str, Any]:
        """Phase 1 gap 2: corrects/publishes an ALREADY-INDEXED document's
        jurisdiction metadata (wrong State, wrong effective date, an
        `issuing_level` that was left `unknown`, ...) without a full
        re-upload -- no re-embedding, and never routed through
        `IndexingPipeline.index_file`, so `DocumentQualityChecker`'s
        duplicate-content-hash check (which would reject this as a
        "duplicate" of the very document being corrected) is never in the
        picture at all.

        `raw_jurisdiction` REPLACES the document's jurisdiction metadata
        wholesale (the same full-object shape the upload route's
        `jurisdiction_metadata` form field takes), not a partial patch --
        an admin correcting one field re-sends the complete, corrected
        picture, so there is never an ambiguous "which old fields survive"
        question. Supplying `verification_status: "verified"` (with
        `verified_by`) alongside complete, known `issuing_level`/
        `applicability`/`source_url` is what actually PUBLISHES the
        document (flips `review_status` to `approved`); leaving any of
        those unknown keeps it `needs_review`.
        """
        document = await self.documents.find_by_id(document_id)
        if document is None:
            raise NotFoundError(f"Knowledge Base document '{document_id}' not found.")
        if document.get("owner_session_id") or document.get("owner_user_id"):
            # Jurisdiction review is a shared-Knowledge-Base concept only --
            # see `kb_jurisdiction.propagate_jurisdiction_metadata`'s own
            # docstring and Part 45/46 for why a private, owner-scoped
            # document must never be touched by this path.
            raise BadRequestError("Cannot set jurisdiction metadata on a private, owner-scoped document.")
        try:
            normalized = normalize_jurisdiction(raw_jurisdiction, provenance=PROVENANCE_MANUAL)
        except JurisdictionMetadataError as exc:
            raise BadRequestError("Invalid jurisdiction metadata.", {"issues": exc.errors}) from exc

        fields = document_metadata_fields(normalized)
        source_document = document.get("filename")
        chunks_updated = await propagate_jurisdiction_metadata(
            self.documents.collection, self.chunks.collection,
            document_id=document_id, source_document=source_document,
            fields=fields, section_overrides=normalized.section_overrides,
        )
        # Gap 3 fix: invalidates both the final-answer cache AND the raw
        # retrieval-results cache in the same instant (shared generation
        # counter -- see `app.cache.semantic_cache.SemanticCache`) so a
        # newly-approved (or newly-withdrawn) document's chunks are neither
        # served stale-excluded nor stale-included for the rest of either
        # cache's TTL.
        await response_cache.bump_generation()
        log.info(
            "kb_jurisdiction_metadata_updated",
            document_id=document_id, source_document=source_document,
            review_status=normalized.metadata["review_status"], chunks_updated=chunks_updated,
        )
        return {
            "document_id": document_id,
            "source_document": source_document,
            "chunks_updated": chunks_updated,
            "review_status": normalized.metadata["review_status"],
            "review_reasons": normalized.metadata["review_reasons"],
        }

    async def list_automation_review_queue(
        self, limit: int = 50, jurisdiction_codes: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        """A READ-ONLY view of documents an automation adapter discovered and
        indexed (`kb_automation_jobs.status == "quarantined"`, the state
        `KnowledgeBaseAutomationService._process_claimed` leaves a job in
        once it has a real `document_id`), for a human reviewer to work
        through with `bulk_review_automated_documents`. It approves nothing.

        These documents already have their structural jurisdiction fields
        (`issuing_level`/`applicability`/`applicable_state_codes`/
        `source_url`/...) set correctly by `_process_claimed` from the real
        adapter/candidate data -- not a guess, the actual portal the
        automation runtime fetched this exact file from. The only thing
        genuinely missing is a human looking at the document and deciding
        `verification_status`. This view surfaces exactly the fields a
        reviewer needs to make that call quickly (title, official URL,
        jurisdiction, act number/year) without opening each job record.
        """
        query: dict[str, Any] = {"status": "quarantined", "document_id": {"$exists": True}}
        if jurisdiction_codes:
            query["candidate.jurisdiction_code"] = {"$in": list(jurisdiction_codes)}
        cursor = mongodb.db[KB_AUTOMATION_JOBS].find(query).sort("updated_at", -1).limit(limit)
        items: list[dict[str, Any]] = []
        async for job in cursor:
            candidate = job.get("candidate") or {}
            document = await self.documents.find_by_id(job["document_id"])
            metadata = (document or {}).get("metadata") or {}
            items.append({
                "document_id": job["document_id"],
                "job_id": job["_id"],
                "title": candidate.get("title"),
                "jurisdiction_code": candidate.get("jurisdiction_code"),
                "applicable_state_codes": metadata.get("applicable_state_codes") or [],
                "document_type": candidate.get("document_type"),
                "act_number": candidate.get("act_number"),
                "enactment_year": candidate.get("enactment_year"),
                "source_url": metadata.get("source_url") or candidate.get("url"),
                "issuing_level": metadata.get("issuing_level"),
                "applicability": metadata.get("applicability"),
                "confidence_score": candidate.get("confidence_score"),
                "updated_at": job.get("updated_at"),
            })
        return items

    async def bulk_review_automated_documents(
        self, document_ids: list[str], verified_by: str,
    ) -> list[dict[str, Any]]:
        """Speeds up human review of automation-discovered documents whose
        structural jurisdiction fields were already set correctly at
        discovery time (see `list_automation_review_queue`'s docstring) --
        the only thing actually missing per document is a human's yes/no on
        `verification_status`. This is NOT a bulk auto-approve: a reviewer
        still looks at and is individually responsible for every
        `document_id` they choose to include; this only removes the need to
        hand-type the full jurisdiction object per document through
        `update_jurisdiction_metadata`, which this method calls unchanged
        for each one (same validation, same `PROVENANCE_MANUAL`-only path to
        `verified`, same `_review_reasons` gate) -- a document whose
        structural fields are still incomplete stays `needs_review` exactly
        as it would through the single-document endpoint, not silently
        approved because it was in a batch.
        """
        if not verified_by.strip():
            raise BadRequestError("verified_by is required for a bulk review.")
        results: list[dict[str, Any]] = []
        for document_id in document_ids:
            try:
                document = await self.documents.find_by_id(document_id)
                if document is None:
                    results.append({"document_id": document_id, "outcome": "failed", "reason": "Document not found."})
                    continue
                existing = dict(document.get("metadata") or {})
                # `metadata_provenance` MUST NOT carry forward from the
                # existing record: it defaults to `"automated_official"` for
                # every automation-discovered document, and
                # `normalize_jurisdiction` reads it straight off this dict
                # (`raw.get("metadata_provenance", provenance)`) IN
                # PREFERENCE to the `provenance=PROVENANCE_MANUAL` argument
                # `update_jurisdiction_metadata` passes -- so leaving it in
                # silently downgrades `verification_status: "verified"` to
                # `"inferred"` (`_resolve_verification`'s `provenance !=
                # PROVENANCE_MANUAL` rule) and the document never actually
                # becomes `approved`, no matter how many times a reviewer
                # submits it. Confirmed live 2026-09-22 against 4 real
                # documents stuck exactly this way. This call IS the human
                # review action, so its own provenance is manual by
                # definition -- `update_jurisdiction_metadata`'s explicit
                # `provenance=PROVENANCE_MANUAL` should be the one that wins.
                existing.pop("metadata_provenance", None)
                raw_jurisdiction = {
                    **existing, "verification_status": "verified", "verified_by": verified_by.strip(),
                }
                outcome = await self.update_jurisdiction_metadata(document_id, raw_jurisdiction)
                results.append({
                    "document_id": document_id, "outcome": outcome["review_status"],
                    "review_reasons": outcome["review_reasons"], "chunks_updated": outcome["chunks_updated"],
                })
            except (NotFoundError, BadRequestError) as exc:
                results.append({"document_id": document_id, "outcome": "failed", "reason": str(exc)})
        return results

    async def list_needs_review_documents(self, limit: int = 200) -> list[dict[str, Any]]:
        """Live, authoritative `needs_review` documents, queried directly from
        `uploaded_documents` -- NOT the staging ledger
        (`admin_operations.staging_records`), whose `review_status` field is
        a write-once snapshot taken at indexing time
        (`kb_ingestion_service.py`'s own `process()`) that never updates
        after a later approval. Confirmed live 2026-09-22: the staging ledger
        reported 2,174 `needs_review` rows against only 21 real ones -- most
        of that gap was documents approved sometime after they were first
        indexed, whose staging record was simply never told. This is the
        query the Streamlit review page should use instead.
        """
        cursor = self.documents.collection.find(
            {"metadata.review_status": "needs_review"},
        ).sort("created_at", -1).limit(limit)
        results: list[dict[str, Any]] = []
        async for document in cursor:
            results.append({
                "document_id": str(document["_id"]),
                "filename": document.get("filename"),
                "jurisdiction_metadata": dict(document.get("metadata") or {}),
            })
        return results

    async def update_machine_verified_metadata(
        self, document_id: str, raw_jurisdiction: dict[str, Any],
        *, excluded_text_patterns: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Publish only metadata carrying the strict automated evidence bundle.

        Kept separate from the admin/human path so an automated caller can
        never acquire `verified` merely by choosing a request field.
        """
        document = await self.documents.find_by_id(document_id)
        if document is None:
            raise NotFoundError(f"Knowledge Base document '{document_id}' not found.")
        metadata = document.get("metadata") or {}
        if any(document.get(key) or metadata.get(key) for key in ("owner_session_id", "owner_user_id")):
            raise BadRequestError("Private documents cannot be machine-published.")
        try:
            normalized = normalize_jurisdiction(
                raw_jurisdiction, provenance=PROVENANCE_AUTOMATED_OFFICIAL
            )
        except JurisdictionMetadataError as exc:
            raise BadRequestError("Invalid machine-verification metadata.", {"issues": exc.errors}) from exc
        fields = document_metadata_fields(normalized)
        if fields.get("verification_status") != "machine_verified" or fields.get("review_status") != "approved":
            raise BadRequestError("Machine-verification evidence did not qualify for publication.")
        source_document = document.get("filename")
        if excluded_text_patterns:
            for pattern in excluded_text_patterns:
                matches = await self.chunks.collection.count_documents({
                    "metadata.source_document": source_document,
                    "text": {"$regex": pattern, "$options": "i"},
                })
                if matches <= 0:
                    raise BadRequestError("Required uncommenced-provision exclusion matched no chunks.")
        chunks_updated = await propagate_jurisdiction_metadata(
            self.documents.collection, self.chunks.collection,
            document_id=document_id, source_document=source_document,
            fields=fields, section_overrides=normalized.section_overrides,
        )
        if chunks_updated <= 0:
            raise BadRequestError("Machine publication produced no searchable chunks.")
        excluded_chunks = 0
        for pattern in excluded_text_patterns:
            excluded = await self.chunks.collection.update_many(
                {"metadata.source_document": source_document, "text": {"$regex": pattern, "$options": "i"}},
                {"$set": {"metadata.in_force": False, "metadata.machine_exclusion_reason": "uncommenced provision"}},
            )
            excluded_chunks += excluded.modified_count
        await response_cache.bump_generation(strict=True)
        return {
            "document_id": document_id, "source_document": document.get("filename"),
            "chunks_updated": chunks_updated, "review_status": fields["review_status"],
            "verification_status": fields["verification_status"], "excluded_chunks": excluded_chunks,
        }

    async def approve_needs_review(self, staging_id: str) -> dict[str, Any]:
        """The only path by which a `needs_review` document reaches the shared
        Knowledge Base. Moves the file back into staging, re-claims the active
        slot and enqueues it, so it runs through the exact same dedup/index
        pipeline as an admin upload."""
        record = await self._require_record(staging_id)
        if record.get("status") != "needs_review":
            raise BadRequestError(
                f"Only needs_review documents can be approved (this one is '{record.get('status')}')."
            )
        return await self._requeue(record, "Approved by admin for indexing.")

    async def retry_failed(self, staging_id: str) -> dict[str, Any]:
        """Re-attempts one previously failed document (e.g. after the source
        file was repaired in the failed-review directory)."""
        record = await self._require_record(staging_id)
        if record.get("status") != "failed":
            raise BadRequestError(f"Only failed documents can be retried (this one is '{record.get('status')}').")
        return await self._requeue(record, "Retry requested by admin.")

    async def archive_rejected(self, staging_id: str, reason: str | None = None) -> dict[str, Any]:
        """Admin rejection: the document is moved to the archive (never
        deleted) and its ledger row closed."""
        record = await self._require_record(staging_id)
        current = self._record_path(record)
        archived_path = (
            self._move(current, settings.archive_dir, record.get("original_filename") or current.name)
            if current is not None and current.exists()
            else None
        )
        await self.staging.release_active(
            staging_id,
            self._ledger_fields(
                original_filename=record.get("original_filename") or "document",
                generated_filename=record.get("generated_filename"),
                status="duplicate",
                content_hash=record.get("content_hash"),
                current_path=archived_path,
                destination_path=settings.archive_dir,
                reason=reason or "Rejected by admin; archived rather than deleted.",
                previous_status=record.get("status"),
            ),
        )
        return {
            "staging_id": staging_id,
            "status": "duplicate",
            "archived_path": str(archived_path) if archived_path else None,
        }

    async def _requeue(self, record: dict[str, Any], reason: str) -> dict[str, Any]:
        from app.services.kb_indexing_queue import kb_indexing_queue as default_queue

        current = self._record_path(record)
        if current is None or not current.exists():
            raise BadRequestError("The document's file is missing; it cannot be indexed.")
        content_hash = record.get("content_hash") or self.quality.hash_file(current)
        if await self.staging.find_active_by_content_hash(content_hash) is not None:
            raise BadRequestError("Identical content already has an active ingestion job.")
        settings.kb_staging_dir.mkdir(parents=True, exist_ok=True)
        staged_path = self._move(current, settings.kb_staging_dir, current.name)
        original_filename = record.get("original_filename") or staged_path.name
        staging_id = str(record["_id"])
        await self.staging.update_by_id(
            staging_id,
            self._ledger_fields(
                original_filename=original_filename,
                generated_filename=record.get("generated_filename"),
                status="pending",
                content_hash=content_hash,
                current_path=staged_path,
                destination_path=settings.knowledge_base_dir,
                reason=reason,
                previous_status=record.get("status"),
                active_key=content_hash,
            ),
        )
        default_queue.enqueue(staging_id, staged_path, original_filename)
        return {"staging_id": staging_id, "status": "pending", "current_path": str(staged_path), "reason": reason}

    async def close_path_missing(self, staging_id: str, reason: str) -> dict[str, Any]:
        """Admin closes a `path_missing` finding, stating why. The reason is
        required: "the file is gone" is a decision someone has to own, not a
        row that quietly disappears."""
        if not reason or not reason.strip():
            raise BadRequestError("A reason is required to close a missing-file record.")
        record = await self._require_record(staging_id)
        current = self._record_path(record)
        # Older stale-job rows predate the reconciliation `path_missing`
        # marker. The admin listing can still prove they are missing from the
        # recorded path (`file_exists=False`), so allow those rows to be
        # closed too. Never trust the UI flag alone: a file that currently
        # exists on disk must remain impossible to close through this route.
        actually_missing = current is None or not current.is_file()
        if not record.get("path_missing") and not actually_missing:
            raise BadRequestError("This record is not flagged as missing a file.")
        await self.staging.update_by_id(
            staging_id,
            {
                "resolution": "closed_missing_file",
                "resolution_reason": reason.strip(),
                "path_missing": False,
                "file_exists": False,
            },
        )
        return {"staging_id": staging_id, "resolution": "closed_missing_file", "reason": reason.strip()}

    async def resource_path_missing(self, staging_id: str, replacement_path: Path) -> dict[str, Any]:
        """Re-sources a `path_missing` record from a replacement the admin
        supplies. Deliberately does NOT go looking in `storage/uploads` or the
        Knowledge Base for something that looks similar -- guessing which file
        was meant is how a private document ends up in the shared corpus."""
        record = await self._require_record(staging_id)
        if not record.get("path_missing"):
            raise BadRequestError("This record is not flagged as missing a file.")
        if not replacement_path.exists() or not replacement_path.is_file():
            raise BadRequestError(f"Replacement file '{replacement_path}' does not exist.")
        settings.kb_staging_dir.mkdir(parents=True, exist_ok=True)
        staged_path = self._reserve_path(settings.kb_staging_dir, replacement_path.name)
        shutil.copy2(replacement_path, staged_path)
        content_hash = self.quality.hash_file(staged_path)
        await self.staging.update_by_id(
            staging_id,
            self._ledger_fields(
                original_filename=record.get("original_filename") or replacement_path.name,
                status="needs_review",
                content_hash=content_hash,
                current_path=staged_path,
                destination_path=None,
                reason="Re-sourced from an admin-supplied replacement file; awaiting approval.",
                previous_status=record.get("status"),
                path_missing=False,
                resolution="resourced",
            ),
        )
        # Still `needs_review`: supplying bytes is not the same as approving
        # them for the shared Knowledge Base.
        return {"staging_id": staging_id, "status": "needs_review", "current_path": str(staged_path)}

    # ------------------------------------------------------ manifest auditing

    def manifest_dir(self) -> Path:
        return settings.operations_output_dir / "kb_reconciliation"

    def list_manifests(self) -> list[dict[str, Any]]:
        """Read-only index of reconciliation manifests, newest first."""
        directory = self.manifest_dir()
        if not directory.exists():
            return []
        return [
            {"name": path.name, "size_bytes": path.stat().st_size, "moves": len(json.loads(path.read_text(encoding="utf-8")).get("moves", []))}
            for path in sorted(directory.glob("*.json"), reverse=True)
        ]

    def read_manifest(self, name: str) -> dict[str, Any]:
        """One manifest, flattened to per-physical-file audit rows.

        The manifest -- not the ledger -- is the per-file record: byte-identical
        staging copies share a content hash and therefore one ledger row, and
        inventing extra Mongo rows to paper over that would fabricate history.
        Every physical file that moved appears here individually.
        """
        path = self.manifest_dir() / Path(name).name
        if not path.exists():
            raise NotFoundError(f"Reconciliation manifest '{name}' not found.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        operations = [
            {
                "operation_timestamp": manifest.get("completed_at") or path.stem.replace("reconciliation_", ""),
                "original_path": item.get("source"),
                "destination": item.get("destination") or item.get("proposed_destination"),
                "content_hash": item.get("content_hash"),
                "previous_status": item.get("previous_status"),
                "reason_code": item.get("reason_code"),
                "outcome": item.get("outcome"),
            }
            for item in manifest.get("moves", [])
        ]
        counts: dict[str, int] = {}
        for operation in operations:
            counts[str(operation["reason_code"])] = counts.get(str(operation["reason_code"]), 0) + 1
        return {
            "manifest": path.name,
            "operation_count": len(operations),
            "reason_code_counts": counts,
            "deleted": 0,
            "auto_indexed": 0,
            "ledger_missing_file": manifest.get("ledger_missing_file", 0),
            "operations": operations,
        }

    # -------------------------------------------------------- reconciliation

    async def sweep_stale_active(self) -> int:
        """No record may stay `pending`/`processing` when its file is gone --
        an interrupted process used to leave rows there permanently, invisible
        to the queue (nothing to requeue) and to the admin (not `failed`)."""
        swept = 0
        for record in await self.staging.find_stale_active():
            await self.staging.release_active(
                str(record["_id"]),
                {
                    "status": "needs_review",
                    "previous_status": record.get("status"),
                    "file_exists": False,
                    "reason": "Stale job: the staged file is missing, so indexing can never complete.",
                },
            )
            swept += 1
        if swept:
            log.warning("kb_stale_active_records_swept", count=swept)
        return swept

    async def reconcile_staging(
        self, *, apply: bool = False, expected_total: int | None = None
    ) -> dict[str, Any]:
        """Reconciles `kb_staging_dir` (disk) against `kb_staging_records`
        (the ledger) -- DRY RUN unless `apply=True`.

        The dry run reads the directory and Mongo and changes neither; it
        reports every proposed move with its source, destination and reason so
        an admin can check the plan before anything is touched. `apply=True`
        first writes a JSON manifest (hashes, old paths, proposed
        destinations, previous statuses) under `operations_output_dir` so
        every move stays reversible by hand, then executes the plan:

        - content already byte-identical to an indexed Knowledge Base document
          -> `archive_dir`, ledger row `duplicate`;
        - unreadable/corrupt, or a tracked `failed` row -> `kb_review/failed`;
        - anything else (orphaned, untracked, unverified provenance) ->
          `kb_review/pending`, ledger row `needs_review`, deliberately NOT
          indexed: embedding a batch of unknown documents nobody has reviewed
          is exactly the "silently do the expensive thing" this pipeline
          avoids elsewhere.

        Nothing is ever deleted, no existing Knowledge Base file is touched,
        and re-running after an apply is a no-op (the staging directory holds
        no terminal files any more), so the operation is idempotent.
        """
        stale_processing = await self.sweep_stale_active() if apply else len(await self.staging.find_stale_active())

        staging_dir = settings.kb_staging_dir
        files = sorted(p for p in staging_dir.iterdir() if p.is_file()) if staging_dir.exists() else []

        records_by_hash: dict[str, dict[str, Any]] = {}
        tracked_paths: dict[str, dict[str, Any]] = {}
        missing_ledger_paths = 0
        for record in await self.staging.find_by_statuses(self.staging.STATUSES):
            if record.get("content_hash"):
                records_by_hash.setdefault(str(record["content_hash"]), record)
            path = record.get("current_path") or record.get("staged_path")
            if path:
                tracked_paths[str(Path(path).resolve())] = record
                if not Path(path).exists():
                    missing_ledger_paths += 1

        kb_hashes = self._knowledge_base_hashes()

        plan: list[dict[str, Any]] = []
        for file_path in files:
            content_hash = self.quality.hash_file(file_path)
            match = tracked_paths.get(str(file_path.resolve())) or records_by_hash.get(content_hash)
            record_status = match.get("status") if match else None
            already_indexed = content_hash in kb_hashes or record_status == "indexed"
            corrupt_reason = None if already_indexed else self._corruption_reason(file_path)

            # Order matters: what the file IS now outranks what the ledger
            # once said about it. A row marked `failed` whose PDF still opens
            # cleanly is a candidate for review, not a corpse -- classifying
            # every tracked failure as corrupt would have quarantined five
            # perfectly readable documents as unrecoverable.
            if already_indexed:
                outcome, destination_dir = "duplicate", settings.archive_dir
                reason_code = "already_indexed_exact_hash"
                reason = "Content is byte-for-byte identical to an already indexed Knowledge Base document."
            elif corrupt_reason is not None:
                outcome, destination_dir = "failed", self.review_failed_dir()
                reason_code = "corrupt_pdf"
                reason = corrupt_reason
            elif record_status == "failed":
                outcome, destination_dir = "needs_review", self.review_pending_dir()
                reason_code = "readable_previous_failure"
                reason = (
                    "Readable file with a failed ingestion history "
                    f"({match.get('reason') if match else 'no recorded reason'}). "
                    "Requires explicit admin approval before indexing."
                )
            else:
                outcome, destination_dir = "needs_review", self.review_pending_dir()
                reason_code = "unverified_provenance"
                reason = (
                    "Untracked or unverified staging file: no Knowledge Base match and no completed ingestion. "
                    "Requires admin approval before indexing."
                )

            ledger_path = match.get("current_path") or match.get("staged_path") if match else None
            plan.append(
                {
                    "source": str(file_path),
                    "destination_dir": str(destination_dir),
                    "proposed_destination": str(destination_dir / file_path.name),
                    "outcome": outcome,
                    "reason_code": reason_code,
                    "reason": reason,
                    "content_hash": content_hash,
                    "staging_id": str(match["_id"]) if match else None,
                    "previous_status": record_status,
                    "orphaned": match is None,
                    "ledger_path_mismatch": bool(
                        match is not None
                        and (ledger_path is None or str(Path(ledger_path).resolve()) != str(file_path.resolve()))
                    ),
                }
            )

        # Rows that expect a live file but have none. They are ledger
        # findings, not physical work: nothing is moved, invented or indexed
        # for them, and they are never added to the physical file totals.
        ledger_missing = [
            {
                "staging_id": str(record["_id"]),
                "original_filename": record.get("original_filename"),
                "recorded_path": record.get("current_path") or record.get("staged_path"),
                "previous_status": record.get("status"),
                "reason_code": "ledger_file_missing",
                "reason": "The ledger records a file at this path but nothing is there.",
            }
            for record in await self.staging.find_by_statuses(
                (*self.staging.ACTIVE_STATUSES, "needs_review")
            )
            if not (record.get("current_path") or record.get("staged_path"))
            or not Path(str(record.get("current_path") or record.get("staged_path"))).exists()
        ]

        summary: dict[str, Any] = {
            "mode": "apply" if apply else "dry_run",
            # Physical files only. `ledger_missing_file` is deliberately NOT
            # added to this: 68 files scanned means 68 files on disk.
            "scanned": len(files),
            "archive": sum(1 for item in plan if item["outcome"] == "duplicate"),
            "failed_review": sum(1 for item in plan if item["outcome"] == "failed"),
            "delete": 0,
            "auto_index": 0,
            "ledger_missing_file": len(ledger_missing),
            "ledger_missing": ledger_missing,
            "files_scanned": len(files),
            "already_indexed": sum(1 for item in plan if item["outcome"] == "duplicate"),
            "duplicates": sum(1 for item in plan if item["outcome"] == "duplicate"),
            "failed_or_corrupt": sum(1 for item in plan if item["outcome"] == "failed"),
            "orphaned": sum(1 for item in plan if item["orphaned"]),
            "needs_review": sum(1 for item in plan if item["outcome"] == "needs_review"),
            "missing_ledger_paths": missing_ledger_paths + sum(1 for item in plan if item["ledger_path_mismatch"]),
            "stale_processing_records": stale_processing,
            "moves": plan,
            "manifest_path": None,
            "applied": False,
        }

        if not apply:
            return summary

        if expected_total is not None and expected_total != len(files):
            raise BadRequestError(f"Refusing to apply: expected {expected_total} staging files, found {len(files)}.")

        summary["manifest_path"] = str(self._write_manifest(summary))
        applied = 0
        for item in plan:
            source = Path(item["source"])
            if not source.exists():
                continue
            destination = self._move(source, Path(item["destination_dir"]), source.name)
            item["destination"] = str(destination)
            await self._record_reconciliation(item, destination)
            applied += 1
        for finding in ledger_missing:
            await self.staging.release_active(
                str(finding["staging_id"]),
                {
                    "status": "needs_review",
                    "previous_status": finding["previous_status"],
                    "file_exists": False,
                    "path_missing": True,
                    "reason_code": "ledger_file_missing",
                    "reason": finding["reason"],
                },
            )
        summary["applied"] = True
        summary["moved"] = applied
        log.info("kb_staging_reconciliation_applied", moved=applied, manifest=summary["manifest_path"])
        return summary

    async def apply_manifest(self, manifest_path: Path) -> dict[str, Any]:
        """Applies a plan produced by an earlier dry run. Entries whose source
        no longer exists are skipped, so replaying a manifest is idempotent."""
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        applied = 0
        skipped = 0
        for item in manifest.get("moves", []):
            source = Path(item["source"])
            if not source.exists():
                skipped += 1
                continue
            destination = self._move(source, Path(item["destination_dir"]), source.name)
            await self._record_reconciliation(item, destination)
            applied += 1
        log.info("kb_staging_manifest_applied", manifest=str(manifest_path), moved=applied, skipped=skipped)
        return {"manifest_path": str(manifest_path), "moved": applied, "skipped": skipped, "applied": True}

    async def _record_reconciliation(self, item: dict[str, Any], destination: Path) -> None:
        fields = self._ledger_fields(
            original_filename=Path(item["source"]).name,
            status=item["outcome"],
            content_hash=item["content_hash"],
            current_path=destination,
            destination_path=Path(item["destination_dir"]),
            reason=item["reason"],
            reason_code=item.get("reason_code"),
            previous_status=item.get("previous_status"),
            ingestion_source="staging_reconciliation",
        )
        if item.get("staging_id"):
            await self.staging.release_active(item["staging_id"], fields)
        else:
            await self.staging.insert(fields)

    def _write_manifest(self, summary: dict[str, Any]) -> Path:
        directory = settings.operations_output_dir / "kb_reconciliation"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"reconciliation_{stamp}.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return path

    def _knowledge_base_hashes(self) -> set[str]:
        if not settings.knowledge_base_dir.exists():
            return set()
        return {self.quality.hash_file(path) for path in settings.knowledge_base_dir.rglob("*") if path.is_file()}

    def _corruption_reason(self, path: Path) -> str | None:
        """Cheap readability probe -- exactly the failure the tracked rows
        already report ("Stream has ended unexpectedly"). Only PDFs are
        probed; other types are left to the indexing pipeline."""
        if path.suffix.lower() != ".pdf":
            return None
        try:
            reader = PdfReader(str(path))
            if not reader.pages:
                return "Corrupt PDF: no readable pages."
        except Exception as exc:  # noqa: BLE001 - any parser error means this file cannot be indexed
            return f"Corrupt PDF: {exc}"
        return None

    # --------------------------------------------------------------- helpers

    def review_pending_dir(self) -> Path:
        return settings.kb_review_dir / "pending"

    def review_failed_dir(self) -> Path:
        return settings.kb_review_dir / "failed"

    async def _require_record(self, staging_id: str) -> dict[str, Any]:
        record = await self.staging.find_by_id(staging_id)
        if record is None:
            raise NotFoundError(f"Staging record '{staging_id}' not found.")
        return record

    def _record_path(self, record: dict[str, Any]) -> Path | None:
        path = record.get("current_path") or record.get("staged_path") or record.get("archived_path")
        return Path(path) if path else None

    def _move(self, source: Path, destination_dir: Path, filename: str) -> Path:
        """Every move in this service goes through here: the destination is
        created, the name is collision-reserved (never overwriting an existing
        file), and the file is moved -- never deleted."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = self._reserve_path(destination_dir, filename)
        shutil.move(str(source), str(destination))
        return destination

    def _ledger_fields(
        self,
        *,
        original_filename: str,
        status: str,
        content_hash: str | None,
        current_path: Path | None,
        destination_path: Path | None,
        reason: str | None = None,
        generated_filename: str | None = None,
        document_id: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """The one shape every ledger row has, so an admin view can rely on
        the same keys being present whatever produced the row."""
        fields: dict[str, Any] = {
            "original_filename": original_filename,
            "generated_filename": generated_filename,
            "content_hash": content_hash,
            "status": status,
            "reason": reason,
            "current_path": str(current_path) if current_path is not None else None,
            "destination_path": str(destination_path) if destination_path is not None else None,
            "file_exists": bool(current_path is not None and Path(current_path).exists()),
            "document_id": document_id,
        }
        # `staged_path` is kept in step with `current_path`: startup recovery
        # and older rows read it, and those two silently diverging is what left
        # 64 of 68 physical files unmatchable by their ledger path.
        if current_path is not None:
            fields["staged_path"] = str(current_path)
        fields.update(extra)
        return fields

    async def _stage_upload(self, file: UploadFile) -> tuple[Path, str]:
        original_filename = Path(file.filename or "document").name
        suffix = Path(original_filename).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            raise BadRequestError(f"Unsupported upload type: {suffix}")

        settings.kb_staging_dir.mkdir(parents=True, exist_ok=True)
        settings.knowledge_base_dir.mkdir(parents=True, exist_ok=True)

        staged_path = self._reserve_path(settings.kb_staging_dir, original_filename)
        await self._write_upload(file, staged_path)
        try:
            await self.scanner.scan(staged_path)
        except ValueError as exc:
            staged_path.unlink(missing_ok=True)
            raise BadRequestError(f"Upload rejected: {exc}") from exc
        return staged_path, original_filename

    async def _process(
        self,
        staged_path: Path,
        original_filename: str,
        staging_id: str | None = None,
        jurisdiction_metadata: dict[str, Any] | None = None,
    ) -> KnowledgeBaseUploadResponse:
        # Transition out of `pending` ("waiting for indexing") the moment
        # indexing actually starts. Only meaningful for the queue path --
        # `ingest()` (staging_id=None) has no prior row to transition.
        if staging_id:
            record = await self.staging.find_by_id(staging_id)
            # The queue's `process(staging_id, staged_path, original_filename)`
            # never receives jurisdiction metadata directly -- it was recorded
            # on the ledger row back in `stage()`/`_requeue`, so it is read
            # back here rather than needing a second queue.
            if jurisdiction_metadata is None and record is not None:
                jurisdiction_metadata = record.get("jurisdiction_metadata")
            await self.staging.update_by_id(
                staging_id,
                {
                    "status": "processing",
                    "current_path": str(staged_path),
                    "staged_path": str(staged_path),
                    "file_exists": staged_path.exists(),
                },
            )

        content_hash = self.quality.hash_file(staged_path)
        if not staged_path.exists():
            return await self._mark_failed(
                staged_path, original_filename, None, content_hash,
                "Staged file is missing; the job cannot stay in processing.", staging_id=staging_id,
            )

        exact_match = await self.versions.find_by_hash(content_hash)
        if exact_match is not None:
            matched = exact_match.get("source_document") or str(exact_match.get("_id"))
            return await self._reject_duplicate(staged_path, original_filename, content_hash, matched, staging_id=staging_id)

        try:
            loaded = await self.loader.load(staged_path)
        except Exception as exc:  # noqa: BLE001 - unreadable content: indexing was attempted and failed
            return await self._mark_failed(staged_path, original_filename, None, content_hash, str(exc), staging_id=staging_id)

        fingerprint = self._normalize_fingerprint(loaded.text)
        near_dup = await self._find_near_duplicate(fingerprint)
        if near_dup is not None:
            return await self._reject_duplicate(staged_path, original_filename, content_hash, near_dup, staging_id=staging_id)

        document_type = self._classify_document_type(loaded.text)
        generated_name = self._generate_filename(original_filename, staged_path)
        renamed_path = staged_path.with_name(generated_name)
        if renamed_path != staged_path:
            staged_path.rename(renamed_path)
            # Rule 3: the ledger's path is updated in the same flow as the
            # move, never left pointing at a name that no longer exists.
            if staging_id:
                await self.staging.update_by_id(
                    staging_id,
                    {
                        "current_path": str(renamed_path),
                        "staged_path": str(renamed_path),
                        "generated_filename": generated_name,
                        "file_exists": renamed_path.exists(),
                    },
                )

        try:
            document_id, language, chunks = await self.pipeline.index_file(
                renamed_path, jurisdiction_metadata=jurisdiction_metadata
            )
        except BadRequestError as exc:
            issues = (exc.details or {}).get("issues", [])
            if any("duplicate" in issue.lower() for issue in issues):
                matched = "an existing knowledge base document (hash matched during indexing)"
                return await self._reject_duplicate(
                    renamed_path, original_filename, content_hash, matched, generated_name, staging_id=staging_id
                )
            return await self._mark_failed(
                renamed_path, original_filename, generated_name, content_hash, exc.message, staging_id=staging_id
            )
        except Exception as exc:  # noqa: BLE001 - any indexing failure is "failed", never a hard 500
            return await self._mark_failed(
                renamed_path, original_filename, generated_name, content_hash, str(exc), staging_id=staging_id
            )

        final_path = settings.knowledge_base_dir / generated_name
        shutil.move(str(renamed_path), str(final_path))
        await self.documents.update_by_id(
            document_id,
            {"original_filename": original_filename, "generated_filename": generated_name, "ingestion_source": "admin_kb_upload"},
        )
        latest_version = await self.versions.latest_for_source(generated_name)
        if latest_version is not None:
            await self.versions.update_by_id(
                str(latest_version["_id"]), {"original_filename": original_filename, "ingestion_source": "admin_kb_upload"}
            )
        # `chunks[0].metadata` is every chunk's jurisdiction metadata merged
        # in identically by `IndexingPipeline` (see `apply_to_chunk`), so any
        # one chunk's `review_status`/`review_reasons` speaks for the whole
        # document. Absent entirely when no jurisdiction metadata was supplied
        # -- an upload with no jurisdiction info is left exactly as visible as
        # it always was, not silently gated.
        review_status = chunks[0].metadata.get("review_status") if chunks else None
        review_reasons = chunks[0].metadata.get("review_reasons") if chunks else None
        # Same signal `document_service.upload_and_index` surfaces to a chat
        # user, recorded here instead for whoever reviews the admin KB
        # ingestion ledger -- a scanned Act/notification that OCR couldn't
        # read reliably (wrong script for `settings.ocr_language`, or the
        # engine unavailable) was previously indexed into the shared
        # knowledge base with no trace of that beyond a server log line. See
        # `DocumentLoader._load_pdf`.
        ocr_degraded = chunks[0].metadata.get("ocr_degraded") if chunks else False
        ocr_degraded_reason = chunks[0].metadata.get("ocr_degraded_reason") if chunks else None
        await self._save_staging(
            staging_id,
            self._ledger_fields(
                original_filename=original_filename,
                generated_filename=generated_name,
                status="indexed",
                content_hash=content_hash,
                document_id=document_id,
                current_path=final_path,
                destination_path=settings.knowledge_base_dir,
                reason=None,
                chunks_indexed=len(chunks),
                document_type=document_type,
                language=language,
                # Cached so later near-duplicate checks reuse it instead of
                # rescanning knowledge_base_dir and reparsing this file.
                fingerprint=fingerprint,
                jurisdiction_metadata=jurisdiction_metadata,
                review_status=review_status,
                ocr_degraded=ocr_degraded,
                ocr_degraded_reason=ocr_degraded_reason,
            ),
        )
        await response_cache.bump_generation()

        return KnowledgeBaseUploadResponse(
            status="indexed",
            original_filename=original_filename,
            generated_filename=generated_name,
            document_id=document_id,
            content_hash=content_hash,
            document_type=document_type,
            language=language,
            chunks_indexed=len(chunks),
            review_status=review_status,
            review_reasons=review_reasons,
        )

    async def _save_staging(self, staging_id: str | None, fields: dict[str, Any]) -> None:
        # `staging_id` is set when this call originated from the background
        # queue (a ledger row already exists -- `pending`, then flipped to
        # `processing` -- to update in place); otherwise (direct `ingest`)
        # there's no existing row, so insert one.
        if staging_id:
            # `release_active` also `$unset`s `active_key`, so the single-active-
            # job claim is given up in the same write that records the outcome --
            # never a window where a finished job still blocks the same content.
            await self.staging.release_active(staging_id, fields)
        else:
            await self.staging.insert(fields)

    async def _reject_duplicate(
        self,
        staged_path: Path,
        original_filename: str,
        content_hash: str,
        matched: str,
        generated_filename: str | None = None,
        staging_id: str | None = None,
    ) -> KnowledgeBaseUploadResponse:
        # Archive system: duplicates are moved, never deleted, preserving the
        # original filename and bytes for review -- but only ONCE per
        # distinct content. Confirmed live: a source repeatedly
        # re-discovered by automation/sync (kb_automation/official_source_
        # sync run on an interval and will encounter the SAME official
        # document again every cycle) kept being archived as if it were a
        # newly-seen duplicate every single time, because `_move` always
        # copies under a fresh, collision-safe name -- 150+ redundant
        # ~1.3MB copies of the same handful of official documents
        # accumulated in real `storage/archive` over time. If an earlier
        # rejection of this EXACT content hash already has a preserved copy
        # on disk, that copy already carries all the review value a new one
        # would; reuse its path and discard this attempt's bytes instead of
        # archiving them again.
        existing = await self.staging.find_duplicate_by_content_hash(content_hash)
        existing_path = Path(existing["archived_path"]) if existing and existing.get("archived_path") else None
        if existing_path is not None and existing_path.exists():
            archive_path = existing_path
            staged_path.unlink(missing_ok=True)
        else:
            archive_path = self._move(staged_path, settings.archive_dir, original_filename)
        reason = f"Duplicate of existing knowledge base document: {matched}"
        await self._save_staging(
            staging_id,
            self._ledger_fields(
                original_filename=original_filename,
                generated_filename=generated_filename,
                status="duplicate",
                content_hash=content_hash,
                reason=reason,
                current_path=archive_path,
                destination_path=settings.archive_dir,
                archived_path=str(archive_path),
            ),
        )
        return KnowledgeBaseUploadResponse(
            status="duplicate",
            original_filename=original_filename,
            generated_filename=generated_filename,
            content_hash=content_hash,
            reason=reason,
        )

    async def _mark_failed(
        self,
        staged_path: Path,
        original_filename: str,
        generated_filename: str | None,
        content_hash: str,
        reason: str,
        staging_id: str | None = None,
    ) -> KnowledgeBaseUploadResponse:
        # Indexing was attempted and did not succeed. The file is moved out
        # of `kb_staging_dir` into the failed review queue -- staging holds
        # only in-flight work, so "still in staging" can no longer mean a file
        # that failed weeks ago. Moved, never deleted, so it stays recoverable
        # and an admin can `retry_failed` it after repairing the source.
        failed_path = (
            self._move(staged_path, self.review_failed_dir(), original_filename)
            if staged_path.exists()
            else None
        )
        await self._save_staging(
            staging_id,
            self._ledger_fields(
                original_filename=original_filename,
                generated_filename=generated_filename,
                status="failed",
                content_hash=content_hash,
                reason=reason,
                current_path=failed_path,
                destination_path=self.review_failed_dir(),
            ),
        )
        return KnowledgeBaseUploadResponse(
            status="failed",
            original_filename=original_filename,
            generated_filename=generated_filename,
            content_hash=content_hash,
            reason=reason,
        )

    async def _write_upload(self, file: UploadFile, destination: Path) -> None:
        await write_upload(file, destination)

    def _reserve_path(self, directory: Path, filename: str) -> Path:
        candidate = directory / filename
        if not candidate.exists():
            return candidate
        stem, suffix = Path(filename).stem, Path(filename).suffix
        counter = 2
        while (directory / f"{stem}_{counter}{suffix}").exists():
            counter += 1
        return directory / f"{stem}_{counter}{suffix}"

    async def _find_near_duplicate(self, fingerprint: str) -> str | None:
        # Compares against fingerprints cached on prior `indexed` ledger rows
        # (stored by `_process` after indexing) -- never rescans
        # `knowledge_base_dir` or reparses files already on disk there.
        if not fingerprint:
            return None
        for record in await self.staging.find_by_status("indexed"):
            existing_fp = record.get("fingerprint")
            if not existing_fp:
                continue
            if difflib.SequenceMatcher(None, fingerprint, existing_fp).ratio() >= FINGERPRINT_SIMILARITY_THRESHOLD:
                return record.get("generated_filename") or record.get("original_filename")
        return None

    def _normalize_fingerprint(self, text: str) -> str:
        words = text.split()[:MAX_FINGERPRINT_WORDS]
        return re.sub(r"[^a-z0-9]", "", " ".join(words).lower())

    def _classify_document_type(self, text: str) -> str:
        for pattern, label in DOCUMENT_TYPE_RULES:
            if pattern.search(text):
                return label
        return "Document"

    def _generate_filename(self, original_filename: str, staged_path: Path | None = None) -> str:
        """Keep the uploaded name as the canonical, admin-visible KB name.

        Exact-content duplicates are rejected before this method. If a
        different document already owns the same filename, `_2`, `_3`, etc.
        is appended rather than overwriting either file.
        """
        candidate = Path(original_filename).name.strip() or "Document"
        suffix = Path(candidate).suffix
        if len(candidate) > MAX_FILENAME_LENGTH:
            keep = MAX_FILENAME_LENGTH - len(suffix)
            candidate = f"{Path(candidate).stem[:keep].rstrip()}{suffix}"
        return self._deduplicate_name(candidate, suffix, ignore_path=staged_path)

    def _deduplicate_name(
        self, candidate: str, suffix: str, *, ignore_path: Path | None = None
    ) -> str:
        existing: set[str] = set()
        if settings.knowledge_base_dir.exists():
            existing.update(p.name.lower() for p in settings.knowledge_base_dir.iterdir())
        if settings.kb_staging_dir.exists():
            existing.update(
                p.name.lower()
                for p in settings.kb_staging_dir.iterdir()
                if ignore_path is None or p != ignore_path
            )
        if candidate.lower() not in existing:
            return candidate
        stem = candidate[: -len(suffix)] if suffix else candidate
        counter = 2
        new_candidate = f"{stem}_{counter}{suffix}"
        while new_candidate.lower() in existing:
            counter += 1
            new_candidate = f"{stem}_{counter}{suffix}"
        return new_candidate
