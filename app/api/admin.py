import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_current_user_claims, require_admin
from app.core.config import settings
from app.core.exceptions import BadRequestError
from app.rag.kb_jurisdiction import (
    PROVENANCE_MANUAL,
    JurisdictionMetadataError,
    document_metadata_fields,
    normalize_jurisdiction,
)
from app.repositories.phase3 import AuditLogRepository
from app.schemas.admin import (
    AssignDocumentOwnershipRequest,
    AutomationReviewQueueItem,
    BulkReviewRequest,
    BulkReviewResponse,
    BulkReviewResultItem,
    CacheFlushRequest,
    KnowledgeBaseJurisdictionUpdateResponse,
    KnowledgeBaseUploadResponse,
)
from app.services import admin_operations
from app.services.kb_indexing_queue import kb_indexing_queue
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService


def _parse_jurisdiction_metadata(raw_json: str | None) -> dict[str, Any]:
    """Shared by every route that accepts jurisdiction metadata as a form
    field. Always returns a normalized dict -- even an admin upload with NO
    `jurisdiction_metadata` field runs `normalize_jurisdiction({})`, which
    resolves to `issuing_level=unknown`/`applicability=unknown`, and therefore
    `review_status=needs_review` (see `kb_jurisdiction._review_reasons`).
    Deliberately not `None`-able: a document nobody described jurisdiction for
    must default to "not yet fit for shared retrieval", never to "unrestricted
    because nothing was said" (objective item 3 / item 4).
    """
    raw: dict[str, Any] = {}
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except (TypeError, ValueError) as exc:
            raise BadRequestError(f"jurisdiction_metadata must be valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise BadRequestError("jurisdiction_metadata must be a JSON object.")
        raw = parsed
    try:
        normalized = normalize_jurisdiction(raw, provenance=PROVENANCE_MANUAL)
    except JurisdictionMetadataError as exc:
        raise BadRequestError("Invalid jurisdiction metadata.", {"issues": exc.errors}) from exc
    return document_metadata_fields(normalized)

# Every route here is admin-only: `require_admin` is a router-level dependency,
# so a non-admin caller is rejected before any handler body runs.
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


async def _audit(action: str, actor_user_id: str, **details: Any) -> None:
    """Records one mutating knowledge-base action. Every state change an admin
    can make to the ingestion ledger goes through here, so "who promoted this
    document into the shared Knowledge Base" always has an answer."""
    await AuditLogRepository().insert(
        {
            "actor_user_id": actor_user_id,
            "action": action,
            "resource": "knowledge_base",
            "details": details,
        }
    )


@router.get("/knowledge-base/status")
async def knowledge_base_status() -> dict[str, Any]:
    return await admin_operations.knowledge_base_status()


@router.get("/logs/status")
async def logs_status() -> dict[str, Any]:
    return await admin_operations.logs_status()


@router.post("/reindex")
async def trigger_incremental_reindex(
    background_tasks: BackgroundTasks,
    root: str | None = None,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> dict[str, Any]:
    """Detects new/updated/deleted files under the knowledge-base root and re-indexes only those.

    Runs in the background so large corpora don't block the request; poll
    ``GET /admin/reindex/{job_id}`` for progress and the final summary.
    """
    runner, job_id, target_root = await admin_operations.queue_reindex(root)
    background_tasks.add_task(runner.run_for_job, job_id, target_root)
    await _audit("kb_reindex_triggered", str(claims["sub"]), job_id=job_id, root=str(target_root))
    return {"job_id": job_id, "status": "queued", "root": str(target_root)}


@router.get("/reindex/{job_id}")
async def reindex_job_status(job_id: str) -> dict[str, Any]:
    return await admin_operations.reindex_job(job_id)


@router.post("/cache/flush")
async def flush_response_cache(
    payload: CacheFlushRequest | None = None,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> dict[str, Any]:
    """Clears the response cache. With no body, invalidates everything (same
    mechanism as an automatic post-reindex invalidation). With `question` +
    `language` + `intent`, removes just that one cached answer -- e.g. after
    spotting a bad cached response without wiping every other cached entry.
    """
    if payload is None:
        result = await admin_operations.flush_cache()
    else:
        result = await admin_operations.flush_cache(payload.question, payload.language, payload.intent)
    await _audit("cache_flush", str(claims["sub"]), **result)
    return result


@router.post("/knowledge-base/upload", response_model=KnowledgeBaseUploadResponse)
async def upload_to_knowledge_base(
    file: UploadFile = File(...),
    jurisdiction_metadata: str | None = Form(
        None,
        description=(
            "JSON object: issuing_level, applicability, applicable_state_codes, "
            "applicable_localities, jurisdiction_source_type, source_url, "
            "source_page_reference, version_label, effective_from/effective_to "
            "(ISO dates), amends/amended_by/supersedes/superseded_by, "
            "verification_status, verified_by, section_overrides. Omitted or "
            "incomplete fields resolve to 'unknown', which routes the document "
            "to needs_review and keeps it out of shared retrieval."
        ),
    ),
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> KnowledgeBaseUploadResponse:
    """Admin Knowledge Base Ingestion Pipeline: stages the file synchronously
    then returns immediately, handing the dedup/index/transfer work to
    `kb_indexing_queue`'s background worker (background indexing queue) --
    no manual indexing step. That worker rejects exact or content-fingerprint
    duplicates (moving them into `archive_dir`, never deleting), renames
    accepted files from a meaningless name to a content-derived one, indexes
    them, and moves them into `knowledge_base_dir` -- or, on a failed
    outcome, leaves the file in the staging directory with status "failed"
    for an admin to review. Separate from `POST /upload` (chat users' own
    private documents, never touched here).
    """
    jurisdiction_fields = _parse_jurisdiction_metadata(jurisdiction_metadata)
    staged = await KnowledgeBaseIngestionService().stage(file, jurisdiction_metadata=jurisdiction_fields)
    if not staged.claimed:
        # Identical bytes already have an active job (possibly in another
        # process). The attempt is recorded and the copy archived; enqueuing it
        # again would index the same document twice.
        await _audit(
            "kb_upload_duplicate", str(claims["sub"]),
            staging_id=staged.staging_id, filename=staged.original_filename, status=staged.status,
        )
        return KnowledgeBaseUploadResponse(
            status=staged.status,
            staging_id=staged.staging_id,
            original_filename=staged.original_filename,
            reason=staged.reason,
        )
    kb_indexing_queue.enqueue(staged.staging_id, staged.staged_path, staged.original_filename)
    await _audit(
        "kb_upload_queued", str(claims["sub"]),
        staging_id=staged.staging_id, filename=staged.original_filename,
    )
    return KnowledgeBaseUploadResponse(
        status="pending",
        staging_id=staged.staging_id,
        original_filename=staged.original_filename,
        reason="Queued for background indexing.",
    )


@router.post("/knowledge-base/backfill-uploads")
async def backfill_knowledge_base_from_uploads(
    source: str | None = None,
    confirm: bool = False,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> dict[str, Any]:
    """Admin-authorized quarantine of documents already sitting in
    `upload_storage_dir` (default) or `source`.

    `upload_storage_dir` holds users' private chat attachments, so this never
    indexes anything and never runs by itself: it requires `confirm=true`, and
    every candidate is copied into the review queue as `needs_review` for an
    admin to approve individually via
    `POST /admin/knowledge-base/staging/{id}/approve`. The originals are left
    untouched.
    """
    source_dir = Path(source) if source else settings.upload_storage_dir
    result = await KnowledgeBaseIngestionService().backfill_from_uploads(source_dir, authorized=confirm)
    await _audit("kb_backfill_uploads", str(claims["sub"]), source=str(source_dir), **result)
    return result


@router.get("/knowledge-base/reconcile-staging")
async def inspect_knowledge_base_reconciliation() -> dict[str, Any]:
    """Read-only reconciliation plan: what the apply would move, where, and
    why. Touches neither disk nor the ledger."""
    return await admin_operations.kb_reconcile(apply=False)


@router.post("/knowledge-base/reconcile-staging")
async def reconcile_knowledge_base_staging(
    apply: bool = False,
    expected_total: int | None = None,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> dict[str, Any]:
    """Fixes drift between `kb_staging_dir` (disk) and `kb_staging_records`
    (the ledger `GET /knowledge-base/staging` reads).

    DRY RUN unless `apply=true` -- the default reports the plan and changes
    nothing. With `apply=true` a JSON manifest (hashes, old paths, proposed
    destinations, previous statuses) is written first, then files are moved:
    already-indexed content to the archive, corrupt/failed documents to the
    failed review queue, and everything unverified to the pending review
    queue as `needs_review` -- never auto-indexed, never deleted. Pass
    `expected_total` to refuse the apply unless the directory holds exactly
    that many files.
    """
    result = await admin_operations.kb_reconcile(apply=apply, expected_total=expected_total)
    if apply:
        await _audit(
            "kb_reconcile_apply",
            str(claims["sub"]),
            moved=result.get("moved"),
            manifest_path=result.get("manifest_path"),
            files_scanned=result.get("files_scanned"),
        )
    return result


@router.post("/knowledge-base/reconcile-staging/apply-manifest")
async def apply_knowledge_base_reconciliation_manifest(
    manifest_path: str,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> dict[str, Any]:
    """Applies a plan produced by an earlier dry run, so what an admin
    reviewed is exactly what gets executed. Replaying it is a no-op for
    entries already moved."""
    result = await admin_operations.kb_apply_manifest(manifest_path)
    await _audit("kb_reconcile_apply_manifest", str(claims["sub"]), **result)
    return result


@router.get("/knowledge-base/reconciliation/manifests")
async def list_reconciliation_manifests() -> dict[str, Any]:
    """Read-only: the reconciliation manifests on disk, newest first."""
    return admin_operations.kb_manifests()


@router.get("/knowledge-base/reconciliation/manifests/{name}")
async def read_reconciliation_manifest(name: str) -> dict[str, Any]:
    """Read-only per-physical-file audit trail for one reconciliation run:
    timestamp, original path, destination, hash, previous status and reason
    code for every file moved -- including byte-identical copies, which share
    one ledger row but appear here individually."""
    return admin_operations.kb_manifest(name)


@router.post("/knowledge-base/staging/{staging_id}/close-missing")
async def close_missing_file_record(
    staging_id: str,
    reason: str,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> dict[str, Any]:
    """Closes a `path_missing` finding. The reason is mandatory."""
    result = await admin_operations.kb_close_missing(staging_id, reason)
    await _audit("kb_close_missing_file", str(claims["sub"]), staging_id=staging_id, reason=reason)
    return result


@router.post("/knowledge-base/staging/{staging_id}/resource")
async def resource_missing_file_record(
    staging_id: str,
    replacement_path: str,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> dict[str, Any]:
    """Re-sources a `path_missing` record from a replacement file the admin
    names. Nothing is searched for automatically, and the record stays
    `needs_review` until it is separately approved."""
    result = await admin_operations.kb_resource_missing(staging_id, replacement_path)
    await _audit(
        "kb_resource_missing_file",
        str(claims["sub"]),
        **{**result, "staging_id": staging_id, "replacement_path": replacement_path},
    )
    return result


@router.put("/knowledge-base/documents/{document_id}/jurisdiction", response_model=KnowledgeBaseJurisdictionUpdateResponse)
async def update_knowledge_base_document_jurisdiction(
    document_id: str,
    jurisdiction_metadata: dict[str, Any] = Body(
        ...,
        description=(
            "The FULL, corrected jurisdiction metadata object (same shape as the upload route's "
            "jurisdiction_metadata field) -- replaces the document's existing jurisdiction metadata "
            "wholesale, not a partial patch. Supplying verification_status='verified' with verified_by "
            "set, alongside a known issuing_level/applicability/source_url, is what publishes the "
            "document into shared retrieval; leaving any of those unknown keeps it needs_review."
        ),
    ),
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> KnowledgeBaseJurisdictionUpdateResponse:
    """Phase 1 gap 2: corrects a wrong State/effective-date/etc. on an
    ALREADY-INDEXED Knowledge Base document, or publishes one that was left
    `needs_review`, without a full re-upload or re-embedding. See
    `KnowledgeBaseIngestionService.update_jurisdiction_metadata`.
    """
    result = await KnowledgeBaseIngestionService().update_jurisdiction_metadata(document_id, jurisdiction_metadata)
    await _audit("kb_jurisdiction_metadata_updated", str(claims["sub"]), **result)
    return KnowledgeBaseJurisdictionUpdateResponse(**result)


@router.get("/knowledge-base/documents/needs-review")
async def list_needs_review_documents(limit: int = Query(200, ge=1, le=500)) -> list[dict[str, Any]]:
    """Live `needs_review` documents, queried directly from
    `uploaded_documents` -- see
    `KnowledgeBaseIngestionService.list_needs_review_documents`'s docstring
    for why this exists instead of the staging-ledger `status=all` listing
    (that ledger's `review_status` is a stale, write-once snapshot).
    """
    return await KnowledgeBaseIngestionService().list_needs_review_documents(limit=limit)


@router.get("/knowledge-base/automation-review-queue", response_model=list[AutomationReviewQueueItem])
async def automation_review_queue(
    limit: int = Query(50, ge=1, le=200),
    jurisdiction_code: list[str] | None = Query(None),
) -> list[AutomationReviewQueueItem]:
    """Read-only queue of automation-discovered documents awaiting human
    review, pre-filled with everything `list_automation_review_queue`
    already knows (title, official source URL, jurisdiction, act number) so
    a reviewer doesn't have to open each job record to decide. Pair with
    `POST .../documents/bulk-review` once a batch has been looked at.
    """
    items = await KnowledgeBaseIngestionService().list_automation_review_queue(
        limit=limit, jurisdiction_codes=tuple(jurisdiction_code or ()),
    )
    return [AutomationReviewQueueItem(**item) for item in items]


@router.post("/knowledge-base/documents/bulk-review", response_model=BulkReviewResponse)
async def bulk_review_automated_documents(
    request: BulkReviewRequest,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> BulkReviewResponse:
    """Marks every `document_id` in the batch `verified` in one call, for a
    reviewer who has already looked at each one via the review queue above.
    NOT an auto-approve: this calls the same per-document
    `update_jurisdiction_metadata` path as the single-document endpoint, so
    a document whose structural fields are still incomplete stays
    `needs_review` exactly as it would one at a time -- being in the batch
    changes nothing about what makes a document fit for shared retrieval.
    """
    results = await KnowledgeBaseIngestionService().bulk_review_automated_documents(
        request.document_ids, request.verified_by,
    )
    await _audit(
        "kb_bulk_review", str(claims["sub"]),
        verified_by=request.verified_by, document_count=len(request.document_ids),
        outcomes={item["document_id"]: item["outcome"] for item in results},
    )
    return BulkReviewResponse(results=[BulkReviewResultItem(**item) for item in results])


@router.post("/knowledge-base/staging/{staging_id}/approve")
async def approve_staging_document(
    staging_id: str,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> dict[str, Any]:
    """The only way a `needs_review` document enters the shared Knowledge
    Base: an admin approves it explicitly and it is queued for indexing."""
    result = await admin_operations.kb_approve(staging_id)
    await _audit(
        "kb_approve_needs_review", str(claims["sub"]), **{**result, "staging_id": staging_id}
    )
    return result


@router.post("/knowledge-base/staging/{staging_id}/retry")
async def retry_staging_document(
    staging_id: str,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> dict[str, Any]:
    """Re-queues one previously failed document for indexing."""
    result = await admin_operations.kb_retry(staging_id)
    await _audit("kb_retry_failed", str(claims["sub"]), **{**result, "staging_id": staging_id})
    return result


@router.post("/knowledge-base/staging/{staging_id}/archive")
async def archive_staging_document(
    staging_id: str,
    reason: str | None = None,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> dict[str, Any]:
    """Rejects one document: its file is moved to the archive (never deleted)
    and its ledger row closed."""
    result = await admin_operations.kb_archive(staging_id, reason)
    await _audit(
        "kb_archive_rejected",
        str(claims["sub"]),
        **{**result, "staging_id": staging_id, "reason": reason},
    )
    return result


@router.get("/knowledge-base/dashboard")
async def knowledge_base_dashboard() -> dict[str, Any]:
    """Admin dashboard status: per-status file counts plus knowledge-base
    storage usage. `pending` (queued, indexing not yet attempted) and
    `failed` (indexing attempted and unsuccessful) are distinct buckets.
    """
    return await admin_operations.kb_dashboard()


@router.get("/documents/unowned")
async def list_unowned_documents() -> dict[str, Any]:
    """Part 45 flagged, but never fixed, that documents indexed before
    per-user ownership existed have no owner recorded and stay globally
    visible forever with no way to even see which ones those are. Surfaces
    them by source document (not per-chunk -- an admin thinks in documents)
    so they can be triaged via `POST /documents/{source_document}/assign-owner`.
    """
    return await admin_operations.unowned_documents()


@router.post("/documents/{source_document}/assign-owner")
async def assign_document_ownership(source_document: str, request: AssignDocumentOwnershipRequest) -> dict[str, Any]:
    """Resolves one entry from `GET /documents/unowned`. Passing
    `owner_user_id` attributes the document to a real user (e.g. an admin
    who knows, from outside this system, who actually uploaded it) --
    afterwards it's visible to that user from any session, same as any
    Part-46 document. Omitting it is an explicit decision that the document
    stays globally visible, recorded so it stops appearing as unreviewed.
    """
    return await admin_operations.assign_owner(source_document, request.owner_user_id)


@router.get("/knowledge-base/staging")
async def knowledge_base_staging_status(status: str | None = None) -> dict[str, Any]:
    """Lists staging-ledger entries. `status=all` returns every record whatever
    its state; the default shows the actionable ones (pending, processing,
    failed, needs_review). Each row carries the file's real current location
    and whether that file is actually present.
    """
    return await admin_operations.staging_records(status)


@router.get("/knowledge-base/staging/{staging_id}")
async def knowledge_base_staging_record(staging_id: str) -> dict[str, Any]:
    """Returns the current state of one upload without exposing filesystem paths."""
    return await admin_operations.staging_record(staging_id)


@router.get("/knowledge-base/staging/{staging_id}/file")
async def download_staging_file(staging_id: str) -> FileResponse:
    """Streams the actual file for one staging record, so an admin can preview
    it (e.g. read the PDF) before approving or rejecting. Read-only; the path
    served is always the one already recorded for this record, never one
    supplied by the caller.
    """
    path, filename = await admin_operations.staging_file_path(staging_id)
    return FileResponse(path, filename=filename)


@router.get("/knowledge-base/content/{content_hash}/status")
async def knowledge_base_content_status(content_hash: str) -> dict[str, Any]:
    """Resolves old UI receipts by exact SHA-256 content identity."""
    if len(content_hash) != 64 or any(character not in "0123456789abcdef" for character in content_hash):
        raise BadRequestError("A lowercase SHA-256 content hash is required.")
    return await admin_operations.staging_record_by_hash(content_hash)
