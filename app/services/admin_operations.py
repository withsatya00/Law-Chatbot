"""Administrative knowledge-base operations, in the service layer.

Same reasoning as `draft_management`: these computations lived in the
`/admin/*` route bodies, and an admin can now ask for them by talking. One
implementation, two callers -- so "how many chunks are indexed" cannot come
back with two different answers depending on how it was asked.

Scheduling is deliberately NOT in here. `queue_reindex` creates the job
record and returns what is needed to start the work; the REST route hands it
to FastAPI's `BackgroundTasks` and the chat workflow schedules it on the
running loop. That is the one part that genuinely differs between callers.
"""

from pathlib import Path
from typing import Any

from app.cache.response_cache import response_cache
from app.core.config import settings
from app.core.exceptions import BadRequestError, NotFoundError
from app.database.mongodb import mongodb
from app.models.collections import (
    DOCUMENT_VERSIONS,
    EMBEDDINGS_METADATA,
    KB_STAGING_RECORDS,
    PROMPT_LOGS,
    QUERY_LOGS,
    SYSTEM_LOGS,
)
from app.rag.incremental import IncrementalReindexRunner
from app.repositories.documents import EmbeddingMetadataRepository
from app.repositories.kb_staging import KnowledgeBaseStagingRepository
from app.repositories.versioning import IndexingJobRepository
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService


async def knowledge_base_status() -> dict[str, Any]:
    chunks = await mongodb.db[EMBEDDINGS_METADATA].count_documents({})
    active_documents = await mongodb.db[DOCUMENT_VERSIONS].count_documents({"document_status": "active"})
    deleted_documents = await mongodb.db[DOCUMENT_VERSIONS].count_documents({"document_status": "deleted"})
    return {
        "indexed_chunks": chunks,
        "active_documents": active_documents,
        "deleted_documents": deleted_documents,
        "status": "ready" if chunks else "empty",
    }


async def logs_status() -> dict[str, Any]:
    return {
        "prompt_logs": await mongodb.db[PROMPT_LOGS].count_documents({}),
        "query_logs": await mongodb.db[QUERY_LOGS].count_documents({}),
        "system_logs": await mongodb.db[SYSTEM_LOGS].count_documents({}),
    }


async def queue_reindex(root: str | None = None) -> tuple[IncrementalReindexRunner, str, Path]:
    """Records a queued incremental re-index and returns how to run it.

    Detects new/updated/deleted files under the knowledge-base root and
    re-indexes only those. The caller schedules `runner.run_for_job(job_id,
    target_root)`; progress and the final summary are read back from the job.
    """
    target_root = Path(root) if root else settings.knowledge_base_dir
    target_root.mkdir(parents=True, exist_ok=True)
    runner = IncrementalReindexRunner()
    job_id = await runner.jobs.insert({"root": str(target_root), "status": "queued"})
    return runner, job_id, target_root


async def reindex_job(job_id: str) -> dict[str, Any]:
    job = await IndexingJobRepository().find_by_id(job_id)
    if job is None:
        raise NotFoundError("Indexing job not found.")
    return job


async def flush_cache(
    question: str | None = None, language: str | None = None, intent: str | None = None
) -> dict[str, Any]:
    """Clears the response cache, entirely or one entry.

    A single entry needs all three parts of its key: purging "the one bad
    answer" without them would quietly purge nothing and report success.
    """
    if question:
        if not (language and intent):
            raise BadRequestError("language and intent are required to purge a specific cache entry.")
        removed = await response_cache.purge_entry(question, language, intent)
        return {"status": "purged" if removed else "not_found", "scope": "entry"}
    await response_cache.bump_generation()
    return {"status": "purged", "scope": "all"}


def _count_files(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if p.is_file()) if directory.exists() else 0


async def kb_dashboard() -> dict[str, Any]:
    """Ledger counts and disk counts, reported separately.

    The two answer different questions and used to be conflated, which is how
    "68 files in staging" and "0 pending records" could both be true with
    nobody able to see the contradiction. `ledger` counts rows per status;
    `disk` counts actual files in each directory. `stale_or_missing` is the
    number of active rows whose file is not where the ledger says it is.
    """
    repository = KnowledgeBaseStagingRepository()
    counts = await repository.status_counts()
    stale = len(await repository.find_stale_active())
    # Ledger findings, not files: reconciliation flagged these rows as having
    # nothing behind their recorded path. `find_stale_active` cannot see them
    # (they are no longer pending/processing), which is why they read as zero.
    path_missing = len(await repository.find_path_missing())
    service = KnowledgeBaseIngestionService()
    storage_bytes = (
        sum(p.stat().st_size for p in settings.knowledge_base_dir.rglob("*") if p.is_file())
        if settings.knowledge_base_dir.exists()
        else 0
    )
    return {
        "ledger": {
            "pending": counts["pending"],
            "processing": counts["processing"],
            "indexed": counts["indexed"],
            "duplicate": counts["duplicate"],
            "failed": counts["failed"],
            "needs_review": counts["needs_review"],
            "stale_or_missing": stale,
            "path_missing": path_missing,
            "total": counts["total"],
        },
        "disk": {
            "staging_files": _count_files(settings.kb_staging_dir),
            "review_pending_files": _count_files(service.review_pending_dir()),
            "review_failed_files": _count_files(service.review_failed_dir()),
            "review_queue_files": _count_files(settings.kb_review_dir),
            "knowledge_base_files": _count_files(settings.knowledge_base_dir),
            "archive_files": _count_files(settings.archive_dir),
        },
        # Retained keys so existing dashboard callers keep working.
        "total_files": counts["total"],
        "indexed_files": counts["indexed"],
        "processing_files": counts["processing"],
        "pending_files": counts["pending"],
        "duplicate_files": counts["duplicate"],
        "failed_files": counts["failed"],
        "needs_review_files": counts["needs_review"],
        "storage_usage_bytes": storage_bytes,
    }


async def staging_records(status: str | None = None) -> dict[str, Any]:
    """Staging-ledger entries.

    `status=all` lists every record whatever its state; the default lists the
    actionable ones (pending, processing, failed, needs_review) -- what is
    either in flight or waiting on an admin. Each row reports the file's real
    current location and whether it is actually there, so a record pointing at
    a path that no longer exists is visible rather than merely stale.
    """
    repository = KnowledgeBaseStagingRepository()
    if status == "all":
        query: dict[str, Any] = {}
        status_filter = "all"
    elif status:
        # A closed missing-file finding remains in the immutable audit ledger
        # with its original status, but it is no longer waiting for review.
        # Keep it visible under `status=all` while excluding it from every
        # actionable queue/filter.
        query = {"status": status, "resolution": {"$exists": False}}
        status_filter = status
    else:
        query = {
            "status": {"$in": list(repository.ACTIONABLE_STATUSES)},
            "resolution": {"$exists": False},
        }
        status_filter = "actionable"
    cursor = mongodb.db[KB_STAGING_RECORDS].find(query).sort("created_at", -1)
    records = []
    for record in [item async for item in cursor]:
        current = record.get("current_path") or record.get("staged_path") or record.get("archived_path")
        record["current_path"] = current
        record["file_exists"] = bool(current and Path(current).exists())
        records.append(record)
    return {"status_filter": status_filter, "count": len(records), "records": records}


def _public_staging_record(record: dict[str, Any]) -> dict[str, Any]:
    """Browser-safe subset of an ingestion ledger row."""
    return {
        "staging_id": str(record["_id"]),
        "status": str(record.get("status") or "pending"),
        "original_filename": record.get("original_filename"),
        "generated_filename": record.get("generated_filename"),
        "document_id": record.get("document_id"),
        "chunks_indexed": record.get("chunks_indexed"),
        "reason": record.get("reason"),
    }


async def staging_record(staging_id: str) -> dict[str, Any]:
    """Upload-specific status for chat UI polling."""
    record = await KnowledgeBaseStagingRepository().find_by_id(staging_id)
    if record is None:
        raise NotFoundError("Knowledge Base upload record not found.")
    return _public_staging_record(record)


async def staging_record_by_hash(content_hash: str) -> dict[str, Any]:
    """Canonical outcome for a legacy receipt that predates tracking IDs.

    Prefer the successful indexed row over a later duplicate-attempt row, so
    the UI reports whether these exact bytes are available in the KB.
    """
    repository = KnowledgeBaseStagingRepository()
    record = await repository.find_indexed_by_content_hash(content_hash)
    if record is None:
        record = await repository.find_by_content_hash(content_hash)
    if record is None:
        raise NotFoundError("Knowledge Base upload record not found.")
    return _public_staging_record(record)


async def staging_file_path(staging_id: str) -> tuple[Path, str]:
    """Resolves one staging record to the actual file on disk, for an admin
    preview before approving. Never takes a path from the caller -- only the
    location already recorded against this specific ledger row -- so this
    cannot be used to read anything outside the KB staging/review/archive
    directories."""
    record = await KnowledgeBaseStagingRepository().find_by_id(staging_id)
    if record is None:
        raise NotFoundError("Knowledge Base upload record not found.")
    path_str = record.get("current_path") or record.get("staged_path") or record.get("archived_path")
    if not path_str or not Path(path_str).is_file():
        raise NotFoundError("The file for this record is not on disk.")
    filename = str(record.get("original_filename") or Path(path_str).name)
    return Path(path_str), filename


async def kb_reconcile(*, apply: bool = False, expected_total: int | None = None) -> dict[str, Any]:
    """Dry run by default; `apply=True` writes a manifest first, then moves."""
    return await KnowledgeBaseIngestionService().reconcile_staging(apply=apply, expected_total=expected_total)


async def kb_apply_manifest(manifest_path: str) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        raise NotFoundError(f"Reconciliation manifest '{manifest_path}' not found.")
    return await KnowledgeBaseIngestionService().apply_manifest(path)


async def kb_close_missing(staging_id: str, reason: str) -> dict[str, Any]:
    return await KnowledgeBaseIngestionService().close_path_missing(staging_id, reason)


async def kb_resource_missing(staging_id: str, replacement_path: str) -> dict[str, Any]:
    return await KnowledgeBaseIngestionService().resource_path_missing(staging_id, Path(replacement_path))


def kb_manifests() -> dict[str, Any]:
    manifests = KnowledgeBaseIngestionService().list_manifests()
    return {"count": len(manifests), "manifests": manifests}


def kb_manifest(name: str) -> dict[str, Any]:
    return KnowledgeBaseIngestionService().read_manifest(name)


async def kb_approve(staging_id: str) -> dict[str, Any]:
    return await KnowledgeBaseIngestionService().approve_needs_review(staging_id)


async def kb_retry(staging_id: str) -> dict[str, Any]:
    return await KnowledgeBaseIngestionService().retry_failed(staging_id)


async def kb_archive(staging_id: str, reason: str | None = None) -> dict[str, Any]:
    return await KnowledgeBaseIngestionService().archive_rejected(staging_id, reason)


async def unowned_documents() -> dict[str, Any]:
    """Documents indexed before per-user ownership existed.

    They have no owner recorded and stay globally visible forever with no way
    to even see which ones those are. Surfaced by source document (not
    per-chunk -- an admin thinks in documents) so they can be triaged.
    """
    documents = await EmbeddingMetadataRepository().list_unowned_documents()
    return {"count": len(documents), "documents": documents}


async def assign_owner(source_document: str, owner_user_id: str | None) -> dict[str, Any]:
    """Resolves one unowned document.

    Passing `owner_user_id` attributes the document to a real user;
    omitting it is an explicit decision that it stays globally visible,
    recorded so it stops appearing as unreviewed.
    """
    modified = await EmbeddingMetadataRepository().assign_document_ownership(source_document, owner_user_id)
    if modified == 0:
        raise NotFoundError(f"No chunks found for source document '{source_document}'.")
    return {"source_document": source_document, "chunks_updated": modified, "owner_user_id": owner_user_id}
