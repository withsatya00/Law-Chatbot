"""Sequential background worker for admin Knowledge Base uploads.

`POST /admin/knowledge-base/upload` stages the file synchronously (fast) and
calls `enqueue` so the request returns immediately. A single persistent
consumer task then runs `KnowledgeBaseIngestionService.process` for each job
in arrival order -- uploads must not be indexed concurrently, since duplicate
detection reads the current on-disk state of `knowledge_base_dir`.

Phase 3A "KB Indexing Queue Resilience": one job raising an exception that
escapes `process()` (most anticipated failures are already caught inside
`KnowledgeBaseIngestionService.process` and recorded as a "failed" staging
status, but the consumer loop must not depend on that -- a bug in `process`
itself, or an error during its own failure handling, must not be able to
take down every future upload) must never stop the loop from picking up the
next job. See `_run`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from app.repositories.kb_staging import KnowledgeBaseStagingRepository
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService

log = structlog.get_logger(__name__)


class KnowledgeBaseIndexingQueue:
    def __init__(self, service: KnowledgeBaseIngestionService | None = None) -> None:
        self._queue: asyncio.Queue[tuple[str, Path, str]] = asyncio.Queue()
        self._service = service or KnowledgeBaseIngestionService()
        self._started = False
        self._task: asyncio.Task[None] | None = None
        self._pending_ids: set[str] = set()

    def enqueue(self, staging_id: str, staged_path: Path, original_filename: str) -> None:
        if staging_id not in self._pending_ids:
            self._pending_ids.add(staging_id)
            self._queue.put_nowait((staging_id, staged_path, original_filename))
        if self._task is None or self._task.done():
            self._started = True
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            staging_id, staged_path, original_filename = await self._queue.get()
            try:
                await self._service.process(staging_id, staged_path, original_filename)
            except Exception as exc:  # noqa: BLE001 - one bad job must never kill this loop (Phase 3A)
                # `Exception`, not `BaseException`: asyncio.CancelledError must still
                # propagate so the task can be cancelled normally (e.g. on shutdown).
                log.warning(
                    "kb_indexing_job_failed",
                    staging_id=staging_id,
                    staged_path=str(staged_path),
                    original_filename=original_filename,
                    error=str(exc),
                )
            finally:
                self._pending_ids.discard(staging_id)
                self._queue.task_done()

    async def close(self) -> None:
        """Cancel the consumer; persisted pending/processing jobs recover next boot."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._started = False

    async def join(self) -> None:
        """Blocks until every currently-enqueued job has been processed.
        Used by one-time bulk operations (e.g. the uploads backfill) that
        need a synchronous outcome summary; the upload route never calls
        this since it must return immediately."""
        await self._queue.join()


kb_indexing_queue = KnowledgeBaseIndexingQueue()


async def recover_pending_jobs(
    queue: KnowledgeBaseIndexingQueue | None = None,
    repository: KnowledgeBaseStagingRepository | None = None,
) -> int:
    """Startup recovery: a record left at status="pending" (staged, queued,
    never dequeued) or "processing" (dequeued but indexing didn't finish)
    means the process was interrupted -- e.g. a crash/restart -- before that
    file reached a terminal status (indexed/duplicate/failed). Requeues each
    one via the existing `enqueue` -- reusing its staging record and the file
    already sitting in `kb_staging_dir`, never re-uploading. Records missing
    `staged_path` (pre-recovery ledger rows) or whose file no longer exists
    on disk are skipped.
    """
    target = queue or kb_indexing_queue
    repo = repository or KnowledgeBaseStagingRepository()
    stuck = [*await repo.find_by_status("pending"), *await repo.find_by_status("processing")]
    requeued = 0
    for record in stuck:
        staged_path = record.get("current_path") or record.get("staged_path")
        if not staged_path or not Path(staged_path).exists():
            # A tracked job whose file is gone can never finish. Leaving it at
            # "processing" is what produced permanently stuck records that no
            # admin view explained, so it is closed as `needs_review` here and
            # the active-content claim is released with it.
            await repo.release_active(
                record["_id"],
                {
                    "status": "needs_review",
                    "previous_status": record.get("status"),
                    "file_exists": False,
                    "reason": "Stale job: the staged file is missing, so indexing can never complete.",
                },
            )
            log.warning("kb_indexing_job_stale_file_missing", staging_id=record["_id"])
            continue
        target.enqueue(record["_id"], Path(staged_path), record["original_filename"])
        requeued += 1
    return requeued
