from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from app.cache.response_cache import response_cache
from app.core.constants import ALLOWED_UPLOAD_EXTENSIONS
from app.repositories.versioning import DocumentVersionRepository, IndexingJobRepository

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FileChangeSet:
    new_files: list[Path]
    updated_files: list[Path]
    deleted_sources: list[str]
    unchanged_files: list[Path]


class IncrementalIndexPlanner:
    def __init__(
        self, versions: DocumentVersionRepository | None = None, jobs: "IndexingJobRepository | None" = None,
    ) -> None:
        self.versions = versions or DocumentVersionRepository()
        self.jobs = jobs or IndexingJobRepository()

    async def plan(self, root: Path, allowed_suffixes: set[str]) -> FileChangeSet:
        current_files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed_suffixes]
        new_files: list[Path] = []
        updated_files: list[Path] = []
        unchanged_files: list[Path] = []
        current_names = {path.name for path in current_files}
        # `document_versions` is global -- it also holds every owner-scoped
        # per-user chat upload (Part 45/46), indexed through a completely
        # different path (`DocumentService.upload_and_index`) that has nothing
        # to do with any `root` a reindex ever runs against. Querying it
        # unfiltered here previously treated "every other document that
        # exists anywhere in the database" as "deleted from THIS root" --
        # confirmed directly: reindexing `storage/knowledge_base` deindexed 28
        # unrelated per-user upload documents (~1470 chunks) that were never
        # under that root to begin with. Scoping to this root's own indexing
        # history (only files THIS runner has ever indexed from THIS root)
        # makes "deleted" mean "was here, now isn't" instead of "exists
        # somewhere in the whole system, but not here".
        known_sources = await self._known_sources_for_root(root)
        for path in current_files:
            latest = await self.versions.latest_for_source(path.name)
            if latest is None:
                new_files.append(path)
            else:
                current_mtime = path.stat().st_mtime
                if float(latest.get("source_modified_time", 0)) < current_mtime:
                    updated_files.append(path)
                else:
                    unchanged_files.append(path)
        deleted_sources = sorted(known_sources - current_names)
        return FileChangeSet(new_files, updated_files, deleted_sources, unchanged_files)

    async def _known_sources_for_root(self, root: Path) -> set[str]:
        sources: set[str] = set()
        cursor = self.jobs.collection.find({"root": str(root)})
        async for job in cursor:
            sources.update(job.get("new_indexed") or [])
            sources.update(job.get("updated_indexed") or [])
        return sources


class IncrementalReindexRunner:
    """Re-indexes only what changed under a knowledge-base directory.

    Never rebuilds the whole vector database: new files are indexed, updated files
    are re-indexed as a new version, deleted files are soft-deleted (version history
    and audit trail are preserved), and untouched files are skipped entirely. Every
    run is recorded as an ``IndexingJob`` document so progress and failures for large
    batches are inspectable and re-runnable instead of being lost on a crash.
    """

    def __init__(
        self,
        pipeline: Any | None = None,
        planner: IncrementalIndexPlanner | None = None,
        jobs: IndexingJobRepository | None = None,
    ) -> None:
        from app.rag.pipeline import IndexingPipeline

        self.pipeline = pipeline or IndexingPipeline()
        self.planner = planner or IncrementalIndexPlanner()
        self.jobs = jobs or IndexingJobRepository()

    async def run(self, root: Path, allowed_suffixes: set[str] | None = None) -> dict[str, Any]:
        job_id = await self.jobs.insert({"root": str(root), "status": "running", "started_at": datetime.now(UTC)})
        return await self.run_for_job(job_id, root, allowed_suffixes)

    async def run_for_job(self, job_id: str, root: Path, allowed_suffixes: set[str] | None = None) -> dict[str, Any]:
        await self.jobs.update_by_id(job_id, {"status": "running", "started_at": datetime.now(UTC)})
        try:
            change_set = await self.planner.plan(root, allowed_suffixes or ALLOWED_UPLOAD_EXTENSIONS)
        except Exception as exc:
            log.exception("incremental_reindex_planning_failed", root=str(root))
            await self.jobs.update_by_id(job_id, {"status": "failed", "error": str(exc), "finished_at": datetime.now(UTC)})
            raise

        indexed: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []
        failed: list[dict[str, str]] = []

        for path in change_set.new_files:
            try:
                await self.pipeline.index_file(path)
                indexed.append(path.name)
            except Exception as exc:  # noqa: BLE001 - per-file isolation: one unreadable document must not abandon the rest of the re-index
                log.error("incremental_reindex_file_failed", file=path.name, error=str(exc))
                failed.append({"file": path.name, "error": str(exc)})

        for path in change_set.updated_files:
            try:
                await self.pipeline.index_file(path)
                updated.append(path.name)
            except Exception as exc:  # noqa: BLE001 - per-file isolation: one unreadable document must not abandon the rest of the re-index
                log.error("incremental_reindex_file_failed", file=path.name, error=str(exc))
                failed.append({"file": path.name, "error": str(exc)})

        for source in change_set.deleted_sources:
            try:
                await self.pipeline.deindex_source(source)
                deleted.append(source)
            except Exception as exc:  # noqa: BLE001 - per-source isolation: one failed de-index must not abandon the rest
                log.error("incremental_deindex_failed", source=source, error=str(exc))
                failed.append({"file": source, "error": str(exc)})

        summary: dict[str, Any] = {
            "new_indexed": indexed,
            "updated_indexed": updated,
            "deleted": deleted,
            "unchanged": [path.name for path in change_set.unchanged_files],
            "failed": failed,
            "status": "completed" if not failed else "completed_with_errors",
            "finished_at": datetime.now(UTC),
        }
        await self.jobs.update_by_id(job_id, summary)
        if indexed or updated or deleted:
            # Knowledge base content actually changed: stop serving any response
            # cached before this point rather than surgically finding affected
            # entries -- correctness (never serve stale legal information) over
            # cache efficiency.
            await response_cache.bump_generation()
        return {"job_id": job_id, **summary}


async def recover_pending_reindex_jobs(runner: "IncrementalReindexRunner | None" = None) -> int:
    """Startup recovery for `/admin/reindex` jobs (mirrors
    `kb_indexing_queue.recover_pending_jobs`, which only covers admin
    KB-*upload* staging, not this incremental-reindex path).

    `queue_reindex` records a job at status="queued" and hands it to
    FastAPI `BackgroundTasks`, which is in-process only -- a crash or
    restart before `run_for_job` reaches "completed"/"failed" leaves the job
    stuck at "queued" or "running" forever, with nothing to resume it. Safe
    to just re-run: `IncrementalIndexPlanner.plan` recomputes new/updated/
    deleted files from current mtimes on every call, so a file already
    indexed before the crash reads back as unchanged and is skipped, not
    re-indexed.
    """
    target = runner or IncrementalReindexRunner()
    cursor = target.jobs.collection.find({"status": {"$in": ["queued", "running"]}})
    stuck = [job async for job in cursor]
    recovered = 0
    for job in stuck:
        root = job.get("root")
        if not root or not Path(root).is_dir():
            await target.jobs.update_by_id(
                job["_id"],
                {
                    "status": "failed",
                    "error": "Recovery found no such root directory; the job cannot be resumed.",
                    "finished_at": datetime.now(UTC),
                },
            )
            continue
        try:
            await target.run_for_job(job["_id"], Path(root))
            recovered += 1
        except Exception as exc:  # noqa: BLE001 - one unresumable job must not block recovering the rest
            log.error("incremental_reindex_recovery_failed", job_id=job["_id"], error=str(exc))
    return recovered
