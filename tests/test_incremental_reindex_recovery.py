"""Startup recovery for `/admin/reindex` jobs interrupted by a crash/restart.

`queue_reindex` (app/services/admin_operations.py) hands the job to FastAPI
`BackgroundTasks`, which is in-process only, so a job stuck at status
"queued" or "running" has nothing left to resume it after a restart unless
something re-runs it -- that "something" is `recover_pending_reindex_jobs`.
"""
import asyncio
from pathlib import Path
from typing import Any

from app.rag.incremental import (
    FileChangeSet,
    IncrementalReindexRunner,
    recover_pending_reindex_jobs,
)


class _FakeCursor:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = list(docs)

    def __aiter__(self) -> "_FakeCursor":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._docs:
            raise StopAsyncIteration
        return self._docs.pop(0)


class _FakeCollection:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def find(self, query: dict[str, Any]) -> _FakeCursor:
        wanted = set(query["status"]["$in"])
        return _FakeCursor([doc for doc in self.docs if doc["status"] in wanted])


class _FakeJobs:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.by_id = {doc["_id"]: doc for doc in docs}
        self.collection = _FakeCollection(list(docs))

    async def update_by_id(self, job_id: str, updates: dict[str, Any]) -> bool:
        self.by_id[job_id].update(updates)
        return True


class _EmptyPlanner:
    """Reports nothing changed, so `run_for_job` never touches a real pipeline."""

    async def plan(self, root: Path, allowed_suffixes: set[str]) -> FileChangeSet:
        return FileChangeSet(new_files=[], updated_files=[], deleted_sources=[], unchanged_files=[])


def test_recover_pending_reindex_jobs_reruns_queued_and_running_jobs(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    root.mkdir()
    jobs = _FakeJobs([
        {"_id": "job-queued", "status": "queued", "root": str(root)},
        {"_id": "job-running", "status": "running", "root": str(root)},
        {"_id": "job-done", "status": "completed", "root": str(root)},
    ])
    runner = IncrementalReindexRunner(pipeline=object(), planner=_EmptyPlanner(), jobs=jobs)

    recovered = asyncio.run(recover_pending_reindex_jobs(runner))

    assert recovered == 2
    assert jobs.by_id["job-queued"]["status"] == "completed"
    assert jobs.by_id["job-running"]["status"] == "completed"
    # An already-finished job is left alone -- recovery only touches
    # queued/running rows, never re-runs completed work.
    assert jobs.by_id["job-done"]["status"] == "completed"
    assert "new_indexed" not in jobs.by_id["job-done"]


def test_recover_pending_reindex_jobs_fails_closed_when_root_is_gone(tmp_path: Path) -> None:
    missing_root = tmp_path / "no-longer-here"
    jobs = _FakeJobs([{"_id": "job-orphaned", "status": "running", "root": str(missing_root)}])
    runner = IncrementalReindexRunner(pipeline=object(), planner=_EmptyPlanner(), jobs=jobs)

    recovered = asyncio.run(recover_pending_reindex_jobs(runner))

    assert recovered == 0
    assert jobs.by_id["job-orphaned"]["status"] == "failed"


def test_recover_pending_reindex_jobs_is_a_noop_with_nothing_stuck() -> None:
    jobs = _FakeJobs([])
    runner = IncrementalReindexRunner(pipeline=object(), planner=_EmptyPlanner(), jobs=jobs)

    recovered = asyncio.run(recover_pending_reindex_jobs(runner))

    assert recovered == 0
