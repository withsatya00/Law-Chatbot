"""Regression tests for the Admin Knowledge Base Ingestion Pipeline
(`app/services/kb_ingestion_service.py`) -- duplicate detection, content-based
renaming, successful transfer, failed indexing, and staging cleanup.

Follows the same conventions as `test_document_service_ownership.py`: sync
`def test_...()` wrapping `asyncio.run(...)`, the shared `IndexingPipeline`
mocked out (it's the existing, separately-tested pipeline -- not exercised for
real here), and a real project-local scratch directory with manual cleanup
(this environment's `tmp_path` fixture isn't reliable, same reason
`test_bm25_index.py` avoids it). `DocumentLoader`/fingerprinting/renaming run
for real against plain `.txt` fixtures, since that's the actual new logic
under test.
"""

import asyncio
import shutil
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from app.core.config import settings
from app.core.exceptions import BadRequestError
from app.rag.types import LoadedDocument
from app.repositories.kb_staging import KnowledgeBaseStagingRepository
from app.services.kb_indexing_queue import KnowledgeBaseIndexingQueue, recover_pending_jobs
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService
from conftest import session_storage_root

# A project-local scratch dir rather than pytest's `tmp_path` fixture (this
# sandbox's OS temp directory isn't reliably writable here -- see
# `test_bm25_index.py`'s matching comment) -- but NOT under the repository's
# real `storage/` tree the way this used to be (`Path("storage") /
# "_test_kb_ingestion"`). `_kb_dirs()` below temporarily points
# `settings.archive_dir`/etc. at subdirectories of this root and restores
# them afterward; that restore is a manual try/finally, not `monkeypatch`,
# so a rare failure mode (an exception during teardown, an interrupted run)
# could in principle leave stale files behind. Rooting this under the
# session's OWN already-isolated temp directory (`conftest.py`'s
# `session_storage_root()`, proven reliable -- the whole suite's storage
# isolation depends on it) means that even a failed cleanup can never leave
# anything inside the real `storage/` tree, which is what
# `_real_storage_is_never_written` (in `conftest.py`) actually guards.
# Confirmed live: an earlier version of this file, scoped under `storage/`,
# intermittently left real files in `storage/archive/` across a full suite
# run.
_SCRATCH_ROOT = session_storage_root() / "_test_kb_ingestion"


class _FakeUploadFile:
    """Minimal stand-in for FastAPI's `UploadFile` -- `ingest` only ever
    touches `.filename` and awaits `.read(size)`, returning `b""` once
    exhausted (the same protocol `UploadFile.read` follows)."""

    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self._content = content
        self._exhausted = False

    async def read(self, _size: int) -> bytes:
        if self._exhausted:
            return b""
        self._exhausted = True
        return self._content


@contextmanager
def _kb_dirs():
    staging_dir = _SCRATCH_ROOT / "staging"
    kb_dir = _SCRATCH_ROOT / "kb"
    archive_dir = _SCRATCH_ROOT / "archive"
    review_dir = _SCRATCH_ROOT / "review"
    operations_dir = _SCRATCH_ROOT / "operations"
    for directory in (staging_dir, kb_dir, archive_dir, review_dir):
        directory.mkdir(parents=True, exist_ok=True)
    originals = (
        settings.kb_staging_dir,
        settings.knowledge_base_dir,
        settings.archive_dir,
        settings.kb_review_dir,
        settings.operations_output_dir,
    )
    settings.kb_staging_dir = staging_dir
    settings.knowledge_base_dir = kb_dir
    settings.archive_dir = archive_dir
    settings.kb_review_dir = review_dir
    settings.operations_output_dir = operations_dir
    try:
        yield staging_dir, kb_dir, archive_dir
    finally:
        (
            settings.kb_staging_dir,
            settings.knowledge_base_dir,
            settings.archive_dir,
            settings.kb_review_dir,
            settings.operations_output_dir,
        ) = originals
        shutil.rmtree(_SCRATCH_ROOT, ignore_errors=True)


def _service() -> KnowledgeBaseIngestionService:
    service = KnowledgeBaseIngestionService()
    service.versions = MagicMock()
    service.versions.find_by_hash = AsyncMock(return_value=None)
    service.versions.latest_for_source = AsyncMock(return_value=None)
    service.versions.update_by_id = AsyncMock(return_value=True)
    service.documents = MagicMock()
    service.documents.update_by_id = AsyncMock(return_value=True)
    service.staging = MagicMock()
    service.staging.STATUSES = KnowledgeBaseStagingRepository.STATUSES
    service.staging.insert = AsyncMock(return_value="staging-record-id")
    service.staging.claim_active = AsyncMock(return_value="staging-record-id")
    service.staging.release_active = AsyncMock(return_value=True)
    service.staging.update_by_id = AsyncMock(return_value=True)
    service.staging.find_by_status = AsyncMock(return_value=[])
    service.staging.find_by_statuses = AsyncMock(return_value=[])
    service.staging.find_stale_active = AsyncMock(return_value=[])
    service.staging.find_active_by_content_hash = AsyncMock(return_value=None)
    service.staging.find_duplicate_by_content_hash = AsyncMock(return_value=None)
    # Looked up by `_process` to recover any jurisdiction metadata recorded on
    # the ledger row at `stage()` time (see `kb_jurisdiction`/Phase 1); tests
    # in this file never set that up, so `None` here matches this suite's
    # pre-Phase-1 behavior (no jurisdiction metadata applied).
    service.staging.find_by_id = AsyncMock(return_value=None)
    service.pipeline = MagicMock()
    service.pipeline.index_file = AsyncMock(return_value=("doc-abc123", "english", [MagicMock(metadata={})]))
    return service


# ---- 0. Malware scanning ---------------------------------------------------
#
# This admin manual-upload route was previously the one path writing an
# untrusted file to disk with no scan at all -- `DocumentService.
# upload_and_index` (end-user chat attachments) and `KnowledgeBaseAutomation
# Service` (the automated fetch pipeline) both already scanned. See
# `KnowledgeBaseIngestionService.__init__`.


def test_a_flagged_upload_is_rejected_before_indexing_and_cleaned_up() -> None:
    with _kb_dirs() as (staging_dir, _kb_dir, _archive_dir):
        service = _service()
        service.scanner = AsyncMock()
        service.scanner.scan = AsyncMock(side_effect=ValueError("Malware test signature detected."))

        try:
            asyncio.run(service.ingest(_FakeUploadFile("invoice.pdf", b"malicious bytes")))
            raised = False
        except BadRequestError as exc:
            raised = True
            assert "rejected" in str(exc).lower()

        assert raised
        service.pipeline.index_file.assert_not_awaited()
        service.staging.claim_active.assert_not_awaited()
        # The staged copy must not survive a failed scan -- otherwise the
        # flagged bytes just sit on disk indefinitely.
        assert list(staging_dir.iterdir()) == []


def test_a_clean_upload_is_scanned_before_indexing() -> None:
    with _kb_dirs() as (_staging_dir, _kb_dir, _archive_dir):
        service = _service()
        service.scanner = AsyncMock()
        service.scanner.scan = AsyncMock(return_value=None)

        asyncio.run(service.ingest(_FakeUploadFile("clean.txt", b"perfectly ordinary content")))

        service.scanner.scan.assert_awaited_once()
        service.pipeline.index_file.assert_awaited_once()


# ---- 1. Duplicate detection ------------------------------------------------


def test_exact_hash_duplicate_is_rejected_without_calling_index_file() -> None:
    with _kb_dirs() as (staging_dir, _kb_dir, _archive_dir):
        service = _service()
        service.versions.find_by_hash = AsyncMock(return_value={"source_document": "Existing_Act.pdf", "_id": "v1"})

        result = asyncio.run(service.ingest(_FakeUploadFile("1827364.pdf", b"duplicate bytes")))

        assert result.status == "duplicate"
        service.pipeline.index_file.assert_not_awaited()
        assert list(staging_dir.iterdir()) == []


def test_near_duplicate_content_is_rejected_via_cached_fingerprint() -> None:
    # No file is dropped into kb_dir here -- the match must come from a
    # cached fingerprint on a prior `indexed` ledger row, never a directory
    # rescan (`_find_near_duplicate` no longer touches `knowledge_base_dir`).
    with _kb_dirs() as (staging_dir, _kb_dir, _archive_dir):
        service = _service()
        body = (
            "This agreement outlines terms and conditions for lease of residential "
            "property located under applicable law. "
        ) * 40
        cached_fingerprint = service._normalize_fingerprint(body)
        service.staging.find_by_status = AsyncMock(
            return_value=[{"generated_filename": "Existing_Lease_Agreement.txt", "fingerprint": cached_fingerprint}]
        )
        upload_content = (body + " Additional clause appended only in the new copy.").encode("utf-8")

        result = asyncio.run(service.ingest(_FakeUploadFile("998271.txt", upload_content)))

        assert result.status == "duplicate"
        assert result.reason is not None and "Existing_Lease_Agreement.txt" in result.reason
        service.pipeline.index_file.assert_not_awaited()
        service.staging.find_by_status.assert_awaited_once_with("indexed")
        assert list(staging_dir.iterdir()) == []


# ---- 2. Renaming ------------------------------------------------------------


def test_numeric_uploaded_filename_is_preserved_for_admin_visibility() -> None:
    with _kb_dirs() as (_staging_dir, _kb_dir, _archive_dir):
        service = _service()
        content = (
            b"Consumer Protection Act Manual\n\nThis manual explains consumer rights "
            b"and grievance redressal procedures under the Consumer Protection Act."
        )

        result = asyncio.run(service.ingest(_FakeUploadFile("5512783.txt", content)))

        assert result.status == "indexed"
        name = result.generated_filename
        assert name is not None
        assert name == "5512783.txt"


def test_generic_extracted_title_falls_back_to_a_meaningful_original_filename() -> None:
    service = _service()
    loaded = LoadedDocument(
        document_id="doc-1",
        filename="Bharat_Ka_Samvidhan_Hindi_English_Official_2026.pdf",
        text="",
    )

    name = service._generate_filename(loaded.filename)

    assert name == "Bharat_Ka_Samvidhan_Hindi_English_Official_2026.pdf"


def test_uuid_uploaded_filename_is_preserved_for_traceability() -> None:
    service = _service()

    name = service._generate_filename("3f9632b8d37342109dc146217d9bb8bd.pdf")

    assert name == "3f9632b8d37342109dc146217d9bb8bd.pdf"


def test_same_filename_with_different_content_gets_visible_numeric_suffix() -> None:
    with _kb_dirs() as (_staging_dir, kb_dir, _archive_dir):
        (kb_dir / "Official_Act.pdf").write_bytes(b"existing different document")

        name = _service()._generate_filename("Official_Act.pdf")

        assert name == "Official_Act_2.pdf"


# ---- 3. Successful transfer + 5. Upload cleanup ----------------------------


def test_successful_indexing_moves_file_into_knowledge_base_and_clears_staging() -> None:
    with _kb_dirs() as (staging_dir, kb_dir, _archive_dir):
        service = _service()
        content = b"Motor Vehicles Act guidance document explaining registration and licensing rules for vehicle owners."

        result = asyncio.run(service.ingest(_FakeUploadFile("998271.txt", content)))

        assert result.status == "indexed"
        assert (kb_dir / result.generated_filename).exists()
        assert not (staging_dir / result.generated_filename).exists()
        assert list(staging_dir.iterdir()) == []


# ---- Fingerprint caching -----------------------------------------------------


def test_successful_indexing_caches_the_fingerprint_on_the_ledger_row() -> None:
    with _kb_dirs() as (_staging_dir, _kb_dir, _archive_dir):
        service = _service()
        content = b"Motor Vehicles Act guidance document explaining registration and licensing rules for vehicle owners."

        result = asyncio.run(service.ingest(_FakeUploadFile("998271.txt", content)))

        assert result.status == "indexed"
        service.staging.insert.assert_awaited_once()
        (saved_fields,) = service.staging.insert.await_args.args
        expected_fingerprint = service._normalize_fingerprint(content.decode("utf-8"))
        assert saved_fields["fingerprint"] == expected_fingerprint
        assert saved_fields["fingerprint"]


# ---- 4. Failed indexing -----------------------------------------------------


def test_failed_indexing_moves_the_file_to_the_failed_review_queue() -> None:
    with _kb_dirs() as (staging_dir, kb_dir, _archive_dir):
        service = _service()
        service.pipeline.index_file = AsyncMock(side_effect=RuntimeError("embedding provider unavailable"))
        content = b"Cyber Crime Complaint Guidelines for reporting online fraud to the cyber cell."

        result = asyncio.run(service.ingest(_FakeUploadFile("1827364.txt", content)))

        assert result.status == "failed"
        assert result.generated_filename is not None
        # Staging holds in-flight work only, so a file that failed weeks ago
        # can no longer masquerade as one still being processed.
        assert list(staging_dir.iterdir()) == []
        assert [p.name for p in service.review_failed_dir().iterdir()] == ["1827364.txt"]
        assert list(kb_dir.iterdir()) == []


# ---- 5. Upload cleanup (duplicate path) ------------------------------------


def test_duplicate_rejection_cleans_up_the_staged_copy() -> None:
    with _kb_dirs() as (staging_dir, _kb_dir, _archive_dir):
        service = _service()
        service.versions.find_by_hash = AsyncMock(return_value={"source_document": "Existing.pdf"})

        asyncio.run(service.ingest(_FakeUploadFile("1827364.pdf", b"duplicate bytes")))

        assert list(staging_dir.iterdir()) == []


# ---- 6. Archive movement -----------------------------------------------------


def test_duplicate_is_archived_not_deleted_preserving_original_filename() -> None:
    with _kb_dirs() as (staging_dir, _kb_dir, archive_dir):
        service = _service()
        service.versions.find_by_hash = AsyncMock(return_value={"source_document": "Existing.pdf"})

        asyncio.run(service.ingest(_FakeUploadFile("Original_Name.pdf", b"duplicate bytes")))

        assert list(staging_dir.iterdir()) == []
        archived = archive_dir / "Original_Name.pdf"
        assert archived.exists()
        assert archived.read_bytes() == b"duplicate bytes"


def test_duplicate_archival_never_touches_the_repositorys_real_storage_tree() -> None:
    """Direct, localized proof for the dedup/archive path specifically --
    `conftest.py`'s session-scoped `_real_storage_is_never_written` already
    guards the whole suite, but only compares one snapshot taken at session
    start against one taken at session end, which can be confused by
    anything else writing into the real `storage/` tree during that whole
    multi-minute window (in particular: a live dev server process running
    on this machine independently syncs/archives official KB sources into
    this exact tree on its own schedule -- confirmed running, separately
    from any test, while investigating this). Snapshotting immediately
    before and after ONE call, inside ONE test, is far less likely to catch
    an unrelated external write and gives fast, precise attribution if this
    ever regresses.
    """
    real_archive = Path(__file__).resolve().parent.parent / "storage" / "archive"
    before = {p for p in real_archive.rglob("*") if p.is_file()} if real_archive.exists() else set()

    with _kb_dirs() as (_staging_dir, _kb_dir, archive_dir):
        service = _service()
        service.versions.find_by_hash = AsyncMock(return_value={"source_document": "Existing.pdf"})
        asyncio.run(service.ingest(_FakeUploadFile("Real_Storage_Probe.pdf", b"duplicate bytes")))
        # The duplicate really was archived -- just into the isolated
        # `_kb_dirs()` root, not the real tree checked below.
        assert (archive_dir / "Real_Storage_Probe.pdf").exists()

    after = {p for p in real_archive.rglob("*") if p.is_file()} if real_archive.exists() else set()
    assert after == before, f"real storage/archive changed: {sorted(str(p) for p in after - before)}"


def test_the_same_duplicate_content_is_archived_only_once() -> None:
    """Confirmed live: a source repeatedly re-discovered by automation/sync
    (kb_automation/official_source_sync run on an interval and encounter
    the SAME official document again every cycle) kept being archived as a
    "new" duplicate every single time -- 150+ redundant ~1.3MB copies of a
    handful of official documents accumulated in real `storage/archive`
    over time, because `_reject_duplicate` unconditionally called `_move`
    (which always reserves a fresh, collision-safe filename) with no check
    for "have we already preserved this exact content before". This proves
    the fix: rejecting the SAME content hash a second time reuses the
    FIRST archived copy's path rather than creating another one.
    """
    with _kb_dirs() as (staging_dir, _kb_dir, archive_dir):
        service = _service()
        service.versions.find_by_hash = AsyncMock(return_value={"source_document": "Existing.pdf"})

        asyncio.run(service.ingest(_FakeUploadFile("Official_Gazette.pdf", b"same official bytes")))
        first_archived = list(archive_dir.iterdir())
        assert len(first_archived) == 1

        # `find_duplicate_by_content_hash` (real `KnowledgeBaseStagingRepository`
        # behavior) would now find the row `_save_staging` just wrote for the
        # first rejection -- reproduced here directly, since `service.staging`
        # is a mock with no real backing collection.
        service.staging.find_duplicate_by_content_hash = AsyncMock(
            return_value={"archived_path": str(first_archived[0])}
        )

        asyncio.run(service.ingest(_FakeUploadFile("Official_Gazette_again.pdf", b"same official bytes")))

        assert list(staging_dir.iterdir()) == [], "the second attempt's staged copy must not be left behind"
        assert list(archive_dir.iterdir()) == first_archived, (
            "a second rejection of the SAME content must not create another archive copy"
        )


# ---- 7. Background queue + automatic indexing --------------------------------


def test_stage_then_process_indexes_automatically_via_the_same_ledger_row() -> None:
    with _kb_dirs() as (_staging_dir, kb_dir, _archive_dir):
        service = _service()
        service.staging.update_by_id = AsyncMock(return_value=True)
        content = b"Consumer Protection Act guidance for grievance redressal procedures."

        staged = asyncio.run(service.stage(_FakeUploadFile("77123.txt", content)))
        staging_id = staged.staging_id
        result = asyncio.run(service.process(staging_id, staged.staged_path, staged.original_filename))

        assert staged.claimed is True
        assert staging_id == "staging-record-id"
        assert result.status == "indexed"
        assert (kb_dir / result.generated_filename).exists()
        # pending -> processing on `update_by_id`, then the terminal
        # `release_active` write that records "indexed" and drops the claim.
        assert service.staging.update_by_id.await_args_list[0].args[1]["status"] == "processing"
        assert service.staging.release_active.await_args_list[-1].args[1]["status"] == "indexed"


def test_queue_processes_uploads_sequentially_in_arrival_order() -> None:
    calls: list[str] = []

    class _FakeService:
        async def process(self, staging_id: str, staged_path: Path, original_filename: str) -> None:
            calls.append(staging_id)

    async def _drain() -> None:
        queue = KnowledgeBaseIndexingQueue(service=_FakeService())
        queue.enqueue("job-1", Path("a.txt"), "a.txt")
        queue.enqueue("job-2", Path("b.txt"), "b.txt")
        queue.enqueue("job-3", Path("c.txt"), "c.txt")
        await queue._queue.join()

    asyncio.run(_drain())

    assert calls == ["job-1", "job-2", "job-3"]


def test_queue_survives_a_failed_job_and_keeps_processing_subsequent_ones_in_order() -> None:
    """Phase 3A "KB Indexing Queue Resilience" regression test: before this
    fix, an exception escaping `process()` propagated out of `_run()`'s
    consumer loop, permanently ending that task -- every job enqueued after
    the failure was accepted but never processed until a process restart.
    """
    calls: list[str] = []

    class _FlakyService:
        async def process(self, staging_id: str, staged_path: Path, original_filename: str) -> None:
            if staging_id == "job-2":
                raise RuntimeError("simulated ingestion failure")
            calls.append(staging_id)

    async def _drain() -> None:
        queue = KnowledgeBaseIndexingQueue(service=_FlakyService())
        queue.enqueue("job-1", Path("a.txt"), "a.txt")
        queue.enqueue("job-2", Path("b.txt"), "b.txt")  # raises -- must not kill the worker
        queue.enqueue("job-3", Path("c.txt"), "c.txt")
        await queue._queue.join()

    asyncio.run(_drain())

    # job-2's failure is swallowed (logged, not silently dropped -- covered by
    # inspection of `_run`'s except block) but must not prevent job-1 (already
    # done before the failure) or job-3 (enqueued after) from completing, and
    # ordering among the jobs that DID succeed must be preserved.
    assert calls == ["job-1", "job-3"]


# ---- 8. Dashboard status ------------------------------------------------------


def test_dashboard_status_counts_reads_from_the_staging_ledger() -> None:
    class _FakeCollection:
        def __init__(self, counts: dict) -> None:
            self._counts = counts

        async def count_documents(self, query: dict) -> int:
            if not query:
                return sum(self._counts.values())
            return self._counts.get(query.get("status"), 0)

    fake = _FakeCollection(
        {"indexed": 5, "processing": 1, "pending": 2, "duplicate": 3, "failed": 4, "needs_review": 6}
    )

    counts = asyncio.run(KnowledgeBaseStagingRepository().status_counts(fake))

    assert counts == {
        "pending": 2,
        "processing": 1,
        "indexed": 5,
        "duplicate": 3,
        "failed": 4,
        "needs_review": 6,
        "total": 21,
    }


# ---- 9. Queue recovery --------------------------------------------------------


class _FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, Path, str]] = []

    def enqueue(self, staging_id: str, staged_path: Path, original_filename: str) -> None:
        self.enqueued.append((staging_id, staged_path, original_filename))


class _FakeStagingRepo:
    def __init__(self, records: list[dict]) -> None:
        self.released: list[tuple[str, dict]] = []
        self._records = records

    async def release_active(self, item_id: str, updates: dict) -> bool:
        self.released.append((item_id, updates))
        return True

    async def find_by_status(self, status: str) -> list[dict]:
        return [record for record in self._records if record.get("status") == status]


def test_startup_recovery_requeues_stuck_pending_and_processing_records_without_reupload() -> None:
    with _kb_dirs() as (staging_dir, _kb_dir, _archive_dir):
        pending_path = staging_dir / "Pending_Doc.txt"
        pending_path.write_text("pending content", encoding="utf-8")
        processing_path = staging_dir / "Processing_Doc.txt"
        processing_path.write_text("processing content", encoding="utf-8")
        repo = _FakeStagingRepo(
            [
                {"_id": "job-1", "original_filename": "Pending_Doc.txt", "staged_path": str(pending_path), "status": "pending"},
                {
                    "_id": "job-2",
                    "original_filename": "Processing_Doc.txt",
                    "staged_path": str(processing_path),
                    "status": "processing",
                },
            ]
        )
        queue = _FakeQueue()

        count = asyncio.run(recover_pending_jobs(queue, repo))

        assert count == 2
        assert queue.enqueued == [
            ("job-1", pending_path, "Pending_Doc.txt"),
            ("job-2", processing_path, "Processing_Doc.txt"),
        ]


def test_recovery_closes_records_whose_staged_file_no_longer_exists() -> None:
    """A tracked job whose file is gone can never finish. It used to be
    skipped and left at `processing` forever -- invisible to the queue and
    unexplained in every admin view."""
    with _kb_dirs() as (staging_dir, _kb_dir, _archive_dir):
        missing_path = staging_dir / "Gone.txt"
        repo = _FakeStagingRepo(
            [{"_id": "job-99", "original_filename": "Gone.txt", "staged_path": str(missing_path), "status": "processing"}]
        )
        queue = _FakeQueue()

        count = asyncio.run(recover_pending_jobs(queue, repo))

        assert count == 0
        assert queue.enqueued == []
        assert repo.released[0][0] == "job-99"
        assert repo.released[0][1]["status"] == "needs_review"
        assert repo.released[0][1]["file_exists"] is False


# ---- 10. Existing-uploads backfill (privacy-gated) ---------------------------


def test_backfill_refuses_to_run_without_explicit_admin_authorization() -> None:
    with _kb_dirs() as (_staging_dir, _kb_dir, _archive_dir):
        uploads_dir = _SCRATCH_ROOT / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        (uploads_dir / "private.pdf").write_bytes(b"a private user upload")

        service = _service()
        try:
            asyncio.run(service.backfill_from_uploads(uploads_dir))
        except BadRequestError as exc:
            assert "authorization" in exc.message.lower()
        else:  # pragma: no cover - the guard is the point of the test
            raise AssertionError("backfill ran without authorization")
        service.pipeline.index_file.assert_not_awaited()


def test_authorized_backfill_quarantines_uploads_as_needs_review_and_never_indexes() -> None:
    with _kb_dirs() as (staging_dir, kb_dir, _archive_dir):
        uploads_dir = _SCRATCH_ROOT / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        (uploads_dir / "private.pdf").write_bytes(b"a private user upload")

        service = _service()
        service.staging.find_by_content_hash = AsyncMock(return_value=None)

        result = asyncio.run(service.backfill_from_uploads(uploads_dir, authorized=True))

        assert result == {
            "files_processed": 1,
            "needs_review_count": 1,
            "skipped_known_count": 0,
            "indexed_count": 0,
            "duplicate_count": 0,
            "failed_count": 0,
        }
        # The private document is quarantined for review, never indexed, and
        # the user's own copy is left exactly where it was.
        assert [p.name for p in service.review_pending_dir().iterdir()] == ["private.pdf"]
        assert (uploads_dir / "private.pdf").exists()
        assert list(kb_dir.iterdir()) == []
        assert list(staging_dir.iterdir()) == []
        service.pipeline.index_file.assert_not_awaited()
        assert service.staging.insert.await_args.args[0]["status"] == "needs_review"
