"""KB staging lifecycle: idempotent claiming, accurate paths, safe (dry-run
first) reconciliation, the review queue, and the privacy boundary around
`storage/uploads`.

Same conventions as `test_kb_ingestion_service.py`: sync `def test_...()`
wrapping `asyncio.run(...)`, the shared `IndexingPipeline` mocked out, and a
project-local scratch directory with manual cleanup. The staging ledger is
backed by `_FakeStagingCollection` below -- a minimal in-memory stand-in that
enforces the same sparse-unique `active_key` constraint Mongo does, so the
"only one active job per content" rule is genuinely exercised rather than
mocked away.
"""

import asyncio
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import DuplicateKeyError

import app.main as main_module
from app.core.config import settings
from app.core.exceptions import BadRequestError
from app.repositories.kb_staging import KnowledgeBaseStagingRepository
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService

_SCRATCH_ROOT = Path("storage") / "_test_kb_reconciliation"


class _FakeUploadFile:
    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self._content = content
        self._exhausted = False

    async def read(self, _size: int) -> bytes:
        if self._exhausted:
            return b""
        self._exhausted = True
        return self._content


class _FakeStagingCollection:
    """In-memory `kb_staging_records`. Enforces the sparse unique index on
    `active_key` -- the constraint that makes "one active ingestion job per
    content" hold across processes."""

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []

    def _matches(self, doc: dict[str, Any], query: dict[str, Any]) -> bool:
        for key, expected in query.items():
            value = doc.get(key)
            if isinstance(expected, dict) and "$in" in expected:
                if value not in expected["$in"]:
                    return False
            elif isinstance(expected, dict) and "$exists" in expected:
                if (key in doc) != expected["$exists"]:
                    return False
            elif value != expected:
                return False
        return True

    async def insert_one(self, document: dict[str, Any]) -> Any:
        active = document.get("active_key")
        if active is not None and any(d.get("active_key") == active for d in self.docs):
            raise DuplicateKeyError("duplicate active_key")
        self.docs.append(dict(document))
        return MagicMock(inserted_id=document["_id"])

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        return next((dict(d) for d in self.docs if self._matches(d, query)), None)

    def find(self, query: dict[str, Any], _projection: Any = None) -> Any:
        matches = [dict(d) for d in self.docs if self._matches(d, query)]

        class _Cursor:
            def __aiter__(self) -> Any:
                async def _gen() -> Any:
                    for item in matches:
                        yield item

                return _gen()

            def sort(self, *_args: Any) -> Any:
                return self

        return _Cursor()

    async def update_one(self, query: dict[str, Any], update: dict[str, Any]) -> Any:
        for doc in self.docs:
            if self._matches(doc, query):
                doc.update(update.get("$set", {}))
                for key in update.get("$unset", {}):
                    doc.pop(key, None)
                return MagicMock(modified_count=1)
        return MagicMock(modified_count=0)

    async def count_documents(self, query: dict[str, Any]) -> int:
        return sum(1 for d in self.docs if self._matches(d, query))


class _FakeStagingRepository(KnowledgeBaseStagingRepository):
    def __init__(self) -> None:
        self._collection = _FakeStagingCollection()

    @property
    def collection(self) -> Any:  # type: ignore[override]
        return self._collection


@contextmanager
def _kb_dirs():
    staging = _SCRATCH_ROOT / "staging"
    kb = _SCRATCH_ROOT / "kb"
    archive = _SCRATCH_ROOT / "archive"
    review = _SCRATCH_ROOT / "review"
    operations = _SCRATCH_ROOT / "operations"
    for directory in (staging, kb, archive, review):
        directory.mkdir(parents=True, exist_ok=True)
    originals = (
        settings.kb_staging_dir,
        settings.knowledge_base_dir,
        settings.archive_dir,
        settings.kb_review_dir,
        settings.operations_output_dir,
    )
    settings.kb_staging_dir = staging
    settings.knowledge_base_dir = kb
    settings.archive_dir = archive
    settings.kb_review_dir = review
    settings.operations_output_dir = operations
    try:
        yield staging, kb, archive, review
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
    service.staging = _FakeStagingRepository()
    service.versions = MagicMock()
    service.versions.find_by_hash = AsyncMock(return_value=None)
    service.versions.latest_for_source = AsyncMock(return_value=None)
    service.versions.update_by_id = AsyncMock(return_value=True)
    service.documents = MagicMock()
    service.documents.update_by_id = AsyncMock(return_value=True)
    service.pipeline = MagicMock()
    service.pipeline.index_file = AsyncMock(return_value=("doc-1", "english", [MagicMock(metadata={})]))
    return service


def _records(service: KnowledgeBaseIngestionService) -> list[dict[str, Any]]:
    return service.staging.collection.docs  # type: ignore[attr-defined]


# ---- 1. Startup does not touch the private uploads directory ------------------


def test_startup_lifespan_has_no_uploads_backfill_hook() -> None:
    """The API used to fire `_reconcile_kb_uploads` on every boot, which
    scanned every PDF in `storage/uploads` (620 of them here) and indexed them
    into the shared Knowledge Base. Nothing on the startup path may reference
    the uploads directory any more."""
    source = Path("app/main.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))

    assert not hasattr(main_module, "_reconcile_kb_uploads")
    assert "backfill_from_uploads" not in code
    assert "upload_storage_dir" not in code
    assert "KnowledgeBaseIngestionService" not in code
    # Genuinely tracked jobs are still recovered.
    assert "recover_pending_jobs" in code


# ---- 2. One active job per content, across processes --------------------------


def test_two_concurrent_submissions_of_identical_bytes_create_one_active_job() -> None:
    with _kb_dirs() as (staging, _kb, archive, _review):
        service = _service()
        content = b"Identical bytes submitted twice at the same moment."

        async def _both() -> tuple[Any, Any]:
            return await asyncio.gather(
                service.stage(_FakeUploadFile("a.txt", content)),
                service.stage(_FakeUploadFile("b.txt", content)),
            )

        first, second = asyncio.run(_both())

        claimed = [outcome for outcome in (first, second) if outcome.claimed]
        assert len(claimed) == 1
        active = [r for r in _records(service) if r["status"] in ("pending", "processing")]
        assert len(active) == 1
        # The losing attempt is preserved as history, not silently dropped,
        # and its bytes are archived rather than deleted.
        assert len([r for r in _records(service) if r["status"] == "duplicate"]) == 1
        assert len(list(archive.iterdir())) == 1
        assert len(list(staging.iterdir())) == 1
        # The hash is known before the first row exists -- never null.
        assert all(r["content_hash"] for r in _records(service))


def test_a_finished_job_releases_the_claim_so_the_content_can_be_staged_again() -> None:
    with _kb_dirs() as (_staging, _kb, _archive, _review):
        service = _service()
        content = b"Consumer Protection Act guidance for grievance redressal."

        staged = asyncio.run(service.stage(_FakeUploadFile("first.txt", content)))
        asyncio.run(service.process(staged.staging_id, staged.staged_path, staged.original_filename))

        row = _records(service)[0]
        assert row["status"] == "indexed"
        assert "active_key" not in row


# ---- 3-5. Terminal outcomes leave nothing in staging ---------------------------


def test_successful_indexing_leaves_no_file_in_staging() -> None:
    with _kb_dirs() as (staging, kb, _archive, _review):
        service = _service()
        staged = asyncio.run(service.stage(_FakeUploadFile("77123.txt", b"Legal Notice for recovery of dues.")))

        result = asyncio.run(service.process(staged.staging_id, staged.staged_path, staged.original_filename))

        assert result.status == "indexed"
        assert list(staging.iterdir()) == []
        assert (kb / result.generated_filename).exists()
        row = _records(service)[0]
        assert row["current_path"] == str(kb / result.generated_filename)
        assert row["file_exists"] is True
        assert row["document_id"] == "doc-1"


def test_duplicate_files_move_to_the_archive() -> None:
    with _kb_dirs() as (staging, _kb, archive, _review):
        service = _service()
        service.versions.find_by_hash = AsyncMock(return_value={"source_document": "Existing.pdf"})
        staged = asyncio.run(service.stage(_FakeUploadFile("copy.txt", b"already indexed content")))

        result = asyncio.run(service.process(staged.staging_id, staged.staged_path, staged.original_filename))

        assert result.status == "duplicate"
        assert list(staging.iterdir()) == []
        assert [p.name for p in archive.iterdir()] == ["copy.txt"]
        assert _records(service)[0]["current_path"] == str(archive / "copy.txt")


def test_failed_files_move_to_the_failed_review_directory() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        service.pipeline.index_file = AsyncMock(side_effect=RuntimeError("Stream has ended unexpectedly"))
        staged = asyncio.run(service.stage(_FakeUploadFile("broken.txt", b"Cyber Crime Complaint about fraud.")))

        result = asyncio.run(service.process(staged.staging_id, staged.staged_path, staged.original_filename))

        assert result.status == "failed"
        assert list(staging.iterdir()) == []
        assert [p.name for p in service.review_failed_dir().iterdir()] == ["broken.txt"]
        row = _records(service)[0]
        assert row["status"] == "failed"
        assert row["current_path"] == str(service.review_failed_dir() / "broken.txt")


# ---- 6. Orphans need review, and are never auto-indexed ------------------------


def test_unknown_orphan_files_become_needs_review_and_are_not_auto_indexed() -> None:
    with _kb_dirs() as (staging, kb, _archive, _review):
        service = _service()
        (staging / "mystery.txt").write_bytes(b"a file nobody has a ledger row for")

        asyncio.run(service.reconcile_staging(apply=True))

        assert list(staging.iterdir()) == []
        assert [p.name for p in service.review_pending_dir().iterdir()] == ["mystery.txt"]
        assert [r["status"] for r in _records(service)] == ["needs_review"]
        assert list(kb.iterdir()) == []
        service.pipeline.index_file.assert_not_awaited()


def test_needs_review_reaches_the_knowledge_base_only_via_explicit_approval() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        (staging / "mystery.txt").write_bytes(b"An Affidavit sworn before a notary public.")
        asyncio.run(service.reconcile_staging(apply=True))
        record = _records(service)[0]

        queue = MagicMock()
        with pytest.MonkeyPatch.context() as patch:
            import app.services.kb_indexing_queue as queue_module

            patch.setattr(queue_module, "kb_indexing_queue", queue)
            result = asyncio.run(service.approve_needs_review(record["_id"]))

        assert result["status"] == "pending"
        assert queue.enqueue.call_count == 1
        # Approval moves the file back into staging and says so in the ledger.
        assert [p.name for p in staging.iterdir()] == ["mystery.txt"]
        assert _records(service)[0]["current_path"] == str(staging / "mystery.txt")


def test_a_document_that_is_not_needs_review_cannot_be_approved() -> None:
    with _kb_dirs() as (_staging, _kb, _archive, _review):
        service = _service()
        staged = asyncio.run(service.stage(_FakeUploadFile("x.txt", b"An RTI Application under the Act.")))
        asyncio.run(service.process(staged.staging_id, staged.staged_path, staged.original_filename))

        with pytest.raises(BadRequestError):
            asyncio.run(service.approve_needs_review(staged.staging_id))


# ---- 7. Paths stay accurate ----------------------------------------------------


def test_rename_updates_the_ledger_current_path_in_the_same_flow() -> None:
    with _kb_dirs() as (staging, kb, _archive, _review):
        service = _service()
        recorded: list[str] = []
        original_update = service.staging.update_by_id

        async def _spy(item_id: str, updates: dict[str, Any]) -> bool:
            if updates.get("current_path"):
                recorded.append(updates["current_path"])
            return await original_update(item_id, updates)

        service.staging.update_by_id = _spy  # type: ignore[method-assign]
        staged = asyncio.run(service.stage(_FakeUploadFile("99999.txt", b"Power Of Attorney granted to the agent.")))

        result = asyncio.run(service.process(staged.staging_id, staged.staged_path, staged.original_filename))

        renamed = str(staging / result.generated_filename)
        assert renamed in recorded  # the rename was recorded, not just the move
        row = _records(service)[0]
        assert row["current_path"] == str(kb / result.generated_filename)
        assert row["staged_path"] == row["current_path"]


# ---- 8-10. Reconciliation safety ------------------------------------------------


def test_dry_run_changes_neither_files_nor_records() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        (staging / "orphan.txt").write_bytes(b"untracked staging file")

        report = asyncio.run(service.reconcile_staging())

        assert report["mode"] == "dry_run"
        assert report["applied"] is False
        assert report["files_scanned"] == 1
        assert report["needs_review"] == 1
        assert report["orphaned"] == 1
        move = report["moves"][0]
        assert move["source"] == str(staging / "orphan.txt")
        assert move["proposed_destination"] == str(service.review_pending_dir() / "orphan.txt")
        # Nothing moved, nothing written.
        assert [p.name for p in staging.iterdir()] == ["orphan.txt"]
        assert _records(service) == []
        assert not settings.operations_output_dir.exists()


def test_apply_requires_explicit_confirmation_and_writes_a_manifest() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        (staging / "orphan.txt").write_bytes(b"untracked staging file")

        asyncio.run(service.reconcile_staging())
        assert [p.name for p in staging.iterdir()] == ["orphan.txt"]  # default did nothing

        # A mismatch between the reviewed count and reality refuses to apply.
        with pytest.raises(BadRequestError):
            asyncio.run(service.reconcile_staging(apply=True, expected_total=99))
        assert [p.name for p in staging.iterdir()] == ["orphan.txt"]

        applied = asyncio.run(service.reconcile_staging(apply=True, expected_total=1))

        assert applied["applied"] is True
        manifest = Path(applied["manifest_path"])
        assert manifest.exists()
        body = manifest.read_text(encoding="utf-8")
        assert "orphan.txt" in body and "content_hash" in body and "previous_status" in body
        assert list(staging.iterdir()) == []


def test_reconciliation_is_idempotent_when_run_twice() -> None:
    with _kb_dirs() as (staging, kb, archive, _review):
        service = _service()
        kb_file = kb / "Already_Indexed.txt"
        kb_file.write_bytes(b"content already in the knowledge base")
        (staging / "stray_copy.txt").write_bytes(b"content already in the knowledge base")
        (staging / "orphan.txt").write_bytes(b"never processed")

        first = asyncio.run(service.reconcile_staging(apply=True))
        records_after_first = len(_records(service))
        second = asyncio.run(service.reconcile_staging(apply=True))

        assert first["duplicates"] == 1
        assert first["needs_review"] == 1
        assert second["files_scanned"] == 0
        assert second["moved"] == 0
        assert len(_records(service)) == records_after_first
        # Nothing terminal is left in staging, and the KB file is untouched.
        assert list(staging.iterdir()) == []
        assert [p.name for p in archive.iterdir()] == ["stray_copy.txt"]
        assert kb_file.read_bytes() == b"content already in the knowledge base"


def test_reconciliation_never_deletes_a_document() -> None:
    with _kb_dirs() as (staging, kb, _archive, _review):
        service = _service()
        (kb / "Indexed.txt").write_bytes(b"in the kb")
        (staging / "dup.txt").write_bytes(b"in the kb")
        (staging / "orphan.txt").write_bytes(b"unknown")

        asyncio.run(service.reconcile_staging(apply=True))

        surviving = sorted(
            p.name for p in _SCRATCH_ROOT.rglob("*") if p.is_file() and p.suffix == ".txt"
        )
        assert surviving == ["Indexed.txt", "dup.txt", "orphan.txt"]


# ---- 11. Missing-path processing records cannot remain processing ---------------


def test_processing_record_with_a_missing_file_cannot_stay_processing() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        asyncio.run(
            service.staging.claim_active(
                {
                    "original_filename": "vanished.txt",
                    "status": "processing",
                    "content_hash": "abc123",
                    "current_path": str(staging / "vanished.txt"),
                }
            )
        )

        swept = asyncio.run(service.sweep_stale_active())

        assert swept == 1
        row = _records(service)[0]
        assert row["status"] == "needs_review"
        assert row["file_exists"] is False
        assert "active_key" not in row  # the claim is released with it


def test_dry_run_reports_stale_processing_records_without_changing_them() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        asyncio.run(
            service.staging.claim_active(
                {
                    "original_filename": "vanished.txt",
                    "status": "processing",
                    "content_hash": "abc123",
                    "current_path": str(staging / "vanished.txt"),
                }
            )
        )

        report = asyncio.run(service.reconcile_staging())

        assert report["stale_processing_records"] == 1
        assert report["missing_ledger_paths"] == 1
        assert _records(service)[0]["status"] == "processing"


# ---- 15. The privacy boundary ---------------------------------------------------


def test_private_uploads_never_enter_the_knowledge_base_automatically() -> None:
    with _kb_dirs() as (staging, kb, _archive, _review):
        uploads = _SCRATCH_ROOT / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        (uploads / "private.pdf").write_bytes(b"a user's own document")
        service = _service()

        with pytest.raises(BadRequestError):
            asyncio.run(service.backfill_from_uploads(uploads))

        result = asyncio.run(service.backfill_from_uploads(uploads, authorized=True))

        assert result["indexed_count"] == 0
        assert result["needs_review_count"] == 1
        assert [r["status"] for r in _records(service)] == ["needs_review"]
        assert list(kb.iterdir()) == []
        assert list(staging.iterdir()) == []
        assert (uploads / "private.pdf").exists()  # the user's copy is untouched
        service.pipeline.index_file.assert_not_awaited()


# ---- Classification is decided by the file, not by ledger history ---------------


def test_a_readable_file_with_a_failed_history_goes_to_review_not_to_failed() -> None:
    """A `failed` row whose file still opens cleanly is recoverable. Treating
    every tracked failure as corrupt would bury readable documents in the
    unrecoverable queue -- and it must still never be indexed without an
    admin saying so."""
    with _kb_dirs() as (staging, kb, _archive, _review):
        service = _service()
        readable = staging / "readable.txt"
        readable.write_bytes(b"An Affidavit that parses perfectly well.")
        content_hash = service.quality.hash_file(readable)
        asyncio.run(
            service.staging.insert(
                {
                    "original_filename": "readable.txt",
                    "status": "failed",
                    "content_hash": content_hash,
                    "reason": "embedding provider unavailable",
                }
            )
        )

        report = asyncio.run(service.reconcile_staging())

        move = report["moves"][0]
        assert move["reason_code"] == "readable_previous_failure"
        assert move["outcome"] == "needs_review"
        assert report["needs_review"] == 1
        assert report["failed_review"] == 0

        asyncio.run(service.reconcile_staging(apply=True))
        assert [p.name for p in service.review_pending_dir().iterdir()] == ["readable.txt"]
        assert list(kb.iterdir()) == []
        service.pipeline.index_file.assert_not_awaited()


def test_an_unreadable_pdf_goes_to_the_failed_review_queue_with_a_corrupt_reason_code() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        (staging / "broken.pdf").write_bytes(b"%PDF-1.4 truncated before the EOF marker")

        report = asyncio.run(service.reconcile_staging())

        move = report["moves"][0]
        assert move["reason_code"] == "corrupt_pdf"
        assert move["outcome"] == "failed"
        assert report["failed_review"] == 1
        assert report["needs_review"] == 0


def test_reason_codes_and_physical_totals_are_reported_separately_from_ledger_findings() -> None:
    with _kb_dirs() as (staging, kb, _archive, _review):
        service = _service()
        (kb / "Indexed.txt").write_bytes(b"already in the kb")
        (staging / "dup.txt").write_bytes(b"already in the kb")
        (staging / "broken.pdf").write_bytes(b"%PDF-1.4 truncated")
        readable = staging / "readable.txt"
        readable.write_bytes(b"A Legal Notice that reads fine.")
        asyncio.run(
            service.staging.insert(
                {
                    "original_filename": "readable.txt",
                    "status": "failed",
                    "content_hash": service.quality.hash_file(readable),
                    "reason": "provider timeout",
                }
            )
        )
        # A ledger row with no file behind it: a finding, never physical work.
        asyncio.run(
            service.staging.insert(
                {
                    "original_filename": "vanished.txt",
                    "status": "needs_review",
                    "content_hash": "deadbeef",
                    "current_path": str(staging / "vanished.txt"),
                }
            )
        )

        report = asyncio.run(service.reconcile_staging())

        assert report["scanned"] == 3
        assert report["archive"] == 1
        assert report["failed_review"] == 1
        assert report["needs_review"] == 1
        assert report["delete"] == 0
        assert report["auto_index"] == 0
        # Counted apart from the physical plan: 1 + 1 + 1 == 3 files scanned.
        assert report["ledger_missing_file"] == 1
        assert report["archive"] + report["failed_review"] + report["needs_review"] == report["scanned"]
        assert {item["reason_code"] for item in report["moves"]} == {
            "already_indexed_exact_hash",
            "corrupt_pdf",
            "readable_previous_failure",
        }
        assert report["ledger_missing"][0]["reason_code"] == "ledger_file_missing"


def test_apply_marks_missing_file_rows_without_inventing_or_moving_a_file() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        asyncio.run(
            service.staging.insert(
                {
                    "original_filename": "vanished.txt",
                    "status": "needs_review",
                    "content_hash": "deadbeef",
                    "current_path": str(staging / "vanished.txt"),
                }
            )
        )

        report = asyncio.run(service.reconcile_staging(apply=True))

        assert report["scanned"] == 0
        assert report["moved"] == 0
        assert report["ledger_missing_file"] == 1
        row = _records(service)[0]
        assert row["path_missing"] is True
        assert row["reason_code"] == "ledger_file_missing"
        assert row["file_exists"] is False
        assert not (staging / "vanished.txt").exists()


# ---- Closing / re-sourcing a path_missing finding --------------------------------


def _missing_record(service: KnowledgeBaseIngestionService, staging: Path) -> str:
    return asyncio.run(
        service.staging.insert(
            {
                "original_filename": "vanished.txt",
                "status": "needs_review",
                "content_hash": "deadbeef",
                "current_path": str(staging / "vanished.txt"),
                "path_missing": True,
            }
        )
    )


def test_closing_a_missing_file_record_requires_a_reason() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        staging_id = _missing_record(service, staging)

        with pytest.raises(BadRequestError):
            asyncio.run(service.close_path_missing(staging_id, "   "))
        assert _records(service)[0].get("resolution") is None

        result = asyncio.run(service.close_path_missing(staging_id, "Source document confirmed destroyed."))

        assert result["resolution"] == "closed_missing_file"
        row = _records(service)[0]
        assert row["resolution_reason"] == "Source document confirmed destroyed."
        assert row["path_missing"] is False
        # Closed by decision, not converted into an indexable document.
        assert row["status"] == "needs_review"


def test_closing_a_legacy_stale_record_uses_the_missing_recorded_path() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        staging_id = asyncio.run(
            service.staging.insert(
                {
                    "original_filename": "legacy-vanished.pdf",
                    "status": "needs_review",
                    "current_path": str(staging / "legacy-vanished.pdf"),
                    "file_exists": False,
                }
            )
        )

        result = asyncio.run(service.close_path_missing(staging_id, "Confirmed stale by admin."))

        assert result["resolution"] == "closed_missing_file"
        assert _records(service)[0]["path_missing"] is False


def test_close_missing_refuses_a_legacy_record_whose_file_exists() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        present = staging / "present.pdf"
        present.write_bytes(b"still here")
        staging_id = asyncio.run(
            service.staging.insert(
                {
                    "original_filename": present.name,
                    "status": "needs_review",
                    "current_path": str(present),
                }
            )
        )

        with pytest.raises(BadRequestError):
            asyncio.run(service.close_path_missing(staging_id, "Incorrectly close it."))
        assert _records(service)[0].get("resolution") is None


def test_re_sourcing_requires_an_existing_replacement_file_and_never_searches_for_one() -> None:
    with _kb_dirs() as (staging, kb, _archive, _review):
        service = _service()
        staging_id = _missing_record(service, staging)
        (kb / "Lookalike.txt").write_bytes(b"something that merely looks similar")

        with pytest.raises(BadRequestError):
            asyncio.run(service.resource_path_missing(staging_id, staging / "nope.txt"))
        assert _records(service)[0]["path_missing"] is True

        replacement = _SCRATCH_ROOT / "replacement.txt"
        replacement.write_bytes(b"the real replacement supplied by an admin")
        result = asyncio.run(service.resource_path_missing(staging_id, replacement))

        # Re-sourced, still awaiting approval -- supplying bytes is not
        # approving them, and nothing was pulled from the KB automatically.
        assert result["status"] == "needs_review"
        assert _records(service)[0]["path_missing"] is False
        assert replacement.exists()
        service.pipeline.index_file.assert_not_awaited()


def test_manifest_lists_byte_identical_copies_individually() -> None:
    with _kb_dirs() as (staging, _kb, _archive, _review):
        service = _service()
        for name in ("copy.pdf", "copy_2.pdf"):
            (staging / name).write_bytes(b"%PDF-1.4 identical truncated bytes")

        report = asyncio.run(service.reconcile_staging(apply=True))
        audit = service.read_manifest(Path(report["manifest_path"]).name)

        # One content hash, two physical files: the manifest keeps both, and
        # no extra ledger row is fabricated to make the numbers look tidy.
        assert audit["operation_count"] == 2
        assert len({op["original_path"] for op in audit["operations"]}) == 2
        assert len({op["content_hash"] for op in audit["operations"]}) == 1
        assert audit["deleted"] == 0 and audit["auto_indexed"] == 0
        assert all(op["reason_code"] == "corrupt_pdf" for op in audit["operations"])
        assert all(op["destination"] and op["operation_timestamp"] for op in audit["operations"])
