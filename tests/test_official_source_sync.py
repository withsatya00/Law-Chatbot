import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.kb_official_source_sync import OfficialSource, OfficialSourceSyncService

SOURCE = OfficialSource(
    "TEST", "official.pdf", "https://example.gov.in/official.pdf", "test-act-2020",
    "Official Gazette", "bare_act", ("testact2020", "no1of2020"),
)


def test_official_source_is_indexed_but_not_auto_approved(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.kb_official_source_sync.settings.knowledge_base_dir", tmp_path)
    monkeypatch.setattr(
        "app.services.kb_official_source_sync.normalized_text",
        lambda _: "thetestact2020no1of2020",
    )
    response = SimpleNamespace(
        status="indexed", generated_filename="official.pdf", document_id="doc-1",
        chunks_indexed=3, review_status="needs_review",
        review_reasons=["Applicability is unknown."],
    )
    ingestion = SimpleNamespace(ingest=AsyncMock(return_value=response))
    result = asyncio.run(OfficialSourceSyncService(
        AsyncMock(return_value=(b"%PDF-test", "application/pdf")), ingestion,
    ).sync(SOURCE))
    assert result["status"] == "indexed_needs_review"
    metadata = ingestion.ingest.await_args.kwargs["jurisdiction_metadata"]
    assert metadata["source_url"] == SOURCE.url
    assert metadata["review_status"] == "needs_review"
    assert metadata["verification_status"] == "unverified"


def test_identity_failure_is_held_and_never_ingested(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.kb_official_source_sync.settings.knowledge_base_dir", tmp_path)
    monkeypatch.setattr("app.services.kb_official_source_sync.normalized_text", lambda _: "wrongdocument")
    ingestion = SimpleNamespace(ingest=AsyncMock())
    result = asyncio.run(OfficialSourceSyncService(
        AsyncMock(return_value=(b"%PDF-test", "application/pdf")), ingestion,
    ).sync(SOURCE))
    assert result["status"] == "held"
    assert "identity" in result["reason"].lower()
    ingestion.ingest.assert_not_awaited()


def test_exact_canonical_file_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.kb_official_source_sync.settings.knowledge_base_dir", tmp_path)
    monkeypatch.setattr(
        "app.services.kb_official_source_sync.normalized_text", lambda _: "testact2020no1of2020",
    )
    body = b"%PDF-test"
    (tmp_path / SOURCE.filename).write_bytes(body)
    ingestion = SimpleNamespace(ingest=AsyncMock())
    result = asyncio.run(OfficialSourceSyncService(
        AsyncMock(return_value=(body, "application/pdf")), ingestion,
    ).sync(SOURCE))
    assert result["status"] == "already_present"
    ingestion.ingest.assert_not_awaited()


def test_scheduler_does_not_start_when_disabled(monkeypatch):
    from app.services.kb_official_source_sync import OfficialSourceSyncScheduler

    monkeypatch.setattr("app.services.kb_official_source_sync.settings.official_source_sync_enabled", False)
    scheduler = OfficialSourceSyncScheduler(service=AsyncMock())
    scheduler.start()
    assert scheduler.task is None


def test_scheduler_starts_exactly_one_task_when_enabled(monkeypatch):
    from app.services.kb_official_source_sync import OfficialSourceSyncScheduler

    monkeypatch.setattr("app.services.kb_official_source_sync.settings.official_source_sync_enabled", True)
    scheduler = OfficialSourceSyncScheduler(service=SimpleNamespace(run=AsyncMock(return_value={})))

    async def start_and_stop():
        scheduler.start()
        task = scheduler.task
        assert task is not None and not task.done()
        scheduler.start()  # calling again must not spawn a second task
        assert scheduler.task is task
        await scheduler.close()
        assert scheduler.task is None

    asyncio.run(start_and_stop())


def test_scheduler_close_is_safe_when_never_started():
    from app.services.kb_official_source_sync import OfficialSourceSyncScheduler

    asyncio.run(OfficialSourceSyncScheduler(service=AsyncMock()).close())


def test_scheduler_loop_catches_a_failed_cycle_instead_of_dying(monkeypatch):
    # Same shape as `GapAutoFetchScheduler`/`KnowledgeBaseAutomationScheduler`'s
    # own `_loop` -- exercised directly (bypassing `start()`'s real `sleep`)
    # rather than through the live scheduler, which neither of those two
    # existing schedulers' tests do either.
    from app.services.kb_official_source_sync import OfficialSourceSyncScheduler

    service = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("boom")))
    scheduler = OfficialSourceSyncScheduler(service=service)

    async def run_one_iteration():
        sleep_calls = 0

        async def fake_sleep(_seconds: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            raise asyncio.CancelledError

        monkeypatch.setattr("app.services.kb_official_source_sync.asyncio.sleep", fake_sleep)
        with pytest.raises(asyncio.CancelledError):
            await scheduler._loop()
        assert sleep_calls == 1  # reached the sleep, i.e. the RuntimeError was caught, not raised

    asyncio.run(run_one_iteration())
