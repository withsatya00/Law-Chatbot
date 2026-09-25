from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.kb_automation import (
    KnowledgeBaseAutomationService,
    ManifestSourceAdapter,
    ManualAccessRequiredError,
    OfficialDownloader,
    SourceCandidate,
    normalize_batch_codes,
)
from app.services.kb_central_adapters import (
    CatalogueConfig,
    OfficialCatalogueAdapter,
    central_source_adapters,
)


def _candidate() -> SourceCandidate:
    return SourceCandidate(
        adapter="test", external_id="act-1", title="Example Act, 2026",
        url="https://example.gov.in/act.pdf", jurisdiction_code="IN",
        document_type="bare_act", version="Act 1 of 2026", filename="example.pdf",
        act_number="1", enactment_year=2026,
    )


def test_canonical_identity_is_stable_across_urls() -> None:
    first = _candidate()
    second = SourceCandidate(**{**first.__dict__, "url": "https://other.gov.in/copy.pdf"})
    assert first.canonical_document_key == second.canonical_document_key
    assert first.source_key == second.source_key


def test_downloader_rejects_mime_spoof_and_truncated_pdf() -> None:
    with pytest.raises(ValueError, match="PDF signature"):
        OfficialDownloader._validate_payload(b"<html>not pdf</html>", "application/pdf")
    with pytest.raises(ValueError, match="EOF"):
        OfficialDownloader._validate_payload(b"%PDF-1.7 incomplete", "application/pdf")


def test_all_requested_central_adapters_are_registered() -> None:
    assert {adapter.name for adapter in central_source_adapters()} == {
        "india_code", "legislative_department", "egazette", "supreme_court",
        "rbi", "sebi", "mca", "irdai", "cbdt", "cbic_gst",
        "selected_ministries",
    }


def test_catalogue_adapter_extracts_version_language_and_document_link() -> None:
    html = b"""
    <html><body><table><tr><td>12-04-2026</td><td>
      <a href="/files/act-12-hindi.pdf">Act No. 12 of 2026 Hindi</a>
    </td></tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = OfficialCatalogueAdapter(CatalogueConfig(
        "fixture", "Test Ministry", ("https://law.gov.in/catalogue",),
        "bare_act", r"act", 10,
    ), fetcher=fetcher)

    candidates, checkpoint = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    assert candidates[0].url == "https://law.gov.in/files/act-12-hindi.pdf"
    assert candidates[0].act_number == "12"
    assert candidates[0].enactment_year == 2026
    assert candidates[0].language == "hindi"
    assert checkpoint and len(checkpoint) == 64


def test_catalogue_adapter_accepts_icon_only_download_links() -> None:
    """Regression for a real defect found against the live upvidhai.gov.in
    Ordinances page: every one of its real PDF links is an icon-only anchor
    with no visible text (`<a href="...pdf"><img alt=""/></a>`), which the
    parser previously discarded outright because it required the ANCHOR's
    own text to be non-empty, before ever checking `include_pattern`. The row
    text must still be used for matching and for the candidate's title."""
    html = b"""
    <html><body><table><tr><td>15-08-2026</td><td>Ordinance No. 7 of 2026</td><td>
      <a href="/Upload/Ordinance/7of2026.pdf"><img src="/images/download.png" alt="" /></a>
    </td></tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = OfficialCatalogueAdapter(CatalogueConfig(
        "fixture", "Test Department", ("https://law.gov.in/ordinances",),
        "ordinance", r"ordinance|\.pdf", 10,
    ), fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    assert candidates[0].url == "https://law.gov.in/Upload/Ordinance/7of2026.pdf"
    assert candidates[0].title  # never empty, even though the anchor's own text was
    assert "7" in candidates[0].title or "2026" in candidates[0].title


def test_manifest_adapter_discovers_configured_official_source(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        '{"sources":[{"external_id":"x","title":"Act X","url":"https://law.gov.in/x.pdf",'
        '"jurisdiction_code":"IN","document_type":"bare_act","version":"v1",'
        '"filename":"x.pdf"}]}',
        encoding="utf-8",
    )
    candidates, checkpoint = asyncio.run(ManifestSourceAdapter(manifest).discover(None))
    assert candidates[0].external_id == "x"
    assert checkpoint and len(checkpoint) == 64


def test_download_scan_stage_index_ends_hidden_pending_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.kb_automation as module

    monkeypatch.setattr(module.settings, "kb_staging_dir", tmp_path)
    candidate = _candidate()
    body = b"%PDF-1.7 legal bytes\n%%EOF"

    downloader = AsyncMock()
    downloader.fetch = AsyncMock(return_value=(body, "application/pdf", candidate.url))
    scanner = AsyncMock()
    scanner.scan = AsyncMock(return_value=None)
    ingestion = AsyncMock()

    async def stage(upload, jurisdiction_metadata):
        path = tmp_path / "staged.pdf"
        path.write_bytes(upload.file.read())
        assert jurisdiction_metadata["review_status"] == "needs_review"
        assert jurisdiction_metadata["document_key"] == candidate.canonical_document_key
        return SimpleNamespace(
            claimed=True, staging_id="stage-1", staged_path=path, original_filename="example.pdf",
        )

    ingestion.stage = AsyncMock(side_effect=stage)
    ingestion.process = AsyncMock(return_value=SimpleNamespace(
        status="indexed", document_id="doc-1", chunks_indexed=4,
        review_status="needs_review", reason=None, generated_filename="Example_Act.pdf",
    ))
    jobs = AsyncMock()
    jobs.find_one = AsyncMock(return_value=None)
    jobs.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1))
    db = {
        "kb_automation_jobs": jobs,
        "kb_automation_state": AsyncMock(),
        "operational_events": AsyncMock(),
    }
    service = KnowledgeBaseAutomationService(
        db=db, adapters=[], downloader=downloader, scanner=scanner, ingestion=ingestion,
    )
    job = {
        "_id": "job-1", "lease_token": "lease-1", "candidate": candidate.__dict__, "attempts": 0,
    }

    outcome = asyncio.run(service._process_claimed(job))

    assert outcome == "quarantined"
    scanner.scan.assert_awaited_once()
    ingestion.stage.assert_awaited_once()
    ingestion.process.assert_awaited_once()
    update = jobs.update_one.await_args.args[1]
    assert update["$set"]["status"] == "quarantined"
    assert update["$set"]["document_id"] == "doc-1"
    assert update["$set"]["chunks_indexed"] == 4


def test_failed_security_gate_retries_without_ingestion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.kb_automation as module

    monkeypatch.setattr(module.settings, "kb_staging_dir", tmp_path)
    candidate = _candidate()
    downloader = AsyncMock()
    downloader.fetch = AsyncMock(return_value=(b"%PDF-1.7 x\n%%EOF", "application/pdf", candidate.url))
    scanner = AsyncMock()
    scanner.scan = AsyncMock(side_effect=ValueError("infected"))
    ingestion = AsyncMock()
    jobs = AsyncMock()
    jobs.find_one = AsyncMock(return_value=None)
    jobs.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1))
    events = AsyncMock()
    events.insert_one = AsyncMock(return_value=None)
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": jobs, "kb_automation_state": AsyncMock(), "operational_events": events},
        adapters=[], downloader=downloader, scanner=scanner, ingestion=ingestion,
    )

    outcome = asyncio.run(service._process_claimed({
        "_id": "job-2", "lease_token": "lease-2", "candidate": candidate.__dict__, "attempts": 0,
    }))

    assert outcome == "retry"
    ingestion.stage.assert_not_awaited()
    events.insert_one.assert_awaited_once()


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def sort(self, *args):
        return self

    def limit(self, count):
        self.rows = self.rows[:count]
        return self

    def __aiter__(self):
        async def iterator():
            for row in self.rows:
                yield row
        return iterator()


def test_approved_document_must_pass_real_benchmark_contract_before_publication() -> None:
    candidate = _candidate()
    jobs = AsyncMock()
    jobs.find = lambda query: _Cursor([{
        "_id": "job-3", "status": "quarantined", "document_id": "doc-3",
        "source_document": "example.pdf", "candidate": candidate.__dict__,
    }])
    jobs.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1))
    chunks = AsyncMock()
    chunks.count_documents = AsyncMock(side_effect=[3, 3])
    benchmark = AsyncMock()
    benchmark.evaluate = AsyncMock(return_value={"passed": True, "matches": 1, "top_score": 0.7})
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": jobs, "kb_automation_state": AsyncMock(),
            "operational_events": AsyncMock(), "embeddings_metadata": chunks},
        adapters=[], downloader=AsyncMock(), scanner=AsyncMock(), ingestion=AsyncMock(), benchmark=benchmark,
    )

    result = asyncio.run(service.promote_approved())

    assert result == {"checked": 1, "published": 1, "failed": 0}
    benchmark.evaluate.assert_awaited_once_with(candidate, "example.pdf")
    assert jobs.update_one.await_args.args[1]["$set"]["status"] == "published"


def test_changed_official_bytes_create_a_new_version_job() -> None:
    candidate = _candidate()
    jobs = AsyncMock()
    jobs.find = lambda query: _Cursor([{
        "_id": "old-job", "status": "published", "checksum": "old-checksum",
        "candidate": candidate.__dict__,
    }])
    jobs.update_one = AsyncMock(side_effect=[
        SimpleNamespace(upserted_id="new-job", modified_count=0),
        SimpleNamespace(upserted_id=None, modified_count=1),
    ])
    downloader = AsyncMock()
    downloader.fetch = AsyncMock(return_value=(
        b"%PDF-1.7 changed\n%%EOF", "application/pdf", candidate.url,
    ))
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": jobs, "kb_automation_state": AsyncMock(),
            "operational_events": AsyncMock(), "embeddings_metadata": AsyncMock()},
        adapters=[], downloader=downloader, scanner=AsyncMock(), ingestion=AsyncMock(), benchmark=AsyncMock(),
    )

    result = asyncio.run(service.probe_due_sources())

    assert result == {"checked": 1, "changed": 1, "unchanged": 0, "failed": 0}
    inserted = jobs.update_one.await_args_list[0].args[1]["$setOnInsert"]
    assert inserted["detected_from_job_id"] == "old-job"
    assert inserted["candidate"]["version"].startswith("Act 1 of 2026-content-")
    assert inserted["canonical_document_key"] == candidate.canonical_document_key


def test_state_candidate_gets_real_applicability_not_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A state adapter already knows the jurisdiction -- unlike a manual
    upload with no metadata, this must not be forced through `unknown`."""
    import app.services.kb_automation as module

    monkeypatch.setattr(module.settings, "kb_staging_dir", tmp_path)
    candidate = SourceCandidate(
        adapter="up_acts", external_id="act-1", title="UP Example Act, 2026",
        url="https://upvidhai.gov.in/act.pdf", jurisdiction_code="UP",
        document_type="bare_act", version="v1", filename="up-act.pdf",
    )
    downloader = AsyncMock()
    downloader.fetch = AsyncMock(return_value=(b"%PDF-1.7 x\n%%EOF", "application/pdf", candidate.url))
    scanner = AsyncMock()
    scanner.scan = AsyncMock(return_value=None)
    ingestion = AsyncMock()
    seen_metadata: dict = {}

    async def stage(upload, jurisdiction_metadata):
        seen_metadata.update(jurisdiction_metadata)
        return SimpleNamespace(claimed=True, staging_id="s", staged_path=tmp_path / "x.pdf", original_filename="x.pdf")

    ingestion.stage = AsyncMock(side_effect=stage)
    ingestion.process = AsyncMock(return_value=SimpleNamespace(
        status="indexed", document_id="doc-1", chunks_indexed=1,
        review_status="needs_review", reason=None, generated_filename="x.pdf",
    ))
    jobs = AsyncMock()
    jobs.find_one = AsyncMock(return_value=None)
    jobs.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1))
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": jobs, "kb_automation_state": AsyncMock(), "operational_events": AsyncMock()},
        adapters=[], downloader=downloader, scanner=scanner, ingestion=ingestion,
    )

    asyncio.run(service._process_claimed({
        "_id": "job-up", "lease_token": "lease-up", "candidate": candidate.__dict__, "attempts": 0,
    }))

    assert seen_metadata["issuing_level"] == "state"
    assert seen_metadata["applicability"] == "specific_states"
    assert seen_metadata["applicable_state_codes"] == ["UP"]
    assert seen_metadata["verification_status"] == "unverified"
    # Advisory-only fields never leak into jurisdiction metadata.
    assert "confidence_score" not in seen_metadata
    assert "change_type" not in seen_metadata


def test_captcha_page_becomes_manual_access_required_not_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.kb_automation as module

    monkeypatch.setattr(module.settings, "kb_staging_dir", tmp_path)
    candidate = _candidate()
    downloader = AsyncMock()
    downloader.fetch = AsyncMock(side_effect=ManualAccessRequiredError("captcha detected"))
    jobs = AsyncMock()
    jobs.find_one = AsyncMock(return_value=None)
    jobs.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1))
    events = AsyncMock()
    events.insert_one = AsyncMock(return_value=None)
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": jobs, "kb_automation_state": AsyncMock(), "operational_events": events},
        adapters=[], downloader=downloader, scanner=AsyncMock(), ingestion=AsyncMock(),
    )

    outcome = asyncio.run(service._process_claimed({
        "_id": "job-captcha", "lease_token": "lease-c", "candidate": candidate.__dict__, "attempts": 0,
    }))

    assert outcome == "manual_access_required"
    update = jobs.update_one.await_args.args[1]
    assert update["$set"]["status"] == "manual_access_required"
    assert "next_attempt_at" not in update["$set"]  # not on the retry/backoff schedule
    event = events.insert_one.await_args.args[0]
    assert event["event_type"] == "kb_manual_access_required"


def test_retry_refuses_manual_access_required_jobs() -> None:
    jobs = AsyncMock()
    jobs.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=0))
    jobs.find_one = AsyncMock(return_value={"_id": "job-captcha", "status": "manual_access_required"})
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": jobs, "kb_automation_state": AsyncMock(), "operational_events": AsyncMock()},
        adapters=[],
    )

    requeued = asyncio.run(service.retry("job-captcha"))

    assert requeued is False
    query = jobs.update_one.await_args.args[0]
    assert query["$or"] == [
        {"status": "retry"},
        {"status": "quarantined", "last_error": {"$exists": True}},
    ]


def test_retrying_a_nonexistent_job_is_not_found_not_a_silent_false() -> None:
    """Security/correctness finding G5: a 0-match update meant either "this
    job exists but isn't retryable right now" (a real `requeued: false`,
    exercised above) or "this job_id was never registered at all" -- these
    must not read the same to a caller."""
    from app.core.exceptions import NotFoundError

    jobs = AsyncMock()
    jobs.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=0))
    jobs.find_one = AsyncMock(return_value=None)
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": jobs, "kb_automation_state": AsyncMock(), "operational_events": AsyncMock()},
        adapters=[],
    )

    with pytest.raises(NotFoundError):
        asyncio.run(service.retry("this-job-id-was-never-registered"))


def test_downloader_treats_403_as_manual_access_required(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status_code = 403

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, headers=None):
            return _Response()

    import app.services.kb_automation as module

    monkeypatch.setattr(module.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(OfficialDownloader, "_validate_public", AsyncMock(return_value=None))
    downloader = OfficialDownloader()
    with pytest.raises(ManualAccessRequiredError):
        asyncio.run(downloader.fetch("https://blocked.gov.in/doc.pdf"))


def test_layout_change_on_previously_productive_single_url_adapter() -> None:
    class _FlatlinedAdapter:
        name = "flatlined"
        config = SimpleNamespace(urls=("https://law.gov.in/listing",))

        async def discover(self, checkpoint):
            return [], "same-checksum"

    state = AsyncMock()
    state.find_one = AsyncMock(return_value={"last_discovered_count": 12, "checkpoint": "old"})
    state.update_one = AsyncMock(return_value=None)
    events = AsyncMock()
    events.insert_one = AsyncMock(return_value=None)
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": AsyncMock(), "kb_automation_state": state, "operational_events": events},
        adapters=[_FlatlinedAdapter()],
    )

    counts = asyncio.run(service.discover())

    assert counts == {"discovered": 0, "already_known": 0, "adapter_failures": 1}
    event = events.insert_one.await_args.args[0]
    assert event["event_type"] == "kb_adapter_layout_changed"
    state_update = state.update_one.await_args.args[1]["$set"]
    assert state_update["last_error"] == "PortalLayoutChangedError"
    assert state_update["layout_changed"] is True


def test_layout_change_check_skips_paginated_multi_url_adapters() -> None:
    class _PaginatedAdapter:
        name = "paginated"
        config = SimpleNamespace(urls=("https://law.gov.in/page1", "https://law.gov.in/page2"))

        async def discover(self, checkpoint):
            return [], "next-page"

    state = AsyncMock()
    state.find_one = AsyncMock(return_value={"last_discovered_count": 25})
    state.update_one = AsyncMock(return_value=None)
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": AsyncMock(), "kb_automation_state": state, "operational_events": AsyncMock()},
        adapters=[_PaginatedAdapter()],
    )

    counts = asyncio.run(service.discover())

    assert counts == {"discovered": 0, "already_known": 0, "adapter_failures": 0}


def test_downloader_rejects_captcha_marker_distinctly() -> None:
    with pytest.raises(ManualAccessRequiredError):
        OfficialDownloader._validate_payload(
            b"<html><body>Please complete the g-recaptcha challenge</body></html>", "text/html",
        )


def test_batch_scope_is_validated_and_normalized() -> None:
    assert normalize_batch_codes("up, DL,up") == ("UP", "DL")
    assert normalize_batch_codes(None) == ()
    with pytest.raises(ValueError, match="XX"):
        normalize_batch_codes("UP,XX")


def test_relationship_candidate_cannot_publish_before_human_linkage_review() -> None:
    candidate = SourceCandidate(**{**_candidate().__dict__, "change_type": "amendment"})
    jobs = AsyncMock()
    jobs.find = lambda query: _Cursor([{
        "_id": "amendment", "status": "quarantined", "document_id": "doc",
        "source_document": "amendment.pdf", "candidate": candidate.__dict__,
    }])
    chunks = AsyncMock()
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": jobs, "kb_automation_state": AsyncMock(),
            "operational_events": AsyncMock(), "embeddings_metadata": chunks},
        adapters=[], benchmark=AsyncMock(),
    )
    assert asyncio.run(service.promote_approved()) == {
        "checked": 0, "published": 0, "failed": 0,
    }
    chunks.count_documents.assert_not_awaited()


def test_relationship_review_is_audited_and_requires_existing_related_jobs() -> None:
    jobs = AsyncMock()
    jobs.find_one = AsyncMock(return_value={
        "_id": "amendment", "candidate": {"change_type": "amendment"},
    })
    jobs.count_documents = AsyncMock(return_value=1)
    events = AsyncMock()
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": jobs, "kb_automation_state": AsyncMock(),
            "operational_events": events, "embeddings_metadata": AsyncMock()}, adapters=[],
    )
    result = asyncio.run(service.record_relationship_review(
        "amendment", ["principal-act"], "Compared against the official amendment.", "admin",
    ))
    assert result["status"] == "reviewed"
    assert jobs.update_one.await_args.args[1]["$set"]["relationship_reviewed_by"] == "admin"
    assert events.insert_one.await_args.args[0]["event_type"] == "kb_legal_relationship_reviewed"


def test_scoped_claim_adds_jurisdiction_filter() -> None:
    jobs = AsyncMock()
    jobs.find_one_and_update = AsyncMock(return_value=None)
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": jobs, "kb_automation_state": AsyncMock(),
            "operational_events": AsyncMock(), "embeddings_metadata": AsyncMock()}, adapters=[],
    )
    assert asyncio.run(service._claim(("UP",))) is None
    query = jobs.find_one_and_update.await_args.args[0]
    assert query["$and"][1]["$or"][0] == {
        "candidate.jurisdiction_code": {"$in": ["UP"]},
    }


def test_failed_recurring_canary_quarantines_chunks_and_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.cache.response_cache import response_cache

    candidate = _candidate()
    jobs = AsyncMock()
    jobs.find = lambda query: _Cursor([{
        "_id": "published-job", "status": "published", "document_id": "doc-1",
        "source_document": "example.pdf", "candidate": candidate.__dict__,
    }])
    jobs.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1))
    chunks = AsyncMock()
    benchmark = AsyncMock()
    benchmark.evaluate = AsyncMock(return_value={"passed": False, "matches": 0})
    invalidate = AsyncMock()
    monkeypatch.setattr(response_cache, "bump_generation", invalidate)
    events = AsyncMock()
    service = KnowledgeBaseAutomationService(
        db={"kb_automation_jobs": jobs, "kb_automation_state": AsyncMock(),
            "operational_events": events, "embeddings_metadata": chunks},
        adapters=[], benchmark=benchmark,
    )

    result = asyncio.run(service.revalidate_published())

    assert result == {"checked": 1, "passed": 0, "rolled_back": 1}
    update = chunks.update_many.await_args.args[1]["$set"]
    assert update["metadata.review_status"] == "needs_review"
    assert update["metadata.document_status"] == "quarantined"
    invalidate.assert_awaited_once_with(strict=True)
    assert events.insert_one.await_args.args[0]["event_type"] == "kb_canary_rollback"
