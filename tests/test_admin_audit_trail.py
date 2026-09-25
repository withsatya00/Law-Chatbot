"""`_audit` (app/api/admin.py) already covered 9 mutating KB actions
(backfill, reconcile, close-missing, resource, jurisdiction-update, approve,
retry, archive) -- but `/admin/reindex`, `/admin/cache/flush`, and
`/admin/knowledge-base/upload` (arguably the three most consequential
mutating admin actions: triggering a full reindex, wiping the response
cache, and promoting a new document into the shared Knowledge Base) had no
audit call at all. These lock in that they now do, by calling the route
handlers directly (same style as test_kb_admin_review.py) and asserting on
`admin_api._audit` rather than re-testing the underlying KB/cache machinery.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.api import admin as admin_api
from app.schemas.admin import CacheFlushRequest


def _claims(sub: str = "admin-1") -> dict[str, object]:
    return {"sub": sub, "role": "admin"}


def test_reindex_trigger_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    audited = AsyncMock()
    monkeypatch.setattr(admin_api, "_audit", audited)
    monkeypatch.setattr(
        admin_api.admin_operations, "queue_reindex",
        AsyncMock(return_value=(AsyncMock(), "job-1", "storage/knowledge_base")),
    )
    background_tasks = admin_api.BackgroundTasks()

    result = asyncio.run(admin_api.trigger_incremental_reindex(background_tasks, root=None, claims=_claims()))

    assert result["job_id"] == "job-1"
    audited.assert_awaited_once()
    assert audited.await_args.args[0] == "kb_reindex_triggered"
    assert audited.await_args.args[1] == "admin-1"


def test_cache_flush_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    audited = AsyncMock()
    monkeypatch.setattr(admin_api, "_audit", audited)
    monkeypatch.setattr(
        admin_api.admin_operations, "flush_cache", AsyncMock(return_value={"status": "purged", "scope": "all"})
    )

    result = asyncio.run(admin_api.flush_response_cache(payload=None, claims=_claims()))

    assert result["status"] == "purged"
    audited.assert_awaited_once_with("cache_flush", "admin-1", status="purged", scope="all")


def test_cache_flush_of_one_entry_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    audited = AsyncMock()
    monkeypatch.setattr(admin_api, "_audit", audited)
    monkeypatch.setattr(
        admin_api.admin_operations, "flush_cache", AsyncMock(return_value={"status": "purged", "scope": "entry"})
    )
    payload = CacheFlushRequest(question="what is bail", language="english", intent="definition")

    asyncio.run(admin_api.flush_response_cache(payload=payload, claims=_claims()))

    audited.assert_awaited_once_with("cache_flush", "admin-1", status="purged", scope="entry")


def test_kb_upload_queued_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    audited = AsyncMock()
    monkeypatch.setattr(admin_api, "_audit", audited)

    class _Staged:
        staging_id = "staging-1"
        staged_path = "storage/kb_staging/x.pdf"
        original_filename = "x.pdf"
        status = "pending"
        claimed = True
        reason = None

    monkeypatch.setattr(
        admin_api, "KnowledgeBaseIngestionService",
        lambda: type("_Svc", (), {"stage": AsyncMock(return_value=_Staged())})(),
    )
    monkeypatch.setattr(admin_api.kb_indexing_queue, "enqueue", lambda *a, **k: None)

    class _FakeUpload:
        filename = "x.pdf"

    result = asyncio.run(
        admin_api.upload_to_knowledge_base(file=_FakeUpload(), jurisdiction_metadata=None, claims=_claims())
    )

    assert result.status == "pending"
    audited.assert_awaited_once_with("kb_upload_queued", "admin-1", staging_id="staging-1", filename="x.pdf")


def test_kb_upload_duplicate_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    audited = AsyncMock()
    monkeypatch.setattr(admin_api, "_audit", audited)

    class _Staged:
        staging_id = "staging-2"
        staged_path = "storage/kb_staging/y.pdf"
        original_filename = "y.pdf"
        status = "duplicate"
        claimed = False
        reason = "Identical content already staged."

    monkeypatch.setattr(
        admin_api, "KnowledgeBaseIngestionService",
        lambda: type("_Svc", (), {"stage": AsyncMock(return_value=_Staged())})(),
    )

    class _FakeUpload:
        filename = "y.pdf"

    result = asyncio.run(
        admin_api.upload_to_knowledge_base(file=_FakeUpload(), jurisdiction_metadata=None, claims=_claims())
    )

    assert result.status == "duplicate"
    audited.assert_awaited_once_with(
        "kb_upload_duplicate", "admin-1", staging_id="staging-2", filename="y.pdf", status="duplicate"
    )
