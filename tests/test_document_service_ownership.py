"""Part 46 "Authenticated User Ownership": `DocumentService._ensure_document_
access` -- closes the remaining risk Part 45 flagged (`analyze()` had zero
ownership check at all, unlike the RAG-retrieval path). Pure function over a
synthetic metadata dict, no DB needed.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError
from app.schemas.document import DocumentAnalysisRequest
from app.services.document_service import DocumentService
from app.services.phase3 import AuditService


def _service() -> DocumentService:
    return DocumentService()


def test_global_document_allowed_for_anyone() -> None:
    _service()._ensure_document_access({}, authenticated_user_id=None, session_id=None)
    _service()._ensure_document_access({}, authenticated_user_id="user-A", session_id="s1")


def test_user_owned_document_allowed_for_its_owner() -> None:
    metadata = {"owner_user_id": "user-A", "owner_session_id": "session-A-original"}
    _service()._ensure_document_access(metadata, authenticated_user_id="user-A", session_id=None)


def test_user_owned_document_allowed_for_owner_from_a_different_session() -> None:
    metadata = {"owner_user_id": "user-A", "owner_session_id": "session-A-original"}
    _service()._ensure_document_access(metadata, authenticated_user_id="user-A", session_id="brand-new-session")


def test_user_owned_document_denied_for_a_different_authenticated_user() -> None:
    metadata = {"owner_user_id": "user-A", "owner_session_id": "session-A-original"}
    with pytest.raises(ForbiddenError):
        _service()._ensure_document_access(metadata, authenticated_user_id="user-B", session_id=None)


def test_user_owned_document_denied_for_anonymous_caller() -> None:
    metadata = {"owner_user_id": "user-A", "owner_session_id": "session-A-original"}
    with pytest.raises(ForbiddenError):
        _service()._ensure_document_access(metadata, authenticated_user_id=None, session_id=None)


def test_user_owned_document_denied_even_with_a_matching_guessed_session_id() -> None:
    metadata = {"owner_user_id": "user-A", "owner_session_id": "session-A-original"}
    with pytest.raises(ForbiddenError):
        _service()._ensure_document_access(metadata, authenticated_user_id=None, session_id="session-A-original")


def test_legacy_session_only_document_allowed_for_the_original_session() -> None:
    metadata = {"owner_session_id": "session-A-original"}
    _service()._ensure_document_access(metadata, authenticated_user_id=None, session_id="session-A-original")


def test_legacy_session_only_document_denied_for_a_different_session() -> None:
    metadata = {"owner_session_id": "session-A-original"}
    with pytest.raises(ForbiddenError):
        _service()._ensure_document_access(metadata, authenticated_user_id=None, session_id="some-other-session")


class _FakeUploadFile:
    """Minimal stand-in for FastAPI's `UploadFile` -- `upload_and_index` only
    ever touches `.filename` and awaits `.read(size)`, returning `b""` once
    exhausted (the same protocol `UploadFile.read` follows).
    """

    def __init__(self, filename: str, content: bytes = b"%PDF-1.4 fake content") -> None:
        self.filename = filename
        self._content = content
        self._exhausted = False

    async def read(self, _size: int) -> bytes:
        if self._exhausted:
            return b""
        self._exhausted = True
        return self._content


def _service_with_mocked_pipeline() -> DocumentService:
    """Part 51 "Uploaded Document Conversation Context": `upload_and_index`
    calling `self.memory.update(...)` on success is the specific behavior
    under test here -- `self.pipeline.index_file` (the actual chunking/
    embedding work) and disk I/O are mocked out since they're unrelated to
    that behavior and covered by their own tests elsewhere.
    """
    service = _service()
    service.pipeline.index_file = AsyncMock(return_value=("doc-abc123", "english", [MagicMock(metadata={})]))
    service.memory.update = AsyncMock(return_value={})
    return service


def test_successful_upload_records_last_uploaded_document_id() -> None:
    service = _service_with_mocked_pipeline()
    asyncio.run(service.upload_and_index(_FakeUploadFile("notice.pdf"), session_id="session-1", user_id="user-A"))
    service.memory.update.assert_awaited_once()
    session_id, updates = service.memory.update.await_args.args[0], service.memory.update.await_args.kwargs
    assert session_id == "session-1"
    assert updates["last_uploaded_document_id"] == "doc-abc123"


def test_successful_upload_appends_to_the_conversation_document_list() -> None:
    """Phase 3: the LIST is what lets a user say "compare these two" or pick
    between uploads by name. With only the latest id, a second upload
    silently replaced the first."""
    service = _service_with_mocked_pipeline()
    asyncio.run(service.upload_and_index(_FakeUploadFile("notice.pdf"), session_id="session-1", user_id="user-A"))
    documents = service.memory.update.await_args.kwargs["uploaded_documents"]
    assert [item["document_id"] for item in documents] == ["doc-abc123"]
    assert documents[0]["filename"] == "notice.pdf"


def test_upload_without_a_session_id_never_touches_memory() -> None:
    # Part 45's existing design: no `session_id` means no conversation to
    # remember the upload FOR (a script/admin-tooling/anonymous-API upload).
    service = _service_with_mocked_pipeline()
    asyncio.run(service.upload_and_index(_FakeUploadFile("notice.pdf"), session_id=None, user_id="user-A"))
    service.memory.update.assert_not_awaited()


def test_failed_upload_never_touches_memory() -> None:
    # An unsupported file extension is rejected before indexing is ever
    # attempted -- the memory-recording line must never be reached.
    service = _service_with_mocked_pipeline()
    with pytest.raises(BadRequestError):
        asyncio.run(
            service.upload_and_index(_FakeUploadFile("malware.exe"), session_id="session-1", user_id="user-A")
        )
    service.memory.update.assert_not_awaited()


def test_rtf_upload_is_accepted() -> None:
    service = _service_with_mocked_pipeline()
    asyncio.run(
        service.upload_and_index(_FakeUploadFile("notice.rtf", content=b"{\\rtf1 Hello}"), session_id="session-1")
    )
    service.memory.update.assert_awaited_once()


def test_odt_upload_is_accepted() -> None:
    service = _service_with_mocked_pipeline()
    asyncio.run(
        service.upload_and_index(_FakeUploadFile("notice.odt", content=b"fake odt bytes"), session_id="session-1")
    )
    service.memory.update.assert_awaited_once()


class _FakeCursor:
    """Minimal stand-in for the Motor async cursor `self.embeddings.collection.
    find(...)` returns -- `analyze()` only ever does `async for chunk in cursor`.
    """

    def __init__(self, chunks: list[dict]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


def test_analyze_end_to_end_denies_user_b_access_to_user_a_document() -> None:
    # Part 51 "Uploaded Document Conversation Context" test 9: proves
    # `analyze()` itself -- not just the isolated `_ensure_document_access`
    # function -- actually enforces ownership on a real call, closing the
    # gap between "the ownership function is correct" (the tests above) and
    # "the code path the chat service actually calls uses it."
    service = _service()
    service.embeddings = MagicMock()
    service.embeddings.collection.find = MagicMock(
        return_value=_FakeCursor(
            [{"text": "Confidential contract text.", "metadata": {"owner_user_id": "user-A", "owner_session_id": "session-A"}}]
        )
    )
    with pytest.raises(ForbiddenError):
        asyncio.run(
            service.analyze(
                DocumentAnalysisRequest(document_id="doc-A", session_id="session-B"), authenticated_user_id="user-B",
            )
        )


def test_analyze_end_to_end_allows_owner_to_analyze_their_own_document() -> None:
    service = _service()
    service.embeddings = MagicMock()
    service.embeddings.collection.find = MagicMock(
        return_value=_FakeCursor(
            [{"text": "This is a rental agreement.", "metadata": {"owner_user_id": "user-A", "owner_session_id": "session-A"}}]
        )
    )
    result = asyncio.run(
        service.analyze(
            DocumentAnalysisRequest(document_id="doc-A", session_id="session-A"), authenticated_user_id="user-A",
        )
    )
    assert "rental agreement" in result.executive_summary.lower()


def test_analyzing_a_stored_document_records_an_audit_trail_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stored document being read back (as opposed to raw pasted `text`,
    which is never persisted) previously left no trace of who accessed
    which document -- unlike admin KB actions and draft export, which
    already went through `AuditService.record`.
    """
    calls: list[dict] = []

    async def fake_record(self: AuditService, **kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(AuditService, "record", fake_record)
    service = _service()
    service.embeddings = MagicMock()
    service.embeddings.collection.find = MagicMock(
        return_value=_FakeCursor(
            [{"text": "This is a rental agreement.", "metadata": {"owner_user_id": "user-A", "owner_session_id": "session-A"}}]
        )
    )

    asyncio.run(
        service.analyze(
            DocumentAnalysisRequest(document_id="doc-A", session_id="session-A"), authenticated_user_id="user-A",
        )
    )

    assert len(calls) == 1
    assert calls[0]["action"] == "document_accessed"
    assert calls[0]["resource_id"] == "doc-A"
    assert calls[0]["actor_user_id"] == "user-A"


def test_upload_indexing_failure_never_touches_memory() -> None:
    # A genuine indexing failure (corrupt file, pipeline error) must not
    # record a document ID that was never actually successfully indexed.
    service = _service_with_mocked_pipeline()
    service.pipeline.index_file = AsyncMock(side_effect=RuntimeError("indexing exploded"))
    with pytest.raises(RuntimeError):
        asyncio.run(
            service.upload_and_index(_FakeUploadFile("notice.pdf"), session_id="session-1", user_id="user-A")
        )
    service.memory.update.assert_not_awaited()


# ---------------------------------------------------------------------------
# PDF Q&A acceptance pass (2026-09-12): `delete_owned` -- previously there was
# no way to delete an uploaded document at all (`DELETE /session`/`DELETE
# /me/data` erase conversation memory and chat history, never indexed
# document content), so a "deleted" document stayed fully queryable
# indefinitely.
# ---------------------------------------------------------------------------


def _service_with_mocked_deletion(document: dict | None) -> DocumentService:
    service = _service()
    service.documents = MagicMock()
    service.documents.find_by_id = AsyncMock(return_value=document)
    service.documents.delete_by_id = AsyncMock(return_value=True)
    service.pipeline = MagicMock()
    service.pipeline.vector_store.delete_version_chunks = AsyncMock(return_value=7)
    return service


def test_delete_owned_removes_every_chunk_and_the_document_record() -> None:
    service = _service_with_mocked_deletion(
        {"_id": "doc-A", "owner_user_id": "user-A", "owner_session_id": "session-A"}
    )
    deleted = asyncio.run(service.delete_owned("doc-A", authenticated_user_id="user-A", session_id="session-A"))
    assert deleted == 7
    service.pipeline.vector_store.delete_version_chunks.assert_awaited_once_with("doc-A")
    service.documents.delete_by_id.assert_awaited_once_with("doc-A")


def test_delete_owned_denies_a_different_users_document() -> None:
    service = _service_with_mocked_deletion({"_id": "doc-A", "owner_user_id": "user-A"})
    with pytest.raises(ForbiddenError):
        asyncio.run(service.delete_owned("doc-A", authenticated_user_id="user-B", session_id=None))
    service.pipeline.vector_store.delete_version_chunks.assert_not_awaited()
    service.documents.delete_by_id.assert_not_awaited()


def test_delete_owned_missing_document_raises_not_found() -> None:
    service = _service_with_mocked_deletion(None)
    with pytest.raises(NotFoundError):
        asyncio.run(service.delete_owned("doc-missing", authenticated_user_id="user-A", session_id=None))
    service.pipeline.vector_store.delete_version_chunks.assert_not_awaited()


def test_delete_owned_records_an_audit_trail_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    async def fake_record(self: AuditService, **kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(AuditService, "record", fake_record)
    service = _service_with_mocked_deletion({"_id": "doc-A", "owner_user_id": "user-A"})
    asyncio.run(service.delete_owned("doc-A", authenticated_user_id="user-A", session_id="session-A"))
    assert len(calls) == 1
    assert calls[0]["action"] == "document_deleted"
    assert calls[0]["resource_id"] == "doc-A"
