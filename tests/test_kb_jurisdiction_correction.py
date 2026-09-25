"""Phase 1 "Jurisdiction-Aware Knowledge Base" gap 2: correcting/publishing an
ALREADY-INDEXED document's jurisdiction metadata
(`KnowledgeBaseIngestionService.update_jurisdiction_metadata`) without a
re-upload or re-embedding, and without ever touching
`DocumentQualityChecker`'s content-hash dedup gate.

Same in-memory fake-collection convention as `test_backfill_kb_jurisdiction.py`
-- no live MongoDB in this suite.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import BadRequestError, NotFoundError
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService


def _get_path(doc: dict[str, Any], dotted_key: str) -> Any:
    value: Any = doc
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


class _FakeCollection:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def find(self, query: dict[str, Any] | None = None, *_a: Any, **_k: Any) -> Any:
        matches = [d for d in self.docs if all(_get_path(d, k) == v for k, v in (query or {}).items())]

        class _Cursor:
            def __aiter__(self) -> Any:
                async def _gen() -> Any:
                    for item in matches:
                        yield item

                return _gen()

        return _Cursor()

    async def update_one(self, query: dict[str, Any], update: dict[str, Any]) -> Any:
        modified = 0
        for doc in self.docs:
            if all(_get_path(doc, k) == v for k, v in query.items()):
                for dotted_key, value in update.get("$set", {}).items():
                    parts = dotted_key.split(".")
                    cursor = doc
                    for part in parts[:-1]:
                        cursor = cursor.setdefault(part, {})
                    cursor[parts[-1]] = value
                modified += 1
        return MagicMock(modified_count=modified)

    async def update_many(self, query: dict[str, Any], update: dict[str, Any]) -> Any:
        modified = 0
        for doc in self.docs:
            if all(_get_path(doc, k) == v for k, v in query.items()):
                for dotted_key, value in update.get("$set", {}).items():
                    parts = dotted_key.split(".")
                    cursor = doc
                    for part in parts[:-1]:
                        cursor = cursor.setdefault(part, {})
                    cursor[parts[-1]] = value
                modified += 1
        return MagicMock(modified_count=modified)


class _FakeDocumentRepo:
    def __init__(self, doc: dict[str, Any]) -> None:
        self._doc = doc
        self.collection = _FakeCollection([doc])

    async def find_by_id(self, document_id: str) -> dict[str, Any] | None:
        return self._doc if self._doc.get("_id") == document_id else None


def _service_for(document: dict[str, Any], chunks: list[dict[str, Any]]) -> KnowledgeBaseIngestionService:
    service = KnowledgeBaseIngestionService()
    service.documents = _FakeDocumentRepo(document)
    service.chunks = MagicMock()
    service.chunks.collection = _FakeCollection(chunks)
    return service


def _document(doc_id: str, filename: str, metadata: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return {"_id": doc_id, "filename": filename, "metadata": metadata or {}, **extra}


def _chunk(chunk_id: str, source_document: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"_id": chunk_id, "metadata": {"source_document": source_document, **(metadata or {})}}


def test_correcting_state_and_verifying_publishes_the_document(monkeypatch) -> None:
    document = _document("d1", "mh_act.pdf", metadata={"issuing_level": "state", "applicability": "unknown"})
    chunks = [_chunk("c1", "mh_act.pdf"), _chunk("c2", "mh_act.pdf")]
    service = _service_for(document, chunks)
    monkeypatch.setattr("app.services.kb_ingestion_service.response_cache.bump_generation", AsyncMock())

    result = asyncio.run(
        service.update_jurisdiction_metadata(
            "d1",
            {
                "issuing_level": "state",
                "applicability": "specific_states",
                "applicable_state_codes": ["MH"],
                "jurisdiction_source_type": "bare_act",
                "source_url": "https://example.gov.in/mh-act",
                "verification_status": "verified",
                "verified_by": "admin@example.com",
            },
        )
    )

    assert result["review_status"] == "approved"
    assert result["chunks_updated"] == 2
    for chunk in chunks:
        assert chunk["metadata"]["applicability"] == "specific_states"
        assert chunk["metadata"]["applicable_state_codes"] == ["MH"]
        assert chunk["metadata"]["review_status"] == "approved"
    assert document["metadata"]["review_status"] == "approved"


def test_incomplete_correction_stays_needs_review() -> None:
    document = _document("d1", "act.pdf")
    chunks = [_chunk("c1", "act.pdf")]
    service = _service_for(document, chunks)
    service_result = asyncio.run(
        service.update_jurisdiction_metadata("d1", {"issuing_level": "central"})
    )
    assert service_result["review_status"] == "needs_review"
    assert chunks[0]["metadata"]["review_status"] == "needs_review"


def test_invalid_state_code_is_rejected_without_writing_anything() -> None:
    document = _document("d1", "act.pdf")
    chunks = [_chunk("c1", "act.pdf")]
    service = _service_for(document, chunks)

    with pytest.raises(BadRequestError) as excinfo:
        asyncio.run(
            service.update_jurisdiction_metadata(
                "d1", {"applicability": "specific_states", "applicable_state_codes": ["ZZ"]}
            )
        )
    assert any("ZZ" in issue for issue in excinfo.value.details["issues"])
    assert "review_status" not in document["metadata"]
    assert "review_status" not in chunks[0]["metadata"]


def test_unknown_document_id_is_rejected() -> None:
    service = _service_for(_document("d1", "act.pdf"), [])
    with pytest.raises(NotFoundError):
        asyncio.run(service.update_jurisdiction_metadata("does-not-exist", {"issuing_level": "central"}))


def test_private_document_cannot_be_given_jurisdiction_metadata() -> None:
    """A private, owner-scoped document is never a jurisdiction-review
    target -- this is a Knowledge Base-only concept (Part 45/46 preserved)."""
    document = _document("d1", "my_lease.pdf", owner_user_id="user-42")
    service = _service_for(document, [_chunk("c1", "my_lease.pdf")])
    with pytest.raises(BadRequestError):
        asyncio.run(service.update_jurisdiction_metadata("d1", {"issuing_level": "state"}))


def test_correction_does_not_call_the_quality_checker_or_pipeline(monkeypatch) -> None:
    """Gap 2's core requirement: a metadata-only correction must never be
    routed through indexing/dedup -- verified here by asserting the pipeline
    and quality checker are simply never touched."""
    document = _document("d1", "act.pdf")
    chunks = [_chunk("c1", "act.pdf")]
    service = _service_for(document, chunks)
    service.pipeline = MagicMock()
    service.pipeline.index_file = AsyncMock(side_effect=AssertionError("must not be called"))
    service.quality = MagicMock()
    service.quality.assess = AsyncMock(side_effect=AssertionError("must not be called"))
    monkeypatch.setattr("app.services.kb_ingestion_service.response_cache.bump_generation", AsyncMock())

    asyncio.run(
        service.update_jurisdiction_metadata(
            "d1",
            {
                "issuing_level": "central", "applicability": "all_india",
                "jurisdiction_source_type": "bare_act", "source_url": "https://example.gov.in/act",
                "verification_status": "verified", "verified_by": "admin",
            },
        )
    )

    service.pipeline.index_file.assert_not_called()
    service.quality.assess.assert_not_called()
