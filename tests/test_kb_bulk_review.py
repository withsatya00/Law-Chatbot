"""Speeds up human review of automation-discovered documents
(`KnowledgeBaseIngestionService.list_automation_review_queue` /
`bulk_review_automated_documents`) without weakening the existing
`verification_status`/`review_status` safety gate: every result still goes
through the same `update_jurisdiction_metadata` path, one document at a
time, so a batch can never approve something the single-document endpoint
would have rejected.

Same in-memory fake-collection convention as `test_kb_jurisdiction_correction.py`.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import BadRequestError
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService


def _get_path(doc: dict[str, Any], dotted_key: str) -> Any:
    value: Any = doc
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _matches(value: Any, expected: Any, present: bool) -> bool:
    if isinstance(expected, dict) and "$exists" in expected:
        return present == expected["$exists"]
    if isinstance(expected, dict) and "$in" in expected:
        return value in expected["$in"]
    return value == expected


class _FakeCollection:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def find(self, query: dict[str, Any] | None = None, *_a: Any, **_k: Any) -> Any:
        def matches_doc(d: dict[str, Any]) -> bool:
            for key, expected in (query or {}).items():
                present = _get_path(d, key) is not None
                if not _matches(_get_path(d, key), expected, present):
                    return False
            return True

        matches = [d for d in self.docs if matches_doc(d)]

        class _Cursor:
            def __init__(self, items: list[dict[str, Any]]) -> None:
                self.items = items

            def sort(self, *_a: Any, **_k: Any) -> Any:
                return self

            def limit(self, n: int, *_a: Any, **_k: Any) -> Any:
                return _Cursor(self.items[:n])

            def __aiter__(self) -> Any:
                async def _gen() -> Any:
                    for item in self.items:
                        yield item

                return _gen()

        return _Cursor(matches)

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
        return await self.update_one(query, update)


class _FakeDocumentRepo:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = {d["_id"]: d for d in docs}
        self.collection = _FakeCollection(docs)

    async def find_by_id(self, document_id: str) -> dict[str, Any] | None:
        return self._docs.get(document_id)


class _FakeDB:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self._collections = {"kb_automation_jobs": _FakeCollection(jobs)}

    def __getitem__(self, name: str) -> _FakeCollection:
        return self._collections[name]


def _document(doc_id: str, filename: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"_id": doc_id, "filename": filename, "metadata": metadata or {}}


def _chunk(chunk_id: str, source_document: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"_id": chunk_id, "metadata": {"source_document": source_document, **(metadata or {})}}


def _service_for(
    documents: list[dict[str, Any]], chunks: list[dict[str, Any]],
) -> KnowledgeBaseIngestionService:
    service = KnowledgeBaseIngestionService()
    service.documents = _FakeDocumentRepo(documents)
    service.chunks = MagicMock()
    service.chunks.collection = _FakeCollection(chunks)
    return service


def _job(job_id: str, document_id: str, candidate: dict[str, Any], status: str = "quarantined") -> dict[str, Any]:
    return {"_id": job_id, "document_id": document_id, "status": status, "candidate": candidate, "updated_at": "2026-09-22"}


def test_review_queue_surfaces_already_known_structural_fields(monkeypatch) -> None:
    document = _document(
        "d1", "odisha_act.pdf",
        metadata={"source_url": "https://law.odisha.gov.in/act.pdf", "issuing_level": "state", "applicability": "specific_states", "applicable_state_codes": ["OR"]},
    )
    job = _job("j1", "d1", {
        "title": "The Odisha Repealing Act, 2021", "jurisdiction_code": "OR",
        "document_type": "bare_act", "act_number": "12", "enactment_year": 2021,
        "url": "https://law.odisha.gov.in/act.pdf", "confidence_score": 0.5,
    })
    service = _service_for([document], [])
    monkeypatch.setattr("app.services.kb_ingestion_service.mongodb", MagicMock(db=_FakeDB([job])))

    items = asyncio.run(service.list_automation_review_queue(limit=10))

    assert len(items) == 1
    assert items[0]["document_id"] == "d1"
    assert items[0]["title"] == "The Odisha Repealing Act, 2021"
    assert items[0]["jurisdiction_code"] == "OR"
    assert items[0]["source_url"] == "https://law.odisha.gov.in/act.pdf"
    assert items[0]["issuing_level"] == "state"


def test_review_queue_filters_by_jurisdiction_code(monkeypatch) -> None:
    jobs = [
        _job("j1", "d1", {"title": "MN Act", "jurisdiction_code": "MN"}),
        _job("j2", "d2", {"title": "NL Act", "jurisdiction_code": "NL"}),
    ]
    documents = [_document("d1", "mn.pdf"), _document("d2", "nl.pdf")]
    service = _service_for(documents, [])
    monkeypatch.setattr("app.services.kb_ingestion_service.mongodb", MagicMock(db=_FakeDB(jobs)))

    items = asyncio.run(service.list_automation_review_queue(jurisdiction_codes=("MN",)))

    assert len(items) == 1
    assert items[0]["title"] == "MN Act"


def test_review_queue_only_shows_quarantined_jobs_with_a_document(monkeypatch) -> None:
    jobs = [
        _job("j1", "d1", {"title": "Ready"}, status="quarantined"),
        _job("j2", "d2", {"title": "Still downloading"}, status="claimed"),
    ]
    service = _service_for([_document("d1", "a.pdf"), _document("d2", "b.pdf")], [])
    monkeypatch.setattr("app.services.kb_ingestion_service.mongodb", MagicMock(db=_FakeDB(jobs)))

    items = asyncio.run(service.list_automation_review_queue())

    assert [item["title"] for item in items] == ["Ready"]


def test_bulk_review_approves_every_document_that_already_has_complete_metadata(monkeypatch) -> None:
    documents = [
        _document("d1", "act1.pdf", metadata={
            "issuing_level": "state", "applicability": "specific_states",
            "applicable_state_codes": ["MN"], "source_url": "https://assembly.mn.gov.in/act1.pdf",
            "jurisdiction_source_type": "bare_act",
        }),
        _document("d2", "act2.pdf", metadata={
            "issuing_level": "state", "applicability": "specific_states",
            "applicable_state_codes": ["NL"], "source_url": "https://nagaland.gov.in/act2.pdf",
            "jurisdiction_source_type": "bare_act",
        }),
    ]
    chunks = [_chunk("c1", "act1.pdf"), _chunk("c2", "act2.pdf")]
    service = _service_for(documents, chunks)
    monkeypatch.setattr("app.services.kb_ingestion_service.response_cache.bump_generation", AsyncMock())

    results = asyncio.run(service.bulk_review_automated_documents(["d1", "d2"], "adv.sharma@example.com"))

    assert {r["document_id"]: r["outcome"] for r in results} == {"d1": "approved", "d2": "approved"}
    assert documents[0]["metadata"]["verification_status"] == "verified"
    assert documents[0]["metadata"]["verified_by"] == "adv.sharma@example.com"
    assert chunks[0]["metadata"]["review_status"] == "approved"
    assert chunks[1]["metadata"]["review_status"] == "approved"


def test_bulk_review_does_not_approve_a_document_with_incomplete_structural_fields(monkeypatch) -> None:
    """A legacy document with no issuing_level/source_url must stay
    needs_review even when it rides along in a batch with good documents --
    this is the core "not a bulk auto-approve" guarantee."""
    documents = [
        _document("d1", "good.pdf", metadata={
            "issuing_level": "state", "applicability": "specific_states",
            "applicable_state_codes": ["MN"], "source_url": "https://assembly.mn.gov.in/good.pdf",
        }),
        _document("d2", "legacy.pdf", metadata={}),
    ]
    chunks = [_chunk("c1", "good.pdf"), _chunk("c2", "legacy.pdf")]
    service = _service_for(documents, chunks)
    monkeypatch.setattr("app.services.kb_ingestion_service.response_cache.bump_generation", AsyncMock())

    results = asyncio.run(service.bulk_review_automated_documents(["d1", "d2"], "reviewer@example.com"))

    by_id = {r["document_id"]: r for r in results}
    assert by_id["d1"]["outcome"] == "approved"
    assert by_id["d2"]["outcome"] == "needs_review"
    assert "Issuing level is unknown." in by_id["d2"]["review_reasons"]
    assert chunks[1]["metadata"]["review_status"] == "needs_review"


def test_bulk_review_does_not_let_automated_provenance_downgrade_verified_to_inferred(monkeypatch) -> None:
    """Regression: `normalize_jurisdiction` reads `metadata_provenance`
    straight off the submitted dict (`raw.get("metadata_provenance",
    provenance)`), IN PREFERENCE to `update_jurisdiction_metadata`'s own
    `provenance=PROVENANCE_MANUAL`. Every automation-discovered document
    carries `metadata_provenance: "automated_official"` on its existing
    record; spreading that into the bulk-review submission unchanged made
    `verification_status: "verified"` silently become `"inferred"` instead
    (`_resolve_verification`'s "a non-manual provenance can never claim
    verified" rule) -- confirmed live 2026-09-22 against 4 real documents
    stuck exactly this way, never reaching `approved` no matter how many
    times they were submitted."""
    documents = [_document("d1", "act1.pdf", metadata={
        "issuing_level": "state", "applicability": "specific_states",
        "applicable_state_codes": ["MH"], "source_url": "https://bombayhighcourt.gov.in/act1.pdf",
        "jurisdiction_source_type": "bare_act", "metadata_provenance": "automated_official",
    })]
    chunks = [_chunk("c1", "act1.pdf")]
    service = _service_for(documents, chunks)
    monkeypatch.setattr("app.services.kb_ingestion_service.response_cache.bump_generation", AsyncMock())

    results = asyncio.run(service.bulk_review_automated_documents(["d1"], "reviewer@example.com"))

    assert results[0]["outcome"] == "approved"
    assert documents[0]["metadata"]["verification_status"] == "verified"
    assert documents[0]["metadata"]["review_status"] == "approved"


def test_bulk_review_reports_a_missing_document_without_failing_the_whole_batch(monkeypatch) -> None:
    documents = [_document("d1", "good.pdf", metadata={
        "issuing_level": "state", "applicability": "specific_states",
        "applicable_state_codes": ["MN"], "source_url": "https://assembly.mn.gov.in/good.pdf",
    })]
    chunks = [_chunk("c1", "good.pdf")]
    service = _service_for(documents, chunks)
    monkeypatch.setattr("app.services.kb_ingestion_service.response_cache.bump_generation", AsyncMock())

    results = asyncio.run(service.bulk_review_automated_documents(["d1", "does-not-exist"], "reviewer@example.com"))

    by_id = {r["document_id"]: r for r in results}
    assert by_id["d1"]["outcome"] == "approved"
    assert by_id["does-not-exist"]["outcome"] == "failed"
    assert by_id["does-not-exist"]["reason"] == "Document not found."


def test_bulk_review_requires_a_verified_by_name() -> None:
    service = _service_for([_document("d1", "a.pdf")], [])
    with pytest.raises(BadRequestError):
        asyncio.run(service.bulk_review_automated_documents(["d1"], "  "))


# ------------------------------------------------------------- live needs-review listing


def test_list_needs_review_documents_reads_live_status_not_a_stale_snapshot() -> None:
    """Regression for the bug that made the Streamlit review page report
    2,174 documents against 21 real ones: it used to read the staging
    ledger's write-once `review_status` snapshot from indexing time, which
    never updates after a later approval. This queries `uploaded_documents`
    directly, so it only ever reflects the CURRENT status."""
    documents = [
        _document("d1", "act1.pdf", metadata={"review_status": "needs_review", "source_url": "https://a.gov.in/1.pdf"}),
        _document("d2", "act2.pdf", metadata={"review_status": "approved", "source_url": "https://a.gov.in/2.pdf"}),
        _document("d3", "act3.pdf", metadata={"review_status": "needs_review", "source_url": "https://a.gov.in/3.pdf"}),
    ]
    service = _service_for(documents, [])

    items = asyncio.run(service.list_needs_review_documents())

    assert {item["document_id"] for item in items} == {"d1", "d3"}
    d1 = next(item for item in items if item["document_id"] == "d1")
    assert d1["filename"] == "act1.pdf"
    assert d1["jurisdiction_metadata"]["source_url"] == "https://a.gov.in/1.pdf"


def test_list_needs_review_documents_respects_the_limit() -> None:
    documents = [
        _document(f"d{i}", f"act{i}.pdf", metadata={"review_status": "needs_review"})
        for i in range(5)
    ]
    service = _service_for(documents, [])

    items = asyncio.run(service.list_needs_review_documents(limit=2))

    assert len(items) == 2
