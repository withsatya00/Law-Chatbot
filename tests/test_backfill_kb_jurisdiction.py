"""Phase 1 "Jurisdiction-Aware Knowledge Base": `scripts/backfill_kb_jurisdiction.py`.

In-memory fake `documents`/`embeddings_metadata` collections, same shape and
convention as `test_kb_staging_reconciliation.py`'s `_FakeStagingCollection`
(this suite never touches a live MongoDB) -- extended with dotted-path
`metadata.*` query matching and `update_many`, which the backfill's queries
and writes both rely on.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from scripts.backfill_kb_jurisdiction import backfill


def _get_path(doc: dict[str, Any], dotted_key: str) -> Any:
    value: Any = doc
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _has_path(doc: dict[str, Any], dotted_key: str) -> bool:
    value: Any = doc
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


class _FakeCollection:
    """Minimal in-memory Motor-collection stand-in supporting exactly the
    query/update shapes `backfill_kb_jurisdiction.py` issues: `$in`/`$exists`
    in queries, dotted `metadata.*` keys in both queries and `$set`."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def _matches(self, doc: dict[str, Any], query: dict[str, Any]) -> bool:
        for key, expected in query.items():
            if isinstance(expected, dict) and "$in" in expected:
                if _get_path(doc, key) not in expected["$in"]:
                    return False
            elif isinstance(expected, dict) and "$exists" in expected:
                if _has_path(doc, key) != expected["$exists"]:
                    return False
            elif _get_path(doc, key) != expected:
                return False
        return True

    def find(self, query: dict[str, Any] | None = None, *_args: Any, **_kwargs: Any) -> Any:
        matches = [doc for doc in self.docs if self._matches(doc, query or {})]

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
            if self._matches(doc, query):
                self._apply_set(doc, update.get("$set", {}))
                modified += 1
                break
        return MagicMock(modified_count=modified)

    async def update_many(self, query: dict[str, Any], update: dict[str, Any]) -> Any:
        modified = 0
        for doc in self.docs:
            if self._matches(doc, query):
                self._apply_set(doc, update.get("$set", {}))
                modified += 1
        return MagicMock(modified_count=modified)

    @staticmethod
    def _apply_set(doc: dict[str, Any], updates: dict[str, Any]) -> None:
        for dotted_key, value in updates.items():
            parts = dotted_key.split(".")
            cursor = doc
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value


def _document(doc_id: str, filename: str, metadata: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return {"_id": doc_id, "filename": filename, "metadata": metadata or {}, **extra}


def _chunk(chunk_id: str, source_document: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"_id": chunk_id, "metadata": {"source_document": source_document, **(metadata or {})}}


def test_dry_run_reports_counts_without_writing_anything() -> None:
    documents = _FakeCollection([_document("d1", "central_act.pdf")])
    chunks = _FakeCollection([_chunk("c1", "central_act.pdf")])

    result = asyncio.run(backfill(apply=False, documents=documents, chunks=chunks))

    assert result["dry_run"] is True
    assert result["documents_scanned"] == 1
    assert result["documents_marked_needs_review"] == 1  # no jurisdiction info at all -> unknown
    # Read-only: nothing was written to either fake collection.
    assert "jurisdiction_schema_version" not in documents.docs[0]["metadata"]
    assert "jurisdiction_schema_version" not in chunks.docs[0]["metadata"]


def test_apply_marks_missing_applicability_unknown_and_needs_review_on_documents_and_chunks() -> None:
    documents = _FakeCollection([_document("d1", "central_act.pdf")])
    chunks = _FakeCollection([_chunk("c1", "central_act.pdf"), _chunk("c2", "central_act.pdf")])

    result = asyncio.run(backfill(apply=True, documents=documents, chunks=chunks))

    assert result["dry_run"] is False
    assert result["documents_marked_needs_review"] == 1
    assert result["chunks_updated"] == 2
    doc_metadata = documents.docs[0]["metadata"]
    assert doc_metadata["applicability"] == "unknown"
    assert doc_metadata["review_status"] == "needs_review"
    assert doc_metadata["jurisdiction_schema_version"] == 1
    for chunk in chunks.docs:
        assert chunk["metadata"]["review_status"] == "needs_review"
        assert chunk["metadata"]["applicable_state_codes"] == []  # never guessed


def test_apply_never_guesses_a_state_or_date_from_missing_metadata() -> None:
    documents = _FakeCollection([_document("d1", "some_act.pdf")])
    chunks = _FakeCollection([_chunk("c1", "some_act.pdf")])

    asyncio.run(backfill(apply=True, documents=documents, chunks=chunks))

    metadata = documents.docs[0]["metadata"]
    assert metadata["effective_from"] is None
    assert metadata["effective_to"] is None
    assert metadata["applicable_state_codes"] == []


def test_apply_preserves_existing_trustworthy_metadata() -> None:
    """A document that already has SOME jurisdiction fields on record (e.g. a
    prior manual fix) keeps them -- the backfill only fills in what's
    actually missing, never overwrites what's already there."""
    documents = _FakeCollection(
        [_document("d1", "mh_act.pdf", metadata={"issuing_level": "state", "applicable_state_codes": ["MH"]})]
    )
    chunks = _FakeCollection([_chunk("c1", "mh_act.pdf")])

    asyncio.run(backfill(apply=True, documents=documents, chunks=chunks))

    metadata = documents.docs[0]["metadata"]
    assert metadata["issuing_level"] == "state"
    assert metadata["applicable_state_codes"] == ["MH"]
    # applicability was never supplied, so it's still unknown/needs_review --
    # existing fields are preserved, not used to fabricate the missing one.
    assert metadata["applicability"] == "unknown"
    assert metadata["review_status"] == "needs_review"


def test_repeated_apply_is_idempotent() -> None:
    documents = _FakeCollection([_document("d1", "central_act.pdf")])
    chunks = _FakeCollection([_chunk("c1", "central_act.pdf")])

    first = asyncio.run(backfill(apply=True, documents=documents, chunks=chunks))
    assert first["documents_updated"] == 1
    assert first["chunks_updated"] == 1

    second = asyncio.run(backfill(apply=True, documents=documents, chunks=chunks))
    # Every candidate was already stamped with jurisdiction_schema_version by
    # the first run, so the query that selects candidates finds none the
    # second time -- zero further writes.
    assert second["documents_scanned"] == 0
    assert second["documents_updated"] == 0
    assert second["chunks_updated"] == 0


def test_private_owned_documents_are_never_touched_or_promoted() -> None:
    """A per-user private upload (`owner_user_id` set) must never be selected
    by the backfill -- promoting it into the shared/reviewed jurisdiction
    scheme is exactly the "private documents shared KB mein promote na hon"
    failure this excludes structurally, not just by convention."""
    documents = _FakeCollection(
        [
            _document("d1", "central_act.pdf"),  # curated KB document
            _document("d2", "my_lease.pdf", owner_user_id="user-42"),  # private
            _document("d3", "my_note.pdf", owner_session_id="session-9"),  # private
        ]
    )
    chunks = _FakeCollection(
        [_chunk("c1", "central_act.pdf"), _chunk("c2", "my_lease.pdf"), _chunk("c3", "my_note.pdf")]
    )

    result = asyncio.run(backfill(apply=True, documents=documents, chunks=chunks))

    assert result["documents_scanned"] == 1
    private_doc_1 = next(d for d in documents.docs if d["_id"] == "d2")
    private_doc_2 = next(d for d in documents.docs if d["_id"] == "d3")
    assert "jurisdiction_schema_version" not in private_doc_1["metadata"]
    assert "jurisdiction_schema_version" not in private_doc_2["metadata"]
    private_chunk_1 = next(c for c in chunks.docs if c["_id"] == "c2")
    private_chunk_2 = next(c for c in chunks.docs if c["_id"] == "c3")
    assert "review_status" not in private_chunk_1["metadata"]
    assert "review_status" not in private_chunk_2["metadata"]


def test_apply_invalidates_the_response_cache(monkeypatch) -> None:
    documents = _FakeCollection([_document("d1", "central_act.pdf")])
    chunks = _FakeCollection([_chunk("c1", "central_act.pdf")])

    fake_bump = AsyncMock()
    import app.cache.response_cache as response_cache_module

    monkeypatch.setattr(response_cache_module.response_cache, "bump_generation", fake_bump)

    asyncio.run(backfill(apply=True, documents=documents, chunks=chunks))

    fake_bump.assert_awaited_once()
