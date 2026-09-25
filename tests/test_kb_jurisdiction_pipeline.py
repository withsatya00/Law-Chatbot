"""Phase 1 "Jurisdiction-Aware Knowledge Base": metadata document -> chunk
propagation through the REAL `IndexingPipeline.index_file` (loader, section-
aware chunker, and metadata extractor all run for real over a plain-text
fixture), plus the retrieval-side review-status gate.

Follows `test_kb_ingestion_service.py`'s convention of faking out only the
Mongo-backed repositories/vector store (no live MongoDB in this suite) while
exercising the real chunking/metadata code path under test.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

from app.rag import kb_jurisdiction as kj
from app.rag.chunker import SectionAwareChunker
from app.rag.loader import DocumentLoader
from app.rag.pipeline import IndexingPipeline
from app.rag.quality import DocumentQualityChecker


class _EmptyCursor:
    def __aiter__(self) -> Any:
        async def _gen() -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

        return _gen()


class _FakeVectorStore:
    """Mirrors just enough of `MongoVectorStore`'s P0-2 staging/activation
    contract (`count_by_document_id`/`activate_version`/
    `delete_version_chunks`) for the real `IndexingPipeline.index_file` to run
    end to end -- tracked in-memory by `document_id` rather than Mongo."""

    def __init__(self) -> None:
        self.upserted: list[Any] = []
        self.activated: list[tuple[str, str | None]] = []
        self.deleted_document_ids: list[str] = []

    async def upsert_chunks(self, chunks: list[Any]) -> None:
        self.upserted.extend(chunks)

    async def delete_by_source(self, source_document: str) -> int:
        return 0

    async def count_by_document_id(self, document_id: str) -> int:
        return sum(1 for chunk in self.upserted if chunk.document_id == document_id)

    async def activate_version(self, new_document_id: str, old_document_id: str | None) -> None:
        self.activated.append((new_document_id, old_document_id))
        for chunk in self.upserted:
            if chunk.document_id == new_document_id:
                chunk.metadata["document_status"] = "active"
            elif old_document_id and chunk.document_id == old_document_id:
                chunk.metadata["document_status"] = "superseded"

    async def delete_version_chunks(self, document_id: str) -> int:
        self.deleted_document_ids.append(document_id)
        before = len(self.upserted)
        self.upserted = [chunk for chunk in self.upserted if chunk.document_id != document_id]
        return before - len(self.upserted)


class _FakeEmbeddings:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeCollection:
    def find(self, *_args: Any, **_kwargs: Any) -> Any:
        return _EmptyCursor()


class _FakeVersionRepository:
    def __init__(self) -> None:
        self.collection = _FakeCollection()
        self.inserted: list[dict[str, Any]] = []
        self.activated: list[tuple[str, dict[str, Any]]] = []
        self.deleted_ids: list[str] = []

    async def latest_for_source(self, _source_document: str, **_scope: Any) -> dict[str, Any] | None:
        return None

    async def insert(self, document: dict[str, Any]) -> str:
        self.inserted.append(document)
        return str(uuid4())

    async def update_by_id(self, *_args: Any, **_kwargs: Any) -> bool:
        return True

    async def claim_staging(self, document: dict[str, Any]) -> str | None:
        return await self.insert(document)

    async def activate(self, version_id: str, updates: dict[str, Any]) -> bool:
        self.activated.append((version_id, updates))
        return True

    async def delete_by_id(self, version_id: str) -> bool:
        self.deleted_ids.append(version_id)
        return True


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.inserted: list[dict[str, Any]] = []
        self.collection = _FakeCollection()

    async def insert(self, document: dict[str, Any]) -> str:
        self.inserted.append(document)
        return str(document["_id"])

    async def update_by_id(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


_ACT_TEXT = (
    "THE SAMPLE STATE ACT, 2020\n"
    "(ACT NO. 5 OF 2020)\n\n"
    "5. Short title.—This Act may be called the Sample State Act, 2020.\n\n"
    "12. Grievance redressal.—(1) Every complaint shall be filed with the "
    "designated authority within thirty days.\n\n"
    "20. Penalty.—Any person contravening section 12 shall be liable to a fine.\n"
)


def _build_pipeline() -> tuple[IndexingPipeline, _FakeVectorStore]:
    vector_store = _FakeVectorStore()
    pipeline = IndexingPipeline(
        loader=DocumentLoader(),
        chunker=SectionAwareChunker(),
        embeddings=_FakeEmbeddings(),
        vector_store=vector_store,
    )
    pipeline.documents = _FakeDocumentRepository()  # type: ignore[assignment]
    pipeline.versions = _FakeVersionRepository()  # type: ignore[assignment]
    pipeline.quality_checker = DocumentQualityChecker()
    return pipeline, vector_store


def test_document_jurisdiction_metadata_reaches_every_searchable_chunk(tmp_path) -> None:
    """Objective item 4: "Metadata document record se searchable chunks tak
    preserve karo" -- a State Act's jurisdiction metadata, supplied once at
    the document level, must land on every chunk the section-aware chunker
    produces, not just the first."""
    source = tmp_path / "sample_state_act.txt"
    source.write_text(_ACT_TEXT, encoding="utf-8")

    normalized = kj.normalize_jurisdiction(
        {
            "issuing_level": "state",
            "applicability": "specific_states",
            "applicable_state_codes": ["MH"],
            "jurisdiction_source_type": "bare_act",
            "source_url": "https://example.gov.in/mh-sample-act",
            "verification_status": "verified",
            "verified_by": "admin@example.com",
        }
    )
    jurisdiction_fields = kj.document_metadata_fields(normalized)

    pipeline, vector_store = _build_pipeline()
    document_id, _language, chunks = asyncio.run(
        pipeline.index_file(source, jurisdiction_metadata=jurisdiction_fields)
    )

    assert len(chunks) >= 2  # the section-aware chunker actually split the Act
    for chunk in chunks:
        assert chunk.metadata["issuing_level"] == "state"
        assert chunk.metadata["applicability"] == "specific_states"
        assert chunk.metadata["applicable_state_codes"] == ["MH"]
        assert chunk.metadata["verification_status"] == "verified"
        assert chunk.metadata["review_status"] == kj.REVIEW_APPROVED
    assert vector_store.upserted == chunks
    assert pipeline.documents.inserted[0]["metadata"]["issuing_level"] == "state"  # document record too
    assert document_id


def test_original_filename_is_recorded_separately_from_the_storage_key(tmp_path) -> None:
    """Security/correctness finding K2: `path`'s own name is a server-
    generated storage key, never the caller's real filename -- passing
    `original_filename` must record it as its OWN field, leaving
    `document.filename`/`source_document` (what version lineage/dedup key
    on) untouched."""
    storage_path = tmp_path / "a1b2c3d4-5678-90ab-cdef-1234567890ab.txt"
    storage_path.write_text(_ACT_TEXT, encoding="utf-8")
    pipeline, _vector_store = _build_pipeline()

    document_id, _language, _chunks = asyncio.run(
        pipeline.index_file(storage_path, original_filename="my_rent_agreement.txt")
    )

    assert document_id
    inserted = pipeline.documents.inserted[0]
    assert inserted["original_filename"] == "my_rent_agreement.txt"
    assert inserted["filename"] == storage_path.name  # unchanged -- still the storage-key-derived identity


def test_original_filename_defaults_to_none_when_the_caller_has_no_concept_of_one(tmp_path) -> None:
    """KB ingestion (and every caller before this fix) never passes
    `original_filename` -- must not silently default to the storage key,
    which would make a reader unable to tell "no real name" apart from "the
    filename genuinely is this UUID"."""
    source = tmp_path / "sample_state_act.txt"
    source.write_text(_ACT_TEXT, encoding="utf-8")
    pipeline, _vector_store = _build_pipeline()

    asyncio.run(pipeline.index_file(source))

    assert pipeline.documents.inserted[0]["original_filename"] is None


def test_section_level_override_survives_real_chunking() -> None:
    """Section 12 of the sample Act is commenced only in Maharashtra, while
    the rest of the Act (e.g. section 20) is all-India -- the override must
    land on section 12's own chunk and nowhere else."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "override_act.txt"
        source.write_text(_ACT_TEXT, encoding="utf-8")

        normalized = kj.normalize_jurisdiction(
            {
                "issuing_level": "central",
                "applicability": "all_india",
                "jurisdiction_source_type": "bare_act",
                "source_url": "https://example.gov.in/central-sample-act",
                "verification_status": "verified",
                "verified_by": "admin@example.com",
                "section_overrides": {
                    "12": {"applicability": "specific_states", "applicable_state_codes": ["MH"], "effective_from": "2021-04-01"}
                },
            }
        )
        jurisdiction_fields = kj.document_metadata_fields(normalized)

        pipeline, _vector_store = _build_pipeline()
        _document_id, _language, chunks = asyncio.run(
            pipeline.index_file(source, jurisdiction_metadata=jurisdiction_fields)
        )

        by_section = {chunk.metadata.get("section_number"): chunk for chunk in chunks}
        assert "12" in by_section, f"expected a chunk for section 12, got sections {list(by_section)}"
        section_12 = by_section["12"]
        assert section_12.metadata["applicability"] == "specific_states"
        assert section_12.metadata["applicable_state_codes"] == ["MH"]
        assert section_12.metadata["effective_from"] == "2021-04-01"
        assert section_12.metadata["section_override_applied"] is True

        section_20 = by_section.get("20")
        assert section_20 is not None
        assert section_20.metadata["applicability"] == "all_india"
        assert "section_override_applied" not in section_20.metadata


def test_owner_scoped_upload_without_jurisdiction_metadata_is_unaffected() -> None:
    """Private per-user uploads never pass `jurisdiction_metadata` -- chunks
    must carry no jurisdiction/review_status keys at all, exactly as before
    this phase, so `LegalRetriever`'s new default filter (which matches an
    absent `review_status`) never excludes a user's own document."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "private_upload.txt"
        source.write_text("A short private note about my landlord dispute.", encoding="utf-8")

        pipeline, _vector_store = _build_pipeline()
        _document_id, _language, chunks = asyncio.run(
            pipeline.index_file(source, owner_user_id="user-123")
        )

        assert chunks
        for chunk in chunks:
            assert "review_status" not in chunk.metadata
            assert "issuing_level" not in chunk.metadata


# ---------------------------------------------------------------------------
# Retrieval-side gate: `LegalRetriever.retrieve` must exclude needs_review/
# pending chunks from shared search by default (objective item 4 + the
# explicitly allowed "minimal retrieval ... change" for review-status
# exclusion).
# ---------------------------------------------------------------------------


def test_retriever_defaults_to_excluding_unreviewed_chunks() -> None:
    from app.rag.retriever import LegalRetriever

    retriever = LegalRetriever(embeddings=_FakeEmbeddings(), vector_store=AsyncMock())
    retriever.vector_store.search = AsyncMock(return_value=[])

    asyncio.run(retriever.retrieve("What is the penalty under section 20?", top_k=3))

    called_filters = retriever.vector_store.search.call_args.args[3]
    assert called_filters["review_status"] == kj.REVIEW_APPROVED


def test_chat_service_shared_branch_gates_but_private_branch_does_not() -> None:
    """Gap 1 regression, exercised against the REAL filter-matching code
    (`BM25Index._matches_filters`, which mirrors `MongoVectorStore.
    _mongo_filter`'s semantics) rather than re-deriving the logic in the
    test: the exact `$or` shape `ChatService._prepare_rag_context` builds
    must let an approved shared chunk through, reject a needs_review shared
    chunk, and let a user's own private chunk through regardless of
    review_status."""
    from app.rag.bm25_index import BM25Index

    filters = {
        "$or": [
            {"owner_session_id": [None], "owner_user_id": [None], "review_status": "approved"},
            {"owner_session_id": ["session-1"], "owner_user_id": [None]},
        ]
    }
    index = BM25Index()

    approved_shared = {"review_status": "approved"}
    needs_review_shared = {"review_status": "needs_review"}
    never_reviewed_legacy_shared = {}  # no owner, no review_status at all
    own_private_doc = {"owner_session_id": "session-1"}  # no review_status field

    assert index._matches_filters(approved_shared, filters) is True
    assert index._matches_filters(needs_review_shared, filters) is False
    assert index._matches_filters(never_reviewed_legacy_shared, filters) is False
    assert index._matches_filters(own_private_doc, filters) is True


def test_retriever_does_not_override_a_caller_supplied_review_status_filter() -> None:
    """`setdefault` semantics: admin review tooling that explicitly asks for
    `needs_review` chunks (a capability outside this phase's scope, but the
    mechanism must not accidentally foreclose it) is not silently overridden."""
    from app.rag.retriever import LegalRetriever

    retriever = LegalRetriever(embeddings=_FakeEmbeddings(), vector_store=AsyncMock())
    retriever.vector_store.search = AsyncMock(return_value=[])

    asyncio.run(
        retriever.retrieve("internal review query", top_k=3, filters={"review_status": kj.REVIEW_NEEDS_REVIEW})
    )

    called_filters = retriever.vector_store.search.call_args.args[3]
    assert called_filters["review_status"] == kj.REVIEW_NEEDS_REVIEW
