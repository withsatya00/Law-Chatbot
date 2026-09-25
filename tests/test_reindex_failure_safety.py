"""P0-2 regression: reindexing must never take a source's previously-indexed
content offline before the new content is proven complete and active.

Three layers are exercised:

1. `DocumentVersionRepository.claim_staging`/`latest_for_source` -- the
   cross-process "one in-flight reindex per source" lock and the "a `staging`
   row is never `latest`" rule, against a fake collection that mirrors what a
   real unique sparse index on `reindex_lock_key` does (pre-check + a
   `DuplicateKeyError` on the race).
2. `IndexingPipeline.index_file`'s orchestration -- staged write, validation,
   activate-then-supersede ordering, and cleanup-on-failure -- against fakes
   of `vector_store`/`versions`/`documents` that can be told to fail at a
   specific step, mirroring `test_kb_jurisdiction_pipeline.py`'s convention
   (real loader/chunker, faked Mongo-backed collaborators, no live MongoDB).
3. Actual retrieval VISIBILITY, end to end, through a real (unfaked)
   `BM25Index` fed the same way `MongoVectorStore.upsert_chunks`/
   `activate_version`/`delete_version_chunks` would drive it -- so "the old
   version stays searchable" and "only the new version answers after a
   successful switch" are proven by real `.search()` calls, not by asserting
   a mock was called.
"""

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pymongo.errors import DuplicateKeyError

from app.core.exceptions import BadRequestError
from app.rag.bm25_index import BM25Index
from app.rag.chunker import SectionAwareChunker
from app.rag.loader import DocumentLoader
from app.rag.pipeline import IndexingPipeline
from app.rag.quality import DocumentQualityChecker
from app.rag.types import DocumentChunk
from app.repositories.versioning import DocumentVersionRepository

# ---------------------------------------------------------------------------
# Layer 1: the repository's own lock/latest-version logic.
# ---------------------------------------------------------------------------


class _UpdateResult:
    def __init__(self, modified_count: int) -> None:
        self.modified_count = modified_count


class _FakeVersionCollection:
    """Simulates a unique sparse index on `reindex_lock_key`: at most one
    document with a given non-null `reindex_lock_key` value at a time. Good
    enough to exercise `claim_staging`'s pre-check AND the `DuplicateKeyError`
    branch that closes the actual race a real index closes."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    async def find_one(self, query: dict[str, Any], sort: Any = None) -> dict[str, Any] | None:
        matches = [doc for doc in self.docs.values() if self._matches(doc, query)]
        if sort:
            field, direction = sort[0]
            matches.sort(key=lambda d: d.get(field, 0), reverse=direction < 0)
        return matches[0] if matches else None

    async def insert_one(self, document: dict[str, Any]) -> Any:
        lock_key = document.get("reindex_lock_key")
        if lock_key is not None and any(d.get("reindex_lock_key") == lock_key for d in self.docs.values()):
            raise DuplicateKeyError("uniq_reindex_lock")
        self.docs[document["_id"]] = document
        return document

    async def update_one(self, query: dict[str, Any], update: dict[str, Any]) -> "_UpdateResult":
        modified = 0
        for doc in self.docs.values():
            if self._matches(doc, query):
                doc.update(update.get("$set", {}))
                for key in update.get("$unset", {}):
                    doc.pop(key, None)
                modified = 1
                break
        return _UpdateResult(modified)

    async def delete_one(self, query: dict[str, Any]) -> None:
        for doc_id in [doc_id for doc_id, doc in self.docs.items() if self._matches(doc, query)]:
            del self.docs[doc_id]

    def find(self, query: dict[str, Any]) -> "_FakeCursor":
        return _FakeCursor([doc for doc in self.docs.values() if self._matches(doc, query)])

    @staticmethod
    def _matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
        for key, value in query.items():
            if isinstance(value, dict) and "$ne" in value:
                if doc.get(key) == value["$ne"]:
                    return False
            elif doc.get(key) != value:
                return False
        return True


class _FakeCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __aiter__(self) -> Any:
        async def _gen() -> Any:
            for row in self._rows:
                yield row

        return _gen()


class _FakeMongoForVersions:
    """`DocumentVersionRepository.collection` is a read-only property
    (`mongodb.db[self.collection_name]`) -- monkeypatching the module-level
    `mongodb` singleton, the way `test_index_reconciliation.py`'s `_mongo`
    helper does, is how a real repository instance gets a fake collection
    rather than a live database."""

    def __init__(self, fake: "_FakeVersionCollection") -> None:
        self._fake = fake

    @property
    def db(self) -> Any:
        return {DocumentVersionRepository.collection_name: self._fake}


def _repo_with_fake_collection(monkeypatch: pytest.MonkeyPatch) -> DocumentVersionRepository:
    # `app/repositories/base.py` does `from app.database.mongodb import
    # mongodb` at MODULE level (bound once, at first import) -- patching
    # `app.database.mongodb.mongodb` (as `test_index_reconciliation.py`'s
    # `_mongo` helper does) has no effect here, because that helper's target
    # (`IndexReconciler.analyze`) re-imports `mongodb` fresh INSIDE its
    # function body on every call, while `MongoRepository.collection` reads
    # the name `base.py` already bound. The fix is to patch that binding
    # directly.
    import app.repositories.base as base_module

    fake = _FakeVersionCollection()
    monkeypatch.setattr(base_module, "mongodb", _FakeMongoForVersions(fake))
    return DocumentVersionRepository()


def test_claim_staging_succeeds_when_no_lock_is_held(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _repo_with_fake_collection(monkeypatch)
    version_id = asyncio.run(
        repository.claim_staging({"_id": str(uuid4()), "source_document": "bns.pdf", "document_status": "staging"})
    )
    assert version_id is not None


def test_claim_staging_rejects_a_concurrent_attempt_for_the_same_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core concurrency guarantee: two attempts to reindex the SAME
    source at once -- the second is refused, not raced."""
    repository = _repo_with_fake_collection(monkeypatch)
    first = asyncio.run(
        repository.claim_staging({"_id": str(uuid4()), "source_document": "bns.pdf", "document_status": "staging"})
    )
    second = asyncio.run(
        repository.claim_staging({"_id": str(uuid4()), "source_document": "bns.pdf", "document_status": "staging"})
    )
    assert first is not None
    assert second is None


def test_claim_staging_allows_different_sources_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _repo_with_fake_collection(monkeypatch)
    first = asyncio.run(
        repository.claim_staging({"_id": str(uuid4()), "source_document": "bns.pdf", "document_status": "staging"})
    )
    second = asyncio.run(
        repository.claim_staging({"_id": str(uuid4()), "source_document": "bnss.pdf", "document_status": "staging"})
    )
    assert first is not None
    assert second is not None


def test_claim_staging_allows_a_retry_once_the_lock_is_released(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _repo_with_fake_collection(monkeypatch)
    version_id = asyncio.run(
        repository.claim_staging({"_id": str(uuid4()), "source_document": "bns.pdf", "document_status": "staging"})
    )
    assert version_id is not None
    asyncio.run(repository.activate(version_id, {"document_status": "active"}))
    retry_id = asyncio.run(
        repository.claim_staging({"_id": str(uuid4()), "source_document": "bns.pdf", "document_status": "staging"})
    )
    assert retry_id is not None


def test_latest_for_source_never_returns_a_staging_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """An abandoned `staging` row from a crashed attempt must not be picked
    up as "the version to increment from/supersede" by a later retry."""
    repository = _repo_with_fake_collection(monkeypatch)
    asyncio.run(
        repository.insert(
            {"_id": "v1", "source_document": "bns.pdf", "version_number": 1, "document_status": "active"}
        )
    )
    asyncio.run(
        repository.insert(
            {"_id": "v2-abandoned", "source_document": "bns.pdf", "version_number": 2, "document_status": "staging"}
        )
    )
    latest = asyncio.run(repository.latest_for_source("bns.pdf"))
    assert latest is not None
    assert latest["_id"] == "v1"


def test_find_stale_staging_surfaces_abandoned_rows_for_admin_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _repo_with_fake_collection(monkeypatch)
    asyncio.run(
        repository.insert(
            {"_id": "v2-abandoned", "source_document": "bns.pdf", "version_number": 2, "document_status": "staging"}
        )
    )
    stale = asyncio.run(repository.find_stale_staging())
    assert [row["_id"] for row in stale] == ["v2-abandoned"]


# ---------------------------------------------------------------------------
# Layer 2: `IndexingPipeline.index_file`'s orchestration, with failures
# injected at specific steps.
# ---------------------------------------------------------------------------


class _FailingVectorStore:
    """A configurable fake: `fail_on` names the method that should raise on
    its NEXT call (once, then behaves normally) -- lets a single test target
    one exact step of the staging/activation sequence."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.upserted: list[DocumentChunk] = []
        self.activate_calls: list[tuple[str, str | None]] = []
        self.deleted_document_ids: list[str] = []
        self.miscount_next_validation = False

    def _maybe_fail(self, name: str) -> None:
        if self.fail_on == name:
            self.fail_on = None  # fail exactly once
            raise RuntimeError(f"injected failure in {name}")

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        self._maybe_fail("upsert_chunks")
        self.upserted.extend(chunks)

    async def delete_by_source(self, source_document: str) -> int:
        return 0

    async def count_by_document_id(self, document_id: str) -> int:
        self._maybe_fail("count_by_document_id")
        count = sum(1 for chunk in self.upserted if chunk.document_id == document_id)
        if self.miscount_next_validation:
            self.miscount_next_validation = False
            return max(count - 1, 0)  # simulate a partial write
        return count

    async def activate_version(self, new_document_id: str, old_document_id: str | None) -> None:
        self._maybe_fail("activate_version")
        self.activate_calls.append((new_document_id, old_document_id))
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


class _FakeCollectionHandle:
    def __init__(self, store: dict[str, dict[str, Any]]) -> None:
        self._store = store

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        doc_id = query.get("_id")
        return self._store.get(doc_id)

    async def delete_one(self, query: dict[str, Any]) -> None:
        self._store.pop(query.get("_id"), None)

    def find(self, _query: dict[str, Any], _projection: dict[str, Any] | None = None) -> _FakeCursor:
        # `IndexingPipeline._known_hashes` is the only caller against this
        # handle; an empty result is fine here -- these tests never rely on
        # cross-source duplicate-hash detection, only on the staging/
        # activation sequence.
        return _FakeCursor([])


class _FakeVersionRepository:
    def __init__(self, existing: dict[str, Any] | None = None) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.collection = _FakeCollectionHandle(self.docs)
        self._existing_latest = existing
        self.deleted_ids: list[str] = []
        self.activated: list[tuple[str, dict[str, Any]]] = []
        self.updated: list[tuple[str, dict[str, Any]]] = []
        self.claim_should_fail = False

    async def latest_for_source(self, _source_document: str, **_scope: Any) -> dict[str, Any] | None:
        return self._existing_latest

    async def claim_staging(self, document: dict[str, Any]) -> str | None:
        if self.claim_should_fail:
            return None
        version_id = str(uuid4())
        self.docs[version_id] = {**document, "_id": version_id}
        return version_id

    async def activate(self, version_id: str, updates: dict[str, Any]) -> bool:
        self.activated.append((version_id, updates))
        if version_id in self.docs:
            self.docs[version_id].update(updates)
        return True

    async def update_by_id(self, version_id: str, updates: dict[str, Any]) -> bool:
        self.updated.append((version_id, updates))
        if version_id in self.docs:
            self.docs[version_id].update(updates)
        return True

    async def delete_by_id(self, version_id: str) -> bool:
        self.deleted_ids.append(version_id)
        self.docs.pop(version_id, None)
        return True


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.collection = _FakeCollectionHandle(self.docs)
        self.updated: list[tuple[str, dict[str, Any]]] = []

    async def insert(self, document: dict[str, Any]) -> str:
        self.docs[document["_id"]] = document
        return str(document["_id"])

    async def update_by_id(self, document_id: str, updates: dict[str, Any]) -> bool:
        self.updated.append((document_id, updates))
        if document_id in self.docs:
            self.docs[document_id].update(updates)
        return True


class _FakeEmbeddings:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("injected embedding provider failure")
        return [[0.1, 0.2, 0.3] for _ in texts]


_ACT_TEXT = (
    "5. Short title.—This Act may be called the Sample Act, 2020.\n\n"
    "12. Grievance redressal.—Every complaint shall be filed within thirty days.\n\n"
    "20. Penalty.—Contravention of section 12 is punishable with a fine.\n"
)


def _build_pipeline(
    vector_store: Any, versions: _FakeVersionRepository, documents: _FakeDocumentRepository | None = None
) -> IndexingPipeline:
    pipeline = IndexingPipeline(
        loader=DocumentLoader(),
        chunker=SectionAwareChunker(),
        embeddings=_FakeEmbeddings(),
        vector_store=vector_store,
    )
    pipeline.documents = documents or _FakeDocumentRepository()  # type: ignore[assignment]
    pipeline.versions = versions  # type: ignore[assignment]
    pipeline.quality_checker = DocumentQualityChecker()
    return pipeline


def _source(tmp_path: Path, name: str = "sample_act.txt") -> Path:
    path = tmp_path / name
    path.write_text(_ACT_TEXT, encoding="utf-8")
    return path


@pytest.mark.parametrize("scope", [{"owner_session_id": "new-chat"}, {"owner_user_id": "another-user"}])
def test_private_upload_ignores_hashes_from_other_documents(tmp_path, monkeypatch, scope):
    from unittest.mock import AsyncMock

    pipeline = _build_pipeline(_FailingVectorStore(), _FakeVersionRepository())
    source = _source(tmp_path)
    known = AsyncMock(return_value={pipeline.quality_checker.hash_file(source)})
    monkeypatch.setattr(pipeline, "_known_hashes", known)
    _, _, chunks = asyncio.run(pipeline.index_file(source, **scope))
    assert chunks
    known.assert_not_awaited()
    for key, value in scope.items():
        assert all(chunk.metadata[key] == value for chunk in chunks)


def test_shared_kb_still_rejects_duplicate_hash(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    pipeline = _build_pipeline(_FailingVectorStore(), _FakeVersionRepository())
    source = _source(tmp_path)
    monkeypatch.setattr(pipeline, "_known_hashes", AsyncMock(return_value={pipeline.quality_checker.hash_file(source)}))
    with pytest.raises(BadRequestError, match="quality checks"):
        asyncio.run(pipeline.index_file(source))


@pytest.mark.parametrize(
    "fail_on",
    ["upsert_chunks", "activate_version"],
)
def test_a_failure_during_staging_or_activation_leaves_no_orphaned_version_or_chunks(
    tmp_path: Path, fail_on: str
) -> None:
    vector_store = _FailingVectorStore(fail_on=fail_on)
    versions = _FakeVersionRepository(existing=None)
    pipeline = _build_pipeline(vector_store, versions)

    with pytest.raises(RuntimeError):
        asyncio.run(pipeline.index_file(_source(tmp_path)))

    # Cleanup ran: the staging version row and any chunks it wrote are gone.
    assert versions.docs == {}
    assert vector_store.upserted == []
    assert len(versions.deleted_ids) == 1


def test_a_reindex_failure_with_an_existing_previous_version_leaves_it_fully_intact(tmp_path: Path) -> None:
    """The headline P0-2 guarantee: a source being RE-indexed (not a brand
    new one) that fails must leave its previously-indexed content exactly as
    it was -- not partially deleted, not marked superseded, still the only
    thing active. This is the scenario the original bug broke: the old
    pipeline deleted the previous version's chunks BEFORE writing the new
    ones, so any failure after that point (an embedding-provider error, a
    partial `bulk_write`, anything) left the source with NOTHING searchable.
    """
    vector_store = _FailingVectorStore(fail_on="activate_version")
    old_chunks = [
        DocumentChunk(
            chunk_id="old-c1", document_id="doc-old", text="old content one",
            metadata={"document_status": "active", "source_document": "sample_act.txt"},
        ),
        DocumentChunk(
            chunk_id="old-c2", document_id="doc-old", text="old content two",
            metadata={"document_status": "active", "source_document": "sample_act.txt"},
        ),
    ]
    vector_store.upserted.extend(old_chunks)
    previous = {
        "_id": "v1", "document_id": "doc-old", "source_document": "sample_act.txt",
        "version_number": 1, "document_status": "active",
    }
    versions = _FakeVersionRepository(existing=previous)
    pipeline = _build_pipeline(vector_store, versions)

    with pytest.raises(RuntimeError):
        asyncio.run(pipeline.index_file(_source(tmp_path)))

    # The previous version's own chunks: untouched, still active -- and
    # nothing from the failed attempt survives alongside them.
    assert len(vector_store.upserted) == 2
    assert all(chunk.document_id == "doc-old" for chunk in vector_store.upserted)
    assert all(chunk.metadata["document_status"] == "active" for chunk in vector_store.upserted)
    # The previous version RECORD itself was never touched -- no supersede
    # was ever recorded against it.
    assert versions.updated == []


def test_a_partial_staged_write_is_caught_by_validation_and_cleaned_up(tmp_path: Path) -> None:
    vector_store = _FailingVectorStore()
    vector_store.miscount_next_validation = True
    versions = _FakeVersionRepository(existing=None)
    pipeline = _build_pipeline(vector_store, versions)

    with pytest.raises(BadRequestError, match="incomplete"):
        asyncio.run(pipeline.index_file(_source(tmp_path)))

    assert versions.docs == {}
    assert vector_store.upserted == []


def test_embedding_provider_failure_leaves_no_orphaned_version(tmp_path: Path) -> None:
    versions = _FakeVersionRepository(existing=None)
    pipeline = _build_pipeline(_FailingVectorStore(), versions)
    pipeline.embeddings = _FakeEmbeddings(fail=True)

    with pytest.raises(RuntimeError):
        asyncio.run(pipeline.index_file(_source(tmp_path)))

    assert versions.docs == {}


def test_a_concurrent_reindex_attempt_is_rejected_without_touching_the_previous_version(tmp_path: Path) -> None:
    vector_store = _FailingVectorStore()
    previous = {"_id": "v1", "document_id": "doc-old", "source_document": "sample_act.txt", "version_number": 1}
    versions = _FakeVersionRepository(existing=previous)
    versions.claim_should_fail = True  # simulates the unique index rejecting a concurrent claim
    pipeline = _build_pipeline(vector_store, versions)

    with pytest.raises(BadRequestError, match="already in progress"):
        asyncio.run(pipeline.index_file(_source(tmp_path)))

    # No chunking/embedding/write work happened at all -- the lock is
    # claimed BEFORE any of it, so a concurrent rejection is cheap.
    assert vector_store.upserted == []
    assert versions.updated == []


def test_a_successful_reindex_activates_new_before_superseding_old(tmp_path: Path) -> None:
    previous = {"_id": "v1", "document_id": "doc-old", "source_document": "sample_act.txt", "version_number": 1}
    vector_store = _FailingVectorStore()
    versions = _FakeVersionRepository(existing=previous)
    pipeline = _build_pipeline(vector_store, versions)

    document_id, _language, chunks = asyncio.run(pipeline.index_file(_source(tmp_path)))

    assert chunks
    assert vector_store.activate_calls == [(document_id, "doc-old")]
    # The version record chain: new version activated, old marked superseded.
    assert any(updates.get("document_status") == "active" for _vid, updates in versions.activated)
    assert any(updates.get("document_status") == "superseded" for _vid, updates in versions.updated)
    # Best-effort physical cleanup of the old version's chunks ran too.
    assert "doc-old" in vector_store.deleted_document_ids


def test_a_brand_new_source_has_no_previous_version_to_touch(tmp_path: Path) -> None:
    vector_store = _FailingVectorStore()
    versions = _FakeVersionRepository(existing=None)
    pipeline = _build_pipeline(vector_store, versions)

    document_id, _language, chunks = asyncio.run(pipeline.index_file(_source(tmp_path)))

    assert chunks
    assert vector_store.activate_calls == [(document_id, None)]
    assert vector_store.deleted_document_ids == []  # nothing old to clean up


@pytest.mark.parametrize("failure_point", ["partial_activation", "version_metadata", "document_metadata"])
def test_failure_after_switch_restores_old_searchable_content(tmp_path, monkeypatch, failure_point):
    """Failures AFTER old chunks become superseded must not leave both versions gone."""
    from unittest.mock import AsyncMock

    previous = {"_id": "v1", "document_id": "doc-old", "source_document": "sample_act.txt",
                "version_number": 1, "document_status": "active"}
    vectors = _FailingVectorStore()
    vectors.upserted.append(DocumentChunk(chunk_id="old-c1", document_id="doc-old", text="old evidence",
                                         metadata={"document_status": "active"}))
    versions = _FakeVersionRepository(existing=previous)
    pipeline = _build_pipeline(vectors, versions)
    if failure_point == "partial_activation":
        original = vectors.activate_version
        first = True

        async def activate(new_id, old_id):
            nonlocal first
            await original(new_id, old_id)
            if first:
                first = False
                raise RuntimeError("BM25 save failed after Mongo switch")
        monkeypatch.setattr(vectors, "activate_version", activate)
    elif failure_point == "version_metadata":
        monkeypatch.setattr(versions, "activate", AsyncMock(side_effect=RuntimeError("version write failed")))
    else:
        monkeypatch.setattr(pipeline.documents, "update_by_id", AsyncMock(side_effect=RuntimeError("document write failed")))

    with pytest.raises(RuntimeError):
        asyncio.run(pipeline.index_file(_source(tmp_path)))
    assert [chunk.chunk_id for chunk in vectors.upserted] == ["old-c1"]
    assert vectors.upserted[0].metadata["document_status"] == "active"
    assert "doc-old" not in vectors.deleted_document_ids


def test_identical_filenames_have_separate_owner_locks(monkeypatch):
    repository = _repo_with_fake_collection(monkeypatch)

    async def exercise():
        base = {"source_document": "contract.pdf", "document_status": "staging"}
        first = await repository.claim_staging({**base, "owner_user_id": "alice"})
        second = await repository.claim_staging({**base, "owner_user_id": "bob"})
        duplicate = await repository.claim_staging({**base, "owner_user_id": "alice"})
        assert first and second and first != second
        assert duplicate is None
    asyncio.run(exercise())


def test_latest_version_lookup_includes_owner_scope(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.database.mongodb import mongodb

    collection = SimpleNamespace(find_one=AsyncMock(return_value=None))
    monkeypatch.setattr(type(mongodb), "db", property(lambda _self: {"document_versions": collection}))
    repository = DocumentVersionRepository()
    asyncio.run(repository.latest_for_source("contract.pdf", owner_user_id="alice", owner_session_id="session"))
    query = collection.find_one.call_args.args[0]
    assert query["owner_user_id"] == "alice"
    assert "owner_session_id" not in query  # authenticated ownership survives session changes
    asyncio.run(repository.latest_for_source("contract.pdf"))
    query = collection.find_one.call_args.args[0]
    assert query["owner_user_id"] is None and query["owner_session_id"] is None


# ---------------------------------------------------------------------------
# Ownership isolation (Part 45/46): two different owners' documents,
# reindexed independently, must never share a lock or an activation scope.
# ---------------------------------------------------------------------------


def test_two_different_owners_reindexing_at_once_do_not_interfere(tmp_path: Path) -> None:
    vector_store_a = _FailingVectorStore()
    vector_store_b = _FailingVectorStore()
    versions_a = _FakeVersionRepository(existing=None)
    versions_b = _FakeVersionRepository(existing=None)
    pipeline_a = _build_pipeline(vector_store_a, versions_a)
    pipeline_b = _build_pipeline(vector_store_b, versions_b)

    # Real per-user uploads get a fresh uuid4-based filename each time (see
    # `DocumentService.upload_and_index`) -- source identity is therefore
    # already unique per upload; this proves the reindex machinery doesn't
    # ACCIDENTALLY reintroduce a cross-owner collision on top of that.
    source_a = _source(tmp_path, "doc-owner-a.txt")
    source_b = _source(tmp_path, "doc-owner-b.txt")

    doc_id_a, _lang_a, chunks_a = asyncio.run(
        pipeline_a.index_file(source_a, owner_user_id="user-A")
    )
    doc_id_b, _lang_b, chunks_b = asyncio.run(
        pipeline_b.index_file(source_b, owner_user_id="user-B")
    )

    assert doc_id_a != doc_id_b
    for chunk in chunks_a:
        assert chunk.metadata["owner_user_id"] == "user-A"
    for chunk in chunks_b:
        assert chunk.metadata["owner_user_id"] == "user-B"
    # Each pipeline's own fake vector store only ever saw its own owner's chunks.
    assert all(c.metadata["owner_user_id"] == "user-A" for c in vector_store_a.upserted)
    assert all(c.metadata["owner_user_id"] == "user-B" for c in vector_store_b.upserted)


# ---------------------------------------------------------------------------
# Layer 3: real retrieval visibility through a real `BM25Index`, driven the
# way `MongoVectorStore` drives it -- proves the answer chat actually gets,
# not just that the right internal methods were called.
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, document_id: str, text: str, status: str, **extra: Any) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id, document_id=document_id, text=text,
        metadata={"source_document": "bns.pdf", "document_status": status, **extra},
    )


# BM25's IDF term goes non-positive when a token appears in every document in
# the corpus, and `BM25Index.search` drops non-positive scores -- with only
# one or two near-identical "theft" documents, every query term would appear
# in all of them (mirrors `test_index_reconciliation.py`'s own `_FILLER`).
# These vocabulary-disjoint fillers keep the query genuinely discriminative.
_FILLER = [
    _chunk("filler-1", "doc-filler-1", "The Motor Vehicles Act governs driving licences and registration.", "active"),
    _chunk("filler-2", "doc-filler-2", "Consumer complaints about defective goods go to the consumer commission.", "active"),
    _chunk("filler-3", "doc-filler-3", "Income tax returns must be filed before the assessment deadline.", "active"),
    _chunk("filler-4", "doc-filler-4", "Registration of documents happens at the sub-registrar office.", "active"),
]


def _seeded(index: BM25Index, chunks: list[DocumentChunk]) -> None:
    """Seeds directly via `_set_corpus` -- the same seam
    `_rebuild_from_mongo`/`_load_from_disk` funnel through (see
    `test_bm25_index.py`'s own convention) -- rather than
    `add_or_update_chunks`, which touches real Mongo the moment nothing is
    loaded yet (production assumes Mongo is always reachable there; this
    suite has no live MongoDB)."""
    index._set_corpus(
        [chunk.chunk_id for chunk in chunks],
        [chunk.text for chunk in chunks],
        [chunk.metadata for chunk in chunks],
    )


def test_old_version_stays_searchable_while_a_new_one_is_only_staged(tmp_path: Path) -> None:
    index = BM25Index(cache_path=tmp_path / "bm25.pkl")
    old_chunk = _chunk("old-1", "doc-old", "Section 100 defines theft under the old text.", "active")
    _seeded(index, [old_chunk, *_FILLER])

    # A new reindex attempt writes its chunks as "staging" -- exactly what
    # `IndexingPipeline.index_file` does before activation.
    staging_chunk = _chunk("new-1", "doc-new", "Section 100 defines theft under the REVISED text.", "staging")
    asyncio.run(index.add_or_update_chunks([staging_chunk]))

    results = index.search("Section 100 defines theft", top_k=5)
    result_ids = {r.chunk_id for r in results}
    assert "old-1" in result_ids
    assert "new-1" not in result_ids  # staging is never visible


def test_activation_makes_new_visible_and_excludes_the_old_version(tmp_path: Path) -> None:
    index = BM25Index(cache_path=tmp_path / "bm25.pkl")
    old_chunk = _chunk("old-1", "doc-old", "Section 100 defines theft under the old text.", "active")
    new_chunk = _chunk("new-1", "doc-new", "Section 100 defines theft under the REVISED text.", "staging")
    _seeded(index, [old_chunk, new_chunk, *_FILLER])

    # Mirrors `MongoVectorStore.activate_version`: new -> active, old -> superseded.
    asyncio.run(index.mark_document_status({"new-1"}, "active"))
    asyncio.run(index.mark_document_status({"old-1"}, "superseded"))

    results = index.search("Section 100 defines theft", top_k=5)
    result_ids = {r.chunk_id for r in results}
    assert result_ids == {"new-1"}


def test_a_failed_activation_leaves_the_old_version_as_the_only_visible_one(tmp_path: Path) -> None:
    """Simulates `activate_version` raising after marking the new version
    active but before superseding the old one would be the WORST case (both
    briefly visible) -- this test instead confirms the actually-implemented,
    safer failure: if activation never runs at all (the more common failure
    mode, e.g. the staged write or its validation failed first), the old
    version is the ONLY thing visible throughout."""
    index = BM25Index(cache_path=tmp_path / "bm25.pkl")
    old_chunk = _chunk("old-1", "doc-old", "Section 100 defines theft under the old text.", "active")
    staging_chunk = _chunk("new-1", "doc-new", "Section 100 defines theft under the REVISED text.", "staging")
    _seeded(index, [old_chunk, staging_chunk, *_FILLER])
    # Activation never happens (the caller hit an exception first) -- the
    # staging chunk is cleaned up, mirroring `index_file`'s except block.
    asyncio.run(index.remove_chunk_ids({"new-1"}))

    results = index.search("Section 100 defines theft", top_k=5)
    assert {r.chunk_id for r in results} == {"old-1"}
