"""Phase 2 Milestone D: BM25/Mongo reconciliation and ownership safety.

MongoDB is the source of truth. The rules this module defends:

  * dry-run writes nothing, ever;
  * apply prunes ONLY BM25 entries whose chunk id is absent from Mongo;
  * a private upload's chunks never reach another user, and a stale index entry
    cannot resurrect one;
  * reconciliation is idempotent.

The BM25 index is driven for real (a genuine `BM25Index` over a temp cache
file), with MongoDB replaced by an in-memory double -- so the pruning logic and
the on-disk round-trip are exercised, not mocked away.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from app.rag.bm25_index import BM25Index
from app.rag.reconciliation import IndexReconciler, _ownership_problem


class _FakeCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __aiter__(self) -> Any:
        async def _gen() -> Any:
            for row in self._rows:
                yield row

        return _gen()


class _FakeCollection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def find(self, query: dict[str, Any], projection: dict[str, Any] | None = None) -> _FakeCursor:
        return _FakeCursor(self.rows)


class _FakeDb:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._collection = _FakeCollection(rows)

    def __getitem__(self, name: str) -> _FakeCollection:
        return self._collection


def _mongo(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> None:
    import app.database.mongodb as mongodb_module

    class _FakeMongo:
        db = _FakeDb(rows)

    monkeypatch.setattr(mongodb_module, "mongodb", _FakeMongo())


def _index(tmp_path: Path, entries: list[tuple[str, str, dict[str, Any]]]) -> BM25Index:
    index = BM25Index(cache_path=tmp_path / "bm25.pkl")
    index._set_corpus(
        [entry[0] for entry in entries],
        [entry[1] for entry in entries],
        # P0-2 "Failure-safe reindexing": `BM25Index.search` now requires
        # `document_status="active"` unconditionally (mirrors
        # `MongoVectorStore`'s own gate) -- every real chunk carries this
        # field, so every entry's metadata here does too, regardless of what
        # the individual test supplied.
        [{**entry[2], "document_status": "active"} for entry in entries],
    )
    return index


def _row(chunk_id: str, **metadata: Any) -> dict[str, Any]:
    return {"_id": chunk_id, "document_id": metadata.get("document_id", "doc-1"), "metadata": metadata}

# BM25's IDF term goes non-positive when a token appears in EVERY document,
# and `BM25Index.search` drops non-positive scores -- so a two-document corpus
# where both documents match the query scores nothing. That is an artefact of
# the fixture size, not of the ownership filter, so these unrelated documents
# are added to make the query terms genuinely discriminative.
_FILLER: list[tuple[str, str, dict[str, Any]]] = [
    ("filler-1", "The Motor Vehicles Act governs driving licences and vehicle registration.", {}),
    ("filler-2", "Consumer complaints about defective goods go to the consumer commission.", {}),
    ("filler-3", "Income tax returns must be filed before the assessment deadline.", {}),
    ("filler-4", "Registration of documents happens at the sub-registrar office.", {}),
]


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def test_dry_run_reports_stale_entries_without_changing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mongo(monkeypatch, [_row("live-1", source_document="bns.pdf")])
    index = _index(tmp_path, [
        ("live-1", "Section 318. Cheating.", {"source_document": "bns.pdf"}),
        ("stale-1", "Old generation of the same text.", {"source_document": "bns.pdf"}),
        ("stale-2", "Another orphan.", {"source_document": "gone.pdf"}),
    ])

    report = asyncio.run(IndexReconciler(index).analyze())

    assert report.mode == "dry-run"
    assert report.mongo_chunk_count == 1
    assert report.bm25_chunk_count == 3
    assert sorted(report.stale_bm25_chunk_ids) == ["stale-1", "stale-2"]
    assert report.drifted is True
    # Nothing written.
    assert index.chunk_count == 3
    assert not (tmp_path / "bm25.pkl").exists()


def test_dry_run_reports_chunks_missing_from_bm25(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mongo(monkeypatch, [_row("a"), _row("b"), _row("c")])
    index = _index(tmp_path, [("a", "text a", {})])

    report = asyncio.run(IndexReconciler(index).analyze())
    assert report.missing_from_bm25_count == 2


def test_apply_never_backfills_chunks_missing_from_bm25(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding them means tokenizing text the reconciler has not read. That is
    `BM25Index.rebuild()`'s job and an explicit operator decision."""
    _mongo(monkeypatch, [_row("a"), _row("b")])
    index = _index(tmp_path, [("a", "text a", {})])

    report = asyncio.run(IndexReconciler(index).apply())
    assert report.missing_from_bm25_count == 1
    assert index.chunk_count == 1


def test_duplicate_chunk_ids_inside_the_index_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mongo(monkeypatch, [_row("dup")])
    index = _index(tmp_path, [("dup", "one", {}), ("dup", "two", {})])
    report = asyncio.run(IndexReconciler(index).analyze())
    assert report.duplicate_bm25_chunk_ids == ["dup"]


def test_orphaned_source_documents_are_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mongo(monkeypatch, [_row("live", source_document="current.pdf")])
    index = _index(tmp_path, [
        ("live", "current text", {"source_document": "current.pdf"}),
        ("stale", "withdrawn text", {"source_document": "withdrawn.pdf"}),
    ])
    report = asyncio.run(IndexReconciler(index).analyze())
    assert report.orphaned_source_documents == ["withdrawn.pdf"]


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def test_apply_prunes_only_the_stale_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mongo(monkeypatch, [_row("live-1"), _row("live-2")])
    index = _index(tmp_path, [
        ("live-1", "kept one", {}),
        ("stale-1", "dropped", {}),
        ("live-2", "kept two", {}),
    ])

    report = asyncio.run(IndexReconciler(index).apply())

    assert report.mode == "apply"
    assert report.pruned_count == 1
    assert report.bm25_chunk_count == 3
    assert report.bm25_chunk_count_after == 2
    assert set(index._chunk_ids) == {"live-1", "live-2"}
    # The surviving entries keep their own text; pruning is not a rebuild.
    assert index._texts == ["kept one", "kept two"]


def test_reconciliation_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mongo(monkeypatch, [_row("live")])
    index = _index(tmp_path, [("live", "kept", {}), ("stale", "dropped", {})])

    first = asyncio.run(IndexReconciler(index).apply())
    second = asyncio.run(IndexReconciler(index).apply())

    assert first.pruned_count == 1
    assert second.pruned_count == 0
    assert second.drifted is False
    assert index.chunk_count == 1


def test_apply_on_a_consistent_index_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mongo(monkeypatch, [_row("a"), _row("b")])
    index = _index(tmp_path, [("a", "text a", {}), ("b", "text b", {})])

    report = asyncio.run(IndexReconciler(index).apply())
    assert report.pruned_count == 0
    assert report.drifted is False
    assert index.chunk_count == 2


def test_apply_persists_the_pruned_corpus_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the next process start reloads the stale entries from the pickle."""
    _mongo(monkeypatch, [_row("live")])
    index = _index(tmp_path, [("live", "kept", {}), ("stale", "dropped", {})])
    asyncio.run(IndexReconciler(index).apply())

    reloaded = BM25Index(cache_path=tmp_path / "bm25.pkl")
    assert reloaded._load_from_disk() is True
    assert reloaded._chunk_ids == ["live"]


# ---------------------------------------------------------------------------
# Ownership safety
# ---------------------------------------------------------------------------


def test_a_stale_private_chunk_is_counted_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale entry carrying owner metadata is a retrievable copy of a private
    upload MongoDB no longer has -- a privacy consequence, not a ranking one."""
    _mongo(monkeypatch, [])
    index = _index(tmp_path, [
        ("stale-private", "my lease terms", {"owner_user_id": "user-1", "visibility": "private"}),
        ("stale-public", "a public act", {"source_document": "bns.pdf"}),
    ])
    report = asyncio.run(IndexReconciler(index).analyze())
    assert report.stale_private_chunk_count == 1


def test_pruning_preserves_owner_metadata_on_surviving_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mongo(monkeypatch, [_row("mine", owner_user_id="user-1", visibility="private")])
    index = _index(tmp_path, [
        ("mine", "my private lease", {"owner_user_id": "user-1", "visibility": "private"}),
        ("stale", "orphan", {}),
    ])
    asyncio.run(IndexReconciler(index).apply())
    assert index._metadatas == [{"owner_user_id": "user-1", "visibility": "private", "document_status": "active"}]


def test_one_users_private_chunk_is_not_retrievable_by_another(tmp_path: Path) -> None:
    """The isolation that reconciliation must not disturb, checked through the
    index's own filter path rather than by inspecting metadata."""
    index = _index(tmp_path, [
        ("u1", "confidential lease agreement for tenant Rahul", {"owner_user_id": "user-1", "visibility": "private"}),
        ("u2", "confidential lease agreement for tenant Priya", {"owner_user_id": "user-2", "visibility": "private"}),
        *_FILLER,
    ])
    results = index.search("confidential lease agreement", top_k=5, filters={"owner_user_id": "user-1"})
    assert results, "expected the owner's own chunk to be retrievable"
    owners = {chunk.metadata.get("owner_user_id") for chunk in results}
    assert owners == {"user-1"}
    assert all("Priya" not in chunk.text for chunk in results)


def test_isolation_still_holds_after_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mongo(monkeypatch, [
        _row("u1", owner_user_id="user-1", visibility="private"),
        _row("u2", owner_user_id="user-2", visibility="private"),
        # The filler is live in Mongo too, otherwise reconciliation correctly
        # prunes it and the corpus shrinks back to the degenerate two-document
        # case where BM25's IDF cannot discriminate.
        *(_row(chunk_id) for chunk_id, _, _ in _FILLER),
    ])
    index = _index(tmp_path, [
        ("u1", "confidential lease agreement for tenant Rahul", {"owner_user_id": "user-1", "visibility": "private"}),
        ("u2", "confidential lease agreement for tenant Priya", {"owner_user_id": "user-2", "visibility": "private"}),
        ("stale", "confidential lease agreement for a deleted tenant", {"owner_user_id": "user-3", "visibility": "private"}),
        *_FILLER,
    ])
    asyncio.run(IndexReconciler(index).apply())

    for owner in ("user-1", "user-2"):
        results = index.search("confidential lease agreement", top_k=5, filters={"owner_user_id": owner})
        assert results, f"expected {owner} to still retrieve their own chunk"
        assert {chunk.metadata.get("owner_user_id") for chunk in results} == {owner}
    # The deleted third user's text is gone from the index entirely.
    assert "user-3" not in {meta.get("owner_user_id") for meta in index._metadatas}


def test_shared_knowledge_base_chunks_do_not_leak_into_a_private_filter(tmp_path: Path) -> None:
    index = _index(tmp_path, [
        ("kb", "Section 318 of the Bharatiya Nyaya Sanhita defines cheating", {"source_document": "bns.pdf"}),
        ("mine", "Section 318 cheating appears in my uploaded case notes", {"owner_user_id": "user-1", "visibility": "private"}),
        *_FILLER,
    ])
    results = index.search("Section 318 cheating", top_k=5, filters={"owner_user_id": "user-1"})
    assert results, "expected the owner's own chunk to be retrievable"
    assert {chunk.chunk_id for chunk in results} == {"mine"}


# ---------------------------------------------------------------------------
# Ownership metadata problems are reported, never repaired
# ---------------------------------------------------------------------------


def test_private_chunk_without_an_owner_is_flagged() -> None:
    problem = _ownership_problem("c1", {"visibility": "private"})
    assert problem == {"chunk_id": "c1", "problem": "private_without_owner"}


def test_owned_chunk_marked_shared_is_flagged() -> None:
    problem = _ownership_problem("c2", {"owner_user_id": "user-1", "visibility": "public"})
    assert problem == {"chunk_id": "c2", "problem": "owned_chunk_marked_shared"}


def test_a_well_formed_chunk_is_not_flagged() -> None:
    assert _ownership_problem("c3", {"owner_user_id": "user-1", "visibility": "private"}) is None
    assert _ownership_problem("c4", {"source_document": "bns.pdf"}) is None


def test_ownership_problems_are_reported_but_not_corrected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deciding an ambiguous chunk is public, or assigning it an owner, is a
    visibility decision no automated pass should make."""
    _mongo(monkeypatch, [_row("bad", visibility="private")])
    index = _index(tmp_path, [("bad", "ambiguous", {"visibility": "private"})])

    report = asyncio.run(IndexReconciler(index).apply())

    assert report.ownership_metadata_problems == [{"chunk_id": "bad", "problem": "private_without_owner"}]
    assert index._metadatas == [{"visibility": "private", "document_status": "active"}]


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


def test_the_report_carries_no_document_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconciliation output is logged and surfaced on /health/index-drift."""
    _mongo(monkeypatch, [])
    index = _index(tmp_path, [("stale", "confidential lease terms for tenant Rahul", {})])
    payload = str(asyncio.run(IndexReconciler(index).analyze()).as_dict())
    assert "Rahul" not in payload
    assert "lease terms" not in payload


def test_remove_chunk_ids_with_an_empty_set_is_a_no_op(tmp_path: Path) -> None:
    index = _index(tmp_path, [("a", "text", {})])
    assert asyncio.run(index.remove_chunk_ids(set())) == 0
    assert index.chunk_count == 1
