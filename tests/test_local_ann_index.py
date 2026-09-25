"""QA pass 2026-09-24 (60,000-chunk local vector scan latency/timeout):
`LocalAnnIndex`'s FAISS HNSW graph, tested in isolation from MongoDB --
`_build_index_sync`/`_add_vectors` (the same seam `_rebuild_from_mongo_locked`
funnels through) are used directly to seed a small synthetic corpus, mirroring
`test_bm25_index.py`'s own `_set_corpus` pattern. Text hydration (`_hydrate_
text`, a real Mongo `_id`-`$in` lookup) is exercised by the live pipeline
verification instead, not here.
"""

import asyncio

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from app.rag.local_ann_index import LocalAnnIndex  # noqa: E402


def _unit(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    return (array / np.linalg.norm(array)).tolist()


_DIM = 8
# Three well-separated synthetic "topics" so approximate search has an
# unambiguous right answer regardless of HNSW's approximation error.
_TENANCY = _unit([1, 0, 0, 0, 0, 0, 0, 0])
_CHEQUE = _unit([0, 1, 0, 0, 0, 0, 0, 0])
_EVIDENCE = _unit([0, 0, 1, 0, 0, 0, 0, 0])


def _seeded_index() -> LocalAnnIndex:
    index = LocalAnnIndex()
    vectors = [_TENANCY, _CHEQUE, _EVIDENCE]
    ids = ["chunk-tenancy", "chunk-cheque", "chunk-evidence"]
    metadatas = [
        {"act_name": "Rent Control Act", "document_status": "active", "owner_session_id": None},
        {"act_name": "Negotiable Instruments Act", "document_status": "active", "owner_session_id": None},
        {"act_name": "Bharatiya Sakshya Adhiniyam", "document_status": "active", "owner_session_id": "session-X"},
    ]
    index._index = faiss.IndexHNSWFlat(_DIM, 16, faiss.METRIC_INNER_PRODUCT)
    index._index.hnsw.efConstruction = 40
    index._dim = _DIM
    index._add_vectors(vectors, ids, metadatas)
    return index


def test_search_returns_none_when_never_built() -> None:
    index = LocalAnnIndex()
    result = asyncio.run(index.search(_TENANCY, limit=5, filters={}))
    assert result is None


def test_search_returns_none_on_dimension_mismatch() -> None:
    index = _seeded_index()
    result = asyncio.run(index.search([0.0] * (_DIM + 1), limit=5, filters={}))
    assert result is None


def test_search_ranks_the_closest_topic_first() -> None:
    index = _seeded_index()
    results = asyncio.run(index.search(_TENANCY, limit=3, filters={}))
    assert results is not None
    assert results[0].chunk_id == "chunk-tenancy"
    assert results[0].score > results[1].score


def test_search_applies_ownership_filter_like_bm25_and_mongo() -> None:
    index = _seeded_index()
    # chunk-evidence is owned by session-X; a caller scoped to "no owner"
    # must never see it, even though nothing else in the tiny corpus is
    # closer to the query vector.
    results = asyncio.run(
        index.search(_EVIDENCE, limit=3, filters={"owner_session_id": [None]})
    )
    assert results is not None
    assert all(chunk.chunk_id != "chunk-evidence" for chunk in results)


def test_search_score_is_never_negative_pydantic_ge_zero() -> None:
    index = _seeded_index()
    # A query orthogonal/opposite to every indexed vector could in principle
    # score a negative cosine -- RetrievedChunk.score requires >= 0.0
    # (Field(ge=0.0)), so this must not raise a validation error.
    opposite = _unit([-1, -1, -1, 0, 0, 0, 0, 0])
    results = asyncio.run(index.search(opposite, limit=3, filters={}))
    assert results is not None
    assert all(chunk.score >= 0.0 for chunk in results)


def test_add_or_update_chunks_makes_a_new_vector_searchable() -> None:
    from app.rag.types import DocumentChunk

    index = _seeded_index()
    new_topic = _unit([0, 0, 0, 1, 0, 0, 0, 0])
    chunk = DocumentChunk(
        chunk_id="chunk-contract", document_id="doc-contract", text="",
        metadata={"act_name": "Indian Contract Act", "document_status": "active", "owner_session_id": None},
        embedding=new_topic,
    )
    asyncio.run(index.add_or_update_chunks([chunk]))
    results = asyncio.run(index.search(new_topic, limit=4, filters={}))
    assert results is not None
    assert results[0].chunk_id == "chunk-contract"


def test_add_or_update_chunks_updates_metadata_in_place_for_existing_id() -> None:
    from app.rag.types import DocumentChunk

    index = _seeded_index()
    chunk = DocumentChunk(
        chunk_id="chunk-tenancy", document_id="doc-tenancy", text="",
        metadata={"act_name": "Rent Control Act", "document_status": "superseded", "owner_session_id": None},
        embedding=_TENANCY,
    )
    asyncio.run(index.add_or_update_chunks([chunk]))
    # Superseded now excluded by a document_status="active" filter, even
    # though it remains the single closest vector in the graph.
    results = asyncio.run(index.search(_TENANCY, limit=3, filters={"document_status": "active"}))
    assert results is not None
    assert all(chunk.chunk_id != "chunk-tenancy" for chunk in results)


def test_remove_chunk_ids_hides_without_deleting_the_vector() -> None:
    index = _seeded_index()
    removed = asyncio.run(index.remove_chunk_ids({"chunk-cheque"}))
    assert removed == 1
    results = asyncio.run(index.search(_CHEQUE, limit=3, filters={}))
    assert results is not None
    assert all(chunk.chunk_id != "chunk-cheque" for chunk in results)
    # The underlying HNSW node count is untouched (see module docstring --
    # true removal happens only at the next full rebuild).
    assert index._index.ntotal == 3


def test_mark_document_status_updates_cache_without_touching_faiss() -> None:
    index = _seeded_index()
    updated = asyncio.run(index.mark_document_status({"chunk-cheque"}, "superseded"))
    assert updated == 1
    assert index._metadatas[1]["document_status"] == "superseded"
    assert index._index.ntotal == 3


def test_ensure_current_generation_does_not_block_on_a_stale_already_loaded_index() -> None:
    # QA pass 2026-09-24: confirmed live -- a blocking rebuild here stalled
    # a real `/chat` request's retrieval leg for ~90s when the KB-automation
    # background job bumped `kb-generation` mid-request. An already-loaded
    # index must return immediately (serving the stale-but-correct data)
    # and rebuild in the background instead.
    index = _seeded_index()
    index._kb_generation = b"1"
    rebuild_started = asyncio.Event()
    rebuild_may_finish = asyncio.Event()

    async def _slow_fetch_and_build():
        rebuild_started.set()
        await rebuild_may_finish.wait()
        return None

    index._fetch_and_build = _slow_fetch_and_build  # type: ignore[method-assign]

    class _FakeRedisClient:
        async def get(self, _key: str) -> bytes:
            return b"2"

    class _FakeRedis:
        client = _FakeRedisClient()

    import app.rag.local_ann_index as local_ann_index_module

    async def run() -> None:
        import app.cache.redis_client as redis_client_module
        import app.cache.response_cache as response_cache_module

        original_client = redis_client_module.redis_client
        redis_client_module.redis_client = _FakeRedis()
        try:
            result = await asyncio.wait_for(index.ensure_current_generation(), timeout=2.0)
        finally:
            redis_client_module.redis_client = original_client
        assert result is True
        # The call above must have returned WITHOUT waiting for the slow
        # rebuild -- the background task should have just started.
        await asyncio.wait_for(rebuild_started.wait(), timeout=2.0)
        assert not rebuild_may_finish.is_set()
        rebuild_may_finish.set()
        await index._background_rebuild_task

    asyncio.run(run())
    assert index._kb_generation == b"2"


def test_kick_off_background_rebuild_does_not_start_a_second_one_while_running() -> None:
    index = _seeded_index()
    call_count = 0
    release = asyncio.Event()

    async def _slow_fetch_and_build():
        nonlocal call_count
        call_count += 1
        await release.wait()
        return None

    index._fetch_and_build = _slow_fetch_and_build  # type: ignore[method-assign]

    async def run() -> None:
        index._kick_off_background_rebuild(b"gen-1")
        index._kick_off_background_rebuild(b"gen-1")  # should be a no-op -- one already in flight
        await asyncio.sleep(0)
        release.set()
        await index._background_rebuild_task

    asyncio.run(run())
    assert call_count == 1
