"""Regression tests for `search mode=semantic` (QA session 2026-09-24, from
`QA_REPORT_100Q_RETEST_20260924.md` section 7 item 5): `SearchService.search`
read `request.mode` for `"section"`/`"act"` (BUG-109/BUG-111, already fixed
and covered by `test_search_mode_section_and_act.py`), but every other mode
value -- including `"semantic"`, the one the QA retest explicitly flagged as
still byte-for-byte identical to `"hybrid"` -- fell straight through to
`LegalRetriever.retrieve()`, which always ran the full two-leg (BM25 +
embedding) hybrid search regardless of what the caller asked for.

`MongoVectorStore.search()` now takes a `mode` parameter: `"semantic"` skips
the BM25 leg entirely (pure vector similarity, no RRF blending with lexical
scores), `"keyword"` skips the embedding leg (pure BM25), and everything
else (including the default `"hybrid"` and the still-unhandled
`"metadata"`) keeps running the original two-leg search unchanged.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from app.rag.vector_store import MongoVectorStore
from app.schemas.common import RetrievedChunk
from app.schemas.search import SearchRequest
from app.services.search_service import SearchService


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=f"text {chunk_id}", score=score, metadata={})


def _store_with_mocked_legs(vector_results: list[RetrievedChunk], lexical_results: list[RetrievedChunk]) -> MongoVectorStore:
    store = MongoVectorStore()
    store._local_cosine_leg = AsyncMock(return_value=vector_results)  # type: ignore[method-assign]
    store._atlas_vector_leg = AsyncMock(return_value=vector_results)  # type: ignore[method-assign]
    return store


def test_mode_semantic_skips_the_bm25_leg_entirely() -> None:
    store = _store_with_mocked_legs([_chunk("v1", 0.9), _chunk("v2", 0.5)], [])
    with patch("app.rag.vector_store.bm25_index") as mock_bm25:
        mock_bm25.ensure_current_generation = AsyncMock()
        mock_bm25.search = lambda *a, **k: (_ for _ in ()).throw(AssertionError("BM25 must not run in semantic mode"))
        results = asyncio.run(store.search([0.1, 0.2], "query", top_k=5, filters={}, mode="semantic"))
    store._local_cosine_leg.assert_awaited_once()
    mock_bm25.ensure_current_generation.assert_not_awaited()
    assert [chunk.chunk_id for chunk in results] == ["v1", "v2"]


def test_mode_keyword_skips_the_embedding_leg_entirely() -> None:
    store = _store_with_mocked_legs([_chunk("v1", 0.9)], [])
    with patch("app.rag.vector_store.bm25_index") as mock_bm25:
        mock_bm25.ensure_current_generation = AsyncMock()
        mock_bm25.search.return_value = [_chunk("l1", 0.8), _chunk("l2", 0.4)]
        results = asyncio.run(store.search([0.1, 0.2], "query", top_k=5, filters={}, mode="keyword"))
    store._local_cosine_leg.assert_not_awaited()
    store._atlas_vector_leg.assert_not_awaited()
    assert [chunk.chunk_id for chunk in results] == ["l1", "l2"]


def test_mode_hybrid_default_runs_both_legs_unchanged() -> None:
    store = _store_with_mocked_legs([_chunk("v1", 0.9)], [])
    with patch("app.rag.vector_store.bm25_index") as mock_bm25:
        mock_bm25.ensure_current_generation = AsyncMock()
        mock_bm25.search.return_value = [_chunk("l1", 0.8)]
        results = asyncio.run(store.search([0.1, 0.2], "query", top_k=5, filters={}))
    store._local_cosine_leg.assert_awaited_once()
    mock_bm25.ensure_current_generation.assert_awaited_once()
    assert {chunk.chunk_id for chunk in results} == {"v1", "l1"}


def test_search_service_threads_semantic_mode_through_to_the_retriever() -> None:
    service = SearchService()
    service.retriever.retrieve = AsyncMock(return_value=("q", []))
    asyncio.run(service.search(SearchRequest(query="cheque bounce", mode="semantic", top_k=5)))
    service.retriever.retrieve.assert_awaited_once()
    assert service.retriever.retrieve.call_args.kwargs["mode"] == "semantic"


def test_search_service_defaults_unhandled_modes_to_hybrid() -> None:
    # "metadata" has no dedicated leg-skipping behavior -- must not be
    # passed through as a literal mode the vector store doesn't understand.
    service = SearchService()
    service.retriever.retrieve = AsyncMock(return_value=("q", []))
    asyncio.run(service.search(SearchRequest(query="cheque bounce", mode="metadata", top_k=5)))
    assert service.retriever.retrieve.call_args.kwargs["mode"] == "hybrid"
