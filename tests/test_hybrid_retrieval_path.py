"""The single hybrid retrieval path, end to end over its own components.

Phase 1 deleted `app/rag/hybrid_legal_retriever.py`, a second, never-imported
implementation of dense search + BM25 + reciprocal rank fusion + cross-encoder
reranking. These tests pin the surviving one -- the path chat actually runs --
so "there is only one hybrid retriever" stays true and is enforced rather than
asserted in a comment.

The four stages, and where each now lives:

    dense   `MongoVectorStore._atlas_vector_leg` / `._local_cosine_leg`
    sparse  `app.rag.bm25_index.bm25_index.search`
    fusion  `app.rag.fusion.reciprocal_rank_fusion`
    rerank  `app.rag.reranker.LegalReranker.rerank`

The removed module's cross-encoder also silently substituted fabricated scores
(`1.0 / (1 + i)` -- position, not relevance) whenever `sentence_transformers`
was missing. `test_reranker_scores_are_derived_from_content` covers the
surviving reranker's opposite behaviour: its score comes from the chunk's own
text, so an unrelated chunk cannot inherit a good score from its position.
"""

from typing import Any

import pytest

from app.rag.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from app.rag.reranker import LegalReranker
from app.rag.vector_store import MongoVectorStore
from app.schemas.common import RetrievedChunk


def _chunk(chunk_id: str, text: str, score: float, **metadata: Any) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, score=score, metadata=metadata)


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def test_rrf_ranks_a_chunk_found_by_both_legs_above_either_leg_alone() -> None:
    """The whole point of running two legs: agreement between them outranks a
    strong showing in one. `dense_only` is rank 1 on its own leg and still
    loses to `both`, which is rank 2 on both."""
    dense = [_chunk("dense_only", "vector leg favourite", 0.91), _chunk("both", "agreed", 0.88)]
    sparse = [_chunk("sparse_only", "lexical leg favourite", 7.4), _chunk("both", "agreed", 5.1)]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert fused[0].chunk_id == "both"
    assert {chunk.chunk_id for chunk in fused} == {"both", "dense_only", "sparse_only"}
    # Scored by RANK, not by the two legs' incomparable raw scales (cosine
    # similarity vs unbounded BM25).
    assert fused[0].score == pytest.approx(1 / (DEFAULT_RRF_K + 2) * 2)


def test_rrf_degrades_to_a_single_leg_when_the_other_returns_nothing() -> None:
    """`MongoVectorStore.search` catches a BM25 failure and passes `[]`; the
    query must still be answered from the vector leg alone."""
    dense = [_chunk("a", "first", 0.9), _chunk("b", "second", 0.7)]

    fused = reciprocal_rank_fusion([dense, []])

    assert [chunk.chunk_id for chunk in fused] == ["a", "b"]


def test_rrf_deduplicates_by_chunk_id_and_keeps_the_first_seen_content() -> None:
    dense = [_chunk("shared", "text from the vector leg", 0.9, source_document="bns.pdf")]
    sparse = [_chunk("shared", "text from the lexical leg", 6.2, source_document="bns.pdf")]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert len(fused) == 1
    assert fused[0].text == "text from the vector leg"
    assert fused[0].metadata["source_document"] == "bns.pdf"


# --------------------------------------------------------------------------
# Dense + sparse, merged by the production search method
# --------------------------------------------------------------------------


async def test_vector_store_search_merges_both_legs_and_reports_the_hybrid_route(monkeypatch, caplog) -> None:
    store = MongoVectorStore()
    dense = [_chunk("dense_only", "cognizable offence recorded by the officer in charge", 0.88),
             _chunk("both", "an FIR may be registered at any police station", 0.71)]
    sparse = [_chunk("both", "an FIR may be registered at any police station", 6.9),
              _chunk("sparse_only", "zero FIR and its transfer to the correct station", 4.2)]

    async def fake_dense(embedding: list[float], candidate_k: int, filters: dict[str, Any]) -> list[RetrievedChunk]:
        return dense

    monkeypatch.setattr(store, "_local_cosine_leg", fake_dense)
    monkeypatch.setattr("app.rag.vector_store.bm25_index.search", lambda *args, **kwargs: sparse)

    async def already_loaded() -> None:
        return None

    monkeypatch.setattr("app.rag.vector_store.bm25_index.ensure_current_generation", already_loaded)

    results = await store.search([0.1, 0.2, 0.3], "how do I register an FIR", top_k=3, filters={})

    assert [chunk.chunk_id for chunk in results] == ["both", "dense_only", "sparse_only"]


async def test_vector_store_search_survives_a_failing_sparse_leg(monkeypatch) -> None:
    """A BM25 fault degrades the query to the vector leg -- it must not raise,
    because the alternative is a chat request that returns nothing at all."""
    store = MongoVectorStore()
    dense = [_chunk("a", "vector leg still works", 0.9)]

    async def fake_dense(embedding: list[float], candidate_k: int, filters: dict[str, Any]) -> list[RetrievedChunk]:
        return dense

    def exploding_search(*args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        raise RuntimeError("bm25 index corrupt")

    async def already_loaded() -> None:
        return None

    monkeypatch.setattr(store, "_local_cosine_leg", fake_dense)
    monkeypatch.setattr("app.rag.vector_store.bm25_index.search", exploding_search)
    monkeypatch.setattr("app.rag.vector_store.bm25_index.ensure_current_generation", already_loaded)

    results = await store.search([0.1], "anything", top_k=3, filters={})

    assert [chunk.chunk_id for chunk in results] == ["a"]


# --------------------------------------------------------------------------
# Reranking
# --------------------------------------------------------------------------


async def test_reranker_scores_are_derived_from_content_not_candidate_position() -> None:
    """The deleted module's cross-encoder fallback returned `1.0 / (1 + i)` --
    a score derived purely from a chunk's position in the candidate list, which
    would have handed an unrelated chunk a top score for arriving first. The
    surviving reranker scores from the chunk's own text, so an off-topic chunk
    placed first stays below an on-topic one placed last."""
    reranker = LegalReranker()
    chunks = [
        _chunk("pizza", "A classic Margherita pizza uses tomato sauce and fresh mozzarella.", 0.5),
        _chunk("bail", "Anticipatory bail may be sought by a person apprehending arrest.", 0.5),
    ]

    ranked = await reranker.rerank("what is anticipatory bail", chunks, top_k=2)

    assert ranked[0].chunk_id == "bail"
    by_id = {chunk.chunk_id: chunk.score for chunk in ranked}
    assert by_id["bail"] > by_id["pizza"]


async def test_reranker_preserves_chunk_identity_and_text() -> None:
    """Reranking reorders and rescores; it must never rewrite what the LLM is
    then asked to ground its answer in."""
    reranker = LegalReranker()
    original = _chunk("s173", "173. Information in cognizable cases.", 0.6, source_document="bnss.pdf")

    ranked = await reranker.rerank("section 173 bnss", [original], top_k=1)

    assert len(ranked) == 1
    assert ranked[0].chunk_id == "s173"
    assert ranked[0].text == original.text
    assert ranked[0].metadata["source_document"] == "bnss.pdf"
