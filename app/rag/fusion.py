from app.schemas.common import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(result_lists: list[list[RetrievedChunk]], k: int = DEFAULT_RRF_K) -> list[RetrievedChunk]:
    """Part 40 section 7: merges independently-ranked retrieval legs (embedding
    similarity, BM25) by RANK rather than raw score.

    Embedding cosine similarity and BM25's unbounded lexical score live on
    completely different scales, so adding them directly (the previous
    `VECTOR_WEIGHT`/`LEXICAL_WEIGHT` linear fusion in `vector_store.py`) lets
    whichever leg happens to produce bigger numbers dominate regardless of
    actual relevance. RRF sidesteps that: only a result's POSITION within its
    own list matters, contributing `1/(k+rank)` (1-indexed) per list it
    appears in. A chunk found by only one leg still contributes that leg's
    share -- it isn't zeroed out just because another leg missed it, which is
    also what makes this degrade gracefully into a single-leg ranking when
    one retrieval system is empty/unavailable (Part 40 section 17's fallback
    falls out of this for free, no special-case code needed).

    Generic over any number of ranked lists (not hardcoded to exactly two),
    and deduplicates by `chunk_id` -- a chunk present in more than one list
    keeps the metadata/text from whichever list it was first seen in, only
    its `score` is replaced with the fused RRF score.
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks_by_id.setdefault(chunk.chunk_id, chunk)
    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [chunks_by_id[chunk_id].model_copy(update={"score": scores[chunk_id]}) for chunk_id in ranked_ids]
