"""Benchmark (2026-09-24, QA pass "60,000-chunk local vector scan latency"):
does `LocalAnnIndex` (FAISS HNSW) actually beat `MongoVectorStore.
_local_cosine_leg_brute_force` on the REAL live corpus, in both latency and
result accuracy (recall against the exact brute-force ranking)?

Read-only against the real database -- builds the ANN index (an in-memory
structure, nothing written to Mongo) and runs `.search()`/the brute-force leg
directly; never inserts, updates, or deletes a single document.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.mongodb import mongodb
from app.rag.embeddings import EmbeddingProvider
from app.rag.local_ann_index import LocalAnnIndex
from app.rag.vector_store import MongoVectorStore

QUERIES = [
    "mera landlord mera security deposit wapas nahi de raha hai",
    "admissibility of electronic evidence Bharatiya Sakshya Adhiniyam",
    "cheque bounce hone par kya karna chahiye",
    "what is anticipatory bail",
    "consumer complaint deficiency in service",
]


async def main() -> None:
    await mongodb.connect()
    collection = mongodb.db["embeddings_metadata"]
    corpus_size = await collection.count_documents({})
    print(f"=== Live corpus size: {corpus_size} chunks ===\n")

    ann = LocalAnnIndex()
    print("Building FAISS HNSW index from the live corpus (id + embedding only)...")
    build_started = time.perf_counter()
    await ann.rebuild()
    build_ms = (time.perf_counter() - build_started) * 1000
    print(f"Build complete: {ann.chunk_count} vectors indexed in {build_ms:.0f}ms\n")

    store = MongoVectorStore()
    embeddings = EmbeddingProvider()
    filters = {"document_status": "active"}
    limit = 80  # matches candidate_k for top_k=10, candidate_multiplier=8

    total_ann_ms = 0.0
    total_brute_ms = 0.0
    for query in QUERIES:
        [vector] = await embeddings.embed_batch([query])

        ann_started = time.perf_counter()
        ann_results = await ann.search(vector, limit, filters)
        ann_ms = (time.perf_counter() - ann_started) * 1000
        total_ann_ms += ann_ms

        brute_started = time.perf_counter()
        brute_results = await store._local_cosine_leg_brute_force(vector, limit, filters)
        brute_ms = (time.perf_counter() - brute_started) * 1000
        total_brute_ms += brute_ms

        ann_ids = {c.chunk_id for c in (ann_results or [])[:10]}
        brute_ids = {c.chunk_id for c in brute_results[:10]}
        overlap = len(ann_ids & brute_ids)

        print(f"Query: {query!r}")
        print(f"  ANN:   {ann_ms:8.1f}ms  {len(ann_results or [])} results  top1={(ann_results or [{}])[0].metadata.get('source_document') if ann_results else None}")
        print(f"  Brute: {brute_ms:8.1f}ms  {len(brute_results)} results  top1={brute_results[0].metadata.get('source_document') if brute_results else None}")
        print(f"  Top-10 overlap: {overlap}/10\n")

    print("=== Summary ===")
    print(f"ANN total:   {total_ann_ms:.0f}ms  (avg {total_ann_ms/len(QUERIES):.1f}ms/query)")
    print(f"Brute total: {total_brute_ms:.0f}ms  (avg {total_brute_ms/len(QUERIES):.1f}ms/query)")
    print(f"Speedup: {total_brute_ms/total_ann_ms:.1f}x")

    await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
