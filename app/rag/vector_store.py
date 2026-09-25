import asyncio
import heapq
import re
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar
from uuid import uuid4

import numpy as np
import structlog
from pymongo import UpdateOne
from pymongo.errors import PyMongoError

from app.core.config import settings
from app.database.mongodb import mongodb
from app.models.collections import EMBEDDINGS_METADATA
from app.observability.metrics import metrics
from app.rag.bm25_index import bm25_index
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.kb_jurisdiction import TEMPORAL_FILTER_KEY, mongo_temporal_and_clauses
from app.rag.local_ann_index import local_ann_index
from app.rag.types import DocumentChunk
from app.schemas.common import RetrievedChunk

log = structlog.get_logger(__name__)


class VectorStore(ABC):
    @abstractmethod
    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self, query_embedding: list[float], query: str, top_k: int, filters: dict[str, Any],
        mode: str = "hybrid",
    ) -> list[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    async def delete_by_source(self, source_document: str) -> int:
        raise NotImplementedError

    @abstractmethod
    async def find_by_section_number(self, section_number: str, filters: dict[str, Any]) -> list[RetrievedChunk]:
        raise NotImplementedError

    async def find_constitution_article(self, article_number: str, filters: dict[str, Any]) -> list[RetrievedChunk]:
        """Compatibility hook for indexes created before article metadata."""
        return []

    async def find_named_section(
        self, section_number: str, act_name: str, filters: dict[str, Any]
    ) -> list[RetrievedChunk]:
        """Compatibility hook for chunks whose overlap metadata is stale."""
        return []

    # ------------------------------------------------------------------
    # P0-2 failure-safe reindexing. Not abstract: a test double that never
    # exercises `IndexingPipeline.index_file` (most of the existing suite)
    # need not implement these. `MongoVectorStore` overrides all three; a
    # store that DOES back a real reindex path and leaves these unimplemented
    # will fail loudly (`NotImplementedError`), never silently skip the
    # staging/activation safety this exists for.
    # ------------------------------------------------------------------

    async def count_by_document_id(self, document_id: str) -> int:
        """How many chunks are currently written under this document id --
        used to validate a staged write completed before activating it."""
        raise NotImplementedError

    async def activate_version(self, new_document_id: str, old_document_id: str | None) -> None:
        """Flips the new version's chunks to `document_status="active"`
        (retrievable), then the old version's chunks (if any) to
        `"superseded"` (excluded from every retrieval path) -- in that order,
        so a crash between the two steps leaves both briefly visible rather
        than neither."""
        raise NotImplementedError

    async def delete_version_chunks(self, document_id: str) -> int:
        """Physically removes every chunk written under this document id
        (Mongo and the BM25 mirror). Best-effort cleanup, not a correctness
        requirement: a `staging` or `superseded` chunk is already excluded
        from retrieval by `document_status` alone."""
        raise NotImplementedError


class MongoVectorStore(VectorStore):
    """Hybrid (vector + BM25) retrieval over MongoDB, merged by Reciprocal Rank
    Fusion (Part 40 "Hybrid Legal Retrieval").

    The vector leg uses a native ``$vectorSearch`` aggregation against the configured
    Atlas Search index in production (``vector_search_backend=atlas``), degrading to an
    in-process cosine scan when Atlas Search isn't available (e.g. the community MongoDB
    used in local docker-compose) — so retrieval keeps working, just without the
    index-backed speed and scale. The lexical leg always queries the persisted,
    in-memory `app.rag.bm25_index.bm25_index` singleton (see that module for why it has
    to live outside this per-request object graph), not MongoDB's ``$text`` index.
    """

    # QA session 2026-09-24 ("BUG-101"): process-wide, not per-instance --
    # `MongoVectorStore()` is constructed fresh per request (see
    # `LegalRetriever`), so a per-instance semaphore would never actually
    # limit anything. Bounds how many concurrent `_local_cosine_leg` scans
    # (each a `local_vector_scan_limit`-document fetch + BLAS score) run at
    # once across the whole process; see `settings.local_vector_scan_max_concurrency`.
    _local_scan_semaphore: ClassVar[asyncio.Semaphore] = asyncio.Semaphore(
        settings.local_vector_scan_max_concurrency
    )

    def __init__(self) -> None:
        # A large, topically-dense document (e.g. a 75-chunk judgment that's
        # entirely about FIR) can crowd a narrower-but-more-precise chunk
        # from a different, smaller document out of the candidate pool
        # entirely -- confirmed directly: "What is Zero FIR?" only surfaced
        # the one chunk that actually discusses Zero FIR (in a police-FAQ
        # document) once the candidate pool was widened enough for both the
        # vector and lexical legs to see past the judgment's sheer volume of
        # generically-similar FIR chunks. 3x was too narrow for a corpus this
        # topically concentrated; 8x fixed it without a meaningful latency
        # cost since both legs already do a full local scan either way.
        self.candidate_multiplier = 8
        # Local (non-Atlas) fallback scan cap -- see `settings.local_vector_scan_limit`
        # for why this is a cap at all and why it's configurable rather than hardcoded.
        self.local_scan_limit = settings.local_vector_scan_limit

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        collection = mongodb.db[EMBEDDINGS_METADATA]
        operations = [
            UpdateOne(
                {"_id": chunk.chunk_id},
                {
                    "$set": {
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "embedding": chunk.embedding,
                        "namespace": chunk.metadata.get("namespace", "default"),
                    }
                },
                upsert=True,
            )
            for chunk in chunks
        ]
        await collection.bulk_write(operations, ordered=False)
        # Part 40 section 14: keeps the BM25 lexical index synchronized with
        # the same chunk IDs the vector index just received -- through this
        # one shared write path (called by both the full `IndexingPipeline`
        # and `IncrementalReindexRunner`), never a separate ingestion route.
        await bm25_index.add_or_update_chunks(chunks)
        # QA pass 2026-09-24 (60k-chunk local-scan latency): same reasoning,
        # for the local ANN fallback -- see `local_ann_index`'s module
        # docstring for why this exists alongside `_local_cosine_leg`.
        await local_ann_index.add_or_update_chunks(chunks)

    async def search(
        self, query_embedding: list[float], query: str, top_k: int, filters: dict[str, Any],
        mode: str = "hybrid",
    ) -> list[RetrievedChunk]:
        """Part 40 "Hybrid Legal Retrieval": embedding (vector) and BM25
        (lexical) retrieval run as two independent legs over the same
        candidate pool size, merged by Reciprocal Rank Fusion rather than a
        raw weighted score sum (see `reciprocal_rank_fusion`'s docstring for
        why) -- then, unchanged, handed to `LegalReranker` by the caller
        (`LegalRetriever.retrieve`) as the final relevance filter before the
        LLM. Either leg failing/returning nothing degrades gracefully to a
        single-leg ranking (section 17's fallback), not an error.

        QA retest 2026-09-24 (`search mode=semantic`, BUG from
        `QA_REPORT_100Q_RETEST_20260924.md` section 7 item 5): `mode` used to
        be accepted by `SearchService`/`SearchRequest` but never reached this
        method at all -- every mode ran byte-for-byte the same hybrid search.
        `mode="semantic"` now genuinely skips the BM25 leg (pure vector
        similarity, no lexical scoring/RRF blending at all) and
        `mode="keyword"` genuinely skips the embedding leg (pure BM25, no
        vector scoring) -- both real, distinguishable retrieval strategies an
        API caller can now actually choose between. Any other value
        (including the existing default `"hybrid"`, and `"metadata"`, which
        has no dedicated leg-skipping behavior of its own) runs exactly the
        unchanged two-leg RRF-fused search every mode has always run.
        """
        query_id = str(uuid4())
        # P0-2: every retrieval path requires `document_status="active"`,
        # unconditionally -- never caller-overridable, unlike `review_status`.
        # A `staging` chunk is an in-flight, unvalidated reindex attempt and a
        # `superseded` chunk is content a later version has already replaced;
        # neither has a legitimate reason to answer a query through ANY
        # caller, admin tooling included. See `_document_status_filter`.
        filters = self._document_status_filter(filters or {})
        candidate_k = max(top_k * self.candidate_multiplier, top_k)

        embedding_start = time.perf_counter()
        vector_results: list[RetrievedChunk] = []
        if mode != "keyword":
            if settings.vector_search_backend == "atlas":
                try:
                    vector_results = await self._atlas_vector_leg(query_embedding, candidate_k, filters)
                    metrics.increment("atlas_vector_search_ok")
                except PyMongoError as exc:
                    # This dev environment has no Atlas backend to exercise this
                    # branch against live traffic (see `tests/test_atlas_vector_
                    # leg.py` for the structural pipeline-shape coverage that IS
                    # possible without one) -- this counter is the intended way
                    # to confirm, once actually deployed against Atlas, whether
                    # `$vectorSearch` is succeeding or silently falling back on
                    # every call.
                    metrics.increment("atlas_vector_search_fallback")
                    log.warning("atlas_vector_search_unavailable", error=str(exc))
            if not vector_results:
                vector_results = await self._local_cosine_leg(query_embedding, candidate_k, filters)
        embedding_ms = (time.perf_counter() - embedding_start) * 1000

        bm25_start = time.perf_counter()
        lexical_results: list[RetrievedChunk] = []
        if mode != "semantic":
            try:
                await bm25_index.ensure_current_generation()
                # `BM25Index.search` is CPU-bound, pure-Python (a per-candidate
                # filter loop over the WHOLE corpus, same shape as the fix just
                # applied to `_local_cosine_leg` above) -- run off the event
                # loop for the same reason: left inline, it can stall every
                # other in-flight coroutine on this process, including an
                # unrelated request's own `/health` Mongo/Redis ping.
                lexical_results = await asyncio.to_thread(bm25_index.search, query, candidate_k, filters)
            # defensive: a BM25 hiccup must never break the whole request
            except Exception as exc:  # noqa: BLE001 - the BM25 leg is one of two; RRF degrades to the vector leg alone rather than failing the query
                log.warning("bm25_retrieval_failed", error=str(exc))
                lexical_results = []
        bm25_ms = (time.perf_counter() - bm25_start) * 1000

        rrf_start = time.perf_counter()
        if mode == "semantic":
            # A single leg needs no rank fusion -- its own embedding
            # similarity score is already the ranking.
            final = sorted(vector_results, key=lambda chunk: chunk.score, reverse=True)[:top_k]
            merged = final
        elif mode == "keyword":
            final = sorted(lexical_results, key=lambda chunk: chunk.score, reverse=True)[:top_k]
            merged = final
        else:
            merged = reciprocal_rank_fusion([vector_results, lexical_results], k=settings.retrieval_rrf_k)
            final = merged[:top_k]
        rrf_ms = (time.perf_counter() - rrf_start) * 1000

        log_fields: dict[str, Any] = {
            "query_id": query_id,
            "embedding_retrieval_ms": round(embedding_ms, 2),
            "bm25_retrieval_ms": round(bm25_ms, 2),
            "rrf_merge_ms": round(rrf_ms, 2),
            "embedding_candidate_count": len(vector_results),
            "bm25_candidate_count": len(lexical_results),
            "merged_candidate_count": len(merged),
            "final_candidate_count": len(final),
            "top_chunk_ids": [chunk.chunk_id for chunk in final[:5]],
            "retrieval_route": mode,
        }
        if settings.retrieval_debug:
            # Part 40 section 16 debug mode: raw per-leg rankings, gated
            # behind the flag so production logs never carry full query
            # text/content by default (section 15).
            log_fields["debug_query"] = query
            log_fields["debug_top_bm25"] = [
                (c.chunk_id, round(c.score, 4), c.metadata.get("source_document")) for c in lexical_results[:5]
            ]
            log_fields["debug_top_embedding"] = [
                (c.chunk_id, round(c.score, 4), c.metadata.get("source_document")) for c in vector_results[:5]
            ]
            log_fields["debug_rrf_ranking"] = [
                (c.chunk_id, round(c.score, 4), c.metadata.get("source_document")) for c in merged[:5]
            ]
        log.info("hybrid_retrieval_timing", **log_fields)

        return final

    async def delete_by_source(self, source_document: str) -> int:
        collection = mongodb.db[EMBEDDINGS_METADATA]
        ids = [
            str(item["_id"])
            async for item in collection.find({"metadata.source_document": source_document}, {"_id": 1})
        ]
        result = await collection.delete_many({"metadata.source_document": source_document})
        await bm25_index.remove_by_source(source_document)
        if ids:
            await local_ann_index.remove_chunk_ids(set(ids))
        return result.deleted_count

    def _document_status_filter(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Forces `document_status="active"` into a filters dict headed for
        `_mongo_filter`/`_atlas_filter`, unconditionally overriding anything a
        caller supplied.

        Unlike `review_status` (a business/product gate `LegalRetriever`
        deliberately lets admin tooling override -- see
        `test_retriever_does_not_override_a_caller_supplied_review_status_filter`),
        `document_status` is a pure data-integrity invariant: a `staging`
        chunk is an in-flight, unvalidated reindex attempt and a `superseded`
        chunk is content a later version already replaced. No caller,
        including admin tooling, has a legitimate reason to retrieve either
        through a shared search path, so this is never `setdefault`.
        """
        return {**filters, "document_status": "active"}

    async def count_by_document_id(self, document_id: str) -> int:
        collection = mongodb.db[EMBEDDINGS_METADATA]
        return await collection.count_documents({"document_id": document_id})

    async def activate_version(self, new_document_id: str, old_document_id: str | None) -> None:
        collection = mongodb.db[EMBEDDINGS_METADATA]
        new_ids = [str(item["_id"]) async for item in collection.find({"document_id": new_document_id}, {"_id": 1})]
        await collection.update_many(
            {"document_id": new_document_id}, {"$set": {"metadata.document_status": "active"}}
        )
        await bm25_index.mark_document_status(set(new_ids), "active")
        await local_ann_index.mark_document_status(set(new_ids), "active")
        if not old_document_id:
            return
        # Scoped by the OLD document id specifically (not just
        # `document_status="active"` for this source) -- the new version's
        # chunks were just activated above under a DIFFERENT document id, so a
        # source-only scope here would immediately re-supersede them too.
        old_query = {"document_id": old_document_id, "metadata.document_status": "active"}
        old_ids = [str(item["_id"]) async for item in collection.find(old_query, {"_id": 1})]
        await collection.update_many(old_query, {"$set": {"metadata.document_status": "superseded"}})
        await bm25_index.mark_document_status(set(old_ids), "superseded")
        await local_ann_index.mark_document_status(set(old_ids), "superseded")

    async def delete_version_chunks(self, document_id: str) -> int:
        collection = mongodb.db[EMBEDDINGS_METADATA]
        query = {"document_id": document_id}
        ids = [str(item["_id"]) async for item in collection.find(query, {"_id": 1})]
        result = await collection.delete_many(query)
        if ids:
            await bm25_index.remove_chunk_ids(set(ids))
            await local_ann_index.remove_chunk_ids(set(ids))
        return result.deleted_count

    async def find_by_section_number(self, section_number: str, filters: dict[str, Any]) -> list[RetrievedChunk]:
        """Phase 8 "Low-Number Section Recall": a pure metadata equality
        lookup, bypassing similarity scoring entirely -- `search()` above
        (BM25 + vector, both similarity-ranked) can simply never surface a
        section's own chunk at all, regardless of candidate pool width.
        Confirmed live: BNS's own Section 2 chunk never enters the top-10
        semantic candidates for the bare query "Section 2" on either leg, so
        no amount of reranking or score-flooring downstream can rescue a
        chunk that was never retrieved in the first place. `LegalRetriever`
        calls this only for SECTION_LOOKUP queries with a resolved number
        (never widens candidate limits for anything else), and merges the
        result into the ordinary candidate pool rather than replacing it.
        """
        collection = mongodb.db[EMBEDDINGS_METADATA]
        mongo_filter = self._mongo_filter(
            self._document_status_filter({**filters, "section_number": section_number})
        )
        results: list[RetrievedChunk] = []
        async for item in collection.find(mongo_filter):
            results.append(
                RetrievedChunk(
                    chunk_id=str(item["_id"]),
                    text=item.get("text", ""),
                    score=0.0,
                    metadata=item.get("metadata", {}),
                )
            )
        return results

    async def find_constitution_article(
        self, article_number: str, filters: dict[str, Any]
    ) -> list[RetrievedChunk]:
        """Find an Article in both current and legacy Constitution indexes.

        Legacy indexes stored bare Constitution headings as section numbers
        and may have an opaque source filename.  Identify the instrument from
        its bilingual official title, requiring both signatures so an Act
        that merely mentions the Constitution cannot be swept in.
        """
        collection = mongodb.db[EMBEDDINGS_METADATA]
        source_sets: list[set[str]] = []
        for signature in ("THE CONSTITUTION OF INDIA", "भारत का संविधान"):
            names: set[str] = set()
            async for item in collection.find(
                {"text": {"$regex": signature, "$options": "i"}},
                {"metadata.source_document": 1},
            ):
                name = str((item.get("metadata") or {}).get("source_document") or "")
                if name:
                    names.add(name)
            source_sets.append(names)
        sources = set.intersection(*source_sets) if source_sets else set()
        if not sources:
            return []
        safe_filters = {
            key: value
            for key, value in filters.items()
            if key not in {"article_number", "section_number", "instrument_type"}
        }
        mongo_filter = self._mongo_filter(safe_filters)
        mongo_filter["metadata.document_status"] = "active"
        mongo_filter["metadata.source_document"] = {"$in": sorted(sources)}
        mongo_filter["$and"] = [
            {
                "$or": [
                    {"metadata.article_number": article_number},
                    {"metadata.section_number": article_number},
                ]
            }
        ]
        results: list[RetrievedChunk] = []
        async for item in collection.find(mongo_filter):
            metadata = dict(item.get("metadata", {}))
            # Normalize only the in-memory retrieval result.  The stored KB
            # record remains untouched, while prompts/citations correctly
            # call this an Article of the Constitution rather than "Section
            # 21 of Document.pdf".
            metadata["instrument_type"] = "constitution"
            metadata["act_name"] = "The Constitution of India"
            metadata["article_number"] = article_number
            metadata.pop("section_number", None)
            results.append(
                RetrievedChunk(
                    chunk_id=str(item["_id"]),
                    text=item.get("text", ""),
                    score=0.0,
                    metadata=metadata,
                )
            )
        return results

    async def find_named_section(
        self, section_number: str, act_name: str, filters: dict[str, Any]
    ) -> list[RetrievedChunk]:
        """Return chunks containing the provision's own heading in the named Act.

        Older oversized/overlap chunks can contain ``138. Dishonour...`` but
        carry the next/previous heading as metadata. Heading text plus source
        identity is stronger evidence than that stale tag and avoids choosing
        a sample notice merely because its metadata says 138.
        """
        collection = mongodb.db[EMBEDDINGS_METADATA]
        safe_filters = {
            key: value
            for key, value in filters.items()
            if key not in {"act_name", "section_number", "article_number", "instrument_type"}
        }
        mongo_filter = self._mongo_filter(safe_filters)
        mongo_filter["metadata.document_status"] = "active"
        mongo_filter["text"] = {
            # Operative Indian-statute headings carry an em/en dash before
            # their body.  Requiring it excludes the same numbered line in
            # the arrangement-of-sections table of contents.
            "$regex": rf"^[ \t]*{re.escape(section_number)}\.\s+[^\n]{{0,180}}[—–]",
            "$options": "m",
        }
        requested_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", act_name.lower())
            if token not in {"the", "act", "code", "of", "and"}
        }
        results: list[RetrievedChunk] = []
        async for item in collection.find(mongo_filter):
            metadata = dict(item.get("metadata", {}))
            identity = " ".join(
                str(metadata.get(key) or "") for key in ("source_document", "source", "act_name")
            ).lower().replace("_", " ")
            identity_tokens = set(re.findall(r"[a-z0-9]+", identity))
            if requested_tokens and not requested_tokens <= identity_tokens:
                continue
            metadata["act_name"] = act_name
            metadata["section_number"] = section_number
            metadata["section_number_provenance"] = "heading"
            metadata["source_type"] = "statute"
            results.append(
                RetrievedChunk(
                    chunk_id=str(item["_id"]),
                    text=item.get("text", ""),
                    score=0.0,
                    metadata=metadata,
                )
            )
        # A long s.138 provision can be split into a heading chunk and a
        # continuation chunk. The latter holds provisos (b)/(c) but inherits
        # s.139 metadata because that next heading occurs near its tail.
        confirmed_sources = {
            str(result.metadata.get("source_document") or "")
            for result in results
            if result.metadata.get("source_document")
        }
        if section_number == "138" and confirmed_sources:
            continuation_filter = self._mongo_filter(safe_filters)
            continuation_filter["metadata.document_status"] = "active"
            continuation_filter["metadata.source_document"] = {"$in": sorted(confirmed_sources)}
            continuation_filter["text"] = {"$regex": "within thirty days", "$options": "i"}
            existing_ids = {result.chunk_id for result in results}
            async for item in collection.find(continuation_filter):
                text = item.get("text", "")
                if "within fifteen days" not in text.lower() or str(item["_id"]) in existing_ids:
                    continue
                metadata = dict(item.get("metadata", {}))
                metadata.update(
                    {
                        "act_name": act_name,
                        "section_number": section_number,
                        "section_number_provenance": "heading_continuation",
                        "source_type": "statute",
                    }
                )
                results.append(
                    RetrievedChunk(
                        chunk_id=str(item["_id"]), text=text, score=0.0, metadata=metadata
                    )
                )
        return results

    async def _atlas_vector_leg(
        self, query_embedding: list[float], limit: int, filters: dict[str, Any]
    ) -> list[RetrievedChunk]:
        atlas_filter = self._atlas_filter(filters)
        vector_search_stage: dict[str, Any] = {
            "index": settings.mongodb_vector_index,
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": max(limit * 20, 200),
            "limit": limit,
        }
        if atlas_filter:
            vector_search_stage["filter"] = atlas_filter
        pipeline: list[dict[str, Any]] = [
            {"$vectorSearch": vector_search_stage},
            {
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        collection = mongodb.db[EMBEDDINGS_METADATA]
        results: list[RetrievedChunk] = []
        async for item in collection.aggregate(pipeline):
            results.append(
                RetrievedChunk(
                    chunk_id=str(item["_id"]),
                    text=item.get("text", ""),
                    score=float(item.get("score", 0.0)),
                    metadata=item.get("metadata", {}),
                )
            )
        return results

    async def _local_cosine_leg(
        self, query_embedding: list[float], limit: int, filters: dict[str, Any]
    ) -> list[RetrievedChunk]:
        """QA pass 2026-09-24 (60k-chunk local-scan latency/timeout, see
        `local_ann_index`'s module docstring): tries the FAISS ANN index
        first -- O(log n) per query instead of this method's own O(n)
        brute-force scan below, with no per-query MongoDB fetch of the full
        candidate pool's text/metadata/embeddings at all (only the winning
        handful get hydrated, see `LocalAnnIndex._hydrate_text`). Falls
        through to the unchanged brute-force scan when the ANN index isn't
        usable for this call (FAISS not installed, never successfully
        built, or a dimension mismatch) -- `LocalAnnIndex.search` returns
        `None`, not `[]`, specifically so "ANN unavailable" is never
        confused with "zero real matches" here.
        """
        if await local_ann_index.ensure_current_generation():
            ann_results = await local_ann_index.search(query_embedding, limit, filters)
            if ann_results is not None:
                return ann_results
        return await self._local_cosine_leg_brute_force(query_embedding, limit, filters)

    async def _local_cosine_leg_brute_force(
        self, query_embedding: list[float], limit: int, filters: dict[str, Any]
    ) -> list[RetrievedChunk]:
        """The original, exact (non-approximate) local fallback: fetches
        candidates via the async Mongo cursor (real I/O, yields to the event
        loop between batches as normal), then hands the actual similarity
        scoring off to a worker thread via `asyncio.to_thread` -- mirroring
        `EmbeddingProvider._encode_sync`'s own reasoning one module over.
        Scoring thousands of candidates one Python `sum()` generator
        expression at a time (the pre-fix shape) ran entirely on the event
        loop with no `await` in the loop body to yield control back, so it
        could stall EVERY other in-flight coroutine on this same process --
        including this same request's own Mongo/Redis health pings -- for as
        long as the scan took. Confirmed live: concurrent `/health` calls
        during retrieval-heavy chat traffic reported Mongo/Redis as
        "unavailable" (a ~30s `pymongo`/`redis` client-side timeout) purely
        because the event loop was too busy to schedule their awaits in
        time, not because Mongo or Redis were actually unreachable.

        Kept as the safety-net fallback for `_local_cosine_leg` above (the
        ANN index being unusable for any reason) rather than removed --
        still exact, still correct, just the pre-ANN latency profile.
        """
        collection = mongodb.db[EMBEDDINGS_METADATA]
        mongo_filter = self._mongo_filter(filters)
        async with self._local_scan_semaphore:
            scan_started = time.perf_counter()
            cursor = collection.find(mongo_filter).limit(self.local_scan_limit)
            chunk_ids: list[str] = []
            texts: list[str] = []
            metadatas: list[dict[str, Any]] = []
            embeddings: list[list[float]] = []
            async for item in cursor:
                chunk_ids.append(str(item["_id"]))
                texts.append(item.get("text", ""))
                metadatas.append(item.get("metadata", {}))
                embeddings.append(item.get("embedding") or [])
            results = await self._scan_and_score(
                query_embedding, chunk_ids, texts, metadatas, embeddings, limit, scan_started
            )
        return results

    async def _scan_and_score(
        self,
        query_embedding: list[float],
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
        limit: int,
        scan_started: float,
    ) -> list[RetrievedChunk]:
        if len(chunk_ids) >= self.local_scan_limit:
            # The cursor came back exactly at the cap: the corpus (after
            # `filters`) is at least this large, and Mongo's natural-order
            # scan may have silently dropped whichever documents were
            # indexed most recently. Visible in logs/metrics instead of a
            # silent truncation -- see `settings.local_vector_scan_limit`.
            metrics.increment("local_vector_scan_truncated")
            log.warning(
                "local_vector_scan_hit_cap",
                scan_limit=self.local_scan_limit,
                hint="Raise LOCAL_VECTOR_SCAN_LIMIT or configure Atlas vector search.",
            )
        scan_ms = (time.perf_counter() - scan_started) * 1000
        score_started = time.perf_counter()
        results = await asyncio.to_thread(
            self._score_and_rank_sync, query_embedding, chunk_ids, texts, metadatas, embeddings, limit
        )
        log.info("local_vector_timing", candidate_count=len(chunk_ids), mongo_scan_ms=round(scan_ms, 2),
                 score_ms=round((time.perf_counter() - score_started) * 1000, 2))
        return results

    def _score_and_rank_sync(
        self,
        query_embedding: list[float],
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
        limit: int,
    ) -> list[RetrievedChunk]:
        scores = self._cosine_batch(query_embedding, embeddings)
        # Only materialize the winners, not thousands of discarded Pydantic
        # objects per request. nlargest preserves input order for tied scores.
        winners = heapq.nlargest(limit, range(len(scores)), key=scores.__getitem__)
        return [
            RetrievedChunk(chunk_id=chunk_ids[i], text=texts[i], score=scores[i], metadata=metadatas[i])
            for i in winners
        ]

    def _mongo_filter(self, filters: dict[str, Any]) -> dict[str, Any]:
        # Part 45 "Per-User Document Isolation": a list/tuple value means
        # "must be one of these" (membership), not "equals this list" --
        # used for `owner_session_id: [None, session_id]` ("mine, or
        # nobody's"). `$in` with `None` in the list also matches documents
        # where the field is absent entirely (standard Mongo behavior), so
        # every pre-existing document (curated or uploaded, none of which
        # have this field) stays visible with no reindex. Scalar values keep
        # the original plain-equality behavior, unchanged.
        #
        # Part 46 "Authenticated User Ownership": a reserved `"$or"` key
        # (list of sub-filter-dicts) expresses "global OR mine (by
        # session) OR mine (by authenticated user)" -- a boolean shape the
        # single flat AND-of-memberships above can't express on its own.
        # Recurses through this same method for each branch, then relies on
        # MongoDB's native `$or` (valid mixed with sibling equality/`$in`
        # keys as an implicit AND, standard Mongo query syntax).
        result: dict[str, Any] = {}
        or_branches = filters.get("$or")
        if or_branches:
            result["$or"] = [self._mongo_filter(branch) for branch in or_branches]
        # Jurisdiction Routing (Phase 2): candidate-selection-level date
        # eligibility -- see `kb_jurisdiction.TEMPORAL_FILTER_KEY`'s own
        # comment for why this must not be left to the retriever's
        # post-filter alone.
        as_of_date = filters.get(TEMPORAL_FILTER_KEY)
        if as_of_date:
            result["$and"] = mongo_temporal_and_clauses(as_of_date)
        for key, value in filters.items():
            if key in ("$or", TEMPORAL_FILTER_KEY) or not value:
                continue
            if isinstance(value, (list, tuple)):
                result[f"metadata.{key}"] = {"$in": list(value)}
            else:
                result[f"metadata.{key}"] = value
        return result

    def _atlas_filter(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """Confirmed live (2026-08-24) against a real Atlas Search deployment
        (`mongodb/mongodb-atlas-local` -- no Atlas cloud cluster exists in
        this dev environment, but that image bundles a real mongot, so this
        is a genuine `$vectorSearch` filter engine, not a guess): a literal
        `null` INSIDE an `$in` array is rejected outright --
        `"filter[0][0].metadata.owner_session_id.$in[0]" value type cannot
        be null`. This would have broken EVERY Part 45/46 ownership-scoped
        query (`owner_session_id: [None, session_id]` is exactly this
        shape) the very first time this app ran against real Atlas, always
        silently falling back to `_local_cosine_leg` on the resulting
        `PyMongoError` -- functionally fine today (no Atlas backend has
        ever been live), but broken the moment one is. Confirmed `$eq: null`
        (a bare scalar equality, not inside `$in`) is accepted and correctly
        matches an explicit-null field the way this app's data actually
        stores it (`owner_user_id: None`, a present key, not an absent
        one -- confirmed `$exists: false` alone does NOT match it, only
        `$eq: null` or `$exists: true` do). `_mongo_filter` above (the
        local/non-Atlas leg) is unaffected -- plain MQL `$in` already
        handles a `None` list element correctly there.
        """
        clauses: list[dict[str, Any]] = []
        or_branches = filters.get("$or")
        if or_branches:
            sub_clauses = [c for c in (self._atlas_filter(branch) for branch in or_branches) if c]
            if sub_clauses:
                clauses.append({"$or": sub_clauses})
        # Jurisdiction Routing (Phase 2): same candidate-selection-level date
        # constraint as `_mongo_filter` above. The explicit `{path: None}`
        # branch inside `mongo_temporal_and_clauses` (not `$exists` alone) is
        # what makes this correct here specifically -- see this method's own
        # docstring: `$exists: false` alone does NOT match a present-but-null
        # field on this Atlas deployment, only `$eq`/bare equality to `None`
        # does, and each temporal clause already includes that branch.
        as_of_date = filters.get(TEMPORAL_FILTER_KEY)
        if as_of_date:
            clauses.extend(mongo_temporal_and_clauses(as_of_date))
        for key, value in filters.items():
            if key in ("$or", TEMPORAL_FILTER_KEY) or not value:
                continue
            if isinstance(value, (list, tuple)):
                path = f"metadata.{key}"
                non_null = [item for item in value if item is not None]
                has_null = any(item is None for item in value)
                if has_null and non_null:
                    clauses.append({"$or": [{path: {"$eq": None}}, {path: {"$in": non_null}}]})
                elif has_null:
                    clauses.append({path: {"$eq": None}})
                else:
                    clauses.append({path: {"$in": non_null}})
            else:
                clauses.append({f"metadata.{key}": {"$eq": value}})
        if not clauses:
            return None
        return {"$and": clauses} if len(clauses) > 1 else clauses[0]

    def _cosine_batch(self, query_embedding: list[float], candidate_embeddings: list[list[float]]) -> list[float]:
        """Vectorized replacement for the old one-pair-at-a-time `_cosine`:
        a single BLAS matrix-vector multiply over every candidate instead of
        thousands of individual Python-level `sum()` generator expressions.
        Same defensive semantics as the original (a missing or
        wrong-dimension candidate embedding scores 0.0 rather than raising),
        just evaluated as a mask over the whole batch instead of an
        early-return per call.
        """
        if not candidate_embeddings:
            return []
        dim = len(query_embedding)
        query = np.asarray(query_embedding, dtype=np.float64)
        query_norm = float(np.linalg.norm(query))
        if dim == 0 or query_norm == 0.0:
            return [0.0] * len(candidate_embeddings)
        matrix = np.zeros((len(candidate_embeddings), dim), dtype=np.float64)
        valid = np.zeros(len(candidate_embeddings), dtype=bool)
        for index, embedding in enumerate(candidate_embeddings):
            if embedding and len(embedding) == dim:
                matrix[index] = embedding
                valid[index] = True
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        scores = (matrix @ query) / (norms * query_norm)
        scores[~valid] = 0.0
        return [float(score) for score in scores.tolist()]
