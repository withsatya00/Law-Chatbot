import asyncio
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from app.database.mongodb import mongodb
from app.models.collections import EMBEDDINGS_METADATA
from app.rag.metadata_filter import matches_filters
from app.rag.types import DocumentChunk
from app.schemas.common import RetrievedChunk

log = structlog.get_logger(__name__)

# ASCII alphanumeric runs are kept whole (so "173", "bnss", "cognizable"
# survive intact -- exactly the section numbers and legal terms a heavier
# stemmer/stopword-remover would risk mangling), and each Indic script's runs
# are kept whole as their own tokens for that language's corpus text and
# queries alike (Part 40 section 3). A hyphenated compound like "e-FIR"
# splits into ["e", "fir"] rather than being preserved as one token -- not
# ideal, but a query for "e-FIR" tokenizes the identical way, so the match
# still happens via the shared "fir"/"e" tokens either side.
#
# Originally Devanagari-only. Confirmed live (qa-40q-multilingual-20260921
# BUG-03): a Punjabi (Gurmukhi) query on a topic proven present in the KB --
# the same mobile-theft/police-complaint question that succeeded in Hinglish
# and in Devanagari Hindi -- got the bare "no verified document" refusal.
# `tokenize()` returned an empty list for the Gurmukhi text (no character in
# it matched `[a-zA-Z0-9]+|[ऀ-ॿ]+`), so its BM25 leg of retrieval contributed
# nothing regardless of how well the embedding/concept-bridge legs did.
# Extended here to every script this app's 22 scheduled languages actually
# use, so no supported language silently loses its BM25 leg the way Gurmukhi,
# Gujarati, Tamil, Telugu, Kannada, Malayalam, Bengali/Assamese, Odia,
# Perso-Arabic (Urdu/Sindhi/Kashmiri), Ol Chiki (Santali) and Meetei Mayek
# (Manipuri) all did before this fix.
_TOKEN_RE = re.compile(
    r"[a-zA-Z0-9]+"
    r"|[ऀ-ॿ]+"  # Devanagari: Hindi, Marathi, Sanskrit, Nepali, Konkani, Maithili, Bodo, Dogri
    r"|[ঀ-৿]+"  # Bengali script: Bengali, Assamese
    r"|[਀-੿]+"  # Gurmukhi: Punjabi
    r"|[઀-૿]+"  # Gujarati
    r"|[଀-୿]+"  # Odia
    r"|[஀-௿]+"  # Tamil
    r"|[ఀ-౿]+"  # Telugu
    r"|[ಀ-೿]+"  # Kannada
    r"|[ഀ-ൿ]+"  # Malayalam
    r"|[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]+"  # Perso-Arabic: Urdu, Sindhi, Kashmiri
    r"|[᱐-᱿]+"  # Ol Chiki: Santali
    r"|[ꫠ-꫿ꯀ-꯿]+"  # Meetei Mayek: Manipuri
)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


_DEFAULT_CACHE_PATH = Path("./storage/bm25_index/index.pkl")


class _FrequencyBM25(BM25Okapi):  # type: ignore[misc]  # rank_bm25 ships no type stubs; BM25Okapi is Any to mypy.
    """Reuse unchanged document term counts, recomputing global IDF exactly."""

    def _initialize(self, corpus: list[dict[str, int]]) -> "Counter[str]":
        frequencies: Counter[str] = Counter()
        self.doc_freqs = list(corpus)
        self.doc_len = [sum(document.values()) for document in corpus]
        self.corpus_size = len(corpus)
        self.avgdl = sum(self.doc_len) / self.corpus_size
        for document in corpus:
            frequencies.update(document.keys())
        return frequencies


class BM25Index:
    """Persisted, reusable BM25 lexical index over the same chunks the vector
    store indexes (Part 40 sections 2/13) -- built once (loaded from disk, or
    scanned from Mongo if no cache file exists yet) and kept in memory across
    requests. Never rebuilt per query: `search()` only ever reads the
    already-built `BM25Okapi` instance.

    Updated incrementally by `add_or_update_chunks`/`remove_by_source`,
    called from `MongoVectorStore.upsert_chunks`/`delete_by_source` so BM25
    stays synchronized with the vector index through the exact same
    ingestion/deindex hooks (section 14) -- no separate ingestion path.
    """

    def __init__(self, cache_path: Path = _DEFAULT_CACHE_PATH) -> None:
        self.cache_path = cache_path
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._lock = asyncio.Lock()
        self._snapshot: tuple[_FrequencyBM25 | None, list[str], list[str], list[dict[str, Any]]] = (
            None, [], [], [],
        )

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_ids)

    async def ensure_loaded(self) -> None:
        """Idempotent: a no-op once loaded. Called both eagerly at app
        startup (`app/main.py`'s `lifespan`) and defensively before every
        search, so a query still works correctly (just pays the one-time
        load cost) even if startup loading was skipped or failed.
        """
        if self._bm25 is not None:
            return
        async with self._lock:
            if self._bm25 is not None:
                return
            if not await asyncio.to_thread(self._load_from_disk):
                await self._rebuild_from_mongo()

    async def rebuild(self) -> int:
        """Full rebuild from Mongo, bypassing any disk cache -- for an
        explicit admin-triggered resync if the two ever drift.
        """
        async with self._lock:
            return await self._rebuild_from_mongo()

    async def ensure_current_generation(self) -> None:
        """Refresh worker-local/disk metadata after any shared KB publication.

        Redis failure propagates to the vector store's existing lexical-leg
        fallback: serving no keyword results is preferable to stale approval
        or effective dates. First use rebuilds instead of trusting an undated
        disk snapshot. Each process observes the same generation independently.
        """
        from app.cache.redis_client import redis_client
        from app.cache.response_cache import CACHE_KEY_PREFIX

        generation = await redis_client.client.get(f"{CACHE_KEY_PREFIX}:kb-generation") or b"0"
        async with self._lock:
            if getattr(self, "_kb_generation", None) != generation or self._bm25 is None:
                await self._rebuild_from_mongo()
                self._kb_generation = generation

    async def add_or_update_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        async with self._lock:
            if self._bm25 is None and not await asyncio.to_thread(self._load_from_disk):
                # Nothing loaded yet -- a fresh rebuild from Mongo already
                # picks up these chunks (upsert_chunks writes to Mongo
                # before calling this), so there's nothing left to merge.
                await self._rebuild_from_mongo()
                return
            by_id = {
                chunk_id: (text, metadata)
                for chunk_id, text, metadata in zip(self._chunk_ids, self._texts, self._metadatas, strict=True)
            }
            for chunk in chunks:
                by_id[chunk.chunk_id] = (chunk.text, chunk.metadata)
            started = time.perf_counter()
            await asyncio.to_thread(self._set_corpus, list(by_id.keys()), [v[0] for v in by_id.values()], [v[1] for v in by_id.values()])
            rebuild_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            await asyncio.to_thread(self._save_to_disk)
            log.info("bm25_update_timing", chunk_count=len(by_id), updated_count=len(chunks),
                     rebuild_ms=round(rebuild_ms, 2), persist_ms=round((time.perf_counter() - started) * 1000, 2))

    async def remove_by_source(self, source_document: str) -> None:
        async with self._lock:
            if self._bm25 is None and not await asyncio.to_thread(self._load_from_disk):
                await self._rebuild_from_mongo()
                return
            kept = [
                (chunk_id, text, metadata)
                for chunk_id, text, metadata in zip(self._chunk_ids, self._texts, self._metadatas, strict=True)
                if metadata.get("source_document") != source_document
            ]
            chunk_ids = [item[0] for item in kept]
            texts = [item[1] for item in kept]
            metadatas = [item[2] for item in kept]
            await asyncio.to_thread(self._set_corpus, chunk_ids, texts, metadatas)
            await asyncio.to_thread(self._save_to_disk)

    async def mark_document_status(self, chunk_ids: set[str], status: str) -> int:
        """Patches `metadata.document_status` in place for chunk ids already
        in the corpus (P0-2).

        `MongoVectorStore.activate_version` changes this ONE field on
        existing chunk rows via a raw `update_many` -- it does not go through
        `add_or_update_chunks` (which would re-tokenize text that has not
        changed), so without this, the BM25 leg's in-memory copy of these
        chunks' metadata would keep answering `search()`'s `document_status=
        "active"` filter with a stale value: a freshly-activated version
        invisible to the lexical leg until the next generation-bump rebuild,
        or a just-superseded version still answering queries through it in
        the meantime. Cheap by design -- only the metadata dict changes, so
        the tokenized corpus itself is never rebuilt.
        """
        if not chunk_ids:
            return 0
        async with self._lock:
            if self._bm25 is None and not await asyncio.to_thread(self._load_from_disk):
                return 0
            updated = 0
            for index, chunk_id in enumerate(self._chunk_ids):
                if chunk_id in chunk_ids:
                    self._metadatas[index] = {**self._metadatas[index], "document_status": status}
                    updated += 1
            if updated:
                await asyncio.to_thread(self._save_to_disk)
            return updated

    async def remove_chunk_ids(self, chunk_ids: set[str]) -> int:
        """Drop specific chunk ids from the corpus. Returns how many went.

        The removal primitive this index was missing. `remove_by_source` keys
        on `source_document`, so re-indexing a document -- which mints fresh
        `uuid4` chunk ids -- left the previous generation's entries in the
        pickle forever: 5598 BM25 chunks against 2079 in Mongo at Phase 2
        baseline. A stale entry is still retrievable and still carries the
        ownership metadata it had when it was written, so this is a
        correctness and privacy fix, not tidying.

        Deliberately takes explicit ids rather than a predicate: the caller
        (`app/rag/reconciliation.py`) has already checked each one against
        MongoDB, and an index that can delete by its own guesswork is an index
        that can delete the wrong thing.
        """
        if not chunk_ids:
            return 0
        async with self._lock:
            if self._bm25 is None and not await asyncio.to_thread(self._load_from_disk):
                # Nothing is loaded, so there is nothing stale to prune. A
                # rebuild here would be a much larger action than asked for.
                return 0
            kept = [
                (chunk_id, text, metadata)
                for chunk_id, text, metadata in zip(self._chunk_ids, self._texts, self._metadatas, strict=True)
                if chunk_id not in chunk_ids
            ]
            removed = len(self._chunk_ids) - len(kept)
            if not removed:
                return 0
            await asyncio.to_thread(self._set_corpus,
                [item[0] for item in kept],
                [item[1] for item in kept],
                [item[2] for item in kept],
            )
            await asyncio.to_thread(self._save_to_disk)
            log.info("bm25_chunks_removed", removed=removed, remaining=len(kept))
            return removed

    def search(self, query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        """Synchronous by design: `BM25Okapi.get_scores` is a fast, already
        in-memory numpy computation over the pre-built index, not I/O -- no
        need to force callers through `await` for it. Returns `[]` (never
        raises) if the index isn't loaded yet or the query has no tokens, so
        a caller can treat "BM25 unavailable" and "BM25 found nothing" the
        same way (Part 40 section 17's fallback).
        """
        bm25, chunk_ids, texts, metadatas = self._snapshot
        if bm25 is None or not chunk_ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        max_score = float(max(scores)) if len(scores) else 0.0
        if max_score <= 0:
            return []
        # P0-2: mirrors `MongoVectorStore.search`'s own unconditional
        # `document_status="active"` gate -- see that method's comment for
        # why this is never caller-overridable. Forced here too rather than
        # trusted to arrive already-set, since `BM25Index.search` is also
        # called directly (not only through `MongoVectorStore.search`).
        filters = {**(filters or {}), "document_status": "active"}
        candidates: list[tuple[str, str, dict[str, Any], float]] = []
        for chunk_id, text, metadata, score in zip(chunk_ids, texts, metadatas, scores, strict=True):
            if score <= 0:
                continue
            if not self._matches_filters(metadata, filters):
                continue
            candidates.append((chunk_id, text, metadata, float(score)))
        candidates.sort(key=lambda item: item[3], reverse=True)
        return [
            RetrievedChunk(chunk_id=chunk_id, text=text, score=min(1.0, score / max_score), metadata=metadata)
            for chunk_id, text, metadata, score in candidates[:top_k]
        ]

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        # Part 45/46 ownership + Jurisdiction Routing (Phase 2) semantics --
        # see `app.rag.metadata_filter.matches_filters`'s own docstring.
        # Extracted there (2026-09-24) so `LocalAnnIndex`'s own in-memory
        # filtering leg cannot silently drift from this one.
        return matches_filters(metadata, filters)

    def _set_corpus(self, chunk_ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        previous = {
            key: (text, counts)
            for key, text, counts in zip(
                self._chunk_ids, self._texts,
                self._bm25.doc_freqs if self._bm25 is not None else [],
            )
        }
        counts = []
        for key, text in zip(chunk_ids, texts, strict=True):
            old = previous.get(key)
            counts.append(old[1] if old is not None and old[0] == text else dict(Counter(tokenize(text))))
        # BM25's IDF calculation requires at least one term for an empty corpus.
        bm25 = _FrequencyBM25(counts) if any(counts) else None
        self._chunk_ids, self._texts, self._metadatas = chunk_ids, texts, metadatas
        self._bm25 = bm25
        # Search captures one immutable tuple so an off-thread rebuild cannot
        # pair new scores with old IDs/ownership metadata.
        self._snapshot = (bm25, chunk_ids, texts, metadatas)

    async def _rebuild_from_mongo(self) -> int:
        collection = mongodb.db[EMBEDDINGS_METADATA]
        chunk_ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        cursor = collection.find({}, {"text": 1, "metadata": 1})
        async for item in cursor:
            chunk_ids.append(str(item["_id"]))
            texts.append(item.get("text", ""))
            metadatas.append(item.get("metadata", {}))
        # `_set_corpus` tokenizes every chunk in the corpus and builds a
        # fresh `BM25Okapi` index -- pure-Python, CPU-bound, and (like
        # `search()`, see the call site in `vector_store.py`) scales with
        # total corpus size, not top-k. Left inline, a rebuild triggered by
        # `ensure_current_generation()` mid-request stalls the WHOLE
        # process's event loop -- every other in-flight request, including
        # unrelated ones -- for as long as the rebuild takes. `_save_to_disk`
        # is blocking file I/O for the same reason. Both run off-loop.
        await asyncio.to_thread(self._set_corpus, chunk_ids, texts, metadatas)
        await asyncio.to_thread(self._save_to_disk)
        log.info("bm25_index_rebuilt", chunk_count=len(chunk_ids))
        return len(chunk_ids)

    def _save_to_disk(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("wb") as handle:
                pickle.dump(
                    {"chunk_ids": self._chunk_ids, "texts": self._texts, "metadatas": self._metadatas}, handle
                )
        except OSError as exc:
            log.warning("bm25_index_persist_failed", error=str(exc))

    def _load_from_disk(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            with self.cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            self._set_corpus(payload["chunk_ids"], payload["texts"], payload["metadatas"])
            log.info("bm25_index_loaded_from_disk", chunk_count=len(self._chunk_ids))
            return True
        except (OSError, pickle.PickleError, KeyError, EOFError) as exc:
            log.warning("bm25_index_disk_load_failed", error=str(exc))
            return False


# Module-level singleton, mirroring `app/cache/semantic_cache.py`'s
# `response_cache`/`retrieval_cache` pattern -- `ChatService` (and therefore
# `LegalRetriever`/`MongoVectorStore`) is reconstructed fresh on EVERY
# request (see `app/api/chat.py`), so anything that must persist/be loaded
# only once has to live outside that per-request object graph. Imported by
# `app/main.py`'s `lifespan` (eager load at startup) and by
# `MongoVectorStore` (query-time search + ingestion-time sync).
bm25_index = BM25Index()
