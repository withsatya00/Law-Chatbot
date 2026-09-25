"""In-process approximate-nearest-neighbor index for the local (non-Atlas)
retrieval fallback.

QA pass 2026-09-24 ("60,000-chunk local vector scan latency/timeout"):
`MongoVectorStore._local_cosine_leg` fetched up to `settings.
local_vector_scan_limit` full documents (id + text + metadata + the full
1024-dim embedding vector) from MongoDB on EVERY query, then brute-force
scored every one of them against the query vector -- an O(corpus size) cost
paid per query, not a one-time cost. Raising that limit to keep the whole
corpus reachable (see `local_vector_scan_limit_must_track_corpus_size`) fixed
correctness but made the cost worse: a live `/chat` call measured exceeding
the 150s request budget under concurrent KB-automation contention at 60,000.

This module trades that O(n)-per-query brute-force scan for an O(log n)
approximate one, using an in-memory FAISS HNSW graph (cosine similarity via
L2-normalized vectors + inner product) built once and kept warm across
requests -- mirroring `app.rag.bm25_index.BM25Index`'s own pattern exactly
(disk-cached, refreshed on the same Redis `kb-generation` counter
`ResponseCache.bump_generation()` already exists to invalidate, incrementally
updated on ordinary ingestion writes). Real MongoDB Atlas Vector Search
(`MongoVectorStore._atlas_vector_leg`) remains the preferred backend when
configured (`settings.vector_search_backend="atlas"`) -- this index only
exists for the `"local"` fallback this dev environment actually runs
(community MongoDB, no `mongot`/Atlas Search engine available; confirmed via
`.env`'s `VECTOR_SEARCH_BACKEND=local` and the running `mongo:7` container,
not `mongodb/mongodb-atlas-local`).

FAISS is not authoritative for `document_status`/ownership/jurisdiction
filtering -- an ANN graph has no query-time filter awareness. Every search
over-fetches (asks the graph for more neighbors than `limit`) and applies
`app.rag.metadata_filter.matches_filters` in Python afterward, the exact same
predicate `BM25Index` already uses, so a narrow filter (a specific session's
own private uploads) still gets a fair shot at enough true candidates rather
than being starved by an over-fetch window sized for the unfiltered case.
`search()` returns `None` (not `[]`) when the index cannot be used at all
(FAISS unavailable, never built, embedding-dimension mismatch) so the caller
can fall back to the exact, slower `_local_cosine_leg` brute-force scan --
this index is an optimization of the local fallback, never a silent
correctness downgrade from it.
"""

import asyncio
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from pymongo.errors import PyMongoError

from app.core.config import settings
from app.database.mongodb import mongodb
from app.models.collections import EMBEDDINGS_METADATA
from app.observability.metrics import metrics
from app.rag.metadata_filter import matches_filters
from app.rag.types import DocumentChunk
from app.schemas.common import RetrievedChunk

log = structlog.get_logger(__name__)

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only on a host without faiss-cpu installed
    faiss = None  # type: ignore[assignment]

_DEFAULT_CACHE_DIR = Path("./storage/local_ann_index")
_HNSW_M = 32
_HNSW_EF_CONSTRUCTION = 80
_HNSW_EF_SEARCH_FLOOR = 64
# How many extra candidates the ANN graph is asked for beyond `limit`, before
# `matches_filters` narrows them down -- widened progressively (see `search`)
# only if the first, cheaper attempt starves under a narrow filter.
_OVER_FETCH_MULTIPLIER = 6


@dataclass(frozen=True)
class _BuiltIndex:
    """A freshly built FAISS index plus its parallel id/metadata arrays,
    not yet installed as `LocalAnnIndex`'s live state -- see `_fetch_and_
    build`/`_swap_in`'s own docstrings for why construction and installation
    are two separate steps."""

    index: Any
    chunk_ids: list[str]
    metadatas: list[dict[str, Any]]
    dim: int


class LocalAnnIndex:
    """Disk-cached, incrementally-updated FAISS HNSW index over
    `embeddings_metadata`'s embedding vectors. See module docstring."""

    def __init__(self, cache_dir: Path = _DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self._index: Any = None
        self._chunk_ids: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._removed_positions: set[int] = set()
        self._dim: int = 0
        self._lock = asyncio.Lock()
        self._kb_generation: bytes | None = None
        self._background_rebuild_task: asyncio.Task[None] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_ids) - len(self._removed_positions)

    async def ensure_current_generation(self) -> bool:
        """Idempotent per-generation refresh, mirroring `BM25Index.
        ensure_current_generation`'s trigger (the same Redis `kb-generation`
        counter, so an operator who already knows to call `ResponseCache.
        bump_generation()` after a retrieval-affecting change invalidates
        this index too) but deliberately NOT its blocking rebuild -- see
        `_kick_off_background_rebuild`'s own comment for why. Returns
        whether the index is usable after this call; `False` (FAISS
        missing, Mongo unreachable, empty corpus, first build still running)
        tells the caller to fall back to the brute-force scan for this
        request rather than blocking it on a retry loop.
        """
        if faiss is None:
            return False
        try:
            from app.cache.redis_client import redis_client
            from app.cache.response_cache import CACHE_KEY_PREFIX

            generation = await redis_client.client.get(f"{CACHE_KEY_PREFIX}:kb-generation") or b"0"
        except Exception as exc:  # noqa: BLE001 - Redis unavailable must not block retrieval; keep serving whatever generation is already loaded
            log.warning("local_ann_generation_check_failed", error=str(exc))
            generation = self._kb_generation
        if self._index is not None and generation == self._kb_generation:
            return True
        if self._index is not None:
            # QA pass 2026-09-24: confirmed live -- a background KB-
            # automation job bumps `kb-generation` while this dev machine's
            # ingestion runs continuously (see [[local-vector-scan-limit-
            # must-track-corpus-size]]), and a synchronous rebuild here (the
            # original design, mirroring BM25's own blocking behavior)
            # measured a live `/chat` request's OWN retrieval leg stalling
            # for ~90s waiting on a full 56,000-vector HNSW rebuild it
            # happened to trigger -- reintroducing, via a different
            # mechanism, exactly the multi-second retrieval-latency problem
            # this index exists to fix. An index that is already built and
            # merely stale is still correct enough to serve immediately
            # (new chunks since the last rebuild are ALSO individually
            # reachable already via `add_or_update_chunks`'s incremental
            # `index.add` -- a full rebuild here only additionally cleans up
            # `remove_chunk_ids`/`mark_document_status` staleness, not
            # reachability of new content) -- so a stale-but-loaded index is
            # served as-is, and a fresh rebuild is kicked off in the
            # background rather than blocking this or any other in-flight
            # request on it.
            self._kick_off_background_rebuild(generation)
            return True
        # Cold start (`self._index` still `None` for this process): unlike
        # the already-loaded case above, there is nothing to serve yet, so
        # blocking IS correct here -- but ONLY the first caller should pay
        # for it. Confirmed live (2026-09-24): a single `/chat` request's
        # OWN several concurrent query-expansion legs (`LegalRetriever.
        # _search_legs` runs each variant's `vector_store.search()`
        # concurrently), each independently reaching this method before the
        # first had finished, each started ITS OWN full Mongo fetch + FAISS
        # build with no coordination at all -- multiple simultaneous ~75s
        # rebuilds racing each other. A non-blocking `locked()` check
        # (rather than every concurrent caller queuing behind the lock)
        # sends every caller but the first straight to the brute-force
        # fallback for just this one request instead -- self-healing the
        # moment the first build finishes and swaps `self._index` in.
        if self._lock.locked():
            return False
        async with self._lock:
            if self._index is not None:
                self._kb_generation = generation
                return True
            if await asyncio.to_thread(self._load_from_disk):
                self._kb_generation = generation
                return True
            built = await self._fetch_and_build()
            self._swap_in(built)
            self._kb_generation = generation
        if built is not None:
            await asyncio.to_thread(self._save_to_disk)
        return self._index is not None

    async def rebuild(self) -> int:
        """Explicit full rebuild, bypassing any disk cache -- for an
        admin-triggered resync if incremental adds and the real corpus ever
        drift (mirrors `BM25Index.rebuild`). Same no-lock-during-the-
        expensive-part shape as the background path below."""
        built = await self._fetch_and_build()
        async with self._lock:
            self._swap_in(built)
        if built is not None:
            await asyncio.to_thread(self._save_to_disk)
        return self.chunk_count

    def _kick_off_background_rebuild(self, generation: bytes) -> None:
        """Fire-and-forget rebuild, de-duplicated so a burst of concurrent
        requests observing the same stale generation starts at most ONE
        rebuild rather than one per request. The expensive Mongo fetch +
        FAISS build (`_fetch_and_build`, seconds to ~90s on this corpus)
        runs with NO lock held at all -- `search()`/`add_or_update_chunks()`
        keep reading/writing the OLD `self._index` throughout, completely
        unaffected. Only the final swap (`_swap_in`, plain attribute
        assignment, effectively instant) briefly holds `self._lock`.
        """
        if self._background_rebuild_task is not None and not self._background_rebuild_task.done():
            return

        async def _run() -> None:
            try:
                built = await self._fetch_and_build()
                async with self._lock:
                    self._swap_in(built)
                    self._kb_generation = generation
                if built is not None:
                    await asyncio.to_thread(self._save_to_disk)
            except Exception as exc:  # noqa: BLE001 - a failed background rebuild must not crash the process; the stale index keeps serving until the next attempt
                log.warning("local_ann_background_rebuild_failed", error=str(exc))

        self._background_rebuild_task = asyncio.ensure_future(_run())

    async def add_or_update_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Incremental hook, called from `MongoVectorStore.upsert_chunks`
        alongside the BM25 hook -- so newly-ingested content is searchable
        immediately rather than waiting for the next `kb-generation` bump.

        A chunk id the index has never seen is appended (new HNSW node); a
        chunk id already indexed only has its cached metadata refreshed in
        place (the embedding itself essentially never changes for an
        existing id in this pipeline -- `BM25Index.add_or_update_chunks`'s
        own comment notes a re-index mints fresh `uuid4` ids rather than
        reusing one). HNSW has no cheap in-place vector update; a genuine
        embedding change for an existing id is picked up at the next full
        rebuild (`kb-generation` bump or `rebuild()`), same as BM25's own
        documented limitation for `remove_chunk_ids` in between rebuilds.
        """
        if faiss is None or not chunks:
            return
        if self._index is None and self._lock.locked():
            # Another caller (e.g. a concurrent `/chat` request's own
            # cold-start build, see `ensure_current_generation`) is already
            # doing the one-time first build, which fetches fresh from
            # Mongo -- these chunks (already written to Mongo by the caller
            # before this hook runs) will be picked up by it. Starting a
            # SECOND redundant full rebuild here would only add contention.
            return
        if self._index is None and not await asyncio.to_thread(self._load_from_disk):
            # Nothing loaded yet -- a fresh rebuild from Mongo already picks
            # these chunks up (upsert_chunks writes to Mongo first). Runs
            # the expensive fetch+build unlocked, same reasoning as
            # `ensure_current_generation`/`_kick_off_background_rebuild`.
            built = await self._fetch_and_build()
            async with self._lock:
                self._swap_in(built)
            if built is not None:
                await asyncio.to_thread(self._save_to_disk)
            return
        changed = False
        async with self._lock:
            existing = {chunk_id: pos for pos, chunk_id in enumerate(self._chunk_ids)}
            new_vectors: list[list[float]] = []
            new_ids: list[str] = []
            new_metadatas: list[dict[str, Any]] = []
            for chunk in chunks:
                pos = existing.get(chunk.chunk_id)
                if pos is not None:
                    self._metadatas[pos] = chunk.metadata
                    continue
                if not chunk.embedding or len(chunk.embedding) != self._dim:
                    continue
                new_vectors.append(chunk.embedding)
                new_ids.append(chunk.chunk_id)
                new_metadatas.append(chunk.metadata)
            if new_vectors:
                self._add_vectors(new_vectors, new_ids, new_metadatas)
                changed = True
        # `index.add()` on an HNSW graph of this size is a few milliseconds
        # for a handful of new vectors (unlike a full rebuild) -- held
        # inside the lock above as plain sync code, no `asyncio.to_thread`,
        # since the point of moving work off-lock is avoiding SECONDS-long
        # holds, not eliminating single-digit-millisecond ones. Only the
        # slower disk persist happens outside the lock.
        if changed:
            await asyncio.to_thread(self._save_to_disk)

    async def remove_chunk_ids(self, chunk_ids: set[str]) -> int:
        """Marks ids invisible to future searches (see module docstring's
        note on HNSW having no cheap removal) -- their vectors stay in the
        graph, wasting a little memory/search time until the next full
        rebuild, exactly like an over-fetch-and-filter miss rather than a
        correctness gap."""
        if not chunk_ids:
            return 0
        async with self._lock:
            if self._index is None:
                return 0
            removed = 0
            for pos, chunk_id in enumerate(self._chunk_ids):
                if chunk_id in chunk_ids and pos not in self._removed_positions:
                    self._removed_positions.add(pos)
                    removed += 1
        if removed:
            await asyncio.to_thread(self._save_to_disk)
        return removed

    async def mark_document_status(self, chunk_ids: set[str], status: str) -> int:
        """Mirrors `BM25Index.mark_document_status` -- patches the cached
        `document_status` in place (called from `MongoVectorStore.
        activate_version`) without touching the FAISS graph itself."""
        if not chunk_ids:
            return 0
        async with self._lock:
            if self._index is None:
                return 0
            updated = 0
            for pos, chunk_id in enumerate(self._chunk_ids):
                if chunk_id in chunk_ids:
                    self._metadatas[pos] = {**self._metadatas[pos], "document_status": status}
                    updated += 1
        if updated:
            await asyncio.to_thread(self._save_to_disk)
        return updated

    async def search(
        self, query_embedding: list[float], limit: int, filters: dict[str, Any]
    ) -> list[RetrievedChunk] | None:
        """Approximate cosine search over the HNSW graph. Returns `None`
        (never `[]` for "unusable") when the index cannot answer this query
        at all, so `MongoVectorStore._local_cosine_leg` can fall back to its
        exact brute-force scan rather than silently treating "ANN
        unavailable" as "zero real matches" for the user.
        """
        dim = self._dim
        if self._index is None or dim == 0 or len(query_embedding) != dim:
            return None
        query = np.asarray([query_embedding], dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        query = query / norm
        search_started = time.perf_counter()
        # Holds `self._lock` for the actual FAISS calls below (not just a
        # snapshot read): `index.search()` is not guaranteed safe to run
        # concurrently with `index.add()` (`add_or_update_chunks`/a
        # rebuild's `_swap_in`) on the same underlying FAISS object.
        # Concurrent SEARCHES do serialize against each other too as a
        # result, but each one is single-digit-to-low-double-digit
        # milliseconds on this corpus (confirmed live via `local_ann_search_
        # timing`) -- negligible next to the seconds-to-90s rebuild cost
        # this lock is NOT held for (see `_fetch_and_build`/`_swap_in`).
        # Progressive over-fetch: the cheap common case (broad/no filter)
        # succeeds on the first, narrow attempt; a genuinely narrow filter
        # (e.g. one session's own private uploads among a 60k-chunk shared
        # corpus) widens up to the full graph rather than returning a
        # false-empty result the way a single fixed over-fetch window would.
        results: list[RetrievedChunk] = []
        async with self._lock:
            index = self._index
            chunk_ids = self._chunk_ids
            metadatas = self._metadatas
            removed = self._removed_positions
            if index is None or not chunk_ids:
                return None
            total = len(chunk_ids)
            for attempt_k in (
                min(total, max(limit * _OVER_FETCH_MULTIPLIER, limit)),
                min(total, limit * _OVER_FETCH_MULTIPLIER * 4),
                total,
            ):
                scores, ids = await asyncio.to_thread(self._search_sync, index, query, attempt_k)
                results = []
                for score, position in zip(scores[0], ids[0]):
                    if position < 0 or position in removed:
                        continue
                    metadata = metadatas[position]
                    if not matches_filters(metadata, filters):
                        continue
                    results.append(
                        RetrievedChunk(
                            chunk_id=chunk_ids[position], text="", score=float(max(0.0, score)), metadata=metadata
                        )
                    )
                if len(results) >= limit or attempt_k >= total:
                    break
        search_ms = (time.perf_counter() - search_started) * 1000
        winners = results[:limit]
        hydrate_started = time.perf_counter()
        await self._hydrate_text(winners)
        log.info(
            "local_ann_search_timing",
            corpus_size=total,
            winner_count=len(winners),
            search_ms=round(search_ms, 2),
            hydrate_ms=round((time.perf_counter() - hydrate_started) * 1000, 2),
        )
        metrics.increment("local_ann_search_used")
        return winners

    @staticmethod
    def _search_sync(index: Any, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        index.hnsw.efSearch = max(_HNSW_EF_SEARCH_FLOOR, k * 2)
        return index.search(query, k)

    async def _hydrate_text(self, chunks: list[RetrievedChunk]) -> None:
        """The FAISS graph and its metadata cache never hold chunk `text`
        (only the embedding + metadata, to keep the resident index small) --
        fetched here, once, only for the handful of winning ids a query
        actually needs, via an indexed `_id`-`$in` lookup. This is the whole
        reason this index is cheaper than the brute-force scan it replaces:
        that leg fetched full text + metadata + embedding for every one of
        up to `local_vector_scan_limit` candidates on every query; this leg
        fetches full documents for only the (typically tens of) winners.
        """
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        texts: dict[str, str] = {}
        try:
            collection = mongodb.db[EMBEDDINGS_METADATA]
            async for item in collection.find({"_id": {"$in": ids}}, {"text": 1}):
                texts[str(item["_id"])] = item.get("text", "")
        # defensive: a Mongo hiccup during hydration must not crash retrieval
        # -- the caller already committed to these `chunk_id`s/scores/
        # metadata from the ANN search itself; losing only the display text
        # degrades the answer, it does not make it wrong. RuntimeError also
        # covers `mongodb.db` raising "MongoDB client is not connected." in a
        # unit test that seeds this index directly without a live connection.
        except (PyMongoError, RuntimeError) as exc:  # noqa: BLE001 - see comment above
            log.warning("local_ann_hydrate_text_failed", error=str(exc))
        for chunk in chunks:
            chunk.text = texts.get(chunk.chunk_id, "")

    def _add_vectors(self, vectors: list[list[float]], ids: list[str], metadatas: list[dict[str, Any]]) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms
        self._index.add(matrix)
        self._chunk_ids.extend(ids)
        self._metadatas.extend(metadatas)

    async def _fetch_and_build(self) -> "_BuiltIndex | None":
        """No lock held at all -- both the Mongo fetch (network I/O) and the
        FAISS build (`asyncio.to_thread`, CPU-bound, ~30-90s on this corpus)
        operate on entirely local variables, never touching `self._index`.
        Concurrent `search()`/`add_or_update_chunks()` calls keep reading/
        writing the OLD index throughout, completely unaffected -- see
        `_swap_in` for the only step that actually touches shared state.
        Streams id + embedding only (never `text`, see `_hydrate_text`) --
        for a 55,000-chunk corpus at 1024 dims this is ~220MB of vector data
        over the wire once, versus the same payload PLUS full chunk text on
        every single query the brute-force leg it replaces used to pay.
        Returns `None` (not a `_BuiltIndex` with an empty index) when the
        corpus itself is empty, so `_swap_in` can tell "rebuild found
        nothing" apart from "rebuild wasn't attempted" in its own caller.
        """
        if faiss is None:
            return None
        collection = mongodb.db[EMBEDDINGS_METADATA]
        chunk_ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        dim = settings.embedding_dimensions
        started = time.perf_counter()
        async for item in collection.find({}, {"metadata": 1, "embedding": 1}):
            embedding = item.get("embedding") or []
            if len(embedding) != dim:
                continue
            chunk_ids.append(str(item["_id"]))
            metadatas.append(item.get("metadata", {}))
            vectors.append(embedding)
        if not vectors:
            log.warning("local_ann_index_rebuild_found_no_vectors")
            return None
        index = await asyncio.to_thread(self._build_index_sync, vectors, dim)
        log.info(
            "local_ann_index_built",
            chunk_count=len(chunk_ids),
            build_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return _BuiltIndex(index=index, chunk_ids=chunk_ids, metadatas=metadatas, dim=dim)

    def _swap_in(self, built: "_BuiltIndex | None") -> None:
        """Caller must already hold `self._lock`. Plain attribute
        assignment only -- no `await` anywhere in this method -- so it is
        atomic with respect to every other coroutine on this event loop:
        nothing can observe a torn mix of old/new `self._index`/`self.
        _chunk_ids`/`self._metadatas`. `built=None` means `_fetch_and_build`
        found an empty corpus; clears the index rather than leaving a stale
        one silently answering queries against a corpus that no longer
        exists (matches the pre-split behavior for this case).
        """
        if built is None:
            self._index = None
            self._chunk_ids, self._metadatas, self._removed_positions = [], [], set()
            return
        self._index = built.index
        self._chunk_ids = built.chunk_ids
        self._metadatas = built.metadatas
        self._removed_positions = set()
        self._dim = built.dim
        metrics.increment("local_ann_index_rebuilt")

    @staticmethod
    def _build_index_sync(vectors: list[list[float]], dim: int) -> Any:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms
        index = faiss.IndexHNSWFlat(dim, _HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = _HNSW_EF_CONSTRUCTION
        index.add(matrix)
        return index

    def _load_from_disk(self) -> bool:
        if faiss is None:
            return False
        index_path = self.cache_dir / "index.faiss"
        meta_path = self.cache_dir / "meta.pkl"
        if not index_path.exists() or not meta_path.exists():
            return False
        try:
            index = faiss.read_index(str(index_path))
            with meta_path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, RuntimeError, pickle.PickleError) as exc:
            log.warning("local_ann_index_load_failed", error=str(exc))
            return False
        self._index = index
        self._chunk_ids = payload.get("chunk_ids", [])
        self._metadatas = payload.get("metadatas", [])
        self._removed_positions = payload.get("removed_positions", set())
        self._dim = payload.get("dim", settings.embedding_dimensions)
        return True

    def _save_to_disk(self) -> None:
        if self._index is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.cache_dir / "index.faiss"))
            with (self.cache_dir / "meta.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "chunk_ids": self._chunk_ids,
                        "metadatas": self._metadatas,
                        "removed_positions": self._removed_positions,
                        "dim": self._dim,
                    },
                    handle,
                )
        except OSError as exc:
            log.warning("local_ann_index_persist_failed", error=str(exc))


local_ann_index = LocalAnnIndex()
