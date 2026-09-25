import asyncio
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, ClassVar

import structlog

from app.cache.redis_client import CACHE_UNAVAILABLE_ERRORS, redis_client
from app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = structlog.get_logger(__name__)


class EmbeddingProvider:
    """Async-safe wrapper around a sentence-transformers model.

    Models are heavy to load, so instances are cached per model name at the class
    level and shared across every ``EmbeddingProvider()`` constructed in the process.
    Inference is synchronous/CPU-or-GPU bound, so it must not block the event loop --
    but it also must NOT run on asyncio's default thread pool (`asyncio.to_thread`),
    which happily hands out a separate worker thread to every concurrent caller.

    QA session 9: confirmed live that N concurrent chat/upload requests each
    calling `embed_batch` independently made this stage 100-1000x slower under
    concurrency than alone (a solo call: ~350ms; 3 concurrent calls: ~320
    SECONDS each) -- consistent with multiple Python threads all driving
    inference calls into the SAME CUDA-backed `SentenceTransformer` instance at
    once, which contends for the one GPU context rather than parallelizing
    (`chat_service.py` even has a standing comment from an earlier session
    describing an 8-minute stall attributed to exactly this: "GPU/Mongo
    contention from a concurrent ingestion job"). Routed through
    `_ENCODE_EXECUTOR`, a SINGLE dedicated worker thread shared by every
    `EmbeddingProvider` instance, instead: concurrent embed calls now queue and
    run one at a time on that one thread rather than fighting each other on
    the GPU from several threads simultaneously. Still fully non-blocking for
    the event loop (other coroutines proceed normally while a call waits its
    turn); it trades true multi-threaded concurrency -- which this model
    cannot actually exploit safely/efficiently anyway -- for a queue, which
    measured live as dramatically faster in aggregate (five sequential
    ~0.3-1s encodes beat five GPU-thrashing concurrent ones taking minutes
    each).
    """

    _model_cache: ClassVar[dict[str, "SentenceTransformer"]] = {}
    _load_lock: ClassVar[threading.Lock] = threading.Lock()
    _cache_semaphore: ClassVar[asyncio.Semaphore] = asyncio.Semaphore(20)
    # One worker, shared process-wide: every concurrent `embed_batch` call
    # (across every `EmbeddingProvider()` instance and every request) queues
    # onto this SAME thread rather than each getting its own from asyncio's
    # default pool. Named for visibility in thread dumps/profilers.
    _ENCODE_EXECUTOR: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="embedding-encode"
    )

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device or None

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, reusing cached vectors and computing only the misses."""
        if not texts:
            return []

        cache_keys = [self._cache_key(text) for text in texts]
        resolved: list[list[float] | None] = list(await asyncio.gather(*(self._cache_get(key) for key in cache_keys)))

        pending_indexes = [index for index, value in enumerate(resolved) if value is None]
        if pending_indexes:
            pending_texts = [texts[index] for index in pending_indexes]
            loop = asyncio.get_running_loop()
            computed = await loop.run_in_executor(self._ENCODE_EXECUTOR, self._encode_sync, pending_texts)
            await asyncio.gather(
                *(
                    self._cache_set(cache_keys[index], vector)
                    for index, vector in zip(pending_indexes, computed, strict=True)
                )
            )
            for index, vector in zip(pending_indexes, computed, strict=True):
                resolved[index] = vector

        return [vector for vector in resolved if vector is not None]

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=settings.indexing_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors.tolist()]

    def _get_model(self) -> "SentenceTransformer":
        cached = EmbeddingProvider._model_cache.get(self.model_name)
        if cached is not None:
            return cached
        with EmbeddingProvider._load_lock:
            cached = EmbeddingProvider._model_cache.get(self.model_name)
            if cached is not None:
                return cached
            from sentence_transformers import SentenceTransformer

            device = self.device or self._detect_device()
            log.info("embedding_model_loading", model=self.model_name, device=device)
            model = SentenceTransformer(self.model_name, device=device)
            EmbeddingProvider._model_cache[self.model_name] = model
            log.info("embedding_model_loaded", model=self.model_name, device=device)
            return model

    def _detect_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embedding:{self.model_name}:{settings.embedding_version}:{digest}"

    async def _cache_get(self, key: str) -> list[float] | None:
        async with self._cache_semaphore:
            try:
                cached = await redis_client.get_json(key)
            # `RuntimeError` alone only covers "connect() was never called";
            # a real Redis outage (the common case in production) raises
            # `RedisError`/`OSError`, which used to propagate straight out of
            # `embed_batch` and fail the whole indexing/retrieval call --
            # confirmed live: a stopped local Redis broke every indexing
            # attempt with "Error 22 connecting to localhost:6379" instead of
            # just skipping the cache. Matches `CACHE_UNAVAILABLE_ERRORS`'s own
            # contract, which every other cache call site in this codebase follows.
            except CACHE_UNAVAILABLE_ERRORS:
                return None
        return list(cached) if cached is not None else None

    async def _cache_set(self, key: str, vector: list[float]) -> None:
        async with self._cache_semaphore:
            try:
                await redis_client.set_json(key, vector, ttl_seconds=30 * 24 * 3600)
            except CACHE_UNAVAILABLE_ERRORS:
                pass
