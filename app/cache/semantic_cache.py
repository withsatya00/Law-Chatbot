import hashlib
from typing import Any

from app.cache.redis_client import CACHE_UNAVAILABLE_ERRORS, redis_client
from app.mlops.registry import RuntimeVersionRegistry

RETRIEVAL_CACHE_POLICY_VERSION = "retrieval-policy-v3"

# Phase 1 "Jurisdiction-Aware Knowledge Base" gap 3: the SAME redis counter
# `app.cache.response_cache.ResponseCache.bump_generation` increments
# (`f"{CACHE_KEY_PREFIX}:kb-generation"`, `CACHE_KEY_PREFIX == "legal"`) --
# not a second, independent counter. Every point that already calls
# `response_cache.bump_generation()` when the Knowledge Base's content OR
# reviewed status changes (indexing, the jurisdiction backfill, the
# jurisdiction-correction endpoint) must invalidate `retrieval_cache` (this
# module's raw retrieval-results cache) in the exact same instant, or a
# stale retrieval cached from BEFORE a document was withdrawn/marked
# needs_review could keep serving that document's chunks for the rest of its
# TTL even after the final-answer cache was correctly invalidated. Sharing
# one counter, read into the key on every `get`/`set`, makes that
# structurally impossible rather than a second invalidation call to
# remember at each of those sites.
_KB_GENERATION_KEY = "legal:kb-generation"


class SemanticCache:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.registry = RuntimeVersionRegistry()

    async def _generation(self) -> int:
        try:
            raw = await redis_client.client.get(_KB_GENERATION_KEY)
        except CACHE_UNAVAILABLE_ERRORS:
            # Redis unavailable: generation 0 is what every other lookup in
            # this degraded state already gets (same failure mode `get`/`set`
            # below fall back to) -- never a reason to raise out of a cache path.
            return 0
        return int(raw) if raw else 0

    async def key(self, text: str, language: str | None = None) -> str:
        normalized = " ".join(text.lower().split())
        digest = hashlib.sha256(f"{language or 'auto'}:{normalized}".encode()).hexdigest()
        generation = await self._generation()
        return (
            f"{self.prefix}:{RETRIEVAL_CACHE_POLICY_VERSION}:"
            f"{self.registry.cache_namespace()}:{generation}:{digest}"
        )

    async def get(self, text: str, language: str | None = None) -> Any | None:
        try:
            return await redis_client.get_json(await self.key(text, language))
        # `RuntimeError` alone only covers "connect() was never called"; a
        # real Redis outage raises `RedisError`/`OSError` instead, which used
        # to propagate out of here and fail the caching caller entirely --
        # inconsistent with `_generation()` just above, which already
        # degrades to 0 on the exact same class of failure.
        except CACHE_UNAVAILABLE_ERRORS:
            return None

    async def set(self, text: str, value: Any, language: str | None = None, ttl_seconds: int = 3600) -> None:
        try:
            await redis_client.set_json(await self.key(text, language), value, ttl_seconds)
        except CACHE_UNAVAILABLE_ERRORS:
            pass


response_cache = SemanticCache("response-cache")
retrieval_cache = SemanticCache("retrieval-cache")
prompt_cache = SemanticCache("prompt-cache")
