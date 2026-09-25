import hashlib
import re
import time
from datetime import UTC, datetime
from typing import Any

import structlog

from app.cache.redis_client import CACHE_UNAVAILABLE_ERRORS, redis_client
from app.intent.detector import parse_section_lookup
from app.language.normalizer import QueryNormalizer
from app.mlops.registry import RuntimeVersionRegistry
from app.observability.metrics import metrics
from app.rag.embeddings import EmbeddingProvider

log = structlog.get_logger(__name__)

CACHE_KEY_PREFIX = "legal"
# Increment when answer/retrieval policy changes in a way that makes an old
# generated response unsafe to replay.  This is deliberately code-owned (not
# an operator env var): deploying the fix must invalidate the bad answer even
# when PROMPT_VERSION was left at its default.
ANSWER_CACHE_POLICY_VERSION = "answer-policy-v2"
# The embedding call is a different tier from Redis: it reaches a
# sentence-transformers model that may not be downloaded, may not fit in
# memory, and on first use pulls ~2.3 GB over the network. A semantic-cache
# lookup that cannot embed is simply a cache miss -- the caller falls through
# to a real LLM call -- so it must never propagate, but it is still narrowed
# past `Exception` so a `TypeError`/`AttributeError` in this module's own key
# handling is not laundered into "cache miss".
#   OSError     -- model download / local model directory failures.
#   RuntimeError -- torch device and out-of-memory failures.
#   ValueError  -- malformed input rejected by the tokenizer.
#   ImportError -- sentence-transformers or torch not installed on this host.
EMBEDDING_UNAVAILABLE_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    RuntimeError,
    ValueError,
    ImportError,
)
HIGH_CONFIDENCE_THRESHOLD = 0.6
# Calibrated empirically against bge-m3 on real legal questions, not guessed: genuine
# paraphrases of the same question scored 0.62-0.98, but same-intent-bucket questions
# that deserve a genuinely different answer (e.g. "How do I file an FIR?" vs "What is
# an FIR?") scored up to 0.73. A lower threshold would confidently serve the wrong
# legal answer to a different question; 0.90 sits above that false-positive ceiling
# at the cost of missing a few very terse genuine paraphrases (those just fall through
# to a fresh LLM call, which is the safer failure mode here).
SIMILARITY_THRESHOLD = 0.90
BUCKET_MAX_ENTRIES = 40
DEFINITIONAL_TTL_SECONDS = 7 * 24 * 3600
FAQ_TTL_SECONDS = 24 * 3600
DEFINITIONAL_PATTERN = re.compile(
    r"^\s*(what is|what does|what are|define|explain|meaning of|kya hota hai|kya hai|kya matlab)\b", re.IGNORECASE
)

# $ per 1K tokens (blended input+output), illustrative only -- self-hosted providers
# (ollama) run at $0 API cost. Update to match real provider pricing if precision matters.
COST_PER_1K_TOKENS: dict[str, float] = {
    "openai": 0.005,
    "gemini": 0.0015,
    "claude": 0.006,
    "deepseek": 0.001,
    "groq": 0.0008,
    "ollama": 0.0,
    "openrouter": 0.0,  # z-ai/glm-5.3-free (the configured default) is a free-tier model
}


class ResponseCache:
    """Redis-backed semantic cache for full legal-answer responses.

    Two-tier lookup per request: an exact normalized-key hit (single Redis GET,
    typically single-digit ms) falls back to a bounded brute-force cosine search
    over recent entries in the same language+intent bucket (catches paraphrases
    and cross-language duplicates like "What is FIR?" / "FIR kya hota hai?").
    Keys are namespaced by model/prompt/embedding versions (`RuntimeVersionRegistry`)
    and a knowledge-base generation counter, so swapping models or reindexing the
    knowledge base automatically stops serving stale entries without a scan/delete.
    """

    def __init__(self) -> None:
        self.registry = RuntimeVersionRegistry()
        self.embeddings = EmbeddingProvider()
        self.normalizer = QueryNormalizer()

    def normalize(self, question: str) -> str:
        return self.normalizer.normalize(question).lower().strip()

    def ttl_for(self, question: str) -> int:
        return DEFINITIONAL_TTL_SECONDS if DEFINITIONAL_PATTERN.search(question.strip()) else FAQ_TTL_SECONDS

    def is_cacheable(self, confidence: float, has_sources: bool, llm_error: str | None) -> bool:
        return confidence >= HIGH_CONFIDENCE_THRESHOLD and has_sources and not llm_error

    async def lookup(
        self, question: str, language: str, intent: str, jurisdiction_key: str | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Returns (cached_entry, hit_type) where hit_type is 'exact' | 'semantic' | 'miss'.

        `jurisdiction_key` (Jurisdiction Routing, Phase 2) -- built by
        `ChatService` from the resolved matter's State(s)/locality/as-of
        date -- scopes BOTH the bucket and the entry key, the same way
        `intent`'s own `SECTION_LOOKUP` disambiguation already does (see
        `_keys`). Two callers asking the identical question under different
        matter contexts land in different buckets entirely, so the semantic
        (embedding-similarity) leg can never match across a jurisdiction
        boundary -- it only ever compares candidates already inside the same
        bucket. `None` (every pre-existing caller) is unchanged behavior.
        """
        started = time.perf_counter()
        normalized = self.normalize(question)
        bucket_key, entry_key = await self._keys(language, intent, normalized, jurisdiction_key)

        exact = await self._get(entry_key)
        if exact is not None:
            await self._record_hit(time.perf_counter() - started, exact)
            return exact, "exact"

        candidate_keys = await self._bucket_members(bucket_key)
        if candidate_keys:
            best_entry, best_score = await self._best_semantic_match(normalized, candidate_keys)
            if best_entry is not None and best_score >= SIMILARITY_THRESHOLD:
                await self._record_hit(time.perf_counter() - started, best_entry)
                return best_entry, "semantic"

        await self._record_miss(time.perf_counter() - started)
        return None, "miss"

    async def store(
        self,
        question: str,
        language: str,
        intent: str,
        response: dict[str, Any],
        jurisdiction_key: str | None = None,
    ) -> None:
        normalized = self.normalize(question)
        bucket_key, entry_key = await self._keys(language, intent, normalized, jurisdiction_key)
        try:
            embedding = await self.embeddings.embed_text(normalized)
        except EMBEDDING_UNAVAILABLE_ERRORS as exc:
            log.warning("response_cache_embedding_failed", error=str(exc))
            return
        ttl = self.ttl_for(question)
        entry = {
            "original_query": question,
            "normalized_query": normalized,
            "language": language,
            "intent": intent,
            "response": response,
            "embedding": embedding,
            "cached_at": datetime.now(UTC).isoformat(),
        }
        await self._set(entry_key, entry, ttl)
        await self._add_to_bucket(bucket_key, entry_key, ttl)

    async def purge_entry(self, question: str, language: str, intent: str) -> bool:
        """Removes one specific cached response (e.g. after spotting a bad
        answer) without invalidating the rest of the cache."""
        normalized = self.normalize(question)
        bucket_key, entry_key = await self._keys(language, intent, normalized)
        try:
            removed = await redis_client.client.delete(entry_key)
            await redis_client.client.lrem(bucket_key, 0, entry_key)
            return bool(removed)
        except CACHE_UNAVAILABLE_ERRORS as exc:
            log.warning("response_cache_purge_failed", error=str(exc))
            return False

    async def bump_generation(self, *, strict: bool = False) -> None:
        """Invalidates every previously-cached response by moving all future
        cache keys to a new generation; old entries simply age out via TTL."""
        try:
            await redis_client.client.incr(f"{CACHE_KEY_PREFIX}:kb-generation")
        except CACHE_UNAVAILABLE_ERRORS as exc:
            log.warning("response_cache_generation_bump_failed", error=str(exc))
            if strict:
                raise

    async def stats(self) -> dict[str, Any]:
        try:
            hits = int((await redis_client.client.get(f"{CACHE_KEY_PREFIX}:stats:hits")) or 0)
            misses = int((await redis_client.client.get(f"{CACHE_KEY_PREFIX}:stats:misses")) or 0)
            tokens_saved = float((await redis_client.client.get(f"{CACHE_KEY_PREFIX}:stats:tokens_saved")) or 0)
            cost_saved = float((await redis_client.client.get(f"{CACHE_KEY_PREFIX}:stats:cost_saved")) or 0)
        except CACHE_UNAVAILABLE_ERRORS as exc:
            log.warning("response_cache_stats_read_failed", error=str(exc))
            hits = misses = 0
            tokens_saved = cost_saved = 0.0
        total = hits + misses
        timing = metrics.snapshot().get("timings", {}).get("response_cache_lookup_latency_ms", {})
        return {
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_ratio": round(hits / total, 4) if total else 0.0,
            "avg_lookup_latency_ms": timing.get("avg_ms"),
            "estimated_tokens_saved": tokens_saved,
            "estimated_cost_saved_usd": round(cost_saved, 4),
        }

    async def _best_semantic_match(
        self, normalized_query: str, candidate_keys: list[str]
    ) -> tuple[dict[str, Any] | None, float]:
        try:
            query_embedding = await self.embeddings.embed_text(normalized_query)
        except EMBEDDING_UNAVAILABLE_ERRORS as exc:
            log.warning("response_cache_query_embedding_failed", error=str(exc))
            return None, 0.0
        best_entry: dict[str, Any] | None = None
        best_score = 0.0
        for candidate_key in candidate_keys:
            candidate = await self._get(candidate_key)
            if not candidate or not candidate.get("embedding"):
                continue
            score = self._cosine(query_embedding, candidate["embedding"])
            if score > best_score:
                best_score, best_entry = score, candidate
        return best_entry, best_score

    async def _keys(
        self, language: str, intent: str, normalized_query: str, jurisdiction_key: str | None = None,
    ) -> tuple[str, str]:
        namespace = self.registry.cache_namespace()
        generation = await self._generation()
        lang = (language or "auto").lower()
        intent_slug = self._slug(intent or "general")
        # Jurisdiction Routing (Phase 2): same disambiguation mechanism as
        # SECTION_LOOKUP's own section-number suffix just below -- folded
        # into `intent_slug` so it affects BOTH the bucket and entry key in
        # one place.
        if jurisdiction_key:
            intent_slug = f"{intent_slug}:jx-{self._slug(jurisdiction_key)}"
        # Root cause of the "Section 420 IPC" -> BNSS Section 420 bug: every
        # SECTION_LOOKUP query shares ONE bucket regardless of which section
        # number it names, so the semantic leg below only has to clear
        # SIMILARITY_THRESHOLD -- and it does: "section 420" vs "section 420
        # ipc" measures 0.9048 cosine on the real bge-m3 model, just over
        # 0.90, even though `parse_section_lookup`'s IPC->BNS crosswalk means
        # they ask for two entirely different provisions (BNSS 420 vs BNS
        # 318). Narrowing the bucket to the query's OWN parsed section number
        # keeps semantic matching scoped to genuine paraphrases of the SAME
        # section ("BNS 318" / "IPC 420" / "Section 420 IPC" all correctly
        # share one bucket, since they all resolve to BNS 318) while making
        # different sections structurally unable to collide, without
        # touching SIMILARITY_THRESHOLD's calibration for every other
        # intent.
        if intent == "SECTION_LOOKUP":
            section_number = parse_section_lookup(normalized_query)
            if section_number:
                intent_slug = f"{intent_slug}-{section_number.lower()}"
        bucket = (
            f"{CACHE_KEY_PREFIX}:bucket:{ANSWER_CACHE_POLICY_VERSION}:"
            f"{namespace}:{generation}:{lang}:{intent_slug}"
        )
        digest = hashlib.sha256(normalized_query.encode()).hexdigest()[:24]
        entry = (
            f"{CACHE_KEY_PREFIX}:entry:{ANSWER_CACHE_POLICY_VERSION}:"
            f"{namespace}:{generation}:{lang}:{intent_slug}:{digest}"
        )
        return bucket, entry

    async def _generation(self) -> int:
        try:
            raw = await redis_client.client.get(f"{CACHE_KEY_PREFIX}:kb-generation")
        except CACHE_UNAVAILABLE_ERRORS as exc:
            # Generation 0 means "serve from the ungenerationed keyspace",
            # which is safe, but it also means a KB re-index would not
            # invalidate anything -- worth knowing about.
            log.warning("response_cache_generation_read_failed", error=str(exc))
            return 0
        return int(raw) if raw else 0

    def _slug(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "general"

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        return dot / (left_norm * right_norm or 1.0)

    async def _get(self, key: str) -> dict[str, Any] | None:
        try:
            return await redis_client.get_json(key)
        except CACHE_UNAVAILABLE_ERRORS as exc:
            log.warning("response_cache_get_failed", error=str(exc))
            return None

    async def _set(self, key: str, value: dict[str, Any], ttl: int) -> None:
        try:
            await redis_client.set_json(key, value, ttl)
        except CACHE_UNAVAILABLE_ERRORS as exc:
            log.warning("response_cache_set_failed", error=str(exc))

    async def _bucket_members(self, bucket_key: str) -> list[str]:
        try:
            raw = await redis_client.client.lrange(bucket_key, 0, BUCKET_MAX_ENTRIES - 1)
        except CACHE_UNAVAILABLE_ERRORS as exc:
            log.warning("response_cache_bucket_read_failed", error=str(exc))
            return []
        return [item.decode() if isinstance(item, bytes) else item for item in raw]

    async def _add_to_bucket(self, bucket_key: str, entry_key: str, ttl: int) -> None:
        try:
            pipe = redis_client.client.pipeline()
            pipe.lpush(bucket_key, entry_key)
            pipe.ltrim(bucket_key, 0, BUCKET_MAX_ENTRIES - 1)
            pipe.expire(bucket_key, ttl)
            await pipe.execute()
        except CACHE_UNAVAILABLE_ERRORS as exc:
            log.warning("response_cache_bucket_update_failed", error=str(exc))

    async def _record_hit(self, elapsed_seconds: float, entry: dict[str, Any]) -> None:
        elapsed_ms = elapsed_seconds * 1000
        metrics.increment("response_cache_hit")
        metrics.observe_ms("response_cache_lookup_latency_ms", elapsed_ms)
        tokens = self._tokens(entry)
        cost = self._cost(entry, tokens)
        try:
            pipe = redis_client.client.pipeline()
            pipe.incr(f"{CACHE_KEY_PREFIX}:stats:hits")
            if tokens:
                pipe.incrbyfloat(f"{CACHE_KEY_PREFIX}:stats:tokens_saved", tokens)
            if cost:
                pipe.incrbyfloat(f"{CACHE_KEY_PREFIX}:stats:cost_saved", cost)
            await pipe.execute()
        except CACHE_UNAVAILABLE_ERRORS as exc:
            log.warning("response_cache_stats_update_failed", error=str(exc))

    async def _record_miss(self, elapsed_seconds: float) -> None:
        metrics.increment("response_cache_miss")
        metrics.observe_ms("response_cache_lookup_latency_ms", elapsed_seconds * 1000)
        try:
            await redis_client.client.incr(f"{CACHE_KEY_PREFIX}:stats:misses")
        except CACHE_UNAVAILABLE_ERRORS as exc:
            log.warning("response_cache_stats_update_failed", error=str(exc))

    def _tokens(self, entry: dict[str, Any]) -> int:
        response = entry.get("response") or {}
        return int(response.get("prompt_tokens") or 0) + int(response.get("completion_tokens") or 0)

    def _cost(self, entry: dict[str, Any], tokens: int) -> float:
        response = entry.get("response") or {}
        provider = (response.get("llm_provider") or "").lower()
        rate = COST_PER_1K_TOKENS.get(provider, 0.0)
        return (tokens / 1000) * rate


response_cache = ResponseCache()
