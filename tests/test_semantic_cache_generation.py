"""Phase 1 "Jurisdiction-Aware Knowledge Base" gap 3: `retrieval_cache`
(`app.cache.semantic_cache.SemanticCache`) must be invalidated by the exact
same event that already invalidates `response_cache`
(`app.cache.response_cache.ResponseCache.bump_generation`) -- a document
withdrawn/marked needs_review by the jurisdiction backfill or the correction
endpoint must not keep being served from either cache for the rest of its
TTL.

Same fake-redis-client convention as
`test_response_cache.py::test_purge_entry_deletes_the_exact_key_and_removes_it_from_the_bucket`
(`monkeypatch.setattr(redis_client, "_client", fake_client)`).
"""

import asyncio

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache.redis_client import redis_client
from app.cache.response_cache import CACHE_KEY_PREFIX, ResponseCache
from app.cache.semantic_cache import SemanticCache


class _FakeRedisClient:
    """Enough of the redis-py async interface for `.get`/`.incr` -- a plain
    in-memory dict, not a mock, so `incr` actually behaves like Redis's
    (parse-as-int, default 0, store back as bytes-like string)."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def incr(self, key: str) -> int:
        value = int(self.store.get(key, "0")) + 1
        self.store[key] = str(value)
        return value


def test_retrieval_cache_key_changes_when_response_cache_bumps_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(redis_client, "_client", fake_client)

    retrieval_cache = SemanticCache("retrieval-cache")
    key_before = asyncio.run(retrieval_cache.key("what is an fir", "english"))

    asyncio.run(ResponseCache().bump_generation())

    key_after = asyncio.run(retrieval_cache.key("what is an fir", "english"))
    assert key_before != key_after


def test_response_cache_and_retrieval_cache_share_one_generation_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not two independent counters -- ONE bump invalidates both caches at
    once, so a jurisdiction correction only ever needs a single call."""
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(redis_client, "_client", fake_client)

    asyncio.run(ResponseCache().bump_generation())
    assert fake_client.store[f"{CACHE_KEY_PREFIX}:kb-generation"] == "1"

    retrieval_cache = SemanticCache("retrieval-cache")
    generation = asyncio.run(retrieval_cache._generation())
    assert generation == 1


def test_generation_defaults_to_zero_when_redis_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A degraded Redis tier must never raise out of a cache lookup -- see
    `CACHE_UNAVAILABLE_ERRORS`; generation 0 is the same safe fallback
    `ResponseCache._generation` already uses."""

    class _BrokenClient:
        async def get(self, _key: str) -> str:
            raise RuntimeError("redis unavailable")

    monkeypatch.setattr(redis_client, "_client", _BrokenClient())
    cache = SemanticCache("retrieval-cache")
    assert asyncio.run(cache._generation()) == 0


def test_get_and_set_degrade_to_a_miss_on_a_live_redis_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirmed live: with Redis actually reachable-but-refusing-connections
    (a stopped server, not just "connect() was never called"), `.get`/`.set`
    used to only catch bare `RuntimeError` and let `redis.exceptions.
    ConnectionError` (a `RedisError`) propagate straight out -- which broke
    every caller of `response_cache`/`retrieval_cache`/`prompt_cache`, not
    just this class's own `_generation()` (already covered by the test
    above). Must match `CACHE_UNAVAILABLE_ERRORS`'s contract exactly like
    `_generation` does.
    """

    class _DownClient:
        async def get(self, _key: str) -> str:
            raise RedisConnectionError("Error 22 connecting to localhost:6379.")

        async def set(self, *_args: object, **_kwargs: object) -> None:
            raise RedisConnectionError("Error 22 connecting to localhost:6379.")

    monkeypatch.setattr(redis_client, "_client", _DownClient())
    cache = SemanticCache("response-cache")

    assert asyncio.run(cache.get("what is an fir", "english")) is None
    # Must not raise.
    asyncio.run(cache.set("what is an fir", {"answer": "..."}, "english"))
