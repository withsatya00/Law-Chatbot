"""`EmbeddingProvider`'s Redis cache must degrade to "no cache" on any Redis
outage, never raise -- confirmed live: with Redis actually stopped (not just
"connect() was never called"), indexing failed outright with
"Error 22 connecting to localhost:6379" instead of just recomputing the
embedding without a cache hit. `_cache_get`/`_cache_set` used to only catch
bare `RuntimeError`, missing `redis.exceptions.ConnectionError` (a
`RedisError`) -- inconsistent with `CACHE_UNAVAILABLE_ERRORS`, the contract
every other cache call site in this codebase already follows.
"""
import asyncio

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache.redis_client import redis_client
from app.rag.embeddings import EmbeddingProvider


class _DownClient:
    async def get(self, _key: str) -> bytes:
        raise RedisConnectionError("Error 22 connecting to localhost:6379.")

    async def set(self, *_args: object, **_kwargs: object) -> None:
        raise RedisConnectionError("Error 22 connecting to localhost:6379.")


def test_cache_get_returns_none_on_a_live_redis_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redis_client, "_client", _DownClient())
    provider = EmbeddingProvider()

    assert asyncio.run(provider._cache_get("embedding:some-key")) is None


def test_cache_set_does_not_raise_on_a_live_redis_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redis_client, "_client", _DownClient())
    provider = EmbeddingProvider()

    asyncio.run(provider._cache_set("embedding:some-key", [0.1, 0.2, 0.3]))
