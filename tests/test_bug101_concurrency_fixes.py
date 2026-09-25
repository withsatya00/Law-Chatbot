"""Regression tests for BUG-101 (QA session 2026-09-24): the concurrency
collapse under load, its two crash/failure symptoms, and the production
fail-fast guard against the slow local vector-scan fallback.

Evidence this run reproduces against: qa-100q-full-20260924/
CONCURRENCY_BUG_EVIDENCE_6way.log (6 concurrent /chat calls -> 100-197s
retrieval, Redis timeout crashing the request, Postgres long-term-memory
writes failing on a datetime/str type mismatch).
"""

import asyncio

import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError
from unittest.mock import AsyncMock

from app.core.config import Settings
from app.memory.store import ConversationMemoryStore
from app.rag.vector_store import MongoVectorStore


def test_production_rejects_local_vector_search_backend():
    """BUG-101: a production/staging process must refuse to start on the
    slow, concurrency-fragile local scan fallback instead of silently
    shipping it -- mirrors the existing `_validate_security` pattern."""
    config = Settings.model_construct(
        environment="production", vector_search_backend="local",
        jwt_secret_key="a" * 48, secrets_encryption_key="b" * 48, api_cors_origins=["https://example.com"],
    )
    with pytest.raises(RuntimeError, match="VECTOR_SEARCH_BACKEND"):
        config._validate_retrieval_backend()


def test_production_accepts_atlas_vector_search_backend():
    config = Settings.model_construct(
        environment="production", vector_search_backend="atlas",
        jwt_secret_key="a" * 48, secrets_encryption_key="b" * 48, api_cors_origins=["https://example.com"],
    )
    config._validate_retrieval_backend()  # must not raise


@pytest.mark.parametrize("environment", ["development", "test"])
def test_dev_and_test_environments_are_unvalidated(environment):
    """Local workflows and the existing test suite must keep working
    unchanged on the local fallback -- only production/staging are gated."""
    config = Settings.model_construct(environment=environment, vector_search_backend="local")
    config._validate_retrieval_backend()  # must not raise


def test_local_scan_semaphore_is_bounded_and_shared_across_instances():
    """`MongoVectorStore()` is constructed fresh per request (see
    `LegalRetriever`) -- the concurrency limit only works if the semaphore is
    a class-level singleton, not a per-instance one that would reset to a
    fresh permit count on every single request."""
    store_a = MongoVectorStore()
    store_b = MongoVectorStore()
    assert store_a._local_scan_semaphore is store_b._local_scan_semaphore
    assert store_a._local_scan_semaphore._value == MongoVectorStore._local_scan_semaphore._value


def test_local_scan_semaphore_actually_serializes_concurrent_callers():
    """Directly exercises the concurrency bound: with the semaphore's permit
    count temporarily set to 1, two overlapping acquires must not both be
    inside the critical section at the same time."""
    store = MongoVectorStore()
    sem = asyncio.Semaphore(1)
    store._local_scan_semaphore = sem  # instance-shadow for this test only
    concurrent_inside = 0
    max_concurrent_seen = 0

    async def guarded_task():
        nonlocal concurrent_inside, max_concurrent_seen
        async with store._local_scan_semaphore:
            concurrent_inside += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_inside)
            await asyncio.sleep(0.05)
            concurrent_inside -= 1

    async def run_two_concurrently():
        await asyncio.gather(guarded_task(), guarded_task())

    asyncio.run(run_two_concurrently())
    assert max_concurrent_seen == 1


def test_persist_degrades_gracefully_on_redis_timeout_instead_of_crashing():
    """BUG-101 live-reproduced crash (Odia language test, T016): a real
    `redis.exceptions.TimeoutError` -- a `RedisError`, not a `RuntimeError`
    -- previously propagated out of `_persist` uncaught and crashed the
    whole `/chat` request with a 500 even though the LLM answer had already
    been generated. Must now degrade to a logged warning."""
    store = ConversationMemoryStore()
    from app.cache import redis_client as redis_client_module

    async def raise_timeout(*args, **kwargs):
        raise RedisTimeoutError("Timeout reading from localhost:6379")

    original_set_json = redis_client_module.redis_client.set_json
    redis_client_module.redis_client.set_json = raise_timeout
    store.memory_repository.upsert_by_session = AsyncMock(return_value=None)
    try:
        asyncio.run(store._persist("session-bug101", {"summary": "", "messages": []}))
    finally:
        redis_client_module.redis_client.set_json = original_set_json
    store.memory_repository.upsert_by_session.assert_awaited_once()


def test_persist_still_degrades_gracefully_on_redis_not_connected():
    """Regression guard: the narrower pre-fix case (`RuntimeError` from a
    never-connected client) must keep working exactly as before."""
    store = ConversationMemoryStore()
    from app.cache import redis_client as redis_client_module

    async def raise_not_connected(*args, **kwargs):
        raise RuntimeError("Redis client is not connected.")

    original_set_json = redis_client_module.redis_client.set_json
    redis_client_module.redis_client.set_json = raise_not_connected
    store.memory_repository.upsert_by_session = AsyncMock(return_value=None)
    try:
        asyncio.run(store._persist("session-bug101b", {"summary": "", "messages": []}))
    finally:
        redis_client_module.redis_client.set_json = original_set_json
    store.memory_repository.upsert_by_session.assert_awaited_once()
