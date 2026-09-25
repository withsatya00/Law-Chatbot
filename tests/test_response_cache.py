import asyncio
from unittest.mock import AsyncMock

import pytest

from app.cache.redis_client import redis_client
from app.cache.response_cache import DEFINITIONAL_TTL_SECONDS, FAQ_TTL_SECONDS, ResponseCache
from app.core.constants import no_verified_context_message
from app.llm.base import LLMResponse
from app.schemas.chat import ChatRequest
from app.schemas.common import RetrievedChunk
from app.services.chat_service import ChatService


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    cache = ResponseCache()
    assert cache.normalize("  What   IS an FIR?  ") == "what is an fir?"


def test_ttl_for_definitional_question_is_longer_than_faq() -> None:
    cache = ResponseCache()
    assert cache.ttl_for("What is an FIR?") == DEFINITIONAL_TTL_SECONDS
    assert cache.ttl_for("Explain the concept of bail") == DEFINITIONAL_TTL_SECONDS
    assert cache.ttl_for("My landlord won't return my deposit") == FAQ_TTL_SECONDS
    assert DEFINITIONAL_TTL_SECONDS > FAQ_TTL_SECONDS


def test_is_cacheable_requires_high_confidence_sources_and_no_error() -> None:
    cache = ResponseCache()
    assert cache.is_cacheable(confidence=0.65, has_sources=True, llm_error=None) is True
    assert cache.is_cacheable(confidence=0.5, has_sources=True, llm_error=None) is False
    assert cache.is_cacheable(confidence=0.9, has_sources=False, llm_error=None) is False
    assert cache.is_cacheable(confidence=0.9, has_sources=True, llm_error="connection_failed") is False


def test_jurisdiction_key_scopes_the_bucket_so_states_never_share_a_cache_entry() -> None:
    """Jurisdiction Routing (Phase 2), objective item 6: two callers asking
    the identical question under different resolved States must land in
    different buckets, the same disambiguation mechanism SECTION_LOOKUP's own
    section-number suffix already uses (see the test right below)."""
    cache = ResponseCache()

    async def bucket_for(jurisdiction_key: str | None) -> str:
        normalized = cache.normalize("what is the eviction notice period")
        bucket_key, _ = await cache._keys("english", "GENERAL", normalized, jurisdiction_key)
        return bucket_key

    mp_bucket = asyncio.run(bucket_for("MP::"))
    up_bucket = asyncio.run(bucket_for("UP::"))
    no_jurisdiction_bucket = asyncio.run(bucket_for(None))

    assert mp_bucket != up_bucket
    assert mp_bucket != no_jurisdiction_bucket


def test_section_lookup_bucket_key_is_scoped_to_parsed_section_number() -> None:
    # Regression for the "Section 420 IPC" bug: "section 420" and "section 420
    # ipc" measure 0.9048 cosine similarity on the real embedding model --
    # comfortably over SIMILARITY_THRESHOLD (0.90) -- yet the IPC->BNS
    # crosswalk means they ask for two entirely different provisions (BNSS
    # 420 vs BNS 318). Before this fix both shared one bucket
    # ("section-lookup"), so the semantic leg would serve BNSS 420's cached
    # answer for a BNS 318 question. Bucketing by the query's own parsed
    # section number keeps genuine same-section paraphrases ("IPC 420", "BNS
    # 318", "Section 420 IPC" -- all resolve to BNS 318) sharing one bucket,
    # while making a different target section structurally unable to collide.
    cache = ResponseCache()

    async def bucket_for(question: str) -> str:
        normalized = cache.normalize(question)
        bucket_key, _ = await cache._keys("english", "SECTION_LOOKUP", normalized)
        return bucket_key

    bucket_420 = asyncio.run(bucket_for("Section 420"))
    bucket_420_ipc = asyncio.run(bucket_for("Section 420 IPC"))
    bucket_ipc_420 = asyncio.run(bucket_for("IPC 420"))
    bucket_bns_318 = asyncio.run(bucket_for("BNS 318"))

    assert bucket_420 != bucket_420_ipc
    assert bucket_420_ipc == bucket_ipc_420 == bucket_bns_318


def test_cosine_similarity_identical_vectors_is_one() -> None:
    cache = ResponseCache()
    vector = [0.6, 0.8, 0.0]
    assert abs(cache._cosine(vector, vector) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    cache = ResponseCache()
    assert cache._cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_purge_entry_deletes_the_exact_key_and_removes_it_from_the_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: a bad cached answer (wrong retrieval + wrong generation, not
    # a cache-matching bug) needs a targeted way to clear just that one entry
    # without invalidating every other cached response.
    cache = ResponseCache()
    fake_client = AsyncMock()
    fake_client.delete = AsyncMock(return_value=1)
    fake_client.lrem = AsyncMock(return_value=1)
    monkeypatch.setattr(redis_client, "_client", fake_client)
    monkeypatch.setattr(cache, "_generation", AsyncMock(return_value=0))

    removed = asyncio.run(cache.purge_entry("fir kya hota hai", "hinglish", "FIR"))

    assert removed is True
    fake_client.delete.assert_awaited_once()
    fake_client.lrem.assert_awaited_once()
    (entry_key,) = fake_client.delete.await_args.args
    assert entry_key.startswith("legal:entry:")
    assert "hinglish" in entry_key
    assert "fir" in entry_key


def _service_with_mocks() -> ChatService:
    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])
    service.memory.append = AsyncMock(
        return_value={"messages": [], "summary": "", "current_intent": None, "legal_category": None}
    )
    service.memory.update = AsyncMock(return_value={})
    service.memory.summarize_if_needed = AsyncMock(return_value={})
    service.history.insert = AsyncMock(return_value="history-id")
    service.query_log.insert = AsyncMock(return_value="query-log-id")
    return service


def _rag_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        text="An FIR is a First Information Report registered under BNSS.",
        score=0.7,
        metadata={"source_document": "BNSS", "section_number": "173", "act_name": "BNSS"},
    )


def test_cache_hit_skips_retrieval_and_llm_entirely() -> None:
    service = _service_with_mocks()
    cached_entry = {
        "response": {
            "answer": "An FIR is a First Information Report.",
            "sources": [{"source_document": "BNSS", "section": "173"}],
            "confidence": 0.8,
            "confidence_label": "High",
            "llm_provider": "ollama",
            "llm_model": "qwen3:8b",
            "prompt_tokens": 120,
            "completion_tokens": 40,
        }
    }
    service.response_cache.lookup = AsyncMock(return_value=(cached_entry, "exact"))
    service.retriever.retrieve = AsyncMock(side_effect=AssertionError("retrieval should be skipped on a cache hit"))
    service.llm.chat = AsyncMock(side_effect=AssertionError("LLM should not be called on a cache hit"))

    request = ChatRequest(question="What is FIR?")
    response = asyncio.run(service.answer(request))

    assert response.answer == "An FIR is a First Information Report."
    assert response.cache_hit == "exact"
    assert response.prompt_tokens == 120
    assert response.completion_tokens == 40
    service.response_cache.lookup.assert_awaited_once()


def test_cache_miss_falls_through_to_normal_rag_and_stores_high_confidence_result() -> None:
    service = _service_with_mocks()
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    chunk = _rag_chunk()
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="An FIR is a First Information Report.", model="test", provider="test",
            prompt_tokens=100, completion_tokens=30,
        )
    )

    request = ChatRequest(question="What is FIR?")
    response = asyncio.run(service.answer(request))

    assert response.cache_hit is None
    service.response_cache.store.assert_awaited_once()
    *_stored_key, stored_payload = service.response_cache.store.await_args.args
    assert stored_payload["answer"] == response.answer
    assert stored_payload["prompt_tokens"] == 100
    assert stored_payload["completion_tokens"] == 30


def test_insufficient_context_response_is_not_cached() -> None:
    # Strict-RAG guardrail: zero relevant chunks short-circuits before the LLM
    # is ever called (the `side_effect` below would fail loudly if that ever
    # regressed back into calling it).
    service = _service_with_mocks()
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    service.retriever.retrieve = AsyncMock(return_value=("some obscure query", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(side_effect=AssertionError("no context means no LLM call"))

    request = ChatRequest(question="Some obscure query with no matching source")
    response = asyncio.run(service.answer(request))

    assert response.confidence == 0.0
    assert response.answer.startswith(no_verified_context_message("english"))
    service.response_cache.store.assert_not_awaited()
