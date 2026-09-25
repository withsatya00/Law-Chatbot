import asyncio
from unittest.mock import AsyncMock

from app.llm.base import LLMResponse
from app.memory.store import ConversationMemoryStore, _empty_memory


def test_summarize_if_needed_ignores_provider_error_response() -> None:
    # Regression: a provider outage (e.g. Ollama unreachable) must not
    # overwrite the rolling conversation summary with its friendly error
    # sentence -- that pollution would persist across every later turn until
    # the next successful summarization, which may never happen while the
    # provider stays down.
    store = ConversationMemoryStore()
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(14)]
    memory = {
        "summary": "existing summary",
        "messages": messages,
        "language_preference": None,
        "current_intent": None,
        "legal_category": None,
        "uploaded_documents": [],
    }
    store.load = AsyncMock(return_value=memory)
    store._persist = AsyncMock(return_value=None)
    fake_llm = AsyncMock()
    fake_llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="Local LLM is unavailable. Please start Ollama.",
            model="qwen3:8b",
            provider="ollama",
            error="connection_failed",
        )
    )

    result = asyncio.run(store.summarize_if_needed("session-1", fake_llm))

    assert result["summary"] == "existing summary"


def test_empty_memory_includes_last_uploaded_document_id() -> None:
    assert _empty_memory()["last_uploaded_document_id"] is None


def test_typed_facts_keep_corrections_and_allow_deletion() -> None:
    store = ConversationMemoryStore()
    memory = _empty_memory()
    store.load = AsyncMock(return_value=memory)
    store._persist = AsyncMock(return_value=None)

    asyncio.run(store.set_fact("session-1", "confirmed", "claim_amount", "1000"))
    asyncio.run(store.set_fact("session-1", "confirmed", "claim_amount", "1500"))
    assert memory["confirmed_facts"] == {"claim_amount": "1500"}
    assert memory["fact_corrections"][0]["previous"] == "1000"
    assert memory["fact_corrections"][0]["current"] == "1500"

    asyncio.run(store.delete_fact("session-1", "confirmed", "claim_amount"))
    assert memory["confirmed_facts"] == {}


def test_missing_details_are_trimmed_and_deduplicated() -> None:
    store = ConversationMemoryStore()
    memory = _empty_memory()
    store.load = AsyncMock(return_value=memory)
    store._persist = AsyncMock(return_value=None)

    asyncio.run(store.set_missing_details("session-1", ["court", " court ", "", "filing date"]))
    assert memory["missing_details"] == ["court", "filing date"]


def test_last_uploaded_document_id_survives_a_cold_mongo_reload() -> None:
    # Part 51 "Uploaded Document Conversation Context": a new field added
    # only via `update(session_id, new_field=...)` round-trips through the
    # Redis cache fine, but is silently dropped on a cold Mongo reload
    # unless it's also read back in `load()`'s Mongo-fallback reconstruction
    # -- this reproduces exactly that path (Redis miss -> Mongo hit).
    store = ConversationMemoryStore()
    store.memory_repository = AsyncMock()
    store.memory_repository.find_by_session = AsyncMock(
        return_value={
            "summary": "", "recent_messages": [], "language_preference": None, "current_intent": None,
            "legal_category": None, "uploaded_documents": [], "last_failed_question": None,
            "last_failed_reason": None, "last_successful_response": None, "entities": [], "owner_user_id": None,
            "last_uploaded_document_id": "doc-123",
        }
    )

    async def _run():
        import app.memory.store as store_module

        original_get_json = store_module.redis_client.get_json
        original_set_json = store_module.redis_client.set_json
        store_module.redis_client.get_json = AsyncMock(side_effect=RuntimeError("no redis in this test"))
        store_module.redis_client.set_json = AsyncMock(side_effect=RuntimeError("no redis in this test"))
        try:
            return await store.load("session-1")
        finally:
            store_module.redis_client.get_json = original_get_json
            store_module.redis_client.set_json = original_set_json

    memory = asyncio.run(_run())
    assert memory["last_uploaded_document_id"] == "doc-123"


def test_persist_writes_last_uploaded_document_id_to_mongo() -> None:
    store = ConversationMemoryStore()
    store.memory_repository = AsyncMock()
    store.memory_repository.upsert_by_session = AsyncMock(return_value=None)

    async def _run():
        import app.memory.store as store_module

        original_set_json = store_module.redis_client.set_json
        store_module.redis_client.set_json = AsyncMock(side_effect=RuntimeError("no redis in this test"))
        try:
            memory = _empty_memory()
            memory["last_uploaded_document_id"] = "doc-456"
            await store._persist("session-1", memory)
        finally:
            store_module.redis_client.set_json = original_set_json

    asyncio.run(_run())
    written = store.memory_repository.upsert_by_session.call_args.args[1]
    assert written["last_uploaded_document_id"] == "doc-456"
