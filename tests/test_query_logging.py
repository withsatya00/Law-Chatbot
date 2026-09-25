import asyncio
from unittest.mock import AsyncMock

from app.llm.base import LLMResponse
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService


def _service_with_mocks(messages: list[dict[str, str]] | None = None) -> ChatService:
    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])
    service.memory.append = AsyncMock(
        return_value={"messages": messages or [], "summary": "", "current_intent": None, "legal_category": None}
    )
    service.memory.update = AsyncMock(return_value={})
    service.memory.summarize_if_needed = AsyncMock(return_value={})
    service.history.insert = AsyncMock(return_value="history-id")
    service.query_log.insert = AsyncMock(return_value="query-log-id")
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    return service


def test_general_conversation_writes_query_log_matching_response() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Happy to help!", model="test", provider="test"))
    request = ChatRequest(question="Thanks, that helps")
    response = asyncio.run(service.answer(request))

    service.query_log.insert.assert_awaited_once()
    (entry,) = service.query_log.insert.await_args.args
    assert entry["_id"] == response.message_id
    assert entry["message_id"] == response.message_id
    assert entry["question"] == request.question
    assert entry["answer"] == response.answer
    assert entry["confidence"] == response.confidence
    assert entry["conversation_intent"] == "General Conversation"
    assert entry["llm_provider"] == response.llm_provider
    assert entry["rating"] is None


def test_query_log_written_via_background_task_when_provided() -> None:
    from fastapi import BackgroundTasks

    service = _service_with_mocks()
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Happy to help!", model="test", provider="test"))
    request = ChatRequest(question="Thanks, that helps")
    background_tasks = BackgroundTasks()

    asyncio.run(service.answer(request, background_tasks))

    # Scheduled, not awaited inline, when background_tasks is supplied.
    service.query_log.insert.assert_not_awaited()
    assert len(background_tasks.tasks) == 2  # memory.summarize_if_needed + query_log.insert
