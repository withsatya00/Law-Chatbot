"""Direct unit tests for `app.services.chat_support.analytics` (Phase 1
god-object split out of `ChatService`). `log_query` takes the repository as
an explicit parameter rather than being a method on a constructed
collaborator -- see the module's own docstring for why (the
`service.query_log = _NoopRepository()` reassignment in
`test_multi_turn_conversations.py`).
"""
import asyncio

import structlog

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import LawyerRecommendation
from app.services.chat_support.analytics import log_query, log_routing_decision


def _response(**overrides: object) -> ChatResponse:
    base: dict[str, object] = {
        "answer": "You may apply for bail under Section 480 BNSS.",
        "sources": [],
        "confidence": 0.7,
        "lawyer_recommendation": LawyerRecommendation(category="Criminal Law", confidence=0.5, reason="bail matter"),
        "detected_language": "english",
        "detected_intent": "Bail",
        "latency_ms": 120.0,
        "llm_provider": "gemini",
        "llm_model": "gemini-2.0-flash",
    }
    base.update(overrides)
    return ChatResponse(**base)


class _FakeQueryLog:
    def __init__(self) -> None:
        self.inserted: list[dict] = []

    async def insert(self, entry: dict) -> None:
        self.inserted.append(entry)


def test_log_routing_decision_emits_a_structured_log_line_with_masked_pii() -> None:
    with structlog.testing.capture_logs() as logs:
        log_routing_decision(
            session_id="s1", message_id="m1", question="My phone number is 9876543210",
            conversation_intent="Bail", route="rag",
            memory_hit=False, rag_used=True, response_modification=False, draft_mode=False,
        )
    assert len(logs) == 1
    entry = logs[0]
    assert entry["event"] == "chat_routing_decision"
    assert entry["session_id"] == "s1"
    assert entry["route"] == "rag"
    assert "9876543210" not in entry["question"]


def test_log_query_awaits_insert_directly_when_no_background_tasks() -> None:
    fake = _FakeQueryLog()
    request = ChatRequest(question="what is bail")
    response = _response()

    asyncio.run(log_query(fake, request, "s1", "m1", response, background_tasks=None))

    assert len(fake.inserted) == 1
    entry = fake.inserted[0]
    assert entry["message_id"] == "m1"
    assert entry["session_id"] == "s1"
    assert entry["llm_provider"] == "gemini"
    assert entry["answer"] == response.answer


def test_log_query_schedules_via_background_tasks_when_given() -> None:
    from fastapi import BackgroundTasks

    fake = _FakeQueryLog()
    background_tasks = BackgroundTasks()
    request = ChatRequest(question="what is bail")
    response = _response()

    asyncio.run(log_query(fake, request, "s1", "m1", response, background_tasks=background_tasks))

    # Not inserted synchronously -- scheduled as a background task instead.
    assert fake.inserted == []
    assert len(background_tasks.tasks) == 1
