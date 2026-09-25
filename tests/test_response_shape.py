import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.constants import no_verified_context_message
from app.llm.base import LLMResponse
from app.schemas.chat import ChatRequest
from app.schemas.common import RetrievedChunk
from app.services.chat_service import ChatService


@pytest.mark.parametrize(
    "value,expected_label",
    [
        (0.2, "Low"),  # insufficient-context hardcoded value
        (0.34, "Low"),
        (0.35, "Medium"),
        (0.55, "Medium"),  # realistic "General Legal Query" fallback + solid retrieval blend
        (0.6, "High"),
        (0.85, "High"),
    ],
)
def test_confidence_label_bucketing(value: float, expected_label: str) -> None:
    service = ChatService()
    assert service._confidence_label(value) == expected_label


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


def _rag_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        text="An FIR is a First Information Report registered under BNSS.",
        score=0.7,
        metadata={"source_document": "BNSS", "section_number": "173", "act_name": "BNSS"},
    )


def test_related_questions_degrade_to_empty_when_second_llm_call_fails() -> None:
    service = _service_with_mocks()
    chunk = _rag_chunk()
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    main_answer = "An FIR is a First Information Report..."
    service.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(content=main_answer, model="test", provider="test"),
            RuntimeError("provider unavailable"),
        ]
    )
    request = ChatRequest(question="What is FIR?")
    response = asyncio.run(service.answer(request))
    assert response.related_questions == []
    assert main_answer in response.answer


def test_related_questions_degrade_to_empty_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # Shrink the real timeout constant so this test exercises the actual
    # `asyncio.wait_for` wrapper in `_generate_related_questions` without
    # waiting the real 6-second production timeout.
    monkeypatch.setattr("app.services.chat_service.RELATED_QUESTIONS_TIMEOUT_SECONDS", 0.2)

    service = _service_with_mocks()
    chunk = _rag_chunk()
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    main_answer = "An FIR is a First Information Report..."
    call_count = {"n": 0}

    async def _chat_side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(content=main_answer, model="test", provider="test")
        await asyncio.sleep(10)
        return LLMResponse(content="ignored", model="test", provider="test")

    service.llm.chat = _chat_side_effect
    request = ChatRequest(question="What is FIR?")
    response = asyncio.run(service.answer(request))
    assert response.related_questions == []
    assert main_answer in response.answer


def test_followup_resolution_ignores_provider_error_response() -> None:
    # Regression: an LLM provider that's down (e.g. Ollama unreachable)
    # returns a friendly error sentence as `LLMResponse.content` rather than
    # raising -- if that sentence were trusted as a resolved question, it
    # would get fed straight into retrieval instead of the user's real one.
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="Local LLM is unavailable. Please start Ollama.",
            model="qwen3:8b",
            provider="ollama",
            error="connection_failed",
        )
    )
    resolved = asyncio.run(service._resolve_followup_question("What about that?", {"messages": [], "summary": ""}))
    assert resolved == "What about that?"


def test_related_questions_ignore_provider_error_response() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="Local LLM is unavailable. Please start Ollama.",
            model="qwen3:8b",
            provider="ollama",
            error="connection_failed",
        )
    )
    related = asyncio.run(service._generate_related_questions("What is FIR?", "An FIR is...", "english"))
    assert related == []


def test_no_verified_context_never_reaches_the_llm() -> None:
    # Strict-RAG guardrail (replaces the old general-knowledge-fallback
    # prompt tests): zero relevant chunks must short-circuit before any LLM
    # call is made at all -- there is no longer a general-knowledge prompt
    # path to exercise.
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("some obscure question", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock()
    request = ChatRequest(question="Some obscure legal question with no KB match")
    response = asyncio.run(service.answer(request))

    service.llm.chat.assert_not_awaited()
    assert response.answer.startswith(no_verified_context_message("english"))


def test_rag_system_prompt_enforces_strict_grounding() -> None:
    # Strict-RAG guardrail: `system_prompt.md` must instruct the model to
    # answer only from the provided <context>, never blend in general
    # knowledge of Indian law, and never substitute a hedging disclaimer for
    # the exact fallback line. This exercises the MAIN RAG path (a real,
    # well-scoring chunk), which is what renders `system_prompt.md`.
    service = _service_with_mocks()
    chunk = _rag_chunk()
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="An FIR is a First Information Report...", model="test", provider="test")
    )
    request = ChatRequest(question="What is FIR?")
    asyncio.run(service.answer(request))

    sent_system_prompt = service.llm.chat.call_args_list[0].args[0][0].content
    assert "never use general knowledge" in sent_system_prompt.lower()
    # Regression for qa-40q-multilingual-20260921 BUG-01/BUG-02's deeper root
    # cause: Rule 2 used to hardcode the Hindi fallback line and rely on the
    # LLM to translate it into the reply language on the fly, but the "write
    # everything in {language}" rule elsewhere in the same prompt told it to
    # do exactly that -- so the two rules conflicted, and the LLM produced a
    # fresh (sometimes mixed-script) paraphrase every time instead of the
    # literal string `is_no_verified_context()` can recognize. The prompt now
    # gets handed the ALREADY-CORRECT string for the request's own language
    # (english here, since `ChatRequest` above names no language), so it only
    # has to copy it, never translate it.
    assert no_verified_context_message("english").lower() in sent_system_prompt.lower()


def test_rag_system_prompt_carries_the_fallback_line_in_the_requests_own_language() -> None:
    # Same guardrail as above, but for a non-English/non-Hindi request --
    # this is the case that was actually broken: Rule 2's line must be
    # PUNJABI already (not something the model has to translate itself).
    service = _service_with_mocks()
    chunk = _rag_chunk()
    service.retriever.retrieve = AsyncMock(return_value=("fir ki jankari", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="FIR ਇੱਕ ਪਹਿਲੀ ਜਾਣਕਾਰੀ ਰਿਪੋਰਟ ਹੈ...", model="test", provider="test")
    )
    request = ChatRequest(question="FIR ਕੀ ਹੈ?", language="punjabi")
    asyncio.run(service.answer(request))

    sent_system_prompt = service.llm.chat.call_args_list[0].args[0][0].content
    assert no_verified_context_message("punjabi") in sent_system_prompt
    assert no_verified_context_message("hindi") not in sent_system_prompt


def test_suggested_actions_present_for_short_circuit_and_rag_branch() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(side_effect=AssertionError("lawyer recommendation should not call the LLM"))
    request = ChatRequest(question="Can you recommend a lawyer for cheque bounce")
    response = asyncio.run(service.answer(request))
    assert response.suggested_actions

    service2 = _service_with_mocks()
    chunk = _rag_chunk()
    service2.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service2.reranker.rerank = AsyncMock(return_value=[chunk])
    service2.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(content="An FIR is a First Information Report...", model="test", provider="test"),
            LLMResponse(content="What is Zero FIR?\nCan FIR be withdrawn?", model="test", provider="test"),
        ]
    )
    request2 = ChatRequest(question="What is FIR?")
    response2 = asyncio.run(service2.answer(request2))
    assert response2.suggested_actions
    assert response2.related_questions == ["What is Zero FIR?", "Can FIR be withdrawn?"]
