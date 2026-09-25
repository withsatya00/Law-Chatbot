"""Greetings must never reach the knowledge base.

Reported: "Namste" -- a one-letter misspelling of "Namaste" -- was answered
with "No verified document related to this question is currently available in
the Knowledge Base." That is the worst possible first impression: it is the
literal first message a user sends.

The typo turned out to be the small half of the problem. Only the exact Latin
spelling "namaste" was recognised, so EVERY native-script greeting failed too
-- in an application that advertises all 22 Eighth Schedule languages.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.intent.classifier import _GENERAL_CONVERSATION_PATTERN
from app.llm.base import LLMResponse
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService


@pytest.mark.parametrize(
    "greeting",
    [
        # The reported case and its spelling neighbours.
        "Namste", "Namaste", "namaskar", "Namaskaar", "namaskaram", "Namskar", "namastey",
        # Other romanised Indian greetings.
        "pranam", "salaam", "adaab", "sat sri akal", "vanakkam", "kem cho",
        # Native scripts, one per major language.
        "नमस्ते", "नमस्ते।", "नमस्कार", "धन्यवाद",
        "নমস্কার", "প্রণাম", "ধন্যবাদ",
        "નમસ્તે", "કેમ છો",
        "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", "ਧੰਨਵਾਦ",
        "ନମସ୍କାର",
        "வணக்கம்", "நன்றி",
        "నమస్కారం", "ధన్యవాదాలు",
        "ನಮಸ್ಕಾರ",
        "നമസ്കാരം", "നന്ദി",
        "السلام علیکم", "آداب", "شکریہ",
        # Pre-existing coverage that must not regress.
        "hi", "hii", "hello", "hloo", "thanks", "shukriya", "dhanyavad", "theek hai",
    ],
)
def test_greetings_are_recognised_in_every_supported_language(greeting: str) -> None:
    assert _GENERAL_CONVERSATION_PATTERN.match(greeting), f"{greeting!r} was not recognised as a greeting"


@pytest.mark.parametrize(
    "message",
    [
        # A greeting followed by a real question must NOT decompose -- the
        # anchored whole-message rule is what keeps this safe.
        "Namaste, my landlord will not return my deposit",
        "Namste, mujhe complaint karni hai",
        "नमस्ते, मुझे शिकायत दर्ज करनी है",
        "vanakkam, enakku oru complaint venum",
        "thanks for nothing, now explain section 138",
        # Words that merely start like a greeting.
        "name change kaise kare",
        "namune ke liye draft banao",
        # Ordinary legal questions.
        "What is anticipatory bail?",
        "धन्यवाद के बारे में कानून क्या कहता है",
    ],
)
def test_real_questions_are_never_swallowed_as_greetings(message: str) -> None:
    assert not _GENERAL_CONVERSATION_PATTERN.match(message), f"{message!r} was wrongly treated as a greeting"


def _service() -> ChatService:
    """A ChatService whose retrieval raises if reached.

    A greeting reaching retrieval is the bug this file exists to prevent, so
    it must fail loudly rather than quietly returning a refusal.
    """
    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])
    empty = {"messages": [], "summary": "", "current_intent": None, "legal_category": None}
    service.memory.append = AsyncMock(return_value=empty)
    service.memory.load = AsyncMock(return_value=empty)
    service.memory.update = AsyncMock(return_value={})
    service.memory.summarize_if_needed = AsyncMock(return_value={})
    service.history.insert = AsyncMock(return_value="h")
    service.query_log.insert = AsyncMock(return_value="q")
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    service.retriever.retrieve = AsyncMock(side_effect=AssertionError("a greeting must never reach retrieval"))
    service.reranker.rerank = AsyncMock(side_effect=AssertionError("a greeting must never reach reranking"))
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Namaste! How can I help?", model="t", provider="t"))
    return service


@pytest.mark.parametrize("greeting", ["Namste", "नमस्ते", "வணக்கம்", "السلام علیکم", "namaskar"])
def test_a_greeting_never_returns_the_knowledge_base_refusal(greeting: str) -> None:
    """End to end: the exact reported symptom."""
    service = _service()
    response = asyncio.run(service.answer(ChatRequest(question=greeting)))

    assert response.conversation_intent == "General Conversation"
    assert "Knowledge Base" not in response.answer
    assert "सत्यापित दस्तावेज़" not in response.answer
    service.retriever.retrieve.assert_not_awaited()
