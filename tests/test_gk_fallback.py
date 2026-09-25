"""Coverage for the controlled General Knowledge (GK) fallback
(`app.core.gk_fallback`) -- the opt-in path that lets an out-of-KB legal
question be answered from the LLM's general knowledge instead of the bare
strict-RAG refusal, gated so it can never fabricate a citation, never fire
for a query needing a specific/current provision, and never override the
hardcoded safety-guidance categories.
"""

import pytest

from app.core.constants import general_knowledge_disclaimer, general_knowledge_label
from app.core.gk_fallback import (
    eligible,
    general_knowledge_answer,
    is_urgent_safety_category,
    requires_current_or_specific_provision,
)
from app.llm.base import ChatMessage, LLMProvider, LLMResponse


class _FakeProvider(LLMProvider):
    def __init__(self, response: LLMResponse) -> None:
        self.provider_name = "fake"
        self.model = "fake-model"
        self._response = response
        self.calls = 0
        self.last_messages: list[ChatMessage] | None = None

    async def chat(self, messages, temperature: float = 0.1) -> LLMResponse:
        self.calls += 1
        self.last_messages = messages
        return self._response.model_copy()

    async def stream(self, messages, temperature: float = 0.1):
        yield self._response.content

    async def health(self) -> bool:
        return True


def _response(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="fake-model", provider="fake")


# --- eligibility gating -----------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Section 302 kya hai",
        "What is Section 138 of the NI Act?",
        "IPC 420 explain karo",
    ],
)
def test_specific_provision_queries_are_never_eligible(question: str) -> None:
    assert requires_current_or_specific_provision(question)
    assert not eligible(question)


@pytest.mark.parametrize(
    "question",
    [
        "What is the latest amendment to the Consumer Protection Act?",
        "What is the current law on cheque bounce?",
        "vartaman kanoon kya hai is baare mein",
    ],
)
def test_current_law_queries_are_never_eligible(question: str) -> None:
    assert requires_current_or_specific_provision(question)
    assert not eligible(question)


def test_urgent_safety_categories_defer_to_fixed_guidance() -> None:
    question = "my husband beats me and threatens me, what do I do"
    assert is_urgent_safety_category(question)
    assert not eligible(question)


def test_ordinary_out_of_kb_question_is_eligible() -> None:
    assert eligible("What is the general process for registering a partnership firm in India?")


def test_empty_question_is_not_eligible() -> None:
    assert not eligible("")
    assert not eligible("   ")


# --- general_knowledge_answer: acceptance -----------------------------------


@pytest.mark.asyncio
async def test_accepts_medium_confidence_answer_with_no_citations() -> None:
    provider = _FakeProvider(_response(
        "CONFIDENCE: MEDIUM\n\nIn general, partnership firms in India are governed by partnership "
        "law and may optionally be registered with the Registrar of Firms."
    ))
    answer = await general_knowledge_answer(
        provider, "How do I register a partnership firm?", "english"
    )
    assert answer is not None
    assert general_knowledge_label("english") in answer
    assert general_knowledge_disclaimer("english") in answer
    assert "Registrar of Firms" in answer
    assert "CONFIDENCE" not in answer


@pytest.mark.asyncio
async def test_accepts_high_confidence_answer() -> None:
    provider = _FakeProvider(_response("CONFIDENCE: HIGH\n\nGenerally speaking, this is a civil matter."))
    answer = await general_knowledge_answer(provider, "Is this a civil or criminal matter generally?", "english")
    assert answer is not None


@pytest.mark.asyncio
async def test_labels_and_disclaims_in_requested_language() -> None:
    provider = _FakeProvider(_response("CONFIDENCE: MEDIUM\n\nयह सामान्य जानकारी है।"))
    answer = await general_knowledge_answer(provider, "generic legal process question", "hindi")
    assert answer is not None
    assert general_knowledge_label("hindi") in answer
    assert general_knowledge_disclaimer("hindi") in answer


# --- general_knowledge_answer: rejection ------------------------------------


@pytest.mark.asyncio
async def test_rejects_low_confidence() -> None:
    provider = _FakeProvider(_response("CONFIDENCE: LOW\n\nI'm not sure about this."))
    answer = await general_knowledge_answer(provider, "some obscure legal question", "english")
    assert answer is None


@pytest.mark.asyncio
async def test_rejects_missing_confidence_tag() -> None:
    provider = _FakeProvider(_response("Here is an answer with no confidence tag at all."))
    answer = await general_knowledge_answer(provider, "some legal question", "english")
    assert answer is None


@pytest.mark.asyncio
async def test_rejects_fabricated_section_citation() -> None:
    provider = _FakeProvider(_response(
        "CONFIDENCE: HIGH\n\nThis is governed by Section 42 of the relevant Act, 2015."
    ))
    answer = await general_knowledge_answer(provider, "generic legal process question", "english")
    assert answer is None


@pytest.mark.asyncio
async def test_rejects_fabricated_case_citation() -> None:
    provider = _FakeProvider(_response(
        "CONFIDENCE: HIGH\n\nAs held in Sharma v. State, this principle generally applies."
    ))
    answer = await general_knowledge_answer(provider, "generic legal process question", "english")
    assert answer is None


@pytest.mark.asyncio
async def test_rejects_on_provider_error() -> None:
    provider = _FakeProvider(LLMResponse(
        content="", model="fake-model", provider="fake", error="connection failed", error_kind="timeout",
    ))
    answer = await general_knowledge_answer(provider, "generic legal process question", "english")
    assert answer is None


@pytest.mark.asyncio
async def test_never_calls_llm_for_ineligible_question() -> None:
    provider = _FakeProvider(_response("CONFIDENCE: HIGH\n\nShould never be reached."))
    answer = await general_knowledge_answer(provider, "Section 302 kya hai", "english")
    assert answer is None
    assert provider.calls == 0
