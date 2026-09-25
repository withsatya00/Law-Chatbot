import asyncio
import json
from unittest.mock import AsyncMock

from app.llm.base import ChatMessage, LLMProvider, LLMResponse
from app.schemas.common import LawyerRecommendation
from app.schemas.document import DocumentAnalysisRequest
from app.services.document_service import DocumentService


class FakeLLM(LLMProvider):
    provider_name = "test"

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.messages: list[ChatMessage] = []

    async def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> LLMResponse:
        self.messages = messages
        return self.response

    async def stream(self, messages: list[ChatMessage], temperature: float = 0.1):
        yield ""

    async def health(self) -> bool:
        return True


def _recommendation() -> LawyerRecommendation:
    return LawyerRecommendation(category="General Law", confidence=0.5, reason="test")


def _service(response: LLMResponse) -> tuple[DocumentService, FakeLLM]:
    llm = FakeLLM(response)
    service = DocumentService(llm=llm)
    service.recommendations.recommend = AsyncMock(return_value=_recommendation())
    return service, llm


def _payload() -> dict:
    return {
        "executive_summary": "A tenancy agreement between two parties.",
        "legal_summary": "The tenant must pay rent and may face a penalty for late payment.",
        "important_clauses": ["Late payment penalty applies."],
        "analyzed_clauses": [
            {
                "name": "Late payment",
                "text": "A penalty applies after seven days.",
                "explanation": "The tenant owes an additional charge after the grace period.",
                "risk_level": "medium",
                "risk_reason": "The amount is not capped.",
            }
        ],
        "important_dates": ["1 January 2026 - commencement"],
        "important_names": ["Asha Verma", "Raj Traders"],
        "important_sections": ["Clause 4"],
        "key_risks": ["Uncapped late payment penalty."],
        "action_items": ["Confirm the penalty amount in writing."],
        "missing_information": ["The governing jurisdiction is absent."],
        "structured_data": {"parties": ["Asha Verma", "Raj Traders"], "obligations": ["Pay rent"]},
        "confidence": 0.91,
    }


def test_llm_analysis_populates_semantic_clauses_names_and_document_specific_missing_info() -> None:
    service, _ = _service(LLMResponse(content=json.dumps(_payload()), model="test", provider="test"))
    result = asyncio.run(service.analyze(DocumentAnalysisRequest(text="Rental agreement text.")))

    assert result.important_names == ["Asha Verma", "Raj Traders"]
    assert result.missing_information == ["The governing jurisdiction is absent."]
    assert result.analyzed_clauses[0].risk_reason == "The amount is not capped."
    assert result.structured_data["parties"] == ["Asha Verma", "Raj Traders"]
    assert result.confidence == 0.91


def test_llm_analysis_accepts_markdown_json_and_bounds_prompt_text(monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "document_analysis_max_chars", 40)
    service, llm = _service(LLMResponse(content=f"```json\n{json.dumps(_payload())}\n```", model="test", provider="test"))
    asyncio.run(service.analyze(DocumentAnalysisRequest(text="x" * 100)))

    prompt = llm.messages[0].content
    assert "x" * 40 in prompt
    assert "x" * 41 not in prompt


def test_malformed_or_failed_llm_output_uses_compatibility_fallback() -> None:
    for response in (
        LLMResponse(content="not json", model="test", provider="test"),
        LLMResponse(content="", model="test", provider="test", error="connection_failed"),
    ):
        service, _ = _service(response)
        result = asyncio.run(service.analyze(DocumentAnalysisRequest(text="Payment is due on 01/02/2026.")))
        assert result.confidence == 0.64
        assert result.structured_data["analysis_source"] == "deterministic_fallback"
        assert result.important_dates == ["01/02/2026"]


def test_document_analysis_wires_entities_and_returns_chronological_timeline() -> None:
    service, _ = _service(LLMResponse(content="not json", model="test", provider="test"))
    text = (
        "Section 138 was signed on 15 मार्च 2024. Payment was due on 12/03/2024. "
        "The notice was issued on 05-01-2023."
    )
    result = asyncio.run(service.analyze(DocumentAnalysisRequest(text=text, language="hindi")))

    assert result.structured_data["entities"]["section_number"] == ["138"]
    assert result.structured_data["entities"]["date"] == ["12/03/2024", "05-01-2023"]
    assert [event.date for event in result.timeline] == ["2023-01-05", "2024-03-12", "2024-03-15"]
    assert "notice was issued" in result.timeline[0].event_description
