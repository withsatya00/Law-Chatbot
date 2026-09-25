"""Legal-context retrieval: draft generation is grounded in the ACTUAL
Knowledge Base text of a template's hinted Acts/sections (see
`LegalDraftEngine._retrieve_legal_context`), not just the hint names plus the
model's own recollection -- the root-cause fix for the citation-hallucination
risk `citation_audit.py` only catches after the fact.

Off by default for the whole test session (`conftest.py` sets
`DRAFT_LEGAL_CONTEXT_ENABLED=false`), since turning it on unconditionally
would load a real embedding model on every one of the ~50 other
`LegalDraftEngine()` call sites across the drafting test suite that don't
care about it. Tests here turn it back on explicitly and mock
`engine.retriever.retrieve` so retrieval stays fast and deterministic.
"""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.drafting.engine import LegalDraftEngine
from app.drafting.templates import get_template
from app.llm.base import LLMResponse
from app.schemas.common import RetrievedChunk
from app.schemas.drafting import DraftPreviewRequest

_CHEQUE_FIELDS: dict[str, str] = {
    "applicant_name": "Sunita Rao",
    "applicant_address": "9, Model Town, Lucknow, Uttar Pradesh - 226001",
    "applicant_mobile": "9812345672",
    "respondent_name": "Vikram Singh",
    "respondent_address": "14, Aliganj, Lucknow, Uttar Pradesh - 226024",
    "cheque_number": "004512",
    "cheque_amount": "1,50,000",
    "cheque_date": "2026-07-20",
    "bank_name": "State Bank of India, Aliganj Branch",
    "dishonour_reason": "Funds insufficient",
    "dishonour_date": "2026-07-28",
    "facts": "The cheque was issued towards repayment of a loan and was returned unpaid.",
    "place": "Lucknow, Uttar Pradesh",
}

_NOTICE_HEADINGS = (
    "Recipient", "Subject", "Sender Details", "Introduction", "Facts of the Case",
    "Legal Position", "Consequences", "Prayer", "Signature Block",
)


def _fake_sections() -> str:
    return "\n\n".join(f"## {heading}\nContent." for heading in _NOTICE_HEADINGS)


@pytest.fixture(autouse=True)
def _enable_legal_context(monkeypatch):
    monkeypatch.setattr(settings, "draft_legal_context_enabled", True)


def test_retrieved_context_reaches_the_generation_prompt() -> None:
    engine = LegalDraftEngine()
    engine.retriever.retrieve = AsyncMock(
        return_value=(
            "Section 138 Negotiable Instruments Act",
            [
                RetrievedChunk(
                    chunk_id="c1",
                    text="Where any cheque drawn by a person is returned by the bank unpaid.",
                    score=0.9,
                    metadata={"act_name": "Negotiable Instruments Act, 1881", "section_number": "138"},
                )
            ],
        )
    )
    captured: dict[str, str] = {}

    async def _fake_chat(messages, **kwargs):
        captured["text"] = messages[0].content
        return LLMResponse(content=_fake_sections(), model="test", provider="test")

    engine.llm.chat = _fake_chat
    request = DraftPreviewRequest(draft_id="cheque_bounce_notice", language="english", fields=_CHEQUE_FIELDS)
    asyncio.run(engine.preview(request))

    engine.retriever.retrieve.assert_awaited_once()
    query = engine.retriever.retrieve.call_args.args[0]
    assert "Section 138" in query
    assert "Negotiable Instruments Act, 1881" in query
    assert "Negotiable Instruments Act, 1881, Section 138" in captured["text"]
    assert "returned by the bank unpaid" in captured["text"]


def test_retrieval_failure_never_blocks_generation() -> None:
    engine = LegalDraftEngine()
    engine.retriever.retrieve = AsyncMock(side_effect=RuntimeError("mongo unreachable"))
    captured: dict[str, str] = {}

    async def _fake_chat(messages, **kwargs):
        captured["text"] = messages[0].content
        return LLMResponse(content=_fake_sections(), model="test", provider="test")

    engine.llm.chat = _fake_chat
    request = DraftPreviewRequest(draft_id="cheque_bounce_notice", language="english", fields=_CHEQUE_FIELDS)
    response = asyncio.run(engine.preview(request))

    assert response.generated_by_llm is True
    assert "None retrieved." in captured["text"]


def test_retrieval_empty_results_render_as_none_retrieved() -> None:
    engine = LegalDraftEngine()
    engine.retriever.retrieve = AsyncMock(return_value=("Section 138", []))
    context = asyncio.run(engine._retrieve_legal_context(get_template("cheque_bounce_notice")))
    assert context == "None retrieved."


def test_a_template_with_no_hints_at_all_skips_retrieval() -> None:
    engine = LegalDraftEngine()
    engine.retriever.retrieve = AsyncMock(side_effect=AssertionError("must not be called with no hints"))
    template = get_template("cheque_bounce_notice")
    assert template is not None
    bare_template = replace(template, applicable_acts_hint=[], applicable_sections_hint=[])

    context = asyncio.run(engine._retrieve_legal_context(bare_template))

    assert context == "None retrieved."
    engine.retriever.retrieve.assert_not_awaited()


def test_disabled_by_setting_skips_retrieval() -> None:
    engine = LegalDraftEngine()
    engine.retriever.retrieve = AsyncMock(side_effect=AssertionError("must not be called when disabled"))
    settings.draft_legal_context_enabled = False
    try:
        context = asyncio.run(engine._retrieve_legal_context(get_template("cheque_bounce_notice")))
    finally:
        settings.draft_legal_context_enabled = True

    assert context == "None retrieved."
    engine.retriever.retrieve.assert_not_awaited()


def test_context_is_truncated_to_the_configured_character_budget() -> None:
    engine = LegalDraftEngine()
    long_text = "X" * 5000
    engine.retriever.retrieve = AsyncMock(
        return_value=("query", [RetrievedChunk(chunk_id="c1", text=long_text, score=0.9, metadata={})])
    )
    settings.draft_legal_context_max_chars = 100
    try:
        context = asyncio.run(engine._retrieve_legal_context(get_template("cheque_bounce_notice")))
    finally:
        settings.draft_legal_context_max_chars = 2500

    assert len(context) <= 100
