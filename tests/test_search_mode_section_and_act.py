"""Regression tests for BUG-109/BUG-111 (QA session 2026-09-24):
`SearchService.search` used to ignore `SearchRequest.mode` entirely -- every
mode silently ran the same hybrid retrieval, so `mode="act"`/`mode="section"`
had zero effect on the actual query behavior despite being distinct,
documented, client-selectable options.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.schemas.common import RetrievedChunk
from app.schemas.search import SearchRequest
from app.services.search_service import SearchService


def _chunk(chunk_id: str, act_name: str | None, section_number: str | None, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=f"text for {chunk_id}", score=0.0,
        metadata={"act_name": act_name, "section_number": section_number, "source_document": source},
    )


def test_mode_section_with_a_bare_number_uses_the_exact_metadata_lookup_not_similarity_ranking() -> None:
    service = SearchService()
    service.retriever.vector_store.find_by_section_number = AsyncMock(
        return_value=[_chunk("c1", "Negotiable Instruments Act", "138", "ni_act.pdf")]
    )
    service.retriever.retrieve = AsyncMock(return_value=("should not be called", []))

    result = asyncio.run(service.search(SearchRequest(query="138", mode="section", top_k=5)))

    service.retriever.vector_store.find_by_section_number.assert_awaited_once()
    called_section_number = service.retriever.vector_store.find_by_section_number.call_args.args[0]
    assert called_section_number == "138"
    service.retriever.retrieve.assert_not_awaited()
    assert len(result.results) == 1
    assert result.results[0].metadata["source_document"] == "ni_act.pdf"


def test_mode_section_still_applies_the_review_status_safety_gate() -> None:
    """The exact-lookup shortcut must not bypass the same shared-corpus
    review gate `LegalRetriever.retrieve` applies -- see security finding C10."""
    service = SearchService()
    service.retriever.vector_store.find_by_section_number = AsyncMock(return_value=[])
    service.retriever.retrieve = AsyncMock(return_value=("q", []))

    asyncio.run(service.search(SearchRequest(query="138", mode="section", top_k=5)))

    forwarded_filters = service.retriever.vector_store.find_by_section_number.call_args.args[1]
    assert forwarded_filters.get("review_status") is not None


def test_mode_section_with_non_bare_query_falls_back_to_hybrid_retrieval() -> None:
    """A full-sentence question isn't a bare section-number citation --
    `mode="section"` must still fall back to ordinary retrieval for it
    rather than finding nothing and returning empty."""
    service = SearchService()
    service.retriever.vector_store.find_by_section_number = AsyncMock(return_value=[])
    service.retriever.retrieve = AsyncMock(return_value=("rewritten", [_chunk("c1", "Act", "1", "doc.pdf")]))

    result = asyncio.run(service.search(SearchRequest(query="what happens if a cheque bounces", mode="section", top_k=5)))

    service.retriever.vector_store.find_by_section_number.assert_not_awaited()
    service.retriever.retrieve.assert_awaited_once()
    assert len(result.results) == 1


def test_mode_section_falls_back_to_hybrid_when_exact_lookup_finds_nothing() -> None:
    service = SearchService()
    service.retriever.vector_store.find_by_section_number = AsyncMock(return_value=[])
    service.retriever.retrieve = AsyncMock(return_value=("rewritten", [_chunk("c1", "Act", "999", "doc.pdf")]))

    result = asyncio.run(service.search(SearchRequest(query="999", mode="section", top_k=5)))

    service.retriever.retrieve.assert_awaited_once()
    assert len(result.results) == 1


def test_mode_act_filters_by_the_named_act_not_free_text_similarity() -> None:
    service = SearchService()
    service.retriever.retrieve = AsyncMock(return_value=("should not be called", []))
    ni_act_chunks = [
        _chunk("c1", "Negotiable Instruments Act", "138", "ni_act.pdf"),
        _chunk("c2", "Negotiable Instruments Act", "6", "ni_act.pdf"),
    ]
    service._find_by_act_name = AsyncMock(return_value=ni_act_chunks)

    result = asyncio.run(service.search(SearchRequest(query="Negotiable Instruments Act", mode="act", top_k=5)))

    service._find_by_act_name.assert_awaited_once()
    service.retriever.retrieve.assert_not_awaited()
    assert {c.metadata["source_document"] for c in result.results} == {"ni_act.pdf"}


def test_mode_act_falls_back_to_hybrid_when_no_exact_act_name_match() -> None:
    service = SearchService()
    service._find_by_act_name = AsyncMock(return_value=[])
    service.retriever.retrieve = AsyncMock(return_value=("rewritten", [_chunk("c1", "Some Act", "1", "doc.pdf")]))

    result = asyncio.run(service.search(SearchRequest(query="a vague act description", mode="act", top_k=5)))

    service.retriever.retrieve.assert_awaited_once()
    assert len(result.results) == 1
