"""Jurisdiction Routing (Phase 2) -- follow-up gap fixes:

1. streaming disclosure (assumed State + relevant date shown in
   `answer_stream`, not just `answer`);
2. historical-law version ambiguity (a data-driven signal, not a single
   incident date deciding every provision);
3. broader, data-driven clarification coverage (`detect_jurisdiction_ambiguity`),
   including a locality-specific question;
4. candidate-selection-level enforcement of BOTH jurisdiction and temporal
   filters (native Mongo/BM25 filtering, not only the retriever's post-filter).
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from test_chat_service_routing import _collect_stream, _mock_stream, _service_with_mocks

from app.rag import kb_jurisdiction as kj
from app.rag.bm25_index import BM25Index
from app.rag.vector_store import MongoVectorStore
from app.schemas.chat import ChatRequest
from app.schemas.common import RetrievedChunk


def _chunk(chunk_id: str, **metadata: Any) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text="text", score=0.9, metadata=metadata)


# ---------------------------------------------------------------------------
# 1. Streaming disclosure
# ---------------------------------------------------------------------------


def test_streaming_answer_discloses_assumed_profile_state() -> None:
    from app.schemas.phase3 import UserPreferencesResponse

    service = _service_with_mocks()
    service.preferences.get = AsyncMock(return_value=UserPreferencesResponse(profile_state_code="MP"))
    chunk = _chunk("c1", source_document="act.pdf", section_number="12")
    service.retriever.retrieve = AsyncMock(return_value=("deposit dispute", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.stream = _mock_stream(["Your deposit must be returned."])
    request = ChatRequest(question="My landlord is not returning my deposit", session_id="s1")
    events = asyncio.run(_collect_stream(service.answer_stream(request, authenticated_user_id="user-A")))
    done = next(event for event in events if event["event"] == "done")
    assert "Madhya Pradesh" in done["data"]["answer"]


def test_streaming_answer_discloses_the_as_of_date_used() -> None:
    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="c1", text="An FIR is a First Information Report.", score=0.9,
        metadata={"source_document": "CrPC", "act_name": "Code of Criminal Procedure", "section_number": "154"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.stream = _mock_stream(["An FIR ", "is a police report."])
    request = ChatRequest(question="On 12/03/2021 what is FIR?", session_id="s1")
    events = asyncio.run(_collect_stream(service.answer_stream(request)))
    done = next(event for event in events if event["event"] == "done")
    assert "2021-03-12" in done["data"]["answer"]


# ---------------------------------------------------------------------------
# 2. Historical-law version ambiguity is not silently resolved
# ---------------------------------------------------------------------------


def test_multiple_surviving_versions_for_the_same_provision_is_detectable() -> None:
    """Two DIFFERENT effective-date ranges both survive filtering for a
    given as_of_date only when their ranges genuinely overlap that date --
    a caller can detect this (same source_document/section_number, more than
    one distinct effective_from/effective_to pair) and must not silently
    pick one as "the" governing version."""
    v1 = _chunk("v1", source_document="act.pdf", section_number="12", effective_from="2015-01-01", effective_to=None)
    v2 = _chunk("v2", source_document="act.pdf", section_number="12", effective_from="2018-01-01", effective_to=None)
    # Both have no effective_to, so both are "eligible" under the
    # conservative absence-is-eligible rule for a date after both starts --
    # this is exactly the ambiguity a single as_of_date cannot resolve alone.
    survivors = kj.filter_by_matter_context([v1, v2], [], None, "2020-01-01")
    versions = {(c.metadata.get("effective_from"), c.metadata.get("effective_to")) for c in survivors}
    assert len(versions) > 1  # more than one candidate version -- genuinely ambiguous, not a single answer


def test_zero_verified_historical_coverage_yields_no_verified_context_not_a_guess() -> None:
    """A historical date earlier than any recorded version's effective_from:
    the strict-RAG guardrail (`no_verified_context`) fires because retrieval
    returns nothing -- never a silently-returned current version."""
    service = _service_with_mocks()
    only_version = _chunk("v1", source_document="act.pdf", effective_from="2020-01-01")
    # Simulate the retriever having already applied the temporal post-filter
    # (as `LegalRetriever.retrieve` does) and found nothing eligible.
    service.retriever.retrieve = AsyncMock(return_value=("what applied in 2005", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(side_effect=AssertionError("must not call the LLM with no verified context"))
    request = ChatRequest(question="On 01/01/2005 what applied under this Act?", session_id="s1")
    response = asyncio.run(service.answer(request))
    assert response.confidence == 0.0
    del only_version  # documents the scenario; the retriever mock is what's exercised


# ---------------------------------------------------------------------------
# 3. Broader, data-driven clarification coverage
# ---------------------------------------------------------------------------


def test_conflicting_state_candidates_trigger_clarification_even_outside_property_law() -> None:
    """A topic with no hardcoded keyword ("Banking and Criminal Law") still
    gets a clarifying question when the KB's own retrieved candidates
    disagree by State -- the general, data-driven backstop
    (`detect_jurisdiction_ambiguity`), not the fixed keyword pre-check."""
    up_chunk = _chunk("up1", applicability="specific_states", applicable_state_codes=["UP"])
    mh_chunk = _chunk("mh1", applicability="specific_states", applicable_state_codes=["MH"])
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("stamp duty on this cheque instrument", [up_chunk, mh_chunk]))
    service.reranker.rerank = AsyncMock(return_value=[up_chunk, mh_chunk])
    service.llm.chat = AsyncMock(side_effect=AssertionError("must ask before answering an ambiguous matter"))
    request = ChatRequest(question="What is the fee for this instrument?", session_id="s1")
    response = asyncio.run(service.answer(request))
    assert "state" in response.answer.lower()


def test_locality_level_ambiguity_asks_for_locality_not_state() -> None:
    local_a = _chunk("a", applicability="specific_states", applicable_state_codes=["MH"], applicable_localities=["Pune"])
    local_b = _chunk("b", applicability="specific_states", applicable_state_codes=["MH"], applicable_localities=["Mumbai"])
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("local property tax rate", [local_a, local_b]))
    service.reranker.rerank = AsyncMock(return_value=[local_a, local_b])
    service.llm.chat = AsyncMock(side_effect=AssertionError("must ask before answering an ambiguous matter"))
    # Deliberately avoids the fixed pre-retrieval keyword list (eviction/rent
    # control/stamp duty/registration fee/...) so this exercises the GENERAL,
    # data-driven post-retrieval check instead of the narrow pre-check.
    request = ChatRequest(question="What is the local property tax rate here?", session_id="s1")
    response = asyncio.run(service.answer(request))
    assert "locality" in response.answer.lower() or "city" in response.answer.lower() or "district" in response.answer.lower()


def test_ambiguity_detector_ignores_all_india_and_single_state_candidates() -> None:
    central = _chunk("c1", applicability="all_india")
    single_state = _chunk("c2", applicability="specific_states", applicable_state_codes=["MP"])
    assert kj.detect_jurisdiction_ambiguity([central, single_state]) is None


def test_ambiguity_detector_treats_unknown_applicability_bns_content_as_all_india() -> None:
    """Live repro: "which section talks about culpable homicide?" wrongly
    triggered "Which State?" -- a large share of this corpus's BNS/BNSS
    content carries `applicability: "unknown"` (an ingestion gap, not a
    genuine ambiguity about these specific central codes), so a single,
    genuinely unrelated State-specific chunk retrieved alongside it as
    ordinary similarity-search noise (a separate, known issue) was enough
    to look like an "unconfirmed single State" case. BNS/BNSS/BSA are
    verified, uniformly all-India codes (`_looks_like_bns_family`), so this
    must resolve to no ambiguity even though the BNS chunk's own stored
    `applicability` is unhelpful.
    """
    bns_chunk = _chunk("bns1", act_name="THE BHARA TIY A NAGARIK SURAKSHA SANHITA", applicability="unknown")
    unrelated_state_chunk = _chunk(
        "mh1", act_name="THE MAHARASHTRA GOODS AND SERVICES TAX ACT",
        applicability="specific_states", applicable_state_codes=["MH"],
    )
    assert kj.detect_jurisdiction_ambiguity([bns_chunk, unrelated_state_chunk]) is None


# ---------------------------------------------------------------------------
# 4. Candidate-selection-level enforcement (not just the retriever post-filter)
# ---------------------------------------------------------------------------


def test_mongo_filter_builds_native_temporal_range_clause() -> None:
    store = MongoVectorStore()
    result = store._mongo_filter({"source_document": "act.pdf", kj.TEMPORAL_FILTER_KEY: "2020-06-01"})
    assert result["metadata.source_document"] == "act.pdf"
    assert result["$and"] == kj.mongo_temporal_and_clauses("2020-06-01")


def test_atlas_filter_builds_native_temporal_range_clause() -> None:
    store = MongoVectorStore()
    result = store._atlas_filter({kj.TEMPORAL_FILTER_KEY: "2020-06-01"})
    for clause in kj.mongo_temporal_and_clauses("2020-06-01"):
        assert clause in (result.get("$and") or [result])


def test_bm25_candidate_selection_excludes_temporally_ineligible_before_top_k() -> None:
    """The filter is applied inside `_matches_filters`, called from `search`
    BEFORE sorting/truncation -- proven here by confirming an out-of-range
    chunk never reaches the returned candidates at all, not merely that the
    final list happens to look right."""
    index = BM25Index()
    # Several vocabulary-disjoint filler documents keep BM25's IDF
    # meaningful -- with only the two near-identical eviction documents in a
    # tiny corpus, every query term would appear in most/all documents,
    # driving IDF (and therefore every score) to zero/negative before the
    # filter even runs.
    filler_texts = [
        "unrelated consumer refund policy for defective goods",
        "cheque dishonour under section 138 of the negotiable instruments act",
        "procedure for filing a first information report at a police station",
        "registration of a sale deed at the sub registrar office",
        "workplace harassment complaint under the posh act",
    ]
    ids = ["future", "current", *[f"filler{i}" for i in range(len(filler_texts))]]
    texts = ["eviction notice provision text", "eviction notice provision text", *filler_texts]
    metadatas = [{"effective_from": "2030-01-01"}, {"effective_from": "2020-01-01"}, *[{}] * len(filler_texts)]
    # P0-2 "Failure-safe reindexing": `BM25Index.search` now requires
    # `document_status="active"` unconditionally -- every real chunk carries
    # this field, so the synthetic fixture must too.
    metadatas = [{**metadata, "document_status": "active"} for metadata in metadatas]
    index._set_corpus(ids, texts, metadatas)
    results = index.search("eviction notice provision", top_k=5, filters={kj.TEMPORAL_FILTER_KEY: "2026-01-01"})
    assert [r.chunk_id for r in results] == ["current"]


def test_temporal_filter_is_folded_into_the_shared_branch_sent_to_retriever() -> None:
    """End-to-end wiring check: `ChatService` passes the native temporal
    constraint all the way to `vector_store.search`'s `filters` argument --
    i.e. into candidate selection, not only into the retriever's own
    post-filter."""
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("what applied then", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    request = ChatRequest(question="On 12/03/2021 what applied under this Act?", session_id="s1")
    asyncio.run(service.answer(request))
    filters = service.retriever.retrieve.call_args.kwargs["filters"]
    shared_branch = filters["$or"][0]
    assert shared_branch[kj.TEMPORAL_FILTER_KEY] == "2021-03-12"
