"""Jurisdiction Routing (Phase 2): applicability/version eligibility at
retrieval time (`app.rag.kb_jurisdiction.jurisdiction_or_branches`,
`chunk_matches_jurisdiction`, `chunk_temporally_eligible`,
`filter_by_matter_context`), plus `LegalRetriever.retrieve`'s wiring of the
post-filter and the BM25 array-containment fix it depends on.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from app.rag import kb_jurisdiction as kj
from app.rag.bm25_index import BM25Index
from app.rag.retriever import LegalRetriever
from app.schemas.common import RetrievedChunk


def _chunk(chunk_id: str, **metadata: Any) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text="text", score=0.9, metadata=metadata)


# ---------------------------------------------------------------------------
# jurisdiction_or_branches / chunk_matches_jurisdiction
# ---------------------------------------------------------------------------


def test_mp_matter_excludes_unrelated_up_state_law() -> None:
    mp_chunk = _chunk("c1", applicability="specific_states", applicable_state_codes=["MP"])
    up_chunk = _chunk("c2", applicability="specific_states", applicable_state_codes=["UP"])
    assert kj.chunk_matches_jurisdiction(mp_chunk.metadata, ["MP"], None) is True
    assert kj.chunk_matches_jurisdiction(up_chunk.metadata, ["MP"], None) is False


def test_central_and_state_provisions_can_both_appear() -> None:
    central = _chunk("c1", applicability="all_india")
    state = _chunk("c2", applicability="specific_states", applicable_state_codes=["MP"])
    results = kj.filter_by_matter_context([central, state], ["MP"], None, None)
    assert {c.chunk_id for c in results} == {"c1", "c2"}


def test_unknown_applicability_is_never_an_unrestricted_fallback() -> None:
    unknown = _chunk("c1", applicability="unknown")
    assert kj.chunk_matches_jurisdiction(unknown.metadata, ["MP"], None) is False


def test_multi_state_matter_matches_either_state_not_forced_into_one() -> None:
    up_chunk = _chunk("c1", applicability="specific_states", applicable_state_codes=["UP"])
    mh_chunk = _chunk("c2", applicability="specific_states", applicable_state_codes=["MH"])
    ka_chunk = _chunk("c3", applicability="specific_states", applicable_state_codes=["KA"])
    results = kj.filter_by_matter_context([up_chunk, mh_chunk, ka_chunk], ["UP", "MH"], None, None)
    assert {c.chunk_id for c in results} == {"c1", "c2"}


def test_locality_branch_only_matches_when_locality_given() -> None:
    local_chunk = _chunk("c1", applicability="specific_states", applicable_localities=["Pune"])
    assert kj.chunk_matches_jurisdiction(local_chunk.metadata, ["MH"], None) is False
    assert kj.chunk_matches_jurisdiction(local_chunk.metadata, ["MH"], "Pune") is True


def test_no_state_resolved_applies_no_jurisdiction_constraint() -> None:
    assert kj.jurisdiction_or_branches([], None) == []
    state_only_chunk = _chunk("c1", applicability="specific_states", applicable_state_codes=["MH"])
    assert kj.filter_by_matter_context([state_only_chunk], [], None, None) == [state_only_chunk]


# ---------------------------------------------------------------------------
# chunk_temporally_eligible -- historical/current/future-effective
# ---------------------------------------------------------------------------


def test_future_effective_provision_excluded_from_current_law() -> None:
    future = _chunk("c1", effective_from="2027-01-01")
    assert kj.chunk_temporally_eligible(future.metadata, "2026-09-05") is False


def test_current_provision_with_no_end_date_is_eligible_today() -> None:
    current = _chunk("c1", effective_from="2020-01-01")
    assert kj.chunk_temporally_eligible(current.metadata, "2026-09-05") is True


def test_historical_question_retrieves_the_version_in_force_on_that_date() -> None:
    old_version = _chunk("c1", effective_from="2010-01-01", effective_to="2019-12-31")
    new_version = _chunk("c2", effective_from="2020-01-01", effective_to=None)
    assert kj.chunk_temporally_eligible(old_version.metadata, "2015-06-01") is True
    assert kj.chunk_temporally_eligible(new_version.metadata, "2015-06-01") is False
    assert kj.chunk_temporally_eligible(old_version.metadata, "2022-01-01") is False
    assert kj.chunk_temporally_eligible(new_version.metadata, "2022-01-01") is True


def test_missing_temporal_metadata_never_guessed_either_way() -> None:
    """Objective: "Dates/relationships insufficient hon toh...guess mat
    karo" -- absence of effective_from/effective_to is never treated as
    proof of eligibility OR ineligibility beyond what's actually known."""
    undated = _chunk("c1")
    assert kj.chunk_temporally_eligible(undated.metadata, "2026-09-05") is True
    assert kj.chunk_temporally_eligible(undated.metadata, "2015-01-01") is True


def test_old_versions_are_filtered_not_deleted() -> None:
    """A historical query about 2015 correctly excludes the 2020+ version
    from RESULTS -- `filter_by_matter_context` never mutates or drops the
    chunk from the corpus, only from this one query's eligible set."""
    old_version = _chunk("c1", effective_from="2010-01-01", effective_to="2019-12-31")
    new_version = _chunk("c2", effective_from="2020-01-01")
    results = kj.filter_by_matter_context([old_version, new_version], [], None, "2015-06-01")
    assert [c.chunk_id for c in results] == ["c1"]


def test_missing_temporal_coverage_returns_empty_not_a_wrong_version() -> None:
    """A historical date earlier than ANY version's effective_from means no
    verified version covers it -- the filter must return empty, never fall
    back to the current version as if it applied retroactively."""
    only_version = _chunk("c1", effective_from="2020-01-01")
    results = kj.filter_by_matter_context([only_version], [], None, "2005-01-01")
    assert results == []


def test_zero_results_never_silently_drops_the_filter() -> None:
    """Every candidate is state-mismatched -- the correct outcome is an
    EMPTY list, never a fallback to the unfiltered candidates."""
    wrong_state = _chunk("c1", applicability="specific_states", applicable_state_codes=["KA"])
    results = kj.filter_by_matter_context([wrong_state], ["MP"], None, None)
    assert results == []


# ---------------------------------------------------------------------------
# BM25Index._matches_filters -- array-containment fix
# ---------------------------------------------------------------------------


def test_bm25_matches_filters_treats_list_valued_metadata_as_membership() -> None:
    index = BM25Index()
    metadata = {"applicable_state_codes": ["MP", "UP"]}
    assert index._matches_filters(metadata, {"applicable_state_codes": "MP"}) is True
    assert index._matches_filters(metadata, {"applicable_state_codes": "KA"}) is False


def test_bm25_matches_filters_scalar_metadata_unaffected() -> None:
    index = BM25Index()
    assert index._matches_filters({"source_document": "bns.pdf"}, {"source_document": "bns.pdf"}) is True
    assert index._matches_filters({"source_document": "bns.pdf"}, {"source_document": "bnss.pdf"}) is False


# ---------------------------------------------------------------------------
# LegalRetriever.retrieve -- uniform enforcement across every internal path,
# private-document isolation preserved, cache keyed by matter context.
# ---------------------------------------------------------------------------


class _FakeEmbeddings:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] for _ in texts]


def test_retrieve_applies_matter_filter_regardless_of_which_internal_path_matched() -> None:
    """Simulates the section-number-floor fallback path surfacing an
    out-of-state chunk alongside the normal fusion leg -- both must be
    filtered uniformly by the SAME post-filter before `retrieve` returns."""
    mp_chunk = _chunk("mp1", section_number="12", applicability="specific_states", applicable_state_codes=["MP"])
    up_chunk = _chunk("up1", section_number="12", applicability="specific_states", applicable_state_codes=["UP"])

    vector_store = AsyncMock()
    vector_store.search = AsyncMock(return_value=[mp_chunk, up_chunk])
    vector_store.find_named_section = AsyncMock(return_value=[])
    vector_store.find_constitution_article = AsyncMock(return_value=[])

    retriever = LegalRetriever(embeddings=_FakeEmbeddings(), vector_store=vector_store)
    _rewritten, results = asyncio.run(
        retriever.retrieve(
            "Section 12", top_k=5, matter_context={"state_codes": ["MP"], "locality": None, "as_of_date": None},
        )
    )
    assert [c.chunk_id for c in results] == ["mp1"]


def test_retrieve_private_document_metadata_is_never_jurisdiction_filtered() -> None:
    """A private per-user chunk has no `applicability` field at all --
    `matter_context` is a SHARED-corpus concept only; `retrieve`'s post-filter
    must never be applied to a caller that never resolved a State (verifies
    the empty-matter_context no-op path private retrieval relies on)."""
    private_chunk = _chunk("priv1", owner_user_id="user-1")
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(return_value=[private_chunk])
    retriever = LegalRetriever(embeddings=_FakeEmbeddings(), vector_store=vector_store)
    _rewritten, results = asyncio.run(retriever.retrieve("my document", top_k=5, matter_context=None))
    assert [c.chunk_id for c in results] == ["priv1"]


def test_retrieval_cache_key_differs_by_resolved_state() -> None:
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(return_value=[])
    retriever = LegalRetriever(embeddings=_FakeEmbeddings(), vector_store=vector_store)
    asyncio.run(retriever.retrieve("what about eviction", top_k=5, matter_context={"state_codes": ["MP"], "locality": None, "as_of_date": None}))
    asyncio.run(retriever.retrieve("what about eviction", top_k=5, matter_context={"state_codes": ["UP"], "locality": None, "as_of_date": None}))
    # Each distinct matter context must cause its OWN vector search (a shared
    # cache entry would mean the second call short-circuits before reaching
    # `vector_store.search` at all).
    assert vector_store.search.await_count == 2
