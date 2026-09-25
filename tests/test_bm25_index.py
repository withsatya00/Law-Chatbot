"""Part 40 "Hybrid Legal Retrieval" -- BM25 tokenizer, persisted index, and
Reciprocal Rank Fusion regression suite.

None of this touches MongoDB: `BM25Index._set_corpus` (the same seam
`_rebuild_from_mongo`/`_load_from_disk` funnel through) is used directly to
seed a small synthetic legal corpus, mirroring how the rest of this test
suite avoids real DB/LLM dependencies (see `test_multi_turn_conversations.py`'s
`FakeMemoryStore`).
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from app.rag.bm25_index import BM25Index, tokenize
from app.rag.fusion import reciprocal_rank_fusion
from app.schemas.common import RetrievedChunk

# ---------------------------------------------------------------------------
# Tokenizer (section 3): must keep section numbers, Act names, and Hindi/
# Hinglish terms searchable rather than mangling them.
# ---------------------------------------------------------------------------


def test_tokenize_keeps_section_numbers_intact() -> None:
    assert tokenize("Section 173 BNSS") == ["section", "173", "bnss"]


def test_tokenize_keeps_act_abbreviations_intact() -> None:
    assert tokenize("Section 138 NI Act") == ["section", "138", "ni", "act"]


def test_tokenize_handles_hyphenated_legal_terms() -> None:
    # "e-FIR" splits into ["e", "fir"] rather than being preserved as one
    # token -- but a query for "e-FIR" tokenizes identically, so the match
    # still happens via the shared tokens either side (documented tradeoff).
    assert tokenize("e-FIR") == ["e", "fir"]
    assert tokenize("cheque bounce") == ["cheque", "bounce"]
    assert tokenize("cognizable offence") == ["cognizable", "offence"]


def test_tokenize_keeps_devanagari_words_whole() -> None:
    assert tokenize("एफआईआर क्या होती है") == ["एफआईआर", "क्या", "होती", "है"]


# Regression tests for qa-40q-multilingual-20260921 BUG-03: `_TOKEN_RE` was
# Devanagari-only, so any other Indic/Perso-Arabic script tokenized to `[]`
# and contributed nothing to the BM25 leg of retrieval -- confirmed live with
# a Gurmukhi (Punjabi) "police complaint" question that fell back to "no
# verified document" while the same topic succeeded in Hinglish, English and
# Devanagari Hindi. Every script below must now tokenize to a non-empty list
# instead of silently losing its BM25 leg.
@pytest.mark.parametrize(
    "text",
    [
        "ਪੁਲਿਸ ਨੂੰ ਸ਼ਿਕਾਇਤ",  # Gurmukhi (Punjabi)
        "પોલીસ ફરિયાદ",  # Gujarati
        "காவல் புகார்",  # Tamil
        "పోలీసు ఫిర్యాదు",  # Telugu
        "ಪೊಲೀಸ್ ದೂರು",  # Kannada
        "പോലീസ് പരാതി",  # Malayalam
        "পুলিশ অভিযোগ",  # Bengali / Assamese
        "ପୋଲିସ ଅଭିଯୋଗ",  # Odia
        "پولیس شکایت",  # Perso-Arabic (Urdu/Sindhi/Kashmiri)
    ],
)
def test_tokenize_keeps_every_scheduled_script_whole(text: str) -> None:
    tokens = tokenize(text)
    assert tokens
    assert all(tokens)


# ---------------------------------------------------------------------------
# A small synthetic legal corpus covering the spec's own regression queries
# (section 18, items 1/2/4/5/6/7) -- seeded directly via `_set_corpus`.
# ---------------------------------------------------------------------------

_CORPUS = {
    "sec173_bnss": (
        (
            "173. Information in cognizable cases. Every information relating to the commission of a "
            "cognizable offence shall be recorded by the officer in charge of a police station under the "
            "Bharatiya Nagarik Suraksha Sanhita, and a copy shall be given to the informant. Refusal by "
            "police to register an FIR for a cognizable offence can be challenged before the Superintendent "
            "of Police or by approaching the Magistrate."
        ),
        {"act_name": "Bharatiya Nagarik Suraksha Sanhita", "section_number": "173", "source_document": "bnss.pdf"},
    ),
    "sec138_ni_act": (
        (
            "138. Dishonour of cheque for insufficiency of funds. Where a cheque drawn by a person for "
            "discharge of a debt is returned unpaid, dishonour of the cheque under the Negotiable "
            "Instruments Act constitutes an offence, and the payee may file a criminal complaint after "
            "issuing a legal notice within the prescribed period."
        ),
        {"act_name": "Negotiable Instruments Act", "section_number": "138", "source_document": "ni_act.pdf"},
    ),
    "fir_faq": (
        (
            "What is an FIR? An FIR, or First Information Report, is a written document prepared by police "
            "when they receive information about the commission of a cognizable offence. Filing an e-FIR "
            "online is also possible in several states for certain categories of crime."
        ),
        {"source_type": "faq", "source_document": "police_faq.pdf"},
    ),
    "bail_faq": (
        (
            "What is bail? Bail is the temporary release of an accused person awaiting trial, subject to "
            "conditions set by the court. Anticipatory bail may be sought before arrest under the Bharatiya "
            "Nagarik Suraksha Sanhita."
        ),
        {"source_type": "faq", "source_document": "bail_faq.pdf"},
    ),
    "cognizable_vs_noncognizable": (
        (
            "Difference between cognizable and non-cognizable offence: in a cognizable offence the police "
            "may arrest without a warrant and start investigation without the Magistrate's permission; in a "
            "non-cognizable offence the police cannot arrest without a warrant and require court permission "
            "to investigate."
        ),
        {"source_type": "definition", "source_document": "bnss_commentary.pdf"},
    ),
    "unrelated_pizza": (
        (
            "A classic Margherita pizza recipe uses tomato sauce, fresh mozzarella, and basil leaves baked "
            "at high temperature on a thin crust."
        ),
        {"source_document": "recipes.pdf"},
    ),
}


# The name says it: these tests exercise in-memory search only and never
# persist. It still has to be a real path, and a RELATIVE one dropped a stray
# `unused-in-this-test.pkl` in the repository root on every run -- so it points
# into the system temp directory, where nothing is left behind if a future
# change does start writing.
_UNUSED_CACHE_PATH = Path(tempfile.gettempdir()) / "legalai-bm25-unused-in-this-test.pkl"


def _seeded_index() -> BM25Index:
    index = BM25Index(cache_path=_UNUSED_CACHE_PATH)
    chunk_ids = list(_CORPUS.keys())
    texts = [_CORPUS[cid][0] for cid in chunk_ids]
    # P0-2 "Failure-safe reindexing": `BM25Index.search` now requires
    # `document_status="active"` unconditionally -- every real chunk carries
    # this field, so the synthetic fixture must too.
    metadatas = [{**_CORPUS[cid][1], "document_status": "active"} for cid in chunk_ids]
    index._set_corpus(chunk_ids, texts, metadatas)
    return index


@pytest.mark.parametrize(
    "query,expected_chunk_id",
    [
        ("What is Section 173 BNSS?", "sec173_bnss"),
        ("What is Section 138 of the Negotiable Instruments Act?", "sec138_ni_act"),
        ("fir kya hota hai", "fir_faq"),
        ("what is bail", "bail_faq"),
        ("what is the difference between cognizable and non cognizable offence", "cognizable_vs_noncognizable"),
        ("cheque bounce ke baad kya karna chahiye", "sec138_ni_act"),
    ],
)
def test_bm25_ranks_correct_chunk_highest(query: str, expected_chunk_id: str) -> None:
    index = _seeded_index()
    results = index.search(query, top_k=3)
    assert results, f"expected at least one result for {query!r}"
    assert results[0].chunk_id == expected_chunk_id


def test_bm25_finds_fir_material_for_hinglish_police_refusal_query() -> None:
    """Spec item 3: "Police FIR nhi likh rhi kya kru?" -- BM25 should still
    surface FIR/police-registration material via the shared "police"/"fir"
    tokens, even though most of the sentence is Hinglish filler BM25 has no
    special handling for.
    """
    index = _seeded_index()
    results = index.search("Police FIR nhi likh rhi kya kru?", top_k=3)
    chunk_ids = [chunk.chunk_id for chunk in results]
    assert "fir_faq" in chunk_ids or "sec173_bnss" in chunk_ids


def test_bm25_irrelevant_query_returns_nothing() -> None:
    """Spec item 9: an irrelevant query must never surface random LEGAL
    chunks -- no shared vocabulary with any corpus text means every BM25
    score is zero, and `search()` treats an all-zero score distribution as
    "no result" rather than returning the least-bad match. (Matching the
    corpus's own unrelated pizza-recipe chunk to a pizza query would be
    CORRECT lexical retrieval, so this deliberately queries something with
    zero overlap anywhere in the corpus, legal or not.)
    """
    index = _seeded_index()
    assert index.search("aeroplane turbine engine maintenance schedule", top_k=3) == []


def test_bm25_pizza_query_never_surfaces_a_legal_chunk() -> None:
    """The corpus's own unrelated pizza-recipe chunk legitimately matching a
    pizza query is fine -- what must never happen is a LEGAL chunk (which
    shares no real vocabulary with a recipe) being dragged along with it.
    """
    index = _seeded_index()
    chunk_ids = {chunk.chunk_id for chunk in index.search("How to make pizza?", top_k=3)}
    assert chunk_ids <= {"unrelated_pizza"}


def test_bm25_respects_metadata_filters() -> None:
    index = _seeded_index()
    results = index.search("cognizable offence", top_k=5, filters={"source_document": "bnss_commentary.pdf"})
    assert results
    assert all(chunk.metadata.get("source_document") == "bnss_commentary.pdf" for chunk in results)


# ---------------------------------------------------------------------------
# Part 45 "Per-User Document Isolation": a filter value that's a list/tuple
# means "must be one of these" (membership), used for
# `owner_session_id: [None, session_id]` -- "mine, or nobody's". A chunk
# with no `owner_session_id` key at all (every document indexed before this
# field existed, curated or uploaded) must match a filter list containing
# `None`, the same way MongoDB's `$in: [None, ...]` treats an absent field
# as `null`.
# ---------------------------------------------------------------------------


def _owner_scoped_index() -> BM25Index:
    index = BM25Index(cache_path=_UNUSED_CACHE_PATH)
    chunk_ids = ["owned_by_a", "ownerless_global", "owned_by_b"]
    texts = [
        "Private legal document A discusses a specific rental dispute with landlord Sharma regarding flat 4B.",
        "The FIR process under criminal procedure requires the officer to record cognizable information immediately.",
        "Private legal document B discusses a specific cheque dishonour matter involving payee Verma.",
    ]
    metadatas = [
        {"source_document": "doc_a.pdf", "owner_session_id": "session-A", "document_status": "active"},
        # No `owner_session_id` key -- simulates every pre-existing document.
        # `document_status` IS present though: every real chunk has carried
        # it unconditionally since before ownership scoping existed, unlike
        # ownership itself (see the P0-2 gate this exercises implicitly via
        # `_matches_filters`).
        {"source_document": "bnss_commentary.pdf", "document_status": "active"},
        {"source_document": "doc_b.pdf", "owner_session_id": "session-B", "document_status": "active"},
    ]
    index._set_corpus(chunk_ids, texts, metadatas)
    return index


def test_owner_filter_hides_another_sessions_document() -> None:
    index = _owner_scoped_index()
    results = index.search(
        "rental dispute landlord Sharma flat", top_k=5, filters={"owner_session_id": [None, "session-B"]}
    )
    assert "owned_by_a" not in {chunk.chunk_id for chunk in results}


def test_owner_filter_returns_the_owning_sessions_own_document() -> None:
    index = _owner_scoped_index()
    results = index.search(
        "rental dispute landlord Sharma flat", top_k=5, filters={"owner_session_id": [None, "session-A"]}
    )
    assert "owned_by_a" in {chunk.chunk_id for chunk in results}


def test_owner_filter_keeps_ownerless_documents_visible_to_every_session() -> None:
    index = _owner_scoped_index()
    for session in ("session-A", "session-B", "some-brand-new-session"):
        results = index.search(
            "FIR cognizable information officer procedure", top_k=5, filters={"owner_session_id": [None, session]}
        )
        assert "ownerless_global" in {chunk.chunk_id for chunk in results}


# ---------------------------------------------------------------------------
# Part 46 "Authenticated User Ownership": a reserved `"$or"` key expresses
# "global OR mine (by session, only if no owner_user_id is set) OR mine (by
# authenticated user, from any session)". Uses the exact filter shape
# `ChatService._prepare_rag_context` builds.
# ---------------------------------------------------------------------------


def _user_scoped_index() -> BM25Index:
    index = BM25Index(cache_path=_UNUSED_CACHE_PATH)
    chunk_ids = ["global_chunk", "userA_chunk", "userB_chunk"]
    texts = [
        "The FIR process under criminal procedure requires the officer to record cognizable information immediately.",
        "Private legal document A discusses a specific rental dispute with landlord Sharma regarding flat 4B.",
        "Private legal document B discusses a specific cheque dishonour matter involving payee Verma.",
    ]
    metadatas = [
        {"source_document": "bnss_commentary.pdf", "document_status": "active"},
        {
            "source_document": "doc_a.pdf", "owner_user_id": "user-A", "owner_session_id": "session-A-original",
            "document_status": "active",
        },
        {
            "source_document": "doc_b.pdf", "owner_user_id": "user-B", "owner_session_id": "session-B-original",
            "document_status": "active",
        },
    ]
    index._set_corpus(chunk_ids, texts, metadatas)
    return index


def _owner_or_filter(session_id: str, user_id: str | None) -> dict:
    branches = [{"owner_session_id": [None, session_id], "owner_user_id": [None]}]
    if user_id:
        branches.append({"owner_user_id": [user_id]})
    return {"$or": branches}


def test_user_owned_document_visible_to_its_owner_from_any_session() -> None:
    index = _user_scoped_index()
    # a brand new session, never used to upload the document -- account
    # ownership must survive this, unlike Part 45's session-only mechanism.
    results = index.search(
        "rental dispute landlord Sharma flat", top_k=5,
        filters=_owner_or_filter("brand-new-session", "user-A"),
    )
    assert "userA_chunk" in {chunk.chunk_id for chunk in results}


def test_user_owned_document_never_visible_to_a_different_authenticated_user() -> None:
    index = _user_scoped_index()
    results = index.search(
        "rental dispute landlord Sharma flat", top_k=5,
        filters=_owner_or_filter("brand-new-session", "user-B"),
    )
    assert "userA_chunk" not in {chunk.chunk_id for chunk in results}


def test_user_owned_document_never_leaks_via_a_guessed_matching_session_id() -> None:
    # Hardening beyond what was strictly asked: even if an attacker's
    # CURRENT session_id happens to exactly equal the document's ORIGINAL
    # upload session_id, a document with a real owner_user_id must still
    # only be reachable through the exact user_id match, never through
    # session-guessing (the attacker is unauthenticated here: user_id=None).
    index = _user_scoped_index()
    results = index.search(
        "rental dispute landlord Sharma flat", top_k=5,
        filters=_owner_or_filter("session-A-original", None),
    )
    assert "userA_chunk" not in {chunk.chunk_id for chunk in results}


def test_anonymous_request_never_sees_a_user_owned_document() -> None:
    index = _user_scoped_index()
    results = index.search(
        "rental dispute landlord Sharma flat", top_k=5,
        filters=_owner_or_filter("some-anonymous-session", None),
    )
    assert "userA_chunk" not in {chunk.chunk_id for chunk in results}


def test_global_document_visible_to_authenticated_and_anonymous_alike() -> None:
    index = _user_scoped_index()
    for session_id, user_id in [("s1", "user-A"), ("s2", "user-B"), ("s3", None)]:
        results = index.search(
            "FIR cognizable information officer procedure", top_k=5,
            filters=_owner_or_filter(session_id, user_id),
        )
        assert "global_chunk" in {chunk.chunk_id for chunk in results}


# ---------------------------------------------------------------------------
# Incremental update / delete (section 14): BM25 must track the exact same
# chunk IDs the vector index has, kept in sync through the same hooks.
# ---------------------------------------------------------------------------


def test_add_or_update_chunks_is_reflected_immediately() -> None:
    from app.rag.types import DocumentChunk

    index = _seeded_index()
    assert index.search("landlord security deposit", top_k=3) == []

    new_chunk = DocumentChunk(
        chunk_id="deposit_notice",
        document_id="doc-1",
        text="A tenant may send a legal notice demanding return of the security deposit from the landlord.",
        metadata={"source_document": "tenancy.pdf", "document_status": "active"},
    )
    asyncio.run(index.add_or_update_chunks([new_chunk]))
    results = index.search("landlord security deposit", top_k=3)
    assert results
    assert results[0].chunk_id == "deposit_notice"


def test_remove_by_source_drops_only_that_sources_chunks() -> None:
    index = _seeded_index()
    assert index.chunk_count == len(_CORPUS)
    asyncio.run(index.remove_by_source("recipes.pdf"))
    assert index.chunk_count == len(_CORPUS) - 1
    assert "unrelated_pizza" not in index._chunk_ids
    # Everything else survives untouched.
    assert index.search("Section 173 BNSS", top_k=1)[0].chunk_id == "sec173_bnss"


# ---------------------------------------------------------------------------
# Disk persistence (section 2/13): must survive a process restart without
# rescanning Mongo, and must never be rebuilt mid-query.
# ---------------------------------------------------------------------------


def test_index_persists_to_disk_and_reloads() -> None:
    # A project-local scratch dir rather than pytest's `tmp_path` fixture --
    # this sandbox's Windows temp directory isn't writable here, but the
    # repo's own filesystem is.
    cache_path = Path("storage") / "_test_bm25_index_persistence_test.pkl"
    try:
        index = BM25Index(cache_path=cache_path)
        chunk_ids = list(_CORPUS.keys())
        texts = [_CORPUS[cid][0] for cid in chunk_ids]
        # P0-2 "Failure-safe reindexing": `BM25Index.search` now requires
        # `document_status="active"` unconditionally (mirrors
        # `MongoVectorStore`'s own gate) -- every real chunk carries this
        # field, so the synthetic fixture must too, or every search below
        # would (correctly) find nothing.
        metadatas = [{**_CORPUS[cid][1], "document_status": "active"} for cid in chunk_ids]
        index._set_corpus(chunk_ids, texts, metadatas)
        index._save_to_disk()
        assert cache_path.exists()

        reloaded = BM25Index(cache_path=cache_path)
        assert not reloaded.is_loaded
        asyncio.run(reloaded.ensure_loaded())
        assert reloaded.is_loaded
        assert reloaded.chunk_count == len(_CORPUS)
        assert reloaded.search("Section 173 BNSS", top_k=1)[0].chunk_id == "sec173_bnss"
    finally:
        cache_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (section 7): rank-based, not score-based; graceful
# single-leg degradation (section 17's fallback).
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=f"text for {chunk_id}", score=score, metadata={})


def test_rrf_ranks_chunk_found_by_both_legs_highest() -> None:
    vector_results = [_chunk("a", 0.91), _chunk("b", 0.80), _chunk("c", 0.70)]
    bm25_results = [_chunk("b", 55.0), _chunk("d", 40.0), _chunk("a", 12.0)]
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=60)
    # "b" is rank 2 in vector and rank 1 in BM25 (1/62 + 1/61); "a" is rank 1
    # in vector and rank 3 in BM25 (1/61 + 1/63) -- "b" ekes out "a" despite
    # BM25's raw score for "a" (12.0) being much smaller in absolute terms
    # than vector's raw score for "b" (0.80), which is exactly the point:
    # rank position drives the fusion, not the incomparable raw scales.
    assert fused[0].chunk_id == "b"
    assert {chunk.chunk_id for chunk in fused} == {"a", "b", "c", "d"}


def test_rrf_keeps_single_leg_chunk_with_only_that_legs_contribution() -> None:
    fused = reciprocal_rank_fusion([[_chunk("only_in_vector", 0.5)], []], k=60)
    assert len(fused) == 1
    assert fused[0].chunk_id == "only_in_vector"
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_degrades_to_single_leg_ranking_when_other_leg_is_empty() -> None:
    """Section 17's fallback falls out of RRF for free: if BM25 is
    unavailable, fusing `[vector_results, []]` just reproduces the vector
    leg's own order.
    """
    vector_results = [_chunk("x", 0.9), _chunk("y", 0.5), _chunk("z", 0.2)]
    fused = reciprocal_rank_fusion([vector_results, []], k=60)
    assert [chunk.chunk_id for chunk in fused] == ["x", "y", "z"]


def test_rrf_both_legs_empty_returns_empty() -> None:
    assert reciprocal_rank_fusion([[], []], k=60) == []
