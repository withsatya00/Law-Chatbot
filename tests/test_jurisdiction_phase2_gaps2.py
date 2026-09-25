"""Jurisdiction Routing (Phase 2) -- second round of follow-up gap fixes:

1. a single surviving historical version is not silently presented as
   conclusively governing (`kb_jurisdiction.detect_version_ambiguity` +
   `matter_context.version_ambiguity_note`);
2. a locality clarification reply is captured into structured `MatterContext.
   locality` and actually reaches the next retrieval's jurisdiction filter,
   not just acknowledged;
3. the missing-state ambiguity backstop is checked against the WIDE,
   pre-rerank candidate pool, so reranking/relevance-filtering narrowing the
   list to one State cannot silently defeat it.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from test_chat_service_routing import _service_with_mocks

from app.rag import kb_jurisdiction as kj
from app.rag import matter_context as mc
from app.schemas.chat import ChatRequest
from app.schemas.common import RetrievedChunk


def _chunk(chunk_id: str, **metadata: Any) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text="text", score=0.9, metadata=metadata)


# ---------------------------------------------------------------------------
# 1. Version ambiguity is disclosed/downgraded, not silently resolved
# ---------------------------------------------------------------------------


def test_detect_version_ambiguity_flags_two_overlapping_versions_of_one_section() -> None:
    v1 = _chunk("v1", source_document="act.pdf", section_number="12", effective_from="2015-01-01", effective_to=None)
    v2 = _chunk("v2", source_document="act.pdf", section_number="12", effective_from="2018-01-01", effective_to=None)
    result = kj.detect_version_ambiguity([v1, v2])
    assert result is not None
    assert len(result) == 2


def test_detect_version_ambiguity_is_none_for_a_single_version() -> None:
    v1 = _chunk("v1", source_document="act.pdf", section_number="12", effective_from="2015-01-01")
    assert kj.detect_version_ambiguity([v1]) is None


def test_detect_version_ambiguity_ignores_different_sections() -> None:
    """Two DIFFERENT sections each having their own single version is not
    ambiguity -- only multiple versions of the SAME provision are."""
    a = _chunk("a", source_document="act.pdf", section_number="12", effective_from="2015-01-01")
    b = _chunk("b", source_document="act.pdf", section_number="20", effective_from="2018-01-01")
    assert kj.detect_version_ambiguity([a, b]) is None


def test_historical_answer_with_version_ambiguity_gets_a_limitation_note_and_lower_confidence() -> None:
    v1 = RetrievedChunk(
        chunk_id="v1", text="Notice period is thirty days.", score=0.9,
        metadata={"source_document": "act.pdf", "act_name": "Sample Act", "section_number": "12", "effective_from": "2015-01-01"},
    )
    v2 = RetrievedChunk(
        chunk_id="v2", text="Notice period is thirty days.", score=0.9,
        metadata={"source_document": "act.pdf", "act_name": "Sample Act", "section_number": "12", "effective_from": "2018-01-01"},
    )
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("notice period on 12/03/2020", [v1, v2]))
    service.reranker.rerank = AsyncMock(return_value=[v1, v2])
    from app.llm.base import LLMResponse

    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Notice period is thirty days.", model="t", provider="t"))
    request = ChatRequest(question="On 12/03/2020 what was the notice period under this Act?", session_id="s1")
    response = asyncio.run(service.answer(request))
    assert response.confidence <= 0.35  # "Low" (`_confidence_label`'s <0.35) -- the stronger, KB-evidenced case
    assert "not" in response.answer.lower() and ("verif" in response.answer.lower() or "confirm" in response.answer.lower())


def test_historical_answer_with_only_one_surviving_version_still_gets_a_limitation_and_capped_confidence() -> None:
    """Objective: the limitation trigger is NOT "multiple versions found" --
    a single surviving version is not guaranteed applicable either, since
    this KB has no transition/savings metadata to confirm that either way.
    The disclosure text and the confidence cap must both fire here too, not
    only when `detect_version_ambiguity` finds a literal conflict."""
    v1 = RetrievedChunk(
        chunk_id="v1", text="Notice period is thirty days under this Act.", score=0.9,
        metadata={"source_document": "act.pdf", "act_name": "Sample Act", "section_number": "12", "effective_from": "2015-01-01"},
    )
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("notice period on 12/03/2020", [v1]))
    service.reranker.rerank = AsyncMock(return_value=[v1])
    from app.llm.base import LLMResponse

    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="Notice period is thirty days under this Act.", model="t", provider="t")
    )
    request = ChatRequest(question="On 12/03/2020 what was the notice period under this Act?", session_id="s1")
    response = asyncio.run(service.answer(request))
    # Capped short of "High" (`_confidence_label`'s >=0.6), but not pushed as
    # low as the genuine multi-version-conflict case above.
    assert 0.35 < response.confidence <= 0.55
    assert "not" in response.answer.lower() and ("verif" in response.answer.lower() or "confirm" in response.answer.lower())


def test_current_law_question_is_not_subject_to_the_historical_confidence_cap() -> None:
    """No `as_of_date` resolved at all -- an ordinary current-law question
    must not be penalized by either historical-answer cap."""
    v1 = RetrievedChunk(
        chunk_id="v1", text="An FIR is a First Information Report.", score=0.9,
        metadata={"source_document": "CrPC", "act_name": "Code of Criminal Procedure", "section_number": "154"},
    )
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [v1]))
    service.reranker.rerank = AsyncMock(return_value=[v1])
    from app.llm.base import LLMResponse

    service.llm.chat = AsyncMock(return_value=LLMResponse(content="An FIR is a First Information Report.", model="t", provider="t"))
    request = ChatRequest(question="What is FIR?", session_id="s1")
    response = asyncio.run(service.answer(request))
    assert "transition" not in response.answer.lower() and "savings" not in response.answer.lower()


# ---------------------------------------------------------------------------
# 2. Locality clarification reply is captured and reused
# ---------------------------------------------------------------------------


def test_locality_reply_resolves_into_matter_context_locality() -> None:
    memory = {
        "matter_context": {
            "state_codes": [], "locality": None, "as_of_date": None,
            "context_source": None, "legal_category": "Property Law",
        }
    }
    result = mc.resolve_matter_context("Pune", memory, None, "Property Law", awaiting_locality_clarification=True)
    assert result.locality == "Pune"
    assert result.context_source == mc.CONTEXT_SOURCE_EXPLICIT
    assert result.needs_clarification is False


def test_locality_reply_persists_and_the_next_turn_carries_it_forward() -> None:
    """Simulates the two-turn exchange end to end using the real
    `ConversationMemoryStore`-shaped dict a caller would persist: the
    locality survives into the SAME matter_context slot future turns read."""
    memory: dict[str, Any] = {}
    first = mc.resolve_matter_context(
        "What is the local property tax rate here?", memory, None, "Property Law", awaiting_locality_clarification=False,
    )
    memory["matter_context"] = first.to_memory()  # nothing resolved yet -- would trigger the ambiguity backstop
    reply = mc.resolve_matter_context("Pune", memory, None, "Property Law", awaiting_locality_clarification=True)
    memory["matter_context"] = reply.to_memory()
    assert memory["matter_context"]["locality"] == "Pune"

    followup = mc.resolve_matter_context("What about the exemption limit?", memory, None, "Property Law")
    assert followup.locality == "Pune"
    assert followup.context_source == mc.CONTEXT_SOURCE_CONFIRMED


def test_locality_only_context_reaches_the_jurisdiction_filter() -> None:
    """Objective: a resolved locality must actually narrow retrieval, not
    just be acknowledged and dropped -- state_codes is empty here on
    purpose (the user answered "which locality?" without naming a State)."""
    branches = kj.jurisdiction_or_branches([], "Pune")
    assert {"applicability": "specific_states", "applicable_localities": "Pune"} in branches
    pune_chunk = _chunk("a", applicability="specific_states", applicable_localities=["Pune"])
    mumbai_chunk = _chunk("b", applicability="specific_states", applicable_localities=["Mumbai"])
    survivors = kj.filter_by_matter_context([pune_chunk, mumbai_chunk], [], "Pune", None)
    assert [c.chunk_id for c in survivors] == ["a"]


def test_locality_answer_service_level_uses_the_saved_locality_in_retrieval_filters() -> None:
    """End-to-end: after a locality reply, the VERY NEXT `answer()` call
    passes that locality into the filters handed to `retriever.retrieve`."""
    service = _service_with_mocks()
    memory_state = {
        "matter_context": {
            "state_codes": [], "locality": None, "as_of_date": None,
            "context_source": None, "legal_category": "Property Law",
        },
        "pending_clarification": "matter_jurisdiction_locality",
        "messages": [], "summary": "", "current_intent": None, "legal_category": "Property Law",
    }
    service.memory.append = AsyncMock(return_value=memory_state)
    service.retriever.retrieve = AsyncMock(return_value=("local property tax", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    request = ChatRequest(question="Pune", session_id="s1")
    asyncio.run(service.answer(request))
    filters = service.retriever.retrieve.call_args.kwargs["filters"]
    shared_branch = filters["$or"][0]
    assert {"applicability": "specific_states", "applicable_localities": "Pune"} in shared_branch["$or"]


# ---------------------------------------------------------------------------
# 3. The ambiguity backstop is checked against the WIDE candidate pool
# ---------------------------------------------------------------------------


def test_ambiguity_backstop_survives_reranking_narrowing_to_one_state() -> None:
    """Retrieval returns two conflicting-State candidates, but reranking
    (mocked to simulate a real reranker favoring one phrasing) keeps only
    ONE of them -- the ambiguity must still be caught, because it's checked
    against the retrieval-stage pool, not the post-rerank one."""
    up_chunk = _chunk("up1", applicability="specific_states", applicable_state_codes=["UP"])
    mh_chunk = _chunk("mh1", applicability="specific_states", applicable_state_codes=["MH"])
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("stamp duty on this instrument", [up_chunk, mh_chunk]))
    # Reranking keeps only ONE candidate -- exactly the scenario that would
    # defeat a check performed against `ranked` alone.
    service.reranker.rerank = AsyncMock(return_value=[up_chunk])
    service.llm.chat = AsyncMock(side_effect=AssertionError("must ask before answering an ambiguous matter"))
    request = ChatRequest(question="What is the fee for this instrument?", session_id="s1")
    response = asyncio.run(service.answer(request))
    assert "state" in response.answer.lower()


def test_single_unconfirmed_state_with_no_all_india_fallback_still_asks() -> None:
    """A conflict between States is not the only failure mode: exactly ONE
    State's specific-applicability candidate, with no all-India provision to
    fall back on and no State ever resolved for this matter, STILL assumes
    the user is in that one State if answered directly -- "conflicting
    candidates na milna jurisdiction clear hone ka proof nahi hai". Absence
    of a detected conflict must not be read as "jurisdiction is settled"."""
    mh_chunk = _chunk("mh1", applicability="specific_states", applicable_state_codes=["MH"])
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("stamp duty on this instrument", [mh_chunk]))
    service.reranker.rerank = AsyncMock(return_value=[mh_chunk])
    service.llm.chat = AsyncMock(side_effect=AssertionError("must ask before answering an unconfirmed single-State matter"))
    request = ChatRequest(question="What is the fee for this instrument?", session_id="s1")
    response = asyncio.run(service.answer(request))
    assert "which state" in response.answer.lower()


def test_single_state_candidate_alongside_all_india_coverage_does_not_ask() -> None:
    """The SAME single-State candidate is fine when an all-India provision
    is ALSO available -- there is a general answer to fall back on, so
    nothing here silently assumes the user is in that one State."""
    central_chunk = _chunk("central1", applicability="all_india")
    mh_chunk = _chunk("mh1", applicability="specific_states", applicable_state_codes=["MH"])
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("stamp duty on this instrument", [central_chunk, mh_chunk]))
    service.reranker.rerank = AsyncMock(return_value=[central_chunk, mh_chunk])
    from app.llm.base import LLMResponse

    service.llm.chat = AsyncMock(return_value=LLMResponse(content="The fee depends on the instrument value.", model="t", provider="t"))
    request = ChatRequest(question="What is the fee for this instrument?", session_id="s1")
    response = asyncio.run(service.answer(request))
    assert "which state" not in response.answer.lower()


def test_untagged_legacy_content_never_triggers_the_ambiguity_backstop() -> None:
    """Content with no `applicability` at all (the vast majority of today's
    corpus, indexed before Phase 1) must not spuriously trigger this signal
    -- that gap is the fixed category+keyword pre-check's job, not this
    data-driven one, which only activates once genuinely jurisdiction-tagged
    content is present."""
    untagged_chunk = _chunk("legacy1", source_document="bns.pdf")
    assert kj.detect_jurisdiction_ambiguity([untagged_chunk]) is None
