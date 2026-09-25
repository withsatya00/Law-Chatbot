"""Phase 2 Milestone E: source currency and amendment handling.

The rule this module defends: **repealed law is never silently presented as
current, and current law is never silently presented as settled when nobody has
verified it.**

The BNS, BNSS and BSA took effect on 1 July 2024 and are not retrospective, so
"what is the punishment for cheating" and "what was the punishment when it
happened in 2019" have genuinely different correct answers. The system must
prefer current law by default, keep legacy law reachable, and say when the
answer depends on the incident date rather than quietly choosing.
"""

from app.rag.reranker import LegalReranker
from app.rag.statute_currency import (
    asks_about_a_past_incident,
    currency_notice,
    status_ranking_adjustment,
)
from app.schemas.common import RetrievedChunk


def _chunk(chunk_id: str, text: str, score: float, **metadata: object) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, score=score, metadata=metadata)


# ---------------------------------------------------------------------------
# Ranking adjustment
# ---------------------------------------------------------------------------


def test_in_force_and_verified_sources_are_preferred() -> None:
    assert status_ranking_adjustment({"amendment_status": "in_force"}) > 0
    assert status_ranking_adjustment(
        {"amendment_status": "in_force", "verification_status": "verified"}
    ) > status_ranking_adjustment({"amendment_status": "in_force"})


def test_repealed_and_superseded_sources_are_pushed_down() -> None:
    assert status_ranking_adjustment({"amendment_status": "repealed"}) < 0
    assert status_ranking_adjustment({"amendment_status": "superseded"}) < 0


def test_a_rejected_source_is_penalised_hardest() -> None:
    """A rejected source is one a reviewer looked at and refused."""
    rejected = status_ranking_adjustment({"verification_status": "rejected"})
    assert rejected < status_ranking_adjustment({"amendment_status": "repealed"})


def test_an_ungoverned_chunk_is_treated_as_neutral() -> None:
    """Most of the existing corpus carries no governance metadata. Penalising
    it would rank by how much review has happened, not by relevance."""
    assert status_ranking_adjustment({}) == 0.0
    assert status_ranking_adjustment({"source_document": "bns.pdf"}) == 0.0
    assert status_ranking_adjustment({"verification_status": "unverified"}) == 0.0


def test_the_adjustment_is_a_preference_not_a_filter() -> None:
    """A repealed provision is still the correct answer for an incident that
    happened while it was in force, so it must stay reachable. The adjustment
    is therefore bounded well below what relevance contributes."""
    assert abs(status_ranking_adjustment({"amendment_status": "repealed"})) < 0.25


async def test_current_law_outranks_repealed_law_on_a_near_tie() -> None:
    reranker = LegalReranker()
    chunks = [
        _chunk(
            "ipc", "420. Cheating and dishonestly inducing delivery of property.", 0.60,
            source_document="ipc.pdf", act_name="Indian Penal Code",
            amendment_status="repealed", verification_status="verified",
        ),
        _chunk(
            "bns", "318. Cheating and dishonestly inducing delivery of property.", 0.60,
            source_document="bns.pdf", act_name="Bharatiya Nyaya Sanhita",
            amendment_status="in_force", verification_status="verified",
        ),
    ]
    ranked = await reranker.rerank("punishment for cheating", chunks, top_k=2)
    assert ranked[0].chunk_id == "bns"


async def test_repealed_law_remains_retrievable() -> None:
    """Down-ranked, not excluded. A user asking about a 2019 offence needs it."""
    reranker = LegalReranker()
    chunks = [
        _chunk(
            "ipc", "420. Cheating under the Indian Penal Code.", 0.7,
            source_document="ipc.pdf", amendment_status="repealed",
        ),
    ]
    ranked = await reranker.rerank("cheating under the IPC", chunks, top_k=5)
    assert [chunk.chunk_id for chunk in ranked] == ["ipc"]
    assert ranked[0].score > 0


# ---------------------------------------------------------------------------
# Incident-date dependence
# ---------------------------------------------------------------------------


def test_a_present_tense_question_is_not_treated_as_a_past_incident() -> None:
    for question in (
        "What is the punishment for cheating?",
        "How do I file an FIR?",
        "dhokhadhadi ki saza kya hai",
    ):
        assert asks_about_a_past_incident(question) is False, question


def test_a_dated_incident_is_detected() -> None:
    for question in (
        "My case was filed in 2019, what happens now?",
        "The offence occurred in 2021 under which section?",
        "This happened before July 2024, does the BNS apply?",
        "It is an old case, which law applies?",
        "My pending trial started earlier, which code governs it?",
    ):
        assert asks_about_a_past_incident(question) is True, question


def test_a_dated_question_produces_a_retrospectivity_warning() -> None:
    notice = currency_notice(
        [{"source_document": "bns.pdf", "amendment_status": "in_force", "verification_status": "verified"}],
        question="My case was filed in 2019, which section applies?",
    )
    assert "1 July 2024" in notice
    assert "retrospectively" in notice
    assert "depends on when the incident occurred" in notice


def test_a_present_tense_question_gets_no_retrospectivity_warning() -> None:
    notice = currency_notice(
        [{"source_document": "bns.pdf", "amendment_status": "in_force", "verification_status": "verified"}],
        question="What is the punishment for cheating?",
    )
    assert "retrospectively" not in notice


# ---------------------------------------------------------------------------
# Disclosure
# ---------------------------------------------------------------------------


def test_a_repealed_source_is_never_presented_as_current() -> None:
    notice = currency_notice(
        [{"source_document": "ipc.pdf", "amendment_status": "repealed", "verification_status": "verified"}]
    )
    assert "ipc.pdf" in notice
    assert "repealed or superseded" in notice
    assert "not as current law" in notice


def test_a_superseded_source_is_disclosed_the_same_way() -> None:
    notice = currency_notice(
        [{"source_document": "crpc.pdf", "amendment_status": "superseded", "verification_status": "verified"}]
    )
    assert "crpc.pdf" in notice
    assert "repealed or superseded" in notice


def test_an_unknown_status_is_disclosed_rather_than_assumed_current() -> None:
    notice = currency_notice(
        [{"source_document": "faq.pdf", "amendment_status": "unknown", "verification_status": "verified"}]
    )
    assert "not recorded" in notice


def test_an_unverified_source_is_disclosed() -> None:
    notice = currency_notice(
        [{"source_document": "bns.pdf", "amendment_status": "in_force", "verification_status": "unverified"}]
    )
    assert "not been verified" in notice
    assert "issuing authority" in notice


def test_a_verified_in_force_source_produces_no_noise() -> None:
    """A clean record should not generate a warning; otherwise every answer
    carries one and readers stop reading them."""
    notice = currency_notice(
        [{"source_document": "bns.pdf", "amendment_status": "in_force", "verification_status": "verified"}]
    )
    assert notice == ""


def test_no_sources_produces_no_notice() -> None:
    assert currency_notice([]) == ""


def test_the_notice_never_asserts_a_source_is_current() -> None:
    """The registry records claims; it does not establish them. No wording here
    may tell a reader a source IS the current law."""
    for status in ("in_force", "amended", "unknown", "repealed", "superseded"):
        for verification in ("verified", "unverified", "pending_review", "rejected"):
            notice = currency_notice(
                [{"source_document": "x.pdf", "amendment_status": status, "verification_status": verification}]
            )
            lowered = notice.lower()
            assert "is the current law" not in lowered
            assert "guaranteed" not in lowered
            assert "up to date" not in lowered


# ---------------------------------------------------------------------------
# Response wiring
# ---------------------------------------------------------------------------


def test_the_chat_response_exposes_currency_notice_and_defaults_empty() -> None:
    from app.schemas.chat import ChatResponse
    from app.schemas.common import LawyerRecommendation

    response = ChatResponse(
        answer="a", sources=[], confidence=0.5,
        lawyer_recommendation=LawyerRecommendation(category="c", confidence=0.5, reason="r"),
        detected_language="english", detected_intent="General", latency_ms=1.0,
    )
    assert response.currency_notice == ""


def test_the_service_helper_builds_a_notice_from_its_citations() -> None:
    from app.schemas.common import SourceCitation
    from app.services.chat_service import _currency_notice_for

    notice = _currency_notice_for(
        [SourceCitation(source_document="ipc.pdf", amendment_status="repealed", verification_status="verified")],
        "what is the punishment for cheating",
    )
    assert "ipc.pdf" in notice
    assert "repealed or superseded" in notice
