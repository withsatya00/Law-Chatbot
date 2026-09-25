"""QA pass 2026-09-24 (T026/T027/T028 multi-turn grounding):
`LegalCitationEngine.validate_grounding` regression coverage for the
per-Act section-scoping fix -- see that method's own comment for the live
false-rejection this addresses (a correct, well-grounded "Section 18 of the
Bombay Rents... Act" answer discarded because two OTHER, unrelated retrieved
Acts in the same candidate pool happened to carry real section-number
metadata that the answer's own Act did not).
"""

from app.rag.citation import LegalCitationEngine
from app.schemas.common import RetrievedChunk, SourceCitation

_CHUNKS = [RetrievedChunk(chunk_id="c1", text="irrelevant for this check", score=0.5, metadata={})]


def _engine() -> LegalCitationEngine:
    return LegalCitationEngine()


def test_validate_grounding_accepts_answer_when_its_own_act_has_no_tagged_section() -> None:
    citations = [
        SourceCitation(act_name="ORISSA ACT", section="26", source_document="orissa.pdf"),
        SourceCitation(act_name="BOMBAY ACT", section=None, source_document="bombay.pdf"),
        SourceCitation(act_name="Chapter V of Transfer of Property Act", section="44", source_document="top.pdf"),
    ]
    answer = (
        "Bombay Rents, Hotel and Lodging House Rates Control Act, 1947 ke mutabiq, Section 18 ke tahat aap "
        "deposit wapas maang sakte hain."
    )

    is_grounded, reason = _engine().validate_grounding(answer, citations, _CHUNKS)

    assert is_grounded is True
    assert reason is None


def test_validate_grounding_still_rejects_a_genuinely_mismatched_section_for_the_same_act() -> None:
    # The fix scopes the comparison to the NAMED Act's own citations -- it
    # must not disable the check entirely. A section number that conflicts
    # with THAT Act's own tagged section is still a real mismatch.
    citations = [
        SourceCitation(act_name="BOMBAY ACT", section="40", source_document="bombay.pdf"),
    ]
    answer = "Bombay Rents, Hotel and Lodging House Rates Control Act, 1947 ke Section 18 ke tahat..."

    is_grounded, reason = _engine().validate_grounding(answer, citations, _CHUNKS)

    assert is_grounded is False
    assert reason is not None


def test_validate_grounding_uses_full_pool_when_answer_names_no_act() -> None:
    # No Act named clearly enough to narrow the comparison -- falls back to
    # the original, unscoped behavior exactly as before this fix.
    citations = [
        SourceCitation(act_name="ORISSA ACT", section="26", source_document="orissa.pdf"),
    ]
    answer = "Under Section 99, you may recover the amount."

    is_grounded, reason = _engine().validate_grounding(answer, citations, _CHUNKS)

    assert is_grounded is False
    assert reason is not None


def test_validate_grounding_accepts_matching_section_unchanged() -> None:
    citations = [
        SourceCitation(act_name="BOMBAY ACT", section="18", source_document="bombay.pdf"),
    ]
    answer = "Bombay Rents, Hotel and Lodging House Rates Control Act, 1947 ke Section 18 ke tahat..."

    is_grounded, reason = _engine().validate_grounding(answer, citations, _CHUNKS)

    assert is_grounded is True
    assert reason is None
