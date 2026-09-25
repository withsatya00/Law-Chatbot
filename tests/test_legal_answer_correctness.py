"""Post-Phase-3 hardening, Phase 2 milestone B: cheque-bounce answer
correctness, and honest citation presentation.

What this file is about
-----------------------
An observed session answered "Cheque bounce hone par Negotiable Instruments
Act Section 138 ke tahat kya remedy hai" like this:

    "Legal notice ke guidance ke mutabiq, aapko pehle ek notice bhejna hoga
     jisme cheque ki amount ko notice milne ke 15 din ke andar pay karne ki
     demand ki jaye. Agar 15 din ke andar payment nahi milti, tabhi aap
     Section 138 ke under criminal proceedings initiate kar sakte hain."

The reply named no Act, no section, no source and no verification status. The
authority it *did* name -- "legal notice guidance" -- is this product's own
drafting-template note, which is not law. And it stated exactly one of the
statute's four time limits, omitting the thirty-day window for sending the
notice, the six-month presentation requirement, and the one-month limitation
for the complaint. A reader following it would send the notice on day 40 and
lose the offence.

How this file avoids repeating that mistake
-------------------------------------------
Nothing here quotes the law from memory. `tests/reference/
negotiable_instruments_138_142.json` records, for each proposition, a verbatim
quotation ("anchor") from a bare-Act PDF that is already in this repository's
knowledge base, together with that file's sha256 and the page the quotation is
on. `test_every_recorded_proposition_is_still_in_the_source_file` re-reads the
PDF and re-checks every anchor, so a reference entry that drifts from the file
fails the suite rather than becoming a second, unchecked source of "law".

The retrieved chunks the pipeline is then driven with are built from those same
pages, carrying the same governance metadata the reference records -- including
`verification_status: "pending_review"`, because no human in this repository
has compared the file against the issuing authority's publication. The system
is required to DISCLOSE that, not to paper over it.
"""

import asyncio
import json
import re
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.llm.base import LLMResponse
from app.rag.statutory_periods import unsupported_periods
from app.schemas.chat import ChatRequest
from app.schemas.common import RetrievedChunk
from app.services.chat_service import ChatService

REFERENCE = Path(__file__).parent / "reference" / "negotiable_instruments_138_142.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _reference() -> dict[str, Any]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _page_text(page_number: int) -> str:
    """The extracted text of one page of the recorded source file."""
    pypdf = pytest.importorskip("pypdf")
    pdf_path = PROJECT_ROOT / _reference()["source"]["repository_path"]
    if not pdf_path.exists():  # pragma: no cover - guarded by the test below
        pytest.skip(f"Recorded source file is not present: {pdf_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader = pypdf.PdfReader(str(pdf_path))
        return _normalize(reader.pages[page_number - 1].extract_text())


# ---------------------------------------------------------------------------
# The reference itself
# ---------------------------------------------------------------------------


def test_the_recorded_source_file_is_the_one_that_was_measured() -> None:
    """A different file under the same name is a different source."""
    import hashlib

    source = _reference()["source"]
    pdf_path = PROJECT_ROOT / source["repository_path"]
    if not pdf_path.exists():
        pytest.skip(f"Recorded source file is not present: {pdf_path}")
    assert hashlib.sha256(pdf_path.read_bytes()).hexdigest() == source["sha256"]


def test_every_recorded_proposition_is_still_in_the_source_file() -> None:
    """Each anchor must be a verbatim quotation from the page it names.

    This is the check that keeps the reference honest. Without it the JSON
    would be exactly what rule 5 warns about -- a legal-looking file trusted
    because of what it is called.
    """
    payload = _reference()
    pages = {entry["page_number"] for entry in payload["propositions"]}
    texts = {page: _page_text(page) for page in pages}
    missing = [
        f"{entry['key']} (s.{entry['section']}, page {entry['page_number']})"
        for entry in payload["propositions"]
        if entry["anchor"] not in texts[entry["page_number"]]
    ]
    assert not missing, f"Anchors no longer present in the source file: {missing}"


def test_the_reference_does_not_claim_an_unchecked_verification_status() -> None:
    """Nobody here has compared this PDF against the issuing authority, so it
    must not be recorded as verified (docs/SOURCE_GOVERNANCE.md section 1)."""
    source = _reference()["source"]
    assert source["verification_status"] == "pending_review"
    assert source["url"] is None
    assert source["government_source"] is None
    assert source["human_review_task"]


def test_the_four_statutory_periods_are_each_recorded_separately() -> None:
    """The observed defect was collapsing two different periods into one.

    Six months to present, thirty days to send the notice, fifteen days to
    pay, one month to complain -- four distinct limits, each with its own
    quotation behind it.
    """
    entries = {entry["key"]: entry["anchor"] for entry in _reference()["propositions"]}
    assert "six months" in entries["presentation_window"]
    assert "thirty days" in entries["notice_deadline"]
    assert "fifteen days" in entries["payment_opportunity"]
    assert "one month" in entries["cause_of_action_and_limitation"]
    # And the cause of action is tied to proviso (c), not to the dishonour.
    assert "clause (c) of the proviso to section 138" in entries["cause_of_action_and_limitation"]


def test_propositions_this_source_does_not_establish_are_recorded_as_such() -> None:
    """Civil recovery may well be available; this Act's text does not say so.

    Recording that explicitly is what stops it being quietly asserted as a
    section 138 consequence.
    """
    unsupported = _reference()["not_established_by_this_source"]
    assert any("civil suit" in item["claim"].lower() for item in unsupported)
    assert all(item["why_not_recorded"] for item in unsupported)


# ---------------------------------------------------------------------------
# Driving the real pipeline with those chunks
# ---------------------------------------------------------------------------


class _StubLLM:
    """Returns a fixed answer. Nothing about these tests depends on a model:
    what is under test is what the pipeline does with an answer and its
    sources, which must hold whatever the model wrote."""

    provider_name = "stub"
    model = "stub-model"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[Any] = []

    async def chat(self, messages: list[Any], temperature: float = 0.1) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=self.content, model=self.model, provider=self.provider_name)


def _statute_chunks(*, with_pages: bool = True) -> list[RetrievedChunk]:
    """Chunks built from the recorded pages, with the recorded governance
    metadata. `with_pages=False` models the pre-Phase-2 corpus, where page
    identity was never captured and must therefore never be shown."""
    payload = _reference()
    source = payload["source"]
    chunks: list[RetrievedChunk] = []
    by_page: dict[int, list[dict[str, Any]]] = {}
    for entry in payload["propositions"]:
        by_page.setdefault(entry["page_number"], []).append(entry)
    for index, (page, entries) in enumerate(sorted(by_page.items())):
        metadata: dict[str, Any] = {
            "source_document": source["source_document"],
            "act_name": source["act_name"],
            "section_number": entries[0]["section"],
            "verification_status": source["verification_status"],
            "amendment_status": source["amendment_status"],
        }
        if with_pages:
            metadata["page_number"] = page
            metadata["page_start"] = page
            metadata["page_end"] = page
            metadata["extraction_method"] = source["extraction_method"]
        chunks.append(
            RetrievedChunk(
                chunk_id=f"ni-{page}",
                text=" ".join(entry["anchor"] for entry in entries),
                score=0.9 - (index * 0.05),
                metadata=metadata,
            )
        )
    return chunks


def _service(answer: str, chunks: list[RetrievedChunk]) -> ChatService:
    """A real `ChatService` with only I/O and the model replaced."""
    memory: dict[str, Any] = {}
    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])

    async def _load(session_id: str) -> dict[str, Any]:
        return memory

    async def _append(session_id: str, role: str, content: str) -> dict[str, Any]:
        memory.setdefault("messages", []).append({"role": role, "content": content})
        return memory

    async def _update(session_id: str, **values: Any) -> dict[str, Any]:
        memory.update(values)
        return memory

    service.memory.load = _load
    service.memory.append = _append
    service.memory.update = _update
    service.memory.summarize_if_needed = AsyncMock(return_value=memory)
    service.memory.append_intent_event = AsyncMock(return_value=memory)
    service.history.insert = AsyncMock(return_value="history-id")
    service.query_log.insert = AsyncMock(return_value="query-log-id")
    service.intent_events.insert = AsyncMock(return_value="intent-event-id")
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    service.retriever.retrieve = AsyncMock(return_value=("", list(chunks)))
    service.reranker.rerank = AsyncMock(return_value=list(chunks))
    service.llm = _StubLLM(answer)
    return service


_QUESTION = "What remedy does Section 138 of the Negotiable Instruments Act give for a bounced cheque?"

# An answer that says only what the recorded quotations establish, with each
# period attached to the obligation it actually governs.
_FAITHFUL_ANSWER = (
    "Under Section 138 of the Negotiable Instruments Act, 1881, an offence arises where a cheque "
    "drawn on an account is returned by the bank unpaid for insufficiency of funds or because it "
    "exceeds the arranged amount. Three conditions in the proviso must all be met: the cheque must "
    "have been presented within six months of the date it was drawn or within its validity period, "
    "whichever is earlier; the payee must make a written demand for payment within thirty days of "
    "receiving the bank's information that the cheque was returned unpaid; and the drawer must fail "
    "to pay within fifteen days of receiving that notice. The cause of action arises on the expiry "
    "of that fifteen-day window. Under Section 142(1)(b) the complaint must be made within one "
    "month of that date, though the court may take cognizance later if sufficient cause is shown. "
    "Section 142(1)(a) requires the complaint to be in writing by the payee or holder in due "
    "course, and Section 142(1)(c) requires it to be tried by a Metropolitan Magistrate or a "
    "Judicial Magistrate of the first class. Which court has territorial jurisdiction depends on "
    "how the cheque was presented, which is not stated in your question."
)

# The shape of the observed failure: a period the source does not state.
_INVENTED_DEADLINE_ANSWER = (
    "Under Section 138 of the Negotiable Instruments Act, 1881 you must send a legal notice, and "
    "the complaint has to be filed within 45 days of the dishonour."
)


def _ask(service: ChatService, question: str = _QUESTION):
    return asyncio.run(service.answer(ChatRequest(question=question, session_id="s-138")))


def test_the_answer_carries_the_act_and_both_sections_from_metadata() -> None:
    """`applicable_law` is built from citation metadata, never parsed out of
    the prose -- so this asserts the SOURCES identified the provisions, not
    that the model happened to name them."""
    response = _ask(_service(_FAITHFUL_ANSWER, _statute_chunks()))
    assert "Negotiable Instruments Act, 1881 — Section 138" in response.applicable_law
    assert "Negotiable Instruments Act, 1881 — Section 142" in response.applicable_law


def test_the_answer_shows_the_real_page_evidence() -> None:
    response = _ask(_service(_FAITHFUL_ANSWER, _statute_chunks()))
    pages = {page.page_number for page in response.evidence_pages}
    recorded = {entry["page_number"] for entry in _reference()["propositions"]}
    assert pages == recorded
    assert all(page.extraction_method == "embedded_text" for page in response.evidence_pages)


def test_a_source_with_no_page_identity_never_gets_a_guessed_page() -> None:
    """Rule: a page number is the page's real number, or it is absent."""
    response = _ask(_service(_FAITHFUL_ANSWER, _statute_chunks(with_pages=False)))
    assert response.evidence_pages == []
    # The citation itself survives -- no page evidence is not no source.
    assert "Negotiable Instruments Act, 1881 — Section 138" in response.applicable_law


def test_the_unverified_status_of_the_source_is_disclosed() -> None:
    """The file has not been checked against the issuing authority. Saying so
    is the whole point of recording it as `pending_review`."""
    response = _ask(_service(_FAITHFUL_ANSWER, _statute_chunks()))
    notice = response.currency_notice
    assert "have not been verified" in notice
    assert "amendment status" in notice
    assert any(citation.verification_status == "pending_review" for citation in response.sources)


def test_a_faithful_answer_raises_no_time_limit_warning() -> None:
    """Every period in it is in the retrieved text, attached to its own rule."""
    response = _ask(_service(_FAITHFUL_ANSWER, _statute_chunks()))
    assert not [warning for warning in response.warnings if "time limit" in warning]
    assert "Unverified time limit" not in response.answer


def test_a_deadline_the_source_does_not_state_is_marked_not_asserted() -> None:
    """The answer is not deleted -- it may be right about everything else --
    but the number is flagged in the prose and in `warnings`."""
    response = _ask(_service(_INVENTED_DEADLINE_ANSWER, _statute_chunks()))
    assert "Unverified time limit" in response.answer
    assert any("45 days" in warning for warning in response.warnings)


def test_the_period_check_distinguishes_one_month_from_thirty_days() -> None:
    """s.142(1)(b) says ONE MONTH. An answer that says "30 days" is stating a
    period that provision does not, and converting units would hide that.

    Scoped to the limitation quotation alone: the full s.138 text separately
    says "thirty days" (for sending the notice), which would legitimately
    support a 30-day claim -- this check is about the unit, not about whether
    the number appears somewhere else in the Act.
    """
    limitation = next(
        entry["anchor"]
        for entry in _reference()["propositions"]
        if entry["key"] == "cause_of_action_and_limitation"
    )
    assert "one month" in limitation
    stated = unsupported_periods(
        "The complaint must be filed within 30 days of the cause of action.", [limitation]
    )
    assert [claim.describe() for claim in stated] == ["30 days"]
    # And the correct wording is accepted against the same text.
    assert unsupported_periods("The complaint must be filed within one month.", [limitation]) == []


def test_a_period_the_user_narrated_is_not_treated_as_a_deadline() -> None:
    """"the cheque bounced two months ago" is the user's fact, not a rule."""
    source_text = " ".join(chunk.text for chunk in _statute_chunks())
    assert unsupported_periods("The cheque bounced two months ago.", [source_text]) == []


def test_without_adequate_material_the_system_declines_instead_of_answering() -> None:
    """No chunks means no answer -- not an answer from model memory, and not a
    citation-shaped structure with nothing behind it."""
    from app.core.constants import is_no_verified_context

    response = _ask(_service(_FAITHFUL_ANSWER, []))
    assert is_no_verified_context(response.answer)
    assert response.applicable_law == []
    assert response.evidence_pages == []
    assert response.sources == []
    assert response.confidence == 0.0


def test_the_capability_overview_does_not_promise_a_citation_on_every_answer() -> None:
    """It used to say "I cite the Act and section behind every answer". A
    source whose metadata records neither is a normal, honest outcome, and the
    overview must not describe it as a malfunction."""
    from app.core.constants import CAPABILITY_OVERVIEW_MESSAGES, capability_overview

    for language in CAPABILITY_OVERVIEW_MESSAGES:
        text = capability_overview(language, 55)
        assert "every answer" not in text.lower()
        assert "har jawab ke saath act aur section batata hoon" not in text.lower()
        assert "हर जवाब के साथ" not in text


# ---------------------------------------------------------------------------
# Found live, against the running backend and the real index
# ---------------------------------------------------------------------------


def test_a_sentence_fragment_is_never_presented_as_an_act_name() -> None:
    """Observed live. Asked the transcript's own cheque-bounce question against
    the real index, the answer's cited provisions came back as:

        Negotiable Instruments Act - Section 138           (correct)
        Notwithstanding anything contained in the Code - Section 138
        THE CODE - Section 138
        Repealing and Amending Act - Section 138.

    The middle two are ingestion artefacts -- a subordinate clause and a
    truncated heading -- rendered in the citation slot, in `applicable_law`,
    and in the reader-facing source list, where they are indistinguishable
    from the real citation above them.
    """
    from app.rag.citation import usable_act_name

    assert usable_act_name("Notwithstanding anything contained in the Code") is None
    assert usable_act_name("THE CODE") is None
    assert usable_act_name("the said Act") is None
    # And a real title is untouched -- rejecting one of these would remove a
    # correct citation, which is the worse error.
    for real in (
        "Negotiable Instruments Act, 1881",
        "Bharatiya Nyaya Sanhita, 2023",
        "The Constitution of India",
        "Consumer Protection Act, 2019",
        "Transfer of Property Act, 1882",
        "Right to Information Act, 2005",
    ):
        assert usable_act_name(real) == real


def test_a_citation_with_an_unusable_act_name_keeps_its_section_and_document() -> None:
    """Dropping the name must not drop the citation: where the passage came
    from is still true, and the answer is still attributable."""
    from app.rag.citation import citation_from_metadata

    citation = citation_from_metadata(
        {
            "act_name": "Notwithstanding anything contained in the Code",
            "section_number": "138",
            "source_document": "Negotiable_Instruments_Act_1881_Complete_Act.pdf",
        },
        source_document="Negotiable_Instruments_Act_1881_Complete_Act.pdf",
    )
    assert citation.act_name is None
    assert citation.section == "138"
    assert citation.is_identifiable


def test_the_same_provision_is_not_cited_twice_over_a_trailing_full_stop() -> None:
    """Live, one answer cited both "Section 138" and "Section 138." -- the
    second lifted from a heading that ended in a period."""
    from app.rag.citation import citation_from_metadata

    first = citation_from_metadata({"section_number": "138"}, source_document="a.pdf")
    second = citation_from_metadata({"section_number": "138."}, source_document="a.pdf")
    assert first.section == second.section == "138"
