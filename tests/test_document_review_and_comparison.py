"""Milestone E: document-type review checklists and two-document comparison.

The claim these defend is a narrow one and it is the whole point: the review
never says a clause is absent unless the document extracted well enough to
support that, and the comparison never reaches a document the conversation
does not own.
"""

import asyncio
from typing import Any

import pytest

import app.chatops.orchestrator  # noqa: F401  -- registers every workflow
from app.chatops.base import WorkflowTurn
from app.chatops.orchestrator import ChatOrchestrator
from app.core.exceptions import ForbiddenError
from app.services import document_comparison, document_review
from app.services.document_insight import DocumentChunk, LoadedDocument

_USER = {"authenticated_user_id": "u1", "claims": {"sub": "u1", "role": "user"}}


def _orchestrate(message: str, memory: dict[str, Any] | None = None, **kwargs) -> WorkflowTurn | None:
    return asyncio.run(
        ChatOrchestrator().handle_turn(
            session_id="s1",
            message=message,
            language=kwargs.pop("language", "english"),
            memory=memory if memory is not None else {},
            **kwargs,
        )
    )


def _document(text: str, *, document_id: str = "doc-1", filename: str = "agreement.pdf", pages: bool = True) -> LoadedDocument:
    """One chunk per paragraph, numbered from page 1 when `pages` is set."""
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    return LoadedDocument(
        document_id=document_id,
        filename=filename,
        chunks=[
            DocumentChunk(text=paragraph, page_number=(index if pages else None))
            for index, paragraph in enumerate(paragraphs, start=1)
        ],
    )


_RENT_AGREEMENT = """RENT AGREEMENT

This Rent Agreement is made between Shri Ramesh Gupta (Lessor) and Smt. Anita Rao (Lessee).

The premises situated at Flat No. 4B, Sunrise Apartments, Pune 411004 are let for residential use.

The term of this agreement is eleven months commencing from 01/04/2026.

The Lessee shall pay a monthly rent of Rs. 25,000 payable on or before the 5th of each month.

A security deposit of Rs. 1,50,000 has been paid and is refundable on vacating, subject to deductions for damage.

Either party may terminate this agreement by giving two months' prior notice in writing.

The Lessor shall have the right to enter and inspect the premises without prior notice.

The courts at Pune shall have exclusive jurisdiction over any dispute arising from this agreement.

Signed and delivered by the parties in the presence of the witness below.
"""

_REVISED_RENT_AGREEMENT = """RENT AGREEMENT

This Rent Agreement is made between Shri Ramesh Gupta (Lessor) and Smt. Anita Rao (Lessee).

The premises situated at Flat No. 4B, Sunrise Apartments, Pune 411004 are let for residential use.

The term of this agreement is eleven months commencing from 01/04/2026.

The Lessee shall pay a monthly rent of Rs. 32,000 payable on or before the 5th of each month.

A security deposit of Rs. 1,50,000 has been paid and is non-refundable under any circumstances.

Either party may terminate this agreement by giving two months' prior notice in writing.

The Lessor shall have the right to enter and inspect the premises without prior notice.

The courts at Mumbai shall have exclusive jurisdiction over any dispute arising from this agreement.

The Lessee shall indemnify the Lessor against all claims arising from use of the premises.

Signed and delivered by the parties in the presence of the witness below.
"""


# ---------------------------------------------------------------------------
# 1. Type detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This Lease Deed between the Lessor and the Lessee for the premises...", "rental_agreement"),
        ("This Employment Agreement sets out the probation and CTC of the employee.", "employment_agreement"),
        ("This Non-Disclosure Agreement between the Disclosing Party and Receiving Party.", "nda"),
        ("This Service Agreement sets out the scope of services of the service provider.", "service_agreement"),
        ("This Agreement to Sell is between the vendor and the purchaser for conveyance.", "sale_agreement"),
        ("This Partnership Deed records the profit sharing ratio agreed by the partners hereby.", "partnership_deed"),
        ("LEGAL NOTICE under section 138. My client instructs me through my client.", "legal_notice"),
        ("AFFIDAVIT. I solemnly affirm as the deponent, verified at Pune.", "affidavit"),
        ("COMPLAINT under section 12. The complainant states before the commission. Prayer follows.", "complaint"),
        ("POWER OF ATTORNEY. I constitute and appoint the attorney holder.", "power_of_attorney"),
    ],
)
def test_all_ten_document_types_are_detected(text: str, expected: str) -> None:
    detected, confidence = document_review.detect_type(text)
    assert detected == expected
    assert confidence > 0.0


def test_an_unrecognised_document_is_not_forced_into_a_checklist() -> None:
    """Running an employment checklist over a sale deed produces a page of
    confident nonsense."""
    detected, confidence = document_review.detect_type("A shopping list: milk, bread, coffee.")
    assert detected == "unknown"
    assert confidence == 0.0


# ---------------------------------------------------------------------------
# 2. Review: present / missing / unable to determine, kept apart
# ---------------------------------------------------------------------------


def test_present_clauses_are_quoted_with_their_page() -> None:
    result = document_review.review(_document(_RENT_AGREEMENT))
    present = {item.key: item for item in result.by_status("present")}
    assert "termination" in present
    assert present["termination"].source_text
    assert present["termination"].source_page is not None


def test_a_genuinely_absent_clause_is_reported_as_missing() -> None:
    result = document_review.review(_document(_RENT_AGREEMENT))
    missing = {item.key for item in result.by_status("missing")}
    # The agreement has no lock-in and no maintenance clause.
    assert "lock_in" in missing
    assert "maintenance" in missing


def test_a_thin_extraction_never_produces_a_missing_claim() -> None:
    """Telling somebody their agreement has no termination clause, when the
    scan failed, is worse than saying nothing."""
    scanned = _document("RENT AGREEMENT between Lessor and Lessee.\n\n%%%% ### ~~~~ ????")
    result = document_review.review(scanned)
    assert result.extraction_quality == "thin"
    assert result.by_status("missing") == []
    assert result.by_status("unable_to_determine")


def test_a_thin_extraction_is_not_reported_as_low_risk() -> None:
    """An unreadable document is not a safe one."""
    scanned = _document("NDA between Disclosing Party and Receiving Party.\n\n#### ???")
    assert document_review.review(scanned).risk_level == "unknown"


def test_risky_wording_is_reported_with_its_page_and_a_plain_reason() -> None:
    result = document_review.review(_document(_REVISED_RENT_AGREEMENT))
    flagged = {item.label for item in result.risky_clauses}
    assert '"non-refundable"' in flagged
    assert '"without prior notice"' in flagged
    for item in result.risky_clauses:
        assert item.question, "a flagged clause with no explanation is just an alarm"
        assert item.source_text


def test_the_review_never_states_a_legal_conclusion() -> None:
    result = document_review.review(_document(_RENT_AGREEMENT))
    assert any("not legal conclusions" in note for note in result.notes)
    for question in result.suggested_questions:
        assert "?" in question or question.endswith("."), question


def test_a_document_with_no_page_numbers_says_so_rather_than_citing_page_one() -> None:
    result = document_review.review(_document(_RENT_AGREEMENT, pages=False))
    assert result.has_page_evidence is False
    assert all(item.source_page is None for item in result.items)
    assert any("no page numbers" in note for note in result.notes)


def test_the_caller_can_declare_the_type_and_that_wins() -> None:
    result = document_review.review(_document(_RENT_AGREEMENT), declared_type="nda")
    assert result.detected_type == "nda"


# ---------------------------------------------------------------------------
# 3. Comparison
# ---------------------------------------------------------------------------


def test_identical_documents_report_no_material_change() -> None:
    old = _document(_RENT_AGREEMENT, document_id="a")
    new = _document(_RENT_AGREEMENT, document_id="b")
    result = document_comparison.compare(old, new)
    assert not [item for item in result.items if item.change_type in {"added", "removed", "changed"}]
    assert "no material differences" in " ".join(result.unresolved)


def test_reformatting_alone_is_not_reported_as_a_change() -> None:
    """A re-typeset document must not report fifteen spurious changes."""
    old = _document(_RENT_AGREEMENT, document_id="a")
    respaced = "\n\n".join(
        "  ".join(paragraph.split()) for paragraph in _RENT_AGREEMENT.split("\n\n")
    )
    new = _document(respaced, document_id="b")
    result = document_comparison.compare(old, new)
    changed = [item.label for item in result.items if item.change_type == "changed"]
    assert changed == []


def test_a_changed_clause_reports_both_values_and_both_pages() -> None:
    old = _document(_RENT_AGREEMENT, document_id="a", filename="old.pdf")
    new = _document(_REVISED_RENT_AGREEMENT, document_id="b", filename="new.pdf")
    result = document_comparison.compare(old, new)
    jurisdiction = next(item for item in result.items if item.aspect == "jurisdiction")
    assert jurisdiction.change_type == "changed"
    assert "Pune" in jurisdiction.old_value
    assert "Mumbai" in jurisdiction.new_value
    assert jurisdiction.old_page is not None
    assert jurisdiction.new_page is not None
    assert result.has_page_evidence


def test_an_added_clause_is_reported_as_added() -> None:
    old = _document(_RENT_AGREEMENT, document_id="a")
    new = _document(_REVISED_RENT_AGREEMENT, document_id="b")
    result = document_comparison.compare(old, new)
    assert "Indemnity" in result.added_clauses


def test_a_removed_clause_is_reported_as_removed() -> None:
    old = _document(_REVISED_RENT_AGREEMENT, document_id="a")
    new = _document(_RENT_AGREEMENT, document_id="b")
    result = document_comparison.compare(old, new)
    assert "Indemnity" in result.removed_clauses


def test_a_payment_change_is_surfaced_with_a_reason() -> None:
    old = _document(_RENT_AGREEMENT, document_id="a")
    new = _document(_REVISED_RENT_AGREEMENT, document_id="b")
    result = document_comparison.compare(old, new)
    payment = next(item for item in result.items if item.aspect == "payment")
    assert payment.change_type == "changed"
    assert payment.risk_note


def test_a_scanned_document_is_flagged_rather_than_silently_half_compared() -> None:
    old = _document(_RENT_AGREEMENT, document_id="a", filename="good.pdf")
    scanned = _document("RENT AGREEMENT\n\n#### ???", document_id="b", filename="scan.pdf")
    result = document_comparison.compare(old, scanned)
    assert any("scan.pdf" in note for note in result.unresolved)


def test_comparison_without_page_numbers_says_so() -> None:
    old = _document(_RENT_AGREEMENT, document_id="a", pages=False)
    new = _document(_REVISED_RENT_AGREEMENT, document_id="b", pages=False)
    assert document_comparison.compare(old, new).has_page_evidence is False


def test_the_summary_names_what_changed() -> None:
    old = _document(_RENT_AGREEMENT, document_id="a")
    new = _document(_REVISED_RENT_AGREEMENT, document_id="b")
    summary = document_comparison.compare(old, new).executive_summary
    assert "changed" in summary
    assert "Jurisdiction" in summary or "Payment" in summary


# ---------------------------------------------------------------------------
# 4. Through chat, with ownership enforced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("compare these two agreements", "document_compare"),
        ("New agreement me kya change hua?", "document_compare"),
        ("Dono contracts ke risky differences batao", "document_compare"),
        ("दोनों की तुलना करो", "document_compare"),
        ("what is missing from my rent agreement", "document_review"),
        ("check this NDA", "document_review"),
    ],
)
def test_review_and_comparison_are_reachable_by_talking(message: str, expected: str) -> None:
    from app.chatops.registry import best_match

    match = best_match(message, "english")
    assert match is not None, f"nothing matched: {message!r}"
    assert match[0].name == expected


def test_comparison_uses_the_two_documents_this_conversation_uploaded(monkeypatch) -> None:
    loaded: list[str] = []

    async def _load(document_id, user_id, session_id):
        loaded.append(document_id)
        text = _RENT_AGREEMENT if document_id == "doc-1" else _REVISED_RENT_AGREEMENT
        return _document(text, document_id=document_id, filename=f"{document_id}.pdf")

    from app.chatops.workflows import documents

    monkeypatch.setattr(documents.document_insight, "load_document", _load)

    memory: dict[str, Any] = {
        "uploaded_documents": [
            {"document_id": "doc-1", "filename": "original.pdf"},
            {"document_id": "doc-2", "filename": "revised.pdf"},
        ]
    }
    turn = _orchestrate("compare these two agreements", memory, **_USER)
    assert turn is not None and turn.status == "completed"
    assert loaded == ["doc-1", "doc-2"]
    assert "Jurisdiction" in turn.message


def test_a_document_belonging_to_someone_else_is_refused_before_it_is_read(monkeypatch) -> None:
    from app.chatops.workflows import documents

    async def _load(document_id, user_id, session_id):
        raise ForbiddenError("You do not have access to this document.")

    monkeypatch.setattr(documents.document_insight, "load_document", _load)

    memory: dict[str, Any] = {
        "uploaded_documents": [
            {"document_id": "doc-1", "filename": "mine.pdf"},
            {"document_id": "doc-2", "filename": "theirs.pdf"},
        ]
    }
    turn = _orchestrate("compare these two agreements", memory, **_USER)
    assert turn is not None and turn.status == "forbidden"
    assert "theirs.pdf" not in turn.message


def test_comparison_needs_two_documents_and_asks_rather_than_guessing() -> None:
    memory: dict[str, Any] = {
        "uploaded_documents": [{"document_id": "doc-1", "filename": "only-one.pdf"}]
    }
    turn = _orchestrate("compare these two agreements", memory, **_USER)
    assert turn is not None
    assert turn.missing_field in {"old_document_id", "new_document_id"}
    assert "both documents" in turn.message or "Which" in turn.message


def test_three_uploads_produce_a_numbered_choice_for_each_side() -> None:
    memory: dict[str, Any] = {
        "uploaded_documents": [
            {"document_id": "doc-1", "filename": "v1.pdf"},
            {"document_id": "doc-2", "filename": "v2.pdf"},
            {"document_id": "doc-3", "filename": "v3.pdf"},
        ]
    }
    turn = _orchestrate("compare the agreements", memory, **_USER)
    assert turn is not None
    assert "earlier version" in turn.message
    assert "v1.pdf" in turn.message
    assert "doc-1" not in turn.message


def test_an_unknown_document_type_is_not_reviewed_against_a_wrong_checklist(monkeypatch) -> None:
    from app.chatops.workflows import documents

    async def _load(document_id, user_id, session_id):
        return _document("A shopping list: milk, bread, coffee. Nothing legal here at all.", document_id=document_id)

    monkeypatch.setattr(documents.document_insight, "load_document", _load)

    memory: dict[str, Any] = {"uploaded_documents": [{"document_id": "doc-1", "filename": "list.pdf"}]}
    turn = _orchestrate("check this agreement for anything missing", memory, **_USER)
    assert turn is not None
    assert "could not tell what kind of document" in turn.message
