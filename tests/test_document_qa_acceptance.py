"""PDF Q&A acceptance pass (2026-09-12).

The requirement: upload a PDF, ask a SPECIFIC question about it, get an
answer grounded only in THAT document with a supporting page/passage, a
clear "not in this document" when the information genuinely is not there,
follow-up continuity ("explain this clause") on the same PDF, no mixing of
sources when switching between two uploaded PDFs, and no cross-session/
cross-user leakage.

Before this pass there was no `DocumentQuestionWorkflow` at all -- every
other document workflow (`DocumentSummaryWorkflow`, `RiskyClauseWorkflow`,
etc.) reports a FIXED shape regardless of what was actually asked, and a
question naming no analyze/summarize/review verb fell through to plain RAG,
which mixes the shared knowledge base with every document the session/
account owns, with no per-document scoping at all.
"""

import asyncio
from typing import Any

import pytest

import app.chatops.orchestrator  # noqa: F401 - registers every workflow
from app.chatops.base import WorkflowTurn
from app.chatops.orchestrator import ChatOrchestrator
from app.chatops.workflows import documents
from app.core.exceptions import ForbiddenError
from app.services.document_insight import AnsweredQuestion, DocumentChunk, LoadedDocument

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


def _document(
    document_id: str, filename: str, *, pages: bool = True,
    text: str = "The security deposit is Rs. 50,000, refundable within 15 days of vacating.",
) -> LoadedDocument:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    return LoadedDocument(
        document_id=document_id,
        filename=filename,
        chunks=[
            DocumentChunk(text=paragraph, page_number=(index if pages else None))
            for index, paragraph in enumerate(paragraphs, start=1)
        ],
    )


def _two_document_memory() -> dict[str, Any]:
    return {
        "uploaded_documents": [
            {"document_id": "doc-a", "filename": "rent-agreement.pdf"},
            {"document_id": "doc-b", "filename": "loan-contract.pdf"},
        ],
        "last_uploaded_document_id": "doc-b",
    }


def test_answer_is_grounded_only_in_the_selected_document(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_ids: list[str] = []
    asked_questions: list[str] = []

    async def _load(document_id, user_id, session_id):
        loaded_ids.append(document_id)
        return _document(document_id, "rent-agreement.pdf")

    async def _answer(document, question, language):
        asked_questions.append(question)
        assert document.document_id == "doc-a", "must answer from the SELECTED document only"
        return AnsweredQuestion(
            found=True, answer="The deposit is Rs. 50,000.",
            quote="The security deposit is Rs. 50,000, refundable within 15 days of vacating.", page=1,
        )

    monkeypatch.setattr(documents.document_insight, "load_document", _load)
    monkeypatch.setattr(documents.document_insight, "answer_question", _answer)

    memory: dict[str, Any] = {"uploaded_documents": [{"document_id": "doc-a", "filename": "rent-agreement.pdf"}]}
    turn = _orchestrate("What is the security deposit mentioned in this agreement?", memory, **_USER)

    assert turn is not None and turn.status == "completed"
    assert loaded_ids == ["doc-a"]
    assert asked_questions == ["What is the security deposit mentioned in this agreement?"]
    assert "Rs. 50,000" in turn.message
    assert "page 1" in turn.message.lower()
    assert "rent-agreement.pdf" in turn.message


def test_information_not_in_the_document_is_stated_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _load(document_id, user_id, session_id):
        return _document(document_id, "rent-agreement.pdf")

    async def _answer(document, question, language):
        return AnsweredQuestion(found=False)

    monkeypatch.setattr(documents.document_insight, "load_document", _load)
    monkeypatch.setattr(documents.document_insight, "answer_question", _answer)

    memory: dict[str, Any] = {"uploaded_documents": [{"document_id": "doc-a", "filename": "rent-agreement.pdf"}]}
    turn = _orchestrate("Does this agreement mention a parking space?", memory, **_USER)

    assert turn is not None and turn.status == "completed"
    assert "couldn't find that" in turn.message.lower()
    assert "rent-agreement.pdf" in turn.message
    # Must never be dressed up as a real, grounded answer.
    assert "rs." not in turn.message.lower()


def test_a_failed_answering_call_is_distinct_from_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """`answer_question` returning `None` (the LLM call itself failed/timed
    out) must never be presented as "not in this document" -- one is a fact
    about the document, the other is a transient failure the user should
    retry."""
    async def _load(document_id, user_id, session_id):
        return _document(document_id, "rent-agreement.pdf")

    async def _answer(document, question, language):
        return None

    monkeypatch.setattr(documents.document_insight, "load_document", _load)
    monkeypatch.setattr(documents.document_insight, "answer_question", _answer)

    memory: dict[str, Any] = {"uploaded_documents": [{"document_id": "doc-a", "filename": "rent-agreement.pdf"}]}
    turn = _orchestrate("What is the security deposit in this agreement?", memory, **_USER)

    assert turn is not None and turn.status == "completed"
    assert "did not respond in time" in turn.message.lower()
    assert "couldn't find that" not in turn.message.lower()


def test_followup_after_answering_one_document_stays_on_the_same_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """"Explain this clause" (no document named) must resolve to whichever
    document the LAST question was actually answered from, not force a
    "which document?" re-ask and not silently drift to a different upload --
    the workflow itself is popped off the stack the moment it finishes, so
    nothing else would otherwise remember which document was in focus.
    """
    loaded_ids: list[str] = []

    async def _load(document_id, user_id, session_id):
        loaded_ids.append(document_id)
        return _document(document_id, "rent-agreement.pdf" if document_id == "doc-a" else "loan-agreement.pdf")

    async def _answer(document, question, language):
        return AnsweredQuestion(found=True, answer="Two months' notice is required.", quote="two months", page=2)

    monkeypatch.setattr(documents.document_insight, "load_document", _load)
    monkeypatch.setattr(documents.document_insight, "answer_question", _answer)

    memory: dict[str, Any] = _two_document_memory()
    memory["last_uploaded_document_id"] = "doc-a"  # doc-a is the one just discussed

    first = _orchestrate("What is the notice period in this agreement?", memory, **_USER)
    assert first is not None and "rent-agreement.pdf" in first.message

    second = _orchestrate("Explain this clause.", memory, **_USER)
    assert second is not None and second.status == "completed"
    assert "rent-agreement.pdf" in second.message
    assert loaded_ids == ["doc-a", "doc-a"], "the follow-up must stay on the SAME document, not re-ask or drift"


def test_switching_to_a_named_document_does_not_mix_with_the_other(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicitly naming the second document must resolve to it immediately
    (not ask "which one?" again), and the answer must come from ONLY that
    document -- never a blend of both.
    """
    loaded_ids: list[str] = []

    async def _load(document_id, user_id, session_id):
        loaded_ids.append(document_id)
        return _document(document_id, "rent-agreement.pdf" if document_id == "doc-a" else "loan-contract.pdf")

    async def _answer(document, question, language):
        assert document.document_id == "doc-b"
        return AnsweredQuestion(found=True, answer="The interest rate is 9% per annum.", quote="9% per annum", page=3)

    monkeypatch.setattr(documents.document_insight, "load_document", _load)
    monkeypatch.setattr(documents.document_insight, "answer_question", _answer)

    memory: dict[str, Any] = _two_document_memory()
    memory["last_uploaded_document_id"] = "doc-a"  # was discussing doc-a; switching explicitly to doc-b now

    turn = _orchestrate("What is the interest rate in this loan contract?", memory, **_USER)

    assert turn is not None and turn.status == "completed"
    assert loaded_ids == ["doc-b"]
    assert "loan-contract.pdf" in turn.message
    assert "rent-agreement.pdf" not in turn.message
    # Switching updates which document later implicit follow-ups stick to.
    assert memory["last_uploaded_document_id"] == "doc-b"


def test_ambiguous_switch_attempt_asks_rather_than_silently_defaulting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gap `selection.overlapping` closes: naming a document ambiguously
    (both uploaded files happen to share a word) must ask which one is
    meant, never silently fall back to whichever was last discussed and
    risk answering from the WRONG document. Confirmed live before this fix:
    "the loan agreement" against ["rent-agreement.pdf", "loan-agreement.pdf"]
    silently resolved to whichever document was last active, ignoring that
    the user was plainly trying to switch to the other one.
    """
    memory: dict[str, Any] = {
        "uploaded_documents": [
            {"document_id": "doc-a", "filename": "rent-agreement.pdf"},
            {"document_id": "doc-b", "filename": "loan-agreement.pdf"},
        ],
        "last_uploaded_document_id": "doc-a",
    }
    turn = _orchestrate("What is the interest rate in this loan agreement?", memory, **_USER)
    assert turn is not None
    assert turn.missing_field == "document_id"
    assert "which document" in turn.message.lower()


def test_ambiguous_first_question_with_two_documents_asks_rather_than_guessing() -> None:
    """With no prior context (`last_uploaded_document_id` unset/stale) and
    no document named in the message, the workflow must ask which document
    rather than silently picking one."""
    memory: dict[str, Any] = {
        "uploaded_documents": [
            {"document_id": "doc-a", "filename": "rent-agreement.pdf"},
            {"document_id": "doc-b", "filename": "loan-agreement.pdf"},
        ],
    }
    turn = _orchestrate("What is the notice period in this agreement?", memory, **_USER)
    assert turn is not None
    assert turn.missing_field == "document_id"
    assert "which document" in turn.message.lower()


def test_a_scanned_document_with_no_page_evidence_still_answers_but_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _load(document_id, user_id, session_id):
        return _document(document_id, "scanned-notice.pdf", pages=False)

    async def _answer(document, question, language):
        assert not document.has_page_evidence
        return AnsweredQuestion(found=True, answer="The response period is 7 days.", quote="7 days", page=None)

    monkeypatch.setattr(documents.document_insight, "load_document", _load)
    monkeypatch.setattr(documents.document_insight, "answer_question", _answer)

    memory: dict[str, Any] = {"uploaded_documents": [{"document_id": "doc-a", "filename": "scanned-notice.pdf"}]}
    turn = _orchestrate("What is the response period in this notice?", memory, **_USER)

    assert turn is not None and turn.status == "completed"
    assert "7 days" in turn.message
    assert any("no page numbers" in warning.lower() for warning in turn.warnings)


def test_a_document_belonging_to_someone_else_is_refused_before_it_is_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _load(document_id, user_id, session_id):
        raise ForbiddenError("You do not have access to this document.")

    answered = False

    async def _answer(document, question, language):
        nonlocal answered
        answered = True
        return AnsweredQuestion(found=True, answer="leaked", quote="", page=None)

    monkeypatch.setattr(documents.document_insight, "load_document", _load)
    monkeypatch.setattr(documents.document_insight, "answer_question", _answer)

    memory: dict[str, Any] = {"uploaded_documents": [{"document_id": "doc-a", "filename": "someone-elses.pdf"}]}
    turn = _orchestrate("What is the notice period in this agreement?", memory, **_USER)

    assert turn is not None and turn.status == "forbidden"
    assert not answered, "the document's content must never reach the answering step without access"
    assert "someone-elses.pdf" not in turn.message
