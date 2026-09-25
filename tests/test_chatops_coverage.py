"""Milestone C: every user-facing capability is reachable by talking.

These tests are about COVERAGE and SAFETY of the conversational adapters, not
about the services underneath -- those have their own tests. Services are
stubbed here so a workflow's behaviour (what it asks, what it refuses, what
it calls, and how many times) is what is under test.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest

import app.chatops.orchestrator  # noqa: F401  -- registers every workflow
from app.chatops import state
from app.chatops.base import WorkflowTurn
from app.chatops.orchestrator import ChatOrchestrator
from app.chatops.registry import WORKFLOWS, best_match
from app.core.exceptions import ForbiddenError
from app.core.security import Role


def _orchestrate(message: str, memory: dict[str, Any] | None = None, **kwargs) -> WorkflowTurn | None:
    return asyncio.run(
        ChatOrchestrator().handle_turn(
            session_id=kwargs.pop("session_id", "s1"),
            message=message,
            language=kwargs.pop("language", "english"),
            memory=memory if memory is not None else {},
            **kwargs,
        )
    )


_USER = {"authenticated_user_id": "u1", "claims": {"sub": "u1", "role": "user"}}
_ADMIN = {"authenticated_user_id": "a1", "claims": {"sub": "a1", "role": "admin"}}


class _Draft:
    def __init__(self, draft_id: str, name: str) -> None:
        self.draft_id = draft_id
        self.template_id = "legal_notice"
        self.template_name = name
        self.language = "english"
        self.status = "preview_ready"
        self.created_at = "2026-08-01T00:00:00"


# ---------------------------------------------------------------------------
# 1. The capability matrix is actually covered by registered workflows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        # documents
        "document_summary", "risky_clauses", "document_choose",
        # drafts
        "saved_drafts", "draft_manage", "draft_export",
        # cases
        "case_create", "case_list", "case_manage", "evidence_organize", "lawyer_summary",
        # account
        "preferences", "downloads", "background_jobs", "clear_conversation", "delete_my_data",
        # admin
        "admin_knowledge_base", "admin_analytics", "admin_sources",
        # legal + notarization (pre-existing)
        "cyber_fraud", "jurisdiction",
        "notarization_prepare", "notarization_status", "notarization_verify",
        "notary_queue", "notary_admin",
    ],
)
def test_every_capability_family_has_a_registered_workflow(name: str) -> None:
    assert name in WORKFLOWS


@pytest.mark.parametrize(
    "message,expected",
    [
        # English
        ("summarise this document", "document_summary"),
        ("what are the risky clauses", "risky_clauses"),
        ("delete my draft", "draft_manage"),
        ("start a new case", "case_create"),
        ("show my cases", "case_list"),
        ("add a hearing", "case_manage"),
        ("organise my evidence", "evidence_organize"),
        ("show my preferences", "preferences"),
        ("my downloads", "downloads"),
        ("knowledge base status", "admin_knowledge_base"),
        ("show the unanswered questions", "admin_analytics"),
        ("list the legal sources", "admin_sources"),
        # Hinglish
        ("pdf ka summary do", "document_summary"),
        ("mera draft delete kar do", "draft_manage"),
        ("naya case banao", "case_create"),
        ("meri cases dikhao", "case_list"),
        # Hindi
        ("सबूत व्यवस्थित करो", "evidence_organize"),
        ("ड्राफ्ट हटा दो", "draft_manage"),
    ],
)
def test_capabilities_route_from_plain_language(message: str, expected: str) -> None:
    match = best_match(message, "english")
    assert match is not None, f"nothing matched: {message!r}"
    assert match[0].name == expected


def test_no_chat_workflow_makes_an_outbound_network_call() -> None:
    """No workflow may file or submit anything externally.

    Enforced structurally: an HTTP client imported anywhere in the workflow
    package is the mechanism by which that could happen, so its absence is
    checked rather than argued about.
    """
    package = Path(__file__).resolve().parent.parent / "app" / "chatops"
    for module in package.rglob("*.py"):
        source = module.read_text(encoding="utf-8")
        for banned in ("import httpx", "import requests", "import aiohttp", "urllib.request"):
            assert banned not in source, f"{module.name} imports {banned}"


# ---------------------------------------------------------------------------
# 2. Drafts
# ---------------------------------------------------------------------------


def test_deleting_a_draft_requires_confirmation_and_says_what_is_lost(monkeypatch) -> None:
    from app.chatops.workflows import drafts

    deleted: list[str] = []

    async def _owned(engine, draft_id, user_id, session_id):
        return {"_id": draft_id, "user_id": user_id, "draft_type": "legal_notice"}

    async def _delete(draft_id, user_id, session_id):
        deleted.append(draft_id)
        return {"status": "deleted", "draft_id": draft_id, "versions_deleted": 2}

    monkeypatch.setattr(drafts.draft_management, "owned_draft", _owned)
    monkeypatch.setattr(drafts.draft_management, "delete_draft", _delete)

    memory: dict[str, Any] = {"draft_id": "d1"}
    confirmation = _orchestrate("delete my draft", memory, **_USER)
    assert confirmation is not None
    assert confirmation.status == "awaiting_confirmation"
    assert "cannot be undone" in confirmation.message
    assert deleted == [], "nothing may be deleted before the user says yes"

    done = _orchestrate("yes", memory, **_USER)
    assert done is not None and done.status == "completed"
    assert deleted == ["d1"]


def test_declining_the_delete_confirmation_deletes_nothing(monkeypatch) -> None:
    from app.chatops.workflows import drafts

    async def _owned(engine, draft_id, user_id, session_id):
        return {"_id": draft_id, "user_id": user_id, "draft_type": "legal_notice"}

    async def _delete(draft_id, user_id, session_id):  # pragma: no cover - must not run
        raise AssertionError("delete ran after the user declined")

    monkeypatch.setattr(drafts.draft_management, "owned_draft", _owned)
    monkeypatch.setattr(drafts.draft_management, "delete_draft", _delete)

    memory: dict[str, Any] = {"draft_id": "d1"}
    _orchestrate("delete my draft", memory, **_USER)
    turn = _orchestrate("no", memory, **_USER)
    assert turn is not None and turn.status == "cancelled"


def test_another_users_draft_is_refused_by_the_service_not_the_workflow(monkeypatch) -> None:
    from app.chatops.workflows import drafts

    async def _owned(engine, draft_id, user_id, session_id):
        raise ForbiddenError("You do not have access to this draft.")

    monkeypatch.setattr(drafts.draft_management, "owned_draft", _owned)

    memory: dict[str, Any] = {"draft_id": "someone-elses"}
    _orchestrate("approve my draft", memory, **_USER)
    turn = _orchestrate("yes", memory, **_USER)
    assert turn is not None and turn.status == "forbidden"


def test_several_drafts_produce_a_numbered_choice_not_an_id_prompt(monkeypatch) -> None:
    from app.chatops.workflows import drafts

    async def _history(self, request):
        return [_Draft("d1", "Rent agreement notice"), _Draft("d2", "Consumer complaint")]

    monkeypatch.setattr(drafts.LegalDraftEngine, "history", _history)

    memory: dict[str, Any] = {}
    turn = _orchestrate("show my draft versions", memory, **_USER)
    assert turn is not None
    assert "1." in turn.message and "Rent agreement notice" in turn.message
    assert "d1" not in turn.message, "an internal id must never be shown"
    assert turn.missing_field == "draft_id"


def test_a_numbered_reply_selects_the_right_draft(monkeypatch) -> None:
    from app.chatops.workflows import drafts

    seen: list[str] = []

    async def _history(self, request):
        return [_Draft("d1", "Rent agreement notice"), _Draft("d2", "Consumer complaint")]

    async def _owned(engine, draft_id, user_id, session_id):
        seen.append(draft_id)
        return {"_id": draft_id, "user_id": user_id, "draft_type": "legal_notice"}

    async def _versions(self, draft_id):
        return []

    monkeypatch.setattr(drafts.LegalDraftEngine, "history", _history)
    monkeypatch.setattr(drafts.LegalDraftEngine, "list_versions", _versions)
    monkeypatch.setattr(drafts.draft_management, "owned_draft", _owned)

    memory: dict[str, Any] = {}
    _orchestrate("show my draft versions", memory, **_USER)
    _orchestrate("2", memory, **_USER)
    assert seen == ["d2"]


# ---------------------------------------------------------------------------
# 3. Cases
# ---------------------------------------------------------------------------


def test_case_workflows_require_a_signed_in_user() -> None:
    turn = _orchestrate("show my cases", {}, claims={})
    assert turn is not None and turn.status == "forbidden"
    assert "signed in" in turn.message
    assert "notary" not in turn.message


def test_creating_a_case_asks_one_field_at_a_time_then_creates_it(monkeypatch) -> None:
    from app.chatops.workflows import cases

    created: list[Any] = []

    class _Created:
        case_id = "c1"
        case_number = "CRL/123/2026"
        title = "Sharma property dispute"

    async def _create(self, owner, request):
        created.append(request)
        return _Created()

    monkeypatch.setattr(cases.CaseService, "create", _create)

    memory: dict[str, Any] = {}
    first = _orchestrate("start a new case", memory, **_USER)
    assert first is not None and first.missing_field == "case_number"

    second = _orchestrate("CRL/123/2026", memory, **_USER)
    assert second is not None and second.missing_field == "title"

    done = _orchestrate("Sharma property dispute", memory, **_USER)
    assert done is not None and done.status == "completed"
    assert created[0].case_number == "CRL/123/2026"
    assert created[0].title == "Sharma property dispute"
    assert memory["case_id"] == "c1"


def test_deleting_a_case_requires_confirmation(monkeypatch) -> None:
    from app.chatops.workflows import cases

    deleted: list[str] = []

    async def _delete(self, owner, case_id):
        deleted.append(case_id)

    monkeypatch.setattr(cases.CaseService, "delete", _delete)

    memory: dict[str, Any] = {"case_id": "c1"}
    confirmation = _orchestrate("delete the case", memory, **_USER)
    assert confirmation is not None and confirmation.status == "awaiting_confirmation"
    assert deleted == []

    done = _orchestrate("haan", memory, **_USER)
    assert done is not None and done.status == "completed"
    assert deleted == ["c1"]
    assert "filed or submitted" in done.message


def test_a_hearing_without_a_date_is_asked_for_rather_than_guessed(monkeypatch) -> None:
    from app.chatops.workflows import cases

    async def _add(self, owner, case_id, request):  # pragma: no cover - must not run
        raise AssertionError("a hearing was recorded without a date")

    monkeypatch.setattr(cases.CaseService, "add_hearing", _add)

    turn = _orchestrate("add a hearing", {"case_id": "c1"}, **_USER)
    assert turn is not None
    assert turn.missing_field == "hearing_date"
    assert "DD/MM/YYYY" in turn.message


# ---------------------------------------------------------------------------
# 4. Account
# ---------------------------------------------------------------------------


def test_preferences_can_be_read_and_changed_by_talking(monkeypatch) -> None:
    from app.chatops.workflows import account

    class _Prefs:
        language = "hindi"
        explanation_mode = "simple"
        preferred_document_format = "pdf"
        voice_output = False

    captured: list[Any] = []

    async def _update(self, owner, request):
        captured.append(request)
        return _Prefs()

    monkeypatch.setattr(account.PreferenceService, "update", _update)

    turn = _orchestrate("change my settings, answer me in Hindi", {}, **_USER)
    assert turn is not None and turn.status == "completed"
    assert captured[0].language == "hindi"


def test_erasing_account_data_needs_a_typed_phrase_not_just_yes(monkeypatch) -> None:
    from app.services import user_data

    erased: list[str] = []

    async def _erase(user_id):
        erased.append(user_id)
        return {"status": "deleted", "deleted": {"chat_messages": 3}}

    monkeypatch.setattr(user_data, "erase_user_data", _erase)

    memory: dict[str, Any] = {}
    asked = _orchestrate("delete my personal data", memory, **_USER)
    assert asked is not None
    assert asked.missing_field == "typed_confirmation"
    assert "cannot be undone" in asked.message

    ignored = _orchestrate("yes", memory, **_USER)
    assert ignored is not None
    assert erased == [], "a bare yes must not erase an account"

    done = _orchestrate("delete my data", memory, **_USER)
    assert done is not None and done.status == "awaiting_confirmation"
    final = _orchestrate("yes", memory, **_USER)
    assert final is not None and final.status == "completed"
    assert erased == ["u1"]


def test_retrying_a_failed_job_is_confirmed_and_never_runs_twice(monkeypatch) -> None:
    from app.chatops.workflows import account

    class _Job:
        def __init__(self, job_id: str, status: str) -> None:
            self.job_id = job_id
            self.job_type = "export"
            self.status = status
            self.progress = 0
            self.error = "provider timeout" if status == "failed" else None

    retried: list[str] = []

    async def _list(self, owner):
        return [_Job("j1", "failed")]

    async def _retry(self, owner, job_id):
        retried.append(job_id)
        return _Job(job_id, "queued")

    monkeypatch.setattr(account.BackgroundJobService, "list", _list)
    monkeypatch.setattr(account.BackgroundJobService, "retry", _retry)

    memory: dict[str, Any] = {}
    confirmation = _orchestrate("retry the failed job", memory, **_USER)
    assert confirmation is not None and confirmation.status == "awaiting_confirmation"
    assert retried == []

    _orchestrate("yes", memory, **_USER)
    assert retried == ["j1"]

    # Replayed: same job, same request.
    _orchestrate("retry the failed job", memory, **_USER)
    replay = _orchestrate("yes", memory, **_USER)
    assert replay is not None
    assert retried == ["j1"], "a replayed retry ran the job a second time"


# ---------------------------------------------------------------------------
# 5. Admin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    ["knowledge base status", "show the admin dashboard", "list the legal sources"],
)
def test_admin_capabilities_are_refused_to_ordinary_users(message: str) -> None:
    turn = _orchestrate(message, {}, **_USER)
    assert turn is not None and turn.status == "forbidden"


def test_admin_workflows_declare_the_admin_role() -> None:
    for name in ("admin_knowledge_base", "admin_analytics", "admin_sources"):
        assert WORKFLOWS[name].required_role is Role.admin


def test_applying_a_reconciliation_is_confirmed_first(monkeypatch) -> None:
    from app.chatops.workflows import admin

    applied: list[str] = []

    class _Report:
        def as_dict(self):
            return {
                "mode": "apply", "mongo_chunk_count": 10, "bm25_chunk_count": 12,
                "bm25_chunk_count_after": 10, "stale_bm25_records": 2, "stale_private_records": 0,
                "missing_from_bm25": 0, "duplicate_chunk_ids": 0, "orphaned_source_documents": [],
                "ownership_metadata_problems": 0, "pruned": 2, "drifted": True,
            }

    class _Reconciler:
        def __init__(self, index=None) -> None:
            pass

        async def apply(self):
            applied.append("apply")
            return _Report()

        async def analyze(self):  # pragma: no cover - not the path under test
            return _Report()

    from app.rag import reconciliation

    monkeypatch.setattr(reconciliation, "IndexReconciler", _Reconciler)
    assert admin  # the workflow module is what is under test

    memory: dict[str, Any] = {}
    confirmation = _orchestrate("apply the index reconciliation", memory, **_ADMIN)
    assert confirmation is not None and confirmation.status == "awaiting_confirmation"
    assert applied == []

    done = _orchestrate("yes", memory, **_ADMIN)
    assert done is not None and done.status == "completed"
    assert applied == ["apply"]


def test_a_legal_source_cannot_be_marked_verified_without_evidence(monkeypatch) -> None:
    from app.chatops.workflows import admin

    class _Source:
        source_id = "s1"
        act_name = "Consumer Protection Act"
        section_number = "2"
        verification_status = "unverified"
        stale = False

    async def _list(self, *, stale_only=False, status=None):
        return [_Source()]

    async def _review(self, source_id, admin_user_id, request):  # pragma: no cover - must not run
        raise AssertionError("a verification was recorded with no evidence URL")

    monkeypatch.setattr(admin.LegalUpdateService, "list", _list)
    monkeypatch.setattr(admin.LegalUpdateService, "review", _review)

    memory: dict[str, Any] = {}
    _orchestrate("verify a legal source", memory, **_ADMIN)
    turn = _orchestrate("1", memory, **_ADMIN)
    assert turn is not None
    assert turn.missing_field == "evidence_url"
    assert "issuing authority" in turn.message


# ---------------------------------------------------------------------------
# 6. Documents
# ---------------------------------------------------------------------------


def test_two_uploads_produce_a_choice_rather_than_a_guess() -> None:
    memory: dict[str, Any] = {
        "uploaded_documents": [
            {"document_id": "doc-1", "filename": "rent-agreement.pdf", "uploaded_at": "2026-08-01"},
            {"document_id": "doc-2", "filename": "employment-contract.pdf", "uploaded_at": "2026-08-02"},
        ]
    }
    turn = _orchestrate("summarise this document", memory, **_USER)
    assert turn is not None
    assert turn.missing_field == "document_id"
    assert "rent-agreement.pdf" in turn.message
    assert "doc-1" not in turn.message


def test_a_document_request_with_nothing_uploaded_asks_for_the_file() -> None:
    turn = _orchestrate("what are the risky clauses", {}, **_USER)
    assert turn is not None
    assert "Attach the document" in turn.message


def test_the_documents_chosen_from_are_only_this_conversations_uploads() -> None:
    """A user must not be able to reach a document by naming an id."""
    from app.chatops.base import WorkflowContext
    from app.chatops.workflows.documents import _conversation_documents

    context = WorkflowContext(
        session_id="s1", message="doc-belonging-to-someone-else", language="english",
        memory={"uploaded_documents": [{"document_id": "mine", "filename": "mine.pdf"}]},
    )
    assert [choice.key for choice in _conversation_documents(context)] == ["mine"]


# ---------------------------------------------------------------------------
# 7. Cancellation applies everywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opening",
    ["start a new case", "delete my personal data", "add a hearing"],
)
def test_any_workflow_can_be_cancelled_mid_flow(opening: str) -> None:
    memory: dict[str, Any] = {"case_id": "c1"}
    _orchestrate(opening, memory, **_USER)
    turn = _orchestrate("cancel", memory, **_USER)
    assert turn is not None and turn.status == "cancelled"
    assert state.active_name(memory) is None
