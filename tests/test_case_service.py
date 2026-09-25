"""Case Management: unit tests for `CaseService`'s ownership boundary and
business logic (next-hearing-date computation, note/hearing/document
appends). Mocking pattern mirrors `test_draft_lifecycle.py`: the
repository's own methods are monkeypatched with `AsyncMock` rather than
hitting a real MongoDB, since these are tests of the service's own logic,
not the repository layer.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError
from app.schemas.case import (
    AddHearingRequest,
    AddNoteRequest,
    CaseCreateRequest,
    CaseUpdateRequest,
)
from app.schemas.phase2 import TaskEntry
from app.services.case_service import CaseService


def _case(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    base = {
        "_id": "case-1",
        "owner_user_id": "user-A",
        "case_number": "CASE/2026/001",
        "title": "Suresh Kumar vs State",
        "status": "active",
        "case_type": "Criminal",
        "court_name": "CMM Patiala House",
        "client_name": "Suresh Kumar",
        "client_contact": "9876543210",
        "opposite_party": "State (NCT of Delhi)",
        "opposite_party_advocate": "",
        "filing_date": now,
        "next_hearing_date": None,
        "hearings": [],
        "notes": [],
        "document_ids": [],
        "created_at": now,
        "updated_at": now,
    }
    base.update(overrides)
    return base


def _service_with_case(case: dict[str, object]) -> CaseService:
    service = CaseService()
    service.repository.find_by_id = AsyncMock(return_value=case)
    service.repository.insert = AsyncMock(return_value=case["_id"])
    service.repository.update_by_id = AsyncMock(return_value=True)
    return service


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_sets_owner_and_defaults() -> None:
    service = CaseService()
    service.repository.insert = AsyncMock(return_value="new-id")
    request = CaseCreateRequest(case_number="CASE/2026/002", title="New Matter")
    response = await service.create("user-A", request)
    assert response.status == "active"
    assert response.hearings == []
    assert response.document_ids == []
    inserted = service.repository.insert.call_args.args[0]
    assert inserted["owner_user_id"] == "user-A"


# ---------------------------------------------------------------------------
# ownership boundary -- shared by get/update/delete/add_hearing/add_note/attach_document
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_missing_case_raises_not_found() -> None:
    service = CaseService()
    service.repository.find_by_id = AsyncMock(return_value=None)
    with pytest.raises(NotFoundError):
        await service.get("user-A", "does-not-exist")


@pytest.mark.asyncio
async def test_get_wrong_owner_raises_forbidden_not_not_found() -> None:
    """A case that EXISTS but belongs to someone else is 403, not 404 --
    same distinction `DocumentService._ensure_document_access` already
    draws (Part 46), so a caller can tell "never existed" from "not yours"."""
    service = _service_with_case(_case(owner_user_id="user-A"))
    with pytest.raises(ForbiddenError):
        await service.get("user-B", "case-1")


@pytest.mark.asyncio
async def test_get_correct_owner_succeeds() -> None:
    service = _service_with_case(_case(owner_user_id="user-A"))
    response = await service.get("user-A", "case-1")
    assert response.case_id == "case-1"


# ---------------------------------------------------------------------------
# update() -- only touches fields actually supplied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_only_writes_supplied_fields() -> None:
    service = _service_with_case(_case())
    await service.update("user-A", "case-1", CaseUpdateRequest(status="closed"))
    updates = service.repository.update_by_id.call_args.args[1]
    assert updates == {"status": "closed"}


@pytest.mark.asyncio
async def test_update_with_no_fields_does_not_call_repository_write() -> None:
    service = _service_with_case(_case())
    await service.update("user-A", "case-1", CaseUpdateRequest())
    service.repository.update_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_update_wrong_owner_raises_before_writing() -> None:
    service = _service_with_case(_case(owner_user_id="user-A"))
    with pytest.raises(ForbiddenError):
        await service.update("user-B", "case-1", CaseUpdateRequest(status="closed"))
    service.repository.update_by_id.assert_not_called()


# ---------------------------------------------------------------------------
# add_hearing() -- next_hearing_date is the earliest UPCOMING hearing, not
# just whichever was added last
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_hearing_sets_next_hearing_date() -> None:
    service = _service_with_case(_case())
    future = datetime.now(UTC) + timedelta(days=10)
    await service.add_hearing("user-A", "case-1", AddHearingRequest(hearing_date=future, purpose="Framing of charges"))
    updates = service.repository.update_by_id.call_args.args[1]
    assert updates["next_hearing_date"] == future
    assert len(updates["hearings"]) == 1


@pytest.mark.asyncio
async def test_add_hearing_picks_earliest_upcoming_not_most_recently_added() -> None:
    now = datetime.now(UTC)
    sooner = now + timedelta(days=5)
    later = now + timedelta(days=20)
    existing_case = _case(hearings=[{"hearing_date": later, "purpose": "", "notes": "", "added_at": now}])
    service = _service_with_case(existing_case)
    await service.add_hearing("user-A", "case-1", AddHearingRequest(hearing_date=sooner))
    updates = service.repository.update_by_id.call_args.args[1]
    assert updates["next_hearing_date"] == sooner
    assert len(updates["hearings"]) == 2


@pytest.mark.asyncio
async def test_add_hearing_ignores_past_dates_for_next_hearing() -> None:
    now = datetime.now(UTC)
    past = now - timedelta(days=5)
    service = _service_with_case(_case())
    await service.add_hearing("user-A", "case-1", AddHearingRequest(hearing_date=past))
    updates = service.repository.update_by_id.call_args.args[1]
    assert updates["next_hearing_date"] is None


@pytest.mark.asyncio
async def test_add_hearing_stamps_a_hearing_id_and_reminder_fields() -> None:
    service = _service_with_case(_case())
    future = datetime.now(UTC) + timedelta(days=10)
    await service.add_hearing(
        "user-A", "case-1", AddHearingRequest(hearing_date=future, reminder_days_before=5)
    )
    stamped = service.repository.update_by_id.call_args.args[1]["hearings"][0]
    assert stamped["hearing_id"]
    assert stamped["reminder_days_before"] == 5
    assert stamped["reminder_acknowledged"] is False


# ---------------------------------------------------------------------------
# hearing_reminders() / acknowledge_hearing_reminder()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hearing_reminders_surfaces_only_hearings_within_their_own_threshold() -> None:
    now = datetime.now(UTC)
    due_soon = {
        # +1 hour buffer: `hearing_reminders()` computes its own `now`
        # microseconds after this fixture's `now`, so a bare `days=2` can
        # truncate `.days` down to 1 depending on exact timing.
        "hearing_id": "h-due", "hearing_date": now + timedelta(days=2, hours=1), "purpose": "Arguments",
        "reminder_days_before": 3, "reminder_acknowledged": False,
    }
    too_far = {
        "hearing_id": "h-far", "hearing_date": now + timedelta(days=10), "purpose": "Evidence",
        "reminder_days_before": 3, "reminder_acknowledged": False,
    }
    acknowledged = {
        "hearing_id": "h-ack", "hearing_date": now + timedelta(days=1), "purpose": "Framing",
        "reminder_days_before": 3, "reminder_acknowledged": True,
    }
    case = _case(hearings=[due_soon, too_far, acknowledged])
    service = CaseService()
    service.repository.list_for_owner = AsyncMock(return_value=[case])

    reminders = await service.hearing_reminders("user-A")

    assert [reminder.hearing_id for reminder in reminders] == ["h-due"]
    assert reminders[0].days_until == 2


@pytest.mark.asyncio
async def test_acknowledge_hearing_reminder_marks_only_that_hearing() -> None:
    now = datetime.now(UTC)
    hearing_a = {"hearing_id": "h-a", "hearing_date": now, "purpose": "", "notes": "", "added_at": now}
    hearing_b = {"hearing_id": "h-b", "hearing_date": now, "purpose": "", "notes": "", "added_at": now}
    service = _service_with_case(_case(hearings=[hearing_a, hearing_b]))

    await service.acknowledge_hearing_reminder("user-A", "case-1", "h-a")

    updated = service.repository.update_by_id.call_args.args[1]["hearings"]
    by_id = {hearing["hearing_id"]: hearing for hearing in updated}
    assert by_id["h-a"]["reminder_acknowledged"] is True
    assert by_id["h-b"].get("reminder_acknowledged") is None


@pytest.mark.asyncio
async def test_acknowledge_hearing_reminder_unknown_hearing_id_raises_not_found() -> None:
    service = _service_with_case(_case(hearings=[]))
    with pytest.raises(NotFoundError):
        await service.acknowledge_hearing_reminder("user-A", "case-1", "does-not-exist")


# ---------------------------------------------------------------------------
# add_note() / attach_document()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_note_appends_without_losing_existing_notes() -> None:
    existing_case = _case(notes=[{"text": "First note", "added_at": datetime.now(UTC)}])
    service = _service_with_case(existing_case)
    await service.add_note("user-A", "case-1", AddNoteRequest(text="Second note"))
    updates = service.repository.update_by_id.call_args.args[1]
    assert [n["text"] for n in updates["notes"]] == ["First note", "Second note"]


@pytest.mark.asyncio
async def test_attach_document_is_idempotent() -> None:
    existing_case = _case(document_ids=["doc-1"])
    service = _service_with_case(existing_case)
    service.documents.find_by_id = AsyncMock(return_value={"_id": "doc-1", "owner_user_id": "user-A"})
    await service.attach_document("user-A", "case-1", "doc-1")
    service.repository.update_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_attach_document_adds_new_id() -> None:
    existing_case = _case(document_ids=["doc-1"])
    service = _service_with_case(existing_case)
    service.documents.find_by_id = AsyncMock(return_value={"_id": "doc-2", "owner_user_id": "user-A"})
    await service.attach_document("user-A", "case-1", "doc-2")
    updates = service.repository.update_by_id.call_args.args[1]
    assert updates["document_ids"] == ["doc-1", "doc-2"]


# ---------------------------------------------------------------------------
# attach_document() / link_draft() -- security/correctness finding N4
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attaching_a_nonexistent_document_is_rejected() -> None:
    service = _service_with_case(_case(document_ids=[]))
    service.documents.find_by_id = AsyncMock(return_value=None)
    with pytest.raises(NotFoundError):
        await service.attach_document("user-A", "case-1", "no-such-doc")
    service.repository.update_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_attaching_another_users_private_document_is_rejected() -> None:
    """The core N4 scenario: a caller must not be able to attach a
    DIFFERENT account's private upload to their own case."""
    service = _service_with_case(_case(document_ids=[]))
    service.documents.find_by_id = AsyncMock(return_value={"_id": "doc-1", "owner_user_id": "victim-user"})
    with pytest.raises(NotFoundError):
        await service.attach_document("user-A", "case-1", "doc-1")
    service.repository.update_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_attaching_a_shared_document_with_no_owner_is_allowed() -> None:
    service = _service_with_case(_case(document_ids=[]))
    service.documents.find_by_id = AsyncMock(return_value={"_id": "doc-1", "owner_user_id": None})
    await service.attach_document("user-A", "case-1", "doc-1")
    service.repository.update_by_id.assert_called_once()


@pytest.mark.asyncio
async def test_linking_a_nonexistent_draft_is_rejected() -> None:
    service = _service_with_case(_case(linked_draft_ids=[]))
    service.drafts.find_by_id = AsyncMock(return_value=None)
    with pytest.raises(NotFoundError):
        await service.link_draft("user-A", "case-1", "no-such-draft")
    service.repository.update_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_linking_another_users_draft_is_rejected() -> None:
    service = _service_with_case(_case(linked_draft_ids=[]))
    service.drafts.find_by_id = AsyncMock(return_value={"_id": "draft-1", "user_id": "victim-user"})
    with pytest.raises(NotFoundError):
        await service.link_draft("user-A", "case-1", "draft-1")
    service.repository.update_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_linking_ones_own_draft_succeeds() -> None:
    service = _service_with_case(_case(linked_draft_ids=[]))
    service.drafts.find_by_id = AsyncMock(return_value={"_id": "draft-1", "user_id": "user-A"})
    await service.link_draft("user-A", "case-1", "draft-1")
    updates = service.repository.update_by_id.call_args.args[1]
    assert updates["linked_draft_ids"] == ["draft-1"]


@pytest.mark.asyncio
async def test_add_evidence_extracts_facts_and_assigns_stable_annexure() -> None:
    state = _case(evidence=[])
    service = CaseService()
    service.repository.find_by_id = AsyncMock(side_effect=lambda _case_id: state)

    async def update(_case_id: str, values: dict) -> bool:
        state.update(values)
        return True

    service.repository.update_by_id = AsyncMock(side_effect=update)
    response = await service.add_evidence(
        "user-A",
        "case-1",
        document_id="doc-receipt",
        document_name="upi_receipt.pdf",
        text="Debit Rs 45,000 on 26/08/2026 UTR HDFC1234567890",
        description="Bank receipt",
    )

    assert response.document_ids == ["doc-receipt"]
    assert response.evidence[0]["annexure"] == "Annexure A-1"
    assert {fact["slot"] for fact in response.evidence[0]["extracted_facts"]} >= {
        "amount", "incident_date", "transaction_id"
    }


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_wrong_owner_raises_and_never_deletes() -> None:
    service = _service_with_case(_case(owner_user_id="user-A"))
    service.repository.delete_by_id = AsyncMock(return_value=True)
    with pytest.raises(ForbiddenError):
        await service.delete("user-B", "case-1")
    service.repository.delete_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_delete_correct_owner_deletes() -> None:
    service = _service_with_case(_case(owner_user_id="user-A"))
    service.repository.delete_by_id = AsyncMock(return_value=True)
    await service.delete("user-A", "case-1")
    service.repository.delete_by_id.assert_called_once_with("case-1")


# ---------------------------------------------------------------------------
# resolve_conflict() -- security/correctness finding N5
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolving_a_slot_not_in_unresolved_conflicts_is_rejected() -> None:
    """A caller must not be able to "resolve" an arbitrary, made-up slot
    name -- only a conflict this case actually has may be recorded."""
    service = _service_with_case(_case(unresolved_conflicts=["client_name"], resolved_conflicts={}))
    with pytest.raises(BadRequestError):
        await service.resolve_conflict("user-A", "case-1", "not_a_real_conflict", "some value")
    service.repository.update_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_resolving_when_there_are_no_unresolved_conflicts_is_rejected() -> None:
    service = _service_with_case(_case(unresolved_conflicts=[], resolved_conflicts={}))
    with pytest.raises(BadRequestError):
        await service.resolve_conflict("user-A", "case-1", "client_name", "Suresh Kumar")


@pytest.mark.asyncio
async def test_resolving_a_valid_slot_records_it_and_removes_it_from_unresolved() -> None:
    case = _case(unresolved_conflicts=["client_name", "opposite_party"], resolved_conflicts={})
    service = _service_with_case(case)

    await service.resolve_conflict("user-A", "case-1", "client_name", "Suresh Kumar")

    updates = service.repository.update_by_id.await_args.args[1]
    assert updates["resolved_conflicts"] == {"client_name": "Suresh Kumar"}
    assert updates["unresolved_conflicts"] == ["opposite_party"]


@pytest.mark.asyncio
async def test_resolve_conflict_still_enforces_ownership() -> None:
    service = _service_with_case(_case(owner_user_id="user-A", unresolved_conflicts=["client_name"]))
    with pytest.raises(ForbiddenError):
        await service.resolve_conflict("user-B", "case-1", "client_name", "Someone Else")
    service.repository.update_by_id.assert_not_called()


# ---------------------------------------------------------------------------
# add_task() -- security finding N7
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_client_supplied_task_id_is_ignored_and_replaced() -> None:
    """`task_id` must be server-generated -- a caller handing back an id
    (whether by accident or to collide with an existing task) must never
    have it honoured verbatim."""
    service = _service_with_case(_case(tasks=[]))

    await service.add_task("user-A", "case-1", TaskEntry(task_id="attacker-chosen-id", title="File response"))

    saved_tasks = service.repository.update_by_id.await_args.args[1]["tasks"]
    assert len(saved_tasks) == 1
    assert saved_tasks[0]["task_id"] != "attacker-chosen-id"
    assert saved_tasks[0]["task_id"]  # still a real, non-empty id


@pytest.mark.asyncio
async def test_a_client_cannot_create_a_task_that_is_already_completed() -> None:
    """A new task always starts `pending` -- there is no "add an
    already-completed task" workflow, only "add, then later mark done"."""
    service = _service_with_case(_case(tasks=[]))

    await service.add_task("user-A", "case-1", TaskEntry(title="File response", status="completed"))

    saved_tasks = service.repository.update_by_id.await_args.args[1]["tasks"]
    assert saved_tasks[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_two_added_tasks_never_collide_on_task_id() -> None:
    service = _service_with_case(_case(tasks=[]))

    await service.add_task("user-A", "case-1", TaskEntry(task_id="same-id", title="First"))
    first_call_case = _case(tasks=service.repository.update_by_id.await_args.args[1]["tasks"])
    service.repository.find_by_id = AsyncMock(return_value=first_call_case)
    await service.add_task("user-A", "case-1", TaskEntry(task_id="same-id", title="Second"))

    saved_tasks = service.repository.update_by_id.await_args.args[1]["tasks"]
    assert len(saved_tasks) == 2
    assert saved_tasks[0]["task_id"] != saved_tasks[1]["task_id"]
