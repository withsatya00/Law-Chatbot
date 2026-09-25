from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.casefile.annexures import EvidenceItem, build_evidence_table
from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError
from app.repositories.cases import CaseRepository
from app.repositories.documents import DocumentRepository
from app.repositories.drafts import DraftRepository
from app.schemas.case import (
    AddHearingRequest,
    AddNoteRequest,
    AddTimelineEventRequest,
    CaseCreateRequest,
    CaseResponse,
    CaseUpdateRequest,
    HearingReminderResponse,
)
from app.schemas.phase2 import LawyerSummaryResponse, TaskEntry


class CaseService:
    def __init__(self) -> None:
        self.repository = CaseRepository()
        self.documents = DocumentRepository()
        self.drafts = DraftRepository()

    async def create(self, owner_user_id: str, request: CaseCreateRequest) -> CaseResponse:
        now = datetime.now(UTC)
        document: dict[str, Any] = {
            "_id": str(uuid4()),
            "owner_user_id": owner_user_id,
            "case_number": request.case_number,
            "title": request.title,
            "status": "active",
            "case_type": request.case_type,
            "court_name": request.court_name,
            "client_name": request.client_name,
            "client_contact": request.client_contact,
            "opposite_party": request.opposite_party,
            "opposite_party_advocate": request.opposite_party_advocate,
            "filing_date": request.filing_date,
            "legal_category": request.legal_category or request.case_type,
            "parties": request.parties or [value for value in (request.client_name, request.opposite_party) if value],
            "tasks": [],
            "timeline": [],
            "next_action": request.next_action,
            "reminders": [],
            "linked_draft_ids": [],
            "resolved_conflicts": {},
            "evidence": [],
            "next_hearing_date": None,
            "hearings": [],
            "notes": [],
            "document_ids": [],
            "created_at": now,
            "updated_at": now,
        }
        await self.repository.insert(document)
        return self._to_response(document)

    async def list_for_owner(self, owner_user_id: str, status: str | None = None) -> list[CaseResponse]:
        cases = await self.repository.list_for_owner(owner_user_id, status=status)
        return [self._to_response(case) for case in cases]

    async def upcoming_hearings(self, owner_user_id: str, within_days: int) -> list[CaseResponse]:
        cases = await self.repository.upcoming_hearings(owner_user_id, within_days)
        return [self._to_response(case) for case in cases]

    async def get(self, owner_user_id: str, case_id: str) -> CaseResponse:
        case = await self._get_owned_case(owner_user_id, case_id)
        return self._to_response(case)

    async def update(self, owner_user_id: str, case_id: str, request: CaseUpdateRequest) -> CaseResponse:
        await self._get_owned_case(owner_user_id, case_id)
        updates = request.model_dump(exclude_unset=True)
        if updates:
            await self.repository.update_by_id(case_id, updates)
        case = await self._get_owned_case(owner_user_id, case_id)
        return self._to_response(case)

    async def delete(self, owner_user_id: str, case_id: str) -> None:
        await self._get_owned_case(owner_user_id, case_id)
        await self.repository.delete_by_id(case_id)

    async def add_hearing(self, owner_user_id: str, case_id: str, request: AddHearingRequest) -> CaseResponse:
        case = await self._get_owned_case(owner_user_id, case_id)
        now = datetime.now(UTC)
        entry = {
            "hearing_id": str(uuid4()),
            "hearing_date": request.hearing_date,
            "purpose": request.purpose,
            "notes": request.notes,
            "reminder_days_before": request.reminder_days_before,
            "reminder_acknowledged": False,
            "added_at": now,
        }
        hearings = [*case.get("hearings", []), entry]
        next_hearing = self._earliest_upcoming(hearings, now)
        await self.repository.update_by_id(
            case_id, {"hearings": hearings, "next_hearing_date": next_hearing}
        )
        case = await self._get_owned_case(owner_user_id, case_id)
        return self._to_response(case)

    async def hearing_reminders(self, owner_user_id: str) -> list[HearingReminderResponse]:
        """Distinct from `upcoming_hearings` above (a fixed shared window,
        e.g. "everything in the next 14 days") -- this reads each hearing's
        OWN `reminder_days_before` (set per-hearing in `add_hearing`, default
        3) and only surfaces the ones actually due for a nudge, skipping any
        already acknowledged via `acknowledge_hearing_reminder`. No
        email/SMS/push channel exists anywhere in this app to actually
        deliver a reminder -- this is the polling surface a client (the
        Streamlit UI today, any future frontend) reads on load/refresh,
        same "backend computes what's due, client renders it" shape as
        `KnowledgeBaseIngestionService`'s staging status endpoints.
        """
        cases = await self.repository.list_for_owner(owner_user_id)
        now = datetime.now(UTC)
        due: list[HearingReminderResponse] = []
        for case in cases:
            if case.get("status") == "closed":
                continue
            for hearing in case.get("hearings", []):
                hearing_date = self._as_aware(hearing["hearing_date"])
                days_until = (hearing_date - now).days
                threshold = hearing.get("reminder_days_before", 3)
                if hearing.get("reminder_acknowledged") or days_until < 0 or days_until > threshold:
                    continue
                due.append(
                    HearingReminderResponse(
                        case_id=case["_id"],
                        case_title=case["title"],
                        hearing_id=hearing.get("hearing_id", ""),
                        hearing_date=hearing_date,
                        purpose=hearing.get("purpose", ""),
                        days_until=days_until,
                    )
                )
        return sorted(due, key=lambda reminder: reminder.days_until)

    async def acknowledge_hearing_reminder(self, owner_user_id: str, case_id: str, hearing_id: str) -> CaseResponse:
        case = await self._get_owned_case(owner_user_id, case_id)
        hearings = case.get("hearings", [])
        if not any(hearing.get("hearing_id") == hearing_id for hearing in hearings):
            raise NotFoundError(f"Hearing {hearing_id} not found on case {case_id}.")
        updated_hearings = [
            {**hearing, "reminder_acknowledged": True} if hearing.get("hearing_id") == hearing_id else hearing
            for hearing in hearings
        ]
        await self.repository.update_by_id(case_id, {"hearings": updated_hearings})
        case = await self._get_owned_case(owner_user_id, case_id)
        return self._to_response(case)

    async def add_note(self, owner_user_id: str, case_id: str, request: AddNoteRequest) -> CaseResponse:
        case = await self._get_owned_case(owner_user_id, case_id)
        entry = {"text": request.text, "added_at": datetime.now(UTC)}
        notes = [*case.get("notes", []), entry]
        await self.repository.update_by_id(case_id, {"notes": notes})
        case = await self._get_owned_case(owner_user_id, case_id)
        return self._to_response(case)

    async def attach_document(self, owner_user_id: str, case_id: str, document_id: str) -> CaseResponse:
        """Security/correctness finding N4: `document_id` previously went
        straight into the case's `document_ids` list with no check that it
        even exists, let alone belongs to this account -- a caller could
        attach an arbitrary id, including one naming a DIFFERENT user's
        private upload, and any later reader of this case (a UI fetching
        "attached" documents by id, a lawyer-summary export) would then
        treat that id as something this case's owner was already vetted to
        see. Same three-way visibility rule `DocumentService.
        ensure_document_access` already enforces elsewhere: no owner
        stamped (shared/global) -> allowed; owned by THIS account ->
        allowed; owned by anyone else -> rejected, as a 404 (not 403) so
        this can't be used to confirm a specific document_id exists for
        another account.
        """
        case = await self._get_owned_case(owner_user_id, case_id)
        document = await self.documents.find_by_id(document_id)
        if document is None:
            raise NotFoundError(f"Document '{document_id}' not found.")
        document_owner = document.get("owner_user_id")
        if document_owner and document_owner != owner_user_id:
            raise NotFoundError(f"Document '{document_id}' not found.")
        document_ids = case.get("document_ids", [])
        if document_id not in document_ids:
            document_ids = [*document_ids, document_id]
            await self.repository.update_by_id(case_id, {"document_ids": document_ids})
        case = await self._get_owned_case(owner_user_id, case_id)
        return self._to_response(case)

    async def link_draft(self, owner_user_id: str, case_id: str, draft_id: str) -> CaseResponse:
        """Security/correctness finding N4: identical reasoning and fix as
        `attach_document` immediately above, for drafts -- `draft_id` must
        exist and belong to this account (or have no recorded owner at all,
        the same legacy-record allowance `ensure_draft_access` uses
        elsewhere) before a case may claim it as linked.
        """
        case = await self._get_owned_case(owner_user_id, case_id)
        draft = await self.drafts.find_by_id(draft_id)
        if draft is None:
            raise NotFoundError(f"Draft '{draft_id}' not found.")
        draft_owner = draft.get("user_id")
        if draft_owner and draft_owner != owner_user_id:
            raise NotFoundError(f"Draft '{draft_id}' not found.")
        linked = list(case.get("linked_draft_ids", []))
        if draft_id not in linked:
            linked.append(draft_id)
            await self.repository.update_by_id(case_id, {"linked_draft_ids": linked})
        return self._to_response(await self._get_owned_case(owner_user_id, case_id))

    async def add_evidence(
        self,
        owner_user_id: str,
        case_id: str,
        *,
        document_id: str,
        document_name: str,
        text: str,
        description: str = "",
    ) -> CaseResponse:
        case = await self._get_owned_case(owner_user_id, case_id)
        existing = list(case.get("evidence", []))
        if any(item.get("evidence_id") == document_id for item in existing):
            return self._to_response(case)
        inputs = [
            EvidenceItem(
                evidence_id=item.get("evidence_id", ""),
                document_name=item.get("document_name", "evidence"),
                text=item.get("text", ""),
                description=item.get("description", ""),
                uploaded_at=item.get("uploaded_at", ""),
                annexure=item.get("annexure", ""),
            )
            for item in existing
        ]
        inputs.append(EvidenceItem(
            evidence_id=document_id,
            document_name=document_name,
            text=text,
            description=description,
            uploaded_at=datetime.now(UTC).isoformat(),
        ))
        rows = build_evidence_table(inputs)
        evidence = [
            {
                **item.__dict__,
                "annexure": row.annexure,
                "document_date": row.document_date,
                "relevance": row.relevance,
                "missing": list(row.missing),
                "extracted_facts": [
                    {"slot": fact.slot, "label": fact.label, "value": fact.value}
                    for fact in row.facts
                ],
            }
            for item, row in zip(inputs, rows, strict=True)
        ]
        document_ids = list(case.get("document_ids", []))
        if document_id not in document_ids:
            document_ids.append(document_id)
        await self.repository.update_by_id(case_id, {"evidence": evidence, "document_ids": document_ids})
        return self._to_response(await self._get_owned_case(owner_user_id, case_id))

    async def add_task(self, owner_user_id: str, case_id: str, request: TaskEntry) -> CaseResponse:
        """Security finding N7: `task_id` and `status` are never taken from
        the request -- `task_id` must be unique and server-generated (a
        client-supplied id was previously honoured whenever present, `task.
        get("task_id") or str(uuid4())`, and nothing anywhere enforces
        uniqueness against it, so a caller could hand back an id that
        already names a DIFFERENT task and create an ambiguous duplicate).
        `status` is always `"pending"` for a newly added task regardless of
        what the client sent -- there is no "add an already-completed task"
        concept in this workflow, only "add a task, then later mark it
        done", and honouring a caller-asserted `"completed"` at creation
        would let a task skip whatever real completion means without ever
        having been acted on.
        """
        case = await self._get_owned_case(owner_user_id, case_id)
        task = request.model_dump()
        task["task_id"] = str(uuid4())
        task["status"] = "pending"
        tasks = [*case.get("tasks", []), task]
        reminders = list(case.get("reminders", []))
        if task.get("reminder_at"):
            reminders.append({"reminder_id": str(uuid4()), "task_id": task["task_id"], "remind_at": task["reminder_at"], "acknowledged": False})
        await self.repository.update_by_id(case_id, {"tasks": tasks, "reminders": reminders})
        return self._to_response(await self._get_owned_case(owner_user_id, case_id))

    async def add_timeline_event(self, owner_user_id: str, case_id: str, request: AddTimelineEventRequest) -> CaseResponse:
        case = await self._get_owned_case(owner_user_id, case_id)
        event = {**request.model_dump(), "event_id": str(uuid4()), "added_at": datetime.now(UTC)}
        timeline = [*case.get("timeline", []), event]
        timeline.sort(key=lambda item: (not bool(item.get("occurred_on")), item.get("occurred_on", "")))
        await self.repository.update_by_id(case_id, {"timeline": timeline})
        return self._to_response(await self._get_owned_case(owner_user_id, case_id))

    async def resolve_conflict(self, owner_user_id: str, case_id: str, slot: str, chosen_value: str) -> CaseResponse:
        """Security/correctness finding N5: `slot` previously went straight
        into `resolved_conflicts` with no check that it was ever a real
        conflict this case actually has -- a caller could "resolve" an
        arbitrary, made-up slot name (or one belonging to a wholly
        different case/workflow), polluting the record with a value nothing
        will ever read as authoritative and that a lawyer summary would
        never surface for review since it never appears in the case's own
        `unresolved_conflicts` list. `unresolved_conflicts` is also updated
        here to drop the slot once resolved -- otherwise `lawyer_summary`
        keeps listing it under `unanswered_questions` forever, telling the
        user to resolve something they just did.
        """
        case = await self._get_owned_case(owner_user_id, case_id)
        unresolved = list(case.get("unresolved_conflicts", []))
        if slot not in unresolved:
            raise BadRequestError(
                f"{slot!r} is not an unresolved conflict on this case.", {"field": "slot"},
            )
        resolved = {**case.get("resolved_conflicts", {}), slot: chosen_value}
        unresolved.remove(slot)
        await self.repository.update_by_id(
            case_id, {"resolved_conflicts": resolved, "unresolved_conflicts": unresolved}
        )
        return self._to_response(await self._get_owned_case(owner_user_id, case_id))

    async def lawyer_summary(self, owner_user_id: str, case_id: str) -> LawyerSummaryResponse:
        case = await self._get_owned_case(owner_user_id, case_id)
        parties = case.get("parties") or [
            value for value in (case.get("client_name"), case.get("opposite_party")) if value
        ]
        facts = [
            f"Matter: {case.get('title', '')}",
            f"Category: {case.get('legal_category') or case.get('case_type', '')}",
            f"Parties: {', '.join(parties) if parties else 'Not fully recorded'}",
        ]
        if case.get("next_action"):
            facts.append(f"Next action: {case['next_action']}")
        unanswered: list[str] = []
        if not parties:
            unanswered.append("Who are all of the parties and what are their correct legal names?")
        if not case.get("timeline"):
            unanswered.append("What are the material event dates in chronological order?")
        if not case.get("document_ids"):
            unanswered.append("Which primary documents support the account?")
        for slot in case.get("unresolved_conflicts", []):
            unanswered.append(f"Resolve the conflicting {slot} before filing.")
        summary = "\n".join(facts)
        return LawyerSummaryResponse(
            case_id=case_id,
            summary=summary,
            facts=facts,
            timeline=case.get("timeline", []),
            key_documents=case.get("evidence", []) or [{"document_id": value} for value in case.get("document_ids", [])],
            draft_status=[{"draft_id": value, "status": "linked; review status in draft workspace"} for value in case.get("linked_draft_ids", [])],
            unanswered_questions=unanswered,
            sharing={"shared": False, "automatic_external_sharing": False, "user_action_required": "Export or copy this summary explicitly if you choose to share it."},
        )

    def _earliest_upcoming(self, hearings: list[dict[str, Any]], now: datetime) -> datetime | None:
        upcoming = [h["hearing_date"] for h in hearings if self._as_aware(h["hearing_date"]) >= now]
        return min(upcoming, key=self._as_aware) if upcoming else None

    def _as_aware(self, value: datetime) -> datetime:
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    async def _get_owned_case(self, owner_user_id: str, case_id: str) -> dict[str, Any]:
        """Ownership check shared by every mutating/read operation above --
        a case not found at all and a case that exists but belongs to
        someone else are deliberately distinguished (404 vs 403), same
        boundary `DocumentService._ensure_document_access` already
        establishes for uploaded documents (Part 46).
        """
        case = await self.repository.find_by_id(case_id)
        if case is None:
            raise NotFoundError(f"Case {case_id} not found.")
        if case.get("owner_user_id") != owner_user_id:
            raise ForbiddenError("You do not have access to this case.")
        return case

    def _to_response(self, case: dict[str, Any]) -> CaseResponse:
        return CaseResponse(
            case_id=case["_id"],
            case_number=case["case_number"],
            title=case["title"],
            status=case["status"],
            case_type=case.get("case_type", ""),
            court_name=case.get("court_name", ""),
            client_name=case.get("client_name", ""),
            client_contact=case.get("client_contact", ""),
            opposite_party=case.get("opposite_party", ""),
            opposite_party_advocate=case.get("opposite_party_advocate", ""),
            filing_date=case.get("filing_date"),
            next_hearing_date=case.get("next_hearing_date"),
            hearings=case.get("hearings", []),
            notes=case.get("notes", []),
            document_ids=case.get("document_ids", []),
            created_at=case["created_at"],
            updated_at=case["updated_at"],
            legal_category=case.get("legal_category", case.get("case_type", "")),
            parties=case.get("parties", []),
            tasks=case.get("tasks", []),
            timeline=case.get("timeline", []),
            next_action=case.get("next_action", ""),
            reminders=case.get("reminders", []),
            linked_draft_ids=case.get("linked_draft_ids", []),
            resolved_conflicts=case.get("resolved_conflicts", {}),
            evidence=case.get("evidence", []),
        )
