"""Case files, by talking.

Seventeen REST routes sat behind `CaseService` with no conversational path to
any of them. These four workflows cover all of them:

    case_create     -- start a case file
    case_list       -- my cases, upcoming hearings, reminders
    case_manage     -- open / update / delete / hearings / notes / tasks /
                       attach a document / evidence / timeline / conflicts
    lawyer_summary  -- the lawyer-ready brief

Every one is an adapter: `CaseService` does the work and enforces ownership
through `_get_owned_case`, so a case belonging to someone else is a 403 from
the service, never something this layer decides.
"""

import re
from datetime import UTC, datetime
from typing import Any

from app.chatops import intents, selection
from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.registry import register_workflow
from app.core.security import Role
from app.schemas.case import (
    AddHearingRequest,
    AddNoteRequest,
    AddTimelineEventRequest,
    CaseCreateRequest,
    CaseResponse,
    CaseUpdateRequest,
)
from app.schemas.phase2 import TaskEntry
from app.services.case_service import CaseService

# The conversation's current case, remembered like `draft_id` is, so a user
# who says "add a hearing" right after opening a case is not asked which one.
_MEMORY_CASE_KEY = "case_id"

_DATE_PATTERNS = (
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"),
)


def _parse_date(text: str) -> datetime | None:
    """A date the user typed, or None. Never a guessed one.

    Ambiguous or absent dates come back as `None` and the workflow asks --
    a hearing recorded on the wrong day is a missed hearing.
    """
    match = _DATE_PATTERNS[0].search(text or "")
    if match:
        year, month, day = (int(part) for part in match.groups())
    else:
        match = _DATE_PATTERNS[1].search(text or "")
        if not match:
            return None
        day, month, year = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None


def _case_choices(cases: list[CaseResponse]) -> list[selection.Choice]:
    return [
        selection.Choice(
            key=case.case_id,
            label=case.title or case.case_number,
            detail=f"{case.case_number} · {case.status}",
        )
        for case in cases
    ]


class _OwnerScopedCaseWorkflow(ChatWorkflow):
    """Shared resolution of "which case?" without ever asking for an id."""

    required_role = Role.user
    may_interrupt_draft = True

    async def _resolve_case(self, context: WorkflowContext) -> dict[str, Any]:
        """Facts identifying the case: either `case_id`, or `_choices` to pick from."""
        if context.facts.get("case_id"):
            return {}

        parked = selection.from_dicts(list(context.facts.get("_choices") or []))
        if parked:
            chosen = selection.resolve(context.message, parked)
            if chosen is None:
                return {}
            return {"case_id": chosen.key, "case_label": chosen.label}

        remembered = context.memory.get(_MEMORY_CASE_KEY)
        if remembered:
            return {"case_id": remembered}

        cases = await CaseService().list_for_owner(str(context.authenticated_user_id))
        choices = _case_choices(cases)
        if len(choices) == 1:
            return {"case_id": choices[0].key, "case_label": choices[0].label}
        if choices:
            return {"_choices": [choice.as_dict() for choice in choices]}
        return {}

    def _which_case_question(self, context: WorkflowContext) -> str:
        choices = selection.from_dicts(list(context.facts.get("_choices") or []))
        if choices:
            return "Which case is this about?\n\n" + selection.render(choices)
        return (
            "I do not have a case file to attach this to yet. "
            "Would you like me to start one? Just tell me the case number and a short title."
        )


@register_workflow
class CaseCreateWorkflow(_OwnerScopedCaseWorkflow):
    """"Start a new case file" — collects the minimum, one question at a time."""

    name = "case_create"
    title = "new case file"

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.CASE_CREATE, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        message = context.message.strip()
        # The orchestrator asks one field at a time, so this message answers
        # whichever field is still missing. Values are stored VERBATIM: a
        # case number is an identifier and must never be normalised, expanded
        # or guessed at. The opening request ("start a new case") carries an
        # action cue and is not mistaken for an answer.
        missing = self.missing_fields(context)
        if missing and message and not intents.wants_action(message) and len(message.split()) <= 20:
            facts[missing[0]] = message
        return facts

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["case_number", "title"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        if missing[0] == "case_number":
            return (
                "Let's set up the case file. What is the case number? "
                "(If it has not been filed yet, give me any reference you use for it.)"
            )
        return "And a short title for the case — what should I call it?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        service = CaseService()
        case = await service.create(
            str(context.authenticated_user_id),
            CaseCreateRequest(
                case_number=str(context.facts["case_number"]),
                title=str(context.facts["title"]),
                court_name=str(context.facts.get("court_name", "")),
                client_name=str(context.facts.get("client_name", "")),
            ),
        )
        context.memory[_MEMORY_CASE_KEY] = case.case_id
        return WorkflowTurn(
            message=(
                f"Created the case file **{case.title}** ({case.case_number}).\n\n"
                "You can now add hearings, notes, tasks and evidence to it just by telling me."
            ),
            status="completed",
            allowed_actions=["add a hearing", "add a note", "attach a document", "show my cases"],
            finished=True,
        )


@register_workflow
class CaseListWorkflow(_OwnerScopedCaseWorkflow):
    """"Show my cases", "what hearings are coming up", "any reminders?"."""

    name = "case_list"
    title = "case list"

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.CASE_LIST, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        lowered = context.message.lower()
        if re.search(r"reminder|अनुस्मारक|yaad", lowered):
            return {"view": "reminders"}
        if re.search(r"hearing|sunwai|peshi|सुनवाई|पेशी|upcoming|आगामी", lowered):
            return {"view": "hearings"}
        return {"view": "cases"}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return []

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return "Would you like your case list, upcoming hearings, or reminders?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        service = CaseService()
        owner = str(context.authenticated_user_id)
        view = str(context.facts.get("view", "cases"))

        if view == "reminders":
            reminders = await service.hearing_reminders(owner)
            if not reminders:
                return WorkflowTurn(
                    message="Nothing needs acknowledging right now.", status="completed", finished=True
                )
            lines = "\n".join(
                f"- **{reminder.case_title}** — {reminder.hearing_date:%d %b %Y}"
                f" (in {reminder.days_until} day(s)): {reminder.purpose or 'hearing'}"
                for reminder in reminders
            )
            return WorkflowTurn(
                message=f"**Reminders**\n\n{lines}\n\nSay \"acknowledge\" once you have noted one.",
                status="completed", allowed_actions=["acknowledge reminder"], finished=True,
            )

        if view == "hearings":
            cases = await service.upcoming_hearings(owner, within_days=30)
            if not cases:
                return WorkflowTurn(
                    message="You have no hearings listed in the next 30 days.",
                    status="completed", finished=True,
                )
            lines = "\n".join(
                f"- **{case.title}** ({case.case_number}) — "
                f"{case.next_hearing_date:%d %b %Y}" if case.next_hearing_date else f"- **{case.title}**"
                for case in cases
            )
            return WorkflowTurn(
                message=f"**Hearings in the next 30 days**\n\n{lines}",
                status="completed", allowed_actions=["open a case", "add a hearing"], finished=True,
            )

        cases = await service.list_for_owner(owner)
        if not cases:
            return WorkflowTurn(
                message="You have no case files yet. Say \"start a new case\" and I will set one up.",
                status="completed", allowed_actions=["start a new case"], finished=True,
            )
        return WorkflowTurn(
            message="**Your cases**\n\n" + selection.render(_case_choices(cases))
            + "\n\nSay which one you want to open.",
            status="completed",
            allowed_actions=["open a case", "upcoming hearings", "start a new case"],
            finished=True,
        )


# The closed action set for case management. As with drafts, the action comes
# from a pattern match against this table -- never from a model naming one.
_CASE_ACTIONS: tuple[tuple[str, str], ...] = (
    ("delete", r"delete|remove|hata\s*do|मिटा|हटा"),
    ("close", r"\bclose\b|band\s*kar|बंद\s*कर"),
    ("acknowledge", r"acknowledge|\back\b|noted|dekh\s*liya|देख\s*लिया"),
    ("hearing", r"hearing|sunwai|peshi|सुनवाई|पेशी|tareekh|तारीख"),
    ("note", r"\bnotes?\b|टिप्पणी|नोट"),
    ("task", r"\btasks?\b|to-?do|कार्य"),
    ("attach", r"attach|link|जोड़|संलग्न"),
    ("evidence", r"evidence|sabut|सबूत|annexure|अनुलग्नक"),
    ("event", r"add\s+(?:an?\s+)?event|record\s+(?:an?\s+)?event|घटना\s*(?:जोड़|दर्ज)"),
    ("timeline", r"timeline|chronolog|समयरेखा|टाइमलाइन"),
    ("conflict", r"conflict|contradict|विरोधाभास|टकराव"),
    ("update", r"update|edit|change|badlo|बदल"),
    ("open", r"open|show|dikhao|खोल|दिखाओ"),
)

_CASE_NEEDS_CONFIRMATION = frozenset({"delete", "close"})


def _detect_case_action(message: str) -> str | None:
    for action, pattern in _CASE_ACTIONS:
        if re.search(pattern, message or "", re.IGNORECASE):
            return action
    return None


@register_workflow
class CaseManageWorkflow(_OwnerScopedCaseWorkflow):
    """Everything done TO a case file, from one conversational entry point."""

    name = "case_manage"
    title = "case management"

    def matches_intent(self, message: str, language: str) -> float:
        # "case timeline dikhao" names both a listing verb and a case-file
        # operation; the operation is what was asked for, so it outranks the
        # plain case list.
        return max(
            intents.score(intents.CASE_MANAGE, message, 0.95),
            intents.score(intents.CASE_TIMELINE, message, 0.95),
        )

    def needs_confirmation(self, context: WorkflowContext) -> bool:
        return str(context.facts.get("action", "")) in _CASE_NEEDS_CONFIRMATION

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        if not context.facts.get("action"):
            action = _detect_case_action(context.message)
            if action:
                facts["action"] = action
        facts.update(await self._resolve_case(context))

        action = str(facts.get("action") or context.facts.get("action") or "")
        if action == "hearing" and not context.facts.get("hearing_date"):
            hearing_date = _parse_date(context.message)
            if hearing_date is not None:
                facts["hearing_date"] = hearing_date.isoformat()
        # Only once the case is known, so the message that named the case is
        # not also stored as the note text.
        if (
            action in {"note", "task", "event"}
            and not context.facts.get("text")
            and (facts.get("case_id") or context.facts.get("case_id"))
            and not _detect_case_action(context.message)
            and len(context.message.split()) >= 2
        ):
            facts["text"] = context.message.strip()
        if action == "event" and not context.facts.get("occurred_on"):
            occurred = _parse_date(context.message)
            if occurred is not None:
                facts["occurred_on"] = occurred.date().isoformat()
        return facts

    def required_fields(self, context: WorkflowContext) -> list[str]:
        required = ["action", "case_id"]
        action = str(context.facts.get("action", ""))
        if action == "hearing":
            required.append("hearing_date")
        elif action in {"note", "task"}:
            required.append("text")
        return required

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        field_name = missing[0]
        if field_name == "action":
            return (
                "What would you like to do with the case — open it, add a hearing, add a note, "
                "add a task, attach a document, build its timeline, or close it?"
            )
        if field_name == "case_id":
            return self._which_case_question(context)
        if field_name == "hearing_date":
            return "What is the hearing date? (Please give it as DD/MM/YYYY so I record the right day.)"
        if field_name == "text":
            action = str(context.facts.get("action", ""))
            if action == "note":
                return "What should the note say?"
            return "What is the task?" if action == "task" else "What happened, and on what date?"
        return "Could you tell me a little more?"

    def summarize_for_confirmation(self, context: WorkflowContext) -> str:
        label = str(context.facts.get("case_label") or "this case")
        action = str(context.facts.get("action", ""))
        if action == "delete":
            return (
                f"- Case: {label}\n"
                "- Action: delete the case file, including its hearings, notes, tasks and evidence index\n\n"
                "This cannot be undone."
            )
        return f"- Case: {label}\n- Action: close the case (it stays readable, but is marked closed)"

    def idempotency_key(self, context: WorkflowContext) -> str:
        return (
            f"{self.name}:{context.facts.get('action')}:{context.facts.get('case_id')}"
            f":{context.facts.get('hearing_date', '')}:{str(context.facts.get('text', ''))[:40]}"
        )

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        service = CaseService()
        owner = str(context.authenticated_user_id)
        case_id = str(context.facts["case_id"])
        action = str(context.facts.get("action", "open"))
        context.memory[_MEMORY_CASE_KEY] = case_id

        if action == "delete":
            await service.delete(owner, case_id)
            context.memory.pop(_MEMORY_CASE_KEY, None)
            return WorkflowTurn(
                message="The case file has been deleted. Nothing was filed or submitted anywhere.",
                status="completed", finished=True,
            )
        if action == "close":
            case = await service.update(owner, case_id, CaseUpdateRequest(status="closed"))
            return WorkflowTurn(
                message=f"**{case.title}** is now marked closed. It stays in your list and can be reopened.",
                status="completed", finished=True,
            )
        if action == "hearing":
            hearing_date = _parse_date(str(context.facts["hearing_date"])) or datetime.fromisoformat(
                str(context.facts["hearing_date"])
            )
            case = await service.add_hearing(
                owner, case_id,
                AddHearingRequest(hearing_date=hearing_date, purpose=str(context.facts.get("purpose", ""))),
            )
            return WorkflowTurn(
                message=(
                    f"Hearing added to **{case.title}** for {hearing_date:%d %b %Y}. "
                    "I will include it in your upcoming hearings and reminders."
                ),
                status="completed", allowed_actions=["upcoming hearings", "add a note"], finished=True,
            )
        if action == "acknowledge":
            reminders = await service.hearing_reminders(owner)
            match = next((reminder for reminder in reminders if reminder.case_id == case_id), None)
            if match is None:
                return WorkflowTurn(
                    message="There is no reminder waiting on that case.", status="completed", finished=True
                )
            await service.acknowledge_hearing_reminder(owner, case_id, match.hearing_id)
            return WorkflowTurn(
                message=f"Noted — the reminder for {match.hearing_date:%d %b %Y} is acknowledged.",
                status="completed", finished=True,
            )
        if action == "note":
            case = await service.add_note(owner, case_id, AddNoteRequest(text=str(context.facts["text"])))
            return WorkflowTurn(
                message=f"Note added to **{case.title}** ({len(case.notes)} note(s) in total).",
                status="completed", finished=True,
            )
        if action == "task":
            case = await service.add_task(owner, case_id, TaskEntry(title=str(context.facts["text"])))
            return WorkflowTurn(
                message=f"Task added to **{case.title}** ({len(case.tasks)} task(s) in total).",
                status="completed", finished=True,
            )
        if action == "event":
            case = await service.add_timeline_event(
                owner, case_id,
                AddTimelineEventRequest(
                    occurred_on=str(context.facts.get("occurred_on", "")),
                    description=str(context.facts["text"]),
                ),
            )
            when = str(context.facts.get("occurred_on")) or "no date recorded"
            return WorkflowTurn(
                message=(
                    f"Added to the timeline of **{case.title}** ({when}). "
                    "Undated events stay listed separately so nothing is placed on a date you did not give."
                ),
                status="completed", allowed_actions=["show the timeline"], finished=True,
            )
        if action in {"attach", "evidence"}:
            return await self._attach(context, service, owner, case_id, as_evidence=action == "evidence")
        if action == "timeline":
            case = await service.get(owner, case_id)
            if not case.timeline:
                return WorkflowTurn(
                    message=(
                        f"**{case.title}** has no timeline entries yet. Tell me what happened and when, "
                        "and I will add each event as you give it."
                    ),
                    status="completed", finished=True,
                )
            lines = "\n".join(
                f"- {event.get('occurred_on') or 'date not recorded'} — {event.get('description', '')}"
                for event in case.timeline
            )
            return WorkflowTurn(
                message=f"**{case.title}** — timeline\n\n{lines}",
                status="completed", allowed_actions=["add an event", "lawyer-ready summary"], finished=True,
            )
        if action == "conflict":
            case = await service.get(owner, case_id)
            unresolved = [
                slot for slot in case.resolved_conflicts if not case.resolved_conflicts.get(slot)
            ]
            if not unresolved:
                return WorkflowTurn(
                    message=f"No unresolved conflicting facts are recorded on **{case.title}**.",
                    status="completed", finished=True,
                )
            return WorkflowTurn(
                message=(
                    f"**{case.title}** has conflicting values for: {', '.join(unresolved)}.\n\n"
                    "Tell me which value is correct — I will not choose for you."
                ),
                status="completed", finished=True,
            )
        if action == "update":
            case = await service.update(
                owner, case_id, CaseUpdateRequest(next_action=context.message.strip()[:200])
            )
            return WorkflowTurn(
                message=f"Updated **{case.title}**. Next action recorded as: {case.next_action}",
                status="completed", finished=True,
            )

        case = await service.get(owner, case_id)
        return WorkflowTurn(
            message=self._render_case(case),
            status="completed",
            allowed_actions=["add a hearing", "add a note", "build the timeline", "lawyer-ready summary"],
            finished=True,
        )

    async def _attach(
        self, context: WorkflowContext, service: CaseService, owner: str, case_id: str, *, as_evidence: bool
    ) -> WorkflowTurn:
        """Attaches the document the conversation is already holding.

        Uses only documents uploaded in THIS conversation, so a user cannot
        attach a document by naming an id they happen to know; the service
        then re-checks the case's owner besides.
        """
        document_id = context.document_ids[0] if context.document_ids else context.memory.get(
            "last_uploaded_document_id"
        )
        if not document_id:
            return WorkflowTurn(
                message="Upload the document here first and I will attach it to the case.",
                status="awaiting_upload", upload_required=True,
            )
        if as_evidence:
            case = await service.add_evidence(
                owner, case_id,
                document_id=str(document_id),
                document_name=str(context.memory.get("last_uploaded_filename") or "evidence"),
                text="",
            )
            latest = case.evidence[-1] if case.evidence else {}
            annexure = latest.get("annexure", "")
            return WorkflowTurn(
                message=(
                    f"Filed as evidence on **{case.title}**"
                    + (f", indexed as Annexure {annexure}." if annexure else ".")
                ),
                status="completed", allowed_actions=["organise my evidence", "lawyer-ready summary"], finished=True,
            )
        case = await service.attach_document(owner, case_id, str(document_id))
        return WorkflowTurn(
            message=f"Attached to **{case.title}** ({len(case.document_ids)} document(s) on file).",
            status="completed", finished=True,
        )

    @staticmethod
    def _render_case(case: CaseResponse) -> str:
        parts = [f"**{case.title}** ({case.case_number}) — {case.status}"]
        if case.court_name:
            parts.append(f"Court: {case.court_name}")
        if case.next_hearing_date:
            parts.append(f"Next hearing: {case.next_hearing_date:%d %b %Y}")
        if case.next_action:
            parts.append(f"Next action: {case.next_action}")
        parts.append(
            f"{len(case.hearings)} hearing(s) · {len(case.notes)} note(s) · "
            f"{len(case.tasks)} task(s) · {len(case.document_ids)} document(s)"
        )
        return "\n".join(parts)


@register_workflow
class EvidenceOrganizeWorkflow(_OwnerScopedCaseWorkflow):
    """"Organise my evidence" — the annexure table, from the case's own files.

    Numbering comes from `organize_evidence`, the same deterministic Phase 2
    function `POST /evidence/organize` calls. This workflow only supplies the
    case's stored evidence and renders the result.
    """

    name = "evidence_organize"
    title = "evidence organisation"

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.EVIDENCE_ORGANIZE, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return await self._resolve_case(context)

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["case_id"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return self._which_case_question(context)

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        from app.schemas.phase2 import EvidenceInput, EvidenceOrganizeRequest
        from app.services.phase2_workflow import organize_evidence

        case = await CaseService().get(str(context.authenticated_user_id), str(context.facts["case_id"]))
        if not case.evidence:
            return WorkflowTurn(
                message=(
                    f"**{case.title}** has no evidence on file yet. Upload a document here and say "
                    "\"file this as evidence\" and I will index it."
                ),
                status="completed", allowed_actions=["attach a document"], finished=True,
            )
        result = organize_evidence(
            EvidenceOrganizeRequest(
                evidence=[
                    EvidenceInput(
                        evidence_id=str(item.get("evidence_id", "")),
                        document_name=str(item.get("document_name", "evidence")),
                        text=str(item.get("text", "")),
                        description=str(item.get("description", "")),
                        uploaded_at=str(item.get("uploaded_at", "")),
                        annexure=str(item.get("annexure", "")),
                    )
                    for item in case.evidence
                ]
            )
        )
        rows = "\n".join(
            f"- **{row.get('annexure', '')}** {row.get('document_name', '')}"
            + (f" — {row.get('relevance', '')}" if row.get("relevance") else "")
            + (f" _(missing: {', '.join(row['missing'])})_" if row.get("missing") else "")
            for row in result.evidence_table
        )
        return WorkflowTurn(
            message=f"**{case.title}** — evidence index\n\n{rows}\n\n{result.annexure_index}",
            status="completed",
            allowed_actions=["lawyer-ready summary", "build the timeline"],
            finished=True,
        )


@register_workflow
class LawyerSummaryWorkflow(_OwnerScopedCaseWorkflow):
    """"Make a summary I can take to a lawyer."""

    name = "lawyer_summary"
    title = "lawyer-ready summary"

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.LAWYER_SUMMARY, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return await self._resolve_case(context)

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["case_id"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return self._which_case_question(context)

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        summary = await CaseService().lawyer_summary(
            str(context.authenticated_user_id), str(context.facts["case_id"])
        )
        questions = "\n".join(f"- {question}" for question in summary.unanswered_questions)
        gaps = f"\n\n**Your lawyer will need to know:**\n{questions}" if questions else ""
        documents = (
            f"\n\n{len(summary.key_documents)} document(s) are on file for this matter."
            if summary.key_documents else ""
        )
        return WorkflowTurn(
            message=(
                f"**Summary for your lawyer**\n\n{summary.summary}{documents}{gaps}\n\n"
                "Nothing here has been shared with anyone — copy or export it yourself if you want to send it."
            ),
            status="completed",
            warnings=["This summary is not legal advice and has not been reviewed by an advocate."],
            allowed_actions=["organise my evidence", "build the timeline"],
            finished=True,
        )
