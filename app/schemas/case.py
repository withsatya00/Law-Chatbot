from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.phase2 import TaskEntry

CaseStatus = Literal["active", "on_hold", "closed"]


class HearingEntry(BaseModel):
    hearing_id: str | None = None
    hearing_date: datetime
    purpose: str = ""
    notes: str = ""
    reminder_days_before: int = 3
    reminder_acknowledged: bool = False
    added_at: datetime | None = None


class NoteEntry(BaseModel):
    text: str
    added_at: datetime | None = None


class CaseCreateRequest(BaseModel):
    case_number: str = Field(min_length=1)
    title: str = Field(min_length=1)
    case_type: str = ""
    court_name: str = ""
    client_name: str = ""
    client_contact: str = ""
    opposite_party: str = ""
    opposite_party_advocate: str = ""
    filing_date: datetime | None = None
    legal_category: str = ""
    parties: list[str] = Field(default_factory=list)
    next_action: str = ""


class CaseUpdateRequest(BaseModel):
    """Every field optional -- only the ones present in the request body get
    changed (`CaseService.update` only writes keys the caller actually
    supplied, via `model_dump(exclude_unset=True)`), so a client can patch
    just `status` without needing to resend the whole case.
    """

    case_number: str | None = None
    title: str | None = None
    status: CaseStatus | None = None
    case_type: str | None = None
    court_name: str | None = None
    client_name: str | None = None
    client_contact: str | None = None
    opposite_party: str | None = None
    opposite_party_advocate: str | None = None
    filing_date: datetime | None = None
    legal_category: str | None = None
    parties: list[str] | None = None
    next_action: str | None = None


class AddHearingRequest(BaseModel):
    hearing_date: datetime
    purpose: str = ""
    notes: str = ""
    reminder_days_before: int = Field(default=3, ge=0, le=30)


class AddNoteRequest(BaseModel):
    text: str = Field(min_length=1)


class AttachDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1)
    label: str = ""


class LinkDraftRequest(BaseModel):
    draft_id: str = Field(min_length=1)


class AddTimelineEventRequest(BaseModel):
    occurred_on: str = ""
    description: str = Field(min_length=1)
    source: str = "user"
    reference: str = ""


class ResolveCaseConflictRequest(BaseModel):
    slot: str = Field(min_length=1)
    chosen_value: str = Field(min_length=1)


class CaseResponse(BaseModel):
    case_id: str
    case_number: str
    title: str
    status: CaseStatus
    case_type: str
    court_name: str
    client_name: str
    client_contact: str
    opposite_party: str
    opposite_party_advocate: str
    filing_date: datetime | None
    next_hearing_date: datetime | None
    hearings: list[HearingEntry]
    notes: list[NoteEntry]
    document_ids: list[str]
    created_at: datetime
    updated_at: datetime
    legal_category: str = ""
    parties: list[str] = Field(default_factory=list)
    tasks: list[TaskEntry] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    next_action: str = ""
    reminders: list[dict[str, Any]] = Field(default_factory=list)
    linked_draft_ids: list[str] = Field(default_factory=list)
    resolved_conflicts: dict[str, str] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class CaseListResponse(BaseModel):
    cases: list[CaseResponse]
    count: int


class HearingReminderResponse(BaseModel):
    case_id: str
    case_title: str
    hearing_id: str
    hearing_date: datetime
    purpose: str
    days_until: int


class HearingReminderListResponse(BaseModel):
    reminders: list[HearingReminderResponse]
    count: int
