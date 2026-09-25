from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from app.api.deps import get_current_user_claims
from app.schemas.case import (
    AddHearingRequest,
    AddNoteRequest,
    AddTimelineEventRequest,
    AttachDocumentRequest,
    CaseCreateRequest,
    CaseListResponse,
    CaseResponse,
    CaseUpdateRequest,
    HearingReminderListResponse,
    LinkDraftRequest,
    ResolveCaseConflictRequest,
)
from app.schemas.phase2 import LawyerSummaryResponse, TaskEntry
from app.services.case_service import CaseService
from app.services.document_service import DocumentService

# Every route requires a real authenticated user (Part 46's JWT identity) --
# a case is private advocate/user data, never something an anonymous chat
# session should be able to create or browse the way a draft or an uploaded
# document can be. `get_current_user_claims` (not the optional
# `get_current_user_id`) enforces that at the router level: no valid bearer
# token -> 401 before any handler runs.
router = APIRouter(prefix="/cases", tags=["cases"], dependencies=[Depends(get_current_user_claims)])


def _owner_id(claims: dict[str, Any] = Depends(get_current_user_claims)) -> str:
    return str(claims["sub"])


@router.post("", response_model=CaseResponse)
async def create_case(request: CaseCreateRequest, owner_user_id: str = Depends(_owner_id)) -> CaseResponse:
    return await CaseService().create(owner_user_id, request)


@router.get("", response_model=CaseListResponse)
async def list_cases(
    status: str | None = Query(None, pattern="^(active|on_hold|closed)$"),
    owner_user_id: str = Depends(_owner_id),
) -> CaseListResponse:
    cases = await CaseService().list_for_owner(owner_user_id, status=status)
    return CaseListResponse(cases=cases, count=len(cases))


@router.get("/upcoming-hearings", response_model=CaseListResponse)
async def upcoming_hearings(
    within_days: int = Query(14, ge=1, le=180),
    owner_user_id: str = Depends(_owner_id),
) -> CaseListResponse:
    """Cases with a hearing coming up within `within_days` -- the data
    surface a future Court Calendar view is expected to read from.
    """
    cases = await CaseService().upcoming_hearings(owner_user_id, within_days)
    return CaseListResponse(cases=cases, count=len(cases))


@router.get("/hearing-reminders", response_model=HearingReminderListResponse)
async def hearing_reminders(owner_user_id: str = Depends(_owner_id)) -> HearingReminderListResponse:
    """Distinct from `/upcoming-hearings` above -- reads each hearing's own
    `reminder_days_before` threshold and excludes already-acknowledged ones
    (see `CaseService.hearing_reminders`), so this is the "what actually
    needs a nudge right now" surface, not "everything in a fixed window."
    Registered before `/{case_id}` -- otherwise FastAPI would match this
    path as a `case_id` value (same reason `/upcoming-hearings` precedes it).
    """
    reminders = await CaseService().hearing_reminders(owner_user_id)
    return HearingReminderListResponse(reminders=reminders, count=len(reminders))


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(case_id: str, owner_user_id: str = Depends(_owner_id)) -> CaseResponse:
    return await CaseService().get(owner_user_id, case_id)


@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: str, request: CaseUpdateRequest, owner_user_id: str = Depends(_owner_id)
) -> CaseResponse:
    return await CaseService().update(owner_user_id, case_id, request)


@router.delete("/{case_id}")
async def delete_case(case_id: str, owner_user_id: str = Depends(_owner_id)) -> dict[str, Any]:
    await CaseService().delete(owner_user_id, case_id)
    return {"status": "deleted", "case_id": case_id}


@router.post("/{case_id}/hearings", response_model=CaseResponse)
async def add_hearing(
    case_id: str, request: AddHearingRequest, owner_user_id: str = Depends(_owner_id)
) -> CaseResponse:
    return await CaseService().add_hearing(owner_user_id, case_id, request)


@router.post("/{case_id}/hearings/{hearing_id}/acknowledge-reminder", response_model=CaseResponse)
async def acknowledge_hearing_reminder(
    case_id: str, hearing_id: str, owner_user_id: str = Depends(_owner_id)
) -> CaseResponse:
    return await CaseService().acknowledge_hearing_reminder(owner_user_id, case_id, hearing_id)


@router.post("/{case_id}/notes", response_model=CaseResponse)
async def add_note(case_id: str, request: AddNoteRequest, owner_user_id: str = Depends(_owner_id)) -> CaseResponse:
    return await CaseService().add_note(owner_user_id, case_id, request)


@router.post("/{case_id}/documents", response_model=CaseResponse)
async def attach_document(
    case_id: str, request: AttachDocumentRequest, owner_user_id: str = Depends(_owner_id)
) -> CaseResponse:
    return await CaseService().attach_document(owner_user_id, case_id, request.document_id)


@router.post("/{case_id}/evidence/upload", response_model=CaseResponse)
async def upload_case_evidence(
    case_id: str,
    file: UploadFile = File(...),
    description: str = Form(""),
    owner_user_id: str = Depends(_owner_id),
) -> CaseResponse:
    """Uploads, OCRs/parses, extracts and annexure-numbers private case evidence."""
    cases = CaseService()
    await cases.get(owner_user_id, case_id)  # ownership before file processing
    documents = DocumentService()
    uploaded = await documents.upload_and_index(file, user_id=owner_user_id)
    cursor = documents.embeddings.collection.find({"document_id": uploaded.document_id}, {"text": 1})
    # `str.join` takes a synchronous iterable; handing it an async generator
    # raises `TypeError: can only join an iterable` before a single chunk is
    # read. Materialise the cursor first.
    text = "\n\n".join([chunk.get("text", "") async for chunk in cursor])
    return await cases.add_evidence(
        owner_user_id,
        case_id,
        document_id=uploaded.document_id,
        document_name=uploaded.filename,
        text=text,
        description=description,
    )


@router.post("/{case_id}/drafts", response_model=CaseResponse)
async def link_draft(case_id: str, request: LinkDraftRequest, owner_user_id: str = Depends(_owner_id)) -> CaseResponse:
    return await CaseService().link_draft(owner_user_id, case_id, request.draft_id)


@router.post("/{case_id}/tasks", response_model=CaseResponse)
async def add_task(case_id: str, request: TaskEntry, owner_user_id: str = Depends(_owner_id)) -> CaseResponse:
    return await CaseService().add_task(owner_user_id, case_id, request)


@router.post("/{case_id}/timeline", response_model=CaseResponse)
async def add_timeline_event(case_id: str, request: AddTimelineEventRequest, owner_user_id: str = Depends(_owner_id)) -> CaseResponse:
    return await CaseService().add_timeline_event(owner_user_id, case_id, request)


@router.post("/{case_id}/conflicts/resolve", response_model=CaseResponse)
async def resolve_case_conflict(case_id: str, request: ResolveCaseConflictRequest, owner_user_id: str = Depends(_owner_id)) -> CaseResponse:
    return await CaseService().resolve_conflict(owner_user_id, case_id, request.slot, request.chosen_value)


@router.get("/{case_id}/lawyer-summary", response_model=LawyerSummaryResponse)
async def lawyer_summary(case_id: str, owner_user_id: str = Depends(_owner_id)) -> LawyerSummaryResponse:
    """Builds a private summary. It never sends or shares data externally."""
    return await CaseService().lawyer_summary(owner_user_id, case_id)
