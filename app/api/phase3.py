
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.api.deps import get_current_user_claims
from app.core.config import settings
from app.core.exceptions import ForbiddenError, NotFoundError
from app.repositories.chat_history import ChatRepository
from app.repositories.documents import EmbeddingMetadataRepository
from app.repositories.phase3 import DownloadArtifactRepository
from app.schemas.phase3 import (
    BackgroundJobRequest,
    BackgroundJobResponse,
    FollowUpRequest,
    FollowUpResponse,
    FormConfirmRequest,
    FormPrefillRequest,
    FormUpdateRequest,
    FormWorkflowResponse,
    LegacyReferenceRequest,
    LegacyReferenceResponse,
    UserPreferencesRequest,
    UserPreferencesResponse,
)
from app.services.phase3 import (
    BackgroundJobService,
    FormWorkflowService,
    LegalUpdateService,
    PreferenceService,
    next_follow_up,
)

router = APIRouter(prefix="/assistant", tags=["phase-3"])


def _owner(claims: dict[str, Any] = Depends(get_current_user_claims)) -> str:
    return str(claims["sub"])


@router.post("/follow-up", response_model=FollowUpResponse)
async def proactive_follow_up(request: FollowUpRequest) -> FollowUpResponse:
    return next_follow_up(request)


@router.get("/preferences", response_model=UserPreferencesResponse)
async def get_preferences(owner_user_id: str = Depends(_owner)) -> UserPreferencesResponse:
    return await PreferenceService().get(owner_user_id)


@router.patch("/preferences", response_model=UserPreferencesResponse)
async def update_preferences(
    request: UserPreferencesRequest, owner_user_id: str = Depends(_owner)
) -> UserPreferencesResponse:
    return await PreferenceService().update(owner_user_id, request)


@router.delete("/preferences")
async def delete_preferences(owner_user_id: str = Depends(_owner)) -> dict[str, str]:
    await PreferenceService().delete(owner_user_id)
    return {"status": "deleted"}


@router.post("/forms", response_model=FormWorkflowResponse)
async def create_form(request: FormPrefillRequest, owner_user_id: str = Depends(_owner)) -> FormWorkflowResponse:
    return await FormWorkflowService().create(owner_user_id, request)


@router.get("/forms", response_model=list[FormWorkflowResponse])
async def list_forms(owner_user_id: str = Depends(_owner)) -> list[FormWorkflowResponse]:
    return await FormWorkflowService().list(owner_user_id)


@router.get("/forms/{workflow_id}", response_model=FormWorkflowResponse)
async def get_form(workflow_id: str, owner_user_id: str = Depends(_owner)) -> FormWorkflowResponse:
    return await FormWorkflowService().get(owner_user_id, workflow_id)


@router.patch("/forms/{workflow_id}", response_model=FormWorkflowResponse)
async def update_form(
    workflow_id: str, request: FormUpdateRequest, owner_user_id: str = Depends(_owner)
) -> FormWorkflowResponse:
    return await FormWorkflowService().update(owner_user_id, workflow_id, request.fields)


@router.post("/forms/{workflow_id}/confirm", response_model=FormWorkflowResponse)
async def confirm_form(
    workflow_id: str, request: FormConfirmRequest, owner_user_id: str = Depends(_owner)
) -> FormWorkflowResponse:
    return await FormWorkflowService().confirm(owner_user_id, workflow_id, request.confirmation_text)


@router.post("/jobs", response_model=BackgroundJobResponse)
async def create_background_job(
    request: BackgroundJobRequest, owner_user_id: str = Depends(_owner)
) -> BackgroundJobResponse:
    return await BackgroundJobService().create(owner_user_id, request)


@router.get("/jobs/{job_id}", response_model=BackgroundJobResponse)
async def get_background_job(job_id: str, owner_user_id: str = Depends(_owner)) -> BackgroundJobResponse:
    return await BackgroundJobService().get(owner_user_id, job_id)


@router.get("/jobs", response_model=list[BackgroundJobResponse])
async def list_background_jobs(owner_user_id: str = Depends(_owner)) -> list[BackgroundJobResponse]:
    return await BackgroundJobService().list(owner_user_id)


@router.post("/jobs/{job_id}/retry", response_model=BackgroundJobResponse)
async def retry_background_job(job_id: str, owner_user_id: str = Depends(_owner)) -> BackgroundJobResponse:
    return await BackgroundJobService().retry(owner_user_id, job_id)


@router.get("/downloads")
async def download_center(owner_user_id: str = Depends(_owner)) -> dict[str, Any]:
    items = await DownloadArtifactRepository().list_for_owner(owner_user_id)
    return {
        "items": [
            {"artifact_id": item["_id"], "filename": item["filename"], "format": item["format"], "created_at": item["created_at"]}
            for item in items
        ]
    }


@router.get("/downloads/{artifact_id}")
async def download_artifact(artifact_id: str, owner_user_id: str = Depends(_owner)) -> FileResponse:
    item = await DownloadArtifactRepository().find_by_id(artifact_id)
    if item is None:
        raise NotFoundError("Download artifact not found.")
    if item.get("owner_user_id") != owner_user_id:
        raise ForbiddenError("You do not have access to this download.")
    path = Path(item["path"]).resolve()
    allowed_root = settings.draft_output_dir.resolve()
    if allowed_root not in path.parents or not path.is_file():
        raise NotFoundError("Download file is unavailable.")
    return FileResponse(path, filename=item["filename"])


@router.post("/legacy-reference", response_model=LegacyReferenceResponse)
async def map_legacy_reference(request: LegacyReferenceRequest) -> LegacyReferenceResponse:
    return LegalUpdateService().map_legacy(request.code, request.section)


@router.get("/search")
async def search_my_content(
    q: str = Query(..., min_length=2, max_length=200), owner_user_id: str = Depends(_owner)
) -> dict[str, Any]:
    pattern = re.compile(re.escape(q), re.IGNORECASE)
    chats = await ChatRepository().search_for_user(owner_user_id, q, limit=30)
    document_cursor = EmbeddingMetadataRepository().collection.find(
        {"metadata.owner_user_id": owner_user_id, "text": pattern},
        {"text": 1, "document_id": 1, "metadata.source_document": 1},
    ).limit(30)
    documents = [item async for item in document_cursor]
    return {
        "chat_results": [
            {"session_id": item.get("session_id"), "question": item.get("question", ""), "answer": item.get("answer", "")[:400], "created_at": item.get("created_at")}
            for item in chats
        ],
        "document_results": [
            {"document_id": item.get("document_id"), "document_name": item.get("metadata", {}).get("source_document", ""), "snippet": item.get("text", "")[:500]}
            for item in documents
        ],
    }
