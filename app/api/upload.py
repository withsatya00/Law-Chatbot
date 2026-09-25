from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_current_user_id
from app.chatops.artifacts import MEDIA_TYPES
from app.core.config import settings
from app.core.exceptions import BadRequestError, NotFoundError, UnauthorizedError
from app.memory.store import ConversationMemoryStore
from app.repositories.document_transcripts import DocumentTranscriptRepository
from app.schemas.document import UploadResponse
from app.schemas.document_extraction import DocumentExtractionResponse
from app.services.document_service import DocumentService, ensure_document_access

router = APIRouter(tags=["documents"])


@router.post("/documents/extract", response_model=DocumentExtractionResponse)
async def extract_document(
    file: UploadFile = File(...),
    user_id: str | None = Depends(get_current_user_id),
) -> DocumentExtractionResponse:
    """Transcribe an upload and return editable exports without publishing it to the KB."""
    if settings.environment in {"staging", "production"} and not user_id:
        raise UnauthorizedError("Sign in before extracting a private document.")
    from app.services.document_extraction import DocumentExtractionService

    return await DocumentExtractionService().extract_upload(file)


@router.get("/documents")
async def list_documents(
    session_id: str | None = Query(default=None),
    user_id: str | None = Depends(get_current_user_id),
) -> dict[str, list[dict[str, Any]]]:
    """List only documents owned by the JWT user or supplied conversation."""
    documents = await DocumentService().list_owned(user_id=user_id, session_id=session_id)
    return {"documents": documents}


@router.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    user_id: str | None = Depends(get_current_user_id),
) -> UploadResponse:
    if settings.environment in {"staging", "production"} and not user_id:
        raise UnauthorizedError("Sign in before uploading a private document.")
    if session_id:
        memory_store = ConversationMemoryStore()
        memory = await memory_store.check_access(session_id, user_id)
        if user_id and not memory.get("owner_user_id"):
            await memory_store.update(session_id, owner_user_id=user_id)
    return await DocumentService().upload_and_index(file, session_id=session_id, user_id=user_id)


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    session_id: str | None = Query(default=None),
    user_id: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """PDF Q&A acceptance pass (2026-09-12): previously there was no way to
    delete an uploaded document at all -- `DELETE /session`/`DELETE
    /me/data` erase conversation memory and chat history, never indexed
    document content, so a "deleted" document stayed fully queryable
    (including through the chat-driven document workflows) indefinitely.

    Also drops the document from the session's own `uploaded_documents`
    list/`last_uploaded_document_id` pointer when `session_id` is given, so
    a deleted document can never again be silently selected as "the active
    document" for a later question (`app/chatops/workflows/documents.py`'s
    `_conversation_documents`/`_resolve_document` both read from exactly
    this memory state).
    """
    chunks_deleted = await DocumentService().delete_owned(document_id, authenticated_user_id=user_id, session_id=session_id)
    if session_id:
        memory_store = ConversationMemoryStore()
        memory = await memory_store.check_access(session_id, user_id)
        remaining = [
            item for item in (memory.get("uploaded_documents") or [])
            if isinstance(item, dict) and item.get("document_id") != document_id
        ]
        updates: dict[str, Any] = {"uploaded_documents": remaining}
        if memory.get("last_uploaded_document_id") == document_id:
            updates["last_uploaded_document_id"] = remaining[-1]["document_id"] if remaining else None
        await memory_store.update(session_id, **updates)
    return {"status": "deleted", "document_id": document_id, "chunks_deleted": chunks_deleted}


@router.get("/documents/{document_id}/transcript/export")
async def export_document_transcript(
    document_id: str,
    fmt: Literal["pdf", "docx", "txt"] = Query(...),
    session_id: str | None = Query(default=None),
    user_id: str | None = Depends(get_current_user_id),
) -> FileResponse:
    """Downloads the most recent `DocumentTranscribeWorkflow` retype of an
    uploaded document, in the requested format.

    Re-renders from the STORED transcript (`DocumentTranscriptRepository`),
    never re-runs the LLM transcription -- a download is not the moment to
    pay for, or risk a different result from, another model call. Reuses
    the same `PdfDraftExporter`/`DocxDraftExporter` the drafting engine's
    `/draft/export` already uses, so Unicode/RTL script handling comes for
    free, with `flowing_letter=True` so the already fully-formatted
    `formatted_text` prints as one continuous document instead of being
    split under a drafting template's own section headings.
    """
    record = await DocumentTranscriptRepository().find_by_document_id(document_id)
    if record is None:
        raise NotFoundError(
            "No formatted version of this document has been generated yet. "
            "Ask to have it formatted/transcribed first."
        )
    ensure_document_access(
        {"owner_user_id": record.get("owner_user_id"), "owner_session_id": record.get("owner_session_id")},
        user_id, session_id,
    )
    data = record["data"]
    formatted_text = data.get("formatted_text", "")
    if not formatted_text.strip():
        raise BadRequestError("This document's formatted text is empty; nothing to export.")

    settings.draft_output_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"pdf": ".pdf", "docx": ".docx", "txt": ".txt"}[fmt]
    output_path = settings.draft_output_dir / f"{uuid4()}{suffix}"
    title = data.get("document_type") or "Document"
    # Font/RTL lookup in `app.drafting.export` keys on the app's own lowercase
    # language names ("hindi", "urdu", ...); normalize defensively since the
    # transcription prompt's own free-text `primary_language` field is not
    # guaranteed to come back in that exact casing.
    language = (data.get("primary_language") or "english").strip().lower()

    if fmt == "txt":
        output_path.write_text(f"{title}\n\n{formatted_text}", encoding="utf-8")
    else:
        from app.drafting.export import DocxDraftExporter, ExportOptions, PdfDraftExporter

        exporter = PdfDraftExporter() if fmt == "pdf" else DocxDraftExporter()
        exporter.export(
            title, {"Document": formatted_text}, output_path, language,
            ExportOptions(flowing_letter=True, include_header_footer=True),
        )
    return FileResponse(output_path, filename=f"{document_id}.{fmt}", media_type=MEDIA_TYPES[fmt])
