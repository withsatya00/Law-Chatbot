from typing import Any

from pydantic import BaseModel, Field


class CacheFlushRequest(BaseModel):
    question: str | None = None
    language: str | None = None
    intent: str | None = None


class AssignDocumentOwnershipRequest(BaseModel):
    owner_user_id: str | None = None


class KnowledgeBaseUploadResponse(BaseModel):
    status: str  # "pending" | "processing" | "indexed" | "duplicate" | "failed"
    staging_id: str | None = None
    original_filename: str
    generated_filename: str | None = None
    document_id: str | None = None
    content_hash: str | None = None
    document_type: str | None = None
    language: str | None = None
    chunks_indexed: int | None = None
    reason: str | None = None
    # Phase 1 "Jurisdiction-Aware Knowledge Base": present once jurisdiction
    # metadata was supplied and normalized (see `app.rag.kb_jurisdiction`).
    # `review_status="needs_review"` tells the admin this document was
    # indexed but is NOT yet visible to shared retrieval -- see
    # `review_reasons` for exactly what is missing.
    review_status: str | None = None
    review_reasons: list[str] | None = None


class KnowledgeBaseJurisdictionUpdateResponse(BaseModel):
    """Phase 1 gap 2: response for correcting/publishing an already-indexed
    document's jurisdiction metadata without a re-upload."""

    document_id: str
    source_document: str | None = None
    chunks_updated: int
    review_status: str
    review_reasons: list[str]


class AutomationReviewQueueItem(BaseModel):
    """One row of `KnowledgeBaseIngestionService.list_automation_review_queue`
    -- everything a reviewer needs to decide `verification_status` without
    opening the underlying job record."""

    document_id: str
    job_id: str
    title: str | None = None
    jurisdiction_code: str | None = None
    applicable_state_codes: list[str] = []
    document_type: str | None = None
    act_number: str | None = None
    enactment_year: int | None = None
    source_url: str | None = None
    issuing_level: str | None = None
    applicability: str | None = None
    confidence_score: float | None = None
    updated_at: Any = None


class BulkReviewRequest(BaseModel):
    """A human reviewer's batch decision: every `document_id` here is one
    they looked at and are choosing to mark `verified`. Not a partial-field
    patch and not an auto-approve -- see
    `KnowledgeBaseIngestionService.bulk_review_automated_documents`."""

    document_ids: list[str] = Field(min_length=1, max_length=200)
    verified_by: str = Field(min_length=1)


class BulkReviewResultItem(BaseModel):
    document_id: str
    outcome: str
    review_reasons: list[str] = []
    chunks_updated: int | None = None
    reason: str | None = None


class BulkReviewResponse(BaseModel):
    results: list[BulkReviewResultItem]
