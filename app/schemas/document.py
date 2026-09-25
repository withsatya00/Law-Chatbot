from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import LawyerRecommendation, SourceCitation


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_indexed: int
    detected_language: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # User-facing notices about this specific upload (e.g. a scanned PDF whose
    # OCR was unavailable or added nothing, so it was indexed from its
    # near-empty embedded text layer). Empty on the normal, non-degraded path.
    warnings: list[str] = Field(default_factory=list)


class DocumentAnalysisRequest(BaseModel):
    document_id: str | None = None
    text: str | None = Field(default=None, max_length=120_000)
    analysis_type: str = "general"
    language: str | None = None
    # Part 46 "Authenticated User Ownership": lets a Part-45-era
    # session-scoped (anonymous) document stay analyzable by its original
    # session -- without this, a private document with no `owner_user_id`
    # would become unanalyzable by anyone once ownership checks are added.
    session_id: str | None = None


class AnalyzedClause(BaseModel):
    name: str
    text: str
    explanation: str
    risk_level: str = "low"
    risk_reason: str = ""


class DocumentTimelineEvent(BaseModel):
    date: str
    event_description: str
    source_text: str = ""


class DocumentAnalysisResponse(BaseModel):
    executive_summary: str
    legal_summary: str
    important_clauses: list[str]
    important_dates: list[str]
    important_names: list[str]
    important_sections: list[str]
    key_risks: list[str]
    action_items: list[str]
    missing_information: list[str]
    structured_data: dict[str, Any]
    sources: list[SourceCitation]
    recommended_lawyer: LawyerRecommendation
    confidence: float = Field(ge=0.0, le=1.0)
    analyzed_clauses: list[AnalyzedClause] = Field(default_factory=list)
    timeline: list[DocumentTimelineEvent] = Field(default_factory=list)
