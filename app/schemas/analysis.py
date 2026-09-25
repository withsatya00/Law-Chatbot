from typing import Any

from pydantic import BaseModel, Field

from app.schemas.extraction import (
    ExtractedEntity,
    RoleCandidate,
    RoleConflict,
    TimelineEvent,
)


class IntentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    language: str | None = None


class IntentResponse(BaseModel):
    intent: str
    legal_category: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class EntityRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    language: str | None = None


class EntityResponse(BaseModel):
    entities: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    # Phase 3 structured extraction. Additive and defaulted, so every existing
    # client keeps the response it already parses: `entities` above is
    # unchanged, and these carry the evidence (source text, page, method,
    # confidence) that a flat `{"type": ["value"]}` map cannot express.
    #
    # `unresolved_roles` and `conflicts` are deliberately NOT folded into
    # `structured`: a role the text hints at and a role the text states are
    # different kinds of claim, and merging them would let a suggestion be
    # read as a finding.
    structured: list[ExtractedEntity] = Field(default_factory=list)
    unresolved_roles: list[RoleCandidate] = Field(default_factory=list)
    conflicts: list[RoleConflict] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    undated_events: list[TimelineEvent] = Field(default_factory=list)
    date_contradictions: list[str] = Field(default_factory=list)
    #: What to ask the user before relying on any of the above.
    questions: list[str] = Field(default_factory=list)
