from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SupportedWorkflowLanguage = Literal["english", "hindi", "hinglish"]
FraudType = Literal[
    "upi_fraud", "online_banking_fraud", "card_fraud", "otp_scam",
    "investment_scam", "account_takeover", "unknown",
]


class SourceLink(BaseModel):
    title: str
    url: str
    authority: str
    verified_on: str


class EvidenceInput(BaseModel):
    evidence_id: str
    document_name: str
    text: str = ""
    description: str = ""
    uploaded_at: str = ""
    annexure: str = ""


class CyberFraudRequest(BaseModel):
    narrative: str = Field(min_length=1, max_length=30_000)
    language: SupportedWorkflowLanguage = "english"
    fraud_type: FraudType | None = None
    amount: str | None = None
    transaction_time: str | None = None
    transaction_id: str | None = None
    bank: str | None = None
    platform: str | None = None
    sender_details: str | None = None
    receiver_details: str | None = None
    complainant_name: str | None = None
    complainant_address: str | None = None
    police_station: str | None = None
    evidence: list[EvidenceInput] = Field(default_factory=list)
    resolved_conflicts: dict[str, str] = Field(default_factory=dict)


class CyberFraudResponse(BaseModel):
    detected_fraud_type: FraudType
    risk_level: Literal["medium", "high", "critical"]
    immediate_steps: list[str]
    emergency_options: list[dict[str, Any]]
    required_information: list[dict[str, Any]]
    extracted_facts: list[dict[str, Any]]
    evidence_table: list[dict[str, Any]]
    evidence_checklist: list[dict[str, Any]]
    timeline: list[dict[str, Any]]
    contradictions: list[dict[str, Any]]
    export_blocked: bool
    drafts: dict[str, str]
    sources: list[SourceLink]
    lawyer_escalation_recommended: bool
    warning: str


MatterType = Literal["police_complaint", "consumer_complaint", "rti", "property", "cyber_complaint"]


class JurisdictionRequest(BaseModel):
    matter_type: MatterType
    complainant_location: str | None = None
    respondent_location: str | None = None
    incident_location: str | None = None
    transaction_location: str | None = None
    property_location: str | None = None
    public_authority_location: str | None = None
    branch_location: str | None = None
    online_transaction: bool = False
    language: SupportedWorkflowLanguage = "english"


class JurisdictionResponse(BaseModel):
    tentative_forum: str
    possible_bases: list[str]
    missing_facts: list[str]
    confidence: Literal["low", "medium"]
    verify_before_filing: bool = True
    warning: str


class TimelineRequest(BaseModel):
    messages: list[dict[str, str]] = Field(default_factory=list)
    evidence: list[EvidenceInput] = Field(default_factory=list)
    draft_fields: dict[str, str] = Field(default_factory=dict)
    resolved_conflicts: dict[str, str] = Field(default_factory=dict)


class TimelineResponse(BaseModel):
    timeline: list[dict[str, Any]]
    extracted_facts: list[dict[str, Any]]
    contradictions: list[dict[str, Any]]
    export_blocked: bool


class ConflictResolutionRequest(BaseModel):
    slot: str = Field(min_length=1)
    chosen_value: str = Field(min_length=1)


class TaskEntry(BaseModel):
    task_id: str | None = None
    title: str = Field(min_length=1)
    due_at: datetime | None = None
    status: Literal["pending", "completed"] = "pending"
    reminder_at: datetime | None = None


class LawyerSummaryRequest(BaseModel):
    include_draft_text: bool = False


class LawyerSummaryResponse(BaseModel):
    case_id: str
    summary: str
    facts: list[str]
    timeline: list[dict[str, Any]]
    key_documents: list[dict[str, Any]]
    draft_status: list[dict[str, str]]
    unanswered_questions: list[str]
    sharing: dict[str, Any]


class DraftReviewResponse(BaseModel):
    draft_id: str
    user_facts: dict[str, Any]
    extracted_facts: list[dict[str, Any]]
    legal_references: list[str]
    missing_information: list[str]
    possible_assumptions: list[dict[str, str]]
    evidence_annexures: list[dict[str, Any]]
    editable_sections: dict[str, str]
    version_history: list[dict[str, Any]]
    lifecycle_state: str
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    final_export_blocked: bool = False


class DraftCompareResponse(BaseModel):
    draft_id: str
    original_version: int
    revised_version: int
    original_text: str
    revised_text: str
    unified_diff: str


class DraftDuplicateRequest(BaseModel):
    session_id: str | None = None


class DraftDuplicateResponse(BaseModel):
    source_draft_id: str
    draft_id: str


class EvidenceOrganizeRequest(BaseModel):
    evidence: list[EvidenceInput]
    series: str = Field(default="A", pattern=r"^[A-Z]{1,3}$")


class EvidenceOrganizeResponse(BaseModel):
    evidence_table: list[dict[str, Any]]
    annexure_index: str
