from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

VerificationStatus = Literal["unverified", "pending_review", "verified", "rejected"]
SourceStatus = Literal["in_force", "amended", "repealed", "superseded", "unknown"]


SourceType = Literal[
    "bare_act", "amendment_act", "ordinance", "rules", "regulation", "notification",
    "circular", "judgment", "commentary", "faq", "form", "unknown",
]


class LegalSourceMetadata(BaseModel):
    """Governance record for one legal source.

    Phase 2: the registry is what lets an answer say "this is current law, and
    here is who checked". Every field below is either supplied by a human
    reviewer or derived from the file itself (`checksum`, `ingestion_version`).
    NOTHING here is inferred from a filename -- a document called
    `bns_2023_official.pdf` is still `unverified` until a reviewer records
    evidence, because a plausible filename is not provenance.
    """

    title: str = Field(min_length=1)
    # The source's own title as printed on it, distinct from `title`, which is
    # whatever the uploader called it.
    official_title: str = ""
    source_type: SourceType = "unknown"
    issuing_authority: str = ""
    source_url: str = ""
    jurisdiction: str = "India"
    act_name: str = ""
    section_number: str = ""
    source_version: str = ""
    publication_date: date | None = None
    effective_date: date | None = None
    status: SourceStatus = "unknown"
    amendment_notes: str = ""
    last_verified_date: date | None = None
    update_date: date | None = None
    # Defaults to `unverified`, not `pending_review`: a record nobody has
    # queued for review has not entered the review workflow at all, and
    # conflating the two overstated how much of the corpus was being looked at.
    verification_status: VerificationStatus = "unverified"
    # SHA-256 of the ingested bytes. Lets a later run detect that a "verified"
    # source's file changed underneath the verification.
    checksum: str = ""
    ingestion_version: str = ""
    # Amendment lineage. Both hold `source_id` values in this same registry.
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: str | None = None


class LegalSourceResponse(LegalSourceMetadata):
    source_id: str
    owner_user_id: str
    document_id: str | None = None
    staging_id: str | None = None
    # Chunk ids in `embeddings_metadata` this source governs. Populated by
    # `link_document`, so a governance change can be traced to the exact
    # retrievable text it affects.
    chunk_ids: list[str] = Field(default_factory=list)
    stale: bool = False
    current_as_of: str
    # Who last changed the verification state, and when. Never the reviewer's
    # email or any other PII beyond the account id the admin routes already use.
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_notes: str = ""
    evidence_url: str = ""
    created_at: datetime
    updated_at: datetime


class SourceReviewRequest(BaseModel):
    """Evidence required to move a source OUT of `unverified`.

    `verify` is deliberately not a bare status flip. Marking a source verified
    is an assertion that a human compared it against the issuing authority's
    own text, so the call has to carry what was checked (`evidence_url`), when
    (`last_verified_date`) and what the reviewer concluded (`review_notes`).
    Without those the registry records a claim nobody can audit.
    """

    verification_status: Literal["verified", "rejected", "pending_review"]
    last_verified_date: date
    # `None` = "not supplied, leave whatever is already on the source alone"
    # -- distinct from `""`, which explicitly clears it (security finding
    # G3: a PATCH-style re-review that omits these must not erase evidence a
    # PRIOR review already recorded just because this particular call is
    # about a different field, e.g. moving a previously-verified source to
    # `rejected` without re-typing the same evidence URL).
    evidence_url: str | None = None
    review_notes: str | None = Field(default=None, max_length=2000)
    status: SourceStatus | None = None
    amendment_notes: str | None = None


class SupersedeSourceRequest(BaseModel):
    """Records that `superseded_by` replaces this source in law."""

    superseded_by_source_id: str = Field(min_length=1)
    effective_date: date | None = None
    review_notes: str = Field(default="", max_length=2000)


class VerifySourceRequest(BaseModel):
    """Security finding G2: this previously had no `evidence_url`/
    `review_notes` fields at all, so `POST /verify` could never satisfy
    `LegalUpdateService.review`'s requirement that a `verified` transition
    carry both -- every call attempting to verify a source through this
    endpoint was rejected with `BadRequestError`, regardless of what the
    caller sent, because there was nowhere for it to go. Now the same shape
    as `SourceReviewRequest`, which `verify()` has always delegated to."""

    verification_status: VerificationStatus
    last_verified_date: date
    evidence_url: str | None = None
    review_notes: str | None = Field(default=None, max_length=2000)
    status: SourceStatus | None = None
    amendment_notes: str | None = None


class LinkSourceDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1)


class LegacyReferenceRequest(BaseModel):
    code: Literal["IPC", "CrPC"]
    section: str = Field(min_length=1, max_length=20)


class LegacyReferenceResponse(BaseModel):
    legacy_reference: str
    current_code: str | None
    current_section: str | None
    note: str
    requires_legal_verification: bool = True
    official_source_url: str = "https://www.mha.gov.in/en/commoncontent/new-criminal-laws"


class UserPreferencesRequest(BaseModel):
    language: str | None = None
    explanation_mode: Literal["simple", "detailed", "advocate"] | None = None
    preferred_document_format: Literal["pdf", "docx", "txt", "rtf"] | None = None
    voice_output: bool | None = None
    # Jurisdiction Routing "Matter Context": the user's home State/UT, used as
    # the LOWEST-priority default when a matter's own location isn't stated
    # (see `app.rag.matter_context`). A code or a full name is accepted (same
    # as `kb_jurisdiction.normalize_state_code`); validated and stored as the
    # canonical code by `PreferenceService.update`. Never inferred or
    # auto-set from a chat message -- only this explicit profile write ever
    # changes it, so a one-off out-of-state matter never overwrites it.
    profile_state_code: str | None = None


class UserPreferencesResponse(BaseModel):
    language: str = "english"
    explanation_mode: Literal["simple", "detailed", "advocate"] = "simple"
    preferred_document_format: Literal["pdf", "docx", "txt", "rtf"] = "pdf"
    voice_output: bool = False
    profile_state_code: str | None = None
    updated_at: datetime | None = None


class FollowUpRequest(BaseModel):
    workflow: Literal["cyber_fraud", "police_complaint"]
    messages: list[dict[str, str]] = Field(default_factory=list)
    document_texts: list[str] = Field(default_factory=list)
    known_fields: dict[str, Any] = Field(default_factory=dict)
    asked_fields: list[str] = Field(default_factory=list)
    language: Literal["english", "hindi", "hinglish"] = "english"


class FollowUpResponse(BaseModel):
    field: str | None
    question: str | None
    complete: bool
    remaining_fields: list[str]


FormType = Literal["police_complaint", "cybercrime_complaint", "rti", "consumer_complaint"]


class FormPrefillRequest(BaseModel):
    form_type: FormType
    case_id: str | None = None
    facts: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    language: str = "english"


class FormUpdateRequest(BaseModel):
    fields: dict[str, Any]


class FormConfirmRequest(BaseModel):
    confirmation_text: str


class FormWorkflowResponse(BaseModel):
    workflow_id: str
    form_type: FormType
    fields: dict[str, Any]
    missing_fields: list[str]
    review_required: bool = True
    confirmed: bool = False
    external_submission: bool = False
    warning: str


class BackgroundJobRequest(BaseModel):
    job_type: Literal["ingestion", "ocr", "indexing", "export", "document_analysis"]
    payload: dict[str, Any] = Field(default_factory=dict)
    case_id: str | None = None


class BackgroundJobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: int = 0
    error: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class EvaluationRunRequest(BaseModel):
    benchmark_path: str | None = None


class EvaluationRunResponse(BaseModel):
    run_id: str
    status: str
    scores: dict[str, float]
    cases: int
    failures: list[dict[str, Any]]
    latency_ms: float


class VoiceDraftConfirmationRequest(BaseModel):
    session_id: str | None = None
    confirmation_text: str
