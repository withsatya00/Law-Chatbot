from typing import Literal

from pydantic import BaseModel, Field

from app.core.constants import DRAFT_DISCLAIMER


class DraftFieldSchema(BaseModel):
    key: str
    label: str
    hindi_label: str
    field_type: str
    required: bool
    help_text: str = ""


class DraftTemplateSummary(BaseModel):
    draft_id: str
    name: str
    hindi_name: str
    category: str
    description: str
    # Problem-first discovery metadata (see `DraftTemplateDefinition` in
    # `app/drafting/templates/base.py`). All optional/defaulted so existing
    # `/draft-templates` clients that only read the fields above are
    # completely unaffected -- these are purely additive.
    document_family: str = ""
    domain: str = ""
    subcategory: str = ""
    template_status: str = "production"


class DraftTemplateDetail(DraftTemplateSummary):
    applicable_acts_hint: list[str]
    applicable_sections_hint: list[str]
    required_fields: list[DraftFieldSchema]
    optional_fields: list[DraftFieldSchema]


class DraftGenerateRequest(BaseModel):
    draft_id: str = Field(min_length=1)
    language: str = "english"
    fields: dict[str, str] = Field(default_factory=dict)
    session_id: str | None = None
    user_id: str | None = None
    # "restyle" edit action (QA pass, 2026-09-11): a free-text tone/register
    # directive for regeneration ("polite but firm", "more formal") -- the
    # SAME facts, re-rendered with a different voice. Never a new fact, and
    # never itself written into the document; see `LegalDraftEngine.
    # _render_sections_within_deadline`'s use of it.
    style_instruction: str | None = None


class DraftPreviewRequest(DraftGenerateRequest):
    pass


DraftStatus = Literal["complete", "needs_more_info"]
# "llm": composed by the language model (the normal path).
# "deterministic": every LLM attempt failed; the text is the non-LLM
# skeleton and must be labelled as degraded wherever it is shown.
DraftGenerationMode = Literal["llm", "deterministic"]


class DraftGenerateResponse(BaseModel):
    status: DraftStatus
    draft_id: str | None = None
    template_id: str
    template_name: str
    missing_fields: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    sections: dict[str, str] = Field(default_factory=dict)
    full_text: str = ""
    applicable_acts: list[str] = Field(default_factory=list)
    applicable_sections: list[str] = Field(default_factory=list)
    disclaimer: str = DRAFT_DISCLAIMER
    language: str = "english"
    generated_by_llm: bool = False
    # How the text was actually produced. "llm" is the normal path;
    # "deterministic" means every LLM attempt (including retries and
    # configured fallback providers) failed and the document is the
    # non-LLM skeleton, which is structurally valid but far thinner than a
    # real advocate-style draft. Callers MUST surface "deterministic" to the
    # user -- silently presenting a skeleton as the finished document is the
    # exact failure this field exists to make impossible.
    generation_mode: DraftGenerationMode = "llm"
    # The provider error behind a "deterministic" result, for the chat layer
    # to explain and for logs. Never shown as document content.
    generation_error: str | None = None
    word_count: int = 0
    estimated_page_count: int = 0
    latency_ms: float = 0.0
    # Phase 1 item 6: advisory fact-only audit findings (see
    # `app/drafting/fact_audit.py`). Empty means nothing in the draft went
    # beyond the facts the user supplied. Never blocks generation or export --
    # it tells the reviewer which sentences to check before signing.
    audit_findings: list[dict[str, str]] = Field(default_factory=list)


class DraftTranslateRequest(BaseModel):
    draft_id: str | None = None
    session_id: str | None = None
    text: str | None = None
    target_language: str = Field(min_length=1)
    source_language: str = "english"


class DraftTranslateResponse(BaseModel):
    translated_text: str
    target_language: str
    disclaimer: str = DRAFT_DISCLAIMER


class LegalTermEntry(BaseModel):
    term: str
    meaning: str
    simple_explanation: str
    usage: str
    legal_example: str
    english_equivalent: str


class LegalTermsRequest(BaseModel):
    query: str | None = None


class LegalTermsResponse(BaseModel):
    terms: list[LegalTermEntry]


class DraftHistoryRequest(BaseModel):
    session_id: str | None = None
    user_id: str | None = None
    limit: int = Field(default=50, ge=1, le=200)


class DraftSummary(BaseModel):
    draft_id: str
    template_id: str
    template_name: str
    language: str
    status: str
    created_at: str


class DraftHistoryResponse(BaseModel):
    drafts: list[DraftSummary]


# Part 57 "Drafting Lifecycle Redesign": "selecting"/"collecting"/"preview"
# keep their original wire-format spelling (unchanged, several existing
# tests/frontend assertions depend on the exact strings) and map onto the
# spec's DRAFT_CREATED / COLLECTING_INFORMATION / PREVIEW_READY-and-EDITING
# states respectively -- "preview" covers both PREVIEW_READY and the
# transient EDITING turn (an edit command's reply), since editing is
# synchronous and settles back to preview_ready by the time a reply is
# returned. "approved"/"locked"/"exported" are new, added verbatim from the
# spec since nothing existing depends on their absence.
# Problem-first discovery adds a further set of stages that run BEFORE
# "collecting" for a generic ("draft banana hai") request -- see
# `DraftConversationEngine._start_discovery` in `conversation.py` and
# `app/drafting/discovery.py`. A direct, named request ("rent agreement
# banao") still skips straight to "collecting" exactly as before; these are
# additive, not a replacement for the existing stages.
DraftConversationStage = Literal[
    "selecting", "collecting", "preview", "approved", "locked", "exported",
    "describe_problem", "identify_role", "identify_relief", "identify_case_stage",
    "recommend_template", "confirm_template", "collect_jurisdiction", "confirm_summary",
]

# The persisted draft record's lifecycle state (`legal_drafts.lifecycle_state`
# in Mongo) -- distinct from `DraftConversationStage` above, which also
# covers the pre-persistence "selecting"/"collecting" phases where no draft
# record exists yet. Only meaningful once a draft has actually been
# generated at least once.
DraftLifecycleState = Literal["preview_ready", "editing", "approved", "locked", "exported"]


class DraftTurnInfo(BaseModel):
    """Carries the chatbot's current drafting state to the frontend.

    Embedded in `ChatResponse.draft` -- `None` there means the message wasn't
    drafting-related and was answered by the normal RAG chat flow instead.
    """

    stage: DraftConversationStage
    template_id: str | None = None
    template_name: str | None = None
    collected_summary: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    draft_id: str | None = None
    sections: dict[str, str] = Field(default_factory=dict)
    full_text: str = ""
    disclaimer: str = DRAFT_DISCLAIMER
    available_export_formats: list[str] = Field(default_factory=list)
    word_count: int = 0
    estimated_page_count: int = 0
    # Mirrors `DraftGenerateResponse.generation_mode`/`generation_error` so a
    # degraded (LLM-unavailable) draft is visibly marked in the chat UI, not
    # just in the API response.
    generation_mode: DraftGenerationMode = "llm"
    generation_error: str | None = None
    # Phase 1 item 6: same advisory audit as `DraftGenerateResponse`, surfaced
    # on the chat turn so the preview can flag unsupported sentences at the
    # moment the user first reads the draft.
    audit_findings: list[dict[str, str]] = Field(default_factory=list)
    # Part 57 "Drafting Lifecycle Redesign": the persisted draft's actual
    # `DraftLifecycleState` once one exists (None during "selecting"/
    # "collecting", before any draft record is generated). Distinct from
    # `stage` above -- `stage` also drives which chat handler runs next,
    # while this is purely informational for callers that want the precise
    # lifecycle value without inferring it from `stage`.
    lifecycle_state: str | None = None


class DraftMessageRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)
    language: str | None = None


class DraftMessageResponse(BaseModel):
    reply: str
    draft: DraftTurnInfo | None = None


class DraftEditRequest(BaseModel):
    draft_id: str = Field(min_length=1)
    session_id: str | None = None
    instruction: str | None = None
    target_field: str | None = None
    new_value: str | None = None
    target_language: str | None = None


class DraftExportRequest(BaseModel):
    draft_id: str = Field(min_length=1)
    session_id: str | None = None
    format: Literal["pdf", "docx", "txt", "rtf"] = "pdf"
    # Clean filing copy by default. A watermark remains an explicit opt-in for
    # internal review copies, rather than being stamped across every download.
    watermark: bool = False
    sign: bool = False
    watermark_text: str | None = Field(default=None, max_length=80)
    include_header_footer: bool = True


class DraftAuditRequest(BaseModel):
    draft_id: str = Field(min_length=1)
    session_id: str | None = None


class DraftAuditResponse(BaseModel):
    """Phase 1 item 6: the pre-export fact-only audit, on demand.

    `ok` is True when nothing in the draft went beyond the facts the user
    supplied. Findings are advisory -- export is never blocked -- so a client
    can show them beside the download button as "check these before signing".
    """

    draft_id: str
    ok: bool
    findings: list[dict[str, str]] = Field(default_factory=list)


class DraftLifecycleRequest(BaseModel):
    draft_id: str = Field(min_length=1)
    session_id: str | None = None


class DraftLifecycleResponse(BaseModel):
    draft_id: str
    lifecycle_state: DraftLifecycleState


class DraftRollbackRequest(BaseModel):
    draft_id: str = Field(min_length=1)
    session_id: str | None = None
    version_number: int = Field(ge=1)


class DraftVersionSummary(BaseModel):
    version_number: int
    document_status: str
    language: str
    created_at: str | None = None


class DraftVersionsResponse(BaseModel):
    draft_id: str
    versions: list[DraftVersionSummary]


# ---------------------------------------------------------------------------
# Problem-first draft discovery: matter profile + recommendation + category
# browsing schemas (see `app/drafting/discovery.py`, `recommendation.py`,
# and the new routes in `app/api/drafting.py`).
# ---------------------------------------------------------------------------


class MatterJurisdiction(BaseModel):
    country: str = "IN"
    state: str | None = None
    district: str | None = None


class MatterProfile(BaseModel):
    """Wire/API shape of the structured matter profile `discovery.py` builds
    conversationally and `recommendation.py` scores templates against.
    Conversation memory stores the equivalent plain dict (see
    `discovery.default_matter_profile`); this model exists for the
    `/draft/recommend` request/response boundary and for API clients that
    want to submit or inspect a matter profile directly.
    """

    domain: str | None = None
    subcategory: str | None = None
    issues: list[str] = Field(default_factory=list)
    user_role: str | None = None
    opposite_party_role: str | None = None
    desired_reliefs: list[str] = Field(default_factory=list)
    case_stage: str | None = None
    prior_actions: list[str] = Field(default_factory=list)
    forum: str | None = None
    jurisdiction: MatterJurisdiction = Field(default_factory=MatterJurisdiction)
    urgency: str | None = None
    raw_description: str = ""


class DraftRecommendationCandidate(BaseModel):
    draft_id: str
    name: str
    hindi_name: str
    score: float
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class DraftRecommendationRequest(BaseModel):
    profile: MatterProfile
    # Restricts scoring to this set of template ids (used by the discovery
    # conversation when a shortlist is already narrowed, e.g. by a partial
    # name match) -- omitted/empty means "score the whole catalogue".
    restrict_ids: list[str] = Field(default_factory=list)


class DraftRecommendationResponse(BaseModel):
    tier: Literal["high", "medium", "low", "none"]
    candidates: list[DraftRecommendationCandidate] = Field(default_factory=list)
    no_match_reason: str = ""


class DraftCategorySummary(BaseModel):
    category_id: str
    name: str
    template_count: int = 0


class DraftSubcategorySummary(BaseModel):
    subcategory_id: str
    name: str
    domain: str
    template_count: int = 0


class DraftSearchResult(BaseModel):
    draft_id: str
    name: str
    hindi_name: str
    category: str
    domain: str = ""
    score: float = 0.0


class DraftSearchResponse(BaseModel):
    query: str
    results: list[DraftSearchResult] = Field(default_factory=list)


# Schema-driven drafting API. These models are additive; the original
# conversational and template APIs remain wire-compatible.
class DocumentResolveRequest(BaseModel):
    draft_id: str | None = None
    description: str = ""
    jurisdiction_profile: str = "generic_india"
    language: str = "auto"
    script: str = "auto"


class DocumentResolveResponse(BaseModel):
    document_id: str | None = None
    document_family: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_confirmation: bool = True
    reason: str = ""


class GrammarValidationRequest(BaseModel):
    document_id: str = Field(min_length=1)
    sections: dict[str, str] = Field(default_factory=dict)
    context: dict[str, object] = Field(default_factory=dict)


class GrammarValidationResponse(BaseModel):
    status: Literal["PASS", "WARNING", "ERROR"]
    issues: list[dict[str, str]] = Field(default_factory=list)


class LanguageDetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class SemanticCheckRequest(BaseModel):
    source_text: str = Field(min_length=1, max_length=100000)
    target_text: str = Field(min_length=1, max_length=100000)
    back_translation: str = Field(default="", max_length=100000)


class TemplateAnalyzeRequest(BaseModel):
    document_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]{2,79}$")
    extracted_text: str = Field(min_length=1, max_length=250000)


class VisualQARequest(BaseModel):
    draft_id: str = Field(min_length=1)
    session_id: str | None = None
    format: Literal["pdf", "docx"] = "pdf"


class OcrRouteRequest(BaseModel):
    text_hint: str = Field(default="", max_length=20000)
    handwritten: bool = False


class MatterWorkspaceRequest(BaseModel):
    matter_id: str = Field(min_length=1)
    domain: str = "civil"
    parties: list[dict[str, str]] = Field(default_factory=list)
    fields: dict[str, str] = Field(default_factory=dict)
    events: list[dict[str, str]] = Field(default_factory=list)
    evidence: list[dict[str, str]] = Field(default_factory=list)
    requested_reliefs: list[str] = Field(default_factory=list)
    prior_documents: list[str] = Field(default_factory=list)
    jurisdiction_profile: str = "generic_india"
    case_stage: str = "pre_litigation"
    draft_body: str = ""
    bundle_documents: list[dict[str, object]] = Field(default_factory=list)


class RedlineRiskRequest(BaseModel):
    original_text: str = Field(max_length=250000)
    revised_text: str = Field(max_length=250000)
