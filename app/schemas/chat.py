from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.constants import LEGAL_DISCLAIMER
from app.schemas.common import EvidencePage, LawyerRecommendation, RetrievedChunk, SourceCitation
from app.schemas.drafting import DraftTurnInfo


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    language: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None
    user_id: str | None = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    explanation_mode: Literal["simple", "detailed", "advocate"] | None = None


class ChatResponse(BaseModel):
    message_id: str = ""
    session_id: str = ""
    answer: str
    sources: list[SourceCitation]
    # Page-level evidence for `sources`, when the cited chunks carry it.
    # Additive and defaults to `[]`, so every existing client and every stored
    # response from before Phase 2 stays valid. An empty list means "no page
    # evidence available", never "page 1".
    evidence_pages: list[EvidencePage] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: str = "Low"
    confidence_reason: str = ""
    # Phase 2: the dimensions behind `confidence`, so a reader can tell "we
    # found nothing" apart from "we found good sources but the answer does not
    # rest on them". All default to 0.0 and `confidence` above keeps its exact
    # previous meaning, so existing clients are unaffected.
    #
    # `overall_confidence` mirrors `confidence`; it is duplicated under the
    # explicit name so a client reading the breakdown does not have to know
    # that the legacy field is the same number.
    retrieval_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_grounding_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    model_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # What the governance registry records about how current the cited sources
    # are, in plain words. Empty when there is nothing to disclose. Kept as its
    # own field rather than appended to `answer` so a client can present it as
    # provenance rather than as part of the legal explanation, and so an empty
    # string is unambiguously "nothing to flag" rather than a formatting quirk.
    currency_notice: str = ""
    # True when the settled answer is the strict-RAG refusal. Machine-readable
    # companion to that fixed text, so a client can suppress the whole
    # sources/currency apparatus without string-matching the answer -- and so
    # the API states plainly that nothing supports this reply. Whenever it is
    # true, `sources`, `evidence_pages`, `applicable_law`,
    # `retrieved_sections`, `retrieved_chunks` and `currency_notice` are all
    # empty and `confidence` is 0.0 (see `app/services/safe_decline.py`).
    no_verified_context: bool = False
    # True only when `no_verified_context` would otherwise have been true but
    # the controlled General Knowledge fallback (`app.core.gk_fallback`,
    # opt-in via `settings.general_knowledge_fallback_enabled`) produced an
    # answer instead. `answer` already carries the "General Legal Knowledge"
    # label and unverified-answer disclaimer inline; this is the
    # machine-readable companion so a client can style it distinctly without
    # string-matching the label. `no_verified_context` is `False` whenever
    # this is `True` -- a GK answer is a real (if unverified) answer, not the
    # strict-RAG refusal, so the invariants documented on `no_verified_context`
    # (empty sources, zero confidence) do not apply here; `sources` stays
    # empty because none exist, but `confidence` reflects the GK answer.
    general_knowledge_used: bool = False
    lawyer_recommendation: LawyerRecommendation
    retrieved_sections: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    # Phase 2 structured answer fields. Populated only when the answer path can
    # fill them from grounded material; never invented, so an empty list means
    # "not extracted", not "none apply".
    applicable_law: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    disclaimer: str = LEGAL_DISCLAIMER
    detected_language: str
    detected_intent: str
    conversation_intent: str = "General Legal Information"
    detected_intents: list[str] = Field(default_factory=list)
    intent_reason: str = ""
    explanation_level: str = "citizen"
    suggested_actions: list[str] = Field(default_factory=list)
    related_questions: list[str] = Field(default_factory=list)
    llm_provider: str = ""
    llm_model: str = ""
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float
    draft: DraftTurnInfo | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hit: str | None = None
    # Reliability diagnostics are intentionally coarse and contain no
    # provider secrets or internal URLs. They let the UI distinguish a
    # graceful provider degradation from a transport failure.
    request_id: str = ""
    retryable: bool = False
    failure_category: str | None = None
    saved_progress: bool = False
    # Phase 1 "Multi-Intent Workflow Orchestration": populated only when
    # `detected_intents` matched one of the fixed allowlisted chains in
    # `app/services/workflow_orchestrator.py` -- empty/`None` for every
    # ordinary single-intent turn. `workflow_status` values: "detected"
    # (chain recognized, no bespoke automation for it beyond normal
    # routing), "awaiting_missing_fields", "draft_preview_ready",
    # "failed_no_document", "failed_ownership", "failed_unreadable_document",
    # "failed_no_draft_template".
    workflow_chain: list[str] = Field(default_factory=list)
    workflow_status: str | None = None
    # --- Conversational orchestration (chat-first refactor) ---------------
    # All additive and all optional, so every existing client keeps working
    # unchanged: a caller that ignores these sees exactly the response it saw
    # before. Populated only when `app/chatops` handled the turn.
    #
    # `intent` is the workflow that ran, `active_workflow` the one still in
    # the foreground afterwards (they differ when a workflow completes and
    # a parked one is waiting). `missing_field` names the single field the
    # assistant just asked about -- one at a time, by design.
    intent: str | None = None
    active_workflow: str | None = None
    assistant_message: str = ""
    missing_field: str | None = None
    # Structured progress for a multi-step workflow. `workflow_name` is the
    # workflow that produced this turn (`active_workflow` is what remains in
    # the foreground afterwards). `required_field` is `missing_field` under
    # the name the Phase 3 contract uses; both are populated so neither an
    # existing nor a new client has to know about the other.
    workflow_name: str = ""
    current_step: str = ""
    completed_steps: int = 0
    total_steps: int = 0
    progress_percentage: int = Field(default=0, ge=0, le=100)
    required_field: str | None = None
    collected_facts: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False
    allowed_actions: list[str] = Field(default_factory=list)
    # A produced file, as a SECURE reference (an API route plus ids) -- never
    # a filesystem path. See `app/chatops/artifacts.py`.
    artifact: dict[str, Any] | None = None
    citations: list[dict[str, str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    upload_required: bool = False
    # For an unavoidable third-party step (an external e-sign ceremony).
    secure_action_url: str | None = None
