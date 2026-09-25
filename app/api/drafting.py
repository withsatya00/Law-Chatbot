from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.api.deps import get_current_user_id
from app.casefile.annexures import EvidenceItem
from app.core.exceptions import BadRequestError, NotFoundError
from app.memory.store import ConversationMemoryStore
from app.drafting import discovery, recommendation
from app.drafting import (
    native_grammars as _native_grammars,  # noqa: F401 - registers native schemas
)
from app.drafting.advanced_services import (
    BilingualSemanticValidator,
    LanguageScriptDetector,
    MultilingualOcrRouter,
    TemplateAnalyzer,
    VisualQAService,
)
from app.drafting.conversation import DraftConversationEngine
from app.drafting.document_grammar import (
    DocumentGrammar,
    DocumentGrammarValidator,
    document_schema_registry,
)
from app.drafting.edit_commands import EditCommandInterpreter
from app.drafting.engine import LegalDraftEngine
from app.drafting.export import ExportOptions
from app.drafting.matter_workspace import (
    CaseFacts,
    CourtPleadingBuilder,
    EvidenceTraceabilityEngine,
    FilingBundleGenerator,
    JurisdictionRulePackRegistry,
    MatterConsistencyEngine,
    MatterDocumentChain,
    MatterParty,
    RedlineRiskAnalyzer,
)
from app.drafting.registries import (
    formatting_profile_registry,
    jurisdiction_registry,
    language_registry,
)
from app.drafting.templates import get_template, list_templates
from app.memory.store import ConversationMemoryStore
from app.schemas.drafting import (
    DocumentResolveRequest,
    DocumentResolveResponse,
    DraftAuditRequest,
    DraftAuditResponse,
    DraftCategorySummary,
    DraftEditRequest,
    DraftExportRequest,
    DraftGenerateResponse,
    DraftHistoryRequest,
    DraftHistoryResponse,
    DraftLifecycleRequest,
    DraftLifecycleResponse,
    DraftMessageRequest,
    DraftMessageResponse,
    DraftRecommendationCandidate,
    DraftRecommendationRequest,
    DraftRecommendationResponse,
    DraftRollbackRequest,
    DraftSearchResponse,
    DraftSearchResult,
    DraftSubcategorySummary,
    DraftTemplateDetail,
    DraftTemplateSummary,
    DraftTranslateRequest,
    DraftTranslateResponse,
    DraftVersionsResponse,
    DraftVersionSummary,
    GrammarValidationRequest,
    GrammarValidationResponse,
    LanguageDetectRequest,
    LegalTermsRequest,
    LegalTermsResponse,
    MatterWorkspaceRequest,
    OcrRouteRequest,
    RedlineRiskRequest,
    SemanticCheckRequest,
    TemplateAnalyzeRequest,
    VisualQARequest,
)
from app.schemas.phase2 import (
    ConflictResolutionRequest,
    DraftCompareResponse,
    DraftDuplicateRequest,
    DraftDuplicateResponse,
    DraftReviewResponse,
)
from app.services.draft_management import (
    build_review,
    compare_versions,
    ensure_draft_access,
)
from app.services.draft_management import (
    delete_draft as delete_draft_record,
)
from app.services.draft_management import (
    duplicate_draft as duplicate_draft_record,
)
from app.services.draft_management import (
    resolve_conflict as resolve_conflict_record,
)
from app.services.phase3 import AuditService

router = APIRouter(tags=["drafting"])

_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "rtf": "application/rtf",
}


# The canonical rule now lives in `app/services/draft_management.py`, so the
# chat workflows enforce exactly the same ownership check as these routes.
# Kept under the original private name because every route below reads better
# with it and existing tests exercise it here.
_ensure_draft_access = ensure_draft_access


def _grammar_payload(grammar: DocumentGrammar) -> dict[str, object]:
    return {
        "document_id": grammar.document_id,
        "document_family": grammar.document_family,
        "document_type": grammar.document_type,
        "document_variant": grammar.document_variant,
        "proceeding_type": grammar.proceeding_type,
        "proceeding_stage": grammar.proceeding_stage,
        "forum_type": grammar.forum_type,
        "jurisdiction_profile": grammar.jurisdiction_profile,
        "formatting_profile": grammar.formatting_profile,
        "governing_law": list(grammar.governing_law),
        "legal_provisions": list(grammar.legal_provisions),
        "version": grammar.version,
        "legacy_adapter": grammar.legacy_adapter,
        "structure": [
            {
                "id": section.id,
                "source_heading": section.source_heading,
                "component": section.component.value,
                "heading_mode": section.heading_mode.value,
                "required": section.required,
                "repeatable": section.repeatable,
                "children": list(section.children),
                "pagination": {
                    "keep_together": section.pagination.keep_together,
                    "keep_with_next": section.pagination.keep_with_next,
                    "widow_control": section.pagination.widow_control,
                    "orphan_control": section.pagination.orphan_control,
                    "allow_split": section.pagination.allow_split,
                    "page_break_before": section.pagination.page_break_before,
                },
            }
            for section in grammar.structure
        ],
    }


@router.get("/languages")
async def list_languages() -> list[dict[str, object]]:
    return language_registry.serialized()


@router.get("/languages/{language_code}")
async def get_language(language_code: str) -> dict[str, object]:
    language = language_registry.get(language_code.lower())
    if language is None:
        raise NotFoundError(f"Unknown language: {language_code}")
    from dataclasses import asdict

    return asdict(language)


@router.get("/formatting-profiles")
async def list_formatting_profiles() -> list[dict[str, object]]:
    return formatting_profile_registry.serialized()


@router.get("/jurisdictions")
async def list_jurisdictions() -> list[dict[str, object]]:
    return jurisdiction_registry.serialized()


@router.get("/document-grammars")
async def list_document_grammars() -> list[dict[str, object]]:
    # Materialize adapters lazily so every existing YAML template appears.
    for template in list_templates():
        document_schema_registry.for_template(template)
    return [_grammar_payload(grammar) for grammar in document_schema_registry.list()]


@router.get("/document-grammars/{document_id}")
async def get_document_grammar(document_id: str) -> dict[str, object]:
    template = get_template(document_id)
    grammar = document_schema_registry.get(document_id)
    if grammar is None and template is not None:
        grammar = document_schema_registry.for_template(template)
    if grammar is None:
        raise NotFoundError(f"Unknown document grammar: {document_id}")
    return _grammar_payload(grammar)


@router.post("/draft/resolve", response_model=DocumentResolveResponse)
async def resolve_document(request: DocumentResolveRequest) -> DocumentResolveResponse:
    if request.draft_id:
        template = get_template(request.draft_id)
        if template is None:
            return DocumentResolveResponse(reason="No template or grammar matched the supplied draft_id.")
        grammar = document_schema_registry.for_template(template)
        return DocumentResolveResponse(
            document_id=grammar.document_id, document_family=grammar.document_family,
            confidence=1.0, requires_confirmation=False, reason="Exact draft_id match.",
        )
    matches = recommendation.search_templates(request.description)
    if not matches:
        return DocumentResolveResponse(reason="No safe deterministic match; user confirmation is required.")
    match = matches[0]
    template = get_template(match.draft_id)
    if template is None:
        return DocumentResolveResponse(reason="Matched template is unavailable.")
    grammar = document_schema_registry.for_template(template)
    confidence = max(0.0, min(1.0, float(match.score)))
    return DocumentResolveResponse(
        document_id=grammar.document_id, document_family=grammar.document_family,
        confidence=confidence, requires_confirmation=confidence < 0.85,
        reason="Deterministic template-search match; confirm when confidence is below threshold.",
    )


@router.post("/draft/validate", response_model=GrammarValidationResponse)
async def validate_document(request: GrammarValidationRequest) -> GrammarValidationResponse:
    template = get_template(request.document_id)
    grammar = document_schema_registry.get(request.document_id)
    if grammar is None and template is not None:
        grammar = document_schema_registry.for_template(template)
    if grammar is None:
        raise NotFoundError(f"Unknown document grammar: {request.document_id}")
    issues = DocumentGrammarValidator().validate(grammar, request.sections, request.context)
    status: Literal["PASS", "WARNING", "ERROR"] = "ERROR" if any(
        issue.severity == "error" for issue in issues
    ) else ("WARNING" if issues else "PASS")
    return GrammarValidationResponse(status=status, issues=[issue.as_dict() for issue in issues])


@router.post("/language/detect")
async def detect_language(request: LanguageDetectRequest) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(LanguageScriptDetector().detect(request.text))


@router.post("/ocr/multilingual/route")
async def route_multilingual_ocr(request: OcrRouteRequest) -> dict[str, object]:
    """Returns the safe provider/script route; extraction stays in existing OCR APIs."""
    return MultilingualOcrRouter().route(request.text_hint, handwritten=request.handwritten)


@router.post("/draft/back-translation-check")
async def back_translation_check(request: SemanticCheckRequest) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(BilingualSemanticValidator().compare(
        request.source_text, request.target_text, request.back_translation
    ))


@router.post("/draft/analyze-template")
async def analyze_template(request: TemplateAnalyzeRequest) -> dict[str, object]:
    """Produces an inactive, review-required grammar proposal from untrusted text."""
    analysis = TemplateAnalyzer().analyze(request.extracted_text, request.document_id)
    return {
        "status": analysis.status,
        "detected_headings": list(analysis.detected_headings),
        "letterhead_lines": list(analysis.letterhead_lines),
        "warnings": list(analysis.warnings),
        "proposed_grammar": _grammar_payload(analysis.proposed_grammar),
    }


@router.post("/draft/visual-qa")
async def visual_qa(
    request: VisualQARequest, user_id: str | None = Depends(get_current_user_id)
) -> dict[str, object]:
    from dataclasses import asdict

    engine = LegalDraftEngine()
    draft = await engine.drafts.find_by_id(request.draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {request.draft_id}")
    _ensure_draft_access(draft, user_id, request.session_id)
    path = await engine.export(request.draft_id, request.format, ExportOptions())
    return asdict(VisualQAService().inspect(path))


@router.post("/matter/workspace/analyze")
async def analyze_matter_workspace(request: MatterWorkspaceRequest) -> dict[str, object]:
    """One deterministic review payload spanning the eight matter-workspace layers."""
    from dataclasses import asdict

    matter = CaseFacts(
        matter_id=request.matter_id,
        parties=[MatterParty(**party) for party in request.parties],
        fields=request.fields,
        events=request.events,
        evidence=[EvidenceItem(**item) for item in request.evidence],
        requested_reliefs=request.requested_reliefs,
        prior_documents=request.prior_documents,
        jurisdiction_profile=request.jurisdiction_profile,
        case_stage=request.case_stage,
    )
    rows, traces = EvidenceTraceabilityEngine().build(matter)
    conflicts = MatterConsistencyEngine().check(matter)
    jurisdiction_missing = JurisdictionRulePackRegistry().validate(
        matter.jurisdiction_profile, matter.fields
    )
    pleading = CourtPleadingBuilder().build(matter)
    annexure_issues = EvidenceTraceabilityEngine.validate_body_references(request.draft_body, rows)
    bundle = FilingBundleGenerator().manifest(
        matter, request.bundle_documents, conflicts=conflicts,
        validation_errors=[f"Missing jurisdiction field: {key}" for key in jurisdiction_missing],
    )
    return {
        "matter_id": matter.matter_id,
        "document_chain": [asdict(step) for step in MatterDocumentChain().plan(matter, request.domain)],
        "pleading_workspace": asdict(pleading),
        "annexures": [row.as_dict() for row in rows],
        "evidence_traceability": [asdict(trace) for trace in traces],
        "annexure_issues": annexure_issues,
        "conflicts": conflicts,
        "jurisdiction_missing_fields": jurisdiction_missing,
        "filing_bundle": asdict(bundle),
        "final_export_blocked": bool(conflicts or jurisdiction_missing or any(
            issue["severity"] == "error" for issue in annexure_issues
        )),
    }


@router.post("/draft/redline/risk")
async def redline_risk(request: RedlineRiskRequest) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(RedlineRiskAnalyzer().compare(request.original_text, request.revised_text))


@router.get("/draft-templates", response_model=list[DraftTemplateSummary])
async def list_draft_templates(
    domain: str | None = None, subcategory: str | None = None
) -> list[DraftTemplateSummary]:
    """Unchanged for existing callers that pass no query params (still every
    template, as before). `domain`/`subcategory` are additive filters for
    the new category-browsing UI -- see `GET /draft-subcategories` and
    `CATEGORY_DOMAIN_MAP` in `app/drafting/discovery.py`.
    """
    templates = LegalDraftEngine().list_templates()
    if domain:
        templates = [t for t in templates if t.domain == domain]
    if subcategory:
        templates = [t for t in templates if t.subcategory == subcategory]
    return templates


@router.get("/draft-templates/search", response_model=DraftSearchResponse)
async def search_draft_templates(q: str = "") -> DraftSearchResponse:
    """Ranked, limited (at most 5 by default) free-text template search --
    backs the frontend's "Search a draft" entry action. Declared BEFORE the
    `/draft-templates/{draft_id}` route below so FastAPI doesn't try to
    resolve "search" as a `draft_id` path parameter.
    """
    results = recommendation.search_templates(q)
    return DraftSearchResponse(
        query=q,
        results=[
            DraftSearchResult(
                draft_id=r.draft_id, name=r.name, hindi_name=r.hindi_name,
                category=r.category, domain=r.domain, score=r.score,
            )
            for r in results
        ],
    )


@router.get("/draft-categories", response_model=list[DraftCategorySummary])
async def list_draft_categories() -> list[DraftCategorySummary]:
    """The 6 top-level browse categories (secondary, manual-browse route --
    the primary flow is problem-first discovery via `POST /draft/recommend`
    and the conversational `/draft` endpoint). `template_count` is 0 for a
    category whose templates haven't been tagged with a `domain` yet; see
    `CATEGORY_DOMAIN_MAP`'s docstring for how templates are onboarded.
    """
    templates = list_templates()
    counts: list[DraftCategorySummary] = []
    for category_id, name in discovery.TOP_LEVEL_CATEGORIES:
        domains = discovery.CATEGORY_DOMAIN_MAP.get(category_id, ())
        count = sum(1 for t in templates if t.domain in domains)
        counts.append(DraftCategorySummary(category_id=category_id, name=name, template_count=count))
    return counts


@router.get("/draft-subcategories", response_model=list[DraftSubcategorySummary])
async def list_draft_subcategories(domain: str = "") -> list[DraftSubcategorySummary]:
    """Distinct subcategories under one top-level category, identified by
    its `domain` (e.g. "property"), each with how many templates it holds.
    """
    templates = list_templates()
    seen: dict[str, int] = {}
    for template in templates:
        if domain and template.domain != domain:
            continue
        if not template.subcategory:
            continue
        seen[template.subcategory] = seen.get(template.subcategory, 0) + 1
    return [
        DraftSubcategorySummary(
            subcategory_id=subcategory, name=subcategory.replace("_", " ").title(),
            domain=domain, template_count=count,
        )
        for subcategory, count in sorted(seen.items())
    ]


@router.post("/draft/recommend", response_model=DraftRecommendationResponse)
async def recommend_draft(request: DraftRecommendationRequest) -> DraftRecommendationResponse:
    """Deterministic, LLM-free draft recommendation from a structured matter
    profile -- the same engine (`app/drafting/recommendation.py`) the
    conversational discovery flow (`POST /draft` / `POST /chat`) calls
    internally, exposed here for programmatic/testing access, exactly like
    `POST /draft` is documented above as exposing the conversation engine.
    """
    restrict_ids = frozenset(request.restrict_ids) if request.restrict_ids else None
    result = recommendation.recommend(request.profile.model_dump(), restrict_ids=restrict_ids)
    return DraftRecommendationResponse(
        tier=result.tier,
        candidates=[
            DraftRecommendationCandidate(
                draft_id=c.draft_id, name=c.name, hindi_name=c.hindi_name,
                score=c.score, confidence=c.confidence, reason=c.reason,
            )
            for c in result.candidates
        ],
        no_match_reason=result.no_match_reason,
    )


@router.get("/draft-templates/{draft_id}", response_model=DraftTemplateDetail)
async def get_draft_template(draft_id: str) -> DraftTemplateDetail:
    return LegalDraftEngine().get_template_detail(draft_id)


@router.post("/draft", response_model=DraftMessageResponse)
async def draft_message(
    request: DraftMessageRequest, user_id: str | None = Depends(get_current_user_id)
) -> DraftMessageResponse:
    """Sends a message into the conversational drafting state machine directly.

    This is the same engine `/chat` delegates to internally when it detects
    drafting intent -- exposed here for programmatic/testing access. The
    Streamlit frontend only calls `/chat`; it never calls this directly.
    """
    memory_store = ConversationMemoryStore()
    memory = await memory_store.check_access(request.session_id, user_id)
    if user_id and not memory.get("owner_user_id"):
        memory = await memory_store.update(request.session_id, owner_user_id=user_id)
    language = request.language or memory.get("language_preference") or "english"
    try:
        result = await DraftConversationEngine().handle_turn(request.session_id, request.message, language, memory)
    except Exception:
        await AuditService().event("draft_failed", language=language, details={"session_id": request.session_id})
        raise
    if result is None:
        return DraftMessageResponse(reply="This doesn't look like a drafting request.", draft=None)
    await memory_store.update(request.session_id, **memory)
    return DraftMessageResponse(reply=result.reply_text, draft=result.info)


@router.post("/draft/edit", response_model=DraftGenerateResponse)
async def edit_draft(
    request: DraftEditRequest, user_id: str | None = Depends(get_current_user_id)
) -> DraftGenerateResponse:
    engine = LegalDraftEngine()
    draft = await engine.drafts.find_by_id(request.draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {request.draft_id}")
    _ensure_draft_access(draft, user_id, request.session_id)
    template = get_template(draft["draft_type"])
    if template is None:
        raise NotFoundError(f"Unknown draft template: {draft['draft_type']}")

    target_field = request.target_field
    new_value = request.new_value
    if not target_field and request.instruction:
        command = EditCommandInterpreter().interpret(request.instruction, template)
        if command.action == "translate":
            # Part 52: regenerates and persists the draft in the new language
            # (see `LegalDraftEngine.regenerate`) instead of the old
            # `translate()` call, which returned a translated blob without
            # ever writing it back to the draft record -- a subsequent
            # `/draft/export` for this same `draft_id` would silently keep
            # serving the pre-translation content.
            target_language = command.target_language or request.target_language or "hindi"
            return await engine.regenerate(request.draft_id, {}, language=target_language)
        if command.action not in ("replace_field", "remove_paragraph") or not command.target_field:
            raise BadRequestError("Could not determine which field to edit from the instruction provided.")
        target_field = command.target_field
        new_value = command.new_value if command.action == "replace_field" else ""

    if not target_field:
        raise BadRequestError("Provide either an instruction or target_field/new_value to edit the draft.")

    return await engine.regenerate(request.draft_id, {target_field: new_value or ""})


@router.post("/draft/export")
async def export_draft(
    request: DraftExportRequest, user_id: str | None = Depends(get_current_user_id)
) -> FileResponse:
    engine = LegalDraftEngine()
    draft = await engine.drafts.find_by_id(request.draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {request.draft_id}")
    _ensure_draft_access(draft, user_id, request.session_id)
    unresolved = [
        conflict for conflict in draft.get("conflicts", [])
        if conflict.get("slot") not in draft.get("resolved_conflicts", {})
    ]
    if unresolved:
        raise BadRequestError(
            "Resolve all conflicting facts in the draft review workspace before final export.",
            {"conflicts": unresolved},
        )
    if draft.get("voice_collected") and not draft.get("voice_final_confirmed"):
        raise BadRequestError(
            "Review and explicitly confirm voice-collected facts before final export.",
            {"confirmation_required": True, "draft_id": request.draft_id},
        )
    try:
        path = await engine.export(
            request.draft_id,
            request.format,
            ExportOptions(
                watermark=request.watermark,
                sign=request.sign,
                watermark_text=request.watermark_text or ExportOptions().watermark_text,
                include_header_footer=request.include_header_footer,
            ),
        )
    except Exception:
        await AuditService().event(
            "export_failed", language=draft.get("language", ""),
            details={"draft_id": request.draft_id, "format": request.format},
        )
        await AuditService().record(
            actor_user_id=user_id, action="draft_exported", resource_type="legal_draft",
            resource_id=request.draft_id, outcome="failure", details={"format": request.format},
        )
        raise
    # A completed export is the actual security/compliance-relevant event --
    # someone just downloaded a document dense with another person's PII
    # (name, address, phone, bank/transaction details) -- but until now only
    # the FAILURE path above was ever recorded anywhere; a successful export
    # left no trace of who exported what, when.
    await AuditService().record(
        actor_user_id=user_id, action="draft_exported", resource_type="legal_draft",
        resource_id=request.draft_id, outcome="success", details={"format": request.format},
    )
    return FileResponse(path, filename=f"{request.draft_id}.{request.format}", media_type=_MEDIA_TYPES[request.format])


@router.delete("/draft/{draft_id}")
async def delete_draft(
    draft_id: str, session_id: str | None = None, user_id: str | None = Depends(get_current_user_id)
) -> dict[str, object]:
    """Phase 1 item 7 ("delete-draft"): erase one draft and its version chain.

    A draft is the single most PII-dense record this product creates -- name,
    full postal address, mobile number, bank, transaction reference and a
    free-text account of what happened, all in one document. Users could
    create them and never delete them: there was no delete route at all, and
    `DELETE /session` (added for chat) does not touch `legal_drafts`.

    Ownership is enforced with the same `_ensure_draft_access` rule every
    other mutating draft route uses, so one user cannot delete another's
    draft. The version history goes with it -- leaving `draft_versions` behind
    would retain every earlier revision of exactly the text just deleted.
    """
    return await delete_draft_record(draft_id, user_id, session_id)


@router.post("/draft/audit", response_model=DraftAuditResponse)
async def audit_draft_endpoint(
    request: DraftAuditRequest, user_id: str | None = Depends(get_current_user_id)
) -> DraftAuditResponse:
    """Phase 1 item 6: run the fact-only audit against a stored draft.

    Same check `export` runs and the chat preview already surfaces, exposed
    separately so a client can offer "review before you download" without
    generating the file first.
    """
    engine = LegalDraftEngine()
    draft = await engine.drafts.find_by_id(request.draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {request.draft_id}")
    _ensure_draft_access(draft, user_id, request.session_id)
    template = engine._require_template(draft["draft_type"])
    findings = engine._audit_findings(template, draft["sections"], draft.get("fields", {}))
    return DraftAuditResponse(draft_id=request.draft_id, ok=not findings, findings=findings)


@router.post("/draft/approve", response_model=DraftLifecycleResponse)
async def approve_draft(
    request: DraftLifecycleRequest, user_id: str | None = Depends(get_current_user_id)
) -> DraftLifecycleResponse:
    engine = LegalDraftEngine()
    draft = await engine.drafts.find_by_id(request.draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {request.draft_id}")
    _ensure_draft_access(draft, user_id, request.session_id)
    state = await engine.approve(request.draft_id)
    return DraftLifecycleResponse(draft_id=request.draft_id, lifecycle_state=state)


@router.post("/draft/lock", response_model=DraftLifecycleResponse)
async def lock_draft(
    request: DraftLifecycleRequest, user_id: str | None = Depends(get_current_user_id)
) -> DraftLifecycleResponse:
    engine = LegalDraftEngine()
    draft = await engine.drafts.find_by_id(request.draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {request.draft_id}")
    _ensure_draft_access(draft, user_id, request.session_id)
    state = await engine.lock(request.draft_id)
    return DraftLifecycleResponse(draft_id=request.draft_id, lifecycle_state=state)


@router.post("/draft/unlock", response_model=DraftLifecycleResponse)
async def unlock_draft(
    request: DraftLifecycleRequest, user_id: str | None = Depends(get_current_user_id)
) -> DraftLifecycleResponse:
    engine = LegalDraftEngine()
    draft = await engine.drafts.find_by_id(request.draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {request.draft_id}")
    _ensure_draft_access(draft, user_id, request.session_id)
    state = await engine.unlock(request.draft_id)
    return DraftLifecycleResponse(draft_id=request.draft_id, lifecycle_state=state)


@router.post("/draft/rollback", response_model=DraftGenerateResponse)
async def rollback_draft(
    request: DraftRollbackRequest, user_id: str | None = Depends(get_current_user_id)
) -> DraftGenerateResponse:
    engine = LegalDraftEngine()
    draft = await engine.drafts.find_by_id(request.draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {request.draft_id}")
    _ensure_draft_access(draft, user_id, request.session_id)
    return await engine.rollback(request.draft_id, request.version_number)


@router.get("/draft/{draft_id}/versions", response_model=DraftVersionsResponse)
async def list_draft_versions(
    draft_id: str, session_id: str | None = None, user_id: str | None = Depends(get_current_user_id)
) -> DraftVersionsResponse:
    engine = LegalDraftEngine()
    draft = await engine.drafts.find_by_id(draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {draft_id}")
    _ensure_draft_access(draft, user_id, session_id)
    versions = await engine.list_versions(draft_id)
    return DraftVersionsResponse(
        draft_id=draft_id,
        versions=[
            DraftVersionSummary(
                version_number=version["version_number"],
                document_status=version.get("document_status", "active"),
                language=version.get("language", "english"),
                created_at=version["created_at"].isoformat() if version.get("created_at") else None,
            )
            for version in versions
        ],
    )


@router.post("/draft/translate", response_model=DraftTranslateResponse)
async def translate_draft(
    request: DraftTranslateRequest, user_id: str | None = Depends(get_current_user_id)
) -> DraftTranslateResponse:
    engine = LegalDraftEngine()
    if request.draft_id:
        draft = await engine.drafts.find_by_id(request.draft_id)
        if draft is None:
            raise NotFoundError(f"Draft not found: {request.draft_id}")
        _ensure_draft_access(draft, user_id, request.session_id)
    return await engine.translate(request)


@router.post("/legal-terms", response_model=LegalTermsResponse)
async def legal_terms(request: LegalTermsRequest) -> LegalTermsResponse:
    terms = await LegalDraftEngine().legal_terms(request.query)
    return LegalTermsResponse(terms=terms)


@router.post("/draft-history", response_model=DraftHistoryResponse)
async def draft_history(
    request: DraftHistoryRequest, user_id: str | None = Depends(get_current_user_id)
) -> DraftHistoryResponse:
    # The client-supplied `user_id` is never trusted as-is: an authenticated
    # caller's own JWT identity always wins (so they can't query someone
    # else's history by guessing their user id), and an anonymous caller's
    # claimed `user_id` is dropped entirely, falling back to session-scoped
    # history only.
    # BUG-07 (QA session 2026-09-24): the session-scoped branch below used to
    # go straight to `list_for_session` with no ownership check at all --
    # unlike every other session_id-scoped route (`/history`, `/session`,
    # `/session/facts`, ...), which all gate through this same
    # `check_access`. Anyone who knew or guessed another user's `session_id`
    # could read that user's full draft history. `check_access` is a no-op
    # for a session nobody has claimed yet (todays default), and only
    # rejects (403) when the caller's identity doesn't match a session that
    # IS claimed -- same behaviour `/history` already has.
    if request.session_id:
        await ConversationMemoryStore().check_access(request.session_id, user_id)
    effective_request = request.model_copy(update={"user_id": user_id or None})
    drafts = await LegalDraftEngine().history(effective_request)
    return DraftHistoryResponse(drafts=drafts)


@router.get("/draft/{draft_id}/review", response_model=DraftReviewResponse)
async def review_draft(
    draft_id: str,
    session_id: str | None = None,
    user_id: str | None = Depends(get_current_user_id),
) -> DraftReviewResponse:
    """Structured, read-only review workspace for a stored draft."""
    return await build_review(draft_id, user_id, session_id)


@router.get("/draft/{draft_id}/compare", response_model=DraftCompareResponse)
async def compare_draft_versions(
    draft_id: str,
    original_version: int = Query(1, ge=1),
    revised_version: int = Query(..., ge=1),
    session_id: str | None = None,
    user_id: str | None = Depends(get_current_user_id),
) -> DraftCompareResponse:
    return await compare_versions(draft_id, original_version, revised_version, user_id, session_id)


@router.post("/draft/{draft_id}/duplicate", response_model=DraftDuplicateResponse)
async def duplicate_draft(
    draft_id: str,
    request: DraftDuplicateRequest,
    user_id: str | None = Depends(get_current_user_id),
) -> DraftDuplicateResponse:
    return await duplicate_draft_record(draft_id, user_id, request.session_id)


@router.post("/draft/{draft_id}/conflicts/resolve", response_model=DraftReviewResponse)
async def resolve_draft_conflict(
    draft_id: str,
    request: ConflictResolutionRequest,
    session_id: str | None = None,
    user_id: str | None = Depends(get_current_user_id),
) -> DraftReviewResponse:
    return await resolve_conflict_record(draft_id, request.slot, request.chosen_value, user_id, session_id)
