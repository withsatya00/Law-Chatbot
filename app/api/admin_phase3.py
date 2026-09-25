from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import Response

from app.api.deps import get_current_user_claims, require_admin
from app.database.mongodb import mongodb
from app.models.collections import AUDIT_LOGS, OPERATIONAL_EVENTS
from app.observability.metrics import metrics
from app.schemas.law_monitoring import (
    CatalogueSourceRequest,
    MonitorRequest,
    MonitorReviewRequest,
    RelationshipReviewRequest,
)
from app.schemas.phase3 import (
    EvaluationRunRequest,
    EvaluationRunResponse,
    LegalSourceMetadata,
    LegalSourceResponse,
    LinkSourceDocumentRequest,
    SourceReviewRequest,
    SupersedeSourceRequest,
    VerifySourceRequest,
)
from app.services.analytics_service import AnalyticsService
from app.services.kb_automation import KnowledgeBaseAutomationService, normalize_batch_codes
from app.services.kb_coverage import KnowledgeBaseCoverageService
from app.services.kb_indexing_queue import kb_indexing_queue
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService
from app.services.kb_official_source_sync import OfficialSourceSyncService
from app.services.kb_operations import KBOperationsService
from app.services.kb_production_release import KBProductionReleaseService
from app.services.kb_source_onboarding import SourceOnboardingService
from app.services.kb_source_registry import rollout_readiness
from app.services.law_monitoring import LawMonitoringService
from app.services.phase3 import AuditService, EvaluationService, LegalUpdateService

router = APIRouter(prefix="/admin/phase3", tags=["admin-phase-3"], dependencies=[Depends(require_admin)])


def _json_safe_document(item: dict[str, Any]) -> dict[str, Any]:
    return {**item, "_id": str(item.get("_id", ""))}


def _admin_id(claims: dict[str, Any] = Depends(get_current_user_claims)) -> str:
    return str(claims["sub"])


@router.put("/law-monitors")
async def register_law_monitor(request: MonitorRequest, admin_user_id: str = Depends(_admin_id)) -> dict[str, Any]:
    return await LawMonitoringService().register(request, admin_user_id)


@router.get("/law-monitors/coverage")
async def law_monitor_coverage() -> dict[str, Any]:
    return await LawMonitoringService().coverage()


@router.get("/official-sources/coverage")
async def official_source_coverage() -> dict[str, Any]:
    return await OfficialSourceSyncService().coverage()


@router.get("/knowledge-base/coverage")
async def knowledge_base_coverage() -> dict[str, Any]:
    """Return honest central/state coverage stages for the searchable KB."""
    report = await KnowledgeBaseCoverageService().coverage()
    report["automation"] = await KnowledgeBaseAutomationService().coverage()
    return report


@router.post("/kb-automation/run")
async def run_kb_automation(
    limit: int = Query(20, ge=1, le=100), jurisdictions: str | None = Query(default=None),
) -> dict[str, Any]:
    """Discover and process one bounded batch immediately."""
    from app.core.exceptions import BadRequestError
    try:
        scope = normalize_batch_codes(jurisdictions)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc
    return await KnowledgeBaseAutomationService().run_cycle(limit, scope)


@router.get("/kb-automation/status")
async def kb_automation_status() -> dict[str, Any]:
    return await KnowledgeBaseAutomationService().coverage()


@router.get("/kb-automation/jobs")
async def kb_automation_jobs(
    status: str | None = None, limit: int = Query(100, ge=1, le=500),
) -> list[dict[str, Any]]:
    return await KnowledgeBaseAutomationService().list_jobs(status, limit)


@router.post("/kb-automation/jobs/{job_id}/retry")
async def retry_kb_automation_job(job_id: str) -> dict[str, Any]:
    return {"job_id": job_id, "requeued": await KnowledgeBaseAutomationService().retry(job_id)}


@router.post("/kb-automation/jobs/{job_id}/relationship-review")
async def review_kb_relationship(
    job_id: str, request: RelationshipReviewRequest,
    admin_user_id: str = Depends(_admin_id),
) -> dict[str, Any]:
    from app.core.exceptions import BadRequestError
    try:
        return await KnowledgeBaseAutomationService().record_relationship_review(
            job_id, request.related_job_ids, request.notes, admin_user_id,
        )
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc


@router.get("/state-coverage")
async def state_coverage() -> dict[str, Any]:
    """One composed view for the State/UT coverage dashboard: the per-
    jurisdiction discover/download/index/test matrix, adapter/dead-letter
    health, the open manual-access exceptions an admin needs to fetch by hand,
    and recent portal-layout-change alerts."""
    automation = KnowledgeBaseAutomationService()
    coverage = await KnowledgeBaseCoverageService().coverage()
    manual_access_jobs = await automation.list_jobs(status="manual_access_required", limit=200)
    layout_change_events = [
        _json_safe_document(item) async for item in
        mongodb.db[OPERATIONAL_EVENTS].find({"event_type": "kb_adapter_layout_changed"})
        .sort("created_at", -1).limit(50)
    ]
    return {
        "coverage": coverage,
        "automation": await automation.coverage(),
        "manual_access_required": [_json_safe_document(item) for item in manual_access_jobs],
        "layout_change_alerts": layout_change_events,
    }


@router.get("/kb-rollout/readiness")
async def kb_rollout_readiness() -> dict[str, Any]:
    """Fail-closed Phase-4 readiness gates for every State and UT."""
    automation = await KnowledgeBaseAutomationService().coverage()
    coverage = await KnowledgeBaseCoverageService().coverage()
    onboarded = await SourceOnboardingService().list_sources()
    return rollout_readiness(coverage, automation, onboarded)


@router.put("/kb-sources")
async def register_kb_source(
    request: CatalogueSourceRequest, admin_user_id: str = Depends(_admin_id),
) -> dict[str, Any]:
    """Register a confirmed official catalogue in a disabled state."""
    from app.core.exceptions import BadRequestError
    try:
        return await SourceOnboardingService().register(request, admin_user_id)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc


@router.post("/kb-sources/{name}/probe")
async def probe_kb_source(name: str, admin_user_id: str = Depends(_admin_id)) -> dict[str, Any]:
    """Fetch and parse one source; this never activates or publishes it."""
    from app.core.exceptions import BadRequestError
    try:
        return await SourceOnboardingService().probe(name, admin_user_id)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc


@router.post("/kb-sources/{name}/activate")
async def activate_kb_source(name: str, admin_user_id: str = Depends(_admin_id)) -> dict[str, Any]:
    """Activate only the exact configuration that passed its latest probe."""
    from app.core.exceptions import BadRequestError
    try:
        return await SourceOnboardingService().activate(name, admin_user_id)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc


@router.post("/kb-sources/{name}/pause")
async def pause_kb_source(name: str, admin_user_id: str = Depends(_admin_id)) -> dict[str, Any]:
    """Stop future discovery from one source without deleting its records."""
    from app.core.exceptions import BadRequestError
    try:
        return await SourceOnboardingService().pause(name, admin_user_id)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc


@router.get("/kb-sources")
async def list_kb_sources() -> list[dict[str, Any]]:
    return await SourceOnboardingService().list_sources()


@router.get("/kb-production/readiness")
async def kb_production_readiness() -> dict[str, Any]:
    """Phase-6 go/no-go report; this endpoint never deploys or publishes."""
    return await KBProductionReleaseService().readiness()


@router.get("/kb-operations/status")
async def kb_operations_status() -> dict[str, Any]:
    return await KBOperationsService().status()


@router.post("/kb-operations/audit")
async def audit_kb_operations() -> dict[str, Any]:
    """Create deduplicated operational alerts for current KB backlogs."""
    return await KBOperationsService().audit_and_alert()


@router.get("/kb-benchmarks/jurisdictions")
async def jurisdiction_benchmark_status() -> dict[str, Any]:
    """Per-State/UT retrieval benchmark results from actual automated jobs."""
    return {"jurisdictions": await KnowledgeBaseAutomationService().per_jurisdiction_summary()}


@router.post("/law-monitors/{monitor_id}/check")
async def check_law_monitor(monitor_id: str) -> dict[str, Any]:
    return await LawMonitoringService().check(monitor_id)


@router.get("/law-updates")
async def list_law_updates(status: str = "pending_review", limit: int = Query(100, ge=1, le=500)) -> list[dict[str, Any]]:
    return await LawMonitoringService().list_changes(status, limit)


@router.get("/law-updates/{change_id}/snapshot")
async def law_update_snapshot(change_id: str) -> Response:
    from app.core.exceptions import NotFoundError
    service = LawMonitoringService()
    change = await service.changes.find_one({"_id": change_id})
    if change is None:
        raise NotFoundError("Monitored change not found.")
    return Response(
        change["snapshot"], media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="official-source-snapshot.bin"',
                 "X-Content-Type-Options": "nosniff"},
    )


@router.post("/law-updates/{change_id}/review")
async def review_law_update(change_id: str, request: MonitorReviewRequest, admin_user_id: str = Depends(_admin_id)) -> dict[str, Any]:
    from app.core.exceptions import BadRequestError
    try:
        return await LawMonitoringService().review(change_id, request, admin_user_id)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc


@router.get("/dashboard")
async def advanced_dashboard(days: int = Query(30, ge=1, le=365)) -> dict[str, Any]:
    dashboard = await AnalyticsService().dashboard(days)
    since = datetime.now(UTC) - timedelta(days=days)
    failed_events = [
        event async for event in mongodb.db[OPERATIONAL_EVENTS]
        .find({"created_at": {"$gte": since}, "event_type": {"$in": ["draft_failed", "export_failed"]}})
        .sort("created_at", -1).limit(50)
    ]
    dashboard["failed_draft_export_events"] = [_json_safe_document(item) for item in failed_events]
    service = LegalUpdateService()
    all_sources = await service.list()
    dashboard["stale_sources"] = [
        item.model_dump(mode="json") for item in all_sources if item.stale
    ]
    # Split out because these are different follow-up actions. A STALE source
    # just needs re-checking; one recorded as repealed or superseded is
    # actively wrong to present as current law and needs its replacement
    # linked. An UNVERIFIED source has never been checked at all.
    dashboard["outdated_sources"] = [
        item.model_dump(mode="json")
        for item in all_sources
        if item.status in {"repealed", "superseded"}
    ]
    dashboard["unverified_sources"] = [
        item.model_dump(mode="json")
        for item in all_sources
        if item.verification_status in {"unverified", "pending_review"}
    ]
    dashboard["source_governance_summary"] = {
        "total": len(all_sources),
        "verified": sum(1 for item in all_sources if item.verification_status == "verified"),
        "unverified": sum(1 for item in all_sources if item.verification_status == "unverified"),
        "pending_review": sum(1 for item in all_sources if item.verification_status == "pending_review"),
        "rejected": sum(1 for item in all_sources if item.verification_status == "rejected"),
        "stale": sum(1 for item in all_sources if item.stale),
        "repealed_or_superseded": sum(
            1 for item in all_sources if item.status in {"repealed", "superseded"}
        ),
    }
    return dashboard


@router.get("/metrics")
async def application_metrics() -> dict[str, Any]:
    return metrics.snapshot()


@router.get("/metrics/prometheus")
async def application_metrics_prometheus() -> Response:
    """Same underlying counters as `GET /metrics`, in Prometheus text
    exposition format -- point a Prometheus server (or compatible agent) at
    this on an interval so metric history survives a process restart in
    ITS storage, not this in-process one. Still behind `require_admin` like
    every other route on this router; configure the scraper with a bearer
    token, or put it behind network-level trust instead if the deployment's
    Prometheus can't send one.
    """
    return Response(content=metrics.to_prometheus_text(), media_type="text/plain; version=0.0.4")


@router.get("/audit-logs")
async def audit_logs(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    cursor = mongodb.db[AUDIT_LOGS].find({}).sort("created_at", -1).limit(limit)
    items = [_json_safe_document(item) async for item in cursor]
    return {"items": items, "count": len(items)}


@router.post("/knowledge-gaps/{message_id}/source", response_model=LegalSourceResponse)
async def upload_gap_source(
    message_id: str,
    file: UploadFile = File(...),
    metadata_json: str = Form(...),
    admin_user_id: str = Depends(_admin_id),
) -> LegalSourceResponse:
    metadata = LegalSourceMetadata.model_validate_json(metadata_json)
    staged = await KnowledgeBaseIngestionService().stage(file)
    staging_id = staged.staging_id
    # `claimed` is False when identical bytes already have an active ingestion
    # job: the attempt is recorded and archived, and enqueuing it again would
    # index the same source twice.
    if staged.claimed:
        kb_indexing_queue.enqueue(staging_id, staged.staged_path, staged.original_filename)
    source = await LegalUpdateService().create(admin_user_id, metadata, staging_id=staging_id)
    await AnalyticsService().mark_reviewed(message_id, "reviewed", f"Source queued: {source.source_id}")
    await AuditService().record(
        actor_user_id=admin_user_id, action="knowledge_gap_source_uploaded",
        resource_type="legal_source", resource_id=source.source_id,
        details={"message_id": message_id, "staging_id": staging_id},
    )
    return source


@router.get("/legal-sources", response_model=list[LegalSourceResponse])
async def list_legal_sources(
    stale_only: bool = False, verification_status: str | None = None,
) -> list[LegalSourceResponse]:
    return await LegalUpdateService().list(stale_only=stale_only, status=verification_status)


@router.post("/legal-sources/{source_id}/verify", response_model=LegalSourceResponse)
async def verify_legal_source(
    source_id: str, request: VerifySourceRequest, admin_user_id: str = Depends(_admin_id)
) -> LegalSourceResponse:
    result = await LegalUpdateService().verify(source_id, admin_user_id, request)
    await AuditService().record(
        actor_user_id=admin_user_id, action="legal_source_verified",
        resource_type="legal_source", resource_id=source_id,
        details={"verification_status": request.verification_status},
    )
    return result


@router.post("/legal-sources/{source_id}/review", response_model=LegalSourceResponse)
async def review_legal_source(
    source_id: str, request: SourceReviewRequest, admin_user_id: str = Depends(_admin_id)
) -> LegalSourceResponse:
    """Record a human review decision on one source.

    Admin-only through the router-level `require_admin` dependency, and audited
    unconditionally: a change to whether the platform presents a source as
    verified law is exactly the kind of change that has to be attributable
    afterwards.
    """
    result = await LegalUpdateService().review(source_id, admin_user_id, request)
    await AuditService().record(
        actor_user_id=admin_user_id,
        action="legal_source_reviewed",
        resource_type="legal_source",
        resource_id=source_id,
        details={
            "verification_status": request.verification_status,
            "last_verified_date": request.last_verified_date.isoformat(),
            # The evidence URL is the point of the record; notes are the
            # reviewer's own words and are stored on the source, not duplicated
            # into the audit trail.
            "evidence_url": request.evidence_url,
            "legal_status": request.status,
        },
    )
    return result


@router.post("/legal-sources/{source_id}/supersede", response_model=LegalSourceResponse)
async def supersede_legal_source(
    source_id: str, request: SupersedeSourceRequest, admin_user_id: str = Depends(_admin_id)
) -> LegalSourceResponse:
    """Record that another registered source has replaced this one in law."""
    result = await LegalUpdateService().supersede(source_id, admin_user_id, request)
    await AuditService().record(
        actor_user_id=admin_user_id,
        action="legal_source_superseded",
        resource_type="legal_source",
        resource_id=source_id,
        details={"superseded_by": request.superseded_by_source_id},
    )
    return result


@router.post("/legal-sources/{source_id}/link-document", response_model=LegalSourceResponse)
async def link_source_document(
    source_id: str, request: LinkSourceDocumentRequest, admin_user_id: str = Depends(_admin_id)
) -> LegalSourceResponse:
    result = await LegalUpdateService().link_document(source_id, request.document_id)
    await AuditService().record(
        actor_user_id=admin_user_id, action="legal_source_document_linked",
        resource_type="legal_source", resource_id=source_id, details={"document_id": request.document_id},
    )
    return result


@router.post("/evaluations/run", response_model=EvaluationRunResponse)
async def run_evaluation(
    request: EvaluationRunRequest, admin_user_id: str = Depends(_admin_id)
) -> EvaluationRunResponse:
    result = await EvaluationService().run(request.benchmark_path)
    await AuditService().record(
        actor_user_id=admin_user_id, action="evaluation_run", resource_type="evaluation",
        resource_id=result.run_id, details={"cases": result.cases},
    )
    return result


@router.get("/evaluations", response_model=list[EvaluationRunResponse])
async def list_evaluations(limit: int = Query(50, ge=1, le=500)) -> list[EvaluationRunResponse]:
    """Security/correctness finding G6: `POST /evaluations/run`'s response
    was the only way to ever see a run's result -- once it scrolled off an
    admin's screen, a durably-stored historical run was unreachable."""
    return await EvaluationService().list_runs(limit)


@router.get("/evaluations/{run_id}", response_model=EvaluationRunResponse)
async def get_evaluation(run_id: str) -> EvaluationRunResponse:
    return await EvaluationService().get_run(run_id)
