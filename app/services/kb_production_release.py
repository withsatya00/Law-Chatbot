"""Phase-6 fail-closed production release gates for the legal KB."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.cache.redis_client import redis_client
from app.core.config import settings
from app.database.mongodb import mongodb
from app.models.collections import EVALUATION_RUNS
from app.rag.kb_jurisdiction import STATE_UT_CODES
from app.services.kb_automation import KnowledgeBaseAutomationService
from app.services.kb_coverage import KnowledgeBaseCoverageService
from app.services.kb_operations import KBOperationsService
from app.services.kb_source_onboarding import SourceOnboardingService
from app.services.kb_source_registry import rollout_readiness


def required_jurisdictions(value: str) -> list[str]:
    codes = list(dict.fromkeys(code.strip().upper() for code in value.split(",") if code.strip()))
    unknown = set(codes) - set(STATE_UT_CODES)
    if unknown:
        raise ValueError(f"Unknown required jurisdiction(s): {', '.join(sorted(unknown))}")
    if not codes:
        raise ValueError("At least one production jurisdiction is required.")
    return codes


def evaluate_release_gates(
    *, infrastructure: dict[str, bool], rollout: dict[str, Any],
    latest_evaluation: dict[str, Any] | None, required_codes: list[str],
    now: datetime | None = None, max_age_hours: int = 168, min_score: float = 0.95,
    automation_enabled: bool,
    operations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a release without mutating corpus, source, or deployment state."""
    now = now or datetime.now(UTC)
    rows = {row["code"]: row for row in rollout.get("jurisdictions", [])}
    jurisdiction_gates = {
        code: bool(rows.get(code, {}).get("ready")) for code in required_codes
    }
    evaluation_time = latest_evaluation.get("created_at") if latest_evaluation else None
    if evaluation_time is not None and evaluation_time.tzinfo is None:
        evaluation_time = evaluation_time.replace(tzinfo=UTC)
    evaluation_fresh = bool(
        evaluation_time and now - evaluation_time <= timedelta(hours=max_age_hours)
    )
    scores = latest_evaluation.get("scores", {}) if latest_evaluation else {}
    evaluation_scores_pass = bool(scores) and all(
        float(score) >= min_score for score in scores.values()
    )
    gates = {
        "database_ready": bool(infrastructure.get("database")),
        "redis_ready": bool(infrastructure.get("redis")),
        "automation_enabled": automation_enabled,
        "required_jurisdictions_ready": all(jurisdiction_gates.values()),
        "evaluation_present": latest_evaluation is not None,
        "evaluation_fresh": evaluation_fresh,
        "evaluation_scores_pass": evaluation_scores_pass,
        "no_evaluation_failures": bool(latest_evaluation) and not (
            latest_evaluation or {}
        ).get("failures"),
        "review_sla_clear": int((operations or {}).get("review_overdue_count", 0)) == 0,
        "relationship_review_clear": int(
            (operations or {}).get("relationship_review_pending", 0)
        ) == 0,
        "ocr_quality_clear": int((operations or {}).get("low_ocr_documents", 0)) == 0,
        "adapter_health_clear": not (operations or {}).get("unhealthy_adapters", []),
    }
    blockers = [name for name, passed in gates.items() if not passed]
    blockers.extend(
        f"jurisdiction_not_ready:{code}"
        for code, passed in jurisdiction_gates.items() if not passed
    )
    return {
        "phase": 6,
        "release_status": "go" if not blockers else "blocked",
        "production_ready": not blockers,
        "required_jurisdictions": required_codes,
        "jurisdiction_gates": jurisdiction_gates,
        "gates": gates,
        "blockers": blockers,
        "latest_evaluation": latest_evaluation,
        "generated_at": now.isoformat(),
    }


class KBProductionReleaseService:
    def __init__(self, db: Any = None) -> None:
        self.db = db if db is not None else mongodb.db

    async def readiness(self) -> dict[str, Any]:
        automation_service = KnowledgeBaseAutomationService(self.db)
        automation = await automation_service.coverage()
        coverage = await KnowledgeBaseCoverageService().coverage()
        onboarded = await SourceOnboardingService(self.db).list_sources()
        rollout = rollout_readiness(coverage, automation, onboarded)
        operations = await KBOperationsService(self.db).status()
        latest = await self.db[EVALUATION_RUNS].find_one(
            {"status": "complete"}, sort=[("created_at", -1)],
        )
        infrastructure = {
            "database": await mongodb.ping(),
            "redis": await redis_client.ping(),
        }
        return evaluate_release_gates(
            infrastructure=infrastructure,
            rollout=rollout,
            latest_evaluation=latest,
            required_codes=required_jurisdictions(settings.kb_production_required_jurisdictions),
            max_age_hours=settings.kb_release_evaluation_max_age_hours,
            min_score=settings.kb_release_min_evaluation_score,
            automation_enabled=bool(automation.get("enabled")),
            operations=operations,
        )
