from datetime import UTC, datetime, timedelta

import pytest

from app.services.kb_production_release import (
    evaluate_release_gates,
    required_jurisdictions,
)

NOW = datetime(2026, 9, 7, tzinfo=UTC)


def rollout(ready_codes: set[str]) -> dict:
    return {"jurisdictions": [
        {"code": code, "ready": code in ready_codes} for code in ("UP", "DL", "MH")
    ]}


def evaluation(**changes) -> dict:
    value = {
        "_id": "eval-1", "created_at": NOW - timedelta(hours=1), "status": "complete",
        "scores": {"routing_accuracy": 1.0, "language_accuracy": 0.98}, "failures": [],
    }
    return {**value, **changes}


def test_release_is_go_only_when_every_gate_passes() -> None:
    result = evaluate_release_gates(
        infrastructure={"database": True, "redis": True}, rollout=rollout({"UP", "DL"}),
        latest_evaluation=evaluation(), required_codes=["UP", "DL"], now=NOW,
        automation_enabled=True,
    )
    assert result["release_status"] == "go"
    assert result["production_ready"] is True
    assert result["blockers"] == []


def test_release_reports_actionable_blockers_and_stale_evaluation() -> None:
    result = evaluate_release_gates(
        infrastructure={"database": True, "redis": False}, rollout=rollout({"UP"}),
        latest_evaluation=evaluation(created_at=NOW - timedelta(days=9)),
        required_codes=["UP", "DL"], now=NOW, automation_enabled=False,
    )
    assert result["release_status"] == "blocked"
    assert "redis_ready" in result["blockers"]
    assert "automation_enabled" in result["blockers"]
    assert "evaluation_fresh" in result["blockers"]
    assert "jurisdiction_not_ready:DL" in result["blockers"]


def test_failed_or_missing_evaluation_never_passes() -> None:
    failed = evaluate_release_gates(
        infrastructure={"database": True, "redis": True}, rollout=rollout({"UP"}),
        latest_evaluation=evaluation(scores={"routing_accuracy": 0.8}, failures=[{"x": 1}]),
        required_codes=["UP"], now=NOW, automation_enabled=True,
    )
    assert failed["production_ready"] is False
    assert failed["gates"]["evaluation_scores_pass"] is False
    assert failed["gates"]["no_evaluation_failures"] is False

    missing = evaluate_release_gates(
        infrastructure={"database": True, "redis": True}, rollout=rollout({"UP"}),
        latest_evaluation=None, required_codes=["UP"], now=NOW, automation_enabled=True,
    )
    assert missing["gates"]["evaluation_present"] is False
    assert missing["production_ready"] is False


def test_required_jurisdictions_are_validated_and_deduplicated() -> None:
    assert required_jurisdictions("up, DL,UP") == ["UP", "DL"]
    with pytest.raises(ValueError, match="Unknown"):
        required_jurisdictions("UP,XX")
    with pytest.raises(ValueError, match="At least one"):
        required_jurisdictions(" ")


def test_phase6_readiness_route_is_admin_gated() -> None:
    from app.api.admin_phase3 import router
    from app.api.deps import require_admin

    assert any(dependency.dependency is require_admin for dependency in router.dependencies)
    assert "/admin/phase3/kb-production/readiness" in {route.path for route in router.routes}
