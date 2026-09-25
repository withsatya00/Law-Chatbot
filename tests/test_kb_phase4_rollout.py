from app.rag.kb_jurisdiction import STATE_UT_CODES
from app.services.kb_source_registry import rollout_readiness, rollout_registry


def test_registry_covers_every_state_and_ut_without_overclaiming() -> None:
    rows = rollout_registry()
    assert {row["code"] for row in rows} == set(STATE_UT_CODES)
    assert next(row for row in rows if row["code"] == "UP")["source_status"] == "verified"
    assert next(row for row in rows if row["code"] == "KL")["source_status"] == "verification_required"


def test_readiness_requires_source_health_corpus_and_benchmark() -> None:
    coverage = {"jurisdictions": [{
        "code": "UP", "approved_chunks": 20, "tested_passed": 1,
        "manual_access_required": 0,
    }]}
    automation = {
        "routine_manual_downloads_required": 0,
        "adapters": [
            {"adapter": "up_acts", "status": "healthy"},
            {"adapter": "up_ordinances", "status": "healthy"},
        ],
    }
    result = rollout_readiness(coverage, automation)
    up = next(row for row in result["jurisdictions"] if row["code"] == "UP")
    assert up["ready"] is True
    assert result["production_ready"] is False


def test_active_onboarded_source_extends_registry_without_code_change() -> None:
    result = rollout_readiness(
        {"jurisdictions": []},
        {"adapters": []},
        [{"name": "kerala_laws", "status": "active", "jurisdiction_code": "KL"}],
    )
    kerala = next(row for row in result["jurisdictions"] if row["code"] == "KL")
    assert kerala["source_status"] == "verified"
    assert kerala["adapters"] == ["kerala_laws"]
