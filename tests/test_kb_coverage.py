import asyncio

from app.rag.kb_jurisdiction import STATE_UT_CODES
from app.services.kb_coverage import KnowledgeBaseCoverageService


def test_coverage_matrix_contains_central_all_states_and_honest_stages() -> None:
    report = KnowledgeBaseCoverageService.from_grouped_chunks([
        {
            "_id": {
                "issuing_level": "central",
                "state_codes": [],
                "source_document": "central-act.pdf",
            },
            "chunks": 5,
            "approved_chunks": 5,
        },
        {
            "_id": {
                "issuing_level": "state",
                "state_codes": ["MP"],
                "source_document": "mp-act.pdf",
            },
            "chunks": 3,
            "approved_chunks": 1,
        },
    ])

    assert report["scope"]["jurisdictions"] == 1 + len(STATE_UT_CODES)
    assert report["scope"]["jurisdictions_with_indexed_documents"] == 2
    rows = {row["code"]: row for row in report["jurisdictions"]}
    assert rows["IN"]["status"] == "approved"
    assert rows["MP"]["status"] == "indexed_needs_review"
    assert rows["UP"]["status"] == "catalogued_only"
    assert report["complete"] is False


def test_unmapped_chunks_are_visible_instead_of_counted_as_coverage() -> None:
    report = KnowledgeBaseCoverageService.from_grouped_chunks([
        {
            "_id": {"issuing_level": "unknown", "state_codes": [], "source_document": "legacy.pdf"},
            "chunks": 4,
            "approved_chunks": 0,
        }
    ])

    assert report["unmapped"] == {
        "indexed_documents": 1,
        "indexed_chunks": 4,
        "approved_chunks": 0,
    }


def _base_report() -> dict:
    return KnowledgeBaseCoverageService.from_grouped_chunks([
        {
            "_id": {"issuing_level": "central", "state_codes": [], "source_document": "central-act.pdf"},
            "chunks": 5, "approved_chunks": 5,
        },
        {
            "_id": {"issuing_level": "state", "state_codes": ["UP"], "source_document": "up-act.pdf"},
            "chunks": 3, "approved_chunks": 3,
        },
    ])


def test_merge_automation_requires_a_passing_benchmark_only_when_one_ran() -> None:
    report = _base_report()

    merged = KnowledgeBaseCoverageService.merge_automation(report, {
        "UP": {"discovered": 4, "downloaded": 2, "manual_access_required": 0, "tested_passed": 0, "tested_total": 1},
    })

    rows = {row["code"]: row for row in merged["jurisdictions"]}
    assert rows["UP"]["tested_total"] == 1
    assert rows["UP"]["complete"] is False  # a benchmark ran and failed
    assert rows["IN"]["complete"] is True  # no benchmark was ever run for IN -- not penalized
    assert merged["complete"] is False


def test_merge_automation_passes_once_the_benchmark_passes() -> None:
    report = _base_report()

    merged = KnowledgeBaseCoverageService.merge_automation(report, {
        "UP": {"discovered": 4, "downloaded": 4, "manual_access_required": 0, "tested_passed": 1, "tested_total": 1},
    })

    rows = {row["code"]: row for row in merged["jurisdictions"]}
    assert rows["UP"]["complete"] is True
    # Overall `complete` still requires every one of the 28+8+1 jurisdictions,
    # not just the two populated in this fixture.
    assert merged["complete"] is False


def test_merge_automation_surfaces_manual_access_exceptions_without_forcing_incompleteness() -> None:
    report = _base_report()

    merged = KnowledgeBaseCoverageService.merge_automation(report, {
        "MH": {"discovered": 1, "downloaded": 1, "manual_access_required": 1, "tested_passed": 0, "tested_total": 0},
    })

    assert merged["manual_access_exceptions"] == ["MH"]
    rows = {row["code"]: row for row in merged["jurisdictions"]}
    assert rows["MH"]["manual_access_required"] == 1
    # MH has no approved chunks either way, so it is legitimately incomplete --
    # the exception is reported, not used to fake completeness.
    assert rows["MH"]["complete"] is False


def test_reconciliation_report_classifies_the_furthest_stage_reached() -> None:
    service = KnowledgeBaseCoverageService()
    fake_coverage = {
        "jurisdictions": [
            {"code": "IN", "name": "Central", "complete": True,
             "tested_total": 0, "tested_passed": 0, "indexed_documents": 1, "manual_access_required": 0, "downloaded": 0, "discovered": 0},
            {"code": "UP", "name": "Uttar Pradesh", "complete": False,
             "tested_total": 1, "tested_passed": 0, "indexed_documents": 1, "manual_access_required": 0, "downloaded": 3, "discovered": 4},
            {"code": "MH", "name": "Maharashtra", "complete": False,
             "tested_total": 0, "tested_passed": 0, "indexed_documents": 0, "manual_access_required": 1, "downloaded": 1, "discovered": 1},
            {"code": "GA", "name": "Goa", "complete": False,
             "tested_total": 0, "tested_passed": 0, "indexed_documents": 0, "manual_access_required": 0, "downloaded": 2, "discovered": 2},
            {"code": "SK", "name": "Sikkim", "complete": False,
             "tested_total": 0, "tested_passed": 0, "indexed_documents": 0, "manual_access_required": 0, "downloaded": 0, "discovered": 1},
            {"code": "ML", "name": "Meghalaya", "complete": False,
             "tested_total": 0, "tested_passed": 0, "indexed_documents": 0, "manual_access_required": 0, "downloaded": 0, "discovered": 0},
        ],
        "scope": {"jurisdictions": 6},
        "complete": False,
        "manual_access_exceptions": ["MH"],
    }
    service.coverage = lambda: asyncio.sleep(0, result=fake_coverage)  # type: ignore[method-assign]

    report = asyncio.run(service.reconciliation_report())

    stages = {item["code"]: item["stuck_at"] for item in report["stuck_jurisdictions"]}
    assert stages == {
        "UP": "tested_failing",
        "MH": "manual_access_exception",
        "GA": "downloaded_not_indexed",
        "SK": "discovered_not_downloaded",
        "ML": "not_configured",
    }
    assert "IN" not in stages
