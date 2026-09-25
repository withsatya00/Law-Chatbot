from datetime import date

from fastapi.testclient import TestClient

from app.casefile.annexures import EvidenceItem
from app.drafting.matter_workspace import (
    CaseFacts,
    CourtPleadingBuilder,
    EvidenceTraceabilityEngine,
    FilingBundleGenerator,
    JurisdictionRulePackRegistry,
    LimitationDeadlineEngine,
    MatterConsistencyEngine,
    MatterDocumentChain,
    MatterParty,
    RedlineRiskAnalyzer,
)
from app.main import app


def _matter() -> CaseFacts:
    return CaseFacts(
        "matter-1",
        parties=[
            MatterParty("Asha", "plaintiff", "Delhi"),
            MatterParty("Bharat", "defendant", "Noida"),
        ],
        fields={"jurisdiction_basis": "Cause of action arose in Delhi", "suit_valuation": "Rs. 50,000"},
        events=[{"date": "2026-01-02", "description": "Demand made"},
                {"date": "2026-01-01", "description": "Payment due"}],
        requested_reliefs=["Recovery of Rs. 50,000", "Interim restraint"],
        evidence=[EvidenceItem("e1", "bank_receipt.pdf", "Paid Rs. 50,000 on 01/01/2026 UTR ABC123456789")],
    )


def test_casefacts_document_chain_is_stage_aware() -> None:
    matter = _matter()
    chain = MatterDocumentChain().plan(matter, "civil")
    assert chain[0].status == "available"
    assert chain[1].status == "blocked"
    matter.prior_documents.append("advocate_legal_notice")
    assert MatterDocumentChain().plan(matter, "civil")[1].status == "available"


def test_court_pleading_workspace_orders_dates_and_preserves_placeholders() -> None:
    workspace = CourtPleadingBuilder().build(_matter())
    assert workspace.list_of_dates[0]["date"] == "2026-01-01"
    assert workspace.court_fee == "[COURT FEE REVIEW REQUIRED]"
    assert workspace.cause_title["petitioners"][0]["name"] == "Asha"


def test_annexure_traceability_and_reference_validation() -> None:
    rows, traces = EvidenceTraceabilityEngine().build(_matter())
    assert rows[0].annexure == "Annexure A-1"
    assert traces
    issues = EvidenceTraceabilityEngine.validate_body_references("See Annexure A-2.", rows)
    assert {issue["error_code"] for issue in issues} == {"ANNEXURE_REFERENCE_MISSING", "ANNEXURE_ORPHAN"}


def test_consistency_engine_flags_conflicting_amounts() -> None:
    matter = _matter()
    matter.evidence.append(EvidenceItem("e2", "invoice.pdf", "Invoice amount Rs. 75,000 dated 01/01/2026"))
    conflicts = MatterConsistencyEngine().check(matter)
    assert any(item["slot"] == "amount" and item["resolution_required"] for item in conflicts)


def test_jurisdiction_pack_returns_only_missing_material_fields() -> None:
    missing = JurisdictionRulePackRegistry().validate("delhi_district_courts", _matter().fields)
    assert "court_name" in missing
    assert "court_fee" in missing
    assert "jurisdiction_basis" not in missing


def test_deadline_engine_refuses_unverified_filing_conclusion() -> None:
    engine = LimitationDeadlineEngine()
    advisory = engine.calculate(rule_id="example", anchor_date=date(2026, 1, 1), days=30,
                                legal_basis="Configured example", verification_status="review_required",
                                today=date(2026, 1, 2))
    assert advisory.due_date == "2026-01-31"
    assert advisory.status == "REVIEW_REQUIRED"
    verified = engine.calculate(rule_id="verified", anchor_date=date(2026, 1, 1), days=30,
                                legal_basis="Verified rule", verification_status="verified",
                                today=date(2026, 1, 2))
    assert verified.status == "OPEN"


def test_redline_risk_detects_amount_and_liability_changes() -> None:
    report = RedlineRiskAnalyzer().compare(
        "Liability is capped at Rs. 50,000.", "Liability is capped at Rs. 75,000.",
    )
    assert report.status == "REVIEW_REQUIRED"
    assert "Rs. 50,000" in report.changed_protected_values
    assert any(flag["risk"] == "liability" for flag in report.risk_flags)


def test_filing_bundle_manifest_orders_and_blocks_incomplete_bundle() -> None:
    manifest = FilingBundleGenerator().manifest(_matter(), [
        {"kind": "annexures", "title": "Annexures", "path": "annexures.pdf"},
        {"kind": "index", "title": "Index", "path": "index.pdf"},
        {"kind": "main_pleading", "title": "Plaint", "path": ""},
    ])
    assert manifest.documents[0]["kind"] == "index"
    assert manifest.validation_status == "BLOCKED"
    assert "Plaint" in manifest.bookmarks


def test_matter_workspace_and_redline_endpoints() -> None:
    client = TestClient(app)
    response = client.post("/matter/workspace/analyze", json={
        "matter_id": "m-1", "domain": "civil",
        "parties": [{"name": "A", "role": "plaintiff", "address": "Delhi"},
                    {"name": "B", "role": "defendant", "address": "Noida"}],
        "fields": {"jurisdiction_basis": "Delhi"},
        "requested_reliefs": ["Recovery"],
    })
    assert response.status_code == 200
    assert response.json()["document_chain"][0]["document_id"] == "advocate_legal_notice"
    assert response.json()["filing_bundle"]["validation_status"] == "READY_FOR_ADVOCATE_REVIEW"

    risk = client.post("/draft/redline/risk", json={
        "original_text": "Liability Rs. 10,000", "revised_text": "Liability Rs. 20,000",
    })
    assert risk.status_code == 200
    assert risk.json()["status"] == "REVIEW_REQUIRED"
