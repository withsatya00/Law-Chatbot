"""Phase 2 workflow acceptance tests.

These exercise each workflow from validated request through final response,
without a database or LLM, so emergency behavior remains deterministic.
"""

import pytest

from app.schemas.phase2 import (
    CyberFraudRequest,
    EvidenceInput,
    EvidenceOrganizeRequest,
    JurisdictionRequest,
    TimelineRequest,
)
from app.services.phase2_workflow import (
    analyze_timeline,
    cyber_fraud_workflow,
    jurisdiction_check,
    organize_evidence,
)


@pytest.mark.parametrize(
    ("language", "narrative", "expected_type", "language_marker"),
    [
        ("english", "I lost Rs 45,000 in a UPI fraud on PhonePe.", "upi_fraud", "Contact the bank"),
        ("hindi", "मेरे साथ OTP scam हुआ और 45000 रुपये चले गए।", "otp_scam", "बैंक/पेमेंट"),
        ("hinglish", "Mera account hack hua aur net banking se paise gaye.", "account_takeover", "turant report"),
    ],
)
def test_cyber_fraud_flow_is_multilingual_and_uses_verified_escalation(
    language: str, narrative: str, expected_type: str, language_marker: str
) -> None:
    response = cyber_fraud_workflow(CyberFraudRequest(narrative=narrative, language=language))

    assert response.detected_fraud_type == expected_type
    assert language_marker in response.immediate_steps[0]
    assert {option["action"] for option in response.emergency_options} == {
        "Call 1930 immediately for financial cyber fraud",
        "Report and track the incident at cybercrime.gov.in",
    }
    assert all(source.url.startswith("https://") for source in response.sources)
    assert set(response.drafts) == {"cybercrime_complaint", "bank_dispute_letter", "police_complaint"}
    assert "automatic" not in " ".join(response.drafts.values()).lower()


def test_evidence_extraction_annexures_and_draft_references() -> None:
    evidence = [EvidenceInput(
        evidence_id="ev-1",
        document_name="upi_receipt.pdf",
        text="UPI debit Rs 45,000 on 26/08/2026. UTR: HDFC1234567890",
        description="Payment receipt",
    )]
    organized = organize_evidence(EvidenceOrganizeRequest(evidence=evidence))
    response = cyber_fraud_workflow(CyberFraudRequest(
        narrative="PhonePe UPI fraud",
        amount="Rs 45,000",
        transaction_time="26/08/2026",
        transaction_id="HDFC1234567890",
        bank="HDFC Bank",
        evidence=evidence,
    ))

    row = organized.evidence_table[0]
    assert row["annexure"] == "Annexure A-1"
    assert row["document_date"] == "2026-08-26"
    assert any(fact["slot"] == "transaction_id" for fact in row["extracted_facts"])
    assert "Annexure A-1" in organized.annexure_index
    assert "Annexure A-1" in response.drafts["cybercrime_complaint"]


def test_timeline_detects_contradiction_and_never_silently_resolves_it() -> None:
    request = TimelineRequest(
        messages=[{"role": "user", "content": "I lost Rs 35,000 on 25/08/2026 via UPI."}],
        evidence=[EvidenceInput(
            evidence_id="ev-1",
            document_name="bank_receipt.pdf",
            text="Debit Rs 45,000 on 26/08/2026 UTR HDFC1234567890",
        )],
    )
    unresolved = analyze_timeline(request)
    resolved = analyze_timeline(request.model_copy(update={"resolved_conflicts": {"amount": "45000", "incident_date": "2026-08-26"}}))

    assert unresolved.export_blocked is True
    assert {item["slot"] for item in unresolved.contradictions} >= {"amount", "incident_date"}
    assert all("Which one is correct?" in item["question"] for item in unresolved.contradictions)
    assert resolved.export_blocked is False


@pytest.mark.parametrize("matter_type", ["police_complaint", "consumer_complaint", "rti", "property", "cyber_complaint"])
def test_jurisdiction_is_tentative_and_warns_before_filing(matter_type: str) -> None:
    response = jurisdiction_check(JurisdictionRequest(matter_type=matter_type))

    assert response.verify_before_filing is True
    assert response.confidence == "low"
    assert response.missing_facts
    assert response.warning.startswith("VERIFY BEFORE FILING")

