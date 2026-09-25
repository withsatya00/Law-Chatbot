"""Deterministic Phase 2 legal workflows.

No legal fact or emergency contact in this module is model-generated.  The
source registry is intentionally small and official; UI clients can display
the links beside the action they support.
"""

from __future__ import annotations

from typing import Any

from app.casefile.annexures import (
    EvidenceItem,
    build_evidence_table,
    evidence_checklist,
    render_annexure_index,
)
from app.casefile.timeline import blocking_conflicts, build_timeline, render_timeline
from app.schemas.phase2 import (
    CyberFraudRequest,
    CyberFraudResponse,
    EvidenceOrganizeRequest,
    EvidenceOrganizeResponse,
    FraudType,
    JurisdictionRequest,
    JurisdictionResponse,
    SourceLink,
    TimelineRequest,
    TimelineResponse,
)

_VERIFIED_ON = "2026-09-01"
_NCRP = SourceLink(
    title="National Cyber Crime Reporting Portal",
    url="https://www.cybercrime.gov.in/",
    authority="Indian Cybercrime Coordination Centre, Ministry of Home Affairs",
    verified_on=_VERIFIED_ON,
)
_MHA_1930 = SourceLink(
    title="Citizen Financial Cyber Fraud Reporting and Management System",
    url="https://ncrp-grievanceredressal.mha.gov.in/",
    authority="Ministry of Home Affairs, Government of India",
    verified_on=_VERIFIED_ON,
)
_RBI = SourceLink(
    title="Customer Protection – Limiting Liability of Customers in Unauthorised Electronic Banking Transactions",
    url="https://www.rbi.org.in/commonman/English/scripts/Notification.aspx?Id=2623",
    authority="Reserve Bank of India",
    verified_on=_VERIFIED_ON,
)

_FRAUD_PATTERNS: tuple[tuple[FraudType, tuple[str, ...]], ...] = (
    ("account_takeover", ("account takeover", "account hacked", "sim swap", "खाता हैक", "account hack")),
    ("investment_scam", ("investment", "trading scam", "crypto", "double money", "निवेश", "investment fraud")),
    ("otp_scam", ("otp", "one time password", "ओटीपी")),
    ("card_fraud", ("credit card", "debit card", "card fraud", "कार्ड")),
    ("online_banking_fraud", ("net banking", "online banking", "imps", "neft", "internet banking")),
    ("upi_fraud", ("upi", "gpay", "google pay", "phonepe", "paytm", "bhim", "यूपीआई")),
)

_TEXT: dict[str, dict[str, str]] = {
    "english": {
        "bank": "Contact the bank/payment provider immediately through its official fraud channel; block the affected card/account or disable UPI where appropriate, and keep the complaint number.",
        "secure": "Change banking, email and device passwords from a trusted device; revoke unknown sessions and never share a fresh OTP, PIN or screen-sharing access.",
        "preserve": "Preserve screenshots, messages, call logs, receipts and original files. Do not crop, edit or delete them.",
        "warning": "This is a workflow aid, not a finding of liability. Verify names, amounts, dates, jurisdiction and annexures before filing; have a qualified advocate review the final document.",
    },
    "hindi": {
        "bank": "बैंक/पेमेंट प्रदाता के आधिकारिक फ्रॉड चैनल पर तुरंत सूचना दें; जरूरत के अनुसार कार्ड/खाता ब्लॉक या UPI बंद करें और शिकायत संख्या सुरक्षित रखें।",
        "secure": "विश्वसनीय डिवाइस से बैंकिंग, ईमेल और डिवाइस पासवर्ड बदलें; अनजान सत्र हटाएँ और नया OTP, PIN या स्क्रीन-शेयरिंग एक्सेस किसी को न दें।",
        "preserve": "स्क्रीनशॉट, संदेश, कॉल लॉग, रसीदें और मूल फाइलें सुरक्षित रखें। उन्हें काटें, बदलें या मिटाएँ नहीं।",
        "warning": "यह एक कार्यप्रवाह सहायता है, दायित्व पर अंतिम राय नहीं। फाइल करने से पहले नाम, रकम, तारीख, क्षेत्राधिकार और संलग्नक जाँचें तथा अंतिम दस्तावेज की अधिवक्ता से समीक्षा कराएँ।",
    },
    "hinglish": {
        "bank": "Bank/payment provider ke official fraud channel par turant report karein; zarurat ke hisaab se card/account block ya UPI disable karein aur complaint number sambhal kar rakhein.",
        "secure": "Trusted device se banking, email aur device passwords badlein; unknown sessions hata dein aur naya OTP, PIN ya screen-sharing access kisi ko na dein.",
        "preserve": "Screenshots, messages, call logs, receipts aur original files preserve karein. Unhe crop, edit ya delete na karein.",
        "warning": "Yeh workflow aid hai, liability par final opinion nahi. Filing se pehle names, amount, dates, jurisdiction aur annexures verify karein; final document advocate se review karayein.",
    },
}


def _detect_fraud_type(text: str) -> FraudType:
    lowered = text.casefold()
    for fraud_type, phrases in _FRAUD_PATTERNS:
        if any(phrase.casefold() in lowered for phrase in phrases):
            return fraud_type
    return "unknown"


def _evidence_items(evidence: list[Any]) -> list[EvidenceItem]:
    return [EvidenceItem(**item.model_dump()) for item in evidence]


def organize_evidence(request: EvidenceOrganizeRequest) -> EvidenceOrganizeResponse:
    rows = build_evidence_table(_evidence_items(request.evidence), series=request.series)
    return EvidenceOrganizeResponse(
        evidence_table=[row.as_dict() for row in rows],
        annexure_index=render_annexure_index(rows),
    )


def analyze_timeline(request: TimelineRequest) -> TimelineResponse:
    evidence_rows = build_evidence_table(_evidence_items(request.evidence))
    evidence = [
        {**item.model_dump(), "annexure": row.annexure}
        for item, row in zip(request.evidence, evidence_rows, strict=True)
    ]
    case_facts = build_timeline(
        messages=request.messages,
        evidence=evidence,
        draft_fields=request.draft_fields,
    )
    conflicts = blocking_conflicts(case_facts, request.resolved_conflicts)
    return TimelineResponse(
        timeline=[event.as_dict() for event in case_facts.events],
        extracted_facts=[
            {"slot": fact.slot, "label": fact.label, "value": fact.value, "source": fact.source}
            for fact in case_facts.facts
        ],
        contradictions=[conflict.as_dict() for conflict in conflicts],
        export_blocked=bool(conflicts),
    )


def _collected_fields(request: CyberFraudRequest) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in {
            "amount": request.amount,
            "incident_date": request.transaction_time,
            "transaction_id": request.transaction_id,
            "bank_name": request.bank,
            "fraud_type": request.platform,
            "applicant_name": request.complainant_name,
            "applicant_address": request.complainant_address,
            "police_station": request.police_station,
        }.items()
        if value
    }


def _required_information(request: CyberFraudRequest) -> list[dict[str, Any]]:
    values = {
        "amount": request.amount,
        "transaction_time": request.transaction_time,
        "UTR/reference ID": request.transaction_id,
        "bank": request.bank,
        "platform": request.platform,
        "sender details": request.sender_details,
        "receiver details": request.receiver_details,
        "screenshots": any("screenshot" in item.document_name.casefold() for item in request.evidence),
        "call/chat evidence": any(
            word in f"{item.document_name} {item.description}".casefold()
            for item in request.evidence for word in ("call", "chat", "whatsapp", "message")
        ),
    }
    return [{"field": key, "status": "provided" if value else "missing"} for key, value in values.items()]


def _drafts(request: CyberFraudRequest, fraud_type: FraudType, timeline: str, annexures: str) -> dict[str, str]:
    amount = request.amount or "[amount to be confirmed]"
    when = request.transaction_time or "[transaction date/time to be confirmed]"
    ref = request.transaction_id or "[UTR/reference ID to be confirmed]"
    bank = request.bank or "[bank/payment provider]"
    complainant = request.complainant_name or "[complainant name]"
    details = request.narrative.strip()
    evidence_ref = f"\n\nANNEXURE INDEX\n{annexures}" if annexures else ""
    common = (
        f"Complainant: {complainant}\nNature: {fraud_type.replace('_', ' ').title()}\n"
        f"Amount: {amount}\nTransaction date/time: {when}\nUTR/reference: {ref}\nBank/platform: {bank}"
    )
    cyber = (
        "CYBERCRIME COMPLAINT\n\n" + common + "\n\nFACTS\n" + details
        + "\n\nCHRONOLOGY\n" + (timeline or "[Timeline to be completed]")
        + "\n\nREQUEST\nPlease register and examine this complaint, trace the transaction trail, preserve relevant electronic records, and provide an acknowledgement/complaint number."
        + evidence_ref
    )
    bank_letter = (
        f"TO: The Grievance/Fraud Officer, {bank}\n\nSUBJECT: Dispute of unauthorised/suspected fraudulent transaction {ref}\n\n"
        + common + "\n\n" + details
        + "\n\nPlease block further unauthorised access, register this dispute, preserve transaction/authentication logs, take steps permitted by the applicable RBI customer-protection framework, and provide the complaint number and written outcome."
        + evidence_ref
    )
    police = (
        f"TO: The Station House Officer, {request.police_station or '[police station to be verified]'}\n\n"
        "SUBJECT: Complaint regarding suspected cyber-enabled financial fraud\n\n"
        + common + "\n\n" + details
        + "\n\nPlease receive this complaint, provide an acknowledgement, and take action in accordance with law. The filing police station/jurisdiction must be verified."
        + evidence_ref
    )
    return {"cybercrime_complaint": cyber, "bank_dispute_letter": bank_letter, "police_complaint": police}


def cyber_fraud_workflow(request: CyberFraudRequest) -> CyberFraudResponse:
    fraud_type = request.fraud_type or _detect_fraud_type(request.narrative)
    rows = build_evidence_table(_evidence_items(request.evidence))
    evidence = [
        {**item.model_dump(), "annexure": row.annexure}
        for item, row in zip(request.evidence, rows, strict=True)
    ]
    case_facts = build_timeline(
        messages=[{"role": "user", "content": request.narrative}],
        evidence=evidence,
        draft_fields=_collected_fields(request),
    )
    conflicts = blocking_conflicts(case_facts, request.resolved_conflicts)
    annexure_index = render_annexure_index(rows)
    text = _TEXT[request.language]
    immediate_steps = [
        text["bank"],
        text["secure"],
        text["preserve"],
    ]
    emergency_options = [
        {
            "action": "Call 1930 immediately for financial cyber fraud",
            "availability_note": "National cybercrime helpline; reporting does not guarantee recovery.",
            "source_url": _MHA_1930.url,
        },
        {
            "action": "Report and track the incident at cybercrime.gov.in",
            "availability_note": "Use the official National Cyber Crime Reporting Portal.",
            "source_url": _NCRP.url,
        },
    ]
    extracted = [
        {"slot": fact.slot, "label": fact.label, "value": fact.value, "source": fact.source}
        for fact in case_facts.facts
    ]
    return CyberFraudResponse(
        detected_fraud_type=fraud_type,
        risk_level="critical" if fraud_type == "account_takeover" else "high",
        immediate_steps=immediate_steps,
        emergency_options=emergency_options,
        required_information=_required_information(request),
        extracted_facts=extracted,
        evidence_table=[row.as_dict() for row in rows],
        evidence_checklist=evidence_checklist(rows, ("bank_record", "screenshot", "chat_record", "call_record")),
        timeline=[event.as_dict() for event in case_facts.events],
        contradictions=[conflict.as_dict() for conflict in conflicts],
        export_blocked=bool(conflicts),
        drafts=_drafts(request, fraud_type, render_timeline(case_facts.events), annexure_index),
        sources=[_MHA_1930, _NCRP, _RBI],
        lawyer_escalation_recommended=bool(conflicts) or fraud_type in {"investment_scam", "account_takeover"},
        warning=text["warning"],
    )


def jurisdiction_check(request: JurisdictionRequest) -> JurisdictionResponse:
    missing: list[str] = []
    bases: list[str] = []
    forum = "Jurisdiction cannot yet be narrowed"

    if request.matter_type == "property":
        if request.property_location:
            forum = f"The competent authority/court for the place where the property is situated ({request.property_location})"
            bases.append("The property location is ordinarily a central territorial connection.")
        else:
            missing.append("exact property address, district and State/UT")
    elif request.matter_type == "rti":
        if request.public_authority_location:
            forum = f"The designated PIO of the concerned public authority ({request.public_authority_location})"
            bases.append("RTI routing depends primarily on which public authority holds or controls the requested record.")
        else:
            missing.append("name and level (Central/State/local) of the public authority holding the record")
    elif request.matter_type == "consumer_complaint":
        locations = [request.complainant_location, request.respondent_location, request.transaction_location]
        if any(locations):
            forum = "The appropriate Consumer Commission connected to the complainant, respondent, or cause of action"
            bases.append("Residence/business of the complainant, respondent location, and where the cause of action arose may each matter.")
        for label, value in (("complainant residence/business", request.complainant_location), ("respondent location", request.respondent_location), ("place of transaction/cause of action", request.transaction_location)):
            if not value:
                missing.append(label)
    elif request.matter_type in {"police_complaint", "cyber_complaint"}:
        locations = [request.incident_location, request.complainant_location, request.branch_location]
        if any(locations):
            forum = "A police station/cyber cell with a territorial connection to the incident, victim, account branch, or transaction"
            bases.append("The incident location and locations connected to the victim/transaction help determine territorial handling.")
        for label, value in (("incident location", request.incident_location), ("complainant's current location", request.complainant_location), ("bank/payment branch or transaction location", request.branch_location or request.transaction_location)):
            if not value:
                missing.append(label)
        if request.online_transaction:
            bases.append("Online conduct may connect more than one place; portal routing and police verification remain necessary.")

    return JurisdictionResponse(
        tentative_forum=forum,
        possible_bases=bases,
        missing_facts=missing,
        confidence="medium" if bases and len(missing) <= 1 else "low",
        warning="VERIFY BEFORE FILING: This is a tentative jurisdiction screen, not final legal advice. Confirm the forum, territorial limits and current procedural requirements with the receiving authority or a qualified advocate.",
    )
