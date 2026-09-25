"""Matter-level orchestration for related legal drafts and filing review.

All calculations are advisory.  No limitation period, court fee, jurisdiction
or filing-readiness conclusion is invented when a verified rule is absent.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, ClassVar

from app.casefile.annexures import (
    AnnexureRow,
    EvidenceItem,
    annexure_reference_map,
    build_evidence_table,
)
from app.casefile.facts import ExtractedFact, extract_facts, facts_from_fields
from app.core import clock


@dataclass(frozen=True)
class MatterParty:
    name: str
    role: str
    address: str = ""
    contact: str = ""


@dataclass
class CaseFacts:
    matter_id: str
    parties: list[MatterParty] = field(default_factory=list)
    fields: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, str]] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    requested_reliefs: list[str] = field(default_factory=list)
    prior_documents: list[str] = field(default_factory=list)
    jurisdiction_profile: str = "generic_india"
    case_stage: str = "pre_litigation"

    def extracted_facts(self) -> list[ExtractedFact]:
        facts = facts_from_fields(self.fields)
        for item in self.evidence:
            facts.extend(extract_facts(item.text, item.document_name))
        return facts


@dataclass(frozen=True)
class ChainStep:
    document_id: str
    status: str
    reason: str
    required_before: tuple[str, ...] = ()


class MatterDocumentChain:
    """Stage-aware suggestions; it never auto-files or silently advances."""

    _CHAINS: ClassVar[dict[str, tuple[str, ...]]] = {
        "civil": ("advocate_legal_notice", "civil_plaint", "civil_interim_injunction_application", "general_affidavit"),
        "criminal": ("police_fir_application", "criminal_complaint", "general_affidavit", "bail_application"),
        "consumer": ("consumer_notice", "native_consumer_complaint", "general_affidavit"),
        "tenancy": ("rent_notice", "tenancy_termination_notice", "civil_plaint", "general_affidavit"),
    }

    def plan(self, matter: CaseFacts, domain: str) -> list[ChainStep]:
        chain = self._CHAINS.get(domain, ("advocate_legal_notice",))
        completed = set(matter.prior_documents)
        result: list[ChainStep] = []
        for index, document_id in enumerate(chain):
            prerequisites = tuple(item for item in chain[:index] if item not in completed)
            status = "completed" if document_id in completed else "available" if not prerequisites else "blocked"
            result.append(ChainStep(
                document_id, status,
                "Previously generated." if status == "completed" else
                "All earlier workflow prerequisites are present." if status == "available" else
                "Earlier procedural step requires review or completion.",
                prerequisites,
            ))
        return result


@dataclass(frozen=True)
class PleadingWorkspace:
    cause_title: dict[str, list[dict[str, str]]]
    memo_of_parties: tuple[dict[str, str], ...]
    list_of_dates: tuple[dict[str, str], ...]
    jurisdiction: str
    limitation: str
    valuation: str
    court_fee: str
    main_reliefs: tuple[str, ...]
    interim_reliefs: tuple[str, ...]
    missing_information: tuple[str, ...]


class CourtPleadingBuilder:
    def build(self, matter: CaseFacts) -> PleadingWorkspace:
        petitioners = [asdict(p) for p in matter.parties if p.role in {"petitioner", "plaintiff", "applicant", "complainant"}]
        respondents = [asdict(p) for p in matter.parties if p.role in {"respondent", "defendant", "opposite_party"}]
        missing: list[str] = []
        if not petitioners:
            missing.append("At least one petitioner/plaintiff/applicant is required.")
        if not respondents:
            missing.append("At least one respondent/defendant is required.")
        for party in [*petitioners, *respondents]:
            if not party.get("address"):
                missing.append(f"Address required for {party.get('name') or party.get('role')}.")
        events = tuple(sorted(matter.events, key=lambda item: item.get("date", "9999-99-99")))
        fields = matter.fields
        return PleadingWorkspace(
            {"petitioners": petitioners, "respondents": respondents}, (*petitioners, *respondents), events,
            fields.get("jurisdiction_basis", "[JURISDICTION BASIS REQUIRED]"),
            fields.get("limitation_basis", "[LIMITATION REVIEW REQUIRED]"),
            fields.get("suit_valuation", "[VALUATION REQUIRED]"),
            fields.get("court_fee", "[COURT FEE REVIEW REQUIRED]"),
            tuple(matter.requested_reliefs),
            tuple(value for value in matter.requested_reliefs if "interim" in value.casefold()),
            tuple(dict.fromkeys(missing)),
        )


@dataclass(frozen=True)
class EvidenceTrace:
    annexure: str
    fact_value: str
    source_document: str
    body_reference: str
    review_notes: tuple[str, ...]


class EvidenceTraceabilityEngine:
    def build(self, matter: CaseFacts) -> tuple[list[AnnexureRow], list[EvidenceTrace]]:
        rows = build_evidence_table(matter.evidence)
        references = annexure_reference_map(rows)
        traces: list[EvidenceTrace] = []
        for row in rows:
            for fact in row.facts:
                if references.get(fact.normalized) == row.annexure:
                    traces.append(EvidenceTrace(
                        row.annexure, fact.value, row.document_name,
                        f"Refer {row.annexure}", row.missing,
                    ))
        return rows, traces

    @staticmethod
    def validate_body_references(body: str, rows: Sequence[AnnexureRow]) -> list[dict[str, str]]:
        declared = {row.annexure.casefold() for row in rows}
        cited = {match.group(0).casefold() for match in re.finditer(r"Annexure\s+[A-Z]+-\d+", body, re.IGNORECASE)}
        issues: list[dict[str, str]] = []
        for reference in sorted(cited - declared):
            issues.append({"error_code": "ANNEXURE_REFERENCE_MISSING", "severity": "error",
                           "location": reference, "message": "Body cites an annexure that is not registered.",
                           "suggested_fix": "Attach and register it or remove the reference."})
        for reference in sorted(declared - cited):
            issues.append({"error_code": "ANNEXURE_ORPHAN", "severity": "warning",
                           "location": reference, "message": "Registered annexure is not cited in the body.",
                           "suggested_fix": "Add a relevant body reference or remove it from the filing bundle."})
        return issues


class MatterConsistencyEngine:
    _MATERIAL_SLOTS = frozenset({"amount", "incident_date", "transaction_id", "person", "phone", "email"})

    def check(self, matter: CaseFacts) -> list[dict[str, Any]]:
        values: dict[str, dict[str, set[str]]] = {}
        for fact in matter.extracted_facts():
            if fact.slot not in self._MATERIAL_SLOTS:
                continue
            values.setdefault(fact.slot, {}).setdefault(fact.normalized, set()).add(fact.source)
        conflicts: list[dict[str, Any]] = []
        for slot, by_value in values.items():
            if len(by_value) > 1:
                conflicts.append({"slot": slot, "values": list(by_value),
                                  "sources": {value: sorted(sources) for value, sources in by_value.items()},
                                  "severity": "error", "resolution_required": True})
        return conflicts


@dataclass(frozen=True)
class JurisdictionRulePack:
    profile_id: str
    forum_type: str
    required_fields: tuple[str, ...]
    formatting_profile: str
    require_verification: bool
    require_affidavit: bool
    effective_from: str
    verification_status: str


class JurisdictionRulePackRegistry:
    def __init__(self) -> None:
        self._packs = {
            "generic_india": JurisdictionRulePack(
                "generic_india", "", ("jurisdiction_basis",), "generic_legal_document",
                False, False, "", "review_required",
            ),
            "delhi_district_courts": JurisdictionRulePack(
                "delhi_district_courts", "district_court",
                ("court_name", "jurisdiction_basis", "suit_valuation", "court_fee"),
                "delhi_courts_2022", True, True, "2022-11-01", "source_review_required",
            ),
        }

    def get(self, profile_id: str) -> JurisdictionRulePack | None:
        return self._packs.get(profile_id)

    def validate(self, profile_id: str, fields: Mapping[str, str]) -> list[str]:
        pack = self.get(profile_id)
        if pack is None:
            return ["Unknown jurisdiction profile; advocate review required."]
        return [key for key in pack.required_fields if not str(fields.get(key, "")).strip()]


@dataclass(frozen=True)
class DeadlineReview:
    rule_id: str
    anchor_date: str
    due_date: str
    status: str
    legal_basis: str
    verification_status: str


class LimitationDeadlineEngine:
    """Computes only explicitly configured, verified/advisory rules."""

    def calculate(
        self, *, rule_id: str, anchor_date: date | None, days: int | None,
        legal_basis: str = "", verification_status: str = "review_required", today: date | None = None,
    ) -> DeadlineReview:
        if anchor_date is None or days is None or not legal_basis:
            return DeadlineReview(rule_id, anchor_date.isoformat() if anchor_date else "", "", "REVIEW_REQUIRED",
                                  legal_basis, verification_status)
        due = anchor_date + timedelta(days=days)
        current = today or clock.today()
        status = "EXPIRED" if due < current else "DUE_TODAY" if due == current else "OPEN"
        if verification_status != "verified":
            status = "REVIEW_REQUIRED"
        return DeadlineReview(rule_id, anchor_date.isoformat(), due.isoformat(), status, legal_basis,
                              verification_status)


@dataclass(frozen=True)
class RedlineRiskReport:
    unified_diff: str
    changed_protected_values: tuple[str, ...]
    risk_flags: tuple[dict[str, str], ...]
    status: str


class RedlineRiskAnalyzer:
    _RISK_TERMS: ClassVar[dict[str, str]] = {
        "indemn": "Indemnity allocation changed.", "terminate": "Termination language changed.",
        "jurisdiction": "Jurisdiction language changed.", "arbitration": "Dispute-resolution language changed.",
        "liability": "Liability allocation changed.", "penalty": "Penalty language changed.",
    }

    def compare(self, original: str, revised: str) -> RedlineRiskReport:
        diff = "\n".join(difflib.unified_diff(original.splitlines(), revised.splitlines(),
                                                fromfile="original", tofile="revised", lineterm=""))
        original_values = set(re.findall(r"(?:₹|Rs\.?)\s*[\d,]+|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b", original, re.IGNORECASE))
        revised_values = set(re.findall(r"(?:₹|Rs\.?)\s*[\d,]+|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b", revised, re.IGNORECASE))
        changed = tuple(sorted(original_values ^ revised_values))
        flags: list[dict[str, str]] = []
        lowered_change = " ".join(line[1:].casefold() for line in diff.splitlines() if line.startswith(("+", "-")))
        for token, message in self._RISK_TERMS.items():
            if token in lowered_change:
                flags.append({"risk": token, "severity": "warning", "message": message})
        if changed:
            flags.append({"risk": "protected_value", "severity": "error",
                          "message": "An amount or date changed between versions."})
        return RedlineRiskReport(diff, changed, tuple(flags), "REVIEW_REQUIRED" if flags else "PASS")


@dataclass(frozen=True)
class FilingBundleManifest:
    matter_id: str
    documents: tuple[dict[str, Any], ...]
    page_numbering: str
    bookmarks: tuple[str, ...]
    validation_status: str
    blockers: tuple[str, ...]


class FilingBundleGenerator:
    """Builds a deterministic manifest; physical PDF merge follows approval."""

    _ORDER: ClassVar[tuple[str, ...]] = (
        "index", "memo_of_parties", "synopsis", "list_of_dates", "main_pleading",
        "interim_application", "affidavit", "vakalatnama", "annexure_index", "annexures",
    )

    def manifest(
        self, matter: CaseFacts, documents: Sequence[Mapping[str, Any]], *,
        conflicts: Sequence[Mapping[str, Any]] = (), validation_errors: Sequence[str] = (),
    ) -> FilingBundleManifest:
        indexed = sorted(documents, key=lambda item: self._ORDER.index(str(item.get("kind")))
                         if str(item.get("kind")) in self._ORDER else len(self._ORDER))
        blockers = [*validation_errors]
        if conflicts:
            blockers.append("Unresolved factual conflicts exist.")
        for item in indexed:
            if not item.get("path"):
                blockers.append(f"Missing file for bundle item: {item.get('kind', 'unknown')}")
        return FilingBundleManifest(
            matter.matter_id, tuple(dict(item) for item in indexed), "continuous_page_x_of_y",
            tuple(str(item.get("title") or item.get("kind")) for item in indexed),
            "BLOCKED" if blockers else "READY_FOR_ADVOCATE_REVIEW", tuple(dict.fromkeys(blockers)),
        )
