"""Contracts for document-type review and two-document comparison.

`ReviewStatus` is the important type here. Three outcomes, never two:

* `present` — found, and quoted;
* `missing` — genuinely not in a document that extracted cleanly;
* `unable_to_determine` — the extraction is too thin to support either claim.

Collapsing the third into `missing` is the specific error this schema exists
to prevent: telling somebody their agreement has no termination clause, when
in fact the scan failed, is a confident falsehood about a document they are
about to sign.
"""

from typing import Literal

from pydantic import BaseModel, Field

DocumentType = Literal[
    "rental_agreement",
    "employment_agreement",
    "nda",
    "service_agreement",
    "sale_agreement",
    "partnership_deed",
    "legal_notice",
    "affidavit",
    "complaint",
    "power_of_attorney",
    "unknown",
]

ReviewStatus = Literal["present", "missing", "not_applicable", "unable_to_determine"]

RiskLevel = Literal["low", "medium", "high", "unknown"]

#: What kind of difference a comparison found.
ChangeType = Literal["added", "removed", "changed", "unchanged"]


class ChecklistItem(BaseModel):
    key: str
    label: str
    status: ReviewStatus
    #: True when its absence is a risk, not merely a gap.
    critical: bool = False
    #: The span that evidences a `present` finding. Empty otherwise -- an
    #: absence has no quote, and inventing one would be absurd.
    source_text: str = ""
    source_page: int | None = None
    #: What to ask about this, in plain words. Never phrased as a conclusion.
    question: str = ""


class DocumentReviewResult(BaseModel):
    document_id: str
    filename: str = ""
    detected_type: DocumentType = "unknown"
    type_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    #: `thin` means no absence claim was made about anything.
    extraction_quality: Literal["good", "thin"] = "good"
    has_page_evidence: bool = False
    items: list[ChecklistItem] = Field(default_factory=list)
    #: Wording that is a risk in itself, wherever it appears.
    risky_clauses: list[ChecklistItem] = Field(default_factory=list)
    risk_level: RiskLevel = "unknown"
    suggested_questions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def by_status(self, status: ReviewStatus) -> list[ChecklistItem]:
        return [item for item in self.items if item.status == status]


class ComparisonItem(BaseModel):
    """One difference between two documents, with both sides quoted."""

    aspect: str
    label: str
    change_type: ChangeType
    old_value: str = ""
    new_value: str = ""
    old_page: int | None = None
    new_page: int | None = None
    #: Whether the change is worth a second look, and why.
    risk_note: str = ""
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class ComparisonResult(BaseModel):
    old_document_id: str
    new_document_id: str
    old_filename: str = ""
    new_filename: str = ""
    executive_summary: str = ""
    items: list[ComparisonItem] = Field(default_factory=list)
    #: Clauses present in one document and absent from the other.
    added_clauses: list[str] = Field(default_factory=list)
    removed_clauses: list[str] = Field(default_factory=list)
    #: Anything that stopped the comparison being complete -- a scan that did
    #: not extract, an aspect neither document states. Reported, because a
    #: comparison that silently skipped half a document is worse than none.
    unresolved: list[str] = Field(default_factory=list)
    has_page_evidence: bool = False
