"""Phase 2 item 2: the evidence and annexure organizer.

Indian legal documents attach evidence as numbered annexures and refer to them
by that number in the body ("...as per the bank statement at Annexure A-1").
Before this, uploaded evidence and generated drafts were two unconnected
things: the draft's "Annexures" section was whatever free text the user had
typed into an `available_documents` field, so the numbering was absent, the
dates were unstated, and nothing connected a claim in the body to the document
that proves it.

This module turns a set of uploaded files into that table:

    Annexure A-1 | bank_receipt.pdf | 26 August 2026 | Proves the debit of
                 |                  |                | ₹45,000 | Payee name unclear

`missing` is as important as the rest. An annexure the user believes proves
the transfer but which carries no date, or no transaction reference, is a weak
exhibit -- and the moment to say so is while they can still find a better one,
not after filing.

Numbering is assigned in upload order and is STABLE: once an item has a
number, re-running the organizer keeps it, because the number is already
quoted in a draft the user may have downloaded.
"""

import re
from dataclasses import dataclass, field

from app.casefile.facts import (
    AMOUNT,
    EMAIL,
    INCIDENT_DATE,
    PERSON,
    PHONE,
    TRANSACTION_ID,
    ExtractedFact,
    extract_facts,
)

# Annexures are lettered by series so a case can group them ("A" for financial
# records, "B" for correspondence) the way a filed paperbook does. A single
# default series keeps the common case simple.
DEFAULT_SERIES = "A"

# What a document of each kind is expected to establish, and which facts it
# must carry to actually do so. Drives both `relevance` and `missing`.
_KIND_RULES: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    (
        "bank_record",
        ("statement", "receipt", "passbook", "transaction", "debit", "credit", "utr", "neft", "imps", "upi"),
        "Establishes the disputed transaction and the amount debited",
        (AMOUNT, INCIDENT_DATE, TRANSACTION_ID),
    ),
    (
        "screenshot",
        ("screenshot", "screen shot", "snapshot", "capture", ".png", ".jpg", ".jpeg"),
        "Shows what the applicant saw at the time of the incident",
        (INCIDENT_DATE,),
    ),
    (
        "chat_record",
        ("chat", "whatsapp", "telegram", "message", "sms", "conversation"),
        "Records the exchange with the other party",
        (INCIDENT_DATE, PERSON),
    ),
    (
        "call_record",
        ("call", "cdr", "call log", "call record", "phone log"),
        "Shows the calls exchanged with the other party",
        (INCIDENT_DATE, PHONE),
    ),
    (
        "email",
        ("email", "e-mail", "mail", ".eml", "correspondence"),
        "Records written correspondence with the other party",
        (INCIDENT_DATE, EMAIL),
    ),
    (
        "invoice",
        ("invoice", "bill", "order", "purchase", "tax invoice"),
        "Establishes the purchase and the amount paid",
        (AMOUNT, INCIDENT_DATE),
    ),
    (
        "identity",
        ("aadhaar", "pan", "passport", "voter", "licence", "license", "id proof"),
        "Establishes the applicant's identity",
        (),
    ),
    (
        "official_record",
        ("fir", "acknowledgement", "acknowledgment", "receipt no", "complaint copy", "ticket"),
        "Records that the matter was already reported to an authority",
        (INCIDENT_DATE,),
    ),
)
_FALLBACK_RELEVANCE = "Supporting document supplied by the applicant"

_MISSING_LABELS: dict[str, str] = {
    AMOUNT: "no amount is visible in this document",
    INCIDENT_DATE: "no date is visible in this document",
    TRANSACTION_ID: "no transaction/UTR reference is visible in this document",
    PERSON: "the other party is not named in this document",
    PHONE: "no phone number is visible in this document",
    EMAIL: "no email address is visible in this document",
}


@dataclass
class EvidenceItem:
    """One uploaded file, before it is numbered."""

    evidence_id: str
    document_name: str
    text: str = ""
    description: str = ""
    uploaded_at: str = ""
    annexure: str = ""  # preserved when already assigned


@dataclass(frozen=True)
class AnnexureRow:
    annexure: str
    evidence_id: str
    document_name: str
    document_date: str
    relevance: str
    missing: tuple[str, ...]
    facts: tuple[ExtractedFact, ...] = field(default=())

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict[str, object]:
        return {
            "annexure": self.annexure,
            "evidence_id": self.evidence_id,
            "document_name": self.document_name,
            "document_date": self.document_date,
            "relevance": self.relevance,
            "missing": list(self.missing),
            "is_complete": self.is_complete,
            "extracted_facts": [
                {"slot": fact.slot, "label": fact.label, "value": fact.value} for fact in self.facts
            ],
        }


def classify_document(name: str, text: str, description: str = "") -> tuple[str, str, tuple[str, ...]]:
    """`(kind, relevance, required_facts)` for one document.

    Matched on the filename, the user's own description, and the first part of
    the extracted text -- a file called `IMG_2043.jpg` says nothing, but its
    text ("UPI transaction successful") or the user's note ("payment
    screenshot") usually does.
    """
    haystack = f"{name} {description} {text[:1500]}".lower()
    for kind, keywords, relevance, required in _KIND_RULES:
        if any(keyword in haystack for keyword in keywords):
            return kind, relevance, required
    return "other", _FALLBACK_RELEVANCE, ()


def _next_number(existing: list[str], series: str) -> int:
    used = [
        int(match.group(1))
        for annexure in existing
        if (match := re.search(rf"{re.escape(series)}-(\d+)$", annexure or ""))
    ]
    return max(used, default=0) + 1


def build_evidence_table(items: list[EvidenceItem], *, series: str = DEFAULT_SERIES) -> list[AnnexureRow]:
    """The numbered evidence table.

    An item that already carries an annexure number keeps it. That is not a
    detail: the number is quoted in the body of drafts the user may already
    have downloaded or filed, so renumbering on a later upload would break
    every one of those references.
    """
    assigned = [item.annexure for item in items if item.annexure]
    rows: list[AnnexureRow] = []
    for item in items:
        annexure = item.annexure
        if not annexure:
            annexure = f"Annexure {series}-{_next_number(assigned, series)}"
            assigned.append(annexure)
        facts = extract_facts(item.text, annexure)
        _kind, relevance, required = classify_document(item.document_name, item.text, item.description)
        dates = [fact for fact in facts if fact.slot == INCIDENT_DATE]
        present = {fact.slot for fact in facts}
        missing = tuple(
            _MISSING_LABELS[slot] for slot in required if slot not in present and slot in _MISSING_LABELS
        )
        if not item.text.strip():
            # An image with no OCR text behind it is not useless -- it may be
            # a perfectly good exhibit -- but nothing in it can be read, so
            # say that instead of reporting every required fact as missing.
            missing = ("no readable text could be extracted; check this document manually before filing",)
        rows.append(AnnexureRow(
            annexure=annexure,
            evidence_id=item.evidence_id,
            document_name=item.document_name,
            document_date=dates[0].normalized if dates else "",
            relevance=item.description.strip() or relevance,
            missing=missing,
            facts=tuple(facts),
        ))
    return rows


def render_annexure_index(rows: list[AnnexureRow], *, language: str = "english") -> str:
    """The annexure index as it appears at the end of a document.

    Deliberately plain text rather than a table: it is interpolated into a
    draft's "Annexures" section and rendered by four different exporters
    (PDF/DOCX/TXT/RTF), only one of which has table support.
    """
    if not rows:
        return ""
    lines: list[str] = []
    for row in rows:
        dated = f" (dated {_pretty(row.document_date)})" if row.document_date else ""
        lines.append(f"{row.annexure}: {row.document_name}{dated} — {row.relevance}")
    return "\n".join(lines)


def _pretty(iso: str) -> str:
    from app.casefile.facts import parse_date

    parsed = parse_date(iso)
    return parsed.strftime("%d %B %Y") if parsed else iso


def annexure_reference_map(rows: list[AnnexureRow]) -> dict[str, str]:
    """`{fact value -> annexure number}` so a draft's body can cite the exhibit
    that proves each figure. Only unambiguous facts are mapped: a value that
    appears in two different annexures is left out rather than cited to
    whichever came first."""
    occurrences: dict[str, set[str]] = {}
    for row in rows:
        for fact in row.facts:
            if fact.slot in (AMOUNT, TRANSACTION_ID, INCIDENT_DATE):
                occurrences.setdefault(fact.normalized, set()).add(row.annexure)
    return {value: next(iter(annexures)) for value, annexures in occurrences.items() if len(annexures) == 1}


def evidence_checklist(rows: list[AnnexureRow], expected_kinds: tuple[str, ...]) -> list[dict[str, object]]:
    """What the user has, and what a case of this type usually also needs.

    `expected_kinds` comes from the workflow (the cyber-fraud workflow knows a
    UPI-fraud case wants a bank record, a screenshot and a chat/call record).
    Reported as "still needed", never as an error -- the user may have good
    reasons not to have it, and a checklist that reads as a rejection stops
    being read.
    """
    present_kinds = {
        classify_document(row.document_name, "", row.relevance)[0] for row in rows
    }
    checklist: list[dict[str, object]] = []
    for row in rows:
        checklist.append({
            "annexure": row.annexure,
            "item": row.document_name,
            "status": "provided" if row.is_complete else "provided, needs review",
            "notes": "; ".join(row.missing),
        })
    for kind in expected_kinds:
        if kind in present_kinds:
            continue
        label = next((relevance for name, _, relevance, _ in _KIND_RULES if name == kind), kind)
        checklist.append({
            "annexure": "",
            "item": kind.replace("_", " ").title(),
            "status": "still needed",
            "notes": label,
        })
    return checklist
