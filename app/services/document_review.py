"""Document-type review checklists.

What this is: for ten common document types, the clauses a competent version
of that document normally contains, each with the words that evidence its
presence. What it is NOT: an opinion on whether the document is any good.

Three outcomes, kept strictly apart, because collapsing them is the whole
failure mode this module exists to avoid:

* **present** — wording matching the clause was found, and the span is quoted;
* **missing** — the document extracted cleanly and none of the clause's
  wording appears anywhere in it;
* **unable to determine** — the extraction is too thin to support either
  claim (a scanned page that OCR'd to noise, a document with almost no text).

"Missing" is a claim about the document. It is only ever made when the
extraction is good enough to support it, which is why `_extraction_quality`
gates it: telling somebody their agreement has no termination clause, when
really the OCR failed, is worse than saying nothing at all.

Nothing here is a legal conclusion. The output is a list of things to ask
about, which is what a person without a lawyer actually needs.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from app.schemas.document_review import (
    ChecklistItem,
    DocumentReviewResult,
    DocumentType,
    ReviewStatus,
)
from app.services.document_insight import LoadedDocument

# Below this many characters of extracted text, no absence claim is made
# about anything -- there is not enough document to be absent FROM.
_MIN_TEXT_FOR_ABSENCE = 600
# A document whose text is mostly non-alphabetic is OCR noise, not prose.
_MIN_ALPHA_RATIO = 0.55


@dataclass(frozen=True)
class Clause:
    """One thing a checklist looks for."""

    key: str
    label: str
    #: Wording that evidences the clause. Any one match is enough.
    cues: tuple[str, ...]
    #: True when its absence is a risk worth raising, not merely a gap.
    critical: bool = False
    #: What to ask about it, in plain words.
    question: str = ""
    #: Document types this clause does not apply to at all.
    not_applicable_to: frozenset[str] = field(default_factory=frozenset)


# Wording that identifies the document type itself.
_TYPE_CUES: tuple[tuple[DocumentType, tuple[str, ...]], ...] = (
    ("rental_agreement", ("lease deed", "rent agreement", "rental agreement", "leave and licence", "lessor", "lessee", "किरायानामा")),
    ("employment_agreement", ("employment agreement", "appointment letter", "offer of employment", "employee shall", "ctc", "probation")),
    ("nda", ("non-disclosure", "confidentiality agreement", "nda", "disclosing party", "receiving party")),
    ("service_agreement", ("service agreement", "services agreement", "statement of work", "service provider", "scope of services")),
    ("sale_agreement", ("agreement to sell", "sale deed", "sale agreement", "vendor and the purchaser", "conveyance")),
    ("partnership_deed", ("partnership deed", "deed of partnership", "profit sharing ratio", "partners hereby")),
    ("legal_notice", ("legal notice", "notice under section", "my client", "through my client", "विधिक सूचना")),
    ("affidavit", ("affidavit", "solemnly affirm", "deponent", "verified at", "शपथ पत्र")),
    ("complaint", ("complaint under", "complainant", "before the", "prayer", "परिवाद")),
    ("power_of_attorney", ("power of attorney", "attorney holder", "constitute and appoint", "मुख्तारनामा")),
)

_COMMON_CONTRACT_CLAUSES: tuple[Clause, ...] = (
    Clause("parties", "Parties and their details", ("between", "party of the first part", "hereinafter referred"), critical=True,
           question="Are both parties named in full, with addresses that match their identity documents?"),
    Clause("term", "Term / duration", ("term of", "commencing from", "period of", "duration", "shall remain in force"), critical=True,
           question="When does this start and end, and what happens when it ends?"),
    Clause("payment", "Payment terms", ("payment", "consideration", "fees", "remuneration", "shall pay", "rent of"), critical=True,
           question="Exactly how much is payable, by when, and how is it to be paid?"),
    Clause("termination", "Termination", ("terminate", "termination", "notice period", "cancel this agreement"), critical=True,
           question="Who can end this, on how much notice, and what happens to money already paid?"),
    Clause("dispute_resolution", "Dispute resolution", ("arbitration", "dispute", "mediation", "conciliation"),
           question="If there is a disagreement, where does it get decided and who pays for that?"),
    Clause("jurisdiction", "Governing law and jurisdiction", ("jurisdiction", "courts at", "governed by the laws"), critical=True,
           question="Which city's courts have jurisdiction? Travelling to another state to enforce this is a real cost."),
    Clause("indemnity", "Indemnity", ("indemnify", "indemnity", "hold harmless"),
           question="What are you agreeing to cover if something goes wrong?"),
    Clause("liability", "Limitation of liability", ("limitation of liability", "shall not be liable", "liability shall be limited"),
           question="Is anyone's liability capped, and is the cap one-sided?"),
    Clause("confidentiality", "Confidentiality", ("confidential", "confidentiality", "non-disclosure"),
           question="What information is covered, and for how long after this ends?"),
    Clause("force_majeure", "Force majeure", ("force majeure", "act of god", "beyond the reasonable control"),
           question="What happens if neither side can perform because of something outside their control?"),
    Clause("signatures", "Signatures and execution", ("signature", "in witness whereof", "signed and delivered", "witness"), critical=True,
           question="Is it signed and dated by everyone, and witnessed where that is required?"),
)


def _contract_checklist(*extra: Clause, drop: frozenset[str] = frozenset()) -> tuple[Clause, ...]:
    return tuple(clause for clause in _COMMON_CONTRACT_CLAUSES if clause.key not in drop) + extra


_CHECKLISTS: dict[DocumentType, tuple[Clause, ...]] = {
    "rental_agreement": _contract_checklist(
        Clause("premises", "Description of the premises", ("premises", "situated at", "flat no", "house no", "schedule of property"), critical=True,
               question="Is the property described precisely enough to identify it — address, floor, area?"),
        Clause("security_deposit", "Security deposit and refund", ("security deposit", "advance", "refundable", "deposit of"), critical=True,
               question="How much is the deposit, when is it refunded, and what may be deducted from it?"),
        Clause("maintenance", "Maintenance and repairs", ("maintenance", "repairs", "wear and tear"),
               question="Who pays for what repairs?"),
        Clause("rent_escalation", "Rent escalation", ("escalation", "increase in rent", "enhanced rent"),
               question="Does the rent rise, by how much, and how often?"),
        Clause("lock_in", "Lock-in period", ("lock-in", "lock in period", "minimum period"),
               question="Are you locked in for a minimum period, and what does leaving early cost?"),
        Clause("entry_rights", "Landlord's right of entry", ("right to enter", "inspect the premises", "access to the premises"),
               question="Can the landlord enter without notice? That is worth negotiating."),
    ),
    "employment_agreement": _contract_checklist(
        Clause("role", "Role and duties", ("designation", "position of", "duties", "responsibilities"), critical=True,
               question="Is the role and what you are expected to do written down?"),
        Clause("salary", "Salary and benefits", ("salary", "ctc", "gross", "in-hand", "allowance"), critical=True,
               question="Is the full salary structure set out, including deductions?"),
        Clause("probation", "Probation", ("probation", "probationary period", "confirmation"),
               question="How long is probation and what notice applies during it?"),
        Clause("non_compete", "Non-compete / restrictive covenants", ("non-compete", "non compete", "shall not engage", "restraint of trade"),
               question="Does this restrict where you can work afterwards? Restraints on employment after service are often unenforceable in India — worth asking about."),
        Clause("notice_period", "Notice period", ("notice period", "months' notice", "days' notice"), critical=True,
               question="How much notice must each side give, and is it the same both ways?"),
        Clause("ip_assignment", "Intellectual property", ("intellectual property", "work product", "inventions", "assign all rights"),
               question="Does everything you create belong to the employer, including work done on your own time?"),
    ),
    "nda": _contract_checklist(
        Clause("definition", "Definition of confidential information", ("confidential information means", "defined as", "shall include"), critical=True,
               question="Is it clear what counts as confidential, and what does not?"),
        Clause("exclusions", "Exclusions from confidentiality", ("publicly available", "already known", "independently developed", "exclusions"),
               question="Are the standard carve-outs there — public information, what you already knew?"),
        Clause("duration_of_obligation", "Duration of the obligation", ("survive", "for a period of", "years from"), critical=True,
               question="How long does the obligation last after the relationship ends?"),
        Clause("return_of_information", "Return or destruction of information", ("return", "destroy", "upon termination"),
               question="What must you do with the information when this ends?"),
        Clause("mutuality", "Mutuality", ("mutual", "each party", "both parties"),
               question="Does this bind both sides, or only you?"),
        drop=frozenset({"payment"}),
    ),
    "service_agreement": _contract_checklist(
        Clause("scope", "Scope of services", ("scope of services", "services to be provided", "deliverables"), critical=True,
               question="Is the work described specifically enough to tell whether it has been done?"),
        Clause("timelines", "Timelines and milestones", ("milestone", "timeline", "delivery date", "completion"),
               question="Are there dates, and what happens if they slip?"),
        Clause("acceptance", "Acceptance criteria", ("acceptance", "approval of the deliverables", "sign off"),
               question="Who decides the work is acceptable, and on what basis?"),
        Clause("ip_ownership", "Intellectual property ownership", ("intellectual property", "ownership of the deliverables", "work for hire"),
               question="Who owns what is produced?"),
        Clause("penalties", "Penalties and service levels", ("penalty", "liquidated damages", "service level"),
               question="Are there penalties for late or poor performance, and do they run both ways?"),
    ),
    "sale_agreement": _contract_checklist(
        Clause("property_schedule", "Schedule of property", ("schedule", "described in", "bounded by", "survey no", "plot no"), critical=True,
               question="Is the property described precisely, with boundaries and measurements?"),
        Clause("title", "Title and encumbrances", ("clear title", "marketable title", "free from encumbrance", "encumbrance"), critical=True,
               question="Does the seller warrant clear title? Has an encumbrance certificate been checked?"),
        Clause("possession", "Handover of possession", ("possession", "handed over", "vacant possession"), critical=True,
               question="When is possession given, and in what condition?"),
        Clause("consideration_schedule", "Payment schedule", ("advance", "balance consideration", "on or before", "instalment"), critical=True,
               question="What is paid when, and what happens if a payment is late?"),
        Clause("default", "Consequences of default", ("default", "forfeit", "specific performance"),
               question="What happens if either side backs out?"),
        Clause("registration", "Registration", ("registration", "sub-registrar", "stamp duty"), critical=True,
               question="Who pays stamp duty and registration, and by when must it be registered?"),
    ),
    "partnership_deed": _contract_checklist(
        Clause("capital", "Capital contribution", ("capital", "contributed", "invested by"), critical=True,
               question="Who is putting in how much, and is it money or something else?"),
        Clause("profit_sharing", "Profit and loss sharing", ("profit sharing", "profits and losses", "ratio"), critical=True,
               question="In what ratio are profits AND losses shared?"),
        Clause("management", "Management and authority", ("management", "authority", "managing partner", "decisions"),
               question="Who can bind the firm, and what needs everyone's agreement?"),
        Clause("admission_retirement", "Admission and retirement of partners", ("admission", "retirement", "new partner", "outgoing partner"),
               question="What happens when someone joins or leaves?"),
        Clause("dissolution", "Dissolution", ("dissolution", "winding up", "dissolve the firm"), critical=True,
               question="How is the firm wound up, and how are assets divided?"),
        Clause("accounts", "Accounts and audit", ("books of account", "audit", "financial year"),
               question="Who keeps the books, and can every partner see them?"),
        drop=frozenset({"payment"}),
    ),
    "legal_notice": (
        Clause("sender_details", "Sender and advocate details", ("my client", "advocate", "chamber", "enrolment"), critical=True,
               question="Are the sender's and the advocate's details complete?"),
        Clause("recipient_details", "Recipient details", ("to,", "addressee", "resident of", "having its office"), critical=True,
               question="Is the recipient named and addressed correctly? A notice to the wrong address is often the whole defence."),
        Clause("facts", "Statement of facts", ("facts", "briefly stated", "that on"), critical=True,
               question="Are the facts set out in date order?"),
        Clause("legal_basis", "Legal basis", ("section", "act", "under the provisions"), critical=True,
               question="Which provision is relied on?"),
        Clause("demand", "Demand and time limit", ("hereby call upon", "within", "days", "failing which"), critical=True,
               question="Is there a clear demand and a clear deadline to meet it?"),
        Clause("consequence", "Consequence of non-compliance", ("failing which", "legal proceedings", "at your risk as to cost"),
               question="Does it say what happens if the demand is not met?"),
        Clause("signature", "Signature and date", ("signature", "dated", "advocate"), critical=True,
               question="Is it signed and dated?"),
    ),
    "affidavit": (
        Clause("deponent", "Deponent details", ("deponent", "aged about", "resident of", "s/o", "d/o"), critical=True,
               question="Are the deponent's name, age, parentage and address stated?"),
        Clause("paragraphs", "Numbered paragraphs", ("1.", "2.", "that i"), critical=True,
               question="Are the statements numbered, one fact per paragraph?"),
        Clause("verification", "Verification clause", ("verified", "verification", "true to my knowledge", "believed to be true"), critical=True,
               question="Is there a verification clause distinguishing what you know from what you believe?"),
        Clause("oath", "Oath / solemn affirmation", ("solemnly affirm", "on oath", "state on oath"), critical=True,
               question="Does it record the oath or affirmation?"),
        Clause("attestation", "Attestation", ("before me", "notary", "oath commissioner", "sworn before"), critical=True,
               question="Is there space for the notary or oath commissioner to attest it?"),
        Clause("place_date", "Place and date", ("place:", "date:", "verified at"), critical=True,
               question="Are the place and date of verification filled in?"),
    ),
    "complaint": (
        Clause("forum", "Forum and cause title", ("before the", "in the court of", "commission at"), critical=True,
               question="Is the correct forum named at the top?"),
        Clause("parties", "Parties", ("complainant", "opposite party", "respondent", "versus"), critical=True,
               question="Are both sides named with full addresses?"),
        Clause("jurisdiction_facts", "Facts establishing jurisdiction", ("jurisdiction", "cause of action arose", "resides within"), critical=True,
               question="Does it explain why THIS forum can hear it?"),
        Clause("facts", "Facts in chronological order", ("that on", "thereafter", "subsequently"), critical=True,
               question="Are the facts in date order, with documents referenced?"),
        Clause("limitation", "Limitation", ("within the period of limitation", "limitation", "cause of action arose on"),
               question="Is the complaint within time? Limitation is the first thing the other side will check."),
        Clause("relief", "Relief / prayer", ("prayer", "pray", "relief", "direct the opposite party"), critical=True,
               question="Is it clear exactly what you are asking for?"),
        Clause("annexures", "List of documents", ("annexure", "list of documents", "exhibit"), critical=True,
               question="Are the supporting documents listed and attached?"),
        Clause("verification", "Verification", ("verified", "verification", "true to the best"), critical=True,
               question="Is the complaint verified and signed?"),
    ),
    "power_of_attorney": (
        Clause("principal", "Principal's details", ("i,", "principal", "executant", "aged about", "resident of"), critical=True,
               question="Is the person granting the power fully identified?"),
        Clause("attorney", "Attorney's details", ("attorney", "constitute and appoint", "nominate"), critical=True,
               question="Is the person receiving the power fully identified?"),
        Clause("powers", "Specific powers granted", ("power to", "authorised to", "on my behalf"), critical=True,
               question="Is each power listed specifically? A general power is far broader than most people intend."),
        Clause("property_scope", "Property or matter covered", ("in respect of", "property", "schedule"), critical=True,
               question="What exactly does this cover?"),
        Clause("duration_revocation", "Duration and revocation", ("shall remain in force", "revoke", "until"), critical=True,
               question="When does it end, and how do you cancel it?"),
        Clause("registration_attestation", "Registration / attestation", ("registered", "sub-registrar", "notary", "witness"), critical=True,
               question="Does it need registration or notarisation to be effective for this purpose?"),
    ),
}

# Wording that is a risk in itself, wherever it appears.
_RISK_CUES: tuple[tuple[str, str], ...] = (
    ("sole discretion", "One side decides on its own — check what happens when you disagree."),
    ("without notice", "Something can be done to you without warning."),
    ("without prior notice", "Something can be done to you without warning."),
    ("non-refundable", "Money paid cannot be recovered, whatever the reason."),
    ("shall not be liable", "Liability is being excluded — check how far that goes."),
    ("irrevocable", "This cannot be withdrawn once given."),
    ("indemnify and hold harmless", "You may be covering the other side's losses, including their legal costs."),
    ("liquidated damages", "A fixed penalty applies — check the amount is proportionate."),
    ("automatically renew", "This renews itself unless you actively stop it."),
    ("waives all rights", "Rights are being given up wholesale."),
    ("time is of the essence", "Any delay, however small, may be a breach."),
    ("exclusive jurisdiction", "Disputes go to one named place — check where."),
    ("forfeit", "Money or rights can be lost entirely."),
)


def detect_type(text: str) -> tuple[DocumentType, float]:
    """The document's type and how confidently it was identified.

    Returns `("unknown", 0.0)` rather than a best guess when nothing scores:
    running an employment checklist over a sale deed produces a page of
    confident nonsense.
    """
    lowered = (text or "").lower()
    scores: list[tuple[DocumentType, int]] = []
    for document_type, cues in _TYPE_CUES:
        hits = sum(1 for cue in cues if cue in lowered)
        if hits:
            scores.append((document_type, hits))
    if not scores:
        return "unknown", 0.0
    scores.sort(key=lambda pair: pair[1], reverse=True)
    best, hits = scores[0]
    runner_up = scores[1][1] if len(scores) > 1 else 0
    # Confidence reflects the MARGIN, not the raw hit count: two types
    # scoring equally means the document reads like both.
    confidence = min(0.95, 0.45 + 0.15 * (hits - runner_up) + 0.05 * hits)
    return best, round(confidence, 2)


def _extraction_quality(text: str) -> Literal["good", "thin"]:
    stripped = (text or "").strip()
    if len(stripped) < _MIN_TEXT_FOR_ABSENCE:
        return "thin"
    alphabetic = sum(1 for character in stripped if character.isalpha() or character.isspace())
    return "good" if alphabetic / len(stripped) >= _MIN_ALPHA_RATIO else "thin"


def _find(document: LoadedDocument, cue: str) -> tuple[str, int | None] | None:
    """`(quoted span, page)` for the first chunk containing `cue`, or None."""
    needle = cue.lower()
    for chunk in document.chunks:
        lowered = chunk.text.lower()
        index = lowered.find(needle)
        if index >= 0:
            start = max(0, index - 60)
            end = min(len(chunk.text), index + len(cue) + 120)
            return re.sub(r"\s+", " ", chunk.text[start:end]).strip(), chunk.page_number
    return None


def review(document: LoadedDocument, *, declared_type: DocumentType | None = None) -> DocumentReviewResult:
    """Runs the checklist for this document's type.

    `declared_type` lets a caller override detection (the user saying "this is
    my rent agreement"), which is better evidence than any cue count.
    """
    text = document.text
    document_type, type_confidence = (
        (declared_type, 0.99) if declared_type else detect_type(text)
    )
    quality = _extraction_quality(text)
    checklist = _CHECKLISTS.get(document_type, ())

    items: list[ChecklistItem] = []
    for clause in checklist:
        hit = next(
            (found for found in (_find(document, cue) for cue in clause.cues) if found is not None),
            None,
        )
        if hit is not None:
            quote, page = hit
            status: ReviewStatus = "present"
        elif quality == "thin":
            # The honest answer. Saying "missing" here would be a claim the
            # extraction cannot support.
            quote, page, status = "", None, "unable_to_determine"
        else:
            quote, page, status = "", None, "missing"
        items.append(
            ChecklistItem(
                key=clause.key,
                label=clause.label,
                status=status,
                critical=clause.critical,
                source_text=quote,
                source_page=page,
                question=clause.question,
            )
        )

    risky: list[ChecklistItem] = []
    for cue, why in _RISK_CUES:
        hit = _find(document, cue)
        if hit is None:
            continue
        quote, page = hit
        risky.append(
            ChecklistItem(
                key=cue.replace(" ", "_"),
                label=f'"{cue}"',
                status="present",
                critical=True,
                source_text=quote,
                source_page=page,
                question=why,
            )
        )

    missing_critical = [item for item in items if item.status == "missing" and item.critical]
    undetermined = [item for item in items if item.status == "unable_to_determine"]
    risk_level = _risk_level(len(missing_critical), len(risky), quality)

    return DocumentReviewResult(
        document_id=document.document_id,
        filename=document.filename,
        detected_type=document_type,
        type_confidence=type_confidence,
        extraction_quality=quality,
        has_page_evidence=document.has_page_evidence,
        items=items,
        risky_clauses=risky,
        risk_level=risk_level,
        suggested_questions=[
            *(item.question for item in missing_critical if item.question),
            *(item.question for item in risky[:5] if item.question),
        ][:8],
        notes=_notes(document_type, quality, undetermined, document.has_page_evidence),
    )


def _risk_level(missing_critical: int, risky: int, quality: str) -> Literal["low", "medium", "high", "unknown"]:
    if quality == "thin":
        # Not "low". An unreadable document is not a safe one.
        return "unknown"
    score = missing_critical * 2 + risky
    if score >= 6:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _notes(
    document_type: DocumentType, quality: str, undetermined: list[ChecklistItem], has_pages: bool
) -> list[str]:
    notes: list[str] = []
    if document_type == "unknown":
        notes.append(
            "I could not tell what kind of document this is, so I have not run a checklist over it. "
            "Tell me what it is and I will."
        )
    if quality == "thin":
        notes.append(
            "Not enough text came out of this file for me to say what is or is not in it — it may be a "
            f"scan. {len(undetermined)} check(s) are marked as undetermined rather than missing."
        )
    if not has_pages:
        notes.append("This file carries no page numbers, so I cannot cite where each clause sits.")
    notes.append("These are points to raise with a lawyer, not legal conclusions.")
    return notes
