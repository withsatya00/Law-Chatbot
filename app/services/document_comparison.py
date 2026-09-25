"""Comparing two versions of an agreement.

The question this answers is the one people actually ask: "what changed in
the new one, and does any of it hurt me?" A raw text diff answers a different
question badly — it reports every reflowed line as a change and buries the
one clause that moved the liability.

So the comparison is ASPECT-BASED. For each of the fifteen aspects that
matter in a contract (parties, dates, payment, liability, indemnity,
termination, confidentiality, IP, jurisdiction, dispute resolution, notice
period, penalties, obligations, renewal, governing law), the relevant clause
is located in each document, quoted with its page, and compared. A clause in
one document and not the other is an addition or a removal; a clause in both
whose wording differs materially is a change.

Two safety properties:

* **Both documents are loaded through `document_insight.load_document`**,
  which applies the single ownership rule. A document belonging to somebody
  else raises before any of its text is read, so nothing about it — not its
  existence, not its filename — reaches the comparison.
* **Nothing is compared by filename.** Documents are addressed by id
  throughout; two files called `agreement.pdf` are not assumed to be related,
  and a file named `their-contract.pdf` grants no access to anything.
"""

import re

from app.schemas.document_review import ComparisonItem, ComparisonResult
from app.services.document_insight import LoadedDocument

# The aspects worth comparing, with the wording that locates each one and a
# note on why a change to it matters.
_ASPECTS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("parties", "Parties", ("between", "party of the first part", "hereinafter referred", "lessor", "lessee", "employer", "employee"),
     "A change of party changes who is bound and who you can enforce against."),
    ("dates", "Term and dates", ("term of", "commencing", "effective from", "shall remain in force", "expiry", "until"),
     "A shorter or longer term changes what you are committed to."),
    ("payment", "Payment", ("payment", "consideration", "fees", "salary", "ctc", "rent of", "shall pay", "instalment"),
     "Check the amount, the due date and who bears the taxes."),
    ("liability", "Liability", ("liability", "shall not be liable", "limitation of liability", "aggregate liability"),
     "A liability cap that moved, or became one-sided, is worth arguing about."),
    ("indemnity", "Indemnity", ("indemnify", "indemnity", "hold harmless"),
     "Indemnities decide who pays when a third party sues."),
    ("termination", "Termination", ("terminate", "termination", "cancel this agreement", "expiry of the term"),
     "Who can end it, on what grounds, and what survives."),
    ("notice_period", "Notice period", ("notice period", "days' notice", "months' notice", "prior notice of"),
     "A longer notice period on your side than theirs is a common asymmetry."),
    ("confidentiality", "Confidentiality", ("confidential", "confidentiality", "non-disclosure"),
     "What is covered and for how long after the end."),
    ("intellectual_property", "Intellectual property", ("intellectual property", "copyright", "work product", "inventions", "assign all rights"),
     "Who owns what is created, including anything made before this started."),
    ("jurisdiction", "Jurisdiction", ("jurisdiction", "courts at", "courts of"),
     "Moving jurisdiction to another city can make enforcement uneconomic."),
    ("dispute_resolution", "Dispute resolution", ("arbitration", "arbitrator", "mediation", "conciliation", "dispute shall be"),
     "Arbitration is faster but you pay for it, and you usually lose the right to appeal."),
    ("penalties", "Penalties", ("penalty", "liquidated damages", "forfeit", "interest at the rate"),
     "Check the amount is proportionate and applies to both sides."),
    ("obligations", "Obligations", ("shall provide", "shall deliver", "shall maintain", "undertakes to", "responsible for"),
     "New obligations are the most common quiet addition."),
    ("renewal", "Renewal", ("renew", "renewal", "extended for a further", "automatically"),
     "Automatic renewal binds you again unless you actively stop it."),
    ("governing_law", "Governing law", ("governed by the laws", "governing law", "in accordance with the laws"),
     "Which country's or state's law applies."),
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


_SENTENCE_SPLIT = re.compile(r"(?<=[.।])\s+|\n+")
# Periods that do not end a sentence. Without this, "a monthly rent of
# Rs. 25,000" splits after "Rs." and the quote stops one word before the
# amount -- so a rent rise from 25,000 to 32,000 compared as unchanged.
_ABBREVIATION = re.compile(r"\b(Rs|No|Nos|Mr|Mrs|Ms|Dr|Smt|Shri|Ltd|Pvt|Co|Sec|Art|vs|v)\.", re.IGNORECASE)
_DECIMAL = re.compile(r"(\d)\.(\d)")
_PROTECTED_DOT = "\x00"


def _sentences(text: str) -> list[str]:
    protected = _ABBREVIATION.sub(lambda match: match.group(1) + _PROTECTED_DOT, text or "")
    protected = _DECIMAL.sub(lambda match: f"{match.group(1)}{_PROTECTED_DOT}{match.group(2)}", protected)
    return [
        part.replace(_PROTECTED_DOT, ".").strip()
        for part in _SENTENCE_SPLIT.split(protected)
        if part.strip()
    ]


def _locate(document: LoadedDocument, cues: tuple[str, ...]) -> tuple[str, int | None] | None:
    """The clause matching any cue, quoted with its page, or None.

    The quote is the SENTENCE containing the cue, not a fixed character
    window around it. A window is sensitive to formatting: re-typesetting a
    document shifts every offset, so the two windows capture different
    partial words at their edges and three unchanged clauses report as
    changed. A sentence boundary does not move when the whitespace does.
    """
    for cue in cues:
        needle = cue.lower()
        for chunk in document.chunks:
            if needle not in chunk.text.lower():
                continue
            for sentence in _sentences(chunk.text):
                if needle in sentence.lower():
                    return re.sub(r"\s+", " ", sentence).strip()[:400], chunk.page_number
            # The cue is in the chunk but not in any single sentence (an
            # unpunctuated block); fall back to the whole chunk.
            return re.sub(r"\s+", " ", chunk.text).strip()[:400], chunk.page_number
    return None


def _materially_different(old: str, new: str) -> bool:
    """Whether two clause quotes differ beyond reformatting.

    Compared as TOKEN SEQUENCES after normalising away whitespace, case and
    punctuation, so a re-typeset document reports no changes — and then any
    residual difference counts, however small.

    A similarity threshold was tried first and was wrong for exactly the
    changes that matter: "courts at Pune" → "courts at Mumbai" and
    "Rs. 25,000" → "Rs. 32,000" are ~97% similar as strings and completely
    different as terms. Materiality here is not a matter of degree.
    """
    return _tokens(old) != _tokens(new)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalise(text))


def compare(old: LoadedDocument, new: LoadedDocument) -> ComparisonResult:
    """What changed between two documents, aspect by aspect."""
    items: list[ComparisonItem] = []
    added: list[str] = []
    removed: list[str] = []
    unresolved: list[str] = []

    for aspect, label, cues, risk_note in _ASPECTS:
        old_hit = _locate(old, cues)
        new_hit = _locate(new, cues)

        if old_hit is None and new_hit is None:
            # Neither document addresses it. That is not a change, and
            # reporting it as one would pad the result with noise -- but it
            # IS worth knowing when the aspect is one both should cover.
            continue
        if old_hit is None and new_hit is not None:
            quote, page = new_hit
            added.append(label)
            items.append(
                ComparisonItem(
                    aspect=aspect, label=label, change_type="added",
                    new_value=quote, new_page=page, risk_note=risk_note, confidence=0.7,
                )
            )
            continue
        if new_hit is None and old_hit is not None:
            quote, page = old_hit
            removed.append(label)
            items.append(
                ComparisonItem(
                    aspect=aspect, label=label, change_type="removed",
                    old_value=quote, old_page=page,
                    risk_note=f"This clause is gone from the new version. {risk_note}",
                    confidence=0.7,
                )
            )
            continue

        assert old_hit is not None and new_hit is not None
        old_quote, old_page = old_hit
        new_quote, new_page = new_hit
        changed = _materially_different(old_quote, new_quote)
        items.append(
            ComparisonItem(
                aspect=aspect,
                label=label,
                change_type="changed" if changed else "unchanged",
                old_value=old_quote,
                new_value=new_quote,
                old_page=old_page,
                new_page=new_page,
                risk_note=risk_note if changed else "",
                # Wording-based location is good evidence that the clause was
                # found, and weaker evidence that two found clauses are the
                # same clause -- hence well short of certainty.
                confidence=0.75 if changed else 0.6,
            )
        )

    for document, side in ((old, "the earlier document"), (new, "the newer document")):
        if len(document.text.strip()) < 600:
            unresolved.append(
                f"Very little text extracted from {side} ({document.filename}) — it may be a scan. "
                "Anything reported as missing from it may simply not have been readable."
            )
    if not any(item.change_type in {"added", "removed", "changed"} for item in items):
        unresolved.append(
            "I found no material differences in the clauses I check. That is not the same as "
            "\"the documents are identical\" — wording I do not check for could still differ."
        )

    return ComparisonResult(
        old_document_id=old.document_id,
        new_document_id=new.document_id,
        old_filename=old.filename,
        new_filename=new.filename,
        executive_summary=_summarise(items, added, removed),
        items=items,
        added_clauses=added,
        removed_clauses=removed,
        unresolved=unresolved,
        has_page_evidence=old.has_page_evidence and new.has_page_evidence,
    )


def _summarise(items: list[ComparisonItem], added: list[str], removed: list[str]) -> str:
    changed = [item.label for item in items if item.change_type == "changed"]
    if not changed and not added and not removed:
        return "Nothing I check for has changed between these two documents."
    parts: list[str] = []
    if changed:
        parts.append(f"{len(changed)} clause(s) changed ({', '.join(changed[:4])})")
    if added:
        parts.append(f"{len(added)} added ({', '.join(added[:3])})")
    if removed:
        parts.append(f"{len(removed)} removed ({', '.join(removed[:3])})")
    return "; ".join(parts) + "."
