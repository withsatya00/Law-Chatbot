"""Deterministic confidence scoring for an automated acquisition candidate.

This score is advisory only: it prioritizes the review/processing queue so a
reviewer sees the most promising discoveries first. It must NEVER influence
`verification_status` or `review_status` -- per `app/rag/kb_jurisdiction.py`'s
rule 3, nothing a regex or heuristic infers is "verified", and a high score
here is not an exception to that. `KnowledgeBaseAutomationService` always
starts a machine-acquired candidate at `verification_status="unverified"`
regardless of this value.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from app.services.kb_automation import SourceCandidate

_AMENDMENT_RE = re.compile(r"amendment|amend(?:ing|ed)?|संशोधन", re.IGNORECASE)
_REPEAL_RE = re.compile(r"repeal(?:ed|ing)?|rescind(?:ed|ing)?|निरसन|निरस्त", re.IGNORECASE)
_ORDINANCE_RE = re.compile(r"ordinance|अध्यादेश", re.IGNORECASE)
_COMMENCEMENT_RE = re.compile(
    r"commencement|comes? into force|effective from|प्रवर्तन|लागू\s+(?:होगा|हुई|हुआ)",
    re.IGNORECASE,
)


def classify_change_type(record_text: str) -> str:
    """Advisory tag from a deterministic English/Hindi keyword match on the
    listing text surrounding a candidate link. Never fed into `amends`/
    `supersedes` metadata -- that linkage stays a human decision made through
    `MonitorReviewRequest`."""
    if _REPEAL_RE.search(record_text):
        return "repeal"
    if _AMENDMENT_RE.search(record_text):
        return "amendment"
    if _ORDINANCE_RE.search(record_text):
        return "ordinance"
    if _COMMENCEMENT_RE.search(record_text):
        return "commencement"
    return "unclassified"


def score_candidate(
    *, url: str, title: str, record_text: str, is_pdf: bool,
    has_date: bool, has_act_number: bool, expected_domain_suffixes: tuple[str, ...] = (),
) -> float:
    """0.0-1.0. Each signal is a real, independently-checkable fact about the
    candidate link itself -- not a model opinion -- so the score stays fully
    explainable to a reviewer deciding what to look at first."""
    score = 0.0
    if has_date:
        score += 0.25
    if has_act_number:
        score += 0.25
    if is_pdf:
        score += 0.2
    host = (urlsplit(url).hostname or "").lower()
    if expected_domain_suffixes and any(host.endswith(suffix) for suffix in expected_domain_suffixes):
        score += 0.2
    elif not expected_domain_suffixes and host.endswith((".gov.in", ".nic.in")):
        score += 0.1
    meaningful_title = len(" ".join((title or record_text).split())) >= 12
    if meaningful_title:
        score += 0.1
    return min(1.0, round(score, 4))


def score_source_candidate(candidate: SourceCandidate, record_text: str = "") -> float:
    """Convenience wrapper scoring an already-built `SourceCandidate`."""
    is_pdf = urlsplit(candidate.url).path.casefold().endswith(".pdf")
    has_date = bool(candidate.publication_date)
    has_act_number = bool(candidate.act_number)
    return score_candidate(
        url=candidate.url, title=candidate.title, record_text=record_text or candidate.title,
        is_pdf=is_pdf, has_date=has_date, has_act_number=has_act_number,
    )
