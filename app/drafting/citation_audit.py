"""Verifies that a specific statute section a generated draft cites is one
the template's own curated `applicable_sections_hint` actually names.

Why this exists
----------------
Each template YAML tells the drafting prompt which sections apply to that
document type (`applicable_sections_hint`/`drafting_notes` -- e.g.
`cheque_bounce_notice.yaml` names "Section 138"/"Section 142" of the
Negotiable Instruments Act). Nothing previously checked whether the model's
OUTPUT actually cited one of those, rather than inventing or substituting a
different number. A wrong or hallucinated section number is the single
highest-stakes failure mode in a legal-drafting tool: unlike an unsupported
adjective or an uncited figure (see `fact_audit.py`), a citation is exactly
the kind of detail a lay user copies verbatim into a document they file
themselves, and cannot be expected to independently verify.

This is a narrow, precision-first check, for the same reason `fact_audit.py`
is: a finding the user cannot act on trains them to ignore the audit.

* Only runs when the template HAS curated section hints. An empty hint list
  (every Contract-category template today -- rent/service/partnership
  agreements etc.) means "nothing to verify against", not "nothing may be
  cited": those documents number their OWN clauses internally ("Section 5"/
  "Clause 5" of the agreement), which would otherwise misfire as if it were
  a hallucinated statute citation.
* Only recognizes `Section`/`Sec.`/`धारा`/`कलम` as citation cues -- not the
  broader `clause/schedule/rule/article` set `fact_audit.py` treats as
  citation CONTEXT for its own, different purpose (suppressing the
  unsupported-figure check). Those other words are far more likely to name
  the document's own internal structure than a reference to outside law.
* Only compares the cited NUMBER, not the Act name. Act names have too many
  legitimate short-title, abbreviation and translated forms across the
  languages this app drafts in (BNS/IPC, "the said Act", a full citation
  first then "the Act" thereafter, ...) to verify without false positives;
  the number is the part a hallucination is most likely to get wrong and the
  part a user has no independent way to check.
* Matches on the section's leading digits only ("66C" and "66" both reduce
  to "66"): the goal is catching a plainly different section, not enforcing
  letter/sub-clause-perfect citation.
"""

import re

_CITE_CUE = r"(?:Sections?|Sec\.?|धारा|कलम)"
_SECTION_TOKEN = r"\d+[A-Za-z]?"
_CITED_SECTION_RE = re.compile(rf"{_CITE_CUE}\s*[-:—]?\s*({_SECTION_TOKEN})", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")


def _base_number(token: str) -> str:
    match = re.match(r"\d+", token)
    return match.group(0) if match else token


def _hinted_numbers(applicable_sections_hint: list[str]) -> set[str]:
    numbers: set[str] = set()
    for hint in applicable_sections_hint:
        match = _CITED_SECTION_RE.search(hint)
        if match:
            numbers.add(_base_number(match.group(1)))
    return numbers


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_SPLIT.split(text or "") if sentence.strip()]


def _excerpt(sentence: str, limit: int = 160) -> str:
    return sentence if len(sentence) <= limit else f"{sentence[:limit].rstrip()}..."


def verify_citations(sections: dict[str, str], applicable_sections_hint: list[str]) -> list[dict[str, str]]:
    """Advisory findings in the same `section`/`category`/`excerpt`/`detail`
    shape `fact_audit.DraftAudit.as_dicts`/`prohibited_clauses.scan` produce,
    so all three audits reach the user through one `audit_findings` list.
    """
    if not applicable_sections_hint:
        return []
    hinted_numbers = _hinted_numbers(applicable_sections_hint)
    findings: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for heading, body in sections.items():
        if not body:
            continue
        for sentence in _sentences(body):
            for match in _CITED_SECTION_RE.finditer(sentence):
                cited = match.group(1)
                if _base_number(cited) in hinted_numbers:
                    continue
                key = (heading, cited)
                if key in seen:
                    continue
                seen.add(key)
                findings.append({
                    "section": heading,
                    "category": "unverified_citation",
                    "excerpt": _excerpt(sentence),
                    "detail": (
                        f'Cites "Section {cited}", which is not among the sections this document type is '
                        f"drafted under ({', '.join(sorted(hinted_numbers)) or 'none'}). Verify this citation "
                        "independently before relying on it -- it may be a substituted or invented section number."
                    ),
                })
    return findings
