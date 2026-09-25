"""Part 58 "Answer Quality Audit" issue 25: PII redaction for anything that
leaves the request and lands somewhere it will be retained -- application
logs, the analytics query log, intent-analytics events.

Drafting is an inherently PII-heavy flow: a single police-complaint turn
carries the complainant's full postal address, mobile number, bank name,
account/UTR references and the amount lost, all typed straight into chat.
Those values legitimately belong in the generated DOCUMENT and in the
session's own short-lived working memory -- that is the product. What they
must not do is silently accumulate, in the clear, in operational logs and
long-lived analytics collections that a much wider set of people can read and
that outlive the conversation.

Scope, deliberately narrow:

* This module redacts what gets STORED or LOGGED. It never touches the
  answer shown to the user, the draft's own field values, or the exported
  document -- redacting those would break the feature.
* Redaction is one-way and shape-preserving (`98******10`), so analytics that
  count/segment on "a mobile number was present" still work while the value
  itself is gone.
* Patterns are conservative about false positives on legal text, which is
  full of numbers that are NOT PII (section numbers, years, monetary
  amounts). Anything ambiguous is left alone: over-redacting an analytics
  record is cheap, under-redacting is not, but silently mangling "Section
  318" into "Section ***" would make the logs useless for the debugging they
  exist for.
"""

import re
from collections.abc import Callable
from typing import Any

_MASK = "[redacted]"


def _keep_edges(value: str, lead: int = 2, tail: int = 2) -> str:
    """`9876543210` -> `98******10` -- enough to correlate two turns from the
    same user during an incident investigation, not enough to contact them."""
    digits = re.sub(r"\D", "", value)
    if len(digits) <= lead + tail:
        return "*" * len(digits)
    return f"{digits[:lead]}{'*' * (len(digits) - lead - tail)}{digits[-tail:]}"


# Ordered: the most specific/highest-confidence shapes first, so a value that
# could match two patterns is redacted by the one that actually describes it.
_RULES: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], str]], ...] = (
    # Email -- unambiguous, no legal-text false positives.
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), lambda m: _MASK),
    # Aadhaar: 12 digits, optionally spaced/hyphenated in 4-4-4 groups.
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), lambda m: _keep_edges(m.group(0))),
    # PAN: five letters, four digits, one letter.
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), lambda m: _MASK),
    # Payment card: 13-19 digits in groups.
    (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), lambda m: _keep_edges(m.group(0))),
    # Indian mobile: 10 digits starting 6-9, with an optional +91/0 prefix.
    # Anchored on the leading digit so a 10-digit figure that isn't a phone
    # number (rare in legal chat, and never starting 6-9 by convention for
    # amounts, which carry separators) isn't swept up.
    (re.compile(r"(?:\+?91[ -]?|\b0)?\b[6-9]\d{9}\b"), lambda m: _keep_edges(m.group(0))),
    # UPI virtual payment address.
    (re.compile(r"\b[\w.-]{2,}@(?:okhdfcbank|oksbi|okicici|okaxis|ybl|paytm|upi|apl|ibl)\b", re.IGNORECASE),
     lambda m: _MASK),
    # IFSC: four letters, "0", six alphanumerics.
    (re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), lambda m: _MASK),
    # Bank account / UTR / transaction reference: an explicitly labelled value,
    # or a long alphanumeric reference with both letters and digits (e.g.
    # "HDFC260828458721"). Labelled first so "A/c 123456" is caught even
    # though six digits alone would be too ambiguous to redact on sight.
    (re.compile(
        r"\b(?:a/?c|acc(?:oun)?t|utr|txn|transaction|ref(?:erence)?|rrn)\s*(?:no\.?|number|id)?\s*[:#-]?\s*"
        r"([A-Za-z0-9]{6,24})\b",
        re.IGNORECASE,
    ), lambda m: m.group(0).replace(m.group(1), _keep_edges(m.group(1)) if m.group(1).isdigit() else _MASK)),
    (re.compile(r"\b(?=[A-Za-z0-9]{12,24}\b)(?=[A-Za-z0-9]*\d)[A-Z]{3,6}\d[A-Za-z0-9]{6,}\b"), lambda m: _MASK),
    # Indian PIN code stated as part of an address ("- 226010", "PIN 302016").
    (re.compile(r"\b(?:pin\s*(?:code)?\s*[:-]?\s*)?(\d{6})\b(?=\s*$|[\s,.]|\s*[-–])", re.IGNORECASE),
     lambda m: m.group(0).replace(m.group(1), _keep_edges(m.group(1), lead=2, tail=0))),
)

# House-number-led postal addresses ("24, Shanti Vihar, Gomti Nagar, Lucknow").
# Kept separate from `_RULES` because it's the one rule that redacts free text
# rather than a structured identifier, so it's applied last and only to a
# clearly address-shaped run.
# The trailing segment is lazy and bounded by a lookahead so the run stops at
# the end of the address rather than swallowing whatever label follows it on
# the same line -- real chat input arrives as one line ("Applicant Address:
# 24, Shanti Vihar, ..., Lucknow - 226010 Mobile Number: 9876543210"), and a
# greedy tail ate "Mobile Number" along with the address, leaving mangled
# output that made the log line harder to read than the redaction was worth.
#
# The fixed-width lookbehinds exclude a statutory citation that happens to
# share the shape -- "Section 35, Consumer Protection Act, 2019, applies
# here." is a number followed by three comma-separated segments, and was
# redacted as if it were a street address, destroying exactly the legal
# detail these logs exist to show.
_ADDRESS_RE = re.compile(
    r"(?<!section )(?<!sec\. )(?<!sec )(?<!clause )(?<!article )(?<!rule )(?<!para )(?<!s\. )"
    r"(?<!धारा )(?<!कलम )"
    r"\b\d{1,4}[A-Za-z]?\s*[/,-]\s*"
    r"(?:[^,\n:]{2,40},\s*){2,}"
    r"[^,\n:]{2,40}?"
    r"(?=\s*(?:$|[.;\n]|\s(?:\w+[\s-]+){0,2}\w+\s*:))",
    re.IGNORECASE,
)


def mask_pii(text: str | None) -> str:
    """Returns `text` with personal identifiers redacted. Safe on `None`."""
    if not text:
        return text or ""
    masked = text
    for pattern, replacement in _RULES:
        masked = pattern.sub(replacement, masked)
    masked = _ADDRESS_RE.sub(_MASK, masked)
    return masked


def mask_entities(entities: dict[str, Any] | None) -> dict[str, Any]:
    """`EntityExtractionResponse.entities`-shaped dict with every string value
    (and every string inside a list value) redacted."""
    if not entities:
        return {}
    masked: dict[str, Any] = {}
    for key, value in entities.items():
        if isinstance(value, str):
            masked[key] = mask_pii(value)
        elif isinstance(value, list):
            masked[key] = [mask_pii(item) if isinstance(item, str) else item for item in value]
        else:
            masked[key] = value
    return masked
