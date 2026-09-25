"""Deterministic extraction rules — the layer that produces evidence.

A regex that matched is a fact about the text: the span it matched IS the
evidence, and it can be shown to the user. That is why this layer runs first
and why a model, when one runs at all, may only ADD to what these rules
found, never overrule it.

Naming rule that shapes the whole module: a person's name is only extracted
when the text itself labels it. "Complainant: Rahul Sharma" yields a person
with the role `complainant`; a bare "Rahul Sharma" in a paragraph yields
nothing, because a capitalised pair of words is not evidence of a party, and
guessing here is how the wrong person ends up named as the accused.
"""

import re
from collections.abc import Iterator
from datetime import date

from app.drafting.localized_dates import parse_localized_date
from app.schemas.extraction import EntityType, ExtractedEntity, LegalRole

# A name as it appears after a role label: up to four capitalised words, or a
# Devanagari run. Deliberately narrow -- it only ever runs immediately after a
# label that already established what the name IS.
#
# TWO constraints here are load-bearing and were each wrong at first:
#
# * the word separator is `[ \t]+`, NOT `\s+`. With `\s+` the pattern ran
#   across the newline and "Accused: Vikram Singh\nFIR No. 233/2024" produced
#   the name "Vikram Singh FIR No" -- a party named after a form field.
# * capitalisation is enforced case-SENSITIVELY. The label needs to match
#   case-insensitively ("Complainant"/"complainant"), but applying
#   `re.IGNORECASE` to the whole pattern silently turned `[A-Z]` into "any
#   letter", and "The tenant shall vacate the premises" then yielded a person
#   named "shall vacate the premises" with the role `tenant`. The label is
#   wrapped in a scoped `(?i:...)` instead, so only it is case-insensitive.
_NAME = r"([A-Z][A-Za-z.'-]+(?:[ \t]+[A-Z][A-Za-z.'-]+){0,3}|[ऀ-ॿ][ऀ-ॿ \t.]{2,40}?)"

# Role labels, in the languages parties actually appear in. The label is the
# evidence; the captured group is the value.
_ROLE_LABELS: tuple[tuple[LegalRole, str], ...] = (
    ("complainant", r"complainant|shikayatkarta|शिकायतकर्ता|परिवादी"),
    ("accused", r"accused|aaropi|आरोपी|अभियुक्त"),
    ("applicant", r"applicant|aavedak|आवेदक|प्रार्थी"),
    ("respondent", r"respondent|pratiwadi|प्रतिवादी|उत्तरदाता"),
    ("petitioner", r"petitioner|yachikakarta|याचिकाकर्ता"),
    ("defendant", r"defendant|बचाव\s*पक्ष"),
    ("advocate", r"advocate|counsel|vakil|अधिवक्ता|वकील"),
    ("witness", r"witness|gawah|गवाह|साक्षी"),
    ("landlord", r"landlord|lessor|makan\s*malik|मकान\s*मालिक|पट्टादाता"),
    ("tenant", r"tenant|lessee|kirayedar|किरायेदार|पट्टेदार"),
    ("employer", r"employer|niyokta|नियोक्ता"),
    ("employee", r"employee|karmchari|कर्मचारी"),
    ("buyer", r"buyer|purchaser|kreta|क्रेता|खरीदार"),
    ("seller", r"seller|vendor|vikreta|विक्रेता"),
)

_ROLE_PATTERNS: tuple[tuple[LegalRole, re.Pattern[str]], ...] = tuple(
    (
        role,
        re.compile(
            rf"(?i:{labels})[ \t]*(?i:no\.?[ \t]*\d+[ \t]*)?[:\-–—]?[ \t]*"
            rf"(?i:is|are|shri|smt|mr\.?|mrs\.?|ms\.?|श्री|श्रीमती)?[ \t]*{_NAME}"
        ),
    )
    for role, labels in _ROLE_LABELS
)

# Everything else: one pattern per entity type, each capturing the value in
# group 1 and matching the evidence span as group 0.
_VALUE_PATTERNS: tuple[tuple[EntityType, re.Pattern[str]], ...] = (
    (
        "fir_number",
        re.compile(
            r"(?:f\.?i\.?r\.?|प्राथमिकी|प्रथम\s*सूचना\s*रिपोर्ट)\s*(?:no\.?|number|संख्या|सं\.?)?\s*[:\-]?\s*"
            r"([0-9]{1,6}\s*/\s*[0-9]{2,4})",
            re.IGNORECASE,
        ),
    ),
    (
        "case_number",
        re.compile(
            r"(?:case|suit|petition|appeal|complaint|c\.?c\.?|crl\.?a\.?|o\.?s\.?|w\.?p\.?|मुकदमा|वाद)\s*"
            r"(?:no\.?|number|संख्या|सं\.?)\s*[:\-]?\s*([A-Za-z0-9./\-]{3,30})",
            re.IGNORECASE,
        ),
    ),
    (
        # Scoped `(?i:...)` again: the words "police station" may be written
        # any way, but the station's NAME must still start with a capital,
        # or "at the police station" yields a station called "at the".
        "police_station",
        re.compile(
            r"(?:([A-Z][A-Za-z][A-Za-z \t]{1,28}?)[ \t]+(?i:police[ \t]+station|p\.s\.)"
            r"|(?i:police[ \t]+station|थाना)[ \t]*[:\-]?[ \t]*([A-Zऀ-ॿ][A-Za-zऀ-ॿ \t]{2,30}))",
        ),
    ),
    (
        "court",
        re.compile(
            r"((?:hon'?ble\s+)?(?:supreme\s+court|high\s+court|district\s+court|sessions\s+court|"
            r"family\s+court|consumer\s+(?:forum|commission)|tribunal|magistrate'?s?\s+court)"
            r"(?:\s+(?:of|at)\s+[A-Z][A-Za-z]+)?|[ऀ-ॿ\s]{2,20}न्यायालय)",
            re.IGNORECASE,
        ),
    ),
    (
        "amount",
        re.compile(
            r"((?:rs\.?|inr|₹|रु\.?|रुपये)\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?(?:\s*(?:lakhs?|lakh|crores?|crore|लाख|करोड़))?)",
            re.IGNORECASE,
        ),
    ),
    (
        "vehicle_number",
        re.compile(r"\b([A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,3}\s?[0-9]{3,4})\b"),
    ),
    (
        "property",
        re.compile(
            r"((?:flat|plot|house|shop|survey|khasra|door)\s*(?:no\.?|number|सं\.?)\s*[:\-]?\s*[A-Za-z0-9/\-]{1,15})",
            re.IGNORECASE,
        ),
    ),
    (
        "document_reference",
        re.compile(r"((?:annexure|exhibit|अनुलग्नक|प्रदर्श)\s*[A-Z]{0,3}[-\s]?[0-9]{1,3})", re.IGNORECASE),
    ),
    (
        "email",
        re.compile(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"),
    ),
    (
        "phone",
        re.compile(r"(?<![0-9])((?:\+91[\s-]?)?[6-9][0-9]{9})(?![0-9])"),
    ),
    (
        "organization",
        re.compile(
            r"\b([A-Z][A-Za-z&.\s]{2,40}?(?:Pvt\.?\s*Ltd\.?|Private\s+Limited|Limited|Ltd\.?|LLP|"
            r"Bank|Insurance|Corporation|Company))\b"
        ),
    ),
    (
        # `\b` before the abbreviations is what stops "Rs. 45,000" being read
        # as "s. 45" -- a section number invented out of a rupee amount.
        "section",
        re.compile(
            r"(\b(?:section|sec\.?|s\.|धारा)\s*([0-9]+[A-Z]{0,2}(?:\([0-9a-z]+\))?))",
            re.IGNORECASE,
        ),
    ),
    (
        # Anchored on a leading NUMBER (house/flat/plot) and a trailing PIN,
        # which is what an Indian postal address actually looks like. Started
        # at any capital instead, it swallowed the preceding party name and
        # reported "Rahul Sharma, 12 MG Road, Pune 411001" as an address.
        "address",
        re.compile(
            r"(\b[0-9][A-Za-z0-9,\s./-]{8,60}?\s*(?:-|,)?\s*(?:pin\s*(?:code)?\s*[:\-]?\s*)?[1-9][0-9]{5})\b",
            re.IGNORECASE,
        ),
    ),
)

# Acts are matched by NAME, from a fixed list, because a plausible-looking
# "Act" phrase in free text is very often not a real enactment.
_ACT_NAMES: tuple[str, ...] = (
    "Bharatiya Nyaya Sanhita",
    "Bharatiya Nagarik Suraksha Sanhita",
    "Bharatiya Sakshya Adhiniyam",
    "Indian Penal Code",
    "Code of Criminal Procedure",
    "Indian Evidence Act",
    "Consumer Protection Act",
    "Information Technology Act",
    "Indian Contract Act",
    "Companies Act",
    "Transfer of Property Act",
    "Negotiable Instruments Act",
    "Hindu Marriage Act",
    "Right to Information Act",
    "Rent Control Act",
    "Motor Vehicles Act",
    "Specific Relief Act",
    "Arbitration and Conciliation Act",
)

# Obligation sentences: who must do what. The modal is the evidence.
_OBLIGATION = re.compile(
    r"([^.\n]{10,200}?\b(?:shall|must|is\s+required\s+to|undertakes\s+to|agrees\s+to|"
    r"करना\s+होगा|करेगा|करेंगी|अनिवार्य\s+है)\b[^.\n]{5,200}\.)",
    re.IGNORECASE,
)

# Absolute dates in the forms Indian documents actually use.
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b"),
    re.compile(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-zऀ-ॿ]{3,15})\,?\s+(\d{4})\b",
        re.IGNORECASE,
    ),
)

_ENGLISH_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def parse_date_text(text: str) -> date | None:
    """A date from `text`, or None. Never a guess.

    An ambiguous numeric date is read day-first, which is the Indian
    convention and the one every other date parser in this codebase uses; a
    value that cannot be a day-first date (13/25/2024) returns None rather
    than being silently reinterpreted.
    """
    stripped = (text or "").strip()
    iso = _DATE_PATTERNS[0].search(stripped)
    if iso:
        return _safe_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    numeric = _DATE_PATTERNS[1].search(stripped)
    if numeric:
        day, month, year = (int(part) for part in numeric.groups())
        return _safe_date(year, month, day)
    worded = _DATE_PATTERNS[2].search(stripped)
    if worded:
        day_text, month_word, year_text = worded.groups()
        month_number = _ENGLISH_MONTHS.get(month_word.lower())
        if month_number is not None:
            return _safe_date(int(year_text), month_number, int(day_text))
        return parse_localized_date(f"{day_text} {month_word} {year_text}")
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def iter_dates(text: str) -> Iterator[tuple[str, date | None, int, int]]:
    """Every date-shaped span in `text` as `(raw, parsed, start, end)`.

    A span that does not parse is still yielded with `None`, because "there
    is a date here I could not read" is information the caller must be able
    to act on -- silently dropping it looks identical to there being no date.
    """
    seen: set[tuple[int, int]] = set()
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text or ""):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in seen):
                continue
            seen.add(span)
            yield match.group(0), parse_date_text(match.group(0)), span[0], span[1]


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip(" ,.;:-–—")


def _context(text: str, start: int, end: int, width: int = 60) -> str:
    """The surrounding sentence fragment -- the evidence a user can check."""
    return _clean(text[max(0, start - width) : min(len(text), end + width)])


def extract_roles(text: str) -> list[ExtractedEntity]:
    """People whose role the text itself states.

    The label IS the evidence, so these are the only person entities that
    carry a role. A name with no label is not extracted at all.
    """
    found: list[ExtractedEntity] = []
    seen: set[tuple[str, str]] = set()
    for role, pattern in _ROLE_PATTERNS:
        for match in pattern.finditer(text or ""):
            value = _clean(match.group(1))
            if len(value) < 3 or (value.lower(), role) in seen:
                continue
            seen.add((value.lower(), role))
            found.append(
                ExtractedEntity(
                    value=value,
                    entity_type="person",
                    legal_role=role,
                    # High but not certain: the label is explicit, while the
                    # extent of the NAME after it is the part a human should
                    # still glance at.
                    confidence=0.85,
                    source_text=_context(text, *match.span()),
                    extraction_method="rule",
                )
            )
    return found


def extract_values(text: str) -> list[ExtractedEntity]:
    """Every non-person value the deterministic patterns establish."""
    found: list[ExtractedEntity] = []
    seen: set[tuple[str, str]] = set()

    def _add(value: str, entity_type: EntityType, span: tuple[int, int], confidence: float) -> None:
        cleaned = _clean(value)
        key = (cleaned.lower(), entity_type)
        if not cleaned or key in seen:
            return
        seen.add(key)
        found.append(
            ExtractedEntity(
                value=cleaned,
                entity_type=entity_type,
                confidence=confidence,
                source_text=_context(text, *span),
                extraction_method="rule",
            )
        )

    for entity_type, pattern in _VALUE_PATTERNS:
        for match in pattern.finditer(text or ""):
            value = next((group for group in match.groups() if group), match.group(0))
            _add(value, entity_type, match.span(), 0.8)

    lowered = (text or "").lower()
    for act in _ACT_NAMES:
        index = lowered.find(act.lower())
        if index >= 0:
            _add(act, "act", (index, index + len(act)), 0.9)

    for match in _OBLIGATION.finditer(text or ""):
        _add(match.group(1), "obligation", match.span(), 0.6)

    for raw, parsed, start, end in iter_dates(text or ""):
        _add(parsed.isoformat() if parsed else raw, "date", (start, end), 0.85 if parsed else 0.4)

    return found
