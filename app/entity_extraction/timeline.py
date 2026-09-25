"""Timeline and deadline extraction.

Two things a legal document does with time, and they are not the same thing:

* it RECORDS events ("the agreement was signed on 12/03/2024"), and
* it IMPOSES deadlines ("within 30 days from the date of receipt").

The second kind is what people miss, and it is usually written relatively --
the document does not contain the date that matters, it contains the rule for
computing it. So a relative deadline is kept as what it is: the phrase, the
number of days, and what it runs from. A relative deadline is resolved to a
calendar date ONLY when the text supplies the anchor; otherwise
`normalized_date` stays empty and the anchor is asked for. Computing "30 days
from receipt" without knowing when receipt happened produces a confident,
wrong limitation date, which is the single worst output this module could
have.

Undated events are kept in their own list rather than being given a
placeholder date, because a placeholder puts them in a false chronological
order that reads as fact.
"""

import re
from datetime import date, timedelta

from app.core import clock
from app.entity_extraction.rules import iter_dates, parse_date_text
from app.schemas.extraction import EventCategory, TimelineEvent, TimelineExtractionResult

# Number words, so "within thirty days" reads the same as "within 30 days".
_NUMBER_WORDS: dict[str, int] = {
    "seven": 7, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
    "sixty": 60, "ninety": 90, "one hundred eighty": 180,
    "saat": 7, "pandrah": 15, "bees": 20, "tees": 30, "saath": 60, "nabbe": 90,
    "सात": 7, "पंद्रह": 15, "बीस": 20, "तीस": 30, "साठ": 60, "नब्बे": 90,
}

_DURATION_UNITS: tuple[tuple[str, int], ...] = (
    (r"days?|दिन|din", 1),
    (r"weeks?|सप्ताह|hafte|hafta", 7),
    (r"months?|माह|महीने|महीना|mahine", 30),
    (r"years?|साल|वर्ष", 365),
)

_UNIT_PATTERN = "|".join(unit for unit, _ in _DURATION_UNITS)

# "within 30 days", "30 दिन के भीतर", "30 din ke andar"
_WITHIN = re.compile(
    rf"(?:within|inside|no\s+later\s+than)\s+(\d{{1,3}}|[a-z\-]+)\s+({_UNIT_PATTERN})"
    rf"|(\d{{1,3}})\s+({_UNIT_PATTERN})\s*(?:ke\s+(?:andar|bhitar)|के\s+(?:भीतर|अंदर))",
    re.IGNORECASE,
)

# "on or before 15 March 2026", "15 मार्च 2026 तक"
_ON_OR_BEFORE = re.compile(
    r"(?:on\s+or\s+before|not\s+later\s+than|by)\s+([^,;.\n]{4,40})"
    r"|([^,;.\n]{4,40}?)\s*(?:tak|तक)\b",
    re.IGNORECASE,
)

# What a relative period runs FROM.
_ANCHORS: tuple[tuple[str, str], ...] = (
    ("the date of receipt", r"date\s+of\s+receipt|receipt\s+of\s+this|prapti\s+ki\s+tithi|प्राप्ति\s+की\s+तिथि"),
    ("the date of this notice", r"date\s+of\s+(?:this\s+)?notice|is\s+notice\s+ki\s+tarikh|इस\s+नोटिस\s+की\s+तिथि"),
    ("the date of the agreement", r"date\s+of\s+(?:this\s+)?(?:agreement|deed)|समझौते\s+की\s+तिथि"),
    ("termination", r"termination|समाप्ति|khatm"),
    ("expiry", r"expiry|expiration|अवसान|समाप्ति\s+तिथि"),
    ("the date of the incident", r"date\s+of\s+(?:the\s+)?incident|ghatna\s+ki\s+tarikh|घटना\s+की\s+तिथि"),
)

# "before expiry" -- a deadline with no computable date at all.
_BEFORE_EXPIRY = re.compile(
    r"\b(?:before|prior\s+to)\s+(?:the\s+)?(?:expiry|expiration|termination|due\s+date)\b"
    r"|(?:समाप्ति|अवसान)\s+से\s+पहले|expiry\s+se\s+pehle",
    re.IGNORECASE,
)

_CATEGORY_CUES: tuple[tuple[EventCategory, str], ...] = (
    ("hearing", r"hearing|sunwai|सुनवाई|peshi|पेशी|listed\s+for"),
    ("filing", r"\bfil(?:ed|ing)\b|lodged|registered|दायर|दर्ज"),
    ("notice", r"notice|नोटिस|intimation|सूचना"),
    ("payment", r"payment|paid|remit|invoice|भुगतान|adayagi"),
    ("agreement", r"agreement|deed|contract|executed|समझौता|अनुबंध"),
    ("termination", r"terminat|vacate|quit|समाप्त|समाप्ति"),
    ("incident", r"incident|accident|assault|theft|fraud|घटना|दुर्घटना|धोखा"),
)

# Who must act, when the sentence says so.
_PARTY_CUES: tuple[tuple[str, str], ...] = (
    ("the tenant", r"tenant|lessee|kirayedar|किरायेदार"),
    ("the landlord", r"landlord|lessor|makan\s*malik|मकान\s*मालिक"),
    ("the employer", r"employer|company|नियोक्ता"),
    ("the employee", r"employee|कर्मचारी"),
    ("the buyer", r"buyer|purchaser|क्रेता"),
    ("the seller", r"seller|vendor|विक्रेता"),
    ("the respondent", r"respondent|प्रतिवादी"),
    ("the complainant", r"complainant|शिकायतकर्ता"),
    ("you", r"\byou\b|आप"),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.\n।])\s+")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]


def _category(sentence: str) -> EventCategory:
    for category, pattern in _CATEGORY_CUES:
        if re.search(pattern, sentence, re.IGNORECASE):
            return category
    return "other"


def _responsible(sentence: str) -> str:
    """Who the sentence says must act, or "" -- never inferred."""
    if not re.search(r"\b(?:shall|must|is\s+required|agrees|undertakes|होगा|करेगा)\b", sentence, re.IGNORECASE):
        return ""
    for party, pattern in _PARTY_CUES:
        if re.search(pattern, sentence, re.IGNORECASE):
            return party
    return ""


def _anchor(sentence: str) -> str:
    for label, pattern in _ANCHORS:
        if re.search(pattern, sentence, re.IGNORECASE):
            return label
    return ""


def _duration_days(sentence: str) -> tuple[int, str] | None:
    """`(days, matched_text)` for a "within N <unit>" phrase, or None."""
    match = _WITHIN.search(sentence)
    if not match:
        return None
    quantity_text = match.group(1) or match.group(3) or ""
    unit_text = (match.group(2) or match.group(4) or "").lower()
    quantity = (
        int(quantity_text)
        if quantity_text.isdigit()
        else _NUMBER_WORDS.get(quantity_text.strip().lower(), 0)
    )
    if quantity <= 0:
        return None
    multiplier = next(
        (factor for unit, factor in _DURATION_UNITS if re.fullmatch(unit, unit_text, re.IGNORECASE)),
        1,
    )
    return quantity * multiplier, match.group(0)


def _expiry_status(value: date | None, today: date) -> str:
    if value is None:
        return "unknown"
    if value < today:
        return "expired"
    if value == today:
        return "due_today"
    return "upcoming"


def extract_timeline(
    text: str, *, page_of: dict[str, int] | None = None, today: date | None = None
) -> TimelineExtractionResult:
    """Every dated event and every deadline the text states.

    `page_of` maps a sentence prefix to the page it was read from, so a
    caller that has page provenance can supply it; absent, `source_page`
    stays `None` rather than defaulting to page 1.
    """
    current_day = today or clock.today()
    pages = page_of or {}
    dated: list[TimelineEvent] = []
    undated: list[TimelineEvent] = []
    questions: list[str] = []

    for sentence in _sentences(text):
        page = pages.get(sentence[:40])
        category = _category(sentence)
        party = _responsible(sentence)
        duration = _duration_days(sentence)
        anchor = _anchor(sentence)

        if duration is not None:
            days, phrase = duration
            # The anchor date, when the same sentence supplies one.
            anchor_date = next((parsed for _, parsed, _, _ in iter_dates(sentence) if parsed), None)
            resolved = anchor_date + timedelta(days=days) if anchor_date else None
            event = TimelineEvent(
                original_text=phrase,
                normalized_date=resolved.isoformat() if resolved else "",
                description=sentence[:200],
                category="deadline",
                responsible_party=party,
                source_page=page,
                is_deadline=True,
                expiry_status=_expiry_status(resolved, current_day),  # type: ignore[arg-type]
                # A resolved deadline is as good as its anchor; an unresolved
                # one is a correctly-read rule with a missing input, which is
                # a different and lower kind of certainty.
                confidence=0.8 if resolved else 0.55,
                source_text=sentence[:300],
                relative_to=anchor or ("the date given" if anchor_date else ""),
                relative_days=days,
                extraction_method="rule",
            )
            (dated if resolved else undated).append(event)
            if resolved is None:
                questions.append(
                    f'"{phrase}" runs from {anchor or "a date the document does not state"}. '
                    "What is that date? Until I know it I cannot work out the deadline."
                )
            continue

        if _BEFORE_EXPIRY.search(sentence):
            undated.append(
                TimelineEvent(
                    original_text=_BEFORE_EXPIRY.search(sentence).group(0),  # type: ignore[union-attr]
                    description=sentence[:200],
                    category="deadline",
                    responsible_party=party,
                    source_page=page,
                    is_deadline=True,
                    confidence=0.5,
                    source_text=sentence[:300],
                    relative_to="expiry",
                    extraction_method="rule",
                )
            )
            questions.append(
                "The document sets a deadline tied to an expiry date it does not state. "
                "When does the term expire?"
            )
            continue

        deadline_phrase = _ON_OR_BEFORE.search(sentence)
        for raw, parsed, _, _ in iter_dates(sentence):
            is_deadline = bool(deadline_phrase and raw in (deadline_phrase.group(0) or ""))
            event = TimelineEvent(
                original_text=raw,
                normalized_date=parsed.isoformat() if parsed else "",
                description=sentence[:200],
                category="deadline" if is_deadline else category,
                responsible_party=party,
                source_page=page,
                is_deadline=is_deadline,
                expiry_status=_expiry_status(parsed, current_day) if is_deadline else "unknown",  # type: ignore[arg-type]
                confidence=0.85 if parsed else 0.35,
                source_text=sentence[:300],
                extraction_method="rule",
            )
            if parsed:
                dated.append(event)
            else:
                undated.append(event)
                questions.append(
                    f'I found "{raw}" where a date should be, but could not read it as one. '
                    "What date does that mean?"
                )

    dated.sort(key=lambda event: event.normalized_date)
    return TimelineExtractionResult(
        events=dated,
        undated_events=undated,
        contradictions=find_contradictions(dated, current_day),
        questions=_dedupe(questions),
    )


def find_contradictions(events: list[TimelineEvent], today: date) -> list[str]:
    """Dates that cannot all be true.

    Reported, never resolved: the module has no basis for deciding which of
    two conflicting dates the user meant, and picking one would put an
    unverified date into a document they sign.
    """
    problems: list[str] = []
    parsed = [
        (event, parse_date_text(event.normalized_date))
        for event in events
        if event.normalized_date
    ]

    for event, value in parsed:
        if value is None:
            continue
        if value.year < 1900 or value > today.replace(year=today.year + 50):
            problems.append(
                f'"{event.original_text}" reads as {value.isoformat()}, which is outside any plausible '
                "range for this matter. Please confirm the correct date."
            )

    incidents = [value for event, value in parsed if event.category == "incident" and value]
    filings = [value for event, value in parsed if event.category == "filing" and value]
    if incidents and filings and min(filings) < min(incidents):
        problems.append(
            f"A filing is dated {min(filings).isoformat()}, before the earliest incident "
            f"({min(incidents).isoformat()}). One of those dates is wrong."
        )

    hearings = [value for event, value in parsed if event.category == "hearing" and value]
    if hearings and incidents and min(hearings) < min(incidents):
        problems.append(
            f"A hearing is dated {min(hearings).isoformat()}, before the earliest incident "
            f"({min(incidents).isoformat()}). Please confirm which is correct."
        )
    return problems


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
