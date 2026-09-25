"""Post-Phase-3 hardening (Phase 2, milestone B): a legal time-limit stated in
an answer must be traceable to the retrieved text.

The failure this exists for, from a real session
-----------------------------------------------
Asked "Cheque bounce hone par Negotiable Instruments Act Section 138 ke tahat
kya remedy hai", the assistant replied:

    "Legal notice ke guidance ke mutabiq, aapko pehle ek notice bhejna hoga
     jisme cheque ki amount ko notice milne ke 15 din ke andar pay karne ki
     demand ki jaye."

Two things are wrong with that, and only one of them is about the number.

* The authority cited is "legal notice guidance" -- this product's own
  drafting-template notes. A drafting note is not law. Nothing in the reply
  named an Act, a section, or a source with a verification status.
* The reply stated the fifteen-day payment window and omitted the *thirty-day*
  window in which the notice itself must be sent (s. 138 proviso (b)), the
  six-month presentation requirement (proviso (a)), and the one-month
  limitation for the complaint (s. 142(1)(b)). A reader following it would
  send a notice too late and lose the offence.

The general rule this module enforces
-------------------------------------
A period ("within fifteen days", "तीस दिन के भीतर", "one month") is one of the
few things in a legal answer a reader acts on IMMEDIATELY and irreversibly.
So: every period an answer states must appear in the text that was actually
retrieved. When it does not, the period is not deleted -- deleting text from a
grounded answer risks removing something correct -- it is MARKED, in the
answer and in `ChatResponse.warnings`, as unverified and needing an advocate's
confirmation.

Deliberate limits, so this stays a safety net and not a source of noise:

* Only periods stated in a legal-obligation frame ("within ...", "... ke andar",
  "... के भीतर") are examined. "I bought it two months ago" is a fact the user
  supplied, not a deadline being asserted.
* A period is matched on the normalized (count, unit) pair. "fifteen days" in
  the source supports "15 days" in the answer and vice versa. Units are NOT
  converted into each other: "one month" and "30 days" are different legal
  periods (s. 142(1)(b) says one month, not thirty days), and treating them as
  interchangeable is exactly the conflation that produced the bad answer.
* Nothing here decides what the law IS. It only reports whether the retrieved
  text says what the answer says.
"""

import re
from dataclasses import dataclass

# Canonical units. The keys are what a claim normalizes to; the values are
# every spelling seen in this corpus and in user-facing replies.
_UNIT_WORDS: dict[str, tuple[str, ...]] = {
    "days": ("days", "day", "din", "dino", "dinon", "दिन", "दिनों", "دن"),
    "weeks": ("weeks", "week", "hafte", "hafta", "सप्ताह", "हफ्ते", "हफ़्ते", "ہفتے"),
    "months": ("months", "month", "mahine", "maheene", "mahina", "माह", "महीने", "महीना", "महीनों", "ماہ"),
    "years": ("years", "year", "saal", "varsh", "वर्ष", "साल", "बरस", "سال"),
}

# Number words, English and the Hindi/Hinglish forms that actually turn up in
# statutory prose and in replies. Only the values a legal period is ever
# expressed in -- this is not a general numeral parser.
_NUMBER_WORDS: dict[str, int] = {
    "one": 1, "ek": 1, "एक": 1,
    "two": 2, "do": 2, "दो": 2,
    "three": 3, "teen": 3, "तीन": 3,
    "four": 4, "char": 4, "चार": 4,
    "five": 5, "paanch": 5, "panch": 5, "पांच": 5, "पाँच": 5,
    "six": 6, "chah": 6, "chhah": 6, "chhe": 6, "छह": 6, "छः": 6,
    "seven": 7, "saat": 7, "सात": 7,
    "eight": 8, "aath": 8, "आठ": 8,
    "nine": 9, "nau": 9, "नौ": 9,
    "ten": 10, "das": 10, "दस": 10,
    "twelve": 12, "barah": 12, "बारह": 12,
    "fifteen": 15, "pandrah": 15, "पंद्रह": 15, "पन्द्रह": 15,
    "twenty": 20, "bees": 20, "बीस": 20,
    "thirty": 30, "tees": 30, "तीस": 30,
    "forty": 40, "चालीस": 40,
    "sixty": 60, "साठ": 60,
    "ninety": 90, "नब्बे": 90,
}

_NUMBER_ALTERNATION = "|".join(sorted((re.escape(word) for word in _NUMBER_WORDS), key=len, reverse=True))
_UNIT_ALTERNATION = "|".join(
    sorted((re.escape(word) for words in _UNIT_WORDS.values() for word in words), key=len, reverse=True)
)
_UNIT_LOOKUP: dict[str, str] = {
    word.casefold(): canonical for canonical, words in _UNIT_WORDS.items() for word in words
}

# "<count> <unit>", where count is digits or one of the words above.
_PERIOD_PATTERN = re.compile(
    rf"(?<![\w\d])(?P<count>\d{{1,3}}|{_NUMBER_ALTERNATION})\s*[-–]?\s*(?P<unit>{_UNIT_ALTERNATION})(?![\w])",
    re.IGNORECASE,
)

# Words that make a period an ASSERTED OBLIGATION rather than a narrated fact.
# Checked in a window before/after the period. Without this, "the product
# failed within two months of purchase" -- the user's own supplied fact -- would
# be reported as an unsupported statutory deadline.
_OBLIGATION_CUES = (
    "within", "before", "no later", "not later", "period of", "time limit", "limitation",
    "deadline", "must", "shall", "has to", "have to", "expires", "expiry", "prescribed",
    "ke andar", "ke bhitar", "ke bheetar", "ke ander", "ki avadhi", "samay seema",
    "के भीतर", "के अंदर", "की अवधि", "समय सीमा", "समय-सीमा", "अवधि के", "के पूर्व",
    "کے اندر", "کی مدت",
)
# How far either side of the period to look for a cue. One clause's worth --
# wide enough for "a notice ... must be given within thirty days", tight enough
# that an obligation two sentences away does not license an unrelated number.
_CUE_WINDOW = 60


@dataclass(frozen=True)
class PeriodClaim:
    count: int
    unit: str
    # The exact text matched, for the message shown to the user.
    text: str

    @property
    def normalized(self) -> tuple[int, str]:
        return (self.count, self.unit)

    def describe(self) -> str:
        unit = self.unit[:-1] if self.count == 1 and self.unit.endswith("s") else self.unit
        return f"{self.count} {unit}"


def _to_count(raw: str) -> int | None:
    stripped = raw.strip().casefold()
    if stripped.isdigit():
        value = int(stripped)
        # A four-digit "year" is a date, not a period; three digits is already
        # generous for a statutory limit.
        return value if 0 < value <= 999 else None
    return _NUMBER_WORDS.get(stripped)


def _has_obligation_cue(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - _CUE_WINDOW) : end + _CUE_WINDOW].casefold()
    return any(cue in window for cue in _OBLIGATION_CUES)


def find_periods(text: str, *, require_obligation: bool) -> list[PeriodClaim]:
    """Every period stated in `text`.

    `require_obligation=True` keeps only periods framed as a rule or deadline;
    `False` takes them all, which is what a SOURCE text should be scanned with
    -- a statute that says "within a period of six months" supports an answer's
    "six months" regardless of how the sentence around it is phrased.
    """
    claims: list[PeriodClaim] = []
    for match in _PERIOD_PATTERN.finditer(text or ""):
        count = _to_count(match.group("count"))
        unit = _UNIT_LOOKUP.get(match.group("unit").casefold())
        if count is None or unit is None:
            continue
        if require_obligation and not _has_obligation_cue(text, match.start(), match.end()):
            continue
        claims.append(PeriodClaim(count=count, unit=unit, text=match.group(0).strip()))
    return claims


def unsupported_periods(answer: str, context_texts: list[str]) -> list[PeriodClaim]:
    """Periods the answer asserts as deadlines that no retrieved text states.

    De-duplicated on the normalized pair, in the order they appear, so an
    answer repeating "15 days" three times produces one warning.
    """
    supported = {
        claim.normalized
        for text in context_texts
        for claim in find_periods(text, require_obligation=False)
    }
    seen: set[tuple[int, str]] = set()
    unsupported: list[PeriodClaim] = []
    for claim in find_periods(answer, require_obligation=True):
        if claim.normalized in supported or claim.normalized in seen:
            continue
        seen.add(claim.normalized)
        unsupported.append(claim)
    return unsupported


_CAUTION: dict[str, str] = {
    "english": (
        "Unverified time limit: this answer states {periods}, which does not appear in the source "
        "text retrieved for it. Do not rely on that period -- have it confirmed against the bare Act "
        "by an advocate before acting."
    ),
    "hinglish": (
        "Unverified time limit: is jawab mein {periods} likha hai, lekin jo source text retrieve hua "
        "usme yeh period nahi hai. Is deadline par bharosa mat kijiye -- kisi advocate se bare Act "
        "dekhkar confirm karwa lijiye."
    ),
    "hindi": (
        "असत्यापित समय-सीमा: इस उत्तर में {periods} बताया गया है, जो इसके लिए प्राप्त स्रोत पाठ में नहीं मिला। "
        "इस अवधि पर निर्भर न रहें — कार्रवाई से पहले किसी अधिवक्ता से मूल अधिनियम में इसकी पुष्टि करा लें।"
    ),
}


def period_warnings(answer: str, context_texts: list[str]) -> list[str]:
    """Machine-readable companions to the caveat, for `ChatResponse.warnings`."""
    return [
        f"The stated time limit of {claim.describe()} is not supported by the retrieved source text."
        for claim in unsupported_periods(answer, context_texts)
    ]


def annotate_periods(answer: str, context_texts: list[str], language: str | None) -> str:
    """Append the unverified-time-limit caution to `answer`, or return it as is.

    Additive only: it never edits or removes what the answer already says. A
    grounded answer can be right about everything except one number, and
    deleting prose on a regex's say-so is a worse failure than flagging it.
    """
    claims = unsupported_periods(answer, context_texts)
    if not claims:
        return answer
    key = (language or "english").strip().casefold()
    template = _CAUTION.get(key, _CAUTION["english"])
    periods = ", ".join(f'"{claim.text}"' for claim in claims)
    return f"{answer}\n\n{template.format(periods=periods)}"
