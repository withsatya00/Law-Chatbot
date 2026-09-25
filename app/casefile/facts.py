"""Phase 2: structured fact extraction shared by the evidence organizer, the
case timeline and the contradiction detector.

All three need the same thing -- "what concrete facts does this text assert?"
-- so they share one extractor rather than three drifting ones. That matters
most for contradiction detection: two facts can only be compared if they were
normalised the same way, and a detector built on a second, subtly different
parser would silently miss conflicts (or invent them) whenever the two
disagreed about what "the same date" means.

A fact carries three things beyond its value:

* `slot` -- what KIND of fact it is, which is what makes two facts
  comparable at all;
* `normalized` -- the canonical form conflicts are compared on, so
  "Rs. 35,000", "35000/-" and "₹35,000" are one value rather than three;
* `source` -- where it came from, so a conflict can be shown to the user as
  "your message said X, the receipt says Y" rather than as an unattributed
  contradiction they cannot act on.

Extraction is deliberately regex/table-driven rather than LLM-driven. These
facts decide what goes into a legal document and whether the user is warned
about a conflict, so the process has to be deterministic, inspectable, and
identical on every run -- an LLM that extracts "35,000" on Monday and "35000"
on Tuesday would produce phantom contradictions.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

from app.drafting.localized_dates import parse_localized_date

# Slots are the comparison unit. Two facts conflict only if they share a slot,
# so the slot vocabulary is closed and named here rather than inferred.
AMOUNT = "amount"
INCIDENT_DATE = "incident_date"
TRANSACTION_ID = "transaction_id"
BANK = "bank"
PHONE = "phone"
EMAIL = "email"
POLICE_STATION = "police_station"
ADDRESS = "address"
PERSON = "person"
PLATFORM = "platform"
ACCOUNT = "account"

# Human labels, used in conflict prompts and the evidence table.
SLOT_LABELS: dict[str, str] = {
    AMOUNT: "Amount",
    INCIDENT_DATE: "Date of incident",
    TRANSACTION_ID: "Transaction / UTR reference",
    BANK: "Bank",
    PHONE: "Phone number",
    EMAIL: "Email address",
    POLICE_STATION: "Police station",
    ADDRESS: "Address",
    PERSON: "Name",
    PLATFORM: "Platform / app",
    ACCOUNT: "Account number",
}

# Slots where two different values genuinely contradict each other. A user can
# legitimately have two phone numbers or mention two people, so those are
# collected but never treated as a conflict; there is only one amount lost and
# one date on which it happened.
CONFLICTING_SLOTS: frozenset[str] = frozenset({
    AMOUNT, INCIDENT_DATE, TRANSACTION_ID, BANK, POLICE_STATION, ADDRESS, PLATFORM, ACCOUNT,
})

_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|₹|rupees)\s*([0-9][0-9,]*(?:\.\d{1,2})?)"
    r"|([0-9][0-9,]{3,}(?:\.\d{1,2})?)\s*(?:rs\.?|rupees|/-|रुपये|रुपए|રૂપિયા|টাকা|ரூபாய்|రూపాయలు)",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
# "25 August 2026" / "25 अगस्त 2026" / "25th Aug 2026" -- the month word is
# matched loosely and resolved against the localized month table, so this one
# pattern covers every language that table covers.
_WORD_DATE_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?[\s,]+([^\s,\d]{3,20})[\s,]+(\d{4})\b", re.IGNORECASE)

# A UTR/transaction reference: a long token mixing letters and digits, or a
# 12+ digit run. Anchored on an explicit label where one exists, because an
# unlabelled long number is just as likely to be an account or Aadhaar.
_LABELLED_REFERENCE_RE = re.compile(
    r"(?:utr|txn|transaction|reference|ref|rrn|order|payment)\s*(?:id|no\.?|number)?\s*[:#\-]?\s*"
    r"([A-Za-z0-9][A-Za-z0-9\-/]{5,29})",
    re.IGNORECASE,
)
_BARE_REFERENCE_RE = re.compile(r"\b(?=[A-Z0-9]{10,24}\b)(?=[A-Z0-9]*\d)[A-Z]{3,6}\d[A-Z0-9]{5,}\b")
_ACCOUNT_RE = re.compile(
    r"(?:a/?c|acc(?:oun)?t)\s*(?:no\.?|number)?\s*[:#\-]?\s*(\d{6,18})", re.IGNORECASE
)
_PHONE_RE = re.compile(r"(?:\+?91[\s-]?)?\b([6-9]\d{9})\b")
_EMAIL_RE = re.compile(r"\b([\w.+-]+@[\w-]+\.[\w.-]+)\b")
# The station NAME is at most three words and may not contain punctuation: an
# unbounded run before "Police Station" reaches back across sentence
# boundaries and swallows whatever preceded it (confirmed: it captured
# "example.com. I complained at Gomti Nagar" from a message that mentioned an
# email address two sentences earlier). Real station names are one or two
# words ("Hazratganj", "Gomti Nagar", "Shivajinagar"), so three is already
# generous; the leading grammar word the window still admits is trimmed by
# `_trim_station_name` below.
_POLICE_STATION_RE = re.compile(
    r"((?:[^\W\d_]|[ऀ-෿])+(?:[ \t]+(?:(?:[^\W\d_]|[ऀ-෿])+)){0,2})[ \t]*"
    r"(?:police\s+station|p\.?s\.?\b|thana|थाना|पुलिस\s*स्टेशन|पोलीस\s*स्टेशन|પોલીસ\s*સ્ટેશન|থানা)",
    re.IGNORECASE,
)
# Words that are grammar, not part of a station's name -- trimmed off the
# front of the captured run ("complained at Gomti Nagar Police Station" ->
# "Gomti Nagar Police Station").
_STATION_NAME_STOPWORDS = frozenset({
    "at", "in", "to", "the", "my", "a", "an", "of", "from", "near", "nearest", "local", "concerned",
    "complained", "reported", "filed", "filing", "visited", "went", "contacted", "and", "or",
    "me", "par", "mein", "ko", "se", "wale",
    "fir", "complaint", "report", "registered", "lodged", "lodging", "lodge", "raised", "submitted",
    "मैंने", "मैं", "में", "पर", "को", "से", "मैने", "हमने",
})

# Banks and payment platforms are recognised from a closed list: a free-text
# "which organisation is this?" extractor would pick up every proper noun in
# the sentence, and a wrong bank name in a fraud complaint is worse than none.
_BANKS: tuple[str, ...] = (
    "State Bank of India", "SBI", "HDFC Bank", "HDFC", "ICICI Bank", "ICICI", "Axis Bank", "Axis",
    "Punjab National Bank", "PNB", "Bank of Baroda", "BoB", "Kotak Mahindra", "Kotak", "Yes Bank",
    "IndusInd", "Canara Bank", "Union Bank", "IDFC First", "IDBI", "Bank of India", "Central Bank of India",
    "Indian Bank", "UCO Bank", "Federal Bank", "RBL Bank", "Bandhan Bank", "AU Small Finance", "India Post",
)
_PLATFORMS: tuple[str, ...] = (
    "UPI", "Google Pay", "GPay", "PhonePe", "Paytm", "BHIM", "Amazon Pay", "WhatsApp", "Telegram",
    "Instagram", "Facebook", "Amazon", "Flipkart", "Meesho", "Myntra", "Snapdeal", "OLX", "Quikr",
    "Zerodha", "Groww", "Upstox", "Binance", "Netbanking", "Net Banking", "Debit Card", "Credit Card",
)


@dataclass(frozen=True)
class ExtractedFact:
    slot: str
    value: str
    normalized: str
    source: str
    excerpt: str = ""

    @property
    def label(self) -> str:
        return SLOT_LABELS.get(self.slot, self.slot.replace("_", " ").title())


def _excerpt(text: str, start: int, end: int, window: int = 45) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = " ".join(text[left:right].split())
    return f"...{snippet}..." if (left > 0 or right < len(text)) else snippet


def normalize_amount(value: str) -> str:
    """`Rs. 35,000` / `35000/-` / `₹35,000.00` all normalise to `35000`.

    Trailing `.00` is dropped so a receipt's "35,000.00" and a chat message's
    "35000" are recognised as the same amount rather than reported as a
    conflict.
    """
    digits = re.sub(r"[^\d.]", "", value or "")
    if not digits:
        return ""
    try:
        amount = float(digits)
    except ValueError:
        return re.sub(r"\D", "", digits)
    return str(int(amount)) if amount.is_integer() else f"{amount:.2f}"


def normalize_date(value: str) -> str:
    """Any recognised date shape to ISO `YYYY-MM-DD`, else `""`.

    Ambiguous numeric dates are read as DD/MM/YYYY -- the Indian convention,
    and the one every localized template in this app already writes.
    """
    parsed = parse_date(value)
    return parsed.isoformat() if parsed else ""


def parse_date(value: str) -> date | None:
    text = (value or "").strip()
    if not text:
        return None
    iso = _ISO_DATE_RE.search(text)
    if iso:
        year, month, day = (int(part) for part in iso.groups())
        return _safe_date(year, month, day)
    numeric = _NUMERIC_DATE_RE.search(text)
    if numeric:
        day, month, year = (int(part) for part in numeric.groups())
        if year < 100:
            year += 2000
        return _safe_date(year, month, day)
    localized = parse_localized_date(text)
    if localized:
        return localized
    word = _WORD_DATE_RE.search(text)
    if word:
        day_text, month_word, year_text = word.groups()
        word_month = _month_number(month_word)
        if word_month:
            return _safe_date(int(year_text), word_month, int(day_text))
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


_ENGLISH_MONTH_PREFIXES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _month_number(word: str) -> int | None:
    cleaned = (word or "").strip().strip(".,")
    localized = parse_localized_date(f"1 {cleaned} 2000")
    if localized:
        return localized.month
    return _ENGLISH_MONTH_PREFIXES.get(cleaned[:3].lower())


def _normalize_name(value: str) -> str:
    """Casefold plus whitespace/punctuation collapse, so "Gomti Nagar P.S." and
    "gomti nagar ps" compare equal. Diacritics are left alone: in Indic
    scripts a matra is part of the word, not an accent."""
    stripped = unicodedata.normalize("NFKC", value or "").casefold()
    collapsed = re.sub(r"\s+", " ", stripped)
    # Punctuation is dropped by Unicode CATEGORY, not via `\w`: Python's `\w`
    # excludes combining marks (Mn/Mc), so a `[^\w\s]` filter silently strips
    # every Devanagari/Tamil/Bengali matra and turned "गोमती नगर थाना" into
    # "गमत नगर थन" -- which then failed to match the same station written
    # anywhere else.
    return "".join(
        char for char in collapsed
        if char.isspace() or not unicodedata.category(char).startswith(("P", "S"))
    ).strip()


def _trim_station_name(captured: str, whole_match: str) -> str:
    """Drops leading grammar words from a captured station name and returns it
    together with the "Police Station" suffix that identified it."""
    words = (captured or "").split()
    while words and words[0].casefold() in _STATION_NAME_STOPWORDS:
        words.pop(0)
    suffix = whole_match[len(captured):].strip() if captured else whole_match.strip()
    name = " ".join(words).strip()
    return f"{name} {suffix}".strip() if name else suffix


def extract_facts(text: str, source: str) -> list[ExtractedFact]:
    """Every structured fact `text` asserts, tagged with where it came from.

    Order is stable (slot by slot, then position in the text) so repeated runs
    over the same input produce identical output -- the contradiction detector
    compares across runs and would otherwise report ordering noise as change.
    """
    if not text:
        return []
    facts: list[ExtractedFact] = []
    seen: set[tuple[str, str]] = set()

    def add(slot: str, value: str, normalized: str, start: int, end: int) -> None:
        value = value.strip()
        if not value or not normalized:
            return
        key = (slot, normalized)
        if key in seen:
            return
        seen.add(key)
        facts.append(ExtractedFact(slot, value, normalized, source, _excerpt(text, start, end)))

    for match in _AMOUNT_RE.finditer(text):
        raw = match.group(1) or match.group(2) or ""
        add(AMOUNT, match.group(0).strip(), normalize_amount(raw), match.start(), match.end())

    for pattern in (_ISO_DATE_RE, _NUMERIC_DATE_RE, _WORD_DATE_RE):
        for match in pattern.finditer(text):
            normalized = normalize_date(match.group(0))
            if normalized:
                add(INCIDENT_DATE, match.group(0).strip(), normalized, match.start(), match.end())

    for match in _LABELLED_REFERENCE_RE.finditer(text):
        reference = match.group(1)
        add(TRANSACTION_ID, reference, reference.upper(), match.start(), match.end())
    for match in _BARE_REFERENCE_RE.finditer(text):
        add(TRANSACTION_ID, match.group(0), match.group(0).upper(), match.start(), match.end())

    for match in _ACCOUNT_RE.finditer(text):
        add(ACCOUNT, match.group(1), match.group(1), match.start(), match.end())

    for match in _PHONE_RE.finditer(text):
        add(PHONE, match.group(1), match.group(1), match.start(), match.end())

    for match in _EMAIL_RE.finditer(text):
        add(EMAIL, match.group(1), match.group(1).lower(), match.start(), match.end())

    for match in _POLICE_STATION_RE.finditer(text):
        name = _trim_station_name(match.group(1), match.group(0))
        add(POLICE_STATION, name, _normalize_name(name), match.start(), match.end())

    lowered = text.lower()
    for bank in _BANKS:
        index = lowered.find(bank.lower())
        if index != -1:
            add(BANK, bank, _normalize_name(bank), index, index + len(bank))
            break  # the first/longest match wins; listing every alias would be noise
    for platform in _PLATFORMS:
        index = lowered.find(platform.lower())
        if index != -1:
            add(PLATFORM, platform, _normalize_name(platform), index, index + len(platform))

    return facts


def facts_from_fields(fields: dict[str, str], source: str = "draft_fields") -> list[ExtractedFact]:
    """Facts taken from named draft fields.

    A field's KEY already states its slot, which is far more reliable than
    re-detecting it from the value -- `police_station: "Gomti Nagar"` has no
    "police station" phrase in the value for the regex to find. Falls back to
    free-text extraction for fields with no slot mapping (notably `facts`,
    the free narrative, which is where most facts actually live).
    """
    slot_by_field = {
        "fraud_amount": AMOUNT, "amount_paid": AMOUNT, "amount": AMOUNT,
        "incident_date": INCIDENT_DATE, "purchase_date": INCIDENT_DATE, "transaction_date": INCIDENT_DATE,
        "transaction_id": TRANSACTION_ID, "utr_number": TRANSACTION_ID,
        "bank_name": BANK, "applicant_mobile": PHONE, "applicant_email": EMAIL,
        "police_station": POLICE_STATION, "applicant_address": ADDRESS, "respondent_address": ADDRESS,
        "incident_location": ADDRESS, "place": ADDRESS, "city": ADDRESS,
        "applicant_name": PERSON, "respondent_name": PERSON, "fraud_type": PLATFORM,
    }
    facts: list[ExtractedFact] = []
    seen: set[tuple[str, str]] = set()
    for key, value in (fields or {}).items():
        text = str(value or "").strip()
        if not text:
            continue
        slot = slot_by_field.get(key)
        if slot is None:
            for fact in extract_facts(text, f"{source}:{key}"):
                identity = (fact.slot, fact.normalized)
                if identity in seen:
                    continue
                seen.add(identity)
                facts.append(fact)
            continue
        normalized = (
            normalize_amount(text) if slot == AMOUNT
            else normalize_date(text) if slot == INCIDENT_DATE
            else text.upper() if slot == TRANSACTION_ID
            else _normalize_name(text)
        )
        if not normalized or (slot, normalized) in seen:
            continue
        seen.add((slot, normalized))
        facts.append(ExtractedFact(slot, text, normalized, f"{source}:{key}", text[:120]))
    return facts
