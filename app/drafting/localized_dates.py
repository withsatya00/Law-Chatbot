"""Part 52 (Drafting Workflow Localization): recognizes localized month
names in date fields.

`datetime.strptime`'s `%B`/`%b` directives resolve month names from the
process's OS locale, which is a single global setting -- there is no way to
ask strptime for "Hindi month names" for one request while another request
in the same process uses English ones. A user typing a date the same way
they write everything else in their language ("15 जुलाई 2026") was
therefore always rejected as an invalid date, regardless of server locale.

This module sidesteps the problem entirely: it recognizes the literal
"<day> <localized month name> <year>" shape via a small fixed lookup table
(the closed set of 12 month names, per language) and builds the `date`
directly, instead of trying to make `strptime` locale-aware.
"""

import re
from datetime import date, timedelta

from app.core import clock

_LOCALIZED_MONTHS: dict[int, dict[str, str]] = {
    1: {"hindi": "जनवरी", "tamil": "ஜனவரி", "telugu": "జనవరి", "kannada": "ಜನವರಿ", "bengali": "জানুয়ারি",
        "malayalam": "ജനുവരി", "marathi": "जानेवारी", "gujarati": "જાન્યુઆરી", "punjabi": "ਜਨਵਰੀ",
        "odia": "ଜାନୁଆରୀ", "urdu": "جنوری"},
    2: {"hindi": "फरवरी", "tamil": "பிப்ரவரி", "telugu": "ఫిబ్రవరి", "kannada": "ಫೆಬ್ರವರಿ", "bengali": "ফেব্রুয়ারি",
        "malayalam": "ഫെബ്രുവരി", "marathi": "फेब्रुवारी", "gujarati": "ફેબ્રુઆરી", "punjabi": "ਫਰਵਰੀ",
        "odia": "ଫେବ୍ରୁଆରୀ", "urdu": "فروری"},
    3: {"hindi": "मार्च", "tamil": "மார்ச்", "telugu": "మార్చి", "kannada": "ಮಾರ್ಚ್", "bengali": "মার্চ",
        "malayalam": "മാർച്ച്", "marathi": "मार्च", "gujarati": "માર્ચ", "punjabi": "ਮਾਰਚ",
        "odia": "ମାର୍ଚ୍ଚ", "urdu": "مارچ"},
    4: {"hindi": "अप्रैल", "tamil": "ஏப்ரல்", "telugu": "ఏప్రిల్", "kannada": "ಏಪ್ರಿಲ್", "bengali": "এপ্রিল",
        "malayalam": "ഏപ്രിൽ", "marathi": "एप्रिल", "gujarati": "એપ્રિલ", "punjabi": "ਅਪ੍ਰੈਲ",
        "odia": "ଏପ୍ରିଲ", "urdu": "اپریل"},
    5: {"hindi": "मई", "tamil": "மே", "telugu": "మే", "kannada": "ಮೇ", "bengali": "মে",
        "malayalam": "മേയ്", "marathi": "मे", "gujarati": "મે", "punjabi": "ਮਈ",
        "odia": "ମଇ", "urdu": "مئی"},
    6: {"hindi": "जून", "tamil": "ஜூன்", "telugu": "జూన్", "kannada": "ಜೂನ್", "bengali": "জুন",
        "malayalam": "ജൂൺ", "marathi": "जून", "gujarati": "જૂન", "punjabi": "ਜੂਨ",
        "odia": "ଜୁନ", "urdu": "جون"},
    7: {"hindi": "जुलाई", "tamil": "ஜூலை", "telugu": "జూలై", "kannada": "ಜುಲೈ", "bengali": "জুলাই",
        "malayalam": "ജൂലൈ", "marathi": "जुलै", "gujarati": "જુલાઈ", "punjabi": "ਜੁਲਾਈ",
        "odia": "ଜୁଲାଇ", "urdu": "جولائی"},
    8: {"hindi": "अगस्त", "tamil": "ஆகஸ்ட்", "telugu": "ఆగస్టు", "kannada": "ಆಗಸ್ಟ್", "bengali": "আগস্ট",
        "malayalam": "ഓഗസ്റ്റ്", "marathi": "ऑगस्ट", "gujarati": "ઓગસ્ટ", "punjabi": "ਅਗਸਤ",
        "odia": "ଅଗଷ୍ଟ", "urdu": "اگست"},
    9: {"hindi": "सितंबर", "tamil": "செப்டம்பர்", "telugu": "సెప్టెంబర్", "kannada": "ಸೆಪ್ಟೆಂಬರ್", "bengali": "সেপ্টেম্বর",
        "malayalam": "സെപ്റ്റംബർ", "marathi": "सप्टेंबर", "gujarati": "સપ્ટેમ્બર", "punjabi": "ਸਤੰਬਰ",
        "odia": "ସେପ୍ଟେମ୍ବର", "urdu": "ستمبر"},
    10: {"hindi": "अक्टूबर", "tamil": "அக்டோபர்", "telugu": "అక్టోబర్", "kannada": "ಅಕ್ಟೋಬರ್", "bengali": "অক্টোবর",
        "malayalam": "ഒക്ടോബർ", "marathi": "ऑक्टोबर", "gujarati": "ઓક્ટોબર", "punjabi": "ਅਕਤੂਬਰ",
        "odia": "ଅକ୍ଟୋବର", "urdu": "اکتوبر"},
    11: {"hindi": "नवंबर", "tamil": "நவம்பர்", "telugu": "నవంబర్", "kannada": "ನವೆಂಬರ್", "bengali": "নভেম্বর",
        "malayalam": "നവംബർ", "marathi": "नोव्हेंबर", "gujarati": "નવેમ્બર", "punjabi": "ਨਵੰਬਰ",
        "odia": "ନଭେମ୍ବର", "urdu": "نومبر"},
    12: {"hindi": "दिसंबर", "tamil": "டிசம்பர்", "telugu": "డిసెంబర్", "kannada": "ಡಿಸೆಂಬರ್", "bengali": "ডিসেম্বর",
        "malayalam": "ഡിസംബർ", "marathi": "डिसेंबर", "gujarati": "ડિસેમ્બર", "punjabi": "ਦਸੰਬਰ",
        "odia": "ଡିସେମ୍ବର", "urdu": "دسمبر"},
}

# Flattened reverse lookup: localized month word -> month number. English
# month names aren't included here -- `datetime.strptime`'s `%B`/`%b`
# already handles those natively via `_DATE_FORMATS` in `validation.py`.
MONTH_WORD_TO_NUMBER: dict[str, int] = {
    name: month for month, names_by_language in _LOCALIZED_MONTHS.items() for name in names_by_language.values()
}

_LOCALIZED_DATE_PATTERN = re.compile(r"^\s*(\d{1,2})\s+(\S+)\s+(\d{4})\s*$")


def parse_localized_date(value: str) -> date | None:
    """Parses "<day> <localized month name> <year>" (e.g. "15 जुलाई 2026",
    "5 ஜனவரி 2026"). Returns `None` (never raises) when `value` doesn't
    match this shape or names a month word outside `MONTH_WORD_TO_NUMBER`,
    so callers can safely fall through to their own format list/patterns.
    """
    match = _LOCALIZED_DATE_PATTERN.match(value)
    if not match:
        return None
    day_text, month_word, year_text = match.groups()
    month = MONTH_WORD_TO_NUMBER.get(month_word)
    if month is None:
        return None
    try:
        return date(int(year_text), month, int(day_text))
    except ValueError:
        return None


# English month names, kept as their own fixed table rather than
# `date.strftime("%B")` -- `%B` resolves from the process's single global OS
# locale, so on a non-English-locale host it would silently render an
# ENGLISH document's own "Date" section in the wrong language too, and
# there's no way to ask strftime for "English" on one request while another
# request in the same process needs a different locale. Same reasoning
# `parse_localized_date`'s own docstring gives for why parsing needed this.
_ENGLISH_MONTHS: dict[int, str] = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def format_localized_date(value: date, language: str) -> str:
    """Renders `value` as "<day> <month name> <year>" in `language` -- e.g.
    "13 August 2026", "13 अगस्त 2026", "13 ஆகஸ்ட் 2026" (Part 51 "Draft
    Generation Quality Pass" item 4).

    Covers english plus every language in `_LOCALIZED_MONTHS` above (hindi/
    tamil/telugu/kannada/bengali/malayalam/marathi/gujarati/punjabi/odia/
    urdu). Falls back to numeric DD/MM/YYYY -- the same locale-independent
    format previously used unconditionally for every language -- for any
    language outside that fixed set (hinglish, assamese, sanskrit, etc.),
    rather than guessing at an unreviewed translation.
    """
    if language == "english":
        return f"{value.day} {_ENGLISH_MONTHS[value.month]} {value.year}"
    month_name = _LOCALIZED_MONTHS.get(value.month, {}).get(language)
    if month_name:
        return f"{value.day} {month_name} {value.year}"
    return value.strftime("%d/%m/%Y")


# ---------------------------------------------------------------------------
# Relative dates (draft-quality pass, Sept 2026)
#
# The incident: a user opened with "Kal PhonePe se Rs 15,000 ka fraud hua" and
# was told "'Kal' एक मान्य तिथि जैसा नहीं लगता" -- their incident date was
# discarded, they were asked to supply it again, and the finished complaint
# went out with no date of incident at all. "Kal"/"kal"/"yesterday" is how
# people actually report something that happened to them; refusing it is the
# validator failing the user, not the user failing the validator.
#
# Resolved to an ABSOLUTE date here, at the point of capture, so that
# everything downstream (the draft, the export, the fact audit) sees a real
# date rather than a word whose meaning drifts every midnight.
#
# On the Hindi "kal" ambiguity: कल means both yesterday and tomorrow, and the
# same is true of Marathi "उद्या/काल" cognates, Gujarati "કાલ" and Punjabi
# "ਕੱਲ੍ਹ". These functions resolve it to the PAST reading, because they are
# only ever applied to fields that record something that has already happened
# (an incident date). The caller is expected to tell the user which date it
# assumed -- `DraftFieldValidator` does exactly that -- so a user who meant
# something else can correct it in one turn instead of being blocked.
# ---------------------------------------------------------------------------

# Offset in days from today. Negative is the past.
_RELATIVE_DAY_WORDS: dict[str, int] = {
    # English
    "today": 0, "yesterday": -1, "day before yesterday": -2, "last night": -1, "tonight": 0,
    # Hinglish / romanised
    "aaj": 0, "aj": 0, "kal": -1, "kl": -1, "parso": -2, "parson": -2, "beete kal": -1,
    "kal raat": -1, "aaj subah": 0, "kal subah": -1, "kal shaam": -1,
    # Hindi
    "आज": 0, "कल": -1, "परसों": -2, "बीते कल": -1, "कल रात": -1, "कल सुबह": -1, "कल शाम": -1,
    # Marathi
    "आज रोजी": 0, "काल": -1, "परवा": -2,
    # Bengali
    "আজ": 0, "গতকাল": -1, "গত কাল": -1, "পরশু": -2,
    # Gujarati
    "આજે": 0, "ગઈકાલે": -1, "ગઈ કાલે": -1, "પરમ દિવસે": -2,
    # Punjabi
    "ਅੱਜ": 0, "ਕੱਲ੍ਹ": -1, "ਬੀਤੇ ਕੱਲ੍ਹ": -1, "ਪਰਸੋਂ": -2,
    # Odia
    "ଆଜି": 0, "ଗତକାଲି": -1, "ପରଶୁ": -2,
    # Tamil
    "இன்று": 0, "நேற்று": -1, "முன்தினம்": -2,
    # Telugu
    "ఈరోజు": 0, "నిన్న": -1, "మొన్న": -2,
    # Kannada
    "ಇಂದು": 0, "ನಿನ್ನೆ": -1, "ಮೊನ್ನೆ": -2,
    # Malayalam
    "ഇന്ന്": 0, "ഇന്നലെ": -1, "മിനിഞ്ഞാന്ന്": -2,
    # Urdu
    "آج": 0, "کل": -1, "پرسوں": -2, "گزشتہ کل": -1,
}

# "X days/weeks/months ago", in the same set of languages. Kept separate from
# the fixed word table because the number is a capture group.
_AGO_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\b(\d{1,3})\s*(?:days?|din|दिन|दिवस|দিন|દિવસ|ਦਿਨ|ଦିନ|நாள்|நாட்கள்|రోజుల|ದಿನ|ദിവസ|دن)\s*"
                r"(?:ago|pehle|pahle|back|पहले|आधी|আগে|પહેલાં|ਪਹਿਲਾਂ|ପୂର୍ବେ|முன்பு|క్రితం|ಹಿಂದೆ|മുമ്പ്|پہلے)\b",
                re.IGNORECASE), 1),
    (re.compile(r"\b(\d{1,3})\s*(?:weeks?|hafte|hafta|सप्ताह|हफ्ते|আগে?সপ্তাহ|સપ્તાહ|ਹਫ਼ਤੇ|ସପ୍ତାହ|வாரங்கள்|వారాల|ವಾರ|ആഴ്ച|ہفتے)\s*"
                r"(?:ago|pehle|pahle|back|पहले|আগে|પહેલાં|ਪਹਿਲਾਂ|ପୂର୍ବେ|முன்பு|క్రితం|ಹಿಂದೆ|മുമ്പ്|پہلے)\b",
                re.IGNORECASE), 7),
)


def resolve_relative_date(value: str, today: date | None = None) -> date | None:
    """`value` as an absolute date when it is a relative day expression.

    Returns `None` for anything that isn't one, so callers can fall through to
    the ordinary absolute-date parsers unchanged.
    """
    text = (value or "").strip().lower().rstrip(".,!?।॥")
    if not text:
        return None
    reference = today or clock.today()
    offset = _RELATIVE_DAY_WORDS.get(text)
    if offset is not None:
        return reference + timedelta(days=offset)
    for pattern, multiplier in _AGO_PATTERNS:
        match = pattern.search(text)
        if match:
            return reference - timedelta(days=int(match.group(1)) * multiplier)
    return None


def is_relative_date_expression(value: str) -> bool:
    return resolve_relative_date(value) is not None
