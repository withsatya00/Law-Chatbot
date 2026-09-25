"""Problem-first draft discovery: the conversational wizard that runs BEFORE
template-specific field collection when a user asks for a draft without
naming one (see the "GOAL"/"DISCOVERY DATA MODEL" sections this module
implements). `app/drafting/conversation.py` owns the actual turn-by-turn state
machine; this module holds the data shapes and the deterministic, LLM-free
text-understanding helpers it calls.

Deliberately keyword/regex-based, not LLM-backed: `recommendation.py` (which
this feeds) must "work without an LLM after the matter profile has been
extracted", and building the profile itself with an LLM would both undermine
that determinism (tests would need to mock a model) and risk an LLM inferring
a role/relief the user never actually stated -- exactly the "never invent
facts" rule the rest of the drafting engine already follows. The trade-off is
narrower natural-language understanding than an LLM extractor would have;
discovery therefore leans on short, explicit follow-up questions (role/
relief/stage) rather than trying to parse everything from one free-text
paragraph.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from app.drafting.intent import _typo_tolerant_match
from app.drafting.templates.base import CASE_STAGES

# ---------------------------------------------------------------------------
# Matter profile
# ---------------------------------------------------------------------------


def default_matter_profile() -> dict[str, Any]:
    """The structured matter-profile shape from the spec's "DISCOVERY DATA
    MODEL" section, as a plain dict -- stored directly under
    `memory["matter_profile"]` (conversation memory is a plain dict
    persisted to Redis/Mongo, so a dataclass/pydantic model would just need
    converting back and forth on every turn for no benefit).
    """
    return {
        "domain": None,
        "subcategory": None,
        "issues": [],
        "user_role": None,
        "opposite_party_role": None,
        "desired_reliefs": [],
        "case_stage": None,
        "prior_actions": [],
        "forum": None,
        "jurisdiction": {"country": "IN", "state": None, "district": None},
        "urgency": None,
        # Not part of the spec's model verbatim, but needed by
        # `recommendation.recommend()`'s alias-match scoring and by the
        # discovery reply's "why this matches" text -- the raw first
        # description, kept verbatim (never altered/invented).
        "raw_description": "",
    }


# ---------------------------------------------------------------------------
# Deterministic signal extraction
# ---------------------------------------------------------------------------

# Canonical role -> substrings (English/Hindi/Hinglish) that name it. Only
# covers the roles the currently-metadata-tagged templates actually use
# (see the 8 templates onboarded in this pass) -- a role with no keyword
# entry here is still collectable by the explicit "are you X or Y?" question
# `conversation.py` asks when candidate templates disagree on it.
ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "landlord": (
        "landlord", "makan malik", "मकान मालिक", "ghar malik",
        "வீட்டு உரிமையாளர்", "ஓனர்",  # Tamil
        "ఇంటి యజమాని", "ఓనర్",  # Telugu
        "ಮನೆ ಮಾಲೀಕ",  # Kannada
        "বাড়িওয়ালা", "মালিক",  # Bengali
        "घरमालक",  # Marathi
        "મકાનમાલિક",  # Gujarati
        "مکان مالک",  # Urdu
        "വീട്ടുടമ",  # Malayalam
    ),
    "tenant": (
        "tenant", "kirayedar", "किरायेदार", "renter",
        "வாடகைதாரர்", "குடியிருப்பாளர்",  # Tamil
        "అద్దెదారు", "కిరాయిదారు",  # Telugu
        "ಬಾಡಿಗೆದಾರ",  # Kannada
        "ভাড়াটিয়া",  # Bengali
        "भाडेकरू",  # Marathi
        "ભાડુઆત",  # Gujarati
        "کرایہ دار",  # Urdu
        "വാടകക്കാരൻ",  # Malayalam
    ),
    "payee": ("payee", "cheque lenewala"),
    "creditor": ("creditor", "lender", "udhaar diya", "paisa diya tha"),
    "debtor": ("debtor", "borrower", "udhaar liya", "loan liya"),
    "drawer": ("drawer", "cheque issue kiya"),
    "complainant": (
        "complainant", "victim", "peedit", "पीड़ित",
        "புகார்தாரர்", "பாதிக்கப்பட்டவர்",  # Tamil
        "ఫిర్యాదిదారు", "బాధితుడు",  # Telugu
        "ದೂರುದಾರ", "ಸಂತ್ರಸ್ತ",  # Kannada
        "অভিযোগকারী", "ভুক্তভোগী",  # Bengali
        "तक्रारदार", "पीडित",  # Marathi
        "ફરિયાદી", "પીડિત",  # Gujarati
        "شکایت کنندہ", "متاثرہ",  # Urdu
        "പരാതിക്കാരൻ", "ഇര",  # Malayalam
    ),
    "consumer": (
        "consumer", "customer", "grahak", "ग्राहक", "khareeda",
        "நுகர்வோர்", "வாங்கியவர்",  # Tamil
        "వినియోగదారు",  # Telugu
        "ಗ್ರಾಹಕ",  # Kannada
        "ভোক্তা", "ক্রেতা",  # Bengali
        "ग्राहक",  # Marathi
        "ગ્રાહક",  # Gujarati
        "صارف", "گاہک",  # Urdu
        "ഉപഭോക്താവ്",  # Malayalam
    ),
    "seller": ("seller", "shopkeeper", "dukaandar", "vyapari"),
    "service_provider": ("service provider", "freelancer", "contractor", "consultant"),
    "client": ("client", "customer", "ग्राहक"),
    "aggrieved_person": ("aggrieved", "victim", "peedit"),
}

# The 8-language block appended to several entries below (Tamil, Telugu,
# Kannada, Bengali, Malayalam, Marathi, Gujarati, Urdu) is ONE common phrase
# per language, not the several English/Hindi/Hinglish variants those got --
# breadth (which of the 17 issues/13 reliefs/8 stages have ANY native
# phrasing at all) was prioritised over depth (every way to say it) given
# the size of this vocabulary. An issue/relief with no native block below
# still degrades safely: `_matches_canonical_or_keywords` (see
# `extract_issues`/`extract_reliefs`) also matches the bare English
# canonical term, and `_ask_discovery_question` in conversation.py always
# offers the still-unresolved slot as an explicit follow-up question either
# way -- nothing is ever silently unresolvable, just less likely to be
# caught from the user's very first free-text sentence.
ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "rent_default": (
        "rent nahi diya", "rent nahi de raha", "rent pending", "kiraya nahi diya", "किराया नहीं",
        "வாடகை தரவில்லை", "అద్దె ఇవ్వలేదు", "ಬಾಡಿಗೆ ಕೊಟ್ಟಿಲ್ಲ", "ভাড়া দেয়নি",
        "വാടക കൊടുത്തില്ല", "भाडे दिले नाही", "ભાડું આપ્યું નથી", "کرایہ نہیں دیا",
    ),
    "failure_to_vacate": (
        "vacate nahi", "ghar khali nahi", "khali nahi kar raha", "makan khali",
        "காலி செய்யவில்லை", "ఖాళీ చేయలేదు", "ಖಾಲಿ ಮಾಡಿಲ್ಲ", "খালি করেনি",
        "ഒഴിഞ്ഞില്ല", "रिकामे केले नाही", "ખાલી કર્યું નથી", "خالی نہیں کیا",
    ),
    "new_tenancy_setup": (
        "naya tenant", "new tenant", "rent par dena", "kiraye par",
        "புதிய வாடகைதாரர்", "కొత్త అద్దెదారు", "ಹೊಸ ಬಾಡಿಗೆದಾರ", "নতুন ভাড়াটিয়া",
        "പുതിയ വാടകക്കാരൻ", "नवीन भाडेकरू", "નવો ભાડુઆત", "نیا کرایہ دار",
    ),
    "cheque_bounce": (
        "cheque bounce", "check bounce", "chek bounce", "cheque dishonour", "चेक बाउंस",
        "காசோலை மறுப்பு", "చెక్ బౌన్స్", "ಚೆಕ್ ಬೌನ್ಸ್", "চেক বাউন্স",
        "ചെക്ക് ബൗൺസ്", "चेक बाऊन्स", "ચેક બાઉન્સ", "چیک باؤنس",
    ),
    "payment_default": (
        "paisa nahi diya", "payment nahi", "paisa wapas nahi", "amount due",
        "பணம் தரவில்லை", "డబ్బు ఇవ్వలేదు", "ಹಣ ಕೊಟ್ಟಿಲ್ಲ", "টাকা দেয়নি",
        "പണം തന്നില്ല", "पैसे दिले नाहीत", "પૈસા આપ્યા નથી", "پیسے نہیں دیے",
    ),
    "loan_default": (
        "loan wapas nahi", "udhaar wapas nahi", "loan default",
        "கடன் திரும்பவில்லை", "అప్పు తిరిగి ఇవ్వలేదు", "ಸಾಲ ಹಿಂತಿರುಗಿಸಿಲ್ಲ", "ঋণ ফেরত দেয়নি",
        "കടം തിരികെ തന്നില്ല", "कर्ज परत केले नाही", "લોન પાછી આપી નથી", "قرض واپس نہیں کیا",
    ),
    "theft": (
        "chori", "theft", "stolen", "चोरी",
        "திருட்டு", "దొంగతనం", "ಕಳ್ಳತನ", "চুরি", "മോഷണം", "चोरी", "ચોરી", "چوری",
    ),
    "fraud": (
        "fraud", "cheated", "dhoka", "धोखा", "thagi",
        "மோசடி", "మోసం", "ವಂಚನೆ", "প্রতারণা", "വഞ്ചന", "फसवणूक", "છેતરપિંડી", "دھوکہ",
    ),
    "assault": (
        "assault", "maara", "peeta", "hit me", "beaten",
        "தாக்குதல்", "దాడి", "ಹಲ್ಲೆ", "হামলা", "ആക്രമണം", "हल्ला", "હુમલો", "حملہ",
    ),
    "harassment": (
        "harassment", "pareshan", "परेशान",
        "துன்புறுத்தல்", "వేధింపులు", "ಕಿರುಕುಳ", "হয়রানি", "പീഡനം", "छळ", "પજવણી", "ہراساں کرنا",
    ),
    "cognizable_offence": ("fir", "police case"),
    "defective_goods": (
        "defective", "kharab product", "galat product", "damaged item", "kharab nikla", "kharab nikli",
        "பழுதான பொருள்", "లోపభూయిష్ట వస్తువు", "ದೋಷಪೂರಿತ ಸಾಮಾನು", "ত্রুটিপূর্ণ পণ্য",
        "കേടായ സാധനം", "सदोष वस्तू", "ખામીયુક્ત વસ્તુ", "خراب سامان",
    ),
    "deficient_service": (
        "service kharab", "deficient service", "service theek nahi",
        "குறைபாடான சேவை", "లోపభూయిష్ట సేవ", "ಕೊರತೆಯ ಸೇವೆ", "ত্রুটিপূর্ণ পরিষেবা",
        "കുറവുള്ള സേവനം", "त्रुटीपूर्ण सेवा", "ખામીયુક્ત સેવા", "ناقص خدمت",
    ),
    "unfair_trade_practice": ("cheated by seller", "overcharged", "fake product"),
    "domestic_violence": (
        "domestic violence", "ghar mein maar peet", "husband beats", "पत्नी को मारा", "घरेलू हिंसा",
        "குடும்ப வன்முறை", "గృహ హింస", "ಗೃಹ ಹಿಂಸೆ", "গার্হস্থ্য সহিংসতা",
        "ഗാർഹിക പീഡനം", "घरगुती हिंसाचार", "ઘરેલુ હિંસા", "گھریلو تشدد",
    ),
    "cruelty": (
        "cruelty", "torture", "प्रताड़ित",
        "கொடுமை", "క్రూరత్వం", "ಕ್ರೌರ್ಯ", "নিষ্ঠুরতা", "ക്രൂരത", "क्रूरता", "ક્રૂરતા", "ظلم",
    ),
    "new_engagement_setup": ("naya contract", "service lena hai", "hire karna hai"),
}

RELIEF_KEYWORDS: dict[str, tuple[str, ...]] = {
    "payment": (
        "paisa chahiye", "payment chahiye", "paisa wapas", "recovery of money", "wapas chahiye", "wapas mangna",
        "பணம் வேண்டும்", "డబ్బు కావాలి", "ಹಣ ಬೇಕು", "টাকা চাই",
        "പണം വേണം", "पैसे हवे", "પૈસા જોઈએ", "پیسے چاہیے",
    ),
    "vacant_possession": (
        "ghar khali", "vacate", "possession chahiye", "makan khali karwana", "khali karwana", "khali chahiye",
        "வீடு காலி", "ఇల్లు ఖాళీ", "ಮನೆ ಖಾಲಿ", "বাড়ি খালি",
        "വീട് ഒഴിഞ്ഞു", "घर रिकामे", "ઘર ખાલી", "گھر خالی",
    ),
    "formalize_tenancy": ("agreement banana", "rent agreement chahiye"),
    "fir_registration": (
        "fir darj", "fir register", "fir chahiye",
        "எஃப்ஐஆர் பதிவு", "ఎఫ్ఐఆర్ నమోదు", "ಎಫ್‌ಐಆರ್ ನೋಂದಣಿ", "এফআইআর নিবন্ধন",
        "എഫ്ഐആർ രജിസ്റ്റർ", "एफआयआर नोंदणी", "એફઆઈઆર નોંધણી", "ایف آئی آر درج",
    ),
    "investigation": (
        "investigation chahiye", "jaanch chahiye",
        "விசாரணை", "దర్యాప్తు", "ತನಿಖೆ", "তদন্ত", "അന്വേഷണം", "तपास", "તપાસ", "تحقیقات",
    ),
    "protection": (
        "protection chahiye", "suraksha chahiye",
        "பாதுகாப்பு", "రక్షణ", "ರಕ್ಷಣೆ", "সুরক্ষা", "സംരക്ഷണം", "संरक्षण", "સુરક્ષા", "تحفظ",
    ),
    "refund": (
        "refund chahiye", "paisa wapas chahiye", "deposit wapas",
        "பணத்தைத் திரும்பப் பெற", "రీఫండ్", "ಮರುಪಾವತಿ", "রিফান্ড",
        "റീഫണ്ട്", "परतावा", "રિફંડ", "رقم کی واپسی",
    ),
    "replacement": (
        "replace chahiye", "badalna chahiye",
        "மாற்று", "రీప్లేస్‌మెంట్", "ಬದಲಿ", "প্রতিস্থাপন", "മാറ്റിസ്ഥാപിക്കൽ", "बदली", "રિપ્લેસમેન્ટ", "تبدیلی",
    ),
    "compensation": (
        "compensation chahiye", "muawza chahiye", "हर्जाना",
        "இழப்பீடு", "పరిహారం", "ಪರಿಹಾರ", "ক্ষতিপূরণ",
        "നഷ്ടപരിഹാരം", "नुकसानभरपाई", "વળતર", "معاوضہ",
    ),
    "protection_order": ("protection order",),
    "residence_order": ("residence order", "ghar mein rehna"),
    "monetary_relief": ("monetary relief", "maintenance chahiye", "gujara bhatta"),
    "formalize_engagement": ("contract banana hai", "agreement karna hai"),
}

CASE_STAGE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pre_litigation": (
        "abhi tak kuch nahi", "sirf baat hui", "koi notice nahi", "nothing done yet", "no notice sent yet",
        "இதுவரை எதுவும் நடக்கவில்லை", "ఇప్పటివరకు ఏమీ జరగలేదు", "ಇಲ್ಲಿಯವರೆಗೆ ಏನೂ ಆಗಿಲ್ಲ",
        "এখনও কিছু হয়নি", "ഇതുവരെ ഒന്നും സംഭവിച്ചില്ല", "अजून काहीही झाले नाही",
        "હજુ સુધી કંઈ થયું નથી", "ابھی تک کچھ نہیں ہوا",
    ),
    "notice_sent": (
        "notice bhej diya", "notice de chuka", "already sent a notice", "notice send kar diya", "नोटिस भेज",
        "நோட்டீஸ் அனுப்பப்பட்டது", "నోటీసు పంపారు", "ನೋಟಿಸ್ ಕಳುಹಿಸಲಾಗಿದೆ", "নোটিশ পাঠানো হয়েছে",
        "നോട്ടീസ് അയച്ചു", "नोटीस पाठवली", "નોટિસ મોકલી", "نوٹس بھیجا",
    ),
    "reply_received": (
        "reply mila", "unhone jawab diya", "unka reply aaya", "response mila",
        "பதில் கிடைத்தது", "జవాబు వచ్చింది", "ಉತ್ತರ ಬಂದಿದೆ", "উত্তর পেয়েছি",
        "മറുപടി കിട്ടി", "उत्तर मिळाले", "જવાબ મળ્યો", "جواب مل گیا",
    ),
    "case_pending": (
        "case chal raha", "court mein case", "case pending", "matter is pending", "case daal diya",
        "வழக்கு நிலுவையில்", "కేసు పెండింగ్‌లో", "ಪ್ರಕರಣ ಬಾಕಿ ಇದೆ", "মামলা বিচারাধীন",
        "കേസ് നിലവിലുണ്ട്", "केस प्रलंबित", "કેસ પેન્ડિંગ", "کیس زیر التوا",
    ),
    "evidence_stage": ("evidence stage", "saboot pesh", "evidence de rahe"),
    "order_passed": (
        "order aa gaya", "faisla ho gaya", "order passed", "judgment aa gaya", "court ne order diya",
        "உத்தரவு பிறப்பிக்கப்பட்டது", "ఆర్డర్ వచ్చింది", "ಆದೇಶ ಬಂದಿದೆ", "আদেশ এসেছে",
        "ഉത്തരവ് വന്നു", "आदेश आला", "ઓર્ડર આવ્યો", "حکم آ گیا",
    ),
    "appeal": (
        "appeal karni hai", "appeal file karni", "appeal chahiye",
        "மேல்முறையீடு", "అప్పీల్", "ಮೇಲ್ಮನವಿ", "আপিল", "അപ്പീൽ", "अपील", "અપીલ", "اپیل",
    ),
    "execution": ("execution", "vasuli karni hai order ke baad", "decree ka execution"),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _fuzzy_find(normalized: str, keyword: str) -> int:
    """Like `str.find`, but tolerates one ordinary typo per word (a missing/
    doubled/swapped/wrong letter -- "landlrod" for "landlord", "tenent" for
    "tenant", "rent nai diya" for "rent nahi diya"). A multi-word `keyword`
    is matched by sliding its word count across `normalized`'s own words and
    checking each position pairs up (exactly or via one typo) in order --
    still anchored to a contiguous, correctly-ORDERED run of words, so this
    stays a typo correction (one wrong letter per word) rather than a fuzzy
    word-reordering/omission search. Returns the position of the first word
    in the best match, or -1, exactly like `str.find`.

    Found via live testing: a self-identification typo is the worst kind of
    miss here, not a merely-inconvenient one -- "landlrod hoon, tenant rent
    nai de rha" (a landlord typing their own role with one letter
    transposed) matched NO role keyword for "landlord" at all, so
    `extract_role` fell through to the "earliest OTHER role mentioned"
    fallback and confidently returned "tenant" -- the wrong party, not just
    an unresolved one.
    """
    exact = normalized.find(keyword)
    if exact != -1:
        return exact
    keyword_words = keyword.split(" ")
    if any(len(w) < 4 for w in keyword_words):
        # A short word (an article, a postposition, "hai"/"nahi") is too
        # easy to fuzzy-match by coincidence -- every word in a multi-word
        # phrase must itself be long enough for `_typo_tolerant_match`'s own
        # length gate, or this phrase is only ever matched exactly.
        return -1
    message_words = [m.group() for m in re.finditer(r"[^\s,.!?;:।]+", normalized)]
    message_starts = [m.start() for m in re.finditer(r"[^\s,.!?;:।]+", normalized)]
    span = len(keyword_words)
    for start in range(len(message_words) - span + 1):
        if all(
            _typo_tolerant_match(keyword_words[offset], message_words[start + offset])
            for offset in range(span)
        ):
            return message_starts[start]
    return -1


def _fuzzy_contains(normalized: str, keyword: str) -> bool:
    return _fuzzy_find(normalized, keyword) != -1


def extract_role(text: str) -> str | None:
    """The role the SPEAKER is claiming for themselves -- not just any role
    word that appears in the text.

    A message naming the user's own problem routinely mentions the other
    party's role too ("Tenant hoon, landlord mera security deposit wapas
    nahi de raha" -- "I'm the tenant, the landlord isn't returning my
    deposit"), so a plain "does this role keyword appear anywhere" check
    picks whichever role happens to be defined first in `ROLE_KEYWORDS`
    regardless of which one the speaker actually claimed. Two passes fix
    this: first look for an explicit self-identification pattern ("X hoon",
    "main X", "i am X", "as a X") for ANY role, which settles it outright
    however the two roles are ordered in the sentence; only if neither role
    is self-identified this way does it fall back to whichever role word
    appears EARLIEST in the text (people overwhelmingly state their own
    situation before the other party's).
    """
    normalized = _normalize(text)
    for role, keywords in ROLE_KEYWORDS.items():
        for keyword in keywords:
            pattern = keyword.replace(" ", r"\s+")
            if re.search(rf"\bmain\s+{pattern}\s+hoon\b", normalized):
                return role
            if re.search(rf"\b{pattern}\s+hoon\b", normalized):
                return role
            if re.search(rf"\bi\s*a?m\s+(a\s+|an\s+)?{pattern}\b", normalized):
                return role
            if re.search(rf"\bas\s+(a\s+|an\s+)?{pattern}\b", normalized):
                return role

    best_role: str | None = None
    best_index: int | None = None
    for role, keywords in ROLE_KEYWORDS.items():
        for keyword in keywords:
            index = _fuzzy_find(normalized, keyword)
            if index != -1 and (best_index is None or index < best_index):
                best_index = index
                best_role = role
    return best_role


def _matches_canonical_or_keywords(normalized: str, canonical: str, keywords: tuple[str, ...]) -> bool:
    """True if `normalized` contains any of `keywords`, OR the canonical
    identifier itself written as words (e.g. "vacant_possession" ->
    "vacant possession"). The canonical form is exactly what the discovery
    reply's own option list shows the user (see `_ask_discovery_question` in
    conversation.py, which renders `role.replace("_", " ")`), so a user who
    answers by echoing one of the options back verbatim -- or simply typing
    the plain English term -- is understood even for a language whose
    `ROLE_KEYWORDS`/`ISSUE_KEYWORDS`/`RELIEF_KEYWORDS` entry has no native
    phrasing yet.
    """
    if any(_fuzzy_contains(normalized, keyword) for keyword in keywords):
        return True
    return _fuzzy_contains(normalized, canonical.replace("_", " "))


def extract_issues(text: str) -> list[str]:
    normalized = _normalize(text)
    return [
        issue for issue, keywords in ISSUE_KEYWORDS.items()
        if _matches_canonical_or_keywords(normalized, issue, keywords)
    ]


def extract_reliefs(text: str) -> list[str]:
    normalized = _normalize(text)
    return [
        relief for relief, keywords in RELIEF_KEYWORDS.items()
        if _matches_canonical_or_keywords(normalized, relief, keywords)
    ]


def extract_case_stage(text: str) -> str | None:
    normalized = _normalize(text)
    # Checked in this fixed order (not stage-ascending) so a message that
    # incidentally contains an earlier-stage phrase inside a later-stage one
    # ("order aa gaya, ab execution karni hai") resolves to the LATEST stage
    # actually reached -- order matters here: later entries below win ties by
    # being checked last and overwriting `found`.
    found: str | None = None
    for stage in CASE_STAGES:
        keywords = CASE_STAGE_KEYWORDS.get(stage, ())
        if _matches_canonical_or_keywords(normalized, stage, keywords):
            found = stage
    return found


@dataclass(frozen=True)
class ExtractedSignals:
    role: str | None = None
    issues: list[str] = field(default_factory=list)
    reliefs: list[str] = field(default_factory=list)
    case_stage: str | None = None


def extract_signals(text: str) -> ExtractedSignals:
    return ExtractedSignals(
        role=extract_role(text),
        issues=extract_issues(text),
        reliefs=extract_reliefs(text),
        case_stage=extract_case_stage(text),
    )


def apply_signals_to_profile(profile: dict[str, Any], text: str) -> dict[str, Any]:
    """Merges whatever `extract_signals` finds in `text` into `profile`,
    additively -- never overwrites a slot the profile already has (matches
    the "don't repeatedly ask for / discard information already stored"
    rule), and never invents a value `text` didn't actually contain."""
    signals = extract_signals(text)
    if signals.role and not profile.get("user_role"):
        profile["user_role"] = signals.role
    if signals.issues:
        profile["issues"] = sorted({*profile.get("issues", []), *signals.issues})
    if signals.reliefs:
        profile["desired_reliefs"] = sorted({*profile.get("desired_reliefs", []), *signals.reliefs})
    if signals.case_stage:
        # Case stage is the one slot allowed to be UPDATED by a later
        # message (not just filled once) -- "notice bhej diya" superseding
        # an earlier "abhi tak kuch nahi" reflects the matter having moved
        # on, not a correction to be ignored.
        profile["case_stage"] = signals.case_stage
    return profile


# ---------------------------------------------------------------------------
# Jurisdiction (state/district) parsing
# ---------------------------------------------------------------------------

# Not an exhaustive gazetteer -- the 28 states + 8 union territories of India,
# canonical name first, plus a handful of common short forms actually typed
# in chat. Good enough to recognise a plainly-named state; anything else is
# kept as free text and the jurisdiction is marked unverified rather than
# guessed at (see `parse_jurisdiction_reply`).
INDIAN_STATES: dict[str, str] = {}
for _canonical, *_aliases in [
    ["Andhra Pradesh", "ap", "आंध्र प्रदेश"], ["Arunachal Pradesh", "अरुणाचल प्रदेश"],
    ["Assam", "असम"], ["Bihar", "बिहार"],
    ["Chhattisgarh", "cg", "छत्तीसगढ़"], ["Goa", "गोवा"], ["Gujarat", "गुजरात"], ["Haryana", "हरियाणा"],
    ["Himachal Pradesh", "hp", "हिमाचल प्रदेश"], ["Jharkhand", "झारखंड"],
    ["Karnataka", "कर्नाटक"], ["Kerala", "केरल"],
    ["Madhya Pradesh", "mp", "मध्य प्रदेश"], ["Maharashtra", "महाराष्ट्र"],
    ["Manipur", "मणिपुर"], ["Meghalaya", "मेघालय"],
    ["Mizoram", "मिज़ोरम"], ["Nagaland", "नागालैंड"],
    ["Odisha", "orissa", "ओडिशा"], ["Punjab", "पंजाब"], ["Rajasthan", "राजस्थान"],
    ["Sikkim", "सिक्किम"], ["Tamil Nadu", "tn", "तमिलनाडु"], ["Telangana", "ts", "तेलंगाना"],
    ["Tripura", "त्रिपुरा"],
    ["Uttar Pradesh", "up", "उत्तर प्रदेश"], ["Uttarakhand", "uk", "उत्तराखंड"],
    ["West Bengal", "wb", "पश्चिम बंगाल"],
    ["Andaman and Nicobar Islands", "अंडमान और निकोबार द्वीप समूह"], ["Chandigarh", "चंडीगढ़"],
    ["Dadra and Nagar Haveli and Daman and Diu", "दादरा और नगर हवेली और दमन और दीव"],
    ["Delhi", "ncr", "new delhi", "दिल्ली"],
    ["Jammu and Kashmir", "j&k", "jk", "जम्मू और कश्मीर"], ["Ladakh", "लद्दाख"],
    ["Lakshadweep", "लक्षद्वीप"], ["Puducherry", "pondicherry", "पुडुचेरी"],
]:
    for name in (_canonical, *_aliases):
        INDIAN_STATES[name.lower()] = _canonical


def parse_jurisdiction_reply(text: str) -> dict[str, Any]:
    """Best-effort split of a free-text jurisdiction reply ("Lucknow, Uttar
    Pradesh" / "Delhi" / "Pune, Maharashtra") into state/district.

    Never guesses a state that wasn't named: if no recognised state name (or
    alias) appears anywhere in the text, `state` is left `None` and the whole
    reply is kept as `district` (free text) so nothing the user typed is
    discarded -- the caller marks such jurisdictions as needing verification
    rather than treating this as a confident match.
    """
    normalized = _normalize(text)
    state: str | None = None
    matched_alias: str | None = None
    # Longest alias first, so a full name ("uttar pradesh") is preferred over
    # a short one that happens to also be a substring-adjacent token ("up")
    # when both could plausibly match -- avoids stripping only the shorter
    # alias out of the text and leaving the rest of the state's own name
    # sitting in what becomes the district.
    for alias in sorted(INDIAN_STATES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            state = INDIAN_STATES[alias]
            matched_alias = alias
            break
    if state is None or matched_alias is None:
        return {"country": "IN", "state": None, "district": text.strip() or None, "verified": False}
    # The matched alias is removed from the ORIGINAL (not lower-cased) text
    # -- not just split on comma/slash -- so a reply with no punctuation at
    # all ("UP aur maharanjang") still yields a clean district instead of
    # the state's own name sitting unremoved inside it (confirmed live: the
    # previous comma-only split left `district` equal to the WHOLE input,
    # so the pre-generation summary showed "Jurisdiction: UP aur
    # maharanjang, Uttar Pradesh" -- the state name duplicated).
    remainder = re.sub(rf"\b{re.escape(matched_alias)}\b", " ", text, flags=re.IGNORECASE)
    remainder = re.sub(r"[,/]", " ", remainder)
    remainder = re.sub(r"\s+", " ", remainder).strip(" -")
    district = remainder or None
    return {"country": "IN", "state": state, "district": district, "verified": True}


# ---------------------------------------------------------------------------
# Category browsing taxonomy (secondary, manual-browse route)
# ---------------------------------------------------------------------------

TOP_LEVEL_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("court_and_litigation", "Court and Litigation"),
    ("personal_and_family", "Personal and Family"),
    ("property", "Property"),
    ("business", "Business"),
    ("regulatory", "Regulatory"),
    ("general_documents", "General Documents"),
)

# Translated display names for `TOP_LEVEL_CATEGORIES`, keyed the same way
# `description_translations.py`/`title_translations.py` key their own
# per-item translation tables. Found via live testing: the category-browse
# list ("browse category") stayed in English even inside an otherwise fully
# Tamil/Hindi/... discovery reply, since these six names were interpolated
# as plain strings straight from `TOP_LEVEL_CATEGORIES` with no language
# lookup at all.
_CATEGORY_NAME_TRANSLATIONS: dict[str, dict[str, str]] = {
    "court_and_litigation": {
        "hindi": "न्यायालय और मुकदमा", "hinglish": "Court aur litigation",
        "tamil": "நீதிமன்றம் மற்றும் வழக்கு", "telugu": "న్యాయస్థానం మరియు వ్యాజ్యం",
        "kannada": "ನ್ಯಾಯಾಲಯ ಮತ್ತು ವ್ಯಾಜ್ಯ", "bengali": "আদালত ও মামলা",
        "malayalam": "കോടതിയും വ്യവഹാരവും", "marathi": "न्यायालय आणि खटला",
        "gujarati": "કોર્ટ અને મુકદ્દમા", "punjabi": "ਅਦਾਲਤ ਅਤੇ ਮੁਕੱਦਮਾ",
        "odia": "ନ୍ୟାୟାଳୟ ଏବଂ ମକଦ୍ଦମା", "urdu": "عدالت اور مقدمہ بازی",
    },
    "personal_and_family": {
        "hindi": "व्यक्तिगत और पारिवारिक", "hinglish": "Personal aur family",
        "tamil": "தனிப்பட்ட மற்றும் குடும்பம்", "telugu": "వ్యక్తిగత మరియు కుటుంబం",
        "kannada": "ವೈಯಕ್ತಿಕ ಮತ್ತು ಕುಟುಂಬ", "bengali": "ব্যক্তিগত ও পারিবারিক",
        "malayalam": "വ്യക്തിപരവും കുടുംബവും", "marathi": "वैयक्तिक आणि कौटुंबिक",
        "gujarati": "અંગત અને પારિવારિક", "punjabi": "ਨਿੱਜੀ ਅਤੇ ਪਰਿਵਾਰਕ",
        "odia": "ବ୍ୟକ୍ତିଗତ ଏବଂ ପାରିବାରିକ", "urdu": "ذاتی اور خاندانی",
    },
    "property": {
        "hindi": "संपत्ति", "hinglish": "Property",
        "tamil": "சொத்து", "telugu": "ఆస్తి", "kannada": "ಆಸ್ತಿ", "bengali": "সম্পত্তি",
        "malayalam": "വസ്തു", "marathi": "मालमत्ता", "gujarati": "મિલકત",
        "punjabi": "ਜਾਇਦਾਦ", "odia": "ସମ୍ପତ୍ତି", "urdu": "جائیداد",
    },
    "business": {
        "hindi": "व्यवसाय", "hinglish": "Business",
        "tamil": "வணிகம்", "telugu": "వ్యాపారం", "kannada": "ವ್ಯಾಪಾರ", "bengali": "ব্যবসা",
        "malayalam": "ബിസിനസ്", "marathi": "व्यवसाय", "gujarati": "વ્યવસાય",
        "punjabi": "ਕਾਰੋਬਾਰ", "odia": "ବ୍ୟବସାୟ", "urdu": "کاروبار",
    },
    "regulatory": {
        "hindi": "नियामक", "hinglish": "Regulatory",
        "tamil": "ஒழுங்குமுறை", "telugu": "నియంత్రణ", "kannada": "ನಿಯಂತ್ರಕ", "bengali": "নিয়ন্ত্রক",
        "malayalam": "നിയന്ത്രണം", "marathi": "नियामक", "gujarati": "નિયમનકારી",
        "punjabi": "ਰੈਗੂਲੇਟਰੀ", "odia": "ନିୟାମକ", "urdu": "ریگولیٹری",
    },
    "general_documents": {
        "hindi": "सामान्य दस्तावेज़", "hinglish": "General documents",
        "tamil": "பொது ஆவணங்கள்", "telugu": "సాధారణ పత్రాలు", "kannada": "ಸಾಮಾನ್ಯ ದಾಖಲೆಗಳು",
        "bengali": "সাধারণ নথি", "malayalam": "പൊതു രേഖകൾ", "marathi": "सामान्य कागदपत्रे",
        "gujarati": "સામાન્ય દસ્તાવેજો", "punjabi": "ਆਮ ਦਸਤਾਵੇਜ਼", "odia": "ସାଧାରଣ ଦଲିଲ",
        "urdu": "عام دستاویزات",
    },
}


def localized_category_name(category_id: str, language: str) -> str:
    """Display name for a `TOP_LEVEL_CATEGORIES` entry in `language`. Falls
    back to the English name for any language not covered above or any
    unrecognised `category_id`."""
    english = dict(TOP_LEVEL_CATEGORIES).get(category_id, category_id)
    return _CATEGORY_NAME_TRANSLATIONS.get(category_id, {}).get(language) or english

# Maps a top-level browse category to the `DraftTemplateDefinition.domain`
# values filed under it. Only the templates onboarded with discovery
# metadata so far carry a `domain` at all (see loader.py/README note in
# `recommendation.py`); templates with no `domain` yet simply don't surface
# under category browsing until a later pass tags them -- they remain fully
# reachable via direct name / search, per the "onboard more templates later"
# design note.
CATEGORY_DOMAIN_MAP: dict[str, tuple[str, ...]] = {
    "court_and_litigation": ("criminal",),
    "personal_and_family": ("family",),
    "property": ("property",),
    "business": ("business",),
    "regulatory": ("financial", "consumer"),
    "general_documents": (),
}
