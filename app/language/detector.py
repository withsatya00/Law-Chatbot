import re
import unicodedata

try:
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException
except ModuleNotFoundError:
    DetectorFactory = None
    detect = None

    # Named so the `except` clauses below stay valid when langdetect is absent
    # (they are unreachable in that case -- `detect` is None and never called).
    class LangDetectException(Exception):  # type: ignore[no-redef]
        pass

if DetectorFactory is not None:
    DetectorFactory.seed = 7

# U+0964 DEVANAGARI DANDA ("\u0964") and U+0965 DOUBLE DANDA ("\u0965") live inside the
# Devanagari Unicode block but are SHARED sentence-final punctuation used by
# Gujarati, Bengali, Odia, Punjabi, Marathi and most other Indic scripts. A
# naive full-block `[\u0900-\u097F]` test therefore fired on a pure Gujarati
# sentence that merely ends in "\u0964", sending it down the Devanagari branch of
# `detect()` -- which resolves to Hindi for anything langdetect doesn't call
# mr/ne. Confirmed root cause of a live bug: "\u0AAE\u0ABE\u0AB0\u0AC0 \u0AB8\u0ABE\u0AA5\u0AC7 \u0A9B\u0AC7\u0AA4\u0AB0\u0AAA\u0ABF\u0A82\u0AA1\u0AC0 \u0AA5\u0A88 \u0A9B\u0AC7.
# \u0AAB\u0AB0\u0ABF\u0AAF\u0ABE\u0AA6 \u0AA4\u0AC8\u0AAF\u0ABE\u0AB0 \u0A95\u0AB0\u0ACB\u0964" (entirely Gujarati, one trailing danda) was detected as
# Hindi and the whole drafting conversation was forced into Hindi.
# `DEVANAGARI_PATTERN` now requires an actual Devanagari LETTER/matra;
# `DEVANAGARI_ANY_PATTERN` keeps the original full-block range for anything
# that genuinely wants "any Devanagari codepoint, punctuation included".
DEVANAGARI_PATTERN = r"[\u0900-\u0963\u0966-\u097F]"
DEVANAGARI_ANY_PATTERN = r"[\u0900-\u097F]"

SCRIPT_LANGUAGE_PATTERNS: list[tuple[str, str]] = [
    (r"[஀-௿]", "tamil"),
    (r"[ఀ-౿]", "telugu"),
    (r"[ಀ-೿]", "kannada"),
    (r"[ഀ-ൿ]", "malayalam"),
    (r"[઀-૿]", "gujarati"),
    (r"[਀-੿]", "punjabi"),
    (r"[ঀ-৿]", "bengali"),
    (r"[଀-୿]", "odia"),
    # Eighth Schedule scripts used by exactly one language each, so script
    # presence alone identifies them -- unlike Devanagari and Perso-Arabic
    # below, which several of these languages share.
    (r"[ꯀ-꯿]", "manipuri"),   # Meetei Mayek
    (r"[᱐-᱿]", "santali"),    # Ol Chiki
    # Perso-Arabic stays last: Urdu, Kashmiri and Sindhi all use it, so a bare
    # range match only narrows it to those three. `_perso_arabic_language()`
    # picks between them; plain Urdu remains the fallback.
    (r"[؀-ۿ]", "urdu"),
]

# Letters present in Sindhi's and Kashmiri's Perso-Arabic alphabets but absent
# from Urdu's, so a single occurrence identifies the language. Sindhi's are its
# implosives and four-dot retroflexes; Kashmiri's are its extra vowel forms.
# Letter-based rather than word-based on purpose: these are alphabet
# extensions, so they recur constantly in ordinary text of that language and
# never in ordinary Urdu.
_SINDHI_ONLY_LETTERS = frozenset("ڪڄڃڇڊڏڌڍڦڱڳڻٺٽٿڀ")
_KASHMIRI_ONLY_LETTERS = frozenset("ٲٳۆۄؠٟٕٚٛۍ")


def _devanagari_outweighs_other_scripts(text: str) -> bool:
    """Whether `text` should be resolved down `detect()`'s Devanagari branch.

    `detect()` checks Devanagari FIRST (eight supported languages share it, so
    it needs its own marker-word disambiguation), and every other Indic script
    only afterwards. That ordering is only safe when a stray Devanagari
    character can't outvote a message written overwhelmingly in another
    script -- which is exactly what happened before `DEVANAGARI_PATTERN`
    excluded the shared danda, and what would happen again for any message
    that quotes one Hindi word inside an otherwise Gujarati/Bengali/Tamil
    sentence. Comparing character COUNTS makes the choice proportional
    instead of order-dependent: Devanagari wins only when it's genuinely the
    dominant script, and a single borrowed word never flips the whole reply's
    language.
    """
    devanagari_count = len(re.findall(DEVANAGARI_PATTERN, text))
    if not devanagari_count:
        return False
    other_count = max(
        (len(re.findall(pattern, text)) for pattern, _ in SCRIPT_LANGUAGE_PATTERNS),
        default=0,
    )
    return devanagari_count >= other_count


def _perso_arabic_language(text: str) -> str:
    """Urdu vs Sindhi vs Kashmiri -- all three share the Perso-Arabic script."""
    characters = set(text)
    if characters & _SINDHI_ONLY_LETTERS:
        return "sindhi"
    if characters & _KASHMIRI_ONLY_LETTERS:
        return "kashmiri"
    return "urdu"


# Eight supported languages are written in Devanagari -- Hindi, Marathi,
# Nepali, Konkani, Maithili, Dogri, Bodo and Sanskrit -- so unlike every entry
# in SCRIPT_LANGUAGE_PATTERNS above, the script cannot identify which one a
# message is in. langdetect only ships models for three of them (hi/mr/ne), so
# the rest are recognized by high-frequency function words that do not occur
# in ordinary Hindi.
#
# These are markers, not a language model: chosen for PRECISION (a hit is
# strong evidence) rather than recall. Anything with no hit falls through to
# langdetect and then to Hindi -- exactly the behaviour that existed before.
# Single-character words are excluded even where they are the most frequent
# word in the language (Nepali "chha", Sanskrit "cha"), since a lone
# Devanagari letter turns up too easily inside ordinary Hindi text.
_DEVANAGARI_MARKER_WORDS: tuple[tuple[str, frozenset[str]], ...] = (
    ("sanskrit", frozenset({
        "अस्ति", "भवति", "इति", "एव",
        "तत्र", "यत्र", "वयम्", "अहम्",
        "तस्य", "सर्वम्", "किमपि",
        "उपलभ्यते", "सम्प्रति",
    })),
    ("marathi", frozenset({
        "आहे", "आहेत", "नाही", "मला",
        "तुम्ही", "माझ्या",
        "कोणतेही", "साठी",
        "पाहिजे", "कृपया",
        # Phase 1 item 1: the original marker set covered declarative Marathi
        # ("...आहे") but not the interrogative/imperative forms a QUESTION
        # uses, so "पोलीस तक्रार कशी नोंदवावी?" carried no Marathi marker at
        # all, fell through to langdetect, and came back Hindi -- meaning a
        # Marathi speaker was answered in Hindi. All of these are ordinary
        # high-frequency Marathi words with no Hindi reading.
        "काय", "कसा", "कशी", "कसे",
        "करावा", "करावी", "करावे", "मिळवावा",
        "तक्रार", "आम्ही", "त्यांनी", "झाली", "झाले",
    })),
    ("nepali", frozenset({
        "छैन", "छन्", "गर्नु",
        "गर्ने", "हुन्छ",
        "तपाईं", "भएको", "गरेको",
        "पनि", "हामी", "कुनै",
    })),
    ("maithili", frozenset({
        "अछि", "अहाँ", "हमर", "छथि",
        "एहि", "कोनो", "नहि", "भेल",
    })),
    ("konkani", frozenset({
        "आसा", "म्हाका", "तुका",
        "कित्याक", "हांव", "खंय",
        "आनी", "जाल्यार",
    })),
    ("dogri", frozenset({
        "कन्ने", "नेईं", "मिंजो",
        "तुस", "हून", "दिक्खो",
        "गल्ल", "ऐह",
    })),
    ("bodo", frozenset({
        "थानाय", "मोननो",
        "नोंथाङ", "आरो", "खौ",
        "बादि", "मानो", "आव",
        "खालामनाय",
    })),
)


def _devanagari_language(text: str) -> str | None:
    """Best-supported Devanagari language for `text`, or None when no marker
    word is present (caller then falls back to langdetect, then Hindi).

    Scored rather than first-match, so a message carrying markers from two
    tables goes to whichever language has more evidence behind it rather than
    to whichever happens to be listed first.
    """
    words = set(re.findall(r"[ऀ-ॿ]+", text))
    if not words:
        return None
    best_language, best_score = None, 0
    for language, markers in _DEVANAGARI_MARKER_WORDS:
        score = len(words & markers)
        if score > best_score:
            best_language, best_score = language, score
    return best_language

# Romanized Hindi function/pronoun/verb words with no legitimate standalone
# English reading -- presence of even ONE is a reliable signal on its own.
# Part 39's own flagship example ("Teacher ne mujhe mara" / "Mujhe kisne
# mara tha?") exposed how narrow the original list was: it required one of
# only 4 words (nahi/hai/kya/mera) to CONFIRM a match, and neither sentence
# contains any of them despite being unambiguously Hinglish -- both fell
# through to `langdetect`, which misreads short romanized Hindi as English
# often enough to be the actual root cause of "assistant replies in English"
# complaints, not just an edge case.
_STRONG_HINGLISH_TERMS = {
    "mera", "meri", "mere", "mujhe", "tumhe", "humein", "hume", "nahi", "nahin", "hai", "hain", "kya", "kyun", "kyu", "kaise",
    "kahan", "kaha", "kab", "kaun", "kisne", "kiska", "kiski", "kisko",
    "tha", "thi", "hua", "hui", "hue", "gaya", "gayi", "gaye",
    "kiya", "diya", "liya", "raha", "rahi", "rahe", "chahiye", "matlab",
    "accha", "theek", "bilkul", "sirf", "abhi", "phir", "wapas", "apna",
    "apne", "uska", "uski", "unka", "jiska", "kisi", "koi", "hoga",
    "hogi", "honge", "karu", "kru", "karoon", "karo", "kro", "krna", "karna",
    "banao", "banana", "banani", "likhna", "likhni", "mara", "maara",
    # `ko` (a postposition with no standalone English reading) and `ke`
    # (likewise -- "ke bina," "ke liye") were missing despite being two of
    # the most common Hindi function words; their absence, combined with
    # `kro` (a common colloquial spelling of "karo") also being missing,
    # was the confirmed root cause of genuinely Hinglish messages ("fir ko
    # 30 words me explain kro," "Bail ko simple language me samjhao...")
    # having zero token-level signal and falling through to `langdetect`,
    # which read their majority-English vocabulary as plain English.
    "ko", "ke",
}
# Words that overlap with legitimate standalone English (a genuine English
# sentence can contain "police," "case," or "salary" on its own) -- these
# only count as Hinglish evidence together with a second Hinglish signal,
# so a single ambiguous English word never mis-triggers on its own.
# "the" moved here from the strong list: as a romanized-Hindi past-tense
# marker ("log ache the" -- "people were good") it's a legitimate Hinglish
# signal, but it's also the single most common word in English, so listing
# it as a STRONG (single-word-triggers) signal meant almost any English
# sentence ("What is the difference between FIR and NCR?") false-triggered
# Hinglish detection purely because it contains "the" -- confirmed root
# cause of a live regression where a plain English question got answered in
# Hindi. Needs a second Hinglish signal alongside it now, same as the other
# ambiguous entries below.
_WEAK_HINGLISH_TERMS = {"salary", "kiraya", "police", "case", "shaadi", "talak", "paisa", "the"}

# Romanized Meetei/Manipuri function words with no legitimate standalone
# English reading -- the same precision-first approach as
# `_STRONG_HINGLISH_TERMS` above. Manipuri has a native script
# (`SCRIPT_LANGUAGE_PATTERNS`, Meetei Mayek) but is very commonly typed in
# Latin transliteration instead, which carries no script signal at all and
# previously fell straight through to `langdetect` -- which has no Manipuri
# model and defaults unrecognized codes to English (`LANGDETECT_MAP.get(...,
# "english")`). Confirmed root cause of BUG-110: "check bounce oirabadi India
# da kari punishment oi?" was detected as English, and its Manipuri token
# "kari" ("what") was then treated by `app/language/typo_tolerance.py`'s
# Hindi-verb allowlist as a typo of "kar"/"karo"/"karti", producing a
# nonsensical Hindi spell-check prompt for a Manipuri speaker.
_ROMANIZED_MANIPURI_TERMS = {
    "oirabadi", "oirabasu", "leitrabadi", "haibadi", "haibadu", "haibasi",
    "adudagi", "natragadi", "khudingmakki", "karino", "kadaida", "kanana",
    "kayada", "amasung", "nahakna", "eigi", "nanggi", "mahakki", "makhoigi",
    "kari",
}

LANGDETECT_MAP = {
    "en": "english",
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "gu": "gujarati",
    "mr": "marathi",
    "pa": "punjabi",
    "bn": "bengali",
    "or": "odia",
    "ur": "urdu",
    "as": "assamese",
    # langdetect ships models for exactly one of the newly supported Eighth
    # Schedule languages. Bodo/Dogri/Konkani/Maithili/Sanskrit/Santali/
    # Manipuri/Kashmiri/Sindhi have no langdetect model at all and are
    # identified by script range or marker word instead -- see
    # `_devanagari_language()` and `_perso_arabic_language()`.
    "ne": "nepali",
}


# langdetect's own docs note it's unreliable on short strings; a bare
# follow-up ("Why?", "Fir?", "Uske baad?", "Kitna time?") carries almost no
# signal either way, so trusting a fresh per-message detection on one would
# mean a coin-flip could silently switch the whole conversation's language
# (Part 39 rule 1: "never randomly switch language" / "follow-ups inherit
# previous language unless user changes it"). Chosen to comfortably cover
# every short-follow-up example in the spec ("kitna kharcha?" is 14 chars)
# while still letting a genuinely longer message flip languages.
_SHORT_TEXT_INHERIT_THRESHOLD = 20

# A short message can still carry its own clear signal despite being under
# the length threshold above: a complete English question ("What is bail?",
# 14 chars) is not the kind of ambiguous fragment ("Why?", "Fir?") the
# threshold was built for -- it names its own subject and has an unmistakably
# English grammatical shape. Requires the starter word PLUS at least 4 more
# characters so a bare "Why?"/"How?" (genuinely no signal beyond the word
# itself, and an existing regression-tested inherit case) still doesn't
# match. Confirmed root cause of a live regression: "What is bail?" right
# after a Hindi turn was silently answered in Hindi because it inherited
# instead of being freshly detected.
_ENGLISH_QUESTION_STARTER_PATTERN = re.compile(
    r"^(what|why|how|who|when|where|which|can|could|is|are|does|do|will|should)\b.{4,}", re.IGNORECASE
)


# Part 52 "Multilingual Draft Engine": names/aliases a user might use to
# explicitly request a document be drafted (or re-drafted) in a specific
# language -- distinct from `LANGDETECT_MAP`, which classifies which
# language a message is WRITTEN in. "bangla"/"oriya" are common colloquial
# aliases for bengali/odia that langdetect/ISO codes never produce, so they
# only belong here, not in `LANGDETECT_MAP`.
#
# Part 55 "Draft Audit -- Language Propagation Fix": previously only each
# language's OWN script (plus the plain English name) was listed here --
# "tamil"/"தமிழ்" but never "तमिल" (the Devanagari spelling a Hindi-typing
# user naturally reaches for). A message like "तमिल में पुलिस शिकायत का
# मसौदा तैयार करें।" ("prepare a police complaint draft IN TAMIL," typed
# entirely in Hindi/Devanagari) therefore matched nothing here at all,
# `extract_requested_language` returned `None`, and the whole draft session
# silently fell back to the message's own script-detected language (Hindi)
# instead of the language actually requested -- confirmed root cause of a
# live "explicit Tamil request still drafts in Hindi" bug report. Fixed by
# adding each of the five other-script spellings for every one of this
# app's five fully-localized non-English languages (hindi/tamil/telugu/
# kannada/bengali) plus english -- i.e. every language nameable in every
# other language's own script, not just its own.
_LANGUAGE_REQUEST_ALIASES: dict[str, str] = {
    # "Devanagari" names the SCRIPT, not a language -- Hindi, Marathi,
    # Nepali, Sanskrit, Konkani, Maithili, Bodo and Dogri are all written in
    # it -- but a user replying to "which language?" with just "Devanagari"
    # overwhelmingly means Hindi, by far the most common Devanagari-script
    # language among this app's users. Confirmed live: this reply used to
    # match nothing here, so `app.intent.classifier._find_translation_
    # target` returned `None` and the reply was routed as a brand-new
    # question instead of completing the translation.
    "devanagari": "hindi",
    "hindi": "hindi", "हिंदी": "hindi",
    "இந்தி": "hindi", "హిందీ": "hindi", "ಹಿಂದಿ": "hindi", "হিন্দি": "hindi",
    "english": "english", "अंग्रेजी": "english",
    "ஆங்கிலம்": "english", "ఆంగ్లం": "english", "ಇಂಗ್ಲಿಷ್": "english", "ইংরেজি": "english",
    "hinglish": "hinglish",
    "tamil": "tamil", "தமிழ்": "tamil",
    "तमिल": "tamil", "తమిళం": "tamil", "ತಮಿಳು": "tamil", "তামিল": "tamil",
    "telugu": "telugu", "తెలుగు": "telugu",
    "तेलुगु": "telugu", "தெலுங்கு": "telugu", "ತೆಲುಗು": "telugu", "তেলুগু": "telugu",
    "kannada": "kannada", "ಕನ್ನಡ": "kannada",
    "कन्नड़": "kannada", "कन्नड": "kannada", "கன்னடம்": "kannada", "కన్నడ": "kannada", "কন্নড়": "kannada",
    "malayalam": "malayalam", "മലയാളം": "malayalam",
    "मलयालम": "malayalam", "மலையாளம்": "malayalam", "మలయాళం": "malayalam", "ಮಲಯಾಳಂ": "malayalam", "মালায়ালাম": "malayalam",
    "gujarati": "gujarati", "ગુજરાતી": "gujarati",
    "marathi": "marathi", "मराठी": "marathi",
    "punjabi": "punjabi", "ਪੰਜਾਬੀ": "punjabi",
    "bengali": "bengali", "bangla": "bengali", "বাংলা": "bengali",
    "बंगाली": "bengali", "बांग्ला": "bengali", "வங்காளம்": "bengali", "బెంగాలీ": "bengali", "ಬೆಂಗಾಳಿ": "bengali",
    "odia": "odia", "oriya": "odia", "ଓଡ଼ିଆ": "odia",
    "urdu": "urdu", "اردو": "urdu",
    "assamese": "assamese", "অসমীয়া": "assamese", "asamiya": "assamese", "असमिया": "assamese",
    # ---------------------------------------------------------------
    # The remaining Eighth Schedule languages. `SUPPORTED_LANGUAGES` and the
    # drafting engine already covered all 22, but a user could only ASK for
    # roughly half of them by name -- "অসমীয়াত লিখক" or "संस्कृतम् मध्ये"
    # matched nothing here, so the request fell through silently and the
    # reply came back in whatever language the conversation was already in.
    #
    # Each language is listed under: its own endonym, the roman
    # transliteration people actually type, and the Hindi exonym (by far the
    # most common way an Indian user names another Indian language when
    # writing in Latin script or Devanagari).
    # ---------------------------------------------------------------
    "मराठि": "marathi",
    "konkani": "konkani", "कोंकणी": "konkani", "कोङ्कणी": "konkani", "ಕೊಂಕಣಿ": "konkani",
    "maithili": "maithili", "मैथिली": "maithili",
    "sanskrit": "sanskrit", "संस्कृत": "sanskrit", "संस्कृतम्": "sanskrit", "sanskritam": "sanskrit",
    "nepali": "nepali", "नेपाली": "nepali", "gorkhali": "nepali",
    "dogri": "dogri", "डोगरी": "dogri", "𑠖𑠵𑠌𑠤𑠮": "dogri",
    "bodo": "bodo", "बोड़ो": "bodo", "बोरो": "bodo", "boro": "bodo",
    "santali": "santali", "ᱥᱟᱱᱛᱟᱲᱤ": "santali", "संताली": "santali", "santhali": "santali",
    "manipuri": "manipuri", "ꯃꯅꯤꯄꯨꯔꯤ": "manipuri", "मणिपुरी": "manipuri",
    "meitei": "manipuri", "meiteilon": "manipuri", "ꯃꯩꯇꯩꯂꯣꯟ": "manipuri",
    "kashmiri": "kashmiri", "کٲشُر": "kashmiri", "कश्मीरी": "kashmiri", "koshur": "kashmiri",
    "sindhi": "sindhi", "سنڌي": "sindhi", "सिंधी": "sindhi",
    # Exonyms for languages whose endonym is already listed above.
    "गुजराती": "gujarati",
    "पंजाबी": "punjabi", "panjabi": "punjabi",
    "उड़िया": "odia", "ଓଡ଼ିଶା": "odia",
    "उर्दू": "urdu",
}

# Cue tokens that, immediately before/after a language name, turn a bare
# mention into an explicit request to use that language -- "Kannada ME draft
# karo," "Tamil LA draft pannunga," "tamil BHASHA mein," "in Hindi." Kept as
# a flat token set (checked against a single tokenized word, see
# `_tokenize` below) rather than a regex alternation, for the same reason
# `extract_requested_language` no longer uses `\b`-based regex at all --
# see that function's docstring.
_LANGUAGE_CUE_WORDS = {
    # Latin / Hinglish
    "me", "mein", "main", "la", "bhasha", "basha", "language", "lang",
    "madhye", "vich", "ma", "il", "lo", "alli", "te",
    # Odia locative "-re" ("Odia re notice tiari kara"), romanised. Only ever
    # consulted as the token immediately AFTER a recognised language name, so
    # an ordinary English "re" cannot trigger it on its own.
    "re",
    # Devanagari (Hindi, Marathi, Konkani, Maithili, Nepali, Sanskrit, Dogri, Bodo)
    "में", "मध्ये", "भाषा", "माँ", "मा", "भाषेत", "मध्यें",
    # Bengali / Assamese
    "ভাষায়", "ভাষা", "তে", "এ", "ত",
    # Gujarati
    "માં", "ભાષામાં", "ભાષા",
    # Punjabi
    "ਵਿੱਚ", "ਭਾਸ਼ਾ", "ਵਿਚ",
    # Odia
    "ରେ", "ଭାଷାରେ", "ଭାଷା",
    # Tamil
    "இல்", "ல்", "மொழியில்", "மொழி", "ஆக",
    # Telugu
    "లో", "భాషలో", "భాష",
    # Kannada
    "ನಲ್ಲಿ", "ಭಾಷೆಯಲ್ಲಿ", "ಭಾಷೆ",
    # Malayalam
    "ത്തിൽ", "ഭാഷയിൽ", "ഭാഷ", "ിൽ",
    # Urdu / Kashmiri / Sindhi
    "میں", "زبان",
}
# "to" added after qa-40q-multilingual-20260921 BUG-08: "Convert this notice
# to English" named the target language with a real, unambiguous cue --
# just not one of the two prefixes already covered -- so the whole request
# resolved to no requested language at all and fell through to unrelated
# generic handling instead of re-rendering the draft in English.
_LANGUAGE_PREFIX_CUE_WORDS = {"in", "into", "to"}


# Part 55: Tamil/Telugu/Kannada/Bengali (unlike Hindi/English) express "in
# <language>" via an agglutinative case suffix fused directly onto the
# language's own name -- "தமிழ்" (Tamil) + the locative suffix "-இல்" becomes
# one word, "தமிழில்" ("in Tamil"), with no separate postposition the way
# Hindi's "तमिल में" or English's "in Tamil" have. `_LANGUAGE_CUE_WORDS`
# structurally cannot match this (there is no separate cue token to find),
# so the natural, most common phrasing a Tamil/Telugu/Kannada/Bengali
# speaker actually types for "draft this IN <language>" matched nothing at
# all until this table was added. Curated as complete inflected word forms
# (each already means "in <language>" on its own) rather than derived by a
# generic suffix-stripper: a generic stripper that accepted "language name +
# any short suffix" would also match unrelated words sharing the same root
# -- e.g. Tamil "தமிழ்நாடு" ("Tamil Nadu," the state name, common in a
# complainant's own address) is NOT a request to draft in Tamil, and a
# suffix-based heuristic would eagerly misread it as one.
_INFLECTED_LANGUAGE_REQUEST_PHRASES: dict[str, str] = {
    # Tamil script, "-இல்" locative
    "தமிழில்": "tamil", "தெலுங்கில்": "telugu", "கன்னடத்தில்": "kannada",
    "வங்காளத்தில்": "bengali", "இந்தியில்": "hindi", "ஆங்கிலத்தில்": "english",
    # Telugu script, "-లో" locative
    "తెలుగులో": "telugu", "తమిళంలో": "tamil", "కన్నడంలో": "kannada",
    "బెంగాలీలో": "bengali", "హిందీలో": "hindi", "ఆంగ్లంలో": "english",
    # Kannada script, "-ದಲ್ಲಿ"/"-ನಲ್ಲಿ"/"-ಯಲ್ಲಿ" locative
    "ಕನ್ನಡದಲ್ಲಿ": "kannada", "ತಮಿಳಿನಲ್ಲಿ": "tamil", "ತೆಲುಗಿನಲ್ಲಿ": "telugu",
    "ಬೆಂಗಾಳಿಯಲ್ಲಿ": "bengali", "ಹಿಂದಿಯಲ್ಲಿ": "hindi", "ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ": "english",
    # Bengali script, "-য়"/"-এ"/"-তে" locative
    "বাংলায়": "bengali", "তামিলে": "tamil", "তেলুগুতে": "telugu",
    "কন্নড়ে": "kannada", "হিন্দিতে": "hindi", "ইংরেজিতে": "english",
    "অসমীয়াত": "assamese",
    # Devanagari fused locatives. These were missing entirely, and Devanagari
    # is the script this app sees most: Marathi "मराठीत" ("in Marathi") and
    # Hindi "हिंदीत"/"हिंदीमध्ये" fuse the postposition onto the language name
    # exactly the way the southern scripts above do, so `_LANGUAGE_CUE_WORDS`
    # (which looks for a SEPARATE following token) structurally could not
    # match them. Confirmed: "मराठीत तक्रार तयार करा" -- an unambiguous
    # request, written entirely in Marathi -- resolved to no requested
    # language at all.
    "मराठीत": "marathi", "मराठीमध्ये": "marathi", "मराठीत्": "marathi",
    "हिंदीत": "hindi", "हिंदीमध्ये": "hindi", "हिन्दीमें": "hindi", "हिंदीमें": "hindi",
    "इंग्रजीत": "english", "इंग्लिशमध्ये": "english", "अंग्रेजीमें": "english",
    "संस्कृतम्": "sanskrit", "नेपालीमा": "nepali", "कोंकणीत": "konkani",
    "तमिळमध्ये": "tamil", "बंगालीमध्ये": "bengali",
    # Gujarati "-માં"
    "ગુજરાતીમાં": "gujarati", "હિન્દીમાં": "hindi", "અંગ્રેજીમાં": "english",
    # Malayalam "-ിൽ"
    "മലയാളത്തിൽ": "malayalam", "ഹിന്ദിയിൽ": "hindi", "ഇംഗ്ലീഷിൽ": "english", "തമിഴിൽ": "tamil",
    # Odia "-ରେ"
    "ଓଡ଼ିଆରେ": "odia", "ଓଡିଆରେ": "odia", "ହିନ୍ଦୀରେ": "hindi", "ଇଂରାଜୀରେ": "english",
    # Punjabi "-ਵਿੱਚ" is a separate token (already a cue word); the fused
    # colloquial form is not.
    "ਪੰਜਾਬੀਵਿੱਚ": "punjabi",
}


# Only the ASCII (romanized) alias keys are candidates for typo-tolerant
# matching below -- edit distance on a handful of Latin characters is a
# meaningful "is this a typo" signal, but the same wouldn't hold for
# multi-codepoint Indic-script combining-mark sequences, so those keys are
# deliberately excluded rather than risk a nonsensical near-match there.
_ASCII_LANGUAGE_ALIAS_KEYS: tuple[str, ...] = tuple(
    key for key in _LANGUAGE_REQUEST_ALIASES if key.isascii()
)


def _levenshtein_distance_at_most_one(a: str, b: str) -> bool:
    """True iff `a` and `b` differ by at most one single-character
    insertion, deletion, or substitution. A length gap of 2+ can never be
    within edit distance 1, so this short-circuits before doing any real
    comparison work.
    """
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        mismatches = sum(1 for x, y in zip(a, b, strict=True) if x != y)
        return mismatches <= 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    i = j = 0
    skipped = False
    while i < len(shorter) and j < len(longer):
        if shorter[i] != longer[j]:
            if skipped:
                return False
            skipped = True
            j += 1
            continue
        i += 1
        j += 1
    return True


def resolve_language_token(token: str) -> str | None:
    """Exact alias lookup first; falls through to a single-typo-tolerant
    match ("hiindi" -> "hindi") only when exact lookup fails, scoped to
    tokens at least 4 characters long (`_LANGUAGE_CUE_WORDS` like "me"/"la"
    are shorter, so this can't accidentally swallow a cue word as a typo'd
    language name) so a short, unrelated word doesn't spuriously land
    within edit distance 1 of a real alias.

    Public (no leading underscore) so callers outside this module -- e.g.
    `app.intent.classifier._find_translation_target`, resolving a bare
    one-word reply to "which language would you like this translated
    into?" -- get the same typo tolerance and native-script/exonym
    coverage as `extract_requested_language` below, instead of each
    re-implementing a thinner version of the same lookup.
    """
    exact = _LANGUAGE_REQUEST_ALIASES.get(token)
    if exact is not None:
        return exact
    if len(token) < 4:
        return None
    matches = {
        _LANGUAGE_REQUEST_ALIASES[key]
        for key in _ASCII_LANGUAGE_ALIAS_KEYS
        if _levenshtein_distance_at_most_one(token, key)
    }
    return matches.pop() if len(matches) == 1 else None


def _tokenize(text: str) -> list[str]:
    """Splits `text` into word-like runs using the same Unicode-category
    definition of "word character" as `app/drafting/intent.py`'s
    `_contains_as_word` (letters, plus combining marks Mn/Mc, plus digits) --
    duplicated here rather than imported so this module has no dependency on
    `app.drafting`. Used instead of `re.findall(r"\\w+", text)` because
    Python's `\\w` (and therefore `\\b`) does NOT count Mn/Mc combining marks
    (matras, virama, anusvara) as word characters, so a plain regex tokenizer
    would silently split Devanagari/Tamil/Telugu/Kannada/Bengali words apart
    at their own internal vowel signs -- e.g. "बंगाली" ends in "ी" (category
    Mc); `re.search(r"\\bबंगाली\\b", "बंगाली में")` fails outright because `\\b`
    finds no boundary between that trailing "ी" and the following space
    (neither side reads as "word" under `\\w`'s definition). Confirmed
    directly: with the old `\\b`-based pattern, "बंगाली"/"कन्नड़"/"तेलुगु" (the
    Devanagari spellings of Bengali/Kannada/Telugu) never matched at all,
    even after being added to `_LANGUAGE_REQUEST_ALIASES` -- only "तमिल"
    (which happens to end in a bare consonant, no trailing matra) worked by
    accident.
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        if unicodedata.category(char) in {"Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Nd"}:
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def extract_requested_language(text: str) -> str | None:
    """Finds an explicitly-requested language for a legal draft in free text
    -- "Generate a legal notice in Hindi", "Kannada me draft karo", "Tamil la
    draft pannunga", "तमिल में मसौदा तैयार करें", "தமிழில் ஒரு புகார் வரைவு
    தயார் செய்யுங்கள்" -- so it can take priority over the ambient
    conversation language when the user names one explicitly (Part 52 rule:
    an explicit language request always wins).

    Returns `None` when no explicit language cue is found, so callers can
    fall back to the conversation language. Deliberately requires a cue (an
    "in"/"into" prefix, a following me/mein/la/bhasha/language/में/भाषा
    postposition, or one of the complete South-Indian inflected phrases
    above) rather than a bare language name anywhere in the text, so a
    sentence that merely mentions a language ("I found a lawyer who speaks
    Tamil") isn't mistaken for a request to draft IN that language.
    """
    tokens = _tokenize(text)
    for token in tokens:
        language = _INFLECTED_LANGUAGE_REQUEST_PHRASES.get(token)
        if language:
            return language

    lowered_tokens = [token.lower() for token in tokens]
    for index, token in enumerate(lowered_tokens):
        language = resolve_language_token(token)
        if language is None:
            continue
        previous_token = lowered_tokens[index - 1] if index > 0 else None
        next_token = lowered_tokens[index + 1] if index + 1 < len(lowered_tokens) else None
        if previous_token in _LANGUAGE_PREFIX_CUE_WORDS or next_token in _LANGUAGE_CUE_WORDS:
            return language
    return None


class LanguageDetector:
    def resolve(self, text: str, previous_language: str | None) -> str:
        """Like `detect()`, but a short message with no script-level signal
        (no Devanagari/other Indic script) inherits `previous_language`
        instead of letting an unreliable guess on a handful of characters
        override an already-established conversation language. Any message
        long enough or carrying its own script is still detected fresh, so a
        deliberate language switch (typing in Devanagari, or a real sentence
        in a different language) is always respected.

        A short message with Hinglish tokens (`_STRONG_HINGLISH_TERMS`) or an
        unmistakable English question shape (`_ENGLISH_QUESTION_STARTER_
        PATTERN`) is not actually ambiguous either, despite being short --
        only a genuinely bare fragment ("Why?", "Fir?", "Kitna time?") with
        neither signal falls back to inheriting.
        """
        compact = text.strip()
        has_script_signal = bool(re.search(DEVANAGARI_PATTERN, compact)) or any(
            re.search(pattern, compact) for pattern, _ in SCRIPT_LANGUAGE_PATTERNS
        )
        if previous_language and len(compact) <= _SHORT_TEXT_INHERIT_THRESHOLD and not has_script_signal:
            tokens = set(re.findall(r"[a-zA-Z]+", compact.lower()))
            has_hinglish_token = bool(tokens & _STRONG_HINGLISH_TERMS)
            has_clear_english_shape = bool(_ENGLISH_QUESTION_STARTER_PATTERN.match(compact))
            if not has_hinglish_token and not has_clear_english_shape:
                return previous_language
        return self.detect(text)

    def detect(self, text: str) -> str:
        compact = text.strip()
        lowered = compact.lower()
        if _devanagari_outweighs_other_scripts(compact):
            # Eight supported languages share Devanagari, so the script check
            # alone cannot tell them apart. Marker words first (they cover the
            # five langdetect has no model for), then langdetect for the three
            # it does, then Hindi -- the long-standing default.
            marker_language = _devanagari_language(compact)
            if marker_language is not None:
                return marker_language
            if detect is not None:
                try:
                    code = detect(compact)
                except LangDetectException:
                    # Unprofilable input (too short / no features). The script
                    # is already known to be Devanagari at this point, so Hindi
                    # is the correct default for the region's largest language.
                    return "hindi"
                # Only mr/ne are trusted from langdetect here. Its other
                # Devanagari verdicts are all "hi", and anything else it
                # returns for Devanagari text is noise.
                return LANGDETECT_MAP[code] if code in ("mr", "ne") else "hindi"
            return "hindi"
        for pattern, language in SCRIPT_LANGUAGE_PATTERNS:
            if re.search(pattern, compact):
                # Urdu/Kashmiri/Sindhi share Perso-Arabic; narrow it further.
                return _perso_arabic_language(compact) if language == "urdu" else language
        tokens = set(re.findall(r"[a-zA-Z]+", lowered))
        # Checked before Hinglish: romanized Manipuri and romanized
        # Hindi/Hinglish share the Latin script but not vocabulary (no
        # overlap between this set and `_STRONG_HINGLISH_TERMS`), so order
        # only matters for readability here, not correctness.
        if tokens & _ROMANIZED_MANIPURI_TERMS:
            return "manipuri"
        if tokens & _STRONG_HINGLISH_TERMS:
            return "hinglish"
        if len(tokens & _WEAK_HINGLISH_TERMS) >= 2:
            return "hinglish"
        if detect is not None:
            try:
                return LANGDETECT_MAP.get(detect(compact), "english")
            except LangDetectException:
                return "english"
        return "english"


# ---------------------------------------------------------------------------
# Standalone language-preference commands
# ---------------------------------------------------------------------------
# Post-Phase-3 hardening: `extract_requested_language` already recognised the
# language in "Mujhe simple Hindi mein jawab diya karo" and "Actually Hinglish
# mein jawab do", but the only caller that acted on a standalone preference
# change (`ChatOrchestrator`) gated it behind a regex that matched nothing
# longer than "answer in hindi". Both reported sentences fell through to
# ordinary legal routing -- one classified "Legal Advice", the other "General
# Legal Information" -- and were answered from retrieval.
#
# Recognition is by SUBTRACTION, not by enumerating phrasings: take the
# language cue out, and a preference command is one with nothing left but
# harmless modifiers. Anything with real content left over ("Hindi mein FIR
# kaise file karein?") is a legal question that names a language, not a
# command to switch, and is deliberately not matched.
_LANGUAGE_COMMAND_MODIFIERS = {
    # who/for whom
    "mujhe", "muje", "mujhko", "mereko", "hume", "humein", "mere", "meri", "mera",
    # politeness / discourse openers
    "please", "pls", "plz", "kindly", "kripya", "kripaya", "कृपया", "कृपा",
    "actually", "waise", "ok", "okay", "theek", "thik", "ठीक", "अच्छा", "accha",
    # time / continuation
    "ab", "abhi", "aage", "now", "onwards", "onward", "further", "hamesha",
    "always", "अब", "आगे", "अभी",
    # style modifiers
    "simple", "saral", "सरल", "aasan", "आसान", "easy", "plain", "short", "chhota",
    "asaan", "clear", "साफ", "सादी", "सीधी", "seedhi",
    # the act of answering
    "answer", "answers", "answering", "jawab", "jawaab", "javab", "jvab",
    "जवाब", "उत्तर", "reply", "replies", "respond", "response", "bolo", "bol",
    "batao", "bta", "btao", "bataye", "bataiye", "बताओ", "बताइए", "बताएं",
    "likho", "likhna", "write", "speak", "talk", "samjhao", "समझाओ",
    # imperative tails
    "do", "de", "den", "dena", "dijiye", "diya", "dijiyega", "देना", "दें",
    "दीजिए", "दो", "दिया", "करो", "करें", "कीजिए", "किया", "kar", "karo", "kro",
    "karna", "karen", "karein", "kariye", "kijiye", "kiya", "karte", "karta",
    "raho", "rahe", "rakho", "continue", "switch", "change", "use", "start",
    # the language-naming scaffolding itself
    "in", "into", "me", "mein", "mai", "main", "में", "मे", "language", "lang",
    "bhasha", "भाषा", "भाषामें", "only", "sirf", "केवल", "सिर्फ", "hi", "ही",
    "se", "से", "ke", "ki", "ka", "के", "की", "का",
}

def is_language_preference_command(text: str) -> str | None:
    """The language a STANDALONE preference change asks for, or `None`.

    Returns a language only when the message is nothing but the request:
    "Mujhe simple Hindi mein jawab diya karo", "Hindi me simple answer dena",
    "Actually Hinglish mein jawab do", "Ab English me continue karo",
    "कृपया सरल हिंदी में जवाब दें". A sentence that merely MENTIONS a language
    ("I found a lawyer who speaks Hindi", "Hindi mein FIR kaise file karein?")
    is not one, and returns `None` -- the leftover tokens are real content.

    Deliberately narrower than `extract_requested_language`, which answers a
    different question ("what language should this DRAFT be in?") and is
    happy to find a language inside a longer request.
    """
    language = extract_requested_language(text)
    if language is None:
        return None
    leftover = [
        token
        for token in _tokenize(text)
        if token.lower() not in _LANGUAGE_COMMAND_MODIFIERS
        and resolve_language_token(token.lower()) is None
        and token not in _INFLECTED_LANGUAGE_REQUEST_PHRASES
    ]
    return language if not leftover else None
