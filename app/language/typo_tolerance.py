"""One bounded, reusable typo-tolerant normalizer for ROUTING and RETRIEVAL.

Why this exists
---------------
A reported session lost four turns to single-character typos: "um mere liye
kya-kya kar sakte ho?" (a "t" missing from "tum") was routed to retrieval and
answered with the strict-RAG refusal, while the identical correctly-spelled
sentence was answered from the fixed capability overview. "tum mre liye kya
kya kr skte ho", "aap kon kon se legal kaam kr skte ho" and "what can yu help
me with?" failed the same way, and "chek bouns hone pr secshun 138 me kya hota
hai?" could not retrieve Section 138 material.

What this deliberately is NOT
-----------------------------
It is not an LLM spell-checker. Sending arbitrary user text to a model and
treating its guess as authoritative would silently rewrite a party's name, an
address, an invoice number, a date, an amount or a section number -- in a
legal tool that is a data-integrity failure, not a UX improvement. Every
correction here is a lookup against a curated allowlist, and false-positive
safety is worth more than correcting every typo.

The contract
------------
* the ORIGINAL text is never mutated -- callers keep it for chat history and
  audit (`ChatService` persists `request.question`, unchanged);
* `normalized_for_routing` is a separate value, used only to decide intent and
  to match workflows;
* `retrieval_query` is the original PLUS the safe normalized aliases, so
  retrieval sees both spellings and an exact match on the user's own wording
  can still win;
* exact matching comes first -- an in-vocabulary token is never touched;
* fuzzy matching runs only against the curated allowlist, at edit distance 1,
  and only for ordinary tokens of length >= 4;
* one missing LEADING character is repaired only when the rest of the message
  carries enough control cues to make the result unique ("um ... kya ... kar
  sakte ho" -> "tum");
* a correction is accepted only when exactly one candidate is clearly best --
  a tie is reported as an ambiguity for the caller to ask about, never guessed;
* quoted text, anything containing a digit, anything with identifier-ish
  structure, ALL-CAPS tokens and mid-sentence Capitalized tokens (party names,
  places) are protected and returned byte-for-byte.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Curated vocabularies
# ---------------------------------------------------------------------------
# Conversational-control words: the vocabulary of "talking to the assistant"
# rather than "asking about the law". Only these may trigger a clarification
# question when a correction is ambiguous (see `clarification_question`) --
# an ambiguity among legal terms is left alone and retrieval still sees the
# original spelling, so a genuine legal query is never hijacked by one.
_CONTROL_TERMS: frozenset[str] = frozenset({
    # second person / first person
    "tum", "tumhe", "tumhein", "tumhara", "tumhari", "tumhare", "tumko",
    "aap", "aapka", "aapki", "aapke", "aapko", "you", "your", "yours",
    "mujhe", "mujhko", "mere", "meri", "hamare",
    # interrogatives
    "kya", "kyu", "kyun", "kaise", "kaisi", "kaisa", "kaun", "kaunsa", "kaunsi",
    "kitna", "kitni", "kitne", "kab", "kahan", "what", "which", "how", "who",
    "when", "where", "why", "can", "could", "would", "will", "should",
    # ability / help
    "kar", "karo", "karte", "karti", "karta", "karna", "karne", "karen", "karein",
    "sakta", "sakte", "sakti", "sakoge", "sakenge",
    "madad", "sahayata", "help", "assist", "support", "capable", "capabilities",
    "capability", "feature", "features", "service", "services", "offer",
    # discourse glue
    "liye", "hain", "hoon",
    "please", "kripya", "actually", "abhi", "aage", "phir",
    "batao", "bata", "bataye", "bataiye", "bolo",
    "dena", "dijiye", "diya", "jawab", "jawaab", "javab", "answer", "answers",
    "reply", "response", "respond", "explain", "samjhao", "samjha",
    "simple", "saral", "aasan", "easy", "plain", "short", "detail", "detailed",
    "language", "bhasha", "hindi", "english", "hinglish", "continue", "switch",
    "speak", "talk", "write", "likho", "start", "stop", "cancel", "retry",
    "again", "dobara", "thanks", "thank", "shukriya", "dhanyavad",
    "okay", "theek", "accha",
    "kaam", "legal", "chahiye", "with", "about", "some", "only", "sirf",
    "hota", "hoti", "hote", "kuch",
})

# Workflow terms the product actually supports end to end, so a typo in the
# name of a capability still reaches the workflow that provides it.
_WORKFLOW_TERMS: frozenset[str] = frozenset({
    "draft", "drafts", "drafting", "notice", "notices", "complaint", "complaints",
    "affidavit", "agreement", "application", "petition", "letter", "document",
    "documents", "upload", "download", "export", "preview", "review",
    "compare", "comparison", "timeline", "evidence", "annexure",
    "summary", "summarize", "translate", "translation", "notary", "notarize",
    "notarization", "attest", "attestation", "advocate",
    "vakil", "recommend", "prepare", "tayar", "taiyar", "preferences",
})

# Common legal vocabulary, kept to terms this knowledge base genuinely covers.
_LEGAL_TERMS: frozenset[str] = frozenset({
    "cheque", "bounce", "bounced", "section", "sections",
    "police", "arrest", "magistrate",
    "divorce", "maintenance", "custody", "marriage", "rental", "landlord",
    "tenant", "eviction", "deposit", "consumer", "refund", "warranty", "fraud",
    "cyber", "theft", "harassment", "dowry", "domestic", "violence",
    "property", "inheritance", "succession", "contract", "breach",
    "penalty", "punishment", "imprisonment", "compensation", "damages",
    "summons", "warrant", "appeal", "jurisdiction",
    "salary", "employment", "termination", "gratuity", "provident", "insurance",
    "accident", "injury", "negligence", "defamation", "trademark", "copyright",
    "contempt", "witness", "chargesheet", "investigation", "juvenile",
    "cognizable", "bailable", "quashing", "anticipatory",
    # Confirmed live: "hindu" was missing from every vocabulary this module
    # checks (allowlist, protected, common), so "hindu marriage act" and
    # "hindu ka defination..." had "hindu" silently FUZZY-CORRECTED to
    # "hindi" (a real, distance-1, already-common word) -- actively
    # corrupting a correctly-spelled personal-law query into a nonsensical
    # one before it ever reached retrieval. Personal-law Act names this
    # corpus covers are named after the religion they apply to, so those
    # names need the same protection "hindu" did.
    "hindu", "muslim", "christian", "parsi", "sikh", "buddhist", "jain",
    # Also confirmed live: real misspellings of ordinary legal/procedural
    # vocabulary that this module had no target for, so they passed through
    # uncorrected and then failed the downstream named-Act/topic matching
    # that needs the correct spelling ("code of criminal procidure",
    # "culpable homiside").
    "procedure", "homicide", "culpable",
    # The three post-2024 codes' own names -- "bhartiya nayay sanhita
    # section 12" failed to reach the named-Act-citation path (which
    # requires the exact spelling "Bharatiya Nyaya Sanhita") because none of
    # these words had a correction target either.
    "bharatiya", "nyaya", "sanhita", "adhiniyam",
})

# Confirmed live (2026-09-23): `app.intent.classifier._OFF_DOMAIN_PATTERN`
# checks the SAME typo-normalized text this module produces, but its own
# word list ("python", "pizza", "recipe", ...) is a strict literal match --
# a one-character typo in any of them ("pyton", "pizaa", "recipie") slipped
# past the off-domain check entirely, fell through to the ordinary
# legal-answer pipeline, and got answered (as real Python code / a pizza
# recipe) mislabeled "General Legal Knowledge" by the controlled GK
# fallback -- which has no topic check of its own beyond "did retrieval
# find nothing." Not exhaustive (matching `_OFF_DOMAIN_PATTERN`'s own
# "bounded, representative set, not exhaustive" scope), just the terms a
# real conversation actually mistyped.
_OFF_DOMAIN_TERMS: frozenset[str] = frozenset({
    "python", "pizza", "recipe", "biryani", "coding", "programming",
    "algorithm", "weather", "cricket",
})

_ALLOWLIST: frozenset[str] = _CONTROL_TERMS | _WORKFLOW_TERMS | _LEGAL_TERMS | _OFF_DOMAIN_TERMS

# Explicit exact aliases: shorthand and mistypings whose distance from the
# intended word is more than one edit, so the fuzzy rule below can never reach
# them. Each entry is a deliberate, reviewed decision -- not a guess.
_ALIASES: dict[str, str] = {
    # Hinglish shorthand the report names explicitly.
    "kr": "kar", "kro": "karo", "krna": "karna", "krke": "karke",
    "bta": "bata", "btao": "batao", "btaye": "bataye",
    "htoi": "hota", "pr": "par",
    "skte": "sakte", "skta": "sakta", "skti": "sakti",
    "mre": "mere", "kon": "kaun", "konsa": "kaunsa", "konsi": "kaunsi",
    "tyar": "tayar", "tyaar": "tayar",
    "yu": "you", "plz": "please", "pls": "please",
    "jawb": "jawab", "jwab": "jawab",
    # Legal-term mistypings two or more edits away from the intended term.
    "chek": "cheque", "chq": "cheque", "cheq": "cheque",
    "bouns": "bounce", "bouncee": "bounce", "bounse": "bounce",
    # QA session 2026-09-24 ("BUG-108"): "bonuce" (the "u"/"n" transposed) is
    # edit-distance 2 from "bounce" under plain Levenshtein, one past
    # `_fuzzy_candidates`' edit-distance-1 cutoff -- live-reproduced query
    # "wat is teh punishment for chek bonuce in indai" left "chek" corrected
    # but "bonuce" untouched, so the retrieval-query never contained the
    # contiguous phrase "cheque bounce" needed to trigger
    # `SmartQueryRewriter`'s cheque-bounce concept expansion.
    "bonuce": "bounce",
    "secshun": "section", "sekshan": "section", "sectn": "section",
    "notry": "notary", "notari": "notary",
    # Notarization specifically: a workflow keyword, so a misspelling here
    # loses the whole request rather than degrading the answer. British "-s-"
    # included because it is a spelling, not a typo, and the matchers are
    # keyed on the "-z-" form.
    "notarisation": "notarization", "notarizaton": "notarization",
    "notarizatoin": "notarization", "notorization": "notarization",
    "notarise": "notarize", "notarised": "notarized",
    "compalint": "complaint", "complent": "complaint",
    # "nayay"/"niyay" -> "nyaya": a common phonetic transliteration of the
    # Bharatiya Nyaya Sanhita's own name, but a transposition/multi-position
    # respelling rather than a single edit, so the distance-1 fuzzy rule
    # below can never reach it.
    "nayay": "nyaya", "niyay": "nyaya",
    "afidavit": "affidavit", "affidavid": "affidavit",
}

# Words that are correctly spelled but sit within one edit of an allowlist
# entry. Listed so the exact pass recognises them as in-vocabulary and they
# are therefore never "corrected" into something else -- without "kiya" here,
# for instance, "kiya" would be rewritten to "kya" (one deletion), corrupting
# the narrative facts a user pastes into a draft.
_PROTECTED_WORDS: frozenset[str] = frozenset({
    "kiya", "kiye", "gaya", "gayi", "gaye", "jaye", "jaega", "hua", "hui",
    "mila", "mile", "milta", "lekin", "kaafi", "apne", "apna", "apni",
    "maine", "nahin", "nahi", "chori", "market", "phone", "mobile", "number",
    "search", "call", "receive", "primary", "alternate", "location", "place",
    "station", "address", "name", "date", "time", "amount", "cash", "bill",
    "mail", "male", "sale", "sales", "half", "hall", "ball", "tall",
    "bank", "bond", "band", "land", "hand", "fund", "form", "farm", "firm",
    "term", "team", "book", "look", "took", "cook", "code", "core", "care",
    "cart", "card", "part", "past", "post", "cost", "list", "last", "lost",
    "note", "node", "none", "nine", "line", "life", "like", "wife",
    "site", "size", "side", "ride", "rate", "late", "gate", "data",
    "year", "hear", "near", "dear", "fear", "head", "read", "real", "rear",
    "word", "work", "world", "wound", "round", "found", "sound",
    "shop", "stop", "step", "stay", "star", "smart", "share", "shame",
    "father", "mother", "brother", "sister", "husband",
    "school", "office", "house", "home", "hotel", "hospital",
    "sold", "sell", "cell", "bell", "well", "wall", "tell",
    "kaam", "naam", "gaon", "shahar", "din", "raat", "saal", "mahina",
    "paisa", "paise", "rupay", "rupaye", "rupees", "lakh", "crore", "hazar",
    "check", "checks", "checked", "checking",
    "hone", "hona", "hoke", "hokar", "hoga", "hogi", "honge", "hoye",
    "liya", "diye", "deta", "dete", "dene", "lena", "lene", "leta", "lete",
    "tha", "thi", "kaha", "kahi", "yaha", "waha", "isme", "usme", "iska",
    "uska", "isko", "usko", "unka", "unke", "unki", "sakte", "banao", "bana",
    "beta", "beti", "bola", "bole", "boli", "likha", "likhi", "likhe",
    "suna", "dekha", "dekhi", "padha", "bheja", "bhejo", "bhai",
    "bahut", "thoda", "jyada", "zyada", "pura", "poora", "aadha", "baad",
    "pehle", "abhi", "kabhi", "sabhi", "wahan", "yahan", "jahan", "tab",
    "rahta", "rahte", "rehta", "milna", "milne", "dena", "karta", "hote",
    "page", "pages", "thana", "copy", "sign", "seal", "loan", "emi",
    "ab", "am", "an", "as", "at", "it", "is", "on", "or", "to", "up", "us",
    "simply", "clearly", "briefly", "exactly", "directly", "quickly",
    "easily", "legally", "finally", "really", "sorry", "story", "study",
    "reply", "apply", "party", "carry", "worry", "hurry", "early",
    "demand", "payment", "rupee", "interest", "order", "judgment", "decree",
    "owner", "buyer", "seller", "agent", "broker", "client", "accused",
    "victim", "company", "employer", "employee", "friend", "uncle", "aunt",
    # Very common short glue that must never be reshaped into a control word.
    "main", "hum", "mera", "hai", "ho", "he", "me", "mein", "do", "de",
    "den", "the", "that", "this", "and", "for", "par", "ka", "ki", "ke", "se",
    "bhi", "aur", "sab", "koi", "har", "sath", "raha", "rahe", "raho", "yes",
    "haan", "act", "acts", "case", "cases", "court", "bail", "rent", "will",
    "fine", "tax", "charge", "edit", "gst", "invoice",
})

# The everyday vocabulary a user writes around their question. None of it is
# a correction TARGET -- it exists so the exact pass recognises these words
# as in-vocabulary and the fuzzy pass never reshapes one into an allowlist
# entry it happens to sit one edit away from. Every entry below was either
# a confirmed false positive ("were" -> "where" broke "What were we
# discussing?"; "chain" -> "hain" made "What about the chain?" retrieve
# nothing; "helps" -> "help" sent "Thanks, that helps" to retrieval) or is
# in the same closed class as one. English function words are a closed set,
# so listing them is a bounded, one-time cost -- and it is what keeps the
# fuzzy pass honest without a dictionary.
_COMMON_WORDS: frozenset[str] = frozenset(
    ["a", "about", "above", "across", "after", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "an", "and", "another", "any", "anyone", "anything", "are", "around", "as", "ask", "asked", "asking", "at", "away", "back", "bad", "be", "because", "become", "been", "before", "began", "begin", "behind", "being", "below", "best", "better", "between", "beyond", "big", "both", "bring", "brought", "but", "buy", "by", "call", "called", "came", "cannot", "come", "coming", "could", "country", "day", "days", "did", "different", "do", "does", "doing", "done", "down", "during", "each", "early", "either", "else", "end", "enough", "even", "ever", "every", "everyone", "everything", "example", "far", "feel", "felt", "few", "find", "first", "five", "following", "for", "found", "four", "from", "front", "full", "further", "gave", "general", "get", "gets", "getting", "give", "given", "go", "goes", "going", "gone", "good", "got", "great", "group", "had", "half", "hand", "happen", "happened", "has", "have", "having", "he", "her", "here", "hers", "herself", "high", "him", "himself", "his", "hold", "home", "hour", "how", "however", "i", "if", "important", "in", "into", "is", "it", "its", "itself", "just", "keep", "kept", "kind", "knew", "know", "known", "large", "later", "least", "leave", "left", "less", "let", "letter", "light", "little", "live", "long", "made", "make", "makes", "making", "many", "may", "maybe", "me", "mean", "means", "meant", "men", "might", "mine", "miss", "month", "more", "most", "move", "much", "must", "my", "myself", "name", "near", "need", "needed", "needs", "never", "new", "next", "night", "no", "nor", "not", "nothing", "now", "number", "of", "off", "often", "old", "on", "once", "one", "only", "open", "or", "order", "other", "others", "otherwise", "our", "ours", "out", "over", "own", "page", "part", "people", "per", "perhaps", "person", "place", "point", "possible", "put", "question", "quite", "rather", "reach", "read", "ready", "really", "reason", "receive", "received", "right", "room", "said", "same", "saw", "say", "saying", "says", "second", "see", "seem", "seen", "sell", "send", "sent", "set", "several", "shall", "she", "should", "show", "showed", "shown", "side", "since", "six", "small", "so", "some", "someone", "something", "sometimes", "soon", "sort", "sound", "speak", "spoke", "stand", "start", "state", "still", "such", "sure", "take", "taken", "taking", "tell", "ten", "than", "thank", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "therefore", "these", "they", "thing", "things", "think", "this", "those", "though", "thought", "three", "through", "thus", "time", "times", "to", "today", "together", "told", "too", "took", "toward", "towards", "true", "try", "trying", "turn", "two", "under", "until", "up", "upon", "us", "use", "used", "using", "usually", "very", "want", "wanted", "was", "way", "we", "week", "well", "went", "were", "what", "whatever", "when", "whenever", "where", "whether", "which", "while", "who", "whole", "whom", "whose", "why", "wide", "with", "within", "without", "woman", "women", "word", "words", "work", "worked", "working", "world", "year", "years", "yes", "yet", "you", "young", "your", "yours", "yourself",
     # Everyday words that sit exactly one edit from an allowlist legal
     # term. Confirmed false positive: "Contact me at ..." had "Contact"
     # rewritten to "Contract". Same closed class: "latter"/"letter",
     # "plan"/"plain", "stare"/"start".
     "contact", "contacts", "contacted", "latter", "plan", "plans",
     "planned", "planning", "stare", "star", "stars", "aaj", "aap", "aapko", "abhi", "acha", "achha", "admi", "adhikar", "agar", "age", "ap", "apna", "apne", "apni", "are", "aur", "baad", "baat", "bad", "bahar", "bahut", "bana", "banaya", "bani", "baraabar", "bare", "bas", "bata", "baki", "bhai", "bhej", "bhi", "bhut", "bina", "bola", "bolna", "chahiye", "chaiye", "chal", "chalta", "chaha", "chahta", "chize", "chiz", "de", "dega", "dega", "dege", "den", "dena", "dene", "der", "dete", "dhang", "di", "din", "diya", "do", "dono", "dost", "dusra", "dusre", "ek", "ese", "fir", "gaya", "gaye", "gayi", "ghar", "hai", "hain", "hamara", "hamare", "hamari", "han", "haan", "hi", "hain", "ho", "hoga", "hogi", "hoke", "hona", "hone", "honge", "hota", "hoti", "hote", "hua", "hue", "hui", "hum", "humein", "humko", "is", "isi", "iska", "iske", "iski", "isko", "isliye", "isme", "itna", "jab", "jaise", "jaisa", "jana", "jata", "jate", "jati", "jaye", "jayega", "jitna", "jo", "jyada", "kab", "kabhi", "kaha", "kahan", "kahi", "kam", "kaam", "kar", "karke", "karna", "karta", "karte", "karti", "kaun", "kaise", "ke", "keval", "khud", "ki", "kis", "kisi", "kitna", "kiya", "kiye", "ko", "koi", "kuch", "kul", "kya", "kyu", "kyun", "le", "lekin", "lena", "lene", "liya", "liye", "log", "lugu", "magar", "mai", "main", "mein", "mera", "mere", "meri", "mila", "mile", "mujhe", "na", "nahi", "nai", "nay", "ne", "nhi", "nikal", "par", "pe", "pehle", "phir", "poora", "pura", "raha", "rahe", "rahi", "raho", "rakh", "rakha", "sab", "sabhi", "sakta", "sakte", "sakti", "samay", "samajh", "sath", "se", "sirf", "tab", "tak", "tarah", "tha", "the", "thi", "thoda", "tk", "to", "tum", "tumhara", "tumhare", "tumhari", "unka", "unke", "unki", "unko", "us", "usi", "uska", "uske", "uski", "usko", "vah", "vahan", "wahan", "waise", "wo", "woh", "ye", "yeh", "yaha", "yahan"]
)

_KNOWN_WORDS: frozenset[str] = (
    _ALLOWLIST | _PROTECTED_WORDS | _COMMON_WORDS | frozenset(_ALIASES.values())
)

# Allowlist entries grouped by length, so the edit-distance-1 scan only ever
# looks at candidates that could possibly be one edit away.
_BY_LENGTH: dict[int, list[str]] = {}
for _word in sorted(_ALLOWLIST):
    _BY_LENGTH.setdefault(len(_word), []).append(_word)

# Above this many words a message is pasted content (a narrative, a block of
# labelled draft fields, a quoted document), not a short conversational turn
# with a typo in it. Routing never needs typo tolerance there, and a wrong
# "correction" inside someone's facts costs far more than the benefit, so the
# whole pass is skipped.
MAX_TOKENS_FOR_NORMALIZATION = 30

# The minimum ordinary-token length for the edit-distance-1 rule. Below this a
# single edit is too large a share of the word for the match to mean anything
# ("do" -> "de", "ka" -> "ki" are different words, not typos of each other).
MIN_LENGTH_FOR_FUZZY = 4

# How many control cues the rest of the message must carry before a missing
# leading character is repaired. Two, so one incidental match cannot do it.
_MIN_CUES_FOR_LEADING_CHAR_REPAIR = 2

_LATIN_WORD = re.compile(r"[A-Za-z]+")
_QUOTE_SPAN = re.compile("\"[^\"]*\"|'[^']*'|“[^”]*”|‘[^’]*’")
# A whitespace-delimited chunk carrying identifier structure -- an email, a
# file name, a hyphenated invoice number, an amount, a date. Matched on the
# raw chunk, before word extraction, so "INV-2026/114.pdf" is protected whole.
_IDENTIFIERISH = re.compile(r"[0-9@/\\_]|\.[A-Za-z]")


@dataclass(frozen=True)
class NormalizationResult:
    """The three views of one user message.

    `original` is authoritative for history and audit and is returned
    unchanged in every case, including when normalization is skipped.
    """

    original: str
    normalized_for_routing: str
    retrieval_query: str
    corrections: tuple[tuple[str, str], ...] = ()
    # (token, candidates) for each token whose best correction was a tie.
    ambiguities: tuple[tuple[str, tuple[str, ...]], ...] = field(default=())

    @property
    def changed(self) -> bool:
        return bool(self.corrections)

    @property
    def control_ambiguities(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Ambiguities whose every candidate is a conversational-control word.

        Only these are worth asking the user about: an ambiguity among legal
        terms is resolved by retrieval still seeing the original spelling,
        while an ambiguity in the words that decide ROUTING would otherwise be
        settled by a guess.
        """
        return tuple(
            (token, candidates)
            for token, candidates in self.ambiguities
            if all(candidate in _CONTROL_TERMS for candidate in candidates)
        )


def _within_one_edit(candidate: str, token: str) -> bool:
    """True when `candidate` is reachable from `token` in one substitution,
    insertion or deletion. Written directly rather than via a full Levenshtein
    matrix because only distance 1 is ever allowed."""
    len_candidate, len_token = len(candidate), len(token)
    if abs(len_candidate - len_token) > 1:
        return False
    if len_candidate == len_token:
        differences = sum(1 for left, right in zip(candidate, token, strict=True) if left != right)
        return differences == 1
    longer, shorter = (candidate, token) if len_candidate > len_token else (token, candidate)
    index = 0
    while index < len(shorter) and longer[index] == shorter[index]:
        index += 1
    return longer[index + 1:] == shorter[index:]


# Regular English suffixes. A token that is a known word plus one of these
# is itself a known word, and must never be "corrected" back to the stem:
# without this, "Thanks, that helps" had "helps" rewritten to "help", which
# stopped the whole message decomposing into greeting units and sent a
# thank-you to retrieval.
_REGULAR_SUFFIXES = ("s", "es", "ed", "d", "ing", "er", "ers")


def _is_inflected_known_word(token: str) -> bool:
    return any(
        token.endswith(suffix) and token[: -len(suffix)] in _KNOWN_WORDS
        for suffix in _REGULAR_SUFFIXES
    )


def _fuzzy_candidates(token: str) -> list[str]:
    """Allowlist words one edit from `token`, sharing its first letter.

    The shared-first-letter requirement is what stops the whole class of
    "delete the leading character and land on a different word" false
    positives: without it "chain" became "hain" and "What about the
    chain?" retrieved nothing at all. First-character typos are rare in
    practice, and the one case that matters here -- a dropped leading
    character -- is handled separately, and far more narrowly, by
    `_leading_char_candidates`.
    """
    candidates = {
        word
        for length in (len(token) - 1, len(token), len(token) + 1)
        for word in _BY_LENGTH.get(length, ())
        if word[:1] == token[:1] and _within_one_edit(word, token)
    }
    return sorted(candidates)


# The only words a missing LEADING character may be repaired INTO. Scoped to
# a curated handful rather than the whole allowlist: with the allowlist as the
# candidate pool, "Ab English me continue karo" had its "Ab" ("now") rewritten
# to "kab" ("when"), which is a different word, not a typo of the same one.
# Every entry here is a word whose first keystroke is genuinely easy to drop
# and whose truncation is not itself a common word.
_LEADING_CHAR_REPAIRABLE: frozenset[str] = frozenset({
    "tum", "aap", "kya", "kaise", "kaun", "karo", "kar", "batao", "mujhe",
    "mere", "sakte", "sakta", "sakti", "help", "what", "hain",
})


def _leading_char_candidates(token: str) -> list[str]:
    """Curated words that are `token` with exactly one extra leading
    character -- the "um" -> "tum" case."""
    return sorted({word for word in _LEADING_CHAR_REPAIRABLE if word[1:] == token})


def _match_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Character spans that no correction may touch: quoted text, and every
    whitespace-delimited chunk with identifier structure (digits, `@`, a path
    separator, a file extension). Covers phone and invoice numbers, IMEIs,
    amounts, dates, file names and anything the user put in quotes."""
    spans = [match.span() for match in _QUOTE_SPAN.finditer(text)]
    spans += [match.span() for match in re.finditer(r"\S+", text) if _IDENTIFIERISH.search(match.group())]
    return spans


def _in_any_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


def _is_mid_sentence(text: str, start: int) -> bool:
    """Whether the token at `start` follows real sentence content.

    The first word of the message, and the first word after a sentence
    terminator, are sentence-initial -- their capitalisation says nothing. A
    Capitalized token anywhere else is a name, a place or a proper noun.
    """
    preceding = text[:start].rstrip()
    if not preceding:
        return False
    return preceding[-1] not in ".!?।॥؟۔:"


def normalize_for_routing(text: str) -> NormalizationResult:
    """The one entry point. Never raises, never mutates `text`."""
    original = text or ""
    words = [word.lower() for word in _LATIN_WORD.findall(original)]
    if not words or len(words) > MAX_TOKENS_FOR_NORMALIZATION:
        return NormalizationResult(original, original, original)

    protected = _protected_spans(original)
    cue_count = sum(1 for word in words if word in _CONTROL_TERMS)

    corrections: list[tuple[str, str]] = []
    ambiguities: list[tuple[str, tuple[str, ...]]] = []
    pieces: list[str] = []
    cursor = 0
    for match in _LATIN_WORD.finditer(original):
        start, end = match.span()
        pieces.append(original[cursor:start])
        cursor = end
        pieces.append(
            _correct_token(
                match.group(),
                protected=_in_any_span(start, end, protected),
                mid_sentence=_is_mid_sentence(original, start),
                cue_count=cue_count,
                corrections=corrections,
                ambiguities=ambiguities,
            )
        )
    pieces.append(original[cursor:])

    normalized = "".join(pieces)
    retrieval_query = original if normalized == original else f"{original} {normalized}"
    return NormalizationResult(
        original=original,
        normalized_for_routing=normalized,
        retrieval_query=retrieval_query,
        corrections=tuple(corrections),
        ambiguities=tuple(ambiguities),
    )


def _correct_token(
    token: str,
    *,
    protected: bool,
    mid_sentence: bool,
    cue_count: int,
    corrections: list[tuple[str, str]],
    ambiguities: list[tuple[str, tuple[str, ...]]],
) -> str:
    if protected:
        return token
    lowered = token.lower()
    # Exact match wins outright: an in-vocabulary word is never "corrected",
    # and neither is a regular inflection of one.
    if lowered in _KNOWN_WORDS or _is_inflected_known_word(lowered):
        return token
    if token.isupper() and len(token) > 1:
        # An acronym or an identifier (FIR, BNS, GST, IMEI), never a typo.
        return token
    if mid_sentence and token[:1].isupper():
        # A party name, a place, a document title.
        return token

    alias = _ALIASES.get(lowered)
    if alias is not None:
        corrections.append((token, alias))
        return _match_case(alias, token)

    if len(lowered) >= MIN_LENGTH_FOR_FUZZY:
        candidates = _fuzzy_candidates(lowered)
        if len(candidates) == 1:
            corrections.append((token, candidates[0]))
            return _match_case(candidates[0], token)
        if len(candidates) > 1:
            ambiguities.append((token, tuple(candidates)))
        return token

    if len(lowered) >= 2 and cue_count >= _MIN_CUES_FOR_LEADING_CHAR_REPAIR:
        candidates = _leading_char_candidates(lowered)
        if len(candidates) == 1:
            corrections.append((token, candidates[0]))
            return _match_case(candidates[0], token)
        if len(candidates) > 1:
            ambiguities.append((token, tuple(candidates)))
    return token


def clarification_question(result: NormalizationResult, language: str = "english") -> str | None:
    """A short "did you mean X or Y?" for an ambiguous CONTROL-word typo.

    Returns `None` whenever there is nothing to ask about, so callers can use
    it as the condition itself. Never proposes a correction it has not listed:
    the alternatives shown are exactly the tied candidates.
    """
    control = result.control_ambiguities
    if not control:
        return None
    token, candidates = control[0]
    shown = ", ".join(f"“{candidate}”" for candidate in candidates[:3])
    if language == "hindi":
        return (
            f"मैं “{token}” ठीक से "
            f"समझ नहीं पाया — "
            f"आपका मतलब {shown} में "
            f"से कौन सा था?"
        )
    if language == "hinglish":
        return f"Main “{token}” samajh nahi paya — aapka matlab {shown} me se kaunsa tha?"
    return f"I could not read “{token}” — did you mean {shown}?"


def strip_diacritics(text: str) -> str:
    """A plain fold of Latin diacritics. Not used by the routing pass; kept
    here so there is one implementation for callers that need it."""
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
