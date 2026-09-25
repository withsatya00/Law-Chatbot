import re
import unicodedata
from dataclasses import dataclass

from app.drafting.templates import list_templates
from app.drafting.title_translations import all_localized_names
from app.language.normalizer import QueryNormalizer

# Verbs that are inherently document-authoring -- the ONLY signals that may
# start Draft Mode. Deliberately excludes softer verbs like "need"/"want"/
# "file" (previously allowed when paired with a document noun): "file a
# complaint" and "how can I file a consumer complaint" use the same words
# without the user asking to draft anything, and that ambiguity was
# repeatedly hijacking informational questions into drafting mode. Requiring
# one of these five explicit verbs is a deliberate precision-over-recall
# trade-off -- a softly-phrased genuine drafting wish ("I need an RTI
# application") won't auto-start a draft, but no question can be mistaken
# for one either.
_STRONG_DRAFT_VERB_PATTERN = re.compile(r"\b(create|make|generate|prepare|write|draft)\b", re.IGNORECASE)

# Hindi/Hinglish generic drafting cues (e.g. "RTI application banana hai",
# "legal notice likhni hai") -- specific enough phrasing to count as strong
# evidence on their own, same as the English strong-verb list above.
_STRONG_DRAFT_PHRASE_PATTERNS = [
    re.compile(r"\b(banana|banani|banao|banaiye|likhni|likhna)\s+hai\b", re.IGNORECASE),
    re.compile(r"बनाना\s+है|बनानी\s+है|लिखनी\s+है|लिखना\s+है|बनाओ|बनाइए|ड्राफ्ट"),
    # Part 55 "Draft Audit -- Language Propagation Fix": "तैयार करें/करो/
    # कीजिए/करना" ("prepare [it]") -- the standard, arguably more common
    # Hindi phrasing for asking a document be prepared ("मसौदा तैयार करें" =
    # "prepare the draft") -- was missing entirely from this list even
    # though Part 54 added the equivalent native "prepare" verb stem for
    # Tamil/Telugu/Kannada/Bengali (தயார்/తయారు/ತಯಾರಿಸ, all below). Confirmed
    # root cause of a real bug report's own opening message ("तमिल में
    # पुलिस शिकायत का मसौदा तैयार करें।") never being recognized as a
    # drafting request at all under `_has_drafting_verb` -- it fell through
    # to ordinary chat before `extract_requested_language` ever got a
    # chance to run.
    re.compile(r"तैयार\s+(कर|की)"),
    # "bana do"/"likh do" ("make/write it") -- the casual imperative form of
    # the same request, distinct from "banana/likhna hai" above. Part 32's
    # `_CONFIRMATION_TOKEN` (`app/drafting/conversation.py`) already treats
    # "bna do"/"likh do" as a valid casual command when CONTINUING an
    # in-progress draft; this is the same phrasing recognized for STARTING
    # one (e.g. "Legal notice bana do"), which previously matched no pattern
    # here at all and silently fell through as ordinary chat.
    re.compile(r"\bbana\s*do\b|\blikh\s*do\b", re.IGNORECASE),
    # Bare romanised imperatives. The list above only matched these when
    # followed by "hai" ("banana hai"), and `bana do` only in its two-word
    # form -- so the single most natural way to ask, "police complaint
    # banao", matched NOTHING and never started a draft at all. Confirmed:
    # "Hindi me police complaint banao" produced no draft and no language,
    # while the otherwise-identical "Tamil me police complaint draft karo"
    # worked purely because it happened to contain the English word "draft".
    re.compile(
        r"\b(banao|banaao|bnao|banaiye|banaye|banayen|banaden|banade"
        r"|likho|likhiye|likhen|likhdo|likh\s*de)\b",
        re.IGNORECASE,
    ),
    # "ready kar do" / "taiyar kar do" -- the Hinglish counterpart of the
    # Devanagari "तैयार करें" already matched above. The romanised spelling is
    # what people actually type, and it matched nothing: confirmed live, an
    # assistant offer ("I can prepare a legal notice for you if you'd like")
    # answered with "Document ready kar do" was not recognised as a drafting
    # request at all -- it fell through to retrieval and came back "no
    # verified document in the Knowledge Base", i.e. the app declined its own
    # offer. Requires the explicit imperative tail ("kar/kr do", "karo",
    # "kijiye") so a bare "is the document ready?" is not read as a command.
    re.compile(
        r"\b(ready|taiyar|taiyyar|tayyar|tayar)\s*(kar|kr|kar\s*ke)?\s*"
        r"(do|de|den|do\s*na|dijiye|dijiyega|karo|kro|kariye|kijiye|karein|kren)\b",
        re.IGNORECASE,
    ),
    # Part 54 "Complete Multilingual Drafting Pass" item 2: native
    # create/prepare/write/draft verb stems for Tamil, Telugu, Kannada, and
    # Bengali -- previously a draft could only ever be STARTED with an
    # English or Hindi/Hinglish verb (`_STRONG_DRAFT_VERB_PATTERN`/the two
    # patterns above), even after Part 51 fixed document-NAME recognition
    # for these languages; a message using only a Tamil/Telugu/Kannada/
    # Bengali verb ("சட்ட அறிவிப்பு உருவாக்கு") still failed
    # `_has_drafting_verb` and fell through as ordinary chat. Deliberately
    # no `\b` word-boundary here (see `_contains_as_word` below and its
    # docstring for why `\b` is unreliable on these scripts) -- these are
    # distinctive enough verb stems that a plain substring `re.search` is
    # the same precision/recall trade-off the bare Hindi patterns above
    # already accept (e.g. "बनाओ"/"ड्राफ्ट").
    re.compile(r"உருவாக்|தயார்\s*செய்|தயாரி|எழுது|வரைந்து|வரைய"),  # Tamil: create/prepare/write/draft
    re.compile(r"తయారు\s*చేయ|రాయ|సృష్టించ|డ్రాఫ్ట్"),  # Telugu: prepare/write/create/draft
    re.compile(r"ತಯಾರಿಸ|ಬರೆ|ರಚಿಸ|ಡ್ರಾಫ್ಟ್"),  # Kannada: prepare/write/create/draft
    re.compile(r"তৈরি|লিখ|খসড়া"),  # Bengali: make/write/draft
    # Part 57 "Drafting Lifecycle Redesign": native create/prepare/write/
    # draft verb stems for the 6 languages added to the priority chrome tier
    # (malayalam/marathi/gujarati/punjabi/odia/urdu), mirroring the Tamil/
    # Telugu/Kannada/Bengali block above -- same rationale, same
    # no-`\b`-boundary substring-match approach.
    re.compile(r"തയ്യാറാക്|എഴുത|തയ്യാറ്|ഡ്രാഫ്റ്റ്"),  # Malayalam: prepare/write/draft
    re.compile(r"तयार\s*कर|बनव|लिही|मसुदा"),  # Marathi: prepare/make/write/draft
    re.compile(r"તૈયાર\s*કર|બનાવ|લખ|ડ્રાફ્ટ"),  # Gujarati: prepare/make/write/draft
    re.compile(r"ਤਿਆਰ\s*ਕਰ|ਬਣਾ|ਲਿਖ|ਡਰਾਫਟ"),  # Punjabi: prepare/make/write/draft
    re.compile(r"ପ୍ରସ୍ତୁତ\s*କର|ତିଆରି|ଲେଖ|ଡ୍ରାଫ୍ଟ"),  # Odia: prepare/make/write/draft
    re.compile(r"تیار\s*کر|بنا|لکھ|ڈرافٹ"),  # Urdu: prepare/make/write/draft
    # The remaining Eighth Schedule languages. Same rationale and same
    # no-word-boundary substring approach as the blocks above: without a verb
    # stem here, `_has_drafting_verb` never fires and a drafting request in
    # these languages falls through to ordinary RAG chat no matter how clearly
    # it names the document. Confirmed live for Nepali -- "मलाई प्रहरी उजुरी
    # तयार गर्नुपर्ने छ" was answered as a general legal question, because the
    # only nearby pattern was Marathi's "तयार\s*कर", and Nepali conjugates the
    # same verb as "तयार गर्नु-", with "गर्नु" (not "कर") after the space.
    re.compile(r"तयार\s*(गर|पार)|बनाउन|लेख्न|मस्यौदा|गर्नुहोस्"),  # Nepali
    re.compile(r"रचय|लिख|निर्माण|प्रारूप|करोतु|कुरु"),  # Sanskrit
    re.compile(r"तैयार\s*कर|बनाउ|लिखू|मसौदा"),  # Maithili
    re.compile(r"तयार\s*कर|बरय|करचें|मसुदो"),  # Konkani
    re.compile(r"तैयार\s*कर|बनाओ|लिक्खो|लिखो"),  # Dogri
    # "खालाम" ("do") deliberately excluded: unlike "बानाय"/"लिर" (make/write),
    # it's the same bare verb stem used in the ordinary question "आं मा
    # खालामनो हायो" ("what should I do?"), not just the imperative "do it!".
    # With no word-boundary/imperative distinction (substring match, per this
    # block's own rationale above), including it wrongly launched Draft Mode
    # for a plain FIR-registration question -- the same softness the English
    # strong-verb list already avoids by excluding "do" (see module docstring).
    re.compile(r"बानाय|लिर"),  # Bodo: make/write
    re.compile(r"ꮸꯦꯍꮚꮾ|ꮏꮾ|ꮸꯦꯍꮗꯤ"),  # Manipuri: make/write
    re.compile(r"Მᱴᱞᱠ|ᱚᲞ|ᱤᱴᱚᱠ"),  # Santali: prepare/write/make
    re.compile(r"تيار\s*ڪر|لڪو|ٮاهي|مسود"),  # Sindhi
    re.compile(r"تیار\s*کر|لیکھ|بناو"),  # Kashmiri
    # Assamese "toiyar" has two equally common encodings that Unicode
    # normalization does NOT unify: precomposed U+09DF, and U+09AF followed by
    # the nukta U+09BC. NFC deliberately leaves Bengali/Assamese nukta forms
    # decomposed, so a pattern written with only one of them silently misses
    # every message typed with the other -- both are matched here.
    re.compile(
        r"তৈ(?:য়|য়)াৰ\s*কৰ"
        r"|প্ৰস্তুত|লিখ"
    ),  # Assamese: prepare/write
]


# Gate for `DraftIntentDetector.detect`'s no-template-evidence-at-all
# fallback (see its own comment for the live incident this fixes).
#
# A POSITIVE "must mention something legal" requirement was tried first and
# reverted: it broke 25 existing, deliberately-generic drafting triggers
# ("create draft", "draft chahiye", "Document ready kar do", "ye ready kr
# do") that name no document type and no dispute/legal noun at all -- that
# vagueness is the whole point of the "just tell me what happened" discovery
# flow this fallback starts, so requiring a legal keyword up front defeats
# it. The actual failure mode is narrower than "not legal-sounding enough":
# the message names a CONCRETE, clearly non-legal deliverable instead (code,
# a recipe, a poem) -- something no legal-document template could ever be
# about, as opposed to a genuinely topic-less "please draft something"
# request. A denylist for that specific, narrow shape is precision-safe in
# a way a broad positive requirement is not: it only suppresses a message
# that explicitly names one of these, never a vague one.
_NON_LEGAL_DELIVERABLE_RE = re.compile(
    r"\b(code|program(?:me|ming)?|script|algorithm|function|recipe|dish|"
    r"poem|poetry|song|lyrics|joke|riddle|quiz|essay|story|"
    r"painting|drawing|sketch|rangoli|cake|"
    r"workout|diet\s*plan|"
    r"pizza|biryani)\b",
    re.IGNORECASE,
)


def _names_non_legal_deliverable(text: str) -> bool:
    return bool(_NON_LEGAL_DELIVERABLE_RE.search(text))


# Part 51 "Draft Generation Quality Pass" item 2: Python re's `\b` defines a
# "word" character as whatever `\w` matches, which does NOT include Unicode
# category Mn (non-spacing mark, e.g. Devanagari/Tamil virama "्"/"்") or Mc
# (spacing combining mark, e.g. Tamil/Telugu/Kannada dependent vowel signs
# like "ி"/"ా"/"ಿ"). Tamil/Telugu/Kannada words overwhelmingly END in
# exactly such a combining mark (they're abugida scripts), so `\bphrase\b`
# could essentially never match at the natural end of a Tamil/Telugu/Kannada
# trigger phrase -- confirmed root cause of Tamil document names like
# "சட்ட அறிவிப்பு" never being recognized, even though the exact phrase was
# present in the lookup tables. Hindi happened to mostly dodge this in
# existing test coverage (many Hindi trigger words end in a bare consonant
# with no explicit matra), which is why this went unnoticed until Part 51's
# Tamil examples specifically exposed it.
_WORD_CHAR_CATEGORIES = {"Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Nd"}


def _is_word_char(char: str) -> bool:
    return unicodedata.category(char) in _WORD_CHAR_CATEGORIES


def _contains_as_word(haystack: str, needle: str) -> bool:
    """True if `needle` occurs in `haystack` at a position not immediately
    adjacent to another word-forming character on either side -- a
    script-agnostic replacement for `re.search(rf"\\b{needle}\\b", haystack)`
    that correctly treats Indic combining marks as word-continuing (see
    module docstring above) while still rejecting a partial-word match like
    "fir" inside "confirm" for Latin-script phrases, exactly as `\\b` did.
    """
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index == -1:
            return False
        before_ok = index == 0 or not _is_word_char(haystack[index - 1])
        after_index = index + len(needle)
        after_ok = after_index == len(haystack) or not _is_word_char(haystack[after_index])
        if before_ok and after_ok:
            return True
        start = index + 1


# Words that carry no identifying force inside a document name -- they occur in
# most template names in their language, so overlapping on them alone says
# nothing about WHICH document was meant. Excluded from the partial-match token
# count below (an exact contiguous match is unaffected).
_GENERIC_NAME_TOKENS = frozenset({
    "application", "notice", "complaint", "agreement", "request", "certificate",
    "letter", "affidavit", "general", "for", "to", "the", "of", "a", "an",
    # Problem-first discovery (Part: "Intelligent Draft Discovery"): the
    # drafting-VERB filler words that `_has_drafting_verb`'s own gate
    # requires before `detect()` runs at all ("banana hai", "karo",
    # "chahiye", "draft", ...) previously counted as ordinary signal tokens
    # here. Several templates' own `trigger_phrases` spell out a natural
    # Hinglish imperative in full (e.g. legal_notice.yaml's "legal notice
    # banana hai"), so a completely generic message like "Legal document
    # banana hai" shared enough non-generic-by-the-OLD-definition tokens
    # ("legal", "banana", "hai") with that one template's phrase to score
    # above `_VERB_BACKED_NAME_THRESHOLD` via `_partial_name_overlap` alone
    # -- resolving a generic "I need SOME document" request straight to one
    # specific template with no real signal behind it, exactly the silent
    # unrelated-template mapping the discovery flow (`DraftConversationEngine.
    # _start_discovery`) exists to prevent. These carry no information about
    # WHICH document is meant (that's the entire purpose of this exclusion
    # set), so excluding them can only ever LOWER a loose-match score, never
    # cause a real, distinctively-worded paraphrase (e.g. "mobile chori ki
    # police complaint", which overlaps on "mobile"/"chori" -- untouched by
    # this list) to stop matching.
    "banana", "banani", "banao", "banaiye", "banaye", "banayen", "banwana",
    "likhna", "likhni", "likho", "likhiye", "likhdo",
    "hai", "hain", "karo", "karna", "karni", "kar", "kro", "kr",
    "chahiye", "draft", "drafting", "generate", "prepare", "create", "make", "write", "do",
    # Bare Hindi/Hinglish postpositions and pronouns ("ke liye" = "for",
    # "ka"/"ki"/"ke" = possessive markers, "mujhe" = "to me") -- grammatical
    # connective tissue present in nearly every Hinglish sentence regardless
    # of which document is meant, the exact same false-signal problem as the
    # drafting-verb fillers above. Confirmed: "Mujhe case ke liye draft
    # chahiye" (a `MUJHE_CASE`-style fully generic request) shared "ke"/
    # "liye" with `job_application`'s own "job ke liye application" trigger
    # phrase and scored above the loose-match threshold purely on those two
    # grammatical words, resolving a generic request straight to an
    # unrelated template.
    "mujhe", "mera", "meri", "mere", "ke", "liye", "ka", "ki", "ko", "se", "mein", "case",
    # The SAME false-signal problem as the two blocks above, but in
    # Devanagari script rather than romanised Hinglish -- only the romanised
    # spellings were excluded, so a native-script message in Hindi, Dogri,
    # Marathi, etc. ("मैंकुं दस्तावेज तैयार करना है" -- Dogri for "I need to
    # prepare a document", entirely generic) still shared "करना"/"है" with
    # `account_closure_application`'s own trigger phrase "खाता बंद करना है"
    # and matched that one specific, unrelated template outright. Devanagari
    # is shared across several languages this app supports (Hindi, Marathi,
    # Dogri, Konkani, Maithili, Nepali, Sanskrit), so one addition here
    # covers the same class of false match in all of them at once.
    "करना", "करने", "करो", "करूं", "कीजिए", "किया",
    "है", "हैं", "हो", "था", "थी",
    "बनाना", "बनानी", "बनाओ", "बनाइए", "तैयार",
    "लिखना", "लिखनी", "लिखो", "लिखिए",
    "चाहिए", "मुझे", "मेरा", "मेरी", "मेरे",
    "के", "की", "का", "को", "से", "में", "लिए",
})

# Part 58 issue 17/22: a message asking to STOP drafting ("cancel kr do draft
# ko", "ड्राफ्ट रद्द करें") contains the word "draft" and, in Hinglish, the
# imperative "kar do" -- both strong drafting cues. Without this guard,
# `detect()` read it as an attempt to START an unnamed new document and
# replied with the full 56-template menu, which reads as "your cancel was
# understood as a new request." The chat layer owns cancellation
# (`ChatService._CANCEL_DRAFT_PATTERN`); this only guarantees the detector
# never competes with it.
_CANCEL_INTENT_PATTERN = re.compile(
    r"\b(cancel|discard|delete|remove|abort|scrap)\b"
    r"|\b(cancel|band|bnd|hata|hatao|hata\s*do|rehne\s*do|chhod\s*do|chod\s*do)\b\s*(kar|kr|kro|karo|do)?\b"
    r"|रद्द|निरस्त|हटा\s*द|बंद\s*कर|रहने\s*द"
    r"|रद्द\s*कर|रद्द\s*करा|રદ\s*કર|બંધ\s*કર|বাতিল|ரத்து|రద్దు|ರದ್ದು|റദ്ദ|منسوخ",
    re.IGNORECASE,
)


def _name_tokens(phrase: str) -> list[str]:
    """Content tokens of a document name, with any parenthetical clarifier
    dropped -- "Police Complaint (Application to SHO)" identifies itself by
    "police complaint"; the bracketed part explains where it goes."""
    head = phrase.split("(")[0]
    return [token for token in re.split(r"[\s,/-]+", head.strip()) if token]


def _levenshtein(a: str, b: str, max_dist: int) -> int:
    """Optimal-string-alignment edit distance between `a` and `b` (insertion/
    deletion/substitution, PLUS one adjacent transposition counted as a
    single edit), or `max_dist + 1` the moment it's certain the true
    distance exceeds `max_dist` -- exits early rather than computing the
    full distance, since every caller here only ever asks "is it within N?"
    and this runs across every (template trigger phrase, message token)
    pair on every drafting-verb-classified message.

    Transposition matters here specifically: swapping two adjacent letters
    ("money" -> "moeny", "form" -> "from") is one of the most common typing
    slips there is, and plain Levenshtein charges it 2 (delete + insert, or
    two substitutions) -- which put a completely ordinary transposition
    typo on a short word OUTSIDE the distance-1 budget `_typo_tolerant_match`
    gives 4-5 letter words, silently failing to recognise it while a
    same-cost substitution typo on the same word succeeded.
    """
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    # Two previous rows are kept (not one) so a transposition move can look
    # two rows/columns back -- `prev2_row[j - 2]` is the cell for
    # `a[:i-2]` vs `b[:j-2]`, the state before both transposed characters.
    prev2_row: list[int] = []
    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        row_min = current_row[0]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            best = min(
                previous_row[j] + 1,        # deletion
                current_row[j - 1] + 1,     # insertion
                previous_row[j - 1] + cost,  # substitution
            )
            if (
                i > 1 and j > 1 and cost == 1
                and char_a == b[j - 2] and a[i - 2] == char_b
            ):
                best = min(best, prev2_row[j - 2] + 1)  # adjacent transposition
            current_row[j] = best
            row_min = min(row_min, best)
        if row_min > max_dist:
            # Every entry in this row already exceeds the budget -- no cell
            # in any later row can recover, since each row can only grow
            # from the row before it.
            return max_dist + 1
        prev2_row, previous_row = previous_row, current_row
    return previous_row[-1]


def _typo_tolerant_match(word: str, other: str) -> bool:
    """Whether `word` and `other` are plausibly the same word typed with one
    ordinary slip (a missing/doubled/swapped/wrong letter) -- "renta" vs
    "rent", "polcie" vs "police", "agrement" vs "agreement". Never applied
    to words under 4 characters (too easy for a short, genuinely different
    word to land within distance 1 of another by chance).

    Capped at distance 1, full stop -- not distance 2 for longer words, as a
    previous version of this function allowed. That distance-2 allowance was
    already tightened once ("place" vs "police" coincidentally landing at
    distance 2 purely because "police" is 6 characters -- see the removed
    "shorter word decides the threshold" logic), but the underlying problem
    was the distance-2 budget itself, not just which word's length picked
    it: confirmed live via a QA pass on 2026-09-11, "response period 15
    days" (an ordinary phrase, nothing to do with any document) fuzzy-
    matched "person" (both 6 letters, genuinely distance 2 apart) and, in
    combination with an unrelated genuine "missing" token elsewhere in the
    same message, was alone enough to outscore an exact "legal notice"
    match and launch the wrong draft template (`missing_person_report`
    instead of `legal_notice`). Every legitimate typo case already covered
    by this app's own test suite -- "polcie"/"police", "agrement"/
    "agreement", "domestik"/"domestic", the transposition cases -- is
    distance 1; none of them actually needed distance 2 to be recognised.
    """
    if word == other:
        return True
    if len(word) < 4 or len(other) < 4:
        return False
    return _levenshtein(word, other, 1) <= 1


def _partial_name_overlap(message_tokens: set[str], phrase: str) -> float:
    """Score for a document name the message names LOOSELY rather than
    verbatim.

    `_contains_as_word` only fires on the whole stored name appearing
    contiguously, which misses the way people actually type: a Tamil speaker
    asking for a police complaint writes "kaaval pukaar" (two words), while the
    stored name is "kaaval nilaya pukaar" (three, with an extra word in the
    middle) -- so the request scored 0 against its own template and came back
    as "I don't have a template for that". Confirmed for Tamil, which has a
    complete set of translated names: the gap was never coverage, it was that
    only an exact match counted.

    Scored strictly below an exact contiguous match so a verbatim name always
    outranks a loose one, and requires at least two overlapping NON-generic
    tokens so "complaint" alone can never pick a template -- with several
    complaint templates per language, one shared generic word is noise.

    "Overlapping" tolerates one ordinary typo per word (see
    `_typo_tolerant_match`) as well as an exact match -- "renta agrement
    banao" previously scored 0 against "rent agreement" (neither "renta"
    nor "agrement" is literally "rent"/"agreement"), so a plainly-typed,
    unambiguous direct request fell through to problem-first discovery
    purely because of two ordinary typos, instead of being recognised
    outright the way the same request without the typos already was.
    """
    tokens = {token for token in _name_tokens(phrase.lower()) if token not in _GENERIC_NAME_TOKENS}
    if len(tokens) < 2:
        # A single-word distinctive token ("police" in "police complaint")
        # was tried as a typo-tolerant match on its own and reverted: it
        # kept re-litigating the specificity ordering between a generic
        # parent template and a more specific child sharing that same word
        # ("mobile chori ki police complaint" must still resolve to
        # `mobile_theft_complaint`, not the more generic `police_complaint`
        # -- see `_most_specific_exact_match` below, tuned for exactly this
        # case). Multi-token overlap (below) is where typo-tolerance is
        # applied instead -- it already requires 2+ independently-typed
        # words to agree, which is strong enough evidence that this
        # specificity conflict does not arise there.
        return 0.0
    # A message's own drafting-verb/postposition filler words ("karni",
    # "hai", "ke", ...) are excluded from FUZZY comparison the same way
    # they are already excluded from the phrase side -- confirmed live:
    # "kanuni" ("legal" in Hindi, one template's own distinctive word)
    # happens to sit within typo distance of "karni" (an extremely common
    # Hinglish grammatical filler meaning "to do", present in a huge
    # fraction of ALL Hinglish messages via "... karni hai"). These words
    # carry no information about which document is meant even when spelled
    # correctly, so they must not count as evidence when spelled almost
    # like something else either. Exact matches are unaffected (a phrase
    # token literally equal to a filler word never happens, since filler
    # words are excluded from the phrase side too).
    fuzzy_candidates = message_tokens - _GENERIC_NAME_TOKENS
    shared = {
        token for token in tokens
        if any(_typo_tolerant_match(token, message_token) for message_token in fuzzy_candidates)
    }
    if len(shared) < 2 or len(shared) * 2 < len(tokens):
        return 0.0
    return 0.15 + 0.02 * len(shared)


def _has_drafting_verb(normalized: str) -> bool:
    if _STRONG_DRAFT_VERB_PATTERN.search(normalized):
        return True
    return any(pattern.search(normalized) for pattern in _STRONG_DRAFT_PHRASE_PATTERNS)


def _draft_command_scope(normalized: str) -> str:
    """Return the sentence-like clause containing the drafting command.

    Background facts can mention a different existing document (for example,
    "I filed an online police complaint") before the user explicitly asks for
    a Lost Document Affidavit. Scoring the full narrative let those incidental
    references outvote the document named in the command itself.
    """
    clauses = [part.strip() for part in re.split(r"[.!?\u0964\u0965\u06d4]+", normalized) if part.strip()]
    command_clauses = [clause for clause in clauses if _has_drafting_verb(clause)]
    return " ".join(command_clauses) if command_clauses else normalized


# A message that's just *asking about* a document type ("what is FIR", "how
# do I file an FIR", "I want to know about RTI", "recommend a lawyer for
# cheque bounce") is not a drafting request -- without this guard, a bare
# trigger-phrase match (e.g. the word "FIR" alone) would wrongly hijack
# ordinary informational questions or lawyer-recommendation asks into
# drafting mode. Overridden by an explicit drafting verb (e.g. "how do I
# write ..."), see _has_drafting_verb above.
#
# `_SENTENCE_START` stands in for a bare `^` anchor on patterns that only make
# sense at the start of a question ("what is X", "how do I Y"). A plain `^`
# only matches the very start of the whole message, so a real, common pattern
# -- a short scene-setting sentence followed by the actual question ("I was
# arrested without a warrant. What are my legal rights?" / "My employer
# hasn't paid me. What should I do?") -- silently failed to register as
# informational, because the question wasn't literally the first thing typed.
# Matching after a sentence-ending punctuation mark too closes that gap.
_SENTENCE_START = r"(?:^|[.!?]\s+)"
_INFORMATIONAL_PATTERNS = [
    re.compile(rf"{_SENTENCE_START}\s*what\s+(is|are|does|do)\b", re.IGNORECASE),
    # "what documents/proof/steps is/are ..." -- a noun sitting between "what"
    # and "is/are" (very common phrasing, e.g. "what documents are required")
    # previously fell through this whole pattern set entirely.
    re.compile(rf"{_SENTENCE_START}\s*what\s+\w+\s+(is|are|does|do)\b", re.IGNORECASE),
    # "what should I do" / "what I should do" (either word order -- informal
    # phrasing commonly swaps them) -- a request for guidance on next steps,
    # not a drafting command. Anchored loosely (not just at the very start)
    # since it's commonly preceded by filler ("umm what should i do...").
    re.compile(r"\bwhat\s+(should|can|could|must)\s+(i|we|one)\s+do\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(i|we|one)\s+(should|can|could|must)\s+do\b", re.IGNORECASE),
    re.compile(rf"{_SENTENCE_START}\s*(explain|define)\b", re.IGNORECASE),
    re.compile(rf"{_SENTENCE_START}\s*(how|why|when|where)\s+(do|does|can|to|is|are)\b", re.IGNORECASE),
    # Modal-verb-first questions ("Can police refuse...", "Should I...",
    # "Would this count as...") -- previously unrecognized as informational
    # at all, so a message like "Can police refuse to register an FIR?"
    # (no drafting verb, but incidentally matching a template's trigger
    # phrase like "FIR") fell through to being treated as a drafting request.
    # Excludes a strong drafting verb right after the modal (with an
    # optional "you" in between): "Can you draft the complaint?" is a
    # polite REQUEST addressed to the assistant, not a question about the
    # world, and must still reach `_has_drafting_verb`'s match below --
    # confirmed live, this pattern previously caught it too, so a genuine
    # follow-up drafting request ("Can you draft the complaint?", turn 4 of
    # a conversation that had already established the facts) was wrongly
    # classified as informational and never started drafting.
    re.compile(
        rf"{_SENTENCE_START}\s*(can|could|will|would|should|may|might)\s+"
        rf"(?!(?:you\s+)?(?:create|make|generate|prepare|write|draft)\b)\w+",
        re.IGNORECASE,
    ),
    re.compile(r"\bmeaning of\b", re.IGNORECASE),
    re.compile(r"\bwant to know about\b", re.IGNORECASE),
    re.compile(r"\btell me about\b", re.IGNORECASE),
    re.compile(r"\bguide me\b", re.IGNORECASE),
    re.compile(r"\binformation (on|about)\b", re.IGNORECASE),
    re.compile(r"\brecommend\s+(a|an|me a)?\s*(lawyer|advocate|vakil)\b", re.IGNORECASE),
    re.compile(r"\bfind me\s+(a|an)\s+(lawyer|advocate)\b", re.IGNORECASE),
    re.compile(r"\bkaise\b", re.IGNORECASE),
    # Covers all grammatical-gender/number forms of "what is X" -- "kya hai"
    # (gender-neutral), "kya hota hai" (masc.), "kya hoti hai" (fem., e.g. "FIR
    # kya hoti hai" since FIR/report is feminine in Hindi), "kya hote hain"
    # (plural). Missing the fem./plural forms previously let questions like
    # "FIR kya hoti hai" fall through and false-match the draft trigger
    # phrase "fir" instead of being recognized as an informational question.
    re.compile(r"\bkya\s+(hai|hota\s+hai|hoti\s+hai|hote\s+hain)\b", re.IGNORECASE),
    re.compile(r"\bkise\s+(kahte|kehte)\s+(hai|hain)\b", re.IGNORECASE),
    re.compile(r"\bdifference between\b", re.IGNORECASE),
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\b\w+\s+vs\.?\s+\w+\b", re.IGNORECASE),
    re.compile(r"\bsawaal\b", re.IGNORECASE),
    re.compile(r"\bke\s+baare\s+mein\b", re.IGNORECASE),
    re.compile(r"क्या\s+(है|होता\s+है|होती\s+है|होते\s+हैं)"),
    re.compile(r"किसे\s+(कहते|कहा\s+जाता)\s+(है|हैं)"),
    re.compile(r"कैसे"),
    # Part 58 "Answer Quality Audit" issue 19: a noun sitting between the
    # interrogative and the copula -- "क्या अंतर है" ("what is the
    # difference"), "kya farak hai", "क्या प्रक्रिया है" -- matched none of
    # the patterns above, which all require "क्या"/"kya" to be IMMEDIATELY
    # followed by "है"/"hai". Confirmed root cause of a live regression:
    # "जमानत और अग्रिम जमानत में क्या अंतर है?" asked mid-draft was not
    # recognized as a question at all, so `_is_draft_interruption` returned
    # False and the drafting state machine swallowed a genuine legal
    # question, re-showing its pending-field list instead of answering.
    re.compile(r"\bkya\s+\w+\s+(hai|hain|hota\s+hai|hoti\s+hai|hote\s+hain)\b", re.IGNORECASE),
    re.compile(r"क्या\s+\S+\s+(है|हैं|होता\s+है|होती\s+है|होते\s+हैं)"),
    # Comparison questions in Hindi/Hinglish ("... me kya antar hai", "... में
    # क्या फर्क है"), the direct counterpart of "difference between" above.
    re.compile(r"\b(kya|क्या)\s+(antar|farak|fark|अंतर|फर्क|फ़र्क|अन्तर)\b", re.IGNORECASE),
    re.compile(r"(अंतर|फर्क|फ़र्क|अन्तर)\s+(है|क्या)"),
    # A sentence-initial "which" is interrogative, never a command ("Which
    # draft were we creating?", "Which documents do I need?"). It was the one
    # WH-word none of the patterns above covered, so a memory-recall question
    # containing the word "draft" cleared `_has_drafting_verb` and was
    # classified as a request to start a new draft.
    re.compile(rf"{_SENTENCE_START}\s*which\b", re.IGNORECASE),
]

# Weaker signals than `_INFORMATIONAL_PATTERNS`: a bare interrogative WORD.
# Strong enough to recognize a question in a language whose specific
# question shapes aren't enumerated above, but too weak to trust on its own
# inside a long declarative message -- a "Facts of the Case" field answer can
# legitimately contain "kitna"/"kaun" as ordinary narrative ("mujhe nahi pata
# kitna paisa gaya"), and treating that as a question would pause the very
# draft the user is filling in. `looks_informational` therefore only honours
# these when the message actually reads like a question: it ends with a
# question mark, or it's short enough that an interrogative word is the point
# of the sentence rather than an aside inside it.
_INTERROGATIVE_WORD_PATTERNS = [
    re.compile(r"\b(kyun|kyu|kyon|kab|kahan|kaha|kaun|kaunsi|kaunsa|kitna|kitni|kitne|konsa|konsi)\b", re.IGNORECASE),
    re.compile(r"क्यों|क्यूँ|कब|कहाँ|कहां|कौन|कितना|कितनी|कितने"),
    re.compile(r"काय|कसे|कशी|केव्हा|कुठे|कोण|फरक"),                 # Marathi
    re.compile(r"શું|કેવી\s*રીતે|ક્યારે|ક્યાં|કોણ|તફાવત"),               # Gujarati
    re.compile(r"কী|কিভাবে|কীভাবে|কখন|কোথায়|পার্থক্য"),             # Bengali
    re.compile(r"என்ன|எப்படி|எப்போது|எங்கே|யார்|வித்தியாசம்"),        # Tamil
    re.compile(r"ఏమిటి|ఎలా|ఎప్పుడు|ఎక్కడ|ఎవరు|తేడా"),                # Telugu
    re.compile(r"ಏನು|ಹೇಗೆ|ಯಾವಾಗ|ಎಲ್ಲಿ|ಯಾರು|ವ್ಯತ್ಯಾಸ"),              # Kannada
    re.compile(r"എന്ത|എങ്ങനെ|എപ്പോൾ|എവിടെ|ആരാണ്|വ്യത്യാസം"),     # Malayalam
    re.compile(r"کیا|کیسے|کب|کہاں|کون|فرق"),                        # Urdu
    # Kashmiri (Perso-Arabic): Q10 in the multilingual live suite ended in
    # an Arabic question mark and used "کیانہٕ ... کیا".  Neither the ASCII-only
    # question-mark check below nor the Urdu vocabulary above recognized it,
    # so incidental لکھ ("written" on a received police notice) looked
    # like an instruction to write a new document.
    re.compile(r"کیانہٕ|کیا|کُس|کُتھ|کتھہٕ|کتھ"),
]
# Above this many words, a stray interrogative word is far more likely to be
# narrative filler than the point of the message. A real question -- even a
# wordy one -- almost always still ends in a question mark, which bypasses
# this cap entirely.
_MAX_WORDS_FOR_BARE_INTERROGATIVE = 15


def looks_informational(text: str) -> bool:
    """True if `text` reads as a question/informational ask rather than a
    command. Exposed for reuse by `ChatService`'s draft-interruption check --
    the same "is this a genuine question" signal that decides whether a
    message may *start* a draft also decides whether one should *pause* an
    already-active draft instead of swallowing the message as field input.
    """
    if any(pattern.search(text) for pattern in _INFORMATIONAL_PATTERNS):
        return True
    compact = (text or "").strip()
    if not compact:
        return False
    if not compact.rstrip(" .।॥").endswith(("?", "؟")) and len(compact.split()) > _MAX_WORDS_FOR_BARE_INTERROGATIVE:
        return False
    return any(pattern.search(compact) for pattern in _INTERROGATIVE_WORD_PATTERNS)


# Minimum template score `detect()` accepts, given that a drafting verb has
# already been found. `detect_named_template()` keeps the stricter 0.3, since
# it runs without any verb evidence behind it.
_VERB_BACKED_NAME_THRESHOLD = 0.15


# Part 58 issue 15: how far ahead of the runner-up the top-scoring template
# has to be before it's treated as "the document they meant" rather than "one
# of several this could be". Two templates within this margin describe
# genuinely different documents that need different facts (a Bank Fraud
# Complaint asks for a bank name and UTR number; an Online Fraud Complaint
# doesn't), so silently picking whichever sorted first sends the user off
# collecting fields for a document they never asked for.
_AMBIGUOUS_SCORE_MARGIN = 0.05
# Candidates offered in the "which of these did you mean?" reply. A score
# below this is noise, not a plausible alternative worth naming.
_CANDIDATE_MIN_SCORE = 0.10
_MAX_CANDIDATES = 4


@dataclass(frozen=True)
class DraftIntentMatch:
    matched: bool
    draft_id: str | None = None
    confidence: float = 0.0
    ambiguous: bool = False
    # Draft ids worth offering when `ambiguous` is True and the message DID
    # point at a small set of plausible documents -- lets the caller show
    # those first instead of dumping the whole template catalogue.
    candidates: tuple[str, ...] = ()


class DraftIntentDetector:
    """Detects drafting intent and, where possible, which template it maps to.

    Purely data-driven: scores the message against every template's
    `trigger_phrases` (defined in that template's YAML file, not a separate
    Python rule table) — adding a template automatically teaches the detector
    its trigger phrases too.
    """

    def __init__(self) -> None:
        self.normalizer = QueryNormalizer()

    def detect(self, text: str) -> DraftIntentMatch:
        normalized = self.normalizer.normalize(text).lower()
        # Draft Mode requires an explicit drafting verb, full stop -- no
        # exceptions for a strong trigger-phrase match. Previously a template
        # match alone (score >= 0.3) was enough UNLESS the message also read
        # as a question with no verb -- which caught "what is a police
        # complaint" but not a declarative problem statement like "Police
        # refused to register my FIR" or "My bank account was hacked": no
        # verb, not phrased as a question, but "FIR"/"cyber crime" scored
        # high enough via trigger phrases to start drafting anyway. Verified
        # against 10 such statements: 4 wrongly launched a draft under the
        # old logic. A verb requirement with no exceptions is what "Draft
        # Mode ONLY when the user explicitly uses create/make/generate/
        # prepare/write/draft" actually means.
        if not _has_drafting_verb(normalized):
            return DraftIntentMatch(matched=False)

        # Part 58 issue 17/22: "cancel kr do draft ko" carries both a drafting
        # noun and a Hinglish imperative, so it clears the verb gate above and
        # would otherwise be scored as a request to start some new document.
        # It is the opposite request; the chat layer handles it.
        if _CANCEL_INTENT_PATTERN.search(normalized):
            return DraftIntentMatch(matched=False)

        # A lower bar than `detect_named_template`'s 0.3 on purpose: the
        # drafting-verb gate above has ALREADY established that the user wants
        # a document drafted, so the only question left is which one. At that
        # point a loose name match (`_partial_name_overlap`, which needs two
        # overlapping non-generic tokens covering at least half the stored
        # name) is better evidence than the alternative -- replying "I don't
        # have a template for that document yet" to someone who named a
        # template this app does have, just not in the exact words it stores.
        # Lock onto an explicitly named document in the command-bearing
        # clause before considering incidental document references in facts.
        command_scope = _draft_command_scope(normalized)
        ranked = self._ranked_templates(command_scope)
        if not ranked:
            # Facts outside the command clause may still contain an exact
            # document name, but a loose/fuzzy match there is unsafe.  Q33's
            # ordinary "bank transfer" facts fuzzy-matched both "bank" and
            # "band" in an Account Closure trigger and routed a repayment
            # message into account closure.  Require exact-strength evidence
            # before letting background narrative choose the template.
            full_ranked = self._ranked_templates(normalized)
            # Exact-strength narrative evidence may identify one template.
            # Weaker evidence is useful only when it produces a real
            # shortlist (the generic Gujarati "prepare a complaint" case),
            # never when one fuzzy accident is the sole candidate (Q33).
            plausible = [item for item in full_ranked if item[1] >= _CANDIDATE_MIN_SCORE]
            ranked = (
                full_ranked
                if full_ranked and (full_ranked[0][1] >= 0.3 or len(plausible) > 1)
                else []
            )

        # "Notice" is intentionally generic for fuzzy scoring, but once the
        # user explicitly commands "generate/draft notice" it names the
        # catalogue's general legal-notice template exactly.  Sending that
        # command into discovery is what broke the Q25-Q31 flagship flow.
        if not ranked and _contains_as_word(command_scope, "notice"):
            ranked = [("legal_notice", 0.3)]

        # A question containing an incidental writing verb is not a draft
        # request.  Keep genuine requests such as "can you draft a notice?"
        # because those have template evidence above; only the no-template
        # ambiguous fallback is suppressed.
        if not ranked and looks_informational(text):
            return DraftIntentMatch(matched=False)
        # Confirmed live (2026-09-23): a drafting verb with ZERO template
        # evidence and no question phrasing still isn't enough to assume a
        # LEGAL document is wanted -- "write a python code for adding two
        # numbers" clears every check above (has "write", scores 0 against
        # every template, isn't phrased as a question) exactly the same way
        # a genuine "please draft something for my case, I don't know what
        # it's called" message does. Without this gate the ambiguous
        # "describe your problem" fallback below launched Draft Mode for a
        # plain coding request, and every unrelated follow-up in that same
        # session (a pizza craving, a recipe request, "drinking water")
        # then kept getting swallowed by the drafting state machine instead
        # of answered normally, because `_is_draft_interruption` falls back
        # to this exact same `looks_informational` check to decide whether
        # to escape a draft already in progress. See `_names_non_legal_
        # deliverable`'s own docstring for why this is a narrow denylist,
        # not a positive requirement.
        if not ranked and _names_non_legal_deliverable(text):
            return DraftIntentMatch(matched=False)
        best_draft_id, best_score = (ranked[0] if ranked else (None, 0.0))
        if best_draft_id is not None and best_score >= _VERB_BACKED_NAME_THRESHOLD:
            runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
            if best_score - runner_up_score >= _AMBIGUOUS_SCORE_MARGIN:
                return DraftIntentMatch(matched=True, draft_id=best_draft_id, confidence=min(0.97, 0.6 + best_score))
            # A generic document name can appear inside a more specific one:
            # "mobile chori ki police complaint" genuinely names both the
            # generic Police Complaint and the specialized Mobile Theft
            # Complaint. Prefer the uniquely more distinctive exact trigger
            # instead of asking the user to choose between a parent category
            # and the precise document they already named.
            specific = self._most_specific_exact_match(
                normalized,
                [draft_id for draft_id, score in ranked if best_score - score < _AMBIGUOUS_SCORE_MARGIN],
            )
            if specific is not None:
                return DraftIntentMatch(matched=True, draft_id=specific, confidence=min(0.97, 0.6 + best_score))
            # Too close to call between real alternatives -- ask, and name the
            # ones actually in contention rather than the whole catalogue.
            return DraftIntentMatch(
                matched=True, draft_id=None, confidence=0.5, ambiguous=True,
                candidates=self._candidates(ranked),
            )

        return DraftIntentMatch(
            matched=True, draft_id=None, confidence=0.5, ambiguous=True, candidates=self._candidates(ranked)
        )

    @staticmethod
    def _most_specific_exact_match(normalized: str, candidate_ids: list[str]) -> str | None:
        specificity: list[tuple[str, int]] = []
        for template in list_templates():
            if template.draft_id not in candidate_ids:
                continue
            best = 0
            for phrase in (
                *template.trigger_phrases,
                template.name,
                template.hindi_name,
                *all_localized_names(template.draft_id),
            ):
                if not phrase or not _contains_as_word(normalized, phrase):
                    continue
                tokens = {
                    token for token in re.split(r"[\s,/-]+", phrase.lower())
                    if token and token not in _GENERIC_NAME_TOKENS
                }
                best = max(best, len(tokens))
            if best:
                specificity.append((template.draft_id, best))
        if not specificity:
            return None
        specificity.sort(key=lambda item: item[1], reverse=True)
        if len(specificity) > 1 and specificity[0][1] == specificity[1][1]:
            return None
        return specificity[0][0]

    @staticmethod
    def _candidates(ranked: list[tuple[str, float]]) -> tuple[str, ...]:
        return tuple(draft_id for draft_id, score in ranked[:_MAX_CANDIDATES] if score >= _CANDIDATE_MIN_SCORE)

    def detect_named_template(self, text: str) -> DraftIntentMatch:
        """Like `detect()`, but does NOT require a drafting verb (Part 32
        "Draft State Manager").

        Only meant to be called when the CALLER already knows the user is
        in a template-selection context -- mid-draft "selecting" stage
        (where a list of names was just shown and the reply is expected to
        pick one), or a short mid-"collecting" message being checked for a
        template switch. In that context a bare trigger phrase/template name
        ("Recovery Notice", "Security Deposit Recovery Notice") is
        unambiguously a selection, not a fresh drafting request that needs
        a verb to distinguish it from an ordinary question -- unlike
        `detect()`, called on arbitrary chat messages where that verb
        requirement is what stops "what is a police complaint" or "my bank
        account was hacked" from wrongly launching a draft.

        Returns `matched=False` (never `ambiguous=True`) when nothing
        scores -- unlike `detect()`, a lack of template match here isn't
        evidence of anything (ordinary chat text scores 0 here just as
        often as it does in `detect()`, but without the verb gate there's no
        "they clearly meant to draft something" signal backing an ambiguous
        reply).
        """
        normalized = self.normalizer.normalize(text).lower()
        best_draft_id, best_score = self._score_templates(normalized)
        if best_draft_id is not None and best_score >= 0.3:
            return DraftIntentMatch(matched=True, draft_id=best_draft_id, confidence=min(0.97, 0.6 + best_score))
        return DraftIntentMatch(matched=False)

    def _score_templates(self, normalized: str) -> tuple[str | None, float]:
        """The single best-scoring template. Kept as the public shape every
        existing caller uses; `_ranked_templates` exposes the full ordering
        `detect()` needs to tell a confident match from a near-tie."""
        ranked = self._ranked_templates(normalized)
        return ranked[0] if ranked else (None, 0.0)

    def _ranked_templates(self, normalized: str) -> list[tuple[str, float]]:
        scores: list[tuple[str, float]] = []
        message_tokens = {token for token in re.split(r"[\s,/-]+", normalized) if token}
        for template in list_templates():
            score = 0.0
            # `template.name` (the English canonical name every template
            # menu/listing actually shows, e.g. "Scholarship Application"),
            # `template.hindi_name` (e.g. "साइबर अपराध शिकायत"), and the
            # Tamil/Telugu/Kannada/Bengali names from `title_translations.py`
            # are scored alongside each YAML's own `trigger_phrases` rather
            # than duplicated into them -- a user typing the exact official
            # document name in any of these languages (as shown when
            # templates are listed) must match its own template without a
            # separate, hand-maintained trigger-phrase list per language.
            # `template.name` was missing here: picking the bare English name
            # straight off the menu (e.g. after "cancel draft"/"continue
            # draft" listed it) silently failed to match anything at all for
            # any template whose YAML didn't happen to also list its own
            # English name as a `trigger_phrase` -- confirmed live for
            # "Scholarship Application".
            for phrase in (
                *template.trigger_phrases, template.name, template.hindi_name, *all_localized_names(template.draft_id)
            ):
                if not phrase:
                    continue
                if _contains_as_word(normalized, phrase):
                    score += 0.3 + 0.03 * len(phrase.split())
                else:
                    score += _partial_name_overlap(message_tokens, phrase)
            if score > 0.0:
                scores.append((template.draft_id, score))
        # Ties keep `list_templates()` order, exactly as the previous
        # `score > best_score` loop did -- `sorted` is stable.
        return sorted(scores, key=lambda item: item[1], reverse=True)
