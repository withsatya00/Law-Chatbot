import re
from typing import ClassVar

from app.language.normalizer import QueryNormalizer
from app.rag.multilingual import legal_intent_hint
from app.rag.query_rewriter import IPC_TO_BNS_CROSSWALK
from app.schemas.analysis import IntentResponse

# A bare legal-citation lookup ("Section 302", "sec 302", "s.302", "What is
# Section 302?", "Quote Section 302.", "Show me Section 318.", "Tell me about
# Section 375.") names no legal TOPIC, only a provision number -- treating it
# as e.g. "Cheque Bounce" (whose keyword list happens to include the bare
# digits "138") or the generic "General Legal Query" fallback loses the one
# fact that IS certain about the query: it's a citation lookup, not a
# subject-matter question. Anchored to the WHOLE (normalized) message so a
# section number named as supporting detail inside a longer sentence ("cheque
# bounce case under section 138") is unaffected and still resolves through
# the keyword rules below, unchanged.
#
# Phase 7 "Intent Detection Repair": real queries just as often name the Act
# by its everyday abbreviation instead of (or alongside) the word "section"
# -- "Section 420 IPC", "IPC 420", "420 IPC", "BNS 318", "IPC section 420".
# A closed vocabulary (not a generic "any capitalized word" catch-all) so this
# stays a citation-lookup signal, not a topic-word magnet: every entry is an
# Act/Code/Sanhita abbreviation actually used in this corpus's source
# documents, nothing guessed. Longest-prefix alternatives ("BNSS" before
# "BNS") are ordered first defensively, though Python's backtracking would
# still find the right branch either way.
#
# Security finding C6: the abbreviation-only vocabulary above missed the
# ordinary case of a query spelling the Act out in full ("Section 302 of the
# Bharatiya Nyaya Sanhita") -- confirmed live: that exact question was
# answered from Bharatiya NAGARIK SURAKSHA Sanhita (BNSS) Section 302 (a
# different, real provision that happens to share the number), because
# `parse_section_lookup_act` found no abbreviation anywhere in the text and
# fell through to the Act-agnostic lookup, leaving the two Acts' same-
# numbered chunks tied at an identical floored score. The full names below
# are additional alternatives for the SAME abbreviations, not new Acts --
# `ACT_FULL_NAME_TO_ABBREVIATION` maps whichever form matched back to the
# short form `_ACT_ABBREVIATION_TO_METADATA_NAMES` (in `app/rag/
# retriever.py`) already keys on, so this never diverges into a second,
# differently-cased Act identity.
ACT_FULL_NAME_TO_ABBREVIATION: dict[str, str] = {
    "bharatiya nagarik suraksha sanhita": "BNSS",
    "bharatiya nyaya sanhita": "BNS",
    "bharatiya sakshya adhiniyam": "BSA",
    "indian penal code": "IPC",
    "code of criminal procedure": "CrPC",
    "criminal procedure code": "CrPC",
    "code of civil procedure": "CPC",
    "civil procedure code": "CPC",
    "negotiable instruments act": "NI Act",
    "information technology act": "IT Act",
}
# Full names ordered longest-first (as text) so a shorter name that happens
# to be a substring/prefix of a longer one is never matched short -- not
# actually possible for the specific names above, but kept as a general
# safety property rather than relying on it being accidentally true today.
_ACT_FULL_NAME_PATTERN = "|".join(
    r"\s+".join(re.escape(word) for word in name.split())
    for name in sorted(ACT_FULL_NAME_TO_ABBREVIATION, key=len, reverse=True)
)
_ACT_ABBREVIATION = rf"(?:BNSS|BNS|BSA|IPC|CrPC|CPC|NI\s+Act|IT\s+Act|{_ACT_FULL_NAME_PATTERN})"

_SECTION_LOOKUP_RE = re.compile(
    r"^\s*(?:what\s+is|what's|explain|quote|show\s+me|tell\s+me\s+about|define)?\s*"
    r"(?:section|sec\.?|s\.)\s*(\d{1,4}[A-Z]{0,2})"
    r"(?:\s+(?:of|under)\s+(?:the\s+)?[A-Za-z ]+)?"
    rf"(?:\s+{_ACT_ABBREVIATION})?"
    r"\s*[.?!]*\s*$",
    re.IGNORECASE,
)
# The Act-abbreviation-FIRST shapes ("IPC 420", "IPC section 420", "BNS 318",
# "BNSS 202", "BSA 63") don't fit `_SECTION_LOOKUP_RE` above -- that pattern
# always requires "section"/"sec"/"s." to precede the number. Kept as a
# separate, narrowly-anchored pattern (whole-message, closed vocabulary)
# rather than folding it into one giant alternation, so each shape stays
# independently auditable.
#
# Confirmed live: "BNS ki Section 100" (an entirely ordinary Hinglish
# possessive -- "BNS's Section 100") never matched this pattern, because the
# Hindi possessive particle ("ki"/"ka"/"ke") sits BETWEEN the Act
# abbreviation and "section", and this pattern only allowed whitespace
# there. `_EXPLANATORY_SECTION_RE`'s own reverse-order alternative already
# tolerates "100 ki BNS" (possessive before the Act, after the number) --
# this closes the equivalent gap for the Act-first order. `parse_section_
# lookup`'s explanatory-filler fallback could not rescue this either: it
# only scans for a filler word AFTER the matched "section N", and here the
# possessive sits before it, leaving nothing after "100" to match against.
_ACT_PREFIXED_SECTION_RE = re.compile(
    r"^\s*"
    rf"{_ACT_ABBREVIATION}\s*(?:ki|ka|ke|की|का|के)?\s*(?:section|sec\.?|धारा)?\s*"
    r"(\d{1,4}[A-Z]{0,2})\s*[.?!]*\s*$",
    re.IGNORECASE,
)
# The reverse bare shape: a number immediately followed by the Act
# abbreviation with no "section"/"sec" keyword anywhere ("420 IPC"). Also
# whole-message anchored so it can't swallow a genuine sentence that happens
# to contain a number and an abbreviation elsewhere in it.
_NUMBER_PREFIXED_ACT_RE = re.compile(
    rf"^\s*(\d{{1,4}}[A-Z]{{0,2}})\s*{_ACT_ABBREVIATION}\s*[.?!]*\s*$",
    re.IGNORECASE,
)

# Deliberately NOT extended to cover a leading free-text topic word before
# "under section N" ("Cheating under section 420", "FIR under section 154
# CrPC") -- the abbreviation vocabulary above is closed and auditable, but a
# generic leading `[A-Za-z ]+` capture to swallow "Cheating"/"FIR" would not
# be: it risks reclassifying genuine topical questions ("FIR" already has its
# own strong keyword rule below, and a citation mentioned in passing shouldn't
# hijack it) as bare citation lookups. Left as a documented gap, not a
# silently broadened regex.


# The "explanatory phrasing" fallback used by `parse_section_lookup` when none
# of the three whole-message citation shapes above matches. It finds the
# section NUMBER; `_EXPLANATORY_FILLER_RE` then decides whether what follows
# reads as "...and what does it mean?" rather than arbitrary prose that merely
# mentions a number.
#
# The first alternative ("section 318 kya hai") is the original one. The two
# added alternatives close a confirmed live gap: "BNS 318 kya hai?" named the
# Act instead of the word "section", so it matched NOTHING here -- not
# `_ACT_PREFIXED_SECTION_RE` either, which is whole-message anchored and can't
# tolerate the trailing "kya hai?" -- and the query was never classified
# SECTION_LOOKUP at all. That in turn skipped `_ensure_section_match_survives`
# and `_is_relevant_chunk`'s exact-section-number acceptance rule, so the
# relevance gate rejected every candidate and the user got "no verified
# document" for a section the knowledge base actually indexes (the same
# provision "IPC 420 ab BNS ki kaunsi dhara hai?" answered correctly). Kept
# to the same closed Act vocabulary as the strict patterns, plus the Hindi
# word for "section" ("धारा"), so this stays a citation-lookup signal.
_EXPLANATORY_SECTION_RE = re.compile(
    r"(?:section|sec\.?|s\.|धारा|कलम|பிரிவு|సెక్షన్|ವಿಭಾಗ|ধারা)\s*(\d{1,4}[A-Z]{0,2})"
    rf"|{_ACT_ABBREVIATION}\s*(?:section|sec\.?|धारा)?\s*(\d{{1,4}}[A-Z]{{0,2}})"
    rf"|(\d{{1,4}}[A-Z]{{0,2}})\s*(?:की\s*)?{_ACT_ABBREVIATION}",
    re.IGNORECASE,
)
# Words that turn a bare citation into "...and what does it mean?" -- English,
# Hinglish, and Devanagari forms of "what/is/meaning/explain/tell me".
_EXPLANATORY_FILLER_RE = re.compile(
    r"\b(?:kya|kaunsi|kaun|hai|hain|hota|hoti|ka|ki|matlab|meaning|samjhao|samjha|bataiye|bata|explain|about|says?)\b"
    r"|क्या|है|हैं|होता|होती|मतलब|समझा|बताइए|बताओ|अर्थ|काय|शुं|কি|என்ன|ఏమిటి|ಏನು",
    re.IGNORECASE,
)


def _first_group(match: re.Match[str]) -> str:
    """The first non-empty capture group -- `_EXPLANATORY_SECTION_RE` has one
    per alternative, so exactly one of them holds the section number."""
    return next((group for group in match.groups() if group), "")


def parse_section_lookup(text: str) -> str | None:
    """Returns the section number to look up if `text` is a bare
    citation-lookup shape in any supported form, else `None`.

    The corpus is indexed under BNS/BNSS/BSA (post-2024) numbering, not
    IPC's -- when the citation names IPC and that specific IPC section has a
    VERIFIED BNS mapping (`query_rewriter.IPC_TO_BNS_CROSSWALK`, each entry
    checked against the Act's own section headings), this returns the
    crosswalked BNS number instead of the literal IPC one, so classification
    and every downstream exact-match consumer (`ChatService.
    _ensure_section_match_survives`/`_is_relevant_chunk`) agree on the same
    number `query_rewriter.expand_queries` already searches for. An IPC
    section with no verified mapping yet returns its own literal number
    unchanged -- an explicit, disclosed gap (see the crosswalk's own
    "extend only with equally-verified entries" policy) rather than a
    guessed one.
    """
    stripped = text.strip()
    match = (
        _SECTION_LOOKUP_RE.match(stripped)
        or _ACT_PREFIXED_SECTION_RE.match(stripped)
        or _NUMBER_PREFIXED_ACT_RE.match(stripped)
    )
    if not match:
        # Hindi/Hinglish explanatory phrasing like "Section 30 kya hota hai"
        # or "Section 30 ka matlab kya hai" still asks for a section's
        # meaning, even though the canonical citation regex is intentionally
        # strict about full-message shapes. Keep the fallback narrow: it only
        # accepts a section-number token and a small set of explanatory
        # fillers, not arbitrary free text that merely happens to mention a
        # section number.
        fallback_match = _EXPLANATORY_SECTION_RE.search(stripped)
        if not fallback_match:
            return None
        trailing = stripped[fallback_match.end():]
        if not _EXPLANATORY_FILLER_RE.search(trailing):
            return None
        number = _first_group(fallback_match).upper()
        if re.search(r"\bipc\b", stripped, re.IGNORECASE):
            crosswalk = IPC_TO_BNS_CROSSWALK.get(number.lower())
            if crosswalk:
                return crosswalk[0]
        return number
    number = match.group(1).upper()
    if re.search(r"\bipc\b", stripped, re.IGNORECASE):
        crosswalk = IPC_TO_BNS_CROSSWALK.get(number.lower())
        if crosswalk:
            return crosswalk[0]
    return number


# Standalone (not embedded in the citation regexes above) so it can be
# `.search()`ed independently of *where* in the query the abbreviation sits --
# `parse_section_lookup`'s own regexes only need to know THAT one of these
# words appears, not capture which one.
_ACT_ABBREVIATION_RE = re.compile(_ACT_ABBREVIATION, re.IGNORECASE)


def parse_section_lookup_act(text: str) -> str | None:
    """Returns the Act abbreviation (e.g. "BNS", "BNSS", "BSA") explicitly
    named in `text`'s citation-lookup shape, or `None` if no Act is named --
    the bare-number case ("Section 318") this deliberately leaves
    unresolved, same as `parse_section_lookup` leaves the Act unnamed rather
    than guessing one.

    Root cause this exists to fix: `text` matching a citation-lookup shape
    only proves a section NUMBER was parsed -- retrieval had no equivalent
    signal for which Act, so "BNS 318" and "BNSS 318" (two different, real
    provisions that happen to share the number 318) retrieved BOTH Acts'
    chunks at an IDENTICAL floored score (confirmed live: 0.6175 == 0.6175
    post-rerank for "BNS 318"), leaving Act disambiguation to whichever
    chunk the LLM happened to prefer while generating the answer -- not a
    retrieval guarantee.

    IPC is crosswalk-aware, mirroring `parse_section_lookup` one function
    above: the corpus has no IPC-named chunks at all (repealed, replaced by
    BNS/BNSS/BSA), so "IPC 420" resolves to "BNS" -- the Act the crosswalked
    BNS number 318 actually belongs to -- rather than the literal (and
    unindexed) "IPC". An IPC section with no verified crosswalk entry
    returns `None` here, same disclosed-gap policy as the number crosswalk.
    """
    stripped = text.strip()
    if parse_section_lookup(stripped) is None:
        return None
    act_match = _ACT_ABBREVIATION_RE.search(stripped)
    if not act_match:
        return None
    abbreviation = re.sub(r"\s+", " ", act_match.group(0).upper().strip())
    # A full Act name normalizes to the SAME short form the abbreviation
    # shape already produces (see `ACT_FULL_NAME_TO_ABBREVIATION` above) --
    # every check and lookup below this point is written in terms of the
    # short form and must not care which shape the caller actually typed.
    canonical = ACT_FULL_NAME_TO_ABBREVIATION.get(re.sub(r"\s+", " ", act_match.group(0).lower().strip()))
    if canonical is not None:
        abbreviation = canonical
    if abbreviation == "IPC":
        section_match = (
            _SECTION_LOOKUP_RE.match(stripped)
            or _ACT_PREFIXED_SECTION_RE.match(stripped)
            or _NUMBER_PREFIXED_ACT_RE.match(stripped)
        )
        if section_match is not None:
            number = section_match.group(1).lower()
        else:
            # The explanatory-phrasing shape ("IPC 420 ka matlab kya hai")
            # reaches here too now that `parse_section_lookup` accepts it --
            # without this branch the Act would silently resolve to `None`
            # for exactly the queries the fallback was added to support.
            explanatory_match = _EXPLANATORY_SECTION_RE.search(stripped)
            number = _first_group(explanatory_match).lower() if explanatory_match else ""
        return "BNS" if number in IPC_TO_BNS_CROSSWALK else None
    return abbreviation


class IntentDetector:
    rules: ClassVar[list[tuple[str, str, str, list[str]]]] = [
        ("Salary Issue", "Labour Law", "Mentions unpaid wages or employment dues.", ["salary", "wage", "payment", "appointment letter"]),
        ("Cyber Crime", "Cyber Law", "Mentions online fraud or digital crime.", ["cyber", "online fraud", "upi", "otp", "scam"]),
        ("Cheque Bounce", "Banking and Criminal Law", "Mentions cheque dishonour.", ["cheque", "138", "dishonour", "bounce"]),
        ("Divorce", "Family Law", "Mentions divorce or separation.", ["divorce", "talak", "mutual", "custody", "maintenance"]),
        ("Rental Dispute", "Property Law", "Mentions landlord, tenant, rent, or deposit.", ["landlord", "tenant", "deposit", "rent", "kiraya"]),
        ("Consumer Complaint", "Consumer Law", "Mentions defective goods or service grievance.", ["consumer", "refund", "defective", "warranty"]),
        ("FIR", "Criminal Law", "Mentions FIR or police complaint.", ["fir", "police complaint", "police station"]),
        ("Property Registration", "Property Law", "Mentions deed or registration.", ["sale deed", "gift deed", "registration", "property"]),
        ("GST", "Tax Law", "Mentions GST notice or compliance.", ["gst", "input tax", "itc"]),
        ("Income Tax", "Tax Law", "Mentions income tax notice.", ["income tax", "itr", "assessment"]),
        ("POSH Complaint", "Employment Law", "Mentions workplace harassment.", ["posh", "sexual harassment", "workplace harassment"]),
        ("RTI", "Public Law", "Mentions right to information.", ["rti", "right to information"]),
        ("Bail", "Criminal Law", "Mentions bail or anticipatory bail.", ["bail", "anticipatory bail", "custody"]),
        ("Recovery Agent Harassment", "Banking Law", "Mentions loan recovery agent conduct.", ["recovery agent", "loan recovery", "harassment by bank", "rbi guidelines"]),
        ("Legal Notice", "Civil Law", "Mentions sending or receiving a legal notice.", ["legal notice", "notice period", "send a notice"]),
        # Phase 1 item 5: major legal intents that had no rule at all and so
        # fell into the 0.45-confidence "General Legal Query" catch-all, which
        # gives retrieval no legal-category hint and the reranker no category
        # bonus. Threat/harassment and domestic violence are the two most
        # safety-critical of them -- exactly the questions where landing in a
        # generic bucket costs the most.
        ("Threat and Harassment", "Criminal Law", "Mentions threats, intimidation, stalking, or harassment.",
         ["threat", "threaten", "intimidation", "harass", "stalking", "blackmail", "dhamki", "pareshan"]),
        ("Domestic Violence", "Family Law", "Mentions domestic violence, cruelty, or dowry harassment.",
         ["domestic violence", "dowry", "cruelty", "in-laws", "gharelu hinsa", "dahej"]),
        ("Defamation", "Civil Law", "Mentions defamation or reputational harm.",
         ["defamation", "defamatory", "slander", "libel", "manhani"]),
        ("Employment Dispute", "Labour Law", "Mentions termination, resignation, or workplace dues.",
         ["terminated", "termination", "wrongful dismissal", "resignation", "notice pay", "provident fund"]),
        ("Motor Accident", "Motor Vehicles Law", "Mentions a road accident, challan, or vehicle claim.",
         ["accident", "challan", "motor vehicle", "insurance claim", "driving licence", "driving license"]),
        ("Will and Succession", "Family Law", "Mentions a will, inheritance, or succession dispute.",
         ["will", "inheritance", "succession", "legal heir", "ancestral property"]),
        ("Theft", "Criminal Law", "Mentions theft, robbery, or stolen property.",
         ["theft", "stolen", "robbery", "burglary", "snatching", "chori"]),
        # Deliberately excludes "scam"/"online": those belong to Cyber Crime
        # above, which should keep winning for an online-fraud question.
        ("Cheating and Fraud", "Criminal Law", "Mentions cheating, fraud, or being duped.",
         ["cheat", "cheating", "cheated", "fraud", "duped", "swindle", "dhokha", "thagi"]),
    ]

    def __init__(self) -> None:
        self.normalizer = QueryNormalizer()

    async def detect(self, text: str, language: str | None = None) -> IntentResponse:
        normalized = self.normalizer.normalize(text).lower()
        section_number = parse_section_lookup(normalized)
        if section_number:
            return IntentResponse(
                intent="SECTION_LOOKUP",
                legal_category="General Law",
                reason=f"Bare legal-citation lookup for Section {section_number}.",
                confidence=0.9,
            )
        best: tuple[str, str, str, float] = ("General Legal Query", "General Law", "No strong rule matched.", 0.45)
        for intent, category, reason, keywords in self.rules:
            matches = sum(1 for keyword in keywords if keyword in normalized)
            if matches:
                confidence = min(0.95, 0.62 + matches * 0.11)
                if confidence > best[3]:
                    best = (intent, category, reason, confidence)
        if best[0] == "General Legal Query":
            # Phase 1 item 5 ("preserve language and context"): the rules above
            # are written in English keywords, so a Hindi/Marathi/Gujarati/
            # Tamil/Bengali/Urdu question matched none of them and fell into
            # this 0.45 catch-all -- which then gave retrieval no
            # legal-category hint and the reranker no category bonus, on
            # exactly the questions that need them most. The multilingual
            # concept bridge already knows what such a question is about (it
            # is the same table retrieval uses, so the two agree by
            # construction). Consulted only after the English rules find
            # nothing, so an English question resolves exactly as it always
            # did.
            hint = legal_intent_hint(normalized)
            if hint is not None:
                intent, category = hint
                return IntentResponse(
                    intent=intent,
                    legal_category=category,
                    reason="Matched a legal concept in the question's own language.",
                    confidence=0.8,
                )
        return IntentResponse(intent=best[0], legal_category=best[1], reason=best[2], confidence=best[3])
