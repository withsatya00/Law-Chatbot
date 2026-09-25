"""Jurisdiction Routing (Phase 2): resolves WHICH State(s)/locality and WHICH
date govern the user's actual matter, before retrieval ever runs.

Three sources, in strict priority order (objective item 2):

    1. an EXPLICIT location named in the current message ("in Uttar
       Pradesh", "UP property dispute") -- wins even over an established
       matter, and is NEVER written back to the profile;
    2. a CONFIRMED state already established earlier in this same
       conversation (carried forward turn to turn, but reset the moment the
       conversation moves to a different matter -- see `_is_new_matter`);
    3. the user's saved PROFILE state (`app.services.phase3.PreferenceService`),
       used only as a last-resort default, and always disclosed as an
       assumption rather than presented as confirmed (`context_source`).

Deliberately does NOT treat a bare state-name MENTION as the matter's
location -- "Bihar has a law about X" and "my dispute is in Bihar" look
identical to a naive keyword search, and only the second is a location
statement. `_EXPLICIT_LOCATION_RE` requires a locational preposition/particle
immediately next to the name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from app.rag.jurisdiction import CITY_TO_STATE
from app.rag.kb_jurisdiction import STATE_UT_CODES, normalize_state_code

# Legal categories where the governing rule routinely differs by State (rent
# control, tenancy, stamp duty/registration, local land revenue law) --
# small and deliberately test-driven; extend as more State legislation
# enters the Knowledge Base. `app.intent.detector`'s own category vocabulary
# is reused verbatim rather than inventing a parallel one.
STATE_SENSITIVE_LEGAL_CATEGORIES: frozenset[str] = frozenset({"Property Law"})

# Being in a state-sensitive CATEGORY is necessary but not sufficient to ask a
# clarifying question -- "Property Law" also covers inheritance disputes and
# generic deposit/registration questions this corpus can already answer at a
# general/central level (Transfer of Property Act, general principles), and
# asking "which State?" before every one of those would violate the
# objective's OWN "Har general legal question par unnecessary clarification
# mat lagao" -- confirmed against this codebase's calibrated multi-turn
# regression suite, which treats exactly those phrasings as answerable
# without a State prompt. A clarifying question is reserved for wording that
# ITSELF signals the answer hinges on state-specific rules (rent control,
# eviction procedure, stamp duty/registration rates) -- real Central-vs-State
# fault lines, not the mere presence of the word "property".
_STATE_VARYING_TERM_RE = re.compile(
    r"\b(?:eviction|rent\s*control|stamp\s+duty|registration\s+(?:fee|charges?)|"
    r"notice\s+period|tenancy\s+act|rent\s+agreement\s+registration)\b",
    re.IGNORECASE,
)

CONTEXT_SOURCE_EXPLICIT = "explicit_message"
CONTEXT_SOURCE_CONFIRMED = "confirmed_conversation"
CONTEXT_SOURCE_PROFILE = "profile_default"

# Matches "<State Name>" immediately preceded by an English/Hinglish
# locational cue ("in", "at", "under", a possessive) or followed by a
# Hindi/Hinglish postposition ("mein", "ka", "ki", "ke") -- the shape a real
# location statement takes, as opposed to a bare mention. Longest names
# first so "Andaman and Nicobar Islands" is not shadowed by matching on a
# shorter prefix first.
_STATE_NAMES_BY_LENGTH = sorted(STATE_UT_CODES.values(), key=len, reverse=True)
_STATE_ALTERNATION = "|".join(re.escape(name) for name in _STATE_NAMES_BY_LENGTH)
_EXPLICIT_LOCATION_RE = re.compile(
    rf"\b(?:in|at|under|within|for)\s+({_STATE_ALTERNATION})\b"
    rf"|\b({_STATE_ALTERNATION})\s+(?:mein|me|ka|ki|ke|se|wale|wala)\b",
    re.IGNORECASE,
)
# Used only when a clarification was JUST asked (`awaiting_clarification`):
# the user's one-line reply to "Which State does this concern?" ("Uttar
# Pradesh", "UP") is itself the answer and needs no locational preposition.
_BARE_STATE_RE = re.compile(rf"\b({_STATE_ALTERNATION})\b", re.IGNORECASE)

# A well-known city is also a real location statement ("Mumbai mein hoon",
# "in Pune") -- without this, a user who names their city instead of their
# State (the overwhelmingly common way people actually talk) never satisfies
# `needs_clarification`, and the "which State?" question repeats forever
# because `detect_state` never finds a match to carry into `state_codes`.
# Reuses `app.rag.jurisdiction.CITY_TO_STATE` (already maintained for the
# Model Tenancy Act caveat) rather than a second city list.
_CITY_NAMES_BY_LENGTH = sorted(CITY_TO_STATE, key=len, reverse=True)
_CITY_ALTERNATION = "|".join(re.escape(name) for name in _CITY_NAMES_BY_LENGTH)
_EXPLICIT_CITY_LOCATION_RE = re.compile(
    rf"\b(?:in|at|under|within|for)\s+({_CITY_ALTERNATION})\b"
    rf"|\b({_CITY_ALTERNATION})\s+(?:mein|me|ka|ki|ke|se|wale|wala)\b",
    re.IGNORECASE,
)
# Bare city reply to "which State?" ("Mumbai."), same allowance as
# `_BARE_STATE_RE` for a bare State name.
_BARE_CITY_RE = re.compile(rf"\b({_CITY_ALTERNATION})\b", re.IGNORECASE)

# A State/city name sitting next to a 6-digit Indian PIN code is a real
# postal address, not a bare topic mention -- as reliable a location
# statement as "in Mumbai"/"Mumbai mein", just without either's grammatical
# shape. Confirmed live (qa-40q-multilingual-20260921 BUG-07): "Mera address:
# 21, Andheri West, Mumbai – 400053" named the city adjacent to its PIN code,
# but satisfied neither `_EXPLICIT_CITY_LOCATION_RE` (no preposition/
# postposition next to "Mumbai") nor `allow_bare_mention` (not asked while a
# clarification was pending) -- so `detect_state` returned `None`, no
# Maharashtra filter reached retrieval, and an untagged Hyderabad-specific Act
# chunk was exactly as retrievable as a Maharashtra one for the rest of that
# same tenancy matter. Checked unconditionally (not gated by
# `allow_bare_mention`) because the PIN code makes this address-shaped, not a
# bare mention like "Bihar has a law about X".
_PIN_ADJACENT = r"[\s,\-–—]{0,3}\d{6}\b"
_ADDRESS_STATE_LOCATION_RE = re.compile(
    rf"\b({_STATE_ALTERNATION})\b{_PIN_ADJACENT}|\b\d{{6}}\b[\s,\-–—]{{0,3}}({_STATE_ALTERNATION})\b",
    re.IGNORECASE,
)
_ADDRESS_CITY_LOCATION_RE = re.compile(
    rf"\b({_CITY_ALTERNATION})\b{_PIN_ADJACENT}|\b\d{{6}}\b[\s,\-–—]{{0,3}}({_CITY_ALTERNATION})\b",
    re.IGNORECASE,
)

# `EntityExtractor.patterns["date"]` (reused verbatim -- one date grammar for
# the whole app) matches a raw substring; this module only needs to know
# whether the message names a date at all and, if parseable, whether it is
# in the past (a "historical matter" signal) -- see `extract_as_of_date`.
_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
_DATE_MONTH_NAME_RE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)
_MONTHS = {
    name.lower(): index
    for index, name in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}


def detect_state(text: str, *, allow_bare_mention: bool = False) -> str | None:
    """Returns the canonical State/UT code named as a LOCATION in `text`, or
    `None`. `allow_bare_mention=True` (only when a clarification question was
    just asked) also accepts a bare name with no locational cue. A State name
    is checked before a city so "in Delhi" (a State/UT itself) never gets
    shadowed by a city lookup, and a named State always wins over a same-turn
    city mention."""
    match = _EXPLICIT_LOCATION_RE.search(text)
    if match:
        return normalize_state_code(match.group(1) or match.group(2))
    city_match = _EXPLICIT_CITY_LOCATION_RE.search(text)
    if city_match:
        city = (city_match.group(1) or city_match.group(2)).lower()
        return normalize_state_code(CITY_TO_STATE.get(city))
    address_state_match = _ADDRESS_STATE_LOCATION_RE.search(text)
    if address_state_match:
        return normalize_state_code(address_state_match.group(1) or address_state_match.group(2))
    address_city_match = _ADDRESS_CITY_LOCATION_RE.search(text)
    if address_city_match:
        city = (address_city_match.group(1) or address_city_match.group(2)).lower()
        return normalize_state_code(CITY_TO_STATE.get(city))
    if allow_bare_mention:
        bare = _BARE_STATE_RE.search(text)
        if bare:
            return normalize_state_code(bare.group(1))
        bare_city = _BARE_CITY_RE.search(text)
        if bare_city:
            return normalize_state_code(CITY_TO_STATE.get(bare_city.group(1).lower()))
    return None


def extract_as_of_date(text: str, *, today: date | None = None) -> str | None:
    """A date named in `text`, as ISO, ONLY if it parses AND is not in the
    future relative to `today` -- a future date is not "the incident date",
    it is noise (or a typo), and treating it as historical would silently
    misroute retrieval to a version that does not exist yet."""
    today = today or datetime.now(UTC).date()
    match = _DATE_MONTH_NAME_RE.search(text)
    if match:
        day, month_name, year = match.groups()
        month = _MONTHS.get(month_name.lower())
        try:
            parsed = date(int(year), month, int(day)) if month else None
        except ValueError:
            parsed = None
    else:
        match = _DATE_RE.search(text)
        if not match:
            return None
        day_token, month_token, year_token = match.groups()
        year_int = int(year_token)
        if year_int < 100:  # "12/03/21" -- 2-digit year, assume 2000s
            year_int += 2000
        try:
            parsed = date(year_int, int(month_token), int(day_token))
        except ValueError:
            try:
                parsed = date(year_int, int(day_token), int(month_token))  # DD/MM vs MM/DD ambiguity: try the other order
            except ValueError:
                parsed = None
    if parsed is None or parsed > today:
        return None
    return parsed.isoformat()


@dataclass
class MatterContext:
    state_codes: list[str] = field(default_factory=list)
    locality: str | None = None
    as_of_date: str | None = None
    context_source: str | None = None  # one of the CONTEXT_SOURCE_* constants, or None if unresolved
    needs_clarification: bool = False
    clarification_reason: str | None = None
    # Snapshot of the legal_category this context was resolved under, so the
    # NEXT turn can tell "same matter continuing" from "a new, unrelated
    # question" -- see `_is_new_matter`.
    legal_category: str | None = None

    def to_memory(self) -> dict[str, Any]:
        """The shape persisted into `ConversationMemoryStore` -- never
        includes `needs_clarification`/`clarification_reason` (per-turn
        facts about THIS message, not durable matter state)."""
        return {
            "state_codes": self.state_codes,
            "locality": self.locality,
            "as_of_date": self.as_of_date,
            "context_source": self.context_source,
            "legal_category": self.legal_category,
        }

    def cache_key(self) -> str | None:
        """The response-cache jurisdiction differentiator (objective item 6)
        -- `None` when no State/date is resolved, so a state-insensitive
        question keeps using the SAME cache bucket it always has (no
        pointless cache fragmentation for questions jurisdiction can't
        affect)."""
        if not self.state_codes and not self.as_of_date:
            return None
        return f"{'-'.join(sorted(self.state_codes))}:{self.locality or ''}:{self.as_of_date or ''}"

    def as_filter(self) -> dict[str, Any]:
        """The shape `LegalRetriever.retrieve`/`kb_jurisdiction` filtering
        functions consume. Empty state_codes means "no State constraint" --
        callers must not confuse that with `needs_clarification` (a
        state-sensitive question that was left unresolved should never
        reach retrieval at all; see `resolve_matter_context`)."""
        return {"state_codes": self.state_codes, "locality": self.locality, "as_of_date": self.as_of_date}


def _is_new_matter(prior: dict[str, Any] | None, legal_category: str | None) -> bool:
    """A DIFFERENT legal_category than the one the prior matter context was
    resolved under is treated as a new matter -- coarse, but deterministic
    and testable, and errs toward asking again rather than silently carrying
    a stale State into an unrelated question. `None` current/prior category
    (general chit-chat, or no prior context at all) is never "different".
    """
    if not prior:
        return False
    prior_category = prior.get("legal_category")
    if prior_category is None or legal_category is None:
        return False
    return str(prior_category) != legal_category


def resolve_matter_context(
    question: str,
    memory: dict[str, Any] | None,
    profile_state_code: str | None,
    legal_category: str | None,
    *,
    awaiting_clarification: bool = False,
    awaiting_locality_clarification: bool = False,
) -> MatterContext:
    """Resolves this turn's `MatterContext` from the priority order this
    module's docstring describes. Pure function -- persisting the result and
    deciding whether to short-circuit with a clarifying question are the
    caller's job (`ChatService`), so this stays unit-testable without Mongo,
    Redis, or an LLM.

    `awaiting_locality_clarification=True` means the PREVIOUS turn asked
    "which city/district/local body?" (`clarification_question(...,
    ambiguity="locality")`) -- this turn's message IS that answer. Locality
    has no canonical vocabulary to detect against (unlike State/UT names),
    so the reply is taken close to verbatim: any State name it also happens
    to name is parsed out and kept separately, and whatever text remains
    (trimmed of connecting words/punctuation) becomes `locality` -- captured
    into `MatterContext.locality` here so the CALLER (`ChatService`) can
    persist it into `matter_context` memory and it reaches
    `jurisdiction_or_branches`/`filter_by_matter_context` on every
    subsequent retrieval for this matter, not just acknowledged and dropped.
    """
    memory = memory or {}
    prior = memory.get("matter_context")
    new_matter = _is_new_matter(prior, legal_category)
    effective_prior = None if new_matter else prior

    explicit_state = detect_state(question, allow_bare_mention=awaiting_clarification or awaiting_locality_clarification)
    explicit_date = extract_as_of_date(question)
    explicit_locality: str | None = None
    if awaiting_locality_clarification:
        remainder = _EXPLICIT_LOCATION_RE.sub("", question) if explicit_state else question
        remainder = remainder.strip(" ,.-–—")
        explicit_locality = remainder or None

    if explicit_state:
        state_codes = [explicit_state]
        context_source = CONTEXT_SOURCE_EXPLICIT
    elif explicit_locality:
        # The user answered the LOCALITY question without also naming a
        # State -- a real, resolved answer on its own (objective: don't
        # force a State the user never gave), not something to keep asking
        # about. Falls back to any State already on record so a locality
        # reply doesn't regress a State the conversation already had.
        state_codes = list((effective_prior or {}).get("state_codes") or [])
        if not state_codes and profile_state_code:
            state_codes = [profile_state_code]
        context_source = CONTEXT_SOURCE_EXPLICIT
    elif effective_prior and effective_prior.get("state_codes"):
        state_codes = list(effective_prior["state_codes"])
        context_source = CONTEXT_SOURCE_CONFIRMED
    elif profile_state_code:
        state_codes = [profile_state_code]
        context_source = CONTEXT_SOURCE_PROFILE
    elif effective_prior and effective_prior.get("locality"):
        # No State was ever resolved, but a locality WAS -- e.g. the user
        # answered "which city/district?" without naming a State, and this
        # turn is a follow-up on that same matter. Still a genuinely
        # confirmed context, not "nothing resolved" -- reported as such so a
        # follow-up doesn't wrongly re-trigger the clarification question.
        state_codes = []
        context_source = CONTEXT_SOURCE_CONFIRMED
    else:
        state_codes = []
        context_source = None

    if explicit_date:
        as_of_date = explicit_date
    elif effective_prior:
        as_of_date = effective_prior.get("as_of_date")
    else:
        as_of_date = None

    if explicit_locality:
        locality = explicit_locality
    elif explicit_state:
        locality = None  # a fresh explicit State without a locality resets it, same as before this parameter existed
    else:
        locality = (effective_prior or {}).get("locality")

    needs_clarification = (
        not state_codes
        and not locality
        and legal_category in STATE_SENSITIVE_LEGAL_CATEGORIES
        and bool(_STATE_VARYING_TERM_RE.search(question))
    )
    clarification_reason = (
        f"{legal_category} questions can turn on which State's law applies, and no State is on record for this matter."
        if needs_clarification
        else None
    )

    return MatterContext(
        state_codes=state_codes,
        locality=locality,
        as_of_date=as_of_date,
        context_source=context_source,
        needs_clarification=needs_clarification,
        clarification_reason=clarification_reason,
        legal_category=legal_category,
    )


_CLARIFICATION_QUESTIONS = {
    "english": "Which State (or Union Territory) does this matter concern? Laws on this topic vary by State.",
    "hindi": "यह मामला किस राज्य (या केंद्र शासित प्रदेश) से संबंधित है? इस विषय पर कानून राज्य के अनुसार अलग-अलग होते हैं।",
    "hinglish": "Yeh matter kis State (ya Union Territory) se related hai? Is topic par law State ke hisaab se alag hota hai.",
}

# Used when candidate sources are ambiguous at the LOCALITY level (multiple
# local bodies' provisions found, no locality resolved) rather than at the
# State level -- see `detect_jurisdiction_ambiguity`. Locality detection from
# free text is not implemented in this phase, so a locality-scoped answer can
# ONLY come from the user stating it explicitly; this question is how that's
# asked for rather than silently picked or omitted.
_LOCALITY_CLARIFICATION_QUESTIONS = {
    "english": "Which city, district, or local body does this matter concern? Rules here vary by locality.",
    "hindi": "यह मामला किस शहर, जिले या स्थानीय निकाय से संबंधित है? यहां नियम स्थान के अनुसार अलग-अलग होते हैं।",
    "hinglish": "Yeh matter kis city, district ya local body se related hai? Yahan rules locality ke hisaab se alag hote hain.",
}


def clarification_question(language: str, *, ambiguity: str = "state") -> str:
    """`ambiguity="locality"` asks for the city/district/local body instead
    of the State -- used when `detect_jurisdiction_ambiguity` finds the
    retrieved candidates disagree at the locality level specifically."""
    table = _LOCALITY_CLARIFICATION_QUESTIONS if ambiguity == "locality" else _CLARIFICATION_QUESTIONS
    return table.get(language, table["english"])


_ASSUMED_STATE_NOTE = {
    "english": "Assuming your matter is in {state} (from your saved profile) -- let me know if it's a different State.",
    "hindi": "मान रहा हूँ कि आपका मामला {state} से है (आपकी सेव की गई प्रोफ़ाइल के अनुसार) -- अगर यह किसी और राज्य का है तो बताएं।",
    "hinglish": "Maan raha hoon ki aapka matter {state} ka hai (aapki saved profile ke hisaab se) -- agar kisi aur State ka hai to bataiye.",
}

_AS_OF_DATE_NOTE = {
    "english": (
        "Answered based on the law as it stood on {date} -- the incident date you mentioned. "
        "Note: whether an amendment's transition/savings clause changes this for your specific facts has not been "
        "independently verified -- confirm the exact provision in force if the date is close to when the law changed."
    ),
    "hindi": (
        "यह उत्तर {date} (आपके बताए गए घटना की तारीख) पर लागू कानून के आधार पर दिया गया है। "
        "ध्यान दें: किसी संशोधन का transition/savings प्रावधान आपके मामले पर लागू होता है या नहीं, इसकी स्वतंत्र रूप से पुष्टि नहीं की गई है -- "
        "यदि यह तारीख कानून बदलने के समय के करीब है, तो लागू प्रावधान की पुष्टि कर लें।"
    ),
    "hinglish": (
        "Yeh jawab {date} (aapke bataye gaye incident date) par lagu kanoon ke hisaab se diya gaya hai. "
        "Note: kisi amendment ka transition/savings clause aapke case par lagu hota hai ya nahi, yeh independently verify "
        "nahi kiya gaya hai -- agar yeh date law badalne ke time ke kareeb hai, to exact provision confirm kar lijiye."
    ),
}


def assumed_state_note(language: str, state_code: str) -> str:
    state_name = STATE_UT_CODES.get(state_code, state_code)
    template = _ASSUMED_STATE_NOTE.get(language, _ASSUMED_STATE_NOTE["english"])
    return template.format(state=state_name)


def as_of_date_note(language: str, as_of_date: str) -> str:
    template = _AS_OF_DATE_NOTE.get(language, _AS_OF_DATE_NOTE["english"])
    return template.format(date=as_of_date)


_VERSION_AMBIGUITY_NOTE = {
    "english": (
        "More than one version of this provision could apply around {date}, and which one governs your exact "
        "facts was not conclusively determined here (e.g. transition/savings clauses around an amendment can "
        "change this). Please verify the exact provision in force with a legal professional for this date."
    ),
    "hindi": (
        "{date} के आसपास इस प्रावधान के एक से अधिक संस्करण लागू हो सकते हैं, और आपके सटीक मामले पर कौन सा लागू होता है यह यहां निश्चित रूप से "
        "तय नहीं किया जा सका (जैसे किसी संशोधन के transition/savings प्रावधान इसे बदल सकते हैं)। कृपया इस तारीख के लिए लागू प्रावधान की पुष्टि किसी "
        "कानूनी विशेषज्ञ से करें।"
    ),
    "hinglish": (
        "{date} ke aas-paas is provision ke ek se zyada version lagu ho sakte hain, aur aapke exact case par kaunsa "
        "lagu hota hai yeh yahan conclusively decide nahi kiya ja saka (jaise kisi amendment ka transition/savings "
        "clause isse badal sakta hai). Is date ke liye lagu provision kisi legal professional se confirm kar lijiye."
    ),
}


def version_ambiguity_note(language: str, as_of_date: str) -> str:
    template = _VERSION_AMBIGUITY_NOTE.get(language, _VERSION_AMBIGUITY_NOTE["english"])
    return template.format(date=as_of_date)


def disclosure_note(language: str, matter: MatterContext, *, version_ambiguous: bool = False) -> str | None:
    """The combined "assumed State" + "as of this date" (+ unresolved-
    version-ambiguity limitation, when `version_ambiguous`) disclosure
    appended to a grounded answer (objective item 7: "Jurisdiction/
    assumption, relevant effective date where available ... show karo") --
    built ONCE so the streaming and non-streaming answer paths show
    identical wording rather than two copies that could drift apart.
    """
    lines = []
    if matter.context_source == CONTEXT_SOURCE_PROFILE and matter.state_codes:
        lines.append(assumed_state_note(language, matter.state_codes[0]))
    if matter.as_of_date:
        lines.append(as_of_date_note(language, matter.as_of_date))
        if version_ambiguous:
            lines.append(version_ambiguity_note(language, matter.as_of_date))
    return "\n\n".join(lines) if lines else None
