"""Part 35 "Entity Memory": lightweight, pattern-based extraction and recall
of concrete facts stated during a conversation ("my bike was stolen" ->
entity=bike, fact=stolen), checked BEFORE retrieval so a fact-recall
question ("What was stolen?", "mera kya chori hua tha") never has to fall
back to RAG or a general-knowledge LLM call that has no way to know the
answer -- and never has to be dead-ended with "I don't have access to your
personal history" when the fact was stated earlier in this same session.

Deliberately NOT a general NER/NLU system: a fixed set of entity-type
keyword lists (Person/Vehicle/Property/Place/Document/Incident/
Relationship/Time -- the categories the spec named) and fact-shape
detection built from concrete examples ("my X was stolen", "my X's Y was
injured", "my X won't return Z"). It generalizes across phrasing (Hindi,
English, Hinglish; different word order) via keyword co-occurrence rather
than rigid sentence templates, but it will still miss phrasings outside
its keyword lists -- extending coverage means adding entries to
`_ENTITY_KEYWORDS`/`_FACT_CATEGORIES` below, not rewriting the approach.
"""

import re
from dataclasses import dataclass
from typing import Any

# Part 35's own named categories (Person/Vehicle/Property/Place/Document/
# Incident/Relationship/Time). "Person" and "Relationship" are merged in
# practice -- nearly every person mentioned in a legal conversation is named
# by their relationship to the user ("my brother," "my landlord"), not a
# proper name, so a single keyword list serves both.
# Order matters: `extract_fact` returns the FIRST entity-type match it finds,
# and a message can easily mention more than one ("my LANDLORD won't return
# my DEPOSIT" matches both Relationship and Property). Relationship goes
# first because when a person and an object both appear, the person is
# almost always the more useful "who is this about" answer (the spec's own
# example expects "landlord," not "deposit," as the entity for that
# sentence) -- the object is usually the fact's DETAIL, not its subject.
_ENTITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Relationship": (
        "brother", "sister", "father", "mother", "husband", "wife", "son", "daughter",
        "landlord", "tenant", "neighbor", "neighbour", "employer", "employee", "boss", "friend",
        # Part 39 section 2's own named categories add teacher/police/lawyer
        # -- these weren't in the original Part 35 list, which only covered
        # family/household relationships.
        "teacher", "police", "lawyer", "advocate",
        "bhai", "behan", "bahan", "pita", "papa", "maa", "mummy", "pati", "patni",
        "beta", "beti", "makan malik", "dost", "shikshak", "vakil",
    ),
    "Vehicle": (
        "bike", "motorbike", "motorcycle", "scooter", "car", "truck", "van", "auto",
        "gaadi", "gadi", "gari",
    ),
    "Property": (
        "house", "flat", "pg", "property", "land", "shop", "deposit", "money", "cash",
        "laptop", "phone", "mobile", "jewellery", "jewelry", "wallet", "luggage", "bag",
        "makan", "paisa", "paise",
    ),
    "Document": (
        "fir", "agreement", "contract", "notice", "complaint", "license", "licence", "passport",
        "aadhar", "aadhaar", "certificate", "cheque", "check", "will", "deed",
    ),
    "Place": ("police station", "thana", "court", "office", "home", "ghar", "shop"),
}

# A body part mentioned alongside an "injured" fact sharpens the stored
# value from generic "injury" to "head injury" (matches the spec's own
# "My brother's head was injured" -> "head injury" example).
_BODY_PARTS = (
    "head", "hand", "leg", "arm", "finger", "back", "eye", "face", "shoulder", "knee",
    "sir", "haath", "pair", "aankh", "chehra",
)

# Each category: (fact_key, keyword list, answer-template using {entity}/{value}).
# `fact_key` is the stored, machine-facing label; keywords double as BOTH the
# statement-side trigger ("my bike was STOLEN") and the query-side trigger
# ("what was STOLEN?") -- deliberately symmetric, so recall doesn't need a
# second, separately-maintained set of question phrasings.
_FACT_CATEGORIES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("stolen", ("stolen", "steal", "chori", "churaya", "churaa", "chura"), "Your {entity} was stolen."),
    ("lost", ("lost", "khoya", "kho gaya", "gum ho gaya", "gum"), "Your {entity} was lost."),
    ("injured", ("injured", "injury", "ghayal", "chot", "phat gaya", "phad", "phod", "phoda"), "Your {entity} suffered a {value}."),
    ("damaged", ("damaged", "broken", "toot gaya", "tuta", "kharab"), "Your {entity} was damaged."),
    (
        "dispute",
        (
            "dispute", "issue", "problem", "not returning", "won't return", "wont return",
            "will not return", "not return", "nahi de raha", "wapas nahi", "preshan kar raha", "harass",
        ),
        "Your {entity} was involved in a {value}.",
    ),
    ("cheated", ("cheated", "fraud", "dhoka", "thagi"), "You were cheated by your {entity}."),
    ("accident", ("accident", "durghatna"), "Your {entity} was involved in an accident."),
    (
        # Part 39 section 2's own flagship example ("Teacher ne mujhe mara" /
        # "Mujhe kisne mara tha?") names a fact shape ("X hit/beat me") that
        # had no category here before -- "injured" only fires when a body
        # part or injury word is also present, which "mara" alone doesn't
        # supply.
        "assault",
        (
            "mara", "maara", "maar diya", "peeta", "pitai", "thappad", "slapped",
            "beaten", "beat me", "hit me", "assaulted", "thoka",
        ),
        "Your {entity} hit/beat you.",
    ),
)

# Hindi/Hinglish variants of the templates above, keyed the same way as
# `_FACT_CATEGORIES` (fact_key -> template). `format_answer` falls back to
# the English template in `_FACT_CATEGORIES` for any language not listed
# here (or any fact_key missing from this dict) -- deliberately not a full
# per-language matrix, just enough to stop entity-recall answers coming back
# in English when the whole conversation has been in Hindi/Hinglish (Part 39
# section 1: "never randomly switch language").
_HINDI_FACT_TEMPLATES: dict[str, str] = {
    "stolen": "Aapka {entity} chori ho gaya tha.",
    "lost": "Aapka {entity} kho gaya tha.",
    "injured": "Aapke {entity} ko {value} hui thi.",
    "damaged": "Aapka {entity} damage ho gaya tha.",
    "dispute": "Aapke {entity} ko lekar dispute chal raha hai.",
    "cheated": "Aapke {entity} ne aapke saath dhoka kiya tha.",
    "accident": "Aapka {entity} ek accident mein shaamil tha.",
    "assault": "Aapke {entity} ne aapko mara tha.",
}

_INTERROGATIVE_PATTERN = re.compile(
    r"\b(what|who|whom|kya|kaun|kisne|kiska|kiski|kis)\b", re.IGNORECASE
)

# Guards the entity-only fallback in `find_matching_fact`/`matches_known_
# category` below. Several `_ENTITY_KEYWORDS` (notably "fir", "contract",
# "notice", "complaint" under "Document") are also ordinary legal-topic
# nouns that show up constantly in generic questions with no personal-recall
# intent at all ("kaun FIR darj kar sakta hai" -- who CAN file an FIR, not a
# question about a fact the user shared). Requiring a first-person possessive
# marker restricts the fallback to genuinely personal phrasing ("what
# happened to MY bike?") without reopening that false-positive case.
_POSSESSIVE_PATTERN = re.compile(r"\b(my|mera|meri|mere)\b", re.IGNORECASE)

# Distinguishes a FACT-IDENTIFYING question ("What was stolen?", "kisne
# phoda tha" -- who did X) from an ACTION-SEEKING one ("kya karu", "what
# should I do") that merely happens to share an interrogative word with a
# fact just stated in the SAME message ("mera bike chori ho gya hai kya
# krna chahiye" -- a fresh statement asking for advice, not a question
# about something said earlier). Recall must be suppressed for the latter,
# or it wrongly answers "Your bike was stolen" instead of giving real
# advice about what to do next.
#
# Bare `\bshould\b`/`chahiye` (rather than only the narrower "what should I
# do" phrasing above) catches shapes those miss entirely, e.g. "what
# information should I give" or "mujhe pehle kya information deni chahiye"
# -- "kya" here questions the NOUN ("what information"), not the verb
# ("karu"/"karna"), so the narrower `kya\s+(karu|...)` alternation above
# never matches it, yet it is unambiguously advice-seeking, never a
# fact-recall question (genuine recall in this module is always past-tense
# -- "chori hua tha", "kisne phoda tha" -- and never uses "should"/
# "chahiye"), so this is safe to match unconditionally rather than only in
# combination with a specific verb.
_ACTION_SEEKING_PATTERN = re.compile(
    r"what should i do|what do i do|what to do|what can i do"
    r"|what\s+are\s+my\s+(legal\s+)?(rights|options)\b"
    r"|what\s+rights\s+do\s+i\s+have"
    r"|\bshould\s+i\b"
    r"|chahiye"
    r"|kya\s+(karu|kru|karoon|kare|karna|krna)\b"
    r"|(?:main|mai|hum)\s+kya\s+kar\s+sakt(?:a|i|e)\s+(?:hu|hoon|hain)\b"
    r"|(?:mere|hamare)\s+paas\s+kya\s+(?:legal\s+)?options?\s+(?:hai|hain)\b"
    r"|mere\s+(legal\s+)?adhikar\s+kya\s+(hai|hain)\b",
    re.IGNORECASE,
)

# A second, independent false-trigger: several `_FACT_CATEGORIES` keywords
# ("fraud", "dispute", "accident"...) are ordinary legal-topic nouns, not
# just personal-event words -- "UPI fraud mein customer liability kya hoti
# hai" matches "fraud" and is interrogative, but it's a general definitional
# question ("what IS the liability", habitual present), not a recall
# question about something the user said earlier. Genuine recall in this
# module's own examples is always past-tense ("mera kya chori HUA THA",
# "kisne phoda THA"); a habitual-present "kya hoti/hota/hote hai(n)" shape
# is the reliable signal that distinguishes the two without needing a
# possessive marker (which recall questions like "What was stolen?"
# legitimately omit).
_DEFINITIONAL_PATTERN = re.compile(
    r"kya\s+(hoti|hota|hote)\s+hai(n)?\b|\bwhat\s+is\s+the\b|\bwhat\s+are\s+the\b",
    re.IGNORECASE,
)


def _find_keyword(normalized_text: str, keywords: tuple[str, ...]) -> str | None:
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", normalized_text):
            return keyword
    return None


@dataclass(frozen=True)
class EntityFact:
    entity: str
    entity_type: str
    fact_key: str
    fact_value: str
    confidence: float
    turn_index: int
    source_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity, "entity_type": self.entity_type, "fact_key": self.fact_key,
            "fact_value": self.fact_value, "confidence": self.confidence,
            "turn_index": self.turn_index, "source_text": self.source_text,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EntityFact":
        return EntityFact(
            entity=data["entity"], entity_type=data["entity_type"], fact_key=data["fact_key"],
            fact_value=data["fact_value"], confidence=data["confidence"],
            turn_index=data["turn_index"], source_text=data["source_text"],
        )


def extract_fact(text: str, turn_index: int) -> EntityFact | None:
    """Extracts at most one fact from a single message -- matches the
    spec's own examples, each of which states exactly one fact about one
    entity. A message stating two unrelated facts only has the first
    (by category-list order) captured; good enough for a v1 built directly
    from three concrete examples, not a claim of completeness.
    """
    normalized = text.lower()
    for fact_key, keywords, _template in _FACT_CATEGORIES:
        matched_keyword = _find_keyword(normalized, keywords)
        if matched_keyword is None:
            continue
        for entity_type, entity_keywords in _ENTITY_KEYWORDS.items():
            matched_entity = _find_keyword(normalized, entity_keywords)
            if matched_entity is None:
                continue
            fact_value = matched_keyword
            if fact_key == "injured":
                body_part = _find_keyword(normalized, _BODY_PARTS)
                fact_value = f"{body_part} injury" if body_part else "injury"
            elif fact_key == "dispute":
                fact_value = "dispute"
            return EntityFact(
                entity=matched_entity, entity_type=entity_type, fact_key=fact_key,
                fact_value=fact_value, confidence=0.7, turn_index=turn_index, source_text=text,
            )
    return None


def is_recall_query(text: str) -> bool:
    if _ACTION_SEEKING_PATTERN.search(text) or _DEFINITIONAL_PATTERN.search(text):
        return False
    return bool(_INTERROGATIVE_PATTERN.search(text))


def find_matching_fact(text: str, facts: list[dict[str, Any]]) -> EntityFact | None:
    """Given the CURRENT message and the session's stored facts (as raw
    dicts, `memory["entities"]`'s persisted shape), finds the most recent
    stored fact whose category the message is asking about. Returns `None`
    when nothing in the message names a known fact category, OR when
    nothing stored matches it -- callers fall through to normal routing
    (RAG etc.) either way, this never fabricates an answer.

    If the CURRENT message itself names a specific entity ("Was my PHONE
    stolen?"), that entity is authoritative -- only a stored fact about
    THAT entity can answer it, even if a different entity has a fact in
    the same category ("bike"/stolen). Never silently substitutes a
    different, previously-mentioned entity just because the fact category
    matches; that's exactly the "current message says phone, don't answer
    bike" failure mode this guards against. Only when the message doesn't
    name any entity at all ("What was stolen?") does the most recent
    matching fact, regardless of entity, apply -- there's nothing more
    specific to prefer.
    """
    if not facts:
        return None
    normalized = text.lower()
    queried_entity = None
    for entity_keywords in _ENTITY_KEYWORDS.values():
        keyword = _find_keyword(normalized, entity_keywords)
        if keyword is not None:
            queried_entity = keyword
            break
    for fact_key, keywords, _template in _FACT_CATEGORIES:
        if _find_keyword(normalized, keywords) is None:
            continue
        matching = [EntityFact.from_dict(f) for f in facts if f.get("fact_key") == fact_key]
        if queried_entity is not None:
            matching = [fact for fact in matching if fact.entity == queried_entity]
        if matching:
            return max(matching, key=lambda fact: fact.turn_index)

    # Generic recall phrasing ("What happened to my bike?") names the entity
    # but no category keyword -- the loop above never matches it even though
    # a stored fact about that exact entity exists. Only applies when the
    # message names a specific entity AND uses a first-person possessive
    # (see `_POSSESSIVE_PATTERN`) -- distinguishes "what happened to MY
    # bike?" (personal recall) from a generic entity mention with no
    # recall intent ("kaun FIR darj kar sakta hai").
    if queried_entity is not None and _POSSESSIVE_PATTERN.search(normalized):
        entity_facts = [EntityFact.from_dict(f) for f in facts if f.get("entity") == queried_entity]
        if entity_facts:
            return max(entity_facts, key=lambda fact: fact.turn_index)
    return None


def matches_known_category(text: str) -> bool:
    """True when the message names a fact category this module tracks
    (stolen/lost/injured/...), independent of whether anything with that
    category is actually stored. Distinguishes a genuine "I don't
    remember" case (category named, nothing stored yet) from an ordinary
    interrogative question this module has no opinion about ("kaun FIR
    darj kar sakta hai") that should fall through to RAG/general knowledge
    untouched rather than being hijacked into a bogus no-memory answer.

    Also true when the message names a known entity type (bike, phone,
    ...) WITH a first-person possessive marker ("my", "mera"...) and no
    category keyword -- generic recall phrasing ("What happened to my
    bike?") is just as much "this module has an opinion" as
    category-keyword phrasing; it should abstain with "I don't remember"
    rather than silently falling through when nothing about that entity
    is stored (mirrors the entity-only fallback in `find_matching_fact`).
    The possessive requirement keeps this from firing on generic
    questions that merely mention an entity-type noun with no personal
    recall intent (e.g. "kaun FIR darj kar sakta hai" -- "fir" is a known
    Document-entity keyword, but the question isn't about a stored fact).
    """
    normalized = text.lower()
    if any(_find_keyword(normalized, keywords) is not None for _fact_key, keywords, _template in _FACT_CATEGORIES):
        return True
    if not _POSSESSIVE_PATTERN.search(normalized):
        return False
    return any(_find_keyword(normalized, entity_keywords) is not None for entity_keywords in _ENTITY_KEYWORDS.values())


_NO_MEMORY_ANSWERS: dict[str, str] = {
    "english": "I don't remember that — you haven't shared this information with me earlier in this conversation.",
    "hindi": "Mujhe yaad nahi hai, kyunki aapne yeh jaankari pehle share nahi ki thi.",
    "hinglish": "Mujhe yaad nahi hai, kyunki aapne ye information pehle share nahi ki thi.",
}


def format_no_memory_answer(language: str = "english") -> str:
    return _NO_MEMORY_ANSWERS.get(language, _NO_MEMORY_ANSWERS["english"])


def format_answer(fact: EntityFact, language: str = "english") -> str:
    if language in ("hindi", "hinglish") and fact.fact_key in _HINDI_FACT_TEMPLATES:
        template = _HINDI_FACT_TEMPLATES[fact.fact_key]
    else:
        template = next(t for key, _kw, t in _FACT_CATEGORIES if key == fact.fact_key)
    return template.format(entity=fact.entity, value=fact.fact_value)
