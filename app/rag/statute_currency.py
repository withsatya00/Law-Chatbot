"""Part 58 "Answer Quality Audit" issue 2: keeping repealed-code citations
straight.

The knowledge base mixes post-2024 statute text (BNS/BNSS/BSA) with older
explanatory material that still cites the codes those Acts replaced (IPC,
CrPC, the Indian Evidence Act). Grounded answers therefore drift between the
two depending on which chunk retrieval happened to surface -- confirmed live
in a single session: the first "What is an FIR" answer correctly cited
"Section 173 of the BNSS", while a second, identical question came back
citing "Section 154" with no Act named at all and no indication that 154 is
the OLD CrPC provision. A reader has no way to tell those are the same rule
under two numbering schemes, or which one is currently in force.

Two mechanisms, deliberately separate:

* `currency_directive()` runs BEFORE generation and puts the relevant
  mappings into the prompt, so the model can cite the current provision as
  the operative one in its own words.
* `annotate_answer()` runs AFTER generation as a safety net, appending a
  factual clarification when an answer names a repealed provision and never
  names its successor. It only ever ADDS a note -- it never rewrites or
  contradicts the grounded answer.

Every mapping below is a documented section-to-section correspondence
between the repealed code and its 2023 replacement. Extend only with
equally-verifiable entries: a wrong mapping here is worse than a missing one,
since it would be appended to the answer as a statement of fact.
"""

import re
from dataclasses import dataclass

# Acts whose numbering these mappings translate FROM, with the Act that
# replaced them and the date the replacement took effect.
_REPLACEMENTS: dict[str, tuple[str, str]] = {
    "CrPC": ("Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)", "1 July 2024"),
    "IPC": ("Bharatiya Nyaya Sanhita, 2023 (BNS)", "1 July 2024"),
    "IEA": ("Bharatiya Sakshya Adhiniyam, 2023 (BSA)", "1 July 2024"),
}

# How each repealed Act is named in ordinary prose, for detecting an explicit
# citation ("Section 154 of the CrPC", "under the Code of Criminal Procedure").
_OLD_ACT_PATTERNS: dict[str, re.Pattern[str]] = {
    "CrPC": re.compile(r"\bcr\.?\s?p\.?\s?c\.?\b|code of criminal procedure|दंड प्रक्रिया संहिता", re.IGNORECASE),
    "IPC": re.compile(r"\bi\.?\s?p\.?\s?c\.?\b|indian penal code|भारतीय दंड संहिता", re.IGNORECASE),
    "IEA": re.compile(r"\bindian evidence act\b|\bevidence act, 1872\b|भारतीय साक्ष्य अधिनियम", re.IGNORECASE),
}


@dataclass(frozen=True)
class SectionMapping:
    old_act: str
    old_section: str
    new_act: str
    new_section: str
    subject: str
    # Words that must co-occur with a BARE section number (one named without
    # its Act) before it is treated as this provision. An explicitly
    # Act-qualified citation skips this check. Without it, an answer about a
    # completely different statute that happens to mention "Section 125"
    # would be annotated as if it were CrPC maintenance.
    topic_keywords: tuple[str, ...]


_MAPPINGS: tuple[SectionMapping, ...] = (
    SectionMapping("CrPC", "154", "BNSS", "173", "information in cognizable cases (FIR registration)",
                   ("fir", "first information", "cognizable", "प्राथमिकी", "संज्ञेय")),
    SectionMapping("CrPC", "41", "BNSS", "35", "when police may arrest without a warrant",
                   ("arrest", "warrant", "गिरफ्तार", "वारंट")),
    SectionMapping("CrPC", "41A", "BNSS", "35(3)", "notice of appearance before a police officer",
                   ("notice", "appearance", "arrest", "नोटिस", "गिरफ्तार")),
    SectionMapping("CrPC", "156", "BNSS", "175", "power of police to investigate a cognizable case",
                   ("investigat", "cognizable", "जांच", "अन्वेषण")),
    SectionMapping("CrPC", "161", "BNSS", "180", "examination of witnesses by police",
                   ("witness", "statement", "गवाह", "बयान")),
    SectionMapping("CrPC", "164", "BNSS", "183", "recording of confessions and statements by a Magistrate",
                   ("confession", "magistrate", "statement", "इकबालिया", "मजिस्ट्रेट")),
    SectionMapping("CrPC", "173", "BNSS", "193", "police report on completion of investigation (charge-sheet)",
                   ("charge sheet", "charge-sheet", "chargesheet", "final report", "आरोप पत्र")),
    SectionMapping("CrPC", "436", "BNSS", "478", "bail in bailable offences",
                   ("bail", "जमानत")),
    SectionMapping("CrPC", "437", "BNSS", "480", "bail in non-bailable offences",
                   ("bail", "जमानत")),
    SectionMapping("CrPC", "438", "BNSS", "482", "anticipatory bail",
                   ("bail", "anticipatory", "जमानत", "अग्रिम")),
    SectionMapping("CrPC", "439", "BNSS", "483", "special powers of the High Court/Sessions Court regarding bail",
                   ("bail", "जमानत")),
    SectionMapping("CrPC", "125", "BNSS", "144", "maintenance of wives, children and parents",
                   ("maintenance", "भरण", "पोषण")),
    SectionMapping("CrPC", "144", "BNSS", "163", "orders in urgent cases of nuisance or apprehended danger",
                   ("prohibitory", "nuisance", "apprehended danger", "निषेधाज्ञा")),
    SectionMapping("IPC", "420", "BNS", "318", "cheating and dishonestly inducing delivery of property",
                   ("cheat", "fraud", "धोखा", "ठग")),
    SectionMapping("IPC", "302", "BNS", "101", "murder",
                   ("murder", "हत्या")),
    SectionMapping("IPC", "376", "BNS", "64", "punishment for rape",
                   ("rape", "बलात्कार")),
    SectionMapping("IPC", "379", "BNS", "303", "theft",
                   ("theft", "चोरी")),
    SectionMapping("IPC", "406", "BNS", "316", "criminal breach of trust",
                   ("breach of trust", "misappropriat", "अमानत")),
    SectionMapping("IPC", "498A", "BNS", "85", "cruelty by a husband or his relatives",
                   ("cruelty", "dowry", "क्रूरता", "दहेज")),
    SectionMapping("IPC", "506", "BNS", "351", "criminal intimidation",
                   ("intimidation", "threat", "धमकी")),
    SectionMapping("IPC", "354D", "BNS", "78", "stalking",
                   ("stalk", "पीछा")),
    SectionMapping("IPC", "499", "BNS", "356", "defamation",
                   ("defamation", "मानहानि")),
    SectionMapping("IEA", "65B", "BSA", "63", "admissibility of electronic records",
                   ("electronic", "digital", "इलेक्ट्रॉनिक")),
)

_MAPPINGS_BY_OLD: dict[tuple[str, str], SectionMapping] = {
    (mapping.old_act, mapping.old_section.upper()): mapping for mapping in _MAPPINGS
}

# Any "Section <n>" / "धारा <n>" / bare "IPC 420" style citation in free text.
_SECTION_CITATION_RE = re.compile(
    r"(?:section|sec\.?|s\.|धारा|कलम)\s*(\d{1,4}[A-Za-z]{0,2})"
    r"|\b(?:IPC|CrPC)\s*(\d{1,4}[A-Za-z]{0,2})\b"
    r"|\b(\d{1,4}[A-Za-z]{0,2})\s*(?:of\s+(?:the\s+)?)?(?:IPC|CrPC)\b",
    re.IGNORECASE,
)


def _cited_section_numbers(text: str) -> set[str]:
    numbers: set[str] = set()
    for match in _SECTION_CITATION_RE.finditer(text or ""):
        number = next((group for group in match.groups() if group), "")
        if number:
            numbers.add(number.upper())
    return numbers


def _old_acts_named(text: str) -> set[str]:
    return {act for act, pattern in _OLD_ACT_PATTERNS.items() if pattern.search(text or "")}


def _relevant_mappings(text: str) -> list[SectionMapping]:
    """Mappings whose OLD provision `text` plausibly cites.

    A number qualified by its Act ("Section 154 CrPC", "IPC 420") matches on
    the number alone. A bare number additionally has to co-occur with one of
    that provision's own topic words, so an unrelated statute's Section 125
    is never mistaken for CrPC maintenance.
    """
    numbers = _cited_section_numbers(text)
    if not numbers:
        return []
    lowered = (text or "").lower()
    named_acts = _old_acts_named(text)
    found: list[SectionMapping] = []
    for mapping in _MAPPINGS:
        if mapping.old_section.upper() not in numbers:
            continue
        if mapping.old_act in named_acts:
            found.append(mapping)
            continue
        if any(keyword in lowered for keyword in mapping.topic_keywords):
            found.append(mapping)
    return found


def currency_directive(question: str, context: str) -> str:
    """A prompt block naming every repealed provision this turn's question or
    retrieved context cites, so the answer can lead with the provision that is
    actually in force. Empty string when nothing repealed is in play, so the
    prompt stays unchanged for the overwhelming majority of turns.
    """
    mappings = _relevant_mappings(f"{question}\n{context}")
    if not mappings:
        return ""
    lines = [
        (
            "Statutory currency note -- the retrieved material and/or the question reference provisions of a "
            "REPEALED code. Cite the provision currently in force as the operative one, and mention the old "
            "number only to connect it to what the user may already know, explicitly flagged as pre-1-July-2024:"
        ),
    ]
    seen: set[tuple[str, str]] = set()
    for mapping in mappings:
        key = (mapping.old_act, mapping.old_section)
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- {mapping.old_act} Section {mapping.old_section} (repealed) corresponds to "
            f"{mapping.new_act} Section {mapping.new_section} -- {mapping.subject}."
        )
    lines.append(
        "Never present a repealed section number on its own as the current law, and never cite a bare "
        "section number without naming the Act it belongs to."
    )
    return "\n".join(lines)


# The appended clarification, per language. English is the authored original;
# the rest follow `NO_VERIFIED_CONTEXT_MESSAGES`' pattern of covering the
# languages this product actually replies in, falling back to English.
_NOTE_TEMPLATES: dict[str, str] = {
    "english": (
        "Note on numbering: Section {old_section} belongs to the {old_act_full}, which was replaced by the "
        "{new_act_full} with effect from {effective_date}. The corresponding provision now in force is "
        "Section {new_section} of the {new_act_short} ({subject})."
    ),
    "hindi": (
        "क्रमांक संबंधी सूचना: धारा {old_section} {old_act_full} की है, जिसे {effective_date} से "
        "{new_act_full} द्वारा प्रतिस्थापित कर दिया गया है। वर्तमान में लागू संगत प्रावधान "
        "{new_act_short} की धारा {new_section} है ({subject})।"
    ),
    "hinglish": (
        "Numbering note: Section {old_section} {old_act_full} ka provision hai, jise {effective_date} se "
        "{new_act_full} ne replace kar diya hai. Abhi lagu corresponding provision {new_act_short} "
        "Section {new_section} hai ({subject})."
    ),
    "marathi": (
        "क्रमांकाविषयी सूचना: कलम {old_section} हे {old_act_full} मधील आहे, जे {effective_date} पासून "
        "{new_act_full} ने बदलले आहे. सध्या लागू असलेली संबंधित तरतूद {new_act_short} चे कलम "
        "{new_section} आहे ({subject})."
    ),
    "gujarati": (
        "ક્રમાંક અંગે નોંધ: કલમ {old_section} એ {old_act_full} ની છે, જેને {effective_date} થી "
        "{new_act_full} દ્વારા બદલવામાં આવી છે. હાલમાં અમલમાં રહેલી સંબંધિત જોગવાઈ {new_act_short} ની "
        "કલમ {new_section} છે ({subject})."
    ),
    "bengali": (
        "ধারা-সংখ্যা সম্পর্কিত নোট: ধারা {old_section} {old_act_full}-এর অন্তর্গত, যা {effective_date} "
        "থেকে {new_act_full} দ্বারা প্রতিস্থাপিত হয়েছে। বর্তমানে কার্যকর সংশ্লিষ্ট বিধানটি হল "
        "{new_act_short}-এর ধারা {new_section} ({subject})।"
    ),
}

_ACT_FULL_NAMES: dict[str, str] = {
    "CrPC": "Code of Criminal Procedure, 1973",
    "IPC": "Indian Penal Code, 1860",
    "IEA": "Indian Evidence Act, 1872",
}


def annotate_answer(answer: str, language: str | None) -> str:
    """Appends a factual numbering clarification when `answer` cites a
    repealed provision without ever naming the provision that replaced it.

    Deliberately additive: the grounded answer text is never edited, so this
    can't corrupt a correct answer -- at worst it appends a note the reader
    already knew. Silent (returns `answer` unchanged) whenever the successor
    provision is already mentioned, which is the normal case once
    `currency_directive` has done its job.
    """
    if not answer:
        return answer
    mappings = _relevant_mappings(answer)
    if not mappings:
        return answer
    template = _NOTE_TEMPLATES.get((language or "english").strip().lower(), _NOTE_TEMPLATES["english"])
    notes: list[str] = []
    seen: set[tuple[str, str]] = set()
    for mapping in mappings:
        key = (mapping.old_act, mapping.old_section)
        if key in seen:
            continue
        seen.add(key)
        # Already handled by the answer itself -- both the successor's number
        # and its Act have to be present, so a coincidental "173" elsewhere in
        # an unrelated sentence doesn't suppress a genuinely needed note.
        if mapping.new_section.split("(")[0] in answer and mapping.new_act in answer:
            continue
        new_act_full, effective_date = _REPLACEMENTS[mapping.old_act]
        notes.append(
            template.format(
                old_section=mapping.old_section,
                old_act_full=_ACT_FULL_NAMES[mapping.old_act],
                new_act_full=new_act_full,
                new_act_short=mapping.new_act,
                new_section=mapping.new_section,
                subject=mapping.subject,
                effective_date=effective_date,
            )
        )
    if not notes:
        return answer
    return f"{answer}\n\n" + "\n\n".join(notes)


# ---------------------------------------------------------------------------
# Phase 2: source-status handling for retrieval and for the answer's own
# "how current is this?" disclosure.
# ---------------------------------------------------------------------------

# How much a chunk's ranking is adjusted by the LEGAL STATUS of the source it
# came from. Small, deliberately: this is a preference, not a filter. A repealed
# provision is still the right answer for an incident that happened while it was
# in force, so it must stay reachable -- it just should not outrank the current
# provision for a question about current law.
_STATUS_RANKING_ADJUSTMENT: dict[str, float] = {
    "in_force": 0.06,
    "amended": 0.02,
    "unknown": 0.0,
    "superseded": -0.10,
    "repealed": -0.12,
}

# A verified source is one a human compared against the issuing authority's own
# text (see `app/services/phase3.py`). Unverified is the default and carries no
# penalty -- most of the corpus is unverified, and penalising it would amount to
# ranking by how much review has happened rather than by relevance. Only an
# explicitly REJECTED source is pushed down.
_VERIFICATION_RANKING_ADJUSTMENT: dict[str, float] = {
    "verified": 0.05,
    "machine_verified": 0.03,
    "pending_review": 0.0,
    "unverified": 0.0,
    "rejected": -0.20,
}

# Phrases that mean the user is asking about a PAST incident, where the law in
# force at the time governs rather than the law in force today. Matching any of
# these is what turns "prefer current law" into "say that it depends on when".
_PAST_INCIDENT_PATTERNS = re.compile(
    r"\b(in|back in|during|as of)\s+(19|20)\d{2}\b"
    r"|\bbefore\s+(1\s*july\s*2024|july\s*2024|2024)\b"
    r"|\b(happened|occurred|filed|registered|charged|convicted|arrested)\s+in\s+(19|20)\d{2}\b"
    r"|\bold\s+case\b|\bpending\s+(case|trial)\b|\bat\s+the\s+time\b",
    re.IGNORECASE,
)


def status_ranking_adjustment(metadata: dict[str, object]) -> float:
    """Ranking adjustment for one chunk, from its source's governance metadata.

    Returns 0.0 for a chunk with no governance metadata at all, which is most of
    the existing corpus -- an ungoverned source is treated as neutral, not as
    suspect. Deliberately bounded and small so it can reorder near-ties without
    overriding actual relevance.
    """
    status = str(metadata.get("amendment_status") or metadata.get("status") or "unknown")
    verification = str(metadata.get("verification_status") or "unverified")
    return _STATUS_RANKING_ADJUSTMENT.get(status, 0.0) + _VERIFICATION_RANKING_ADJUSTMENT.get(
        verification, 0.0
    )


def asks_about_a_past_incident(question: str) -> bool:
    """Whether the question is about something that already happened.

    The BNS/BNSS/BSA took effect on 1 July 2024 and did not apply retrospectively
    to offences committed before then, so "what is the punishment for cheating"
    and "what was the punishment when it happened in 2019" have different correct
    answers. This is the signal that the answer must disclose that dependence
    rather than silently pick one.
    """
    return bool(_PAST_INCIDENT_PATTERNS.search(question or ""))


def currency_notice(
    statuses: "list[dict[str, object]]", question: str = "", language: str | None = None
) -> str:
    """A factual note about how current the cited sources are, or "".

    Says only what the governance metadata actually records. It never asserts
    that a source IS current -- an unverified `in_force` record is a claim
    nobody has checked, and reporting it as settled would be exactly the
    overstatement the registry exists to prevent.
    """
    if not statuses:
        return ""
    lines: list[str] = []

    outdated = sorted(
        {
            str(item.get("source_document") or "a cited source")
            for item in statuses
            if str(item.get("amendment_status") or item.get("status") or "") in {"repealed", "superseded"}
        }
    )
    if outdated:
        lines.append(
            "Currency note: "
            + ", ".join(outdated)
            + " is recorded as repealed or superseded. Treat it as the law that WAS in force, "
            "not as current law, and confirm the replacement provision before relying on it."
        )

    unknown = [
        item for item in statuses
        if str(item.get("amendment_status") or item.get("status") or "unknown") == "unknown"
    ]
    if unknown and not outdated:
        lines.append(
            "Currency note: the amendment status of the cited source(s) is not recorded, "
            "so whether they reflect the latest amendments has not been verified."
        )

    unverified = [
        item for item in statuses
        if str(item.get("verification_status") or "unverified") in {"unverified", "pending_review"}
    ]
    if unverified:
        lines.append(
            "Verification note: the cited source(s) have not been verified against the "
            "issuing authority's own publication. Check the official text before acting on this."
        )

    if asks_about_a_past_incident(question):
        lines.append(
            "Because your question refers to an earlier date: the Bharatiya Nyaya Sanhita, "
            "Bharatiya Nagarik Suraksha Sanhita and Bharatiya Sakshya Adhiniyam took effect on "
            "1 July 2024 and do not apply retrospectively to offences committed before that "
            "date. Which law governs therefore depends on when the incident occurred, and that "
            "should be confirmed with a lawyer for your specific dates."
        )
    return "\n\n".join(lines)
