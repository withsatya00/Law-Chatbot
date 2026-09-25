import asyncio
import re
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, cast, get_args
from uuid import uuid4

import structlog

from app.core import clock
from app.core.config import settings
from app.core.constants import DRAFT_DISCLAIMER
from app.core.exceptions import BadRequestError, DraftConflictError, DraftLockedError, NotFoundError
from app.drafting import advocate_register, citation_audit, prohibited_clauses
from app.drafting.document_grammar import (
    DocumentGrammarValidator,
    HeadingMode,
    document_schema_registry,
)
from app.drafting.export import (
    DocxDraftExporter,
    DraftExporter,
    ExportOptions,
    PdfDraftExporter,
    RtfDraftExporter,
    TxtDraftExporter,
)
from app.drafting.fact_audit import audit_draft, strip_invented_details
from app.drafting.fallback_phrases import (
    closing_designator,
    localized_authority_label,
    localized_subject_template,
    phrase,
)
from app.drafting.field_label_translations import localized_field_label
from app.drafting.glossary import LegalTermsLibrary
from app.drafting.heading_translations import translated_heading
from app.drafting.localized_dates import format_localized_date
from app.drafting.officialese import apply_official_scaffolding
from app.drafting.safety import DraftSafetyGuard
from app.drafting.templates import get_template as _get_template_definition
from app.drafting.templates import list_templates as _list_template_definitions
from app.drafting.templates.base import (
    OPTIONAL_SECTIONS,
    DraftField,
    DraftTemplateDefinition,
)
from app.drafting.title_translations import localized_title
from app.drafting.wrapper_messages import msg
from app.llm.base import ChatMessage
from app.llm.deadline import call_with_hard_timeout, deadline
from app.llm.factory import LLMFactory
from app.llm.prompts import prompt_registry
from app.rag.retriever import LegalRetriever
from app.repositories.drafts import DraftRepository, DraftVersionRepository
from app.schemas.drafting import (
    DraftFieldSchema,
    DraftGenerateRequest,
    DraftGenerateResponse,
    DraftGenerationMode,
    DraftHistoryRequest,
    DraftLifecycleState,
    DraftPreviewRequest,
    DraftSummary,
    DraftTemplateDetail,
    DraftTemplateSummary,
    DraftTranslateRequest,
    DraftTranslateResponse,
    LegalTermEntry,
)
from app.utils.prompt_security import PromptInjectionScanner

log = structlog.get_logger(__name__)

ExportFormat = Literal["pdf", "docx", "txt", "rtf"]

# Part 57 "Drafting Lifecycle Redesign": the persisted `legal_drafts.
# lifecycle_state` values from which content edits (regenerate/translate/
# replace_field/remove_paragraph/rollback) are permitted -- everything past
# "approved" is frozen until an explicit `unlock()`. A record with no
# `lifecycle_state` at all (any draft persisted before this change) defaults
# to "locked" everywhere it's read (see `_lifecycle_state_of` below), not to
# an editable state -- treating a pre-existing generated draft as
# already-approved/exportable matches what it already behaved like (exports
# were always open), rather than silently unlocking untouched old drafts.
_EDITABLE_STATES = {"preview_ready", "editing"}
_LOCKABLE_STATES = {"approved", "locked", "exported"}
_LEGACY_DEFAULT_LIFECYCLE_STATE: DraftLifecycleState = "locked"
_LIFECYCLE_STATES: frozenset[str] = frozenset(get_args(DraftLifecycleState))
# States a download is permitted from. Part 57 originally restricted this to
# {"locked", "exported"}, forcing every user through an "approve" -> "lock"
# confirmation pair before any PDF/DOCX/TXT/RTF button appeared. That gate is
# removed by product decision: a generated draft is downloadable immediately,
# and stays editable afterwards, so the conversation reads as
# "here's your draft -> download it, or tell me what to change" with no
# ceremony in between. `approve()`/`lock()` are retained (see below) purely so
# existing API clients and the DRAFT_VERSIONS lifecycle keep working.
_EXPORTABLE_STATES = _EDITABLE_STATES | _LOCKABLE_STATES


# The single wording for "the model replied, but not in the shape a draft
# needs". A named constant because `_render_sections_within_deadline` branches
# on it to decide whether one format re-ask is worth trying -- comparing
# against a sentence literal in two places is how those two drift apart.
_UNPARSABLE_RESPONSE_ERROR = "The drafting model's response could not be parsed into the required sections."


def _created_at_isoformat(value: datetime | str) -> str:
    """`record["created_at"]` is a native `datetime` when read straight from
    Mongo, but a plain ISO string when read via the Postgres JSONB payload
    path (`postgres_json.dump_payload`/`load_payload` round-trip datetimes
    through `json.dumps(..., default=_json_default)`, which serialises them
    to strings and never converts them back). Calling `.isoformat()`
    unconditionally crashed `history()` with `AttributeError: 'str' object
    has no attribute 'isoformat'` every time `POSTGRESQL_ENABLED=true` (the
    default), for every draft, on both the session- and user-scoped lookup.
    """
    if isinstance(value, str):
        return value
    return value.isoformat()


def _lifecycle_state_of(draft: dict[str, Any]) -> DraftLifecycleState:
    """The draft's lifecycle state, as the closed literal the API responses use.

    The stored value is whatever is in Mongo -- including one written by an
    older build, or absent entirely. An unrecognised value used to travel all
    the way to `DraftLifecycleResponse` and fail there as a pydantic error
    while serialising an otherwise successful operation; it now degrades to the
    same conservative `locked` default an absent value already got, and says so.
    """
    stored = draft.get("lifecycle_state")
    if stored in _LIFECYCLE_STATES:
        return cast(DraftLifecycleState, stored)
    if stored:
        log.warning(
            "unrecognised_draft_lifecycle_state",
            stored=stored, using=_LEGACY_DEFAULT_LIFECYCLE_STATE,
        )
    return _LEGACY_DEFAULT_LIFECYCLE_STATE

# Part 42/52 originally forced numeric DD/MM/YYYY unconditionally here (e.g.
# "13/08/2026") to avoid `date.strftime("%B")` leaking the host process's
# single global OS-locale month name into a document written in a different
# language. Part 51 "Draft Generation Quality Pass" fixes the actual root
# cause instead (`format_localized_date` in `localized_dates.py`, a
# locale-independent per-language month-name lookup) so the "Date" section
# can properly read "13 August 2026" / "13 अगस्त 2026" / "13 ஆகஸ்ட் 2026" in
# the document's own language. This constant remains only as the numeric
# fallback for languages `format_localized_date` doesn't cover, and inside
# the (now-dead-by-the-time-`_render_sections`-overwrites-it) deterministic
# section builders below.
_DATE_FORMAT = "%d/%m/%Y"


# How many times a short LLM draft may be sent back for expansion. Two passes
# is a deliberate ceiling: each one costs a full generation round-trip, and
# past the second pass a model that still has not reached the target is
# telling us the supplied facts genuinely do not support more text. Pressing
# further would only buy length by invention.
_MAX_EXPANSION_PASSES = 2


@dataclass(frozen=True)
class _RenderedDraft:
    """Everything `_render_sections` learned while producing a draft.

    Replaces the previous `(sections, generated_by_llm, word_count)` tuple so
    that HOW the text was produced -- and, when the LLM path failed, WHY --
    travels with it instead of being discarded at the return statement. That
    discarded information is precisely what let a provider outage reach the
    user as an unlabelled one-page stub.
    """

    sections: dict[str, str]
    generated_by_llm: bool
    word_count: int
    page_count: int
    generation_mode: DraftGenerationMode
    generation_error: str | None


# Sentence-terminating punctuation across the scripts this app drafts in:
# the Latin full stop/question/exclamation, the Devanagari/Bengali/Odia danda
# and double danda, and the Urdu full stop. Used to tell a complete-sentence
# relief ("...दर्ज की जाए।") from a bare noun phrase ("recovery of Rs. 15,000"),
# which decides whether the Prayer INTRODUCES it or inflects it.
_SENTENCE_ENDINGS = ".?!\u0964\u0965\u06d4"
# A relief long enough to be a clause even without punctuation. Chosen well
# above any realistic noun phrase ("registration of my complaint") so the
# inflecting template keeps working for the short reliefs it reads correctly
# with.
_SENTENCE_WORD_THRESHOLD = 12


def _is_complete_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return stripped[-1] in _SENTENCE_ENDINGS or len(stripped.split()) >= _SENTENCE_WORD_THRESHOLD


# Abbreviations whose trailing full stop is NOT a sentence end. Without this
# guard, "Goods worth Rs. 30000 were not delivered" split into two pleading
# paragraphs ("That Goods worth Rs." / "That 30000 were not delivered"),
# mangling the user's own account of the facts in the one section a legal
# document is actually built from. Indian legal drafting is dense with these:
# currency, section references, honorifics, and company suffixes.
_NON_TERMINAL_ABBREVIATIONS = frozenset({
    "rs", "inr", "no", "nos", "vs", "v", "mr", "mrs", "ms", "dr", "prof", "smt", "shri",
    "sri", "kum", "m/s", "ltd", "pvt", "co", "sec", "secs", "s", "u/s", "art", "cl",
    "sch", "para", "pp", "approx", "etc", "govt", "dept", "hon'ble", "honble", "st",
    "jr", "sr", "ph", "i.e", "e.g", "viz",
})
# The Latin full stop is ambiguous (it also ends abbreviations); the
# Devanagari/Bengali/Odia danda and the Urdu full stop are not, so they always
# terminate a sentence and need no guard.
_UNAMBIGUOUS_TERMINATORS = "\u0964\u0965\u06d4"


def _is_sentence_boundary(text: str, position: int) -> bool:
    """Whether the terminator at `position` really ends a sentence."""
    terminator = text[position]
    if terminator in _UNAMBIGUOUS_TERMINATORS or terminator in "?!":
        return True
    preceding = text[:position]
    last_token = re.split(r"[\s(\[]", preceding)[-1].strip().lower() if preceding.strip() else ""
    if last_token in _NON_TERMINAL_ABBREVIATIONS:
        return False
    # A lone initial ("A." in "A. K. Sharma") is never a sentence end.
    return not (len(last_token) == 1 and last_token.isalpha())


def _split_sentences(text: str) -> list[str]:
    """`text` split into sentences, respecting legal-drafting abbreviations.

    Terminators are kept on the sentence they end, and nothing is altered,
    added, or dropped -- this only decides where paragraph breaks go.
    """
    stripped = (text or "").strip()
    if not stripped:
        return []
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(stripped):
        if char not in _SENTENCE_ENDINGS or not _is_sentence_boundary(stripped, index):
            continue
        # Only break when whitespace (or the end of the text) follows -- a
        # full stop inside "15.08.2026" or "www.x.in" is not a boundary.
        if index + 1 < len(stripped) and not stripped[index + 1].isspace():
            continue
        candidate = stripped[start : index + 1].strip()
        if candidate:
            sentences.append(candidate)
        start = index + 1
    remainder = stripped[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences or [stripped]


def _split_reliefs(text: str) -> list[str]:
    """Splits a multi-part relief into separately enumerable items.

    Splits only on explicit list separators the user themselves wrote --
    newlines, semicolons, and genuine sentence ends. Never on commas or
    conjunctions: "action against the accused and recovery of my money" is
    one relief expressed in one breath, and chopping it in two would put
    words in the applicant's mouth about what they are asking for.
    """
    stripped = (text or "").strip()
    if not stripped:
        return []
    items: list[str] = []
    for chunk in re.split(r"[\n;]+", stripped):
        items.extend(_split_sentences(chunk))
    items = [item for item in items if item.strip()]
    return items or [stripped]


def _numbered_pleading_paragraphs(facts: str, language: str) -> str:
    """The user's facts as numbered "1. That ..." pleading paragraphs.

    One paragraph per line the user wrote, or per sentence when they wrote a
    single block -- which is how an Indian complaint or petition is actually
    laid out, and which is a formatting change only: not one word of the
    user's own text is altered, added to, or dropped.
    """
    stripped = (facts or "").strip()
    if not stripped:
        return ""
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) == 1:
        lines = _split_sentences(stripped)
    return "\n\n".join(
        f"{index}. " + phrase("affidavit_statement_line", language, "That {line}", line=line)
        for index, line in enumerate(lines, start=1)
    )


# Field keys whose presence means the user actually told us about an impact
# they suffered, licensing the "loss, inconvenience and hardship" sentence.
# Without one of these, that sentence is generic filler asserted on their
# behalf (`legal_drafting_prompt.md` rule 15), so a narrower line is used.
#
# Post-Phase-3 hardening (Phase 2, milestone C): the money fields were removed
# from this tuple. It used to also list `fraud_amount`, `dues_amount`,
# `principal_amount`, `amount_paid`, `cheque_amount` and
# `security_deposit_amount`, so filling in ANY sum selected the impact
# sentence. That is what put "the sender has suffered financial loss,
# inconvenience and mental distress, for which you are held responsible" into
# a rent/security-deposit notice whose author had described no distress at
# all: the deposit amount is the sum being claimed, not a statement about how
# the claimant feels. A quantified claim and a pleaded injury are different
# things, and only the second licenses this sentence.
_IMPACT_FIELD_KEYS = ("loss_suffered", "damages_claimed")


def _mentions_impact(fields: dict[str, str]) -> bool:
    return any((fields.get(key) or "").strip() for key in _IMPACT_FIELD_KEYS)


def _looks_like_untrimmed_user_prompt(value: str, fields: dict[str, str]) -> bool:
    """Whether a field contains the whole command/fact prompt, not just relief.

    Extractors intentionally prefer recall over precision for free-text
    fields. In an authority-facing request section, though, pasting the
    entire user prompt repeats facts already pleaded elsewhere and can leave
    drafting commands like "draft karo" inside the document.
    """
    lowered = (value or "").strip().lower()
    if not lowered:
        return False
    facts = (fields.get("facts") or "").strip().lower()
    if facts and len(facts) > 30 and facts in lowered:
        return True
    command_markers = (
        "draft", "banao", "banana", "banaiye", "likho", "prepare", "तैयार", "बनाओ", "लिखो",
    )
    fact_markers = (
        "ko", "near", "paas", "mein", "me", "date", "imei", "₹", "rs.", "दिनांक", "को", "में", "पास",
    )
    return (
        len(lowered.split()) >= 18
        and any(marker in lowered for marker in command_markers)
        and any(marker in lowered for marker in fact_markers)
    )


class _RequireNonBlank(dict[str, Any]):
    """`str.format_map` helper: raises `KeyError` for a missing key exactly
    like a plain dict would, used by `LegalDraftEngine._subject_line`."""

    def __missing__(self, key: str) -> str:
        raise KeyError(key)


# Part 56 "Advocate-Style Draft Redesign": category-conditional instructions
# interpolated into `legal_drafting_prompt.md`'s `{narrative_guidance}` slot,
# telling the LLM how to map the spec's 7-part narrative arc (background /
# chronological sequence / detailed description / consequences faced by the
# applicant / legal implications / need for legal intervention / relief) onto
# each category's own section skeleton (`templates/base.py`). "Consequences"
# here means harm/impact suffered BY THE APPLICANT as a direct result of the
# matter -- not a threat of what will happen to the recipient if they don't
# comply (that belongs in "Prayer", which conventionally closes with exactly
# that line in Indian legal drafting).
_NARRATIVE_ARC = (
    "Narrative structure to follow: expand the given facts across the required sections using this arc -- "
    "background of the matter, the chronological sequence of events, and a detailed description of what "
    "happened (all within \"Facts of the Case\", written as 2-4 separate paragraphs); the consequences and "
    "impact the applicant has personally suffered as a direct result (\"Consequences\", 1 paragraph -- this is "
    "about harm to the applicant, not a threat to the recipient); the legal implications, stated in prima "
    "facie/conditional terms and citing the applicable Act(s)/section(s) given above only where they genuinely "
    "apply to these facts (\"Legal Position\", 1-2 paragraphs); and the specific relief sought (\"Prayer\", 1 "
    "paragraph). \"Introduction\" (exactly 1 paragraph) briefly identifies the applicant and states, in "
    "general terms, why this document has become necessary."
)
# Part 58 "Answer Quality Audit" issue 7: the arc above used to end "...
# including -- where appropriate for this document type -- what the applicant
# will do if that relief is not granted within a reasonable period". Applied
# uniformly, that instruction put an escalation threat into documents where
# the user had asked for no such thing: a police complaint whose stated relief
# was simply "register my complaint and investigate" came back closing "if no
# action is taken within a reasonable period, I shall be compelled to approach
# higher authorities". That is a real, adversarial commitment made on the
# complainant's behalf, in a document they will sign, that they never agreed
# to -- and it can sour the very relationship with the station they need. A
# NOTICE is the one document type whose entire purpose is to state a
# consequence for non-compliance, so it keeps the clause; every other category
# states only the relief actually asked for.
_COMPLAINT_NARRATIVE_GUIDANCE = _NARRATIVE_ARC + (
    " The \"Prayer\" states ONLY the relief the applicant actually asked for. Do not add a deadline, a "
    "warning, or a statement that the applicant will escalate to higher authorities, a court, or any other "
    "forum if the relief is not granted -- unless the applicant's own stated relief says exactly that."
)
_NOTICE_NARRATIVE_GUIDANCE = _NARRATIVE_ARC + (
    " A notice conventionally closes by stating the consequence of non-compliance, so the \"Prayer\" may "
    "include what the sender will do if the demand is not met within the stated period -- but only the "
    "consequence and period the sender actually asked for, never an invented one."
)
_AFFIDAVIT_NARRATIVE_GUIDANCE = (
    "Narrative structure to follow: elaborate the numbered \"Statements\" section into roughly 6-10 short "
    "numbered paragraphs (each beginning \"That...\"), following the same arc as a fuller narrative would -- "
    "background, chronological sequence of events, detailed description, and consequences to the deponent -- "
    "still ending, as always, with the standard verification clause. Do not add a statement about any fact not "
    "given above."
    " \"Deponent Details\" must open with the real Indian affidavit convention, translated into the requested "
    "language: \"I, [name], son/daughter/wife of [father's/husband's name], aged about [age] years, resident "
    "of [address], do hereby solemnly affirm and declare as under:\" -- using only the fields actually "
    "supplied (omit a clause, e.g. age, that was not given, rather than inventing a placeholder value)."
    " \"Verification\" must use the real Indian affidavit verification formula, translated into the requested "
    "language, not an invented paraphrase: \"I, the deponent above named, do hereby verify that the contents "
    "of paragraphs 1 to [N] of this affidavit are true and correct to the best of my knowledge and belief and "
    "that nothing material has been concealed therefrom and no part of it is false.\" -- with [N] replaced by "
    "the actual number of statement paragraphs produced."
)
_APPLICATION_NARRATIVE_GUIDANCE = (
    "Narrative structure to follow: open the \"Request\" section with one framing paragraph explaining the "
    "context and the right being exercised, elaborated with any background the applicant provided, followed by "
    "the information sought written out as a clearly numbered list. Never invent an information category the "
    "applicant did not ask about."
)
# Part 57 "Drafting Lifecycle Redesign": Contract-category documents (NDA/
# MOU/Partnership/Service Agreement/Rent Agreement/Property Sale-Purchase
# Agreement) follow a clause-based arc, not the grievance/petition arc every
# other category uses.
_CONTRACT_NARRATIVE_GUIDANCE = (
    "Narrative structure to follow: \"Recitals\" (1-2 short paragraphs, conventionally starting \"WHEREAS\") "
    "states the background and purpose the parties are entering into this agreement for, using only what is "
    "given above. \"Terms and Conditions\" elaborates the substantive obligations of each party as a clearly "
    "numbered list, grounded strictly in the facts/terms provided -- never invent a term, obligation, price, "
    "or condition not given. \"Term and Termination\" states the agreement's duration and how either party may "
    "end it, using only the duration/termination information given (if none was given, state that the "
    "agreement continues until terminated by mutual written consent). \"Governing Law and Jurisdiction\" names "
    "the applicable law/state given, or India generally if none was specified. Never invent a party's name, "
    "address, date, or monetary figure not provided."
)


def _narrative_guidance(category: str) -> str:
    if category == "Affidavit":
        return _AFFIDAVIT_NARRATIVE_GUIDANCE
    if category == "Application":
        return _APPLICATION_NARRATIVE_GUIDANCE
    if category == "Contract":
        return _CONTRACT_NARRATIVE_GUIDANCE
    if category == "Notice":
        return _NOTICE_NARRATIVE_GUIDANCE
    return _COMPLAINT_NARRATIVE_GUIDANCE


# Words that fit on one A4 body page under the fixed export typography
# (11pt type, 1.5 leading, 0.9in margins -- see `export._LAYOUT`). Used to
# translate the product requirement ("a draft is at least three pages") into
# the only quantity the drafting prompt can actually be steered by: a word
# target.
#
# This is SCRIPT-DEPENDENT and getting it wrong in either direction is a real
# defect. A Devanagari/Tamil word occupies far more horizontal space than an
# English one (longer average grapheme runs, plus matras and conjuncts that
# widen the glyph cluster), so the same word count fills roughly two-thirds
# fewer lines' worth of page. Measured against real WeasyPrint output: a
# 368-word Hindi police complaint rendered to two full body pages, where 368
# English words would not have filled one.
#
# Treating every language as Latin -- which the first version of this did --
# would have asked the model for ~1290 words of Hindi to satisfy a three-page
# minimum that ~800 words already meets, producing a bloated six-page police
# complaint. "Professional" is not a synonym for "long".
_WORDS_PER_BODY_PAGE_LATIN = 430
_WORDS_PER_BODY_PAGE_INDIC = 260

# Scripts whose glyph clusters are materially wider than Latin at the same
# point size. Detected from the TEXT rather than passed in as a language,
# because that keeps the estimate correct for a mixed-script draft (an Indic
# document quoting an English statute name, which is the common case).
_INDIC_RANGES = (
    (0x0900, 0x097F),  # Devanagari  - Hindi/Marathi/Sanskrit/Konkani/Nepali/Maithili/Dogri/Bodo
    (0x0980, 0x09FF),  # Bengali     - Bengali/Assamese
    (0x0A00, 0x0A7F),  # Gurmukhi    - Punjabi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0600, 0x06FF),  # Perso-Arabic - Urdu/Sindhi/Kashmiri
)
# Share of non-space characters that must be Indic before the draft is
# treated as an Indic-script document. Well above incidental use (a stray
# Devanagari word in an English draft) and well below a genuine Indic
# document, which runs far higher even with English proper nouns in it.
_INDIC_TEXT_SHARE = 0.30

# The product's floor. The exported packet's cover sheet and execution page
# are deliberately NOT counted toward it: padding a one-page document with
# two pieces of stationery is exactly the "quality has fallen" complaint this
# addresses, so the minimum applies to the SUBSTANTIVE BODY alone.
# Two substantive pages is the normal floor for notices and authority-facing
# complaints. The previous three-page floor turned a concise SHO application
# into four pages of generic scaffolding, unlike ordinary filing practice.
MINIMUM_BODY_PAGES = 2


# Same 9 templates as `advocate_register.FLOWING_LETTER_DRAFTS` -- kept as
# one alias here since this name is what the "Request" vs "Prayer" heading
# logic below has always used; not a second, independently-maintained set.
_REQUEST_HEADING_DRAFTS = advocate_register.FLOWING_LETTER_DRAFTS


def _is_indic_text(text: str) -> bool:
    meaningful = [char for char in text if not char.isspace()]
    if not meaningful:
        return False
    indic = sum(
        1 for char in meaningful if any(low <= ord(char) <= high for low, high in _INDIC_RANGES)
    )
    return (indic / len(meaningful)) >= _INDIC_TEXT_SHARE


def _words_per_body_page(text: str) -> int:
    return _WORDS_PER_BODY_PAGE_INDIC if _is_indic_text(text) else _WORDS_PER_BODY_PAGE_LATIN


def _target_word_count(fields: dict[str, str], language: str = "english") -> int:
    """The advocate-style *minimum* word target, with no upper ceiling.

    Previously returned a flat 900 for sparse facts, which renders as roughly
    two body pages -- below the three-page minimum this product promises. The
    floor is now whatever three body pages actually costs IN THIS DOCUMENT'S
    SCRIPT, and scales up from there with the volume of material the user
    actually supplied.

    Raising a target does NOT license padding: `legal_drafting_prompt.md`
    rule 8 explicitly instructs the model to return a shorter truthful draft
    rather than invent facts to reach a number, and `fact_audit.py` flags any
    sentence that went beyond the supplied facts. Length here buys fuller
    formal treatment of the SAME facts -- proper recitals, numbered pleading
    paragraphs, an evidence discussion, a carefully qualified legal position
    -- which is what makes a document read as an advocate's work rather than
    a filled-in form.
    """
    # The script of the document being WRITTEN, which is the draft's language
    # -- not the script the user happened to type their facts in. Someone can
    # (and in the reported incident, did) supply Hinglish/English facts for a
    # Hindi draft.
    per_page = _WORDS_PER_BODY_PAGE_INDIC if _language_uses_indic_script(language) else _WORDS_PER_BODY_PAGE_LATIN
    input_word_count = sum(len(value.split()) for value in fields.values() if value and value.strip())
    return max(per_page * MINIMUM_BODY_PAGES, round(input_word_count * 3.0))


# Languages this app drafts in whose script is one of `_INDIC_RANGES`.
# Everything else (english/hinglish, and any unrecognized value) is treated
# as Latin-width.
_INDIC_LANGUAGES = frozenset({
    "hindi", "marathi", "sanskrit", "konkani", "nepali", "maithili", "dogri", "bodo",
    "bengali", "assamese", "punjabi", "gujarati", "odia", "tamil", "telugu", "kannada",
    "malayalam", "urdu", "sindhi", "kashmiri",
})


def _language_uses_indic_script(language: str) -> bool:
    return (language or "").strip().lower() in _INDIC_LANGUAGES


def _word_count(sections: dict[str, str]) -> int:
    return sum(len(text.split()) for text in sections.values())


def estimated_page_count(sections: dict[str, str]) -> int:
    """Approximate rendered BODY page count for `sections`.

    An estimate on purpose: the exact figure needs a full WeasyPrint render
    (`PdfDraftExporter._page_count`), which is far too slow to run inside
    every generation turn. This is accurate enough to decide "does this need
    another expansion pass" and to report progress to the caller.
    """
    text = "\n".join(sections.values())
    return max(1, -(-_word_count(sections) // _words_per_body_page(text)))


class LegalDraftEngine:
    """Independent legal draft generation engine.

    Mirrors `ChatService`'s shape (self-instantiated collaborators, no DI
    container) but is fully decoupled from the chat flow — the chatbot (or
    any other caller) invokes this engine directly when it needs to produce
    a structured legal document.
    """

    def __init__(self) -> None:
        # Retry + provider failover (see `app/llm/resilient.py`). Draft
        # generation is the one place where a transient provider blip used
        # to silently downgrade the product to a one-page skeleton the user
        # might sign and file, so it does not get the bare `create()`.
        self.llm = LLMFactory.create_resilient()
        self.retriever = LegalRetriever()
        self.prompt_scanner = PromptInjectionScanner()
        self.safety_guard = DraftSafetyGuard()
        self.drafts = DraftRepository()
        self.versions = DraftVersionRepository()
        self.glossary = LegalTermsLibrary()
        self.exporters: dict[ExportFormat, DraftExporter] = {
            "pdf": PdfDraftExporter(),
            "docx": DocxDraftExporter(),
            "txt": TxtDraftExporter(),
            "rtf": RtfDraftExporter(),
        }

    def list_templates(self) -> list[DraftTemplateSummary]:
        return [
            DraftTemplateSummary(
                draft_id=template.draft_id,
                name=template.name,
                hindi_name=template.hindi_name,
                category=template.category,
                description=template.description,
                document_family=template.document_family,
                domain=template.domain,
                subcategory=template.subcategory,
                template_status=template.template_status,
            )
            for template in _list_template_definitions()
        ]

    def get_template_detail(self, draft_id: str) -> DraftTemplateDetail:
        template = self._require_template(draft_id)
        return DraftTemplateDetail(
            draft_id=template.draft_id,
            name=template.name,
            hindi_name=template.hindi_name,
            category=template.category,
            description=template.description,
            applicable_acts_hint=template.applicable_acts_hint,
            applicable_sections_hint=template.applicable_sections_hint,
            required_fields=[_field_schema(field, required=True) for field in template.required_fields],
            optional_fields=[_field_schema(field, required=False) for field in template.optional_fields],
        )

    async def preview(self, request: DraftPreviewRequest) -> DraftGenerateResponse:
        return await self._build(request, persist=False)

    async def generate(self, request: DraftGenerateRequest) -> DraftGenerateResponse:
        return await self._build(request, persist=True)

    async def regenerate(
        self, draft_id: str, fields: dict[str, str], language: str | None = None,
        style_instruction: str | None = None,
    ) -> DraftGenerateResponse:
        """Re-renders an existing draft with patched fields and/or a new
        language, for in-chat edits.

        Updates the draft document in place (so the same `draft_id` keeps
        working for translate/export throughout the conversation) while
        recording a proper version-chain entry in `DRAFT_VERSIONS` — the same
        old_version_id/new_version_id/document_status pattern
        `app/repositories/versioning.py` uses for document versions.

        Part 52: also how a mid-conversation language change ("Tamil me
        karo") is applied — passing `language` re-renders the SAME collected
        field values through the normal LLM drafting prompt in the new
        language and persists the result, rather than machine-translating
        already-generated text. That keeps the output a single
        section-structured draft (so PDF/DOCX export, which reads `sections`
        straight from this same persisted record, can never diverge from
        what was just shown in chat) instead of an unstructured blob.

        `style_instruction` ("restyle" edit action, QA pass 2026-09-11) is
        the same mechanism for a tone/register change ("polite but firm",
        "more formal") -- SAME facts, re-rendered with a different voice via
        an added prompt directive (`_render_sections_within_deadline`), not
        a new field and never written into the document as text of its own.

        BUG-015: two overlapping calls to this method (e.g. two rapid edit
        messages in the same session, or two browser tabs) each used to read
        a pre-edit snapshot and write their own full merged snapshot back
        unconditionally -- whichever committed last won in its entirety,
        silently discarding any field the other call had changed, while BOTH
        callers were told "Updated...". Fixed with an optimistic-concurrency
        compare-and-swap (`DraftRepository.compare_and_swap`, keyed on a
        `version` counter): if this call's write loses the race, it re-reads
        the now-current document, re-merges its OWN requested `fields` onto
        THAT fresh base (never onto its own stale read), and retries once.
        Two callers changing DIFFERENT fields therefore both still land
        correctly; only a genuine repeated collision (a third overlapping
        writer landing inside the retry's own window too) raises
        `DraftConflictError` rather than ever reporting a false success.
        """
        started = time.perf_counter()
        draft = await self._require_draft(draft_id)
        for remaining_attempts in (1, 0):
            expected_version = draft.get("version") or 1
            merged_fields, resolved_language, template, sections, rendered = await self._render_regenerated_content(
                draft_id, fields, language, style_instruction, base_draft=draft
            )
            applied = await self._try_apply_regeneration(
                draft_id, expected_version, merged_fields, sections, resolved_language
            )
            if applied:
                break
            if remaining_attempts == 0:
                raise DraftConflictError(
                    f"Draft {draft_id} was changed by another request while this edit was being applied. "
                    "Please retry.",
                    {"draft_id": draft_id},
                )
            # Lost the race -- re-read the document the OTHER writer just
            # committed, so the retry re-merges this call's own requested
            # `fields` onto that fresh base (never onto this attempt's now
            # -stale read), then falls through to try again exactly once.
            draft = await self._require_draft(draft_id)
        await self._append_version(draft_id, sections, resolved_language, merged_fields)

        return DraftGenerateResponse(
            status="complete",
            draft_id=draft_id,
            template_id=template.draft_id,
            template_name=template.name,
            sections=sections,
            full_text=self._render_full_text(sections, resolved_language, template),
            applicable_acts=template.applicable_acts_hint,
            applicable_sections=template.applicable_sections_hint,
            disclaimer=msg("draft_disclaimer", resolved_language, DRAFT_DISCLAIMER),
            language=resolved_language,
            generated_by_llm=rendered.generated_by_llm,
            generation_mode=rendered.generation_mode,
            generation_error=rendered.generation_error,
            word_count=rendered.word_count,
            estimated_page_count=rendered.page_count,
            latency_ms=(time.perf_counter() - started) * 1000,
            audit_findings=self._audit_findings(template, sections, merged_fields),
        )

    async def _render_regenerated_content(
        self,
        draft_id: str,
        fields: dict[str, str],
        language: str | None,
        style_instruction: str | None,
        base_draft: dict[str, Any] | None = None,
    ) -> tuple[dict[str, str], str, DraftTemplateDefinition, dict[str, str], "_RenderedDraft"]:
        """Shared by both attempts in `regenerate`'s retry loop: reads the
        current draft (or reuses `base_draft` if the caller already has a
        fresh one, to avoid a redundant read), merges the caller's requested
        `fields` onto THAT base, and renders. Kept separate from `regenerate`
        itself so the retry attempt can re-run exactly this against a freshly
        re-read document without duplicating the merge/render logic.
        """
        draft = base_draft if base_draft is not None else await self._require_draft(draft_id)
        current_state = _lifecycle_state_of(draft)
        if current_state not in _EDITABLE_STATES:
            raise DraftLockedError(
                f"Draft {draft_id} cannot be edited in state '{current_state}' -- unlock it first.",
                {"lifecycle_state": current_state},
            )
        template = self._require_template(draft["draft_type"])
        merged_fields = {**draft.get("fields", {}), **fields}
        resolved_language = language or draft.get("language", "english")
        request = DraftGenerateRequest(
            draft_id=template.draft_id,
            language=resolved_language,
            fields=merged_fields,
            session_id=draft.get("session_id"),
            user_id=draft.get("user_id"),
            style_instruction=style_instruction,
        )
        rendered = await self._render_sections(template, request)
        return merged_fields, resolved_language, template, rendered.sections, rendered

    async def _try_apply_regeneration(
        self, draft_id: str, expected_version: int, merged_fields: dict[str, str], sections: dict[str, str],
        resolved_language: str,
    ) -> bool:
        return await self.drafts.compare_and_swap(
            draft_id,
            expected_version,
            {
                "fields": merged_fields,
                "sections": sections,
                "status": "complete",
                "language": resolved_language,
                "lifecycle_state": "preview_ready",
            },
        )

    async def translate(self, request: DraftTranslateRequest) -> DraftTranslateResponse:
        # A `draft_id`-tied translation is a real language change to the
        # canonical document, not a scratch text conversion -- delegate to
        # `regenerate`, the same mechanism a mid-conversation "Tamil me karo"
        # edit already uses (see that method's own Part 52 docstring), so the
        # persisted `sections` -- and therefore any later PDF/DOCX export of
        # this SAME draft_id -- can never diverge from the translated text
        # returned here. Confirmed live (qa-40q-multilingual-20260921
        # BUG-08): this branch used to run a standalone LLM pass over the
        # rendered text and return it without writing anything back, while
        # `app/chatops/workflows/drafts.py`'s "translate" action showed that
        # text AND a PDF-download button for the same draft_id in the same
        # reply -- the PDF was the untranslated original.
        if request.draft_id:
            generated = await self.regenerate(request.draft_id, fields={}, language=request.target_language)
            return DraftTranslateResponse(translated_text=generated.full_text, target_language=request.target_language)
        text = request.text
        if not text or not text.strip():
            raise BadRequestError("Provide either draft_id or text to translate.")
        prompt = prompt_registry.render(
            "translation_prompt",
            source_language=request.source_language,
            target_language=request.target_language,
            text=text,
        )
        llm_response = await self.llm.chat([ChatMessage(role="user", content=prompt)])
        translated = llm_response.content.strip() or text
        return DraftTranslateResponse(translated_text=translated, target_language=request.target_language)

    async def get_current(self, draft_id: str) -> DraftGenerateResponse:
        """Re-renders the persisted draft's current sections without calling
        the LLM -- used by preview-stage "continue"/confirmation turns so
        they can show the actual draft text instead of a placeholder."""
        draft = await self._require_draft(draft_id)
        sections = draft["sections"]
        language = draft.get("language", "english")
        return DraftGenerateResponse(
            status="complete",
            draft_id=draft_id,
            template_id=draft["draft_type"],
            template_name=draft["template_name"],
            sections=sections,
            full_text=self._render_full_text(sections, language, self._require_template(draft["draft_type"])),
            language=language,
            word_count=_word_count(sections),
            estimated_page_count=estimated_page_count(sections),
        )

    async def legal_terms(self, query: str | None) -> list[LegalTermEntry]:
        return await self.glossary.lookup(query)

    async def history(self, request: DraftHistoryRequest) -> list[DraftSummary]:
        if request.session_id:
            records = await self.drafts.list_for_session(request.session_id, request.limit)
        elif request.user_id:
            records = await self.drafts.list_for_user(request.user_id, request.limit)
        else:
            raise BadRequestError("Provide session_id or user_id to fetch draft history.")
        return [
            DraftSummary(
                draft_id=str(record["_id"]),
                template_id=record["draft_type"],
                template_name=record["template_name"],
                language=record["language"],
                status=_lifecycle_state_of(record),
                created_at=_created_at_isoformat(record["created_at"]),
            )
            for record in records
        ]

    async def export(self, draft_id: str, fmt: ExportFormat, options: ExportOptions | None = None) -> Path:
        draft = await self._require_draft(draft_id)
        current_state = _lifecycle_state_of(draft)
        # A draft is downloadable as soon as it exists, from any lifecycle
        # state (see `_EXPORTABLE_STATES`). The approve/lock confirmation
        # gate this used to enforce was removed by product decision; the
        # AI-generated-draft disclaimer that every export already carries is
        # what communicates "get this reviewed by an advocate", not a
        # click-through the user had to guess the magic word for.
        if current_state not in _EXPORTABLE_STATES:
            raise DraftLockedError(
                f"Draft {draft_id} cannot be exported from state '{current_state}'.",
                {"lifecycle_state": current_state},
            )
        if options and options.sign and fmt != "pdf":
            raise BadRequestError("Cryptographic digital signing is currently supported for PDF exports only.")
        template = self._require_template(draft["draft_type"])
        # Phase 1 item 6: the pre-export audit. Export is the last moment
        # before a draft leaves this system as a file someone signs and files,
        # so the check runs here as well as at generation -- a draft edited
        # field-by-field since it was generated is re-audited against its
        # CURRENT text. Advisory: it logs what it found and never blocks the
        # download, because refusing someone their document over a heuristic
        # is a worse outcome than an unflagged sentence.
        pre_export_findings = self._audit_findings(template, draft["sections"], draft.get("fields", {}))
        if pre_export_findings:
            log.info(
                "draft_pre_export_audit",
                draft_id=draft_id,
                draft_type=template.draft_id,
                export_format=fmt,
                finding_count=len(pre_export_findings),
                categories=sorted({finding["category"] for finding in pre_export_findings}),
            )
        blocking_findings = [
            finding
            for finding in pre_export_findings
            if str(finding.get("category", "")).startswith(("unsupported_conclusion", "unsupported_statement:"))
        ]
        if blocking_findings:
            raise BadRequestError(
                "Export blocked because the draft contains statements that are not supported by the supplied facts. "
                "Edit or confirm those facts before downloading.",
                {"audit_findings": blocking_findings},
            )
        exporter = self.exporters[fmt]
        settings.draft_output_dir.mkdir(parents=True, exist_ok=True)
        suffix = {"pdf": ".pdf", "docx": ".docx", "txt": ".txt", "rtf": ".rtf"}[fmt]
        output_path = settings.draft_output_dir / f"{uuid4()}{suffix}"
        language = draft.get("language", "english")
        try:
            latest_version = await self.versions.latest_for_draft(draft_id)
        except RuntimeError as exc:
            # Repository-isolated callers/tests can supply a complete draft
            # without connecting MongoDB. Export content never depended on a
            # version lookup before Phase 2, so preserve that compatibility
            # and use version 1 when only the metadata lookup is unavailable.
            if "MongoDB client is not connected" not in str(exc):
                raise
            latest_version = None
        base_options = options or ExportOptions()
        options = replace(
            base_options,
            document_version=(latest_version or {}).get("version_number", 1),
            generated_on=datetime.now(UTC).strftime("%d %B %Y"),
            # Notice grammars mark their narrative/demand blocks as semantic
            # sections with hidden headings. Chat preview already honours
            # that AST policy; exports must do the same instead of reviving
            # internal labels such as "Introduction", "Consequences" and
            # "Prayer". Consumer/court complaints remain headed pleadings.
            flowing_letter=(
                template.category == "Notice"
                or template.draft_id in advocate_register.FLOWING_LETTER_DRAFTS
            ),
        )
        # Part 52/53: exported headings and title must match the language the
        # draft was last (re)generated in -- reading straight from the
        # persisted record keeps this in sync with whatever `regenerate()`
        # last wrote. The title is localized here (export time) rather than
        # on `draft["template_name"]` itself, since that field also serves as
        # the canonical English name used in draft history/chat wrapper text.
        grammar = document_schema_registry.for_template(template)
        grammar_context = {
            "fields": draft.get("fields", {}),
            "attachments": draft.get("fields", {}).get("available_documents", ""),
            "family": grammar.document_family,
            "forum_type": grammar.forum_type,
        }
        structural_issues = DocumentGrammarValidator().validate(grammar, draft["sections"], grammar_context)
        structural_errors = [issue for issue in structural_issues if issue.severity == "error"]
        if structural_errors:
            raise BadRequestError(
                "The draft does not satisfy its document grammar and cannot be exported.",
                {"validation_issues": [issue.as_dict() for issue in structural_errors]},
            )
        ast = document_schema_registry.compile_ast(
            grammar, draft["sections"], language=language, context=grammar_context
        )
        result_path = exporter.export(
            localized_title(template, language), ast.to_legacy_sections(), output_path, language, options
        )
        # Only the locked->exported transition is recorded. Exporting from an
        # editable state deliberately leaves `lifecycle_state` alone, so
        # downloading a draft never freezes it -- the user can keep editing
        # and download again.
        if current_state == "locked":
            await self._transition(draft_id, "exported")
        return result_path

    async def approve(self, draft_id: str) -> DraftLifecycleState:
        draft = await self._require_draft(draft_id)
        current_state = _lifecycle_state_of(draft)
        if current_state not in _EDITABLE_STATES:
            raise DraftLockedError(
                f"Draft {draft_id} cannot be approved from state '{current_state}'.", {"lifecycle_state": current_state}
            )
        await self._transition(draft_id, "approved")
        return "approved"

    async def lock(self, draft_id: str) -> DraftLifecycleState:
        draft = await self._require_draft(draft_id)
        current_state = _lifecycle_state_of(draft)
        if current_state != "approved":
            raise DraftLockedError(
                f"Draft {draft_id} must be approved before it can be locked (current state: '{current_state}').",
                {"lifecycle_state": current_state},
            )
        await self._transition(draft_id, "locked")
        return "locked"

    async def unlock(self, draft_id: str) -> DraftLifecycleState:
        """Explicit user action required by the spec ("unlocking must
        require an explicit user action") -- the chat layer only calls this
        after its own separate Y/N confirmation turn, same pattern as
        `lock()`'s own approve-then-confirm flow.
        """
        draft = await self._require_draft(draft_id)
        current_state = _lifecycle_state_of(draft)
        if current_state not in _LOCKABLE_STATES:
            raise DraftLockedError(
                f"Draft {draft_id} is already editable (state: '{current_state}').", {"lifecycle_state": current_state}
            )
        await self._transition(draft_id, "preview_ready")
        return "preview_ready"

    async def rollback(self, draft_id: str, version_number: int) -> DraftGenerateResponse:
        draft = await self._require_draft(draft_id)
        current_state = _lifecycle_state_of(draft)
        if current_state not in _EDITABLE_STATES:
            raise DraftLockedError(
                f"Draft {draft_id} must be unlocked before rolling back (current state: '{current_state}').",
                {"lifecycle_state": current_state},
            )
        version = await self.versions.get_version(draft_id, version_number)
        if version is None:
            raise NotFoundError(f"Draft {draft_id} has no version {version_number}.")
        sections = version.get("sections") or {}
        if not sections:
            raise BadRequestError(
                f"Version {version_number} of draft {draft_id} was recorded before content snapshots were "
                "stored on versions and can't be rolled back to."
            )
        language = version.get("language") or draft.get("language", "english")
        fields = version.get("fields") or draft.get("fields", {})
        template = self._require_template(draft["draft_type"])
        await self.drafts.update_by_id(
            draft_id, {"sections": sections, "language": language, "fields": fields, "lifecycle_state": "preview_ready"}
        )
        await self._append_version(draft_id, sections, language, fields, note=f"Rolled back to version {version_number}")
        return DraftGenerateResponse(
            status="complete",
            draft_id=draft_id,
            template_id=template.draft_id,
            template_name=template.name,
            sections=sections,
            full_text=self._render_full_text(sections, language, template),
            applicable_acts=template.applicable_acts_hint,
            applicable_sections=template.applicable_sections_hint,
            disclaimer=msg("draft_disclaimer", language, DRAFT_DISCLAIMER),
            language=language,
            generated_by_llm=False,
            word_count=_word_count(sections),
            estimated_page_count=estimated_page_count(sections),
            latency_ms=0.0,
        )

    async def list_versions(self, draft_id: str) -> list[dict[str, Any]]:
        await self._require_draft(draft_id)
        return await self.versions.list_for_draft(draft_id)

    async def _require_draft(self, draft_id: str) -> dict[str, Any]:
        draft = await self.drafts.find_by_id(draft_id)
        if draft is None:
            raise NotFoundError(f"Draft not found: {draft_id}")
        return draft

    async def _transition(self, draft_id: str, new_state: str) -> None:
        draft = await self.drafts.find_by_id(draft_id)
        history = list((draft or {}).get("lifecycle_history") or [])
        history.append({"state": new_state, "at": datetime.now(UTC).isoformat()})
        await self.drafts.update_by_id(draft_id, {"lifecycle_state": new_state, "lifecycle_history": history})

    async def _append_version(
        self, draft_id: str, sections: dict[str, str], language: str, fields: dict[str, str], *, note: str = ""
    ) -> None:
        """Appends a new entry to the draft's content-version chain --
        shared by initial generation, every in-chat edit/translate/
        regenerate, and rollback, so `draft_versions` always carries the
        actual rendered `sections` (not just language/status metadata, which
        is all it stored before this change and which made a real content
        rollback impossible).
        """
        previous_version = await self.versions.latest_for_draft(draft_id)
        new_version_number = (previous_version["version_number"] + 1) if previous_version else 1
        new_version_id = await self.versions.insert(
            {
                "draft_id": draft_id,
                "version_number": new_version_number,
                "document_status": "active",
                "language": language,
                "sections": sections,
                "fields": fields,
                "note": note,
                "old_version_id": previous_version["_id"] if previous_version else None,
                "new_version_id": None,
            }
        )
        if previous_version:
            await self.versions.update_by_id(
                previous_version["_id"], {"new_version_id": new_version_id, "document_status": "superseded"}
            )

    async def _build(self, request: DraftGenerateRequest, persist: bool) -> DraftGenerateResponse:
        started = time.perf_counter()
        template = self._require_template(request.draft_id)

        provided_keys = {key for key, value in request.fields.items() if value and value.strip()}
        missing = sorted(template.required_field_keys() - provided_keys)
        if missing:
            return DraftGenerateResponse(
                status="needs_more_info",
                template_id=template.draft_id,
                template_name=template.name,
                missing_fields=missing,
                follow_up_questions=[self._follow_up_question(template, key, request.language) for key in missing],
                language=request.language,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        rendered = await self._render_sections(template, request)
        sections = rendered.sections

        draft_id = None
        if persist:
            draft_id = str(uuid4())
            await self.drafts.insert(
                {
                    "_id": draft_id,
                    "draft_type": template.draft_id,
                    "template_name": template.name,
                    "language": request.language,
                    "session_id": request.session_id,
                    "user_id": request.user_id,
                    "fields": request.fields,
                    "sections": sections,
                    "status": "complete",
                    # Part 57: the draft starts life reviewable/editable, not
                    # locked -- "locked"/"exported" are only reached via an
                    # explicit approve()/lock() from here.
                    "lifecycle_state": "preview_ready",
                    "lifecycle_history": [{"state": "preview_ready", "at": datetime.now(UTC).isoformat()}],
                    # BUG-015: optimistic-concurrency token for `regenerate`'s
                    # compare-and-swap. `DraftRepository.compare_and_swap`
                    # treats a missing `version` (any draft persisted before
                    # this field existed) as equivalent to 1, so no backfill
                    # migration is needed.
                    "version": 1,
                }
            )
            # Always version 1 -- a freshly-generated draft has no prior
            # version to look up, so this inserts directly rather than
            # going through `_append_version`'s `latest_for_draft` lookup
            # (which exists for regenerate/rollback, where a previous
            # version genuinely might exist).
            await self.versions.insert(
                {
                    "draft_id": draft_id,
                    "version_number": 1,
                    "document_status": "active",
                    "language": request.language,
                    "sections": sections,
                    "fields": request.fields,
                    "old_version_id": None,
                    "new_version_id": None,
                }
            )

        return DraftGenerateResponse(
            status="complete",
            draft_id=draft_id,
            template_id=template.draft_id,
            template_name=template.name,
            sections=sections,
            full_text=self._render_full_text(sections, request.language, template),
            applicable_acts=template.applicable_acts_hint,
            applicable_sections=template.applicable_sections_hint,
            disclaimer=msg("draft_disclaimer", request.language, DRAFT_DISCLAIMER),
            language=request.language,
            generated_by_llm=rendered.generated_by_llm,
            generation_mode=rendered.generation_mode,
            generation_error=rendered.generation_error,
            word_count=rendered.word_count,
            estimated_page_count=rendered.page_count,
            latency_ms=(time.perf_counter() - started) * 1000,
            audit_findings=self._audit_findings(template, sections, request.fields),
        )

    async def _retrieve_legal_context(self, template: DraftTemplateDefinition) -> str:
        """The actual text of the Act(s)/section(s) `template` hints at, pulled
        from the same shared, review-gated Knowledge Base chat retrieval
        already uses -- so the drafting model has real statutory text to draw
        on instead of only the hint NAMES plus whatever it recalls on its
        own. This is the upstream fix for the failure `citation_audit.py`
        only catches downstream: a section number the model never actually
        saw grounded anywhere is a section number it had to guess.

        Best-effort and advisory by construction, like every other check in
        this module -- a retrieval failure, timeout, or empty result must
        never block drafting. It just means the model falls back to
        hints-only, exactly as it always did before this existed.
        """
        if not settings.draft_legal_context_enabled:
            return "None retrieved."
        query = ", ".join(template.applicable_acts_hint + template.applicable_sections_hint)
        if not query.strip():
            return "None retrieved."
        try:
            async with asyncio.timeout(settings.draft_legal_context_timeout_seconds):
                _, chunks = await self.retriever.retrieve(query, top_k=settings.draft_legal_context_top_k)
        except Exception as exc:  # noqa: BLE001 - advisory grounding must never block drafting
            log.warning("draft_legal_context_retrieval_failed", draft_type=template.draft_id, error=str(exc))
            return "None retrieved."
        if not chunks:
            return "None retrieved."
        pieces: list[str] = []
        budget = settings.draft_legal_context_max_chars
        for chunk in chunks:
            act_name = chunk.metadata.get("act_name")
            section_number = chunk.metadata.get("section_number")
            label = ", ".join(part for part in (act_name, f"Section {section_number}" if section_number else None) if part)
            piece = f"[{label}] {chunk.text}".strip() if label else chunk.text
            if budget <= 0:
                break
            piece = piece[:budget]
            pieces.append(piece)
            budget -= len(piece)
        return "\n\n".join(pieces) or "None retrieved."

    async def _render_sections(
        self, template: DraftTemplateDefinition, request: DraftGenerateRequest
    ) -> "_RenderedDraft":
        # This also protects callers of the standalone draft API, which do
        # not enter through ChatService's request-wide deadline. If called
        # from chat, `deadline()` keeps whichever outer/inner expiry is
        # tighter and never extends the request budget.
        with deadline(settings.draft_generation_budget_seconds, label="draft generation"):
            return await self._render_sections_within_deadline(template, request)

    async def _render_sections_within_deadline(
        self, template: DraftTemplateDefinition, request: DraftGenerateRequest
    ) -> "_RenderedDraft":
        combined_text = "\n".join(value for value in request.fields.values() if value)
        risky, findings = self.prompt_scanner.scan(combined_text)
        if risky:
            raise BadRequestError("The request contains unsafe prompt-injection instructions.", {"findings": findings})
        unsafe, unsafe_findings = self.safety_guard.scan(combined_text)
        if unsafe:
            raise BadRequestError(
                "This request cannot be processed: it appears to ask for forged, fabricated, or "
                "otherwise unlawful content, which this platform will never generate.",
                {"findings": unsafe_findings},
            )

        grammar = document_schema_registry.for_template(template)
        grammar_context = {
            "fields": request.fields,
            "attachments": request.fields.get("available_documents", ""),
            "family": grammar.document_family,
            "forum_type": grammar.forum_type,
        }
        structure_sections = grammar.generation_headings(grammar_context)
        retrieved_legal_context = await self._retrieve_legal_context(template)
        base_prompt = prompt_registry.render(
            "legal_drafting_prompt",
            draft_type=template.name,
            language=request.language,
            authority_label=template.authority_label,
            applicable_acts_hint=", ".join(template.applicable_acts_hint) or "None specified.",
            applicable_sections_hint=", ".join(template.applicable_sections_hint) or "None specified.",
            retrieved_legal_context=retrieved_legal_context,
            drafting_notes=template.drafting_notes or "None.",
            legal_enrichment_hints="; ".join(template.legal_enrichment_hints) or "None.",
            facts_and_fields=self._render_fields(template, request.fields, request.language),
            today_date=format_localized_date(clock.today(), request.language),
            section_headings="\n".join(f"## {heading}" for heading in structure_sections),
            target_word_count=_target_word_count(request.fields, request.language),
            narrative_guidance=_narrative_guidance(template.category),
        )
        if request.style_instruction:
            # "restyle" edit action: same facts, different voice. Appended
            # the same way `reask_prompt`/`retry_prompt` below already
            # layer an extra directive onto `base_prompt` rather than
            # threading a new template variable through the whole prompt
            # file -- and explicitly forbidden from adding content, exactly
            # like the expansion-pass directive further down already is,
            # so "make it firmer" can never become licence to insert a new
            # threat, deadline, or allegation the user didn't state.
            base_prompt = (
                f"{base_prompt}\n\nIMPORTANT -- tone/register instruction: rewrite this document so its tone is "
                f'"{request.style_instruction}". Do not change any fact, name, date, amount, deadline, or the '
                "relief sought, and do not add any new allegation, threat, or claim beyond what the facts above "
                "already support -- only the phrasing/register changes."
            )
        target_words = _target_word_count(request.fields, request.language)
        # Wall-clock deadline for this whole generation, expansion passes
        # included. See `settings.draft_generation_budget_seconds`: without
        # it, three sequential LLM calls (each with its own `llm_timeout` and
        # its own retries) could outlive the chat client's own timeout, and
        # the user got a transport error instead of the draft.
        generation_started = time.perf_counter()
        sections, generated_by_llm, llm_error = await self._generate_once(base_prompt, structure_sections)
        format_reasks = 0
        while (
            not generated_by_llm
            and llm_error == _UNPARSABLE_RESPONSE_ERROR
            and format_reasks < self._MAX_FORMAT_REASKS
            and (time.perf_counter() - generation_started) < settings.draft_generation_budget_seconds
        ):
            format_reasks += 1
            log.info(
                "draft_format_reask",
                draft_type=template.draft_id,
                language=request.language,
                attempt=format_reasks,
            )
            reask_prompt = (
                f"{base_prompt}\n\nIMPORTANT: your previous reply did not use the required section "
                "headings, so none of it could be used. Reply with the SAME document, but start every "
                "section with its heading on its own line prefixed by '## ', exactly as listed above, "
                "and write nothing outside those sections. Do not add, remove or reorder headings, and "
                "do not introduce any fact that is not in the details supplied."
            )
            sections, generated_by_llm, llm_error = await self._generate_once(reask_prompt, structure_sections)

        # Bounded expansion passes, only for a response that came back
        # genuinely short of the three-page body minimum. Long documents are
        # accepted as-is: there is no upper ceiling. We stop after
        # `_MAX_EXPANSION_PASSES` and keep whatever we have rather than
        # pressuring the model indefinitely, because repeated "make it
        # longer" demands are exactly what pushes a model into inventing
        # facts -- the one failure mode this engine must never have.
        if generated_by_llm:
            for expansion_pass in range(_MAX_EXPANSION_PASSES):
                current_words = _word_count(sections)
                if current_words >= target_words:
                    break
                elapsed = time.perf_counter() - generation_started
                if elapsed >= settings.draft_generation_budget_seconds:
                    # Out of budget. The first draft is already a complete,
                    # usable document -- shipping it slightly short beats
                    # spending another `llm_timeout` on length and having the
                    # caller's HTTP client disconnect before either arrives.
                    log.info(
                        "draft_expansion_skipped_over_budget",
                        draft_type=template.draft_id,
                        elapsed_seconds=round(elapsed, 1),
                        budget_seconds=settings.draft_generation_budget_seconds,
                        words=current_words,
                        target_words=target_words,
                    )
                    break
                direction = (
                    f"Your previous draft was {current_words} words, which renders as roughly "
                    f"{estimated_page_count(sections)} page(s). This document type requires a substantive body of "
                    f"at least {MINIMUM_BODY_PAGES} pages (about {target_words} words). Expand it by giving the "
                    f"SAME facts fuller formal legal treatment: separate numbered pleading paragraphs for each "
                    f"distinct event in the chronology; a properly developed recital of how the matter arose; an "
                    f"explicit discussion of each document or item of evidence the applicant said they hold, and "
                    f"what it establishes; a carefully hedged treatment of each Act/section that may apply and why; "
                    f"and a precisely framed prayer enumerating each relief separately. "
                    f"ABSOLUTE CONSTRAINT: do not invent or assume any event, injury, loss, amount, document, date, "
                    f"person, demand, or remedy that is not already in the facts above, and do not repeat sentences "
                    f"to add length. If the supplied facts genuinely cannot support this length, return the "
                    f"shorter truthful draft."
                )
                retry_prompt = f"{base_prompt}\n\nIMPORTANT: {direction}"
                retried_sections, retried_ok, retry_error = await self._generate_once(retry_prompt, structure_sections)
                if not retried_ok:
                    # An expansion pass that fails leaves the first, usable
                    # draft in place -- a shorter real document always beats
                    # discarding it over a failed follow-up call.
                    log.warning(
                        "draft_expansion_pass_failed",
                        draft_type=template.draft_id, expansion_pass=expansion_pass + 1, error=retry_error,
                    )
                    break
                if _word_count(retried_sections) <= _word_count(sections):
                    # No progress: stop rather than loop on a model that has
                    # said all it can truthfully say about these facts.
                    break
                sections = retried_sections

            # An otherwise valid provider reply can still stay below the
            # promised three-page body floor after the bounded expansion
            # passes (for example when the provider repeats its first short
            # answer).  The deterministic path already receives the formal,
            # fact-neutral category scaffolding below; use the same authored
            # material for a short LLM draft instead of shipping a polished
            # but two-page document.  This adds no user fact, amount, date,
            # loss or allegation, and remains a no-op for languages for which
            # no reviewed-language scaffolding exists.
            if estimated_page_count(sections) < MINIMUM_BODY_PAGES:
                apply_official_scaffolding(
                    sections,
                    category=template.category,
                    language=request.language,
                )

        if not sections:
            # Every LLM attempt (including `ResilientLLMProvider`'s retries
            # and any configured fallback providers) failed, or returned text
            # that could not be parsed into the required section structure.
            # The deterministic skeleton below is a genuine document, but it
            # is NOT an advocate-style draft -- so this is recorded and
            # propagated as `generation_mode="deterministic"` all the way to
            # the chat reply, which warns the user explicitly. Before this
            # change the degradation was invisible: a provider outage handed
            # someone a one-page stub captioned "here is your draft".
            log.warning(
                "draft_generated_without_llm",
                draft_type=template.draft_id,
                language=request.language,
                category=template.category,
                # What an operator needs to act on this: which provider chain
                # was in play, how long it burned, whether the failure was a
                # transport problem or an unusable reply, and how many format
                # re-asks were spent. None of it is derivable from the message
                # alone, and none of it carries a credential.
                provider=getattr(self.llm, "provider_name", "unknown"),
                format_reasks=format_reasks,
                elapsed_seconds=round(time.perf_counter() - generation_started, 2),
                budget_seconds=settings.draft_generation_budget_seconds,
                error=llm_error,
            )
            sections = self._deterministic_sections(template, request)
        if advocate_register.supports_advocate_register(request.language, template):
            enclosures_text = advocate_register.enclosures_block(request.fields, template, request.language)
            if enclosures_text and "Enclosures" not in sections:
                sections["Enclosures"] = enclosures_text
        else:
            available_documents = request.fields.get("available_documents", "").strip()
            if available_documents and "Annexures" not in sections:
                sections["Annexures"] = available_documents
        if template.category == "Affidavit":
            # Forced unconditionally, regardless of generation path: the
            # "Date" section is always the date this document is generated,
            # never an incident date the user mentioned or something the
            # LLM guessed at. Affidavit keeps its own separate "Place"/"Date"
            # headings -- a sworn statement's closing convention differs
            # from the letter-style closing block below.
            sections["Date"] = format_localized_date(clock.today(), request.language)
        elif template.category == "Contract":
            # Part 57: a contract has TWO signatories, not one -- the
            # single-applicant `_closing_block` below doesn't fit. Forced
            # centrally for the same reason "Date"/the letter closing block
            # are: guarantees the exact structure every time regardless of
            # generation path, and keeps chat preview/PDF/DOCX/TXT trivially
            # in sync since they all read this one persisted value.
            sections["Signatures"] = self._contract_signatures_block(request.fields, request.language)
        else:
            # Part 53 "Professional Layout Audit" / Part 56 "Advocate-Style
            # Draft Redesign": the ENTIRE closing block (place/date
            # inline-labeled, complementary close, signatory name, role
            # designator) is composed here, unconditionally overriding
            # whatever the LLM/deterministic path wrote for the
            # signature-block heading -- "Place" and "Date" are no longer
            # separate headings for Notice/Complaint/Application (see
            # `templates/base.py`), so this is the one place their values
            # actually get rendered. Forcing it centrally (the same
            # philosophy Part 51 already used for "Date" alone) guarantees
            # the exact structure/format every time, regardless of
            # generation path, and makes chat preview/PDF/DOCX/TXT parity
            # trivial since they all read this one persisted value. The
            # unified Notice/Complaint skeleton renamed this heading to
            # "Signature Block" (per the new spec); Application (RTI) keeps
            # its own existing "Signature" heading.
            signature_heading = "Signature" if template.category == "Application" else "Signature Block"
            if template.category == "Complaint":
                sections["Verification"] = self._complaint_verification_text(request.language)
            if advocate_register.supports_advocate_register(request.language, template):
                # The sworn identification opening ("I, X, W/o Y, aged Z,
                # residing at...") replaces whatever the LLM/deterministic
                # path wrote for the party-details heading -- guaranteed
                # centrally, the same way Verification/the closing block
                # already are, regardless of generation path.
                party_heading = "Applicant Details" if template.category == "Application" else "Complainant Details"
                sections[party_heading] = advocate_register.sworn_opening_block(request.fields, request.language)
            sections[signature_heading] = self._closing_block(request.fields, request.language, template.category)
        # Police/SHO representations conventionally make a request to the
        # authority; labelling that paragraph "Prayer" makes them look like a
        # court pleading. Court/commission complaints retain "Prayer".
        if template.draft_id in _REQUEST_HEADING_DRAFTS and "Prayer" in sections:
            sections.pop("Prayer")
            classical_intro = (
                advocate_register.classical_request_intro(
                    request.language, template.applicable_sections_hint, template.applicable_acts_hint
                )
                if advocate_register.supports_advocate_register(request.language, template)
                else None
            )
            request_text = self._police_request_block(
                template, request.fields, request.language, classical_intro=classical_intro
            )
            reordered: dict[str, str] = {}
            for heading, text in sections.items():
                if heading == "Verification":
                    reordered["Request"] = request_text
                reordered[heading] = text
            sections = reordered
        self._normalize_rendered_semantics(sections, template, request.fields, request.language)
        statutory_days = getattr(template, "statutory_response_period_days", None)
        sections = strip_invented_details(
            sections, request.fields, frozenset({str(statutory_days)}) if statutory_days else frozenset()
        )
        # Finding-007 (QA pass, 2026-09-11): guarantees any field flagged
        # `must_appear_verbatim` (e.g. `legal_notice`'s `claim_amount`) is
        # actually present in what gets shown/exported -- covers the
        # LLM-generated, format-reask, expansion-pass, AND deterministic-
        # fallback paths uniformly, since they all converge here before
        # returning. See `DraftField.must_appear_verbatim`'s docstring for
        # the live incident this closes.
        self._ensure_verbatim_fields(sections, template, request.fields)

        # Part 52 (Workflow Localization): the AI-generated-draft disclaimer
        # is deliberately NOT part of the document itself -- it must never
        # appear inside the chat preview, PDF, DOCX, or TXT output. It's
        # still surfaced to the caller via `DraftGenerateResponse.disclaimer`
        # (a separate field, unaffected by this) so the chat layer can show
        # it as a standalone aside outside the actual generated document.
        return _RenderedDraft(
            sections=sections,
            generated_by_llm=generated_by_llm,
            word_count=_word_count(sections),
            page_count=estimated_page_count(sections),
            generation_mode="llm" if generated_by_llm else "deterministic",
            generation_error=None if generated_by_llm else llm_error,
        )

    @staticmethod
    def _normalize_rendered_semantics(
        sections: dict[str, str],
        template: DraftTemplateDefinition,
        fields: dict[str, str],
        language: str,
    ) -> None:
        """Remove model wording that contradicts language or signer mode.

        These are document-level invariants, so they are enforced after both
        the LLM and deterministic paths converge.  This also keeps the chat
        preview and every exporter identical because the normalized sections
        are what get persisted.
        """
        if template.category == "Affidavit" and language == "hindi":
            replacements = {
                "solemnly affirm (सत्यनिष्ठा से प्रतिज्ञान)": "शपथपूर्वक सत्यनिष्ठा से प्रतिज्ञान",
                "solemnly affirm": "शपथपूर्वक सत्यनिष्ठा से प्रतिज्ञान",
            }
            for heading, text in sections.items():
                for source, target in replacements.items():
                    text = text.replace(source, target)
                sections[heading] = text

        representation_mode = fields.get("representation_mode", "").strip().casefold()
        advocate_mode = representation_mode in {"advocate", "lawyer", "through advocate"}
        if template.category != "Notice" or advocate_mode:
            return
        # An omitted representation_mode explicitly defaults to self mode in
        # the notice grammar.  Models sometimes hedge with both alternatives,
        # producing a document that no identifiable person can sign.
        self_mode_replacements = (
            ("अपने अधिवक्ता के माध्यम से (अथवा स्वयं, यदि लागू हो)", "स्वयं"),
            ("अपने अधिवक्ता के माध्यम से अथवा स्वयं", "स्वयं"),
            ("अपने अधिवक्ता के माध्यम से", "स्वयं"),
            ("through my advocate (or personally, if applicable)", "personally"),
            ("through an advocate (or personally, if applicable)", "personally"),
            ("through my advocate", "personally"),
        )
        for heading, text in sections.items():
            for source, target in self_mode_replacements:
                text = re.sub(re.escape(source), target, text, flags=re.IGNORECASE)
            sections[heading] = text

    # One extra attempt, and only for a response that ARRIVED but could not be
    # parsed into the required section headings. Transport failures (timeout,
    # 429, connection reset) are already retried and failed over inside
    # `ResilientLLMProvider`, which deliberately does not retry an unusable
    # success because it cannot know what "usable" means for a draft. That
    # judgement belongs here -- and until this pass, nothing here made it: a
    # model that answered without "## " headings went straight to the
    # deterministic skeleton on the first try. Bounded at one: a second
    # identical prompt that also came back unparsable is a prompt/model
    # problem, and spending a third `llm_timeout` on it costs the user the
    # latency their fallback document was supposed to save.
    _MAX_FORMAT_REASKS = 1

    @staticmethod
    def _redact(error: str | None) -> str | None:
        """Strip anything credential-shaped from a provider error before it is
        logged or returned.

        Provider SDKs put the request URL in their error strings, and a request
        URL can carry an API key in its query string (`?key=...`), which is how
        Google's client builds them. `generation_error` is logged, stored on
        the draft response and returned by `POST /draft/generate`, so an
        unredacted provider error is a credential-disclosure path.
        """
        if not error:
            return error
        redacted = re.sub(
            r"(?i)\b(api[_-]?key|key|token|secret|authorization|bearer)\b\s*[=:]\s*\S+",
            r"\1=[redacted]",
            error,
        )
        redacted = re.sub(r"(?i)([?&](?:key|api_key|access_token)=)[^&\s]+", r"\1[redacted]", redacted)
        # Long opaque strings are how the remaining key formats look.
        redacted = re.sub(r"\b(?:sk|gsk|AIza|hf)[A-Za-z0-9_\-]{16,}\b", "[redacted]", redacted)
        return redacted

    async def _generate_once(
        self, prompt: str, structure_sections: tuple[str, ...]
    ) -> tuple[dict[str, str], bool, str | None]:
        """One drafting call. Returns `(sections, ok, error)`.

        Every provider in this codebase signals failure by RETURNING an
        `LLMResponse` with `error` set and the error sentence in `content`
        (see `app/llm/base.py`) rather than by raising. That `error` field
        was previously never inspected here, so a provider outage arrived as
        content like "Gemini API is currently unavailable", parsed to zero
        sections, and quietly became a deterministic stub. It is checked
        first now, and the reason is returned so the caller can tell the user
        what actually happened.
        """
        try:
            llm_response = await call_with_hard_timeout(
                self.llm.chat([ChatMessage(role="user", content=prompt)]),
                fallback_seconds=settings.draft_generation_budget_seconds,
            )
        except TimeoutError as exc:
            if str(exc):
                # A real `TimeoutError` raised BY THE PROVIDER ITSELF (e.g. an
                # httpx read timeout), propagated unchanged through
                # `call_with_hard_timeout` -- `asyncio.wait_for` never
                # rewrites an inner exception, only ever raises its OWN
                # (always message-less, confirmed above) `TimeoutError` when
                # ITS timeout is what fired. Handled exactly like any other
                # provider exception below so the original, informative
                # message is preserved rather than replaced.
                redacted = self._redact(str(exc))
                log.warning("draft_llm_call_raised", error=redacted, error_type="TimeoutError")
                return {}, False, redacted
            # Backstop, not the primary bound (see `call_with_hard_timeout`'s
            # docstring): `self.llm` (a `ResilientLLMProvider`/`GeminiProvider`
            # chain) already computes and enforces its own timeout from the
            # SAME ambient deadline this draws on, and normally returns
            # (successfully or with `error` set) well before this fires. QA
            # (2026-09-11/12) reproduced that inner enforcement itself
            # stalling past its own computed timeout (317s / 250s+ observed
            # against a ~100s budget), leaving the caller with no bound at
            # all. This guarantees one regardless of why the inner call
            # didn't return on time.
            log.warning(
                "draft_llm_call_hard_timeout",
                budget_seconds=settings.draft_generation_budget_seconds,
            )
            return {}, False, "The drafting service did not respond within its time budget."
        except Exception as exc:  # noqa: BLE001 - never let a provider bug cost the user their document
            redacted = self._redact(str(exc))
            log.warning("draft_llm_call_raised", error=redacted, error_type=type(exc).__name__)
            return {}, False, redacted
        if llm_response.error:
            # Structured, actionable, and free of anything credential-shaped:
            # an operator reading this knows which provider failed and in what
            # category without the message itself having to carry a key.
            log.warning(
                "draft_llm_call_failed",
                provider=llm_response.provider,
                model=llm_response.model,
                error_kind=llm_response.error_kind,
                retry_count=llm_response.retry_count,
                error=self._redact(llm_response.error),
            )
            return {}, False, self._redact(llm_response.error)
        sections = self._parse_sections(llm_response.content, structure_sections)
        if not sections:
            log.warning(
                "draft_llm_response_unparsable",
                provider=llm_response.provider,
                model=llm_response.model,
                response_characters=len(llm_response.content or ""),
            )
            return {}, False, _UNPARSABLE_RESPONSE_ERROR
        return sections, True, None

    def _audit_findings(
        self, template: DraftTemplateDefinition, sections: dict[str, str], fields: dict[str, str]
    ) -> list[dict[str, str]]:
        """Phase 1 item 6: the fact-only audit, run on every generated draft.

        Advisory by design -- it never blocks generation or export. A user who
        needs a complaint is not helped by being refused one over a heuristic;
        they ARE helped by being told which sentences went beyond what they
        said, before they sign it. Failures here are swallowed for the same
        reason: an audit bug must never cost someone their document.
        """
        try:
            fact_findings = audit_draft(
                sections,
                fields,
                category=template.category,
                today_digits=clock.today().strftime("%d%m%Y"),
            ).as_dicts()
        except Exception as exc:  # noqa: BLE001 - an advisory check must never break drafting
            # Previously returned here, which also skipped the two checks
            # below -- one failing advisory check must not disable the other
            # two; each is independent and each already isolates its OWN
            # failure the same way.
            log.warning("draft_fact_audit_failed", draft_type=template.draft_id, error=str(exc))
            fact_findings = []
        try:
            # Post-Phase-3 hardening milestone C: the fact audit above asks
            # "does this sentence assert a fact the user did not supply?".
            # This one asks a narrower, sharper question -- "does this sentence
            # make one of the specific claims a draft must never make?" (silence
            # as admission, a criminal threat, unpleaded mental distress,
            # invented interest/costs, evidence that was never listed). Kept
            # separate because it fires on wording rather than on numbers or
            # names, and because it must also catch text the LLM path wrote.
            unsupported = prohibited_clauses.scan(
                sections, fields, permitted=tuple(template.permitted_statements)
            )
        except Exception as exc:  # noqa: BLE001 - same reasoning as above
            log.warning("draft_clause_scan_failed", draft_type=template.draft_id, error=str(exc))
            unsupported = []
        try:
            # Highest-stakes of the three checks: a wrong or invented section
            # number is exactly the kind of detail a lay user copies verbatim
            # into a document they file themselves. See `citation_audit.py`.
            unverified_citations = citation_audit.verify_citations(
                sections, template.applicable_sections_hint
            )
        except Exception as exc:  # noqa: BLE001 - same reasoning as above
            log.warning("draft_citation_audit_failed", draft_type=template.draft_id, error=str(exc))
            unverified_citations = []
        return [*fact_findings, *unsupported, *unverified_citations]

    def _require_template(self, draft_id: str) -> DraftTemplateDefinition:
        template = _get_template_definition(draft_id)
        if template is None:
            raise NotFoundError(f"Unknown draft template: {draft_id}")
        return template

    def _follow_up_question(self, template: DraftTemplateDefinition, key: str, language: str = "english") -> str:
        for draft_field in template.all_fields():
            if draft_field.key == key:
                if language in {
                    "tamil", "telugu", "kannada", "bengali",
                    "malayalam", "marathi", "gujarati", "punjabi", "odia", "urdu",
                }:
                    label = localized_field_label(template.draft_id, draft_field, language)
                elif language == "hindi":
                    label = f"{draft_field.label} ({localized_field_label(template.draft_id, draft_field, 'hindi')})"
                else:
                    label = draft_field.label
                return msg("follow_up_please_provide", language, "Please provide: {label}.", label=label)
        return f"Please provide a value for: {key}."

    def _render_fields(
        self, template: DraftTemplateDefinition, fields: dict[str, str], language: str = "english"
    ) -> str:
        """The user's collected values, labelled for the drafting prompt.

        Part 58 "Answer Quality Audit" issue 8: labels used to be English-only
        regardless of the document's language, which seeded the model with
        English vocabulary right where it was composing {language} prose --
        a Hindi police complaint came back with "Customer Service", "high
        returns", "transfer" and "Online Investment Scam" embedded in
        otherwise Devanagari sentences, because those were the words it had
        been handed. Labelling in the document's own language removes that
        pull; the English label is kept alongside it so the model can still
        tie a value back to the field the template defines (and so a
        language with no translation degrades to today's behaviour rather
        than to nothing).
        """
        lines = []
        for draft_field in template.all_fields():
            value = fields.get(draft_field.key, "").strip()
            if not value:
                continue
            localized = localized_field_label(template.draft_id, draft_field, language)
            label = draft_field.label if localized == draft_field.label else f"{localized} ({draft_field.label})"
            lines.append(f"{label}: {value}")
        return "\n".join(lines) if lines else "No additional facts provided."

    def _parse_sections(self, content: str, structure_sections: tuple[str, ...]) -> dict[str, str]:
        if not content or not content.strip():
            return {}
        allowed_headings = {*structure_sections, *OPTIONAL_SECTIONS}
        sections: dict[str, str] = {}
        current_heading: str | None = None
        buffer: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                if current_heading is not None:
                    sections[current_heading] = "\n".join(buffer).strip()
                heading = stripped[3:].strip()
                current_heading = heading if heading in allowed_headings else None
                buffer = []
            elif current_heading is not None:
                buffer.append(line)
        if current_heading is not None:
            sections[current_heading] = "\n".join(buffer).strip()

        # If the model didn't follow the required heading format closely
        # enough to recover at least half the expected sections, treat the
        # response as unusable and fall back to deterministic assembly.
        if len(sections) < len(structure_sections) // 2:
            return {}
        # Core sections keep their place in the skeleton even if the model
        # left one blank (a legal document missing "Place"/"Date"/"Signature"
        # is broken, not merely minimal) -- but "Annexures" and any other
        # section outside the required skeleton is dropped entirely when
        # empty, per Part 42: never render an empty section or a "None."
        # placeholder just to satisfy a fixed structure.
        result = {heading: sections.get(heading, "") for heading in structure_sections}
        for heading in OPTIONAL_SECTIONS:
            value = sections.get(heading, "").strip()
            if value and value.lower() not in {"none", "none.", "n/a", "not applicable."}:
                result[heading] = value
        return result

    def _deterministic_sections(self, template: DraftTemplateDefinition, request: DraftGenerateRequest) -> dict[str, str]:
        """Non-LLM fallback, used only when the LLM call fails or returns an
        unparsable response. Dispatches on `template.category` (Notice/
        Complaint/Affidavit/Application) rather than one generic shape, so
        even this fallback path produces a document that matches the
        structure a reader of that document type actually expects (Part 42).
        """
        builders = {
            "Notice": self._deterministic_notice_sections,
            "Complaint": self._deterministic_complaint_sections,
            "Affidavit": self._deterministic_affidavit_sections,
            "Application": self._deterministic_application_sections,
            "Contract": self._deterministic_contract_sections,
        }
        builder = builders.get(template.category, self._deterministic_notice_sections)
        return builder(template, request.fields, request.language)

    @staticmethod
    def _ensure_verbatim_fields(
        sections: dict[str, str], template: DraftTemplateDefinition, fields: dict[str, str]
    ) -> None:
        """Finding-007's fix. Mutates `sections` in place, appending one
        plain, fact-only sentence per `must_appear_verbatim` field whose
        value the generated text doesn't already state -- never inventing
        wording beyond quoting the field's own label and the user's own
        supplied value verbatim (no fabrication risk: nothing here is not
        already something the user typed).

        A no-op (checked first, cheap) for every template with no such
        field, i.e. every template except the ones explicitly opted in.
        """
        verbatim_fields = [f for f in template.all_fields() if f.must_appear_verbatim]
        if not verbatim_fields or not sections:
            return
        combined = " ".join(sections.values())
        combined_no_commas = combined.replace(",", "")
        target_section = "Facts of the Case" if "Facts of the Case" in sections else next(iter(sections))
        for draft_field in verbatim_fields:
            value = fields.get(draft_field.key, "").strip()
            if not value:
                continue
            # A comma-stripped comparison too -- "Rs 65000" vs "Rs. 65,000"
            # is the same figure typed two ways, and a model that DOES
            # restate the number sometimes drops the thousands separator.
            if value in combined or value.replace(",", "") in combined_no_commas:
                continue
            addition = f"{draft_field.label}: {value}."
            sections[target_section] = f"{sections[target_section]}\n\n{addition}".strip()
            combined += f" {addition}"
            combined_no_commas += f" {addition}".replace(",", "")

    def _subject_line(self, template: DraftTemplateDefinition, fields: dict[str, str], language: str = "english") -> str:
        subject = self._subject_line_text(template, fields, language)
        # Advocate-register redesign: a lawyer's Subject line names the exact
        # provision being invoked ("...under section 173 of BNSS read with
        # section 303 of BNS"); this app's was generic. Appends the citation
        # this template already carries (`applicable_sections_hint`) to
        # whichever subject text was actually produced above -- never
        # invents a citation a template doesn't have. `supports_citation` is
        # broader than `supports_advocate_register`: confirmed against real
        # notice formats (cheque-bounce under s.138 NI Act, money-recovery
        # notices) that a Notice's Subject line names its section too, even
        # though Notice gets none of the rest of this register (see
        # `advocate_register.py`'s module docstring).
        if advocate_register.supports_citation(language, template):
            subject = advocate_register.append_citation(
                subject, language, template.applicable_sections_hint, template.applicable_acts_hint
            )
        return subject

    def _subject_line_text(
        self, template: DraftTemplateDefinition, fields: dict[str, str], language: str = "english"
    ) -> str:
        if template.subject_template:
            # Only non-blank field values are exposed to `.format` -- a
            # referenced key that's missing or blank raises `KeyError` (via
            # `dict.__missing__`) exactly like a genuinely absent key would,
            # so a subject_template referencing an unfilled optional field
            # cleanly falls through to the generic subject below instead of
            # rendering "Demand for recovery of Rs. ".
            non_blank = {key: value for key, value in fields.items() if value and value.strip()}
            localized_template = localized_subject_template(template.draft_id, template.subject_template, language)
            try:
                return localized_template.format_map(_RequireNonBlank(non_blank))
            except KeyError:
                pass
        relief = fields.get("expected_relief", "").strip() or fields.get("information_sought", "").strip()
        if relief:
            return relief if len(relief) <= 140 else f"{relief[:137]}..."
        return template.description

    def _applicant_block(self, fields: dict[str, str], language: str = "english") -> str:
        return "\n".join(
            part
            for part in [
                fields.get("applicant_name", ""),
                fields.get("applicant_address", ""),
                f"{phrase('mobile_label', language, 'Mobile')}: {fields['applicant_mobile']}"
                if fields.get("applicant_mobile") else "",
                f"{phrase('email_label', language, 'Email')}: {fields['applicant_email']}"
                if fields.get("applicant_email") else "",
            ]
            if part
        ) or phrase("not_provided", language, "Not provided.")

    def _signature_block(self, fields: dict[str, str], language: str = "english", *, label: str | None = None) -> str:
        lines = [phrase("signed", language, "Sd/-")]
        if label:
            lines.append(label)
        lines.append(fields.get("applicant_name", "") or "___________________")
        if fields.get("applicant_mobile"):
            lines.append(f"{phrase('mobile_label', language, 'Mobile')}: {fields['applicant_mobile']}")
        return "\n".join(lines)

    def _closing_block(self, fields: dict[str, str], language: str, category: str) -> str:
        """The full closing block for Notice/Complaint/Application category
        documents: inline-labeled Place/Date, a complementary close, the
        signatory's name, and a role designator -- e.g.:

            Place: Lucknow
            Date: 13/08/2026

            Yours faithfully,

            (Ajay Kumar)
            Petitioner

        Part 56 "Advocate-Style Draft Redesign": both "Place" and "Date"
        lines are always rendered now, even when the applicant left "Place"
        blank -- a printed underscore placeholder ("Place: ______") for the
        applicant to fill in by hand, matching the spec's own signature-block
        template, rather than silently omitting the line as before.

        Always called from `_render_sections`, which overrides whatever the
        LLM/deterministic path produced for the signature-block heading with
        this -- see that method's comment for why.
        """
        place = fields.get("place", "").strip() or "______"
        lines = [
            f"{translated_heading('Place', language)}: {place}",
            f"{translated_heading('Date', language)}: {format_localized_date(clock.today(), language)}",
            "",
            phrase("closing_salutation", language, "Yours faithfully,"),
            "",
        ]
        representation_mode = fields.get("representation_mode", "").strip().casefold()
        advocate_mode = category == "Notice" and representation_mode in {"advocate", "lawyer", "through advocate"}
        name = (
            fields.get("advocate_name", "").strip()
            if advocate_mode
            else fields.get("applicant_name", "").strip()
        )
        lines.append(f"({name})" if name else "___________________")
        if advocate_mode:
            designator = phrase("advocate_for_sender", language, "Advocate for the Sender")
        else:
            designator = closing_designator(category, language)
        if designator:
            lines.append(designator)
        return "\n".join(lines)

    def _contract_signatures_block(self, fields: dict[str, str], language: str) -> str:
        """The closing block for a Contract-category document -- TWO
        signatories (Party of the First Part / Party of the Second Part),
        not the single applicant `_closing_block` handles. Always called
        from `_render_sections`, which overrides whatever the LLM/
        deterministic path produced for "Signatures" with this, same
        reasoning as `_closing_block`.
        """
        place = fields.get("place", "").strip() or "______"
        party_a_name = fields.get("party_a_name", "").strip() or "___________________"
        party_b_name = fields.get("party_b_name", "").strip() or "___________________"
        signed = phrase("signed", language, "Sd/-")
        lines = [
            f"{translated_heading('Place', language)}: {place}",
            f"{translated_heading('Date', language)}: {format_localized_date(clock.today(), language)}",
            "",
            signed,
            f"({party_a_name})",
            phrase("party_of_first_part", language, "Party of the First Part"),
            "",
            signed,
            f"({party_b_name})",
            phrase("party_of_second_part", language, "Party of the Second Part"),
        ]
        return "\n".join(lines)

    def _deterministic_notice_sections(
        self, template: DraftTemplateDefinition, fields: dict[str, str], language: str = "english"
    ) -> dict[str, str]:
        respondent_block = "\n".join(
            part for part in [fields.get("respondent_name", ""), fields.get("respondent_address", "")] if part
        )
        # Post-Phase-3 hardening (Phase 2, milestone C): where a period comes
        # from, and what happens when there isn't one.
        #
        # This line used to read `fields.get("response_deadline_days") or "15"`.
        # So every notice this app produced -- a rent/security-deposit demand
        # included -- went out telling its recipient they had fifteen days to
        # comply "entirely at your own risk as to costs and consequences",
        # whether or not the sender had chosen a period and whether or not any
        # law fixes one. For a security-deposit demand no statute does. That
        # deadline was invented by this function, printed over the sender's
        # signature, and served on a real person.
        #
        # Two legitimate sources now, and no third:
        #   1. the sender's own `response_deadline_days`, and
        #   2. a period the template records with the provision that fixes it
        #      (`statutory_response_period_days` + `..._basis` -- today only
        #      the s.138 cheque-bounce notice, where the fifteen days ARE the
        #      statute).
        # With neither, the notice states no period at all. A demand with no
        # deadline is a complete, ordinary legal notice; a demand with a
        # fabricated deadline is not.
        chosen = fields.get("response_deadline_days", "").strip()
        deadline = chosen or (
            str(template.statutory_response_period_days)
            if template.statutory_response_period_days is not None
            else ""
        )
        non_compliance = (
            phrase(
                "consequence",
                language,
                "If the above is not complied with within {deadline} days of receipt of this notice, "
                "I/we reserve the right to initiate appropriate legal proceedings against you, entirely "
                "at your own risk as to costs and consequences.",
                deadline=deadline,
            )
            if deadline
            else None
        )
        sections: dict[str, str] = {}
        if respondent_block:
            sections["Recipient"] = respondent_block
        sections["Subject"] = self._subject_line(template, fields, language)
        sections.update(self._deterministic_body_sections(template, fields, language, non_compliance=non_compliance))
        # "Place"/"Date"/the closing itself are deliberately not set here --
        # `_render_sections` always overrides the signature-block heading
        # with the full closing block (Part 53), so anything set here would
        # just be replaced.
        return sections

    def _deterministic_complaint_sections(
        self, template: DraftTemplateDefinition, fields: dict[str, str], language: str = "english"
    ) -> dict[str, str]:
        recipient = localized_authority_label(template.draft_id, template.authority_label, language)
        police_station = fields.get("police_station", "").strip()
        if police_station and police_station.casefold() not in recipient.casefold():
            recipient = f"{recipient}\n{police_station}"
        sections: dict[str, str] = {
            "Recipient": recipient,
            "Subject": self._subject_line(template, fields, language),
        }
        sections.update(self._deterministic_body_sections(template, fields, language, non_compliance=None))
        # "Place"/"Date"/the closing itself are deliberately not set here --
        # see the matching comment in `_deterministic_notice_sections`.
        return sections

    @staticmethod
    def _body_phrase(key: str, category: str, language: str, default: str, **kwargs: str) -> str:
        """`phrase()`, but prefers a Notice-specific variant for notices.

        The shared body phrases were written for a document petitioning an
        authority: they call the document "this application" and describe the
        facts as subject to "verification and investigation by the competent
        authority". Served on a private individual demanding repayment, that
        wording is simply wrong, and it appeared three times in one real Hindi
        recovery notice. Where a `<key>_notice` variant exists for the
        language, it wins; otherwise this is exactly `phrase()`.
        """
        if category == "Notice":
            notice_key = f"{key}_notice"
            variant = phrase(notice_key, language, "", **kwargs)
            if variant:
                return variant
            english_variant = phrase(notice_key, "english", "", **kwargs)
            if english_variant and language in ("english", "hinglish"):
                return english_variant
        return phrase(key, language, default, **kwargs)

    # Fields every category's particulars block shows when they are filled,
    # whatever the template is: these are matter-identifying details, not
    # document-type-specific ones, and a template that also names one in
    # `particulars_fields` is not shown it twice.
    _UNIVERSAL_PARTICULARS: tuple[tuple[str, str, str], ...] = (
        ("police_station", "police_station_label", "Police Station"),
        ("incident_location", "incident_location_label", "Place of incident"),
        ("incident_date", "incident_date_label", "Date of incident"),
        ("accused_details", "accused_details_label", "Accused / opposite party details"),
        ("witnesses", "witnesses_label", "Witnesses"),
    )

    def _particular_lines(
        self, template: DraftTemplateDefinition, fields: dict[str, str], language: str
    ) -> list[str]:
        """The labelled "<label>: <value>" lines for this document's own
        particulars block.

        Post-Phase-3 hardening (Phase 2, milestone C). This used to be a fixed
        list of seven keys -- police station, incident location/date/time,
        IMEI, accused, witnesses -- applied to all 55 templates. It is the
        vocabulary of a police complaint, and it is why the deterministic
        fallback read as generic filler for every other document type: a
        consumer complaint collected the purchase date, the amount paid and
        the product concerned and then rendered none of them as particulars,
        while a rent/security-deposit notice collected the premises address,
        the tenancy dates and the deposit amount and rendered none of those
        either. The values reached the document only if the user happened to
        repeat them inside their free-text facts.

        Now the ORDER and the SELECTION come from the template's own
        `particulars_fields`, and each label is the template's own label for
        that field, localized -- so the block is template-specific by
        construction rather than by adding another `if`. Nothing is invented:
        a field the user left blank contributes no line.
        """
        lines: list[str] = []
        rendered: set[str] = set()
        for key in template.particulars_fields:
            value = (fields.get(key) or "").strip()
            if not value:
                continue
            draft_field = template.get_field(key)
            label = (
                localized_field_label(template.draft_id, draft_field, language)
                if draft_field is not None
                else key.replace("_", " ").title()
            )
            lines.append(f"{label}: {value}")
            rendered.add(key)
        for key, phrase_key, fallback_label in self._UNIVERSAL_PARTICULARS:
            value = (fields.get(key) or "").strip()
            if not value or key in rendered:
                continue
            lines.append(f"{phrase(phrase_key, language, fallback_label)}: {value}")
            rendered.add(key)
        # Two fields that carry a template-specific label and no shared phrase.
        for key, fallback_label in (("incident_time", "Time of incident"), ("imei_number", "IMEI Number")):
            value = (fields.get(key) or "").strip()
            if not value or key in rendered:
                continue
            draft_field = template.get_field(key)
            label = (
                localized_field_label(template.draft_id, draft_field, language)
                if draft_field is not None
                else fallback_label
            )
            lines.append(f"{label}: {value}")
        return lines

    def _deterministic_body_sections(
        self,
        template: DraftTemplateDefinition,
        fields: dict[str, str],
        language: str,
        *,
        non_compliance: str | None,
    ) -> dict[str, str]:
        """Shared body-section assembly for the unified Notice/Complaint
        skeleton: Complainant Details, Introduction, Facts of the Case,
        Legal Position, Consequences, Prayer. `non_compliance`, when given,
        is a Notice-only "if you don't comply within N days..." line appended
        to "Prayer" (Complaint-category documents ask an authority to act,
        not a private party to comply, so they have no equivalent line).

        This path runs ONLY when every LLM attempt failed -- see
        `_render_sections`, which now also reports it to the user as
        `generation_mode="deterministic"` instead of passing it off as a
        finished draft.

        Draft-quality pass (Sept 2026) -- what changed and why:

        The previous version emitted one flat sentence per heading and dumped
        the structured fields as bare "Police Station: X" / "Place of
        incident: Y" label lines INSIDE the narrative. A real user's police
        complaint came out as a single page whose entire "Legal Position" was
        "This matter is governed, inter alia, by <Act>." and whose "Prayer"
        read "...दर्ज की जाए। करने का अनुरोध किया जाता है।" -- a broken
        sentence, because `called_upon`'s "{relief} करने का अनुरोध" frame was
        wrapped around a relief the user had already typed as a complete
        sentence.

        Now it composes an actual pleading from the same values:

        * an opening submission paragraph (not a bare "Sir/Madam,");
        * the case particulars as their own labelled block, kept OUT of the
          narrative;
        * the facts as separately numbered "1. That ..." pleading paragraphs,
          one per sentence/line the user wrote, which is how a complaint is
          actually laid out and which alone roughly triples the body length;
        * a HEDGED legal position ("prima facie appears to attract ...
          subject to verification"), never "is governed by" -- matching the
          same rule `legal_drafting_prompt.md` imposes on the LLM path;
        * an explicit evidence paragraph when the user listed documents;
        * an enumerated prayer that INTRODUCES a complete-sentence relief
          rather than inflecting it, plus a residual-relief clause,
          a cooperation undertaking and an acknowledgement request.

        It still invents nothing: every factual sentence is a value the user
        typed. The added text is legal scaffolding, not new facts.
        """
        # --- Case particulars: labelled, and deliberately kept out of the
        # narrative paragraphs below. A reader of a complaint expects these
        # as a header block, not interleaved with the story.
        particular_lines = self._particular_lines(template, fields, language)

        facts_paragraphs: list[str] = []
        if particular_lines:
            facts_paragraphs.append(
                f"{phrase('matter_particulars', language, 'Particulars of the matter')}:\n"
                + "\n".join(particular_lines)
            )
        numbered = _numbered_pleading_paragraphs(fields.get("facts", ""), language)
        if numbered:
            facts_paragraphs.append(
                self._body_phrase(
                    "facts_lead_in", template.category, language,
                    "The facts and circumstances giving rise to this document are as follows:",
                )
            )
            facts_paragraphs.append(numbered)

        # Notice-category documents head this block "Sender Details"; every
        # other category keeps "Complainant Details". The key must match the
        # category's own skeleton in `templates/base.py`, or `_render_full_text`
        # drops the section entirely.
        party_heading = "Sender Details" if template.category == "Notice" else "Complainant Details"
        sections: dict[str, str] = {
            party_heading: self._applicant_block(fields, language),
            # An "Introduction" whose entire content was the word "Sir/Madam,"
            # is what made the old output read like a form. The salutation
            # stays (it belongs at the top of a letter) but is now followed by
            # a real opening submission.
            "Introduction": (
                phrase("salutation", language, "Sir/Madam,")
                + "\n\n"
                + self._body_phrase(
                    "opening_submission",
                    template.category,
                    language,
                    "The applicant respectfully submits this document before your good office in respect of the "
                    "matter set out below, and states as follows:",
                )
            ),
        }
        if facts_paragraphs:
            sections["Facts of the Case"] = "\n\n".join(facts_paragraphs)

        legal_paragraphs: list[str] = []
        if template.applicable_acts_hint:
            # Hedged, per the same rule the LLM path follows: a draft written
            # from one side's account must never declare the law settled.
            legal_paragraphs.append(
                self._body_phrase(
                    "legal_position_hedged",
                    template.category,
                    language,
                    "The facts stated above, as asserted by the applicant and subject to verification and "
                    "investigation by the competent authority, prima facie appear to attract the provisions of "
                    "{acts}. Whether any offence or liability is in fact made out is a determination for the "
                    "competent authority on the evidence; no such conclusion is asserted here.",
                    acts=", ".join(template.applicable_acts_hint),
                )
            )
        available_documents = fields.get("available_documents", "").strip()
        if available_documents:
            legal_paragraphs.append(
                phrase(
                    "evidence_paragraph",
                    language,
                    "The applicant holds the following material in support of the above and is prepared to "
                    "produce it as and when required: {documents}.",
                    documents=available_documents,
                )
            )
        if legal_paragraphs:
            sections["Legal Position"] = "\n\n".join(legal_paragraphs)

        # "Consequences" is harm to the APPLICANT. The old unconditional
        # "loss, inconvenience and hardship" line is exactly the generic
        # filler `legal_drafting_prompt.md` rule 15 forbids, so it is used
        # only when the user actually described an impact; otherwise a
        # narrower, honest sentence is used instead.
        if not _mentions_impact(fields) and language in {"english", "hinglish"}:
            sections["Consequences"] = (
                "No separate financial loss, physical injury or other consequence is asserted beyond what the "
                "applicant has expressly stated in the facts above. The applicant seeks the intervention of the "
                "competent authority on that stated account."
            )
        else:
            sections["Consequences"] = self._body_phrase(
                "consequences_statement" if _mentions_impact(fields) else "no_consequence_stated",
                template.category,
                language,
                "As a direct result of the above, the applicant has suffered inconvenience and distress, and "
                "seeks the intervention of the competent authority in the matter.",
            )

        classical_intro = (
            advocate_register.classical_prayer_intro(
                language, template.applicable_sections_hint, template.applicable_acts_hint
            )
            if advocate_register.supports_advocate_register(language, template)
            else None
        )
        sections["Prayer"] = self._prayer_block(
            fields, language, non_compliance=non_compliance, category=template.category,
            classical_intro=classical_intro,
        )

        # A provider outage must not collapse a draft into a one-page form.
        # This is procedural/legal scaffolding only: it adds no event, person,
        # loss, document or allegation. The applicant's own narrative above
        # remains the sole factual account.
        #
        # This used to be ~50 lines of English prose inlined here behind
        # `category == "Complaint" and language in {"english", "hinglish"}`.
        # Both halves of that condition were doing real damage: a Hindi
        # RECOVERY NOTICE failed each of them, so it received nothing and went
        # out as a two-page skeleton whose entire Prayer section read
        # "उपलब्ध नहीं कराया गया.", while the equivalent English complaint got
        # a full three-page document. See `app/drafting/officialese.py`, which
        # now holds the text per category (a notice to a private party needs
        # different paragraphs from a complaint to a police station) and per
        # language, and still declines to put English paragraphs into a
        # document written in a language it has not been authored for.
        apply_official_scaffolding(sections, category=template.category, language=language)
        return sections

    def _prayer_block(
        self, fields: dict[str, str], language: str, *, non_compliance: str | None, category: str = "Complaint",
        classical_intro: str | None = None,
    ) -> str:
        """The Prayer section, built so it is grammatical whatever shape the
        user typed their relief in.

        The bug this replaces: `called_upon` is an INFLECTING template --
        Hindi "अतः आपसे {relief} करने का अनुरोध किया जाता है।" It reads
        correctly only when `{relief}` is a bare noun phrase. Users almost
        always type a complete sentence instead ("आरोपी के विरुद्ध उचित
        कानूनी कार्रवाई की जाए तथा मेरी शिकायत दर्ज की जाए।"), which produced
        "अतः आपसे आरोपी के विरुद्ध ... दर्ज की जाए। करने का अनुरोध किया जाता
        है।" -- two sentences fused at a full stop, in the single most
        important paragraph of the document.

        So the shape is detected first: a complete sentence is INTRODUCED by
        `prayer_intro` and enumerated; a bare noun phrase still goes through
        the existing `called_upon` inflection, which is correct for it.
        """
        relief = (fields.get("expected_relief", "") or fields.get("information_sought", "")).strip()
        parts: list[str] = []
        if relief:
            reliefs = _split_reliefs(relief)
            if _is_complete_sentence(relief):
                parts.append(
                    classical_intro
                    or phrase("prayer_intro", language, "The applicant most respectfully prays that:")
                )
                parts.append(
                    "\n".join(f"{index}. {item}" for index, item in enumerate(reliefs, start=1))
                    if len(reliefs) > 1
                    else reliefs[0]
                )
            else:
                parts.append(phrase("called_upon", language, "You are therefore called upon to {relief}.", relief=relief))
            parts.append(
                phrase(
                    "prayer_residual",
                    language,
                    "and to pass such further or other order(s) as the facts and circumstances of the case may "
                    "warrant.",
                )
            )
        else:
            # NEVER "Not provided." here. That is a form-filling placeholder,
            # and the Prayer is the operative paragraph of the document -- the
            # one that says what is actually being demanded. A real reported
            # draft went out with its entire प्रार्थना section reading
            # "उपलब्ध नहीं कराया गया।" ("not provided"), which is worse than
            # useless: it tells the recipient the sender is asking for
            # nothing, and it is the single clearest tell that a document was
            # machine-assembled rather than drafted.
            #
            # A demand notice that states a sum is, by construction, demanding
            # that sum, so that relief is stated explicitly. Anything else
            # gets a properly-worded general prayer. Neither invents a fact:
            # the amount is the user's own figure, and the general form asks
            # only for "such relief as the case warrants".
            amount = next(
                (fields[key].strip() for key in ("principal_amount", "dues_amount") if fields.get(key, "").strip()),
                "",
            )
            if amount:
                parts.append(
                    phrase(
                        "prayer_default_payment_demand",
                        language,
                        "You are therefore called upon to pay the sum of Rs. {amount} to the applicant, together "
                        "with such interest and costs as may be lawfully payable thereon.",
                        amount=amount,
                    )
                )
            else:
                parts.append(
                    phrase(
                        "prayer_default_generic",
                        language,
                        "The applicant most respectfully prays that the matter set out above be considered and "
                        "that such relief be granted as the facts and circumstances of the case may warrant.",
                    )
                )
        if non_compliance:
            parts.append(non_compliance)
        elif category != "Notice":
            # Authority-facing documents only: a complaint or application
            # conventionally offers cooperation and asks for a dated receipt. A
            # notice to a private party does neither -- there is no diary
            # number to issue and no inquiry to cooperate with.
            #
            # This used to key off `non_compliance is None`, which was the same
            # thing only because every notice always had a non-compliance line.
            # Once a notice with no user-chosen and no statutory reply period
            # correctly stopped producing one (see
            # `_deterministic_notice_sections`), a rent/security-deposit notice
            # started asking its recipient -- a private landlord -- to issue an
            # acknowledgement of "this application". The category is what
            # actually decides this, so it is what is checked.
            parts.append(
                phrase(
                    "cooperation_undertaking",
                    language,
                    "The applicant remains willing to extend every necessary cooperation in the inquiry and to "
                    "appear and record a statement whenever required.",
                )
            )
            parts.append(
                phrase(
                    "acknowledgement_request",
                    language,
                    "The applicant further requests that an acknowledgement of this document, bearing its receipt "
                    "number and date, be issued to the applicant.",
                )
            )
        return "\n\n".join(parts)

    def _complaint_verification_text(self, language: str) -> str:
        return phrase(
            "complaint_verification_text", language,
            "Verified that the contents of this complaint/application are true and correct to the best of "
            "my knowledge and belief, and that nothing material has been concealed therefrom.",
        )

    def _police_request_default(
        self, template: DraftTemplateDefinition, language: str
    ) -> str:
        has_section_173 = any("173" in section for section in template.applicable_sections_hint)
        if language == "hindi":
            if has_section_173:
                return (
                    "अतः विनम्र निवेदन है कि उपरोक्त शिकायत को रिकॉर्ड पर लेकर, यदि वर्णित तथ्यों से "
                    "संज्ञेय अपराध प्रकट होता है, तो भारतीय नागरिक सुरक्षा संहिता, 2023 की धारा 173 "
                    "के अंतर्गत FIR दर्ज की जाए और विधि अनुसार जांच/कार्यवाही की जाए।"
                )
            return (
                "अतः विनम्र निवेदन है कि उपरोक्त शिकायत को रिकॉर्ड पर लेकर इसकी जांच की जाए और "
                "विधि अनुसार आवश्यक कार्यवाही की जाए।"
            )
        if has_section_173:
            return (
                "The applicant respectfully requests that the complaint be taken on record and, if the facts "
                "disclose a cognizable offence, an FIR be registered under Section 173 of the Bharatiya "
                "Nagarik Suraksha Sanhita, 2023 and investigated in accordance with law."
            )
        return (
            "The applicant respectfully requests that this complaint be taken on record, examined, and "
            "dealt with in accordance with law."
        )

    def _police_request_block(
        self, template: DraftTemplateDefinition, fields: dict[str, str], language: str,
        *, classical_intro: str | None = None
    ) -> str:
        """Operative request for a police/SHO representation.

        It preserves the complainant's relief and ordinary procedural
        requests without a residual "further or other orders" clause, which
        belongs to a court pleading, not an administrative application.
        `classical_intro`, when given (advocate-register templates only --
        see `advocate_register.py`), prepends the "It is therefore most
        respectfully prayed that this Honorable Authority may kindly be
        pleased to..." framing confirmed live in a real advocate-drafted FIR
        registration application -- the heading stays "Request", not
        "Prayer", but the OPERATIVE PHRASING inside it is exactly what a
        real SHO representation uses.
        """
        relief = fields.get("expected_relief", "").strip()
        relief_text = (
            self._police_request_default(template, language)
            if _looks_like_untrimmed_user_prompt(relief, fields)
            else relief
        )
        relief_text = relief_text or self._police_request_default(template, language)
        parts = [
            f"{classical_intro}\n\n{relief_text}" if classical_intro else relief_text,
            phrase(
                "cooperation_undertaking", language,
                "The applicant remains willing to extend every necessary cooperation in the inquiry and to "
                "appear and record a statement whenever required.",
            ),
            phrase(
                "acknowledgement_request", language,
                "The applicant further requests that an acknowledgement of this document, bearing its receipt "
                "number and date, be issued to the applicant.",
            ),
        ]
        return "\n\n".join(part for part in parts if part.strip())

    def _deterministic_affidavit_sections(
        self, template: DraftTemplateDefinition, fields: dict[str, str], language: str = "english"
    ) -> dict[str, str]:
        name = fields.get("applicant_name", "").strip()
        relation = fields.get("applicant_father_name", "").strip()
        age = fields.get("deponent_age", "").strip()
        address = fields.get("applicant_address", "").strip()
        identity_parts = [name]
        if relation:
            identity_parts.append(
                phrase("son_daughter_wife_of", language, "S/o, D/o, W/o {name}", name=relation)
            )
        if age:
            identity_parts.append(f"{phrase('age_label', language, 'aged about')} {age} years")
        if address:
            identity_parts.append(
                f"निवासी {address}" if language == "hindi" else f"resident of {address}"
            )
        identity = ", ".join(part for part in identity_parts if part)
        if identity and language in {"english", "hinglish"}:
            deponent_block = f"I, {identity}, do hereby solemnly affirm and declare as under:"
        elif identity and language == "hindi":
            deponent_block = f"मैं, {identity}, शपथपूर्वक सत्यनिष्ठा से प्रतिज्ञान और घोषणा करता/करती हूँ कि:"
        else:
            deponent_block = identity or phrase("not_provided", language, "Not provided.")
        purpose = fields.get("affidavit_purpose", "").strip()
        facts = fields.get("facts", "").strip()
        statement_texts: list[str] = []
        if address and language in {"english", "hinglish"}:
            statement_texts.append(
                f"That my present residential address is {address.rstrip('. ')}."
            )
        if purpose:
            clean_purpose = purpose.rstrip(". ")
            if language in {"english", "hinglish"}:
                clean_purpose = re.sub(r"^to\s+", "", clean_purpose, flags=re.IGNORECASE)
                statement_texts.append(f"That I am executing this affidavit to {clean_purpose}.")
            else:
                purpose_statement = phrase(
                    "affidavit_purpose_statement", language,
                    "That this affidavit is sworn for the purpose of {purpose}.", purpose=clean_purpose,
                ).strip()
                statement_texts.append(
                    purpose_statement if _is_complete_sentence(purpose_statement) else purpose_statement + "."
                )
        # A lost-document affidavit has material particulars that may arrive
        # as separately collected fields rather than inside the narrative.
        # State only supplied values, and only when the user's facts do not
        # already contain them, so the fallback neither drops nor duplicates
        # the certificate number/date/place.
        if template.draft_id == "lost_document_affidavit":
            lost_name = fields.get("lost_document_name", "").strip()
            identifier = fields.get("document_identifier", "").strip()
            loss_date = fields.get("loss_date", "").strip()
            loss_place = fields.get("loss_place", "").strip()
            particulars: list[str] = []
            if lost_name and lost_name not in facts:
                particulars.append(lost_name)
            if identifier and identifier not in facts:
                particulars.append(
                    f"प्रमाणपत्र/दस्तावेज़ संख्या {identifier} है" if language == "hindi"
                    else f"the document/certificate number is {identifier}"
                )
            if loss_date and loss_date not in facts:
                particulars.append(f"यह {loss_date} को खो गया" if language == "hindi" else f"it was lost on {loss_date}")
            if loss_place and loss_place not in facts:
                particulars.append(f"स्थान {loss_place} था" if language == "hindi" else f"the place of loss was {loss_place}")
            if particulars:
                joined = ", तथा ".join(particulars) if language == "hindi" else ", and ".join(particulars)
                statement_texts.append(
                    phrase(
                        "affidavit_statement_line", language, "That {line}",
                        line=joined.rstrip(". ") + ("।" if language == "hindi" else "."),
                    )
                )
        fact_items = [line.strip() for line in facts.splitlines() if line.strip()]
        if len(fact_items) <= 1:
            fact_items = _split_sentences(facts)
        for item in fact_items:
            # Users commonly start their own sentence with "That". Strip only
            # that drafting prefix before applying the standard phrase, so the
            # output can never read "That That I...".
            clean_item = re.sub(r"^that\s+", "", item.strip(), flags=re.IGNORECASE)
            clean_item = re.sub(
                r"^(The|This|My)\b", lambda match: match.group(1).lower(), clean_item
            )
            if clean_item:
                statement_texts.append(
                    phrase("affidavit_statement_line", language, "That {line}", line=clean_item)
                )
        if not statement_texts:
            statement_texts.append(phrase("not_provided", language, "Not provided."))
        available_documents = fields.get("available_documents", "").strip()
        if available_documents and language in {"english", "hinglish"}:
            statement_texts.append(
                "That I am relying upon the following documents in support of this declaration: "
                f"{available_documents.rstrip('. ')}."
            )
        statements = "\n\n".join(
            f"{index}. {text}" for index, text in enumerate(statement_texts, start=1)
        )
        statement_count = len(statement_texts)
        verification = (
            "I, the deponent above named, do hereby verify that the contents of paragraphs "
            f"1 to {statement_count} of this affidavit are true and correct to the best of my "
            "knowledge and belief, that nothing material has been concealed therefrom, and no part of it is false."
            if language in {"english", "hinglish"}
            else phrase(
                "verification_text", language,
                "Verified that the contents of the above affidavit are true and correct to the best of "
                "my knowledge and belief, and that nothing material has been concealed therefrom.",
            )
        )
        return {
            "Court / Authority Name": localized_authority_label(template.draft_id, template.authority_label, language),
            "Deponent Details": deponent_block,
            "Statements": statements,
            "Verification": verification,
            "Place": fields.get("place", ""),
            "Date": clock.today().strftime(_DATE_FORMAT),
            "Signature": self._signature_block(fields, language, label=phrase("deponent_label", language, "Deponent")),
        }

    def _deterministic_application_sections(
        self, template: DraftTemplateDefinition, fields: dict[str, str], language: str = "english"
    ) -> dict[str, str]:
        # Part 55: same opening-salutation fix as `_deterministic_complaint_
        # sections` above, for the same reason -- an Application (e.g. RTI)
        # is addressed to an authority exactly like a Complaint, and the LLM
        # path already opens its substantive content with "Sir/Madam,".
        salutation = phrase("salutation", language, "Sir/Madam,")
        request_text = (
            fields.get("information_sought", "").strip()
            or fields.get("facts", "").strip()
            or fields.get("expected_relief", "").strip()
            or phrase("not_provided", language, "Not provided.")
        )
        sections = {
            "To": localized_authority_label(template.draft_id, template.authority_label, language),
            "Subject": self._subject_line(template, fields, language),
            "Applicant Details": self._applicant_block(fields, language),
            "Request": f"{salutation}\n\n{request_text}",
            # "Place"/"Date"/the closing itself are deliberately not set
            # here -- see the matching comment in
            # `_deterministic_notice_sections`.
        }
        # Applications receive the same authored, fact-neutral procedural
        # scaffolding as other authority-facing drafts. This prevents a
        # provider outage from reducing an application to one pasted sentence.
        apply_official_scaffolding(sections, category="Application", language=language)
        additions = [
            sections.pop(heading, "").strip()
            for heading in ("Introduction", "Prayer")
        ]
        additions = [addition for addition in additions if addition]
        if additions:
            sections["Request"] = "\n\n".join([sections["Request"], *additions])
        return sections

    # Security finding D1: fields a contract template collects to describe
    # "what this is about" beyond the generic `contract_purpose` field --
    # the property/rent templates use their own, more specific keys instead.
    # Checked in order; the first one present wins.
    _CONTRACT_SUBJECT_FIELD_KEYS: ClassVar[tuple[str, ...]] = (
        "contract_purpose", "property_address", "rented_premises_address",
    )
    # Security finding D1: money terms a contract template can collect as
    # their OWN dedicated field (as opposed to folded into the free-text
    # `terms` field, which was already rendered) -- a timeout-triggered
    # deterministic fallback for a rent or property agreement previously
    # rendered NONE of these, despite the user having supplied every one.
    # Keyed by field name (not by template id), so a future template reusing
    # the same field key is covered automatically without a code change.
    _CONTRACT_FINANCIAL_FIELD_KEYS: ClassVar[tuple[str, ...]] = (
        "monthly_rent_amount", "security_deposit_amount", "sale_consideration_amount",
    )

    @staticmethod
    def _field_label(template: DraftTemplateDefinition, key: str, language: str) -> str:
        for draft_field in template.all_fields():
            if draft_field.key == key:
                return draft_field.hindi_label if language == "hindi" and draft_field.hindi_label else draft_field.label
        return key

    def _deterministic_contract_sections(
        self, template: DraftTemplateDefinition, fields: dict[str, str], language: str = "english"
    ) -> dict[str, str]:
        not_provided = phrase("not_provided", language, "Not provided.")
        party_a = "\n".join(part for part in [fields.get("party_a_name", ""), fields.get("party_a_address", "")] if part) or not_provided
        party_b = "\n".join(part for part in [fields.get("party_b_name", ""), fields.get("party_b_address", "")] if part) or not_provided
        parties_block = (
            f"{phrase('party_of_first_part', language, 'Party of the First Part')}:\n{party_a}\n\n"
            f"{phrase('party_of_second_part', language, 'Party of the Second Part')}:\n{party_b}"
        )
        subject = next(
            (fields[key].strip() for key in self._CONTRACT_SUBJECT_FIELD_KEYS if fields.get(key, "").strip()),
            "",
        ) or not_provided
        financial_lines = [
            f"{self._field_label(template, key, language)}: {fields[key].strip()}"
            for key in self._CONTRACT_FINANCIAL_FIELD_KEYS
            if fields.get(key, "").strip()
        ]
        terms_body = fields.get("terms", "").strip() or fields.get("facts", "").strip()
        if financial_lines and terms_body:
            terms = "\n".join(financial_lines) + "\n\n" + terms_body
        elif financial_lines:
            terms = "\n".join(financial_lines)
        else:
            terms = terms_body or not_provided
        term_duration = fields.get("term_duration", "").strip()
        governing_law = fields.get("governing_law_state", "").strip() or "India"
        return {
            "Title": template.name,
            "Parties": parties_block,
            "Recitals": subject,
            "Terms and Conditions": terms,
            "Term and Termination": term_duration or not_provided,
            "Governing Law and Jurisdiction": governing_law,
            # "Signatures" is deliberately not set here -- `_render_sections`
            # unconditionally overrides it with `_contract_signatures_block`,
            # same reasoning as the letter-style closing block above.
        }

    def _render_full_text(
        self,
        sections: dict[str, str],
        language: str = "english",
        template: DraftTemplateDefinition | None = None,
    ) -> str:
        # Part 52: the `sections` dict keys stay literal English (the
        # LLM-response-parsing/field-edit contract) -- only the HEADING TEXT
        # shown here is translated, via a small closed lookup table rather
        # than a further LLM call (see `heading_translations.py`).
        #
        # Use bold paragraph labels rather than Markdown headings. This keeps
        # one universal chat text size/font and avoids theme-provided rules or
        # underlines beneath headings while preserving a clear hierarchy.
        modes: dict[str, HeadingMode] = {}
        ordered_headings = list(sections)
        if template is not None:
            grammar = document_schema_registry.for_template(template)
            ast = document_schema_registry.compile_ast(
                grammar,
                sections,
                language=language,
                context={"fields": {}, "family": grammar.document_family, "forum_type": grammar.forum_type},
            )
            modes = {block.source_heading: block.heading_mode for block in ast.blocks}
            ordered_headings = [block.source_heading for block in ast.blocks]
            ordered_headings.extend(heading for heading in sections if heading not in ordered_headings)

        rendered: list[str] = []
        for heading in ordered_headings:
            text = sections.get(heading, "").strip()
            if not text:
                continue
            mode = modes.get(heading, HeadingMode.VISIBLE)
            if mode in {HeadingMode.HIDDEN, HeadingMode.CONTINUATION_ONLY}:
                rendered.append(text)
            elif mode is HeadingMode.INLINE:
                rendered.append(f"**{translated_heading(heading, language)}:** {text}")
            else:
                rendered.append(f"**{translated_heading(heading, language)}**\n\n{text}")
        return "\n\n".join(rendered)


def _field_schema(draft_field: DraftField, *, required: bool) -> DraftFieldSchema:
    return DraftFieldSchema(
        key=draft_field.key,
        label=draft_field.label,
        hindi_label=draft_field.hindi_label,
        field_type=draft_field.field_type,
        required=required,
        help_text=draft_field.help_text,
    )
