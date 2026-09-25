import re
from dataclasses import dataclass, field
from typing import Any, cast, get_args

import structlog

from app.core.config import settings
from app.core.exceptions import DraftConflictError, DraftLockedError, NotFoundError
from app.drafting import discovery, recommendation
from app.drafting.description_translations import localized_description
from app.drafting.edit_commands import EditCommandInterpreter
from app.drafting.engine import LegalDraftEngine
from app.drafting.field_extraction import DraftFieldExtractor, infer_derivable_fields
from app.drafting.field_label_translations import localized_field_label
from app.drafting.intent import DraftIntentDetector, DraftIntentMatch
from app.drafting.language_support import supported_level
from app.drafting.safety import DraftSafetyGuard
from app.drafting.templates import get_template, list_templates
from app.drafting.templates.base import DraftField, DraftTemplateDefinition
from app.drafting.title_translations import localized_document_name, localized_title
from app.drafting.validation import DraftFieldValidator
from app.drafting.wrapper_messages import msg
from app.language.detector import extract_requested_language
from app.llm.base import ChatMessage
from app.llm.deadline import call_with_hard_timeout
from app.llm.prompts import prompt_registry
from app.memory.store import ConversationMemoryStore
from app.schemas.drafting import (
    DraftConversationStage,
    DraftGenerateRequest,
    DraftGenerateResponse,
    DraftTurnInfo,
)
from app.utils.prompt_security import PromptInjectionScanner

log = structlog.get_logger(__name__)

_EXPORT_FORMATS = ["pdf", "docx", "txt", "rtf"]

_DRAFT_STAGES: frozenset[str] = frozenset(get_args(DraftConversationStage))


# `memory` is a plain dict loaded from Redis/Mongo, so anything at all can be
# sitting under "draft_stage" -- a value written by an older build, or one
# corrupted in storage. `DraftTurnInfo.stage` is a closed literal, so an
# unrecognised value reaches pydantic and fails as a 500 while RESPONDING,
# after the turn's real work is already done. Validated on the way out instead,
# where it degrades to a known stage and says so.
def _as_stage(value: object, *, default: DraftConversationStage) -> DraftConversationStage:
    if value in _DRAFT_STAGES:
        return cast(DraftConversationStage, value)
    if value is not None:
        log.warning("unrecognised_draft_stage", stage=value, using=default)
    return default



# The chat layer's post-interruption reminder (`describe_pending`) suggests
# saying this to pick a draft back up -- recognized here as a neutral
# re-engagement phrase so it just re-displays the current collecting-stage
# state instead of being fed into field extraction like an ordinary answer.
_CONTINUE_DRAFT_PATTERN = re.compile(r"^\s*(continue|resume)(\s+(the\s+|this\s+|my\s+)?draft)?\s*$", re.IGNORECASE)


# Phase 1 item 4: what a user actually types to come back to a draft.
#
# The previous version required the literal word "draft" in every phrasing, so
# "resume my complaint" (naming the DOCUMENT instead of the mechanism -- the
# most natural way to say it) and "wahi draft khol do" ("open that draft",
# using an open/show verb rather than a continue/resume one) both fell through
# to RAG and were answered as fresh legal questions, leaving the half-filled
# draft sitting untouched. Split into named parts so each language's
# vocabulary stays independently readable and extendable.
#
# `_RESUME_OBJECT` deliberately covers the document nouns as well as "draft":
# someone who spent five turns filling in a police complaint calls it "my
# complaint", not "my draft".
_RESUME_OBJECT = (
    r"draft|document|application|complaint|notice|petition|affidavit|agreement"
    r"|ड्राफ्ट|मसौदा|शिकायत|आवेदन"
    r"|मसुदा|तक्रार"
    r"|ડ્રાફ્ટ|ફરિયાદ"
    r"|খসড়া|অভিযোগ"
    r"|வரைவு|புகார்"
    r"|డ్రాఫ్ట్|ఫిర్యాదు"
    r"|ڈرافٹ|شکایت"
)
# Verbs meaning "bring it back": continue/resume, plus the open/show/pick-up
# family, plus their Hinglish and Devanagari equivalents.
_RESUME_VERB = (
    r"continue|resume|reopen|re-?open|open|show|bring back|go back to|carry on with|pick up|get back to"
    r"|khol|kholo|khol\s*do|dikha|dikhao|dikha\s*do"
    r"|jari\s*rakho|jaari\s*rakho|chalu\s*karo|chalu\s*kro"
    r"|aage\s*badhao|aage\s*badha|aage\s*badho"
    r"|जारी|खोल|दिखा|फिर\s*से\s*शुरू|आगे\s*बढ़ा"
)
# Leading filler: "I want to", "mujhe", "please", "wahi"/"that same".
_RESUME_PREFIX = (
    r"(?:i\s+(?:want|would like|need)\s+(?:to\s+)?|please\s+|let'?s\s+|"
    r"mujhe\s+|mera\s+|meri\s+|apna\s+|apni\s+|wahi\s+|wo\s+|woh\s+|vahi\s+|"
    r"pehle\s*wala\s+|pehla\s+|purana\s+|last\s+|previous\s+|"
    r"मुझे\s+|मेरा\s+|मेरी\s+|वही\s+|पिछला\s+|पहले\s*वाला\s+)*"
)
# Trailing filler: "karna hai", "kar do", "kijiye", "please".
_RESUME_SUFFIX = (
    r"(?:\s*(?:kar|kr|karna|karni|krna|krni|karo|kro|kijiye|kijie|do|dijiye|hai|h|please|rakhna|rakho|ko|"
    r"करना|करनी|करें|करो|कीजिए|रख|रखना|रखनी|रखें|रखो|दो|दीजिए|है|हैं|को))*\s*[.!?।]*"
)
# Either word order: verb-then-object ("continue my draft", "khol do wo
# draft") or object-then-verb ("draft continue karna hai", "मसौदा जारी रखें"),
# because Indo-Aryan word order puts the object first and English puts it last.
# Filler that can sit BETWEEN the verb and the object. Hinglish puts the
# light verb there -- "continue kro draft ko" -- and with only English
# determiners allowed here that phrasing matched nothing, was read as a
# request for a brand-new document, and answered with the entire template
# list while the half-filled draft sat untouched.
_RESUME_INFIX = (
    r"(?:\s*(?:the|this|my|that|wahi|previous|pehle\s*wala|"
    r"kro|karo|kar|kr|karna|krna|kijiye|kijie|do|dijiye|please|"
    r"करो|कर|कीजिए|दो|दीजिए)\s+)*"
)
_RESUME_DRAFT_PATTERN = re.compile(
    rf"^{_RESUME_PREFIX}"
    rf"(?:"
    rf"(?:{_RESUME_VERB})\s*{_RESUME_INFIX}(?:{_RESUME_OBJECT})"
    rf"|(?:{_RESUME_OBJECT})\s*(?:ko\s+|को\s+)?(?:{_RESUME_VERB})"
    rf")"
    rf"{_RESUME_SUFFIX}$",
    re.IGNORECASE,
)


def _is_resume_draft_request(message: str) -> bool:
    """Recognise natural requests to return to the active draft.

    This is deliberately intent-based rather than requiring the exact UI hint
    ``continue draft``.  A user should not be sent to RAG merely because they
    wrote the same instruction in a normal sentence or Hinglish.
    """
    compact = re.sub(r"\s+", " ", message.strip().lower())
    if _CONTINUE_DRAFT_PATTERN.match(compact):
        return True
    return bool(_RESUME_DRAFT_PATTERN.fullmatch(compact))

# Multi-draft support: previously starting a second, differently-typed draft
# mid-collecting silently discarded the first one's answered fields (see
# `_continue_collecting`'s `new_match.draft_id != template.draft_id` branch,
# which called `_start_collecting` directly with no parking). These two
# patterns are meta-commands about draft MANAGEMENT, checked in `handle_turn`
# before any stage-specific handler -- deliberately distinct from
# `_CONTINUE_DRAFT_PATTERN` above, which means "the one draft I'm already
# in" and has no template name to disambiguate with.
# Phase 1 item 4: "show my drafts", in the languages this product replies in.
# The English-only version missed "mere drafts dikhao" and every native-script
# phrasing, so a Hindi user had no way to see what they had in progress.
_LIST_DRAFTS_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:list|show|see|view)\s+(?:my\s+|all\s+)?drafts?"
    r"|my\s+drafts?"
    r"|(?:mere|mera|meri|sare|sabhi)\s+drafts?(?:\s+(?:dikhao|dikha\s*do|batao|list\s*karo))?"
    r"|drafts?\s+(?:dikhao|dikha\s*do|batao|list\s*karo)"
    r"|kitne\s+drafts?\s+(?:hain|hai)"
    r"|मेरे\s*(?:ड्राफ्ट|मसौदे)\s*(?:दिखाओ|दिखाएं|बताओ)?"
    r"|ड्राफ्ट\s*सूची"
    r")\s*[.!?।]*\s*$",
    re.IGNORECASE,
)
# Phase 1 item 4: the trailing "draft" used to be mandatory, so "switch to
# cyber complaint" -- the phrasing this product's own help text uses -- matched
# nothing and fell through to RAG. It is optional now, and the Hinglish
# postposition order ("cyber complaint pe switch karo") is accepted too. The
# captured name is resolved against real parked drafts by `_try_switch_draft`,
# which returns `None` when nothing matches, so loosening this pattern can
# only fail to switch -- never switch to the wrong document.
_SWITCH_DRAFT_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:switch\s+to|switch|resume|go\s+back\s+to|open)\s+(?:the\s+|my\s+)?(?P<name>.+?)(?:\s+draft)?"
    r"|(?P<name2>.+?)\s*(?:pe|par|pr|me|mein)\s+switch\s*(?:karo|kro|kar\s*do)?"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def _switched_template_name(match: re.Match[str]) -> str:
    """The template name captured by whichever `_SWITCH_DRAFT_PATTERN` branch fired."""
    return (match.group("name") or match.group("name2") or "").strip()


# Phase 1 item 4: "start another draft" -- begin a SECOND document without
# naming it yet. It contains a drafting verb and the word "draft" but no
# template, so `DraftIntentDetector.detect()` classified it as an ambiguous
# new-document request and `_ambiguous_new_draft_reply` answered "I don't have
# a template for that document yet", which is both wrong and confusing: the
# user didn't name a document, they asked to be shown the menu. Recognised
# explicitly instead, so the current draft is parked (recoverable via "my
# drafts"/"switch to ...") and template selection starts cleanly.
_NEW_DRAFT_TRIGGER = (
    r"(?:i\s+(?:want|would like|need)\s+to\s+|please\s+|let'?s\s+|mujhe\s+)?"
    r"(?:start|create|make|begin|open|new|naya|nayi|nava|ek\s+aur|dusra|doosra|another)\s+"
    r"(?:a\s+|an\s+|one\s+)?(?:another|second|new|different|aur\s+ek|ek\s+aur|naya|nayi|dusra|doosra)?\s*"
    r"(?:draft|document|dastavez|ड्राफ्ट|मसौदा|दस्तावेज़|दस्तावेज)"
    r"(?:\s*(?:banao|bana\s*do|start\s*karo|shuru\s*karo|karna\s*hai|chahiye|"
    r"बनाओ|शुरू\s*करें|शुरू\s*करो|चाहिए))?"
)
_NEW_DRAFT_PATTERN = re.compile(rf"^\s*{_NEW_DRAFT_TRIGGER}\s*[.!?।]*\s*$", re.IGNORECASE)
# QA session 8: the bare form above only ever matched "start another draft"
# with NOTHING else -- a real user very often combines the trigger phrase
# with the specific document they actually want in the same breath ("Start a
# new draft: I want to write a police complaint for a stolen phone."), which
# matched neither this nor (before that session's separate fix)
# `_CANCEL_DRAFT_PATTERN`'s unanchored "new draft" -- it fell through to
# `_continue_collecting` and got swallowed as a field answer for whichever
# OTHER draft happened to be active. This prefix variant matches just the
# trigger phrase at the START of the message, with anything after it treated
# as the actual new-draft request (see `handle_turn`, which parks the
# current draft, strips this prefix, and re-runs intent detection on
# whatever remains -- falling back to the old bare-menu behavior only when
# nothing meaningful remains).
_NEW_DRAFT_PREFIX_PATTERN = re.compile(rf"^\s*{_NEW_DRAFT_TRIGGER}\s*[:\-—]?\s*", re.IGNORECASE)

# Part 32 "Draft State Manager" / Part 33 "Natural Confirmation Engine":
# bare confirmation words/phrases ("Yes.", "Generate.", "haan kar do", "ok kr
# do") that mean "do the obvious next thing with the current draft," never
# "start a brand-new one" -- even though "generate"/"create" are themselves
# strong drafting verbs that would otherwise make `DraftIntentDetector.detect()`
# treat a bare, template-less match as an ambiguous NEW drafting attempt.
# Checked everywhere `_CONTINUE_DRAFT_PATTERN` is, and treated identically:
# re-affirm/continue the CURRENT draft rather than asking "which document?"
# or risking the phrase itself being extracted as a field's literal value.
#
# Built from one flat token list rather than separate "affirmation"/"action"
# categories, because real Hinglish confirmations freely combine an
# affirmation with an action word in either role ("haan kar do" = affirm +
# act, "continue kr do" = act + act) -- a single message may be ONE token or
# TWO in sequence, covering both a bare word and a stacked Hinglish phrase
# without having to enumerate every pairing by hand.
_CONFIRMATION_TOKEN = (
    r"haan|hn|ha|han|theek\s*hai|thik\s*hai"
    r"|yes|yeah|yep|sure|okay|ok|fine"
    r"|kar\s*do|kr\s*do|bna\s*do|banao|likh\s*do|likho"
    r"|chalu\s*kro|chalu\s*karo|karo"
    r"|aage\s*bad[ho]o"
    r"|go\s*ahead|proceed|generate|create(\s+it)?|start|continue|resume"
)
_CONFIRMATION_WORD_PATTERN = re.compile(
    rf"^\s*(please\s+)?({_CONFIRMATION_TOKEN})(\s+({_CONFIRMATION_TOKEN}))?"
    rf"(\s+(the\s+|this\s+|my\s+)?draft)?\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def _template_list_line(template: DraftTemplateDefinition, language: str) -> str:
    """One line of the template-selection list, in `language`.

    Previously hardcoded "{English name} / {Hindi name}" regardless of the
    conversation's actual language -- a Tamil/Telugu/Kannada/Bengali user
    (or any other non-Hindi language) always saw the Hindi name as the
    second option, never their own. `localized_title` already covers those
    four languages (used at export time); reused here so the list itself
    follows the same language the rest of the turn was localized to. Falls
    back to the English name alone when `language` has no translation for
    this template (english/hinglish/marathi/etc.), same as `localized_title`
    itself falls back.
    """
    localized = localized_title(template, language)
    if localized != template.name:
        return f"- {template.name} / {localized}"
    return f"- {template.name}"


# Part 54 "Complete Multilingual Drafting Pass" item 1: languages where a
# field label shows ONLY its own translation -- no English, no Hindi mixed
# in -- per the explicit requirement that a South Indian language flow must
# never show a mixed English/Hindi label. Hindi itself keeps the existing
# "English (Hindi)" dual format below (a Hindi conversation showing its own
# two most relevant languages isn't the "mixed label in an unrelated
# language" problem this restricts); english/hinglish/any other language
# also keeps the plain English label, both unchanged from before.
_SINGLE_LANGUAGE_LABEL_LANGUAGES = {
    "tamil", "telugu", "kannada", "bengali",
    "malayalam", "marathi", "gujarati", "punjabi", "odia", "urdu",
    # Post-Phase-3 hardening (Phase 2, milestone D): Hindi joins the set.
    #
    # It was excluded on the reasoning that "a Hindi conversation showing its
    # own two most relevant languages isn't the mixed-label problem". In a real
    # Hindi session that is not how it reads. Ten languages got a reply written
    # entirely in themselves; Hindi -- by far the most used of them here -- got
    # "आइए आपका Rent / Security Deposit Recovery Notice तैयार करते हैं" followed
    # by a numbered list of "Applicant Address (आवेदक का पता)". A user who has
    # asked for Hindi is answered in Hindi, the same as everyone else.
    #
    # Safe for field extraction, which is what the dual label was implicitly
    # protecting: `DraftFieldExtractor._label_candidates` (Phase 1) already
    # registers the English label, the key-as-words, the synonyms, the
    # `hindi_label` AND the label as localized for the collecting language, so
    # an answer typed against either form still matches.
    #
    # Act and section names are unaffected: they are not field labels, and
    # `localized_title`/`localized_authority_label` keep the canonical form
    # wherever no authored translation exists.
    "hindi",
}


def _display_template_name(template: DraftTemplateDefinition, language: str) -> str:
    """The `{template}` value interpolated into every wrapper sentence below
    ("Let's draft your {template}.", "Here is your {template} draft:", ...).

    Part 55: previously always `template.name` (the plain English title)
    regardless of language -- so even a fully Tamil/Telugu/Kannada/Bengali
    reply still named the document in English. Only overridden for those
    four single-language-label languages (same set `_field_label_display`
    already restricts to), matching the existing "no English/Hindi mixed
    into a South Indian language reply" rule; Hindi/English/hinglish/any
    other language keep the original `template.name` behavior unchanged.
    """
    if language in _SINGLE_LANGUAGE_LABEL_LANGUAGES:
        return localized_document_name(template, language)
    return template.name


def _field_label_display(template: DraftTemplateDefinition, draft_field: DraftField, language: str) -> str:
    if language in _SINGLE_LANGUAGE_LABEL_LANGUAGES:
        return localized_field_label(template.draft_id, draft_field, language)
    return draft_field.label


@dataclass
class DraftTurnResult:
    reply_text: str
    info: DraftTurnInfo
    fields_snapshot: dict[str, str] = field(default_factory=dict)



_RECALL_PATTERN = re.compile(
    r"\b(batao|bata\s+do|bataiye|btao|summary|summarize|summarise|recap|yaad|dikhao|dikha\s+do|show\s+me|tell\s+me|"
    r"what\s+(is|was|are|were)|kya\s+(tha|thi|the|hai)|kitna\s+(tha|hai)|kaun\s+(tha|hai))\b",
    re.IGNORECASE,
)
_EDIT_VERB_PATTERN = re.compile(
    r"\b(badlo|badal|change|update|replace|correct\s+kar|sahi\s+kar|jodo|jod\s+do|add|hatao|hata\s+do|remove|delete|"
    r"edit|set|make\s+it)\b",
    re.IGNORECASE,
)


def _is_recall_request(message: str) -> bool:
    """A request to READ BACK what is already known, not to change the draft."""
    return bool(_RECALL_PATTERN.search(message)) and not _EDIT_VERB_PATTERN.search(message)

class DraftConversationEngine:
    """State machine driving drafting entirely inside the chat conversation.

    `handle_turn` returns `None` when the message isn't drafting-related at
    all, so the caller (`ChatService`) knows to fall through to the normal
    RAG chat flow unchanged. All state (`draft_mode`, `draft_stage`,
    `draft_template_id`, `draft_fields`, `draft_id`) lives directly in the
    session's existing `ConversationMemoryStore` dict, mutated in place and
    persisted by the caller exactly like `language_preference`/`current_intent`
    already are.
    """

    # Draft state keys checkpointed by `_checkpoint_collected_fields`. Kept as
    # one list so the checkpoint can never drift from what `_start_collecting`
    # actually sets.
    _CHECKPOINT_KEYS = (
        "draft_mode", "draft_stage", "draft_template_id", "draft_fields",
        "draft_id", "draft_language", "draft_paused",
    )

    def __init__(self, memory_store: ConversationMemoryStore | None = None) -> None:
        self.intent_detector = DraftIntentDetector()
        self.extractor = DraftFieldExtractor()
        self.validator = DraftFieldValidator()
        self.edit_interpreter = EditCommandInterpreter()
        self.prompt_scanner = PromptInjectionScanner()
        self.safety_guard = DraftSafetyGuard()
        self.draft_engine = LegalDraftEngine()
        # Used ONLY to checkpoint collected fields before the long generation
        # call (see `_checkpoint_collected_fields`). Ordinary state
        # persistence is still the caller's job, exactly as the class
        # docstring says -- this does not take that over.
        self.memory_store = memory_store or ConversationMemoryStore()

    async def _checkpoint_collected_fields(self, session_id: str, memory: dict[str, Any]) -> None:
        """Persist the fields collected so far, before generation is attempted.

        `ChatService` normally persists the mutated `memory` dict only after
        `handle_turn` returns. Generation is by far the slowest thing that
        happens inside it -- several LLM calls, each with its own timeout and
        retries -- so anything that goes wrong there (a provider hang, the
        HTTP client disconnecting first) discarded every field collected on
        that turn as well.

        That is exactly what happened in a reported session: the user typed
        out all nine fields of a Mobile Theft Complaint, the request never
        came back, and the very next message was answered with the same
        "I just need 9 more details" list -- their entire answer gone. With
        this checkpoint the fields are already durable, so a retry continues
        from where they got to instead of starting over.

        Best-effort by design: a checkpoint that fails must never cost the
        user the draft turn that was otherwise about to succeed.
        """
        try:
            await self.memory_store.update(
                session_id, **{key: memory.get(key) for key in self._CHECKPOINT_KEYS}
            )
        except Exception as exc:  # noqa: BLE001 - a failed checkpoint must not fail the turn
            log.warning("draft_field_checkpoint_failed", session_id=session_id, error=str(exc))

    async def _dispatch_fresh_request(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult | None:
        """Routes a message with no bearing on any in-progress draft state
        through template detection. Shared by `handle_turn`'s two "nothing
        active yet" entry points: a session with no draft at all, and
        (QA session 8) "start a new draft: <request>" once the previous
        draft has already been parked -- both need the exact same
        matched/ambiguous/candidates dispatch, previously duplicated only in
        the first of the two.
        """
        match = self.intent_detector.detect(message)
        if not match.matched:
            return None
        if match.ambiguous or not match.draft_id:
            if not match.candidates:
                return self._start_discovery(memory, message, language)
            return self._start_selecting(memory, message, language, candidates=match.candidates)
        return await self._start_collecting(session_id, message, language, memory, match.draft_id)

    async def handle_turn(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult | None:
        try:
            return await self._handle_turn_inner(session_id, message, language, memory)
        finally:
            # Everything said up to here belongs to a draft that was already
            # handled; a LATER draft must not be seeded from it (see
            # `_seed_profile_from_history`) or one document's facts leak
            # into the next.
            memory["draft_seen_message_count"] = len(memory.get("messages") or [])

    async def _handle_turn_inner(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult | None:
        if memory.get("draft_mode") or memory.get("parked_drafts"):
            if _LIST_DRAFTS_PATTERN.match(message):
                return self._list_drafts_reply(memory, language)
            switch_match = _SWITCH_DRAFT_PATTERN.match(message)
            if switch_match:
                switched = self._try_switch_draft(memory, _switched_template_name(switch_match), language)
                if switched is not None:
                    return switched
            # Phase 1 item 4: "start another draft" -- park what's in progress
            # (so "my drafts"/"switch to ..." can bring it back) and show the
            # menu, rather than answering "I don't have a template for that".
            # QA session 8: a message combining the trigger phrase with an
            # actual request ("Start a new draft: I want to write a police
            # complaint...") used to match neither this nor the bare
            # `_NEW_DRAFT_PATTERN`, fell through to `_continue_collecting`,
            # and was swallowed as a field answer for the OTHER, still-active
            # draft. The prefix match below strips just the trigger phrase
            # and re-runs detection on whatever request follows it, so the
            # named document actually starts instead of being discarded.
            if memory.get("draft_mode"):
                new_draft_match = _NEW_DRAFT_PREFIX_PATTERN.match(message)
                if new_draft_match:
                    remainder = message[new_draft_match.end():].strip()
                    self._park_current_draft(memory)
                    self._reset(memory)
                    if remainder:
                        dispatched = await self._dispatch_fresh_request(session_id, remainder, language, memory)
                        if dispatched is not None:
                            return dispatched
                    return self._start_selecting(memory, message, language)
        # Part 58 "Answer Quality Audit" issues 19-21: a PAUSED draft answers
        # only to an explicit request to come back to it. Everything else
        # returns `None` -- the "not drafting-related" signal -- so
        # `ChatService` routes the message normally instead of feeding it back
        # into field collection. Without this, an active draft was a trap:
        # a genuine unrelated question, a refusal ("nhi"), and a cancel
        # request all came back as the same pending-field list, with no way
        # out. The draft itself is untouched and fully restorable; only the
        # engine's claim on the NEXT message is released.
        if memory.get("draft_mode") and memory.get("draft_paused"):
            template = get_template(memory.get("draft_template_id"))
            recognized_edit = bool(
                template is not None
                and self.edit_interpreter.interpret(
                    message, template, str(memory.get("draft_language") or "english")
                ).action != "unknown"
            )
            resumes = (
                _is_resume_draft_request(message)
                or bool(_CONFIRMATION_WORD_PATTERN.match(message))
                or recognized_edit
                # Naming a document outright ("Threat Complaint", "police
                # complaint bana do") is as clear a re-entry into drafting as
                # "continue draft" is, and shouldn't need the magic phrase.
                or self.intent_detector.detect(message).matched
                or self.intent_detector.detect_named_template(message).matched
            )
            if not resumes:
                return None
            memory["draft_paused"] = False
        if not memory.get("draft_mode"):
            # See `_dispatch_fresh_request` for the matched/ambiguous/
            # candidates dispatch this shares with the "start a new draft:
            # <request>" branch above.
            return await self._dispatch_fresh_request(session_id, message, language, memory)

        stage = memory.get("draft_stage")
        if stage == "selecting":
            return await self._continue_selecting(session_id, message, language, memory)
        if stage == "describe_problem":
            return await self._continue_discovery_describe(session_id, message, language, memory)
        if stage in ("identify_role", "identify_relief", "identify_case_stage"):
            return await self._continue_discovery_question(session_id, message, language, memory)
        if stage == "confirm_template":
            return await self._continue_confirm_template(session_id, message, language, memory)
        if stage == "collect_jurisdiction":
            return await self._continue_collect_jurisdiction(session_id, message, language, memory)
        if stage == "confirm_summary":
            return await self._continue_confirm_summary(session_id, message, language, memory)
        if stage == "collecting":
            return await self._continue_collecting(session_id, message, language, memory)
        if stage == "preview":
            return await self._continue_preview(session_id, message, language, memory)
        if stage == "approved":
            return await self._continue_approved(session_id, message, language, memory)
        if stage in ("locked", "exported"):
            return await self._continue_locked(session_id, message, language, memory)

        self._reset(memory)
        return None

    def _snapshot_current_draft(self, memory: dict[str, Any]) -> dict[str, Any] | None:
        if not memory.get("draft_mode") or not memory.get("draft_template_id"):
            return None
        return {
            "draft_stage": memory.get("draft_stage"),
            "draft_template_id": memory.get("draft_template_id"),
            "draft_fields": dict(memory.get("draft_fields") or {}),
            "draft_id": memory.get("draft_id"),
            "draft_language": memory.get("draft_language"),
        }

    def _park_current_draft(self, memory: dict[str, Any]) -> None:
        snapshot = self._snapshot_current_draft(memory)
        if snapshot is None:
            return
        memory["parked_drafts"] = [*(memory.get("parked_drafts") or []), snapshot]

    def _restore_draft_snapshot(self, memory: dict[str, Any], snapshot: dict[str, Any]) -> None:
        memory["draft_mode"] = True
        memory["draft_stage"] = snapshot["draft_stage"]
        memory["draft_template_id"] = snapshot["draft_template_id"]
        memory["draft_fields"] = dict(snapshot.get("draft_fields") or {})
        memory["draft_id"] = snapshot.get("draft_id")
        memory["draft_language"] = snapshot.get("draft_language")
        memory["draft_awaiting_unlock_confirm"] = False

    def _list_drafts_reply(self, memory: dict[str, Any], language: str) -> DraftTurnResult:
        entries: list[str] = []
        if memory.get("draft_mode") and memory.get("draft_template_id"):
            template = get_template(memory["draft_template_id"])
            if template is not None:
                entries.append(f"- {_display_template_name(template, language)} (active, {memory.get('draft_stage')})")
        for snapshot in memory.get("parked_drafts") or []:
            template = get_template(snapshot["draft_template_id"])
            name = _display_template_name(template, language) if template else snapshot["draft_template_id"]
            entries.append(f"- {name} (paused, {snapshot.get('draft_stage')})")
        if not entries:
            reply = msg("no_drafts_in_progress", language, "You don't have any drafts in progress right now.")
        else:
            reply = (
                msg("your_drafts_intro", language, "Here are your drafts:")
                + "\n" + "\n".join(entries)
                + "\n\n" + msg(
                    "switch_draft_hint", language,
                    'Say "switch to <name> draft" to work on a different one.',
                )
            )
        info = DraftTurnInfo(stage=memory.get("draft_stage") or "selecting", template_id=memory.get("draft_template_id"))
        return DraftTurnResult(reply_text=reply, info=info)

    def _try_switch_draft(self, memory: dict[str, Any], template_query: str, language: str) -> DraftTurnResult | None:
        """Returns `None` (never a reply) when `template_query` doesn't match
        any PARKED draft -- lets `handle_turn` fall through to normal
        handling instead of confusingly claiming to "switch" to nothing.
        """
        parked = list(memory.get("parked_drafts") or [])
        named = self.intent_detector.detect_named_template(template_query)
        index = None
        for position, snapshot in enumerate(parked):
            if named.matched and named.draft_id and snapshot["draft_template_id"] == named.draft_id:
                index = position
                break
            template = get_template(snapshot["draft_template_id"])
            if template and template_query.strip().lower() in template.name.lower():
                index = position
                break
        if index is None:
            return None
        # `get_template` returns None for an id this build no longer ships --
        # a parked draft can outlive a renamed or withdrawn template, and
        # every use below dereferences it (`.name`, `.required_field_keys()`).
        # Resolving it BEFORE mutating any state means a stale snapshot falls
        # through to normal handling with the user's parked drafts untouched,
        # rather than half-switching and then raising `AttributeError` on
        # `NoneType`.
        target = get_template(parked[index]["draft_template_id"])
        if target is None:
            log.warning(
                "parked_draft_template_missing",
                template_id=parked[index].get("draft_template_id"),
            )
            return None
        self._park_current_draft(memory)
        parked = list(memory.get("parked_drafts") or [])
        snapshot = parked.pop(index)
        memory["parked_drafts"] = parked
        self._restore_draft_snapshot(memory, snapshot)
        template = target
        resumed_language = snapshot.get("draft_language") or language
        reply = msg(
            "draft_switched", resumed_language, 'Switched back to your {template} draft.',
            template=_display_template_name(template, resumed_language),
        )
        if snapshot.get("draft_stage") == "collecting":
            draft_fields = snapshot.get("draft_fields") or {}
            missing = sorted(template.required_field_keys() - {key for key, value in draft_fields.items() if value})
            reply += "\n\n" + self._compose_collecting_reply(template, draft_fields, missing, [], resumed_language)
            info = self._collecting_info(template, memory, missing_fields=missing)
        else:
            info = DraftTurnInfo(
                stage=snapshot.get("draft_stage") or "preview",
                template_id=snapshot["draft_template_id"],
                draft_id=snapshot.get("draft_id"),
            )
        return DraftTurnResult(reply_text=reply, info=info)

    # ------------------------------------------------------------------
    # Problem-first draft discovery.
    #
    # Entered only for a genuinely generic drafting request (see the
    # `not match.candidates` branch in `handle_turn`) -- a direct, named
    # request ("rent agreement banao") never reaches any method below; it
    # goes straight to `_start_collecting` exactly as it always has.
    #
    # Runs BEFORE the existing "collecting" stage, not instead of it: once a
    # template is confirmed and jurisdiction collected,
    # `_enter_field_collection_from_discovery` hands off into the SAME
    # "collecting" stage / `_process_collecting_message` machinery every
    # direct draft already uses, just with `draft_discovery_mode=True` set
    # so the one behavioural difference -- a pre-generation summary
    # confirmation -- kicks in (see `_finalize_generation_reply`'s call
    # site in `_process_collecting_message`).
    # ------------------------------------------------------------------

    _DISCOVERY_QUESTION_ORDER: tuple[str, ...] = ("user_role", "desired_reliefs", "case_stage")

    def _seed_profile_from_history(self, memory: dict[str, Any]) -> str:
        """Recent user turns already sitting in session memory, joined into
        one description string.

        A drafting request often doesn't fire until several fact-laden turns
        in ("landlord won't return my deposit..." ... "...draft a notice"),
        because `DraftIntentDetector` only looks for a drafting verb in the
        CURRENT message (see `intent.py`). Without this, `_start_discovery`
        would start the matter profile blank and discard everything the user
        already said, forcing them to repeat it. `memory["messages"]` already
        has the current triggering message appended as its last entry (see
        `ChatService.answer`), so that's excluded here the same way the RAG
        prompt-builders in `chat_service.py` exclude it.
        """
        seen = int(memory.get("draft_seen_message_count") or 0)
        prior_messages = (memory.get("messages") or [])[seen:-1]
        prior_user_texts = [
            m.get("content", "").strip()
            for m in prior_messages
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content")
        ]
        return " ".join(t for t in prior_user_texts if t)

    def _start_discovery(self, memory: dict[str, Any], message: str, language: str) -> DraftTurnResult:
        memory["draft_mode"] = True
        memory["draft_discovery_mode"] = True
        memory["draft_stage"] = "describe_problem"
        profile = discovery.default_matter_profile()
        prior_context = self._seed_profile_from_history(memory)
        if prior_context:
            profile["raw_description"] = prior_context
            discovery.apply_signals_to_profile(profile, prior_context)
        memory["matter_profile"] = profile
        memory["recommended_template_ids"] = []
        memory["recommendation_confidence"] = None
        memory["selected_template_id"] = None
        memory["recommendation_confirmed"] = False
        memory["jurisdiction_confirmed"] = False
        memory["generation_summary_confirmed"] = False
        memory["draft_discovery_questions_asked"] = []
        memory["draft_discovery_extra_clarification_asked"] = False
        memory["draft_discovery_browse_step"] = None
        memory["draft_discovery_browse_candidates"] = []
        memory["draft_fields"] = {}
        memory["draft_template_id"] = None
        memory["draft_id"] = None
        memory["draft_paused"] = False
        requested_language = extract_requested_language(message)
        if requested_language:
            memory["draft_language_hint"] = requested_language
        resolved_language = requested_language or language
        memory["draft_language"] = resolved_language
        reply = msg(
            "discovery_generic_prompt",
            resolved_language,
            "You don't need to know the legal name of the document. Just tell me in 1-2 lines what "
            "happened, which side you're on, and what result you want -- I'll suggest a suitable draft.",
        )
        return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="describe_problem"))

    async def _continue_discovery_describe(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult | None:
        resolved_language = memory.get("draft_language") or language
        # DIRECT REQUEST BEHAVIOUR applies at any point discovery is waiting
        # on a reply, not only on the very first message: a user who now
        # names the exact document they want ("actually, a rent agreement")
        # should skip the rest of discovery entirely, same as if they had
        # opened with that name. Mirrors the equivalent check the pre-
        # existing "selecting" stage already made via `detect_named_template`.
        # Only checked for SHORT messages -- a real problem description can
        # legitimately contain a template's trigger words as ordinary
        # narrative ("landlord ne mera rent nahi diya..."), and that must
        # still be read as the description it is, not misfired into a bare
        # selection. Mirrors the same length guard `_continue_collecting`
        # already applies for exactly this reason.
        if len(message.split()) <= 6:
            named = self.intent_detector.detect_named_template(message)
            if named.matched and named.draft_id:
                memory["draft_discovery_mode"] = False
                return await self._start_collecting(session_id, message, resolved_language, memory, named.draft_id)
        risky, _findings = self.prompt_scanner.scan(message)
        if risky:
            return DraftTurnResult(
                reply_text=msg(
                    "unsafe_prompt_injection", resolved_language,
                    "That message contains content I can't process for safety reasons. Could you rephrase it?",
                ),
                info=DraftTurnInfo(stage="describe_problem"),
            )
        profile = memory.get("matter_profile") or discovery.default_matter_profile()
        profile["raw_description"] = " ".join(p for p in [profile.get("raw_description"), message.strip()] if p)
        discovery.apply_signals_to_profile(profile, message)
        memory["matter_profile"] = profile
        return self._advance_discovery(memory, resolved_language)

    async def _continue_discovery_question(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult:
        resolved_language = memory.get("draft_language") or language
        profile = memory.get("matter_profile") or discovery.default_matter_profile()
        discovery.apply_signals_to_profile(profile, message)
        memory["matter_profile"] = profile
        return self._advance_discovery(memory, resolved_language)

    def _advance_discovery(self, memory: dict[str, Any], language: str) -> DraftTurnResult:
        """Decides what the discovery flow needs next: another targeted
        question (role -> relief -> case stage, only those not already
        answered/asked -- see the "don't repeatedly ask for information
        already stored" rule), or, once those are exhausted, a
        recommendation. Called after every discovery-stage answer is merged
        into the profile, regardless of which specific question it answered.
        """
        profile = memory.get("matter_profile") or discovery.default_matter_profile()
        asked = set(memory.get("draft_discovery_questions_asked") or [])
        slot_filled = {
            "user_role": bool(profile.get("user_role")),
            "desired_reliefs": bool(profile.get("desired_reliefs")),
            "case_stage": bool(profile.get("case_stage")),
        }
        for slot in self._DISCOVERY_QUESTION_ORDER:
            if not slot_filled[slot] and slot not in asked:
                return self._ask_discovery_question(memory, slot, language)

        result = recommendation.recommend(profile)
        if result.tier in ("low", "none") and not memory.get("draft_discovery_extra_clarification_asked"):
            memory["draft_discovery_extra_clarification_asked"] = True
            memory["draft_stage"] = "describe_problem"
            reply = msg(
                "discovery_need_more_detail",
                language,
                "I need a little more detail to suggest the right document -- could you describe what "
                "happened in a couple more lines?",
            )
            return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="describe_problem"))
        return self._show_recommendation(memory, language, result)

    def _candidate_templates_for_profile(self, profile: dict[str, Any]) -> list[DraftTemplateDefinition]:
        issues = set(profile.get("issues") or [])
        tagged = [t for t in list_templates() if t.domain]
        if not issues:
            return tagged
        narrowed = [t for t in tagged if issues & set(t.supported_issues)]
        return narrowed or tagged

    def _ask_discovery_question(self, memory: dict[str, Any], slot: str, language: str) -> DraftTurnResult:
        asked = set(memory.get("draft_discovery_questions_asked") or [])
        asked.add(slot)
        memory["draft_discovery_questions_asked"] = sorted(asked)
        profile = memory.get("matter_profile") or discovery.default_matter_profile()
        candidates = self._candidate_templates_for_profile(profile)

        if slot == "user_role":
            memory["draft_stage"] = "identify_role"
            roles: set[str] = set()
            for template in candidates:
                roles.update(template.user_roles)
                roles.update(template.opposite_party_roles)
            options = " / ".join(sorted(r.replace("_", " ") for r in roles)[:4]) or "your role in this matter"
            reply = msg("discovery_ask_role", language, "Are you the {options}?", options=options)
            return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="identify_role"))

        if slot == "desired_reliefs":
            memory["draft_stage"] = "identify_relief"
            reliefs: set[str] = set()
            for template in candidates:
                reliefs.update(template.desired_reliefs)
            options = " / ".join(sorted(r.replace("_", " ") for r in reliefs)[:4]) or "the outcome you want"
            reply = msg("discovery_ask_relief", language, "What do you want -- {options}?", options=options)
            return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="identify_relief"))

        memory["draft_stage"] = "identify_case_stage"
        reply = msg(
            "discovery_ask_case_stage",
            language,
            "What has happened so far in this matter -- just talks, a notice already sent, a case "
            "pending in court, or an order already passed?",
        )
        return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="identify_case_stage"))

    def _recommendation_actions_text(self, language: str) -> str:
        return msg(
            "discovery_actions_hint",
            language,
            'Say "continue" to go with this, "show alternatives" for other options, "my situation is '
            'different" to redescribe your matter, or "browse category" to look through document types yourself.',
        )

    def _show_recommendation(
        self, memory: dict[str, Any], language: str, result: recommendation.RecommendationResult
    ) -> DraftTurnResult:
        memory["recommended_template_ids"] = [c.draft_id for c in result.candidates]
        memory["recommendation_confidence"] = result.candidates[0].confidence if result.candidates else 0.0
        memory["draft_stage"] = "confirm_template"

        if result.tier == "none" or not result.candidates:
            reply = (
                msg(
                    "discovery_no_match",
                    language,
                    "No template in the catalogue exactly matches this matter yet. You can describe it "
                    "differently, browse a category yourself, or I can prepare a general working draft "
                    "from what you've told me.",
                )
                + "\n\n"
                + self._recommendation_actions_text(language)
            )
            return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="confirm_template"))

        if len(result.candidates) == 1:
            candidate = result.candidates[0]
            template = get_template(candidate.draft_id)
            name = _display_template_name(template, language) if template else candidate.name
            usage = localized_description(template, language) if template else ""
            reply = (
                msg(
                    "discovery_recommend_one", language, "So '{template}' is the suitable document. {usage}",
                    template=name, usage=usage,
                )
                + "\n\n" + self._recommendation_actions_text(language)
            )
            return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="confirm_template"))

        lines = [msg("discovery_recommend_many", language, "Here are a few documents that could fit your matter:")]
        for index, candidate in enumerate(result.candidates, start=1):
            template = get_template(candidate.draft_id)
            name = _display_template_name(template, language) if template else candidate.name
            lines.append(f"{index}. {name} -- {candidate.reason}")
        lines.append("")
        lines.append(self._recommendation_actions_text(language))
        return DraftTurnResult(reply_text="\n".join(lines), info=DraftTurnInfo(stage="confirm_template"))

    _BROWSE_ACTION_PATTERN = re.compile(r"browse\s*categor|category\s*dikhao|श्रेणी|categories", re.IGNORECASE)
    _ALTERNATIVES_ACTION_PATTERN = re.compile(
        r"alternative|other\s*option|different\s*template|aur\s*dikhao|dusra\s*dikhao", re.IGNORECASE
    )
    _DIFFERENT_SITUATION_PATTERN = re.compile(
        r"situation\s*is\s*different|different\s*situation|mera\s*case\s*alag|alag\s*hai|not\s*this\s*one",
        re.IGNORECASE,
    )

    async def _continue_confirm_template(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult:
        resolved_language = memory.get("draft_language") or language
        normalized = message.strip().lower()

        if memory.get("draft_discovery_browse_step") == "category":
            return self._handle_category_choice(memory, message, resolved_language)
        if memory.get("draft_discovery_browse_step") == "template":
            chosen_id = self._handle_browsed_template_choice(memory, message)
            if chosen_id is not None:
                return await self._confirm_recommended_template(session_id, resolved_language, memory, chosen_id)
            reply = msg(
                "select_template_retry", resolved_language, "I didn't catch that. Please pick one of the options above:"
            )
            return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="confirm_template"))

        recommended_ids: list[str] = memory.get("recommended_template_ids") or []
        selected_id: str | None = None
        if normalized.isdigit():
            index = int(normalized) - 1
            if 0 <= index < len(recommended_ids):
                selected_id = recommended_ids[index]
        if selected_id is None:
            for draft_id in recommended_ids:
                template = get_template(draft_id)
                if template and (template.name.lower() in normalized or draft_id.replace("_", " ") in normalized):
                    selected_id = draft_id
                    break
        if (
            selected_id is None and recommended_ids
            and (_CONFIRMATION_WORD_PATTERN.match(message) or "continue" in normalized)
        ):
            selected_id = recommended_ids[0]
        if selected_id:
            return await self._confirm_recommended_template(session_id, resolved_language, memory, selected_id)

        if self._ALTERNATIVES_ACTION_PATTERN.search(normalized):
            profile = memory.get("matter_profile") or {}
            full_result = recommendation.recommend(profile)
            already_shown = set(memory.get("recommended_template_ids") or [])
            more = [c for c in full_result.candidates if c.draft_id not in already_shown]
            if not more:
                reply = msg(
                    "discovery_no_alternatives", resolved_language,
                    "I don't have any other close matches beyond what's already shown.",
                )
                return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="confirm_template"))
            merged = recommendation.RecommendationResult(
                tier=full_result.tier, candidates=tuple([*full_result.candidates][:3])
            )
            return self._show_recommendation(memory, resolved_language, merged)

        if self._DIFFERENT_SITUATION_PATTERN.search(normalized):
            memory["matter_profile"] = discovery.default_matter_profile()
            memory["draft_discovery_questions_asked"] = []
            memory["draft_discovery_extra_clarification_asked"] = False
            memory["draft_stage"] = "describe_problem"
            reply = msg(
                "discovery_generic_prompt", resolved_language,
                "No problem -- tell me again in 1-2 lines what happened, which side you're on, and what "
                "result you want.",
            )
            return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="describe_problem"))

        if self._BROWSE_ACTION_PATTERN.search(normalized):
            return self._start_category_browse(memory, resolved_language)

        # DIRECT REQUEST BEHAVIOUR applies here too: a message that
        # confidently names a DIFFERENT, specific template ("Cheque bounce
        # notice banao") is a fresh direct request, not an answer to "which
        # of these did you mean?" -- the same "name it and skip the rest of
        # discovery" rule every other discovery stage already honours (see
        # `_continue_discovery_describe`'s equivalent check), and the same
        # park-and-switch pattern `_continue_collecting`/`_continue_preview`/
        # etc. already use for a mid-draft template switch. Checked only
        # after every confirm_template-specific command above has had a
        # chance to match, so "continue"/"show alternatives"/etc. are never
        # mistaken for a new drafting request.
        new_match = self.intent_detector.detect(message)
        if new_match.matched and new_match.draft_id:
            memory["draft_discovery_mode"] = False
            started = await self._start_collecting(session_id, message, resolved_language, memory, new_match.draft_id)
            if started is not None:
                return started
            # `_start_collecting` returns None only when the matched id has no
            # template (see its own docstring/the equivalent check in
            # `_continue_selecting`) -- falls through to the same "didn't
            # catch that" retry below rather than returning None from a
            # function declared to always return a `DraftTurnResult`.
            log.warning("draft_selection_template_missing", draft_id=new_match.draft_id)

        reply = (
            msg("select_template_retry", resolved_language, "I didn't catch that. Here are the options again:")
            + "\n\n" + self._recommendation_actions_text(resolved_language)
        )
        return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="confirm_template"))

    def _start_category_browse(self, memory: dict[str, Any], language: str) -> DraftTurnResult:
        memory["draft_stage"] = "confirm_template"
        memory["draft_discovery_browse_step"] = "category"
        lines = [msg("discovery_browse_categories_intro", language, "Sure -- pick a category:")]
        for index, (category_id, _name) in enumerate(discovery.TOP_LEVEL_CATEGORIES, start=1):
            lines.append(f"{index}. {discovery.localized_category_name(category_id, language)}")
        return DraftTurnResult(reply_text="\n".join(lines), info=DraftTurnInfo(stage="confirm_template"))

    def _handle_category_choice(self, memory: dict[str, Any], message: str, language: str) -> DraftTurnResult:
        normalized = message.strip().lower()
        categories = discovery.TOP_LEVEL_CATEGORIES
        chosen_id: str | None = None
        if normalized.isdigit():
            index = int(normalized) - 1
            if 0 <= index < len(categories):
                chosen_id = categories[index][0]
        if chosen_id is None:
            for category_id, name in categories:
                localized_name = discovery.localized_category_name(category_id, language)
                if (
                    name.lower() in normalized
                    or category_id.replace("_", " ") in normalized
                    or localized_name.lower() in normalized
                ):
                    chosen_id = category_id
                    break
        if chosen_id is None:
            lines = [msg("select_template_retry", language, "I didn't catch that. Pick a category:")]
            for index, (category_id, _name) in enumerate(categories, start=1):
                lines.append(f"{index}. {discovery.localized_category_name(category_id, language)}")
            return DraftTurnResult(reply_text="\n".join(lines), info=DraftTurnInfo(stage="confirm_template"))

        domains = discovery.CATEGORY_DOMAIN_MAP.get(chosen_id, ())
        matching = [t for t in list_templates() if t.domain in domains] if domains else []
        memory["draft_discovery_browse_step"] = "template"
        memory["draft_discovery_browse_candidates"] = [t.draft_id for t in matching]
        if not matching:
            reply = msg(
                "discovery_no_templates_in_category", language,
                "No templates are organized under this category yet -- try \"search a draft\" instead, "
                "or describe your matter and I'll suggest something.",
            )
            memory["draft_discovery_browse_step"] = None
            return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="confirm_template"))
        lines = [msg("discovery_recommend_many", language, "Here are the documents in this category:")]
        for index, template in enumerate(matching, start=1):
            lines.append(f"{index}. {_display_template_name(template, language)}")
        return DraftTurnResult(reply_text="\n".join(lines), info=DraftTurnInfo(stage="confirm_template"))

    def _handle_browsed_template_choice(self, memory: dict[str, Any], message: str) -> str | None:
        """Returns the chosen template's `draft_id` if `message` (a reply to
        the browsed-category template list) picked one, else `None` -- the
        caller (`_continue_confirm_template`, already async) awaits
        `_confirm_recommended_template` itself; this stays a plain sync
        lookup rather than duplicating that async confirmation logic here.
        """
        normalized = message.strip().lower()
        candidate_ids: list[str] = memory.get("draft_discovery_browse_candidates") or []
        chosen: str | None = None
        if normalized.isdigit():
            index = int(normalized) - 1
            if 0 <= index < len(candidate_ids):
                chosen = candidate_ids[index]
        if chosen is None:
            for draft_id in candidate_ids:
                template = get_template(draft_id)
                if template and template.name.lower() in normalized:
                    chosen = draft_id
                    break
        if chosen is None:
            return None
        memory["draft_discovery_browse_step"] = None
        memory["draft_discovery_browse_candidates"] = []
        memory["recommended_template_ids"] = [chosen]
        return chosen

    async def _confirm_recommended_template(
        self, session_id: str, language: str, memory: dict[str, Any], draft_id: str
    ) -> DraftTurnResult:
        template = get_template(draft_id)
        if template is None:
            memory["draft_stage"] = "describe_problem"
            reply = msg(
                "discovery_no_match", language,
                "That document isn't available. Let's try again -- describe your matter once more.",
            )
            return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="describe_problem"))
        memory["selected_template_id"] = draft_id
        memory["recommendation_confirmed"] = True
        memory["draft_discovery_browse_step"] = None
        profile = memory.get("matter_profile") or {}
        jurisdiction = profile.get("jurisdiction") or {}
        if jurisdiction.get("state"):
            memory["jurisdiction_confirmed"] = True
            return await self._enter_field_collection_from_discovery(session_id, language, memory, template)
        memory["draft_stage"] = "collect_jurisdiction"
        reply = msg("discovery_ask_jurisdiction", language, "Which state and district is this matter related to?")
        return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="collect_jurisdiction"))

    async def _continue_collect_jurisdiction(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult | None:
        resolved_language = memory.get("draft_language") or language
        template = get_template(memory.get("selected_template_id"))
        if template is None:
            self._reset(memory)
            return None
        profile = memory.get("matter_profile") or discovery.default_matter_profile()
        parsed = discovery.parse_jurisdiction_reply(message)
        profile["jurisdiction"] = {"country": "IN", "state": parsed["state"], "district": parsed["district"]}
        memory["matter_profile"] = profile
        memory["jurisdiction_confirmed"] = True
        return await self._enter_field_collection_from_discovery(session_id, resolved_language, memory, template)

    async def _enter_field_collection_from_discovery(
        self, session_id: str, language: str, memory: dict[str, Any], template: DraftTemplateDefinition
    ) -> DraftTurnResult:
        memory["draft_mode"] = True
        memory["draft_stage"] = "collecting"
        memory["draft_template_id"] = template.draft_id
        # Changing the recommended draft (e.g. "change draft type" at the
        # confirm_summary stage, or picking a different candidate at
        # confirm_template) reaches this same method for the NEW template --
        # so any field already collected for a DIFFERENT, incompatible
        # template must not silently ride along into this one's generation
        # request. A field is kept only when its key is one this template
        # actually defines; shared keys (both templates have a "place" or
        # "facts" field, say) survive exactly because the user doesn't have
        # to retype something still genuinely applicable.
        template_keys = template.field_keys()
        memory["draft_fields"] = {
            key: value for key, value in (memory.get("draft_fields") or {}).items() if key in template_keys
        }
        memory["draft_id"] = None
        memory["draft_paused"] = False
        memory["draft_language"] = memory.get("draft_language") or language
        collecting_language = memory["draft_language"]

        # Carries forward only what the user has already, explicitly typed
        # (the jurisdiction just collected) -- never invents a fact. Most
        # templates' closing-block "place" field is exactly this.
        profile = memory.get("matter_profile") or {}
        jurisdiction = profile.get("jurisdiction") or {}
        place_value = jurisdiction.get("district") or jurisdiction.get("state")
        draft_fields: dict[str, str] = memory["draft_fields"]
        if place_value and template.get_field("place") and not draft_fields.get("place"):
            draft_fields["place"] = place_value

        # Everything the user already said while the problem was being
        # described (own words, before this template was even chosen -- see
        # `_start_discovery`/`_continue_discovery_describe`) is exactly the
        # kind of fact this template's fields ask for. Run it through the
        # same extractor real field-collection turns use so a deposit
        # amount, a name, or a date already given isn't asked for again.
        raw_description = (profile.get("raw_description") or "").strip()
        if raw_description:
            extracted = await self.extractor.extract(raw_description, template, collecting_language)
            for key, value in extracted.items():
                if key in template_keys and value and not draft_fields.get(key):
                    draft_fields[key] = value

        missing_now = sorted(template.required_field_keys() - {k for k, v in draft_fields.items() if v})
        reply = self._compose_collecting_reply(template, draft_fields, missing_now, [], collecting_language)
        return DraftTurnResult(
            reply_text=reply, info=self._collecting_info(template, memory, missing_fields=missing_now)
        )

    _CHANGE_DRAFT_TYPE_PATTERN = re.compile(
        r"change\s*(the\s*)?draft\s*type|different\s*draft\s*type|draft\s*badlo|dusra\s*draft\s*chahiye",
        re.IGNORECASE,
    )

    def _build_pregeneration_summary(
        self, template: DraftTemplateDefinition, memory: dict[str, Any], language: str
    ) -> str:
        draft_fields: dict[str, str] = memory.get("draft_fields") or {}
        profile: dict[str, Any] = memory.get("matter_profile") or {}
        jurisdiction: dict[str, Any] = profile.get("jurisdiction") or {}
        lines = [msg("discovery_confirm_summary_intro", language, "Before I generate this, please confirm:")]
        lines.append(f"- {msg('discovery_summary_document', language, 'Document')}: {_display_template_name(template, language)}")
        if profile.get("user_role"):
            lines.append(f"- {msg('discovery_summary_role', language, 'Your role')}: {profile['user_role']}")
        if profile.get("opposite_party_role"):
            lines.append(
                f"- {msg('discovery_summary_opposite_party', language, 'Opposite party')}: {profile['opposite_party_role']}"
            )
        jurisdiction_text = ", ".join(p for p in [jurisdiction.get("district"), jurisdiction.get("state")] if p)
        if jurisdiction_text:
            lines.append(f"- {msg('discovery_summary_jurisdiction', language, 'Jurisdiction')}: {jurisdiction_text}")
        if profile.get("case_stage"):
            lines.append(f"- {msg('discovery_summary_stage', language, 'Matter stage')}: {profile['case_stage']}")
        if profile.get("desired_reliefs"):
            lines.append(
                f"- {msg('discovery_summary_relief', language, 'Relief requested')}: "
                + ", ".join(profile["desired_reliefs"])
            )
        for key in template.field_keys():
            if key == "place" or not draft_fields.get(key):
                continue
            draft_field = template.get_field(key)
            label = _field_label_display(template, draft_field, language) if draft_field else key
            lines.append(f"- {label}: {draft_fields[key]}")
        if template.get_field("available_documents") and not draft_fields.get("available_documents"):
            lines.append("")
            lines.append(
                msg(
                    "discovery_documents_offer", language,
                    "If you have the agreement, notice, order or receipt, you can upload it and I can "
                    "extract relevant details -- or just tell me what supporting documents you have.",
                )
            )
        lines.append("")
        lines.append(
            msg(
                "discovery_summary_actions", language,
                'Say "generate" to proceed, tell me what to correct or add, or say "change draft type" '
                "to pick a different document.",
            )
        )
        return "\n".join(lines)

    async def _continue_confirm_summary(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult | None:
        template = get_template(memory.get("draft_template_id"))
        if template is None:
            self._reset(memory)
            return None
        resolved_language = memory.get("draft_language") or language
        normalized = message.strip().lower()

        if self._CHANGE_DRAFT_TYPE_PATTERN.search(normalized):
            memory["draft_stage"] = "confirm_template"
            memory["generation_summary_confirmed"] = False
            recommended_ids: list[str] = memory.get("recommended_template_ids") or []
            profile = memory.get("matter_profile") or {}
            if recommended_ids:
                result = recommendation.recommend(profile, restrict_ids=frozenset(recommended_ids))
                if result.candidates:
                    return self._show_recommendation(memory, resolved_language, result)
            return self._start_category_browse(memory, resolved_language)

        command = self.edit_interpreter.interpret(message, template, resolved_language)
        if command.action == "replace_field" and command.target_field:
            if not command.new_value:
                draft_field = template.get_field(command.target_field)
                label = _field_label_display(template, draft_field, resolved_language) if draft_field else command.target_field
                reply = msg("ask_new_value", resolved_language, "Sure -- what should the new {label} be?", label=label)
                return DraftTurnResult(reply_text=reply, info=self._collecting_info(template, memory, missing_fields=[]))
            draft_fields = dict(memory.get("draft_fields") or {})
            draft_fields[command.target_field] = command.new_value
            memory["draft_fields"] = draft_fields
            reply = self._build_pregeneration_summary(template, memory, resolved_language)
            return DraftTurnResult(reply_text=reply, info=self._collecting_info(template, memory, missing_fields=[]))

        if _is_resume_draft_request(message) or _CONFIRMATION_WORD_PATTERN.match(message) or "generate" in normalized:
            memory["generation_summary_confirmed"] = True
            await self._checkpoint_collected_fields(session_id, memory)
            return await self._finalize_generation_reply(session_id, template, memory, resolved_language)

        reply = self._build_pregeneration_summary(template, memory, resolved_language)
        return DraftTurnResult(reply_text=reply, info=self._collecting_info(template, memory, missing_fields=[]))

    def _start_selecting(
        self, memory: dict[str, Any], message: str, language: str, candidates: tuple[str, ...] = ()
    ) -> DraftTurnResult:
        memory["draft_mode"] = True
        memory["draft_stage"] = "selecting"
        memory["draft_fields"] = {}
        memory["draft_template_id"] = None
        memory["draft_id"] = None
        memory["draft_paused"] = False
        # Part 52: a language named in this first, still-templateless message
        # ("Kannada me draft karo") would otherwise be lost by the time a
        # follow-up message picks the actual template -- carried forward so
        # `_start_collecting` can still honor it once selection completes.
        # Also used, right below, to resolve which language THIS reply
        # itself is written in (Part 52 "Workflow Localization" -- the whole
        # conversation, not just the eventual document, must match the
        # requested language).
        requested_language = extract_requested_language(message)
        if requested_language:
            memory["draft_language_hint"] = requested_language
        resolved_language = requested_language or language
        reply = (
            msg("select_template_prompt", resolved_language, "Sure, I can help you draft a document. Which type would you like?")
            + "\n\n"
            + self._template_options(resolved_language, candidates)
        )
        return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="selecting"))

    def _template_options(self, language: str, candidates: tuple[str, ...] = ()) -> str:
        """The selectable-template block.

        Part 58 issue 15: when the request DID narrow things down but not far
        enough to pick one -- "I've been cheated, prepare a complaint" is a
        Bank Fraud Complaint, an Online Fraud Complaint, or a Cyber Crime
        Complaint, and they ask for different facts -- the shortlist goes
        first, so the reply reads as "which of these did you mean?" rather
        than as a catalogue dump that ignored what the user just said. The
        full list still follows, since the shortlist is a guess.
        """
        full_list = "\n".join(_template_list_line(template, language) for template in list_templates())
        shortlist = [
            template for template in (get_template(draft_id) for draft_id in candidates)
            if template is not None
        ]
        if not shortlist:
            return full_list
        return (
            msg("did_you_mean_one_of", language, "Did you mean one of these?")
            + "\n"
            + "\n".join(_template_list_line(template, language) for template in shortlist)
            + "\n\n"
            + msg("or_choose_from_all", language, "Or choose from the full list:")
            + "\n"
            + full_list
        )

    async def _continue_selecting(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult:
        match = self.intent_detector.detect(message)
        if not (match.matched and match.draft_id):
            # Part 32: at THIS stage the user was just shown a list of
            # template names to pick from -- requiring a drafting verb again
            # ("draft"/"generate"/...) is wrong here specifically, since a
            # bare name/trigger phrase from that list ("Recovery Notice",
            # "RTI Application") is unambiguously a selection, not a fresh
            # question that needs a verb to tell it apart from one.
            match = self.intent_detector.detect_named_template(message)
        if match.matched and match.draft_id:
            started = await self._start_collecting(session_id, message, language, memory, match.draft_id)
            if started is not None:
                return started
            # `_start_collecting` returns None only when the matched id has no
            # template. Falling through re-shows the picker rather than
            # answering the user's selection with nothing at all.
            log.warning("draft_selection_template_missing", draft_id=match.draft_id)
        resolved_language = memory.get("draft_language_hint") or language
        reply = (
            msg("select_template_retry", resolved_language, "I didn't catch which document you need. Please choose one of:")
            + "\n\n"
            + self._template_options(resolved_language, match.candidates)
        )
        return DraftTurnResult(reply_text=reply, info=DraftTurnInfo(stage="selecting"))

    async def _start_collecting(
        self, session_id: str, message: str, language: str, memory: dict[str, Any], draft_id: str
    ) -> DraftTurnResult | None:
        template = get_template(draft_id)
        if template is None:
            self._reset(memory)
            return None
        memory["draft_mode"] = True
        memory["draft_stage"] = "collecting"
        memory["draft_template_id"] = draft_id
        memory["draft_fields"] = {}
        memory["draft_id"] = None
        memory["draft_paused"] = False
        # Part 52 "Multilingual Draft Engine": a language named explicitly in
        # this message -- or carried over from the templateless message that
        # started selection ("Kannada me draft karo") -- always wins over the
        # ambient conversation language; only falls back to `language`
        # (resolved from the ongoing conversation) when nothing was named.
        requested_language = extract_requested_language(message) or memory.pop("draft_language_hint", None)
        # A language the user CHOSE earlier in this conversation ("hindi me")
        # is a conversation-wide preference and survives a template change --
        # unlike the previous draft's facts, parties and addresses, which are
        # deliberately not carried over (`draft_fields` is reset above). A
        # user who switched the assistant into Hindi should not be answered in
        # English again merely because they started a second document.
        # `draft_language` of the draft being replaced counts as the same
        # preference: it is only ever set from an explicit request or from the
        # ambient language, and in the second case it equals `language`, so
        # carrying it forward changes nothing there.
        remembered_language = memory.get("draft_language_preference") or memory.get("draft_language")
        memory["draft_language"] = requested_language or remembered_language or language
        if requested_language:
            memory["draft_language_preference"] = requested_language
        # Continuous drafting commands commonly point back to facts already
        # supplied in this session ("in facts par ... draft karo", "ab
        # notice generate karo").  Direct template selection used to bypass
        # discovery's history seeding entirely, so fixing the route still
        # produced an empty draft and asked the user to repeat everything.
        # Only join history when the command explicitly refers backward;
        # unrelated fresh drafts in a long session remain isolated.
        initial_message = message
        if re.search(r"\b(in|these|those|same|above|previous|prior)\s+facts?\b|\b(inhi|isi|unhi)\b|\bab\b", message, re.IGNORECASE):
            prior_context = self._seed_profile_from_history(memory)
            if prior_context:
                initial_message = f"{prior_context}\n{message}"
                # Deterministic: the user's own earlier narrative IS the
                # facts. Left to the (optional, rate-limited) LLM pass, a
                # failed call meant the facts were silently lost.
                if template.get_field("facts") and not (memory.get("draft_fields") or {}).get("facts"):
                    memory.setdefault("draft_fields", {})["facts"] = prior_context
        return await self._process_collecting_message(session_id, initial_message, language, template, memory)

    async def _continue_collecting(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult | None:
        template = get_template(memory.get("draft_template_id"))
        if template is None:
            self._reset(memory)
            return None
        # A discovery-originated draft's "change draft type" command is
        # meant for the confirm_summary stage (see `_continue_confirm_
        # summary`), but the word "draft" alone is one of
        # `DraftIntentDetector`'s strong verbs -- said a turn early, while
        # still mid-collection, it would otherwise reach `detect()` below,
        # score no template (there's no drafting-verb-backed name in
        # "change draft type" itself), and come back "ambiguous with no
        # candidates" -- which `_ambiguous_new_draft_reply` answers by
        # dumping the full ~55-template catalogue, exactly what discovery
        # exists to avoid. Recognised here first and sent back to the
        # recommendation instead, same destination `_continue_confirm_
        # summary` sends it to when said at the right stage.
        if memory.get("draft_discovery_mode") and self._CHANGE_DRAFT_TYPE_PATTERN.search(message.lower()):
            resolved_language = memory.get("draft_language") or language
            memory["draft_stage"] = "confirm_template"
            recommended_ids: list[str] = memory.get("recommended_template_ids") or []
            profile = memory.get("matter_profile") or {}
            if recommended_ids:
                result = recommendation.recommend(profile, restrict_ids=frozenset(recommended_ids))
                if result.candidates:
                    return self._show_recommendation(memory, resolved_language, result)
            return self._start_category_browse(memory, resolved_language)
        # Part 32: checked BEFORE `DraftIntentDetector` runs at all -- "yes"/
        # "generate"/"create" mean "do the obvious next thing with THIS
        # draft," but "generate"/"create" are themselves strong drafting
        # verbs, so running `detect()` on them first would misread a bare,
        # template-less match as an ambiguous request to start a brand-new
        # draft (the "I don't have a template for that yet" reply below),
        # discarding all progress on the current one.
        if _is_resume_draft_request(message) or _CONFIRMATION_WORD_PATTERN.match(message):
            return await self._process_collecting_message(session_id, message, language, template, memory)
        new_match = self.intent_detector.detect(message)
        if not (new_match.matched and new_match.draft_id) and len(message.split()) <= 8:
            # A bare template NAME with no drafting verb ("Rent / Security
            # Deposit Recovery Notice", copied straight from the template
            # list this engine itself just showed) is unambiguously a
            # selection, not field content -- but only checked for short
            # messages, since a real "Facts of the Case"-style field answer
            # can legitimately contain a template's trigger words (e.g.
            # "security deposit") as ordinary narrative, not a selection.
            named_match = self.intent_detector.detect_named_template(message)
            if named_match.matched and named_match.draft_id and named_match.draft_id != template.draft_id:
                new_match = named_match
        if (
            new_match.matched
            and new_match.draft_id == "legal_notice"
            and template.draft_id != "legal_notice"
            and "notice" in template.draft_id
        ):
            # "Ab notice generate karo" said while already collecting a more
            # specific notice (e.g. `demand_notice`) only reached
            # `legal_notice` through `detect()`'s generic-"notice" fallback --
            # it names no different document, it means "generate THIS one".
            # Switching parked the real draft and restarted with an empty
            # field bag, so every fact collected so far was asked for again
            # (confirmed in the Q27-Q31 retest).
            new_match = DraftIntentMatch(matched=False)
        if new_match.matched and new_match.draft_id and new_match.draft_id != template.draft_id:
            # Multi-draft support: previously discarded every field/stage
            # already reached on `template` with no trace. Parking first
            # means "switch to <template> draft"/"my drafts" (see
            # `handle_turn`) can bring it back later, state intact.
            self._park_current_draft(memory)
            return await self._start_collecting(session_id, message, language, memory, new_match.draft_id)
        if new_match.matched and new_match.ambiguous and not new_match.draft_id:
            missing_now = sorted(
                template.required_field_keys() - {key for key, value in (memory.get("draft_fields") or {}).items() if value}
            )
            return self._ambiguous_new_draft_reply(
                template, self._collecting_info(template, memory, missing_fields=missing_now), memory,
                candidates=new_match.candidates,
            )
        return await self._process_collecting_message(session_id, message, language, template, memory)

    def _ambiguous_new_draft_reply(
        self, template: DraftTemplateDefinition, info: DraftTurnInfo, memory: dict[str, Any],
        candidates: tuple[str, ...] = (),
    ) -> DraftTurnResult:
        """The user clearly asked to draft *something* new (a strong verb like
        "generate"/"draft"), but it didn't match any known template -- e.g.
        "Generate a Rental Agreement" when no rental-agreement template
        exists. Previously this fell through silently and got treated as a
        field answer (or edit command) for the unrelated in-progress draft,
        which is worse than unhelpful: it looks like the new request was
        understood and ignored. `info` is built by the caller using whatever
        stage-appropriate builder matches the draft's actual current stage
        (collecting vs. preview), since this reply doesn't itself change it.
        """
        resolved_language = memory.get("draft_language") or "english"
        options = self._template_options(resolved_language, candidates)
        reply = (
            msg("unknown_template_prefix", resolved_language, "I don't have a template for that document yet. Available templates:")
            + f"\n\n{options}\n\n"
            + msg(
                "draft_still_saved",
                resolved_language,
                'Your {template} draft is still saved -- say "continue draft" to pick it back up, '
                'or "cancel draft" to discard it.',
                template=_display_template_name(template, resolved_language),
            )
        )
        return DraftTurnResult(reply_text=reply, info=info)

    async def _process_collecting_message(
        self, session_id: str, message: str, language: str, template: DraftTemplateDefinition, memory: dict[str, Any]
    ) -> DraftTurnResult:
        # An explicitly named language wins at ANY point during collection,
        # not only on the message that started the draft. Previously a user
        # who opened with an English/Hinglish trigger ("police complaint
        # banana hai") and then answered "hindi me Applicant Address: ..."
        # had that request read as ordinary field content, so the finished
        # document still came out in English -- the language was pinned by
        # the very first message and never revisited.
        mid_flow_language = extract_requested_language(message)
        if mid_flow_language:
            memory["draft_language"] = mid_flow_language
            # Remembered for the NEXT draft too -- see `_start_collecting`.
            memory["draft_language_preference"] = mid_flow_language
        collecting_language = memory.get("draft_language") or language
        risky, _findings = self.prompt_scanner.scan(message)
        if risky:
            return DraftTurnResult(
                reply_text=msg(
                    "unsafe_prompt_injection",
                    collecting_language,
                    "That message contains content I can't process for safety reasons. Could you rephrase it?",
                ),
                info=self._collecting_info(template, memory, missing_fields=[]),
            )
        unsafe, _unsafe_findings = self.safety_guard.scan(message)
        if unsafe:
            return DraftTurnResult(
                reply_text=msg(
                    "unsafe_forgery",
                    collecting_language,
                    "I can't help fabricate or forge information. I can only draft based on facts you "
                    "genuinely want recorded -- let's continue with accurate details.",
                ),
                info=self._collecting_info(template, memory, missing_fields=[]),
            )

        if _is_resume_draft_request(message) or _CONFIRMATION_WORD_PATTERN.match(message):
            # Reached directly (not via `_continue_collecting`'s own gate)
            # for "yes"/"proceed"/"okay"/"go ahead" -- none of those contain
            # a drafting verb, so `DraftIntentDetector.detect()` never
            # matches them at all and `_continue_collecting` falls straight
            # through to here. Handled the same as `_CONTINUE_DRAFT_PATTERN`:
            # re-show what's still needed rather than letting the extractor
            # below treat "yes" itself as a field's literal value (a real
            # risk specifically when exactly one field is still missing --
            # see the deterministic single-field fallback further down).
            draft_fields: dict[str, str] = dict(memory.get("draft_fields") or {})
            missing_now = sorted(template.required_field_keys() - {key for key, value in draft_fields.items() if value})
            if missing_now:
                reply = self._compose_collecting_reply(template, draft_fields, missing_now, [], collecting_language)
                return DraftTurnResult(
                    reply_text=reply, info=self._collecting_info(template, memory, missing_fields=missing_now)
                )
            # Nothing left to ask for. Re-showing the (empty) pending-field
            # list here was a dead end, and specifically the dead end a user
            # hits after a generation timeout: their nine fields are safely
            # checkpointed, they say "continue"/"haan" to retry, and the
            # engine answers by listing nothing and generating nothing. Fall
            # through to generation instead -- which is also what makes retry
            # idempotent, since `draft_fields` is reused rather than
            # re-collected and the failed attempt persisted no draft.
            log.info(
                "draft_resume_triggers_generation",
                session_id=session_id, template=template.draft_id, field_count=len(draft_fields),
            )

        draft_fields = dict(memory.get("draft_fields") or {})
        missing_before = sorted(template.required_field_keys() - {key for key, value in draft_fields.items() if value})

        # The labels the user was SHOWN are the labels they answer with, so
        # extraction is told which language those were printed in.
        extracted = await self.extractor.extract(message, template, collecting_language)
        template_keys = template.field_keys()
        for key, value in extracted.items():
            # Nothing is stored against a key this template does not define.
            # The extractors already filter, but this draft's field bag is
            # the thing a legal document is rendered from, so it validates
            # its own contents rather than trusting a caller.
            if key not in template_keys:
                log.info("draft_field_outside_template_dropped", template=template.draft_id, field=key)
                continue
            if value and not draft_fields.get(key):
                draft_fields[key] = value
        for key, value in infer_derivable_fields(template, draft_fields).items():
            draft_fields.setdefault(key, value)

        # Deterministic fallback so the flow never gets stuck waiting on a
        # freeform field regex/LLM extraction can't reliably parse (e.g. a
        # police station name) when it's the one thing left to ask for.
        # Not for a command ("Isi notice ko English mein convert karo",
        # "ab notice generate karo"): it is an instruction, not the value of
        # the one remaining field (it was stored as the Place of a notice).
        is_command = bool(extract_requested_language(message)) or self.intent_detector.detect(message).matched
        if (
            len(missing_before) == 1
            and not draft_fields.get(missing_before[0])
            and message.strip()
            and not is_command
        ):
            draft_fields[missing_before[0]] = message.strip()

        # `resolved_dates` collects any relative date expression the validator
        # turned into an absolute one ("kal" -> "31/08/2026"). Previously such
        # a value was simply rejected as invalid, so the user's incident date
        # was dropped and the finished document carried none; now it is
        # resolved and the assumption is stated back to them.
        resolved_dates: list[tuple[str, str, str]] = []
        issues = self.validator.validate(template, draft_fields, collecting_language, resolved_dates)
        for field_key, _message in issues:
            draft_fields.pop(field_key, None)
        date_notices = [
            msg(
                "date_assumed_notice",
                collecting_language,
                "I read '{original}' as {resolved}. If that's wrong, just tell me the correct date.",
                original=original,
                resolved=resolved,
            )
            for _key, original, resolved in resolved_dates
        ]

        memory["draft_fields"] = draft_fields
        missing_now = sorted(template.required_field_keys() - {key for key, value in draft_fields.items() if value})

        if issues or missing_now:
            reply = self._compose_collecting_reply(template, draft_fields, missing_now, issues, collecting_language)
            if date_notices:
                reply = "\n\n".join(date_notices) + "\n\n" + reply
            return DraftTurnResult(
                reply_text=reply, info=self._collecting_info(template, memory, missing_fields=missing_now)
            )

        # Every required field is in hand. A draft that went through
        # problem-first discovery gets one more stop -- a factual
        # pre-generation summary the user must explicitly confirm (the spec's
        # "Do not generate the document until required template fields are
        # present and the user has confirmed the summary") -- before
        # `_finalize_generation_reply` is ever called. A direct, named draft
        # (no discovery involved) is completely unaffected: it always went
        # straight to generation, and still does.
        if memory.get("draft_discovery_mode") and not memory.get("generation_summary_confirmed"):
            await self._checkpoint_collected_fields(session_id, memory)
            memory["draft_stage"] = "confirm_summary"
            reply = self._build_pregeneration_summary(template, memory, collecting_language)
            if date_notices:
                reply = "\n\n".join(date_notices) + "\n\n" + reply
            return DraftTurnResult(
                reply_text=reply, info=self._collecting_info(template, memory, missing_fields=[])
            )

        # Save them first -- see `_checkpoint_collected_fields`.
        await self._checkpoint_collected_fields(session_id, memory)
        return await self._finalize_generation_reply(session_id, template, memory, collecting_language)

    async def _finalize_generation_reply(
        self, session_id: str, template: DraftTemplateDefinition, memory: dict[str, Any], collecting_language: str
    ) -> DraftTurnResult:
        """Generates the document and builds the first-preview reply. Split
        out of `_process_collecting_message` so the discovery flow's
        `_continue_confirm_summary` can call exactly this same generation
        path once the user confirms the pre-generation summary, without
        duplicating the request-building/degraded-warning/reply-assembly
        logic below.
        """
        draft_fields: dict[str, str] = memory.get("draft_fields") or {}
        request = DraftGenerateRequest(
            draft_id=template.draft_id,
            language=collecting_language,
            fields=draft_fields,
            session_id=session_id,
            # BUG-014: this was the ONE place that actually creates a draft
            # record (LegalDraftEngine.generate persists version 1 here), and
            # it never passed `user_id` even though `memory["owner_user_id"]`
            # is already reliably set by this point -- `_claim_session_if_
            # unowned` (chat_service.py) and the `/draft` route
            # (app/api/drafting.py) both stamp it into memory before
            # `handle_turn` ever runs. Every OTHER mutating draft route
            # (`/draft/edit`, `/draft/export`, ...) already calls
            # `ensure_draft_access(draft, user_id, session_id)` correctly --
            # it simply had nothing to check for a draft created through this
            # conversational engine, since `owner_user_id` was never stamped
            # on the record in the first place. Without this, `session_id`
            # was a de facto bearer credential for every draft created via
            # chat, regardless of authentication (live-reproduced: QA session
            # 4, two real accounts, User B read User A's draft by supplying
            # A's session_id despite User B's own valid, different token).
            user_id=memory.get("owner_user_id"),
        )
        # A language this engine can only partly localize -- body yes, section
        # headings no (see `app/drafting/language_support.py`) -- is called out
        # before the document appears, rather than leaving the user to discover
        # a mixed-language PDF after they have downloaded it.
        partial_language_notice = (
            msg(
                "partial_language_support_notice",
                collecting_language,
                "Note: for this language the body of the document will be written in your language, but the "
                "section headings will remain in English.",
            )
            if supported_level(collecting_language) == "body_only"
            else ""
        )
        result = await self.draft_engine.generate(request)
        memory["draft_stage"] = "preview"
        memory["draft_id"] = result.draft_id
        # When every LLM attempt failed, the engine still returns a valid
        # document assembled from the user's own values -- but it is a
        # skeleton, not an advocate-style draft. Saying so is not optional:
        # this is a document the user may sign and file, and presenting a
        # degraded fallback under the same "Here is your draft" wrapper as a
        # full one is how a provider outage reached a user as a finished
        # legal document.
        degraded_warning = (
            msg(
                "draft_degraded_warning",
                collecting_language,
                # Leads with what the user is most anxious about after a
                # failure: whether the details they just typed out survived.
                # They did -- `_checkpoint_collected_fields` persists them
                # before generation is even attempted -- and saying so is the
                # difference between "retry" and "start the whole thing over".
                "Your details are saved, but enhanced drafting is temporarily unavailable. I created a "
                "basic draft below from the details you gave -- you can review it, download it, or say "
                '"regenerate" to try the full version again.',
            )
            if result.generation_mode == "deterministic"
            else ""
        )
        reply = (
            (f"{degraded_warning}\n\n" if degraded_warning else "")
            + (f"{partial_language_notice}\n\n" if partial_language_notice else "")
            + msg(
                "draft_ready_intro",
                collecting_language,
                "Here is your draft {template}:",
                template=_display_template_name(template, collecting_language),
            )
            + f"\n\n{result.full_text}\n\n"
            # The draft is ready to use right away: either download it, or say
            # what to change. No approve/lock step stands in between.
            + msg(
                "draft_edit_hint",
                collecting_language,
                'If this looks right, say "download PDF" (DOCX, TXT and RTF also work). '
                'To change something, just tell me what -- e.g. "change the police station '
                'to Hazratganj" -- or ask me to translate it into another language.',
            )
            # Part 52 "Workflow Localization": the AI-generated-draft
            # disclaimer is shown here, once, as a standalone chat aside --
            # deliberately OUTSIDE `result.full_text`, since it must never
            # appear inside the document itself (chat preview, PDF, DOCX, or
            # TXT). `result.disclaimer` stays in English regardless of draft
            # language (a fixed compliance statement, not document content).
            + f"\n\n{result.disclaimer}"
        )
        return DraftTurnResult(
            reply_text=reply,
            info=DraftTurnInfo(
                stage="preview",
                template_id=template.draft_id,
                template_name=template.name,
                draft_id=result.draft_id,
                sections=result.sections,
                full_text=result.full_text,
                disclaimer=result.disclaimer,
                # Downloads are offered immediately alongside the first
                # preview -- see `_preview_info` for why this is no longer gated.
                available_export_formats=list(_EXPORT_FORMATS),
                word_count=result.word_count,
                estimated_page_count=result.estimated_page_count,
                generation_mode=result.generation_mode,
                generation_error=result.generation_error,
                lifecycle_state="preview_ready",
                # Phase 1 item 6: advisory fact-only audit findings, surfaced
                # at the moment the user first reads the draft rather than
                # only at export -- that is when they are actually reviewing
                # it and can still say "I never said that".
                audit_findings=result.audit_findings,
            ),
        )

    def _find_named_parked_draft(self, memory: dict[str, Any], message: str) -> int | None:
        """Index of the ONE parked draft whose template name is explicitly
        named in `message`, or `None` if zero or more than one match
        (ambiguous -- safer to fall through to editing the active draft than
        guess). Without this, an edit that names a different, non-active
        draft by name (e.g. "General Legal Notice wale draft mein amount
        ... kar do" while a Cheque Bounce Notice draft happens to be active)
        silently applies to whichever draft is active instead of the one
        actually named -- see BUG-013.
        """
        parked = memory.get("parked_drafts") or []
        if not parked:
            return None
        lowered = message.lower()
        matches = [
            position
            for position, snapshot in enumerate(parked)
            if (candidate := get_template(snapshot["draft_template_id"])) is not None
            and candidate.name.lower() in lowered
        ]
        return matches[0] if len(matches) == 1 else None

    async def _regenerate_or_conflict_reply(
        self,
        draft_id: str,
        fields: dict[str, str],
        template: DraftTemplateDefinition,
        preview_language: str,
        *,
        language: str | None = None,
        style_instruction: str | None = None,
    ) -> DraftGenerateResponse | DraftTurnResult:
        """Thin wrapper shared by every edit action in `_continue_preview`
        that mutates a draft through `LegalDraftEngine.regenerate`.

        BUG-015: `regenerate` already retries once internally on a real
        concurrent-write conflict (see its own docstring) -- `
        DraftConflictError` reaching here means that retry ALSO lost the
        race. Turned into the same friendly, actionable chat reply at every
        call site rather than repeating a try/except six times, or letting
        it surface as an unhandled 500 through `ChatService`. The caller
        must check `isinstance(result, DraftTurnResult)` and return it
        directly when true, instead of treating it as a `DraftGenerateResponse`.

        Forwards `language`/`style_instruction` only when actually given,
        reproducing each call site's exact prior call signature (existing
        unit tests assert on it directly) rather than always passing both
        explicitly.
        """
        kwargs: dict[str, str] = {}
        if language is not None:
            kwargs["language"] = language
        if style_instruction is not None:
            kwargs["style_instruction"] = style_instruction
        try:
            return await self.draft_engine.regenerate(draft_id, fields, **kwargs)
        except DraftConflictError:
            reply = msg(
                "draft_edit_conflict",
                preview_language,
                "Someone or something else just changed this draft while I was applying your edit, so your "
                "change wasn't saved. Please say what you'd like changed again to retry.",
            )
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id))

    async def _continue_preview(
        self, session_id: str, message: str, language: str, memory: dict[str, Any], *, _retargeted: bool = False
    ) -> DraftTurnResult | None:
        template = get_template(memory.get("draft_template_id"))
        draft_id = memory.get("draft_id")
        if template is None or draft_id is None:
            self._reset(memory)
            return None

        # Part 32: same ordering fix as `_continue_collecting` -- "generate"/
        # "create" are strong drafting verbs, so without this check they'd
        # hit the ambiguous-new-draft branch below and wrongly claim "I
        # don't have a template for that" about a draft that's already
        # fully generated and sitting right here in preview.
        preview_language = memory.get("draft_language") or "english"

        if not _retargeted:
            retarget_index = self._find_named_parked_draft(memory, message)
            if retarget_index is not None:
                previous_name = _display_template_name(template, preview_language)
                self._park_current_draft(memory)
                parked = list(memory.get("parked_drafts") or [])
                snapshot = parked.pop(retarget_index)
                memory["parked_drafts"] = parked
                self._restore_draft_snapshot(memory, snapshot)
                retargeted_result = await self._continue_preview(
                    session_id, message, language, memory, _retargeted=True
                )
                if retargeted_result is not None:
                    new_language = memory.get("draft_language") or "english"
                    new_template = get_template(memory.get("draft_template_id"))
                    new_name = _display_template_name(new_template, new_language) if new_template else ""
                    switch_note = msg(
                        "auto_switched_named_draft",
                        new_language,
                        "(Switched to your {new} draft, since you named it -- your {previous} draft is "
                        "untouched and still saved.)\n\n",
                        new=new_name,
                        previous=previous_name,
                    )
                    retargeted_result.reply_text = switch_note + retargeted_result.reply_text
                return retargeted_result

        if _is_resume_draft_request(message) or _CONFIRMATION_WORD_PATTERN.match(message):
            result = await self.draft_engine.get_current(draft_id)
            reply = (
                f"{result.full_text}\n\n"
                + msg(
                    "draft_ready_for_review",
                    preview_language,
                    "This is your generated draft.\n\n"
                    "Would you like to:\n\n"
                    "1. Edit any details\n"
                    "2. Translate the draft\n"
                    "3. Regenerate the draft\n"
                    "4. Generate a PDF using this version",
                    template=_display_template_name(template, preview_language),
                )
            )
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id, result))

        command = self.edit_interpreter.interpret(message, template, preview_language)
        if command.action in {"replace_field", "append_field"} and _is_recall_request(message):
            # "Mere tenancy matter ka summary do", "ab corrected deposit
            # amount ... batao" ask what is ALREADY on record; they were read
            # as edits (the text was appended to the draft's Facts / an
            # amount field was queried for a new value). Hand them back to
            # normal routing, which answers from conversation memory.
            return None
        # Some current-draft actions contain drafting verbs ("generate a PDF")
        # or broad words ("approve", "lock") that should stay attached to
        # the preview. Field edits are different: a fresh prompt like
        # "Consumer complaint तैयार करो" can mention "email"/"address" and
        # otherwise look like an edit to the old draft. Give a confident
        # different-template request priority over field-changing commands.
        current_draft_action = command.action not in {"unknown", "replace_field", "append_field", "remove_paragraph"}
        new_match = None if current_draft_action else self.intent_detector.detect(message)
        if new_match and new_match.matched and new_match.draft_id and new_match.draft_id != template.draft_id:
            # Multi-draft support: previously discarded every field/stage
            # already reached on `template` with no trace. Parking first
            # means "switch to <template> draft"/"my drafts" (see
            # `handle_turn`) can bring it back later, state intact.
            self._park_current_draft(memory)
            return await self._start_collecting(session_id, message, language, memory, new_match.draft_id)
        if new_match and new_match.matched and new_match.ambiguous and not new_match.draft_id:
            return self._ambiguous_new_draft_reply(
                template, self._preview_info(template, draft_id), memory, candidates=new_match.candidates
            )

        if command.action == "translate":
            target_language = command.target_language or (
                "english" if memory.get("draft_language") == "hindi" else "hindi"
            )
            # Part 52: re-renders the SAME collected fields through the
            # normal drafting prompt in the new language and persists the
            # result (see `LegalDraftEngine.regenerate`'s docstring) instead
            # of the old `translate()` path, which returned a translated blob
            # that was never written back to the draft record -- so a
            # PDF/DOCX download after a language change silently kept
            # serving the pre-translation content. Regenerating also keeps
            # the document section-structured, which the exporters and the
            # one-page compaction logic both depend on.
            regen_outcome = await self._regenerate_or_conflict_reply(
                draft_id, {}, template, preview_language, language=target_language
            )
            if isinstance(regen_outcome, DraftTurnResult):
                return regen_outcome
            result = regen_outcome
            memory["draft_language"] = target_language
            reply = (
                msg("draft_translated_intro", target_language, f"Here is the draft in {target_language.capitalize()}:")
                + f"\n\n{result.full_text}"
            )
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id, result))

        if command.action in ("export", "approve", "lock"):
            # Downloads need no approval step. The frontend renders the buttons
            # from `available_export_formats`; this reply just points at them.
            # "approve"/"lock" still parse -- users who learned the old flow,
            # and existing tests, keep working -- but they gate nothing now:
            # the draft was already downloadable and stays editable after.
            reply = msg(
                "draft_download_ready",
                preview_language,
                "Your draft is ready to download below as PDF, DOCX, TXT, or RTF. "
                "You can still ask me to change anything afterwards -- just tell me what.",
            )
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id))

        if command.action == "unlock":
            reply = msg("already_editable", preview_language, "This draft is already editable -- no need to unlock it.")
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id))

        if command.action == "rollback" and command.version_number:
            try:
                result = await self.draft_engine.rollback(draft_id, command.version_number)
            except NotFoundError:
                reply = msg(
                    "no_such_version", preview_language, "I couldn't find version {number} of this draft.",
                    number=str(command.version_number),
                )
                return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id))
            reply = (
                msg(
                    "rollback_intro", preview_language, "Rolled back to version {number}. Here is that version:",
                    number=str(command.version_number),
                )
                + f"\n\n{result.full_text}"
            )
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id, result))

        if command.action in ("regenerate", "format"):
            regen_outcome = await self._regenerate_or_conflict_reply(draft_id, {}, template, preview_language)
            if isinstance(regen_outcome, DraftTurnResult):
                return regen_outcome
            result = regen_outcome
            if command.action == "format":
                reply = (
                    msg("reformatted_intro", preview_language, "Reformatted for a single professional page. Here is the updated draft:")
                    + f"\n\n{result.full_text}"
                )
            else:
                reply = msg("regenerated_intro", preview_language, "Here is the regenerated draft:") + f"\n\n{result.full_text}"
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id, result))

        if command.action == "restyle":
            if not command.new_value:
                reply = msg("ask_new_value", preview_language, "Sure -- what tone should I use?")
                return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id))
            regen_outcome = await self._regenerate_or_conflict_reply(
                draft_id, {}, template, preview_language, style_instruction=command.new_value
            )
            if isinstance(regen_outcome, DraftTurnResult):
                return regen_outcome
            result = regen_outcome
            reply = (
                msg(
                    "restyled_intro", preview_language,
                    'Rewritten in a "{tone}" tone -- same facts throughout. Here is the revised draft:',
                    tone=command.new_value,
                )
                + f"\n\n{result.full_text}"
            )
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id, result))

        if command.action == "suggest":
            # Finding-010's fix: unlike every other branch here, this one
            # must NEVER call `regenerate` (or any other draft-mutating
            # call) -- it only reads the current, already-persisted draft
            # and comments on it. Structural guarantee, not just a careful
            # prompt: nothing below touches `self.draft_engine.drafts`/
            # `.versions`, so the safety invariant holds even if the LLM
            # call itself fails.
            current = await self.draft_engine.get_current(draft_id)
            reply = await self._suggest_improvements(template, current.full_text, preview_language)
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id, current))

        if command.action == "replace_field" and command.target_field:
            draft_field = template.get_field(command.target_field)
            label = _field_label_display(template, draft_field, preview_language) if draft_field else command.target_field
            if not command.new_value:
                reply = msg("ask_new_value", preview_language, "Sure -- what should the new {label} be?", label=label)
                return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id))
            regen_outcome = await self._regenerate_or_conflict_reply(
                draft_id, {command.target_field: command.new_value}, template, preview_language
            )
            if isinstance(regen_outcome, DraftTurnResult):
                return regen_outcome
            result = regen_outcome
            reply = (
                msg("updated_field_intro", preview_language, "Updated {label}. Here is the revised draft:", label=label)
                + f"\n\n{result.full_text}"
            )
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id, result))

        if command.action == "append_field" and command.target_field:
            # BUG-010's fix: adds NEW content to a field without discarding
            # what is already there -- `replace_field` above always
            # overwrites, which is exactly wrong for "add a fact".
            draft_field = template.get_field(command.target_field)
            label = _field_label_display(template, draft_field, preview_language) if draft_field else command.target_field
            if not command.new_value:
                reply = msg("ask_new_value", preview_language, "Sure -- what should I add to {label}?", label=label)
                return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id))
            # The current, PERSISTED value (not the freshly-rendered prose,
            # which may already be an LLM's paraphrase) is what gets merged
            # with -- the same base every other edit action reads from, so a
            # concurrent/stale edit is no more or less protected here than
            # any other action already going through `regenerate` (rule 10
            # inherits whatever guarantee `LegalDraftEngine.regenerate`
            # already provides; this action adds no NEW risk on that front).
            current_draft = await self.draft_engine.drafts.find_by_id(draft_id)
            existing_value = (current_draft or {}).get("fields", {}).get(command.target_field, "")
            merged_value = self.edit_interpreter.append_value(existing_value, command.new_value)
            if merged_value == existing_value.strip():
                # Rule 7: a duplicate request is a no-op, not a re-append --
                # and not silently claiming a change that didn't happen.
                current = await self.draft_engine.get_current(draft_id)
                reply = msg(
                    "fact_already_present", preview_language,
                    "That's already in {label} -- no change made. Here is the current draft:", label=label,
                ) + f"\n\n{current.full_text}"
                return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id, current))
            regen_outcome = await self._regenerate_or_conflict_reply(
                draft_id, {command.target_field: merged_value}, template, preview_language
            )
            if isinstance(regen_outcome, DraftTurnResult):
                return regen_outcome
            result = regen_outcome
            # Rule 4: when no location was named, the reply says which one
            # was chosen (here, always the case where `target_field` fell
            # back to the template's own narrative field) rather than
            # silently deciding -- the label is stated either way, so this
            # reads identically whether the user named the section or not.
            reply = (
                msg("added_to_field_intro", preview_language, "Added to {label}. Here is the revised draft:", label=label)
                + f"\n\n{result.full_text}"
            )
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id, result))

        if command.action == "remove_paragraph" and command.target_field:
            draft_field = template.get_field(command.target_field)
            label = _field_label_display(template, draft_field, preview_language) if draft_field else command.target_field
            regen_outcome = await self._regenerate_or_conflict_reply(
                draft_id, {command.target_field: ""}, template, preview_language
            )
            if isinstance(regen_outcome, DraftTurnResult):
                return regen_outcome
            result = regen_outcome
            reply = (
                msg("removed_field_intro", preview_language, "Removed {label}. Here is the revised draft:", label=label)
                + f"\n\n{result.full_text}"
            )
            return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id, result))

        # Part 38 "Draft Auto-Pause Engine": previously fell back to a generic
        # "I can edit specific details... what would you like to do?" reply
        # here for ANY unrecognized message -- including a genuine new
        # question or problem statement ("My wife is throwing me out of the
        # house.") that has nothing to do with the draft. At preview stage
        # there's no field collection left to protect (unlike collecting
        # stage, where a declarative statement is ambiguously "a field
        # answer" and swallowing it is the right default), so returning
        # `None` here -- the same "not drafting-related" signal used when a
        # fresh message doesn't match any drafting verb at all -- lets
        # `ChatService.answer()` fall through to its normal routing and
        # actually answer the question, instead of conversation-locking the
        # user inside the draft. The chat layer is responsible for
        # reminding them the draft is still here (see
        # `describe_pending`/`interrupting_draft` handling), not this
        # method -- it only decides "is this about the draft or not."
        return None

    async def _suggest_improvements(
        self, template: DraftTemplateDefinition, draft_text: str, language: str,
    ) -> str:
        """Finding-010 (QA pass, 2026-09-11): "review this and suggest
        improvements, but don't change the draft" had no distinct capability
        -- every existing action either mutates the draft or refuses to
        engage with it at all. Read-only by construction: the caller
        (`_continue_preview`'s "suggest" branch) never passes the result
        anywhere near `regenerate`/`drafts`/`versions`, so a bug or a failure
        in THIS method can never corrupt or lose the persisted draft -- the
        worst it can do is return a poor suggestion or the fallback message
        below.

        Bounded by the same `call_with_hard_timeout` backstop every other
        drafting LLM call now goes through (QA 2026-09-11/12, BUG-013a /
        "New latency finding") rather than a bare `await self.draft_engine.
        llm.chat(...)` -- a suggestion request is exactly the kind of call
        that could otherwise stall well past a user's patience with nothing
        to show for it.
        """
        prompt = prompt_registry.render(
            "draft_suggestions_prompt", template_name=template.name, language=language, draft_text=draft_text,
        )
        try:
            response = await call_with_hard_timeout(
                self.draft_engine.llm.chat([ChatMessage(role="user", content=prompt)]),
                fallback_seconds=settings.draft_generation_budget_seconds,
            )
        except TimeoutError:
            response = None
        except Exception as exc:  # noqa: BLE001 - a suggestion failure must never look like a draft mutation failure
            log.warning("draft_suggestion_call_raised", error=str(exc), error_type=type(exc).__name__)
            response = None
        if response is None or response.error or not response.content.strip():
            fallback = msg(
                "suggestions_unavailable", language,
                "I couldn't generate suggestions just now. Your draft is unchanged -- you can try again, or "
                "tell me exactly what to change.",
            )
            return fallback
        intro = msg(
            "suggestions_intro", language,
            "Here are some suggestions -- your draft has NOT been changed:",
        )
        return f"{intro}\n\n{response.content.strip()}"

    def _compose_collecting_reply(
        self,
        template: DraftTemplateDefinition,
        draft_fields: dict[str, str],
        missing_now: list[str],
        issues: list[tuple[str, str]],
        language: str = "english",
    ) -> str:
        lines = [msg("lets_draft", language, "Let's draft your {template}.", template=_display_template_name(template, language))]
        known_keys = [key for key in template.field_keys() if draft_fields.get(key)]
        if known_keys:
            lines.append("\n" + msg("already_have_info", language, "I already have the following information:"))
            for key in known_keys:
                draft_field = template.get_field(key)
                label = _field_label_display(template, draft_field, language) if draft_field else key
                lines.append(f"✓ {label}")
        if issues:
            lines.append("\n" + msg("need_correction", language, "A couple of details need to be corrected:"))
            for field_key, issue_message in issues:
                draft_field = template.get_field(field_key)
                label = _field_label_display(template, draft_field, language) if draft_field else field_key
                lines.append(f"- {label}: {issue_message}")
        if missing_now:
            # Lists every still-missing field in one message -- a prior
            # version of this asked for only 2 at a time to feel more
            # conversational, but for a multi-field template that read as
            # confusing/incomplete (the header said "9 more" while only 2
            # were actually listed). Reverted per a later, opposite user
            # request. The field extractor already handles a user answering
            # several fields in one message regardless of how many were
            # listed, so this only changes what's displayed, not how answers
            # are parsed.
            count = len(missing_now)
            shown = missing_now
            lines.append(
                "\n"
                + msg(
                    "need_more_details",
                    language,
                    "I just need {count} more detail" + ("s" if count != 1 else "") + ":",
                    count=str(count),
                )
            )
            for index, key in enumerate(shown, start=1):
                draft_field = template.get_field(key)
                label = _field_label_display(template, draft_field, language) if draft_field else key
                lines.append(f"{index}. {label}")
        return "\n".join(lines)

    def _collecting_info(
        self, template: DraftTemplateDefinition, memory: dict[str, Any], *, missing_fields: list[str]
    ) -> DraftTurnInfo:
        draft_fields = memory.get("draft_fields") or {}
        collected_summary = []
        for key in template.field_keys():
            if draft_fields.get(key):
                draft_field = template.get_field(key)
                collected_summary.append(draft_field.label if draft_field else key)
        return DraftTurnInfo(
            stage="collecting",
            template_id=template.draft_id,
            template_name=template.name,
            collected_summary=collected_summary,
            missing_fields=missing_fields,
        )

    def _preview_info(
        self, template: DraftTemplateDefinition, draft_id: str, result: DraftGenerateResponse | None = None
    ) -> DraftTurnInfo:
        # Downloads are available from the moment a draft exists -- the
        # approve/lock gate that used to keep this empty was removed by
        # product decision. `LegalDraftEngine.export` permits the same set of
        # states, so the buttons the frontend renders from this list always work.
        return DraftTurnInfo(
            stage="preview",
            template_id=template.draft_id,
            template_name=template.name,
            draft_id=draft_id,
            sections=result.sections if result else {},
            full_text=result.full_text if result else "",
            available_export_formats=list(_EXPORT_FORMATS),
            word_count=result.word_count if result else 0,
            lifecycle_state="preview_ready",
            audit_findings=result.audit_findings if result else [],
        )

    def _lifecycle_info(
        self,
        template: DraftTemplateDefinition,
        draft_id: str,
        stage: DraftConversationStage,
        result: DraftGenerateResponse | None = None,
    ) -> DraftTurnInfo:
        """Shared builder for the "approved"/"locked"/"exported" stages --
        downloads only ever populate for the two stages an export is
        actually permitted from (see `LegalDraftEngine.export`'s own guard,
        which enforces this independently of what the chat layer shows)."""
        return DraftTurnInfo(
            stage=stage,
            template_id=template.draft_id,
            template_name=template.name,
            draft_id=draft_id,
            sections=result.sections if result else {},
            full_text=result.full_text if result else "",
            available_export_formats=list(_EXPORT_FORMATS) if stage in ("locked", "exported") else [],
            word_count=result.word_count if result else 0,
            lifecycle_state=stage,
        )

    async def _continue_approved(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult | None:
        template = get_template(memory.get("draft_template_id"))
        draft_id = memory.get("draft_id")
        if template is None or draft_id is None:
            self._reset(memory)
            return None
        approved_language = memory.get("draft_language") or "english"

        new_match = self.intent_detector.detect(message)
        if new_match.matched and new_match.draft_id and new_match.draft_id != template.draft_id:
            # Multi-draft support: previously discarded every field/stage
            # already reached on `template` with no trace. Parking first
            # means "switch to <template> draft"/"my drafts" (see
            # `handle_turn`) can bring it back later, state intact.
            self._park_current_draft(memory)
            return await self._start_collecting(session_id, message, language, memory, new_match.draft_id)
        if new_match.matched and new_match.ambiguous and not new_match.draft_id:
            return self._ambiguous_new_draft_reply(
                template, self._lifecycle_info(template, draft_id, "approved"), memory,
                candidates=new_match.candidates,
            )

        command = self.edit_interpreter.interpret(message, template, approved_language)

        if command.action == "lock" or _CONFIRMATION_WORD_PATTERN.match(message):
            try:
                await self.draft_engine.lock(draft_id)
            except DraftLockedError:
                # Defensive: another session/tab already moved this draft
                # out of "approved" (e.g. already locked). Re-sync and show
                # the current state instead of crashing the turn.
                memory["draft_stage"] = "locked"
                reply = msg("draft_locked_intro", approved_language, "Document locked. You can download it below.")
                return DraftTurnResult(reply_text=reply, info=self._lifecycle_info(template, draft_id, "locked"))
            memory["draft_stage"] = "locked"
            reply = msg(
                "draft_locked_intro",
                approved_language,
                'Document locked. You can now download it as PDF, DOCX, TXT, or RTF below. Say "unlock" if you '
                "need to make changes.",
            )
            return DraftTurnResult(reply_text=reply, info=self._lifecycle_info(template, draft_id, "locked"))

        # Part 57: anything else -- an explicit decline, or the user simply
        # continuing to ask for an edit -- cancels the pending approval and
        # returns to the normal editable preview flow. The spec's "editing
        # must remain available until the user explicitly approves" reads as
        # protecting against ACCIDENTAL post-approval edits, not trapping
        # someone who changes their mind before actually confirming the lock.
        await self.draft_engine.unlock(draft_id)
        memory["draft_stage"] = "preview"
        return await self._continue_preview(session_id, message, language, memory)

    async def _continue_locked(
        self, session_id: str, message: str, language: str, memory: dict[str, Any]
    ) -> DraftTurnResult | None:
        template = get_template(memory.get("draft_template_id"))
        draft_id = memory.get("draft_id")
        if template is None or draft_id is None:
            self._reset(memory)
            return None
        locked_language = memory.get("draft_language") or "english"
        current_stage = _as_stage(memory.get("draft_stage"), default="locked")

        # `detect_named_template()`, not `detect()`: the other three stage
        # handlers (collecting/preview/approved) deliberately require a
        # drafting VERB here, since the user is expected to be typing field
        # values at that point and a bare mention of another template's name
        # inside ordinary prose shouldn't hijack the draft. Locked/exported
        # is different -- there are no more fields to type, so any message
        # is either a command (unlock/download) or a genuine new request, and
        # requiring a verb here only breaks the common case of literally
        # naming the next document off the app's own menu (e.g. "Scholarship
        # Application") with no "create"/"make"/"draft" attached -- confirmed
        # live: that exact bare-name reply fell through to plain RAG (a "not
        # found in KB" answer) instead of starting the new draft.
        new_match = self.intent_detector.detect_named_template(message)
        if new_match.matched and new_match.draft_id and new_match.draft_id != template.draft_id:
            # Multi-draft support: previously discarded every field/stage
            # already reached on `template` with no trace. Parking first
            # means "switch to <template> draft"/"my drafts" (see
            # `handle_turn`) can bring it back later, state intact.
            self._park_current_draft(memory)
            return await self._start_collecting(session_id, message, language, memory, new_match.draft_id)
        if new_match.matched and new_match.ambiguous and not new_match.draft_id:
            return self._ambiguous_new_draft_reply(
                template, self._lifecycle_info(template, draft_id, current_stage), memory,
                candidates=new_match.candidates,
            )

        if memory.get("draft_awaiting_unlock_confirm"):
            memory["draft_awaiting_unlock_confirm"] = False
            confirmed = _CONFIRMATION_WORD_PATTERN.match(message) or (
                self.edit_interpreter.interpret(message, template, locked_language).action == "unlock"
            )
            if confirmed:
                await self.draft_engine.unlock(draft_id)
                memory["draft_stage"] = "preview"
                reply = msg("draft_unlocked", locked_language, "Draft unlocked -- you can make changes again.")
                return DraftTurnResult(reply_text=reply, info=self._preview_info(template, draft_id))
            reply = msg("draft_stays_locked", locked_language, "Okay, the draft stays locked.")
            return DraftTurnResult(reply_text=reply, info=self._lifecycle_info(template, draft_id, current_stage))

        command = self.edit_interpreter.interpret(message, template, locked_language)

        if command.action == "unlock":
            # Part 57: "unlocking must require an explicit user action" --
            # a separate confirmation turn, mirroring the lock flow's own
            # approve-then-confirm pattern, rather than unlocking on the
            # first "unlock" message alone.
            memory["draft_awaiting_unlock_confirm"] = True
            reply = msg(
                "confirm_unlock", locked_language, "Unlock this draft? This will allow edits again. (yes/no)"
            )
            return DraftTurnResult(reply_text=reply, info=self._lifecycle_info(template, draft_id, current_stage))

        if command.action == "export":
            export_format = command.export_format or "pdf"
            reply = msg(
                "export_ready", locked_language, "Your {format} is ready -- use the download button below.",
                format=export_format.upper(),
            )
            return DraftTurnResult(reply_text=reply, info=self._lifecycle_info(template, draft_id, current_stage))

        if command.action == "rollback":
            reply = msg(
                "rollback_needs_unlock",
                locked_language,
                'Unlock the draft first -- say "unlock" -- before rolling back to an earlier version.',
            )
            return DraftTurnResult(reply_text=reply, info=self._lifecycle_info(template, draft_id, current_stage))

        if command.action in ("replace_field", "remove_paragraph", "regenerate", "format", "translate", "approve", "lock"):
            reply = msg(
                "draft_is_locked",
                locked_language,
                'This draft is locked and can\'t be edited. Say "unlock" if you need to make changes.',
            )
            return DraftTurnResult(reply_text=reply, info=self._lifecycle_info(template, draft_id, current_stage))

        # Not drafting-related at all (Part 38 philosophy, same as
        # `_continue_preview`'s own final branch) -- fall through to normal
        # chat rather than trapping the user inside a locked draft.
        return None

    def reset(self, memory: dict[str, Any]) -> None:
        """Discards every in-progress draft (used by the chat layer's "cancel
        draft" command).

        Part 58 issue 18: this used to leave `parked_drafts` behind, so a
        session with a paused second draft was still in drafting state right
        after the system said "I've discarded that draft" -- and the next
        message got pulled straight back into the flow the user had just
        asked to leave. "Cancel" means the whole drafting flow stops; a
        specific one can always be restarted by name.
        """
        memory["parked_drafts"] = []
        self._reset(memory)

    # Stages where the engine has just asked a short, specific question and
    # is waiting on nothing but the answer to it -- see `pause`'s docstring.
    # "confirm_summary" is deliberately NOT included here: by that stage
    # real collected field data exists (same as "collecting"/"preview"),
    # so an interruption pausing it and requiring an explicit "continue
    # draft" to come back is the correct, data-protecting behaviour, not
    # this exemption.
    _UNPAUSABLE_STAGES = frozenset({
        "selecting", "describe_problem", "identify_role", "identify_relief",
        "identify_case_stage", "confirm_template", "collect_jurisdiction",
    })

    def pause(self, memory: dict[str, Any]) -> None:
        """Suspends the draft engine's claim on incoming messages while
        keeping the draft itself intact (see `handle_turn`'s paused branch).
        Cleared by an explicit "continue draft"/confirmation, by starting a
        new draft, or by cancelling.

        Deliberately a no-op in `_UNPAUSABLE_STAGES`. Everywhere else, the
        engine is waiting on data it has already asked for and can afford to
        wait longer; at these stages it has just asked a short question (a
        template name at "selecting", a role/relief/case-stage/jurisdiction
        answer during discovery) and the user's next message is expected to
        be a short, specific reply -- which is neither a resume phrase nor a
        confirmation, so pausing there would make the answer to the engine's
        own question fall through to ordinary chat instead of being read as
        the answer it is. Confirmed live: an unrelated interruption
        ("Translate it.") during discovery's "describe_problem" stage paused
        the draft, and the VERY NEXT message -- a perfectly ordinary reply to
        the still-pending question -- fell through to general chat instead
        of continuing discovery, because only "selecting" was originally
        exempted from this pause. There are also no collected fields at any
        of these stages, so nothing is trapped: the user can simply ignore
        the question, and "cancel draft" still works from here.
        """
        if memory.get("draft_mode") and memory.get("draft_stage") not in self._UNPAUSABLE_STAGES:
            memory["draft_paused"] = True

    def describe_pending(self, memory: dict[str, Any], fallback_language: str | None = None) -> str | None:
        """One-line reminder of the in-progress draft, for the chat layer to
        append after answering a question that interrupted it. `None` when
        there's nothing to remind about.

        Deliberately a single line, not a re-listing of every pending field --
        that's already shown once in the collecting-stage reply itself,
        repeating the full list on every interruption reads as a robotic
        "form" rather than a reminder that a real conversation was paused.
        """
        if not memory.get("draft_mode"):
            return None
        stage = memory.get("draft_stage")
        template = get_template(memory.get("draft_template_id")) if memory.get("draft_template_id") else None
        # Post-Phase-3 hardening milestone D: falling straight back to English
        # printed this reminder in English under a Hindi answer whenever the
        # user had never NAMED a language but had simply been writing in one.
        # The conversation's own detected preference is a better answer than
        # English, and is what every other reminder in the chat layer uses.
        language = memory.get("draft_language") or fallback_language or "english"

        if stage == "selecting" or template is None:
            return msg(
                "still_in_progress_generic",
                language,
                'You still have a document draft in progress -- say "continue draft" to pick it back up, '
                'or "cancel draft" to discard it.',
            )
        if stage == "preview":
            # Part 38 "Draft Auto-Pause Engine": deliberately just this one
            # line -- no re-listing of edit/translate/download options (that
            # was already shown once when the draft was generated), no
            # "cancel draft" mention either. This is the reminder shown
            # after AUTO-PAUSING to answer an unrelated question, not the
            # draft's own reply, so it should read as a quiet aside, not
            # another decision to make.
            # Part 58 issue 21: the way OUT has to be as visible as the way
            # back in. Without naming "cancel draft" here, a paused draft
            # looked like something the user was stuck with.
            return msg(
                "previous_draft_saved",
                language,
                'Your previous {template} draft is still saved -- type "continue draft" whenever you want, '
                'or "cancel draft" to discard it.',
                template=_display_template_name(template, language),
            )
        if stage == "approved":
            return msg(
                "previous_draft_approved",
                language,
                'Your {template} draft is approved and waiting for you to confirm "lock".',
                template=_display_template_name(template, language),
            )
        if stage in ("locked", "exported"):
            return msg(
                "previous_draft_locked",
                language,
                'Your {template} draft is locked and ready to download. Say "unlock" if you need to make changes.',
                template=_display_template_name(template, language),
            )

        draft_fields = memory.get("draft_fields") or {}
        missing = sorted(template.required_field_keys() - {key for key, value in draft_fields.items() if value})
        if not missing:
            return None
        count = len(missing)
        return msg(
            "draft_saved_with_count",
            language,
            "Your {template} draft is saved ({count} detail" + ("s" if count != 1 else "") + " left) -- "
            'say "continue draft" whenever you\'re ready, or "cancel draft" to discard it.',
            template=_display_template_name(template, language),
            count=str(count),
        )

    def _reset(self, memory: dict[str, Any]) -> None:
        memory["draft_mode"] = False
        memory["draft_stage"] = None
        memory["draft_template_id"] = None
        memory["draft_fields"] = {}
        memory["draft_id"] = None
        memory["draft_awaiting_unlock_confirm"] = False
        # Part 58 issue 18: `draft_paused`/`draft_language` outlived the draft
        # they belonged to, so a NEW draft started in the same session
        # inherited the previous one's language and its paused state.
        memory["draft_paused"] = False
        memory["draft_language"] = None
        memory.pop("draft_language_hint", None)
