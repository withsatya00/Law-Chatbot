"""Controlled General Knowledge (GK) fallback for legal queries that clear
zero chunks from the Knowledge Base.

The strict-RAG guardrail (`app.core.constants.NO_VERIFIED_CONTEXT_MESSAGES`,
enforced end-to-end by `app.services.safe_decline`) stays exactly as it is:
an answer grounded in retrieved KB chunks is still never abandoned in favor
of this. This module only fires in the branch that already produces the bare
"no verified document" refusal -- i.e. retrieval found nothing usable -- and
even there it is opt-in (`settings.general_knowledge_fallback_enabled`,
default off) and gated three separate ways before anything reaches the user:

1. `eligible()` refuses outright for a question that needs a SPECIFIC
   provision (`app.intent.detector.parse_section_lookup`) or the CURRENT/
   latest state of the law (`_CURRENCY_CUE_RE`) -- an LLM's unverified
   training knowledge is exactly the wrong source for either, and the spec
   this module implements says so explicitly. It also refuses for any
   question the fixed, human-authored safety-guidance categories in
   `app.core.fallback_guidance` already cover (domestic violence, threats,
   cyber/financial fraud) -- those hardcoded helplines and steps are safer
   than a freeform LLM answer for a safety-critical situation.
2. The model is instructed to self-report a confidence level as the first
   line of its response and to never cite a specific section, Act-with-year,
   or case law -- `_CONFIDENCE_TAG_RE`/`_CITATION_LIKE_RE` below parse and
   enforce that mechanically rather than trusting prompt compliance alone: a
   `LOW` self-rating, a missing tag, or any citation-shaped text still in the
   answer discards the whole response and the caller falls back to the
   ordinary strict-RAG refusal.
3. Whatever survives is never presented as if it came from the Knowledge
   Base: `label_and_disclaim()` prepends a fixed "General Legal Knowledge"
   header and appends a fixed disclaimer, in the user's language, and the
   caller (`ChatService`) marks the response with `general_knowledge_used`
   rather than folding it into a grounded answer.
"""

import re

import structlog

from app.core.config import settings
from app.core.constants import general_knowledge_disclaimer, general_knowledge_label
from app.core.fallback_guidance import detect_category
from app.intent.detector import parse_section_lookup
from app.llm.base import ChatMessage, LLMProvider
from app.llm.deadline import call_with_hard_timeout

log = structlog.get_logger(__name__)

# Cues that the user wants the CURRENT/latest state of the law, or names a
# recent change -- deliberately broad and best-effort: a false positive here
# only costs the (already-safe) strict-RAG refusal instead of a GK answer, a
# false negative would let unverified, possibly-stale LLM knowledge answer a
# question that specifically asked "is this still the law".
_CURRENCY_CUE_RE = re.compile(
    r"\b(latest|current(ly)?|up[- ]?to[- ]?date|recently|newly|updated?)\b"
    r"|\b(recent|new|latest)\s+amendment\b"
    r"|\bas\s+of\s+(today|now|this\s+year|\d{4})\b"
    r"|अभी\s*का|नया\s*(कानून|संशोधन)|ताज़ा|हालिया|वर्तमान\s*कानून"
    r"|\b(latest|naya|vartaman|abhi\s*ka)\s*kanoon\b",
    re.IGNORECASE,
)

# Fabrication guard: matches ANY text shaped like a specific legal citation --
# a section/clause number, an Act-with-year, a law-report citation, or a
# "X v. Y" case name. The system prompt already tells the model never to
# produce these; this is the mechanical backstop for when it does anyway.
_CITATION_LIKE_RE = re.compile(
    r"\bsection\s+\d|\bsec\.?\s*\d|\bs\.\s*\d{1,4}\b|धारा\s*\d|कलम\s*\d"
    r"|\b(act|sanhita|adhiniyam|code|संहिता|अधिनियम)[,]?\s*(19|20)\d{2}\b"
    r"|\bAIR\s+(19|20)\d{2}\b|\bSCC\b|\(\d{4}\)\s*\d+\s*SCC"
    r"|\b[A-Z][a-zA-Z.]+\s+v\.?s?\.?\s+[A-Z][a-zA-Z.]+\b",
    re.IGNORECASE,
)

_CONFIDENCE_TAG_RE = re.compile(r"^\s*CONFIDENCE\s*:\s*(HIGH|MEDIUM|LOW)\s*\n+", re.IGNORECASE)

# Self-reported confidence levels the model may still be used at.
_ACCEPTED_CONFIDENCE_LEVELS = {"high", "medium"}

_SYSTEM_PROMPT = (
    "You are answering a legal question that this application's verified Knowledge Base has NO "
    "matching document for. You may answer ONLY from general legal knowledge, under strict rules:\n\n"
    "1. Never state or imply that your answer is verified, official, or from a Knowledge Base.\n"
    "2. Never cite a specific section number, clause number, Act name with its year, statute, "
    "regulation, or case law/judgment -- not even one you believe is correct. Speak only in general "
    "terms (e.g. 'this is generally covered under Indian consumer protection law', never 'Section 12 "
    "of the Consumer Protection Act, 2019').\n"
    "3. Never fabricate a citation, case name, or numbered provision under any circumstance.\n"
    "4. If the question needs the CURRENT or latest legal position, a specific provision, or anything "
    "you are not confident is still accurate, say plainly that you cannot verify this without an "
    "authoritative source, instead of guessing.\n"
    "5. Keep the answer concise, practical, and in the same language/style as the question.\n\n"
    "Output format (mandatory): the FIRST line must be exactly 'CONFIDENCE: HIGH', "
    "'CONFIDENCE: MEDIUM', or 'CONFIDENCE: LOW' -- your genuine confidence that the general answer "
    "below is accurate and safe to show unverified. Rate LOW whenever rule 4 applies. Then a blank "
    "line, then the answer itself with no other preamble."
)


def requires_current_or_specific_provision(question: str) -> bool:
    """True when the query needs an authoritative, current answer that this
    fallback must never attempt from an LLM's unverified training data."""
    if parse_section_lookup(question):
        return True
    return bool(_CURRENCY_CUE_RE.search(question or ""))


def is_urgent_safety_category(question: str) -> bool:
    """True when `app.core.fallback_guidance`'s fixed, reviewed safety
    categories (domestic violence / threats / cyber-financial fraud) already
    cover this question -- their hardcoded helplines and steps are safer
    than a freeform LLM answer here, so GK fallback defers to them."""
    return detect_category(question) is not None


def eligible(question: str) -> bool:
    """Whether `question` may even be attempted through GK fallback.

    Does not by itself guarantee a GK answer is produced -- the model can
    still self-rate LOW confidence or produce a citation-shaped answer,
    either of which discards the result just as this gate would have.
    """
    if not (question or "").strip():
        return False
    if requires_current_or_specific_provision(question):
        return False
    return not is_urgent_safety_category(question)


def _strip_confidence_tag(text: str) -> tuple[str, str | None]:
    match = _CONFIDENCE_TAG_RE.match(text or "")
    if not match:
        return (text or "").strip(), None
    return text[match.end():].strip(), match.group(1).lower()


async def general_knowledge_answer(llm: LLMProvider, question: str, language: str) -> str | None:
    """A labeled, disclaimed general-knowledge answer, or `None`.

    `None` means the caller should fall back to the ordinary strict-RAG
    refusal -- either because this question was never eligible, the model
    itself rated its confidence LOW, the call failed/timed out, or the
    answer still contained a citation-shaped fragment despite the system
    prompt forbidding it.
    """
    if not eligible(question):
        log.info("gk_fallback_ineligible")
        return None
    log.info("gk_fallback_attempting")
    messages = [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=question),
    ]
    try:
        response = await call_with_hard_timeout(
            llm.chat(messages), fallback_seconds=settings.chat_request_budget_seconds,
        )
    except TimeoutError:
        log.warning("gk_fallback_timeout")
        return None
    if response.error:
        log.warning("gk_fallback_llm_error", error_kind=response.error_kind)
        return None
    body, level = _strip_confidence_tag(response.content)
    if level not in _ACCEPTED_CONFIDENCE_LEVELS:
        log.info(
            "gk_fallback_rejected", reason="low_or_missing_confidence", level=level,
            content_head=response.content[:120],
        )
        return None
    if not body:
        # Confidence tag parsed but nothing followed it -- logged (unlike a
        # bare early return) because this is otherwise indistinguishable
        # from the flag being off or the question being ineligible: all
        # three produce a silent `None` with zero telemetry, confirmed live
        # to make a real production discrepancy (GK enabled, question
        # eligible, LLM responding with real content) take real effort to
        # even localize to this function.
        log.info("gk_fallback_rejected", reason="empty_body_after_confidence_tag", level=level)
        return None
    if _CITATION_LIKE_RE.search(body):
        log.warning("gk_fallback_rejected", reason="citation_like_text_detected")
        return None
    log.info("gk_fallback_accepted", level=level)
    return (
        f"{general_knowledge_label(language)}\n\n{body}\n\n{general_knowledge_disclaimer(language)}"
    )
