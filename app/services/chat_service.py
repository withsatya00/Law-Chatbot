import asyncio
import re
import time
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import structlog
from fastapi import BackgroundTasks
from structlog.contextvars import get_contextvars

from app.cache.redis_client import CACHE_UNAVAILABLE_ERRORS, redis_client
from app.cache.response_cache import ResponseCache
from app.core.config import settings
from app.core.constants import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    LEGAL_DISCLAIMER,
    capability_overview,
    is_no_verified_context,
    language_preference_acknowledgement,
    no_verified_context_message,
    unverified_source_disclosure,
)
from app.core.exceptions import BadRequestError, ForbiddenError
from app.core.fallback_guidance import (
    actionable_guidance,
    detect_category,
    generic_next_steps_guidance,
)
from app.core.gk_fallback import general_knowledge_answer
from app.drafting.conversation import DraftConversationEngine, DraftTurnResult
from app.drafting.intent import looks_informational
from app.drafting.templates import get_template, list_templates
from app.drafting.title_translations import localized_document_name
from app.drafting.wrapper_messages import msg
from app.entity_extraction.extractor import EntityExtractor
from app.intent.classifier import (
    CONVERSATION_INTENT_ACTIONS,
    CONVERSATION_INTENT_HINTS,
    ConversationIntentClassifier,
    ConversationIntentMatch,
)
from app.intent.detector import IntentDetector, parse_section_lookup
from app.language.detector import LanguageDetector, extract_requested_language
from app.language.explanation_level import EXPLANATION_LEVEL_INSTRUCTIONS, detect_explanation_level
from app.language.typo_tolerance import normalize_for_routing
from app.llm.base import ChatMessage, LLMResponse
from app.llm.deadline import call_with_hard_timeout, deadline, remaining_seconds
from app.llm.factory import LLMFactory
from app.llm.prompts import prompt_registry
from app.llm.safety import failure_category, looks_like_llm_error, safe_llm_text
from app.memory import entity_memory
from app.memory.store import ConversationMemoryStore
from app.observability.metrics import metrics
from app.rag.answer_quality import QualityVerdict
from app.rag.answer_quality import Severity as AnswerSeverity
from app.rag.answer_quality import evaluate as evaluate_answer_quality
from app.rag.chat_confidence import (
    _confidence_fields,
    confidence_label,
    confidence_reason,
    confidence_score,
)
from app.rag.citation import LegalCitationEngine, citation_from_metadata
from app.rag.jurisdiction import annotate_answer as annotate_jurisdiction
from app.rag.jurisdiction import jurisdiction_directive, jurisdiction_warnings
from app.rag.kb_jurisdiction import (
    REVIEW_NEEDS_REVIEW,
    detect_jurisdiction_ambiguity,
    detect_version_ambiguity,
    jurisdiction_or_branches,
    shared_retrieval_filters,
    temporal_filter,
)
from app.rag.matter_context import (
    MatterContext,
    clarification_question,
    disclosure_note,
    resolve_matter_context,
)
from app.rag.relevance import (
    _MIN_ACCEPTED_CONTEXT_SCORE,
    filter_relevant_context,
    is_cheque_notice_timing_query,
    is_ni_notice_period_evidence,
    is_relevant_chunk,
    most_relevant_chunk,
    reorder_by_relevance,
)
from app.rag.reranker import LegalReranker
from app.rag.retriever import LegalRetriever
from app.rag.statute_currency import annotate_answer, currency_directive, currency_notice
from app.rag.statutory_periods import annotate_periods, period_warnings
from app.recommendation.engine import LawyerRecommendationEngine, render_recommendation
from app.repositories.analytics import (
    IntentEventRepository,
    IntentFeedbackRepository,
    QueryLogRepository,
)
from app.repositories.chat_history import ChatRepository
from app.repositories.users import UserRepository
from app.schemas.analysis import EntityResponse, IntentResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import (
    LawyerRecommendation,
    RetrievedChunk,
    SourceCitation,
    evidence_pages_from_citations,
)
from app.schemas.document import DocumentAnalysisRequest, DocumentAnalysisResponse
from app.services.chat_support.analytics import log_query, log_routing_decision
from app.services.document_service import DocumentService
from app.services.kb_gap_autofetch import GapAutoFetchService
from app.services.phase3 import PreferenceService
from app.services.safe_decline import clear_support, prune_unreferenced_sources, sanitize_payload
from app.services.workflow_orchestrator import detect_chain, map_facts_to_draft_fields
from app.utils.pii import mask_pii
from app.utils.prompt_security import PromptInjectionScanner

log = structlog.get_logger(__name__)


def _machine_verification_disclosure(language: str, sources: list[SourceCitation]) -> str | None:
    if not any(source.verification_status == "machine_verified" for source in sources):
        return None
    if language == "hindi":
        return "स्रोत की आधिकारिक सरकारी प्रति से स्वचालित जाँच हुई है; मानव कानूनी समीक्षा नहीं हुई है।"
    if language == "hinglish":
        return "Source official government copy se automatically verify hua hai; human legal review nahi hua hai."
    return "The source was automatically verified against an official government copy; it has not had human legal review."

# Kept as a module-level constant (rather than inlined) so tests can shrink it
# to exercise the timeout path without actually waiting several seconds.
# Lowered from 6.0s (Part 28 perf sprint): related questions are purely
# supplementary UI content, now fired concurrently with the trailing
# cache/memory/history writes rather than fully serial on the critical path
# (see the `related_questions_task` handling in `answer()`), but a still-slow
# call would otherwise blow well past the <4s "Information Question" target
# on its own -- better to return an empty list than eat that budget.
RELATED_QUESTIONS_TIMEOUT_SECONDS = 3.0

# QA (session 2026-09-11/12, `docs/qa/QA_TEST_MATRIX_20260911.md`, BUG-013a
# and the "New latency finding"): `chat_request_budget_seconds`/
# `draft_generation_budget_seconds` are an ambient, advisory deadline
# (`app/llm/deadline.py`) -- honoured only by call sites that check it, and
# a live repro showed even a call site that DOES check it (`GeminiProvider.
# chat()`) still stalling 250-317s against a ~100s budget it had computed
# itself, because cancelling an in-flight call is not the same as it
# actually stopping on time. `handle_turn` below is the one place a real,
# unconditional ceiling applies: whatever happens underneath, the caller
# gets a definite, clearly-labelled response within the budget plus this
# fixed grace window -- never an indefinitely hanging connection.
_HARD_TIMEOUT_GRACE_SECONDS = 20.0
# Best-effort guard against two requests for the SAME session running the
# full pipeline concurrently -- the concrete risk a client retry (or a
# resend of the identical question, see BUG-013a) creates if the ORIGINAL,
# stalled request is still running server-side: both could independently
# persist a draft or a chat-history entry for what the user experiences as
# one logical request. TTL always exceeds `handle_turn`'s own hard ceiling,
# so a lock can never outlive the request it guards, and a Redis outage
# degrades to "no guard" (same as every other best-effort cache use in this
# codebase) rather than ever wedging a session shut.
_TURN_LOCK_KEY_PREFIX = "chat:turn-lock:"

# Phase 1 item 3: the ceiling a grounded answer's confidence is capped at
# when it came back in the wrong language twice. Below
# `ResponseCache.HIGH_CONFIDENCE_THRESHOLD` (0.6) on purpose, so such an
# answer is structurally uncacheable and the next asker gets a fresh
# attempt rather than a permanently-wrong-language cached reply.
_LANGUAGE_MISMATCH_CONFIDENCE_CAP = 0.4
# Jurisdiction Routing (Phase 2), objective item 5: NO document in this KB
# carries transition/savings-clause metadata, so for EVERY historical
# (as_of_date-resolved) answer -- not only when `detect_version_ambiguity`
# literally finds two conflicting versions -- whether such a clause changes
# the outcome for the user's exact facts is unverified. Keeps the label at
# "Medium" (`_confidence_label`'s >=0.6 threshold for "High") rather than
# silently letting a single-version historical answer read as fully settled.
_HISTORICAL_TRANSITION_UNVERIFIED_CONFIDENCE_CAP = 0.55
# The strictly worse case: `detect_version_ambiguity` found the KB itself
# offering more than one candidate version for the same provision on this
# date -- genuine internal disagreement, not just an unverified caveat.
# Pushed into "Low" (`_confidence_label`'s <0.35) rather than reusing the
# lighter cap above, since this is a stronger, KB-evidenced uncertainty.
_HISTORICAL_VERSION_AMBIGUITY_CONFIDENCE_CAP = 0.3
# Fixed confidence for a controlled General Knowledge fallback answer
# (`app.core.gk_fallback`, see the `no_verified_context` branches below).
# Deliberately below `_HISTORICAL_VERSION_AMBIGUITY_CONFIDENCE_CAP` and
# `ResponseCache.HIGH_CONFIDENCE_THRESHOLD` -- this answer is not grounded in
# any retrieved source at all, only the model's own unverified training
# knowledge, so "Low" (`_confidence_label`'s <0.35) is the only honest label
# for it regardless of how confident the model itself claimed to be.
_GENERAL_KNOWLEDGE_CONFIDENCE = 0.25
# QA pass 2026-09-24 ("LLM non-deterministic refusal despite good context",
# QA_REPORT_100Q_FINAL_20260924.md section 7.2): `safe_decline.py`'s own
# module docstring already documents that the LLM itself, not retrieval, can
# non-deterministically emit Rule 2's exact refusal line even when `ranked`
# holds genuinely on-topic, well-scored context -- confirmed live (same
# question grounded on 1 of 3 identical `/chat` calls for T026, 0 of 2 for
# T018 Sanskrit) AFTER the retrieval-layer bug those same questions also had
# was independently fixed and verified via direct pipeline replication. This
# is the deterministic, code-level check gating a bounded retry (see
# `_answer_with_grounding_retry`) rather than silently accepting the
# refusal: comfortably above `relevance._MIN_ACCEPTED_CONTEXT_SCORE` (0.12)
# and `relevance._CROSS_LINGUAL_CONTEXT_SCORE` (0.24) -- both paired with a
# real relevance-gate pass already, so a chunk clearing THIS bar has already
# survived the gate AND scored well past its minimum -- and comfortably below
# `retriever._EXACT_SECTION_SCORE_FLOOR` (0.65, an authoritative exact-match
# floor). Matches the actual score range observed for the confirmed
# false-refusal cases (0.38-0.58 post-rerank) with headroom, not tuned to
# those exact numbers.
_STRONG_GROUNDING_RETRY_SCORE = 0.30


# Part 58 issue 24: how an internal context block refers to itself when the
# model leaks the label into the answer. Covers the bare "Source 2", the
# bracketed "[Source 2]", and the Hindi/Hinglish forms an answer written in
# those languages produces ("Source 2 ke mutabiq", "स्रोत 2 के अनुसार").
_SOURCE_ORDINAL_PATTERN = re.compile(
    r"\[?\s*(?:source|स्रोत|स्त्रोत|सोर्स)\s*[-#:]?\s*(\d{1,2})\s*\]?",
    re.IGNORECASE,
)
# What a leaked ordinal becomes when the chunk it pointed at carries no
# usable citation metadata at all -- vague, but honest, and never a number
# the reader cannot resolve.
_UNNAMED_SOURCE_LABELS: dict[str, str] = {
    "english": "the retrieved source material",
    "hindi": "प्राप्त स्रोत सामग्री",
    "hinglish": "retrieved source material",
}


def _source_label(chunk: RetrievedChunk) -> str:
    """A human-readable citation for one retrieved chunk -- "Bharatiya Nagarik
    Suraksha Sanhita, 2023, Section 173 (bnss_2023.pdf)" -- built from
    whichever of `act_name`/`section_number`/`article_number`/`chapter`/
    `source_document` the chunk's metadata actually carries. Falls back to
    the document filename, and finally to a generic phrase, rather than
    inventing an Act name.
    """
    metadata = getattr(chunk, "metadata", None) or {}
    act = metadata.get("act_name")
    parts: list[str] = []
    if act:
        parts.append(str(act))
    if metadata.get("section_number"):
        parts.append(f"Section {metadata['section_number']}")
    if metadata.get("article_number"):
        parts.append(f"Article {metadata['article_number']}")
    if not parts and metadata.get("chapter"):
        parts.append(str(metadata["chapter"]))
    document = metadata.get("source_document") or metadata.get("source")
    if parts:
        return f"{', '.join(parts)} ({document})" if document else ", ".join(parts)
    return str(document) if document else _UNNAMED_SOURCE_LABELS["english"]


def _replace_source_ordinals(answer: str, ranked: list[Any]) -> str:
    """Rewrites any "Source N" the model copied out of the prompt into the
    citation that block actually stands for.

    Purely a naming fix: it substitutes one reference for a more useful one
    and never touches the surrounding claim. An out-of-range ordinal (the
    model inventing "Source 9" when six blocks were supplied) degrades to the
    generic phrase rather than pointing at the wrong Act.
    """
    if not answer or "source" not in answer.lower() and "स्रोत" not in answer:
        return answer
    labels = [_source_label(chunk) for chunk in ranked]

    def _replacement(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        if 0 <= index < len(labels):
            return labels[index]
        return _UNNAMED_SOURCE_LABELS["english"]

    return _SOURCE_ORDINAL_PATTERN.sub(_replacement, answer)


def _normalize_ni_138_heading(chunk: RetrievedChunk) -> None:
    """Repair stale overlap metadata on a retrieved NI Act heading in memory."""
    metadata = chunk.metadata or {}
    identity = " ".join(
        str(metadata.get(key) or "")
        for key in ("act_name", "source_document", "source")
    ).lower().replace("_", " ")
    if "negotiable instruments" not in identity:
        return
    has_heading = bool(
        re.search(r"(?m)^[ \t]*138\.\s+Dishonour\s+of\s+cheque\b", chunk.text, re.IGNORECASE)
    )
    lowered = chunk.text.lower()
    has_provisos = (
        "within thirty days" in lowered
        and "within fifteen days" in lowered
        and "drawer" in lowered
        and "notice" in lowered
    )
    if not (has_heading or has_provisos):
        return
    chunk.metadata = {
        **metadata,
        "act_name": "Negotiable Instruments Act",
        "section_number": "138",
        "section_number_provenance": "heading",
        "source_type": "statute",
    }


def _correct_ni_notice_attribution(
    answer: str, question: str, ranked: list[RetrievedChunk]
) -> str:
    """Correct the observed s.139/s.138 notice-period misattribution.

    This fires only for a cheque-notice timing question, only when retrieved
    NI Act s.138 text itself supports the thirty-day period, and only on a
    sentence that calls s.139 the source of the requirement.  It is therefore
    a source-backed correction, not a general legal-memory rewrite.
    """
    if not is_cheque_notice_timing_query(question) or not re.search(
        r"\bsection\s+139\b", answer, re.IGNORECASE
    ):
        return answer
    supporting = [chunk for chunk in ranked if is_ni_notice_period_evidence(chunk)]
    if not any(re.search(r"\b(?:thirty|30)\s+days?\b", chunk.text, re.IGNORECASE) for chunk in supporting):
        return answer

    def replace_sentence(match: re.Match[str]) -> str:
        sentence = match.group(0)
        if not re.search(r"\b(requirement|notice|under|tahat|mutabiq)\b", sentence, re.IGNORECASE):
            return sentence
        return re.sub(r"\bSection\s+139\b", "Section 138 proviso (b)", sentence, flags=re.IGNORECASE)

    return re.sub(
        r"[^.!?\n]*\bSection\s+139\b[^.!?\n]*[.!?]?",
        replace_sentence,
        answer,
        flags=re.IGNORECASE,
    )


_CANCEL_DRAFT_PATTERN = re.compile(
    r"\b(cancel|discard)\s+(the\s+|this\s+|my\s+)?draft\b"
    # Part 38 "Draft Auto-Pause Engine" Rule 2: bare dismissal words, on
    # their own with nothing else in the message (so a genuine field value
    # that happens to just be "Done" -- unlikely but not impossible -- is
    # never at risk; a real one would virtually never be JUST one of these
    # words and nothing else). Distinct from the "pause, don't lose it"
    # behavior below: these mean "I'm done with this draft," full stop.
    r"|^\s*(please\s+)?(nothing(\s+else)?|leave it|not now|cancel|close|dismiss|exit|stop|done)\s*[.!?]*\s*$"
    # QA session 8: "start over"/"new draft" used to be unanchored substring
    # matches above, so ANY message that merely mentioned "new draft" as
    # part of a longer, specific request -- e.g. "Start a new draft: I want
    # to write a police complaint..." -- was misread as a CANCEL of the
    # current draft instead of a request to park it and start the named one.
    # Confirmed live: that exact message produced "Okay, I've discarded that
    # draft," discarding the active draft and never starting the requested
    # one. Anchored here to a BARE "start over"/"new draft" with nothing
    # else, matching every other dismissal phrase in this pattern -- a
    # message that combines this phrasing with an actual request is handled
    # by `_NEW_DRAFT_PATTERN` (app/drafting/conversation.py), never as a
    # plain cancel.
    r"|^\s*(please\s+)?(start over|new draft|another draft)\s*[.!?]*\s*$"
    # Phase 1 item 4: the bare-word branch above only matched "cancel" ALONE,
    # so the Hinglish imperative that actually gets typed -- "cancel kr do",
    # "cancel karo", "band karo" -- named no object and matched nothing. The
    # object is implicit in Hinglish; there is only one thing being cancelled.
    r"|^\s*(please\s+)?(cancel|discard|delete|remove|band|bnd|hata|hatao|rehne|chhod|chod)\s*"
    r"(kar|kr|karo|kro|kar\s*do|kr\s*do|do|de|den|dijiye|dena)?\s*(do|dijiye)?\s*[.!?]*\s*$"
    # Part 58 "Answer Quality Audit" issue 17: the two branches above only
    # recognised the exact English phrase "cancel draft" (verb immediately
    # before the noun) or a message that was nothing but "cancel". Real
    # users of a Hindi/Hinglish product do not type either. Confirmed live:
    # "cancel kr do draft ko" -- unambiguous, in the conversation's own
    # language -- matched neither, fell through to the conversation
    # classifier, was scored by `DraftIntentDetector` as a request to draft
    # some unnamed document (it contains "draft" AND the imperative "kr
    # do"), and came back as the full 56-template menu with the draft still
    # active. These alternatives accept the verb on EITHER side of the noun,
    # in English, Hinglish and the Indic scripts this product replies in.
    #
    # "rehne" is deliberately NOT in this proximity-based word list (removed
    # QA pass 2026-09-11, BUG-012 -- it IS still in the bare-word branch
    # above, where the risk is much lower). "rehne do" is genuinely
    # ambiguous in Hindi/Hinglish: it can mean "leave it, forget it" (the
    # dismissal sense this pattern exists for) OR "let it remain/stay as it
    # is" -- the OPPOSITE, a preservation instruction. Confirmed live: "Ye
    # draft mujhe Hinglish mein samjha do, lekin draft Hindi mein hi rehne
    # do, badlo mat." ("explain this draft to me in Hinglish, but let the
    # draft STAY in Hindi, don't change it") -- an explanation request that
    # explicitly asked NOT to modify the draft -- matched this pattern
    # purely because "draft" and "rehne" both appeared within 20 characters
    # of each other, and the entire draft (every edit accumulated across the
    # conversation) was deleted. In the bare-word branch, "rehne do" alone
    # with nothing else IS a safe, narrow context to read as dismissal (the
    # same rationale Part 38 already used); loosely near the word "draft"
    # anywhere in an arbitrary, possibly long sentence is not.
    r"|\b(?:draft|drafting|document|मसौदा|मसुदा|ड्राफ्ट|ડ્રાફ્ટ|খসড়া|வரைவு|డ్రాఫ్ట్|ಡ್ರಾಫ್ಟ್|ഡ്രാഫ്റ്റ്|ڈرافٹ)\b"
    r"[^\n]{0,20}?\b(?:cancel|discard|delete|remove|hata|hatao|band)\b"
    r"|\b(?:cancel|discard|delete|remove)\b[^\n]{0,20}?"
    r"\b(?:draft|drafting|document|मसौदा|मसुदा|ड्राफ्ट|ડ્રાફ્ટ|খসড়া|வரைவு|డ్రాఫ్ట్|ಡ್ರಾಫ್ಟ್|ഡ്രാഫ്റ്റ്|ڈرافٹ)\b"
    # Native-script cancel verbs, which carry the intent on their own -- a
    # Hindi/Marathi/Gujarati/Bengali speaker writing "रद्द करें" or "રદ કરો"
    # mid-draft means exactly one thing.
    r"|रद्द\s*कर|निरस्त\s*कर|रद्द\s*करा|हटा\s*द|मिटा\s*द"
    r"|રદ\s*કર|બંધ\s*કર|বাতিল\s*কর|ரத்து\s*செய்|రద్దు\s*చేయ|ರದ್ದು\s*ಮಾಡ|റദ്ദാക്ക|منسوخ\s*کر",
    re.IGNORECASE,
)

# Part 58 issue 20: a bare "no" mid-draft. The user is declining to keep
# filling in fields -- not answering the pending field, and not necessarily
# throwing the draft away either. Previously this matched nothing at all, so
# it fell through to `_process_collecting_message`, which re-printed the same
# nine-field list it had just printed; the user said "no" and got the exact
# reply they were saying no TO. Handled as a pause (see
# `_respond_with_draft_paused`): the draft survives, but it stops claiming
# every subsequent message. Whole-message anchored, so a genuine field value
# containing "no" ("Nohar, Rajasthan") is unaffected.
_BARE_REFUSAL_PATTERN = re.compile(
    r"^\s*(?:no|nope|nah|naa+h?|not now|no thanks|no thank you"
    r"|nahi|nahin|nhi|nai|na|nako|nakko"
    r"|नहीं|नही|ना|नको|मत|ન|નહીં|না|নয়|இல்லை|வேண்டாம்|కాదు|వద్దు|ಇಲ್ಲ|ಬೇಡ|ഇല്ല|വേണ്ട|ਨਹੀਂ|ନାହିଁ|نہیں|نہ)"
    r"\s*[.!।॥]*\s*$",
    re.IGNORECASE,
)

_MODIFICATION_TRANSFORM_LABELS = {
    "summarize": "Summarize the text below concisely.",
    "shorten": "Shorten the text below (respect any specific word/length limit named in the instruction).",
    "expand": "Expand the text below with more depth and detail.",
    "simplify": "Rewrite the text below in simple, beginner-friendly language.",
    "bullets": "Convert the text below into clear bullet points.",
    "rewrite": "Rewrite the text below in the style named in the instruction.",
    "table": "Convert the comparable parts of the text below into a markdown table.",
    "examples": "Add real-life or practical examples to the text below.",
    "timeline": "Restructure the text below as a clear step-by-step timeline/procedure.",
    "visual": (
        "Restructure the text below as a text-based visual outline (indentation and arrows to show "
        "structure/flow) -- this is a chat reply, so it can't render an actual diagram."
    ),
}

# Conversation intents confident/specific enough that a message matching one
# mid-draft is almost certainly a genuine unrelated question, not an attempt
# to answer the next pending field (e.g. "what is FIR" vs. a police-station
# name) -- so these interrupt the draft flow instead of being swallowed into
# it. Deliberately excludes the low-confidence catch-all buckets (`General
# Legal Information`'s 0.4 fallback, `Legal Dictionary`'s 0.5 bare-term-lookup
# path) and `General Conversation`, whose greeting-only pattern overlaps with
# the "approve this draft" edit command (e.g. "perfect", "great").
_DRAFT_INTERRUPT_INTENTS = {
    "Lawyer Recommendation",
    "Legal Dictionary",
    "Law Comparison",
    "Legal Procedure",
    "Document Analysis",
    "Legal Explanation",
    "Summarization",
    "Translation",
    # Part 31: a pending failure is about a DIFFERENT, already-abandoned
    # request, not the in-progress draft -- letting it interrupt (rather
    # than being swallowed as a field answer) mirrors "Lawyer Recommendation"
    # above.
    "Retry Failed Request",
    # "What can you do?" is never a police station name or an address -- it
    # requires all three of an interrogative, a second-person reference and an
    # ability word, which no field value carries.
    "Capability Question",
    # "Language Preference" is deliberately NOT here: an active draft has
    # its own language handling (`DraftConversationEngine` switches the
    # draft's language and re-asks the pending field in it), which is a
    # better answer than interrupting the draft to acknowledge a
    # preference. The standalone case, with no draft open, is routed to
    # "Language Preference" by the classifier and never reaches here.
    # "Legal Advice" is deliberately NOT here (Part 32 fix; it used to be)
    # -- `_LEGAL_ADVICE_PATTERN` matches ANY first-person narrative ("mera",
    # "mujhe", "my", "I received"...), which is exactly what a genuine
    # answer to a "Facts of the Case"/"reason for complaint" field looks
    # like. Unconditionally interrupting on it meant pasting real field
    # content (e.g. "Mera makan malik mera deposit nahi de raha hai...")
    # into an active draft got silently answered as a fresh legal-advice
    # question instead of being collected as field data, and the draft
    # never progressed. Falling through to the `looks_informational()`
    # check below (same as the other catch-all buckets) still lets a
    # genuine embedded question interrupt -- just not a bare declarative
    # narrative that reads exactly like the field it's answering.
}
# Part 30 "Memory-First Routing Engine" logging: maps the six conv-intent
# branches that share `_finalize_intent_response` to the routing-decision
# log's `route` slug and whether they count as a "memory hit" (answered from
# `memory["messages"]`/`memory["summary"]` rather than fresh retrieval).
# Lawyer Recommendation and General Conversation are deterministic/small-talk
# replies, not memory recall, so they're `False` here.
_ROUTE_SLUG_BY_CONV_INTENT = {
    "General Conversation": "general_conversation",
    "Lawyer Recommendation": "lawyer_recommendation",
    "Translation": "translation",
    "Response Modification": "response_modification",
    "Conversation Memory": "conversation_memory",
    "Summarization": "conversation_summary",
    "Language Preference": "language_preference",
    "Typo Clarification": "typo_clarification",
}
_MEMORY_HIT_CONV_INTENTS = {"Translation", "Response Modification", "Conversation Memory", "Summarization", "Entity Memory"}

# The bot's own two clarification replies from `_respond_with_translation`'s
# ambiguous branch. `memory.append` stores these as ordinary assistant
# messages, same as any real answer -- so when the user replies "Hindi" to
# "Which language would you like this translated into?", a naive "most
# recent assistant message" search finds this clarification QUESTION, not
# the substantive answer the user actually meant to translate, and hands the
# LLM a nonsensical translation target (confirmed root cause of a live
# regression: the LLM, given its own "translate only when explicitly
# requested" question as the text to translate, replied by parroting that
# instruction back instead of translating anything). Skipped when scanning
# backwards for the real last reply in `_respond_with_translation` below.
_TRANSLATION_NO_TARGET_MESSAGE = "Which language would you like this translated into?"
_TRANSLATION_NO_PRIOR_REPLY_MESSAGE = "There's no earlier reply yet to translate — could you share the text you'd like translated?"
_TRANSLATION_CLARIFICATION_MESSAGES = {_TRANSLATION_NO_TARGET_MESSAGE, _TRANSLATION_NO_PRIOR_REPLY_MESSAGE}

# Part 51 "Uploaded Document Conversation Context": language-matched replies
# for `_respond_with_document_analysis`, same style as `entity_memory._NO_
# MEMORY_ANSWERS` -- keyed by the same lowercase language strings
# `LanguageDetector`/`ConversationMemoryStore` already use, falling back to
# English for any language not listed.
_NO_UPLOADED_DOCUMENT_MESSAGES: dict[str, str] = {
    "english": "Which PDF or document would you like explained? Please upload it first.",
    "hindi": "Kaunsi PDF/file explain karni hai? Pehle document upload kar dijiye.",
    "hinglish": "Kaunsi PDF/file explain karni hai? Pehle document upload kar dijiye.",
}
_DOCUMENT_ACCESS_DENIED_MESSAGES: dict[str, str] = {
    "english": "I couldn't access that document — it doesn't belong to this session or account.",
    "hindi": "Main us document ko access nahi kar saka — yeh is session ya account ka nahi hai.",
    "hinglish": "Main us document ko access nahi kar paaya — yeh is session ya account ka nahi hai.",
}
# A document pasted inline into a message (see `_maybe_answer_from_inline_document`).
_INLINE_QUOTE_PATTERN = re.compile(r"[“\"]([^”\"]{100,})[”\"]", re.DOTALL)
_INLINE_DOC_CUE_PATTERN = re.compile(r"\b(excerpt|agreement|contract|clause|document|deed|lease|extract)\b", re.IGNORECASE)
_INLINE_FOLLOWUP_PATTERN = re.compile(
    r"\b(clause\s*\d+|excerpt|(is|us|yeh|ye|this|that)\s+(agreement|document|clause)|"
    r"(agreement|document)\s+(mein|me|ke\s+(andar|mutabiq|anusar))|isme|according\s+to\s+the\s+(agreement|excerpt|document))\b",
    re.IGNORECASE,
)
_INLINE_DOC_NOT_FOUND_MESSAGES: dict[str, str] = {
    "english": "The excerpt you shared does not say this, so I can't state it -- and I won't assume anything that "
    "isn't in the text you gave me.",
    "hindi": "Aapke diye gaye excerpt mein yeh baat nahi hai, isliye main ise bata nahi sakta -- aur jo text aapne "
    "diya usmein nahi hai, use main maan nahi loonga.",
    "hinglish": "Aapke diye gaye excerpt mein yeh baat nahi hai, isliye main ise bata nahi sakta -- aur jo text aapne "
    "diya usmein nahi hai, use main assume nahi karunga.",
}


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in an LLM reply (tolerates ```json fences)."""
    import json

    match = re.search(r"{.*}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


_DOCUMENT_UNAVAILABLE_MESSAGES: dict[str, str] = {
    "english": "I couldn't read that document right now. Please try uploading it again.",
    "hindi": "Main abhi us document ko padh nahi paaya. Kripya use dobara upload karein.",
    "hinglish": "Main abhi us document ko read nahi kar paaya. Please use dobara upload karein.",
}
# Phase 1 "General Clarification Mode": asked when a message names both a
# draft trigger and a document-analysis reference with no explicit
# "based on it/that" link (see `_WORKFLOW_LINK_PATTERN` in
# app/intent/classifier.py) -- genuinely ambiguous whether the draft should
# be grounded in the document review or is a second, unrelated request.
_WORKFLOW_CLARIFICATION_MESSAGES: dict[str, str] = {
    "english": "Should I review the document first and base the draft on what I find, or start drafting directly?",
    "hindi": "Pehle document ka analysis karun aur usi ke aadhar par draft banau, ya seedha draft karna shuru karun?",
    "hinglish": "Pehle PDF ka analysis karun ya directly draft karna shuru karun?",
}
# Reply keywords resolving the clarification above. "Direct" is checked
# first: a reply naming both a document AND "direct/seedha" (e.g. "no,
# directly draft, skip the PDF review") should honor the explicit skip
# request rather than the incidental document mention.
_WORKFLOW_CLARIFICATION_DIRECT_PATTERN = re.compile(r"\b(direct|directly|seedha|seedhe|skip)\b", re.IGNORECASE)
_WORKFLOW_CLARIFICATION_CHAIN_PATTERN = re.compile(
    r"\b(pdf|analy[sz]e|analysis|review|pehle|first)\b", re.IGNORECASE
)


def _resolve_workflow_clarification_choice(text: str) -> str | None:
    if _WORKFLOW_CLARIFICATION_DIRECT_PATTERN.search(text):
        return "direct"
    if _WORKFLOW_CLARIFICATION_CHAIN_PATTERN.search(text):
        return "chain"
    return None

# "Response Modification" (Part 24: shorten/expand/simplify/bullets/rewrite/
# table/examples/timeline/visual) is deliberately NOT in this set, mirroring
# how "Translation" is excluded in preview stage below -- mid-draft, a
# generic style command like "bullet points" or "shorter" is far more likely
# to mean "reformat my draft" (already handled by the drafting engine's own
# edit-command interpreter in `app/drafting/edit_commands.py`) than "reformat
# the last chat reply." Not wiring it up avoids a real risk of hijacking the
# existing, tested draft-editing flow for an untested new one.
# Same reasoning excludes "Conversation Memory" (Part 26): mid-draft,
# "Continue." already means "resume collecting draft fields" via
# `DraftConversationEngine`'s own `_CONTINUE_DRAFT_PATTERN`/`describe_pending`,
# and "What draft were we making?" is answered more precisely by that same
# engine than by a generic memory-recall LLM call.



def _currency_notice_for(sources: list[SourceCitation], question: str) -> str:
    """The response's currency disclosure, built from the cited sources.

    Reads only what the governance registry recorded on each chunk. It never
    infers that a source is current from the fact that it was retrieved: an
    unverified source with no recorded status produces a note saying exactly
    that, which is the honest answer and the one that tells a reader to check.
    """
    return currency_notice(
        [
            {
                "source_document": citation.source_document,
                "amendment_status": citation.amendment_status,
                "verification_status": citation.verification_status,
            }
            for citation in sources
        ],
        question=question,
    )




def _applicable_law_from(sources: list[SourceCitation]) -> list[str]:
    """The distinct provisions the cited sources actually identify.

    Built from citation METADATA, never parsed out of the generated answer: a
    model naming "Section 420 IPC" in prose is not evidence that any retrieved
    source says so, and echoing it back as `applicable_law` would launder an
    unsupported citation into a structured field that looks authoritative.

    A source that identifies no Act or section contributes nothing, so an empty
    list means "the retrieved sources did not identify a specific provision" --
    not "no law applies".
    """
    entries: list[str] = []
    for citation in sources:
        if not citation.act_name:
            continue
        if citation.section:
            entry = f"{citation.act_name} — Section {citation.section}"
        elif citation.article:
            entry = f"{citation.act_name} — Article {citation.article}"
        else:
            entry = citation.act_name
        if entry not in entries:
            entries.append(entry)
    return entries


class ChatService:
    def __init__(self) -> None:
        self.language_detector = LanguageDetector()
        self.intent_detector = IntentDetector()
        self.entity_extractor = EntityExtractor()
        self.retriever = LegalRetriever()
        self.reranker = LegalReranker()
        self.recommendations = LawyerRecommendationEngine()
        self.memory = ConversationMemoryStore()
        self.history = ChatRepository()
        self.query_log = QueryLogRepository()
        self.intent_events = IntentEventRepository()
        self.intent_feedback = IntentFeedbackRepository()
        self.response_cache = ResponseCache()
        self.preferences = PreferenceService()
        # Chat answers need the same retry/failover contract as drafting.
        # The request-wide deadline below keeps the combined attempts bounded.
        self.llm = LLMFactory.create_resilient()
        self.prompt_scanner = PromptInjectionScanner()
        self.draft_conversation = DraftConversationEngine()
        self.conversation_classifier = ConversationIntentClassifier(draft_detector=self.draft_conversation.intent_detector)
        self.document_service = DocumentService()
        self.citation_engine = LegalCitationEngine()

    async def _run_shared_preflight(
        self, request: ChatRequest, authenticated_user_id: str | None,
    ) -> tuple[str, dict[str, Any]]:
        """Phase 2B: the only sequence `answer()` and `answer_stream()` run
        identically, in the same order, before either makes a routing
        decision -- prompt-injection scanning, session id resolution, and
        the session-ownership check (Part 46; `check_access` also loads
        memory, so its return doubles as a load).

        Deliberately NOT extended to cover language detection, conversation
        classification, or draft-interruption detection even though both
        methods perform all three: `answer()` runs them after a real,
        persisted `memory.append()` of the user's message, while
        `answer_stream()` runs them first, against a local *preview* of
        memory it never persists, specifically so it can fall back to
        calling `answer()` outright (for drafting/translation/etc.) without
        that call's own `memory.append()` double-recording the same turn --
        see `answer_stream`'s docstring. Folding those steps into this same
        preflight would force one order on both callers and risk exactly
        that double-append on the fallback path, so they stay in place,
        unchanged, in each method.
        """
        risky, findings = self.prompt_scanner.scan(request.question)
        if risky:
            raise BadRequestError("The request contains unsafe prompt-injection instructions.", {"findings": findings})
        session_id = request.session_id or str(uuid4())
        # Part 46 "Authenticated User Ownership": reject before touching
        # anything if this session already belongs to a different account
        # (or an anonymous request is trying to reach an account-owned
        # session) -- an unclaimed session is unaffected (open, as always).
        memory = await self.memory.check_access(session_id, authenticated_user_id)
        return session_id, memory

    async def _log_intent_event(
        self, session_id: str, event: dict[str, Any], owner_user_id: str | None = None,
    ) -> None:
        """Phase 1 "Intent Analytics": single call site for every
        classification-event write -- dual-writes to the existing
        session-scoped `intent_history` (`ConversationMemoryStore.
        append_intent_event`, capped at 50, used for within-conversation
        state) AND the new cross-session `intent_events` collection
        (`IntentEventRepository`, unbounded, what `AnalyticsService.
        dashboard()`'s intent-analytics fields read from). The persistent
        write is best-effort -- an analytics write must never break a real
        chat turn, matching `ConversationMemoryStore._persist`'s own
        `long_term_memory_write_failed` fallback for the same reason.
        """
        append_intent_event = getattr(self.memory, "append_intent_event", None)
        if append_intent_event is not None:
            await append_intent_event(session_id, event)
        # Part 58 issue 25: `intent_events` is explicitly the UNBOUNDED,
        # cross-session collection (the session-scoped copy above is capped at
        # 50 and expires with the session), so the message text it carries
        # outlives the conversation by design. Redacted here rather than at
        # the call sites, so no future caller can forget.
        event = {**event, "question": mask_pii(event.get("question", ""))}
        if event.get("corrected_text"):
            event["corrected_text"] = mask_pii(event["corrected_text"])
        try:
            # Security finding C2: `intent_events` never carried a `user_id`
            # field at all until this fix, so `erase_user_data`'s
            # `IntentEventRepository().delete_by_user(user_id)` silently
            # matched nothing for anyone -- same symptom, same root cause
            # (a client-account-scoped deletion query with no matching
            # server-verified field to filter on) as the `chats`/
            # `query_logs` fix above.
            await self.intent_events.insert({**event, "session_id": session_id, "user_id": owner_user_id})
        except Exception as exc:  # noqa: BLE001 - analytics write must never break a chat turn
            log.warning("intent_event_write_failed", session_id=session_id, error=str(exc))

    async def _dispatch_conversation_intent(
        self, request: ChatRequest, session_id: str, language: str, memory: dict[str, Any], started: float,
        background_tasks: BackgroundTasks | None, authenticated_user_id: str | None,
    ) -> tuple[ChatResponse | None, ConversationIntentMatch | None, list[str], bool]:
        """Phase 2C: branch-selection/dispatch only, moved verbatim out of
        `answer()` -- every `_respond_with_*` handler is unchanged, called
        exactly as it was, from exactly the same place in the same order.

        Returns `(response, conv_intent, suggested_actions, interrupting_draft)`.
        `response` is not None iff one of these early-return branches
        matched, in which case the caller must return it immediately, as
        `answer()` used to inline. When `response` is None, no branch
        matched (including the draft-cancellation branch, which returns
        before `conv_intent` is ever computed -- callers must not use
        `conv_intent`/`suggested_actions`/`interrupting_draft` in that case)
        and the caller continues into the default RAG/general-knowledge
        path using the other three return values exactly as `answer()`
        used to compute them inline.
        """
        draft_was_active = bool(memory.get("draft_mode"))

        if memory.get("pending_clarification") == "section_lookup_act":
            resolved_response = await self._resolve_section_lookup_act_clarification(
                request, session_id, language, memory, started, background_tasks, authenticated_user_id,
            )
            if resolved_response is not None:
                return resolved_response
            # Reply didn't name one of the Acts that were offered (an
            # unrelated new message) -- expire the stale pending state and
            # fall through to normal classification of THIS message.
            memory["pending_clarification"] = None
            memory.pop("pending_section_query", None)

        if memory.get("pending_clarification") == "workflow_intent":
            resolved_response = await self._resolve_workflow_clarification(
                request, session_id, language, memory, started, background_tasks, authenticated_user_id,
            )
            if resolved_response is not None:
                return resolved_response
            # Reply didn't answer the clarification (an unrelated new
            # message) -- expire the stale pending state (one-turn-only,
            # mirroring `pending_clarification="translation_target"`) and
            # fall through to normal classification of THIS message.
            memory["pending_clarification"] = None
            memory.pop("pending_workflow_question", None)

        if draft_was_active and _CANCEL_DRAFT_PATTERN.search(request.question):
            response = await self._respond_with_draft_cancelled(
                request, session_id, language, memory, started, background_tasks
            )
            return response, None, [], False

        # Part 58 issue 20: checked right after cancellation and before the
        # conversation classifier, for the same reason -- a bare "nhi" carries
        # no topical content for the classifier to work with, so left to fall
        # through it would be swallowed by the draft engine as a field answer.
        if draft_was_active and _BARE_REFUSAL_PATTERN.match(request.question.strip()):
            response = await self._respond_with_draft_paused(
                request, session_id, language, memory, started, background_tasks
            )
            return response, None, [], False

        # --- Conversational workflow orchestration -----------------------
        # Every capability that used to need its own page or sidebar button
        # (notarization, cyber-fraud triage, jurisdiction, saved drafts,
        # exports, document verification, notary/admin actions) is reachable
        # by talking, through `app/chatops`.
        #
        # Runs BEFORE the classifier so a workflow already mid-flight keeps
        # the turn, and AFTER the draft guards above so an in-progress draft
        # still wins -- the drafting engine owns its own state machine and
        # chatops deliberately does not try to take it over.
        #
        # Returns `None` for anything it does not claim, and the rest of this
        # method then runs exactly as it did before.
        # A draft in progress normally keeps the turn: mid-collection, a
        # message is almost always a field answer, and handing it to the
        # orchestrator would lose the user's work. The exception is a message
        # that EXPLICITLY names another capability -- their saved drafts, a
        # document, a case, a notarization step. In an observed session each
        # of those was swallowed by an open draft and answered from
        # retrieval ("no verified document is available"), which is neither
        # what was asked nor recoverable. `interrupts_draft` is a
        # registry-driven check on declared workflow metadata, never a guess.
        #
        # QA session 9: a SECOND exception, alongside an explicit request --
        # a chatops workflow that is ALREADY mid-flight and holding the floor
        # (e.g. `SavedDraftsWorkflow` having just asked "Which draft would
        # you like to open?") must keep it for its own very next reply, even
        # while an unrelated draft is separately active. Without this, a bare
        # "1"/a draft's name in reply to that question named no capability
        # explicitly, so `_explicit_workflow_request` scored it 0, this whole
        # branch was skipped, and the number was instead swallowed either as
        # a field answer for the OTHER, still-collecting draft or (once that
        # other draft had nothing left to ask) misrouted to the domain-intent
        # classifier and RAG -- confirmed live both ways (session 8). Reading
        # `chatops.state.active` costs nothing extra: it is the exact same
        # stack `_dispatch_chatops` itself consults a few lines below.
        from app.chatops import state as chatops_state

        chatops_workflow_pending = chatops_state.active(memory) is not None
        if not draft_was_active or chatops_workflow_pending or self._explicit_workflow_request(request.question, language):
            workflow_response = await self._dispatch_chatops(
                request, session_id, language, memory, started, background_tasks, authenticated_user_id,
            )
            if workflow_response is not None:
                return workflow_response, None, [], False

        conv_intent = await self.conversation_classifier.classify_advanced(request.question, memory, self.llm, language)

        if conv_intent.is_correction and conv_intent.corrected_text:
            # The user is rejecting their previous request and restating what
            # they actually want (e.g. "No, I meant bail" / "Document
            # analysis nahi, legal research chahiye"). Log the original
            # (wrong) message once for audit, tagged as superseded, then
            # route the corrected request through the real handler by
            # recursing into `answer()` exactly like the "Retry Failed
            # Request" replay above -- retrieval/entity extraction/answer
            # generation must see only the corrected text, never the
            # original. The recursive call logs its own intent event for the
            # corrected text, so this single audit entry is the only write
            # for the original message -- no duplicate.
            await self._log_intent_event(
                session_id,
                {
                    "question": request.question,
                    "primary_intent": conv_intent.intent,
                    "detected_intents": list(conv_intent.detected_intents) or [conv_intent.intent],
                    "confidence": conv_intent.confidence,
                    "reason": f"correction detected; rerouted to: {conv_intent.corrected_text!r}",
                    "classifier_source": "correction",
                    "is_correction": True,
                    "corrected_text": conv_intent.corrected_text,
                },
                owner_user_id=memory.get("owner_user_id") or authenticated_user_id,
            )
            response = await self.answer(
                ChatRequest(
                    question=conv_intent.corrected_text,
                    session_id=session_id,
                    conversation_id=request.conversation_id,
                    user_id=request.user_id,
                    language=request.language,
                    metadata_filters=request.metadata_filters,
                ),
                background_tasks,
            )
            return response, conv_intent, CONVERSATION_INTENT_ACTIONS.get(conv_intent.intent, []), False

        await self._log_intent_event(
            session_id,
            {
                "question": request.question,
                "primary_intent": conv_intent.intent,
                "detected_intents": list(conv_intent.detected_intents) or [conv_intent.intent],
                "confidence": conv_intent.confidence,
                "reason": conv_intent.reason,
                "classifier_source": conv_intent.classifier_source,
                "is_correction": conv_intent.is_correction,
                "corrected_text": conv_intent.corrected_text,
            },
            owner_user_id=memory.get("owner_user_id") or authenticated_user_id,
        )
        suggested_actions = CONVERSATION_INTENT_ACTIONS.get(conv_intent.intent, [])

        if not draft_was_active:
            # Never hijack an already-in-progress draft session into a
            # fresh workflow-driven one -- that would silently discard
            # whatever fields the user already gave. Only ever consults
            # `workflow_orchestrator.ALLOWED_CHAINS`, never an arbitrary
            # LLM-proposed sequence. Excludes "Workflow Clarification"
            # (shares the same `detected_intents` pair by construction,
            # see `classify()`'s draft-match branch) -- that intent means
            # "ask before auto-executing," the opposite of this block.
            chain = detect_chain(conv_intent.detected_intents) if conv_intent.intent != "Workflow Clarification" else None
            if chain is not None:
                workflow_response = await self._dispatch_workflow_chain(
                    request, session_id, language, memory, conv_intent, chain, started, background_tasks,
                    authenticated_user_id,
                )
                if workflow_response is not None:
                    return workflow_response, conv_intent, suggested_actions, False

        interrupting_draft = draft_was_active and self._is_draft_interruption(conv_intent, memory, request.question)

        if draft_was_active and not interrupting_draft:
            # In-progress draft session, and this message isn't a confident
            # unrelated question: let the state machine handle continuation
            # itself, unchanged.
            draft_result = await self.draft_conversation.handle_turn(session_id, request.question, language, memory)
            if draft_result is not None:
                response = await self._respond_with_draft_turn(
                    request, session_id, language, memory, draft_result, started, background_tasks
                )
                return response.model_copy(update={
                    "detected_intents": list(conv_intent.detected_intents) or [conv_intent.intent],
                    "intent_reason": conv_intent.reason,
                }), conv_intent, suggested_actions, interrupting_draft
            # Part 38 "Draft Auto-Pause Engine": `handle_turn` returning
            # `None` here (only possible from `_continue_preview` now, see
            # its docstring) means the draft engine itself decided this
            # message isn't about the draft after all -- treated the same
            # as `_is_draft_interruption` having said so up front, so the
            # normal routing below still gets the "your draft is still
            # saved" reminder appended to whatever it answers with.
            interrupting_draft = True

        if conv_intent.intent == "Draft Generation" and not draft_was_active:
            draft_result = await self.draft_conversation.handle_turn(session_id, request.question, language, memory)
            if draft_result is not None:
                response = await self._respond_with_draft_turn(
                    request, session_id, language, memory, draft_result, started, background_tasks
                )
                return response.model_copy(update={
                    "detected_intents": list(conv_intent.detected_intents) or [conv_intent.intent],
                    "intent_reason": conv_intent.reason,
                }), conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Retry Failed Request":
            # Part 31 "State Manager & Recovery Engine" Rule 1/Rule 4: replay
            # the exact question that failed last time by recursing into
            # `answer()` with it substituted in as if the user asked it fresh
            # -- reuses every existing branch (cache/RAG/general-knowledge/
            # translation/etc.) unchanged rather than duplicating any of
            # their logic here. Cleared optimistically *before* recursing so
            # that if the retry ALSO fails, the recursive call's own
            # RAG-tail failure handling re-records it correctly (see
            # `failure_reason` in the RAG-tail below) instead of this outer
            # turn racing it.
            retry_question = memory.get("last_failed_question") or request.question
            await self.memory.update(
                session_id, last_failed_question=None, last_failed_reason=None, retry_reminder_shown=False
            )
            response = await self.answer(
                ChatRequest(
                    question=retry_question,
                    session_id=session_id,
                    conversation_id=request.conversation_id,
                    user_id=request.user_id,
                    language=request.language,
                    metadata_filters=request.metadata_filters,
                ),
                background_tasks,
            )
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "General Conversation":
            response = await self._respond_with_general_conversation(
                request, session_id, language, memory, started, background_tasks, suggested_actions
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Capability Question":
            response = await self._respond_with_capability_overview(
                request, session_id, language, memory, started, background_tasks, suggested_actions
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Language Preference":
            response = await self._respond_with_language_preference(
                request, session_id, language, memory, conv_intent, started, background_tasks, suggested_actions
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Typo Clarification":
            response = await self._respond_with_typo_clarification(
                request, session_id, language, memory, conv_intent, started, background_tasks, suggested_actions
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Out of Domain":
            response = await self._respond_with_out_of_domain(
                request, session_id, language, memory, started, background_tasks, suggested_actions
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Non-Indian Jurisdiction":
            response = await self._respond_with_jurisdiction_scope(
                request, session_id, language, memory, started, background_tasks, suggested_actions
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Lawyer Recommendation":
            response = await self._respond_with_lawyer_recommendation(
                request, session_id, language, memory, started, background_tasks, suggested_actions
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Translation":
            response = await self._respond_with_translation(
                request, session_id, language, memory, conv_intent, started, background_tasks, suggested_actions
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Response Modification":
            response = await self._respond_with_response_modification(
                request, session_id, language, memory, conv_intent, started, background_tasks, suggested_actions
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Conversation Memory":
            response = await self._respond_with_conversation_memory(
                request, session_id, language, memory, conv_intent, started, background_tasks, suggested_actions
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Summarization":
            response = await self._respond_with_conversation_summary(
                request, session_id, language, memory, started, background_tasks, suggested_actions
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Document Analysis":
            response = await self._respond_with_document_analysis(
                request, session_id, language, memory, started, background_tasks, suggested_actions,
                authenticated_user_id,
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            return response, conv_intent, suggested_actions, interrupting_draft
        elif conv_intent.intent == "Workflow Clarification":
            response = await self._respond_with_workflow_clarification(
                request, session_id, language, memory, started, background_tasks, suggested_actions
            )
            return response, conv_intent, suggested_actions, False
        elif conv_intent.intent == "Intent Feedback":
            response = await self._respond_with_intent_feedback(
                request, session_id, language, memory, conv_intent, started, background_tasks,
                suggested_actions, authenticated_user_id,
            )
            return response, conv_intent, suggested_actions, False

        return None, conv_intent, suggested_actions, interrupting_draft

    @staticmethod
    def _explicit_workflow_request(question: str, language: str) -> bool:
        """Whether this message names a capability that outranks an open draft."""
        from app.chatops.orchestrator import orchestrator  # noqa: F401 - registers workflows
        from app.chatops.registry import interrupts_draft

        try:
            return interrupts_draft(question, language)
        except Exception as exc:  # noqa: BLE001 - a routing hint must never break a turn
            log.warning("chatops_interrupt_check_failed", error=str(exc))
            return False

    async def _dispatch_chatops(
        self, request: ChatRequest, session_id: str, language: str, memory: dict[str, Any], started: float,
        background_tasks: BackgroundTasks | None, authenticated_user_id: str | None,
    ) -> ChatResponse | None:
        """Runs the conversational workflow orchestrator for this turn.

        Returns a fully-formed `ChatResponse` when a workflow handled the
        message, or `None` to fall through to the existing pipeline unchanged.

        Role comes from the database record for the authenticated subject --
        never from anything in the message. A user cannot talk their way into
        a notary or admin workflow, and the workflows that matter re-check
        authorization inside the service besides.
        """
        from app.chatops import state as chatops_state
        from app.chatops.orchestrator import orchestrator

        claims: dict[str, Any] = {}
        if authenticated_user_id:
            try:
                user = await UserRepository().find_by_id(authenticated_user_id)
                claims = {"sub": authenticated_user_id, "role": (user or {}).get("role", "user")}
            except Exception as exc:  # noqa: BLE001 - a lookup failure must not break chat
                log.warning("chatops_claims_lookup_failed", error=str(exc))
                claims = {"sub": authenticated_user_id, "role": "user"}

        try:
            turn = await orchestrator.handle_turn(
                session_id=session_id,
                message=request.question,
                language=language,
                memory=memory,
                authenticated_user_id=authenticated_user_id,
                claims=claims,
                document_ids=list(memory.get("document_ids") or []),
            )
        except Exception as exc:
            log.exception("chatops_dispatch_failed", error=str(exc))
            return None
        if turn is None:
            return None

        # Persist the workflow stack alongside the rest of conversation
        # memory, so pause/resume survives across turns and processes.
        draft_state_keys = (*DraftConversationEngine._CHECKPOINT_KEYS,
                            "parked_drafts", "draft_awaiting_unlock_confirm")
        await self.memory.update(
            session_id,
            **{chatops_state.MEMORY_KEY: memory.get(chatops_state.MEMORY_KEY, [])},
            **{key: memory[key] for key in draft_state_keys if key in memory},
        )

        active_workflow = chatops_state.active_name(memory)
        response = await self._finalize_intent_response(
            request,
            session_id,
            language,
            memory,
            turn.message,
            "Workflow",
            started,
            background_tasks,
            await self._contextual_recommendation(memory, "General"),
            turn.allowed_actions,
            "Deterministic workflow response; no retrieval was needed.",
        )
        # The orchestration contract is attached additively, so every field a
        # pre-existing client reads is untouched.
        return response.model_copy(
            update={
                "intent": active_workflow or turn.status,
                "active_workflow": active_workflow,
                "assistant_message": turn.message,
                "workflow_status": turn.status,
                "missing_field": turn.missing_field,
                "workflow_name": turn.workflow_name,
                "current_step": turn.current_step,
                "completed_steps": turn.completed_steps,
                "total_steps": turn.total_steps,
                "progress_percentage": turn.progress_percentage,
                "required_field": turn.missing_field,
                "collected_facts": turn.collected_facts,
                "requires_confirmation": turn.requires_confirmation,
                "allowed_actions": turn.allowed_actions,
                "artifact": turn.artifact,
                "citations": turn.citations,
                "warnings": turn.warnings,
                "upload_required": turn.upload_required,
                "secure_action_url": turn.secure_action_url,
            }
        )

    async def _dispatch_workflow_chain(
        self, request: ChatRequest, session_id: str, language: str, memory: dict[str, Any],
        conv_intent: ConversationIntentMatch, chain: tuple[str, str], started: float,
        background_tasks: BackgroundTasks | None, authenticated_user_id: str | None,
    ) -> ChatResponse | None:
        """Phase 1 "Multi-Intent Workflow Orchestration": `chain` is always
        one of the fixed pairs `workflow_orchestrator.ALLOWED_CHAINS`
        allows -- never an arbitrary LLM-proposed sequence, and the
        allowlist itself caps every chain at `MAX_CHAIN_LENGTH` (2) steps.

        Only "Document Analysis -> Draft Generation" has real cross-step
        automation (the flagship use case: "review this PDF, flag risky
        clauses, draft a notice based on it"). The other three allowlisted
        chains are logged here as detected -- for the "workflow-chain
        usage" analytics this feeds -- but otherwise return `None`,
        meaning "no bespoke handling, fall through to normal single-intent
        routing" unchanged: "Legal Research -> Draft Generation" already
        works turn-by-turn (the research answer this turn, an ordinary
        `DraftIntentDetector` trigger the next), "Translation -> Response
        Modification" already works via `_respond_with_response_modification`
        chaining onto the latest version, and "Document Analysis -> Legal
        Research" is answered by the normal RAG path with the analysis
        already on the conversation record. Manufacturing bespoke
        cross-step automation for those three would be speculative
        complexity with no missing capability behind it.
        """
        await self._log_intent_event(
            session_id,
            {
                "question": request.question,
                "primary_intent": conv_intent.intent,
                "detected_intents": list(conv_intent.detected_intents),
                "confidence": conv_intent.confidence,
                "reason": f"Workflow chain detected: {chain[0]} -> {chain[1]}",
                "classifier_source": "workflow_chain",
                "workflow_chain": list(chain),
            },
            owner_user_id=memory.get("owner_user_id") or authenticated_user_id,
        )
        if chain != ("Document Analysis", "Draft Generation"):
            return None
        return await self._run_document_analysis_then_draft(
            request, session_id, language, memory, conv_intent, chain, started, background_tasks,
            authenticated_user_id,
        )

    async def _run_document_analysis_then_draft(
        self, request: ChatRequest, session_id: str, language: str, memory: dict[str, Any],
        conv_intent: ConversationIntentMatch, chain: tuple[str, str], started: float,
        background_tasks: BackgroundTasks | None, authenticated_user_id: str | None,
    ) -> ChatResponse:
        """Steps 1-9 of the flagship workflow: resolve the document (same
        resolution order as a normal `Document Analysis` turn), verify
        ownership (`DocumentService.analyze` re-checks unconditionally,
        exactly like a standalone analysis turn -- no workflow-specific
        bypass), analyze it once (no duplicate LLM call), map extractable
        facts into the target draft template's own fields via the
        conservative `map_facts_to_draft_fields` (never fabricates a
        value), then seed `DraftConversationEngine`'s existing collecting
        state so its own already-tested "ask only for what's missing" /
        "auto-preview once nothing's missing" logic runs unchanged.

        Never reaches step 11 automatically: `_process_collecting_message`
        only ever advances a fresh draft to `preview_ready`, never
        `approved`/`locked`/`exported` -- those all require the user's own
        explicit next message (see `DraftConversationEngine._continue_preview`
        / `_continue_approved`), completely unmodified by this workflow.
        """
        workflow_chain = list(chain)
        document_id = (request.metadata_filters or {}).get("document_id") or memory.get("last_uploaded_document_id")
        if not document_id:
            answer = _NO_UPLOADED_DOCUMENT_MESSAGES.get(language, _NO_UPLOADED_DOCUMENT_MESSAGES["english"])
            recommendation = await self._contextual_recommendation(memory, "Document Analysis")
            response = await self._finalize_intent_response(
                request, session_id, language, memory, answer, "Document Analysis", started, background_tasks,
                recommendation, [], "Workflow chain needs an uploaded document; none found.", confidence=0.3,
            )
            return response.model_copy(update={
                "workflow_chain": workflow_chain, "workflow_status": "failed_no_document",
                "detected_intents": list(conv_intent.detected_intents) or [conv_intent.intent],
                "intent_reason": conv_intent.reason,
            })

        draft_match = self.draft_conversation.intent_detector.detect(request.question)
        template = get_template(draft_match.draft_id) if draft_match.matched and draft_match.draft_id else None
        if template is None:
            # Named a document-analysis-plus-draft intent pair but no
            # recognizable draft template in the message itself -- analyze
            # the document (still genuinely useful) rather than guessing
            # which template was meant.
            response = await self._respond_with_document_analysis(
                request, session_id, language, memory, started, background_tasks, [], authenticated_user_id,
            )
            return response.model_copy(update={
                "workflow_chain": workflow_chain, "workflow_status": "failed_no_draft_template",
                "detected_intents": list(conv_intent.detected_intents) or [conv_intent.intent],
                "intent_reason": conv_intent.reason,
            })

        try:
            analysis = await self.document_service.analyze(
                DocumentAnalysisRequest(document_id=document_id, language=language, session_id=session_id),
                authenticated_user_id=authenticated_user_id,
            )
        except ForbiddenError:
            answer = _DOCUMENT_ACCESS_DENIED_MESSAGES.get(language, _DOCUMENT_ACCESS_DENIED_MESSAGES["english"])
            recommendation = await self._contextual_recommendation(memory, "Document Analysis")
            response = await self._finalize_intent_response(
                request, session_id, language, memory, answer, "Document Analysis", started, background_tasks,
                recommendation, [], "Document access denied by ownership check during workflow.", confidence=0.2,
            )
            return response.model_copy(update={
                "workflow_chain": workflow_chain, "workflow_status": "failed_ownership",
                "detected_intents": list(conv_intent.detected_intents) or [conv_intent.intent],
                "intent_reason": conv_intent.reason,
            })
        except BadRequestError:
            answer = _DOCUMENT_UNAVAILABLE_MESSAGES.get(language, _DOCUMENT_UNAVAILABLE_MESSAGES["english"])
            recommendation = await self._contextual_recommendation(memory, "Document Analysis")
            response = await self._finalize_intent_response(
                request, session_id, language, memory, answer, "Document Analysis", started, background_tasks,
                recommendation, [], "Resolved document had no readable content during workflow.", confidence=0.2,
            )
            return response.model_copy(update={
                "workflow_chain": workflow_chain, "workflow_status": "failed_unreadable_document",
                "detected_intents": list(conv_intent.detected_intents) or [conv_intent.intent],
                "intent_reason": conv_intent.reason,
            })

        mapped_fields = map_facts_to_draft_fields(template, analysis)
        memory["draft_mode"] = True
        memory["draft_stage"] = "collecting"
        memory["draft_template_id"] = template.draft_id
        memory["draft_fields"] = mapped_fields
        memory["draft_id"] = None
        memory["draft_language"] = language

        draft_result = await self.draft_conversation.handle_turn(session_id, request.question, language, memory)
        if draft_result is None:
            # Shouldn't happen (draft_mode was just set True above), but a
            # workflow turn must never crash -- degrade to reporting the
            # analysis alone rather than a 500.
            response = await self._respond_with_document_analysis(
                request, session_id, language, memory, started, background_tasks, [], authenticated_user_id,
            )
            return response.model_copy(update={
                "workflow_chain": workflow_chain, "workflow_status": "failed_draft_start",
                "detected_intents": list(conv_intent.detected_intents) or [conv_intent.intent],
                "intent_reason": conv_intent.reason,
            })

        response = await self._respond_with_draft_turn(
            request, session_id, language, memory, draft_result, started, background_tasks
        )
        workflow_status = "draft_preview_ready" if memory.get("draft_stage") == "preview" else "awaiting_missing_fields"
        return response.model_copy(update={
            "workflow_chain": workflow_chain,
            "workflow_status": workflow_status,
            "detected_intents": list(conv_intent.detected_intents) or [conv_intent.intent],
            "intent_reason": f"Workflow chain {workflow_chain[0]} -> {workflow_chain[1]}: {conv_intent.reason}",
        })

    async def _respond_with_workflow_clarification(
        self, request: ChatRequest, session_id: str, language: str, memory: dict[str, Any], started: float,
        background_tasks: BackgroundTasks | None, suggested_actions: list[str],
    ) -> ChatResponse:
        """Phase 1 "General Clarification Mode": asks which of two intents
        with similar/ambiguous confidence the user actually meant, stores
        the resolution state, and stops -- never guesses. The clarifying
        reply itself is never treated as a legal question/fact (it's parsed
        only by `_resolve_workflow_clarification_choice`'s fixed keyword
        set on the next turn); the ORIGINAL question is what actually gets
        analyzed/drafted once resolved.
        """
        answer = _WORKFLOW_CLARIFICATION_MESSAGES.get(language, _WORKFLOW_CLARIFICATION_MESSAGES["english"])
        memory["pending_clarification"] = "workflow_intent"
        memory["pending_workflow_question"] = request.question
        await self.memory.update(session_id, pending_clarification="workflow_intent", pending_workflow_question=request.question)
        recommendation = await self._contextual_recommendation(memory, "Document Analysis")
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Workflow Clarification", started, background_tasks,
            recommendation, suggested_actions,
            "Ambiguous document-analysis + draft-generation phrasing; asked which one instead of guessing.",
            confidence=0.5, pending_clarification="workflow_intent",
        )

    async def _resolve_workflow_clarification(
        self, request: ChatRequest, session_id: str, language: str, memory: dict[str, Any], started: float,
        background_tasks: BackgroundTasks | None, authenticated_user_id: str | None,
    ) -> tuple[ChatResponse, ConversationIntentMatch, list[str], bool] | None:
        """Resolves a pending `_respond_with_workflow_clarification` on the
        very next turn. Returns `None` if this message doesn't answer the
        clarification at all (an unrelated new question) -- the caller
        treats that as an expired, stale clarification and falls through to
        normal classification of the actual message, never forcing an
        interpretation onto text that was never a legal question in the
        first place.

        Deliberately does NOT recurse through `answer()`/re-classify the
        original question -- it still lacks the "based on it/that" link
        that made it ambiguous in the first place, so reclassifying it
        would just ask the same clarifying question again. Instead directly
        invokes the resolved path (the same chain-execution/draft-turn
        methods the normal routes use) with a synthetic, already-resolved
        `ConversationIntentMatch`.
        """
        choice = _resolve_workflow_clarification_choice(request.question)
        if choice is None:
            return None
        original_question = memory.get("pending_workflow_question") or request.question
        memory["pending_clarification"] = None
        memory.pop("pending_workflow_question", None)
        await self.memory.update(session_id, pending_clarification=None, pending_workflow_question=None)
        synthetic_conv_intent = ConversationIntentMatch(
            "Draft Generation", 0.85, False, f"resolved pending workflow clarification: {choice}",
            detected_intents=("Document Analysis", "Draft Generation") if choice == "chain" else (),
        )
        workflow_request = request.model_copy(update={"question": original_question})
        if choice == "chain":
            response = await self._run_document_analysis_then_draft(
                workflow_request, session_id, language, memory, synthetic_conv_intent,
                ("Document Analysis", "Draft Generation"), started, background_tasks, authenticated_user_id,
            )
            return response, synthetic_conv_intent, [], False

        draft_result = await self.draft_conversation.handle_turn(session_id, original_question, language, memory)
        if draft_result is None:
            # Shouldn't happen -- `original_question` matched a draft
            # trigger when the clarification was first asked -- but a
            # workflow turn must never crash.
            response = await self.answer(workflow_request, background_tasks, authenticated_user_id=authenticated_user_id)
            return response, synthetic_conv_intent, [], False
        response = await self._respond_with_draft_turn(
            workflow_request, session_id, language, memory, draft_result, started, background_tasks
        )
        return response.model_copy(update={
            "workflow_status": "resolved_direct_draft",
            "detected_intents": [synthetic_conv_intent.intent],
            "intent_reason": synthetic_conv_intent.reason,
        }), synthetic_conv_intent, [], False

    async def _resolve_section_lookup_act_clarification(
        self, request: ChatRequest, session_id: str, language: str, memory: dict[str, Any], started: float,
        background_tasks: BackgroundTasks | None, authenticated_user_id: str | None,
    ) -> tuple[ChatResponse, ConversationIntentMatch, list[str], bool] | None:
        """Resolves a pending `_render_act_disambiguation_question` ("This
        section number exists in more than one Act. Which one do you mean?")
        on the very next turn. Returns `None` if the reply doesn't name one
        of the Acts that were actually offered (an unrelated new message) --
        the caller treats that as an expired, stale clarification and falls
        through to normal classification of the actual message.

        Root cause this fixes: a bare "Section N" reply asking which Act was
        never persisted anywhere, so a reply naming just the Act ("Hindu
        Marriage Act") lost the section number entirely and was classified as
        a brand new, vague query -- one that fails retrieval's relevance gate
        and answers "not found" even for an Act this KB genuinely indexes.
        Recomposing "Section N of <Act>" and recursing through `answer()`
        gives it `LegalRetriever.retrieve`'s `_NAMED_SECTION_CITATION_RE`
        fast path -- a hard `section_number`+`act_name` filter, the same one
        a user who typed the full citation in one message gets.
        """
        offered_acts: list[str] = memory.get("pending_section_acts") or []
        reply = request.question.strip().lower()
        matched_act = next((act for act in offered_acts if act.lower() in reply or reply in act.lower()), None)
        if matched_act is None:
            return None
        original_question = memory.get("pending_section_query") or request.question
        section_number = parse_section_lookup(original_question.lower().strip())
        memory["pending_clarification"] = None
        memory.pop("pending_section_query", None)
        memory.pop("pending_section_acts", None)
        await self.memory.update(
            session_id, pending_clarification=None, pending_section_query=None, pending_section_acts=None
        )
        composed_question = (
            f"Section {section_number} of {matched_act}" if section_number else f"{original_question} ({matched_act})"
        )
        resolved_request = request.model_copy(update={"question": composed_question})
        response = await self.answer(resolved_request, background_tasks, authenticated_user_id=authenticated_user_id)
        synthetic_conv_intent = ConversationIntentMatch(
            "General Legal Query", 0.85, False, f"resolved pending section-lookup Act clarification: {matched_act}",
        )
        return response, synthetic_conv_intent, [], False

    async def _respond_with_intent_feedback(
        self, request: ChatRequest, session_id: str, language: str, memory: dict[str, Any],
        conv_intent: ConversationIntentMatch, started: float, background_tasks: BackgroundTasks | None,
        suggested_actions: list[str], authenticated_user_id: str | None,
    ) -> ChatResponse:
        """Phase 1 "Intent Feedback": explicit post-hoc correction of the
        PREVIOUS turn's intent classification ("Wrong intent", "I wanted
        legal research", "This is document analysis").

        Stored in `intent_feedback` (`IntentFeedbackRepository`) --
        deliberately NOT `app/memory/entity_memory.py`'s facts store or
        anywhere the RAG/entity-extraction pipeline reads from: this is
        metadata ABOUT how the conversation was routed, never treated as a
        legal fact about the user's situation. Ownership-scoped the same
        way `/feedback` is: `session_id` always, plus the JWT-verified
        `authenticated_user_id` when present -- never a client-supplied one.
        The feedback message itself is never re-classified as a legal
        question or forwarded to retrieval.
        """
        history = memory.get("intent_history") or []
        original_intent = history[-1].get("primary_intent") if history else None
        corrected_intent = conv_intent.feedback_corrected_intent
        try:
            await self.intent_feedback.insert({
                "session_id": session_id,
                "owner_user_id": authenticated_user_id,
                "message_text": request.question,
                "original_intent": original_intent,
                "corrected_intent": corrected_intent,
            })
        except Exception as exc:  # noqa: BLE001 - a feedback-storage hiccup must not break the chat turn
            log.warning("intent_feedback_write_failed", session_id=session_id, error=str(exc))

        if corrected_intent:
            answer = (
                f"Thanks for the correction — noted that you actually wanted \"{corrected_intent}\". "
                "Feel free to ask your question again and I'll route it correctly this time."
            )
        elif original_intent:
            answer = (
                f"Thanks for flagging that \"{original_intent}\" wasn't the right read on your last message. "
                "Could you tell me what you'd actually like me to do?"
            )
        else:
            answer = "Thanks for the feedback — could you tell me what you'd actually like me to do?"

        recommendation = await self._contextual_recommendation(memory, "Intent Feedback")
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Intent Feedback", started, background_tasks,
            recommendation, suggested_actions,
            "Explicit intent-correction feedback stored, kept separate from legal facts.", confidence=0.5,
        )

    async def handle_turn(
        self, request: ChatRequest, background_tasks: BackgroundTasks | None = None,
        authenticated_user_id: str | None = None,
    ) -> ChatResponse:
        """The real entry point for `POST /chat` -- a hard, unconditional
        wall-clock ceiling plus a per-session in-flight guard around
        `answer()`.

        Deliberately NOT folded into `answer()` itself: `answer()` recurses
        into itself for the "Retry Failed Request" intent (same session,
        same method, see its dispatch below), and a lock/hard-timeout
        acquired once per logical turn must not fight that recursion by
        trying to acquire its own session's lock a second time from inside
        itself. Wrapping here, once, at the actual request boundary, keeps
        `answer()` recursion-safe and unchanged.
        """
        started = time.perf_counter()
        session_id, _ = await self._run_shared_preflight(request, authenticated_user_id)
        # Every downstream call (including `answer()`'s own preflight) then
        # resolves the SAME session_id instead of `request.session_id or
        # str(uuid4())` minting a second, different one for a brand-new
        # session.
        request = request.model_copy(update={"session_id": session_id})
        lock_token = await self._acquire_turn_lock(session_id)
        if lock_token is None:
            return await self._turn_in_progress_response(request, session_id, started)
        try:
            try:
                return await asyncio.wait_for(
                    self.answer(request, background_tasks, authenticated_user_id),
                    timeout=settings.chat_request_budget_seconds + _HARD_TIMEOUT_GRACE_SECONDS,
                )
            except TimeoutError:
                log.warning(
                    "chat_request_hard_timeout",
                    session_id=session_id,
                    budget_seconds=settings.chat_request_budget_seconds,
                    elapsed_seconds=round(time.perf_counter() - started, 2),
                )
                return await self._timeout_response(request, session_id, started)
        finally:
            await self._release_turn_lock(session_id, lock_token)

    async def _acquire_turn_lock(self, session_id: str) -> str | None:
        """`None` means "another request for this session is already being
        processed"; any other return value is an opaque token to hand back to
        `_release_turn_lock`. Best-effort: a Redis outage returns a fresh
        token immediately (degrades to "no guard", never to blocking chat)."""
        token = str(uuid4())
        ttl_seconds = int(settings.chat_request_budget_seconds + _HARD_TIMEOUT_GRACE_SECONDS + 10)
        try:
            acquired = await redis_client.client.set(
                f"{_TURN_LOCK_KEY_PREFIX}{session_id}", token, nx=True, ex=ttl_seconds,
            )
        except CACHE_UNAVAILABLE_ERRORS:
            return token
        return token if acquired else None

    async def _release_turn_lock(self, session_id: str, token: str) -> None:
        key = f"{_TURN_LOCK_KEY_PREFIX}{session_id}"
        try:
            current = await redis_client.client.get(key)
            if current is not None and current == token.encode():
                await redis_client.client.delete(key)
        except CACHE_UNAVAILABLE_ERRORS:
            pass

    async def _turn_in_progress_response(
        self, request: ChatRequest, session_id: str, started: float,
    ) -> ChatResponse:
        """Returned instead of starting a second full pipeline run when this
        session already has one in flight -- the direct fix for the
        duplicate-draft/duplicate-message risk a client retry (or resend)
        creates while the original call is still running. Deliberately does
        NOT append to memory/history: this turn was never actually
        processed, so there is nothing real to record, and recording it
        would itself create the kind of duplicate entry this guard exists
        to prevent.
        """
        memory = await self.memory.load(session_id)
        answer = (
            "I'm still working on your previous message in this conversation. Please wait a few seconds for "
            "it to finish before sending it again -- resending now risks creating a duplicate."
        )
        recommendation = await self._contextual_recommendation(memory, "Request In Progress")
        return ChatResponse(
            message_id=str(uuid4()), session_id=session_id, answer=answer, sources=[],
            confidence=0.0, confidence_label=self._confidence_label(0.0),
            confidence_reason="A previous request for this session is still being processed.",
            lawyer_recommendation=recommendation,
            detected_language=request.language or "english",
            detected_intent="Request In Progress", conversation_intent="Request In Progress",
            latency_ms=(time.perf_counter() - started) * 1000,
            retryable=True, failure_category="in_progress",
            request_id=str(get_contextvars().get("request_id") or ""),
        )

    async def _timeout_response(
        self, request: ChatRequest, session_id: str, started: float,
    ) -> ChatResponse:
        """Built when `handle_turn`'s hard ceiling fires. Records the
        question as a genuine failure via the SAME `last_failed_question`/
        `last_failed_reason` fields the ordinary RAG-tail failure path
        writes (see `_answer_within_deadline`) -- so the very next turn,
        whether the user types "retry" or (BUG-013a fix, see
        `ConversationIntentClassifier.classify`) simply resends the same
        question, is recognised as a retry rather than routed as a brand
        new, unrelated message.
        """
        memory = await self.memory.update(
            session_id, last_failed_question=request.question, last_failed_reason="request_timed_out",
            retry_reminder_shown=False,
        )
        answer = (
            "This request took longer than expected and could not be completed in time. Nothing was changed "
            'or saved. Say "retry", or just resend your message, to try again.'
        )
        await self.memory.append(session_id, "assistant", answer)
        message_id = str(uuid4())
        await self.history.insert(
            {
                "message_id": message_id,
                "session_id": session_id,
                "conversation_id": request.conversation_id,
                # Security finding C2: this must be the SERVER-VERIFIED owner
                # (`memory["owner_user_id"]`, set by `check_access`/
                # `_claim_session_if_unowned` from the caller's JWT), never
                # the client-supplied `request.user_id` -- stamping the
                # latter here made `erase_user_data`'s
                # `ChatRepository().delete_by_user(user_id)` match nothing
                # for a real authenticated user (their messages were never
                # actually tagged with their real id), so `DELETE /me/data`
                # reported success while leaving every chat message behind.
                "user_id": memory.get("owner_user_id"),
                "question": request.question,
                "answer": answer,
                "intent": {},
                "conversation_intent": "Request Timeout",
                "entities": {},
                "sources": [],
            }
        )
        recommendation = await self._contextual_recommendation(memory, "Request Timeout")
        return ChatResponse(
            message_id=message_id, session_id=session_id, answer=answer, sources=[],
            confidence=0.0, confidence_label=self._confidence_label(0.0),
            confidence_reason="The request exceeded its time budget before an answer could be produced.",
            lawyer_recommendation=recommendation,
            detected_language=request.language or "english",
            detected_intent="Request Timeout", conversation_intent="Request Timeout",
            latency_ms=(time.perf_counter() - started) * 1000,
            retryable=True, failure_category=self._failure_category("request timed out"),
            saved_progress=False,
            request_id=str(get_contextvars().get("request_id") or ""),
        )

    async def answer(
        self, request: ChatRequest, background_tasks: BackgroundTasks | None = None,
        authenticated_user_id: str | None = None,
    ) -> ChatResponse:
        """Answer one turn under a single wall-clock LLM budget.

        Nested calls used for retry/correction keep the tighter outer
        deadline because `deadline()` never extends an existing deadline.
        """
        with deadline(settings.chat_request_budget_seconds, label="POST /chat"):
            response = await self._answer_within_deadline(
                request, background_tasks, authenticated_user_id
            )
        return response.model_copy(
            update={"request_id": str(get_contextvars().get("request_id") or "")}
        )

    async def _answer_within_deadline(
        self, request: ChatRequest, background_tasks: BackgroundTasks | None = None,
        authenticated_user_id: str | None = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        session_id, _ = await self._run_shared_preflight(request, authenticated_user_id)
        memory = await self.memory.append(session_id, "user", request.question)
        memory = await self._claim_session_if_unowned(session_id, memory, authenticated_user_id)
        await self._store_entity_fact_if_any(session_id, memory, request.question)
        language = (
            request.language
            or extract_requested_language(request.question)
            or self.language_detector.resolve(request.question, memory.get("language_preference"))
        )

        inline_response = await self._maybe_answer_from_inline_document(
            request, session_id, language, memory, started, background_tasks
        )
        if inline_response is not None:
            return inline_response

        dispatched_response, conv_intent, suggested_actions, interrupting_draft = await self._dispatch_conversation_intent(
            request, session_id, language, memory, started, background_tasks, authenticated_user_id
        )
        if dispatched_response is not None:
            return dispatched_response
        assert conv_intent is not None, (
            "_dispatch_conversation_intent returns conv_intent=None only alongside a "
            "non-None response; reaching here without one means that contract broke."
        )

        question_for_pipeline = request.question
        if conv_intent.intent == "Follow-up Question":
            question_for_pipeline = await self._resolve_followup_question(request.question, memory)

        # Part 35 "Entity Memory": checked before retrieval, on the RAW
        # message (not `question_for_pipeline`'s possibly LLM-resolved
        # rephrasing) -- a fact-recall question ("What was stolen?", "mera
        # kya chori hua tha") should never fall through to RAG/general
        # knowledge, which has no way to answer it and no business trying
        # (that's exactly what produced "I don't have access to your
        # personal history" dead-ends for questions the answer to was
        # already sitting in this session). Deliberately checked here
        # rather than earlier alongside the other conv-intent branches --
        # a fact-recall phrasing rarely matches any of those more specific
        # discourse patterns, so it would otherwise reach this point anyway.
        #
        # `is_recall_query` itself distinguishes a fact-IDENTIFYING question
        # ("What was stolen?", "kisne phoda tha") from an action-SEEKING one
        # ("mera bike chori ho gya hai, kya karu") that merely shares an
        # interrogative word with a fact just stated in the same message --
        # the latter must fall through to RAG/general knowledge for real
        # advice, not answer "Your bike was stolen" and stop there.
        matched_fact = (
            entity_memory.find_matching_fact(request.question, memory.get("entities") or [])
            if entity_memory.is_recall_query(request.question)
            else None
        )
        if matched_fact is not None:
            answer = entity_memory.format_answer(matched_fact, language)
            recommendation = await self._contextual_recommendation(memory, "Entity Memory")
            response = await self._finalize_intent_response(
                request, session_id, language, memory, answer, "Entity Memory", started, background_tasks,
                recommendation, [],
                f"Answered from a fact stated earlier in this conversation (turn {matched_fact.turn_index}); "
                "no retrieval performed.",
            )
            return await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)

        if entity_memory.is_recall_query(request.question) and entity_memory.matches_known_category(request.question):
            # Genuinely a fact-recall question about a category this module
            # tracks (stolen/lost/injured/...), but nothing stored matches
            # it -- answer "I don't remember" directly instead of falling
            # through to RAG/general knowledge, which has no way to know and
            # would otherwise either dead-end or invent an answer about an
            # entity that was never actually mentioned in this session.
            answer = entity_memory.format_no_memory_answer(language)
            recommendation = await self._contextual_recommendation(memory, "Entity Memory")
            response = await self._finalize_intent_response(
                request, session_id, language, memory, answer, "Entity Memory", started, background_tasks,
                recommendation, [],
                "Recall question named a known fact category but no stored fact matched; "
                "answered with an explicit no-memory response instead of falling through to retrieval.",
            )
            return await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)

        timings: dict[str, float] = {}
        explanation_level = request.explanation_mode or detect_explanation_level(request.question) or "citizen"

        stage_start = time.perf_counter()
        intent = await self.intent_detector.detect(question_for_pipeline, language)
        timings["intent_detection_ms"] = (time.perf_counter() - stage_start) * 1000

        matter, matter_response = await self._resolve_matter_context(
            request, session_id, language, memory, intent, started, background_tasks, authenticated_user_id
        )
        if matter_response is not None:
            return await self._append_conversation_reminders(matter_response, memory, interrupting_draft, session_id)

        stage_start = time.perf_counter()
        cached_entry, cache_hit_type = await self.response_cache.lookup(
            question_for_pipeline, language, intent.intent, jurisdiction_key=matter.cache_key()
        )
        timings["cache_lookup_ms"] = (time.perf_counter() - stage_start) * 1000
        if cached_entry is not None:
            response = await self._respond_from_cache(
                request, session_id, language, memory, intent, cached_entry, cache_hit_type, started, background_tasks
            )
            return await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)

        entities, ranked, sources, recommendation, jurisdiction_ambiguity = await self._prepare_rag_context(
            question_for_pipeline, language, request, intent, timings, session_id, authenticated_user_id, memory,
            matter_context=matter.as_filter(),
        )
        related_questions: list[str] = []
        related_questions_task: asyncio.Task[list[str]] | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        # Strict-RAG guardrail: `ranked` already only contains chunks that
        # cleared `_is_relevant_chunk`'s relevance gate (`_prepare_rag_context`
        # -> `_filter_relevant_context`) -- zero surviving chunks means no
        # verified document backs this question, so short-circuit with the
        # exact hardcoded fallback string and never call the LLM at all.
        no_verified_context = not ranked
        general_knowledge_used = False
        ambiguous_acts = self._section_lookup_ambiguous_acts(intent.intent, ranked)
        # Jurisdiction Routing (Phase 2), objective item 3/4: `jurisdiction_
        # ambiguity` was computed by `_prepare_rag_context` against the RAW
        # pre-rerank candidate pool (see its own comment for why) -- the
        # general, data-driven clarification backstop, broader than the fixed
        # category+keyword pre-check in `_resolve_matter_context`.
        # Part 31 "State Manager & Recovery Engine" Rule 1: `None` means this
        # turn's LLM call genuinely succeeded; any other value is recorded as
        # `last_failed_question`/`last_failed_reason` below rather than
        # silently treating the fallback/recovery text as a real answer.
        failure_reason: str | None = None
        if jurisdiction_ambiguity:
            answer = clarification_question(language, ambiguity=jurisdiction_ambiguity["ambiguity"])
            confidence = 0.5
            confidence_reason = (
                f"Retrieved sources are tied to specific {jurisdiction_ambiguity['ambiguity']}(s) "
                f"({', '.join(jurisdiction_ambiguity['candidates'])}) with no confirmed match to this matter "
                "and no all-India provision to fall back on; asked which one applies."
            )
            timings["llm_call_ms"] = 0.0
            pending = "matter_jurisdiction_locality" if jurisdiction_ambiguity["ambiguity"] == "locality" else "matter_jurisdiction"
            await self.memory.update(session_id, pending_clarification=pending)
        elif ambiguous_acts:
            answer = self._render_act_disambiguation_question(ambiguous_acts, language)
            confidence = 0.5
            confidence_reason = "Multiple Acts define this section number; asked the user which one they mean."
            timings["llm_call_ms"] = 0.0
        elif no_verified_context:
            stage_start = time.perf_counter()
            gk_answer = (
                await general_knowledge_answer(self.llm, request.question, language)
                if settings.general_knowledge_fallback_enabled else None
            )
            timings["llm_call_ms"] = (time.perf_counter() - stage_start) * 1000
            if gk_answer is not None:
                answer = gk_answer
                confidence = _GENERAL_KNOWLEDGE_CONFIDENCE
                confidence_reason = (
                    "No chunk in the knowledge base cleared the relevance threshold for this question; "
                    "answered from general legal knowledge instead, labeled and unverified."
                )
                general_knowledge_used = True
                if settings.kb_gap_autofetch_enabled:
                    asyncio.create_task(GapAutoFetchService().record_gap(request.question))
            else:
                answer = await self._no_verified_context_answer(request.question, language)
                confidence = 0.0
                confidence_reason = "No chunk in the knowledge base cleared the relevance threshold for this question."
                timings["llm_call_ms"] = 0.0
        else:
            messages = self._build_rag_messages(
                question_for_pipeline, language, intent, conv_intent, entities, ranked, explanation_level
            )
            stage_start = time.perf_counter()
            # `call_with_hard_timeout`'s deadline-backstop reasoning (see its
            # own docstring) lives inside `_call_llm_with_grounding_retry`
            # now, shared with its own bounded retry-on-refusal -- see that
            # method's docstring for why a second call is sometimes made here.
            llm_response = await self._call_llm_with_grounding_retry(messages, ranked, session_id)
            timings["llm_call_ms"] = (time.perf_counter() - stage_start) * 1000
            timings["llm_retry_count"] = llm_response.retry_count
            failure_reason = llm_response.error
            answer = self._safe_llm_text(llm_response, self._fallback_answer(intent.intent, ranked, question_for_pipeline))
            if LEGAL_DISCLAIMER not in answer:
                answer = f"{answer}\n\n{LEGAL_DISCLAIMER}"
            if failure_reason is None:
                # Grounding validation: a genuinely LLM-generated answer must
                # be backed by real source citations, not just by chunks
                # having been retrieved -- `_prepare_rag_context` only fails
                # to produce a citation for a chunk when its metadata is
                # missing `source_document`, which does happen for some
                # ingested content. An answer generated from such chunks
                # would otherwise go out with real prose but zero verifiable
                # sources behind it, indistinguishable from a properly
                # grounded one. Confidence is downgraded to 0 and the answer
                # replaced with the same safe insufficient-context message
                # used when retrieval finds nothing at all, rather than
                # presenting an unsupported claim as verified law. Not
                # applied to the `_fallback_answer` branch below (LLM call
                # itself failed) -- that text is quoted directly out of the
                # retrieved chunk, inherently grounded regardless of whether
                # a formal `source_document` tag happens to be present.
                is_grounded, grounding_reason = self.citation_engine.validate_grounding(answer, sources, ranked)
                # The LLM itself follows the system prompt's Rule 2 and emits
                # this exact fallback line whenever IT judges the retrieved
                # chunks don't actually answer the question -- e.g. retrieval
                # surfacing a same-numbered section from the wrong Act
                # alongside the real one. That's a real refusal, not a
                # citation-formatting problem `validate_grounding` checks
                # for, so it must be treated the same way: confidence forced
                # to 0 so it's never treated as cacheable (`is_cacheable`
                # below), rather than inheriting whatever retrieval-strength
                # confidence `_confidence` would otherwise compute and
                # permanently caching this refusal as if it were a real,
                # high-confidence answer for the next 24h-7d TTL.
                if not is_grounded or is_no_verified_context(answer):
                    answer = await self._no_verified_context_answer(request.question, language)
                    confidence = 0.0
                    confidence_reason = grounding_reason or "The retrieved sources do not sufficiently support this answer."
                else:
                    answer = self._polish_grounded_answer(answer, ranked, language, request.question)
                    # Phase 1 item 3: the last thing between a generated
                    # answer and the user. `validate_grounding` above only
                    # asks "are there citations"; this asks whether what was
                    # actually written is a legal answer at all -- see
                    # `app/rag/answer_quality.py`.
                    answer, quality = await self._apply_quality_gate(
                        answer, request=request, messages=messages, language=language,
                        sources=sources, ranked=ranked,
                    )
                    if quality.severity is AnswerSeverity.REJECT:
                        confidence = 0.0
                        confidence_reason = quality.reason or "The generated answer did not pass the legal-answer quality gate."
                    else:
                        confidence = self._confidence(ranked, intent.confidence, entities.confidence)
                        confidence_reason = self._confidence_reason(ranked, intent.intent)
                        if quality.severity is AnswerSeverity.RETRY_LANGUAGE:
                            # The retry didn't land either. The answer is kept
                            # (discarding correct legal information over a
                            # presentation problem helps nobody) but must not
                            # be presented as high-confidence, and must never
                            # be cached in this state for the next 24h-7d.
                            confidence = min(confidence, _LANGUAGE_MISMATCH_CONFIDENCE_CAP)
                            confidence_reason = quality.reason or confidence_reason
            else:
                # A provider-failure notice is not a legal answer, even when
                # retrieval found a strong candidate. Reporting Medium/High
                # confidence here made the degraded response look verified.
                confidence = 0.0
                confidence_reason = "Answer generation failed; no legal conclusion was presented."
            # Fired as a background task rather than awaited here -- it's
            # purely supplementary "related questions" UI content with no
            # bearing on `answer`/`confidence`/anything else in this
            # response, so there's no reason to block on it before doing the
            # (also-independent) cache-store/memory-persist/history-insert
            # work below. Awaited once, right before the final `ChatResponse`
            # is built, by which point it's very likely already finished.
            if failure_reason is None:
                related_questions_task = asyncio.ensure_future(
                    self._generate_related_questions(question_for_pipeline, answer, language)
                )
            prompt_tokens = llm_response.prompt_tokens
            completion_tokens = llm_response.completion_tokens
            # Jurisdiction Routing (Phase 2), objective item 7: a
            # profile-default State/an as-of date used for this answer must
            # be disclosed, not presented as an unstated fact -- appended
            # before caching so a cache hit later shows the same disclosure a
            # fresh answer would. Only reached here in the real-LLM-answer
            # branch (`no_verified_context`/`ambiguous_acts` both take an
            # earlier branch entirely).
            # Objective item 5: the trigger is "this is a historical answer at
            # all" (`matter.as_of_date` resolved), NOT only "multiple versions
            # were literally detected" -- a SINGLE surviving version is not
            # guaranteed applicable either, since this KB has no transition/
            # savings-clause metadata to confirm it either way. Both the
            # disclosure text and `confidence` react to the broader condition;
            # `detect_version_ambiguity` only escalates to the stronger cap
            # when the KB itself evidences a genuine conflict.
            historical_answer = bool(matter.as_of_date)
            version_ambiguous = bool(historical_answer and detect_version_ambiguity(ranked))
            if version_ambiguous:
                confidence = min(confidence, _HISTORICAL_VERSION_AMBIGUITY_CONFIDENCE_CAP)
                confidence_reason = (
                    "More than one version of a retrieved provision could apply on the given date; "
                    "which one governs was not conclusively determined."
                )
            elif historical_answer:
                confidence = min(confidence, _HISTORICAL_TRANSITION_UNVERIFIED_CONFIDENCE_CAP)
                confidence_reason = (
                    "Answered using the version on record for the date given; whether a transition/savings "
                    "clause changes this for the exact facts was not independently verified."
                )
            note = disclosure_note(language, matter, version_ambiguous=version_ambiguous)
            if note:
                answer = f"{answer}\n\n{note}"
            machine_note = _machine_verification_disclosure(language, sources)
            if machine_note:
                answer = f"{answer}\n\n{machine_note}"
            if self.response_cache.is_cacheable(confidence, bool(sources), llm_response.error):
                await self.response_cache.store(
                    question_for_pipeline,
                    language,
                    intent.intent,
                    {
                        "answer": answer,
                        "sources": [source.model_dump() for source in sources],
                        "confidence": confidence,
                        "confidence_label": self._confidence_label(confidence),
                        "confidence_reason": confidence_reason,
                        "retrieved_sections": [
                            str(chunk.metadata.get("section_number")) for chunk in ranked if chunk.metadata.get("section_number")
                        ],
                        "llm_provider": self.llm.provider_name,
                        "llm_model": getattr(self.llm, "model", ""),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "prompt_version": settings.prompt_version,
                    },
                    jurisdiction_key=matter.cache_key(),
                )
        return await self._finalize_rag_response(
            request, session_id, language, memory, intent, conv_intent, entities, sources, ranked, recommendation,
            explanation_level, suggested_actions, interrupting_draft, no_verified_context,
            answer, confidence, confidence_reason, prompt_tokens, completion_tokens, failure_reason,
            question_for_pipeline, related_questions, related_questions_task, timings, started, background_tasks,
            ambiguous_acts=ambiguous_acts, general_knowledge_used=general_knowledge_used,
        )

    async def _finalize_rag_response(
        self, request: ChatRequest, session_id: str, language: str, memory: dict[str, Any],
        intent: IntentResponse, conv_intent: ConversationIntentMatch, entities: EntityResponse,
        sources: list[SourceCitation], ranked: list[RetrievedChunk],
        recommendation: LawyerRecommendation, explanation_level: str, suggested_actions: list[str],
        interrupting_draft: bool, no_verified_context: bool, answer: str, confidence: float,
        confidence_reason: str, prompt_tokens: int | None, completion_tokens: int | None, failure_reason: str | None,
        question_for_pipeline: str, related_questions: list[str],
        related_questions_task: "asyncio.Task[list[str]] | None", timings: dict[str, float], started: float,
        background_tasks: BackgroundTasks | None, ambiguous_acts: list[str] | None = None,
        general_knowledge_used: bool = False,
    ) -> ChatResponse:
        """The shared post-LLM tail for BOTH `answer()` (via
        `_answer_within_deadline`) and `answer_stream()` -- the code the
        no-verified-context short-circuit and the normal RAG branch fall
        through into on either entry point, once `answer`/`confidence`/
        `confidence_reason`/`prompt_tokens`/`completion_tokens`/
        `failure_reason` are set. Memory updates, history insertion,
        awaiting the related-questions task (skipped when the caller passes
        `related_questions_task=None`, as `answer_stream` does -- see its
        own call site for why), response construction, conversation
        reminders, and logging all go through here exactly once, rather than
        `answer_stream` re-implementing an independent, drifting copy of all
        of it (which is what used to cause the streaming-only
        `pending_section_acts` bug -- see the call site's comment).
        """
        # The one place the strict-RAG refusal invariant is applied on this
        # path, and it is applied BEFORE the memory update and the history
        # insert below -- a refusal that reached here with retrieval results
        # still in scope (grounding validation failed, the LLM emitted the
        # refusal itself, or the quality gate rejected the generated answer)
        # would otherwise be STORED with citations that do not support it,
        # and replayed with them forever. See `app/services/safe_decline.py`.
        sources, ranked, refusal_confidence, is_refusal = clear_support(answer, sources, ranked)
        if is_refusal:
            confidence = refusal_confidence
            no_verified_context = True
            # Confirmed live: retrieval can surface chunks that pass the
            # initial relevance gate (`ranked` non-empty, so the caller took
            # the real-RAG branch, not the `elif no_verified_context:`
            # short-circuit) yet still fail grounding/quality validation
            # further down -- the settled `answer` collapses to the exact
            # same bare refusal string either way. GK fallback's own gate
            # (`app.core.gk_fallback.eligible`) was previously wired only
            # into the zero-retrieval branch, so a Hinglish out-of-KB
            # question whose noisy embedding match dragged in an irrelevant
            # chunk never reached it at all, indistinguishable from GK being
            # disabled from the caller's side. This is the one place both
            # routes converge on the settled refusal, so it is also the one
            # place to give GK fallback a chance regardless of which branch
            # produced it.
            if not general_knowledge_used and settings.general_knowledge_fallback_enabled:
                gk_answer = await general_knowledge_answer(self.llm, request.question, language)
                if gk_answer is not None:
                    answer = gk_answer
                    confidence = _GENERAL_KNOWLEDGE_CONFIDENCE
                    confidence_reason = (
                        "No chunk in the knowledge base cleared the relevance threshold for this question; "
                        "answered from general legal knowledge instead, labeled and unverified."
                    )
                    general_knowledge_used = True
                    is_refusal = False
                    if settings.kb_gap_autofetch_enabled:
                        asyncio.create_task(GapAutoFetchService().record_gap(request.question))
        if not is_refusal:
            sources, ranked = prune_unreferenced_sources(answer, sources, ranked)
        # A General Knowledge fallback answer is a real (if unverified)
        # answer, not the strict-RAG refusal -- `is_refusal` above is already
        # `False` for it (the labeled GK text never matches the refusal
        # string), but `no_verified_context` was set `True` by the caller
        # before it knew GK fallback would succeed. Correct it here, in the
        # one place this invariant is enforced, rather than at every call
        # site: `no_verified_context=True` promises callers empty sources and
        # zero confidence (see its schema docstring), neither of which holds
        # for a GK answer.
        if general_knowledge_used:
            no_verified_context = False

        memory_updates: dict[str, object] = {
            "language_preference": language,
            "current_intent": intent.intent,
            "legal_category": intent.legal_category,
            # A bare "Section N" query with more than one same-numbered Act
            # asked the user which Act they mean (see
            # `_section_lookup_ambiguous_acts`/`_render_act_disambiguation_
            # question`) -- without persisting the original question here, a
            # reply naming just the Act ("Hindu Marriage Act") loses the
            # section number entirely and gets classified as a brand new,
            # vague query, which fails retrieval's relevance gate and answers
            # "not found" even though the Act the user just named is one this
            # KB actually indexes. `_resolve_section_lookup_act_clarification`
            # (see `_dispatch_conversation_intent`) recomposes "Section N of
            # <Act>" from these two fields on the next turn.
            "pending_clarification": "section_lookup_act" if ambiguous_acts else None,
            "pending_section_query": question_for_pipeline if ambiguous_acts else None,
            "pending_section_acts": ambiguous_acts,
        }
        if failure_reason is not None:
            # Rule 1: everything else this turn already wrote (language,
            # topic, etc.) is kept -- only `last_failed_question`/
            # `last_failed_reason` are set. `last_successful_response` is
            # deliberately left untouched, not cleared, so a prior good
            # answer survives this failure for as long as it takes the user
            # to retry it (Rule 4, see `_append_retry_reminder`).
            memory_updates["last_failed_question"] = question_for_pipeline
            memory_updates["last_failed_reason"] = failure_reason
            # Part 49: a fresh failure gets its own one-time reminder, even
            # if an earlier still-pending failure's reminder was already
            # shown and not yet retried (see `_append_retry_reminder`).
            memory_updates["retry_reminder_shown"] = False
        else:
            memory_updates["last_successful_response"] = answer
        await self.memory.update(session_id, **memory_updates)
        await self.memory.append(session_id, "assistant", answer)
        if background_tasks is not None:
            background_tasks.add_task(self.memory.summarize_if_needed, session_id, self.llm)
        else:
            await self.memory.summarize_if_needed(session_id, self.llm)
        message_id = str(uuid4())
        await self.history.insert(
            {
                "message_id": message_id,
                "session_id": session_id,
                "conversation_id": request.conversation_id,
                # Security finding C2: this must be the SERVER-VERIFIED owner
                # (`memory["owner_user_id"]`, set by `check_access`/
                # `_claim_session_if_unowned` from the caller's JWT), never
                # the client-supplied `request.user_id` -- stamping the
                # latter here made `erase_user_data`'s
                # `ChatRepository().delete_by_user(user_id)` match nothing
                # for a real authenticated user (their messages were never
                # actually tagged with their real id), so `DELETE /me/data`
                # reported success while leaving every chat message behind.
                "user_id": memory.get("owner_user_id"),
                "question": request.question,
                "answer": answer,
                "intent": intent.model_dump(),
                "conversation_intent": conv_intent.intent,
                "entities": entities.entities,
                "sources": [source.model_dump() for source in sources],
            }
        )
        if related_questions_task is not None:
            # By this point the cache-store/memory-persist/history-insert
            # writes above have already run concurrently with this task, so
            # in the common case it's finished (or nearly so) and this await
            # adds little to nothing -- only a genuinely slow related-
            # questions call still costs real time here, same as before, but
            # no longer stacked serially on top of the unrelated writes too.
            stage_start = time.perf_counter()
            related_questions = await related_questions_task
            timings["related_questions_ms"] = (time.perf_counter() - stage_start) * 1000
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        log.info("chat_pipeline_timing", session_id=session_id, message_id=message_id, intent=intent.intent, **timings)
        response = ChatResponse(
            message_id=message_id,
            session_id=session_id,
            answer=answer,
            sources=sources,
            evidence_pages=evidence_pages_from_citations(sources),
            currency_notice=_currency_notice_for(sources, request.question),
            applicable_law=_applicable_law_from(sources),
            no_verified_context=no_verified_context,
            general_knowledge_used=general_knowledge_used,
            **_confidence_fields(sources, ranked, confidence),
            confidence=confidence,
            confidence_label=self._confidence_label(confidence),
            confidence_reason=confidence_reason,
            lawyer_recommendation=recommendation,
            retrieved_sections=[str(chunk.metadata.get("section_number")) for chunk in ranked if chunk.metadata.get("section_number")],
            retrieved_chunks=ranked,
            detected_language=language,
            detected_intent=intent.intent,
            conversation_intent=conv_intent.intent,
            detected_intents=list(conv_intent.detected_intents) or [conv_intent.intent],
            intent_reason=conv_intent.reason,
            explanation_level=explanation_level,
            suggested_actions=suggested_actions,
            related_questions=related_questions,
            llm_provider=self.llm.provider_name,
            llm_model=getattr(self.llm, "model", ""),
            extracted_entities=entities.entities,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retryable=failure_reason is not None,
            failure_category=self._failure_category(failure_reason),
            # Machine-readable companion to the model-law caveat appended by
            # `_polish_grounded_answer`, so a UI can surface "this depends on
            # your State" without parsing the prose.
            warnings=[
                *jurisdiction_warnings(answer, request.question),
                *period_warnings(answer, [chunk.text for chunk in ranked]),
            ],
        )
        response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
        self._log_routing_decision(
            session_id, message_id, request.question, conv_intent.intent,
            "general_knowledge_fallback" if general_knowledge_used else ("no_verified_context" if no_verified_context else "rag"),
            # True whenever the question actually answered came from resolving
            # a "Follow-up Question" against conversation memory (e.g. "Why?"
            # -> "Why should I file an FIR for the bike theft?") rather than
            # the user's literal text -- still `rag_used=True` alongside it,
            # since resolving the reference is what MADE a real retrieval
            # possible, not a substitute for one.
            memory_hit=question_for_pipeline != request.question,
            rag_used=True, response_modification=False, draft_mode=bool(memory.get("draft_mode")),
            request_failed=failure_reason is not None,
        )
        await self._log_query(
            request, session_id, message_id, response, background_tasks,
            owner_user_id=memory.get("owner_user_id"),
        )
        return response

    async def _resolve_matter_context(
        self, request: ChatRequest, session_id: str, language: str, memory: dict[str, Any],
        intent: IntentResponse, started: float, background_tasks: BackgroundTasks | None,
        authenticated_user_id: str | None,
    ) -> tuple[MatterContext, ChatResponse | None]:
        """Jurisdiction Routing (Phase 2): resolves WHICH State(s)/locality/
        date govern this matter (`app.rag.matter_context.resolve_matter_context`)
        and, if that resolution says the question cannot be safely answered
        without knowing the State, builds the clarifying-question response.

        Called from BOTH `_answer_within_deadline` and `answer_stream`,
        BEFORE the response-cache lookup in each -- the cache key itself must
        vary by resolved jurisdiction (objective item 6), so this cannot run
        any later than intent detection.

        The profile default (`app.services.phase3.PreferenceService`) is
        looked up ONLY from `authenticated_user_id` -- the server-verified
        JWT subject, never any client-supplied identifier
        (`ChatRequest` carries no user id field at all) -- so an anonymous
        session simply has no profile default, which is the correct,
        unexploitable behavior rather than a gap to paper over.
        """
        profile_state_code = None
        if authenticated_user_id:
            prefs = await self.preferences.get(authenticated_user_id)
            profile_state_code = prefs.profile_state_code
        pending = memory.get("pending_clarification")
        matter = resolve_matter_context(
            request.question, memory, profile_state_code, intent.legal_category,
            awaiting_clarification=pending == "matter_jurisdiction",
            awaiting_locality_clarification=pending == "matter_jurisdiction_locality",
        )
        await self.memory.update(
            session_id,
            matter_context=matter.to_memory(),
            pending_clarification="matter_jurisdiction" if matter.needs_clarification else None,
        )
        if not matter.needs_clarification:
            return matter, None
        answer = clarification_question(language)
        recommendation = await self._contextual_recommendation(memory, "Matter Jurisdiction Clarification")
        response = await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Matter Jurisdiction Clarification", started,
            background_tasks, recommendation, [], matter.clarification_reason or "State not resolved.",
            confidence=0.5, pending_clarification="matter_jurisdiction",
        )
        return matter, response

    async def _prepare_rag_context(
        self, question_for_pipeline: str, language: str, request: ChatRequest,
        intent: IntentResponse, timings: dict[str, float],
        session_id: str, authenticated_user_id: str | None = None, memory: dict[str, Any] | None = None,
        matter_context: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        """Shared by `answer()` and `answer_stream()`: entity extraction, vector
        retrieval, and the lawyer-recommendation lookup have no data
        dependency on each other -- recommendation only needs `intent`
        (already resolved by the caller), and retrieval doesn't take
        `entities` as an input at all (only used later, in prompt
        rendering/confidence/logging). Running them concurrently instead of
        as three sequential awaits is a pure latency win with no behavior
        change. `reranker.rerank` is the only one of the three with a real
        dependency (on the retrieved chunks), so it still runs after.
        """
        # Part 45 "Per-User Document Isolation" + Part 46 "Authenticated User
        # Ownership": injected last so it always wins even if a client tries
        # to pass its own `owner_session_id`/`owner_user_id` in
        # `metadata_filters` to spoof another session or account -- never
        # trust the client for the security-relevant key.
        #
        # Three-way visibility, expressed as an actual `$or` (see
        # `MongoVectorStore._mongo_filter`/`BM25Index._matches_filters`):
        # branch 1 (`shared_branch`) is the truly-unowned Knowledge Base --
        # Phase 1 "Jurisdiction-Aware Knowledge Base" gap 1 fix: this is the
        # ONLY branch that requires `review_status=approved`
        # (`kb_jurisdiction.shared_retrieval_filters`), and it is embedded
        # HERE, inside the shared branch specifically, rather than as a
        # top-level filter -- a top-level `review_status` key would AND
        # against every branch below too, and a private document never
        # carries that field at all, so it would make a user's OWN uploads
        # invisible to themselves. An earlier version of this fix used
        # `[None, "approved"]` at the top level, which let any chunk with NO
        # `review_status` at all through the shared branch too -- silently
        # re-opening exactly the hole Phase 1 exists to close for every
        # document indexed before this phase, migrated or not. Branch 2
        # (`session_owned_branch`) is Part 45's session-ownership check,
        # split out from the shared branch so it carries no review-status
        # requirement -- additionally requires `owner_user_id` be absent, so
        # a chunk that DOES have a real owner can only ever be reached
        # through branch 3's exact-match check, never through a
        # guessed/coincidental `session_id` match. Branch 3 (only added when
        # authenticated) is what makes account ownership survive a brand new
        # session -- unlike `owner_session_id`, `owner_user_id` isn't tied to
        # this one conversation.
        shared_branch = {"owner_session_id": [None], "owner_user_id": [None], **shared_retrieval_filters()}
        # Jurisdiction Routing (Phase 2): the resolved State(s)/locality
        # constraint, folded into the SHARED branch only -- same reasoning as
        # `review_status` immediately above (a private document has no
        # `applicability` field at all, so a top-level jurisdiction filter
        # would make a user's own uploads invisible to themselves). Empty
        # when the question is State-insensitive (`jurisdiction_or_branches`
        # returns `[]`), in which case no jurisdiction constraint is added at
        # all -- see that function's own docstring.
        matter_state_codes = (matter_context or {}).get("state_codes") or []
        jurisdiction_or = jurisdiction_or_branches(matter_state_codes, (matter_context or {}).get("locality"))
        if jurisdiction_or:
            shared_branch["$or"] = jurisdiction_or
        # Candidate-selection-level date eligibility (same shared-branch-only
        # placement as jurisdiction/review_status just above, and the same
        # reasoning: a private document has no `effective_from`/`effective_to`
        # either, so this must never reach an ownership branch). See
        # `kb_jurisdiction.TEMPORAL_FILTER_KEY`'s comment for why this is
        # enforced here -- inside the Mongo/Atlas query itself -- rather than
        # relying solely on `LegalRetriever.retrieve`'s post-filter.
        shared_branch.update(temporal_filter((matter_context or {}).get("as_of_date")))
        session_owned_branch = {"owner_session_id": [session_id], "owner_user_id": [None]}
        or_branches = [shared_branch, session_owned_branch]
        if authenticated_user_id:
            or_branches.append({"owner_user_id": [authenticated_user_id]})
        # A client-supplied `review_status` in `metadata_filters` must never
        # reach the query as a top-level key -- see the retriever-side
        # comment on why a top-level `review_status` breaks private-document
        # visibility; only `shared_branch` above is allowed to carry it.
        client_filters = {k: v for k, v in (request.metadata_filters or {}).items() if k != "review_status"}
        filters = {**client_filters, "$or": or_branches}
        # Bare "Section 2"-style queries that name no Act at all are
        # genuinely ambiguous from the current question alone (see
        # `LegalRetriever._apply_section_number_floor`'s multi-Act
        # tie-break) -- the last couple of turns are the cheapest available
        # disambiguation signal (no new persistence, `memory["messages"]`
        # already loaded for this turn), used only as a soft hint, never a
        # filter, so a stale/irrelevant recent turn just fails to help
        # rather than steering retrieval anywhere wrong.
        recent_messages = (memory or {}).get("messages") or []
        context_hint = " ".join(msg.get("content", "") for msg in recent_messages[-3:])
        # Conservative typo tolerance for RETRIEVAL only. `retrieval_query` is
        # the user's original wording PLUS the safe normalized aliases (see
        # `app/language/typo_tolerance.py`), never a replacement for it -- so
        # an exact match on what the user actually typed can still win, and a
        # word the normalizer declined to touch is still searched as typed.
        # Confirmed failure this addresses: "chek bouns hone pr secshun 138 me
        # kya hota hai?" retrieved nothing and was answered "no verified
        # document ... is available", because no chunk contains "chek",
        # "bouns" or "secshun".
        #
        # Deliberately NOT applied to `entity_extractor.extract`, which reads
        # names, dates and amounts out of the message and must see exactly
        # what the user wrote; nor to the prompt, which is built from
        # `question_for_pipeline` unchanged.
        retrieval_query = normalize_for_routing(question_for_pipeline).retrieval_query
        stage_start = time.perf_counter()
        gathered = asyncio.gather(
            self.entity_extractor.extract(question_for_pipeline, language),
            self.retriever.retrieve(
                retrieval_query, top_k=10, filters=filters, intent=intent.intent, context_hint=context_hint,
                matter_context=matter_context,
            ),
            self.recommendations.recommend(intent.intent, intent.legal_category),
        )
        # Unlike every LLM call in this pipeline, retrieval (embedding
        # inference + the Mongo/BM25 legs) had NO deadline enforcement at
        # all -- the request-wide `deadline()` context (set by `answer()`,
        # see its docstring's incident writeup) only bounds `self.llm.chat()`
        # calls, which read it via `has_time_for()`/`clamp_timeout()`;
        # nothing here ever checked it. Confirmed live: a request's
        # `entities_retrieval_recommendation_ms` measured 494146ms (8m14s) --
        # the embedding leg stalled, almost certainly on GPU/Mongo contention
        # from a concurrent ingestion job -- while `remaining_seconds()` had
        # already gone to -344s by the time control reached the LLM stage
        # that actually checks it. `asyncio.wait_for` here closes that gap:
        # a stuck retrieval now degrades to the same graceful
        # `no_verified_context` fallback empty retrieval already produces,
        # inside the SAME budget every other stage respects, instead of
        # holding the connection open for minutes with no ceiling at all.
        budget = remaining_seconds()
        try:
            if budget is not None and budget <= 0:
                raise TimeoutError
            entities, (rewritten, retrieved), recommendation = await (
                asyncio.wait_for(gathered, timeout=budget) if budget is not None else gathered
            )
        except TimeoutError:
            log.warning("retrieval_deadline_exhausted", remaining_seconds=budget)
            entities = EntityResponse(entities={}, confidence=0.0)
            rewritten, retrieved = retrieval_query, []
            recommendation = await self.recommendations.recommend(intent.intent, intent.legal_category)
        if is_cheque_notice_timing_query(retrieval_query):
            for chunk in retrieved:
                _normalize_ni_138_heading(chunk)
        timings["entities_retrieval_recommendation_ms"] = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        # Snapshotted BEFORE `reranker.rerank()` runs, for the jurisdiction
        # check below. Confirmed live (2026-09-16): `LegalReranker.rerank`
        # mutates each `RetrievedChunk.score` IN PLACE on the very same
        # objects `retrieved` holds -- it does not return copies. The
        # jurisdiction check's own comment below says it deliberately reads
        # `retrieved`'s RAW pre-rerank scores, but by the time that line
        # runs, those exact chunk objects have already been overwritten by
        # this rerank() call, so it was silently reading POST-rerank scores
        # the entire time. Confirmed impact: "bharatiya nyaya sanhita
        # section 12" (no real match in this corpus) retrieved 10 completely
        # unrelated Maharashtra/UP Acts at a genuine raw score of ~0.015 --
        # correctly below `_MIN_ACCEPTED_CONTEXT_SCORE` (0.12) and therefore
        # correctly excluded from the jurisdiction check -- but the
        # reranker's own heuristic bonuses (a flat +0.08 for merely having
        # `section_number` metadata, +0.10 for a coincidental legal_category
        # match, plus raw word overlap on generic terms like "section")
        # pushed those same mutated objects' scores up to ~0.23-0.27,
        # clearing the threshold and producing a false multi-State
        # ambiguity purely from noise the relevance gate would otherwise
        # have rejected.
        raw_scores_by_chunk_id = {chunk.chunk_id: chunk.score for chunk in retrieved}
        ranked = await self.reranker.rerank(rewritten, retrieved, top_k=6, legal_category=intent.legal_category)
        if intent.intent == "SECTION_LOOKUP":
            ranked = self._ensure_section_match_survives(retrieval_query, retrieved, ranked)
        ranked = self._ensure_article_match_survives(retrieval_query, retrieved, ranked)
        if settings.retrieval_debug:
            log.info(
                "reranker_debug",
                query=question_for_pipeline[:200],
                reranked_top=[
                    (chunk.chunk_id, round(chunk.score, 4), chunk.metadata.get("source_document")) for chunk in ranked
                ],
            )
        ranked = self._reorder_by_relevance(retrieval_query, ranked)
        ranked = self._filter_relevant_context(
            retrieval_query, rewritten, ranked, intent=intent.intent, language=language
        )
        timings["reranking_ms"] = (time.perf_counter() - stage_start) * 1000

        sources: list[SourceCitation] = []
        seen_sources: set[tuple[str, str | None, str | None]] = set()
        for chunk in ranked:
            if not chunk.metadata.get("source_document"):
                continue
            citation = self._citation_from_chunk(chunk)
            key = (citation.source_document, citation.section, citation.article)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            sources.append(citation)
        # Jurisdiction Routing (Phase 2): checked against `retrieved` (the
        # RAW pre-rerank/pre-relevance-filter candidate pool, top_k=10),
        # never `ranked` (top_k=6, further narrowed by `_filter_relevant_
        # context`'s relevance gate) -- reranking legitimately favors
        # whichever State's phrasing happens to match this ONE query best,
        # which can silently drop the OTHER State's conflicting chunk out of
        # `ranked` entirely. Checking the wider pool is what keeps a genuine
        # multi-State disagreement from being filtered away before the
        # ambiguity backstop (`ChatService`'s `jurisdiction_ambiguity` check)
        # ever sees it -- computed here, once, rather than in both callers
        # against two different (and differently-vulnerable) chunk lists.
        # Still the wider pre-relevance-gate pool (see comment above), but a
        # bare similarity-score floor first -- without it, a query carrying
        # no real legal content (a greeting, a typo, small talk) can still
        # surface a handful of State-tagged chunks purely by embedding-space
        # coincidence once the corpus has enough State-specific content
        # indexed (confirmed: "hlo kya haal h" triggered a false "which
        # State?" clarification once UP/Maharashtra/MP bulk ingestion made
        # State-tagged chunks common enough to appear in ANY top-10 pool).
        # `_MIN_ACCEPTED_CONTEXT_SCORE` is deliberately low and shared with
        # the main relevance gate -- this excludes only genuine noise
        # matches, not a real-but-modest cross-State conflict. Reads the
        # snapshot taken above `reranker.rerank()`, not `chunk.score`
        # directly -- see that snapshot's own comment for why reading
        # `retrieved`'s scores here, after rerank() has run, would silently
        # observe post-rerank values instead.
        jurisdiction_candidates = [
            chunk for chunk in retrieved
            if raw_scores_by_chunk_id.get(chunk.chunk_id, 0.0) >= _MIN_ACCEPTED_CONTEXT_SCORE
        ]
        jurisdiction_ambiguity = (
            None if (matter_context or {}).get("state_codes") else detect_jurisdiction_ambiguity(jurisdiction_candidates)
        )
        return entities, ranked, sources, recommendation, jurisdiction_ambiguity

    @staticmethod
    def _ensure_article_match_survives(query: str, retrieved: list[Any], ranked: list[Any]) -> list[Any]:
        """Keep an exact Article metadata hit when reranking truncates it."""
        match = re.search(r"\barticle\s+(\d{1,3}[A-Z]?)\b", query, re.IGNORECASE)
        if not match:
            return ranked
        number = match.group(1).upper()
        ranked_ids = {chunk.chunk_id for chunk in ranked}
        return [
            *ranked,
            *(
                chunk
                for chunk in retrieved
                if chunk.chunk_id not in ranked_ids
                and str(
                    chunk.metadata.get("article_number") or chunk.metadata.get("section_number") or ""
                ).upper() == number
            ),
        ]

    def _ensure_section_match_survives(self, query: str, retrieved: list[Any], ranked: list[Any]) -> list[Any]:
        """Phase 6 "Retrieval Recall Fix, continued": `reranker.rerank()`'s fixed
        `top_k=6` can discard a chunk that exactly matches a SECTION_LOOKUP query's
        section number but scores too low in the reranker's blended formula to make
        the cut -- confirmed live for "Section 420" (BM25 rank #1 on its own, but
        reranked-and-truncated away since its raw retrieval score is a small
        fraction of what topic-bonus-heavy unrelated candidates get). Mirrors
        `_is_relevant_chunk`'s exact-match guarantee (Phase 4) one stage earlier:
        if `retrieved` contains a chunk whose `section_number` exactly matches the
        query and reranking already dropped it, append it (never reorders or
        removes anything reranker/rerank already decided) so the filter stage
        still gets a chance to accept it via that same exact-match rule. Scoped to
        SECTION_LOOKUP only (checked by the caller) and to an exact numeric match
        only -- never touches ranking/acceptance for any other intent or query.
        """
        query_section_number = parse_section_lookup(query.lower().strip())
        if not query_section_number:
            return ranked
        ranked_ids = {chunk.chunk_id for chunk in ranked}
        for chunk in retrieved:
            if chunk.chunk_id in ranked_ids:
                continue
            if chunk.metadata.get("section_number") == query_section_number:
                ranked.append(chunk)
        return ranked

    def _filter_relevant_context(
        self, query: str, rewritten_query: str, chunks: list[RetrievedChunk],
        intent: str | None = None, language: str | None = None
    ) -> list[Any]:
        """Delegates to `app.rag.relevance.filter_relevant_context` (Phase 1
        god-object split) -- kept as a same-named method because several
        tests call it directly on a `ChatService` instance."""
        return filter_relevant_context(query, rewritten_query, chunks, intent=intent, language=language)

    def _is_relevant_chunk(
        self, query: str, rewritten_query: str, chunk: RetrievedChunk,
        intent: str | None = None, language: str | None = None
    ) -> bool:
        """Delegates to `app.rag.relevance.is_relevant_chunk` -- see
        `_filter_relevant_context`'s docstring for why this stays a method."""
        return is_relevant_chunk(query, rewritten_query, chunk, intent=intent, language=language)

    def _reorder_by_relevance(self, query: str, chunks: list[Any]) -> list[Any]:
        """Delegates to `app.rag.relevance.reorder_by_relevance` -- see
        `_filter_relevant_context`'s docstring for why this stays a method."""
        return reorder_by_relevance(query, chunks)

    def _build_rag_messages(
        self,
        question_for_pipeline: str,
        language: str,
        intent: IntentResponse,
        conv_intent: ConversationIntentMatch,
        entities: EntityResponse,
        ranked: list[RetrievedChunk],
        explanation_level: str,
    ) -> list[ChatMessage]:
        # Each chunk is capped defensively -- most are already reasonably
        # sized at ingest time, but nothing upstream enforces an upper bound,
        # and an unusually long chunk would otherwise inflate the prompt (and
        # LLM latency) for no benefit past what the model can usefully use
        # anyway. Was 1200 chars, which turned out to be too tight for some
        # FAQ-style ingested chunks that open with a different, unrelated
        # Q&A before the part that actually answers the current question --
        # at 1200 chars that relevant part (and sometimes the operative
        # sentence of the matching Q&A itself) got sliced off entirely,
        # leaving the model with only the unrelated lead-in and no way to
        # answer. With at most `top_k=6` chunks ever reaching this point,
        # even the raised cap keeps total context well under any provider's
        # window.
        # Part 58 "Answer Quality Audit" issue 24: each block used to be
        # labelled "[Source 1]" plus a raw `dict` repr of the metadata. The
        # model duly cited what it was given -- confirmed live, an answer
        # opened "Source 2 ke mutabiq..." ("according to Source 2"), which is
        # meaningless to a reader who cannot see the numbered blocks and does
        # not name the Act, the section, or the document. Leading each block
        # with a readable citation gives the model something worth quoting;
        # the ordinal is retained only so the model can tell two blocks apart
        # internally, and the system prompt forbids surfacing it.
        context = "\n\n".join(
            f"[Source {index + 1} — {_source_label(chunk)}]\n{chunk.text[:3000]}"
            for index, chunk in enumerate(ranked)
        )
        system_prompt = prompt_registry.render(
            "system_prompt",
            language=language,
            explanation_level_instruction=EXPLANATION_LEVEL_INSTRUCTIONS[explanation_level],
            # Rule 2 used to hardcode the Hindi refusal and rely on the LLM to
            # translate it into {language} on the fly -- but the "write
            # everything in {language}" rule elsewhere in this same prompt
            # told it to do exactly that, so the two rules conflicted and the
            # LLM produced a fresh paraphrase (sometimes mixed-script, e.g.
            # Punjabi text with literal untranslated Hindi words in it) every
            # time, almost never matching one of `NO_VERIFIED_CONTEXT_
            # MESSAGES` byte-for-byte. `is_no_verified_context()` -- and
            # therefore the whole safe-decline path (`_no_verified_context_
            # answer`'s guidance block, `safe_decline.py`'s citation-clearing)
            # -- never fired for those paraphrases: a refusal-shaped answer
            # sailed through as a normal, confident answer with unrelated
            # citations still attached. Handing the model the ALREADY-CORRECT
            # string for this exact language removes the translation step
            # (and its variance) entirely -- it only has to copy, never
            # compose -- so the exact-match check below reliably recognizes
            # it again.
            no_verified_context_line=no_verified_context_message(language),
        )
        rag_prompt = prompt_registry.render(
            "rag_prompt",
            question=question_for_pipeline,
            language=language,
            intent=intent.intent,
            conversation_intent=f"{conv_intent.intent} — {CONVERSATION_INTENT_HINTS.get(conv_intent.intent, '')}",
            entities=entities.entities,
            context=context,
            # Part 58 issue 2: empty for the overwhelming majority of turns;
            # populated only when the question or the retrieved material
            # actually cites a repealed IPC/CrPC/Evidence Act provision.
            statutory_currency_note=currency_directive(question_for_pipeline, context) or "None.",
            # Same shape as the currency note above and empty just as often:
            # populated only when a MODEL law (one that binds nobody until a
            # State enacts it) is in play. Reported live: the Model Tenancy
            # Act, 2021 was described as though it governed tenancies
            # nationwide, naming a "Rent Authority" that may not exist in the
            # user's State.
            jurisdiction_note=jurisdiction_directive(question_for_pipeline, context) or "None.",
        )
        return [ChatMessage(role="system", content=system_prompt), ChatMessage(role="user", content=rag_prompt)]

    @staticmethod
    def _has_strong_grounding(ranked: list[Any]) -> bool:
        """Whether `ranked` (already relevance-gated -- see `_prepare_rag_
        context`) contains at least one chunk scored well past the ordinary
        acceptance bar. See `_STRONG_GROUNDING_RETRY_SCORE`'s own comment for
        the exact threshold and why it's set there."""
        return any(chunk.score >= _STRONG_GROUNDING_RETRY_SCORE for chunk in ranked)

    async def _call_llm_with_grounding_retry(
        self, messages: list[ChatMessage], ranked: list[Any], session_id: str,
    ) -> LLMResponse:
        """QA pass 2026-09-24 ("LLM non-deterministic refusal despite good
        context", see `_STRONG_GROUNDING_RETRY_SCORE`'s comment): one bounded
        retry when the LLM's own Rule 2 refusal contradicts already
        relevance-gated, strongly-scored retrieval evidence. Never retries a
        genuine LLM/provider failure (`response.error` set -- that's
        `call_with_hard_timeout`'s job, not this one) or a refusal over
        weak/absent context, where the refusal is very likely correct and a
        retry would only add latency to the common, legitimate case. Still
        never bypasses `validate_grounding`/the quality gate downstream in
        `answer()` -- this only gives the LLM a second chance to use context
        it was already given; it cannot force an answer through if the retry
        itself doesn't produce one that passes those checks.

        Live-confirmed (2026-09-24, restarted backend, real corpus, real
        LLM provider): an identical-messages retry at this app's default
        `temperature=0.1` frequently reproduces the SAME refusal verbatim --
        the non-determinism the QA report observed across separate `/chat`
        calls (1 of 3 grounded) came from LLM-provider-level sampling
        variance across calls, not from anything this retry alone
        reintroduces by resending byte-identical input at a near-zero
        temperature. The retry attempt therefore raises `temperature` (to
        `_GROUNDING_RETRY_TEMPERATURE`) and appends one extra system message
        naming the concrete failure mode (Rule 2 fired despite the context
        actually answering the question) -- never new facts, never a
        suggested answer, only a nudge to re-read the SAME `<context>` it
        was already given. The retry is still held to every rule the first
        attempt was: Rule 2 remains fully available to it, and every
        downstream check (`validate_grounding`, the quality gate) applies to
        whatever it produces exactly as it would to a first attempt.
        """
        _GROUNDING_RETRY_TEMPERATURE = 0.6
        _RETRY_REINFORCEMENT = ChatMessage(
            role="system",
            content=(
                "Your previous attempt at this exact question output Rule 2's exact refusal line, but the "
                "<context> block above was independently verified as strongly relevant to this question. "
                "Re-read the <context> carefully before deciding again. If, on this closer reading, it genuinely "
                "does answer the question, answer from it and cite it normally, following every other rule "
                "unchanged. If it genuinely does not, Rule 2 still applies -- output its exact line again. Do not "
                "invent facts or citations not present in the <context> either way."
            ),
        )

        async def _call_once(temperature: float, extra_messages: list[ChatMessage] | None = None) -> LLMResponse:
            call_messages = messages if not extra_messages else [*messages, *extra_messages]
            try:
                return await call_with_hard_timeout(
                    self.llm.chat(call_messages, temperature=temperature),
                    fallback_seconds=settings.chat_request_budget_seconds,
                )
            except TimeoutError:
                log.warning("chat_llm_call_hard_timeout", session_id=session_id)
                return LLMResponse(
                    content="The assistant did not respond within its time budget.",
                    model=getattr(self.llm, "model", ""),
                    provider=getattr(self.llm, "provider_name", ""),
                    error="Request exceeded its time budget before an answer could be produced.",
                    error_kind="timeout",
                )

        response = await _call_once(temperature=0.1)
        if (
            response.error is None
            and is_no_verified_context(self._safe_llm_text(response, ""))
            and self._has_strong_grounding(ranked)
        ):
            metrics.increment("grounding_refusal_retry_triggered")
            log.info(
                "llm_refusal_retry_triggered", session_id=session_id,
                top_score=round(max((chunk.score for chunk in ranked), default=0.0), 4),
            )
            retry_response = await _call_once(
                temperature=_GROUNDING_RETRY_TEMPERATURE, extra_messages=[_RETRY_REINFORCEMENT]
            )
            if retry_response.error is None:
                still_refused = is_no_verified_context(self._safe_llm_text(retry_response, ""))
                metrics.increment("grounding_refusal_retry_still_refused" if still_refused else "grounding_refusal_retry_recovered")
                response = retry_response
        return response

    async def _no_verified_context_answer(self, question: str, language: str) -> str:
        """The strict-RAG refusal, plus concrete next steps when the question
        is one where a bare refusal strands the user (Part 58 issues 16/23).

        The refusal itself is unchanged and still comes first, verbatim -- the
        guardrail against answering from pre-trained knowledge is untouched.
        What follows it is fixed, reviewed, non-legal text: published
        Government of India helplines and generic evidence/reporting steps
        that hold whichever provision turns out to apply. Confirmed live
        failure this addresses: "a man keeps phoning me, harassing and
        threatening me -- how do I write a police complaint?" received the
        bare refusal and nothing else.
        """
        refusal = no_verified_context_message(language)
        category = detect_category(question)
        template = get_template(category.draft_id) if category and category.draft_id else None
        guidance = actionable_guidance(
            question,
            language,
            template_name=localized_document_name(template, language) if template else None,
            template_command=template.name if template else None,
        )
        # No urgency category matched -- this is an ordinary informational
        # question ("BNS Section 12 kya hai", "how to send a legal notice"),
        # not a safety situation, but a bare refusal still stranded the user
        # exactly the same way (Part 58's own reasoning, applied to the
        # non-urgent case it didn't originally cover). `generic_next_steps_
        # guidance` never returns "", so this always has SOME guidance to
        # append once the urgency check has already been tried and missed.
        if not guidance:
            guidance = generic_next_steps_guidance(language)
        # Fire-and-forget: schedules a background attempt to identify and
        # auto-fetch whichever Act this question was about, so a later user
        # asking the same thing may get a real answer instead of this
        # refusal again. `record_gap` itself no-ops instantly unless an
        # operator has opted in (`settings.kb_gap_autofetch_enabled`), and
        # never raises -- this must never affect, delay, or fail the refusal
        # already being returned. See `app/services/kb_gap_autofetch.py`.
        if settings.kb_gap_autofetch_enabled:
            asyncio.create_task(GapAutoFetchService().record_gap(question))
        parts = [refusal, guidance] if guidance else [refusal]
        disclosure = await self._unverified_candidate_disclosure(question, language)
        if disclosure:
            parts.append(disclosure)
        return "\n\n".join(parts)

    async def _unverified_candidate_disclosure(self, question: str, language: str) -> str | None:
        """A `needs_review` excerpt to disclose alongside the refusal above, or
        `None`.

        Deliberately separate from the confident-answer path in every way:
        - queried only after the real (`review_status=approved`) search has
          already returned nothing for this question, never instead of it;
        - the excerpt is quoted VERBATIM in `unverified_source_disclosure` --
          no LLM call sits between the retrieved chunk and what the user
          sees, so this can never itself state something the source doesn't;
        - never writes `review_status`/`verification_status` anywhere -- a
          document a human later rejects was simply shown, disclosed as
          unverified, once;
        - held to the SAME relevance bar (`_MIN_ACCEPTED_CONTEXT_SCORE`) the
          confident path uses, so this isn't a back door for lower-quality
          matches than the strict-RAG guardrail would otherwise accept -- it
          only ever surfaces something as strong as what a confident answer
          would have used, just not yet human-checked.

        Fails silent (returns `None`) on any error: a broken secondary lookup
        must never turn an already-decided refusal into an unhandled
        exception on the user's request.
        """
        if not settings.chat_unverified_disclosure_enabled:
            return None
        try:
            rewritten, candidates = await self.retriever.retrieve(
                question, top_k=3,
                filters={"review_status": REVIEW_NEEDS_REVIEW, "owner_session_id": [None], "owner_user_id": [None]},
            )
            relevant = filter_relevant_context(question, rewritten, candidates)
            if not relevant:
                return None
            top = relevant[0]
            excerpt = top.text.strip()
            if not excerpt:
                return None
            if len(excerpt) > 700:
                excerpt = excerpt[:700].rsplit(" ", 1)[0] + "…"
            source_label = str(
                top.metadata.get("act_name") or top.metadata.get("source_document") or "Unnamed source"
            )
            return unverified_source_disclosure(language, source_label, excerpt)
        except Exception as exc:  # noqa: BLE001 - a broken secondary lookup must not break the refusal it augments
            log.warning("unverified_candidate_disclosure_failed", error=type(exc).__name__)
            return None

    async def _apply_quality_gate(
        self, answer: str, *, request: ChatRequest, messages: list[ChatMessage], language: str,
        sources: list[SourceCitation], ranked: list[RetrievedChunk],
    ) -> tuple[str, QualityVerdict]:
        """Phase 1 item 3: runs the legal-answer quality gate and acts on it.

        Returns `(answer, verdict)`. A `REJECT` verdict swaps in the same safe
        verified-context fallback used when retrieval finds nothing -- an
        answer that leaks a provider error, quotes an internal source ordinal,
        cites nothing identifiable, or is about a different topic entirely is
        worse than no answer, because the reader has no way to tell.

        A `RETRY_LANGUAGE` verdict gets ONE bounded regeneration with the
        language requirement restated, rather than being discarded: the answer
        is probably legally correct and only mis-presented, and this is the
        single failure the model can reliably fix when told plainly. Bounded
        at one retry -- repeatedly re-prompting for language costs latency on
        the critical path and has never been observed to succeed on a third
        attempt when it failed twice. The returned verdict reflects the
        RETRY's outcome, so the caller can still cap confidence if it missed
        again.
        """
        verdict = evaluate_answer_quality(
            answer, question=request.question, language=language, sources=sources, ranked_chunks=ranked,
            disclaimer=LEGAL_DISCLAIMER,
        )
        if verdict.severity is AnswerSeverity.PASS:
            return answer, verdict

        if verdict.severity is AnswerSeverity.REJECT:
            log.warning(
                "answer_quality_rejected",
                failed_check=verdict.failed_check,
                reason=verdict.reason,
                language=language,
            )
            return await self._no_verified_context_answer(request.question, language), verdict

        if verdict.severity is AnswerSeverity.RETRY_SECTION_ATTRIBUTION:
            # Confirmed live: an otherwise-good FIR-procedure answer
            # attached the right to escalate to the Superintendent of
            # Police to "Section 64" -- a real, genuinely retrieved
            # section, just an entirely unrelated one (rape/arrest
            # procedure, not FIR refusal). `section_grounding` above only
            # confirms the cited NUMBER appears somewhere in context, never
            # that this SPECIFIC claim belongs to it -- see `_section_
            # attribution_mismatch`'s own docstring. Bounded to one retry,
            # same reasoning as RETRY_LANGUAGE below: the answer is very
            # likely mostly correct, and telling the model exactly which
            # claim was misattributed is a fix it can reliably make, or
            # honestly walk back to "no specific section" if genuinely
            # nothing in context supports it.
            log.info("answer_quality_section_attribution_retry", reason=verdict.reason)
            retried, retry_verdict = await self._retry_with_reinforcement(
                f"{verdict.reason} Look again at the retrieved context: either cite the section that "
                "actually supports this specific claim, or -- if nothing retrieved states it -- remove "
                "the specific section-number citation for that one claim and describe it as a general "
                "step without asserting a provision number. Keep every other part of the answer exactly "
                "as it was. Output only the rewritten answer.",
                answer=answer, verdict=verdict, messages=messages, request=request,
                language=language, sources=sources, ranked=ranked,
            )
            # Unlike a language retry (still a useful answer even if it
            # misses again), a repeated section-attribution failure means
            # the model could not honestly ground this specific claim on a
            # second, explicitly-guided attempt -- confirmed live: told
            # "Section 173 doesn't support this", the retry simply swapped
            # in a DIFFERENT wrong number ("Section 64") instead of
            # dropping the citation. A confidently-cited but unsupported
            # section is actively misleading, not merely imperfect, so this
            # escalates to the same safe refusal an outright REJECT gets
            # rather than shipping a second wrong guess. `_retry_with_
            # reinforcement` itself already reverts a NEWLY-broken retry
            # (REJECT) to the original -- that original also still carries
            # its own unresolved mismatch, so it must escalate here too,
            # not just a retry that repeats the SAME check.
            if retry_verdict.severity in (AnswerSeverity.REJECT, AnswerSeverity.RETRY_SECTION_ATTRIBUTION):
                log.warning(
                    "answer_quality_section_attribution_unresolved",
                    failed_check=retry_verdict.failed_check,
                    reason=retry_verdict.reason,
                    language=language,
                )
                return await self._no_verified_context_answer(request.question, language), retry_verdict
            return retried, retry_verdict

        log.info("answer_quality_language_retry", language=language, reason=verdict.reason)
        return await self._retry_with_reinforcement(
            f"Your previous reply was not written in {language}. Rewrite that same answer, keeping every "
            f"legal point, citation and section number exactly as it was, entirely in {language}. "
            "Statutory names, section numbers and text quoted verbatim keep their original form; "
            "everything else must be in " + language + ". Output only the rewritten answer.",
            answer=answer, verdict=verdict, messages=messages, request=request,
            language=language, sources=sources, ranked=ranked,
        )

    async def _retry_with_reinforcement(
        self, reinforcement: str, *, answer: str, verdict: QualityVerdict, messages: list[ChatMessage],
        request: ChatRequest, language: str, sources: list[SourceCitation], ranked: list[RetrievedChunk],
    ) -> tuple[str, QualityVerdict]:
        """Shared bounded-single-retry mechanic behind every `RETRY_*`
        severity: ask the model to fix ONE identified, narrow issue while
        preserving everything else, then re-run the same quality gate. A
        retry that comes back broken in some NEW way (an error leaked, the
        citations dropped out) is worse than the original's narrower
        problem, so the original is kept in that case -- never a retry
        cascading into a worse answer than not retrying at all. Unchanged
        behavior for every caller -- a caller for whom a REPEATED failure
        (not just a REJECT) means the content itself cannot be trusted
        checks the returned verdict's own severity itself; see the
        section-attribution caller above.
        """
        retry_messages = [*messages, ChatMessage(role="user", content=reinforcement)]
        retry_response = await self.llm.chat(retry_messages)
        if retry_response.error or not retry_response.content.strip():
            return answer, verdict
        retried = self._polish_grounded_answer(retry_response.content.strip(), ranked, language, request.question)
        if LEGAL_DISCLAIMER not in retried:
            retried = f"{retried}\n\n{LEGAL_DISCLAIMER}"
        retry_verdict = evaluate_answer_quality(
            retried, question=request.question, language=language, sources=sources, ranked_chunks=ranked,
            disclaimer=LEGAL_DISCLAIMER,
        )
        if retry_verdict.severity is AnswerSeverity.REJECT:
            return answer, verdict
        return retried, retry_verdict

    def _polish_grounded_answer(self, answer: str, ranked: list[Any], language: str, question: str = "") -> str:
        """Post-processing applied to every genuinely LLM-generated, grounded
        answer: replace any leaked internal source ordinal with the real
        citation, then add a statutory-currency note if a repealed provision
        was cited without its successor. Both are additive/naming-only -- they
        never change what the answer asserts.

        Post-Phase-3 hardening milestone B adds a third, equally additive pass:
        a legal time limit the answer states but the retrieved text does not is
        marked as unverified rather than presented as settled. See
        `app/rag/statutory_periods.py` for why a period, specifically, is worth
        its own check -- it is the one thing in a legal answer a reader acts on
        the same day, and acting on the wrong one can extinguish the remedy."""
        polished = annotate_answer(_replace_source_ordinals(answer, ranked), language)
        polished = annotate_jurisdiction(polished, language, question)
        polished = _correct_ni_notice_attribution(polished, question, ranked)
        return annotate_periods(polished, [getattr(chunk, "text", "") for chunk in ranked], language)

    async def handle_turn_stream(
        self, request: ChatRequest, background_tasks: BackgroundTasks | None = None,
        authenticated_user_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """The real entry point for `POST /chat/stream` -- streaming
        counterpart to `handle_turn()` above, same reasoning: a hard,
        unconditional wall-clock ceiling plus the same per-session in-flight
        guard, kept OUTSIDE `answer_stream()` itself for the same recursion-
        safety reason (its own `needs_full_pipeline_fallback` branch calls
        `self.answer(...)`, not `self.handle_turn(...)`, precisely so it
        never re-acquires this same lock).

        Also what makes a client disconnect mid-stream leave the session
        usable again: closing this generator (what FastAPI's
        `StreamingResponse` does the moment the client goes away) raises
        `GeneratorExit` at whatever `yield` is currently suspended, which
        skips straight to `finally` below -- releasing the lock and closing
        the inner generator -- rather than leaving either held until the
        lock's own TTL expires.
        """
        started = time.perf_counter()
        session_id, _ = await self._run_shared_preflight(request, authenticated_user_id)
        request = request.model_copy(update={"session_id": session_id})
        lock_token = await self._acquire_turn_lock(session_id)
        if lock_token is None:
            response = await self._turn_in_progress_response(request, session_id, started)
            yield {"event": "token", "data": response.answer}
            yield {"event": "done", "data": response.model_dump(mode="json")}
            return
        deadline_at = time.monotonic() + settings.chat_request_budget_seconds + _HARD_TIMEOUT_GRACE_SECONDS
        with deadline(settings.chat_request_budget_seconds, label="POST /chat/stream"):
            inner = self.answer_stream(request, background_tasks, authenticated_user_id)
            try:
                try:
                    while True:
                        remaining = deadline_at - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("chat stream exceeded its time budget")
                        try:
                            event = await asyncio.wait_for(inner.__anext__(), timeout=remaining)
                        except StopAsyncIteration:
                            return
                        yield event
                except TimeoutError:
                    log.warning(
                        "chat_stream_hard_timeout",
                        session_id=session_id,
                        budget_seconds=settings.chat_request_budget_seconds,
                        elapsed_seconds=round(time.perf_counter() - started, 2),
                    )
                    response = await self._timeout_response(request, session_id, started)
                    yield {"event": "token", "data": response.answer}
                    yield {"event": "done", "data": response.model_dump(mode="json")}
            finally:
                await inner.aclose()
                await self._release_turn_lock(session_id, lock_token)

    async def answer_stream(
        self, request: ChatRequest, background_tasks: BackgroundTasks | None = None,
        authenticated_user_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Part 28 Step 4: token-level streaming for `/chat/stream`.

        Yields `{"event": "token", "data": <text chunk>}` as text becomes
        available, then exactly one final `{"event": "done", "data": <full
        ChatResponse as a dict>}`.

        Only the grounded-RAG branch (a normal legal question with a good
        retrieval match -- the most common AND by far the slowest case,
        since it's the one real free-form LLM generation over several
        retrieved chunks) streams token-by-token. Every other branch
        (drafting, translation, response modification, conversation memory,
        summarization, lawyer recommendation, general conversation, a cache
        hit, cancelling a draft) reuses `answer()` completely unchanged and
        is emitted as a single "token" event: those are already fast (one
        short LLM call, or none at all), so per-token streaming wouldn't
        meaningfully change perceived latency, and delegating to `answer()`
        outright means zero risk of behavior drift between the streaming and
        non-streaming endpoints for those paths. The strict-RAG no-verified-
        context short-circuit (retrieval found nothing relevant) mirrors
        `answer()`'s branch inline below instead, since it's a plain
        constant-string assignment with no LLM call or error/retry handling
        to duplicate.

        Streamed answers are deliberately NOT written to the response cache.
        `LLMProvider.stream()` reports a failed call as a plain, friendly
        yielded string (see e.g. `GeminiProvider.stream`) rather than the
        structured `LLMResponse.error` field `.chat()` provides -- there's no
        reliable way here to tell a genuine answer from a swallowed-error
        message, so caching is skipped entirely for this path rather than
        risk caching an error as if it were a real answer. Caching still
        works normally through the unchanged, non-streaming `/chat` path.
        """
        started = time.perf_counter()
        # Classification needs the current turn present as the last message
        # (see `_has_prior_assistant_turn`), but if this turn ends up falling
        # back to `answer()` below, THAT call performs its own real, persisted
        # `memory.append` for the user's message -- appending it here too
        # would double-record the same turn. So this builds a local, never-
        # persisted preview of what memory will look like, used only to
        # decide whether to fall back; the real append happens further down,
        # only on the path that doesn't delegate to `answer()`.
        session_id, existing_memory = await self._run_shared_preflight(request, authenticated_user_id)
        preview_memory = {
            **existing_memory,
            "messages": [*existing_memory.get("messages", []), {"role": "user", "content": request.question}],
        }
        draft_was_active = bool(preview_memory.get("draft_mode"))
        language = (
            request.language
            or extract_requested_language(request.question)
            or self.language_detector.resolve(request.question, existing_memory.get("language_preference"))
        )
        conv_intent = await self.conversation_classifier.classify_advanced(
            request.question, preview_memory, self.llm, language
        )
        interrupting_draft = draft_was_active and self._is_draft_interruption(conv_intent, preview_memory, request.question)
        non_streaming_intents = {
            "Draft Generation", "General Conversation", "Lawyer Recommendation", "Translation",
            "Response Modification", "Conversation Memory", "Summarization", "Retry Failed Request",
            # Part 51: document analysis is deterministic keyword scanning
            # over the document's own chunks (`DocumentService.analyze`),
            # not a token-by-token LLM generation -- nothing to stream.
            "Document Analysis",
            # A fixed description of this product, assembled with no LLM call
            # at all -- there is nothing to stream.
            "Capability Question",
            # Fixed acknowledgement text and a fixed clarification question
            # respectively; neither involves a generation to stream.
            "Language Preference",
            "Typo Clarification",
        }
        # Part 35 "Entity Memory": a fact-recall question falls back to the
        # full (non-streaming) `answer()` pipeline too -- it already
        # performs this exact check before RAG, so there's no reason to
        # duplicate the answer-construction logic here for a case that's
        # a single deterministic lookup, not a real streamed generation.
        entity_recall_match = (
            entity_memory.find_matching_fact(request.question, preview_memory.get("entities") or [])
            if entity_memory.is_recall_query(request.question)
            else None
        )
        needs_full_pipeline_fallback = (
            (draft_was_active and not interrupting_draft)
            or (draft_was_active and _CANCEL_DRAFT_PATTERN.search(request.question))
            or conv_intent.intent in non_streaming_intents
            or entity_recall_match is not None
        )
        if needs_full_pipeline_fallback:
            # `answer()` below runs its own `_dispatch_conversation_intent`,
            # which re-classifies against the real (persisted) memory and
            # logs its own intent-history event -- logging one here too
            # (against just a local, never-persisted preview) would double-
            # record this exact turn. Only the genuinely-streamed path below
            # (which never calls `answer()`) logs its own event.
            response = await self.answer(request, background_tasks, authenticated_user_id=authenticated_user_id)
            yield {"event": "token", "data": response.answer}
            yield {"event": "done", "data": response.model_dump(mode="json")}
            return

        await self._log_intent_event(
            session_id,
            {
                "question": request.question,
                "primary_intent": conv_intent.intent,
                "detected_intents": list(conv_intent.detected_intents) or [conv_intent.intent],
                "confidence": conv_intent.confidence,
                "reason": conv_intent.reason,
                "classifier_source": conv_intent.classifier_source,
            },
            owner_user_id=authenticated_user_id,
        )
        memory = await self.memory.append(session_id, "user", request.question)
        memory = await self._claim_session_if_unowned(session_id, memory, authenticated_user_id)
        await self._store_entity_fact_if_any(session_id, memory, request.question)
        question_for_pipeline = request.question
        if conv_intent.intent == "Follow-up Question":
            question_for_pipeline = await self._resolve_followup_question(request.question, memory)
        explanation_level = request.explanation_mode or detect_explanation_level(request.question) or "citizen"
        intent = await self.intent_detector.detect(question_for_pipeline, language)

        matter, matter_response = await self._resolve_matter_context(
            request, session_id, language, memory, intent, started, background_tasks, authenticated_user_id
        )
        if matter_response is not None:
            matter_response = await self._append_conversation_reminders(
                matter_response, memory, interrupting_draft, session_id
            )
            yield {"event": "token", "data": matter_response.answer}
            yield {"event": "done", "data": matter_response.model_dump(mode="json")}
            return

        cached_entry, cache_hit_type = await self.response_cache.lookup(
            question_for_pipeline, language, intent.intent, jurisdiction_key=matter.cache_key()
        )
        if cached_entry is not None:
            response = await self._respond_from_cache(
                request, session_id, language, memory, intent, cached_entry, cache_hit_type, started, background_tasks
            )
            response = await self._append_conversation_reminders(response, memory, interrupting_draft, session_id)
            yield {"event": "token", "data": response.answer}
            yield {"event": "done", "data": response.model_dump(mode="json")}
            return

        timings: dict[str, float] = {}
        entities, ranked, sources, recommendation, jurisdiction_ambiguity = await self._prepare_rag_context(
            question_for_pipeline, language, request, intent, timings, session_id, authenticated_user_id, memory,
            matter_context=matter.as_filter(),
        )

        no_verified_context = not ranked
        general_knowledge_used = False
        ambiguous_acts = self._section_lookup_ambiguous_acts(intent.intent, ranked)
        # `jurisdiction_ambiguity` computed by `_prepare_rag_context` against
        # the raw pre-rerank pool -- see its own comment / the matching check
        # in `_answer_within_deadline`.
        failure_reason: str | None = None
        if jurisdiction_ambiguity:
            answer_text = clarification_question(language, ambiguity=jurisdiction_ambiguity["ambiguity"])
            confidence = 0.5
            confidence_reason = (
                f"Retrieved sources are tied to specific {jurisdiction_ambiguity['ambiguity']}(s) "
                f"({', '.join(jurisdiction_ambiguity['candidates'])}) with no confirmed match to this matter "
                "and no all-India provision to fall back on; asked which one applies."
            )
            prompt_tokens = completion_tokens = None
            pending = "matter_jurisdiction_locality" if jurisdiction_ambiguity["ambiguity"] == "locality" else "matter_jurisdiction"
            await self.memory.update(session_id, pending_clarification=pending)
            yield {"event": "token", "data": answer_text}
        elif ambiguous_acts:
            answer_text = self._render_act_disambiguation_question(ambiguous_acts, language)
            confidence = 0.5
            confidence_reason = "Multiple Acts define this section number; asked the user which one they mean."
            prompt_tokens = completion_tokens = None
            yield {"event": "token", "data": answer_text}
        elif no_verified_context:
            gk_answer = (
                await general_knowledge_answer(self.llm, request.question, language)
                if settings.general_knowledge_fallback_enabled else None
            )
            if gk_answer is not None:
                answer_text = gk_answer
                confidence = _GENERAL_KNOWLEDGE_CONFIDENCE
                confidence_reason = (
                    "No chunk in the knowledge base cleared the relevance threshold for this question; "
                    "answered from general legal knowledge instead, labeled and unverified."
                )
                general_knowledge_used = True
                if settings.kb_gap_autofetch_enabled:
                    asyncio.create_task(GapAutoFetchService().record_gap(request.question))
            else:
                answer_text = await self._no_verified_context_answer(request.question, language)
                confidence = 0.0
                confidence_reason = "No chunk in the knowledge base cleared the relevance threshold for this question."
            prompt_tokens = completion_tokens = None
            yield {"event": "token", "data": answer_text}
        else:
            messages = self._build_rag_messages(
                question_for_pipeline, language, intent, conv_intent, entities, ranked, explanation_level
            )
            chunks: list[str] = []
            stream_failed = False
            async for piece in self.llm.stream(messages):
                # Only the very first chunk is checked -- if generation is
                # already underway, later text legitimately containing one of
                # these substrings shouldn't retroactively be treated as an
                # error (and can't be un-sent to the client anyway).
                if not chunks and self._looks_like_llm_error(piece):
                    stream_failed = True
                    log.warning("streaming_llm_call_failed", detected_text=piece)
                    break
                chunks.append(piece)
                yield {"event": "token", "data": piece}
            if stream_failed or not chunks:
                answer_text = self._fallback_answer(intent.intent, ranked, question_for_pipeline)
                yield {"event": "token", "data": answer_text}
                failure_reason = "streaming LLM call failed"
            else:
                answer_text = "".join(chunks).strip()
            if LEGAL_DISCLAIMER not in answer_text:
                answer_text = f"{answer_text}\n\n{LEGAL_DISCLAIMER}"
            # See the matching check in `answer()`: the LLM emits this exact
            # line itself when it judges the retrieved chunks don't actually
            # answer the question, which must read as a real refusal (0
            # confidence), not a normal grounded answer.
            if is_no_verified_context(answer_text):
                confidence = 0.0
                confidence_reason = "The retrieved sources do not sufficiently support this answer."
            else:
                # Part 58 issues 2/24: the same naming/currency polish
                # `answer()` applies. The tokens already streamed are the raw
                # text, so the polished version only reaches the client in the
                # final "done" payload -- the client renders that as the
                # settled answer, and a mid-stream rewrite isn't possible
                # once tokens are on the wire.
                answer_text = self._polish_grounded_answer(answer_text, ranked, language, request.question)
                confidence = self._confidence(ranked, intent.confidence, entities.confidence)
                confidence_reason = self._confidence_reason(ranked, intent.intent)
                # Phase 1 item 3: the quality gate runs here too, but only as
                # far as it usefully can. Tokens are already on the wire, so a
                # rejected answer cannot be un-sent -- what it CAN do is stop
                # the rejected text from being recorded as the settled answer:
                # the "done" payload (which clients render as the final
                # answer), conversation memory, chat history and the analytics
                # log all read `answer_text` from here. `RETRY_LANGUAGE` is
                # deliberately not acted on in this path: a second generation
                # after streaming has finished would replace text the user has
                # already watched appear, which is more confusing than the
                # language mismatch itself. See the module docstring's note on
                # streaming for the resulting gap.
                stream_verdict = evaluate_answer_quality(
                    answer_text, question=request.question, language=language, sources=sources,
                    ranked_chunks=ranked, disclaimer=LEGAL_DISCLAIMER,
                )
                if stream_verdict.severity is AnswerSeverity.REJECT:
                    log.warning(
                        "answer_quality_rejected_after_streaming",
                        failed_check=stream_verdict.failed_check,
                        reason=stream_verdict.reason,
                        language=language,
                    )
                    answer_text = await self._no_verified_context_answer(request.question, language)
                    confidence = 0.0
                    confidence_reason = stream_verdict.reason or confidence_reason
                else:
                    # Objective item 5: same broadened historical-answer
                    # check as the non-streaming path -- see its comment.
                    historical_answer = bool(matter.as_of_date)
                    version_ambiguous = bool(historical_answer and detect_version_ambiguity(ranked))
                    if version_ambiguous:
                        confidence = min(confidence, _HISTORICAL_VERSION_AMBIGUITY_CONFIDENCE_CAP)
                        confidence_reason = (
                            "More than one version of a retrieved provision could apply on the given date; "
                            "which one governs was not conclusively determined."
                        )
                    elif historical_answer:
                        confidence = min(confidence, _HISTORICAL_TRANSITION_UNVERIFIED_CONFIDENCE_CAP)
                        confidence_reason = (
                            "Answered using the version on record for the date given; whether a transition/savings "
                            "clause changes this for the exact facts was not independently verified."
                        )
                    # Jurisdiction Routing (Phase 2), objective item 7: same
                    # disclosure as the non-streaming path
                    # (`_answer_within_deadline`) -- not streamed token-by-
                    # token (matching how `LEGAL_DISCLAIMER` above is also
                    # only appended post-stream), but present in the settled
                    # `answer_text` the "done" payload/cache/history all read.
                    note = disclosure_note(language, matter, version_ambiguous=version_ambiguous)
                    if note:
                        answer_text = f"{answer_text}\n\n{note}"
                    machine_note = _machine_verification_disclosure(language, sources)
                    if machine_note:
                        answer_text = f"{answer_text}\n\n{machine_note}"
            prompt_tokens = completion_tokens = None

        # Streaming's tail used to independently re-implement everything
        # `_finalize_rag_response` already does for `answer()` -- refusal
        # sanitization, memory/history persistence, response construction,
        # reminders, and logging -- and had drifted from it in three ways:
        # it never persisted `pending_section_query`/`pending_section_acts`
        # (so a streamed "Section 302" -> "which Act?" -> "IPC" follow-up was
        # silently treated as a brand-new vague query instead of resolving),
        # it left `detected_intents`/`intent_reason`/`retryable`/
        # `failure_category`/`warnings` at schema defaults instead of
        # computing them, and it never logged `chat_pipeline_timing`. Calling
        # the one shared implementation fixes all three by construction --
        # `_finalize_rag_response` was already general enough to take this
        # call with no signature changes (`related_questions_task=None` hits
        # its existing `if ... is not None` guard exactly like a genuinely
        # skipped background task would).
        suggested_actions = CONVERSATION_INTENT_ACTIONS.get(conv_intent.intent, [])
        response = await self._finalize_rag_response(
            request, session_id, language, memory, intent, conv_intent, entities, sources, ranked, recommendation,
            explanation_level, suggested_actions, interrupting_draft, no_verified_context,
            answer_text, confidence, confidence_reason, prompt_tokens, completion_tokens, failure_reason,
            question_for_pipeline, [], None, timings, started, background_tasks,
            ambiguous_acts=ambiguous_acts, general_knowledge_used=general_knowledge_used,
        )
        yield {"event": "done", "data": response.model_dump(mode="json")}

    def _is_draft_interruption(self, conv_intent: ConversationIntentMatch, memory: dict[str, Any], question: str) -> bool:
        template = get_template(memory.get("draft_template_id"))
        if template is not None and memory.get("draft_stage") in {"preview", "approved", "locked", "exported"}:
            draft_language = str(memory.get("draft_language") or "english")
            if self.draft_conversation.edit_interpreter.interpret(
                question, template, draft_language
            ).action != "unknown":
                # The draft engine, not the general conversation/RAG router,
                # owns a recognized command against the current draft.
                return False
        if conv_intent.intent == "Translation" and memory.get("draft_stage") == "preview":
            # "translate" already means "translate the draft" in preview stage,
            # handled by the drafting engine's own edit-command interpreter.
            return False
        if (
            conv_intent.intent == "Translation"
            and memory.get("draft_stage") == "collecting"
            and extract_requested_language(question)
        ):
            # Post-Phase-3 hardening (Phase 2, milestone D). Mid-collection,
            # "hindi me" does not mean "translate your last message". It means
            # "ask me the rest of the questions, and write the document, in
            # Hindi" -- which `DraftConversationEngine._process_collecting_
            # message` already implements, by setting `draft_language` and
            # `draft_language_preference` from the same `extract_requested_
            # language`.
            #
            # Confirmed in a real session: the user typed "hindi me" while a
            # rent-notice draft was collecting. It was classified Translation,
            # treated as an interruption, and answered by translating the
            # previous reply -- so the field list APPEARED in Hindi and the
            # user reasonably assumed the draft had switched. It had not:
            # `draft_language` was never set, so the next turn's questions came
            # back in English and the finished notice was built with English
            # scaffolding ("Sir/Madam,", "Introduction") wrapped around Hindi
            # values. The translation was cosmetic and lasted one turn.
            #
            # Narrow on purpose: only when the message actually NAMES a
            # language. "translate this to simple words" mid-draft is still an
            # interruption, exactly as before.
            return False
        if conv_intent.intent in _DRAFT_INTERRUPT_INTENTS:
            if conv_intent.intent == "Legal Dictionary" and conv_intent.confidence < 0.7:
                # The 0.5-confidence bare-term-lookup path is too weak a signal
                # -- a short field value (e.g. a name) can match it just as
                # easily as a genuine "define X" question.
                pass
            else:
                return True
        # Post-Phase-3 hardening (Phase 2, milestone D), found in a LIVE run.
        #
        # Before the heuristic below gets a vote, ask the component that
        # actually knows: does this message carry labelled answers for fields
        # of the template being collected right now? If it does, it is a field
        # answer, whatever it looks like.
        #
        # The heuristic's own comment claimed "a genuine field value (a name,
        # address, amount, date) never looks like a question, so this doesn't
        # cost any recall on real field answers". That holds for ONE short
        # value and fails for the way users actually answer a printed list of
        # ten fields: one 1,143-character block of labelled Hindi values whose
        # `facts` and `expected_relief` entries are full sentences.
        # `looks_informational` returns True for that block, so the whole
        # answer was routed to retrieval and every value in it was discarded --
        # observed live, against the running backend, on the third turn of a
        # Hindi rent-notice conversation.
        #
        # Two or more resolved fields, not one: a single match could be
        # coincidence in a genuine question ("what should the applicant address
        # be?"), while two labelled values in one message is a form being
        # filled in.
        template = get_template(memory.get("draft_template_id"))
        if template is not None and memory.get("draft_stage") == "collecting":
            labelled = self.draft_conversation.extractor._extract_by_label_prefix(
                question, template, str(memory.get("draft_language") or "english")
            )
            if len(labelled) >= 2:
                return False

        # `ConversationIntentClassifier` sorts a message into ~12 discourse
        # buckets, but plenty of genuine unrelated questions don't match any
        # of its specific patterns and fall into the low-confidence "General
        # Legal Information" catch-all (deliberately excluded from
        # `_DRAFT_INTERRUPT_INTENTS` above, since that bucket is too weak a
        # signal on its own -- a bare field value falls into it too). Falling
        # through to the fallback bucket isn't itself evidence either way, so
        # ask the same "does this look like a question" check the drafting
        # detector uses to decide whether to *start* a draft.
        return looks_informational(question)

    async def _append_draft_reminder(
        self, response: ChatResponse, memory: dict[str, Any], interrupting_draft: bool, session_id: str
    ) -> ChatResponse:
        """Part 43 "Stale Draft Protection": shown at most ONCE per pause, not
        on every subsequent unrelated turn -- a user asking three unrelated
        questions in a row while a draft sits paused should see the reminder
        after the first one, not have it tacked onto every single answer
        after that (which reads as a nagging, mechanically-repeated aside
        rather than a helpful one-time nudge). `draft_reminder_shown` is
        cleared again in `_respond_with_draft_turn` the moment the draft
        engine actually handles a turn directly (a fresh start OR a genuine
        continuation), so the next pause shows the reminder again.
        """
        if not interrupting_draft:
            return response
        # Part 58 issues 19/21: an interruption now PAUSES the draft, not just
        # annotates the reply. Without this, only the first unrelated message
        # escaped the draft -- `_is_draft_interruption` has to re-earn its
        # verdict on every single turn, and the moment one message failed that
        # test (a short follow-up, a bare "no", a question phrased in a way the
        # patterns don't cover) the user was pulled straight back into field
        # collection with no way out. Pausing makes the escape stick until
        # they explicitly say "continue draft". Done before the
        # already-shown check below, so the pause is applied on every
        # interrupting turn even when the reminder text itself is only shown
        # once.
        self.draft_conversation.pause(memory)
        # `pause` declines to pause a draft still in its "selecting" stage
        # (see its docstring), so the persisted flag must follow what it
        # actually did rather than assume it paused.
        paused = bool(memory.get("draft_paused"))
        if memory.get("draft_reminder_shown"):
            await self.memory.update(session_id, draft_paused=paused)
            return response
        reminder = self.draft_conversation.describe_pending(
            memory, fallback_language=str(memory.get("language_preference") or "") or None
        )
        if reminder:
            response.answer = f"{response.answer}\n\n---\n{reminder}"
        memory["draft_reminder_shown"] = True
        await self.memory.update(session_id, draft_reminder_shown=True, draft_paused=paused)
        return response

    async def _append_retry_reminder(self, response: ChatResponse, memory: dict[str, Any], session_id: str) -> ChatResponse:
        """Part 31 Rule 4, narrowed by Part 49: if an earlier turn's request
        failed and hasn't been retried yet, remind the user they can retry
        it -- but only ONCE, on the very next turn, not on every subsequent
        turn until they retry. `last_failed_question`/`last_failed_reason`
        themselves are left untouched (nothing but an explicit retry clears
        those, so "retry" still works any time later), only whether the
        reminder TEXT has already been shown once.

        Confirmed production regression: with the reminder repeated on every
        turn, a single stale failure kept surfacing "by the way, your
        earlier question... ran into a temporary issue" after 10+ later,
        completely unrelated answers (translations, follow-ups, a fresh
        topic) -- read as a nagging, mechanically-repeated aside contaminating
        every reply rather than a helpful one-time nudge. Mirrors
        `_append_draft_reminder`'s existing `draft_reminder_shown` pattern.

        Checked against `memory` as loaded at the START of this turn, so a
        failure recorded DURING this same turn is never double-mentioned in
        its own response -- that failure's own fallback/recovery answer
        already covers it; the reminder is for turns *after* that one.
        """
        last_failed_question = memory.get("last_failed_question")
        if not last_failed_question or memory.get("retry_reminder_shown"):
            return response
        reminder = (
            f'By the way, your earlier question -- "{last_failed_question}" -- ran into a temporary issue and '
            'didn\'t get a proper answer. Say "retry" anytime to try it again.'
        )
        response.answer = f"{response.answer}\n\n---\n{reminder}"
        memory["retry_reminder_shown"] = True
        await self.memory.update(session_id, retry_reminder_shown=True)
        return response

    async def _append_conversation_reminders(
        self, response: ChatResponse, memory: dict[str, Any], interrupting_draft: bool, session_id: str
    ) -> ChatResponse:
        response = await self._append_draft_reminder(response, memory, interrupting_draft, session_id)
        response = await self._append_retry_reminder(response, memory, session_id)
        return response

    async def _respond_with_draft_cancelled(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        started: float,
        background_tasks: BackgroundTasks | None = None,
    ) -> ChatResponse:
        # Read BEFORE the reset clears it: the cancellation confirmation has
        # to be written in the language the drafting conversation was actually
        # being held in, not whichever language this one-word "cancel kr do"
        # happens to be detected as.
        draft_language = memory.get("draft_language") or language
        self.draft_conversation.reset(memory)
        # `ConversationMemoryStore.update` re-loads the persisted copy rather than
        # writing the caller's local `memory` dict, so the reset draft fields must
        # be passed explicitly here -- otherwise `_finalize_intent_response`'s own
        # `update(language_preference=...)` call would merge onto a still-active
        # persisted draft_mode and silently leave the draft "cancelled" locally
        # but still in progress in storage.
        await self.memory.update(
            session_id,
            draft_mode=False,
            draft_stage=None,
            draft_template_id=None,
            draft_fields={},
            draft_id=None,
            # Part 58 issue 18: these four were left behind by the old update,
            # so right after "I've discarded that draft" the session still had
            # a parked draft, a stale draft language, and a pause flag -- and
            # the next message was pulled straight back into the flow the user
            # had just cancelled. The system even said so itself ("your Bank
            # Fraud Complaint draft is still saved").
            parked_drafts=[],
            draft_paused=False,
            draft_language=None,
            draft_reminder_shown=False,
        )
        answer = msg(
            "draft_cancelled",
            draft_language,
            "Okay, I've discarded that draft. Let me know if you'd like to start a new one or if I can "
            "help with anything else.",
        )
        recommendation = await self._contextual_recommendation(memory, "General Conversation")
        return await self._finalize_intent_response(
            request, session_id, draft_language, memory, answer, "General Conversation", started, background_tasks,
            recommendation, [], "Draft discarded at your request.",
        )

    async def _respond_with_draft_paused(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        started: float,
        background_tasks: BackgroundTasks | None = None,
    ) -> ChatResponse:
        """Part 58 issue 20/21: the reply to a bare refusal mid-draft.

        Keeps every collected field, but releases the draft engine's claim on
        the next message (see `DraftConversationEngine.handle_turn`'s paused
        branch), and says out loud what BOTH exits are -- resume or cancel.
        The previous behaviour re-printed the pending-field list, which read
        as the assistant ignoring an explicit "no".
        """
        draft_language = memory.get("draft_language") or language
        template = get_template(memory.get("draft_template_id")) if memory.get("draft_template_id") else None
        template_name = template.name if template else "document"
        self.draft_conversation.pause(memory)
        await self.memory.update(session_id, draft_paused=True, draft_reminder_shown=True)
        answer = msg(
            "draft_paused",
            draft_language,
            'Okay, I\'ve paused the {template} draft -- your details are saved. Say "continue draft" to pick '
            'it back up, or "cancel draft" to discard it. In the meantime, ask me anything.',
            template=template_name,
        )
        recommendation = await self._contextual_recommendation(memory, "General Conversation")
        return await self._finalize_intent_response(
            request, session_id, draft_language, memory, answer, "General Conversation", started, background_tasks,
            recommendation, [], "Explicit refusal mid-draft: the draft was paused, not discarded.",
        )

    async def _respond_with_draft_turn(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        draft_result: DraftTurnResult,
        started: float,
        background_tasks: BackgroundTasks | None = None,
    ) -> ChatResponse:
        """Builds the `ChatResponse` for a message the drafting flow handled.

        `handle_turn` mutated `memory` in place (draft_mode/draft_stage/
        draft_fields/draft_id); persist it the same way `language_preference`
        etc. are persisted for the normal RAG path, and skip retrieval/LLM
        chat entirely -- the draft engine already produced the reply.
        """
        memory["pending_clarification"] = None
        # The draft engine handled this turn directly -- a fresh start or a
        # genuine continuation, either way the user is actively engaged with
        # it again, so the next time it gets auto-paused for an unrelated
        # question, the reminder should be shown (once) again too.
        memory["draft_reminder_shown"] = False
        await self.memory.update(session_id, **memory)
        await self.memory.append(session_id, "assistant", draft_result.reply_text)
        if background_tasks is not None:
            background_tasks.add_task(self.memory.summarize_if_needed, session_id, self.llm)
        else:
            await self.memory.summarize_if_needed(session_id, self.llm)
        recommendation = await self.recommendations.recommend("Document Drafting", "General Law")
        message_id = str(uuid4())
        await self.history.insert(
            {
                "message_id": message_id,
                "session_id": session_id,
                "conversation_id": request.conversation_id,
                # Security finding C2: this must be the SERVER-VERIFIED owner
                # (`memory["owner_user_id"]`, set by `check_access`/
                # `_claim_session_if_unowned` from the caller's JWT), never
                # the client-supplied `request.user_id` -- stamping the
                # latter here made `erase_user_data`'s
                # `ChatRepository().delete_by_user(user_id)` match nothing
                # for a real authenticated user (their messages were never
                # actually tagged with their real id), so `DELETE /me/data`
                # reported success while leaving every chat message behind.
                "user_id": memory.get("owner_user_id"),
                "question": request.question,
                "answer": draft_result.reply_text,
                "intent": {
                    "intent": "Document Drafting",
                    "legal_category": "General Law",
                    "reason": f"Drafting flow ({draft_result.info.stage}).",
                    "confidence": 1.0,
                },
                "entities": {},
                "sources": [],
            }
        )
        response = ChatResponse(
            message_id=message_id,
            session_id=session_id,
            answer=draft_result.reply_text,
            sources=[],
            confidence=1.0,
            confidence_label=self._confidence_label(1.0),
            confidence_reason="Structured drafting flow, not a retrieval-based answer.",
            lawyer_recommendation=recommendation,
            detected_language=language,
            detected_intent=(
                f"Document Drafting: {draft_result.info.template_name}"
                if draft_result.info.template_name
                else "Document Drafting"
            ),
            conversation_intent="Draft Generation",
            llm_provider=self.llm.provider_name,
            llm_model=getattr(self.llm, "model", ""),
            latency_ms=(time.perf_counter() - started) * 1000,
            draft=draft_result.info,
            retryable=bool(draft_result.info.generation_error),
            failure_category=self._failure_category(draft_result.info.generation_error),
            saved_progress=True,
        )
        self._log_routing_decision(
            session_id, message_id, request.question, "Draft Generation", "draft_workflow",
            memory_hit=True, rag_used=False, response_modification=False, draft_mode=True,
        )
        await self._log_query(
            request, session_id, message_id, response, background_tasks,
            owner_user_id=memory.get("owner_user_id"),
        )
        return response

    async def _respond_with_general_conversation(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
    ) -> ChatResponse:
        prompt = prompt_registry.render("general_conversation_prompt", language=language, message=request.question)
        llm_response = await self.llm.chat([ChatMessage(role="user", content=prompt)])
        answer = self._safe_llm_text(llm_response, "Happy to help — what would you like to ask?")
        recommendation = await self._contextual_recommendation(memory, "General Conversation")
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "General Conversation", started, background_tasks,
            recommendation, suggested_actions or [], "Conversational reply, not a legal determination.",
        )

    async def _respond_with_capability_overview(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
    ) -> ChatResponse:
        """"What can you help me with?" -- answered from a fixed description of
        what this app actually does, never from retrieval.

        Confirmed live failure this replaces: the second message of a real
        session, "mujhe batao ki tum meri kin legal problems me madad kar
        sakte ho?", was routed to RAG and answered with the strict-RAG
        refusal. No statute describes this product, so retrieval could not
        have succeeded -- and an LLM asked to describe its own capabilities
        invents them, which is worse than a refusal for a legal tool. The
        text below is fixed and enumerates only capabilities that exist:
        `_dispatch_conversation_intent`'s own branches (drafting, document
        analysis, translation, lawyer recommendation) plus the RAG pipeline.
        The template count is read from the live registry rather than
        hardcoded, so it cannot drift as templates are added.
        """
        template_count = len(list_templates())
        answer = capability_overview(language, template_count)
        recommendation = await self._contextual_recommendation(memory, "Capability Question")
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Capability Question", started, background_tasks,
            recommendation, suggested_actions or [],
            "Described this assistant's own capabilities; no retrieval performed.",
        )

    async def _respond_with_language_preference(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        conv_intent: ConversationIntentMatch,
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
    ) -> ChatResponse:
        """"Mujhe simple Hindi mein jawab diya karo" -- store the preference,
        acknowledge it in that language, and stop.

        No retrieval, no reranking, no answer-generation call: a request to
        change the reply language has no answer in any statute, and both
        reported phrasings of it were previously routed to RAG ("Legal
        Advice" and "General Legal Information" respectively) and answered
        with the strict-RAG refusal.

        The preference is persisted by passing it to
        `_finalize_intent_response` AS the turn's language, so the stored
        `language_preference` is the language the user asked for -- not the
        language this particular sentence happened to be detected as, which
        for "Mujhe simple Hindi mein jawab diya karo" (Latin script, Hinglish
        tokens) is "hinglish", the opposite of what was requested.
        """
        preferred = conv_intent.resolved_language_preference or language
        answer = language_preference_acknowledgement(preferred, bool(memory.get("draft_mode")))
        recommendation = await self._contextual_recommendation(memory, "Language Preference")
        return await self._finalize_intent_response(
            request, session_id, preferred, memory, answer, "Language Preference", started, background_tasks,
            recommendation, suggested_actions or [],
            "Recorded the requested reply language; no retrieval was needed.",
        )

    async def _respond_with_typo_clarification(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        conv_intent: ConversationIntentMatch,
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
    ) -> ChatResponse:
        """One unreadable control word with more than one equally-good
        correction: ask which was meant instead of guessing.

        The alternatives offered are exactly the tied candidates the
        normalizer found -- never a preferred guess dressed up as a question.
        """
        answer = conv_intent.clarification_message or "Sorry, could you rephrase that?"
        recommendation = await self._contextual_recommendation(memory, "Typo Clarification")
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Typo Clarification", started, background_tasks,
            recommendation, suggested_actions or [],
            "Asked which word was meant; no retrieval was performed on an unreadable message.",
        )

    async def _respond_with_out_of_domain(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
    ) -> ChatResponse:
        """Part 42 "Answer Quality & Intent Accuracy": a message clearly
        outside the legal domain (weather, coding, storytelling, ...) never
        runs RAG/general-knowledge and never gets `LEGAL_DISCLAIMER` -- it
        isn't legal advice, legal interpretation, or anything disclaimer-
        worthy, just a scope note. Mirrors `_respond_with_general_conversation`'s
        pattern exactly.
        """
        prompt = prompt_registry.render("out_of_domain_prompt", language=language, message=request.question)
        llm_response = await self.llm.chat([ChatMessage(role="user", content=prompt)])
        answer = self._safe_llm_text(
            llm_response,
            "I'm focused on legal questions, so I can't help with that here — happy to help with anything legal though.",
        )
        recommendation = await self._contextual_recommendation(memory, "Out of Domain")
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Out of Domain", started, background_tasks,
            recommendation, suggested_actions or [],
            "Message was outside the legal assistant's domain; no retrieval performed.",
        )

    async def _respond_with_jurisdiction_scope(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
    ) -> ChatResponse:
        """Part 42: a legal question naming a non-Indian jurisdiction never
        gets answered with Indian law -- and never runs RAG against this
        India-only knowledge base, since the retrieved context would be
        genuinely wrong for the jurisdiction asked about.
        """
        prompt = prompt_registry.render("jurisdiction_scope_prompt", language=language, message=request.question)
        llm_response = await self.llm.chat([ChatMessage(role="user", content=prompt)])
        answer = self._safe_llm_text(
            llm_response,
            "I'm focused on Indian law, so I can't reliably give you the process for another country's legal "
            "system from my available legal sources.",
        )
        recommendation = await self._contextual_recommendation(memory, "Non-Indian Jurisdiction")
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Non-Indian Jurisdiction", started, background_tasks,
            recommendation, suggested_actions or [],
            "Question was about a non-Indian jurisdiction; this assistant only covers Indian law, no retrieval performed.",
        )

    async def _respond_with_lawyer_recommendation(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
    ) -> ChatResponse:
        recommendation = await self._contextual_recommendation(memory, "Lawyer Recommendation", language)
        answer = f"{render_recommendation(recommendation, language)}\n\n{LEGAL_DISCLAIMER}"
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Lawyer Recommendation", started, background_tasks,
            recommendation, suggested_actions or [], "Deterministic recommendation based on the legal category discussed so far.",
        )

    async def _respond_with_translation(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        conv_intent: ConversationIntentMatch,
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
    ) -> ChatResponse:
        recommendation = await self._contextual_recommendation(memory, "Translation")
        if conv_intent.ambiguous:
            no_target = conv_intent.resolved_translation_target is None
            answer = _TRANSLATION_NO_TARGET_MESSAGE if no_target else _TRANSLATION_NO_PRIOR_REPLY_MESSAGE
            # Only the "which language" case is a real pending clarification --
            # remembered so a bare one-word reply next turn (e.g. "Hindi.")
            # completes the translation without repeating "translate" (see
            # `ConversationIntentClassifier`'s pending_clarification check).
            return await self._finalize_intent_response(
                request, session_id, language, memory, answer, "Translation", started, background_tasks,
                recommendation, suggested_actions or [], "Clarification needed before translating.", confidence=0.3,
                pending_clarification="translation_target" if no_target else None,
            )
        last_reply = next(
            (
                message["content"]
                for message in reversed(memory.get("messages", [])[:-1])
                if message.get("role") == "assistant" and message.get("content") not in _TRANSLATION_CLARIFICATION_MESSAGES
            ),
            "",
        )
        prompt = prompt_registry.render(
            "translation_prompt",
            # An explicit "into Hindi" request makes this TURN's response
            # language Hindi, but the text being translated is still in the
            # conversation's previous language.  Using `language` for both
            # sides produces a nonsensical Hindi -> Hindi instruction.
            source_language=memory.get("language_preference") or language,
            target_language=conv_intent.resolved_translation_target,
            text=last_reply,
        )
        llm_response = await self.llm.chat([ChatMessage(role="user", content=prompt)])
        answer = self._safe_llm_text(llm_response, last_reply)
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Translation", started, background_tasks,
            recommendation, suggested_actions or [], "Deterministic translation of the prior reply.",
        )

    async def _respond_with_response_modification(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        conv_intent: ConversationIntentMatch,
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
    ) -> ChatResponse:
        """Part 24 "Response Modification Engine": transforms the last assistant
        reply (shorten/expand/simplify/bullets/rewrite/table/examples/timeline/
        visual/last-response-summary) without ever re-running retrieval.

        Chaining ("30 words" after "translate into Hindi" acts on the Hindi
        version, not the English original) falls out for free: `memory["messages"]`
        already holds whatever this method itself appended as the prior turn's
        answer, same as every other conversation-intent branch, so pulling the
        latest assistant entry always yields the CURRENT version.
        """
        recommendation = await self._contextual_recommendation(memory, "Response Modification")
        if conv_intent.ambiguous:
            answer = "I don't have a previous response to modify. Please ask a question first."
            return await self._finalize_intent_response(
                request, session_id, language, memory, answer, "Response Modification", started, background_tasks,
                recommendation, suggested_actions or [], "No previous response exists to transform.", confidence=0.3,
            )
        last_reply = next(
            (message["content"] for message in reversed(memory.get("messages", [])[:-1]) if message.get("role") == "assistant"),
            "",
        )
        language_hint = (
            f"Also render the result in {conv_intent.resolved_translation_target}."
            if conv_intent.resolved_translation_target
            else ""
        )
        prompt = prompt_registry.render(
            "response_modification_prompt",
            transform_label=_MODIFICATION_TRANSFORM_LABELS.get(
                conv_intent.modification_type or "",
                "Transform the text below as the instruction requests.",
            ),
            instruction=request.question,
            language_hint=language_hint,
            text=last_reply,
        )
        llm_response = await self.llm.chat([ChatMessage(role="user", content=prompt)], temperature=0.2)
        answer = self._safe_llm_text(llm_response, last_reply)
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Response Modification", started, background_tasks,
            recommendation, suggested_actions or [],
            f"Transformed the prior reply ({conv_intent.modification_type}); no retrieval performed.",
        )

    async def _respond_with_conversation_memory(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        conv_intent: ConversationIntentMatch,
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
    ) -> ChatResponse:
        """Part 26 "Conversation Memory Engine": answers questions ABOUT the
        conversation itself ("What did I ask?", "What were we discussing?",
        "What was stolen?", "Continue.") purely from `memory["summary"]`/
        `memory["messages"]` -- never retrieval, since there's nothing to look
        up in the knowledge base for a question about what was already said.
        """
        recommendation = await self._contextual_recommendation(memory, "Conversation Memory")
        if conv_intent.ambiguous:
            answer = "We haven't talked about anything yet in this conversation -- go ahead and ask me something to get started."
            return await self._finalize_intent_response(
                request, session_id, language, memory, answer, "Conversation Memory", started, background_tasks,
                recommendation, suggested_actions or [], "Nothing has been discussed yet in this conversation.", confidence=0.3,
            )
        transcript = "\n".join(
            f"{message['role']}: {message['content']}" for message in memory.get("messages", [])[:-1]
        )
        prompt = prompt_registry.render(
            "conversation_memory_prompt",
            summary=memory.get("summary") or "None",
            conversation=transcript or "None",
            question=request.question,
        )
        llm_response = await self.llm.chat([ChatMessage(role="user", content=prompt)], temperature=0.0)
        answer = self._safe_llm_text(llm_response, "I don't have that in our conversation so far.")
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Conversation Memory", started, background_tasks,
            recommendation, suggested_actions or [], "Answered from conversation memory; no retrieval performed.",
        )

    async def _respond_with_conversation_summary(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
    ) -> ChatResponse:
        recommendation = await self._contextual_recommendation(memory, "Summarization")
        transcript = "\n".join(
            f"{message['role']}: {message['content']}" for message in memory.get("messages", [])[:-1]
        )
        prompt = prompt_registry.render(
            "summarization_prompt", existing_summary=memory.get("summary") or "None", conversation=transcript or "None"
        )
        llm_response = await self.llm.chat([ChatMessage(role="user", content=prompt)], temperature=0.0)
        summary = self._safe_llm_text(llm_response, memory.get("summary") or "We haven't discussed much yet.")
        answer = f"Here's a summary of our conversation so far:\n\n{summary}"
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Summarization", started, background_tasks,
            recommendation, suggested_actions or [], "Summary generated from the conversation so far.",
        )

    async def _maybe_answer_from_inline_document(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        started: float,
        background_tasks: BackgroundTasks | None = None,
    ) -> ChatResponse | None:
        """A document pasted INTO the message ("Read only this agreement
        excerpt: "..."") is conversation context, not an upload -- so nothing
        remembered it. A follow-up in the same session ("does Clause 7 permit
        painting charges?", "what page is this excerpt from?") fell through to
        the Knowledge-Base pipeline and came back "no verified document",
        while the answer was sitting in the conversation.

        Captures such an excerpt when one arrives (returning `None` so the
        turn is answered normally) and answers later turns that clearly refer
        back to it strictly from its text.
        """
        question = request.question or ""
        quote = _INLINE_QUOTE_PATTERN.search(question)
        if quote and _INLINE_DOC_CUE_PATTERN.search(question[: quote.start()]):
            memory["inline_document"] = {"text": quote.group(1).strip()[:8000]}
            await self.memory.update(session_id, inline_document=memory["inline_document"])
            return None
        document = memory.get("inline_document")
        if not document or not _INLINE_FOLLOWUP_PATTERN.search(question):
            return None
        prompt = prompt_registry.render(
            "document_question_prompt",
            question=question, language=language, document_text=document["text"],
        )
        llm_response = await self.llm.chat([ChatMessage(role="user", content=prompt)])
        if getattr(llm_response, "error", None):
            return None
        parsed = _parse_json_object(llm_response.content)
        if parsed is None:
            return None
        recommendation = await self._contextual_recommendation(memory, "Document Analysis")
        if parsed.get("found_in_document") and str(parsed.get("answer") or "").strip():
            answer = str(parsed["answer"]).strip()
            quote_text = str(parsed.get("supporting_quote") or "").strip()
            if quote_text:
                answer += f'\n\n> "{quote_text}"'
        else:
            answer = _INLINE_DOC_NOT_FOUND_MESSAGES.get(language, _INLINE_DOC_NOT_FOUND_MESSAGES["english"])
        answer = f"{answer}\n\n{LEGAL_DISCLAIMER}"
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Document Analysis", started, background_tasks,
            recommendation, [], "Answered strictly from the excerpt pasted earlier in this conversation.",
            confidence=0.7 if parsed.get("found_in_document") else 0.3,
        )

    async def _respond_with_document_analysis(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        started: float,
        background_tasks: BackgroundTasks | None = None,
        suggested_actions: list[str] | None = None,
        authenticated_user_id: str | None = None,
    ) -> ChatResponse:
        """Part 51 "Uploaded Document Conversation Context": routes to the
        EXISTING `/document-analysis` logic (`DocumentService.analyze`)
        instead of falling through to generic RAG -- reused unchanged, not
        duplicated.

        Document resolution order: (1) an explicit `document_id` passed via
        `request.metadata_filters` (the same field the RAG ownership filter
        already merges in -- there's no dedicated `document_id` field on
        `ChatRequest`, and inventing one isn't needed when this already
        works); (2) `memory["last_uploaded_document_id"]`, set by
        `DocumentService.upload_and_index` on a successful upload -- covers
        every pronoun/short-reference phrasing ("explain this," "isme kya
        hai," "isko explain kro," "ye pdf samjhao," "key points batao")
        identically, since none of them name a different document from each
        other; (3) neither present -> a short clarification, never a guess,
        and never any KB document sent to the LLM in its place.

        `DocumentService.analyze` re-verifies ownership against the
        document's own stored metadata unchanged (`_ensure_document_
        access`) -- `last_uploaded_document_id` is only ever a resolution
        HINT here, never trusted as proof of access on its own.
        """
        document_id = (request.metadata_filters or {}).get("document_id") or memory.get("last_uploaded_document_id")
        if not document_id:
            answer = _NO_UPLOADED_DOCUMENT_MESSAGES.get(language, _NO_UPLOADED_DOCUMENT_MESSAGES["english"])
            recommendation = await self._contextual_recommendation(memory, "Document Analysis")
            return await self._finalize_intent_response(
                request, session_id, language, memory, answer, "Document Analysis", started, background_tasks,
                recommendation, suggested_actions or [],
                "No uploaded document found to analyze; asked for clarification instead of guessing.", confidence=0.3,
            )
        try:
            result = await self.document_service.analyze(
                DocumentAnalysisRequest(document_id=document_id, language=language, session_id=session_id),
                authenticated_user_id=authenticated_user_id,
            )
        except ForbiddenError:
            answer = _DOCUMENT_ACCESS_DENIED_MESSAGES.get(language, _DOCUMENT_ACCESS_DENIED_MESSAGES["english"])
            recommendation = await self._contextual_recommendation(memory, "Document Analysis")
            return await self._finalize_intent_response(
                request, session_id, language, memory, answer, "Document Analysis", started, background_tasks,
                recommendation, suggested_actions or [], "Document access denied by ownership check.", confidence=0.2,
            )
        except BadRequestError:
            answer = _DOCUMENT_UNAVAILABLE_MESSAGES.get(language, _DOCUMENT_UNAVAILABLE_MESSAGES["english"])
            recommendation = await self._contextual_recommendation(memory, "Document Analysis")
            return await self._finalize_intent_response(
                request, session_id, language, memory, answer, "Document Analysis", started, background_tasks,
                recommendation, suggested_actions or [], "Resolved document had no readable content.", confidence=0.2,
            )
        answer = f"{self._format_document_analysis_answer(result)}\n\n{LEGAL_DISCLAIMER}"
        return await self._finalize_intent_response(
            request, session_id, language, memory, answer, "Document Analysis", started, background_tasks,
            result.recommended_lawyer, suggested_actions or [],
            "Answered from the uploaded document's own indexed content.", confidence=result.confidence,
        )

    def _format_document_analysis_answer(self, result: DocumentAnalysisResponse) -> str:
        """Keeps only what's genuinely useful for a chat reply -- `result`'s
        `action_items`/`missing_information` fields are identical static
        boilerplate on every single analysis regardless of document content
        (see `DocumentService.analyze`), so surfacing them here would
        reintroduce exactly the always-present-regardless-of-relevance
        boilerplate earlier parts already removed from chat answers.
        """
        parts = [result.executive_summary.strip()]
        if result.key_risks:
            parts.append("**Key risks noticed:**\n" + "\n".join(f"- {risk}" for risk in result.key_risks))
        if result.important_clauses:
            parts.append("**Clauses worth noting:**\n" + "\n".join(f"- {clause}" for clause in result.important_clauses))
        return "\n\n".join(part for part in parts if part)

    async def _resolve_followup_question(self, question: str, memory: dict[str, Any]) -> str:
        transcript = "\n".join(
            f"{message['role']}: {message['content']}" for message in memory.get("messages", [])[:-1]
        )
        prompt = prompt_registry.render(
            "conversation_prompt",
            summary=memory.get("summary") or "None",
            messages=transcript or "None",
            question=question,
        )
        llm_response = await self.llm.chat([ChatMessage(role="user", content=prompt)], temperature=0.0)
        if llm_response.error:
            # Pronoun-only follow-ups are unusable retrieval queries on their
            # own. If rewriting is unavailable, attach the most recent real
            # user topic deterministically instead of searching for "iska".
            prior_user_turn = next(
                (
                    message.get("content", "").strip()
                    for message in reversed(memory.get("messages", [])[:-1])
                    if message.get("role") == "user" and len(message.get("content", "").split()) >= 3
                ),
                "",
            )
            return f"{prior_user_turn}\nFollow-up instruction: {question}" if prior_user_turn else question
        return llm_response.content.strip() or question

    async def _contextual_recommendation(
        self, memory: dict[str, Any], label: str, language: str = ""
    ) -> LawyerRecommendation:
        intent = memory.get("current_intent") or label
        category = memory.get("legal_category") or "General Law"
        return await self.recommendations.recommend(intent, category, language=language)

    async def _claim_session_if_unowned(
        self, session_id: str, memory: dict[str, Any], authenticated_user_id: str | None,
    ) -> dict[str, Any]:
        """Part 46 "Authenticated User Ownership": the first authenticated
        turn on a still-unclaimed session records that ownership permanently
        -- `check_access` already ran and would have rejected a mismatched
        account, so reaching here means either unclaimed or already this
        same account. A no-op (no extra write) once already claimed.
        """
        if authenticated_user_id and not memory.get("owner_user_id"):
            memory = await self.memory.update(session_id, owner_user_id=authenticated_user_id)
        return memory

    async def _store_entity_fact_if_any(self, session_id: str, memory: dict[str, Any], text: str) -> None:
        """Part 35 "Entity Memory": runs on every DECLARATIVE turn (cheap,
        pure regex, no LLM), extracting and persisting at most one fact if
        `text` states one. Appends rather than overwrites, so a fact stated
        several turns ago stays recallable after the topic has moved on.

        Skips a message that's itself a fact-identifying recall query
        (Part 39) -- "What happened to my phone, was it also stolen?"
        mentions "phone"+"stolen" too, but it's asking, not declaring;
        storing it as a confirmed fact would let the mere act of asking an
        uncertain question about an entity manufacture a "fact" for it,
        defeating the guarantee that a DIFFERENT entity than the one
        actually stored never gets answered for. An action-seeking message
        ("mera bike chori ho gya hai kya karu") still stores normally --
        `is_recall_query` already treats that as NOT a recall query.
        """
        if entity_memory.is_recall_query(text):
            return
        turn_index = len(memory.get("messages") or [])
        fact = entity_memory.extract_fact(text, turn_index)
        if fact is None:
            return
        facts = [*(memory.get("entities") or []), fact.to_dict()]
        memory["entities"] = facts
        await self.memory.update(session_id, entities=facts)

    async def _finalize_intent_response(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        answer: str,
        conv_intent_label: str,
        started: float,
        background_tasks: BackgroundTasks | None,
        recommendation: LawyerRecommendation,
        suggested_actions: list[str],
        confidence_reason: str,
        confidence: float = 1.0,
        pending_clarification: str | None = None,
    ) -> ChatResponse:
        """Shared tail for every conversation-intent short-circuit branch (General
        Conversation, Lawyer Recommendation, Translation, Summarization): persists
        memory/history the same way `_respond_with_draft_turn` does, and skips
        retrieval/sources entirely since none of these branches use RAG.

        `pending_clarification` defaults to `None`, clearing any earlier pending
        state on every turn except the one branch that deliberately sets it
        (the "which language?" translation prompt) -- so a stale clarification
        flag never outlives the single turn it was meant to be answered by.
        """
        await self.memory.update(session_id, language_preference=language, pending_clarification=pending_clarification)
        await self.memory.append(session_id, "assistant", answer)
        if background_tasks is not None:
            background_tasks.add_task(self.memory.summarize_if_needed, session_id, self.llm)
        else:
            await self.memory.summarize_if_needed(session_id, self.llm)
        message_id = str(uuid4())
        await self.history.insert(
            {
                "message_id": message_id,
                "session_id": session_id,
                "conversation_id": request.conversation_id,
                # Security finding C2: this must be the SERVER-VERIFIED owner
                # (`memory["owner_user_id"]`, set by `check_access`/
                # `_claim_session_if_unowned` from the caller's JWT), never
                # the client-supplied `request.user_id` -- stamping the
                # latter here made `erase_user_data`'s
                # `ChatRepository().delete_by_user(user_id)` match nothing
                # for a real authenticated user (their messages were never
                # actually tagged with their real id), so `DELETE /me/data`
                # reported success while leaving every chat message behind.
                "user_id": memory.get("owner_user_id"),
                "question": request.question,
                "answer": answer,
                "intent": {
                    "intent": conv_intent_label,
                    "legal_category": memory.get("legal_category") or "General Law",
                    "reason": f"Conversation intent: {conv_intent_label}.",
                    "confidence": confidence,
                },
                "conversation_intent": conv_intent_label,
                "entities": {},
                "sources": [],
            }
        )
        response = ChatResponse(
            message_id=message_id,
            session_id=session_id,
            answer=answer,
            sources=[],
            confidence=confidence,
            confidence_label=self._confidence_label(confidence),
            confidence_reason=confidence_reason,
            lawyer_recommendation=recommendation,
            detected_language=language,
            detected_intent=conv_intent_label,
            conversation_intent=conv_intent_label,
            suggested_actions=suggested_actions,
            llm_provider=self.llm.provider_name,
            llm_model=getattr(self.llm, "model", ""),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self._log_routing_decision(
            session_id,
            message_id,
            request.question,
            conv_intent_label,
            _ROUTE_SLUG_BY_CONV_INTENT.get(conv_intent_label, conv_intent_label.lower().replace(" ", "_")),
            memory_hit=conv_intent_label in _MEMORY_HIT_CONV_INTENTS,
            rag_used=False,
            response_modification=conv_intent_label == "Response Modification",
            draft_mode=bool(memory.get("draft_mode")),
        )
        await self._log_query(
            request, session_id, message_id, response, background_tasks,
            owner_user_id=memory.get("owner_user_id"),
        )
        return response

    async def _respond_from_cache(
        self,
        request: ChatRequest,
        session_id: str,
        language: str,
        memory: dict[str, Any],
        intent: IntentResponse,
        cached_entry: dict[str, Any],
        hit_type: str,
        started: float,
        background_tasks: BackgroundTasks | None,
    ) -> ChatResponse:
        """Builds a `ChatResponse` from a response-cache hit without touching
        retrieval or the LLM. `cached_entry["response"]` was written by the
        cache-store call at the end of the main RAG path in `answer()`.
        """
        # A refusal cached before this invariant existed still carries the
        # citations it was stored with. Sanitized on the way out rather than
        # trusted, so a legacy entry cannot reintroduce the defect.
        cached = sanitize_payload(cached_entry["response"])
        sources = [SourceCitation(**item) for item in cached.get("sources", [])]
        recommendation = await self.recommendations.recommend(intent.intent, intent.legal_category)
        await self.memory.update(
            session_id,
            language_preference=language,
            current_intent=intent.intent,
            legal_category=intent.legal_category,
            pending_clarification=None,
        )
        await self.memory.append(session_id, "assistant", cached["answer"])
        if background_tasks is not None:
            background_tasks.add_task(self.memory.summarize_if_needed, session_id, self.llm)
        else:
            await self.memory.summarize_if_needed(session_id, self.llm)
        message_id = str(uuid4())
        await self.history.insert(
            {
                "message_id": message_id,
                "session_id": session_id,
                "conversation_id": request.conversation_id,
                # Security finding C2: this must be the SERVER-VERIFIED owner
                # (`memory["owner_user_id"]`, set by `check_access`/
                # `_claim_session_if_unowned` from the caller's JWT), never
                # the client-supplied `request.user_id` -- stamping the
                # latter here made `erase_user_data`'s
                # `ChatRepository().delete_by_user(user_id)` match nothing
                # for a real authenticated user (their messages were never
                # actually tagged with their real id), so `DELETE /me/data`
                # reported success while leaving every chat message behind.
                "user_id": memory.get("owner_user_id"),
                "question": request.question,
                "answer": cached["answer"],
                "intent": intent.model_dump(),
                "conversation_intent": "General Legal Query",
                "entities": {},
                "sources": [source.model_dump() for source in sources],
            }
        )
        response = ChatResponse(
            message_id=message_id,
            session_id=session_id,
            answer=cached["answer"],
            sources=sources,
            evidence_pages=evidence_pages_from_citations(sources),
            currency_notice=_currency_notice_for(sources, request.question),
            applicable_law=_applicable_law_from(sources),
            no_verified_context=bool(cached.get("no_verified_context")),
            # A cache hit skips retrieval by design, so there are no chunks to
            # score this turn. Grounding still comes from the stored citations.
            **_confidence_fields(sources, [], cached.get("confidence", 1.0)),
            confidence=cached.get("confidence", 1.0),
            confidence_label=cached.get("confidence_label", self._confidence_label(cached.get("confidence", 1.0))),
            confidence_reason=cached.get("confidence_reason", "Served from response cache."),
            lawyer_recommendation=recommendation,
            retrieved_sections=cached.get("retrieved_sections", []),
            detected_language=language,
            detected_intent=intent.intent,
            conversation_intent="General Legal Query",
            llm_provider=cached.get("llm_provider", ""),
            llm_model=cached.get("llm_model", ""),
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=cached.get("prompt_tokens"),
            completion_tokens=cached.get("completion_tokens"),
            cache_hit=hit_type,
        )
        self._log_routing_decision(
            session_id, message_id, request.question, intent.intent, "cache_hit",
            memory_hit=False, rag_used=False, response_modification=False, draft_mode=bool(memory.get("draft_mode")),
        )
        await self._log_query(
            request, session_id, message_id, response, background_tasks,
            cache_hit=hit_type, owner_user_id=memory.get("owner_user_id"),
        )
        return response

    def _log_routing_decision(
        self,
        session_id: str,
        message_id: str,
        question: str,
        conversation_intent: str,
        route: str,
        *,
        memory_hit: bool,
        rag_used: bool,
        response_modification: bool,
        draft_mode: bool,
        request_failed: bool = False,
    ) -> None:
        """Delegates to `app.services.chat_support.analytics.
        log_routing_decision` (Phase 1 god-object split) -- kept as a
        same-named method for call-site consistency with `_log_query`."""
        log_routing_decision(
            session_id, message_id, question, conversation_intent, route,
            memory_hit=memory_hit, rag_used=rag_used, response_modification=response_modification,
            draft_mode=draft_mode, request_failed=request_failed,
        )

    async def _log_query(
        self,
        request: ChatRequest,
        session_id: str,
        message_id: str,
        response: ChatResponse,
        background_tasks: BackgroundTasks | None,
        cache_hit: str | None = None,
        owner_user_id: str | None = None,
    ) -> None:
        """Delegates to `app.services.chat_support.analytics.log_query`.
        Reads `self.query_log` fresh on every call (not captured once in
        `__init__`) -- `test_multi_turn_conversations.py` reassigns
        `service.query_log` after construction and relies on this method
        picking up the replacement.
        """
        await log_query(
            self.query_log, request, session_id, message_id, response, background_tasks,
            cache_hit=cache_hit, owner_user_id=owner_user_id,
        )

    def _safe_llm_text(self, llm_response: LLMResponse, fallback: str) -> str:
        """Delegates to `app.llm.safety.safe_llm_text` (Phase 1 god-object
        split) -- kept as a same-named method since it's called from ~10
        sites across this class as `self._safe_llm_text(...)`."""
        return safe_llm_text(llm_response, fallback)

    def _failure_category(self, error: str | None) -> str | None:
        """Delegates to `app.llm.safety.failure_category` -- see
        `_safe_llm_text`'s docstring for why this stays a method."""
        return failure_category(error)

    def _looks_like_llm_error(self, text: str) -> bool:
        """Delegates to `app.llm.safety.looks_like_llm_error` -- see
        `_safe_llm_text`'s docstring for why this stays a method."""
        return looks_like_llm_error(text)

    def _section_lookup_ambiguous_acts(self, intent: str, ranked: list[Any]) -> list[str] | None:
        """A bare "Section N" query with no Act named and no conversation-
        context hint (see `LegalRetriever._apply_section_number_floor`'s
        `_find_act_hint`) floors every same-numbered chunk from every Act to
        the identical score and tags each one's metadata with
        `ambiguous_acts` -- the retriever's own signal that it genuinely
        could not pick one, rather than chat_service re-deriving the same
        "was there a hint" logic a second time. Checking only the top-ranked
        chunk is a deliberate, cheap heuristic: the floor pushes every tied
        match's score well above ordinary similarity scores, so one of them
        reliably ends up first after reranking.
        """
        if intent != "SECTION_LOOKUP" or not ranked:
            return None
        acts = ranked[0].metadata.get("ambiguous_acts")
        return acts if acts and len(acts) > 1 else None

    def _render_act_disambiguation_question(self, acts: list[str], language: str) -> str:
        act_list = "\n".join(f"- {act}" for act in acts)
        if language == "hi":
            return (
                "Yeh section number ek se zyada Acts mein maujood hai. Kripya batayein "
                f"aap kis Act ki baat kar rahe hain:\n{act_list}"
            )
        return f"This section number exists in more than one Act. Which one do you mean?\n{act_list}"

    def _fallback_answer(self, intent: str, chunks: list[Any], query: str = "") -> str:
        """Used only when the primary LLM call itself failed (see
        `failure_reason` in `answer()`/`answer_stream()`) -- there's no LLM
        available here to judge whether a chunk is actually topically
        relevant (a vector-similarity/rerank score alone doesn't reliably
        tell a genuine match from a same-domain-but-wrong-topic one, e.g. a
        motor-accident-compensation provision scoring well for a bike THEFT
        question).

        Part 41 relevance gate: previously used `chunks[0]` unconditionally,
        which meant a reranked-but-topically-wrong chunk (confirmed live: a
        public-nuisance notice template surfacing for "accha fir kya hota
        hai", ranked ABOVE a chunk that actually discussed FIR) got dumped
        as "closest matching material". `_most_relevant_chunk` promotes a
        lower-ranked chunk over the reranker's own top pick when it shares
        real vocabulary with `query` and the top pick doesn't -- see its
        own docstring for why this deliberately does NOT reject outright
        when nothing overlaps at all (a real risk of false rejections for
        Hindi/Hinglish queries against this corpus's largely English text).
        Framed as raw retrieved material from a failed attempt either way,
        not a considered answer, and points at Part 31's retry mechanism.
        """
        if not chunks:
            return INSUFFICIENT_CONTEXT_MESSAGE
        best = (self._most_relevant_chunk(query, chunks) if query else None) or chunks[0]
        metadata = best.metadata or {}
        act = str(metadata.get("act_name") or "").strip()
        # Generic parser placeholders are not meaningful citations and were
        # exposed verbatim in the reported response as "under This Act".
        if act.casefold() in {"this act", "the act", "act"}:
            act = ""
        document = str(metadata.get("source_document") or metadata.get("source") or "").strip()
        section = str(metadata.get("section_number") or "").strip()
        label_parts = [part for part in (act or document, f"Section {section}" if section else "") if part]
        source_label = ", ".join(label_parts) or "a retrieved legal source"
        # Never dump a character-sliced statute fragment. It can begin/end in
        # the middle of a sentence and, without an available model, the app
        # cannot safely explain whether that fragment actually answers the
        # user's question. Preserve the verified citation and be explicit
        # about what did not complete.
        return (
            "I found potentially relevant verified material in "
            f"{source_label}, but the answer-generation service did not complete. "
            "I have not turned the source into a legal conclusion because its relevance could not be fully "
            'checked. Say "retry" to try the same question again.'
        )

    def _most_relevant_chunk(self, query: str, chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
        """Delegates to `app.rag.relevance.most_relevant_chunk` -- see
        `_filter_relevant_context`'s docstring for why this stays a method."""
        return most_relevant_chunk(query, chunks)

    def _citation_from_chunk(self, chunk: RetrievedChunk) -> SourceCitation:
        """Delegates to the one citation mapping in `app/rag/citation.py`.

        It used to build the citation here, and the two mappings had drifted:
        this one dropped every governance field, so the citations chat users
        actually saw never carried a verification or amendment status. See
        `citation_from_metadata` for what that cost.
        """
        metadata = chunk.metadata
        return citation_from_metadata(
            metadata,
            source_document=str(metadata.get("source_document") or metadata.get("source") or "legal_source"),
        )

    def _confidence(
        self, chunks: list[RetrievedChunk], intent_confidence: float, entity_confidence: float
    ) -> float:
        """Delegates to `app.rag.chat_confidence.confidence_score` (Phase 1
        god-object split) -- kept as a same-named method because it's
        called from ~10 sites across this class."""
        return confidence_score(chunks, intent_confidence, entity_confidence)

    def _confidence_label(self, value: float) -> str:
        """Delegates to `app.rag.chat_confidence.confidence_label` -- see
        `_confidence`'s docstring for why this stays a method (also called
        directly on a `ChatService` instance by `test_response_shape.py`)."""
        return confidence_label(value)

    def _confidence_reason(self, ranked: list[Any], subject_matter_intent: str) -> str:
        """Delegates to `app.rag.chat_confidence.confidence_reason` -- see
        `_confidence`'s docstring for why this stays a method."""
        return confidence_reason(ranked, subject_matter_intent)

    async def _generate_related_questions(self, question: str, answer: str, language: str) -> list[str]:
        """Best-effort follow-up-question suggestions; never breaks the main answer.

        Wrapped in `wait_for` (not just try/except) because, unlike
        `summarize_if_needed` (fire-and-forget via `background_tasks`), this
        call sits on the synchronous response critical path -- a slow-but-
        eventually-200 provider response could otherwise add the provider's
        full timeout to every RAG answer instead of failing fast.
        """
        prompt = prompt_registry.render("related_questions_prompt", question=question, answer=answer, language=language)
        try:
            llm_response = await asyncio.wait_for(
                self.llm.chat([ChatMessage(role="user", content=prompt)], temperature=0.3),
                timeout=RELATED_QUESTIONS_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - related questions are a suggestion strip; a failure returns none and never affects the answer
            log.warning("related_questions_generation_failed", error=str(exc))
            return []
        if llm_response.error:
            return []
        lines = [re.sub(r"^[\-\*\d.)]+\s*", "", line).strip() for line in llm_response.content.splitlines()]
        return [line for line in lines if line][:4]
