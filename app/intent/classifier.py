import json
import re
from dataclasses import dataclass, replace
from typing import Any

from app.core.config import settings
from app.drafting.intent import DraftIntentDetector
from app.language.detector import is_language_preference_command, resolve_language_token
from app.language.normalizer import QueryNormalizer
from app.language.typo_tolerance import clarification_question, normalize_for_routing
from app.llm.base import ChatMessage, LLMProvider
from app.llm.prompts import prompt_registry

# The 12 discourse-level intents from the "Intent Priority Engine" spec. This
# is a DIFFERENT axis from `app.intent.detector.IntentDetector` (which
# classifies legal SUBJECT MATTER, e.g. "Cyber Crime"/"Divorce") -- this
# classifies the *kind of ask* (question vs draft vs translate vs small
# talk, etc.) so `ChatService` can route to the right workflow before
# spending an LLM call on the default RAG path.
CONVERSATION_INTENTS = [
    "Lawyer Recommendation",
    "Draft Generation",
    "Translation",
    "Response Modification",
    "Conversation Memory",
    "Summarization",
    "Legal Dictionary",
    "Law Comparison",
    "Legal Procedure",
    "Document Analysis",
    "Follow-up Question",
    "General Conversation",
    "Legal Explanation",
    "Legal Advice",
    "General Legal Information",
    "Out of Domain",
    "Non-Indian Jurisdiction",
    # A question about the ASSISTANT rather than about the law ("what can you
    # do?", "tum meri kin legal problems me madad kar sakte ho?"). It has no
    # answer in any statute, so routing it through retrieval could only ever
    # produce the strict-RAG refusal -- which is what it did: a user's second
    # message in a real session, asking what the product is for, was answered
    # "no verified document related to this question is available in the
    # Knowledge Base."
    "Capability Question",
    # A turn that is nothing but "reply to me in <language> from now on".
    # Conversation control, not a legal question: it has no answer in any
    # statute, and routing it through retrieval produced exactly the
    # strict-RAG refusal it did for "Capability Question" -- observed twice
    # in one session ("Mujhe simple Hindi mein jawab diya karo" classified
    # "Legal Advice", "Actually Hinglish mein jawab do" classified "General
    # Legal Information", both answered from retrieval).
    "Language Preference",
    # A short conversational turn whose only unreadable token has more than
    # one equally-good correction. Asking which was meant is the honest
    # move; guessing one and routing on it is not. See
    # `app/language/typo_tolerance.py`.
    "Typo Clarification",
]

# Shaping hints handed to `rag_prompt.md` for the intents that fall through to
# the normal RAG pipeline unchanged -- these only nudge tone/structure, they
# never change routing, so precision between them is not load-bearing. Kept as
# optional building blocks, not a checklist -- `system_prompt.md` already
# instructs the model to never force a fixed template onto every reply, and
# these hints are written to stay consistent with that ("only if it helps",
# "only if genuinely relevant") rather than contradict it.
# Shared formatting rules for the genuinely informational intents (a fresh
# question about what something is, how it works, or how it compares --
# NOT drafting, personal-situation advice, translation, summarization, or a
# follow-up, each of which already has its own natural shape and shouldn't
# be forced into this one). Two of the requested sections aren't produced
# here on purpose: "Related Questions" is already a separate, dedicated
# field (`ChatResponse.related_questions`, its own LLM call in
# `_generate_related_questions`) so repeating it inline would duplicate
# what the frontend already renders as its own list; "Disclaimer" is
# already appended by `chat_service.py` after every answer regardless of
# intent, and the model is separately told not to generate its own
# redundant one (see system_prompt.md).
_INFORMATIONAL_RESPONSE_FORMAT = (
    "Let the question's own shape and complexity decide the answer's shape and length -- don't run every "
    "informational question through the same template. A bare definition ('FIR kya hoti hai?') deserves "
    "2-4 plain-language sentences and nothing else: no heading, no bullets, no example. A procedural "
    "question ('FIR kaise file karte hain?') deserves a short numbered list of steps. A comparison deserves "
    "a compact side-by-side, table only if it genuinely clarifies more than prose would. Only reach for "
    "headings/bullets once the answer has enough real substance to need organizing (roughly: more than "
    "5-6 lines of content). Only include an example when the concept is genuinely hard to picture without "
    "one, the user explicitly asked for one, or it materially improves understanding -- most definitions and "
    "simple procedures don't need one at all. Mention the relevant Act/section only when it's actually asked "
    "for or materially changes what the user should do, woven naturally into the explanation rather than "
    "listed separately. Explain any legal term in plain English (or Hinglish, matching the user) the first "
    "time it's used. Default target length: 2-6 lines for a simple question, 6-15 for a normal one -- go "
    "longer only if the question itself asks for more detail ('explain in detail', 'complete', 'A-Z', 'step "
    "by step') or the user has already indicated they want depth."
)

CONVERSATION_INTENT_HINTS = {
    "General Legal Information": f"answer plainly and directly. {_INFORMATIONAL_RESPONSE_FORMAT}",
    "Legal Explanation": f"a definition/concept query. {_INFORMATIONAL_RESPONSE_FORMAT}",
    "Law Comparison": (
        "a comparison query — use a short comparison table only if it genuinely clarifies the distinction, "
        f"otherwise a compare/contrast paragraph; end with the key difference in one line. {_INFORMATIONAL_RESPONSE_FORMAT}"
    ),
    "Legal Procedure": (
        "a process query — walk through the steps in order; mention a rough timeline or documents required "
        f"only if genuinely relevant, and note the expected outcome. {_INFORMATIONAL_RESPONSE_FORMAT}"
    ),
    "Document Analysis": "a document-analysis query — summarize what's relevant, flag any risk or important clauses the retrieved material surfaces, and suggest next steps.",
    "Legal Dictionary": f"a definition query — give a concise, precise definition first. {_INFORMATIONAL_RESPONSE_FORMAT}",
    "Follow-up Question": "a follow-up to the earlier conversation — answer the resolved question directly, referencing what was already discussed only where it adds clarity.",
    # The user described a personal legal problem (e.g. "my landlord won't
    # return my deposit") -- NOT a request to draft anything, even though a
    # document (a notice, a complaint) may ultimately be the right next step.
    # Advice-first: explain the situation before ever mentioning drafting,
    # and never start collecting draft fields from this response.
    "Legal Advice": (
        "the user described a personal legal problem, not a request to draft anything. If the situation is "
        "one a person would naturally feel upset or worried about (theft, injury, an accident, a bounced "
        "cheque, harassment, domestic violence), open with exactly one short, warm line acknowledging that "
        "before getting into the answer — don't overdo the sympathy or repeat it. Then respond like a "
        "knowledgeable friend talking them through it in plain conversation: lead with what they should "
        "actually do, in the order they'd naturally do it (e.g. 'file an FIR first, then block the SIM, "
        "then...'). Only bring up their legal rights, alternative options, required documents, or the "
        "specific law/section where it genuinely adds something for THIS situation — never as a fixed "
        "checklist you run through every time, and never under rigid labels like 'Legal Rights' / "
        "'Realistic Options' / 'Practical Next Steps' unless the answer is long enough to actually need that "
        "structure. Close with at most one line offering to prepare the relevant notice/application/complaint "
        "if they'd like — phrased as an offer ('I can also prepare a ... if you'd like'), never as a question "
        "that starts collecting their name, address, or other draft fields, and never repeated elsewhere in "
        "the same answer."
    ),
}

# Action-suggestion labels shown after a response, keyed by conversation
# intent. Deliberately limited to capabilities that already work end-to-end
# today (Find Lawyer, Translate, Summarize, drafting, document upload) --
# no labels for out-of-scope features (e.g. "Document Checklist," "Legal
# Strategy") since those would be dead-end buttons with no backend yet.
# "Draft Generation" is intentionally absent/empty: the drafting flow already
# has its own preview/export UI via the `draft` field, no duplicate chips.
CONVERSATION_INTENT_ACTIONS = {
    "General Legal Information": ["Find Lawyer", "Translate", "Summarize"],
    "Legal Explanation": ["Find Lawyer", "Translate", "Summarize"],
    "Law Comparison": ["Find Lawyer", "Translate"],
    "Legal Procedure": ["Generate Legal Notice", "Find Lawyer", "Translate"],
    "Document Analysis": ["Upload Documents", "Find Lawyer", "Summarize"],
    "Legal Dictionary": ["Find Lawyer", "Translate"],
    "Follow-up Question": ["Find Lawyer", "Translate", "Summarize"],
    "Legal Advice": ["Generate Legal Notice", "Find Lawyer", "Upload Documents", "Translate"],
    "Lawyer Recommendation": ["Generate Legal Notice", "Translate"],
    "Translation": [],
    "Response Modification": [],
    "Conversation Memory": [],
    "Summarization": ["Find Lawyer"],
    "General Conversation": [],
    "Draft Generation": [],
    "Out of Domain": [],
    "Non-Indian Jurisdiction": ["Find Lawyer"],
    "Capability Question": ["Generate Legal Notice", "Upload Documents", "Find Lawyer"],
    # Both are single-turn conversation control: the next thing the user does
    # is ask their real question, and a chip row would only get in the way.
    "Language Preference": [],
    "Typo Clarification": [],
}

_LAWYER_RECOMMENDATION_PATTERN = re.compile(
    r"\brecommend\s+(a|an|me a)?\s*(lawyer|advocate|vakil)\b"
    r"|\bfind me\s+(a|an)\s+(lawyer|advocate)\b"
    r"|\bwhich\s+(lawyer|advocate)\s+should\b"
    r"|\bvakil\s+chahiye\b",
    re.IGNORECASE,
)

_TRANSLATE_PATTERN = re.compile(r"\btranslat|\btarjuma", re.IGNORECASE)
_SUMMARIZE_PATTERN = re.compile(r"\bsummar(y|ize|ise)\b|\bsankshep\b|\btl;?dr\b", re.IGNORECASE)

# Whole-conversation summarization ("Summarize our conversation") is a
# different, pre-existing feature from Part-24's "Response Modification"
# summarize (which acts on just the last reply, see `_MODIFICATION_PATTERNS`
# below) -- checked first, and only matches when the message explicitly names
# the conversation/chat as the target, so it doesn't swallow the new
# last-response-only phrasing ("Summarize this.", "TL;DR").
_CONVERSATION_SUMMARY_PATTERN = re.compile(
    r"\bsummar(y|ize|ise)\b.{0,20}\b(conversation|chat|everything|so far|discussion|session)\b"
    r"|\b(conversation|chat|discussion)\b.{0,20}\bsummar(y|ize|ise)\b",
    re.IGNORECASE,
)

# Part 24 "Response Modification Engine": natural-language commands that
# transform the LAST assistant reply (never trigger retrieval). Each pattern
# is anchored to the *whole* message (optionally wrapped in "please"/
# trailing punctuation) rather than a loose substring match -- a bare
# "Compare." or "Table." after a previous answer means "reformat that reply,"
# but "Compare FIR and NCR" or "What is the process to file an FIR?" carry
# real new topical content and must still fall through to the existing
# Law Comparison/Legal Procedure/etc. routing below. This mirrors how
# `_GENERAL_CONVERSATION_PATTERN` is anchored for the same reason.
def _anchored(*phrases: str) -> re.Pattern[str]:
    # The optional "(can|could|will|would) you (please)?" lead-in exists
    # because the bare-phrase-only anchor below rejects anything but the
    # fixed command itself -- so "Make it shorter" matched but the equally
    # common "Can you make it shorter?" fell through all the way to
    # `DraftIntentDetector` instead (it contains the drafting verb "make"
    # with no template match, so it wrongly started an ambiguous "which
    # document would you like?" drafting flow). Still a full-message anchor
    # (`$` at the end), so it only fires when nothing follows the phrase --
    # "Can you make it shorter and also check my contract?" still falls
    # through to normal routing, unchanged.
    body = "|".join(phrases)
    return re.compile(
        rf"^\s*(please\s+)?((can|could|will|would)\s+you\s+(please\s+)?)?({body})\s*[.!?]*\s*$", re.IGNORECASE
    )


_MODIFICATION_WORD_COUNT_PATTERN = re.compile(r"^\s*(please\s+)?(in\s+)?\d{1,4}\s*words?\s*[.!?]*\s*$", re.IGNORECASE)

_MODIFICATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "shorten": _anchored(
        "one sentence", "one paragraph", "make it shorter", "make this shorter", "short answer",
        "keep it short", "shorten this", "shorten", "shorter please", "short",
    ),
    "expand": _anchored(
        "explain in detail", "explain deeply", "give more details", "give me more details", "more details",
        "expand this", "expand", "elaborate this", "elaborate", "advanced explanation", "legal explanation",
    ),
    "simplify": _anchored(
        "explain simply", "explain in easy language", "explain like i'?m a beginner", "explain like i am a beginner",
        "eli5", "simple english", "simple hindi", "simple marathi", "simple tamil", "simple telugu", "easy language",
        "easy words", "simplify this", "simplify", "in simple words", "simple words", "explain in simple words",
        "explain like a beginner",
    ),
    "bullets": _anchored(
        "bullet points", "bullets", "key points", "important points", "convert into bullets",
        "convert into bullet points", "list format", "checklist", "in bullet points", "give me bullet points",
    ),
    "rewrite": _anchored(
        "rewrite professionally", "rewrite legally", "rewrite formally", "rewrite politely",
        "rewrite for a client", "rewrite for a judge", "rewrite in simple words", "rewrite this", "rewrite",
        "professionally", "professional",
    ),
    "table": _anchored(
        "convert into table", "table format", "table", "comparison table", "difference table", "pros and cons",
        "compare", "comparison", "compare this",
    ),
    "examples": _anchored(
        "give examples", "give me examples", "give an example", "real-life example", "real life example",
        "practical example", "case example", "example", "examples",
    ),
    "timeline": _anchored(
        "timeline", "procedure", "step by step", "step-by-step", "what happens next",
        "explain step by step", "explain step-by-step",
    ),
    "visual": _anchored("mind map", "flow chart", "flowchart", "decision tree", "hierarchy"),
    "summarize": _anchored(
        "summarize", "summarize this", "summarise", "summarise this", "give me a summary", "give a summary",
        "short summary", "summary", "tl;?dr", "tldr", "in brief", "briefly explain", "in short", "briefly",
    ),
}
# Part 26 "Conversation Memory Engine": questions ABOUT the conversation
# itself (what was asked, what topic, what draft, what was uploaded, resume)
# -- these must never invoke RAG, since there's nothing to retrieve; the
# answer lives entirely in `memory["summary"]`/`memory["messages"]`. Distinct
# from `Follow-up Question` below, which resolves a pronoun into a NEW
# standalone legal question that still needs retrieval (e.g. "What about
# self-defense?" after a discussion of assault) -- this bucket is for
# questions about the conversation's own history, not the law.
_CONVERSATION_MEMORY_PATTERN = _anchored(
    "what did i ask", "what did i ask first", "what did i ask you", "what did i ask earlier",
    "what was my question", "what was my first question", "what was my previous question", "what was my last question",
    "what were we discussing", "what were we talking about",
    "which topic were we talking about", "which topic are we on", "what topic are we on", "what topic were we on",
    "what did you just explain", "what did you explain", "what did you just say", "what did you say",
    "can you remind me", "remind me", "remind me what we discussed",
    "what did i upload", "what document did i upload", "which document did i upload", "what file did i upload",
    "which draft were we creating", "which draft were we making", "what draft were we making", "what draft were we creating",
    "continue", "continue from before", "continue please", "continue from where we left off", "let's continue",
)
# Hindi/Hinglish memory-recall cues, checked separately from
# `_CONVERSATION_MEMORY_PATTERN` above because Hinglish phrasing varies far
# more than the fixed English phrase list can enumerate (e.g. "maine
# previous chat me kon sa question pucha tha" mixes English nouns into
# Hindi sentence structure) -- a `.search()` over a few characteristic
# substrings generalizes across that variance where a whole-message
# `_anchored()` list can't. Deliberately narrower than a bare keyword like
# "pucha" alone would be (that'd false-match "police ko pucha ki..." etc.);
# every alternative pairs the recall verb/noun with a first-person or
# "previous/pichla" anchor.
_CONVERSATION_MEMORY_HINGLISH_PATTERN = re.compile(
    # "Ab corrected deposit amount, disputed deduction, landlord ka naam aur keys
    # handover date batao" / "Mera naam, landlord ka naam ... batao": reading back
    # facts the user gave earlier. Anchored on possessive/context words so a
    # general "<law> ka amount batao" is still a legal question.
    r"(?:mera|meri|mere|landlord\s+ka|corrected|disputed|pending|keys?\s+handover|handover\s+date)\b[^.?!]{0,90}\b(?:batao|bata\s+do|bataiye|btao)\b"
    r"|pichl[ae]\s+(sawaal|prashn|question)|pehl[ae]\s+(sawaal|prashn|question)"
    r"|maine\s+(pehle\s+|previous\s+)?(kya|kaun\s*sa|kon\s*sa)\s+(sawaal|prashn|question)"
    r"|maine\s+kya\s+(poo?cha|pucha)|maine.{0,20}?(poo?cha|pucha)\s+tha"
    r"|previous\s+(chat|conversation|message)\s+(me|mein)\b"
    r"|hum\s+kya\s+(baat|discuss)\s+kar\s+rahe"
    r"|hum\s+kis\s+(baare|topic)\s+mein\s+baat"
    # Part 49: "previous question kya tha?" -- the English nouns "previous"/
    # "last"/"first"/"question" code-switched directly into Hindi sentence
    # structure ("X kya tha?"), rather than the Hindi "pichla"/"pehla
    # sawaal" phrasing already covered above. A real, common alternative
    # phrasing of the same recall question that had no matching alternative.
    r"|(previous|last|first)\s+question\s+kya\s+(tha|thi)",
    re.IGNORECASE,
)
# A narrower, separately-anchored fallback for recalling a specific
# previously-stated fact ("My phone was stolen." -> "What was stolen?").
# Deliberately restricted to exactly one trailing word -- "What was stolen?"
# recalls conversation state, but "What was the maximum bail amount?" or
# "What was the outcome of Kesavananda Bharati?" are genuine new legal
# questions and must keep going to RAG; real questions in this shape almost
# always carry more than one word after "was".
_CONVERSATION_MEMORY_FACT_RECALL_PATTERN = re.compile(r"^\s*what\s+was\s+\w+\??\s*$", re.IGNORECASE)

# Part 31 "State Manager & Recovery Engine": only meaningful when
# `memory["last_failed_question"]` is set (checked alongside this pattern in
# `classify()`, never on its own) -- "continue" is deliberately included
# here even though it already means two OTHER things elsewhere ("resume the
# in-progress draft" mid-draft, "continue the conversation generically" via
# `_CONVERSATION_MEMORY_PATTERN` otherwise). Both of those are unaffected:
# the draft case is decided in `ChatService.answer()` before this
# classifier bucket is ever consulted, and the generic case only exists
# when nothing failed, i.e. exactly when this gate is false.
_RETRY_FAILED_REQUEST_PATTERN = _anchored(
    "retry", "try again", "please retry", "retry that", "try that again", "retry it", "continue"
)

# QA 2026-09-11/12 (BUG-013a, `docs/qa/QA_TEST_MATRIX_20260911.md`): a real
# user who doesn't know the magic word "retry" does the natural thing
# instead -- resends their own exact question, sometimes literally
# copy-pasted -- and that got routed as a brand-new "Follow-up Question"
# into an unrelated no-verified-context RAG answer, silently discarding the
# resend instead of resuming the failed drafting/answering attempt. Only
# `_RETRY_FAILED_REQUEST_PATTERN` above (a fixed set of magic words) was ever
# recognised. This compares the incoming message against
# `memory["last_failed_question"]` case/whitespace/punctuation-insensitively
# -- deliberately an equality check, not a fuzzy one: it exists to catch an
# exact or near-exact resend (what a client-side "retry my last message"
# button, or a user literally repeating themselves, actually sends), not to
# guess that a merely similar-looking new question is the same request.
def _looks_like_resend_of_failed_question(candidate: str, last_failed_question: str) -> bool:
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower()).strip(" .!?।")

    normalized_candidate = _normalize(candidate)
    return bool(normalized_candidate) and normalized_candidate == _normalize(last_failed_question)

_DICTIONARY_PATTERN = re.compile(r"\bmeaning of\b|\bdefine\b|\bka matlab\b|\bmatlab kya\b", re.IGNORECASE)
_COMPARISON_PATTERN = re.compile(
    r"\bdifference between\b|\bcompare\b|\bcomparison\b|\bversus\b|\b\w+\s+vs\.?\s+\w+\b"
    # Confirmed live: "Arbitration aur mediation mein kya antar hai?" fell
    # through this pattern entirely (no Hindi/Hinglish alternative existed)
    # and landed in the generic "General Legal Information" intent instead
    # of "Law Comparison" -- this didn't block retrieval itself, but it did
    # mean the comparison-specific prompt hint never applied.
    r"|\b(kya|क्या)\s+(antar|farak|fark|अंतर|फर्क|फ़र्क|अन्तर)\b"
    r"|(अंतर|फर्क|फ़र्क|अन्तर)\s+(है|क्या)",
    re.IGNORECASE,
)
_PROCEDURE_PATTERN = re.compile(
    r"\bprocess (of|to|for)\b|\bprocedure (for|to)\b|\bsteps to\b|\bhow to file\b|\bhow do i file\b"
    # `karwa\w*`/`banwa\w*` are the causative "get X done/get X made" forms
    # (karwayein/karwaen/karwana/karwao, banwayein/banwana) -- distinct from
    # generic "karte/karti/karta" (deliberately excluded: that bare form is
    # far too generic, matching countless non-legal "how do you do X"
    # phrasings, e.g. "car kaise kaam karta hai"). `mil\w*` covers
    # milega/milegi/milti/milta ("kaise milegi" -- how do I get/obtain X).
    r"|\bkaise\s+(\w+\s+){0,2}(kare|karen|karein|karwa\w*|banwa\w*|mil\w*|hota|hoti|hote|hoga|hogi)\b",
    re.IGNORECASE,
)

# The canonical Hindi definition-question shape ("FIR kya hota hai?", "bail
# kya hoti hai?" -- literally "FIR what is?"). Deliberately narrower than a
# bare "kya hai" check (too ambiguous on its own -- could be almost any
# question shape) so this only fires on the specific 3-word copula
# construction that reliably signals "what is X", mirroring what
# `_EXPLANATION_PATTERN` already does for the English "what is X" phrasing.
_HINDI_DEFINITION_PATTERN = re.compile(r"\bkya\s+(hota|hoti)\s+hai\b", re.IGNORECASE)
# Verb + document-noun co-occurrence anywhere in the message (not rigid
# adjacency) so adjectives in between ("review my rental agreement") still
# match.
# Part 50: "explain"/"summariz(e|se)"/Hindi "batao" ("tell me") were missing
# -- a real, common way to ask about an uploaded file ("pdf explain kro",
# "summarize the pdf") fell through to a generic definition/summarization
# hint instead of "Document Analysis"'s summarize-and-flag-risks shape.
# Note: this only fixes phrasings that NAME the document ("pdf"/"file"/
# etc.) -- a pronoun-only follow-up ("explain this", "isme kya hai") is
# handled separately by `_DOCUMENT_PRONOUN_REFERENCE_PATTERN` below, gated
# on `memory["last_uploaded_document_id"]` (Part 51) rather than keywords.
# "samjhao"/"samjha do" ("explain it to me") is the single most common way a
# Hinglish speaker asks for an explanation of a file they just uploaded, and
# it was missing entirely -- confirmed live: uploading a PDF and asking "iss
# document ko smjhao" matched no verb here, so the message never reached
# Document Analysis at all (with a draft in progress it was swallowed as
# field input, and the user was re-shown the same pending-field list). The
# vowel-dropped spellings ("smjhao", "smjha do") are how it is actually
# typed, so both are matched rather than only the fully-spelled form.
_DOCUMENT_ANALYSIS_VERB_PATTERN = re.compile(
    r"\b(analyz(e|se)|review|check|explain|summariz(e|se)|batao|bataiye"
    r"|samjhao|samjhaao|samajhao|smjhao|smjao|samjhaiye|samjha\s*do|smjha\s*do)\b"
    r"|समझा(ओ|इए|ना)|बताओ|बताइए",
    re.IGNORECASE,
)
_DOCUMENT_ANALYSIS_NOUN_PATTERN = re.compile(r"\b(document|agreement|contract|pdf|file|upload(ed)?)\b", re.IGNORECASE)
# Multi-Intent Workflow Orchestration: distinguishes an unambiguous chain
# request ("review this PDF ... draft a notice BASED ON IT") from two
# possibly-unrelated asks listed together ("review the PDF AND make a
# notice") -- only the former should auto-execute the
# Document-Analysis-then-Draft-Generation chain; the latter is genuinely
# ambiguous about whether the draft should be grounded in the review at
# all, and must ask (see "General Clarification Mode") rather than guess.
_WORKFLOW_LINK_PATTERN = re.compile(
    r"based on (it|that|this)|isi\s*ke\s*(basis|aadhar|aadhaar)|usi\s*ke\s*(basis|aadhar|aadhaar)|"
    r"us\s*ke\s*(basis|aadhar|aadhaar)\s*par|on that basis|accordingly",
    re.IGNORECASE,
)

# Phase 1 "Intent Feedback": explicit post-hoc correction of what the
# PREVIOUS turn was classified as -- a genuinely different thing from
# `_extract_correction`'s "here is a new/restated request" (that one
# extracts replacement TEXT for a fresh question; this one names an INTENT
# CATEGORY the user believes should have been used instead). "You
# misunderstood my request" alone (no named category) is already covered by
# `_extract_correction`'s signal-only branch; this only fires when a
# specific category is actually named, or the generic "wrong intent" is
# used without a corrected request.
_INTENT_FEEDBACK_GENERIC_PATTERN = re.compile(r"\bwrong\s+intent\b", re.IGNORECASE)
_INTENT_FEEDBACK_NAMED_PATTERN = re.compile(
    r"\b(?:i\s+wanted|i\s+meant|i\s+need(?:ed)?|this\s+is|this\s+was)\s+(.+)$", re.IGNORECASE
)
# Natural phrase -> the `CONVERSATION_INTENTS` value it names. "legal
# research" has no single matching intent (the classifier has no intent by
# that literal name -- see `workflow_orchestrator.RAG_INTENTS`); mapped to
# "General Legal Information", the generic RAG-fallthrough bucket, which is
# the closest real routing destination.
_INTENT_FEEDBACK_ALIASES: dict[str, str] = {
    "document analysis": "Document Analysis",
    "legal research": "General Legal Information",
    "draft generation": "Draft Generation",
    "drafting": "Draft Generation",
    "draft": "Draft Generation",
    "translation": "Translation",
    "summarization": "Summarization",
    "summary": "Summarization",
    "lawyer recommendation": "Lawyer Recommendation",
}
# Part 51 "Uploaded Document Conversation Context": pronoun/short references
# to "the thing I just uploaded" -- deliberately a bounded, literal phrase
# list (not a bare "this"/"it"/"isme" alone, which would be far too generic
# and risk swallowing ordinary follow-up questions) covering the spec's own
# named examples. Only ever checked when `memory["last_uploaded_document_id"]`
# is set (see `classify()` below), so it can't misfire in a session with no
# upload at all.
_DOCUMENT_PRONOUN_REFERENCE_PATTERN = re.compile(
    r"\bexplain\s+this\b|\bsummariz(e|se)\s+(it|this)\b"
    r"|\bisme\s+kya\b|\bisme\s+(kuch|koi)\s+important\b"
    r"|\bisko\s+(explain|samjhao|smjhao|summariz(e|se))\b|\bye\s+pdf\s+samjhao\b|\byeh\s+pdf\s+samjhao\b"
    r"|\bsummary\s+batao\b|\bkey\s+points\s+batao\b|\bimportant\s+points\s+kya\s+(hain|hai)\b"
    # "iss/is/ye/yeh <document|pdf|file> ko samjhao" -- the demonstrative
    # spelled out in front of the noun, rather than fused into "isko". The
    # list above only covered the fused form, so the equally common "iss
    # document ko smjhao" matched nothing here.
    r"|\b(is|iss|ye|yeh|yah|us|uss|wo|woh)\s+(document|documnet|dastavez|pdf|file|paper)\b",
    re.IGNORECASE,
)
_FOLLOWUP_PATTERN = re.compile(
    r"^\s*(what about|and if|what else|what if|and what about|what happens then|and then|why)\b"
    # "What documents/proof/papers/evidence do I need?" only makes sense
    # relative to whatever situation was just discussed -- it never names
    # the situation itself, so without a prior turn to resolve against it's
    # not answerable at all. `_is_followup` already gates every branch of
    # this pattern on `_has_prior_assistant_turn`, so this alternative can
    # only ever fire mid-conversation, never on a session's first message.
    r"|^\s*(what|which)\s+(documents?|proof|papers?|evidence)\b"
    # Part 49: the Hindi/Hinglish equivalent of "and if"/"what if" ("agar",
    # optionally preceded by "aur"/"kya") was missing -- a genuine
    # conditional follow-up like "aur agar police mana kar de?" ("and if
    # police refuse?") has 6 tokens, over the generic short-message-with-
    # prior-turn fallback's 5-token cap, and no English-pattern match here
    # either, so it fell all the way through to the generic "General Legal
    # Information" catch-all instead of staying on the active topic.
    r"|^\s*((aur|kya)\s+)?agar\b"
    # A message naming its own legal topic ("Information Technology Act,
    # 2000 isko btao") is exactly the shape `_STANDALONE_LEGAL_SIGNAL_PATTERN`
    # exists to keep OUT of Follow-up Question (a topic name alone usually
    # means a fresh question) -- but "isko"/"iske baare mein"/"ispar" ("about
    # THIS", "on THIS") is an explicit deictic pointer back at whatever the
    # conversation was just covering, not a bare topic mention. That cue
    # should win regardless of the surrounding message's word count (unlike
    # the length-capped fallback below, `_FOLLOWUP_PATTERN` isn't
    # token-count-gated at all), because a message can freely restate the
    # topic's full name AND still mean "tell me more about THAT" in the same
    # breath -- word count says nothing about which one it is. Confirmed
    # live: "Information Technology Act, 2000 isko btao" (6 tokens, over the
    # 5-token fallback cap) fell through to a bare, question-less retrieval
    # query and wrongly reported "not in the Knowledge Base" for an Act the
    # conversation had just been discussing.
    # "iska"/"iski"/"iske"/"isme"/"ismein" and their "us-" counterparts are
    # the same explicit deictic pointer as "isko" -- "THIS one's / of THIS".
    # Only "isko" was listed, so "Iska detailed advocate-style answer do."
    # (6 tokens, over the 5-token short-message fallback cap) matched no
    # pattern anywhere and fell through to the "General Legal Information"
    # catch-all, which sent a pronoun-only message to retrieval with no
    # topic in it at all and came back "not in the Knowledge Base".
    r"|\b(isko|iska|iski|iske|isme|ismein|isem|usko|uska|uski|uske|usme|usmein)\b"
    r"|\b(iske\s+baare\s+m(ein|e)|is\s*ke\s+baare\s+m(ein|e)|ispar|is\s+par)\b"
    r"|\b(about\s+(this|that|it)|on\s+(this|that|it)|more\s+(about|on)\s+(this|that|it))\b",
    re.IGNORECASE,
)
_EXPLANATION_PATTERN = re.compile(r"^\s*(explain|what\s+is|what\s+are|why\s+does|why\s+is)\b", re.IGNORECASE)

# A first-person reference ("my", "I", "mera"/"meri"/"mujhe") is what
# distinguishes "my landlord won't return my deposit" (a personal problem —
# advice) from "what is a security deposit" (a concept — explanation) once
# both have already failed to match the drafting/explanation/procedure/etc.
# patterns above. Checked last, right before the generic fallback, so it
# only reclassifies the subset of that catch-all that's actually about the
# user's own situation, not truly ambiguous fragments.
_LEGAL_ADVICE_PATTERN = re.compile(r"\b(my|i've|i'm|i\s+was|i\s+received|i\s+got|mera|meri|mujhe)\b", re.IGNORECASE)

# Part 42 "Answer Quality & Intent Accuracy": a bounded, representative set of
# clearly non-legal topics (not exhaustive -- a novel off-domain topic not
# listed here still falls through to the existing LLM-prompt-level "I'm a
# legal assistant" soft refusal in system_prompt.md/general_legal_knowledge_
# prompt.md, just without this code-level guarantee). Checked early in
# classify() so it preempts the bare-term/short-message fallbacks below --
# an off-domain message must never be swallowed into stale legal context
# just because it's short and a prior assistant turn exists.
_OFF_DOMAIN_PATTERN = re.compile(
    r"\bweather\b|\bmausam\b|\btemperature\b|\brain(?:y|fall)?\s+(today|kal|forecast)"
    r"|\bpython\b|\bjavascript\b|\bjava\b|\bcoding\b|\bprogramming\b|\balgorithm\b|\bsource\s*code\b"
    r"|\b(horror|funny|love|bedtime)\s+story\b|\btell\s+me\s+a\s+story\b|\bkahani\s+sunao\b"
    r"|\bjoke\s+sunao\b|\btell\s+me\s+a\s+joke\b|\bsunao\s+ek\s+joke\b"
    r"|\bcricket\s+score\b|\bmatch\s+score\b|\brecipe\b|\bmovie\s+recommend"
    # Food/entertainment chit-chat ("pizza khana hai") used to fall through
    # to the legal RAG pipeline and come back as a "no verified document in
    # the Knowledge Base" refusal, as if it were a legal question.
    r"|\b(pizza|burger|biryani|pasta|samosa|momos?|chai|coffee|ice\s*cream)\b"
    r"|\b(khana|khaana|peena|khelna|gaana|gana)\s+(hai|chahiye|hain)\b"
    r"|\b(song|gaana)\s+(sunao|suna\s+do|play)\b|\bbored\b|\bmovie\s+(dekhni|dekhna|suggest)",
    re.IGNORECASE,
)

# Non-Indian jurisdiction cue, only meaningful alongside a legal-shaped
# message (checked together with `_STANDALONE_LEGAL_SIGNAL_PATTERN` below) --
# a bare mention of "US"/"UK" on its own isn't a jurisdiction question.
# Excludes messages that also name India/Indian, so "US aur India dono me
# divorce ka process" style comparisons aren't wrongly refused.
_NON_INDIAN_JURISDICTION_PATTERN = re.compile(
    r"\b(us|usa|america|american|uk|united\s+kingdom|britain|british|canada|canadian|australia|australian)\b",
    re.IGNORECASE,
)
_INDIA_PATTERN = re.compile(r"\bindia|indian\b", re.IGNORECASE)

# Part 42 "Answer Quality & Intent Accuracy": a short message that names its
# own legal topic ("Bail chahiye", "Divorce ka process") is a self-contained
# new question, not a continuation of whatever came before -- it should
# never be forced into "Follow-up Question" just because it's short and a
# prior assistant turn exists. Used to narrow both the bare-term-lookup
# branch and the final short-message fallback below. Deliberately a single
# bounded keyword list (not per-phrase patches) covering the common legal
# nouns/verbs this corpus actually deals with.
_STANDALONE_LEGAL_SIGNAL_PATTERN = re.compile(
    r"\b(bail|fir|vakil|advocate|lawyer|divorce|talaq|talak|cheque|police|court|adalat|kanoon|complaint"
    r"|notice|rights?|arrest|warrant|custody|property|rent|kiraya|consumer|tax|harassment|assault|theft"
    r"|chori|dispute|contract|agreement|marriage|shaadi|alimony|maintenance|inheritance|will|wasiyat"
    r"|lawsuit|case|legal|section|act\b|ipc|bnss|crpc|rti)\b",
    re.IGNORECASE,
)

# Part 44: same list as `_STANDALONE_LEGAL_SIGNAL_PATTERN`, minus `fir` and
# `case`. Those two are uniquely bad standalone-topic signals -- unlike
# "bail"/"divorce"/etc., which overwhelmingly mean the user is naming a fresh
# topic even alone, "fir"/"case" are also extremely common in genuine short
# follow-ups (Hinglish "fir" ~ "phir" ("then"), and "case" as a generic word
# for "my situation"). Confirmed by tracing real chained examples: "phone
# chori ho gaya" -> "fir?"/"FIR kaise karu?"/"mere case me?" were all wrongly
# diverted to Legal Dictionary/General Legal Information instead of
# Follow-up Question, losing the phone-theft entity entirely. Used only at
# the two short-message gates below (`_is_bare_term_lookup`'s branch and the
# <=5-token fallback) -- the full pattern (with fir/case) is kept everywhere
# else, including the jurisdiction check, where this ambiguity doesn't apply.
_STRONG_STANDALONE_LEGAL_SIGNAL_PATTERN = re.compile(
    r"\b(bail|vakil|advocate|lawyer|divorce|talaq|talak|cheque|police|court|adalat|kanoon|complaint"
    r"|notice|rights?|arrest|warrant|custody|property|rent|kiraya|consumer|tax|harassment|assault|theft"
    r"|chori|dispute|contract|agreement|marriage|shaadi|alimony|maintenance|inheritance|will|wasiyat"
    r"|lawsuit|legal|section|act\b|ipc|bnss|crpc|rti)\b",
    re.IGNORECASE,
)

# Whole-message check: the entire message must decompose into one or more of
# these fixed greeting/pleasantry units (each optionally trailed by
# punctuation) and nothing else. Deliberately NOT a token-strip-and-check-
# empty approach -- that would also strip ordinary Hindi/Hinglish grammar
# words ("kaise", "kya", "hai") that appear constantly in real legal
# questions, wrongly emptying them out. Anchoring the whole message instead
# means "Hi, my landlord won't return my deposit" can never match (the
# non-greeting remainder breaks the full-string decomposition), while "Hi",
# "Thanks, that helps", and "Namaste, kaise ho?" all correctly do.
_GREETING_UNIT = (
    # Part 41: "hi" had no "+" repetition, unlike "hello+"/"hey+" right next
    # to it -- so the extremely common casual "hii"/"hiii" fell through this
    # whole pattern, landed in `_is_bare_term_lookup`'s "Legal Dictionary"
    # fallback instead of "General Conversation", and reached the RAG/
    # general-knowledge pipeline for a message with zero legal content
    # (confirmed root cause of the "hii" -> unrelated-answer regression).
    r"(hi+|hello+|hlo+|hey+|namaste|good (morning|evening|afternoon)|thanks?|thank you|shukriya|dhanyavad"
    r"|that helps|that'?s helpful|a lot|so much|very much|got it|ok(ay)?|great|cool|nice|sounds good"
    r"|perfect|bye|goodbye|see you|kaise\s+(ho|hain)"
    # Part 41 regression test item 9: "aur kya haal hai" ("so what's up")
    # is small talk, not a legal question, but had no matching unit here at
    # all -- it fell to the same "General Legal Information" fallback as
    # any genuine unmatched legal question and reached RAG unnecessarily.
    # "h" is the extremely common Hinglish shorthand for "hai" ("kya haal h"
    # is at least as common in real typing as the full "kya haal hai").
    r"|(aur\s+)?kya\s+haal(\s+(hai|h|chal\s+raha\s+hai))?|sab\s+kaisa\s+hai"
    # Part 48: "accha" ("oh, I see"/"okay") and "theek hai"/"theek" ("fine"/
    # "okay") are two of the most common Hinglish casual acknowledgments --
    # same class of bug as the "hii" regression above (bare word with no
    # legal content, no matching unit here, fell through to the "Legal
    # Dictionary" bare-term-lookup fallback and reached RAG). The anchored
    # whole-message check this unit feeds into still requires the ENTIRE
    # message to decompose into greeting units, so a real question that
    # happens to start with "accha" ("accha fir kya hota hai") is unaffected.
    r"|accha|theek(\s+hai)?"
    # ------------------------------------------------------------------
    # Multilingual greetings.
    #
    # Reported bug: "Namste" -- a one-letter misspelling of "Namaste" --
    # was answered with the strict-RAG "no verified document in the
    # Knowledge Base" refusal. The typo was the small half of it: only the
    # exact Latin spelling "namaste" was listed, so EVERY native-script
    # greeting failed too. In an app that advertises all 22 Eighth Schedule
    # languages, greeting it in your own language got a knowledge-base
    # refusal on the first message a user ever sends.
    #
    # Romanised, with the spellings people actually type. `namast`/`namask`
    # prefixes with a permissive tail cover namaste/namastey/namste/
    # namaskar/namaskaar/namaskaram/namaskara in one alternative each.
    # `namst`/`namsk` are NOT redundant with `namast`/`namask`: dropping the
    # second vowel ("Namste", "Namskar") is the single most common way this
    # word is mistyped, and it is what was actually reported.
    r"|namast\w*|namst\w*|namask\w*|namsk\w*|pranaam|pranam|parnam"
    r"|sat\s*sri\s*akal|sasriakal|salaam|salam|assalam\w*|walaikum\w*|aadab|adaab"
    r"|vanakkam|vanakam|namaskara\w*|nomoskar\w*|khoda\s*hafiz|kem\s*cho|ki\s*khobor"
    r"|shubh\s*(prabhat|ratri)|subh\s*(prabhat|ratri)|alvida|phir\s*milenge"
    r"|dhanyawad|dhanyavaad|shukria|meherbani|bahut\s*bahut\s*(dhanyavad|shukriya)"
    # Native scripts. Kept as literal words rather than script ranges: a
    # bare "any Devanagari text" rule would swallow real Hindi questions,
    # which is the exact failure mode this pattern's whole-message
    # anchoring exists to prevent.
    r"|नमस्ते|नमस्कार|नमस्कारम्?|प्रणाम|राम\s*राम|जय\s*हिंद|शुभ\s*(प्रभात|रात्रि)"
    r"|धन्यवाद|शुक्रिया|आभार|अलविदा|फिर\s*मिलेंगे|ठीक\s*है|अच्छा|हाँ|जी"
    r"|नमस्कार|धन्यवाद|नमस्ते"                                   # Marathi (shares Devanagari)
    r"|નમસ્તે|નમસ્કાર|કેમ\s*છો|આભાર|ધન્યવાદ"                      # Gujarati
    r"|নমস্কার|নমস্তে|প্রণাম|ধন্যবাদ|কেমন\s*আছেন|আদাব"             # Bengali / Assamese
    r"|ਸਤਿ\s*ਸ੍ਰੀ\s*ਅਕਾਲ|ਨਮਸਕਾਰ|ਧੰਨਵਾਦ|ਸ਼ੁਕਰੀਆ"                    # Punjabi
    r"|ନମସ୍କାର|ଧନ୍ୟବାଦ|ପ୍ରଣାମ"                                    # Odia
    r"|வணக்கம்|நன்றி|நல்வரவு"                                      # Tamil
    r"|నమస్కారం|నమస్తే|ధన్యవాదాలు|ధన్యవాదములు"                     # Telugu
    r"|ನಮಸ್ಕಾರ|ನಮಸ್ತೆ|ಧನ್ಯವಾದ\w*"                                  # Kannada
    r"|നമസ്കാരം|നന്ദി"                                            # Malayalam
    r"|السلام\s*علیکم|وعلیکم\s*السلام|آداب|شکریہ|خدا\s*حافظ|بہت\s*شکریہ"  # Urdu
    r")"
)
# `।` (danda) and `॥` are sentence terminators in Devanagari/Bengali/Odia and
# `؟`/`۔` in Perso-Arabic. Without them here, "नमस्ते।" -- how the greeting is
# actually punctuated in those scripts -- failed to decompose even once the
# word itself was recognised.
_GREETING_PUNCTUATION = r"[,.!?।॥؟۔…]*"
_GENERAL_CONVERSATION_PATTERN = re.compile(
    rf"^(\s*{_GREETING_PUNCTUATION}\s*{_GREETING_UNIT}\s*{_GREETING_PUNCTUATION}\s*)+$",
    re.IGNORECASE,
)

# A question about what THIS ASSISTANT can do, as opposed to a question about
# the law. Recognised by the co-occurrence of three things -- an interrogative,
# a second-person reference, and an ability/help word -- rather than by an
# enumerated phrase list, because the Hinglish phrasings vary far too much for
# a fixed list ("tum kya kar sakte ho", "aap kaise madad karte hain", "mujhe
# batao ki tum meri kin legal problems me madad kar sakte ho"). Requiring all
# three together is what keeps it narrow: "kya main FIR file kar sakta hoon"
# (about the USER) has no second-person reference, and "kya aap mujhe process
# bata sakte hain" has no ability/help word, so neither matches.
_CAPABILITY_INTERROGATIVE_PATTERN = re.compile(
    r"\b(what|which|how|who)\b|\b(kya|kaise|kaisi|kaun\s*sa|kaun\s*si|kaun\s*se|konsa|konsi|kin|kis|kitn\w+)\b"
    r"|क्या|कैसे|कौन|किन|किस",
    re.IGNORECASE,
)
_CAPABILITY_SECOND_PERSON_PATTERN = re.compile(
    r"\b(you|your|yours|tum|tumhe|tumhein|tumhara|tumhari|tumhare|aap|aapka|aapki|aapke|aapko)\b"
    r"|तुम|आप",
    re.IGNORECASE,
)
_CAPABILITY_ABILITY_PATTERN = re.compile(
    r"\b(do|help|assist|offer|support|capable|capabilit\w+|feature|features|service|services|use|uses|useful)\b"
    r"|\b(kar\s+sakt\w+|karte\s+ho|karti\s+ho|karte\s+hain|madad|sahayata|kaam\s+aat\w+|help\s+kar\w*)\b"
    r"|कर\s*सकत|मदद|सहायता|सेवा",
    re.IGNORECASE,
)
# A message carrying all three capability cues is nonetheless a real legal
# question when it also carries a conditional clause ("... agar landlord
# notice hi na de?") or names a concrete legal subject. Confirmed against the
# reported list: "Aap kya kar sakte hain agar landlord notice na de?" was
# classified "Capability Question" and answered with the product overview
# instead of the tenancy question actually asked.
#
# Deliberately a list of SUBJECTS, not of legal-sounding words: "legal" and
# "kaam" are exactly what a genuine capability question uses ("aap kon kon se
# legal kaam kar sakte ho"), so neither may disqualify one.
_CAPABILITY_DISQUALIFIER_PATTERN = re.compile(
    r"\b(agar|if|jab|when|unless|suppose|in\s+case)\b|अगर|यदि|जब"
    r"|\b(fir|f\.i\.r|police|thana|cheque|check|notice|landlord|tenant|kirayedar|makan\s*malik"
    r"|court|adalat|bail|zamanat|divorce|talaq|rent|kiraya|salary|tankhwah|contract"
    r"|property|jaydad|arrest|giraftar|warrant|summons|complaint|shikayat|gst|tax|fraud"
    r"|theft|chori|dowry|dahej|maintenance|custody|eviction|refund|insurance)\b"
    r"|पुलिस|अदालत|ज़मानत|जमानत|तलाक|किराया|मकान\s*मालिक|संपत्ति|गिरफ़्तार|शिकायत",
    re.IGNORECASE,
)

# Above this many words a message that happens to contain all three cues is
# far more likely to be a real legal question that merely addresses the
# assistant ("aap kya kar sakte hain agar landlord notice hi na de aur ...")
# than a plain "what are you for?". Real capability questions are short.
_MAX_WORDS_FOR_CAPABILITY_QUESTION = 15

_TOKEN_PATTERN = re.compile(r"[\w']+")
_STOPWORDS = {
    "what", "about", "that", "this", "it", "is", "are", "the", "a", "an", "of", "to", "and", "or", "in", "on",
}


@dataclass(frozen=True)
class ConversationIntentMatch:
    intent: str
    confidence: float
    ambiguous: bool
    reason: str
    resolved_translation_target: str | None = None
    # Set only on "Language Preference": the language the user asked all
    # future replies to be in. `ChatService` persists it and answers with the
    # fixed acknowledgement, without retrieval or an answer-generation call.
    resolved_language_preference: str | None = None
    # Set only on "Typo Clarification": the short "did you mean X or Y?" the
    # normalizer produced, so the handler never has to re-derive it.
    clarification_message: str | None = None
    modification_type: str | None = None
    detected_intents: tuple[str, ...] = ()
    is_correction: bool = False
    corrected_text: str | None = None
    # "deterministic" for every rule-based match (including the workflow-chain
    # augmentation on the Draft Generation branch below, which populates
    # `detected_intents` without an LLM call) -- only the genuine
    # `classify_advanced` LLM-success path sets this to "llm". Kept explicit
    # rather than inferred from `detected_intents` being non-empty, which
    # stopped being a reliable signal once a deterministic branch could
    # populate it too.
    classifier_source: str = "deterministic"
    # Phase 1 "Intent Feedback": set when this message is an explicit
    # post-hoc correction of the PREVIOUS turn's intent classification
    # ("Wrong intent", "I wanted legal research") -- `feedback_corrected_
    # intent` is the resolved `CONVERSATION_INTENTS` value when the user
    # named one, `None` for a bare "wrong intent" with nothing specific
    # named.
    is_intent_feedback: bool = False
    feedback_corrected_intent: str | None = None


class ConversationIntentClassifier:
    """Pure `(text, memory) -> match` rule-based classifier, no I/O or LLM calls.

    Same style as `IntentDetector`/`DraftIntentDetector`: deterministic,
    fast, testable. Delegates the Draft Generation check entirely to
    `DraftIntentDetector` rather than reimplementing its guard logic.
    """

    def __init__(self, draft_detector: DraftIntentDetector | None = None) -> None:
        self.draft_detector = draft_detector or DraftIntentDetector()
        self.normalizer = QueryNormalizer()

    def classify(self, text: str, memory: dict[str, Any], language: str = "english") -> ConversationIntentMatch:
        # Every rule below reads the typo-normalized view, never the raw
        # message: "um mere liye kya-kya kar sakte ho?" (one missing "t")
        # otherwise fails the second-person cue and falls through to the
        # "General Legal Information" catch-all, which sends a question about
        # the assistant to retrieval and answers it with the strict-RAG
        # refusal. `normalize_for_routing` corrects only against a curated
        # allowlist and returns the text untouched for anything it is not
        # sure about (see `app/language/typo_tolerance.py`); the ORIGINAL
        # `text` is what `ChatService` persists to history and audit.
        raw_text = text
        routing = normalize_for_routing(text)
        text = routing.normalized_for_routing
        normalized = self.normalizer.normalize(text).lower()

        # A bare reply to the assistant's own "Which language would you like
        # this translated into?" clarifying question (e.g. just "Hindi.")
        # never contains the word "translate," so `_TRANSLATE_PATTERN` below
        # wouldn't catch it on its own -- `_respond_with_translation` marks
        # this pending state in memory right after asking the question, and
        # clears it after every other turn, so this only fires as a direct
        # answer to that specific prompt.
        if memory.get("pending_clarification") == "translation_target":
            target = self._find_translation_target(normalized)
            tokens = _TOKEN_PATTERN.findall(text)
            if target and len(tokens) <= 3:
                return ConversationIntentMatch(
                    "Translation", 0.85, False, "resolved pending translation-target reply", resolved_translation_target=target
                )

        last_failed_question = memory.get("last_failed_question")
        if last_failed_question and _RETRY_FAILED_REQUEST_PATTERN.match(normalized.strip()):
            return ConversationIntentMatch(
                "Retry Failed Request", 0.95, False, "retry command with a pending failed request"
            )
        if last_failed_question and _looks_like_resend_of_failed_question(raw_text, last_failed_question):
            return ConversationIntentMatch(
                "Retry Failed Request", 0.9, False,
                "message resends the previously failed question verbatim -- treated as an equivalent retry",
            )

        is_feedback, corrected_intent = self._extract_intent_feedback(normalized)
        if is_feedback:
            return ConversationIntentMatch(
                "Intent Feedback", 0.9, False,
                f"explicit intent-correction feedback{f' naming {corrected_intent!r}' if corrected_intent else ''}",
                is_intent_feedback=True, feedback_corrected_intent=corrected_intent,
            )

        if _LAWYER_RECOMMENDATION_PATTERN.search(normalized):
            return ConversationIntentMatch("Lawyer Recommendation", 0.9, False, "explicit lawyer/advocate ask")

        # Checked before `DraftIntentDetector` below: a bare command like
        # "Make it shorter" contains one of the detector's drafting verbs
        # ("make") with no template trigger phrase alongside it, which the
        # detector treats as an ambiguous drafting attempt on its own (see
        # `DraftIntentDetector.detect`'s fallback `ambiguous=True` branch).
        # None of the ~80 anchored command phrases below textually equal any
        # genuine drafting request ("Generate FIR draft.", "Write legal
        # notice.", etc.), so moving this check earlier only reroutes the
        # handful of phrases that would otherwise be misread as drafting.
        modification_type = self._match_modification(normalized)
        if modification_type is not None:
            has_prior_reply = self._has_prior_assistant_turn(memory)
            # "Simple Hindi"/"Simple Marathi" etc. under `simplify` double as a
            # language request -- reuse `resolved_translation_target` so the
            # response-modification handler can ask the LLM to render the
            # simplified text in that language too, without a separate field.
            target = self._find_translation_target(normalized) if modification_type == "simplify" else None
            return ConversationIntentMatch(
                "Response Modification",
                0.8,
                not has_prior_reply,
                "response modification command" if has_prior_reply else "response modification command, no prior reply",
                resolved_translation_target=target,
                modification_type=modification_type,
            )

        # Conversation control, checked before any legal-intent rule below
        # and therefore long before the "General Legal Information" fallback
        # that used to swallow it. A turn that is nothing but "reply in
        # <language> from now on" has no answer in any statute -- it must
        # never reach the retriever, the reranker or the answer LLM.
        # `is_language_preference_command` matches by subtraction (take the
        # language cue out; a preference command is one with nothing left but
        # harmless modifiers), so "Hindi mein FIR kaise file karein?" and "I
        # found a lawyer who speaks Hindi" are deliberately NOT preference
        # commands and keep their existing routing.
        preferred_language = is_language_preference_command(text)
        if preferred_language:
            return ConversationIntentMatch(
                "Language Preference", 0.9, False,
                f"standalone request to reply in {preferred_language}",
                resolved_language_preference=preferred_language,
            )

        # Resolve an explicit drafting command before broad conversation-
        # memory patterns.  Q27's "In facts par ... notice draft karo" was
        # previously swallowed as a request to recall facts, so the drafting
        # state was never created and every following turn lost continuity.
        draft_match = self.draft_detector.detect(text)

        stripped_for_memory = normalized.strip()
        if (
            not draft_match.matched
            and (
                _CONVERSATION_MEMORY_PATTERN.match(stripped_for_memory)
                or _CONVERSATION_MEMORY_FACT_RECALL_PATTERN.match(stripped_for_memory)
                or _CONVERSATION_MEMORY_HINGLISH_PATTERN.search(stripped_for_memory)
            )
        ):
            has_prior_reply = self._has_prior_assistant_turn(memory)
            return ConversationIntentMatch(
                "Conversation Memory",
                0.8,
                not has_prior_reply,
                "conversation-memory recall question" if has_prior_reply else "conversation-memory question, nothing discussed yet",
            )

        if draft_match.matched:
            references_document = bool(
                _DOCUMENT_ANALYSIS_VERB_PATTERN.search(normalized) and _DOCUMENT_ANALYSIS_NOUN_PATTERN.search(normalized)
            )
            if references_document and not _WORKFLOW_LINK_PATTERN.search(normalized):
                # General Clarification Mode: a message that both names a
                # draft template/verb AND references analyzing/reviewing a
                # document, but with NO explicit "based on it/that" style
                # link, is genuinely ambiguous -- "review the PDF and draft
                # a notice" could mean "draft grounded in the review" or
                # "do these two separate things." Ask, rather than silently
                # picking one; `ChatService` stores this as
                # `pending_clarification="workflow_intent"` and resolves it
                # on the next turn (mirrors the existing
                # `pending_clarification="translation_target"` pattern
                # above).
                return ConversationIntentMatch(
                    "Workflow Clarification", 0.6, False,
                    "ambiguous document-analysis + draft-generation phrasing, no linking phrase",
                    detected_intents=("Document Analysis", "Draft Generation"),
                )
            workflow_intents: tuple[str, ...] = ("Document Analysis", "Draft Generation") if references_document else ()
            # Multi-Intent Workflow Orchestration: an unambiguous chain
            # request ("review this PDF, flag risky clauses, draft a notice
            # BASED ON IT") is the flagship Document-Analysis-then-Draft-
            # Generation chain -- deliberately NOT gated on
            # `memory["last_uploaded_document_id"]` already being set:
            # whether a document actually exists to analyze is the
            # orchestrator's own step-1 check
            # (`ChatService._run_document_analysis_then_draft`), which asks
            # the user for one rather than silently falling back to plain
            # drafting when it's missing. Surfaced via `detected_intents`
            # only -- `intent` itself stays "Draft Generation" with its own
            # unchanged confidence/ambiguity, so every existing
            # draft-routing test/behavior is unaffected -- purely
            # deterministic (no LLM call) so
            # `workflow_orchestrator.detect_chain` can recognize the chain
            # without waiting on one.
            return ConversationIntentMatch(
                "Draft Generation", draft_match.confidence, draft_match.ambiguous, "draft verb/template match",
                detected_intents=workflow_intents,
            )

        # Checked AFTER the drafting detector on purpose: "kya aap police
        # complaint draft kar sakte hain?" carries capability cues but names a
        # document and a drafting verb, and the user is better served by the
        # drafting flow starting than by being told drafting exists. Only a
        # message with no drafting request left in it reaches here.
        if self._is_capability_question(normalized):
            return ConversationIntentMatch(
                "Capability Question", 0.85, False, "question about the assistant's own capabilities"
            )

        # Part 51 "Uploaded Document Conversation Context": checked before
        # `_CONVERSATION_SUMMARY_PATTERN`/`_SUMMARIZE_PATTERN` below -- a
        # pronoun/short reference ("explain this," "isme kya hai," "isko
        # explain kro," "ye pdf samjhao," "summary batao," "key points
        # batao") can't be told apart from an ordinary informational
        # question (or, for "summary batao" specifically, a whole-
        # conversation summary request) by keyword pattern alone -- "this"/
        # "isme"/"batao" name nothing on their own. Gated on
        # `memory["last_uploaded_document_id"]` actually being set (this
        # classifier already receives `memory`) so it can only ever fire in
        # a session that has genuinely uploaded a document -- the exact same
        # phrase in a session with no upload at all still falls through to
        # its normal routing below, unaffected (verified: "summary batao"
        # with no upload still means "summarize our chat").
        if memory.get("last_uploaded_document_id") and _DOCUMENT_PRONOUN_REFERENCE_PATTERN.search(normalized):
            return ConversationIntentMatch(
                "Document Analysis", 0.75, False, "pronoun reference to the last uploaded document"
            )

        if _CONVERSATION_SUMMARY_PATTERN.search(normalized):
            return ConversationIntentMatch("Summarization", 0.8, False, "whole-conversation summary request")

        if _TRANSLATE_PATTERN.search(normalized):
            target = self._find_translation_target(normalized)
            has_prior_reply = self._has_prior_assistant_turn(memory)
            ambiguous = target is None or not has_prior_reply
            reason = "translate keyword" if target else "translate keyword, no target language named"
            return ConversationIntentMatch("Translation", 0.8, ambiguous, reason, resolved_translation_target=target)

        if _SUMMARIZE_PATTERN.search(normalized):
            return ConversationIntentMatch("Summarization", 0.8, False, "summarize keyword")

        if _DICTIONARY_PATTERN.search(normalized):
            return ConversationIntentMatch("Legal Dictionary", 0.7, False, "define/meaning-of pattern")

        if _COMPARISON_PATTERN.search(normalized):
            return ConversationIntentMatch("Law Comparison", 0.75, False, "vs/difference/compare pattern")

        if _PROCEDURE_PATTERN.search(normalized):
            return ConversationIntentMatch("Legal Procedure", 0.7, False, "process/steps/how-to pattern")

        if _DOCUMENT_ANALYSIS_VERB_PATTERN.search(normalized) and _DOCUMENT_ANALYSIS_NOUN_PATTERN.search(normalized):
            return ConversationIntentMatch("Document Analysis", 0.7, False, "analyze/review pattern")

        # Part 42: checked before `_is_followup`/the bare-term/short-message
        # fallbacks below -- an off-domain or non-Indian-jurisdiction message
        # must never be swallowed into stale legal context or forced into
        # "Follow-up Question" just because it's short and a prior assistant
        # turn exists.
        if _OFF_DOMAIN_PATTERN.search(normalized):
            return ConversationIntentMatch("Out of Domain", 0.8, False, "off-domain topic, outside legal scope")

        if (
            _NON_INDIAN_JURISDICTION_PATTERN.search(normalized)
            and not _INDIA_PATTERN.search(normalized)
            and (
                _STANDALONE_LEGAL_SIGNAL_PATTERN.search(normalized)
                or _PROCEDURE_PATTERN.search(normalized)
                or _COMPARISON_PATTERN.search(normalized)
            )
        ):
            return ConversationIntentMatch(
                "Non-Indian Jurisdiction", 0.75, False, "legal question about a non-Indian jurisdiction"
            )

        if self._is_followup(text, memory):
            return ConversationIntentMatch("Follow-up Question", 0.6, False, "pronoun continuation with prior turn")

        if _GENERAL_CONVERSATION_PATTERN.match(normalized.strip()):
            return ConversationIntentMatch("General Conversation", 0.85, False, "greeting/pleasantry only")

        if _EXPLANATION_PATTERN.search(normalized) or _HINDI_DEFINITION_PATTERN.search(normalized):
            return ConversationIntentMatch("Legal Explanation", 0.55, False, "what-is/explain pattern")

        # Checked before the bare-term-lookup fallback below: a short
        # first-person sentence ("My cheque bounced.") is a complete clause
        # describing a problem, not a bare topic lookup ("cheque bounce") --
        # but at 3 tokens it would otherwise satisfy _is_bare_term_lookup's
        # word-count check and get misread as one.
        if _LEGAL_ADVICE_PATTERN.search(normalized):
            return ConversationIntentMatch("Legal Advice", 0.55, False, "first-person problem statement")

        if self._is_bare_term_lookup(text):
            has_prior_reply = self._has_prior_assistant_turn(memory)
            # A bare term that names its own legal topic ("Bail chahiye",
            # "FIR") is a self-contained new question -- never assume it
            # continues whatever was discussed before just because it's
            # short. Checked before the prior-reply branch below so it
            # applies whether or not there's a prior turn.
            if has_prior_reply and _STRONG_STANDALONE_LEGAL_SIGNAL_PATTERN.search(normalized):
                return ConversationIntentMatch(
                    "Legal Dictionary", 0.6, False, "bare short term naming its own legal topic"
                )
            if has_prior_reply:
                # A bare language name with nothing else, once a real reply
                # already exists, overwhelmingly means "translate that into
                # this language" -- real usage skips straight to naming the
                # language instead of saying "Translate." first (verified
                # against a real conversation trace: "What is FIR?" -> "30
                # words" -> "Hindi" clearly meant "translate the shortened
                # answer," not "define the word Hindi"). Supersedes the
                # older, more cautious design (still correct in spirit: a
                # bare word alone is ambiguous) now that there's concrete
                # evidence of what it's ambiguous BETWEEN, not whether to
                # guess at all.
                implicit_target = self._find_translation_target(normalized)
                if implicit_target:
                    return ConversationIntentMatch(
                        "Translation", 0.75, False, "bare language name with prior reply",
                        resolved_translation_target=implicit_target,
                    )
                # Any other short, standalone fragment ("Example", "NCR")
                # with nothing else, once there's a prior turn to resolve
                # against, almost always means "apply that to what we were
                # just discussing" -- not "define this word in isolation."
                return ConversationIntentMatch("Follow-up Question", 0.6, False, "short fragment with prior turn")
            return ConversationIntentMatch("Legal Dictionary", 0.5, False, "bare short term, no verb")

        # Same reasoning as the bare-term-lookup branch above, for slightly
        # longer fragments that aren't literally <=3 tokens ("What should I
        # do" after "My bike was stolen" -- 4 tokens, no first-person
        # pronoun pattern to catch it as Legal Advice, no modal-question
        # pattern either): once there's a prior turn, a short message with
        # no stronger match anywhere above is far more likely to be
        # continuing that thread than a genuinely new, self-contained
        # question. Capped at 5 tokens (not more) specifically because
        # "What was the maximum bail amount?" (6 tokens) is a real existing
        # test case for a genuinely new, self-contained question that must
        # stay "General Legal Information" -- a cap that swallowed it too
        # would trade one real bug for reintroducing a previously-fixed one.
        if (
            self._has_prior_assistant_turn(memory)
            and len(_TOKEN_PATTERN.findall(text)) <= 5
            and not _STRONG_STANDALONE_LEGAL_SIGNAL_PATTERN.search(normalized)
        ):
            return ConversationIntentMatch("Follow-up Question", 0.55, False, "short message with prior turn")

        # Nothing above matched, and the only reason may be a token the
        # normalizer could not read with confidence. If its candidates are all
        # conversational-control words, the honest move is to ask which was
        # meant rather than to guess one and route on it -- and asking is
        # strictly better than the previous outcome, which was to send an
        # unreadable message to retrieval and refuse it. Scoped to control
        # vocabulary and to short messages on purpose: an ambiguity among
        # LEGAL terms is left alone, because retrieval still sees the user's
        # original spelling and can resolve it without an interruption.
        # BUG-110: the correction allowlist this draws from (`_CONTROL_TERMS`
        # in `app/language/typo_tolerance.py`) is English plus romanized
        # Hindi -- it has no vocabulary for any other language. Offering it
        # against a message DETECTED in some other language (e.g. romanized
        # Manipuri "kari" fuzzy-matching Hindi "kar"/"karo"/"karti") produces
        # a nonsensical prompt in the wrong language's vocabulary; better to
        # fall through to the ordinary "General Legal Information" path
        # below, which lets retrieval/GK-fallback give an honest answer (or
        # an honest "no verified answer") instead.
        clarification = (
            clarification_question(routing)
            if language in ("english", "hindi", "hinglish")
            else None
        )
        if clarification is not None and len(_TOKEN_PATTERN.findall(text)) <= 8:
            return ConversationIntentMatch(
                "Typo Clarification", 0.6, False,
                "unreadable control token with more than one equally-good correction",
                clarification_message=clarification,
            )

        return ConversationIntentMatch("General Legal Information", 0.4, False, "fallback")

    async def classify_advanced(
        self, text: str, memory: dict[str, Any], llm: LLMProvider, language: str = "english",
    ) -> ConversationIntentMatch:
        is_correction, corrected_text = self._extract_correction(text)
        deterministic = self.classify(corrected_text or text, memory, language)
        if is_correction:
            deterministic = replace(deterministic, is_correction=True, corrected_text=corrected_text)
        if not settings.conversation_intent_llm_enabled:
            return deterministic
        protected_intents = {
            "Draft Generation", "Translation", "Response Modification", "Conversation Memory",
            "Retry Failed Request", "General Conversation", "Out of Domain", "Non-Indian Jurisdiction",
            # Deterministic, and the answer is a fixed description of this
            # product -- there is nothing for an LLM reclassification to
            # improve, and a downgrade to a RAG intent reintroduces exactly
            # the "no verified document" refusal this branch exists to stop.
            "Capability Question",
            # Both are conversation control resolved deterministically, with
            # fixed reviewed text as the answer -- there is nothing for an
            # LLM reclassification to improve, and a downgrade to a RAG
            # intent reintroduces exactly the refusal these exist to stop.
            "Language Preference",
            "Typo Clarification",
            # Deliberately below `conversation_intent_llm_min_confidence`
            # (0.6 < the 0.70 default) -- without protection, a genuinely
            # ambiguous case would proceed to the LLM and could get silently
            # resolved to a guess, exactly what asking was meant to avoid.
            "Workflow Clarification",
            # Explicit intent-correction feedback (0.9 confidence) -- an
            # LLM reclassification of the FEEDBACK message itself would be
            # nonsensical (it isn't a legal question), so protected for
            # clarity even though 0.9 already clears the confidence gate.
            "Intent Feedback",
        }
        if deterministic.intent in protected_intents and not deterministic.ambiguous:
            return deterministic
        if not deterministic.ambiguous and deterministic.confidence >= settings.conversation_intent_llm_min_confidence:
            return deterministic
        recent_messages = memory.get("messages", [])[-settings.conversation_intent_llm_max_history:]
        context = "\n".join(
            f"{message.get('role', 'unknown')}: {str(message.get('content', ''))[:1200]}"
            for message in recent_messages
        )
        try:
            prompt = prompt_registry.render(
                "conversation_intent_classification_prompt",
                intents=json.dumps(CONVERSATION_INTENTS),
                question=(corrected_text or text)[:8000],
                context=context,
                current_intent=deterministic.intent,
            )
            response = await llm.chat([ChatMessage(role="user", content=prompt)], temperature=0.0)
            if response.error or not response.content.strip():
                return deterministic
            content = response.content.strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
            payload = json.loads(content)
            candidates = payload.get("intents")
            primary = payload.get("primary_intent")
            if not isinstance(candidates, list) or not isinstance(primary, str):
                return deterministic
            valid: list[tuple[str, float, str]] = []
            for item in candidates:
                if not isinstance(item, dict) or item.get("intent") not in CONVERSATION_INTENTS:
                    continue
                try:
                    confidence = float(item.get("confidence", 0))
                except (TypeError, ValueError):
                    continue
                if 0 <= confidence <= 1:
                    valid.append((item["intent"], confidence, str(item.get("reason", "LLM classification"))))
            if primary not in {intent for intent, _, _ in valid}:
                return deterministic
            selected = next(item for item in valid if item[0] == primary)
            if selected[1] < settings.conversation_intent_llm_min_confidence:
                return deterministic
            detected = tuple(dict.fromkeys(item[0] for item in sorted(valid, key=lambda value: value[1], reverse=True)))
            return ConversationIntentMatch(
                primary, selected[1], False, selected[2],
                resolved_translation_target=deterministic.resolved_translation_target,
                modification_type=deterministic.modification_type,
                detected_intents=detected,
                is_correction=is_correction,
                corrected_text=corrected_text,
                classifier_source="llm",
            )
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            return deterministic
        except Exception:  # noqa: BLE001 - classifier must preserve deterministic routing on provider failure
            return deterministic

    def _extract_correction(self, text: str) -> tuple[bool, str | None]:
        """Detect a user correcting their own prior request.

        Returns `(is_correction, corrected_text)`. `corrected_text` is the
        replacement request to actually route (retrieval/entity
        extraction/answer generation must use this, never the original
        wrong message) -- it's `None` when the user only signals that the
        previous turn was misunderstood without restating what they want
        (e.g. "you misunderstood"), since there's nothing new to extract.
        """
        stripped = text.strip()
        match = re.search(
            r"(?:no|nah|nahi)[,\s]+(?:i\s+)?(?:meant|mean|mera\s+matlab|actually)\s*[:,-]?\s*(.+)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if match and match.group(1).strip():
            return True, match.group(1).strip()
        # Hinglish "X nahi, Y" correction pattern, e.g. "Document analysis
        # nahi, legal research chahiye" -- the part after "nahi" is the
        # actual request; the part before is what the user is rejecting.
        hinglish_match = re.search(
            r"^(?P<wrong>[^,]+?)\s+nahi\s*,\s*(?P<right>.+)$", stripped, flags=re.IGNORECASE,
        )
        if hinglish_match and len(hinglish_match.group("right").split()) >= 2:
            return True, hinglish_match.group("right").strip()
        if re.search(
            r"you\s+misunderstood|that'?s\s+not\s+what\s+i\s+(?:meant|asked)|you\s+got\s+it\s+wrong",
            stripped,
            flags=re.IGNORECASE,
        ):
            return True, None
        return False, None

    def _extract_intent_feedback(self, normalized: str) -> tuple[bool, str | None]:
        """Detect explicit post-hoc feedback about the PREVIOUS turn's
        intent classification ("Wrong intent", "I wanted legal research",
        "This is document analysis").

        Returns `(is_intent_feedback, corrected_intent)` -- `corrected_
        intent` is the resolved `CONVERSATION_INTENTS` value when a
        specific category was named (via `_INTENT_FEEDBACK_ALIASES`),
        `None` for a bare "wrong intent" or for named text that doesn't
        match any known category (never guessed/invented -- e.g. "I wanted
        a refund" stays ordinary text, not feedback). The named-text match
        is capped at 5 words specifically so this doesn't swallow a
        genuine new question merely phrased as "I wanted X" ("I wanted
        legal research on cheque bounce notices" is a real question, not
        feedback about a prior turn).
        """
        if _INTENT_FEEDBACK_GENERIC_PATTERN.search(normalized):
            return True, None
        named_match = _INTENT_FEEDBACK_NAMED_PATTERN.search(normalized)
        if named_match:
            named_text = named_match.group(1).strip().rstrip(".!?")
            if len(named_text.split()) <= 5:
                for phrase, intent in _INTENT_FEEDBACK_ALIASES.items():
                    if phrase in named_text:
                        return True, intent
        return False, None

    def _match_modification(self, normalized: str) -> str | None:
        stripped = normalized.strip()
        if _MODIFICATION_WORD_COUNT_PATTERN.match(stripped):
            return "shorten"
        for modification_type, pattern in _MODIFICATION_PATTERNS.items():
            if pattern.match(stripped):
                return modification_type
        return None

    def _find_translation_target(self, normalized: str) -> str | None:
        # Was a per-language substring regex over `SUPPORTED_LANGUAGES` --
        # exact spelling only, and script names ("Devanagari") never
        # matched at all since they aren't language names. Confirmed live:
        # a bare one-word reply to "which language would you like this
        # translated into?" ("kasmiri" -- a real, common misspelling of
        # "kashmiri"; "Devanagari") returned `None` here, so the pending
        # translation-target clarification (see the `pending_clarification
        # == "translation_target"` branch above) was silently dropped and
        # the reply was routed as a brand-new question instead of
        # completing the translation. `resolve_language_token` gives the
        # same typo tolerance and native-script/exonym coverage
        # `extract_requested_language` already relies on, token by token.
        for token in _TOKEN_PATTERN.findall(normalized):
            language = resolve_language_token(token.lower())
            if language and language != "hinglish":
                return language
        return None

    def _has_prior_assistant_turn(self, memory: dict[str, Any]) -> bool:
        # `memory["messages"]` already includes the just-appended current
        # user turn by the time the classifier runs (chat_service appends
        # before classifying) -- exclude it, or every first turn would
        # false-match as a follow-up/have-a-prior-reply.
        messages = memory.get("messages") or []
        return any(message.get("role") == "assistant" for message in messages[:-1])

    def _is_followup(self, text: str, memory: dict[str, Any]) -> bool:
        if not self._has_prior_assistant_turn(memory):
            return False
        return bool(_FOLLOWUP_PATTERN.search(text))

    def _is_capability_question(self, normalized: str) -> bool:
        """True when the message asks what this assistant itself can do.

        All three cues must be present (see the pattern definitions above),
        and the message must be short -- a long message that happens to
        contain all three is a legal question addressed to the assistant, not
        a question about it.
        """
        if len(_TOKEN_PATTERN.findall(normalized)) > _MAX_WORDS_FOR_CAPABILITY_QUESTION:
            return False
        if _CAPABILITY_DISQUALIFIER_PATTERN.search(normalized):
            return False
        return bool(
            _CAPABILITY_INTERROGATIVE_PATTERN.search(normalized)
            and _CAPABILITY_SECOND_PERSON_PATTERN.search(normalized)
            and _CAPABILITY_ABILITY_PATTERN.search(normalized)
        )

    def _is_bare_term_lookup(self, text: str) -> bool:
        tokens = _TOKEN_PATTERN.findall(text)
        if not tokens or len(tokens) > 3:
            return False
        return not all(token.lower() in _STOPWORDS for token in tokens)
