"""RAG relevance-gate: which retrieved chunks are actually on-topic.

Extracted out of `app.services.chat_service.ChatService` (Phase 1 of the
god-object split) -- these functions have zero dependency on chat state,
they take chunks/queries in and return filtered chunks/booleans out.
`ChatService` keeps its original private method names as one-line
delegators to these, since several tests call those methods directly on a
`ChatService` instance.
"""
import re

import structlog

from app.core.config import settings
from app.intent.detector import parse_section_lookup
from app.schemas.common import RetrievedChunk

log = structlog.get_logger(__name__)

# Part 41 relevance gate (`_fallback_answer`/`_most_relevant_chunk`): common
# English and Hindi/Hinglish function words excluded from the lexical-overlap
# check so two unrelated texts that both happen to contain "hai"/"kya"/"the"
# don't register as a real topical match.
_FALLBACK_STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "do", "does", "did",
    "kya", "hai", "hain", "hota", "hoti", "hote", "ka", "ki", "ke", "ko", "se", "mein", "hi", "accha", "aur",
    "tha", "thi", "hua", "hui", "legal", "law", "act", "section", "rights", "general", "information",
    # Bureaucratic-process nouns generic across nearly every Indian statute
    # (an "application" gets "file"d and a "process" happens under RTI, GST,
    # any Companies-Act filing, a Motor Vehicles licence, a passport renewal
    # -- the word carries no topic-discriminating power at all). Confirmed
    # live: a GST "removal of difficulty" administrative notice that merely
    # repeats "file application for revocation" twice in passing outscored
    # the actual Right to Information Act's own chunks for "RTI application
    # kaise file karte hain?" purely on this overlap, before either word was
    # excluded -- this significant-terms check is a low-score FALLBACK
    # signal meant to catch genuine topical overlap on DISTINCTIVE terms,
    # not a magnet for whichever chunk happens to reuse the most generic
    # procedural vocabulary. "kaise"/"karte"/"karna" (Hindi "how"/"do") are
    # the same kind of empty procedural filler as "kya"/"hai" already above.
    "application", "applications", "file", "filing", "filed", "process", "kaise", "karte", "karna", "karein",
    # Confirmed live: "which section talks about culpable homiside" shared
    # NOTHING topical with an unrelated Maharashtra GST Act chunk, yet
    # `is_relevant_chunk`'s generic fallback accepted it as a genuine
    # "source" purely because both texts happened to contain "which" and
    # "about" -- ordinary English function/question words that appear in
    # nearly every sentence of statutory text, carrying exactly as little
    # topic-discriminating power as "what"/"is"/"the" already excluded
    # above. This was the actual mechanism behind irrelevant sources
    # (unrelated State Acts) being displayed alongside otherwise-correct
    # answers -- not a retrieval-ranking gap, a stopword-list gap in the
    # gate meant to catch exactly this.
    "which", "who", "whom", "whose", "when", "where", "why", "how",
    "about", "talks", "talk", "talking", "this", "that", "these", "those",
    "with", "from", "by", "as", "it", "its", "can", "could", "will", "would",
    "should", "shall", "not", "no", "if", "than", "such", "any", "all",
    "each", "other", "there", "here", "also", "under", "into", "onto",
}
_MIN_ACCEPTED_CONTEXT_SCORE = 0.12
# Phase 1 item 1: the score a chunk must reach to be accepted for a
# non-English question on embedding strength alone, with no literal token
# overlap required. Higher than `_MIN_ACCEPTED_CONTEXT_SCORE` (which is paired
# with an overlap requirement) precisely because nothing else is corroborating
# it. Unchanged in value from the number this rule has always used -- named
# here only so the script-based and language-based branches provably share it.
_CROSS_LINGUAL_CONTEXT_SCORE = 0.24
# Word-boundary match for the FIR relevance-gate override below -- "fir" as a
# bare substring would also fire on "first", "firm", "confirm", etc.
_FIR_QUERY_RE = re.compile(r"\bfir\b", re.IGNORECASE)

# Offence names carry topic evidence; generic penal language (fine,
# imprisonment, punishment) occurs in unrelated regulatory Acts too.
# Inspect passage text, not metadata titles or query-derived tags.
_BNS_QUERY_RE = re.compile(r"\bbns\b|\bbharatiya\s+nyaya\s+sanhita\b", re.IGNORECASE)
_BNS_OFFENCE_TOPICS = (
    re.compile(r"\brape\b", re.IGNORECASE),
    re.compile(r"\b(?:culpable\s+homicide|murder)\b", re.IGNORECASE),
    re.compile(r"\b(?:theft|steal|steals|stealing|stolen)\b", re.IGNORECASE),
    re.compile(r"\bcheat(?:ing|s|ed)?\b|\bdishonest(?:ly)?\s+induc", re.IGNORECASE),
    re.compile(r"\bcriminal\s+breach\s+of\s+trust\b|\bmisappropriat", re.IGNORECASE),
    re.compile(r"\bforger(?:y|ies)\b|\bfalse\s+document\b", re.IGNORECASE),
    re.compile(r"\bcriminal\s+intimidation\b|\bthreat\w*\s+to\s+cause\s+injury\b", re.IGNORECASE),
    re.compile(r"\bextortion\b|\bfear\s+of\s+(?:any\s+)?injury\b", re.IGNORECASE),
    re.compile(r"\bdowry\b", re.IGNORECASE),
)


def significant_terms(text: str, allow_short_numeric: bool = False) -> set[str]:
    """`allow_short_numeric` defaults to `False`, leaving every existing caller
    (`_most_relevant_chunk`'s fallback-answer path, `_is_relevant_chunk`'s
    generic branch for every non-SECTION_LOOKUP intent) byte-for-byte
    unchanged for ordinary WORDS. Phase 4: a bare citation like "Section 2"
    names a section NUMBER, not a topic -- the length filter below exists to
    drop noise tokens ("a", "I") from ordinary topical questions, but it was
    silently dropping the one token that actually identifies a single-digit
    section (`len("2") == 1`), while "section" itself is already a stopword.
    Passed `True` only for SECTION_LOOKUP queries (see `_is_relevant_chunk`),
    never globally -- ordinary questions keep filtering out every 1-character
    token, numeric or not.

    ALL digit-only tokens, not just single-digit ones, are gated behind
    `allow_short_numeric` -- confirmed live: "bharatiya nyaya sanhita
    section 12" (intent NOT SECTION_LOOKUP -- this is a named-Act citation,
    a different code path in `LegalRetriever`) shared nothing topical with a
    completely unrelated Maharashtra Act's own "12. ..." section heading,
    but both texts contained the bare digits "12" -- `len("12") > 1` treated
    that coincidence as a genuine topical match regardless of
    `allow_short_numeric`, which only ever gated LENGTH-1 numbers. A
    2-or-more-digit section number is exactly as likely to coincidentally
    recur across unrelated Acts as a 1-digit one (173 and 154 both do,
    confirmed elsewhere in this corpus) -- there is no length at which a
    bare number stops being a citation-matching concern and starts being a
    genuine topic word, so every digit-only token needs the same gate.
    """
    terms: set[str] = set()
    for token in re.findall(r"[\w']+", (text or "").lower()):
        if token in _FALLBACK_STOPWORDS:
            continue
        if token.isdigit():
            if allow_short_numeric:
                terms.add(token)
            continue
        if len(token) > 1:
            terms.add(token)
    return terms


# QA pass 2026-09-24 (T074): mirrors `LegalRetriever._NAMED_SECTION_CITATION_
# PATTERNS`'s own act-name shape (a capitalized run ending in a common
# instrument suffix) -- see `_drop_named_instrument_terms`'s docstring for why
# a query that names an Act this way needs those specific words excluded from
# `most_relevant_chunk`'s overlap comparison.
_NAMED_INSTRUMENT_RE = re.compile(r"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+(?:Act|Sanhita|Adhiniyam|Code)\b")


def _drop_named_instrument_terms(query: str, terms: set[str]) -> set[str]:
    """An Act/Sanhita/Adhiniyam/Code's own name, when the query names it
    explicitly, is not a discriminating signal for WHICH of its own chunks is
    most on-topic: statutory prose states what a provision says, not what Act
    it belongs to, so a genuine chunk of that Act only rarely repeats the
    Act's own full name -- while a document that merely NAMES or cross-
    references the Act (a FAQ, a commentary, another Act citing it) can
    repeat that name freely, inflating its lexical-overlap score for a
    reason that has nothing to do with actually answering the question.

    Confirmed live (QA pass 2026-09-24, T074, `QA_REPORT_100Q_FINAL_
    20260924.md` section 7.4): after `LegalReranker` was fixed to correctly
    rank the Bharatiya Sakshya Adhiniyam's own section 63 (the electronic-
    evidence-admissibility provision) first for "admissibility of electronic
    evidence Bharatiya Sakshya Adhiniyam", `reorder_by_relevance` still
    overrode it back to a Delhi Police Academy FAQ document -- that FAQ
    chunk's own running page header ("The Bharatiya Sakshya Adhiniyam")
    alone gave it 6/6 significant-term overlap against the query, against
    the real section-63 chunk's 2/6 (that specific chunk's page happened to
    carry no header). Only strips the NAMED Act's own words; a query's other
    vocabulary, and any query that does not name an Act this way at all, is
    completely unaffected. Falls back to the unstripped set if stripping
    would leave nothing (the query consists of nothing but the Act's name),
    rather than making every candidate tie at zero overlap.
    """
    matches = _NAMED_INSTRUMENT_RE.findall(query or "")
    if not matches:
        return terms
    instrument_terms: set[str] = set()
    for match in matches:
        instrument_terms.update(re.findall(r"[a-z0-9]+", match.lower()))
    remaining = terms - instrument_terms
    return remaining or terms


def is_cheque_notice_timing_query(text: str) -> bool:
    lowered = (text or "").lower()
    cheque = any(term in lowered for term in ("cheque bounce", "check bounce", "dishonour", "dishonor"))
    notice = "notice" in lowered
    timing = any(
        term in lowered
        for term in ("kitne time", "kitne din", "kab", "when", "how long", "within", "deadline", "time limit")
    )
    return cheque and notice and timing


def is_ni_section_138(chunk: RetrievedChunk) -> bool:
    metadata = chunk.metadata or {}
    identity = " ".join(
        str(metadata.get(key) or "")
        for key in ("act_name", "source_document", "source")
    ).lower().replace("_", " ")
    return (
        str(metadata.get("section_number") or "").strip().rstrip(".") == "138"
        and "negotiable instruments" in identity
    )


def is_ni_notice_period_evidence(chunk: RetrievedChunk) -> bool:
    if not is_ni_section_138(chunk):
        return False
    lowered = chunk.text.lower()
    return (
        bool(re.search(r"\b(?:within\s+)?(?:thirty|30)\s+days?\b", lowered))
        and bool(re.search(r"\b(?:within\s+)?(?:fifteen|15)\s+days?\b", lowered))
        and "drawer" in lowered
        and "notice" in lowered
    )


def is_relevant_chunk(
    query: str, rewritten_query: str, chunk: RetrievedChunk,
    intent: str | None = None, language: str | None = None,
) -> bool:
    query_text = f"{query} {rewritten_query}".lower()
    chunk_text = f"{chunk.text} {chunk.metadata}".lower()
    # These topic patterns are English. Preserve multilingual semantic
    # matching for passages whose alphabet cannot be checked by this list.
    has_non_latin_text = any(char.isalpha() and ord(char) > 127 for char in chunk.text)
    if _BNS_QUERY_RE.search(query_text) and not has_non_latin_text:
        offence_topics = [topic for topic in _BNS_OFFENCE_TOPICS if topic.search(query_text)]
        if offence_topics and not any(topic.search(chunk.text) for topic in offence_topics):
            return False
    if is_cheque_notice_timing_query(query_text):
        # The notice-dispatch and drawer-payment periods are both in
        # NI Act s.138.  Same-Act sections 90/136/139 and same-numbered
        # provisions from GST/Social Security are not authority for this
        # question, however high their similarity score happens to be.
        return is_ni_notice_period_evidence(chunk)
    if chunk.score >= 0.6:
        return True
    # Phase 4: a SECTION_LOOKUP query names an exact provision NUMBER, not
    # a topic -- the topic/lexical-overlap checks below are the wrong test
    # for it (a bare "Section 2" carries no topic vocabulary to overlap
    # on at all, see `_significant_terms`). An exact match against the
    # chunk's own `section_number` metadata is the correct, narrow
    # acceptance rule: same number, no Act comparison (Phase 1 already
    # owns Act disambiguation), no score floor reintroduced, and scoped
    # to this intent only -- every other intent falls through unchanged.
    if intent == "SECTION_LOOKUP":
        query_section_number = parse_section_lookup(query.lower().strip())
        if query_section_number and chunk.metadata.get("section_number") == query_section_number:
            return True
    article_match = re.search(r"\barticle\s+(\d{1,3}[A-Z]?)\b", query, re.IGNORECASE)
    if article_match and str(
        chunk.metadata.get("article_number") or chunk.metadata.get("section_number") or ""
    ).upper() == article_match.group(1).upper():
        return True
    # The corpus is predominantly English, while the embedding model is
    # multilingual.  Requiring literal token overlap after a successful
    # semantic match rejects Hindi/other-script questions by design: the
    # query and the correct English provision share no characters.  Keep
    # the lexical guard for Latin-script questions, but allow a genuinely
    # strong multilingual embedding match through to the grounded-answer
    # prompt, which still rejects context that does not answer the query.
    if any(ord(char) > 127 for char in query) and chunk.score >= _CROSS_LINGUAL_CONTEXT_SCORE:
        return True
    # Phase 1 item 1: the check above keys off SCRIPT, so it covered Hindi
    # and Tamil but not Hinglish -- romanized Hindi is pure ASCII, shares
    # no more vocabulary with an English statute than Devanagari does, and
    # was being held to the full English lexical-overlap standard. A
    # Hinglish speaker asking a question this app explicitly supports got
    # "no verified document" for material an English speaker was answered
    # from. The allowance is granted on the same two conditions as the
    # script-based one: the language genuinely isn't English, and the
    # embedding match is strong enough to stand on its own.
    if (
        language
        and language.strip().lower() not in ("english", "")
        and chunk.score >= _CROSS_LINGUAL_CONTEXT_SCORE
    ):
        return True
    if "zero fir" in query_text:
        return "zero fir" in chunk_text or ("police station" in chunk_text and "jurisdiction" in chunk_text)
    if "bail" in query_text:
        return "bail" in chunk_text
    if any(term in query_text for term in ("security deposit", "landlord", "tenant", "rental deposit", "pg deposit")):
        return any(term in chunk_text for term in ("security deposit", "landlord", "tenant", "rent agreement"))
    if "legal notice" in query_text:
        return "legal notice" in chunk_text
    if any(term in query_text for term in ("cheque bounce", "dishonour", "section 138")):
        return any(term in chunk_text for term in ("cheque", "dishonour", "section 138"))
    if "recovery agent" in query_text:
        return any(term in chunk_text for term in ("recovery agent", "borrower", "rbi"))
    if "blackmail" in query_text:
        # Part 43 "RAG Retrieval Audit": the corpus's actual coverage of
        # this topic (BNS) uses the statutory term "extortion"/"criminal
        # intimidation", never the colloquial "blackmail" -- confirmed via
        # a direct retrieval-only test, the correct BNS extortion section
        # WAS retrieved into the candidate pool at a reasonable rank, but
        # the reranked score (~0.10) fell just under the generic
        # `_MIN_ACCEPTED_CONTEXT_SCORE` floor (0.12) with no topic-specific
        # override to save it, so a genuinely relevant match was rejected
        # right at the gate.
        return any(term in chunk_text for term in ("extortion", "criminal intimidation", "blackmail"))
    if any(term in query_text for term in ("upi", "unauthorized", "unauthorised", "electronic banking")):
        return any(
            term in chunk_text
            for term in ("upi", "unauthorized", "unauthorised", "electronic transaction", "banking transaction", "customer liability")
        )
    if any(term in query_text for term in ("rti", "right to information")):
        return any(term in chunk_text for term in ("right to information", "public authority", "information officer"))
    if "gst" in query_text or "goods and services tax" in query_text:
        return any(term in chunk_text for term in ("goods and services tax", "gst", "input tax credit"))
    if "posh" in query_text or ("sexual harassment" in query_text and "workplace" in query_text):
        return any(term in chunk_text for term in ("sexual harassment", "internal committee", "posh"))
    # "itr" must be a whole word: a bare substring check matched "arbitration"/
    # "arbitrator"/"arbitral" (which all contain "itr"), confirmed live once
    # an arbitration/mediation query-expansion term entered `query_text` --
    # it was wrongly caught by this income-tax branch before ever reaching
    # the arbitration override below.
    if "income tax" in query_text or re.search(r"\bitr\b", query_text):
        return "income-tax" in chunk_text or "income tax" in chunk_text
    if any(term in query_text for term in (
        "valid contract", "essential elements", "void agreement",
        "voidable contract", "offer and acceptance", "contract act",
    )):
        return any(term in chunk_text for term in ("contract", "agreement", "consideration", "offer", "acceptance"))
    if any(term in query_text for term in (
        "consumer complaint", "consumer forum", "deficiency in service",
        "unfair trade practice", "consumer protection", "district commission",
    )):
        return any(
            term in chunk_text
            for term in ("consumer", "complaint", "deficiency", "unfair trade practice", "commission", "limitation period")
        )
    # Confirmed live (2026-09-25): "Bharatiya Nyaya Sanhita mein cheating ke
    # aavashyak tatva kya hain?" -- the query_rewriter.py "cheat" expansion
    # got BNS Section 318's chunk into the candidate pool, but this gate had
    # no override for it (unlike bail/FIR/RTI/GST/... below), only the
    # negative `_BNS_OFFENCE_TOPICS` guard above -- so it fell to the generic
    # score/overlap fallback and lost. Mirrors the matching reranker bonus.
    if re.search(r"\bcheat(?:ing|s|ed)?\b|\bdhokha|\bdhokhadhadi\b|\bthagi\b|\bfraud(?:ulent(?:ly)?)?\b", query_text):
        return any(
            term in chunk_text
            for term in ("cheating", "dishonestly inducing", "dishonest inducement", "delivery of property")
        )
    # BNSS s.173 (FIR registration) confirmed live to be the corpus's
    # actual coverage of this topic -- the operative text says "cognizable
    # offence" / "reduced to writing" / "officer in charge of a police
    # station" and never spells out "FIR" itself, so the gate has to
    # accept those statutory terms rather than requiring the colloquial
    # abbreviation the query used.
    if _FIR_QUERY_RE.search(query_text) or (
        "police" in query_text and any(term in query_text for term in ("complaint", "report", "darj", "shikayat"))
    ):
        return any(
            term in chunk_text
            for term in ("first information report", "cognizable offence", "reduced to writing", "officer in charge of a police station")
        )
    # A bare "cognizable vs non-cognizable" classification comparison names
    # neither "FIR" nor "police complaint", so the override above never
    # applied to it -- confirmed live, this let the generic score/overlap
    # fallback below reject the correct BNSS chunk. See the matching
    # reranker bonus in `LegalReranker._topic_bonus` for the full trace.
    if "cognizable" in query_text:
        return any(
            term in chunk_text
            for term in ("cognizable", "non-cognizable", "non cognizable", "first schedule", "classification of offences")
        )
    if any(term in query_text for term in ("arbitration", "mediation", "conciliation", "alternative dispute resolution")):
        return any(
            term in chunk_text
            for term in ("arbitral tribunal", "arbitration agreement", "arbitral award", "arbitrator", "mediator", "mediation", "conciliation", "conciliator")
        )
    if any(term in query_text for term in ("living separately", "restitution of conjugal rights", "conjugal rights")) or (
        any(term in query_text for term in ("wife", "husband", "spouse")) and "separately" in query_text
    ):
        return any(term in chunk_text for term in ("restitution of conjugal rights", "conjugal rights", "judicial separation"))
    allow_short_numeric = intent == "SECTION_LOOKUP"
    query_terms = significant_terms(query_text, allow_short_numeric=allow_short_numeric)
    chunk_terms = significant_terms(chunk_text, allow_short_numeric=allow_short_numeric)
    return chunk.score >= _MIN_ACCEPTED_CONTEXT_SCORE and bool(query_terms & chunk_terms)


def filter_relevant_context(
    query: str, rewritten_query: str, chunks: list[RetrievedChunk],
    intent: str | None = None, language: str | None = None,
) -> list[RetrievedChunk]:
    accepted = [
        chunk for chunk in chunks
        if is_relevant_chunk(query, rewritten_query, chunk, intent=intent, language=language)
    ]
    if settings.retrieval_debug:
        log.info(
            "rag_relevance_gate",
            query=query[:200],
            normalized_query=rewritten_query[:400],
            candidate_count=len(chunks),
            accepted_count=len(accepted),
            accepted_chunks=[
                (chunk.chunk_id, round(chunk.score, 4), chunk.metadata.get("source_document"))
                for chunk in accepted
            ],
            rejected_chunks=[
                (chunk.chunk_id, round(chunk.score, 4), chunk.metadata.get("source_document"))
                for chunk in chunks
                if chunk not in accepted
            ],
            confidence="low" if not accepted else ("high" if accepted[0].score >= 0.6 else "medium"),
            fallback_reason=None if accepted else "no top chunk passed lexical/topic relevance validation",
        )
    return accepted


def most_relevant_chunk(query: str, chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    """Prefers a chunk that shares real vocabulary with `query` over the
    reranker's own top pick (`chunks[0]`), when one exists among the
    candidate pool -- a sanity check on the reranker's output, not a
    replacement for it: the reranker already ordered these, this only
    overrides that order when there's positive lexical evidence a
    lower-ranked chunk is actually on-topic and the top one isn't.

    Deliberately falls back to `chunks[0]` (the reranker's choice,
    unchanged) whenever NO candidate shares any significant vocabulary
    with the query at all, rather than treating that as proof nothing
    is relevant -- a Hindi/Hinglish query ("mera bike chori ho gaya")
    can legitimately match an English chunk ("Vehicle theft...") that
    shares zero literal tokens with it, and rejecting that outright
    would be a false negative, not a safety improvement.
    """
    if not chunks:
        return None
    query_terms = significant_terms(query)
    if not query_terms:
        return chunks[0]
    query_terms = _drop_named_instrument_terms(query, query_terms)
    best_chunk = chunks[0]
    best_overlap = len(query_terms & significant_terms(chunks[0].text))
    for chunk in chunks[1:]:
        overlap = len(query_terms & significant_terms(chunk.text))
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = chunk
    return best_chunk


def reorder_by_relevance(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Part 42 "Answer Quality & Intent Accuracy": applies the same
    lexical-overlap sanity check `_most_relevant_chunk` already uses on
    the LLM-failure fallback path to the normal RAG success path too --
    a reranked-but-topically-wrong top chunk (inflated by metadata
    bonuses rather than genuine similarity, e.g. a Legal Notice template
    outranking real bail material for a bail question) should not
    become the PRIMARY context just because the reranker's blended score
    cleared the relevance threshold. Only ever reorders, never drops a
    candidate -- same reasoning as `_most_relevant_chunk` for not
    penalizing a legitimate Hindi/Hinglish query against largely
    English-text chunks with zero literal overlap.
    """
    if not chunks:
        return chunks
    best = most_relevant_chunk(query, chunks)
    if best is None or best is chunks[0]:
        return chunks
    return [best] + [chunk for chunk in chunks if chunk is not best]
