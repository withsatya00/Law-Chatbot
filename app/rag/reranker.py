import re

from app.core.config import settings
from app.rag.neural_reranker import NeuralReranker
from app.rag.query_rewriter import IPC_SECTION_RE, IPC_TO_BNS_CROSSWALK
from app.rag.statute_currency import status_ranking_adjustment
from app.schemas.common import RetrievedChunk

# English definition questions are question-word-first ("what is FIR"), but Hindi/
# Hinglish ones are subject-first with the question word LAST ("FIR kya hota hai",
# "FIR kise kahte hai") -- anchoring both forms to ^\s* (as response_cache.py's
# DEFINITIONAL_PATTERN does, for cache-TTL purposes only) silently never matches any
# Hindi/Hinglish phrasing, since the query doesn't start with "kya hota hai". Verified
# against the live corpus: every Hindi FIR variant fell through to 0.0 bonus and case
# law outranked the statute text as a result. English stays prefix-anchored (matches
# real usage); Hindi/Hinglish forms are checked as a suffix instead.
DEFINITION_QUERY_RE = re.compile(
    r"^\s*(what is|what does|what are|define|explain|meaning of)\b"
    r"|(kya hota hai|kya hoti hai|kya hai|kya matlab|kise kahte hai|kise kahte hain|kise kehte hain)\s*[?？]?\s*$",
    re.IGNORECASE,
)

# For a definition-seeking query, a statute's own defining clause or operative section
# text should always outrank a judgment discussing that provision or a commentary
# summarizing it -- a case law citation doesn't stop being useful, it's just not what
# answers "what is X". Non-definitional queries (e.g. "what did the court say about
# FIR misuse") get no such boost, since case law/commentary is exactly the right
# answer there.
DEFINITION_QUERY_SOURCE_TYPE_BONUS = {
    "definition": 0.22,
    "statute": 0.16,
    "faq": 0.10,
    "case_law": 0.02,
    "commentary": 0.0,
}

# When several chunks claim the same `section_number`, the one whose match came
# from that section's own heading line (`MetadataExtractor.SECTION_NUMBER_
# PROVENANCE_HEADING`) is the section's actual text -- a chunk that only
# mentions the number as a cross-reference or a loose mid-sentence match is
# talking ABOUT the section, not necessarily reproducing it. Deliberately a
# small tie-breaker (well under `legal_bonus`/`domain_bonus`), not a hard
# filter: a chunk with no `section_number_provenance` at all (indexed before
# Part [Phase 3] added the field) gets the same 0.0 as an explicit "fallback",
# never penalized below that.
_SECTION_PROVENANCE_BONUS = {
    "heading": 0.05,
    "cross_reference": 0.02,
    "fallback": 0.0,
}

# Phase 7 "TOC Contamination, stage 2": `metadata.is_toc_chunk()` stops a
# chapter/part index chunk from claiming a spurious `section_number`, which
# is enough to stop it being wrongly ACCEPTED as an exact SECTION_LOOKUP
# match -- but its raw BM25+vector fused score (dominant at 0.75 weight
# below) can still comfortably outrank the real section's own heading chunk
# on lexical density alone (confirmed live: an index chunk's raw fused score
# of 0.217 vs. the real chunk's 0.142 for "Section 318" -- a ~0.075 gap no
# combination of the other bonuses here closes). A flat down-weight, not a
# retrieval-time exclusion, so an index chunk can still surface as
# supporting context; it just stops winning the primary/top slot.
INDEX_CHUNK_PENALTY = 0.2


# Unpaid wages/salary. The multilingual bridge expands every native-language
# salary word to "unpaid salary wages employment dues ...", and the English
# rewriter to "salary pending ... wages", so the English forms below cover
# both those expansions and plain English questions.
_UNPAID_WAGE_QUERY_RE = re.compile(
    r"unpaid\s+(?:salary|wages?)|salary\s+pending"
    r"|(?:salary|wages?)\b[^.?!]{0,40}\b(?:not\s+(?:been\s+)?(?:paid|received)|withheld|overdue|delayed)"
    r"|\b(?:withh[eo]ld|delay(?:ed)?)\w*\s+(?:of\s+)?(?:my\s+)?(?:salary|wages?)",
    re.IGNORECASE,
)
# Preferred central-law source for wage questions. The Maharashtra-hosted
# Payment of Wages Act, 1936 stays a fully eligible source (it governs periods
# before the Code applied) and receives the same topical bonus, just not the
# central-law preference.
_CENTRAL_WAGE_CODE_KEY = "code-on-wages-2019"
_HISTORICAL_WAGE_ACT_KEY_FRAGMENT = "payment-of-wages-act"
_WAGE_TOPIC_BONUS = 0.12
_CENTRAL_WAGE_CODE_BONUS = 0.10


class LegalReranker:
    def __init__(self, neural: NeuralReranker | None = None) -> None:
        # Constructed unconditionally but harmless when unused: `NeuralReranker`
        # only imports/loads its actual model lazily, on first `.score()` call
        # -- see `settings.use_neural_reranker`.
        self._neural = neural or NeuralReranker()

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 6, legal_category: str | None = None
    ) -> list[RetrievedChunk]:
        query_terms = set(query.lower().split())
        is_definition_query = bool(DEFINITION_QUERY_RE.search(query.strip()))
        query_lower = query.lower()
        rescored: list[RetrievedChunk] = []
        for chunk in chunks:
            chunk_lower = f"{chunk.text} {chunk.metadata}".lower()
            legal_bonus = 0.08 if any(key in chunk.metadata for key in ["act_name", "section_number", "article"]) else 0.0
            source_type_bonus = 0.0
            if is_definition_query:
                source_type_bonus = DEFINITION_QUERY_SOURCE_TYPE_BONUS.get(str(chunk.metadata.get("source_type") or ""), 0.0)
            # `IntentDetector` already classifies the query's legal domain (e.g. a
            # landlord/security-deposit question -> "Property Law") before retrieval
            # runs in chat_service.py, but that classification previously only
            # softened the query text via query_rewriter -- it never touched ranking,
            # so a Transfer of Property Act chunk about lessor rights could outscore
            # actual tenant-deposit content purely on embedding similarity. Boosting
            # same-domain chunks (soft, not a hard filter) lets a wrong/uncertain
            # domain guess still fall back to whatever embeddings found, rather than
            # zeroing out results the way a MongoDB filter on legal_category would.
            domain_bonus = 0.0
            if legal_category and chunk.metadata.get("legal_category", "").lower() == legal_category.lower():
                domain_bonus = 0.10
            topic_bonus = self._topic_bonus(query_lower, chunk_lower)
            topic_bonus += self._wage_source_bonus(query_lower, chunk_lower, chunk.metadata)
            provenance_bonus = _SECTION_PROVENANCE_BONUS.get(str(chunk.metadata.get("section_number_provenance") or ""), 0.0)
            index_penalty = INDEX_CHUNK_PENALTY if chunk.metadata.get("source_type") == "index" else 0.0
            # Phase 2: prefer a source recorded as in force and verified over
            # one recorded as repealed or superseded. A preference, not a
            # filter -- a repealed provision is still the correct answer for an
            # incident that happened while it was in force, so it must stay
            # reachable rather than being excluded from the candidate pool.
            currency_adjustment = status_ranking_adjustment(chunk.metadata)
            overlap = len(query_terms & set(chunk.text.lower().split())) / max(len(query_terms), 1)
            chunk.score = min(
                1.0,
                max(
                    0.0,
                    0.75 * chunk.score + 0.25 * overlap + legal_bonus + source_type_bonus + domain_bonus
                    + topic_bonus + provenance_bonus + currency_adjustment - index_penalty,
                ),
            )
            rescored.append(chunk)
        if settings.use_neural_reranker and rescored:
            await self._apply_neural_signal(query, rescored)
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]

    async def _apply_neural_signal(self, query: str, chunks: list[RetrievedChunk]) -> None:
        """Blends in the optional cross-encoder score (see `NeuralReranker`)
        as `final = (1 - weight) * heuristic + weight * neural` -- additive
        to every bonus already folded into `chunk.score` above, never a
        replacement. A `None` result (model unavailable/failed) leaves every
        `chunk.score` exactly as the heuristics alone computed it.
        """
        neural_scores = await self._neural.score(query, [chunk.text for chunk in chunks])
        if neural_scores is None:
            return
        weight = settings.neural_reranker_weight
        for chunk, neural_score in zip(chunks, neural_scores, strict=True):
            chunk.score = min(1.0, max(0.0, (1 - weight) * chunk.score + weight * neural_score))

    @staticmethod
    def _wage_source_bonus(query_lower: str, chunk_lower: str, metadata: dict[str, object]) -> float:
        """Unpaid-wage questions: lift wage text from the central Code on Wages,
        2019 over its predecessor, without dropping the Payment of Wages Act."""
        if not _UNPAID_WAGE_QUERY_RE.search(query_lower):
            return 0.0
        key = str(metadata.get("document_key") or "").lower()
        is_code = key == _CENTRAL_WAGE_CODE_KEY
        if not (is_code or _HISTORICAL_WAGE_ACT_KEY_FRAGMENT in key):
            return 0.0
        if "wage" not in chunk_lower and "remuneration" not in chunk_lower:
            return 0.0
        return _WAGE_TOPIC_BONUS + (_CENTRAL_WAGE_CODE_BONUS if is_code else 0.0)

    def _topic_bonus(self, query_lower: str, chunk_lower: str) -> float:
        bonus = 0.0
        # CPC s.80: the mandatory notice before suing the Government or a public
        # officer. The indexed section reads "no suits shall be instituted
        # against the Government ... or against a public officer"; without a
        # bonus it lost the top-k cut to unrelated "notice" chunks.
        if (
            "suit against government" in query_lower or "suing the government" in query_lower
            or ("section 80" in query_lower and ("civil procedure" in query_lower or "cpc" in query_lower))
        ) and "instituted against the government" in chunk_lower:
            bonus += 0.30
        if "zero fir" in query_lower:
            if "zero fir" in chunk_lower:
                bonus += 0.35
            if "police station" in chunk_lower and "jurisdiction" in chunk_lower:
                bonus += 0.12
        if "bail" in query_lower and "bail" in chunk_lower:
            bonus += 0.16
            if "apprehending arrest" in chunk_lower or "anticipatory" in chunk_lower:
                bonus += 0.08
        if any(term in query_lower for term in ("security deposit", "landlord", "tenant", "rental deposit", "pg deposit")):
            # QA pass 2026-09-24 (T026/T027/T028 multi-turn security-deposit
            # follow-ups): the bare single-term match ("landlord" alone) let
            # an unrelated Odisha Stamp Act schedule outrank the real Bombay
            # Rent Control Act text -- confirmed live via direct pipeline
            # replication: that OCR'd stamp-duty schedule mentions
            # "landlord's share of cesses" in a wholly different (rent-as-
            # taxable-instrument) context and scored 0.41, ahead of the
            # genuine tenancy-dispute chunk at 0.38, intermittently pushing
            # bad context into the LLM's top "Source 1" slot and producing
            # what looked like non-deterministic refusals but was actually
            # this reranking gap. "landlord"/"tenant" now must co-occur
            # (real tenancy-relationship prose, not a schedule entry that
            # happens to name one party) unless the chunk already contains
            # an unambiguous deposit/tenancy phrase.
            if (
                "security deposit" in chunk_lower or "rental deposit" in chunk_lower or "rent agreement" in chunk_lower
                or ("landlord" in chunk_lower and "tenant" in chunk_lower)
            ):
                bonus += 0.18
        if "legal notice" in query_lower and "legal notice" in chunk_lower:
            bonus += 0.18
        # Phase 2 benchmark finding: the trigger list previously required the
        # noun phrase "cheque bounce", so an ordinary user asking "what can I do
        # if a cheque given to me bounced?" matched nothing. Section 138 was in
        # the candidate pool at rank ~12 and never reached the top 8, giving a
        # correct Act with the wrong provision. Colloquial forms -- which is how
        # this is actually asked -- are now recognised, and the chunk side
        # matches the Act's own wording ("insufficiency of funds") rather than
        # only the words the question used.
        if any(
            term in query_lower
            for term in (
                "cheque bounce", "cheque bounced", "bounced cheque", "cheque return",
                "cheque returned", "dishonour", "dishonor", "dishonoured", "dishonored",
                "section 138", "check bounce",
            )
        ) or ("cheque" in query_lower and "bounce" in query_lower):
            if any(
                term in chunk_lower
                for term in (
                    "dishonour", "dishonor", "insufficiency of funds",
                    "section 138", "138. dishonour",
                )
            ):
                bonus += 0.30
            elif "cheque" in chunk_lower:
                bonus += 0.10
        if any(term in query_lower for term in ("upi", "unauthorized", "unauthorised", "electronic banking")):
            if any(term in chunk_lower for term in ("upi", "unauthorized", "unauthorised", "electronic transaction", "banking transaction", "customer liability")):
                bonus += 0.16
        if "recovery agent" in query_lower and any(term in chunk_lower for term in ("recovery agent", "borrower", "rbi")):
            bonus += 0.16
        # Mirrors the pattern above for the newly-added Acts (RTI, GST,
        # POSH, Income-tax 2025): confirmed live these need it just as much
        # as "bail"/"cheque bounce" did -- without a topic bonus, RTI's own
        # actual Section 6/19 chunks scored ~0.06 after reranking (query
        # expansion in `query_rewriter.py` at least got them INTO the
        # candidate pool, but raw fusion score alone still left them well
        # under `_MIN_ACCEPTED_CONTEXT_SCORE`), while nothing distinguished
        # them from unrelated chunks that merely share generic words.
        if any(term in query_lower for term in ("rti", "right to information")):
            if any(term in chunk_lower for term in ("right to information", "public authority", "information officer")):
                bonus += 0.18
        if "gst" in query_lower or "goods and services tax" in query_lower:
            if any(term in chunk_lower for term in ("goods and services tax", "gst", "input tax credit")):
                bonus += 0.16
        if "posh" in query_lower or ("sexual harassment" in query_lower and "workplace" in query_lower):
            if any(term in chunk_lower for term in ("sexual harassment", "internal committee", "posh")):
                bonus += 0.18
        # "itr" must be a whole word: a bare substring check matched
        # "arbitration"/"arbitrator"/"arbitral" (all contain "itr"),
        # confirmed live once the arbitration/mediation topic bonus below was
        # added -- this branch fired first and stole the bonus.
        if "income tax" in query_lower or re.search(r"\bitr\b", query_lower):
            if "income-tax" in chunk_lower or "income tax" in chunk_lower:
                bonus += 0.16
        # Same pattern again for three more topics confirmed live to need it:
        # a user session got "no verified document" for all three even though
        # the corpus holds the exact right provision (Indian Contract Act
        # s.10, Consumer Protection Act s.69, BNSS s.173) -- without a topic
        # bonus these lose the top_k=6 cut to shorter, lexically denser
        # chunks (FAQs, case captions) that merely share generic words.
        if any(term in query_lower for term in (
            "valid contract", "essential elements", "void agreement",
            "voidable contract", "offer and acceptance", "contract act",
        )):
            if any(term in chunk_lower for term in (
                "free consent", "lawful consideration", "lawful object",
                "competent to contract", "voidable", "coercion", "misrepresentation",
            )):
                bonus += 0.18
        if any(term in query_lower for term in (
            "consumer complaint", "consumer forum", "deficiency in service",
            "unfair trade practice", "consumer protection", "district commission",
        )):
            if any(term in chunk_lower for term in (
                "district commission", "deficiency", "unfair trade practice", "limitation period",
            )):
                bonus += 0.18
        # QA pass 2026-09-24 (T074, QA_REPORT_100Q_FINAL_20260924.md section
        # 2.5/7.4): "admissibility of electronic evidence Bharatiya Sakshya
        # Adhiniyam" had no topic bonus at all, unlike every other topic
        # covered above -- confirmed live via direct pipeline replication:
        # raw retrieval already ranked BSA's own section 63 (the electronic-
        # evidence admissibility provision) #1/#3/#5 among the fused
        # candidates at near-tied raw scores (~0.016), but a Delhi Police
        # Academy FAQ document about the new criminal laws generally
        # outscored it after reranking (0.36 vs 0.28) purely from `legal_
        # bonus`/`domain_bonus`/lexical overlap on the Act's own name, which
        # the FAQ repeats as a running page header ("The Bharatiya Sakshya
        # Adhiniyam") though it never actually quotes the provision. Chunk-
        # side markers are the section's own genuine operative vocabulary
        # ("computer output", "deemed to be also a document" -- confirmed
        # live: absent from the FAQ chunk's text, so this cannot also lift
        # it) rather than "admissibility" alone, which the FAQ's own
        # commentary does use.
        if any(term in query_lower for term in (
            "electronic evidence", "electronic record", "admissibility of electronic", "secondary evidence",
        )):
            if any(term in chunk_lower for term in (
                "computer output", "deemed to be also a document", "electronic record",
                "admissibility of electronic",
            )):
                bonus += 0.30
        # Confirmed live (2026-09-25): "Bharatiya Nyaya Sanhita mein cheating
        # ke aavashyak tatva kya hain?" -- named the query_rewriter.py "cheat"
        # expansion above and DID retrieve BNS Section 318's own chunk into
        # the candidate pool, but it had no topic bonus of its own: the only
        # existing cheating-related bonus lives entirely inside the IPC-
        # crosswalk branch below, which requires the query to literally say
        # "IPC" (e.g. "Section 420 IPC"). A plain "cheating under BNS"
        # question -- no IPC mention at all -- got zero bonus and lost the
        # top_k=6 cut, the same gap already fixed above for arbitration/
        # mediation and cognizable/non-cognizable.
        if re.search(r"\bcheat(?:ing|s|ed)?\b|\bdhokha|\bdhokhadhadi\b|\bthagi\b|\bfraud(?:ulent(?:ly)?)?\b", query_lower):
            if any(term in chunk_lower for term in (
                "cheating", "dishonestly inducing", "dishonest inducement", "delivery of property",
            )):
                bonus += 0.20
        if re.search(r"\bfir\b", query_lower) or "first information report" in query_lower or (
            "police" in query_lower and any(term in query_lower for term in ("complaint", "report", "darj", "shikayat"))
        ):
            if any(term in chunk_lower for term in (
                "first information report", "cognizable offence", "reduced to writing", "officer in charge of a police station",
            )):
                bonus += 0.18
        # Confirmed live (2026-09-25): "Cognizable aur non-cognizable offence
        # mein kya antar hai?" -- a bare classification COMPARISON, naming
        # neither "FIR" nor "police complaint" -- only ever reached this
        # bonus indirectly (nested in the FIR gate above), so the correct
        # BNSS chunk squeaked in at the very edge of the relevance floor
        # while three unrelated State Acts outranked it. A standalone gate
        # on "cognizable" alone covers this comparison shape directly.
        if "cognizable" in query_lower:
            if any(term in chunk_lower for term in (
                "cognizable", "non-cognizable", "non cognizable", "first schedule", "classification of offences",
            )):
                bonus += 0.18
        # Confirmed live: "Arbitration aur mediation mein kya antar hai?"
        # DID retrieve both Acts' chunks into the raw candidate pool (the
        # query already names them), but neither had a topic bonus of its
        # own, so they lost the top_k=6 cut to unrelated State Acts on
        # `legal_bonus`/lexical overlap alone -- one Mediation Act chunk was
        # scored down to exactly 0.0 by the blended formula.
        if any(term in query_lower for term in ("arbitration", "mediation", "conciliation", "alternative dispute resolution")):
            if any(term in chunk_lower for term in (
                "arbitral tribunal", "arbitration agreement", "arbitral award", "arbitrator",
                "mediator", "mediation", "conciliation", "conciliator",
            )):
                bonus += 0.18
        # Confirmed live: "meri patni bina karan alag reh rahi hai" never
        # even retrieved the Hindu Marriage Act (fixed by the query
        # expansion in `query_rewriter.py`); this bonus keeps the correct
        # Section 9 chunk from then losing the rerank cut the way the two
        # topics above did.
        if any(term in query_lower for term in ("living separately", "restitution of conjugal rights", "conjugal rights")) or (
            any(term in query_lower for term in ("wife", "husband", "spouse")) and "separately" in query_lower
        ):
            if any(term in chunk_lower for term in ("restitution of conjugal rights", "conjugal rights", "judicial separation")):
                bonus += 0.22
        ipc_match = IPC_SECTION_RE.search(query_lower)
        if ipc_match:
            ipc_section = (ipc_match.group(1) or ipc_match.group(2) or "").lower()
            crosswalk = IPC_TO_BNS_CROSSWALK.get(ipc_section)
            if crosswalk:
                bns_section, topic = crosswalk
                # `f"{bns_section}. "` anchors on the heading form ("318. Cheating"),
                # not a bare digit match -- a bare "318" would just recreate the same
                # numeric-collision problem this bonus exists to fix.
                if f"{bns_section}. " in chunk_lower and any(word in chunk_lower for word in topic.split()):
                    bonus += 0.3
        return bonus
