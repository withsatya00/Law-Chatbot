import asyncio
import re
from typing import Any

import structlog

from app.cache.semantic_cache import retrieval_cache
from app.core.config import settings
from app.intent.detector import (
    ACT_FULL_NAME_TO_ABBREVIATION,
    parse_section_lookup,
    parse_section_lookup_act,
)
from app.observability.metrics import Timer, metrics
from app.rag.embeddings import EmbeddingProvider
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.kb_jurisdiction import filter_by_matter_context, shared_retrieval_filters
from app.rag.query_rewriter import (
    FIR_BNSS_SECTION_NUMBER,
    FIR_CONCEPT_RE,
    RAPE_BNS_SECTION_NUMBERS,
    RAPE_CONCEPT_RE,
    THEFT_BNS_SECTION_NUMBER,
    THEFT_CONCEPT_RE,
    SmartQueryRewriter,
)
from app.rag.vector_store import MongoVectorStore, VectorStore
from app.schemas.common import RetrievedChunk

log = structlog.get_logger(__name__)

# A direct citation ("Section 318 of the Bharatiya Nyaya Sanhita") names one exact
# provision -- similarity search is structurally the wrong tool for it. An act's own
# operative text never repeats its own name, so neither BM25 nor embedding similarity
# can rank that act's own section highly against this exact query shape; a
# cross-referencing document that DOES mention the act by name (e.g. BNSS citing BNS
# sections throughout) dominates instead. Confirmed live against the corpus: BNS
# Section 318's own chunk ranked #120/821 on BM25 and #197/200 on embedding cosine
# for "What is Section 318 of the Bharatiya Nyaya Sanhita?" -- both outside either
# leg's candidate pool, so it never reached RRF, reranking, or the LLM. An exact
# metadata filter sidesteps similarity scoring for this query shape entirely.
#
# Deliberately requires the Act to be NAMED alongside the section number ("... of
# <Act>"), not just a bare "Section <N>" -- every Act in the corpus numbers its own
# provisions independently (BNS, the GST Act, the Special Marriage Act, and the
# Domestic Violence Act each have their own "Section 2"), so a bare number carries no
# Act information at all. Confirmed live (historical bug, since fixed): gating this
# floor on a bare-number pattern floored EVERY retrieved chunk's score unconditionally
# once a bare "Section 2" was seen, including a GST/Consumer/Special-Marriage-Act
# chunk that only happened to be *retrieved*, not one whose OWN section_number
# actually matched -- making every candidate indistinguishable downstream. This regex
# stays scoped to the named-Act shape only, applying its filter+floor to every
# retrieved result unconditionally (safe here specifically because the Mongo/BM25
# filter above already constrains `results` to matching chunks before this runs).
# Phase 8 "Section-Number-Aware Retrieval" adds a SEPARATE, narrower mechanism below
# for the bare-number case that doesn't repeat that mistake -- see `_apply_section_
# number_floor`: it floors (or merges in) only the specific chunks whose OWN
# `section_number` metadata equals the parsed number, chunk by chunk, never the whole
# candidate pool by query shape alone.
_NAMED_SECTION_CITATION_PATTERNS = (
    # "Section 138 of the Negotiable Instruments Act" and the equally
    # ordinary "Section 138 Negotiable Instruments Act explain karo".
    #
    # QA retest 2026-09-24 (T072/T074, from
    # `QA_REPORT_100Q_RETEST_20260924.md` section 7 item 6): the whole
    # pattern is `re.IGNORECASE`, which also relaxes the `[A-Z]` anchor that
    # is supposed to mark where the ACT NAME starts -- so on a query with a
    # lowercase topical phrase between the section number and the real Act
    # name ("Section 138 cheque dishonour Negotiable Instruments Act"), the
    # non-greedy `act` group had nowhere case-sensitive to stop and swallowed
    # the topic words too, capturing "cheque dishonour Negotiable Instruments
    # Act" as the "Act name". That garbled string doesn't match any real
    # `act_name`/`source_document` metadata, so the `act_name`-filtered
    # search below found 0 candidates, AND `find_named_section`'s own
    # `requested_tokens <= identity_tokens` check (which needs every token of
    # the SUPPLIED act name to appear in the chunk's identity) rejected the
    # real Negotiable Instruments Act chunks too, because "cheque"/
    # "dishonour" aren't in that document's filename. Confirmed live: this
    # query retrieved an unrelated Maharashtra Court Fees Act "Section 138"
    # ahead of the real Negotiable Instruments Act one, matching the QA
    # report's exact complaint. `(?-i: ... )` makes only this anchor
    # case-sensitive again (the rest of the pattern, and the OTHER citation
    # pattern below -- which matches a fixed enumerated list of Act names
    # instead of "any capitalized run," so it never had this problem --
    # keep working case-insensitively) -- a genuine Act name is always
    # written capitalized in practice, so requiring a real capital letter to
    # start the capture is precision, not a new failure mode: a query with no
    # capitalized Act name simply fails to match here (as it always could)
    # and falls through to the ordinary hybrid search that already ran.
    re.compile(
        r"\bsection\s+(?P<number>\d{1,4}[A-Z]{0,2})\s+(?:of\s+(?:the\s+)?)?"
        # Up to 4 case-sensitively-LOWERCASE filler words ("cheque
        # dishonour") may sit between the number and the Act name -- kept
        # non-greedy and bounded so it tries the shortest skip first (zero
        # filler words, matching the original adjacent-Act-name shape
        # unchanged) and can never run away across an unrelated capitalized
        # word much later in a long query.
        r"(?-i:(?:[a-z]+\s+){0,4}?(?P<act>[A-Z][A-Za-z ]*?(?:Act|Sanhita|Adhiniyam|Code)))\b",
        re.IGNORECASE,
    ),
    # "Negotiable Instruments Act Section 138 ..." -- the form used in the
    # failed manual-testing turn.  Previously this fell into an Act-agnostic
    # section search and mixed CGST and unrelated same-numbered provisions
    # into both the prompt and the visible Sources list.
    re.compile(
        r"\b(?P<act>(?:Negotiable\s+Instruments|Information\s+Technology|Consumer\s+Protection|"
        r"Hindu\s+Marriage|Domestic\s+Violence|Right\s+to\s+Information|Indian\s+Contract|"
        r"Transfer\s+of\s+Property|Central\s+Goods\s+and\s+Services\s+Tax|Companies)\s+Act|"
        r"Bharatiya\s+(?:Nyaya|Nagarik\s+Suraksha)\s+Sanhita|Bharatiya\s+Sakshya\s+Adhiniyam)\s+"
        r"section\s+(?P<number>\d{1,4}[A-Z]{0,2})\b",
        re.IGNORECASE,
    ),
)

_ARTICLE_CITATION_RE = re.compile(r"\barticle\s+(?P<number>\d{1,3}[A-Z]?)\b", re.IGNORECASE)


def _named_section_citation(query: str) -> tuple[str, str] | None:
    for pattern in _NAMED_SECTION_CITATION_PATTERNS:
        match = pattern.search(query)
        if match:
            return match.group("number").upper(), match.group("act").strip()
    return None
# A metadata-exact section match is more authoritative than any similarity score --
# flooring here (before reranker.rerank()'s 0.75x blend) guarantees the chunk clears
# both chat_service.py's 0.08 fallback gate and 0.12 relevance gate, without touching
# chat_service.py. Verified live: floor of 0.65 -> post-rerank scores of 0.5675-0.8775
# across 8 section-citation test queries. Reused by the bare-number path below.
_EXACT_SECTION_SCORE_FLOOR = 0.65

# `parse_section_lookup_act` returns the ABBREVIATION a query named (e.g.
# "BNS"), but the corpus's own `act_name` metadata never stores abbreviations
# -- confirmed live via `embeddings_metadata.distinct("metadata.act_name")`
# against the real corpus (18 distinct values, checked directly, not
# guessed): "The Bharatiya Nyaya Sanhita", "The Bharatiya Nagarik Suraksha
# Sanhita" (also inconsistently stored as bare "Suraksha Sanhita" in one
# source document), "The Bharatiya Sakshya Adhiniyam", "Negotiable
# Instruments Act" / "The Negotiable Instruments Act", etc. Only Acts this
# corpus actually indexes are listed; an abbreviation `detector.
# _ACT_ABBREVIATION` recognizes but this corpus has no chunks for (e.g.
# "CrPC", "CPC") has no entry here and falls through to the Act-agnostic
# lookup in `_apply_section_number_floor` below, same as if no Act had been
# named -- never a hard failure for an Act this corpus doesn't cover.
_ACT_ABBREVIATION_TO_METADATA_NAMES: dict[str, list[str]] = {
    # Confirmed live (2026-08-25): BNS's own `act_name` metadata never
    # actually stored "The Bharatiya Nyaya Sanhita" at all -- the corpus's
    # two BNS-content documents stored "Bhartiya Nyay Sanhita" (a 22-chunk
    # FAQ) and, after a corrupted-metadata backfill fix, "Bharatiya Nyaya
    # Sanhita" (the 189-chunk document actually containing bare-act text) --
    # so this Act-scoped filter had silently matched zero chunks and always
    # fallen through to the unscoped, ambiguous lookup below. All three
    # spelling variants actually seen in the corpus are listed so a future
    # re-ingest under any of them still resolves correctly.
    # "BHARA TIY A NY A Y A SANHITA" added 2026-09-16: a fourth real variant
    # (237 chunks, the corpus's actual full BNS penal-code document --
    # theft, rape, murder, all core offences -- confirmed live via
    # `distinct("metadata.act_name")` scoped to that document's own
    # `source_document`), missed by the original three-variant sweep above
    # because that scan predates this document's approval out of
    # `needs_review`.
    "BNS": [
        "The Bharatiya Nyaya Sanhita", "Bharatiya Nyaya Sanhita", "Bhartiya Nyay Sanhita",
        "BHARA TIY A NY A Y A SANHITA",
    ],
    # Confirmed live (2026-09-14): the same class of gap as BNS's own three
    # variants above, this time for BNSS -- a `distinct("metadata.act_name")`
    # regex scan for "suraksha" turned up FIVE real corpus values, only two
    # of which were listed here. The single biggest one by chunk count,
    # "THE BHARA TIY A NAGARIK SURAKSHA SANHITA" (405 chunks, a garbled-
    # spacing ingest artifact, not a typo introduced here), was completely
    # unmatched -- every exact `act_name`-scoped BNSS section lookup was
    # silently missing it. This is exactly what caused BNSS Section 173 (the
    # FIR-registration provision, see `FIR_CONCEPT_RE` below) to be
    # unreachable via the Act-scoped path: its one `document_status=active,
    # review_status=approved` chunk is tagged under this exact garbled
    # variant. Deliberately excludes "THE MAHARASHTRA SURAKSHA DAL ACT",
    # which also matched the regex scan but is a genuinely unrelated Act.
    "BNSS": [
        "The Bharatiya Nagarik Suraksha Sanhita",
        "Bharatiya Nagarik Suraksha Sanhita",
        "Suraksha Sanhita",
        "THE BHARA TIY A NAGARIK SURAKSHA SANHITA",
        "Chapter XIII of the Bharatiya Nagarik Suraksha Sanhita",
    ],
    "BSA": ["The Bharatiya Sakshya Adhiniyam"],
    "NI ACT": ["Negotiable Instruments Act", "The Negotiable Instruments Act"],
}

# Multi-Act disambiguation for a BARE section number (no Act named, no
# abbreviation matched by `parse_section_lookup_act` either -- e.g. plain
# "Section 2"). Unlike `_ACT_ABBREVIATION_TO_METADATA_NAMES` above, this is
# never used as a hard `find_by_section_number` filter (no corpus-exact-value
# verification has been done for these terms) -- only as a soft, substring
# tie-breaker among chunks the Act-agnostic lookup already returned, so a
# wrong or partial match can never hide the correct chunk, only fail to
# prefer it (leaving the pre-existing flat-tie behavior). Reuses
# `EntityExtractor.act_terms` verbatim -- the same vocabulary this app
# already uses elsewhere to recognize an Act mention in free text.
_ACT_HINT_TERMS: tuple[str, ...] = (
    "Bharatiya Nyaya Sanhita", "Bharatiya Nagarik Suraksha Sanhita", "Bharatiya Sakshya Adhiniyam",
    "Consumer Protection Act", "Information Technology Act", "Indian Contract Act", "Companies Act",
    "Transfer of Property Act", "Negotiable Instruments Act", "Hindu Marriage Act",
    "Domestic Violence Act", "RTI Act",
)
_ACT_HINT_SCORE_BONUS = 0.03

# Colloquial/topical phrasing that names no Act but unambiguously implies one
# -- "cheque bounce hone ke baad Section 138 ke andar kya hota hai?" never
# says "Negotiable Instruments Act", yet is exactly as unambiguous to a
# person as if it had. Before this project's corpus grew to include hundreds
# of State Acts (each with its own numbered sections), a bare "Section 138"
# rarely collided with another Act's section 138 even without a hint; once it
# did, this exact query started asking "which Act do you mean?" for one of
# the most common real questions this app gets. Checked as a fallback ONLY
# after `_ACT_HINT_TERMS` finds no literal Act name, and, like it, is a soft
# tie-breaker among chunks already returned -- never a hard filter -- so an
# imperfect or missing entry here can only fail to disambiguate, not hide the
# correct chunk.
_TOPIC_HINT_TERMS: dict[str, str] = {
    "cheque bounce": "Negotiable Instruments Act",
    "cheque bounced": "Negotiable Instruments Act",
    "cheque dishonour": "Negotiable Instruments Act",
    "cheque dishonor": "Negotiable Instruments Act",
    "dishonour of cheque": "Negotiable Instruments Act",
    "dishonor of cheque": "Negotiable Instruments Act",
    "consumer complaint": "Consumer Protection Act",
    "deficiency in service": "Consumer Protection Act",
    "unfair trade practice": "Consumer Protection Act",
    "right to information": "RTI Act",
    "cyber crime": "Information Technology Act",
    "cyber fraud": "Information Technology Act",
    "online fraud": "Information Technology Act",
}


def _find_act_hint(*texts: str | None) -> str | None:
    combined = " ".join(text for text in texts if text).lower()
    for term in _ACT_HINT_TERMS:
        if term.lower() in combined:
            return term
    for phrase, act in _TOPIC_HINT_TERMS.items():
        if phrase in combined:
            return act
    return None


class LegalRetriever:
    def __init__(
        self,
        embeddings: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.embeddings = embeddings or EmbeddingProvider()
        self.vector_store = vector_store or MongoVectorStore()
        self.query_rewriter = SmartQueryRewriter()

    async def rewrite_query(self, query: str, intent: str | None = None, context_summary: str | None = None) -> str:
        with Timer("query_rewrite_latency_ms"):
            return await self.query_rewriter.rewrite(query, intent=intent, context_summary=context_summary)

    async def retrieve(
        self, query: str, top_k: int = 8, filters: dict[str, Any] | None = None, intent: str | None = None,
        context_hint: str | None = None, matter_context: dict[str, Any] | None = None,
        mode: str = "hybrid",
    ) -> tuple[str, list[RetrievedChunk]]:
        """`matter_context` (from `app.rag.matter_context.MatterContext.as_filter()`)
        is the Jurisdiction Routing (Phase 2) applicability/version
        constraint -- `{"state_codes": [...], "locality": ..., "as_of_date": ...}`,
        or `None`/all-empty when the question is state-insensitive. Applied
        as a POST-filter over `results` right before this method returns
        (see `kb_jurisdiction.filter_by_matter_context`'s own docstring for
        why that single choke point, not a per-path change, is what makes
        this uniform across every internal retrieval path). `ChatService`
        additionally folds the SAME State constraint into its shared-branch
        Mongo `$or` (via `kb_jurisdiction.jurisdiction_or_branches`) so the
        DB-level candidate pool is narrowed too -- this post-filter is the
        backstop that holds even if a future retrieval path forgets to.
        """
        filters = dict(filters or {})
        state_codes = (matter_context or {}).get("state_codes") or []
        locality = (matter_context or {}).get("locality")
        as_of_date = (matter_context or {}).get("as_of_date")
        # Phase 1 "Jurisdiction-Aware Knowledge Base" (gap 1 fix): a chunk not
        # explicitly `review_status=approved` must never surface through
        # shared retrieval. Applied ONLY when the caller has no ownership
        # `$or` of its own (e.g. `SearchService`'s plain KB search) -- there,
        # every result is shared/unowned by construction, so a flat top-level
        # filter is correct and sufficient.
        #
        # When a caller DOES carry an ownership `$or` (`ChatService.
        # _prepare_rag_context`, Part 45/46), this must NOT be added here: a
        # top-level `review_status` key ANDs against every branch of that
        # `$or`, including a user's OWN private documents -- which never
        # carry a `review_status` field at all -- and would make a user's own
        # uploads invisible to themselves. That caller instead builds
        # `shared_retrieval_filters()` directly into its own shared/unowned
        # branch and leaves its private-ownership branches untouched; see its
        # own comments for the exact split.
        if "$or" not in filters:
            filters.setdefault("review_status", shared_retrieval_filters()["review_status"])
        named_section = _named_section_citation(query)
        known_variants: list[str] | None = None
        article_match = _ARTICLE_CITATION_RE.search(query)
        if named_section:
            section_number, act_name = named_section
            filters.setdefault("section_number", section_number)
            # Security finding C6: chunk metadata stores `act_name`
            # inconsistently -- not just "Bharatiya Nyaya Sanhita" vs "The
            # Bharatiya Nyaya Sanhita", but real, confirmed-live corpus
            # variants like "Bhartiya Nyay Sanhita" and a garbled-ingest
            # "BHARA TIY A NY A Y A SANHITA" (see `_ACT_ABBREVIATION_TO_
            # METADATA_NAMES`'s own history below). The two-variant guess
            # this used to build matched NEITHER of those for BNS, so the
            # search below found nothing, fell through to the Act-agnostic
            # fallback a few lines down, and both BNS's and BNSS's Section
            # 302 chunks got the SAME unconditional floor at line ~407 --
            # this is the actual mechanism behind the live QA repro ("Section
            # 302 of the Bharatiya Nyaya Sanhita" answered from BNSS). When
            # this citation's Act resolves to one of the abbreviations that
            # list already tracks every known corpus spelling for, use that
            # full list; only fall back to the two-variant guess for an Act
            # this corpus has no tracked spellings for at all.
            known_variants = _ACT_ABBREVIATION_TO_METADATA_NAMES.get(
                ACT_FULL_NAME_TO_ABBREVIATION.get(act_name.lower(), "")
            )
            filters.setdefault("act_name", known_variants or [act_name, f"The {act_name}"])
        elif article_match:
            # Article lookups have the same structural problem as short
            # section lookups: the operative text does not repeat the query's
            # topic words.  Constrain retrieval to the article recorded by the
            # chunker instead of asking similarity search to rediscover it.
            filters.setdefault("article_number", article_match.group("number").upper())
            filters.setdefault("instrument_type", "constitution")
        # Jurisdiction Routing (Phase 2), objective item 6: resolved State(s),
        # locality, and the as-of date are cache-key material, not just
        # filter material -- two callers asking byte-identical questions
        # under different matter contexts must never share a cached result
        # (`filters`'s own repr already differs when the $or embeds a
        # different State, but `as_of_date` lives OUTSIDE `filters` as a
        # post-filter input, so it needs to be named here explicitly too).
        # `mode` included: `mode="semantic"`/`"keyword"` run a genuinely
        # different single-leg search (see `MongoVectorStore.search`), so a
        # cached `mode="hybrid"` result for the same query/filters must never
        # be served back for a `mode="semantic"` call and vice versa.
        cache_key = f"{query}:{filters}:{intent}:{sorted(state_codes)}:{locality}:{as_of_date}:{mode}"
        cache_payload = await retrieval_cache.get(cache_key, language=None)
        if cache_payload:
            metrics.increment("retrieval_cache_hit")
            return cache_payload["rewritten"], [RetrievedChunk(**item) for item in cache_payload["results"]]
        metrics.increment("retrieval_cache_miss")
        expanded_queries = self.query_rewriter.expand_queries(query, intent=intent)
        rewritten = self.query_rewriter.combine(expanded_queries)

        async def _search_legs(search_filters: dict[str, Any]) -> list[RetrievedChunk]:
            with Timer("embedding_latency_ms"):
                embeddings = await self.embeddings.embed_batch(expanded_queries)
            with Timer("retriever_latency_ms"):
                # Each `vector_store.search()` call is an independent hybrid
                # (embedding + BM25) lookup for one query VARIANT -- with
                # nothing but `search_filters`/`top_k` shared between them,
                # not a data dependency in sight. Previously awaited one at a
                # time in a for-loop, so `expand_queries`' own point (asking
                # the same question several ways to widen recall) directly
                # multiplied this method's latency by however many variants
                # it produced: confirmed live, a single chat turn logged 2-3
                # separate `hybrid_retrieval_timing` events back to back,
                # each ~5-8s, for one request. `reciprocal_rank_fusion` below
                # sums each list's own per-document rank contribution and is
                # insensitive to which order the lists arrive in, so running
                # them concurrently changes only wall-clock time, not the
                # merged result.
                result_lists = list(
                    await asyncio.gather(
                        *(
                            self.vector_store.search(embedding, expanded_query, top_k, search_filters, mode=mode)
                            for expanded_query, embedding in zip(expanded_queries, embeddings, strict=True)
                        )
                    )
                )
            return reciprocal_rank_fusion(result_lists, k=settings.retrieval_rrf_k)[:top_k]

        results = await _search_legs(filters or {})
        if named_section:
            section_number, act_name = named_section
            heading_matches = await self.vector_store.find_named_section(
                section_number, act_name, filters
            )
            if heading_matches:
                # The named Act's own heading beats both similarity results
                # and stale section metadata from overlap chunks.
                results = heading_matches[:top_k]
        # The Constitution PDF already present in some installations was
        # indexed before `article_number` metadata existed: its bare
        # "21. Protection of life..." headings were stored as
        # `section_number=21`.  Prefer the correct field for every new index,
        # but keep those existing indexes usable without a destructive
        # migration.  The fallback remains constrained to the exact number;
        # semantic ranking then selects the matching article text among any
        # same-numbered statutory sections.
        if article_match and not results and "article_number" in filters:
            results = await self.vector_store.find_constitution_article(
                article_match.group("number").upper(), filters
            )
        # A named-Act citation ("Section 30 of the Information Technology
        # Act") sets an `act_name` filter above, but per-chunk `act_name`
        # extraction at ingest time is unreliable for documents that
        # reference OTHER Acts in their own text (confirmed live: the IT
        # Act's own chunks carry `act_name` values like "An Act"/"Copyright
        # Act"/"Indian Penal Code" -- whatever Act the chunk's surrounding
        # text happened to mention, not the document's own Act) -- an
        # `act_name` filter can zero out EVERY candidate on both legs even
        # though the actual section is sitting right there under a
        # differently-labeled chunk. Retrying without it (section_number
        # alone, which extraction gets right far more reliably) mirrors
        # `_apply_section_number_floor`'s own "never a hard failure over a
        # named-Act mismatch" fallback one function below -- this citation
        # path just never had the equivalent safety net until now.
        act_agnostic_fallback_used = False
        if named_section and not results and "act_name" in filters:
            act_agnostic_filters = {key: value for key, value in filters.items() if key != "act_name"}
            results = await _search_legs(act_agnostic_filters)
            act_agnostic_fallback_used = True
        # Confirmed live (2026-09-16): "bhartiya nayay sanhita section 12"
        # (BNS has no Section 12 in this corpus at all) fell through to the
        # Act-agnostic fallback above, which returned 8 chunks from 8
        # completely different, genuinely unrelated Acts (Maharashtra
        # Devdasi Act, Bombay Smoke-Nuisances Act, a UP act, ...), each from
        # its own distinct source document -- then got floored to
        # `_EXACT_SECTION_SCORE_FLOOR` UNCONDITIONALLY, presented as
        # confident "sources" for a BNS question, and (since one was
        # UP-specific and the rest Maharashtra-specific) triggered a
        # spurious "Which State?" clarification on top. The comment above
        # this fallback describes the case it exists for -- the IT Act's
        # OWN chunks mislabeled with a wrong `act_name` like "Copyright
        # Act"/"Indian Penal Code" by unreliable per-chunk extraction --
        # which is a single real Act whose messy per-chunk act_name text
        # still overwhelmingly clusters around ONE source document. A
        # section number this thinly and diversely spread across many
        # distinct source documents is the opposite signal: the number is
        # simply common across many real, unrelated Acts, not a single
        # mislabeled one, so nothing here has actually been confirmed to
        # answer the cited Act at all. Flooring is skipped in that case --
        # results still fall through to the ordinary similarity-based
        # relevance gate downstream, same as any ungrounded query, rather
        # than being asserted as authoritative.
        _MAX_FALLBACK_SOURCE_DOCUMENTS = 2
        fallback_too_diverse = act_agnostic_fallback_used and len(
            {chunk.metadata.get("source_document") for chunk in results}
        ) > _MAX_FALLBACK_SOURCE_DOCUMENTS
        # These concept regexes (`FIR_CONCEPT_RE`/`RAPE_CONCEPT_RE`/
        # `THEFT_CONCEPT_RE`) only match literal English/Hinglish trigger
        # words ("fir", "police complaint", "rape", "theft"...), so checking
        # only the raw `query` meant a native-script question about the exact
        # same concept never triggered the section-floor guarantee below --
        # confirmed live (qa-40q-multilingual-20260921 BUG-03): a Gurmukhi
        # "police complaint" question got the bare refusal while the
        # identical topic in Hinglish/English/Devanagari succeeded. `rewritten`
        # already carries `multilingual.py`'s English concept-bridge
        # expansion for any recognised native-language term (see
        # `legal_english_variants`), so checking it too catches those queries
        # without needing a native-script variant of each regex.
        concept_probe = f"{query} {rewritten}"
        if (named_section or article_match) and not fallback_too_diverse:
            if named_section and known_variants:
                # Security finding C6: even after the act-agnostic fallback
                # above, `results` can still hold a DIFFERENT Act's own
                # chunk for this same section number (that's the whole
                # reason the fallback exists) -- only a chunk whose OWN
                # `act_name` metadata actually matches the NAMED Act may
                # receive the "this is authoritative" floor. A different
                # Act's same-numbered chunk keeps its ordinary, much lower
                # similarity score instead of tying with the real answer,
                # exactly like `_apply_section_number_floor`'s per-chunk
                # check does for the bare-number case below.
                for chunk in results:
                    if chunk.metadata.get("act_name") in known_variants:
                        chunk.score = max(chunk.score, _EXACT_SECTION_SCORE_FLOOR)
            else:
                for chunk in results:
                    chunk.score = max(chunk.score, _EXACT_SECTION_SCORE_FLOOR)
        elif intent == "SECTION_LOOKUP":
            results = await self._apply_section_number_floor(query, results, filters, top_k, context_hint)
        elif FIR_CONCEPT_RE.search(concept_probe):
            results = await self._ensure_fir_section_present(results, filters, top_k)
        if RAPE_CONCEPT_RE.search(concept_probe):
            results = await self._ensure_sections_present(
                results, filters, top_k,
                list(RAPE_BNS_SECTION_NUMBERS), _ACT_ABBREVIATION_TO_METADATA_NAMES["BNS"],
            )
        if THEFT_CONCEPT_RE.search(concept_probe):
            results = await self._ensure_sections_present(
                results, filters, top_k,
                [THEFT_BNS_SECTION_NUMBER], _ACT_ABBREVIATION_TO_METADATA_NAMES["BNS"],
            )
        # Jurisdiction Routing (Phase 2): the single choke point every
        # internal path above (RRF fusion, named-section/article exact
        # lookup, the section-number floor fallback) funnels through before
        # this method returns -- see this function's own docstring and
        # `kb_jurisdiction.filter_by_matter_context`'s for why this is
        # deliberately here and not duplicated into each path above.
        results = filter_by_matter_context(results, state_codes, locality, as_of_date)
        if settings.retrieval_debug:
            log.info(
                "expanded_retrieval_debug",
                query=query[:200],
                normalized_query=expanded_queries[0][:200] if expanded_queries else "",
                expanded_queries=expanded_queries,
                final_results=[
                    (chunk.chunk_id, round(chunk.score, 4), chunk.metadata.get("source_document"))
                    for chunk in results[:10]
                ],
            )
        await retrieval_cache.set(
            cache_key,
            {"rewritten": rewritten, "results": [result.model_dump() for result in results]},
            ttl_seconds=900,
        )
        return rewritten, results

    async def _apply_section_number_floor(
        self, query: str, results: list[RetrievedChunk], filters: dict[str, Any], top_k: int,
        context_hint: str | None = None,
    ) -> list[RetrievedChunk]:
        """Phase 8 "Section-Number-Aware Retrieval": a bare "Section <N>" query
        (no Act named, so `_NAMED_SECTION_CITATION_RE` above doesn't apply) has
        two confirmed, independent failure modes that similarity ranking alone
        never recovers from:

        1. "Adjacent-section ranking" (e.g. "Section 302" -> Section 305):
           the reranker's lexical-overlap term rewards a chunk that mentions
           the query's tokens in ordinary prose ("...sub-section (1) of
           section 302...") over the section's OWN heading chunk, whose text
           starts "302." (number-plus-period, never matching the bare token
           "302" against `chunk.text.lower().split()`, which doesn't strip
           punctuation). Confirmed live: Section 302's own chunk scores
           overlap=0.5 against the query "Section 302"; the wrong-winning
           Section 305 chunk (which merely cross-references 302 mid-sentence)
           scores overlap=1.0 -- a 0.125-point swing at the reranker's 0.25
           weight, comfortably more than the ~0.001 raw-fusion-score gap
           between them. A second, independent contributor for some pairs
           (confirmed for Section 337 -> 338): the correct section's own text
           got chunked across an overlap boundary, so the specific chunk
           carrying its `section_number` doesn't start with the heading line
           at all and only earned "fallback" provenance (0.0 bonus) against
           the wrong-winner's "heading" provenance (0.05 bonus).
        2. "Low-number section recall" (Section 2, Section 5): the correct
           chunk doesn't merely lose the ranking -- it never enters the
           top-10 fused candidate pool on EITHER leg at all, confirmed live
           via a fresh per-leg trace. No amount of downstream reranking or
           flooring can rescue a chunk that was never retrieved.

        Both are the same underlying gap: nothing between retrieval and the
        LLM ever authoritatively checks "does this chunk's OWN metadata say
        it IS the requested section" -- everything upstream is similarity-
        or lexical-overlap-based, which a short, self-numbering citation
        query defeats structurally. `MongoVectorStore.find_by_section_number`
        is a direct metadata lookup, bypassing similarity scoring entirely,
        fixing (2); flooring any exact match to `_EXACT_SECTION_SCORE_FLOOR`
        (same constant, same magnitude already proven sufficient for the
        named-Act case) fixes (1). Scoped to chunk-level metadata equality,
        never to the whole candidate pool by query shape -- the exact mistake
        `_NAMED_SECTION_CITATION_RE`'s own history warns against (see its
        comment above), so a different Act's same-numbered section is
        floored too (still a genuine exact match, just not necessarily the
        Act the user meant -- an unresolved, separate ambiguity, not
        reintroduced here) rather than every merely-retrieved candidate.

        Re-truncates to `top_k` after merging -- a bare, multi-Act-ambiguous
        number (confirmed live: "Section 2" exact-matches 36 chunks across
        10 different Acts) would otherwise silently grow `results` past the
        caller's own candidate limit, which is exactly the "silently
        increase candidate limits" failure mode this phase was told to
        avoid. Since every exact match is floored well above any ordinary
        retrieval score, a plain re-sort-and-truncate keeps the exact
        matches (whichever `top_k` of them there are) and only drops the
        weakest NON-matching candidates -- it never drops an exact match in
        favor of one that isn't.

        Named-Act disambiguation (this addition): the "different Act, same
        number" ambiguity documented two paragraphs up was flagged but left
        unresolved by design -- until a query like "BNS 318" makes the Act
        explicit, at which point leaving it unresolved is a real, confirmed
        bug, not a documented gap. Live before this fix: "BNS 318" exact-
        matched BOTH BNS Section 318 ("Cheating") AND BNSS Section 318
        ("Record in High Court") and floored them to the IDENTICAL score
        (0.6175 == 0.6175 post-rerank) -- the final answer was only correct
        because the LLM happened to prefer the right chunk while generating
        text, not because retrieval ever picked one. When
        `parse_section_lookup_act` resolves a named Act to a corpus
        `act_name` this corpus actually has, `find_by_section_number` is
        scoped to that Act ALONE for this call -- the other Act's same-
        numbered chunk is simply never returned by it, so it never enters
        the floor loop below and keeps its low raw fused score instead of
        tying. Falls back to the unscoped, Act-agnostic lookup (identical to
        today's behavior) whenever no Act is named, or a named Act maps to
        nothing this corpus indexes -- never a hard failure, never a
        candidate-limit change of its own.

        Soft multi-Act tie-break (this addition): when the query names no
        Act at all (`act_abbreviation` is None) and the exact matches still
        span more than one distinct `act_name`, the prior behavior floored
        every one of them to the IDENTICAL `_EXACT_SECTION_SCORE_FLOOR`,
        leaving the choice among them to whatever the LLM happened to prefer
        while generating text -- not a retrieval decision at all.
        `_find_act_hint` checks the query text and a caller-supplied
        `context_hint` (recent conversation text) for a literal Act
        name/synonym and adds `_ACT_HINT_SCORE_BONUS` to matches from that
        Act only, never excluding the others. No hint found (the fully bare,
        no-context case) intentionally falls through to the exact prior
        flat-tie behavior, covered by
        `test_unnamed_act_bare_number_lookup_is_unchanged` -- there is no
        reliable signal to prefer one Act over another at that point, so
        this deliberately does not invent one (e.g. from raw fusion score,
        which is noise at this margin, not a real preference signal).
        """
        normalized_query = query.lower().strip()
        section_number = parse_section_lookup(normalized_query)
        if not section_number:
            return results
        act_abbreviation = parse_section_lookup_act(normalized_query)
        section_filters = filters
        if act_abbreviation:
            metadata_names = _ACT_ABBREVIATION_TO_METADATA_NAMES.get(act_abbreviation)
            if metadata_names:
                section_filters = {**filters, "act_name": metadata_names}
        exact_matches = await self.vector_store.find_by_section_number(section_number, section_filters)
        if not exact_matches and section_filters is not filters:
            # The named Act had no corpus match under any known naming
            # variant -- fall back to the Act-agnostic lookup rather than
            # silently returning nothing for a section that might still be
            # findable under a different Act.
            exact_matches = await self.vector_store.find_by_section_number(section_number, filters)
        if not exact_matches:
            return results
        act_hint = None
        distinct_acts = {match.metadata.get("act_name") for match in exact_matches}
        genuinely_ambiguous = not act_abbreviation and len(distinct_acts) > 1
        if genuinely_ambiguous:
            act_hint = _find_act_hint(query, context_hint)
        # Still ambiguous even after the soft hint above -- no code here can
        # safely guess, so the ONLY safe signal is telling the caller
        # (`ChatService._section_lookup_ambiguous_acts`) so it can ask the
        # user directly instead of silently answering for a possibly-wrong
        # Act. Tags every tied match's own metadata (rides through
        # reranking, which only ever touches `.score`) rather than changing
        # this function's return type -- `retrieve()` has three other call
        # sites (`search_service.py`, a standalone script, this file's own
        # tests) that would all need updating for a wider return tuple.
        ambiguous_act_names = (
            sorted(str(name) for name in distinct_acts if name)
            if genuinely_ambiguous and not act_hint
            else None
        )
        by_id = {chunk.chunk_id: chunk for chunk in results}
        hinted_chunk_ids: set[str] = set()
        for match in exact_matches:
            floor = _EXACT_SECTION_SCORE_FLOOR
            # `act_name` is not reliable on every chunk -- some documents
            # (confirmed live: the NI Act's own Section 138 chunk) carry a
            # mis-extracted `act_name` like "Notwithstanding anything
            # contained in the Code", a nearby sentence fragment rather than
            # the real Act name. `source_document` (the ingested filename)
            # is the one field this whole pipeline always sets correctly, so
            # it is checked as a second signal -- underscores/hyphens folded
            # to spaces so "Negotiable Instruments Act" matches
            # "Negotiable_Instruments_Act_1881_Complete_Act.pdf".
            act_name = (match.metadata.get("act_name") or "").lower()
            source_document = re.sub(r"[_\-.]", " ", match.metadata.get("source_document") or "").lower()
            # `act_hint and (...)`, wrapped in `bool()`, rather than
            # `bool(act_hint) and (...)` -- mypy narrows `act_hint` to `str`
            # inside the `and`'s right-hand side for the bare-variable form,
            # but not through an opaque `bool(act_hint)` call. Truthiness
            # (empty string treated the same as `None`) is unchanged either way.
            is_hinted = bool(act_hint and (act_hint.lower() in act_name or act_hint.lower() in source_document))
            if is_hinted:
                floor += _ACT_HINT_SCORE_BONUS
                hinted_chunk_ids.add(match.chunk_id)
            existing = by_id.get(match.chunk_id)
            target = existing if existing is not None else match
            if ambiguous_act_names:
                target.metadata = {**target.metadata, "ambiguous_acts": ambiguous_act_names}
            if existing is not None:
                existing.score = max(existing.score, floor)
            else:
                match.score = floor
                results.append(match)
                by_id[match.chunk_id] = match
        # A hint bonus that only nudges `.score` cannot break a tie where
        # both candidates already sit at the reranker's own 1.0 ceiling
        # (confirmed live: NI Act and an unrelated Maharashtra Act's Section
        # 138 chunks both reached 1.0 before this floor ever applied, so
        # `max(existing.score, floor)` left them identical regardless of the
        # bonus). Sorting on the hint as an explicit secondary key breaks the
        # tie directly, independent of what the absolute scores turned out
        # to be.
        results.sort(key=lambda chunk: (chunk.score, chunk.chunk_id in hinted_chunk_ids), reverse=True)
        return results[:top_k]

    async def _ensure_fir_section_present(
        self, results: list[RetrievedChunk], filters: dict[str, Any], top_k: int
    ) -> list[RetrievedChunk]:
        """A FIR/Zero-FIR concept query ("FIR kya hota hai?", "FIR aur
        complaint mein kya farq hota hai?") carries no digits at all, so
        `_apply_section_number_floor`'s own `parse_section_lookup` never
        fires for it -- that path only activates for a literal "Section
        <N>"-shaped query. Confirmed live: even with the query rewriter's
        own "BNSS Section 173" text expansion (`FIR_CONCEPT_RE`'s docstring
        in `query_rewriter.py`), BNSS Section 173 -- the actual FIR-
        registration provision -- never enters the top-8 similarity-fused
        candidates, so the LLM's (correct) citation of "Section 173" then
        fails `answer_quality.evaluate`'s section-grounding check and the
        whole answer is discarded for the generic "no verified document"
        refusal. An exact metadata lookup, scoped to BNSS via the corrected
        `_ACT_ABBREVIATION_TO_METADATA_NAMES["BNSS"]` list above, sidesteps
        similarity ranking entirely -- mirroring `_apply_section_number_
        floor`'s own technique, just triggered by a known CONCEPT rather
        than a literal number in the query.
        """
        return await self._ensure_sections_present(
            results, filters, top_k, [FIR_BNSS_SECTION_NUMBER], _ACT_ABBREVIATION_TO_METADATA_NAMES["BNSS"]
        )

    async def _ensure_sections_present(
        self, results: list[RetrievedChunk], filters: dict[str, Any], top_k: int,
        section_numbers: list[str], act_names: list[str],
    ) -> list[RetrievedChunk]:
        """Shared merge step behind `_ensure_fir_section_present` and the
        BNS-offence companion-section check below: an exact metadata lookup
        for each of `section_numbers` (scoped to `act_names`), merged into
        `results` rather than replacing it -- an exact match already present
        keeps its slot with its score floored; a new one is inserted and the
        weakest non-matching candidate is the one that falls off `top_k`,
        exactly as `_apply_section_number_floor` already behaves for a
        literal citation.
        """
        section_filters = {**filters, "act_name": act_names}
        by_id = {chunk.chunk_id: chunk for chunk in results}
        for section_number in section_numbers:
            exact_matches = await self.vector_store.find_by_section_number(section_number, section_filters)
            for match in exact_matches:
                existing = by_id.get(match.chunk_id)
                if existing is not None:
                    existing.score = max(existing.score, _EXACT_SECTION_SCORE_FLOOR)
                else:
                    match.score = _EXACT_SECTION_SCORE_FLOOR
                    results.append(match)
                    by_id[match.chunk_id] = match
        results.sort(key=lambda chunk: chunk.score, reverse=True)
        return results[:top_k]
