import re
import time
from typing import Any

from app.rag.kb_jurisdiction import shared_retrieval_filters
from app.rag.retriever import LegalRetriever
from app.schemas.search import SearchRequest, SearchResponse

# QA session 2026-09-24 ("BUG-109"/"BUG-111"): `SearchRequest.mode` used to be
# accepted by the schema but never actually read anywhere in this service --
# every mode (semantic/keyword/metadata/section/act) silently ran the exact
# same hybrid retrieval, with no indication to the caller that their explicit
# mode choice had zero effect. Live-reproduced: `mode="act"` with
# `query="Negotiable Instruments Act"` returned a completely unrelated Act
# (score 0.016, pure noise) even though the Negotiable Instruments Act is
# well-indexed and correctly retrievable via `mode="keyword"` in the same
# corpus; `mode="section"` with a bare `query="173"` returned an unrelated
# CGST Act section rather than the contextually-likely BNSS Section 173.
# `section`/`act` now route through the vector store's own exact-metadata
# lookups (`find_by_section_number`, an `act_name` filter via the same
# `_mongo_filter`/`_atlas_filter` mechanism every other filtered search in
# this app already uses) instead of plain similarity ranking on the bare
# number/name -- falling back to ordinary hybrid retrieval whenever the exact
# lookup finds nothing, so this can only ever ADD a more targeted result, never
# remove the previous (already-shipped) hybrid-search behavior for a query
# that isn't a clean bare section number or exact Act name.
_BARE_SECTION_NUMBER_RE = re.compile(r"^\s*(?:section|sec\.?|§)?\s*(\d+[A-Za-z]?(?:\(\d+\))?)\s*$", re.IGNORECASE)

# Security finding C10: `POST /search` takes no authentication at all (see
# the route's own docstring/decision) and `SearchRequest.filters` is an
# arbitrary, fully client-controlled `dict[str, Any]` -- passed straight
# through to `LegalRetriever.retrieve` unsanitized, it let an anonymous
# caller both DEFEAT the shared-corpus safety gate (`LegalRetriever.
# retrieve` only applies its own safe `review_status` default via
# `setdefault`, which never fires once the caller has already supplied that
# key -- a request could set `filters={"review_status": [...]}` to whatever
# it wanted, including unreviewed/rejected content) and attempt to target a
# SPECIFIC private scope directly (`filters={"owner_user_id": "..."}` or
# `{"owner_session_id": "..."}`), neither of which this public, unauthenticated
# endpoint may ever expose. Governance/ownership fields are never something a
# search client legitimately needs to set -- they describe WHO may see a
# chunk, not what it is about -- so they are stripped unconditionally rather
# than validated, keeping this endpoint's own retrieval scope identical to
# every other genuinely public, anonymous-eligible lookup in the app.
_FORBIDDEN_FILTER_KEYS = frozenset({
    "review_status", "owner_user_id", "owner_session_id", "document_status", "$or",
})


def _sanitize_public_filters(filters: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in filters.items() if key not in _FORBIDDEN_FILTER_KEYS}


class SearchService:
    def __init__(self) -> None:
        self.retriever = LegalRetriever()

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        filters = _sanitize_public_filters(request.filters)
        results: list[Any] = []

        if request.mode in ("section", "act"):
            # `find_by_section_number`/`_find_by_act_name` are exact metadata
            # lookups, not full `LegalRetriever.retrieve()` calls -- they skip
            # straight past the `review_status` gate `retrieve()` normally
            # applies via `filters.setdefault(...)` before reaching the vector
            # store. Applying the exact same shared/public-eligible-only
            # constraint here (never overriding a caller-supplied value, same
            # `setdefault` semantics `retrieve()` itself uses) keeps these two
            # exact-lookup shortcuts from exposing unreviewed/rejected KB
            # content that ordinary hybrid search would have excluded.
            exact_lookup_filters = dict(filters)
            if "$or" not in exact_lookup_filters:
                exact_lookup_filters.setdefault("review_status", shared_retrieval_filters()["review_status"])
            if request.mode == "section":
                bare_match = _BARE_SECTION_NUMBER_RE.match(request.query)
                if bare_match:
                    results = await self.retriever.vector_store.find_by_section_number(
                        bare_match.group(1), exact_lookup_filters
                    )
            elif request.query.strip():
                results = await self._find_by_act_name(request.query.strip(), exact_lookup_filters)

        if results:
            return SearchResponse(
                query=request.query, rewritten_query=request.query, results=results[: request.top_k],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # BUG (§7 item 5, `QA_REPORT_100Q_RETEST_20260924.md`): `mode="semantic"`/
        # `"keyword"` are the two modes `MongoVectorStore.search` gives a real,
        # distinct single-leg behavior to -- everything else (including the
        # unhandled `"metadata"` value and the default `"hybrid"`) keeps
        # running the same two-leg RRF-fused search it always has.
        retrieval_mode = request.mode if request.mode in ("semantic", "keyword") else "hybrid"
        rewritten, results = await self.retriever.retrieve(
            request.query, top_k=request.top_k, filters=filters, mode=retrieval_mode
        )
        return SearchResponse(
            query=request.query,
            rewritten_query=rewritten,
            results=results,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def _find_by_act_name(self, act_name: str, filters: dict[str, Any]) -> list[Any]:
        """`act_name` metadata lookup for `mode="act"` -- reuses
        `MongoVectorStore._mongo_filter`'s existing, already-tested filter
        translation (the same one every ownership/governance filter in this
        app already goes through) rather than inventing a second filter
        builder. Tries the name as given, then "The <name>" (the corpus's
        other common prefix form, same pattern `LegalRetriever.retrieve`'s
        own named-section handling already relies on for Act name variants).

        Falls back to a `source_document` filename match (the same
        "significant tokens" approach `MongoVectorStore.find_named_section`
        already uses to identify an Act) when the exact `act_name` match
        returns too few chunks to be a useful result on its own. Necessary,
        not a nice-to-have: `act_name` is extracted from document text at
        ingest time and is unreliable for exactly this reason across this
        corpus (see `app/rag/citation.py`'s own extensive documentation of
        the same noise -- "None", "Repealing and Amending Act", garbled
        OCR spellings, ...). Confirmed live: only 1 of 47
        `Negotiable_Instruments_Act_1881_Complete_Act.pdf` chunks carries
        the clean literal `act_name` "Negotiable Instruments Act"; an
        exact-match-only lookup found that one chunk and stopped, when the
        other 46 -- identified just as reliably by their shared source
        filename -- were sitting right there.
        """
        from app.database.mongodb import mongodb
        from app.models.collections import EMBEDDINGS_METADATA
        from app.schemas.common import RetrievedChunk

        # A result set this small isn't trustworthy as "the Act's own
        # content" on its own -- more likely a single mislabeled chunk from
        # an unrelated document than genuine coverage. Falls through to the
        # filename-based match below instead of returning it as-is.
        _MIN_TRUSTED_EXACT_MATCHES = 3

        store = self.retriever.vector_store
        collection = mongodb.db[EMBEDDINGS_METADATA]
        exact_found: list[Any] = []
        for candidate in (act_name, f"The {act_name}"):
            mongo_filter = store._mongo_filter({**filters, "act_name": candidate})  # type: ignore[attr-defined]
            mongo_filter["metadata.document_status"] = "active"
            found = [
                RetrievedChunk(
                    chunk_id=str(item["_id"]), text=item.get("text", ""),
                    score=0.0, metadata=item.get("metadata", {}),
                )
                async for item in collection.find(mongo_filter).limit(50)
            ]
            if len(found) >= _MIN_TRUSTED_EXACT_MATCHES:
                return found
            exact_found = exact_found or found

        significant_tokens = [
            token for token in re.findall(r"[a-zA-Z0-9]+", act_name.lower())
            if token not in {"the", "act", "of", "and", "for"} and len(token) > 2
        ]
        if significant_tokens:
            filename_pattern = ".*".join(re.escape(token) for token in significant_tokens)
            mongo_filter = store._mongo_filter(dict(filters))  # type: ignore[attr-defined]
            mongo_filter["metadata.document_status"] = "active"
            mongo_filter["metadata.source_document"] = {"$regex": filename_pattern, "$options": "i"}
            by_filename = [
                RetrievedChunk(
                    chunk_id=str(item["_id"]), text=item.get("text", ""),
                    score=0.0, metadata=item.get("metadata", {}),
                )
                async for item in collection.find(mongo_filter).limit(50)
            ]
            if by_filename:
                return by_filename

        return exact_found
