from typing import Any

from app.rag.kb_jurisdiction import TEMPORAL_FILTER_KEY, chunk_temporally_eligible


def matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """In-memory filter predicate shared by every retrieval leg that scores
    candidates outside MongoDB's own query engine (`BM25Index.search`,
    `LocalAnnIndex.search`) rather than expressing filters as a Mongo query
    (`MongoVectorStore._mongo_filter`/`_atlas_filter`).

    Extracted verbatim from `BM25Index._matches_filters` (previously the only
    copy of this logic) so a second in-memory leg cannot silently drift from
    it -- ownership/`$or`/list-membership/temporal-eligibility semantics stay
    identical across every leg a query might route through.
    """
    or_branches = filters.get("$or")
    if or_branches and not any(matches_filters(metadata, branch) for branch in or_branches):
        return False
    as_of_date = filters.get(TEMPORAL_FILTER_KEY)
    if as_of_date and not chunk_temporally_eligible(metadata, as_of_date):
        return False
    for key, value in filters.items():
        if key in ("$or", TEMPORAL_FILTER_KEY) or not value:
            continue
        actual = metadata.get(key)
        if isinstance(value, (list, tuple)):
            if actual not in value:
                return False
        elif isinstance(actual, (list, tuple)):
            if value not in actual:
                return False
        elif actual != value:
            return False
    return True
