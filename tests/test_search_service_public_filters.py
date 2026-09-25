"""Regression tests for security finding C10 (POST /search's client-supplied
`filters` could defeat the review-status gate or target a private scope).

`POST /search` is intentionally public/unauthenticated -- a plain lookup
over the shared, review-approved Knowledge Base. `SearchRequest.filters` is
an arbitrary client-controlled dict, though, and was passed straight
through to `LegalRetriever.retrieve` unsanitized: `retrieve` only applies
its own safe `review_status` default via `setdefault`, which never fires if
the caller already supplied that key, so a request could ask for
unreviewed/rejected content directly, or attempt to target a specific
private owner's documents via `owner_user_id`/`owner_session_id`. Neither is
something a public, unauthenticated endpoint may ever expose.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.schemas.search import SearchRequest
from app.services.search_service import SearchService, _sanitize_public_filters


def test_review_status_override_is_stripped() -> None:
    filters = {"review_status": ["needs_review", "rejected"]}
    assert "review_status" not in _sanitize_public_filters(filters)


def test_owner_scoped_filters_are_stripped() -> None:
    filters = {"owner_user_id": "victim-user-id", "owner_session_id": "victim-session-id"}
    sanitized = _sanitize_public_filters(filters)
    assert "owner_user_id" not in sanitized
    assert "owner_session_id" not in sanitized


def test_a_raw_or_clause_cannot_be_injected() -> None:
    """`retrieve`'s own safe default is skipped entirely when `$or` is
    already present in `filters` -- a public caller must never be able to
    construct that shape itself."""
    filters = {"$or": [{"owner_user_id": None}, {"owner_user_id": "victim-user-id"}]}
    assert "$or" not in _sanitize_public_filters(filters)


def test_legitimate_content_shaped_filters_survive() -> None:
    """The sanitizer must not become a de facto ban on filtering at all --
    ordinary content-shape filters (what a chunk is ABOUT) are unaffected."""
    filters = {"act_name": ["Bharatiya Nyaya Sanhita"], "section_number": "302", "language": "hindi"}
    assert _sanitize_public_filters(filters) == filters


def test_search_service_never_forwards_forbidden_keys_to_the_retriever() -> None:
    """End-to-end through `SearchService.search`, not just the sanitizer in
    isolation -- proves the actual call site uses it."""
    service = SearchService()
    service.retriever.retrieve = AsyncMock(return_value=("query", []))
    request = SearchRequest(
        query="what is section 302",
        filters={"review_status": ["needs_review"], "owner_user_id": "victim-user-id", "act_name": "BNS"},
    )

    asyncio.run(service.search(request))

    forwarded_filters = service.retriever.retrieve.call_args.kwargs["filters"]
    assert "review_status" not in forwarded_filters
    assert "owner_user_id" not in forwarded_filters
    assert forwarded_filters == {"act_name": "BNS"}
