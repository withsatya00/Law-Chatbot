from fastapi import APIRouter

from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import SearchService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Security finding C10: deliberately public/unauthenticated, by design
    -- this is a plain lookup over the SHARED, review-approved Knowledge
    Base (the same corpus `/chat` cites from), analogous to a public
    statute-search feature, not an account- or session-scoped action. It
    must never be able to see anything a genuinely private endpoint (a
    user's own uploads, an unreviewed/rejected KB candidate) would gate --
    `SearchService.search`'s `_sanitize_public_filters` enforces that by
    stripping any governance/ownership filter key a caller supplies, rather
    than trusting the caller not to send one.
    """
    return await SearchService().search(request)
