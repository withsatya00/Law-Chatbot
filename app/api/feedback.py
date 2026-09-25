from fastapi import APIRouter

from app.core.exceptions import ForbiddenError, NotFoundError
from app.repositories.analytics import QueryLogRepository
from app.repositories.chat_history import FeedbackRepository
from app.schemas.history import FeedbackRequest

router = APIRouter(tags=["feedback"])


@router.post("/feedback")
async def feedback(request: FeedbackRequest) -> dict[str, str]:
    if request.message_id:
        # Verify the target message actually belongs to this session before
        # attaching feedback to it -- previously any caller could rate/
        # comment on any message by guessing/enumerating `message_id`, since
        # `attach_feedback` did an unconditional update-by-id with no
        # ownership check at all.
        query_log = QueryLogRepository()
        entry = await query_log.find_by_id(request.message_id)
        if entry is None:
            raise NotFoundError(f"Message not found: {request.message_id}")
        if entry.get("session_id") != request.session_id:
            raise ForbiddenError("You do not have access to this message.")
        if request.category:
            await query_log.attach_feedback(request.message_id, request.rating, request.comment, request.category)
        else:
            # Preserve the Phase 1 call shape for integrations that wrap this
            # repository method while allowing Phase 3 categorical feedback.
            await query_log.attach_feedback(request.message_id, request.rating, request.comment)
    feedback_id = await FeedbackRepository().insert(request.model_dump())
    return {"status": "recorded", "feedback_id": feedback_id}
