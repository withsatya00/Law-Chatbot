import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user_id
from app.memory.store import ConversationMemoryStore
from app.repositories.chat_history import ChatRepository
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, background_tasks: BackgroundTasks,
    user_id: str | None = Depends(get_current_user_id),
) -> ChatResponse:
    return await ChatService().handle_turn(request, background_tasks, authenticated_user_id=user_id)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest, background_tasks: BackgroundTasks,
    user_id: str | None = Depends(get_current_user_id),
) -> StreamingResponse:
    """Part 28 Step 4: additive streaming endpoint alongside the unchanged
    `/chat` above -- existing callers of `/chat` are unaffected.

    Server-Sent Events: zero or more `event: token` frames as text becomes
    available, followed by exactly one `event: done` frame carrying the full
    `ChatResponse` JSON (sources, confidence, draft info, etc.) for the
    client to attach once the answer is complete.
    """
    service = ChatService()

    async def event_source() -> AsyncIterator[str]:
        async for event in service.handle_turn_stream(request, background_tasks, authenticated_user_id=user_id):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.delete("/chat")
async def delete_chat(session_id: str, user_id: str | None = Depends(get_current_user_id)) -> dict[str, str | int]:
    """Was previously a no-op that read/deleted nothing (see
    [[project_orphaned_prototype_cluster]]). Reuses `DELETE /session`'s
    ownership gate (`app/api/history.py`) but goes further -- that route
    only ever clears Redis/long-term conversation MEMORY, never the
    persisted question/answer log itself (`ChatRepository`/`CHATS`), which
    is what this endpoint's name actually promises.
    """
    await ConversationMemoryStore().check_access(session_id, user_id)
    await ConversationMemoryStore().clear(session_id)
    # Security/correctness finding C7: counts actual user+assistant messages
    # (see `ChatRepository.delete_by_session_counting_messages`'s own
    # docstring), not raw turn-document rows -- the two differ by roughly
    # 2x since each stored row holds both sides of one turn.
    deleted_messages = await ChatRepository().delete_by_session_counting_messages(session_id)
    return {"status": "deleted", "session_id": session_id, "messages_deleted": deleted_messages}
