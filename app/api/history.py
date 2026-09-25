from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, Header, Response

from app.api.deps import get_current_user_id
from app.core.exceptions import UnauthorizedError
from app.memory.store import ConversationMemoryStore
from app.repositories.chat_history import ChatRepository
from app.schemas.history import HistoryResponse, MissingDetailsRequest, SessionFactRequest
from app.services.user_data import erase_session_data, erase_user_data

router = APIRouter(tags=["history"])
log = structlog.get_logger(__name__)

# Security finding C1: the credential an anonymous session's owner must
# present on every call after the one that created it. Minted by
# `ConversationMemoryStore.claim_or_verify_ownership` and returned via this
# same header name so a client only ever has to read/send one thing.
OWNER_TOKEN_HEADER = "X-Session-Owner-Token"


def _owner_token(
    x_session_owner_token: str | None = Header(default=None, alias=OWNER_TOKEN_HEADER),
) -> str | None:
    return x_session_owner_token


async def _claim(
    session_id: str, user_id: str | None, owner_token: str | None, response: Response,
) -> dict[str, Any]:
    memory, minted = await ConversationMemoryStore().claim_or_verify_ownership(session_id, user_id, owner_token)
    if minted:
        response.headers[OWNER_TOKEN_HEADER] = minted
    return memory


@router.get("/history", response_model=HistoryResponse)
async def history(session_id: str, user_id: str | None = Depends(get_current_user_id)) -> HistoryResponse:
    await ConversationMemoryStore().check_access(session_id, user_id)
    messages = await ChatRepository().list_session_messages(session_id)
    compact = [
        {"role": "user", "content": message.get("question", "")}
        for message in messages
    ] + [{"role": "assistant", "content": message.get("answer", "")} for message in messages]
    return HistoryResponse(session_id=session_id, messages=compact)


@router.get("/session")
async def session(
    session_id: str, response: Response,
    user_id: str | None = Depends(get_current_user_id),
    owner_token: str | None = Depends(_owner_token),
) -> dict[str, Any]:
    memory = await _claim(session_id, user_id, owner_token, response)
    # `owner_token_hash` is internal bookkeeping for the C1 ownership gate
    # (a hash of the credential this same gate requires) -- it must never
    # be echoed back to a client, unlike the rest of this session's memory,
    # which this route already returned as-is before this fix.
    return {key: value for key, value in memory.items() if key != "owner_token_hash"}


@router.get("/session/facts")
async def session_facts(
    session_id: str, response: Response,
    user_id: str | None = Depends(get_current_user_id),
    owner_token: str | None = Depends(_owner_token),
) -> dict[str, Any]:
    memory = await _claim(session_id, user_id, owner_token, response)
    return {
        "confirmed_facts": memory.get("confirmed_facts", {}),
        "assumptions": memory.get("assumptions", {}),
        "missing_details": memory.get("missing_details", []),
        "corrections": memory.get("fact_corrections", []),
    }


@router.patch("/session/facts")
async def set_session_fact(
    session_id: str, request: SessionFactRequest, response: Response,
    user_id: str | None = Depends(get_current_user_id),
    owner_token: str | None = Depends(_owner_token),
) -> dict[str, Any]:
    store = ConversationMemoryStore()
    await _claim(session_id, user_id, owner_token, response)
    memory = await store.set_fact(session_id, request.kind, request.key, request.value)
    return {"confirmed_facts": memory["confirmed_facts"], "assumptions": memory["assumptions"]}


@router.delete("/session/facts/{kind}/{key}")
async def delete_session_fact(
    session_id: str, kind: Literal["confirmed", "assumption"], key: str, response: Response,
    user_id: str | None = Depends(get_current_user_id),
    owner_token: str | None = Depends(_owner_token),
) -> dict[str, Any]:
    store = ConversationMemoryStore()
    await _claim(session_id, user_id, owner_token, response)
    memory = await store.delete_fact(session_id, kind, key)
    return {"confirmed_facts": memory["confirmed_facts"], "assumptions": memory["assumptions"]}


@router.put("/session/missing-details")
async def set_session_missing_details(
    session_id: str, request: MissingDetailsRequest, response: Response,
    user_id: str | None = Depends(get_current_user_id),
    owner_token: str | None = Depends(_owner_token),
) -> dict[str, Any]:
    store = ConversationMemoryStore()
    await _claim(session_id, user_id, owner_token, response)
    memory = await store.set_missing_details(session_id, request.details)
    return {"missing_details": memory["missing_details"]}


@router.delete("/session")
async def delete_session(
    session_id: str, response: Response,
    user_id: str | None = Depends(get_current_user_id),
    owner_token: str | None = Depends(_owner_token),
) -> dict[str, object]:
    """Erases everything this session left behind, not just its working memory.

    Part 58 "Answer Quality Audit" issue 25: this used to clear only the
    `conversation_memory` document, so a user who deleted their session still
    had the full transcript in `chats` and their message text in the two
    analytics collections -- including, for a drafting session, their postal
    address, mobile number, bank name and transaction references. "Delete my
    session" has to mean deleted.

    The count of removed records is returned per collection so a caller can
    show the user what was actually erased rather than an unverifiable
    "done".

    Uses the strict `claim_or_verify_ownership` gate (security finding C1)
    since this is destructive and a sibling of the other `/session/*`
    routes that gate now protects -- a weaker check here would just move
    the same vulnerability to a more damaging endpoint.
    """
    await _claim(session_id, user_id, owner_token, response)
    return await erase_session_data(session_id)


@router.delete("/me/data")
async def delete_my_data(user_id: str | None = Depends(get_current_user_id)) -> dict[str, object]:
    """Phase 1 item 7 ("delete-user-data"): erase everything this account has
    accumulated, across every session it ever used.

    `DELETE /session` clears one conversation; this clears the account. The
    two are separate on purpose -- deleting a session is routine tidying,
    deleting an account's data is not, and conflating them would make the
    routine action alarmingly destructive.

    Requires authentication and always acts on the CALLER's own id -- there is
    deliberately no user_id parameter, so this route cannot be pointed at
    someone else's data.

    Conversation memory is session-scoped and expires on its own TTL
    (`session_ttl_seconds`); the collections cleared here are the ones that
    persist indefinitely.
    """
    if not user_id:
        raise UnauthorizedError("Sign in to delete your data.")
    return await erase_user_data(user_id)
