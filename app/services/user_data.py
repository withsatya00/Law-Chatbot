"""Erasing a user's own data, in the service layer.

Two operations that must behave identically whether they were asked for as a
REST call or as a sentence in the chat ("delete everything you have about
me"). They were implemented inside the route bodies in `app/api/history.py`;
a chat workflow calling a route is not an option, and a second
implementation of "erase everything" is the last thing this codebase should
have -- a divergence there means data the user believes is gone.

Both act on an identity the CALLER already proved. Neither takes a target
from anything a user typed.
"""

from pathlib import Path
from typing import Any

import structlog

from app.core.config import settings
from app.database.mongodb import mongodb
from app.memory.store import ConversationMemoryStore
from app.models.collections import (
    BACKGROUND_JOBS,
    CASES,
    DOWNLOAD_ARTIFACTS,
    EMBEDDINGS_METADATA,
    FORM_WORKFLOWS,
    USER_PREFERENCES,
)
from app.repositories.analytics import IntentEventRepository, QueryLogRepository
from app.repositories.chat_history import ChatRepository, FeedbackRepository
from app.repositories.conversation_memory import ConversationMemoryRepository
from app.repositories.drafts import DraftRepository, DraftVersionRepository

log = structlog.get_logger(__name__)


async def erase_session_data(session_id: str) -> dict[str, object]:
    """Erases everything this session left behind, not just its working memory.

    Clearing only `conversation_memory` left the full transcript in `chats`
    and the message text in the two analytics collections -- including, for a
    drafting session, a postal address, mobile number, bank name and
    transaction references. "Delete my session" has to mean deleted.

    The count of removed records is returned per collection so a caller can
    show the user what was actually erased rather than an unverifiable "done".
    """
    await ConversationMemoryStore().clear(session_id)
    deleted = {
        "chat_messages": await ChatRepository().delete_by_session(session_id),
        "query_logs": await QueryLogRepository().delete_by_session(session_id),
        "intent_events": await IntentEventRepository().delete_by_session(session_id),
    }
    log.info("session_data_erased", session_id=session_id, **deleted)
    return {"status": "deleted", "session_id": session_id, "deleted": deleted}


async def erase_user_data(user_id: str) -> dict[str, object]:
    """Erases everything this account has accumulated, across every session.

    `erase_session_data` clears one conversation; this clears the account.
    The two are separate on purpose -- deleting a session is routine tidying,
    deleting an account's data is not, and conflating them would make the
    routine action alarmingly destructive.

    Always acts on the id it is given by an authenticated caller; there is
    deliberately no way to point it at a third party's data.

    Security finding C2: this previously reported success while leaving
    chat messages, query-log/intent-event analytics, feedback, and this
    account's own conversation memory/facts fully intact and readable --
    two independent root causes, both fixed here:

    1. `chats`/`query_logs` were stamped with the client-supplied
       `ChatRequest.user_id`, never the server-verified identity, so a
       `delete_many({"user_id": <real id>})` matched nothing; `chats`.
       `intent_events`/`feedback` never carried a `user_id` field at all.
       (`ChatService`/`app/services/chat_support/analytics.py` now stamp
       the real, JWT-verified owner -- see those modules' own comments.)
    2. Even with that fixed going forward, older rows -- and anything
       written while a since-claimed session was still anonymous -- are
       only reachable via `session_id` membership, not `user_id`. This
       resolves every session this account has ever claimed once
       (`ConversationMemoryRepository.list_session_ids_by_owner`) and
       matches on EITHER field (`MongoRepository.delete_by_owner`) so
       account-wide erasure is not silently partial for historical data.

    Also newly cleared here: this account's own `conversation_memory`
    documents (sessions/confirmed facts/assumptions -- previously untouched
    by this route entirely, only ever removed one at a time via `DELETE
    /session`) and its private uploaded documents/indexed chunks (via
    `DocumentService.delete_owned`, the same ownership-checked path used
    everywhere else a document is deleted, so this never diverges into a
    second, independently-written deletion query).
    """
    from app.services.document_service import DocumentService

    memory_repo = ConversationMemoryRepository()
    owned_session_ids = await memory_repo.list_session_ids_by_owner(user_id)

    sessions_deleted = 0
    for session_id in owned_session_ids:
        await ConversationMemoryStore().clear(session_id)
        sessions_deleted += 1

    documents = DocumentService()
    owned_documents = await documents.list_owned(user_id=user_id)
    documents_deleted = 0
    chunks_deleted = 0
    for item in owned_documents:
        chunks_deleted += await documents.delete_owned(item["document_id"], authenticated_user_id=user_id)
        documents_deleted += 1
    # Chunks indexed for this account outside a discrete uploaded-document
    # record (rare, but the ownership field is authoritative regardless of
    # which path wrote it) -- cleared directly so no chunk tagged to this
    # account survives account erasure.
    embeddings_result = await mongodb.db[EMBEDDINGS_METADATA].delete_many({"metadata.owner_user_id": user_id})

    drafts = DraftRepository()
    owned_draft_ids = [str(draft["_id"]) for draft in await drafts.list_for_user(user_id, limit=10_000)]
    versions_deleted = 0
    if owned_draft_ids:
        versions_deleted = await DraftVersionRepository().delete_for_drafts(owned_draft_ids)
    deleted: dict[str, Any] = {
        "sessions": sessions_deleted,
        "chat_messages": await ChatRepository().delete_by_owner(user_id, owned_session_ids),
        "query_logs": await QueryLogRepository().delete_by_owner(user_id, owned_session_ids),
        "intent_events": await IntentEventRepository().delete_by_owner(user_id, owned_session_ids),
        "feedback": await FeedbackRepository().delete_by_owner(user_id, owned_session_ids),
        "drafts": await drafts.delete_by_user(user_id),
        "draft_versions": versions_deleted,
        "uploaded_documents": documents_deleted,
        "indexed_chunks": chunks_deleted + embeddings_result.deleted_count,
    }
    artifacts = [item async for item in mongodb.db[DOWNLOAD_ARTIFACTS].find({"owner_user_id": user_id})]
    allowed_root = settings.draft_output_dir.resolve()
    artifact_files_deleted = 0
    for item in artifacts:
        path = Path(item.get("path", "")).resolve()
        if allowed_root in path.parents and path.is_file():
            path.unlink(missing_ok=True)
            artifact_files_deleted += 1
    for key, collection in {
        "cases": CASES,
        "preferences": USER_PREFERENCES,
        "form_workflows": FORM_WORKFLOWS,
        "background_jobs": BACKGROUND_JOBS,
        "download_artifacts": DOWNLOAD_ARTIFACTS,
    }.items():
        result = await mongodb.db[collection].delete_many({"owner_user_id": user_id})
        deleted[key] = result.deleted_count
    deleted["download_files"] = artifact_files_deleted
    log.info("user_data_erased", user_id=user_id, **deleted)
    return {"status": "deleted", "deleted": deleted}
