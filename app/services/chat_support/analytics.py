"""Per-turn chat analytics: one structured log line plus one persisted
analytics-DB record.

Extracted out of `app.services.chat_service.ChatService` (Phase 1 of the
god-object split) as plain functions, not a constructed collaborator class,
because `test_multi_turn_conversations.py` reassigns `ChatService.query_log`
*after* construction (`service.query_log = _NoopRepository()`) -- a
collaborator built once in `__init__` would keep writing through the
original repository and silently break that test. `ChatService._log_query`
reads `self.query_log` fresh on every call and passes it in here explicitly.
"""
from typing import Any

import structlog
from fastapi import BackgroundTasks

from app.core.config import settings
from app.repositories.analytics import QueryLogRepository
from app.schemas.chat import ChatRequest, ChatResponse
from app.utils.pii import mask_entities, mask_pii

log = structlog.get_logger(__name__)


def log_routing_decision(
    session_id: str,
    message_id: str,
    question: str,
    conversation_intent: str,
    route: str,
    *,
    memory_hit: bool,
    rag_used: bool,
    response_modification: bool,
    draft_mode: bool,
    request_failed: bool = False,
) -> None:
    """Part 30 "Memory-First Routing Engine": one structured log line per
    request, tagging which branch of `answer()`/`answer_stream()` handled
    it -- `route` is one of `draft_workflow`, `general_conversation`,
    `lawyer_recommendation`, `translation`, `response_modification`,
    `conversation_memory`, `conversation_summary`, `cache_hit`,
    `no_verified_context`, or `rag`.

    Exists specifically so "why did this turn hit RAG?" or "how many
    turns never needed retrieval?" can be answered by grepping/querying
    logs instead of re-deriving the decision from the full pipeline --
    `log_query`'s analytics-DB record captures similar fields but isn't
    a queryable application log line, and is fired-and-forgotten via
    `background_tasks` rather than emitted synchronously here.

    `request_failed` (Part 31 "State Manager & Recovery Engine") is
    `True` only for the `rag` route, where a genuine LLM-call failure is
    possible and tracked -- `no_verified_context` never calls the LLM at
    all, and every other route either can't fail this way or already has
    its own failure handling (`safe_llm_text`'s fallback), so it
    defaults to `False` rather than requiring every call site to reason
    about it.
    """
    log.info(
        "chat_routing_decision",
        session_id=session_id,
        message_id=message_id,
        # Part 58 "Answer Quality Audit" issue 25: this log line is
        # emitted on every single turn and retained wherever logs go.
        # Drafting turns put a complainant's full postal address, mobile
        # number, bank name and UTR reference straight into the message
        # text, so the raw question was accumulating exactly that, in the
        # clear, for a far wider audience than the conversation itself.
        # `message_id`/`session_id` still tie the line back to the full
        # record for anyone who genuinely needs it.
        question=mask_pii(question)[:200],
        conversation_intent=conversation_intent,
        route=route,
        memory_hit=memory_hit,
        rag_used=rag_used,
        response_modification=response_modification,
        draft_mode=draft_mode,
        request_failed=request_failed,
    )


async def log_query(
    query_log: QueryLogRepository,
    request: ChatRequest,
    session_id: str,
    message_id: str,
    response: ChatResponse,
    background_tasks: BackgroundTasks | None,
    cache_hit: str | None = None,
    owner_user_id: str | None = None,
) -> None:
    """Persists one analytics record per chat turn to `QUERY_LOGS`.

    Feeds the admin knowledge-improvement analytics (FAQs, knowledge gaps,
    low-confidence/low-retrieval-quality responses); `/feedback` later
    attaches rating/comment to this same document via `message_id`.
    Fire-and-forget like `memory.summarize_if_needed` -- must never add
    latency to, or fail, the user-facing response.
    """
    entry: dict[str, Any] = {
        "_id": message_id,
        "message_id": message_id,
        "session_id": session_id,
        "conversation_id": request.conversation_id,
        # Security finding C2: `owner_user_id` (server-verified, from
        # `memory["owner_user_id"]`) not `request.user_id` (client-supplied
        # and never actually trustworthy) -- see the identical fix and
        # comment on `ChatRepository().insert(...)` in chat_service.py.
        # `erase_user_data`'s `QueryLogRepository().delete_by_user(user_id)`
        # matched nothing for a real account until this was fixed.
        "user_id": owner_user_id,
        # Part 58 issue 25: QUERY_LOGS is an unbounded, cross-session
        # analytics collection read by the admin dashboard (FAQs,
        # knowledge gaps, low-confidence review) -- none of which needs
        # the complainant's address, phone number or transaction ids, all
        # of which a drafting turn puts in `question` verbatim. The
        # redaction is shape-preserving, so grouping/counting on these
        # fields still works.
        "question": mask_pii(request.question),
        "answer": mask_pii(response.answer),
        "language": response.detected_language,
        "intent": response.detected_intent,
        "conversation_intent": response.conversation_intent,
        "entities": mask_entities(response.extracted_entities),
        "confidence": response.confidence,
        "confidence_label": response.confidence_label,
        "retrieved_chunks": [chunk.model_dump() for chunk in response.retrieved_chunks],
        "knowledge_sources": sorted({source.source_document for source in response.sources}),
        "llm_provider": response.llm_provider,
        "llm_model": response.llm_model,
        "prompt_version": settings.prompt_version,
        "latency_ms": response.latency_ms,
        "cache_hit": cache_hit,
        "rating": None,
        "feedback_comment": None,
    }
    if background_tasks is not None:
        background_tasks.add_task(query_log.insert, entry)
    else:
        await query_log.insert(entry)
