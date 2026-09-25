import hashlib
import hmac
import secrets
from datetime import UTC, datetime
from typing import Any, Literal

import structlog

from app.cache.redis_client import CACHE_UNAVAILABLE_ERRORS, redis_client
from app.core.config import settings
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.llm.base import ChatMessage, LLMProvider
from app.llm.prompts import prompt_registry
from app.repositories.conversation_memory import ConversationMemoryRepository

log = structlog.get_logger(__name__)


def _hash_owner_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_owner_token() -> str:
    """A high-entropy credential distinct from `session_id` -- security
    finding C1 requires that an anonymous session's ownership never rest on
    `session_id` alone (it is routinely visible in URLs, logs, and referrer
    headers, so knowing it must not be equivalent to owning the session)."""
    return secrets.token_urlsafe(32)

# Above this many turns, older messages are folded into `summary` and dropped from the
# raw message list so prompt size stays bounded regardless of conversation length.
SUMMARIZE_TRIGGER_MESSAGE_COUNT = 12
KEEP_RECENT_MESSAGES_AFTER_SUMMARY = 6
MAX_RAW_MESSAGES = 20


def _empty_memory() -> dict[str, Any]:
    return {
        "summary": "",
        "messages": [],
        "language_preference": None,
        "current_intent": None,
        "intent_history": [],
        "legal_category": None,
        # Every document uploaded in this conversation, oldest first, each
        # `{document_id, filename, language, uploaded_at}`. Written by
        # `DocumentService.upload_and_index`; read wherever the user refers to
        # a document by name or asks to compare two of them.
        "uploaded_documents": [],
        # Part 31 "State Manager & Recovery Engine": see `app/memory/state.py`
        # for the read side. `last_failed_question`/`last_failed_reason` are
        # set only by a genuine LLM-call failure in the RAG-tail (never
        # cleared just because a later, unrelated turn succeeds -- Rule 1
        # says a failure must not be silently dropped), and cleared only
        # when the failed question is explicitly retried.
        "last_failed_question": None,
        "last_failed_reason": None,
        "last_successful_response": None,
        # Part 35 "Entity Memory": list of `EntityFact.to_dict()` records
        # (see `app/memory/entity_memory.py`), appended to as facts are
        # extracted from each user turn -- never overwritten, so earlier
        # facts stay recallable even after the topic moves on.
        "entities": [],
        # Facts the user confirmed are kept separate from working assumptions
        # and unanswered questions.  Callers must never silently promote an
        # assumption into confirmed_facts.
        "confirmed_facts": {},
        "assumptions": {},
        "missing_details": [],
        "fact_corrections": [],
        # Part 46 "Authenticated User Ownership": `None` = unclaimed (open
        # exactly like every session today -- no forced login, no
        # migration). Set once, on the first authenticated turn on this
        # session, and never overwritten after that -- see `check_access`.
        "owner_user_id": None,
        # Security finding C1: which ownership regime this session is under.
        # `None` = never touched by `claim_or_verify_ownership` (either a
        # brand-new session, or one only ever touched by the older,
        # permissive `check_access` callers -- `/chat`, `/upload`,
        # drafting). `"user"` / `"anonymous"` are set the first time
        # `claim_or_verify_ownership` runs for this session -- see that
        # method in this class for the full model. `owner_token_hash` is the
        # SHA-256 of the anonymous credential; the plaintext is never
        # stored, only returned once, to the caller that minted it.
        "owner_state": None,
        "owner_token_hash": None,
        # Part 51 "Uploaded Document Conversation Context": the most recent
        # document this session successfully uploaded/indexed -- lets a
        # pronoun/short-reference follow-up ("explain this," "isme kya hai")
        # resolve to it without the user repeating an ID. Set only by
        # `DocumentService.upload_and_index` on success (never on a failed
        # upload), read by `ChatService._respond_with_document_analysis`.
        # Ownership is still re-verified against the document's own stored
        # metadata at analysis time (`DocumentService._ensure_document_
        # access`) -- this field is only ever a resolution HINT, never an
        # access grant on its own.
        "last_uploaded_document_id": None,
    }


class ConversationMemoryStore:
    """Two-tier conversation memory: Redis for fast reads, Mongo for durability.

    Every write lands in both tiers. Reads prefer Redis (short-term, TTL-bound) and
    fall back to Mongo (long-term) so a session's summary and preferences survive a
    Redis eviction or restart instead of silently resetting mid-conversation.
    """

    def __init__(self, memory_repository: ConversationMemoryRepository | None = None) -> None:
        self.memory_repository = memory_repository or ConversationMemoryRepository()

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:memory"

    async def load(self, session_id: str) -> dict[str, Any]:
        try:
            cached = await redis_client.get_json(self._key(session_id))
        except RuntimeError:
            cached = None
        if isinstance(cached, dict) and cached:
            return cached
        try:
            persisted = await self.memory_repository.find_by_session(session_id)
        except Exception as exc:  # noqa: BLE001 - long-term memory is an enhancement over the Redis working set; a read failure degrades to no history
            log.warning("long_term_memory_read_failed", session_id=session_id, error=str(exc))
            persisted = None
        if persisted:
            memory = {
                "summary": persisted.get("summary", ""),
                "messages": persisted.get("recent_messages", []),
                "language_preference": persisted.get("language_preference"),
                "current_intent": persisted.get("current_intent"),
                "intent_history": persisted.get("intent_history", []),
                "legal_category": persisted.get("legal_category"),
                "uploaded_documents": persisted.get("uploaded_documents", []),
                "last_failed_question": persisted.get("last_failed_question"),
                "last_failed_reason": persisted.get("last_failed_reason"),
                "last_successful_response": persisted.get("last_successful_response"),
                "entities": persisted.get("entities", []),
                "confirmed_facts": persisted.get("confirmed_facts", {}),
                "assumptions": persisted.get("assumptions", {}),
                "missing_details": persisted.get("missing_details", []),
                "fact_corrections": persisted.get("fact_corrections", []),
                "owner_user_id": persisted.get("owner_user_id"),
                "owner_state": persisted.get("owner_state"),
                "owner_token_hash": persisted.get("owner_token_hash"),
                "last_uploaded_document_id": persisted.get("last_uploaded_document_id"),
                "draft_mode": persisted.get("draft_mode"),
                "draft_stage": persisted.get("draft_stage"),
                "draft_template_id": persisted.get("draft_template_id"),
                "draft_fields": persisted.get("draft_fields", {}),
                "draft_id": persisted.get("draft_id"),
                "draft_language": persisted.get("draft_language"),
                "draft_language_hint": persisted.get("draft_language_hint"),
                "draft_awaiting_unlock_confirm": persisted.get("draft_awaiting_unlock_confirm"),
                "draft_reminder_shown": persisted.get("draft_reminder_shown"),
                "draft_seen_message_count": persisted.get("draft_seen_message_count"),
                "inline_document": persisted.get("inline_document"),
                "parked_drafts": persisted.get("parked_drafts", []),
                # Conversational workflow state. Persisted alongside the rest
                # so "continue" still finds a parked workflow after a Redis
                # eviction or a restart -- the stack IS the authoritative
                # record of what was collected, and losing it silently would
                # look to the user like the assistant forgot what they said.
                "chatops_stack": persisted.get("chatops_stack", []),
                "chatops_executed": persisted.get("chatops_executed", {}),
            }
            try:
                await redis_client.set_json(self._key(session_id), memory, settings.session_ttl_seconds)
            except RuntimeError:
                pass
            return memory
        return _empty_memory()

    async def append(self, session_id: str, role: str, content: str) -> dict[str, Any]:
        memory = await self.load(session_id)
        messages = memory.setdefault("messages", [])
        messages.append({"role": role, "content": content})
        memory["messages"] = messages[-MAX_RAW_MESSAGES:]
        await self._persist(session_id, memory)
        return memory

    async def update(self, session_id: str, **values: object) -> dict[str, Any]:
        memory = await self.load(session_id)
        memory.update(values)
        await self._persist(session_id, memory)
        return memory

    async def append_intent_event(self, session_id: str, event: dict[str, Any]) -> dict[str, Any]:
        memory = await self.load(session_id)
        history = [*(memory.get("intent_history") or []), event]
        memory["intent_history"] = history[-50:]
        await self._persist(session_id, memory)
        return memory

    async def set_fact(
        self, session_id: str, kind: Literal["confirmed", "assumption"], key: str, value: str,
    ) -> dict[str, Any]:
        """Set a typed fact and retain the previous value as correction history."""
        memory = await self.load(session_id)
        bucket_name = "confirmed_facts" if kind == "confirmed" else "assumptions"
        bucket = dict(memory.get(bucket_name) or {})
        previous = bucket.get(key)
        bucket[key] = value
        memory[bucket_name] = bucket
        if previous is not None and previous != value:
            corrections = list(memory.get("fact_corrections") or [])
            corrections.append({
                "kind": kind, "key": key, "previous": previous, "current": value,
                "corrected_at": datetime.now(UTC).isoformat(),
            })
            memory["fact_corrections"] = corrections[-100:]
        await self._persist(session_id, memory)
        return memory

    async def delete_fact(
        self, session_id: str, kind: Literal["confirmed", "assumption"], key: str,
    ) -> dict[str, Any]:
        memory = await self.load(session_id)
        bucket_name = "confirmed_facts" if kind == "confirmed" else "assumptions"
        bucket = dict(memory.get(bucket_name) or {})
        bucket.pop(key, None)
        memory[bucket_name] = bucket
        await self._persist(session_id, memory)
        return memory

    async def set_missing_details(self, session_id: str, details: list[str]) -> dict[str, Any]:
        memory = await self.load(session_id)
        memory["missing_details"] = list(dict.fromkeys(item.strip() for item in details if item.strip()))
        await self._persist(session_id, memory)
        return memory

    async def check_access(self, session_id: str, authenticated_user_id: str | None) -> dict[str, Any]:
        """Part 46 "Authenticated User Ownership": the one shared gate every
        route/service that touches a `session_id` calls before doing
        anything with it. An unclaimed session (`owner_user_id` still
        `None` -- every session today, and any session never used by a
        logged-in request) stays exactly as open as before this change --
        no forced login, no migration. Once a session is claimed (see
        `ChatService.answer`/`answer_stream`), any request whose
        authenticated identity doesn't match -- including an anonymous
        request -- is rejected before it can read or write anything for
        that session (history, memory, drafts, uploaded documents).
        """
        memory = await self.load(session_id)
        owner_user_id = memory.get("owner_user_id")
        if owner_user_id and owner_user_id != authenticated_user_id:
            raise ForbiddenError("This session belongs to a different account.")
        return memory

    async def claim_or_verify_ownership(
        self, session_id: str, authenticated_user_id: str | None, owner_token: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """Security finding C1: strict ownership gate for the session routes
        that expose a user's private state directly -- `GET/PATCH/DELETE
        /session/facts`, `PUT /session/missing-details`, `GET/DELETE
        /session` (`app/api/history.py`).

        `check_access` above stays permissive by design for `/chat`,
        `/upload` and drafting (an anonymous session must keep working with
        no forced login) but that same permissiveness is what made those
        five routes exploitable: an unclaimed session (`owner_user_id`
        still `None` -- every anonymous session, and any session before its
        first authenticated turn) was readable/writable by anyone who knew
        or guessed its `session_id`, for as long as it stayed unclaimed --
        forever, if it never had an authenticated turn.

        Ownership here is established on this method's FIRST call for a
        given `session_id` -- i.e. at that session's creation from this
        gate's point of view, not deferred to a later authenticated `/chat`
        turn:
          * an authenticated caller becomes the permanent owner
            (`owner_user_id`, same field `check_access` already uses, so a
            session claimed by one gate is recognised by the other);
          * an anonymous caller gets a fresh `owner_token` minted server
            side (see `generate_owner_token`) -- only its SHA-256 hash is
            stored; the plaintext is returned ONCE, as this call's second
            return value, for the route to hand back to the client (a
            response header). `session_id` alone is never sufficient again
            after this point.

        On every later call: a `"user"`-owned session requires a matching
        authenticated identity (401 with none presented, 403 with a
        different one, matching `check_access`'s own behaviour); an
        `"anonymous"`-owned session requires the matching `owner_token`
        (401 with neither an identity nor a token presented -- no
        credential offered at all -- 403 with one that doesn't match).
        Presenting the correct `owner_token` while authenticated upgrades
        the session to that account, same as a user logging in mid
        conversation.

        A session already claimed via the permissive `check_access` path
        has `owner_user_id` set with `owner_state` still `None`; that is
        read here as `"user"` so the two gates never disagree about a
        session's owner. A session with real content (messages, facts,
        uploads) but no owner at all is legacy -- created before this
        method existed -- and is claimed now, on this call, exactly like a
        brand-new session: the safe backward-compatible migration is to
        stop it being permanently open from this call onward, not to break
        it outright or leave it open forever.
        """
        memory = await self.load(session_id)
        owner_state = memory.get("owner_state")
        if owner_state is None and memory.get("owner_user_id"):
            owner_state = "user"

        if owner_state is None:
            minted: str | None
            if authenticated_user_id:
                memory["owner_state"] = "user"
                memory["owner_user_id"] = authenticated_user_id
                minted = None
            else:
                minted = generate_owner_token()
                memory["owner_state"] = "anonymous"
                memory["owner_token_hash"] = _hash_owner_token(minted)
            await self._persist(session_id, memory)
            return memory, minted

        if owner_state == "user":
            owner_user_id = memory.get("owner_user_id")
            if not authenticated_user_id:
                raise UnauthorizedError("Sign in to access this session.")
            if authenticated_user_id != owner_user_id:
                raise ForbiddenError("This session belongs to a different account.")
            return memory, None

        # owner_state == "anonymous"
        token_hash = memory.get("owner_token_hash")
        token_matches = bool(
            owner_token and token_hash and hmac.compare_digest(_hash_owner_token(owner_token), token_hash)
        )
        if token_matches:
            if authenticated_user_id:
                memory["owner_state"] = "user"
                memory["owner_user_id"] = authenticated_user_id
                memory["owner_token_hash"] = None
                await self._persist(session_id, memory)
            return memory, None

        if not authenticated_user_id and not owner_token:
            raise UnauthorizedError("This session requires its owner credential.")
        raise ForbiddenError("This session belongs to a different owner.")

    async def clear(self, session_id: str) -> None:
        try:
            await redis_client.client.delete(self._key(session_id))
        except RuntimeError:
            pass
        try:
            await self.memory_repository.delete_by_session(session_id)
        except Exception as exc:  # noqa: BLE001 - a failed durable delete must not stop the session being cleared from Redis
            log.warning("long_term_memory_delete_failed", session_id=session_id, error=str(exc))

    async def summarize_if_needed(self, session_id: str, llm: LLMProvider) -> dict[str, Any]:
        """Folds older messages into the rolling summary once the transcript grows long.

        Intended to be scheduled as a FastAPI background task rather than awaited on
        the chat-response critical path, since it costs one extra LLM call.
        """
        memory = await self.load(session_id)
        messages: list[dict[str, str]] = memory.get("messages", [])
        if len(messages) <= SUMMARIZE_TRIGGER_MESSAGE_COUNT:
            return memory
        to_fold = messages[:-KEEP_RECENT_MESSAGES_AFTER_SUMMARY]
        remaining = messages[-KEEP_RECENT_MESSAGES_AFTER_SUMMARY:]
        transcript = "\n".join(f"{message['role']}: {message['content']}" for message in to_fold)
        prompt = prompt_registry.render(
            "summarization_prompt",
            existing_summary=memory.get("summary") or "None",
            conversation=transcript,
        )
        try:
            response = await llm.chat([ChatMessage(role="user", content=prompt)], temperature=0.0)
            if response.error:
                log.warning("conversation_summarization_failed", session_id=session_id, error=response.error)
                return memory
            new_summary = response.content.strip()
        except Exception as exc:  # noqa: BLE001 - summarization is optional; on any failure the un-summarized memory is returned unchanged
            log.warning("conversation_summarization_failed", session_id=session_id, error=str(exc))
            return memory
        if new_summary:
            memory["summary"] = new_summary
        memory["messages"] = remaining
        await self._persist(session_id, memory)
        return memory

    async def _persist(self, session_id: str, memory: dict[str, Any]) -> None:
        try:
            await redis_client.set_json(self._key(session_id), memory, settings.session_ttl_seconds)
        except CACHE_UNAVAILABLE_ERRORS as exc:
            # QA session 2026-09-24 ("BUG-101" concurrency evidence): under
            # load this narrowly caught only `RuntimeError` (client never
            # connected) and let a real `redis.exceptions.TimeoutError` -- a
            # `RedisError`, not a `RuntimeError` -- propagate straight out of
            # `_persist`, crashing the whole `/chat` request with a 500 even
            # though the LLM answer had already been generated successfully
            # (live-reproduced: Odia language test, 500 on a transient Redis
            # read timeout during otherwise-successful retrieval). Matches
            # `app.cache.redis_client.CACHE_UNAVAILABLE_ERRORS`'s own
            # documented contract exactly: any of these must degrade the
            # session-memory cache write to a no-op, never fail the request --
            # the durable Postgres/Mongo write immediately below is the
            # fallback of record, and is itself already wrapped the same way.
            log.warning("session_memory_cache_write_failed", session_id=session_id, error=str(exc))
        try:
            await self.memory_repository.upsert_by_session(
                session_id,
                {
                    "summary": memory.get("summary", ""),
                    "recent_messages": memory.get("messages", []),
                    "language_preference": memory.get("language_preference"),
                    "current_intent": memory.get("current_intent"),
                    "intent_history": memory.get("intent_history", []),
                    "legal_category": memory.get("legal_category"),
                    "uploaded_documents": memory.get("uploaded_documents", []),
                    "last_failed_question": memory.get("last_failed_question"),
                    "last_failed_reason": memory.get("last_failed_reason"),
                    "last_successful_response": memory.get("last_successful_response"),
                    "entities": memory.get("entities", []),
                    "confirmed_facts": memory.get("confirmed_facts", {}),
                    "assumptions": memory.get("assumptions", {}),
                    "missing_details": memory.get("missing_details", []),
                    "fact_corrections": memory.get("fact_corrections", []),
                    "owner_user_id": memory.get("owner_user_id"),
                    "owner_state": memory.get("owner_state"),
                    "owner_token_hash": memory.get("owner_token_hash"),
                    "last_uploaded_document_id": memory.get("last_uploaded_document_id"),
                    "draft_mode": memory.get("draft_mode"),
                    "draft_stage": memory.get("draft_stage"),
                    "draft_template_id": memory.get("draft_template_id"),
                    "draft_fields": memory.get("draft_fields", {}),
                    "draft_id": memory.get("draft_id"),
                    "draft_language": memory.get("draft_language"),
                    "draft_language_hint": memory.get("draft_language_hint"),
                    "draft_awaiting_unlock_confirm": memory.get("draft_awaiting_unlock_confirm"),
                    "draft_reminder_shown": memory.get("draft_reminder_shown"),
                    "draft_seen_message_count": memory.get("draft_seen_message_count"),
                    "inline_document": memory.get("inline_document"),
                    "parked_drafts": memory.get("parked_drafts", []),
                    "chatops_stack": memory.get("chatops_stack", []),
                    "chatops_executed": memory.get("chatops_executed", {}),
                },
            )
        except Exception as exc:  # noqa: BLE001 - durable write is best-effort behind the Redis working set
            log.warning("long_term_memory_write_failed", session_id=session_id, error=str(exc))
