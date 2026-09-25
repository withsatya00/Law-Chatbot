"""Part 31 "State Manager & Recovery Engine".

A typed, read-only view over a session's `memory` dict (the same dict
`ConversationMemoryStore` persists) that names every piece of conversation
state the routing layer depends on under one roof -- so "what does the
system currently believe about this conversation?" has one answer instead
of requiring a reader to know which of a dozen raw dict keys to check, and
which ones are even populated by any real flow yet.

Only `snapshot()` reads memory; nothing here writes it. Writes still go
through `ConversationMemoryStore.update()`/`.append()` exactly as before,
at the specific call sites in `chat_service.py` that own each piece of
state -- e.g. the RAG-tail owns `last_failed_question`/
`last_successful_response`, the draft engine owns `draft_mode`/
`draft_stage`, `_respond_with_translation` owns `pending_clarification`.
This module exists to make reading that state legible in one place, not to
become a second source of truth for writing it.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConversationState:
    last_user_question: str | None
    last_successful_response: str | None
    last_failed_question: str | None
    last_failed_reason: str | None
    current_topic: str | None
    pending_translation: bool
    pending_modification: bool
    pending_draft: bool
    pending_clarification: str | None
    pending_upload: bool


def snapshot(memory: dict[str, Any]) -> ConversationState:
    messages: list[dict[str, str]] = memory.get("messages", [])
    last_user_question = next(
        (message["content"] for message in reversed(messages) if message.get("role") == "user"), None
    )
    pending_clarification = memory.get("pending_clarification")
    return ConversationState(
        last_user_question=last_user_question,
        last_successful_response=memory.get("last_successful_response"),
        last_failed_question=memory.get("last_failed_question"),
        last_failed_reason=memory.get("last_failed_reason"),
        # Deliberately an alias of `legal_category` (already updated on every
        # successful RAG-tail turn, see `chat_service.py`) rather than a
        # second independently-written field -- two fields meant to hold the
        # same value are two places that can silently drift out of sync the
        # first time one call site updates one and not the other.
        current_topic=memory.get("legal_category"),
        pending_translation=pending_clarification == "translation_target",
        # No product flow currently issues a modification-specific
        # clarifying question the way Translation does -- an ambiguous
        # Response Modification (`_respond_with_response_modification`,
        # `conv_intent.ambiguous`) just tells the user to ask a question
        # first, it doesn't set a state expecting a specific follow-up
        # reply. This stays real-but-always-False today rather than faked,
        # so it's correct the day a flow that DOES set
        # `pending_clarification="modification_target"` gets built.
        pending_modification=pending_clarification == "modification_target",
        pending_draft=bool(memory.get("draft_mode")),
        pending_clarification=pending_clarification,
        # Same reasoning as `pending_modification`: no upload-prompting flow
        # exists in the chat turn loop yet (`app/services/document_service.py`
        # has no `ConversationMemoryStore` interaction at all) -- reserved,
        # not faked.
        pending_upload=bool(memory.get("pending_upload")),
    )
