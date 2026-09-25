import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from app.chatops.orchestrator import ChatOrchestrator
from app.chatops.workflows import drafts
from app.rag.bm25_index import BM25Index, tokenize


@pytest.mark.parametrize("choice", ["2", "Open the General Legal Notice draft"])
def test_saved_selection_restores_editable_state(monkeypatch, choice):
    history = [
        SimpleNamespace(
            draft_id="rent", template_name="Rent Notice", status="complete", created_at="2026-09-14"
        ),
        SimpleNamespace(
            draft_id="notice",
            template_name="General Legal Notice",
            status="complete",
            created_at="2026-09-14",
        ),
    ]
    record = {
        "_id": "notice",
        "user_id": "u1",
        "draft_type": "legal_notice",
        "fields": {"amount": "50000"},
        "language": "english",
        "lifecycle_state": "preview_ready",
    }
    monkeypatch.setattr(drafts.LegalDraftEngine, "history", AsyncMock(return_value=history))
    monkeypatch.setattr(
        drafts.LegalDraftEngine,
        "get_current",
        AsyncMock(return_value=SimpleNamespace(full_text="Saved text")),
    )
    owned = AsyncMock(return_value=record)
    monkeypatch.setattr(drafts.draft_management, "owned_draft", owned)
    generate = AsyncMock(side_effect=AssertionError("Opening must not generate a duplicate"))
    monkeypatch.setattr(drafts.LegalDraftEngine, "generate", generate)
    memory = {}

    async def run():
        bot = ChatOrchestrator()
        args = {
            "session_id": "s1",
            "language": "english",
            "memory": memory,
            "authenticated_user_id": "u1",
        }
        listed = await bot.handle_turn(message="Show my saved drafts", **args)
        assert listed.status == "collecting"
        opened = await bot.handle_turn(message=choice, **args)
        assert opened.status == "completed"
        assert "Saved text" in opened.message

    asyncio.run(run())
    assert memory["draft_id"] == "notice"
    assert memory["draft_fields"] == {"amount": "50000"}
    assert memory["draft_template_id"] == "legal_notice"
    assert memory["draft_mode"] is True and memory["draft_stage"] == "preview"
    assert owned.await_args.args[2:] == ("u1", "s1")
    generate.assert_not_awaited()


def test_cached_bm25_counts_match_full_rebuild_after_update_and_removal(tmp_path):
    index = BM25Index(tmp_path / "index.pkl")
    texts = ["rent rent notice", "deposit refund terms", "criminal complaint police"]
    index._set_corpus(["a", "b", "c"], texts, [{}, {}, {}])
    old_counts = index._bm25.doc_freqs[0]
    changed = [texts[0], "deposit notice notice", "new clause"]
    index._set_corpus(["a", "b", "d"], changed, [{}, {}, {}])
    assert index._bm25.doc_freqs[0] is old_counts
    reference = BM25Okapi([tokenize(t) for t in changed])
    for query in ["notice", "deposit", "criminal", "clause rent"]:
        np.testing.assert_allclose(
            index._bm25.get_scores(tokenize(query)), reference.get_scores(tokenize(query))
        )
    index._set_corpus([], [], [])
    assert index.search("notice", 5) == []


def test_chat_dispatch_persists_restored_draft_state(monkeypatch):
    from app.chatops.base import WorkflowTurn
    from app.chatops.orchestrator import orchestrator
    from app.schemas.chat import ChatRequest
    from app.services.chat_service import ChatService

    async def open_draft(**kwargs):
        kwargs["memory"].update(
            draft_id="saved",
            draft_mode=True,
            draft_stage="preview",
            draft_fields={"amount": "50000"},
            draft_template_id="legal_notice",
            draft_language="english",
            draft_paused=False,
        )
        return WorkflowTurn(message="Opened", status="completed", finished=True)

    monkeypatch.setattr(orchestrator, "handle_turn", open_draft)
    service = ChatService.__new__(ChatService)
    service.memory = SimpleNamespace(update=AsyncMock())
    service._contextual_recommendation = AsyncMock(return_value=None)
    response = SimpleNamespace(model_copy=lambda **kwargs: None)
    service._finalize_intent_response = AsyncMock(return_value=response)
    asyncio.run(
        service._dispatch_chatops(
            ChatRequest(question="Open my notice"), "s", "english", {}, 0, None, None
        )
    )
    saved = service.memory.update.await_args.kwargs
    assert saved["draft_id"] == "saved"
    assert saved["draft_fields"] == {"amount": "50000"}
    assert saved["draft_mode"] is True


def test_named_draft_access_denial_does_not_change_active_draft(monkeypatch):
    from app.chatops.base import WorkflowContext
    from app.core.exceptions import ForbiddenError

    monkeypatch.setattr(
        drafts.draft_management, "owned_draft", AsyncMock(side_effect=ForbiddenError("Denied"))
    )
    memory = {"draft_id": "mine", "draft_fields": {"amount": "100"}}
    context = WorkflowContext(
        session_id="s",
        message="open notice",
        language="english",
        memory=memory,
        facts={"action": "open", "draft_id": "foreign"},
    )
    with pytest.raises(ForbiddenError):
        asyncio.run(drafts.DraftManagementWorkflow().execute(context))
    assert memory == {"draft_id": "mine", "draft_fields": {"amount": "100"}}
