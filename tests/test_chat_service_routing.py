import asyncio
import time
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.core.constants import no_verified_context_message
from app.core.exceptions import ForbiddenError
from app.drafting.templates import get_template
from app.llm.base import LLMResponse
from app.schemas.chat import ChatRequest
from app.schemas.common import LawyerRecommendation, RetrievedChunk
from app.schemas.document import DocumentAnalysisResponse
from app.schemas.drafting import DraftGenerateResponse
from app.schemas.phase3 import UserPreferencesResponse
from app.services import chat_service as chat_service_module
from app.services.chat_service import ChatService


class _FakeRedisClient:
    """Minimal in-memory stand-in for the one Redis command
    `ChatService._acquire_turn_lock`/`_release_turn_lock` needs (`SET NX EX`,
    `GET`, `DELETE`) -- no real Redis connection in these unit tests."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def set(self, key: str, value, nx: bool = False, ex: int | None = None):
        if nx and key in self._store:
            return None
        self._store[key] = value.encode() if isinstance(value, str) else value
        return True

    async def get(self, key: str):
        return self._store.get(key)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


async def _collect_stream(agen) -> list[dict]:
    return [event async for event in agen]


def _mock_stream(chunks: list[str]):
    async def _stream(messages, temperature: float = 0.1):
        for chunk in chunks:
            yield chunk

    return _stream


def _service_with_mocks(messages: list[dict[str, str]] | None = None) -> ChatService:
    """Builds a ChatService with I/O-heavy collaborators mocked out.

    `memory.append`/`update`/`summarize_if_needed` are mocked (rather than left
    real) so tests can control exactly what conversation history the classifier
    sees, without depending on a live Redis/Mongo -- `memory.append` is what
    `ChatService.answer` calls first, so its mocked return value becomes the
    `memory` dict the rest of the turn operates on.
    """
    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])
    service.memory.append = AsyncMock(
        return_value={"messages": messages or [], "summary": "", "current_intent": None, "legal_category": None}
    )
    # `answer_stream` (unlike `answer`) calls `memory.load` directly to build
    # a preview of conversation state before deciding whether to fall back to
    # `answer()` -- mocked here too so streaming tests don't hit a live
    # Redis/Mongo. Existing non-streaming tests never call `.load()` (the
    # real `ConversationMemoryStore.append` does internally, but it's fully
    # replaced by the mock above), so this is a no-op for them.
    service.memory.load = AsyncMock(
        return_value={"messages": messages or [], "summary": "", "current_intent": None, "legal_category": None}
    )
    service.memory.update = AsyncMock(return_value={})
    service.memory.summarize_if_needed = AsyncMock(return_value={})
    # Jurisdiction Routing (Phase 2): `_resolve_matter_context` looks up the
    # caller's saved profile State whenever `authenticated_user_id` is set --
    # mocked here (no profile State) so authenticated-request tests don't
    # need a live Mongo just to resolve matter context.
    service.preferences.get = AsyncMock(return_value=UserPreferencesResponse())
    service.history.insert = AsyncMock(return_value="history-id")
    service.query_log.insert = AsyncMock(return_value="query-log-id")
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    service.retriever.retrieve = AsyncMock(side_effect=AssertionError("retrieval should have been skipped"))
    service.reranker.rerank = AsyncMock(side_effect=AssertionError("reranking should have been skipped"))
    return service


def test_general_conversation_skips_rag() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Happy to help!", model="test", provider="test"))
    request = ChatRequest(question="Thanks, that helps")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "General Conversation"
    assert response.sources == []
    service.llm.chat.assert_called_once()


def test_handle_turn_rejects_a_concurrent_request_for_the_same_session(monkeypatch) -> None:
    """QA (session 2026-09-11/12, `docs/qa/QA_TEST_MATRIX_20260911.md`,
    BUG-013a): a client that gives up on a stalled request and resends it
    (literally or as an equivalent retry) while the ORIGINAL request is
    still running server-side must never let both run the full pipeline --
    that is exactly how a duplicate draft/chat-history entry gets created
    for what the user experiences as one logical request. `handle_turn`'s
    per-session lock is the fix: a second call for the same session must be
    turned away, and MUST NOT invoke `answer()` at all.
    """
    monkeypatch.setattr(chat_service_module.redis_client, "_client", _FakeRedisClient())
    service = _service_with_mocks()
    service.answer = AsyncMock(side_effect=AssertionError("answer() must not run while a turn is already in flight"))
    session_id = "dup-session"
    request = ChatRequest(question="Convert to legal notice.", session_id=session_id)

    async def run():
        held_token = await service._acquire_turn_lock(session_id)
        assert held_token is not None
        return await service.handle_turn(request)

    response = asyncio.run(run())
    service.answer.assert_not_called()
    assert response.retryable is True
    assert response.failure_category == "in_progress"


def test_handle_turn_releases_its_lock_so_the_next_request_is_not_blocked(monkeypatch) -> None:
    monkeypatch.setattr(chat_service_module.redis_client, "_client", _FakeRedisClient())
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Happy to help!", model="test", provider="test"))
    session_id = "release-session"
    request = ChatRequest(question="Thanks, that helps", session_id=session_id)

    async def run():
        await service.handle_turn(request)
        # If the lock from the first call were not released, this second,
        # independent turn would be wrongly rejected as "still in progress".
        return await service.handle_turn(request)

    response = asyncio.run(run())
    assert response.failure_category != "in_progress"
    assert response.conversation_intent == "General Conversation"


def test_handle_turn_returns_a_clear_bounded_timeout_response_when_answer_hangs(monkeypatch) -> None:
    """The other half of the same QA finding: whatever the reason a turn
    doesn't finish in time (here, simulated directly by making `answer()`
    itself never return), the caller must get a real, clearly-labelled
    response within a bounded time -- never an indefinitely hanging
    connection -- and the question must be recorded as a genuine failure so
    the very next turn (see `ConversationIntentClassifier`'s BUG-013a fix,
    or the literal word "retry") resumes it.
    """
    monkeypatch.setattr(settings, "chat_request_budget_seconds", 0.1)
    monkeypatch.setattr(chat_service_module, "_HARD_TIMEOUT_GRACE_SECONDS", 0.05)
    service = _service_with_mocks()

    async def _hang(*args, **kwargs):
        await asyncio.Event().wait()

    service.answer = _hang
    request = ChatRequest(question="Convert to legal notice.", session_id="timeout-session")

    async def run():
        turn_started = time.perf_counter()
        response = await service.handle_turn(request)
        return response, time.perf_counter() - turn_started

    response, elapsed = asyncio.run(run())
    assert elapsed < 5.0, f"handle_turn took {elapsed:.1f}s despite a 0.1s budget -- the hard timeout did not fire"
    assert response.retryable is True
    assert response.failure_category == "timeout"
    assert "took longer than expected" in response.answer.lower()


def test_greeting_hii_skips_rag_and_disclaimer() -> None:
    """Part 41 regression test item 1: "hii" (no prior context) must
    short-circuit to General Conversation -- never reach retrieval/BM25/
    embedding/reranker, and never carry a legal disclaimer, since it's not
    a legal question at all. `_service_with_mocks`'s `retriever.retrieve`/
    `reranker.rerank` mocks raise `AssertionError` if ever called, so this
    fails loudly if "hii" regresses back to the RAG path.
    """
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="Hey! How can I help with a legal question today?", model="test", provider="test")
    )
    response = asyncio.run(service.answer(ChatRequest(question="hii")))
    assert response.conversation_intent == "General Conversation"
    assert response.sources == []
    assert "This information is provided for educational purposes only" not in response.answer


def test_greeting_hello_skips_rag_and_disclaimer() -> None:
    """Part 41 regression test item 2."""
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="Hello! What can I help you with?", model="test", provider="test")
    )
    response = asyncio.run(service.answer(ChatRequest(question="hello")))
    assert response.conversation_intent == "General Conversation"
    assert "This information is provided for educational purposes only" not in response.answer


def test_greeting_aur_kya_haal_hai_skips_rag() -> None:
    """Part 41 regression test item 9: Hindi/Hinglish small talk ("what's
    up") must route the same way plain "hi" does, not fall to the generic
    legal-information bucket and reach RAG.
    """
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Sab badhiya! Aap batayein?", model="test", provider="test"))
    response = asyncio.run(service.answer(ChatRequest(question="aur kya haal hai")))
    assert response.conversation_intent == "General Conversation"
    assert "This information is provided for educational purposes only" not in response.answer


def test_lawyer_recommendation_skips_rag_and_llm() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(side_effect=AssertionError("no LLM call needed for a deterministic recommendation"))
    request = ChatRequest(question="Can you recommend a lawyer for cheque bounce")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "Lawyer Recommendation"
    assert response.lawyer_recommendation is not None


def test_translation_without_target_asks_for_clarification() -> None:
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "Can you translate that for me?"},
    ]
    service = _service_with_mocks(messages)
    service.llm.chat = AsyncMock(side_effect=AssertionError("ambiguous translation should not call the LLM"))
    request = ChatRequest(question="Can you translate that for me?")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "Translation"
    assert "language" in response.answer.lower()


def test_summarization_skips_rag() -> None:
    messages = [
        {"role": "user", "content": "My landlord won't return my deposit"},
        {"role": "assistant", "content": "Here's what you can do..."},
        {"role": "user", "content": "Summarize our conversation"},
    ]
    service = _service_with_mocks(messages)
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="You asked about a deposit dispute.", model="test", provider="test"))
    request = ChatRequest(question="Summarize our conversation")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "Summarization"
    assert "deposit dispute" in response.answer


def test_response_modification_shortens_prior_reply_without_rag() -> None:
    messages = [
        {"role": "user", "content": "What is FIR?"},
        {"role": "assistant", "content": "An FIR is a First Information Report filed with the police..."},
        {"role": "user", "content": "30 words"},
    ]
    service = _service_with_mocks(messages)
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="An FIR is a police report that starts a criminal investigation.", model="test", provider="test"))
    request = ChatRequest(question="30 words")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "Response Modification"
    assert "criminal investigation" in response.answer
    assert response.sources == []
    service.llm.chat.assert_called_once()


def test_response_modification_without_prior_reply_never_calls_llm() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(side_effect=AssertionError("no LLM call needed when there's nothing to modify"))
    request = ChatRequest(question="Bullet points.")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "Response Modification"
    assert "previous response" in response.answer.lower()


def test_response_modification_chains_onto_the_latest_version() -> None:
    # Simulates "Translate into Hindi" already having happened -- the last
    # assistant turn in memory is the Hindi text, not the original English.
    # A follow-up "30 words" must transform THAT version, per Part 24's
    # chaining requirement ("never restart from the original answer").
    messages = [
        {"role": "user", "content": "What is FIR?"},
        {"role": "assistant", "content": "An FIR is..."},
        {"role": "user", "content": "Translate into Hindi"},
        {"role": "assistant", "content": "एफआईआर एक पुलिस रिपोर्ट है..."},
        {"role": "user", "content": "30 words"},
    ]
    service = _service_with_mocks(messages)
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="संक्षिप्त एफआईआर विवरण।", model="test", provider="test"))
    request = ChatRequest(question="30 words")
    asyncio.run(service.answer(request))
    prompt_used = service.llm.chat.call_args[0][0][0].content
    assert "एफआईआर एक पुलिस रिपोर्ट है" in prompt_used
    assert "An FIR is..." not in prompt_used


def test_translation_clarification_then_bare_language_reply_completes_translation() -> None:
    # First turn: "Translate." with no target -- ambiguous, asks which
    # language, and marks `pending_clarification` so the next bare reply
    # resolves it (Part 24's "Translation memory" requirement).
    messages_turn1 = [
        {"role": "user", "content": "What is FIR?"},
        {"role": "assistant", "content": "An FIR is..."},
        {"role": "user", "content": "Translate."},
    ]
    service = _service_with_mocks(messages_turn1)
    service.llm.chat = AsyncMock(side_effect=AssertionError("ambiguous translation should not call the LLM"))
    response1 = asyncio.run(service.answer(ChatRequest(question="Translate.")))
    assert response1.conversation_intent == "Translation"
    update_calls = [call.kwargs for call in service.memory.update.await_args_list]
    assert any(call.get("pending_clarification") == "translation_target" for call in update_calls)

    # Second turn: bare "Hindi" resolves it, using the pending_clarification
    # flag set above.
    messages_turn2 = [
        {"role": "user", "content": "What is FIR?"},
        {"role": "assistant", "content": "An FIR is..."},
        {"role": "user", "content": "Translate."},
        {"role": "assistant", "content": "Which language would you like this translated into?"},
        {"role": "user", "content": "Hindi"},
    ]
    service2 = _service_with_mocks(messages_turn2)
    service2.memory.append = AsyncMock(return_value={"messages": messages_turn2, "pending_clarification": "translation_target"})
    service2.llm.chat = AsyncMock(return_value=LLMResponse(content="एफआईआर क्या है...", model="test", provider="test"))
    response2 = asyncio.run(service2.answer(ChatRequest(question="Hindi")))
    assert response2.conversation_intent == "Translation"
    assert "एफआईआर" in response2.answer


def test_translation_skips_its_own_clarification_question_as_the_source_text() -> None:
    # Part 47.1 regression: reproduces a live failure where "Translate" ->
    # "Which language would you like this translated into?" -> "Hindi"
    # translated the bot's OWN clarification question (the most recent
    # assistant message in `memory["messages"]`) instead of the real prior
    # answer -- the LLM, asked to translate its own "translate only when
    # explicitly requested" prompt text, replied by parroting that
    # instruction back rather than translating anything. `_respond_with_
    # translation` must skip its own known clarification replies when
    # scanning backwards for the text to translate.
    messages = [
        {"role": "user", "content": "What is FIR?"},
        {"role": "assistant", "content": "An FIR is the first report filed with police."},
        {"role": "user", "content": "Translate."},
        {"role": "assistant", "content": "Which language would you like this translated into?"},
        {"role": "user", "content": "Hindi"},
    ]
    service = _service_with_mocks(messages)
    service.memory.append = AsyncMock(return_value={"messages": messages, "pending_clarification": "translation_target"})
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="एफआईआर पहली रिपोर्ट है...", model="test", provider="test"))
    asyncio.run(service.answer(ChatRequest(question="Hindi")))

    sent_prompt = service.llm.chat.call_args_list[0].args[0][0].content
    assert "first report filed with police" in sent_prompt
    assert "Which language would you like this translated into" not in sent_prompt


def test_conversation_memory_question_answered_without_rag() -> None:
    messages = [
        {"role": "user", "content": "My phone was stolen."},
        {"role": "assistant", "content": "I'm sorry to hear that. Here's what you can do..."},
        {"role": "user", "content": "What was stolen?"},
    ]
    service = _service_with_mocks(messages)
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="You mentioned that your phone was stolen.", model="test", provider="test"))
    request = ChatRequest(question="What was stolen?")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "Conversation Memory"
    assert "phone was stolen" in response.answer
    assert response.sources == []
    service.llm.chat.assert_called_once()


def test_conversation_memory_without_prior_turns_never_calls_llm() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(side_effect=AssertionError("no LLM call needed when nothing's been discussed"))
    request = ChatRequest(question="What did I ask?")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "Conversation Memory"
    assert "haven't talked" in response.answer.lower()


def _draft_memory_snapshot(**overrides: object) -> dict[str, object]:
    base = {
        "messages": [{"role": "user", "content": "I want to file an RTI"}, {"role": "assistant", "content": "Sure, let's start."}],
        "summary": "",
        "current_intent": None,
        "legal_category": None,
        "draft_mode": True,
        "draft_stage": "collecting",
        "draft_template_id": "rti_application",
        "draft_fields": {"applicant_name": "Ramesh Kumar"},
        "draft_id": None,
    }
    base.update(overrides)
    return base


def test_unrelated_question_interrupts_draft_and_notes_it_is_saved() -> None:
    memory = _draft_memory_snapshot()
    service = _service_with_mocks(memory["messages"])
    service.memory.append = AsyncMock(return_value=memory)
    service.llm.chat = AsyncMock(side_effect=AssertionError("no LLM call needed for a deterministic recommendation"))
    request = ChatRequest(question="Can you recommend a lawyer for cheque bounce")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "Lawyer Recommendation"
    assert "Public Authority / Department Name" not in response.answer
    # The draft (still mid-collection, 5 fields left) is genuinely
    # interrupted by this unrelated request -- the one-line reminder is
    # appended exactly once for this turn (see `test_multi_turn_conversations
    # .py`'s `test_draft_reminder_shown_once_per_pause_not_every_turn` for the
    # "not spammed on every subsequent unrelated turn" behavior, Part 43
    # section 14).
    assert "draft is saved" in response.answer.lower()
    assert "continue draft" in response.answer.lower()


def test_plain_field_value_still_continues_draft_uninterrupted() -> None:
    # A short freeform value (e.g. a department name) must NOT be treated as
    # an unrelated question -- it should keep flowing into the draft engine
    # exactly like before this feature existed.
    memory = _draft_memory_snapshot()
    service = _service_with_mocks(memory["messages"])
    service.memory.append = AsyncMock(return_value=memory)
    service.llm.chat = AsyncMock(side_effect=AssertionError("draft field collection should not call the LLM"))
    request = ChatRequest(question="Public Works Department")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "Draft Generation"
    assert "still in progress" not in response.answer


def test_cancel_draft_command_resets_draft_state() -> None:
    memory = _draft_memory_snapshot()
    service = _service_with_mocks(memory["messages"])
    service.memory.append = AsyncMock(return_value=memory)
    request = ChatRequest(question="cancel draft")
    response = asyncio.run(service.answer(request))
    assert "discarded" in response.answer.lower()
    update_calls = [call.kwargs for call in service.memory.update.await_args_list]
    assert any(call.get("draft_mode") is False for call in update_calls)


def test_translate_the_draft_in_preview_stage_is_not_treated_as_interruption() -> None:
    # In preview stage, "translate" already has a drafting-specific meaning
    # (translate the generated draft) handled by the edit-command interpreter
    # -- it must not be redirected into the generic conversation Translation
    # handler.
    #
    # Part 52: a language change now goes through `LegalDraftEngine.regenerate`
    # (which re-renders and persists the draft in the new language) rather
    # than the old `translate()` call, which returned a translated blob that
    # was never written back to the draft record -- see `conversation.py`'s
    # `_continue_preview` "translate" branch.
    template = get_template("rti_application")
    assert template is not None
    memory = _draft_memory_snapshot(
        draft_stage="preview",
        draft_id="draft-123",
        draft_fields={key: "value" for key in template.required_field_keys()},
    )
    service = _service_with_mocks(memory["messages"])
    service.memory.append = AsyncMock(return_value=memory)
    service.draft_conversation.draft_engine.regenerate = AsyncMock(
        return_value=DraftGenerateResponse(
            status="complete",
            draft_id="draft-123",
            template_id=template.draft_id,
            template_name=template.name,
            full_text="अनुवादित पाठ",
            language="hindi",
        )
    )
    service.llm.chat = AsyncMock(side_effect=AssertionError("generic translation handler should not run"))
    request = ChatRequest(question="translate this to hindi")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "Draft Generation"
    assert "अनुवादित पाठ" in response.answer
    service.draft_conversation.draft_engine.regenerate.assert_awaited_once_with("draft-123", {}, language="hindi")


def test_no_verified_context_short_circuits_without_calling_llm() -> None:
    # Strict-RAG guardrail: zero chunks cleared the relevance gate -- must
    # return the exact hardcoded fallback string (plus non-urgent next-steps
    # guidance, since this question matches no safety-urgency category --
    # see `generic_next_steps_guidance`) and never call the LLM at all
    # (no general-knowledge answer, no disclaimer appended, nothing cached
    # since `is_cacheable` is only reached on the LLM branch).
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("what are my rights", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock()
    request = ChatRequest(question="What are my rights if my employer doesn't pay overtime?")
    response = asyncio.run(service.answer(request))
    assert response.answer.startswith(no_verified_context_message("english"))
    assert response.confidence == 0.0
    assert response.sources == []
    service.llm.chat.assert_not_awaited()
    service.response_cache.store.assert_not_awaited()


def test_general_knowledge_fallback_answers_an_ordinary_out_of_kb_question(monkeypatch) -> None:
    """The controlled General Knowledge fallback (`app.core.gk_fallback`):
    when the strict-RAG guardrail finds zero verified chunks AND the operator
    has opted in (`general_knowledge_fallback_enabled` -- off by default via
    `conftest.py`'s `_general_knowledge_fallback_off_by_default`, turned on
    here to exercise it), an ordinary legal question with no urgency category
    and no specific-provision/current-law cue gets a real, labeled, LLM-
    generated answer instead of the bare strict-RAG refusal.

    Live example this was built for: "meri wife mujhe chhod ke chali gayi hai
    main kya karu" ("my wife has left me, what do I do") -- an ordinary,
    common family-law question that is neither a domestic-violence/threat/
    cyber-fraud urgency category (so the fixed safety guidance doesn't apply)
    nor a specific-section/current-law lookup (so GK fallback isn't blocked).
    """
    monkeypatch.setattr(settings, "general_knowledge_fallback_enabled", True)
    service = _service_with_mocks()
    question = "meri wife mujhe chhod ke chali gayi hai main kya karu"
    service.retriever.retrieve = AsyncMock(return_value=(question, []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(return_value=LLMResponse(
        content=(
            "CONFIDENCE: MEDIUM\n\n"
            "In general, if your spouse has left the marital home, this is usually treated as a "
            "matter of matrimonial law. You may be able to seek restitution of conjugal rights, or, "
            "after a period of separation, judicial separation or divorce, depending on your "
            "circumstances. A family court advocate can advise on the specific remedy available to you."
        ),
        model="test-model", provider="test",
    ))
    request = ChatRequest(question=question)
    response = asyncio.run(service.answer(request))

    assert "General Legal Knowledge" in response.answer
    # The disclaimer is in whatever language the request resolved to (this
    # Hinglish question resolves to the Hinglish disclaimer, not English) --
    # "verify" appears in every language variant of it (see
    # `app.core.constants.GENERAL_KNOWLEDGE_DISCLAIMER_MESSAGES`).
    assert "verify" in response.answer.lower()
    assert "CONFIDENCE" not in response.answer
    assert "restitution of conjugal rights" in response.answer
    assert response.general_knowledge_used is True
    assert response.no_verified_context is False
    assert response.sources == []
    assert 0.0 < response.confidence < 0.35  # Low, per _GENERAL_KNOWLEDGE_CONFIDENCE
    service.llm.chat.assert_awaited_once()
    # A GK answer is never cached -- see chat_service.py's `elif
    # no_verified_context:` branch, which never calls `response_cache.store`.
    service.response_cache.store.assert_not_awaited()


def test_general_knowledge_fallback_never_fires_when_kb_content_is_available() -> None:
    """Requirement: GK fallback must be used ONLY when nothing in the KB
    cleared the relevance gate -- never as an alternative to, or override
    of, a real grounded answer. `_service_with_mocks()` already asserts this
    structurally (`retriever.retrieve` defaults to raising if called with no
    override), but this test asserts the OUTCOME too: with `ranked`
    non-empty, the normal grounded-RAG branch runs and General Knowledge
    labeling never appears, regardless of whether the flag is on.
    """
    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="c1", text="Section 303: theft should be reported via a police FIR immediately.", score=0.9,
        metadata={"source_document": "BNS", "act_name": "Bharatiya Nyaya Sanhita", "section_number": "303"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("mera bike chori ho gya hai", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="Report the theft at the nearest police station.", model="test", provider="test")
    )
    request = ChatRequest(question="mera bike chori ho gya hai mujhe uske liye kya krna hoga")
    response = asyncio.run(service.answer(request))

    assert "General Legal Knowledge" not in response.answer
    assert response.general_knowledge_used is False


def test_general_knowledge_fallback_fires_when_retrieved_chunks_fail_grounding(monkeypatch) -> None:
    """Regression test for a live production gap (confirmed 2026-09-23 via a
    clean backend restart + a live Hinglish out-of-KB request, so it was not
    a stale-env/config artifact): GK fallback's eligibility was wired only
    into the `elif no_verified_context:` branch, i.e. retrieval finding
    literally zero ranked chunks. But a Hinglish out-of-KB question can just
    as easily retrieve a chunk that clears the initial relevance gate (a
    noisy embedding match) and still collapse to the exact same bare refusal
    once grounding validation discards it for missing `source_document`
    (this is the same setup as `test_ungrounded_answer_is_downgraded_to_
    insufficient_context` above) -- that second route never reached
    `general_knowledge_answer` at all, so a real out-of-KB question
    ("Agar Dubai mein naukri karte waqt company salary time par nahi de rahi
    hai...") got the bare refusal even with the flag enabled right after a
    clean restart. `_finalize_rag_response` is the one place both routes
    converge on the settled refusal, so it is now also the one place GK
    fallback is attempted, regardless of which branch produced it.
    """
    monkeypatch.setattr(settings, "general_knowledge_fallback_enabled", True)
    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="c1", text="Vehicle theft should be reported via a police FIR immediately.", score=0.9,
        metadata={"act_name": "Indian Penal Code", "section_number": "379"},
    )
    question = (
        "Agar Dubai mein naukri karte waqt company salary time par nahi de rahi hai "
        "to employee ke paas kya legal options hain?"
    )
    service.retriever.retrieve = AsyncMock(return_value=(question, [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(side_effect=[
        # 1st call: the normal RAG generation attempt -- discarded by
        # grounding validation regardless of content (the chunk's metadata
        # has no `source_document`, so it yields zero citations).
        LLMResponse(content="Report the theft at the nearest police station.", model="test", provider="test"),
        # 2nd call: `general_knowledge_answer`'s own LLM call, only reached
        # if `_finalize_rag_response` gives GK fallback a chance here.
        LLMResponse(
            content=(
                "CONFIDENCE: MEDIUM\n\n"
                "In general, an employee whose salary goes unpaid abroad can raise the issue with the "
                "local labour ministry or file a complaint in that country's labour court."
            ),
            model="test-model", provider="test",
        ),
    ])
    request = ChatRequest(question=question)
    response = asyncio.run(service.answer(request))

    assert "General Legal Knowledge" in response.answer
    assert "CONFIDENCE" not in response.answer
    assert response.general_knowledge_used is True
    assert response.no_verified_context is False
    assert response.sources == []
    assert 0.0 < response.confidence < 0.35  # Low, per _GENERAL_KNOWLEDGE_CONFIDENCE
    # At least the RAG generation attempt and GK fallback's own call -- a
    # 3rd, best-effort related-questions call may or may not also land on
    # this same mock (unrelated to what this test is checking).
    assert service.llm.chat.await_count >= 2


def test_grounding_refusal_retry_recovers_when_context_is_strongly_scored() -> None:
    """QA pass 2026-09-24 ("LLM non-deterministic refusal despite good
    context"): `safe_decline.py`'s own docstring already documents that the
    LLM itself can emit Rule 2's exact refusal line even when `ranked` holds
    genuinely on-topic, strongly-scored context (confirmed live: the same
    question grounded on 1 of 3 identical `/chat` calls). `_call_llm_with_
    grounding_retry` gives such a case one bounded second attempt before
    accepting the refusal -- this asserts the recovered answer is what the
    user actually gets, not the discarded first-attempt refusal.
    """
    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="s63", text="63. Admissibility of electronic records as documents.", score=0.58,
        metadata={"source_document": "BSA.pdf", "act_name": "Bharatiya Sakshya Adhiniyam", "section_number": "63"},
    )
    service.retriever.retrieve = AsyncMock(
        return_value=("admissibility of electronic evidence Bharatiya Sakshya Adhiniyam", [chunk])
    )
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(side_effect=[
        LLMResponse(content=no_verified_context_message("english"), model="test", provider="test"),
        LLMResponse(
            content=(
                "Under Section 63 of the Bharatiya Sakshya Adhiniyam, electronic records are admissible as "
                "documents when the conditions in that section are met."
            ),
            model="test", provider="test",
        ),
    ])
    request = ChatRequest(question="admissibility of electronic evidence Bharatiya Sakshya Adhiniyam")
    response = asyncio.run(service.answer(request))

    # At least the retry's two calls -- a 3rd, best-effort related-questions
    # call may or may not also land on this same mock (unrelated to what
    # this test is checking, see the GK-fallback test above for the same
    # pattern).
    assert service.llm.chat.await_count >= 2
    assert not response.answer.startswith(no_verified_context_message("english"))
    assert "Section 63" in response.answer
    assert response.no_verified_context is False


def test_grounding_refusal_retry_does_not_fire_for_weakly_scored_context() -> None:
    """The retry must stay scoped to STRONG grounding only -- a refusal over
    weak/marginal context (below `_STRONG_GROUNDING_RETRY_SCORE`) is very
    likely correct and must not pay a second LLM call's latency, nor risk
    manufacturing an answer out of thin evidence.
    """
    service = _service_with_mocks()
    # Scored well below `_STRONG_GROUNDING_RETRY_SCORE` (0.30) but still
    # accepted by the relevance gate's "legal notice" literal-overlap
    # override (`app.rag.relevance.is_relevant_chunk`), so `ranked` is
    # genuinely non-empty and the real LLM branch (not the separate
    # `no_verified_context` short-circuit) is what's under test here.
    chunk = RetrievedChunk(
        chunk_id="weak", text="A legal notice should state the facts and the relief sought.", score=0.15,
        metadata={"source_document": "unrelated.pdf", "act_name": "Some Other Act"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("how to send a legal notice", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    # Isolates this assertion from the separate, unrelated best-effort
    # related-questions call (`_generate_related_questions`), which also
    # calls `self.llm.chat` whenever the RAG generation attempt did not
    # itself fail as a provider error -- see the GK-fallback test above's
    # own comment on the same call.
    service._generate_related_questions = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content=no_verified_context_message("english"), model="test", provider="test")
    )
    request = ChatRequest(question="how to send a legal notice")
    response = asyncio.run(service.answer(request))

    assert service.llm.chat.await_count == 1
    assert response.answer.startswith(no_verified_context_message("english"))


def test_no_verified_context_gives_generic_next_steps_for_an_ordinary_question() -> None:
    """Live repro: an ordinary informational question with no urgency
    category ("BNS Section 12 kya hai", "how to send a legal notice") got a
    bare refusal and no path forward at all -- `actionable_guidance` only
    ever fires for the three safety-urgency categories (domestic violence,
    threat/harassment, cyber fraud), so every other strict-RAG refusal was a
    dead end. `generic_next_steps_guidance` is the non-urgent counterpart
    and must now always be appended instead.
    """
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("bns section 12", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock()
    request = ChatRequest(question="What is Section 12 of the Bharatiya Nyaya Sanhita?")
    response = asyncio.run(service.answer(request))
    assert response.answer.startswith(no_verified_context_message("english"))
    assert len(response.answer) > len(no_verified_context_message("english"))
    assert "advocate" in response.answer.lower()


def test_retrieval_is_scoped_to_the_resolved_session_id() -> None:
    """Part 45 "Per-User Document Isolation" (reshaped by Part 46
    "Authenticated User Ownership" into an explicit `$or`, see
    `test_vector_store_filters.py`/`test_bm25_index.py` for the filter-
    builder-level coverage of that shape): `_prepare_rag_context` must
    inject a session branch scoped to the resolved session_id -- "mine, or
    nobody's (curated KB / pre-existing docs)". Uses the server-generated
    session_id (client sent none) to confirm the *resolved* value is used,
    not `request.session_id` (which is `None` here). No `authenticated_
    user_id` is passed, so only the session branch is present.
    """
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="An FIR is...", model="test", provider="test"))
    response = asyncio.run(service.answer(ChatRequest(question="what is fir")))
    filters = service.retriever.retrieve.call_args.kwargs["filters"]
    # Phase 1 "Jurisdiction-Aware Knowledge Base" gap 1 fix: the shared/
    # unowned branch and the session-owned branch are now separate `$or`
    # entries -- only the shared branch carries `review_status=approved`
    # (never a private document, which has no such field at all).
    assert filters["$or"] == [
        {"owner_session_id": [None], "owner_user_id": [None], "review_status": "approved"},
        {"owner_session_id": [response.session_id], "owner_user_id": [None]},
    ]
    assert response.session_id is not None


def test_client_supplied_metadata_filters_cannot_override_owner_scope() -> None:
    """A client trying to pass its own `"$or"`/`owner_session_id` in
    `metadata_filters` (to spoof another session's documents) must never
    override the server-injected scope -- the owner key is always built
    from the server's own resolved session_id, added last.
    """
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="An FIR is...", model="test", provider="test"))
    request = ChatRequest(
        question="what is fir", session_id="real-session",
        metadata_filters={"$or": [{"owner_session_id": "attacker-session"}], "source_document": "x.pdf"},
    )
    asyncio.run(service.answer(request))
    filters = service.retriever.retrieve.call_args.kwargs["filters"]
    assert filters["$or"] == [
        {"owner_session_id": [None], "owner_user_id": [None], "review_status": "approved"},
        {"owner_session_id": ["real-session"], "owner_user_id": [None]},
    ]
    assert filters["source_document"] == "x.pdf"


def test_authenticated_request_adds_the_user_ownership_branch() -> None:
    """Part 46 "Authenticated User Ownership": an authenticated request adds
    a second `$or` branch for the exact-match `owner_user_id`, on top of
    (not instead of) the unchanged session branch.
    """
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="An FIR is...", model="test", provider="test"))
    request = ChatRequest(question="what is fir", session_id="real-session")
    asyncio.run(service.answer(request, authenticated_user_id="user-A"))
    filters = service.retriever.retrieve.call_args.kwargs["filters"]
    assert filters["$or"] == [
        {"owner_session_id": [None], "owner_user_id": [None], "review_status": "approved"},
        {"owner_session_id": ["real-session"], "owner_user_id": [None]},
        {"owner_user_id": ["user-A"]},
    ]


def test_profile_state_jurisdiction_filter_lands_only_in_shared_branch() -> None:
    """Jurisdiction Routing (Phase 2): an authenticated user's saved profile
    State (MP) is folded into the shared/unowned branch's applicability
    `$or` alongside `review_status` -- the private ownership branches are
    left exactly as Part 45/46 defined them, untouched by jurisdiction."""
    service = _service_with_mocks()
    service.preferences.get = AsyncMock(return_value=UserPreferencesResponse(profile_state_code="MP"))
    service.retriever.retrieve = AsyncMock(return_value=("my landlord deposit dispute", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="...", model="test", provider="test"))
    request = ChatRequest(question="My landlord is not returning my deposit", session_id="real-session")
    asyncio.run(service.answer(request, authenticated_user_id="user-A"))
    filters = service.retriever.retrieve.call_args.kwargs["filters"]
    shared_branch, session_owned_branch, user_branch = filters["$or"]
    assert shared_branch["$or"] == [
        {"applicability": "all_india"},
        {"applicability": "specific_states", "applicable_state_codes": "MP"},
    ]
    assert "$or" not in session_owned_branch
    assert "$or" not in user_branch
    matter_context = service.retriever.retrieve.call_args.kwargs["matter_context"]
    assert matter_context["state_codes"] == ["MP"]


def test_ambiguous_state_sensitive_question_asks_for_state_before_retrieval() -> None:
    """No profile, no explicit State, and eviction wording that genuinely
    turns on State law -- retrieval must never be reached."""
    service = _service_with_mocks()
    request = ChatRequest(question="What is the eviction notice period for my rented flat?", session_id="s1")
    response = asyncio.run(service.answer(request))
    assert "State" in response.answer or "state" in response.answer
    service.retriever.retrieve.assert_not_called()


def test_client_supplied_user_id_on_the_request_is_never_trusted() -> None:
    """`ChatRequest.user_id` is a pre-existing, client-supplied, unverified
    field -- it must never be read for ownership scoping, only the
    JWT-derived `authenticated_user_id` parameter (never passed here).
    """
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="An FIR is...", model="test", provider="test"))
    request = ChatRequest(question="what is fir", session_id="real-session", user_id="spoofed-user-id")
    asyncio.run(service.answer(request))
    filters = service.retriever.retrieve.call_args.kwargs["filters"]
    assert filters["$or"] == [
        {"owner_session_id": [None], "owner_user_id": [None], "review_status": "approved"},
        {"owner_session_id": ["real-session"], "owner_user_id": [None]},
    ]


def test_session_owned_by_one_account_rejects_a_different_authenticated_account() -> None:
    service = _service_with_mocks()
    service.memory.load = AsyncMock(
        return_value={
            "messages": [], "summary": "", "current_intent": None, "legal_category": None,
            "owner_user_id": "user-A",
        }
    )
    request = ChatRequest(question="what is fir", session_id="user-a-session")
    from app.core.exceptions import ForbiddenError

    try:
        asyncio.run(service.answer(request, authenticated_user_id="user-B"))
        assert False, "expected ForbiddenError"
    except ForbiddenError:
        pass


def test_session_owned_by_one_account_rejects_an_anonymous_request() -> None:
    service = _service_with_mocks()
    service.memory.load = AsyncMock(
        return_value={
            "messages": [], "summary": "", "current_intent": None, "legal_category": None,
            "owner_user_id": "user-A",
        }
    )
    request = ChatRequest(question="what is fir", session_id="user-a-session")
    from app.core.exceptions import ForbiddenError

    try:
        asyncio.run(service.answer(request))
        assert False, "expected ForbiddenError"
    except ForbiddenError:
        pass


def test_session_owned_by_its_own_account_is_not_rejected() -> None:
    service = _service_with_mocks()
    service.memory.load = AsyncMock(
        return_value={
            "messages": [], "summary": "", "current_intent": None, "legal_category": None,
            "owner_user_id": "user-A",
        }
    )
    # A real chunk, not an empty list: under the strict-RAG guardrail an empty
    # `ranked` short-circuits before the LLM is ever called, which would never
    # exercise this test's actual point (an owned session isn't rejected).
    chunk = RetrievedChunk(
        chunk_id="c1", text="An FIR is a First Information Report filed under BNSS.", score=0.9,
        metadata={"source_document": "BNSS", "act_name": "Bharatiya Nagarik Suraksha Sanhita", "section_number": "173"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    # Phase 1 item 3: the stub answer has to read like a real answer now --
    # the legal-answer quality gate rejects a 12-character completion as
    # "too short to be a legal answer", which would mask this test's actual
    # subject (that an owned session isn't rejected) behind a quality
    # downgrade. The assertion is unchanged in kind: the LLM's own text is
    # what comes back, not a fallback.
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="An FIR is a First Information Report registered under Section 173 of the BNSS when "
                    "information disclosing a cognizable offence is received.",
            model="test", provider="test",
        )
    )
    request = ChatRequest(question="what is fir", session_id="user-a-session")
    response = asyncio.run(service.answer(request, authenticated_user_id="user-A"))
    assert response.answer.startswith("An FIR is a First Information Report registered under Section 173")


def test_unclaimed_session_is_claimed_on_first_authenticated_turn() -> None:
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="An FIR is...", model="test", provider="test"))
    request = ChatRequest(question="what is fir", session_id="fresh-session")
    asyncio.run(service.answer(request, authenticated_user_id="user-A"))
    # `update` is also legitimately called again later in the turn for
    # unrelated state (language_preference etc.) -- assert the claim call
    # happened, not that it was the only call.
    service.memory.update.assert_any_call("fresh-session", owner_user_id="user-A")


def test_already_claimed_session_is_not_reclaimed() -> None:
    service = _service_with_mocks()
    service.memory.load = AsyncMock(
        return_value={
            "messages": [], "summary": "", "current_intent": None, "legal_category": None,
            "owner_user_id": "user-A",
        }
    )
    service.memory.append = AsyncMock(
        return_value={
            "messages": [], "summary": "", "current_intent": None, "legal_category": None,
            "owner_user_id": "user-A",
        }
    )
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="An FIR is...", model="test", provider="test"))
    request = ChatRequest(question="what is fir", session_id="user-a-session")
    asyncio.run(service.answer(request, authenticated_user_id="user-A"))
    # `update` is legitimately called for other end-of-turn state -- only
    # the claim call (an `owner_user_id` kwarg) must never happen again.
    for call in service.memory.update.call_args_list:
        assert "owner_user_id" not in call.kwargs


def test_rag_llm_error_does_not_leak_as_answer() -> None:
    # Reproduces a real incident: the user's message classified as Legal
    # Advice, retrieval found a relevant chunk, but the Gemini API call
    # itself failed (network unreachable) -- `LLMResponse.content` is the
    # provider's own friendly error string, non-empty, so the pre-fix code
    # (which only checked for *empty* content) used it as the answer
    # verbatim, disclaimer and all.
    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="c1", text="Section 303: theft should be reported via a police FIR immediately.", score=0.9,
        metadata={"source_document": "BNS", "act_name": "Bharatiya Nyaya Sanhita", "section_number": "303"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("mera bike chori ho gya hai", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="Gemini API is unreachable. Check your network connection and try again.",
            model="test", provider="gemini", error="connection_failed",
        )
    )
    request = ChatRequest(question="mera bike chori ho gya hai mujhe uske liye kya krna hoga")
    response = asyncio.run(service.answer(request))
    assert "unreachable" not in response.answer.lower()
    assert "303" in response.answer or "Bharatiya Nyaya Sanhita" in response.answer
    service.response_cache.store.assert_not_awaited()


def test_ungrounded_answer_is_downgraded_to_insufficient_context() -> None:
    # Grounding validation: a chunk that clears the relevance gate but whose
    # metadata is missing `source_document` produces zero source citations
    # (`_citation_from_chunk` skips it) -- the LLM would still happily
    # generate real prose from it, indistinguishable from a properly
    # grounded answer unless something checks that citations actually back
    # it. Must be downgraded to the same safe insufficient-context message
    # used when retrieval finds nothing, with confidence 0 and no caching.
    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="c1", text="Vehicle theft should be reported via a police FIR immediately.", score=0.9,
        metadata={"act_name": "Indian Penal Code", "section_number": "379"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("mera bike chori ho gya hai", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="Report the theft at the nearest police station.", model="test", provider="test")
    )
    request = ChatRequest(question="mera bike chori ho gya hai mujhe uske liye kya krna hoga")
    response = asyncio.run(service.answer(request))
    # The refusal is returned in the language the user asked in -- this question
    # is Hinglish, so a Hinglish speaker gets Hinglish, not the Devanagari
    # original that used to be sent to everyone regardless of language.
    assert response.answer.startswith(no_verified_context_message("hinglish"))
    assert response.confidence == 0.0
    assert response.sources == []
    service.response_cache.store.assert_not_awaited()


def test_grounded_answer_with_real_citations_is_not_downgraded() -> None:
    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="c1", text="Section 303: theft should be reported via a police FIR immediately.", score=0.9,
        metadata={"source_document": "BNS", "act_name": "Bharatiya Nyaya Sanhita", "section_number": "303"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("mera bike chori ho gya hai", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="Report the theft at the nearest police station.", model="test", provider="test")
    )
    request = ChatRequest(question="mera bike chori ho gya hai mujhe uske liye kya krna hoga")
    response = asyncio.run(service.answer(request))
    assert response.confidence > 0.0
    assert len(response.sources) == 1
    assert "police station" in response.answer.lower()


def test_rag_llm_error_fallback_names_relevant_source_without_dumping_raw_chunk() -> None:
    """Part 41 relevance gate regression test: reproduces the exact live
    failure (a public-nuisance notice template outranking a genuinely
    on-topic FIR chunk after reranking, then getting dumped verbatim once
    the primary LLM call timed out). The off-topic chunk is `chunks[0]`
    (the reranker's own top pick) -- `_fallback_answer` must promote the
    lower-ranked but actually-relevant chunk instead.
    """
    service = _service_with_mocks()
    off_topic_chunk = RetrievedChunk(
        chunk_id="nuisance", score=0.25,
        text="appear to me that you have caused an obstruction or nuisance to persons using the public roadway",
        metadata={"act_name": "The Bharatiya Nagarik Suraksha Sanhita", "section_number": None},
    )
    on_topic_chunk = RetrievedChunk(
        chunk_id="fir", score=0.20,
        text="Registration of FIR is mandatory under Section 154 if the information discloses a cognizable offence.",
        metadata={"act_name": "Bharatiya Nagarik Suraksha Sanhita", "section_number": "173"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("What is an FIR?", [off_topic_chunk, on_topic_chunk]))
    service.reranker.rerank = AsyncMock(return_value=[off_topic_chunk, on_topic_chunk])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="", model="test", provider="gemini", error="timeout")
    )
    response = asyncio.run(service.answer(ChatRequest(question="What is an FIR?")))
    assert "obstruction" not in response.answer.lower()
    assert "registration of fir" not in response.answer.lower()
    assert "bharatiya nagarik suraksha sanhita" in response.answer.lower()
    assert "section 173" in response.answer.lower()
    assert response.retryable is True
    assert response.failure_category == "timeout"


def test_general_conversation_llm_error_does_not_leak_as_answer() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="Gemini API is unreachable. Check your network connection and try again.",
            model="test", provider="gemini", error="connection_failed",
        )
    )
    request = ChatRequest(question="Thanks, that helps")
    response = asyncio.run(service.answer(request))
    assert "unreachable" not in response.answer.lower()
    assert response.answer == "Happy to help — what would you like to ask?"


def test_answer_stream_does_not_leak_error_as_first_chunk() -> None:
    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="c1", text="Section 303: theft should be reported via a police FIR immediately.", score=0.9,
        metadata={"source_document": "BNS", "act_name": "Bharatiya Nyaya Sanhita", "section_number": "303"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("mera bike chori ho gya hai", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.stream = _mock_stream(["Gemini API is unreachable. Check your network connection and try again."])
    request = ChatRequest(question="mera bike chori ho gya hai mujhe uske liye kya krna hoga")
    events = asyncio.run(_collect_stream(service.answer_stream(request)))
    token_texts = [event["data"] for event in events if event["event"] == "token"]
    assert not any("unreachable" in text.lower() for text in token_texts)
    done_event = next(event for event in events if event["event"] == "done")
    assert "303" in done_event["data"]["answer"] or "Bharatiya Nyaya Sanhita" in done_event["data"]["answer"]


def test_answer_stream_streams_tokens_for_grounded_rag() -> None:
    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="c1", text="An FIR is a First Information Report.", score=0.9,
        metadata={"source_document": "CrPC", "act_name": "Code of Criminal Procedure", "section_number": "154"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.stream = _mock_stream(["An FIR ", "is a police report."])
    service.llm.chat = AsyncMock(side_effect=AssertionError("streaming path should use .stream(), not .chat()"))
    request = ChatRequest(question="What is FIR?")
    events = asyncio.run(_collect_stream(service.answer_stream(request)))
    token_events = [event for event in events if event["event"] == "token"]
    done_events = [event for event in events if event["event"] == "done"]
    assert [event["data"] for event in token_events] == ["An FIR ", "is a police report."]
    assert len(done_events) == 1
    assert "An FIR is a police report." in done_events[0]["data"]["answer"]
    assert done_events[0]["data"]["sources"]
    recorded_roles = [call.args[1:3] for call in service.memory.append.await_args_list]
    assert ("user", "What is FIR?") in recorded_roles


def _ambiguous_section_lookup_chunk() -> RetrievedChunk:
    # `_section_lookup_ambiguous_acts` (app/services/chat_service.py) reads
    # `ranked[0].metadata["ambiguous_acts"]` -- the retriever's own signal
    # that a bare "Section N" matched more than one Act with no way to tell
    # which. `score=0.9` clears `is_relevant_chunk`'s unconditional-accept
    # floor (0.6) so this chunk survives the relevance gate regardless of
    # any other heuristic.
    return RetrievedChunk(
        chunk_id="c1", text="Whoever commits culpable homicide not amounting to murder...", score=0.9,
        metadata={"section_number": "302", "ambiguous_acts": ["Indian Penal Code", "Bharatiya Nyaya Sanhita"]},
    )


def test_answer_persists_pending_section_lookup_act_disambiguation_state() -> None:
    """Baseline: `answer()` already gets this right -- asserted here so a
    future change to `_finalize_rag_response` can't silently break it while
    only being checked against the streaming test below.
    """
    service = _service_with_mocks()
    chunk = _ambiguous_section_lookup_chunk()
    service.retriever.retrieve = AsyncMock(return_value=("Section 302", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(side_effect=AssertionError("an ambiguous-Act lookup must not call the LLM"))

    response = asyncio.run(service.answer(ChatRequest(question="Section 302")))

    assert "Indian Penal Code" in response.answer and "Bharatiya Nyaya Sanhita" in response.answer
    update_calls = [call.kwargs for call in service.memory.update.await_args_list]
    assert any(
        call.get("pending_clarification") == "section_lookup_act"
        and call.get("pending_section_query") == "Section 302"
        and call.get("pending_section_acts") == ["Indian Penal Code", "Bharatiya Nyaya Sanhita"]
        for call in update_calls
    )


def test_answer_stream_now_also_persists_pending_section_lookup_act_disambiguation_state() -> None:
    """Regression test for the bug the `_finalize_rag_response`/`answer_stream`
    merge fixed: streaming's own tail used to hardcode
    `pending_clarification: None` and never set `pending_section_query`/
    `pending_section_acts` at all, so a streamed "Section 302" -> "which
    Act?" exchange had no state for a follow-up naming the Act (e.g. "IPC")
    to resolve against -- it was silently treated as a brand-new, vague
    query instead. Now that `answer_stream` calls the same
    `_finalize_rag_response` `answer()` does, this must match the baseline
    test above exactly.
    """
    service = _service_with_mocks()
    chunk = _ambiguous_section_lookup_chunk()
    service.retriever.retrieve = AsyncMock(return_value=("Section 302", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.stream = _mock_stream([])
    service.llm.chat = AsyncMock(side_effect=AssertionError("an ambiguous-Act lookup must not call the LLM"))

    events = asyncio.run(_collect_stream(service.answer_stream(ChatRequest(question="Section 302"))))

    done_event = next(event for event in events if event["event"] == "done")
    answer = done_event["data"]["answer"]
    assert "Indian Penal Code" in answer and "Bharatiya Nyaya Sanhita" in answer
    update_calls = [call.kwargs for call in service.memory.update.await_args_list]
    assert any(
        call.get("pending_clarification") == "section_lookup_act"
        and call.get("pending_section_query") == "Section 302"
        and call.get("pending_section_acts") == ["Indian Penal Code", "Bharatiya Nyaya Sanhita"]
        for call in update_calls
    )


def test_answer_stream_now_logs_chat_pipeline_timing_like_answer_does() -> None:
    """Second confirmed divergence the merge fixed: `answer_stream` built its
    own `timings` dict but never logged it, so streaming turns were
    invisible to anything reading the `chat_pipeline_timing` event.
    """
    import structlog

    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="c1", text="An FIR is a First Information Report.", score=0.9,
        metadata={"source_document": "CrPC", "act_name": "Code of Criminal Procedure", "section_number": "154"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.stream = _mock_stream(["An FIR is a police report."])

    with structlog.testing.capture_logs() as logs:
        asyncio.run(_collect_stream(service.answer_stream(ChatRequest(question="What is FIR?"))))

    assert any(entry["event"] == "chat_pipeline_timing" for entry in logs)


def test_streaming_fallback_to_answer_does_not_double_log_the_intent_event() -> None:
    # `answer_stream` falls back to the full `answer()` pipeline for
    # "Lawyer Recommendation" (a `non_streaming_intents` member) --
    # `answer()`'s own `_dispatch_conversation_intent` logs the real intent
    # event against persisted memory; `answer_stream` must not log a second
    # one against its local, unpersisted preview for the same turn.
    service = _service_with_mocks()
    service.memory.append_intent_event = AsyncMock(return_value=None)
    service.llm.chat = AsyncMock(side_effect=AssertionError("no LLM call needed for a deterministic recommendation"))
    request = ChatRequest(question="Can you recommend a lawyer for cheque bounce")
    events = asyncio.run(_collect_stream(service.answer_stream(request)))
    done_event = next(event for event in events if event["event"] == "done")
    assert done_event["data"]["conversation_intent"] == "Lawyer Recommendation"
    assert service.memory.append_intent_event.await_count == 1


def test_streaming_correction_reroutes_to_the_corrected_request() -> None:
    # Correction re-routing must work identically over `/chat/stream` --
    # "Draft Generation"-classified corrections still fall back to the full
    # pipeline (Draft Generation is a `non_streaming_intents` member), which
    # is where the actual reroute-and-recurse happens.
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(side_effect=AssertionError("no LLM call needed for a deterministic recommendation"))
    request = ChatRequest(question="No, I meant recommend a lawyer for cheque bounce")
    events = asyncio.run(_collect_stream(service.answer_stream(request)))
    done_event = next(event for event in events if event["event"] == "done")
    assert done_event["data"]["conversation_intent"] == "Lawyer Recommendation"
    assert done_event["data"]["lawyer_recommendation"] is not None


def test_answer_stream_does_not_double_append_user_message_on_fallback() -> None:
    # "Translate." with no prior reply falls back to the full `answer()`
    # pipeline (Translation isn't streamed) -- `answer()` performs its own
    # `memory.append` for the user's message, so `answer_stream` must not
    # also append it via its own preview step, or the turn gets recorded twice.
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(side_effect=AssertionError("ambiguous translation should not call the LLM"))
    request = ChatRequest(question="Translate.")
    events = asyncio.run(_collect_stream(service.answer_stream(request)))
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["conversation_intent"] == "Translation"
    user_append_calls = [
        call for call in service.memory.append.await_args_list if call.args[1:] == ("user", "Translate.")
    ]
    assert len(user_append_calls) == 1


def test_answer_stream_falls_back_to_full_pipeline_for_non_rag_intents() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Happy to help!", model="test", provider="test"))
    request = ChatRequest(question="Thanks, that helps")
    events = asyncio.run(_collect_stream(service.answer_stream(request)))
    assert len(events) == 2
    assert events[0] == {"event": "token", "data": "Happy to help!"}
    assert events[1]["event"] == "done"
    assert events[1]["data"]["conversation_intent"] == "General Conversation"


def test_answer_stream_does_not_cache_streamed_answers() -> None:
    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="c1", text="An FIR is a First Information Report.", score=0.9,
        metadata={"source_document": "CrPC"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.stream = _mock_stream(["An FIR is a police report."])
    request = ChatRequest(question="What is FIR?")
    asyncio.run(_collect_stream(service.answer_stream(request)))
    service.response_cache.store.assert_not_awaited()


def test_followup_question_falls_through_to_rag_with_resolved_question() -> None:
    # `memory["messages"]` mirrors chat_service's real shape: the current
    # message is already appended as the last entry by the time the
    # classifier/handlers run (memory.append happens before classify).
    messages = [
        {"role": "user", "content": "What is FIR?"},
        {"role": "assistant", "content": "An FIR is..."},
        {"role": "user", "content": "What about that?"},
    ]
    service = _service_with_mocks(messages)
    service.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(content="What is the process to file an FIR?", model="test", provider="test"),
            LLMResponse(content="Here's the process.", model="test", provider="test"),
        ]
    )
    service.retriever.retrieve = AsyncMock(return_value=("what is the process to file an fir", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    request = ChatRequest(question="What about that?")
    response = asyncio.run(service.answer(request))
    assert response.conversation_intent == "Follow-up Question"
    service.retriever.retrieve.assert_called_once()
    called_question = service.retriever.retrieve.call_args[0][0]
    assert called_question == "What is the process to file an FIR?"

def test_repeated_section_misattribution_falls_back_to_safe_refusal() -> None:
    """Live repro: an FIR-refusal answer attached the SP-escalation right to
    "Section 173", then (told 173 doesn't support it) the one bounded retry
    just swapped in a DIFFERENT wrong number ("Section 64") instead of
    dropping the citation -- neither retrieved section's own text supports
    the claim at all. Unlike a repeated language-retry miss (still a
    useful answer), a repeated section-attribution miss means the specific
    claim could not be honestly grounded on a second, explicitly-guided
    attempt, so the final response must be the safe refusal, never either
    wrong guess.
    """
    unrelated_173 = RetrievedChunk(
        chunk_id="c173", text="Furnish to the accused a copy of the police report and the FIR recorded under section 173.",
        score=0.9, metadata={"act_name": "BNSS", "section_number": "173", "source_document": "bnss.pdf"},
    )
    unrelated_64 = RetrievedChunk(
        chunk_id="c64", text="Discharge of person apprehended. Arrest to be made strictly according to the Sanhita.",
        score=0.9, metadata={"act_name": "BNSS", "section_number": "64", "source_document": "bnss.pdf"},
    )
    service = _service_with_mocks()
    service.retriever.retrieve = AsyncMock(return_value=("fir refusal", [unrelated_173, unrelated_64]))
    service.reranker.rerank = AsyncMock(return_value=[unrelated_173, unrelated_64])
    service.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                content=(
                    "Agar police FIR na likhe, to SP ko likhit complaint bhejein. Ye adhikar Section 173 "
                    "mein diya gaya hai."
                ),
                model="test", provider="test",
            ),
            LLMResponse(
                content=(
                    "Agar police FIR na likhe, to SP ko likhit complaint bhejein. Ye adhikar Section 64 "
                    "mein diya gaya hai."
                ),
                model="test", provider="test",
            ),
        ]
    )
    request = ChatRequest(question="agar police FIR na likhe to kya karna chahiye")
    response = asyncio.run(service.answer(request))
    assert response.no_verified_context is True
    assert "Section 173" not in response.answer
    assert "Section 64" not in response.answer
    # Both scripted answers (Section 173, then Section 64) were consumed --
    # confirms the one bounded retry actually ran, rather than the refusal
    # coming from some earlier, unrelated short-circuit.
    assert service.llm.chat.await_count >= 2


# Part 42 "Answer Quality & Intent Accuracy" regression tests.

def test_bail_chahiye_after_unrelated_topic_does_not_drift() -> None:
    """Confirmed root cause: "Bail chahiye" after a Legal Notice discussion
    used to be forced into "Follow-up Question", whose resolver then built
    its LLM prompt from the conversation history WITHOUT the current
    message -- drifting back to "Legal Notice" instead of resolving "Bail
    chahiye" itself. With the classifier fix, a short message naming its own
    legal topic is never routed through that resolver at all -- the
    retriever must see the real question, "Bail chahiye", untouched.
    """
    messages = [
        {"role": "user", "content": "Legal notice kaise likhen?"},
        {"role": "assistant", "content": "Yahan legal notice likhne ka tarika hai..."},
        {"role": "user", "content": "Bail chahiye"},
    ]
    service = _service_with_mocks(messages)
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Bail ka matlab hai...", model="test", provider="test"))
    service.retriever.retrieve = AsyncMock(return_value=("bail chahiye", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    response = asyncio.run(service.answer(ChatRequest(question="Bail chahiye")))
    assert response.conversation_intent != "Follow-up Question"
    service.retriever.retrieve.assert_called_once()
    called_question = service.retriever.retrieve.call_args[0][0]
    assert "bail" in called_question.lower()
    assert "notice" not in called_question.lower()


def test_off_domain_message_skips_rag_and_disclaimer() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="I'm a legal assistant and can't help with the weather.", model="test", provider="test")
    )
    response = asyncio.run(service.answer(ChatRequest(question="Aaj weather kaisa hai?")))
    assert response.conversation_intent == "Out of Domain"
    assert response.sources == []
    assert "This information is provided for educational purposes only" not in response.answer


def test_off_domain_message_with_prior_legal_context_does_not_inherit_it() -> None:
    messages = [
        {"role": "user", "content": "Bail kaise milti hai?"},
        {"role": "assistant", "content": "Bail milne ka process yeh hai..."},
        {"role": "user", "content": "Aaj weather kaisa hai?"},
    ]
    service = _service_with_mocks(messages)
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="I'm a legal assistant and can't help with the weather.", model="test", provider="test")
    )
    response = asyncio.run(service.answer(ChatRequest(question="Aaj weather kaisa hai?")))
    assert response.conversation_intent == "Out of Domain"
    assert response.sources == []


def test_non_indian_jurisdiction_skips_rag_and_disclaimer() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="I'm focused on Indian law...", model="test", provider="test")
    )
    response = asyncio.run(service.answer(ChatRequest(question="US me divorce ka process kya hai?")))
    assert response.conversation_intent == "Non-Indian Jurisdiction"
    assert response.sources == []
    assert "This information is provided for educational purposes only" not in response.answer


def test_response_modification_still_works_after_routing_fixes() -> None:
    messages = [
        {"role": "user", "content": "FIR kya hoti hai?"},
        {"role": "assistant", "content": "FIR ek official record hota hai jab police ko crime ki information di jaati hai."},
        {"role": "user", "content": "simple words"},
    ]
    service = _service_with_mocks(messages)
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Shortened FIR answer.", model="test", provider="test"))
    response = asyncio.run(service.answer(ChatRequest(question="simple words")))
    assert response.conversation_intent == "Response Modification"
    assert response.sources == []


# Part 51 "Uploaded Document Conversation Context" -- routes the "Document
# Analysis" conv_intent to the existing `DocumentService.analyze` instead of
# generic RAG, resolving pronoun/short-reference follow-ups to whichever
# document was most recently uploaded this session.


def _document_analysis_result() -> DocumentAnalysisResponse:
    return DocumentAnalysisResponse(
        executive_summary="This is a rental agreement between two parties for a residential flat.",
        legal_summary="Reviewed against the indexed chunks.",
        important_clauses=["Security deposit is non-refundable if vacated early."],
        important_dates=[],
        important_names=[],
        important_sections=[],
        key_risks=["Landlord may enter without prior notice."],
        action_items=["Verify document authenticity.", "Preserve the original copy.", "Consult a qualified advocate."],
        missing_information=[],
        structured_data={"analysis_type": "general"},
        sources=[],
        recommended_lawyer=LawyerRecommendation(category="Property Law", confidence=0.7, reason="Rental agreement review."),
        confidence=0.64,
    )


def _service_with_last_uploaded_document(document_id: str | None, messages: list[dict[str, str]] | None = None) -> ChatService:
    service = _service_with_mocks(messages)
    memory = {"messages": messages or [], "summary": "", "current_intent": None, "legal_category": None, "last_uploaded_document_id": document_id}
    service.memory.append = AsyncMock(return_value=memory)
    service.memory.load = AsyncMock(return_value=memory)
    return service


def test_pdf_explain_kro_resolves_last_uploaded_document() -> None:
    service = _service_with_last_uploaded_document("doc-abc")
    service.document_service.analyze = AsyncMock(return_value=_document_analysis_result())
    response = asyncio.run(service.answer(ChatRequest(question="pdf explain kro")))
    assert response.conversation_intent == "Document Analysis"
    service.document_service.analyze.assert_awaited_once()
    sent_request = service.document_service.analyze.call_args.args[0]
    assert sent_request.document_id == "doc-abc"
    assert "rental agreement" in response.answer.lower()


def test_isme_kya_hai_resolves_last_uploaded_document() -> None:
    # Proves resolution isn't keyed to the exact "pdf"/"explain" phrasing --
    # any pronoun/short-reference follow-up means the same document.
    service = _service_with_last_uploaded_document("doc-abc")
    service.document_service.analyze = AsyncMock(return_value=_document_analysis_result())
    response = asyncio.run(service.answer(ChatRequest(question="isme kya important hai")))
    assert response.conversation_intent == "Document Analysis"
    assert service.document_service.analyze.call_args.args[0].document_id == "doc-abc"


def test_explicit_document_id_takes_priority_over_last_uploaded() -> None:
    service = _service_with_last_uploaded_document("doc-last-uploaded")
    service.document_service.analyze = AsyncMock(return_value=_document_analysis_result())
    request = ChatRequest(question="pdf explain kro", metadata_filters={"document_id": "doc-explicit"})
    asyncio.run(service.answer(request))
    assert service.document_service.analyze.call_args.args[0].document_id == "doc-explicit"


def test_no_uploaded_document_asks_for_clarification_without_calling_analyze() -> None:
    service = _service_with_last_uploaded_document(None)
    service.document_service.analyze = AsyncMock(side_effect=AssertionError("must not analyze without a resolved document"))
    response = asyncio.run(service.answer(ChatRequest(question="pdf explain kro")))
    assert response.conversation_intent == "Document Analysis"
    assert "upload" in response.answer.lower()
    assert "This information is provided for educational purposes only" not in response.answer


def test_document_access_denied_returns_friendly_message_no_raw_exception() -> None:
    service = _service_with_last_uploaded_document("doc-not-mine")
    service.document_service.analyze = AsyncMock(side_effect=ForbiddenError("You do not have access to this document."))
    response = asyncio.run(service.answer(ChatRequest(question="pdf explain kro")))
    assert response.conversation_intent == "Document Analysis"
    assert "forbiddenerror" not in response.answer.lower()
    assert "traceback" not in response.answer.lower()
    assert "access" in response.answer.lower()


def test_document_analysis_disclaimer_present_on_real_analysis_absent_on_clarification() -> None:
    analyzed = _service_with_last_uploaded_document("doc-abc")
    analyzed.document_service.analyze = AsyncMock(return_value=_document_analysis_result())
    analyzed_response = asyncio.run(analyzed.answer(ChatRequest(question="pdf explain kro")))
    assert "This information is provided for educational purposes only" in analyzed_response.answer

    no_doc = _service_with_last_uploaded_document(None)
    no_doc.document_service.analyze = AsyncMock(side_effect=AssertionError("must not analyze"))
    no_doc_response = asyncio.run(no_doc.answer(ChatRequest(question="pdf explain kro")))
    assert "This information is provided for educational purposes only" not in no_doc_response.answer


def test_ordinary_legal_question_unaffected_by_document_analysis_routing() -> None:
    service = _service_with_last_uploaded_document("doc-abc")
    service.document_service.analyze = AsyncMock(side_effect=AssertionError("must not be called for a normal legal question"))
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="[GENERAL KNOWLEDGE ANSWER]", model="test", provider="test")
    )
    response = asyncio.run(service.answer(ChatRequest(question="What is FIR?")))
    assert response.conversation_intent != "Document Analysis"


def test_streaming_falls_back_to_answer_for_document_analysis() -> None:
    service = _service_with_last_uploaded_document("doc-abc")
    service.document_service.analyze = AsyncMock(return_value=_document_analysis_result())
    events = asyncio.run(_collect_stream(service.answer_stream(ChatRequest(question="pdf explain kro"))))
    done_event = next(event for event in events if event["event"] == "done")
    assert done_event["data"]["conversation_intent"] == "Document Analysis"
    service.document_service.analyze.assert_awaited_once()


def test_correction_reroutes_to_corrected_request_without_duplicate_intent_event() -> None:
    service = _service_with_mocks()
    service.llm.chat = AsyncMock(side_effect=AssertionError("no LLM call needed for a deterministic recommendation"))
    service.memory.append_intent_event = AsyncMock(return_value=None)
    request = ChatRequest(question="No, I meant recommend a lawyer for cheque bounce")
    response = asyncio.run(service.answer(request))
    # Routing/answer content must reflect the corrected request, not the
    # original wrong one.
    assert response.conversation_intent == "Lawyer Recommendation"
    assert response.lawyer_recommendation is not None
    # Exactly two intent-history writes: one audit entry for the original
    # (wrong) message tagged as superseded, and one real classification
    # event for the corrected text logged by the recursive `answer()` call
    # -- not a duplicate write of the same event.
    events = [call.args[1] for call in service.memory.append_intent_event.call_args_list]
    assert len(events) == 2
    assert events[0]["question"] == "No, I meant recommend a lawyer for cheque bounce"
    assert events[0]["is_correction"] is True
    assert events[0]["corrected_text"] == "recommend a lawyer for cheque bounce"
    assert events[1]["question"] == "recommend a lawyer for cheque bounce"
    assert events[1].get("is_correction") is not True


def test_correction_without_extractable_text_is_not_rerouted() -> None:
    messages = [
        {"role": "user", "content": "What is bail?"},
        {"role": "assistant", "content": "Bail is..."},
    ]
    service = _service_with_mocks(messages)
    service.memory.append_intent_event = AsyncMock(return_value=None)
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(content="[GENERAL KNOWLEDGE ANSWER]", model="test", provider="test")
    )
    service.retriever.retrieve = AsyncMock(return_value=("you misunderstood", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    response = asyncio.run(service.answer(ChatRequest(question="You misunderstood")))
    # No new request text to route to -- must not recurse/re-answer, and
    # falls through to normal handling for this message.
    assert response is not None
    events = [call.args[1] for call in service.memory.append_intent_event.call_args_list]
    assert len(events) == 1
    assert events[0]["is_correction"] is True


# Phase 1 "Multi-Intent Workflow Orchestration" -- the flagship chain:
# "review this PDF, flag risky clauses, draft a notice based on it."


_WORKFLOW_QUESTION = "Please review this uploaded PDF document, tell me the risky clauses, and draft a legal notice based on it."


def test_document_analysis_then_draft_generation_asks_only_for_missing_fields() -> None:
    service = _service_with_last_uploaded_document("doc-abc")
    service.document_service.analyze = AsyncMock(return_value=_document_analysis_result())
    service.llm.chat = AsyncMock(side_effect=AssertionError("must not generate a draft while fields are missing"))
    response = asyncio.run(service.answer(ChatRequest(question=_WORKFLOW_QUESTION)))
    assert response.workflow_chain == ["Document Analysis", "Draft Generation"]
    assert response.workflow_status == "awaiting_missing_fields"
    service.document_service.analyze.assert_awaited_once()
    assert response.draft is not None
    assert response.draft.stage == "collecting"


def test_document_analysis_then_draft_generation_reaches_preview_never_auto_approves() -> None:
    service = _service_with_last_uploaded_document("doc-abc")
    service.document_service.analyze = AsyncMock(return_value=_document_analysis_result())
    # `LegalDraftEngine.generate` persists to Mongo internally -- mocked
    # directly (same boundary `test_draft_lifecycle.py` mocks at) so this
    # stays a unit test of the workflow's own field-seeding/status logic,
    # not an integration test of the drafting engine's persistence, which
    # `test_drafting.py`/`test_draft_lifecycle.py` already cover.
    service.draft_conversation.draft_engine.generate = AsyncMock(
        return_value=DraftGenerateResponse(
            status="complete", draft_id="draft-workflow-1", template_id="legal_notice",
            template_name="General Legal Notice", full_text="Generated notice content.", word_count=42,
        )
    )
    all_fields = {
        "applicant_name": "Ramesh Kumar", "applicant_address": "123 MG Road", "applicant_mobile": "9999999999",
        "respondent_name": "Suresh Sharma", "respondent_address": "456 Park Street",
        "facts": "Deposit was withheld.", "expected_relief": "Refund the deposit.", "place": "Delhi",
    }
    with patch("app.services.chat_service.map_facts_to_draft_fields", return_value=all_fields):
        response = asyncio.run(service.answer(ChatRequest(question=_WORKFLOW_QUESTION)))
    assert response.workflow_status == "draft_preview_ready"
    assert response.draft is not None
    assert response.draft.stage == "preview"
    assert response.draft.lifecycle_state == "preview_ready"
    service.draft_conversation.draft_engine.generate.assert_awaited_once()


def test_workflow_never_hijacks_an_already_active_draft_session() -> None:
    messages = [{"role": "user", "content": "start a draft"}]
    service = _service_with_mocks(messages)
    active_draft_memory = {
        "messages": messages, "summary": "", "current_intent": None, "legal_category": None,
        "last_uploaded_document_id": "doc-abc",
        "draft_mode": True, "draft_stage": "collecting", "draft_template_id": "legal_notice",
        "draft_fields": {"applicant_name": "Existing User"}, "draft_id": None,
    }
    service.memory.append = AsyncMock(return_value=active_draft_memory)
    service.memory.load = AsyncMock(return_value=active_draft_memory)
    service.document_service.analyze = AsyncMock(side_effect=AssertionError("must not start a new workflow analysis"))
    response = asyncio.run(service.answer(ChatRequest(question=_WORKFLOW_QUESTION)))
    # The existing draft's own already-collected field must survive --
    # never silently reset by the workflow trigger.
    assert response.workflow_chain == []
    service.document_service.analyze.assert_not_awaited()


def test_workflow_with_no_uploaded_document_asks_for_one_without_analyzing() -> None:
    service = _service_with_last_uploaded_document(None)
    service.document_service.analyze = AsyncMock(side_effect=AssertionError("must not analyze with no document"))
    response = asyncio.run(service.answer(ChatRequest(question=_WORKFLOW_QUESTION)))
    assert response.workflow_chain == ["Document Analysis", "Draft Generation"]
    assert response.workflow_status == "failed_no_document"


def test_workflow_denies_analysis_of_a_document_owned_by_someone_else() -> None:
    service = _service_with_last_uploaded_document("doc-not-mine")
    service.document_service.analyze = AsyncMock(side_effect=ForbiddenError("You do not have access to this document."))
    response = asyncio.run(service.answer(ChatRequest(question=_WORKFLOW_QUESTION)))
    assert response.workflow_chain == ["Document Analysis", "Draft Generation"]
    assert response.workflow_status == "failed_ownership"


# Phase 1 "General Clarification Mode" -- ambiguous document-analysis +
# draft-generation phrasing (no "based on it" link) asks which one instead
# of guessing.

_AMBIGUOUS_WORKFLOW_QUESTION = "Review this PDF and draft a legal notice."


def test_ambiguous_workflow_phrasing_asks_for_clarification_and_stores_pending_state() -> None:
    service = _service_with_last_uploaded_document("doc-abc")
    service.llm.chat = AsyncMock(side_effect=AssertionError("clarification must not call the LLM"))
    service.document_service.analyze = AsyncMock(side_effect=AssertionError("must not analyze before clarification"))
    response = asyncio.run(service.answer(ChatRequest(question=_AMBIGUOUS_WORKFLOW_QUESTION)))
    assert response.conversation_intent == "Workflow Clarification"
    update_calls = [call.kwargs for call in service.memory.update.await_args_list]
    assert any(call.get("pending_clarification") == "workflow_intent" for call in update_calls)
    assert any(call.get("pending_workflow_question") == _AMBIGUOUS_WORKFLOW_QUESTION for call in update_calls)
    service.document_service.analyze.assert_not_awaited()


def test_clarification_reply_choosing_analysis_runs_the_full_chain() -> None:
    messages_turn2 = [
        {"role": "user", "content": _AMBIGUOUS_WORKFLOW_QUESTION},
        {"role": "assistant", "content": "Should I review the document first..."},
        {"role": "user", "content": "Pehle PDF ka analysis karo."},
    ]
    pending_memory = {
        "messages": messages_turn2, "summary": "", "current_intent": None, "legal_category": None,
        "last_uploaded_document_id": "doc-abc",
        "pending_clarification": "workflow_intent", "pending_workflow_question": _AMBIGUOUS_WORKFLOW_QUESTION,
    }
    service = _service_with_mocks(messages_turn2)
    service.memory.append = AsyncMock(return_value=pending_memory)
    service.memory.load = AsyncMock(return_value=pending_memory)
    service.document_service.analyze = AsyncMock(return_value=_document_analysis_result())
    service.llm.chat = AsyncMock(side_effect=AssertionError("must not generate a draft while fields are missing"))
    response = asyncio.run(service.answer(ChatRequest(question="Pehle PDF ka analysis karo.")))
    # Routed using the ORIGINAL stored question (which named "legal
    # notice"), not the clarification reply itself.
    assert response.workflow_chain == ["Document Analysis", "Draft Generation"]
    assert response.workflow_status == "awaiting_missing_fields"
    service.document_service.analyze.assert_awaited_once()


def test_clarification_reply_choosing_direct_starts_plain_drafting() -> None:
    messages_turn2 = [
        {"role": "user", "content": _AMBIGUOUS_WORKFLOW_QUESTION},
        {"role": "assistant", "content": "Should I review the document first..."},
        {"role": "user", "content": "Seedha draft karo, PDF ki zaroorat nahi."},
    ]
    pending_memory = {
        "messages": messages_turn2, "summary": "", "current_intent": None, "legal_category": None,
        "last_uploaded_document_id": "doc-abc",
        "pending_clarification": "workflow_intent", "pending_workflow_question": _AMBIGUOUS_WORKFLOW_QUESTION,
    }
    service = _service_with_mocks(messages_turn2)
    service.memory.append = AsyncMock(return_value=pending_memory)
    service.memory.load = AsyncMock(return_value=pending_memory)
    service.document_service.analyze = AsyncMock(side_effect=AssertionError("direct choice must skip analysis"))
    response = asyncio.run(service.answer(ChatRequest(question="Seedha draft karo, PDF ki zaroorat nahi.")))
    service.document_service.analyze.assert_not_awaited()
    assert response.workflow_status == "resolved_direct_draft"
    assert response.draft is not None
    assert response.draft.stage == "collecting"


def test_unrelated_reply_to_clarification_expires_it_without_treating_reply_as_a_fact() -> None:
    messages_turn2 = [
        {"role": "user", "content": _AMBIGUOUS_WORKFLOW_QUESTION},
        {"role": "assistant", "content": "Should I review the document first..."},
        {"role": "user", "content": "What is the punishment for cheque bounce?"},
    ]
    pending_memory = {
        "messages": messages_turn2, "summary": "", "current_intent": None, "legal_category": None,
        "last_uploaded_document_id": "doc-abc",
        "pending_clarification": "workflow_intent", "pending_workflow_question": _AMBIGUOUS_WORKFLOW_QUESTION,
    }
    service = _service_with_mocks(messages_turn2)
    service.memory.append = AsyncMock(return_value=pending_memory)
    service.memory.load = AsyncMock(return_value=pending_memory)
    service.document_service.analyze = AsyncMock(side_effect=AssertionError("unrelated reply must not trigger analysis"))
    service.retriever.retrieve = AsyncMock(return_value=("cheque bounce punishment", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    response = asyncio.run(service.answer(ChatRequest(question="What is the punishment for cheque bounce?")))
    # Falls through to normal handling of the actual message -- never
    # force-interpreted as an answer to the stale clarification.
    assert response.conversation_intent != "Workflow Clarification"
    service.document_service.analyze.assert_not_awaited()


# Phase 1 "Intent Feedback" -- explicit post-hoc correction of the previous
# turn's intent classification, stored separately from legal facts.


def _service_with_intent_history(primary_intent: str) -> ChatService:
    messages = [
        {"role": "user", "content": "some earlier question"},
        {"role": "assistant", "content": "some earlier answer"},
        {"role": "user", "content": "This is document analysis"},
    ]

    def _fresh_memory(*_args: object, **_kwargs: object) -> dict:
        # A fresh shallow copy per call -- `append_intent_event`'s own
        # internal load-mutate-persist cycle (real `ConversationMemoryStore`
        # code, not mocked) must not corrupt the SAME dict object
        # `_respond_with_intent_feedback` reads `intent_history` from later
        # in this same turn, the way a real backing store never would.
        return {
            "messages": messages, "summary": "", "current_intent": None, "legal_category": None,
            "intent_history": [{"primary_intent": primary_intent, "question": "some earlier question"}],
        }

    service = _service_with_mocks(messages)
    service.memory.append = AsyncMock(side_effect=_fresh_memory)
    service.memory.load = AsyncMock(side_effect=_fresh_memory)
    service.intent_feedback.insert = AsyncMock(return_value="feedback-id")
    service.llm.chat = AsyncMock(side_effect=AssertionError("intent feedback must not call the LLM"))
    return service


def test_named_intent_feedback_is_stored_with_original_and_corrected_intent() -> None:
    service = _service_with_intent_history("Legal Advice")
    response = asyncio.run(service.answer(ChatRequest(question="This is document analysis")))
    assert response.conversation_intent == "Intent Feedback"
    service.intent_feedback.insert.assert_awaited_once()
    stored = service.intent_feedback.insert.await_args.args[0]
    assert stored["original_intent"] == "Legal Advice"
    assert stored["corrected_intent"] == "Document Analysis"
    assert stored["message_text"] == "This is document analysis"


def test_bare_wrong_intent_feedback_stores_no_corrected_intent() -> None:
    service = _service_with_intent_history("Legal Advice")
    response = asyncio.run(service.answer(ChatRequest(question="Wrong intent")))
    assert response.conversation_intent == "Intent Feedback"
    stored = service.intent_feedback.insert.await_args.args[0]
    assert stored["original_intent"] == "Legal Advice"
    assert stored["corrected_intent"] is None


def test_intent_feedback_is_scoped_to_the_authenticated_user_not_client_supplied_id() -> None:
    service = _service_with_intent_history("Legal Advice")
    request = ChatRequest(question="This is document analysis", user_id="spoofed-other-user")
    asyncio.run(service.answer(request, authenticated_user_id="real-authenticated-user"))
    stored = service.intent_feedback.insert.await_args.args[0]
    assert stored["owner_user_id"] == "real-authenticated-user"


def test_anonymous_intent_feedback_has_no_owner_user_id() -> None:
    service = _service_with_intent_history("Legal Advice")
    asyncio.run(service.answer(ChatRequest(question="This is document analysis")))
    stored = service.intent_feedback.insert.await_args.args[0]
    assert stored["owner_user_id"] is None


def test_intent_feedback_message_never_reaches_retrieval() -> None:
    service = _service_with_intent_history("Legal Advice")
    service.retriever.retrieve = AsyncMock(side_effect=AssertionError("feedback message must not be retrieved as a question"))
    response = asyncio.run(service.answer(ChatRequest(question="This is document analysis")))
    assert response.conversation_intent == "Intent Feedback"
    assert response.sources == []


def test_intent_feedback_write_failure_never_breaks_the_chat_turn() -> None:
    service = _service_with_intent_history("Legal Advice")
    service.intent_feedback.insert = AsyncMock(side_effect=RuntimeError("MongoDB client is not connected."))
    response = asyncio.run(service.answer(ChatRequest(question="This is document analysis")))
    assert response.conversation_intent == "Intent Feedback"
    assert "document analysis" in response.answer.lower() or "correction" in response.answer.lower()


# ---------------------------------------------------------------------------
# C5 -- explanation_mode must actually change the prompt and be reported back
# ---------------------------------------------------------------------------


def _rag_service_with_a_real_chunk() -> ChatService:
    service = _service_with_mocks()
    chunk = RetrievedChunk(
        chunk_id="c1", text="An FIR is a First Information Report filed under BNSS.", score=0.9,
        metadata={"source_document": "BNSS", "act_name": "Bharatiya Nagarik Suraksha Sanhita", "section_number": "173"},
    )
    service.retriever.retrieve = AsyncMock(return_value=("what is fir", [chunk]))
    service.reranker.rerank = AsyncMock(return_value=[chunk])
    service.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="An FIR is a First Information Report registered under Section 173 of the BNSS when "
                    "information disclosing a cognizable offence is received.",
            model="test", provider="test",
        )
    )
    return service


def test_explanation_mode_detailed_is_reflected_in_the_prompt_and_response() -> None:
    """Security/correctness finding C5: `explanation_mode` must not be a
    no-op -- the request field has to actually change what is sent to the
    LLM, and the response has to say which level was used."""
    from app.language.explanation_level import EXPLANATION_LEVEL_INSTRUCTIONS

    service = _rag_service_with_a_real_chunk()
    request = ChatRequest(question="what is fir", explanation_mode="detailed")

    response = asyncio.run(service.answer(request))

    assert response.explanation_level == "detailed"
    # The FIRST call is the main RAG answer -- `service.llm.chat` is also
    # invoked again afterward for the unrelated "related questions"
    # suggestion, so `call_args` (the LAST call) is not what was actually
    # asked to answer the user's question.
    system_prompt = service.llm.chat.call_args_list[0].args[0][0].content
    assert EXPLANATION_LEVEL_INSTRUCTIONS["detailed"] in system_prompt


def test_explanation_mode_advocate_and_simple_produce_different_prompts() -> None:
    from app.language.explanation_level import EXPLANATION_LEVEL_INSTRUCTIONS

    advocate_service = _rag_service_with_a_real_chunk()
    advocate_response = asyncio.run(
        advocate_service.answer(ChatRequest(question="what is fir", explanation_mode="advocate"))
    )
    simple_service = _rag_service_with_a_real_chunk()
    simple_response = asyncio.run(
        simple_service.answer(ChatRequest(question="what is fir", explanation_mode="simple"))
    )

    assert advocate_response.explanation_level == "advocate"
    assert simple_response.explanation_level == "simple"
    advocate_prompt = advocate_service.llm.chat.call_args_list[0].args[0][0].content
    simple_prompt = simple_service.llm.chat.call_args_list[0].args[0][0].content
    assert advocate_prompt != simple_prompt
    assert EXPLANATION_LEVEL_INSTRUCTIONS["advocate"] in advocate_prompt
    assert EXPLANATION_LEVEL_INSTRUCTIONS["simple"] in simple_prompt


def test_explanation_mode_omitted_with_no_textual_cue_defaults_to_citizen() -> None:
    """Backward compatibility: a request that never mentions `explanation_mode`
    at all must behave exactly as before this field existed."""
    service = _rag_service_with_a_real_chunk()

    response = asyncio.run(service.answer(ChatRequest(question="what is fir")))

    assert response.explanation_level == "citizen"
