"""Integration coverage for the live failures reported against `POST /chat`.

Every case here was first reproduced against a running FastAPI backend with
the real Gemini provider (see the timeline in `app/llm/deadline.py`); these
tests pin the fixes so the same regressions cannot come back silently.

Three of the reported symptoms had one shared cause -- nothing bounded the
SUM of the individual timeouts:

    llm_timeout (120s) x llm_max_retry_attempts (3) on the primary, plus one
    attempt per fallback, plus a separate field-extraction call, all inside a
    request the Streamlit client abandons at 180s.

The provider-level tests below drive `ResilientLLMProvider`/`GeminiProvider`
directly with fake transports rather than reaching the network, so they are
deterministic and fast. The `/chat`-level tests use the FastAPI app.

A small live-server section at the bottom is skipped unless
`LEGAL_AI_LIVE_BASE_URL` is set, so the default `pytest` run needs no server.
"""

import asyncio
import os
import time

import httpx
import pytest

from app.core.config import settings
from app.llm.base import ChatMessage, LLMProvider, LLMResponse
from app.llm.deadline import deadline, remaining_seconds
from app.llm.gemini import GeminiProvider
from app.llm.resilient import ResilientLLMProvider

LIVE_BASE_URL = os.getenv("LEGAL_AI_LIVE_BASE_URL")


class _FakeProvider(LLMProvider):
    """A provider whose every call takes `latency` and returns `response`."""

    def __init__(self, name: str, response: LLMResponse, latency: float = 0.0) -> None:
        self.provider_name = name
        self.model = f"{name}-model"
        self._response = response
        self._latency = latency
        self.calls = 0

    async def chat(self, messages, temperature: float = 0.1) -> LLMResponse:
        self.calls += 1
        if self._latency:
            await asyncio.sleep(self._latency)
        return self._response.model_copy()

    async def stream(self, messages, temperature: float = 0.1):
        yield self._response.content

    async def health(self) -> bool:
        return True


def _error(kind: str, provider: str = "gemini") -> LLMResponse:
    messages = {
        "timeout": "Gemini API is unreachable. Check your network connection and try again.",
        "rate_limit": "Gemini API rate limit or quota reached. Please wait a moment and try again.",
        "auth": "Gemini API key was rejected. Please check GEMINI_API_KEY.",
        "not_found": "Gemini model 'x' was not found.",
    }
    return LLMResponse(
        content=messages[kind], model="m", provider=provider, error=messages[kind], error_kind=kind
    )


def _ok(text: str = "A grounded answer.") -> LLMResponse:
    return LLMResponse(content=text, model="m", provider="fallback")


# --- The budget contract ----------------------------------------------------

def test_server_budget_is_strictly_inside_the_client_timeout() -> None:
    # The whole point: the backend must give up early enough to still SEND a
    # degraded answer, rather than being cut off mid-flight.
    assert settings.chat_request_budget_seconds < settings.client_request_timeout_seconds
    assert settings.client_request_timeout_seconds - settings.chat_request_budget_seconds >= 5


def test_a_bad_budget_override_fails_at_startup_instead_of_at_runtime() -> None:
    from app.core.config import Settings

    with pytest.raises(RuntimeError, match="CHAT_REQUEST_BUDGET_SECONDS"):
        Settings(chat_request_budget_seconds=200, client_request_timeout_seconds=180)


def test_deadline_never_extends_an_outer_one() -> None:
    # The drafting budget may only ever TIGHTEN the request budget, never
    # loosen it -- otherwise an inner budget could outlive the client.
    with deadline(10, label="outer"), deadline(600, label="inner-too-long"):
        assert remaining_seconds() <= 10


# --- Provider timeout, rate limit, fallback ---------------------------------

def test_retries_stop_when_the_deadline_cannot_fit_another_attempt() -> None:
    slow = _FakeProvider("gemini", _error("timeout"), latency=0.15)
    provider = ResilientLLMProvider(slow, [], max_attempts=5, initial_backoff_seconds=0.05)

    async def run():
        with deadline(0.45, label="test"):
            return await provider.chat([ChatMessage(role="user", content="hi")])

    response = asyncio.run(run())
    assert response.error
    # Without the deadline this would have made all 5 attempts.
    assert slow.calls < 5, f"made {slow.calls} attempts despite the deadline"


def test_a_rate_limited_primary_fails_over_to_the_configured_secondary() -> None:
    primary = _FakeProvider("gemini", _error("rate_limit"))
    secondary = _FakeProvider("groq", _ok("From the fallback."))
    provider = ResilientLLMProvider(primary, [secondary], max_attempts=2, initial_backoff_seconds=0.0)

    response = asyncio.run(provider.chat([ChatMessage(role="user", content="hi")]))
    assert response.error is None
    assert response.content == "From the fallback."
    assert secondary.calls == 1


def test_inline_think_tags_are_stripped_from_a_successful_answer() -> None:
    # Live repro: `gemma-4-26b-a4b-it` sometimes writes its chain-of-thought
    # as literal `<think>...</think>` markup inside the normal text part
    # instead of (or in addition to) using the Gemini API's structured
    # "thought" flag -- which `GeminiProvider._parse_response` already
    # filters, but only for THAT flag, not for inline tags any provider's
    # model could emit. Without this, real system-prompt fragments and
    # internal reasoning ("Rule 2's exact fallback line...") reached the
    # user verbatim.
    raw = (
        "<think>The user hasn't asked anything. Per Rule 2, output the "
        "fallback line and nothing else.</think>\n\nHere is your answer."
    )
    primary = _FakeProvider("gemini", _ok(raw))
    provider = ResilientLLMProvider(primary, [], max_attempts=1)
    response = asyncio.run(provider.chat([ChatMessage(role="user", content="hi")]))
    assert response.error is None
    assert "<think" not in response.content.lower()
    assert "Rule 2" not in response.content
    assert response.content == "Here is your answer."


def test_answer_that_is_pure_thinking_retries_then_fails_over() -> None:
    # If stripping the think-block leaves nothing, that is not a usable
    # "successful" answer -- it must be treated like any other failure
    # (retried, then failed over), never handed to the user as an empty
    # response.
    primary = _FakeProvider("gemini", _ok("<think>only reasoning, no answer</think>"))
    secondary = _FakeProvider("groq", _ok("A real answer."))
    provider = ResilientLLMProvider(primary, [secondary], max_attempts=1, initial_backoff_seconds=0.0)
    response = asyncio.run(provider.chat([ChatMessage(role="user", content="hi")]))
    assert response.error is None
    assert response.content == "A real answer."
    assert secondary.calls == 1


@pytest.mark.parametrize("kind", ["auth", "not_found"])
def test_non_retryable_failures_are_not_retried(kind: str) -> None:
    # A rejected key and a missing model fail identically every time; retrying
    # only burns budget that graceful degradation needs.
    primary = _FakeProvider("gemini", _error(kind))
    provider = ResilientLLMProvider(primary, [], max_attempts=3, initial_backoff_seconds=0.0)
    response = asyncio.run(provider.chat([ChatMessage(role="user", content="hi")]))
    assert response.error
    assert primary.calls == 1, f"{kind} was retried {primary.calls} times"


@pytest.mark.parametrize("kind", ["timeout", "rate_limit"])
def test_retryable_failures_are_retried(kind: str) -> None:
    primary = _FakeProvider("gemini", _error(kind))
    provider = ResilientLLMProvider(primary, [], max_attempts=3, initial_backoff_seconds=0.0)
    asyncio.run(provider.chat([ChatMessage(role="user", content="hi")]))
    assert primary.calls == 3


def test_everything_exhausted_still_returns_a_structured_response() -> None:
    provider = ResilientLLMProvider(_FakeProvider("gemini", _error("timeout")), [], max_attempts=1)
    response = asyncio.run(provider.chat([ChatMessage(role="user", content="hi")]))
    assert isinstance(response, LLMResponse)
    assert response.error and response.error_kind


def test_gemini_clamps_its_own_timeout_to_the_remaining_budget() -> None:
    # Reported live: `gemini_overall_timeout timeout_seconds=120.0` fired 24s
    # AFTER the client had already given up. A call must never be started with
    # a ceiling longer than the caller is still willing to wait.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated", request=request)

    provider = GeminiProvider(
        api_key="test-key", model="m", overall_timeout_seconds=120.0,
        transport=httpx.MockTransport(handler), max_retries=0,
    )

    async def run():
        with deadline(2.0, label="test"):
            started = time.monotonic()
            response = await provider.chat([ChatMessage(role="user", content="hi")])
            return response, time.monotonic() - started

    response, elapsed = asyncio.run(run())
    assert response.error
    assert elapsed < 10, f"took {elapsed:.1f}s despite a 2s deadline"


def test_gemini_reports_a_machine_readable_error_kind() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    provider = GeminiProvider(
        api_key="test-key", model="m", transport=httpx.MockTransport(handler), max_retries=0,
    )
    response = asyncio.run(provider.chat([ChatMessage(role="user", content="hi")]))
    assert response.error_kind == "rate_limit"


def test_an_api_key_is_never_written_to_logs() -> None:
    # Gemini authenticates with the key as a QUERY PARAMETER, and httpx logs
    # full request URLs at INFO -- so the live key was appearing in stdout on
    # every call. Confirmed in this app's real log output.
    import logging

    from app.core.logger import configure_logging

    configure_logging()
    for name in ("httpx", "httpcore"):
        assert logging.getLogger(name).level >= logging.WARNING


# --- RAG failure output -----------------------------------------------------

class _Chunk:
    def __init__(self, text: str, metadata: dict, score: float = 0.5) -> None:
        self.text = text
        self.metadata = metadata
        self.score = score


def _service():
    from app.services.chat_service import ChatService

    return ChatService()


def test_llm_failure_never_dumps_a_raw_truncated_chunk() -> None:
    # The reported answer was "I ran into a temporary issue ..." followed by
    # 1200 characters sliced out of a statute -- beginning and ending
    # mid-sentence, and not necessarily on topic.
    long_text = "Whereas the provisions of this section shall apply " + ("x" * 4000)
    chunks = [_Chunk(long_text, {"act_name": "Some Act", "section_number": "21"})]
    answer = _service()._fallback_answer("Rental Dispute", chunks, "landlord notice")
    assert long_text[:200] not in answer
    assert len(answer) < 700
    assert "retry" in answer.lower()


def test_llm_failure_output_states_no_legal_conclusion() -> None:
    chunks = [_Chunk("Some statutory text.", {"act_name": "Some Act", "section_number": "21"})]
    answer = _service()._fallback_answer("Rental Dispute", chunks, "landlord notice")
    assert "Some Act" in answer  # citation preserved
    assert "did not complete" in answer or "not turned the source into a legal conclusion" in answer


def test_placeholder_act_metadata_is_never_shown_as_a_citation() -> None:
    # "This Act" is a parser placeholder, not an Act name, and it reached the
    # user verbatim as "under This Act, 16".
    chunks = [_Chunk("text", {"act_name": "This Act", "source_document": "tenancy.pdf"})]
    answer = _service()._fallback_answer("Rental Dispute", chunks, "landlord")
    assert "This Act" not in answer


def test_no_chunks_at_all_gives_the_insufficient_context_message() -> None:
    from app.core.constants import INSUFFICIENT_CONTEXT_MESSAGE

    assert _service()._fallback_answer("X", [], "q") == INSUFFICIENT_CONTEXT_MESSAGE


# --- Contextual follow-ups --------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "Iska detailed advocate-style answer do",
        "Isko simple words me samjhao",
        "Isme aur detail batao",
    ],
)
def test_pronoun_only_followups_are_classified_as_followups(message: str) -> None:
    # A pronoun-only message carries no topic, so sending it straight to
    # vector retrieval can only ever miss. It must be resolved against the
    # previous turn first.
    from app.intent.classifier import ConversationIntentClassifier

    memory = {
        "messages": [
            {"role": "user", "content": "Mera landlord bina notice ghar khali karwa raha hai"},
            {"role": "assistant", "content": "Eviction ke liye process follow karna zaroori hai."},
            {"role": "user", "content": message},
        ]
    }
    match = ConversationIntentClassifier().classify(message, memory)
    assert match.intent in {"Follow-up Question", "Response Modification"}


def test_followup_resolution_is_what_reaches_retrieval() -> None:
    # `answer()` must send the RESOLVED standalone question to the pipeline,
    # never the bare pronoun message.
    import inspect

    from app.services.chat_service import ChatService

    source = inspect.getsource(ChatService._answer_within_deadline)
    assert "_resolve_followup_question" in source
    assert "question_for_pipeline" in source


# --- Model-law jurisdiction accuracy ---------------------------------------

def test_model_tenancy_act_is_not_presented_as_binding_nationwide() -> None:
    from app.rag.jurisdiction import annotate_answer

    answer = "Model Tenancy Act, 2021 ke tehat landlord bina notice nahi nikaal sakta."
    annotated = annotate_answer(answer, "hinglish", "Mera landlord Ghaziabad me pareshan kar raha hai")
    assert annotated != answer
    assert "model" in annotated.lower()
    assert "State" in annotated or "राज्य" in annotated


def test_the_caveat_names_the_state_when_the_user_gave_one() -> None:
    from app.rag.jurisdiction import annotate_answer, detect_state

    assert detect_state("Mera ghar Ghaziabad me hai") == "Uttar Pradesh"
    annotated = annotate_answer(
        "Model Tenancy Act, 2021 applies.", "english", "My flat is in Ghaziabad"
    )
    assert "Uttar Pradesh" in annotated


def test_the_caveat_asks_for_the_state_when_none_was_given() -> None:
    from app.rag.jurisdiction import annotate_answer

    annotated = annotate_answer("Model Tenancy Act, 2021 applies.", "english", "landlord issue")
    assert "which State" in annotated


def test_an_already_hedged_answer_is_not_annotated_twice() -> None:
    from app.rag.jurisdiction import annotate_answer

    hedged = (
        "The Model Tenancy Act, 2021 is a model law; tenancy is a State subject, so it applies only "
        "where your State has enacted it."
    )
    assert annotate_answer(hedged, "english", "q") == hedged


def test_an_answer_with_no_model_law_is_untouched() -> None:
    from app.rag.jurisdiction import annotate_answer

    answer = "Under the Bharatiya Nyaya Sanhita, 2023, theft is punishable."
    assert annotate_answer(answer, "english", "q") == answer


def test_jurisdiction_warnings_are_machine_readable() -> None:
    from app.rag.jurisdiction import jurisdiction_warnings

    warnings = jurisdiction_warnings("Model Tenancy Act, 2021 says ...", "landlord")
    assert warnings
    assert any("model law" in warning for warning in warnings)


# --- Drafting: checkpoint, degraded draft, idempotent retry -----------------

_NINE_FIELDS = (
    "Applicant Address: 123, Shastri Nagar, Ghaziabad, Uttar Pradesh - 201002 "
    "Alternate Mobile Number: 9876543210 Applicant Name: Rahul Sharma "
    "Expected Relief / What you want: Mobile trace karke wapas dilaya jaye. "
    "Facts of the Case (in your own words): Mera mobile 1 September 2026 ko market mein chori ho gaya. "
    "Maine kaafi search kiya lekin nahi mila. Maine apne number par call bhi kiya. "
    "IMEI Number: 352099123456789 Location Phone Was Stolen From: Ghanta Ghar Market, Ghaziabad "
    "Place: Ghaziabad, Uttar Pradesh Police Station: Kotwali Nagar Police Station, Ghaziabad"
)
# The Ration Card Application's own five required fields, labelled with the
# template's own labels, exactly as a user pastes them after the engine has
# just printed the list.
_FIVE_FIELDS = (
    "Applicant Name (Head of Family): Sunita Devi "
    "Applicant Address: 44, Civil Lines, Jaipur, Rajasthan - 302006 "
    "Mobile Number: 9812345678 "
    "Place: Jaipur "
    "Action Requested / Family Details (list each point): Mujhe naya ration card banwana hai kyunki "
    "purana kho gaya hai. Parivar mein 4 sadasya hain - main, mere pati, aur do bachche."
)


def _engine_with_failing_llm(exc: Exception | None = None):
    from unittest.mock import AsyncMock

    from app.drafting.conversation import DraftConversationEngine

    engine = DraftConversationEngine()
    engine.memory_store.update = AsyncMock(return_value={})
    engine.draft_engine.drafts.insert = AsyncMock(return_value=None)
    engine.draft_engine.drafts.find_by_id = AsyncMock(return_value=None)
    engine.draft_engine.versions.insert = AsyncMock(return_value=None)
    if exc is not None:
        engine.draft_engine.llm.chat = AsyncMock(side_effect=exc)
    return engine


@pytest.mark.parametrize(
    "template_message,fields_message,required_count",
    [
        ("Mobile Theft Complaint banao", _NINE_FIELDS, 9),
        ("Ration Card Application banao", _FIVE_FIELDS, 5),
    ],
)
def test_all_labelled_fields_are_collected_without_any_llm_call(
    template_message: str, fields_message: str, required_count: int
) -> None:
    from unittest.mock import AsyncMock

    engine = _engine_with_failing_llm(TimeoutError("gemini down"))
    # The extraction LLM must not even be consulted when the deterministic
    # pass already has everything -- an 84s call whose output was rejected
    # anyway is what pushed the reported request past the client timeout.
    engine.extractor.llm.chat = AsyncMock(side_effect=AssertionError("must not call the LLM"))

    async def run():
        memory: dict = {}
        await engine.handle_turn("s", template_message, "english", memory)
        result = await engine.handle_turn("s", fields_message, "english", memory)
        return memory, result

    memory, result = asyncio.run(run())
    assert len(memory["draft_fields"]) >= required_count
    assert result.info.stage == "preview"


def test_fields_are_checkpointed_before_generation_is_attempted() -> None:
    from unittest.mock import AsyncMock

    engine = _engine_with_failing_llm(TimeoutError("gemini down"))
    saved: dict = {}
    engine.memory_store.update = AsyncMock(
        side_effect=lambda sid, **kw: saved.update(kw) or {}
    )

    async def run():
        memory: dict = {}
        await engine.handle_turn("s", "Mobile Theft Complaint banao", "english", memory)
        await engine.handle_turn("s", _NINE_FIELDS, "english", memory)

    asyncio.run(run())
    # Durable BEFORE the slow, retried generation call -- so a timeout there
    # cannot discard what the user typed.
    assert len(saved.get("draft_fields") or {}) >= 9


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), RuntimeError("provider down")])
def test_generation_failure_still_produces_a_usable_draft(failure: Exception) -> None:
    engine = _engine_with_failing_llm(failure)

    async def run():
        memory: dict = {}
        await engine.handle_turn("s", "Mobile Theft Complaint banao", "english", memory)
        return await engine.handle_turn("s", _NINE_FIELDS, "english", memory), memory

    result, memory = asyncio.run(run())
    assert result.info.generation_mode == "deterministic"
    assert result.info.full_text and len(result.info.full_text.split()) > 100
    # The workflow stays alive with the fields intact.
    assert memory["draft_mode"] is True
    assert len(memory["draft_fields"]) >= 9


def test_a_hanging_llm_call_still_completes_the_draft_within_a_bounded_time(monkeypatch) -> None:
    """QA (session 2026-09-11/12, `docs/qa/QA_TEST_MATRIX_20260911.md`,
    BUG-013a / "New latency finding"): a live draft-generation call was
    observed stalling 250-317s against a budget `GeminiProvider.chat()` had
    itself computed as ~100s -- i.e. the existing per-call timeout did not
    actually bound the call. Unlike `test_generation_failure_still_
    produces_a_usable_draft` above (a call that fails FAST), this simulates
    a call that never returns at all, and asserts `_generate_once`'s
    `call_with_hard_timeout` backstop (`app/llm/deadline.py`) still forces a
    real, bounded return.
    """
    from app.llm import deadline as deadline_module

    monkeypatch.setattr(settings, "draft_generation_budget_seconds", 0.2)
    monkeypatch.setattr(deadline_module, "HARD_TIMEOUT_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(deadline_module, "MIN_VIABLE_CALL_SECONDS", 0.05)

    engine = _engine_with_failing_llm(None)

    async def _hangs_forever(messages, temperature: float = 0.1, **kwargs):
        await asyncio.Event().wait()

    engine.draft_engine.llm.chat = _hangs_forever

    async def run():
        memory: dict = {}
        await engine.handle_turn("s", "Mobile Theft Complaint banao", "english", memory)
        turn_started = time.perf_counter()
        result = await engine.handle_turn("s", _NINE_FIELDS, "english", memory)
        return result, time.perf_counter() - turn_started

    result, elapsed = asyncio.run(run())
    assert elapsed < 5.0, f"draft generation took {elapsed:.1f}s despite a 0.2s budget -- the hard timeout did not fire"
    assert result.info.generation_mode == "deterministic"
    assert result.info.full_text and len(result.info.full_text.split()) > 100


def test_the_degraded_reply_says_the_details_were_saved() -> None:
    engine = _engine_with_failing_llm(TimeoutError("down"))

    async def run():
        memory: dict = {}
        await engine.handle_turn("s", "Mobile Theft Complaint banao", "english", memory)
        return await engine.handle_turn("s", _NINE_FIELDS, "english", memory)

    reply = asyncio.run(run()).reply_text
    assert "details are saved" in reply.lower()
    assert "temporarily unavailable" in reply.lower()


def test_retry_reuses_saved_fields_and_never_re_asks() -> None:
    engine = _engine_with_failing_llm(TimeoutError("still down"))

    async def run():
        memory: dict = {}
        await engine.handle_turn("s", "Mobile Theft Complaint banao", "english", memory)
        await engine.handle_turn("s", _NINE_FIELDS, "english", memory)
        # Exactly the state a client-side timeout leaves behind: fields
        # checkpointed, no draft persisted, still collecting.
        memory["draft_stage"] = "collecting"
        memory["draft_id"] = None
        return await engine.handle_turn("s", "continue draft", "english", memory), memory

    result, memory = asyncio.run(run())
    assert "I just need" not in result.reply_text
    assert result.info.stage == "preview"
    assert len(memory["draft_fields"]) >= 9


def test_retry_does_not_create_a_duplicate_draft() -> None:
    from unittest.mock import AsyncMock

    engine = _engine_with_failing_llm(TimeoutError("down"))
    inserted: list = []
    engine.draft_engine.drafts.insert = AsyncMock(side_effect=lambda doc: inserted.append(doc))

    async def run():
        memory: dict = {}
        await engine.handle_turn("s", "Mobile Theft Complaint banao", "english", memory)
        await engine.handle_turn("s", _NINE_FIELDS, "english", memory)
        first = len(inserted)
        # A retry after a failure that persisted nothing must produce exactly
        # one more draft, not one per attempt.
        memory["draft_stage"] = "collecting"
        memory["draft_id"] = None
        await engine.handle_turn("s", "continue draft", "english", memory)
        return first, len(inserted)

    first, total = asyncio.run(run())
    assert first == 1
    assert total == 2, f"retry created {total - first} extra drafts"


def test_expansion_passes_are_skipped_when_the_budget_is_nearly_gone() -> None:
    import inspect

    from app.drafting.engine import LegalDraftEngine

    source = inspect.getsource(LegalDraftEngine._render_sections_within_deadline)
    assert "draft_generation_budget_seconds" in source
    assert "break" in source


# --- Live server (opt-in) ---------------------------------------------------

live = pytest.mark.skipif(
    not LIVE_BASE_URL, reason="set LEGAL_AI_LIVE_BASE_URL to run against a running backend"
)


@live
def test_live_chat_returns_a_structured_response_within_the_client_timeout() -> None:
    started = time.perf_counter()
    with httpx.Client(timeout=settings.client_request_timeout_seconds) as client:
        response = client.post(
            f"{LIVE_BASE_URL}/chat",
            json={"question": "FIR kya hoti hai?", "session_id": "pytest-live"},
        )
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    payload = response.json()
    for field in ("answer", "session_id", "request_id", "retryable", "failure_category"):
        assert field in payload, f"missing {field}"
    assert elapsed < settings.client_request_timeout_seconds


@live
def test_live_drafting_turn_completes_before_the_client_timeout() -> None:
    session = "pytest-live-draft"
    with httpx.Client(timeout=settings.client_request_timeout_seconds) as client:
        client.post(f"{LIVE_BASE_URL}/chat", json={"question": "Mobile Theft Complaint banao", "session_id": session})
        started = time.perf_counter()
        response = client.post(
            f"{LIVE_BASE_URL}/chat", json={"question": _NINE_FIELDS, "session_id": session}
        )
        elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed < settings.client_request_timeout_seconds, f"took {elapsed:.0f}s"
    payload = response.json()
    assert payload["answer"]
    # Whether or not the LLM was reachable, the user gets a draft, not a
    # transport error, and their fields are never discarded.
    assert payload.get("saved_progress") is True or payload.get("draft") is not None
