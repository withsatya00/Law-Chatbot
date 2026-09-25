"""Part 30 "Memory-First Routing Engine" -- multi-turn reference-resolution
regression suite.

`test_chat_service_routing.py` / `test_conversation_intent_classifier.py`
prove each conv-intent bucket routes correctly in isolation, but every one
of those tests hand-builds the `memory` dict to look like "the state at
this point in the conversation" -- none of them let a real, evolving
session drive that state turn by turn. That's exactly the gap for
reference resolution: a follow-up like "Why?" or "30 words." only makes
sense in light of what `memory["messages"]` actually accumulated on the
turns before it.

This suite instead runs real multi-turn conversations through
`ChatService.answer()` against one shared, evolving in-memory session
(`FakeMemoryStore`), and asserts on the same `chat_routing_decision`
structured log line production observability relies on -- so a broken
conversation here means a broken log line in production, not a
disconnected pair of representations.
"""

import asyncio
import copy
import sys
from dataclasses import dataclass
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import structlog

from app.core.constants import LEGAL_DISCLAIMER, is_no_verified_context
from app.llm.base import ChatMessage, LLMResponse
from app.schemas.chat import ChatRequest
from app.schemas.common import RetrievedChunk
from app.services.chat_service import ChatService

# ---------------------------------------------------------------------------
# Fakes: a real, evolving in-memory session store (not a per-call mock) plus
# deterministic stand-ins for the two genuinely I/O-bound collaborators
# (vector retrieval, the LLM). Everything else (entity extraction, subject-
# matter intent detection, lawyer recommendation, the conversation-intent
# classifier itself) is the real, deterministic implementation -- there's
# nothing to fake there, and using the real thing is what makes this a
# routing-behavior test rather than a mock-behavior test.
# ---------------------------------------------------------------------------


class FakeMemoryStore:
    """Minimal stand-in for `ConversationMemoryStore` with no Redis/Mongo --
    a real, accumulating dict per session, so conversation state actually
    persists across `answer()` calls the way it does in production.

    Every public method returns a deep COPY of the canonical per-session
    dict, never the canonical object itself -- this matters: the real
    `ConversationMemoryStore.load()`/`.update()` deserialize a fresh dict
    from Redis/Mongo on every call, so a caller holding an earlier `memory`
    reference from `answer()`'s own `memory = await self.memory.append(...)`
    never sees a later `self.memory.update(...)` call (made further down in
    the SAME turn) retroactively appear on it. A naive fake that returns the
    same object every time breaks that isolation -- e.g. `answer()`'s
    RAG-tail records `last_failed_question` via `.update()` well after
    `memory` was captured, and `_append_retry_reminder` deliberately checks
    that (now-stale, pre-failure) `memory` snapshot so a failure is never
    self-referentially mentioned in its own turn's answer. Aliasing the two
    silently defeats that check.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}

    def _canonical(self, session_id: str) -> dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "summary": "", "messages": [], "language_preference": None,
                "current_intent": None, "legal_category": None, "uploaded_documents": [],
            }
        return self.sessions[session_id]

    async def load(self, session_id: str) -> dict:
        return copy.deepcopy(self._canonical(session_id))

    async def append(self, session_id: str, role: str, content: str) -> dict:
        canonical = self._canonical(session_id)
        canonical.setdefault("messages", []).append({"role": role, "content": content})
        return copy.deepcopy(canonical)

    async def update(self, session_id: str, **values: object) -> dict:
        canonical = self._canonical(session_id)
        canonical.update(values)
        return copy.deepcopy(canonical)

    async def summarize_if_needed(self, session_id: str, llm) -> dict:
        # No scenario in this suite runs long enough to hit the real
        # trigger (`SUMMARIZE_TRIGGER_MESSAGE_COUNT = 12`); kept a no-op so
        # no scenario needs an extra LLM script branch for it.
        return copy.deepcopy(self._canonical(session_id))

    async def check_access(self, session_id: str, authenticated_user_id: str | None) -> dict:
        # Part 46 "Authenticated User Ownership": mirrors the real
        # `ConversationMemoryStore.check_access` -- none of this suite's
        # scenarios authenticate, so every session here stays unclaimed
        # (`owner_user_id` never set) and this is effectively a pass-through
        # `load()`, matching real behavior for an unclaimed session.
        canonical = self._canonical(session_id)
        owner_user_id = canonical.get("owner_user_id")
        if owner_user_id and owner_user_id != authenticated_user_id:
            from app.core.exceptions import ForbiddenError

            raise ForbiddenError("This session belongs to a different account.")
        return copy.deepcopy(canonical)


class _NoopRepository:
    async def insert(self, *args: object, **kwargs: object) -> str:
        return "noop-id"


# Maps a keyword to a plausible (act, section) pair. `_fake_retrieve` treats
# presence of the keyword anywhere in the (possibly follow-up-resolved)
# query as "the knowledge base has something relevant" -- crude, but the
# point of this suite is ROUTING behavior, not retrieval quality, which is
# already covered elsewhere (`test_response_cache.py`, the reranker's own
# tests).
_TOPIC_CHUNKS: dict[str, tuple[str, str]] = {
    "stolen": ("Indian Penal Code", "379"),
    "deposit": ("Rent Control Act", "8"),
    "cheque": ("Negotiable Instruments Act", "138"),
    "cyber": ("Information Technology Act", "66C"),
    "divorce": ("Hindu Marriage Act", "13"),
    "rti": ("Right to Information Act", "6"),
    "salary": ("Payment of Wages Act", "5"),
    "consumer": ("Consumer Protection Act", "35"),
    "property": ("Transfer of Property Act", "54"),
    "bail": ("Bharatiya Nagarik Suraksha Sanhita", "480"),
    "gst": ("CGST Act", "16"),
    "fir": ("Bharatiya Nagarik Suraksha Sanhita", "173"),
}


async def _fake_retrieve(query: str, top_k: int = 10, filters=None, intent=None, context_hint=None, matter_context=None):
    lowered = query.lower()
    for keyword, (act, section) in _TOPIC_CHUNKS.items():
        if keyword in lowered:
            chunk = RetrievedChunk(
                chunk_id=f"{keyword}-1",
                text=f"Verified provision text about {keyword} under {act}, Section {section}.",
                score=0.9,
                metadata={"source_document": act, "act_name": act, "section_number": section},
            )
            return query, [chunk]
    return query, []


async def _fake_rerank(rewritten, retrieved, top_k: int = 6, legal_category=None):
    return retrieved


def _field(prompt: str, label: str) -> str:
    for line in prompt.splitlines():
        if line.startswith(label):
            return line[len(label):].strip()
    return ""


def _second_to_last_user_line(prompt: str) -> str:
    """Best-effort standalone-question synthesis for the scripted follow-up
    resolver below: the conversation-memory prompt's "Recent messages"
    block lists every turn including the just-asked follow-up itself as the
    LAST "user:" line, so the line before it is whatever the follow-up is
    actually referring back to.
    """
    lines = [line[len("user: "):].strip() for line in prompt.splitlines() if line.startswith("user: ")]
    if len(lines) >= 2:
        return lines[-2]
    return lines[-1] if lines else "the earlier topic"


async def _scripted_llm_reply(messages: list[ChatMessage], temperature: float = 0.1, **kwargs) -> LLMResponse:
    """One deterministic responder standing in for every `self.llm.chat(...)`
    call site in `chat_service.py`, dispatched on a unique marker string
    from each prompt template (see `app/llm/prompts/*.md`) -- content is a
    labeled placeholder, not a real legal answer, since this suite verifies
    ROUTING, not answer quality.
    """
    prompt = messages[-1].content
    if "Resolve references, pronouns" in prompt:
        content = f"Standalone question about: {_second_to_last_user_line(prompt)}"
    elif "Transformation requested:" in prompt:
        content = f"[MODIFIED / {_field(prompt, 'Transformation requested:')}] transformed version of the prior reply."
    elif "already confirmed as an explicit translation request" in prompt:
        content = f"[TRANSLATED to {_field(prompt, 'Target language:')}] version of the prior reply."
    elif "THIS CONVERSATION" in prompt:
        content = "[MEMORY ANSWER] recalled from the conversation so far, no retrieval performed."
    elif "Update the running summary" in prompt:
        content = "[SUMMARY] of the conversation so far."
    elif "friendly conversational side" in prompt:
        content = "Happy to help! What legal question can I help you with?"
    else:
        content = "[GROUNDED RAG ANSWER] based on the retrieved, verified legal provision."
    return LLMResponse(content=content, model="test", provider="test")


class FailureInjector:
    """Part 31 "State Manager & Recovery Engine" test hook: lets a test force
    the NEXT N `service.llm.chat(...)` calls to fail exactly the way a real
    provider failure surfaces (`LLMResponse.error` set, non-empty `content`
    -- see `app/llm/base.py`/`app/llm/gemini.py`), then automatically falls
    back to `_scripted_llm_reply` again. Not part of `ChatService`'s real
    API -- stashed on the built instance purely for tests to reach into.
    """

    def __init__(self) -> None:
        self._fail_remaining = 0

    def fail_next(self, count: int = 1) -> None:
        self._fail_remaining = count

    async def __call__(self, messages: list[ChatMessage], temperature: float = 0.1, **kwargs) -> LLMResponse:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            return LLMResponse(
                content="The LLM provider is temporarily unreachable. Please try again.",
                model="test", provider="test", error="simulated_provider_timeout",
            )
        return await _scripted_llm_reply(messages, temperature=temperature, **kwargs)


async def _extractor_llm_chat(messages: list[ChatMessage], temperature: float = 0.0, **kwargs) -> LLMResponse:
    # `DraftFieldExtractor` always attempts an LLM extraction pass on top of
    # its regex pass -- empty content is a legitimate "found nothing extra"
    # reply, letting the regex/label-prefix extraction (still real) do the
    # work for any scenario that provides fields in a labeled/structured way.
    return LLMResponse(content="", model="test", provider="test")


async def _draft_engine_llm_chat(messages: list[ChatMessage], temperature: float = 0.1, **kwargs) -> LLMResponse:
    return LLMResponse(content="[TRANSLATED DRAFT TEXT]", model="test", provider="test")


def _stub_draft_record(
    service: ChatService,
    draft_id: str = "test-draft-id",
    template_id: str = "rti_application",
    template_name: str = "RTI Application",
) -> None:
    """Backs a hand-crafted `memory["draft_id"]` (used throughout this suite
    to jump straight to preview stage without running real field collection)
    with a fake persisted record, so preview-stage turns that now read the
    actual draft from storage (`LegalDraftEngine.get_current`) have
    something to find instead of hitting a real, unconnected Mongo client.
    """
    record = {
        "_id": draft_id,
        "draft_type": template_id,
        "template_name": template_name,
        "language": "english",
        "sections": {"Body": "This is a placeholder draft body for routing tests."},
    }

    async def _fake_find_by_id(requested_id: str):
        return record if requested_id == draft_id else None

    service.draft_conversation.draft_engine.drafts.find_by_id = _fake_find_by_id


def build_service() -> ChatService:
    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])
    service.memory = FakeMemoryStore()
    service.history = _NoopRepository()
    service.query_log = _NoopRepository()

    async def _cache_miss(*args: object, **kwargs: object):
        return None, "miss"

    async def _cache_store(*args: object, **kwargs: object) -> None:
        return None

    service.response_cache.lookup = _cache_miss
    service.response_cache.store = _cache_store
    service.retriever.retrieve = _fake_retrieve
    service.reranker.rerank = _fake_rerank
    failure_injector = FailureInjector()
    service.llm.chat = failure_injector
    service.failure_injector = failure_injector  # test-only, see `FailureInjector`
    # The drafting subsystem creates its own independent LLM client
    # instances (`LLMFactory.create()` inside `DraftFieldExtractor` and
    # `LegalDraftEngine`, not shared with `ChatService.llm`) -- these must
    # be scripted separately or a drafting-stage turn would attempt a real
    # network call to whatever `settings.llm_provider` defaults to.
    service.draft_conversation.extractor.llm.chat = _extractor_llm_chat
    service.draft_conversation.draft_engine.llm.chat = _draft_engine_llm_chat
    return service


# ---------------------------------------------------------------------------
# Turn expectations + trace printing
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    message: str
    expect_memory_hit: bool
    expect_rag_used: bool
    expect_response_modification: bool
    expect_draft_mode: bool
    # `None` when the exact route depends on the fake retriever's crude
    # keyword match (still always correctly `rag_used=True` either way) --
    # otherwise the exact deterministic route slug.
    expect_route: str | frozenset[str] | None = None
    expect_conversation_intent: str | None = None


def _console_safe(text: str) -> str:
    """Draft-template replies can legitimately contain Hindi text (field
    labels like "आवेदक का नाम"). `pytest -s` writes straight to a real
    console, whose encoding on a default Windows terminal is `cp1252`
    (unlike pytest's normal captured-output-on-failure path, which buffers
    through a UTF-8-safe stream) -- printing those bytes directly there
    raises `UnicodeEncodeError` and hides the actual trace/assertion this
    was meant to surface. Round-tripping through the real stdout encoding
    with `backslashreplace` guarantees the print itself can never crash,
    on any platform.
    """
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def _print_trace(scenario_name: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 100}\nCONVERSATION TRACE: {scenario_name}\n{'=' * 100}")
    for index, row in enumerate(rows, start=1):
        print(_console_safe(f"[Turn {index}] USER: {row['message']}"))
        print(f"    Detected Intent........ {row['conversation_intent']}")
        print(f"    Selected Route.......... {row['route']}")
        print(f"    Memory Hit.............. {'Yes' if row['memory_hit'] else 'No'}")
        print(f"    RAG Used................ {'Yes' if row['rag_used'] else 'No'}")
        print(f"    Response Modification... {'Yes' if row['response_modification'] else 'No'}")
        print(f"    Draft Mode.............. {'Yes' if row['draft_mode'] else 'No'}")
        print(_console_safe(f"    Final Response.......... {row['answer'][:180]!r}"))
    print(f"{'-' * 100}\n")


async def _run_turn(service: ChatService, session_id: str, message: str) -> dict:
    with structlog.testing.capture_logs() as logs:
        response = await service.answer(ChatRequest(question=message, session_id=session_id))
    routing = next((entry for entry in logs if entry.get("event") == "chat_routing_decision"), None)
    assert routing is not None, f"no chat_routing_decision log emitted for turn: {message!r}"
    return {
        "message": message,
        "conversation_intent": response.conversation_intent,
        "route": routing["route"],
        "memory_hit": routing["memory_hit"],
        "rag_used": routing["rag_used"],
        "response_modification": routing["response_modification"],
        "draft_mode": routing["draft_mode"],
        "answer": response.answer,
    }


def run_conversation(scenario_name: str, turns: list[Turn]) -> None:
    """Runs `turns` sequentially against one fresh session, printing a full
    trace and asserting each turn's routing decision -- a failure's
    assertion message always names the exact turn, and the trace (printed
    unconditionally, so it's visible in pytest's captured-output-on-failure
    output) shows every turn up to and including it.
    """
    service = build_service()
    session_id = str(uuid4())
    rows: list[dict] = []
    try:
        for turn in turns:
            row = asyncio.run(_run_turn(service, session_id, turn.message))
            rows.append(row)
            if isinstance(turn.expect_route, frozenset):
                assert row["route"] in turn.expect_route, (
                    f"{scenario_name} turn {row['message']!r}: route={row['route']!r} not in {turn.expect_route!r}"
                )
            elif turn.expect_route is not None:
                assert row["route"] == turn.expect_route, (
                    f"{scenario_name} turn {row['message']!r}: route={row['route']!r} != {turn.expect_route!r}"
                )
            assert row["memory_hit"] == turn.expect_memory_hit, (
                f"{scenario_name} turn {row['message']!r}: memory_hit={row['memory_hit']} != {turn.expect_memory_hit}"
            )
            assert row["rag_used"] == turn.expect_rag_used, (
                f"{scenario_name} turn {row['message']!r}: rag_used={row['rag_used']} != {turn.expect_rag_used}"
            )
            assert row["response_modification"] == turn.expect_response_modification, (
                f"{scenario_name} turn {row['message']!r}: response_modification="
                f"{row['response_modification']} != {turn.expect_response_modification}"
            )
            assert row["draft_mode"] == turn.expect_draft_mode, (
                f"{scenario_name} turn {row['message']!r}: draft_mode={row['draft_mode']} != {turn.expect_draft_mode}"
            )
            if turn.expect_conversation_intent is not None:
                assert row["conversation_intent"] == turn.expect_conversation_intent, (
                    f"{scenario_name} turn {row['message']!r}: conversation_intent="
                    f"{row['conversation_intent']!r} != {turn.expect_conversation_intent!r}"
                )
    finally:
        _print_trace(scenario_name, rows)


_RAG_ROUTES = frozenset({"rag", "no_verified_context"})

# ---------------------------------------------------------------------------
# 10 topics x 4 follow-up "recipes" = 40 generated multi-turn scenarios,
# plus 10 hand-written scenarios below (drafting flows, the exact 3
# scenarios from the spec, and a few standalone edge cases) = 50 total.
# ---------------------------------------------------------------------------

_TOPICS = [
    ("bike_theft", "My bike was stolen last night."),
    ("deposit", "My landlord is not returning my security deposit."),
    ("cheque_bounce", "The cheque I received has bounced."),
    ("cyber_fraud", "I lost money in an online cyber fraud."),
    ("divorce", "I want to file for divorce from my husband."),
    ("rti", "I want to file an RTI application."),
    ("unpaid_salary", "My employer has not paid my salary for two months."),
    ("consumer", "I bought a defective product and want to file a consumer complaint."),
    ("property", "My brother is not giving me my share of our father's property."),
    ("gst", "My GST refund has been delayed for months."),
]


def _recipe_why_chain(opener: str) -> list[Turn]:
    return [
        Turn(opener, expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        Turn("Why?", expect_memory_hit=True, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES, expect_conversation_intent="Follow-up Question"),
        Turn("What did I ask first?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_memory", expect_conversation_intent="Conversation Memory"),
        Turn("What documents do I need?", expect_memory_hit=True, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES, expect_conversation_intent="Follow-up Question"),
        Turn("Can you explain simply?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("Translate into Hindi.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="translation", expect_conversation_intent="Translation"),
        Turn("30 words.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
    ]


def _recipe_formatting_chain(opener: str) -> list[Turn]:
    return [
        Turn(opener, expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        Turn("Bullet points.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("Table format.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("Step by step.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("Summarize our conversation.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_summary", expect_conversation_intent="Summarization"),
        Turn("What were we discussing?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_memory", expect_conversation_intent="Conversation Memory"),
    ]


def _recipe_what_about_chain(opener: str) -> list[Turn]:
    return [
        Turn(opener, expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        Turn("What about the time limit?", expect_memory_hit=True, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES, expect_conversation_intent="Follow-up Question"),
        Turn("Give me more details.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("Short answer.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("Translate into Hindi.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="translation", expect_conversation_intent="Translation"),
        Turn("Continue.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_memory", expect_conversation_intent="Conversation Memory"),
    ]


def _recipe_why_not_chain(opener: str) -> list[Turn]:
    return [
        Turn(opener, expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        Turn("Why not?", expect_memory_hit=True, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES, expect_conversation_intent="Follow-up Question"),
        Turn("Which papers are required?", expect_memory_hit=True, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES, expect_conversation_intent="Follow-up Question"),
        Turn("Rewrite professionally.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("One paragraph.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("Translate into Marathi.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="translation", expect_conversation_intent="Translation"),
    ]


_RECIPES = {
    "why_chain": _recipe_why_chain,
    "formatting_chain": _recipe_formatting_chain,
    "what_about_chain": _recipe_what_about_chain,
    "why_not_chain": _recipe_why_not_chain,
}

_GENERATED_SCENARIOS = [
    (f"{topic_key}__{recipe_name}", recipe_fn(opener))
    for topic_key, opener in _TOPICS
    for recipe_name, recipe_fn in _RECIPES.items()
]


@pytest.mark.parametrize("scenario_name,turns", _GENERATED_SCENARIOS, ids=[name for name, _ in _GENERATED_SCENARIOS])
def test_generated_multi_turn_conversation(scenario_name: str, turns: list[Turn]) -> None:
    run_conversation(scenario_name, turns)


def test_generated_scenario_count_is_at_least_forty() -> None:
    assert len(_GENERATED_SCENARIOS) == len(_TOPICS) * len(_RECIPES) == 40


# ---------------------------------------------------------------------------
# 10 hand-written scenarios: the 3 literal examples from the spec (drafting
# state has real, verified edge cases -- see comments below), plus 7 more
# covering things the generated recipes above don't: lawyer recommendation
# mid-conversation, cancelling a draft, a pure greeting not resetting
# context, chained translation-then-modification, and a longer 8-turn
# conversation exercising every route in one session.
# ---------------------------------------------------------------------------


def test_spec_scenario_1_bike_theft_draft_translate_shorten() -> None:
    run_conversation("spec_scenario_1", [
        Turn("My bike was stolen.", expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        Turn("Why?", expect_memory_hit=True, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES, expect_conversation_intent="Follow-up Question"),
        Turn("What was stolen?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_memory", expect_conversation_intent="Conversation Memory"),
        # "the complaint" doesn't name a specific template (no template's
        # trigger phrase matches on its own), so this correctly lands in the
        # ambiguous "which document type?" selecting stage rather than
        # silently guessing -- that's a genuine clarifying question, not a
        # memory-resolution failure.
        Turn("Can you draft the complaint?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=True, expect_route="draft_workflow", expect_conversation_intent="Draft Generation"),
        # Draft is in "selecting" stage (not "preview"), so Translation is
        # still a genuine interruption per `_is_draft_interruption` -- it
        # translates the template-selection prompt itself, draft state is
        # untouched and stays active underneath it.
        Turn("Translate it.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=True, expect_route="translation", expect_conversation_intent="Translation"),
        # By design (see `_DRAFT_INTERRUPT_INTENTS`'s comment in
        # chat_service.py), Response Modification is deliberately NOT a
        # draft interruption -- mid-draft, "30 words." is handed to the
        # draft engine itself, which (correctly) doesn't recognize it as a
        # template choice and re-shows the options rather than silently
        # misinterpreting it as an edit to a draft that was never selected.
        Turn("30 words.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=True, expect_route="draft_workflow", expect_conversation_intent="Draft Generation"),
    ])


def test_spec_scenario_2_fir_explanation_chain() -> None:
    run_conversation("spec_scenario_2", [
        Turn("Explain FIR.", expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES, expect_conversation_intent="Legal Explanation"),
        Turn("Explain simply.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("30 words.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("Translate into Hindi.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="translation", expect_conversation_intent="Translation"),
        Turn("Bullet points.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
    ])


def test_spec_scenario_3_deposit_notice_draft_interrupted_by_translation() -> None:
    run_conversation("spec_scenario_3", [
        Turn("My landlord isn't returning my deposit.", expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        Turn("Why should I send a legal notice?", expect_memory_hit=True, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES, expect_conversation_intent="Follow-up Question"),
        # "Draft it." names no template either (bare pronoun) -- same
        # genuine "which document?" clarification as spec scenario 1.
        Turn("Draft it.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=True, expect_route="draft_workflow", expect_conversation_intent="Draft Generation"),
        # "Change the city to Noida." has no drafting verb, so the draft
        # engine's own selecting-stage handler treats it the same as any
        # other non-matching input: re-shows the template list. Draft state
        # (draft_mode=True) is preserved throughout, not lost.
        Turn("Change the city to Noida.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=True, expect_route="draft_workflow", expect_conversation_intent="Draft Generation"),
        Turn("Translate to Hindi.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=True, expect_route="translation", expect_conversation_intent="Translation"),
    ])


def test_draft_start_with_named_template_then_cancel() -> None:
    run_conversation("draft_named_template_then_cancel", [
        Turn("I want to file an RTI application.", expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        # Names the template explicitly via its trigger phrase ("rti") plus
        # a drafting verb ("draft") -- goes straight to the "collecting"
        # stage for the RTI template instead of the ambiguous selector.
        Turn("Please draft an RTI application for me.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=True, expect_route="draft_workflow", expect_conversation_intent="Draft Generation"),
        # An unrelated, confidently-classified question interrupts (per
        # `_DRAFT_INTERRUPT_INTENTS`) without losing the in-progress draft.
        Turn("What is the difference between RTI and PIL?", expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=True, expect_route=_RAG_ROUTES, expect_conversation_intent="Law Comparison"),
        Turn("cancel draft", expect_memory_hit=False, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="general_conversation"),
    ])


def test_lawyer_recommendation_mid_conversation_preserves_context() -> None:
    run_conversation("lawyer_recommendation_mid_conversation", [
        Turn("My cheque bounced.", expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        Turn("Can you recommend a lawyer for this?", expect_memory_hit=False, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="lawyer_recommendation", expect_conversation_intent="Lawyer Recommendation"),
        # Memory survives the lawyer-recommendation detour: the classic
        # fact-recall pattern still resolves correctly afterward.
        Turn("What did I ask?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_memory", expect_conversation_intent="Conversation Memory"),
    ])


def test_greeting_mid_conversation_does_not_reset_context() -> None:
    run_conversation("greeting_mid_conversation", [
        Turn("My GST refund has been delayed for months.", expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        Turn("Thanks, that helps.", expect_memory_hit=False, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="general_conversation", expect_conversation_intent="General Conversation"),
        # A greeting in between must not have wiped conversation memory --
        # the very next turn still resolves the original topic correctly.
        Turn("What did I ask first?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_memory", expect_conversation_intent="Conversation Memory"),
    ])


def test_chained_translation_then_modification_transforms_translated_version() -> None:
    run_conversation("chained_translation_then_modification", [
        Turn("My employer has not paid my salary for two months.", expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        Turn("Translate into Hindi.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="translation", expect_conversation_intent="Translation"),
        # "30 words." must chain onto the Hindi version just produced, not
        # restart from the original English answer -- verified directly on
        # the prompt sent to the LLM below, not just the route.
        Turn("30 words.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
    ])


def test_translation_without_target_then_bare_language_reply_completes_it() -> None:
    run_conversation("translation_clarification_then_bare_reply", [
        Turn("I want to file for divorce from my husband.", expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        # No target language named -- asks for clarification rather than
        # guessing or dropping the request.
        Turn("Translate this.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="translation", expect_conversation_intent="Translation"),
        # A bare one-word reply to that clarification resolves it via the
        # `pending_clarification` memory flag, without repeating "translate".
        Turn("Hindi", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="translation", expect_conversation_intent="Translation"),
    ])


def test_consecutive_bare_fact_recall_questions_stay_grounded_in_memory() -> None:
    run_conversation("consecutive_bare_fact_recall", [
        Turn("My brother is not giving me my share of our father's property.", expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        Turn("What was withheld?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_memory", expect_conversation_intent="Conversation Memory"),
        Turn("What did I ask first?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_memory", expect_conversation_intent="Conversation Memory"),
        Turn("What were we discussing?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_memory", expect_conversation_intent="Conversation Memory"),
    ])


def test_long_conversation_exercises_every_route_in_one_session() -> None:
    """One extended, realistic 9-turn conversation touching every route this
    suite covers (RAG, follow-up resolution, conversation memory, response
    modification, translation, lawyer recommendation, summarization) in a
    single session -- the scenario closest to real usage, and the one whose
    printed trace is most representative of what a real support ticket
    ("why did turn 7 answer wrong?") would need to debug.
    """
    run_conversation("long_realistic_session", [
        Turn("I lost money in an online cyber fraud.", expect_memory_hit=False, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES),
        Turn("Why did this happen?", expect_memory_hit=True, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES, expect_conversation_intent="Follow-up Question"),
        Turn("What documents do I need?", expect_memory_hit=True, expect_rag_used=True, expect_response_modification=False, expect_draft_mode=False, expect_route=_RAG_ROUTES, expect_conversation_intent="Follow-up Question"),
        Turn("Can you explain simply?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("Translate into Hindi.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="translation", expect_conversation_intent="Translation"),
        Turn("30 words.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=True, expect_draft_mode=False, expect_route="response_modification", expect_conversation_intent="Response Modification"),
        Turn("Can you recommend a lawyer for this?", expect_memory_hit=False, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="lawyer_recommendation", expect_conversation_intent="Lawyer Recommendation"),
        Turn("Summarize our conversation.", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_summary", expect_conversation_intent="Summarization"),
        Turn("What did I ask first?", expect_memory_hit=True, expect_rag_used=False, expect_response_modification=False, expect_draft_mode=False, expect_route="conversation_memory", expect_conversation_intent="Conversation Memory"),
    ])


_HAND_WRITTEN_SCENARIO_COUNT = 10  # kept in sync manually with the test_ functions above


def test_total_scenario_count_is_at_least_fifty() -> None:
    assert len(_GENERATED_SCENARIOS) + _HAND_WRITTEN_SCENARIO_COUNT >= 50


# ---------------------------------------------------------------------------
# Focused content-level check: chaining must act on the CURRENT version of
# the reply, not restart from the original -- verified against the actual
# prompt text sent to the LLM, not just the route (mirrors the equivalent
# single-turn check in test_chat_service_routing.py, but here the Hindi
# version is produced by a real prior turn in the same session instead of
# being hand-seeded into `memory["messages"]`).
# ---------------------------------------------------------------------------


def test_modification_after_translation_chains_onto_translated_text() -> None:
    service = build_service()
    session_id = str(uuid4())
    captured_prompts: list[str] = []
    original_scripted_reply = service.llm.chat

    async def _capturing_llm_chat(messages, temperature: float = 0.1, **kwargs):
        captured_prompts.append(messages[-1].content)
        return await original_scripted_reply(messages, temperature=temperature, **kwargs)

    service.llm.chat = _capturing_llm_chat

    asyncio.run(service.answer(ChatRequest(question="My employer has not paid my salary for two months.", session_id=session_id)))
    asyncio.run(service.answer(ChatRequest(question="Translate into Hindi.", session_id=session_id)))
    asyncio.run(service.answer(ChatRequest(question="30 words.", session_id=session_id)))

    modification_prompt = next(p for p in captured_prompts if "Transformation requested:" in p)
    assert "[TRANSLATED to hindi]" in modification_prompt
    assert "[GROUNDED RAG ANSWER]" not in modification_prompt


# ---------------------------------------------------------------------------
# Part 31 "State Manager & Recovery Engine": failure tracking + retry.
# ---------------------------------------------------------------------------


def test_spec_part31_bail_api_failure_then_retry_recovers() -> None:
    """The literal spec regression test: "What is Bail?" -> (simulated) API
    failure -> "Retry" -> a real answer, with conversation state (not just
    the answer) surviving the failure in between.
    """
    service = build_service()
    session_id = str(uuid4())

    service.failure_injector.fail_next(1)
    with structlog.testing.capture_logs() as logs:
        failed_response = asyncio.run(service.answer(ChatRequest(question="What is Bail?", session_id=session_id)))
    failed_routing = next(entry for entry in logs if entry.get("event") == "chat_routing_decision")
    assert failed_routing["request_failed"] is True
    assert "GROUNDED RAG ANSWER" not in failed_response.answer  # real answer never produced this turn

    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["last_failed_question"] == "What is Bail?"
    assert stored_memory["last_failed_reason"]

    with structlog.testing.capture_logs() as logs:
        retried_response = asyncio.run(service.answer(ChatRequest(question="Retry", session_id=session_id)))
    retried_routing = next(entry for entry in logs if entry.get("event") == "chat_routing_decision")
    assert retried_routing["request_failed"] is False
    assert "[GROUNDED RAG ANSWER]" in retried_response.answer

    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["last_failed_question"] is None
    assert stored_memory["last_failed_reason"] is None
    assert stored_memory["last_successful_response"] == retried_response.answer


def test_failure_does_not_break_subsequent_translation_clarification_flow() -> None:
    """Rule 1 + Rule 2 interaction: a pending failure recorded on one turn
    must (a) survive later, unrelated turns happening in between (Rule 1 --
    nothing but an explicit retry clears it) and (b) not interfere with a
    LATER, unrelated translation-clarification flow starting up normally
    (Rule 2 still resolves a bare-language reply immediately).
    """
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="My cheque bounced.", session_id=session_id)))

    service.failure_injector.fail_next(1)
    # Names the topic explicitly (rather than "this") so `_fake_retrieve`
    # finds a chunk regardless of follow-up resolution -- under the strict-
    # RAG guardrail, an empty-context question short-circuits before the LLM
    # is ever called, which would never exercise `failure_injector` at all.
    asyncio.run(service.answer(ChatRequest(question="What is the punishment for cheque bounce?", session_id=session_id)))
    assert service.memory.sessions[session_id].get("last_failed_question") is not None

    # A later, unrelated translation request still asks the clarifying
    # question and sets up correctly, unaffected by the still-pending failure.
    asyncio.run(service.answer(ChatRequest(question="Translate this.", session_id=session_id)))
    assert service.memory.sessions[session_id].get("pending_clarification") == "translation_target"
    # The pending failure itself is untouched -- Rule 1 (not silently
    # cleared just because unrelated turns happened in between).
    assert service.memory.sessions[session_id].get("last_failed_question") is not None

    response = asyncio.run(service.answer(ChatRequest(question="Hindi", session_id=session_id)))
    assert response.conversation_intent == "Translation"
    assert "[TRANSLATED to hindi]" in response.answer


def test_rule4_reminder_appears_once_then_stops_on_later_unretried_turns() -> None:
    """Part 49 regression: reproduces a real production transcript where a
    single stale failure kept surfacing "by the way, your earlier
    question... ran into a temporary issue" after 10+ later, completely
    unrelated answers (translations, follow-ups, a fresh topic) -- read as
    nagging repetition rather than a helpful one-time nudge. The reminder
    must appear exactly once (the turn right after the failure), then stay
    silent on every later turn even though nothing was retried -- while
    "retry" itself must still work any time later (see the next test).
    """
    service = build_service()
    session_id = str(uuid4())

    service.failure_injector.fail_next(1)
    first_response = asyncio.run(service.answer(ChatRequest(question="What is Bail?", session_id=session_id)))
    # The failing turn's own answer doesn't ALSO get `_append_retry_reminder`'s
    # separate "by the way, your earlier question..." nag layered onto it --
    # `_fallback_answer`'s own text may itself mention retrying, which is
    # fine (that's this turn's own immediate, honest framing of what just
    # happened, not the Rule 4 reminder mechanism being tested here).
    assert "by the way, your earlier question" not in first_response.answer.lower()

    # The next turn (a plain greeting, unrelated) still gets reminded once.
    second_response = asyncio.run(service.answer(ChatRequest(question="Thanks, that helps", session_id=session_id)))
    assert "what is bail" in second_response.answer.lower()
    assert "by the way, your earlier question" in second_response.answer.lower()

    # But several further unrelated turns must NOT keep repeating it --
    # `last_failed_question` itself is still tracked (retry keeps working),
    # only the nagging repeated text stops.
    assert service.memory.sessions[session_id]["last_failed_question"] == "What is Bail?"
    for question in ["Meri bike chori ho gayi hai, kya karu?", "uske baad kya karna hai?", "Explain FIR."]:
        later_response = asyncio.run(service.answer(ChatRequest(question=question, session_id=session_id)))
        assert "by the way, your earlier question" not in later_response.answer.lower()
        assert "ran into a temporary issue" not in later_response.answer.lower()
    assert service.memory.sessions[session_id]["last_failed_question"] == "What is Bail?"


def test_translation_after_failure_never_picks_up_the_retry_reminder_text() -> None:
    """Part 49 item 10 regression: after a failed turn (which gets a Rule 4
    retry reminder appended to what's SHOWN to the user), a later "Translate"
    -> "Hindi" must translate the real fallback answer's own content, never
    the "By the way, your earlier question... ran into a temporary issue..."
    reminder suffix -- the reminder is appended to `response.answer` only
    AFTER `_finalize_intent_response`/the RAG-tail already wrote the clean,
    un-suffixed answer to `memory["messages"]`, so it can never itself
    become the "last reply" a later translation acts on.
    """
    service = build_service()
    session_id = str(uuid4())

    service.failure_injector.fail_next(1)
    asyncio.run(service.answer(ChatRequest(question="What is Bail?", session_id=session_id)))
    reminded_response = asyncio.run(service.answer(ChatRequest(question="Thanks, that helps", session_id=session_id)))
    assert "by the way, your earlier question" in reminded_response.answer.lower()

    asyncio.run(service.answer(ChatRequest(question="Translate.", session_id=session_id)))
    translated_response = asyncio.run(service.answer(ChatRequest(question="Hindi", session_id=session_id)))
    assert translated_response.conversation_intent == "Translation"
    assert "ran into a temporary issue" not in translated_response.answer.lower()
    assert "by the way, your earlier question" not in translated_response.answer.lower()


def test_rule4_reminder_stops_once_retry_succeeds() -> None:
    service = build_service()
    session_id = str(uuid4())

    service.failure_injector.fail_next(1)
    asyncio.run(service.answer(ChatRequest(question="What is Bail?", session_id=session_id)))
    retried_response = asyncio.run(service.answer(ChatRequest(question="Retry", session_id=session_id)))
    assert "say \"retry\"" not in retried_response.answer.lower()

    next_response = asyncio.run(service.answer(ChatRequest(question="Thanks!", session_id=session_id)))
    assert "retry" not in next_response.answer.lower()


def test_resending_the_same_failed_question_is_treated_as_a_retry() -> None:
    """BUG-013a (QA 2026-09-11/12, `docs/qa/QA_TEST_MATRIX_20260911.md`): a
    real user who does not know the magic word "retry" does the natural
    thing instead and simply resends their own exact question -- that must
    resume the failed attempt the same way saying "retry" does (see
    `test_rule4_reminder_stops_once_retry_succeeds` above), not get routed
    as an unrelated brand-new question with the original failure silently
    dropped.
    """
    service = build_service()
    session_id = str(uuid4())

    service.failure_injector.fail_next(1)
    asyncio.run(service.answer(ChatRequest(question="What is Bail?", session_id=session_id)))
    assert service.memory.sessions[session_id]["last_failed_question"] == "What is Bail?"

    resend_response = asyncio.run(service.answer(ChatRequest(question="What is Bail?", session_id=session_id)))
    assert "temporarily unreachable" not in resend_response.answer.lower()
    assert service.memory.sessions[session_id]["last_failed_question"] is None


def test_conversation_state_snapshot_reports_all_part31_fields() -> None:
    """Direct unit test of the State Manager itself (`app/memory/state.py`),
    independent of `ChatService` -- verifies `snapshot()` surfaces every
    field the spec names, correctly derived from a hand-built `memory` dict.
    """
    from app.memory.state import snapshot

    memory = {
        "messages": [
            {"role": "user", "content": "Explain FIR."},
            {"role": "assistant", "content": "An FIR is..."},
            {"role": "user", "content": "Translate."},
        ],
        "legal_category": "Criminal Law",
        "pending_clarification": "translation_target",
        "draft_mode": True,
        "last_failed_question": "What is Bail?",
        "last_failed_reason": "simulated_provider_timeout",
        "last_successful_response": "An FIR is...",
    }
    state = snapshot(memory)
    assert state.last_user_question == "Translate."
    assert state.last_successful_response == "An FIR is..."
    assert state.last_failed_question == "What is Bail?"
    assert state.last_failed_reason == "simulated_provider_timeout"
    assert state.current_topic == "Criminal Law"
    assert state.pending_translation is True
    assert state.pending_modification is False
    assert state.pending_draft is True
    assert state.pending_clarification == "translation_target"
    assert state.pending_upload is False


def test_retry_mid_draft_interrupts_without_losing_draft_state() -> None:
    """A failure recorded before a draft started must still be retryable
    from inside an active draft, without the draft itself being lost --
    "Retry Failed Request" is a `_DRAFT_INTERRUPT_INTENTS` member for
    exactly this reason.
    """
    service = build_service()
    session_id = str(uuid4())

    service.failure_injector.fail_next(1)
    asyncio.run(service.answer(ChatRequest(question="What is Bail?", session_id=session_id)))
    assert service.memory.sessions[session_id]["last_failed_question"] == "What is Bail?"

    asyncio.run(service.answer(ChatRequest(question="I want to file an RTI application.", session_id=session_id)))
    draft_response = asyncio.run(
        service.answer(ChatRequest(question="Please draft an RTI application for me.", session_id=session_id))
    )
    assert draft_response.conversation_intent == "Draft Generation"
    assert service.memory.sessions[session_id]["draft_mode"] is True

    retried_response = asyncio.run(service.answer(ChatRequest(question="Retry", session_id=session_id)))
    assert "[GROUNDED RAG ANSWER]" in retried_response.answer
    # The draft that was in progress when "Retry" interrupted it is untouched.
    assert service.memory.sessions[session_id]["draft_mode"] is True
    assert service.memory.sessions[session_id]["draft_template_id"] == "rti_application"


# ---------------------------------------------------------------------------
# Part 32 "Draft State Manager": template switching, confirmation words, and
# "continue draft" must never lose or mix draft state.
# ---------------------------------------------------------------------------


def test_spec_part32_switch_template_then_confirm_generate() -> None:
    """The literal spec regression test: General Notice -> Recovery Notice
    -> Yes -> Continue Draft -> Generate -> the template switches once and
    stays switched; none of the confirmation words restart or revert it.
    """
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="Generate Legal Notice", session_id=session_id)))
    assert service.memory.sessions[session_id]["draft_template_id"] == "legal_notice"

    asyncio.run(service.answer(ChatRequest(question="Recovery Notice", session_id=session_id)))
    assert service.memory.sessions[session_id]["draft_template_id"] == "recovery_notice"

    for message in ["Yes", "Continue Draft", "Generate"]:
        response = asyncio.run(service.answer(ChatRequest(question=message, session_id=session_id)))
        stored_memory = service.memory.sessions[session_id]
        assert stored_memory["draft_template_id"] == "recovery_notice", f"template lost/reverted on {message!r}"
        assert stored_memory["draft_mode"] is True
        assert response.conversation_intent == "Draft Generation"
        assert "I don't have a template for that" not in response.answer


def test_field_answer_blob_with_first_person_narrative_does_not_interrupt_draft() -> None:
    """Reproduces a real production failure: a multi-field answer pasted as
    one message, whose "Facts of the Case" content happens to read exactly
    like a first-person legal-advice question ("mera makan malik... mujhe
    pareshan kar raha hai"), must be collected as field data -- not
    hijacked into a fresh RAG answer that leaves the draft's fields empty.
    """
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="Generate Legal Notice", session_id=session_id)))
    blob = (
        "Noida Sector 63 9129432906 Satya Prakash Sharma Jail for 4 year "
        "Mera makan malik mera deposit nahi de raha hai aur mujhe bahut pareshan kar raha hai "
        "Glorious PG Chijarshi Mangal Pandey"
    )
    response = asyncio.run(service.answer(ChatRequest(question=blob, session_id=session_id)))

    assert response.conversation_intent == "Draft Generation"
    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["draft_mode"] is True
    # The regex extractor (real, unmocked) picks up at least the mobile
    # number from the blob -- the point being it was extracted AT ALL,
    # proving the message reached the draft engine instead of being
    # answered as a fresh legal-advice question with the draft untouched.
    assert stored_memory["draft_fields"].get("applicant_mobile") == "9129432906"


def test_bare_generate_mid_collecting_does_not_restart_or_get_stored_as_field() -> None:
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="I want to file an RTI application.", session_id=session_id)))
    asyncio.run(service.answer(ChatRequest(question="Please draft an RTI application for me.", session_id=session_id)))
    assert service.memory.sessions[session_id]["draft_template_id"] == "rti_application"

    response = asyncio.run(service.answer(ChatRequest(question="Generate", session_id=session_id)))
    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["draft_template_id"] == "rti_application"
    assert stored_memory["draft_mode"] is True
    assert "I don't have a template for that" not in response.answer
    # "Generate" itself must never end up stored as a field's literal value.
    assert "generate" not in [v.lower() for v in stored_memory["draft_fields"].values()]


def test_continue_draft_resumes_same_template_mid_collecting() -> None:
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="I want to file an RTI application.", session_id=session_id)))
    asyncio.run(service.answer(ChatRequest(question="Please draft an RTI application for me.", session_id=session_id)))

    response = asyncio.run(service.answer(ChatRequest(question="Continue draft", session_id=session_id)))
    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["draft_template_id"] == "rti_application"
    assert "I don't have a template for that" not in response.answer
    assert "RTI" in response.answer


def test_named_template_switch_requires_no_drafting_verb() -> None:
    """Selecting a template purely by its name (no "draft"/"generate"/...
    verb) must still switch -- the verb requirement that stops ordinary
    questions from accidentally starting a draft doesn't apply once the
    user is already inside the drafting flow, whether that's the legacy
    "selecting" stage (a partial name match narrowed things down) or
    problem-first discovery's "describe_problem" stage (a fully generic
    "Draft it." with no candidates at all -- see
    `DraftConversationEngine._start_discovery`).
    """
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="I bought a defective product and want to file a consumer complaint.", session_id=session_id)))
    # "Draft it." matches no specific template -> problem-first discovery's
    # "describe_problem" stage, not a flat list of all ~55 templates.
    first = asyncio.run(service.answer(ChatRequest(question="Draft it.", session_id=session_id)))
    assert first.conversation_intent == "Draft Generation"
    assert service.memory.sessions[session_id]["draft_stage"] == "describe_problem"

    second = asyncio.run(service.answer(ChatRequest(question="RTI Application", session_id=session_id)))
    assert service.memory.sessions[session_id]["draft_template_id"] == "rti_application"
    assert service.memory.sessions[session_id]["draft_stage"] == "collecting"
    assert "I didn't catch which document" not in second.answer


# ---------------------------------------------------------------------------
# Part 33 "Natural Confirmation Engine": Hindi/Hinglish/English confirmation
# phrasings must all map onto the current draft workflow -- never restart,
# never ask an unnecessary clarifying question.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "confirmation_message",
    ["haan", "ok kr do", "generate", "continue", "hn", "theek hai", "kar do", "yeah", "proceed karo", "aage badho"],
)
def test_spec_part33_natural_confirmation_after_template_switch(confirmation_message: str) -> None:
    """The literal spec regression tests (Recovery Notice -> {confirmation}
    -> PASS), parametrized over every listed phrasing: the template must
    stay `recovery_notice` and the draft must stay active, regardless of
    which natural confirmation phrasing is used.
    """
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="Generate Legal Notice", session_id=session_id)))
    asyncio.run(service.answer(ChatRequest(question="Recovery Notice", session_id=session_id)))
    assert service.memory.sessions[session_id]["draft_template_id"] == "recovery_notice"

    response = asyncio.run(service.answer(ChatRequest(question=confirmation_message, session_id=session_id)))
    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["draft_template_id"] == "recovery_notice", (
        f"{confirmation_message!r} lost/changed the template"
    )
    assert stored_memory["draft_mode"] is True
    assert response.conversation_intent == "Draft Generation"
    assert "I don't have a template for that" not in response.answer
    assert "I didn't catch which document" not in response.answer


def test_hinglish_confirmation_mid_collecting_does_not_restart() -> None:
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="I want to file an RTI application.", session_id=session_id)))
    asyncio.run(service.answer(ChatRequest(question="Please draft an RTI application for me.", session_id=session_id)))

    response = asyncio.run(service.answer(ChatRequest(question="haan kar do", session_id=session_id)))
    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["draft_template_id"] == "rti_application"
    assert stored_memory["draft_mode"] is True
    assert "I don't have a template for that" not in response.answer


def test_hinglish_confirmation_at_preview_stage_reaffirms_without_restarting() -> None:
    """Same check as the collecting-stage test above, but with the draft
    already fully generated and sitting in preview -- a bare Hinglish
    confirmation there must reaffirm the existing draft, not misfire the
    ambiguous-new-draft branch (`generate`/`create` are strong drafting
    verbs) or lose the draft entirely.
    """
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="I want to file an RTI application.", session_id=session_id)))
    asyncio.run(service.answer(ChatRequest(question="Please draft an RTI application for me.", session_id=session_id)))
    # Force the session straight to "preview" (all fields already collected)
    # rather than hand-crafting text the real regex extractor would pick up
    # for all of RTI's required fields -- this suite verifies ROUTING, not
    # the field extractor, which has its own tests.
    stored_memory = service.memory.sessions[session_id]
    stored_memory["draft_stage"] = "preview"
    stored_memory["draft_id"] = "test-draft-id"
    _stub_draft_record(service)

    response = asyncio.run(service.answer(ChatRequest(question="haan kar do", session_id=session_id)))
    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["draft_template_id"] == "rti_application"
    assert stored_memory["draft_stage"] == "preview"
    assert stored_memory["draft_mode"] is True
    assert "I don't have a template for that" not in response.answer
    assert "generated draft" in response.answer.lower()


# ---------------------------------------------------------------------------
# Part 34 (unnamed in the request, but the same class of gap as Parts 30/31):
# a real scripted conversation trace exposed three more short-fragment
# reference-resolution gaps once a prior turn exists -- a bare language name
# ("Hindi") without an explicit "Translate." first, a short comparison
# fragment ("Difference with NCR"), and "What should I do" not matching any
# of the existing follow-up/advice/explanation patterns.
# ---------------------------------------------------------------------------


def test_bare_language_name_implies_translation_without_explicit_translate_command() -> None:
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="Explain FIR.", session_id=session_id)))
    asyncio.run(service.answer(ChatRequest(question="30 words", session_id=session_id)))
    response = asyncio.run(service.answer(ChatRequest(question="Hindi", session_id=session_id)))

    assert response.conversation_intent == "Translation"
    assert "[TRANSLATED to hindi]" in response.answer


def test_short_comparison_fragment_resolves_via_memory_not_dictionary_lookup() -> None:
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="Explain FIR.", session_id=session_id)))
    response = asyncio.run(service.answer(ChatRequest(question="Difference with NCR", session_id=session_id)))

    assert response.conversation_intent == "Follow-up Question"


def test_what_should_i_do_after_a_problem_statement_resolves_as_followup() -> None:
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="My bike was stolen.", session_id=session_id)))
    response = asyncio.run(service.answer(ChatRequest(question="What should I do", session_id=session_id)))

    assert response.conversation_intent == "Follow-up Question"


def test_new_self_contained_question_with_prior_turn_is_not_swallowed_as_followup() -> None:
    """The token-count cap that fixes "What should I do" must stay narrow
    enough not to swallow a genuinely new, self-contained question just
    because it happens to be short and a prior turn exists.
    """
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="Explain FIR.", session_id=session_id)))
    response = asyncio.run(service.answer(ChatRequest(question="What was the maximum bail amount?", session_id=session_id)))

    assert response.conversation_intent != "Follow-up Question"


# ---------------------------------------------------------------------------
# Part 35 "Entity Memory": facts stated during a conversation must be
# recallable without retrieval, and never confused with a fresh statement
# that merely shares an interrogative word.
# ---------------------------------------------------------------------------


def test_spec_entity_memory_three_canonical_examples() -> None:
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="My bike was stolen.", session_id=session_id)))
    asyncio.run(service.answer(ChatRequest(question="My brother's head was injured.", session_id=session_id)))
    asyncio.run(service.answer(ChatRequest(question="My landlord won't return my deposit.", session_id=session_id)))

    stolen_response = asyncio.run(service.answer(ChatRequest(question="Who was injured?", session_id=session_id)))
    assert stolen_response.conversation_intent == "Entity Memory"
    assert "brother" in stolen_response.answer.lower()
    assert LEGAL_DISCLAIMER not in stolen_response.answer

    issue_response = asyncio.run(service.answer(ChatRequest(question="What issue was I facing?", session_id=session_id)))
    assert issue_response.conversation_intent == "Entity Memory"
    assert "landlord" in issue_response.answer.lower()
    assert LEGAL_DISCLAIMER not in issue_response.answer


def test_entity_memory_checked_before_rag_never_says_dont_know() -> None:
    service = build_service()
    session_id = str(uuid4())

    with structlog.testing.capture_logs() as logs:
        asyncio.run(service.answer(ChatRequest(question="My bike was stolen.", session_id=session_id)))
    with structlog.testing.capture_logs() as logs:
        response = asyncio.run(service.answer(ChatRequest(question="What was stolen?", session_id=session_id)))
    routing = next(entry for entry in logs if entry.get("event") == "chat_routing_decision")
    # "What was stolen?" already matches the pre-existing Conversation Memory
    # fact-recall pattern (Part 30) -- Entity Memory's job is covering what
    # that pattern list DOESN'T catch (see the Hindi/"who"/"what issue" tests
    # below), so this one is allowed to resolve via either route so long as
    # it never reaches RAG and never leaves the user at a dead end. (The
    # scripted Conversation Memory reply is a generic placeholder, not real
    # content, so this can't also assert "bike" appears -- that's covered
    # by the Entity-Memory-specific tests below instead.)
    assert routing["rag_used"] is False
    assert "i don't have access" not in response.answer.lower()
    assert "i don't know" not in response.answer.lower()


def test_entity_memory_recall_does_not_require_english_phrasing() -> None:
    """Reproduces the real conversation that motivated this feature: a
    Hindi/Hinglish fact-recall question that matches NONE of the existing
    Conversation Memory patterns must still resolve from Entity Memory
    rather than falling through to RAG and dead-ending with "I don't have
    access to your personal history."
    """
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="mera bike chori ho gya hai kya krna chahiye", session_id=session_id)))
    response = asyncio.run(
        service.answer(ChatRequest(question="accha mera kya chori hua tha yaad dila dena", session_id=session_id))
    )
    assert response.conversation_intent == "Entity Memory"
    assert "bike" in response.answer.lower()


def test_new_statement_sharing_a_question_word_is_not_treated_as_recall() -> None:
    """"mera bike chori ho gya hai kya krna chahiye" states a NEW fact and
    asks for advice in the same breath -- "kya" makes it look like a recall
    query, but it must still reach RAG for real advice, not short-circuit
    to "Your bike was stolen" and stop there.
    """
    service = build_service()
    session_id = str(uuid4())

    with structlog.testing.capture_logs() as logs:
        response = asyncio.run(
            service.answer(ChatRequest(question="mera bike chori ho gya hai kya krna chahiye", session_id=session_id))
        )
    routing = next(entry for entry in logs if entry.get("event") == "chat_routing_decision")
    assert response.conversation_intent != "Entity Memory"
    assert routing["rag_used"] is True


def test_spec_part39_teacher_assault_recall_answers_in_hindi() -> None:
    """Part 39 section 15's own literal regression case: "Teacher ne mujhe
    mara." followed by "Mujhe kisne mara tha?" must recall "teacher" -- and,
    since the whole exchange is in Hindi/Hinglish, the recall answer must
    stay in Hindi/Hinglish too (section 1: never randomly switch language).
    """
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="Teacher ne mujhe mara.", session_id=session_id)))
    response = asyncio.run(service.answer(ChatRequest(question="Mujhe kisne mara tha?", session_id=session_id)))

    assert response.conversation_intent == "Entity Memory"
    assert "teacher" in response.answer.lower()
    assert response.detected_language in ("hindi", "hinglish")
    # Localized template, not the English default leaking through.
    assert "your teacher" not in response.answer.lower()


def test_spec_part41_contextual_followup_preserves_topic_for_retrieval() -> None:
    """Part 41 CASE 3: after "Meri bike chori ho gayi." + advice, "accha fir
    kya hota hai" is a genuine (if short) standalone question about FIR, not
    a bike-theft continuation -- once correctly resolved (as the real LLM
    does: confirmed live to resolve to "What is an FIR?"), retrieval must
    target FIR, not lose the topic to "what should I do"-style follow-up
    handling or retrieve something unrelated. `_fake_retrieve` only returns
    a chunk when its query argument contains one of `_TOPIC_CHUNKS`'
    keywords, so this fails loudly if the resolved query drifts off-topic.
    """
    service = build_service()
    session_id = str(uuid4())
    asyncio.run(service.answer(ChatRequest(question="Meri bike chori ho gayi.", session_id=session_id)))

    async def _resolve_to_fir_question(messages, temperature: float = 0.0, **kwargs):
        return LLMResponse(content="What is an FIR?", model="test", provider="test")

    service.llm.chat = _resolve_to_fir_question
    retrieved_chunks: list = []
    inner_retrieve = service.retriever.retrieve

    async def _recording_retrieve(query, **kwargs):
        rewritten, chunks = await inner_retrieve(query, **kwargs)
        retrieved_chunks.extend(chunks)
        return rewritten, chunks

    service.retriever.retrieve = _recording_retrieve
    with structlog.testing.capture_logs() as logs:
        response = asyncio.run(service.answer(ChatRequest(question="accha fir kya hota hai", session_id=session_id)))
    routing = next(entry for entry in logs if entry.get("event") == "chat_routing_decision")
    assert routing["rag_used"] is True
    # What this case is actually about: the query that reached retrieval
    # was on-topic, so `_fake_retrieve` returned the FIR provision.
    assert retrieved_chunks, "retrieval query drifted off the FIR topic"
    assert retrieved_chunks[0].metadata["act_name"] == "Bharatiya Nagarik Suraksha Sanhita"
    assert retrieved_chunks[0].metadata["section_number"] == "173"
    # The stub LLM answers every call with the rewritten question, which
    # the answer-quality gate rejects as too short to be a legal answer --
    # so the SETTLED answer here is the strict-RAG refusal. Post-Phase-3
    # hardening: a refusal must not carry the citations retrieval happened
    # to find, in the response or in stored history (this assertion used to
    # read `assert response.sources`, which is the defect a reported
    # session showed as BNS and GST sources displayed under "no verified
    # document ... is available"). See `app/services/safe_decline.py`.
    assert response.no_verified_context is True
    assert response.sources == []
    assert response.retrieved_chunks == []
    assert response.currency_notice == ""


def test_entity_memory_recall_without_a_stored_fact_answers_i_dont_remember() -> None:
    """Part 42 section 21: a recall-shaped question naming a known fact
    category ("who was injured?") with nothing relevant stored yet must
    answer with an explicit "I don't remember" -- never fall through to
    RAG/general knowledge (which has no way to know and could otherwise
    hallucinate an answer about an entity that was never mentioned), and
    never say something like "I don't have access to your personal life."
    """
    service = build_service()
    session_id = str(uuid4())
    response = asyncio.run(service.answer(ChatRequest(question="Who was injured?", session_id=session_id)))
    assert response.conversation_intent == "Entity Memory"
    assert not response.sources
    assert "don't remember" in response.answer.lower()
    assert "access to your personal" not in response.answer.lower()


def test_fir_procedural_question_naming_meri_fir_is_not_hijacked_into_no_memory() -> None:
    """Live repro: "Police meri FIR register nahi kar rahi... Mujhe pehle
    kya information deni chahiye" is a procedural question asking what to
    do next, not a fact-recall question. The possessive "meri" next to the
    "fir" Document-entity keyword previously satisfied `matches_known_
    category`'s entity-only fallback, and this "kya <noun> chahiye" shape
    slipped past `_ACTION_SEEKING_PATTERN`, so the whole thing wrongly
    answered "I don't remember" on a fresh session instead of reaching
    real retrieval/advice.
    """
    service = build_service()
    session_id = str(uuid4())
    response = asyncio.run(
        service.answer(
            ChatRequest(
                question=(
                    "Police meri FIR register nahi kar rahi. Main Jaipur, Rajasthan mein hoon. "
                    "Mujhe pehle kya information deni chahiye"
                ),
                session_id=session_id,
            )
        )
    )
    assert response.conversation_intent != "Entity Memory"
    assert "don't remember" not in response.answer.lower()
    assert "yaad nahi" not in response.answer.lower()


def test_entity_memory_recall_unrelated_general_question_still_falls_through() -> None:
    """A recall-shaped interrogative that does NOT name any fact category
    this module tracks (no "stolen"/"lost"/"injured"/... keyword) must NOT
    be hijacked into a bogus "I don't remember" -- it's an ordinary
    question with nothing to do with personal fact recall, so it falls
    through to normal routing (RAG/general knowledge) untouched.
    """
    service = build_service()
    session_id = str(uuid4())
    response = asyncio.run(
        service.answer(ChatRequest(question="Who can file an FIR under BNSS?", session_id=session_id))
    )
    assert response.conversation_intent != "Entity Memory"


@pytest.mark.parametrize("message", [
    "Mera landlord security deposit return nahi kar raha. Main kya kar sakta hoon?",
    "Police meri FIR register nahi kar rahi. Mere paas kya legal options hain?",
    # Live repro: "kya" here questions the NOUN ("what information"), not a
    # verb from the narrower `kya\s+(karu|karna|krna|...)` alternation, so
    # this shape slipped past that pattern and was wrongly answered "I
    # don't remember" (matched_known_category fired on possessive "meri" +
    # the "fir" Document-entity keyword) instead of reaching real retrieval.
    "Police meri FIR register nahi kar rahi. Main Jaipur, Rajasthan mein hoon. Mujhe pehle kya information deni chahiye",
    "What information should I give the police first?",
])
def test_hinglish_action_questions_are_not_memory_recall(message: str) -> None:
    from app.memory.entity_memory import is_recall_query

    assert not is_recall_query(message)


def test_entity_fact_extraction_and_recall_unit() -> None:
    from app.memory.entity_memory import extract_fact, find_matching_fact, format_answer

    fact = extract_fact("My bike was stolen.", turn_index=0)
    assert fact is not None
    assert fact.entity == "bike"
    assert fact.entity_type == "Vehicle"
    assert fact.fact_key == "stolen"

    stored = [fact.to_dict()]
    match = find_matching_fact("What was stolen?", stored)
    assert match is not None
    assert format_answer(match) == "Your bike was stolen."


def test_entity_fact_assault_by_teacher_unit() -> None:
    """Part 39 section 2's own flagship example: "Teacher ne mujhe mara"
    (a fact shape -- "X hit/beat me" -- with no injured body part named, so
    the pre-existing "injured" category never fired for it) followed later
    by "Mujhe kisne mara tha?" must recall "teacher," not dead-end.
    """
    from app.memory.entity_memory import extract_fact, find_matching_fact, format_answer

    fact = extract_fact("Teacher ne mujhe mara.", turn_index=0)
    assert fact is not None
    assert fact.entity == "teacher"
    assert fact.entity_type == "Relationship"
    assert fact.fact_key == "assault"

    stored = [fact.to_dict()]
    match = find_matching_fact("Mujhe kisne mara tha?", stored)
    assert match is not None
    assert match.entity == "teacher"
    # English default stays in English; Hindi/Hinglish gets a localized
    # template instead of leaking an English sentence into a Hindi reply.
    assert format_answer(match) == "Your teacher hit/beat you."
    assert format_answer(match, "hindi") == "Aapke teacher ne aapko mara tha."
    assert format_answer(match, "hinglish") == "Aapke teacher ne aapko mara tha."


# ---------------------------------------------------------------------------
# Part 38 "Draft Auto-Pause Engine": a draft sitting in preview must never
# conversation-lock the user -- an unrelated new question pauses it (answers
# normally, reminds once) and a bare dismissal word exits it completely.
# ---------------------------------------------------------------------------


async def _service_with_draft_in_preview() -> tuple:
    service = build_service()
    session_id = str(uuid4())
    await service.answer(ChatRequest(question="I want to file an RTI application.", session_id=session_id))
    await service.answer(ChatRequest(question="Please draft an RTI application for me.", session_id=session_id))
    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["draft_mode"] is True
    # Forced straight to preview (skipping real field collection) -- this
    # suite verifies ROUTING around a ready draft, not the field extractor.
    stored_memory["draft_stage"] = "preview"
    stored_memory["draft_id"] = "test-draft-id"
    _stub_draft_record(service)
    return service, session_id


def test_spec_part38_new_legal_question_pauses_draft_and_answers() -> None:
    service, session_id = asyncio.run(_service_with_draft_in_preview())

    response = asyncio.run(
        service.answer(ChatRequest(question="My wife is throwing me out of the house.", session_id=session_id))
    )
    stored_memory = service.memory.sessions[session_id]

    # Answered as a real question, not swallowed into the draft flow.
    assert response.conversation_intent != "Draft Generation"
    assert "I can edit specific details" not in response.answer
    assert "what would you like to do" not in response.answer.lower()
    # The draft itself survives, untouched, ready to resume.
    assert stored_memory["draft_mode"] is True
    assert stored_memory["draft_stage"] == "preview"
    assert stored_memory["draft_template_id"] == "rti_application"
    # Reminder present, and doesn't re-litigate edit/download options.
    assert 'your previous rti application draft is still saved' in response.answer.lower()
    assert 'type "continue draft"' in response.answer.lower()
    assert "download" not in response.answer.lower()
    assert "translate" not in response.answer.lower()


def test_draft_reminder_shown_once_per_pause_not_every_turn() -> None:
    """Part 43 section 14 "Stale Draft Protection": the reminder must not be
    tacked onto every single unrelated response while a draft sits paused --
    only the first one after it's interrupted. A second, different unrelated
    question asked right after must NOT repeat it (still answered normally,
    draft still untouched and resumable).
    """
    service, session_id = asyncio.run(_service_with_draft_in_preview())

    first = asyncio.run(
        service.answer(ChatRequest(question="My wife is throwing me out of the house.", session_id=session_id))
    )
    assert "still saved" in first.answer.lower()

    second = asyncio.run(
        service.answer(ChatRequest(question="What is Section 498A?", session_id=session_id))
    )
    assert "still saved" not in second.answer.lower()
    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["draft_mode"] is True
    assert stored_memory["draft_stage"] == "preview"

    # Once the user actually re-engages with the draft, the next pause
    # should remind them again.
    resumed = asyncio.run(service.answer(ChatRequest(question="continue draft", session_id=session_id)))
    assert resumed.conversation_intent == "Draft Generation"
    third = asyncio.run(
        service.answer(ChatRequest(question="What is anticipatory bail?", session_id=session_id))
    )
    assert "still saved" in third.answer.lower()


@pytest.mark.parametrize(
    "dismissal_word",
    ["Nothing", "Nothing else", "Leave it", "Not now", "Cancel", "Close", "Dismiss", "Exit", "Stop", "Done"],
)
def test_spec_part38_dismissal_word_exits_draft_completely(dismissal_word: str) -> None:
    service, session_id = asyncio.run(_service_with_draft_in_preview())

    asyncio.run(service.answer(ChatRequest(question=dismissal_word, session_id=session_id)))
    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["draft_mode"] is False
    assert stored_memory["draft_stage"] is None


def test_spec_part38_regression_dismissal_then_greeting() -> None:
    """Draft Ready -> Nothing -> Greeting -> PASS: after exiting, a plain
    greeting must not mention the draft at all -- it's gone, not paused.
    """
    service, session_id = asyncio.run(_service_with_draft_in_preview())

    asyncio.run(service.answer(ChatRequest(question="Nothing", session_id=session_id)))
    response = asyncio.run(service.answer(ChatRequest(question="Hi", session_id=session_id)))
    assert response.conversation_intent == "General Conversation"
    assert "draft" not in response.answer.lower()


def test_spec_part38_regression_new_question_then_advice_mentions_saved_draft() -> None:
    """Draft Ready -> New legal question -> Advice -> Draft Saved -> PASS."""
    service, session_id = asyncio.run(_service_with_draft_in_preview())

    response = asyncio.run(
        service.answer(ChatRequest(question="My wife is throwing me out of the house.", session_id=session_id))
    )
    # No `_TOPIC_CHUNKS` keyword matches this question, so under the strict-RAG
    # guardrail it correctly short-circuits to the exact no-verified-context
    # string rather than reaching the scripted LLM branch -- either is a valid
    # "some advice-shaped answer was produced" outcome here.
    assert "[GROUNDED RAG ANSWER]" in response.answer or is_no_verified_context(response.answer)
    assert "still saved" in response.answer.lower()


def test_recognized_preview_edit_commands_still_work_and_do_not_auto_pause() -> None:
    """The auto-pause fix must not swallow GENUINE preview-stage commands
    (approve/regenerate/translate/replace-field) into the new fallback --
    only messages the edit interpreter truly can't parse should pause.
    """
    service, session_id = asyncio.run(_service_with_draft_in_preview())
    # The draft's Mongo record is faked the same way this suite already fakes
    # the other genuinely I/O-bound collaborators (vector retrieval, the
    # LLM), since `_service_with_draft_in_preview` uses a synthetic
    # `draft_id` with no real backing record.
    draft_engine = service.draft_conversation.draft_engine
    draft_engine.drafts.find_by_id = AsyncMock(return_value={"_id": "test-draft-id", "lifecycle_state": "preview_ready"})
    draft_engine.drafts.update_by_id = AsyncMock(return_value=True)

    response = asyncio.run(service.answer(ChatRequest(question="Looks good, approved.", session_id=session_id)))
    assert response.conversation_intent == "Draft Generation"
    stored_memory = service.memory.sessions[session_id]
    assert stored_memory["draft_mode"] is True
    # "approve" is recognized but no longer gates anything: the approve ->
    # lock confirmation pair was removed, so the draft stays in preview,
    # editable, with downloads already available.
    assert stored_memory["draft_stage"] == "preview"


# ---------------------------------------------------------------------------
# Part 39: Entity Memory must never substitute a DIFFERENT entity than the
# one the current message actually names -- "phone" in the current message
# must never be answered with a stored "bike" fact just because both are
# in the "stolen" category.
# ---------------------------------------------------------------------------


def test_entity_memory_never_substitutes_a_different_named_entity() -> None:
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="My bike was stolen.", session_id=session_id)))
    response = asyncio.run(
        service.answer(ChatRequest(question="What happened to my phone, was it also stolen?", session_id=session_id))
    )

    # A "stolen" fact IS on record, but for a different entity (bike, not
    # phone) -- must never substitute it. Answered with an explicit
    # "I don't remember" (Part 42 section 21/22) rather than falling
    # through to RAG, which has no way to know either and could invent
    # something about the phone.
    assert response.conversation_intent == "Entity Memory"
    assert "your bike was stolen" not in response.answer.lower()
    assert "don't remember" in response.answer.lower()


def test_entity_memory_still_answers_when_the_named_entity_matches() -> None:
    service = build_service()
    session_id = str(uuid4())

    asyncio.run(service.answer(ChatRequest(question="My bike was stolen.", session_id=session_id)))
    response = asyncio.run(
        service.answer(ChatRequest(question="What happened to my bike, was it stolen?", session_id=session_id))
    )

    assert response.conversation_intent == "Entity Memory"
    assert "bike" in response.answer.lower()
