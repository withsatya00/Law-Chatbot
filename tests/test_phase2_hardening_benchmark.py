"""Post-Phase-3 hardening, Phase 2 milestone F: the conversation benchmark.

The Phase-3 benchmark scores one message at a time against the workflow
registry. Every defect Phase 2 is about is a property of a CONVERSATION -- a
field leaking in from the draft two turns ago, a language that reverted on the
next turn, a deadline in the document that appears in nothing the user typed --
so none of them is visible to it.

This runs whole scripted conversations through the real `ChatService.answer()`,
with only I/O stubbed: routing, the draft conversation engine, field
extraction, the ChatOps orchestrator, the drafting engine and the citation and
period guards all execute for real. The benchmark file
(`tests/benchmarks/phase2_hardening_conversations_v1.json`) holds inputs and
expectations only -- `_reject_self_scored` refuses one that ships its own
results -- and every observation comes from the response the service returns.

Retrieval returns nothing, which is a legitimate and deliberate condition: it
is what the observed session actually did on those questions, and it is the
condition under which the system must decline rather than answer from memory.
`provider: "unavailable"` on a conversation additionally puts the drafting
engine on the deterministic path, which is the condition every draft in the
observed transcript was produced under.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.chatops.evaluation import evaluate_hardening_conversations
from app.llm.base import LLMResponse
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService

BENCHMARK = Path(__file__).parent / "benchmarks" / "phase2_hardening_conversations_v1.json"


@dataclass
class _Observation:
    answer: str
    conversation_intent: str
    detected_language: str
    sources: list[Any]
    warnings: list[str]
    memory: dict[str, Any]


def _service(memory: dict[str, Any], *, provider_available: bool) -> ChatService:
    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])

    async def _load(session_id: str) -> dict[str, Any]:
        return memory

    async def _append(session_id: str, role: str, content: str) -> dict[str, Any]:
        memory.setdefault("messages", []).append({"role": role, "content": content})
        return memory

    async def _update(session_id: str, **values: Any) -> dict[str, Any]:
        memory.update(values)
        return memory

    service.memory.load = _load
    service.memory.append = _append
    service.memory.update = _update
    service.memory.summarize_if_needed = AsyncMock(return_value=memory)
    service.memory.append_intent_event = AsyncMock(return_value=memory)
    service.history.insert = AsyncMock(return_value="history-id")
    service.query_log.insert = AsyncMock(return_value="query-log-id")
    service.intent_events.insert = AsyncMock(return_value="intent-event-id")
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    service.retriever.retrieve = AsyncMock(return_value=("", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.draft_conversation.extractor._extract_with_llm = AsyncMock(return_value={})
    # Draft persistence is the only other I/O a completed draft touches.
    # Everything that shapes the DOCUMENT -- the template, the deterministic
    # builders, the clause scan, the fact audit -- still runs for real.
    engine = service.draft_conversation.draft_engine
    engine.drafts.insert = AsyncMock(return_value="draft-id")
    engine.drafts.find_one = AsyncMock(return_value=None)
    engine.versions.insert = AsyncMock(return_value="version-id")
    if not provider_available:
        unavailable = LLMResponse(
            content="The drafting service is currently unavailable. Please try again.",
            model="test",
            provider="test",
            error="The drafting service is currently unavailable. Please try again.",
            error_kind="provider_error",
        )
        service.draft_conversation.draft_engine.llm.chat = AsyncMock(return_value=unavailable)
    return service


def _runner(conversations: dict[str, dict[str, Any]]):
    """One `run_turn` closure over a per-conversation service cache."""
    services: dict[str, ChatService] = {}

    def run_turn(conversation_id: str, message: str, memory: dict[str, Any]) -> _Observation:
        if conversation_id not in services:
            available = conversations.get(conversation_id, {}).get("provider") != "unavailable"
            services[conversation_id] = _service(memory, provider_available=available)
        service = services[conversation_id]
        response = asyncio.run(
            service.answer(ChatRequest(question=message, session_id=f"bench-{conversation_id}"))
        )
        return _Observation(
            answer=response.answer,
            conversation_intent=response.conversation_intent,
            detected_language=response.detected_language,
            sources=list(response.sources),
            warnings=list(response.warnings or []),
            memory=memory,
        )

    return run_turn


def _payload() -> dict[str, Any]:
    import json

    return json.loads(BENCHMARK.read_text(encoding="utf-8"))


def _evaluate():
    payload = _payload()
    by_id = {conversation["id"]: conversation for conversation in payload["conversations"]}
    return evaluate_hardening_conversations(BENCHMARK, _runner(by_id))


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


def test_the_benchmark_covers_every_required_scenario() -> None:
    ids = {conversation["id"] for conversation in _payload()["conversations"]}
    required = {
        "cheque-bounce-question",
        "consumer-complaint-drafting",
        "rent-deposit-notice-after-consumer-complaint",
        "hindi-drafting",
        "provider-fallback-disclosure",
        "edit-and-resume",
        "saved-drafts",
        "notarization-preparation",
        "verification-without-a-token",
        "ordinary-question-during-parked-draft",
    }
    assert required <= ids, f"benchmark is missing: {sorted(required - ids)}"


def test_the_benchmark_records_no_observations_of_its_own() -> None:
    payload = _payload()
    for conversation in payload["conversations"]:
        assert "observed" not in conversation
        for turn in conversation["turns"]:
            assert "observed" not in turn
            assert set(turn) <= {"message", "expect"}


def test_a_self_scored_benchmark_is_rejected(tmp_path: Path) -> None:
    import json

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "version": 1,
                "conversations": [
                    {"id": "x", "turns": [{"message": "hi", "observed": {"intent": "Greeting"}}]}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="observed"):
        evaluate_hardening_conversations(bad, _runner({}))


def test_an_empty_benchmark_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text('{"version": 1, "conversations": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="no cases"):
        evaluate_hardening_conversations(empty, _runner({}))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result():
    return _evaluate()


def test_no_fact_leaks_between_drafts_in_one_session(result) -> None:
    assert result.leakage_free_rate == 1.0, result.failures


def test_no_reply_states_a_deadline_nothing_supports(result) -> None:
    assert result.fabrication_free_rate == 1.0, result.failures


def test_every_turn_is_routed_to_the_intent_the_script_expects(result) -> None:
    assert result.intent_accuracy == 1.0, result.failures


def test_every_turn_is_answered_in_the_language_the_script_expects(result) -> None:
    assert result.language_accuracy == 1.0, result.failures


def test_citations_are_present_exactly_when_there_is_something_to_cite(result) -> None:
    assert result.citation_honesty_rate == 1.0, result.failures


def test_another_accounts_draft_never_becomes_this_sessions_draft(result) -> None:
    assert result.owner_isolation_rate == 1.0, result.failures


def test_the_benchmark_actually_ran_the_conversations(result) -> None:
    """A benchmark that scored nothing would pass every rate above."""
    assert result.conversations >= 10
    assert result.turns >= 18
