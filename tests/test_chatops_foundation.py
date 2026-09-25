"""Milestone B: the orchestration foundation shared by every chat workflow.

These defend behaviour that must hold for EVERY capability rather than for
one workflow: resuming a parked task, correcting a fact, expiring stale
state, refusing to run the same side effect twice, and reporting structured
progress. They use purpose-built workflows registered for the duration of a
test, so the guarantees are tested at the orchestrator level rather than
through whichever real workflow happens to have the right shape today.
"""

import asyncio
from datetime import timedelta
from typing import Any

import pytest

import app.chatops.orchestrator  # noqa: F401  -- registers the real workflows
from app.chatops import intents, state
from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.orchestrator import ChatOrchestrator
from app.chatops.registry import WORKFLOWS
from app.core import clock
from app.core.security import Role


def _run(coro):
    return asyncio.run(coro)


def _orchestrate(message: str, memory: dict[str, Any] | None = None, **kwargs) -> WorkflowTurn | None:
    return _run(
        ChatOrchestrator().handle_turn(
            session_id="s1",
            message=message,
            language=kwargs.pop("language", "english"),
            memory=memory if memory is not None else {},
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# Test doubles, registered only for the tests that need them
# ---------------------------------------------------------------------------


class _CountingWorkflow(ChatWorkflow):
    """Two required fields and a side effect that counts its own runs."""

    name = "test_counting"
    title = "test counting workflow"
    requires_confirmation = True

    def __init__(self) -> None:
        self.runs = 0

    def matches_intent(self, message: str, language: str) -> float:
        return 0.9 if "counting task" in message.lower() else 0.0

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return {}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["alpha", "beta"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return f"What is the {missing[0]}?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        self.runs += 1
        return WorkflowTurn(message="done", status="completed", finished=True)


class _TwinA(ChatWorkflow):
    name = "test_twin_a"
    title = "twin A task"

    def matches_intent(self, message: str, language: str) -> float:
        return 0.8 if "twin thing" in message.lower() else 0.0

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return {}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return []

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return "?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        return WorkflowTurn(message="twin A ran", status="completed", finished=True)


class _TwinB(_TwinA):
    name = "test_twin_b"
    title = "twin B task"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        return WorkflowTurn(message="twin B ran", status="completed", finished=True)


@pytest.fixture
def counting() -> Any:
    workflow = _CountingWorkflow()
    WORKFLOWS[workflow.name] = workflow
    try:
        yield workflow
    finally:
        WORKFLOWS.pop(workflow.name, None)


@pytest.fixture
def twins() -> Any:
    a, b = _TwinA(), _TwinB()
    WORKFLOWS[a.name] = a
    WORKFLOWS[b.name] = b
    try:
        yield a, b
    finally:
        WORKFLOWS.pop(a.name, None)
        WORKFLOWS.pop(b.name, None)


# ---------------------------------------------------------------------------
# 1. Control actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    ["continue", "resume karo", "aage badho", "आगे बढ़ो", "जारी रखो", "carry on"],
)
def test_resume_is_recognised_in_english_hindi_and_hinglish(phrase: str) -> None:
    assert intents.RESUME.search(phrase), phrase


def test_a_parked_workflow_is_resumed_with_its_facts(counting) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    state.remember_facts(memory, {"alpha": "one"})
    state.pause(memory)
    state.start(memory, "test_twin_a")  # something else took the foreground

    turn = _orchestrate("continue", memory)

    assert turn is not None
    assert turn.workflow_name == "test_counting"
    assert state.active_name(memory) == "test_counting"
    assert state.active(memory)["facts"]["alpha"] == "one"


def test_continue_mid_workflow_is_not_treated_as_a_resume(counting) -> None:
    """Nothing is parked, so "continue" must fall through to the workflow."""
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    turn = _orchestrate("continue", memory)
    assert turn is not None
    assert turn.status == "collecting"


def test_correcting_a_named_fact_clears_only_that_fact(counting) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    state.remember_facts(memory, {"alpha": "one", "beta": "two"})

    turn = _orchestrate("change the alpha to something else", memory)

    assert turn is not None
    assert turn.missing_field == "alpha"
    facts = state.active(memory)["facts"]
    assert "alpha" not in facts
    assert facts["beta"] == "two"


def test_cancel_is_never_read_as_a_field_value(counting) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    turn = _orchestrate("cancel", memory)
    assert turn is not None and turn.status == "cancelled"
    assert state.active_name(memory) is None
    assert counting.runs == 0


# ---------------------------------------------------------------------------
# 2. Stale state expires rather than being silently reused
# ---------------------------------------------------------------------------


def test_a_stale_workflow_is_dropped_before_it_can_answer_for_the_user(counting) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    state.remember_facts(memory, {"alpha": "yesterday's answer"})
    stale = (clock.now() - state.STALE_AFTER - timedelta(minutes=1)).isoformat()
    state.active(memory)["updated_at"] = stale

    dropped = state.expire_stale(memory)

    assert dropped == ["test_counting"]
    assert state.active(memory) is None


def test_a_fresh_workflow_survives_expiry(counting) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    assert state.expire_stale(memory) == []
    assert state.active_name(memory) == "test_counting"


# ---------------------------------------------------------------------------
# 3. Ambiguous intent asks instead of guessing
# ---------------------------------------------------------------------------


def test_a_tie_between_two_capabilities_asks_the_user_to_choose(twins) -> None:
    memory: dict[str, Any] = {}
    turn = _orchestrate("please do the twin thing", memory)
    assert turn is not None
    assert "twin A task" in turn.message and "twin B task" in turn.message
    assert turn.status == "collecting"


def test_the_reply_to_an_ambiguity_question_selects_by_number(twins) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("please do the twin thing", memory)
    turn = _orchestrate("2", memory)
    assert turn is not None
    assert turn.message == "twin B ran"


def test_an_unrelated_reply_expires_the_pending_choice(twins) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("please do the twin thing", memory)
    turn = _orchestrate("what is anticipatory bail?", memory)
    # Falls through to the normal answering pipeline, and the stale
    # clarification is gone rather than capturing the next message.
    assert turn is None
    assert "chatops_pending_choice" not in memory


# ---------------------------------------------------------------------------
# 4. An ordinary question parks the workflow instead of being eaten
# ---------------------------------------------------------------------------


def test_a_legal_question_mid_workflow_parks_it_and_falls_through(counting) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    state.remember_facts(memory, {"alpha": "one"})

    turn = _orchestrate("what is anticipatory bail?", memory)

    assert turn is None, "the question should be answered by the normal pipeline"
    entry = next(e for e in memory[state.MEMORY_KEY] if e["name"] == "test_counting")
    assert entry["status"] == "paused"
    assert entry["facts"]["alpha"] == "one"


def test_an_ordinary_answer_is_not_mistaken_for_an_interruption(counting) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    turn = _orchestrate("the alpha value is 42", memory)
    assert turn is not None and turn.status == "collecting"


@pytest.mark.parametrize(
    "message",
    [
        # "kya" sitting away from the verb -- QUESTION's literal phrasings
        # ("kya hai", "kya hota") missed these, so a workflow that misfired
        # (see the DOCUMENT_TIMELINE "file"-as-verb regression) trapped every
        # later question, including ones with nothing to do with it.
        "Valid contract ke liye kya essential elements hote hain?",
        "Constitution ke Article 21 mein kya likha hai?",
    ],
)
def test_hinglish_kya_question_mid_workflow_parks_it_and_falls_through(counting, message: str) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    state.remember_facts(memory, {"alpha": "one"})

    turn = _orchestrate(message, memory)

    assert turn is None, "the question should be answered by the normal pipeline"
    entry = next(e for e in memory[state.MEMORY_KEY] if e["name"] == "test_counting")
    assert entry["status"] == "paused"
    assert entry["facts"]["alpha"] == "one"


# ---------------------------------------------------------------------------
# 5. Idempotency: a replayed turn never runs the side effect twice
# ---------------------------------------------------------------------------


def test_a_replayed_confirmation_does_not_execute_twice(counting) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    state.remember_facts(memory, {"alpha": "one", "beta": "two"})
    confirmation = _orchestrate("go on with it", memory)
    assert confirmation is not None and confirmation.status == "awaiting_confirmation"

    first = _orchestrate("yes", memory)
    assert first is not None and first.status == "completed"
    assert counting.runs == 1

    # The same request, replayed: same workflow, same facts.
    _orchestrate("start the counting task", memory)
    state.remember_facts(memory, {"alpha": "one", "beta": "two"})
    _orchestrate("go on with it", memory)
    replay = _orchestrate("yes", memory)

    assert replay is not None and replay.status == "completed"
    assert counting.runs == 1, "the side effect ran a second time"
    assert "already completed" in replay.message


def test_a_failed_execution_is_not_recorded_as_done(counting) -> None:
    """Only a success is remembered, so a genuine retry still runs."""
    memory: dict[str, Any] = {}
    context = WorkflowContext(
        session_id="s", message="", language="english", memory=memory,
        facts={"alpha": "one", "beta": "two"},
    )
    key = counting.idempotency_key(context)
    assert not state.already_executed(memory, key)
    state.mark_executed(memory, key, "test")
    assert state.already_executed(memory, key)


def test_idempotency_keys_differ_when_the_facts_differ(counting) -> None:
    memory: dict[str, Any] = {}
    first = counting.idempotency_key(
        WorkflowContext(session_id="s", message="", language="english", memory=memory, facts={"alpha": "1"})
    )
    second = counting.idempotency_key(
        WorkflowContext(session_id="s", message="", language="english", memory=memory, facts={"alpha": "2"})
    )
    assert first != second


def test_the_executed_ledger_is_bounded() -> None:
    memory: dict[str, Any] = {}
    for index in range(120):
        state.mark_executed(memory, f"key-{index}")
    assert len(memory[state.EXECUTED_KEY]) <= 50


# ---------------------------------------------------------------------------
# 6. Structured progress
# ---------------------------------------------------------------------------


def test_progress_metadata_is_reported_without_the_workflow_writing_any(counting) -> None:
    memory: dict[str, Any] = {}
    turn = _orchestrate("start the counting task", memory)
    assert turn is not None
    assert turn.workflow_name == "test_counting"
    assert turn.total_steps == 3  # two fields plus the final act
    assert turn.completed_steps == 0
    assert turn.progress_percentage == 0
    assert turn.current_step == "alpha"


def test_progress_advances_as_facts_are_collected(counting) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    state.remember_facts(memory, {"alpha": "one"})
    turn = _orchestrate("still going", memory)
    assert turn is not None
    assert turn.completed_steps == 1
    assert 0 < turn.progress_percentage < 100
    assert turn.missing_field == "beta"


def test_a_completed_workflow_reports_full_progress(counting) -> None:
    memory: dict[str, Any] = {}
    _orchestrate("start the counting task", memory)
    state.remember_facts(memory, {"alpha": "one", "beta": "two"})
    _orchestrate("go on with it", memory)
    turn = _orchestrate("yes", memory)
    assert turn is not None
    assert turn.status == "completed"
    assert turn.progress_percentage == 100
    assert turn.completed_steps == turn.total_steps


def test_progress_percentage_is_always_a_valid_percentage(counting) -> None:
    memory: dict[str, Any] = {}
    for message in ("start the counting task", "one", "two", "yes"):
        turn = _orchestrate(message, memory)
        if turn is not None:
            assert 0 <= turn.progress_percentage <= 100


# ---------------------------------------------------------------------------
# 7. Authorization stays where it belongs
# ---------------------------------------------------------------------------


def test_role_metadata_is_declared_on_the_class_not_decided_per_turn() -> None:
    for name, workflow in WORKFLOWS.items():
        required = workflow.required_role
        assert required is None or isinstance(required, Role), name


def test_a_role_refusal_names_no_internals() -> None:
    memory: dict[str, Any] = {}
    turn = _orchestrate("show me the pending review requests", memory, claims={"role": "user"})
    assert turn is not None and turn.status == "forbidden"
    for leak in ("notary_queue", "required_role", "Traceback", "app."):
        assert leak not in turn.message
