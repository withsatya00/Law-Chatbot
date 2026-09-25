"""Phase 3 milestone F: honest lawyer guidance and language switching."""

import asyncio
from typing import Any

import pytest

from app.chatops import state
from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.orchestrator import ChatOrchestrator
from app.chatops.registry import WORKFLOWS
from app.recommendation.directory import DirectoryListing, NullLawyerDirectory
from app.recommendation.engine import LawyerRecommendationEngine, render_recommendation


class _Directory:
    def __init__(self, *, available: bool = True, fail: bool = False) -> None:
        self.is_available = available
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def available(self) -> bool:
        return self.is_available

    async def search(
        self, *, specialization: str, city: str = "", language: str = "", limit: int = 5
    ) -> list[DirectoryListing]:
        self.calls.append({
            "specialization": specialization, "city": city, "language": language, "limit": limit,
        })
        if self.fail:
            raise ConnectionError("directory unavailable")
        return [
            DirectoryListing(
                full_name="Directory Supplied Name",
                enrolment_number="DIR-101",
                verified_by="Directory Authority",
                city=city,
                languages=[language] if language else [],
            )
        ]


def test_null_directory_never_invents_a_lawyer_listing() -> None:
    result = asyncio.run(
        LawyerRecommendationEngine(NullLawyerDirectory()).recommend("Divorce", "Family Law")
    )
    assert result.category == "Family Lawyer"
    assert result.directory_available is False
    assert result.verified_listings == []
    assert "No verified lawyer directory" in result.directory_notice
    assert result.documents_to_carry
    assert result.questions_to_ask


def test_verified_directory_values_are_copied_not_generated() -> None:
    directory = _Directory()
    result = asyncio.run(
        LawyerRecommendationEngine(directory).recommend(
            "Divorce", "Family Law", city="Pune", language="hindi"
        )
    )
    assert directory.calls == [{
        "specialization": "Family Lawyer", "city": "Pune", "language": "hindi", "limit": 5,
    }]
    assert result.directory_available is True
    assert [item.full_name for item in result.verified_listings] == ["Directory Supplied Name"]
    assert result.verified_listings[0].enrolment_number == "DIR-101"


def test_directory_outage_degrades_to_specialization_only() -> None:
    result = asyncio.run(
        LawyerRecommendationEngine(_Directory(fail=True)).recommend("Divorce", "Family Law")
    )
    assert result.directory_available is False
    assert result.verified_listings == []
    assert "specialization suggestion" in result.directory_notice


def test_urgent_context_is_disclosed_without_inventing_a_deadline() -> None:
    result = asyncio.run(
        LawyerRecommendationEngine().recommend("I received a summons today", "Criminal Law")
    )
    assert result.urgency == "prompt"
    assert all("days" not in question.lower() for question in result.questions_to_ask)


@pytest.mark.parametrize(
    ("language", "expected"),
    [("english", "Documents to take"), ("hindi", "साथ ले जाने वाले दस्तावेज़"),
     ("hinglish", "Saath le jaane wale documents")],
)
def test_guidance_is_rendered_in_the_requested_language(language: str, expected: str) -> None:
    result = asyncio.run(
        LawyerRecommendationEngine().recommend("Divorce", "Family Law", language=language)
    )
    rendered = render_recommendation(result, language)
    assert expected in rendered
    assert "Directory Supplied Name" not in rendered


class _LanguageWorkflow(ChatWorkflow):
    name = "test_language_switch"
    title = "language switch test"

    def matches_intent(self, message: str, language: str) -> float:
        return 0.9 if "start language task" in message else 0.0

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return {}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["fact"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return "What is the fact?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        return WorkflowTurn(message="done", status="completed", finished=True)


@pytest.fixture
def language_workflow() -> _LanguageWorkflow:
    workflow = _LanguageWorkflow()
    WORKFLOWS[workflow.name] = workflow
    try:
        yield workflow
    finally:
        WORKFLOWS.pop(workflow.name, None)


def _turn(message: str, memory: dict[str, Any], language: str = "english") -> WorkflowTurn | None:
    return asyncio.run(
        ChatOrchestrator().handle_turn(
            session_id="s1", message=message, language=language, memory=memory
        )
    )


def test_language_switch_during_workflow_preserves_state(language_workflow: _LanguageWorkflow) -> None:
    memory: dict[str, Any] = {}
    started = _turn("start language task", memory)
    assert started is not None and started.status == "collecting"
    state.remember_facts(memory, {"already_known": "value"})

    switched = _turn("Hindi mein batao", memory)

    assert switched is not None
    assert switched.status == "collecting"
    assert switched.workflow_name == language_workflow.name
    assert "हिंदी" in switched.message
    assert memory["language_preference"] == "hindi"
    assert state.active(memory)["facts"]["already_known"] == "value"  # type: ignore[index]


def test_switching_back_to_english_is_not_collected_as_a_fact(
    language_workflow: _LanguageWorkflow,
) -> None:
    memory: dict[str, Any] = {}
    _turn("start language task", memory)
    switched = _turn("answer in English", memory, language="hindi")
    assert switched is not None and "continue in English" in switched.message
    assert memory["language_preference"] == "english"
    assert "fact" not in state.active(memory)["facts"]  # type: ignore[index]
