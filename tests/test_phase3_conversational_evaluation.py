"""Phase 3 Milestone G: observations come from real routing, not fixture claims."""

import json
from pathlib import Path

import pytest

from app.chatops.evaluation import evaluate_conversations

BENCHMARK = Path(__file__).parent / "benchmarks" / "phase3_conversations_v1.json"


def test_conversation_benchmark_contains_expectations_not_observations() -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    cases = payload["cases"]
    assert len(cases) >= 20
    assert len({case["id"] for case in cases}) == len(cases)
    assert all("observed" not in case for case in cases)
    assert {case["language"] for case in cases} == {"english", "hindi", "hinglish"}
    assert any(case["workflow"] is None for case in cases)


def test_real_conversation_routing_clears_the_recorded_floors() -> None:
    result = evaluate_conversations(BENCHMARK)
    assert result.routing_accuracy >= 0.90, result.failures
    assert result.language_accuracy >= 0.90, result.failures
    assert result.ordinary_question_safety == 1.0, result.failures


def test_empty_or_self_scored_benchmark_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text('{"version": 1, "cases": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="no cases"):
        evaluate_conversations(empty)
