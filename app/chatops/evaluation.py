"""Deterministic conversation-routing benchmark used by Phase 3 gates.

Unlike the old Phase-3 JSON, this dataset contains inputs and expectations
only. Observations are produced from the real language detector and workflow
registry at run time; a benchmark cannot award itself a perfect score by
shipping a hand-written ``observed`` object beside every expectation.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.chatops.orchestrator import orchestrator as _registered_orchestrator
from app.chatops.registry import best_match
from app.language.detector import LanguageDetector
from app.rag.statutory_periods import find_periods


@dataclass(frozen=True)
class ConversationEvaluation:
    cases: int
    routing_accuracy: float
    language_accuracy: float
    ordinary_question_safety: float
    failures: tuple[str, ...]


def evaluate_conversations(path: Path) -> ConversationEvaluation:
    """Run the curated cases through the real deterministic routing path."""
    # Keep the import above observably used: importing the orchestrator is
    # what registers every workflow before `best_match` reads the registry.
    _ = _registered_orchestrator
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("cases"), list):
        raise ValueError("Conversation benchmark must be a version-1 case list.")

    cases: list[dict[str, Any]] = payload["cases"]
    if not cases:
        raise ValueError("Conversation benchmark contains no cases.")
    detector = LanguageDetector()
    route_hits = 0
    language_hits = 0
    ordinary_total = 0
    ordinary_hits = 0
    failures: list[str] = []

    for case in cases:
        case_id = str(case.get("id", "unnamed"))
        message = str(case.get("message", ""))
        expected_language = str(case.get("language", ""))
        expected_workflow = case.get("workflow")
        if not message or expected_language not in {"english", "hindi", "hinglish"}:
            raise ValueError(f"Invalid conversation benchmark case: {case_id}")

        detected_language = detector.detect(message)
        match = best_match(message, detected_language)
        observed_workflow = match[0].name if match else None
        language_ok = detected_language == expected_language
        route_ok = observed_workflow == expected_workflow
        language_hits += int(language_ok)
        route_hits += int(route_ok)
        if expected_workflow is None:
            ordinary_total += 1
            ordinary_hits += int(observed_workflow is None)
        if not route_ok:
            failures.append(
                f"{case_id}: workflow expected={expected_workflow!r} observed={observed_workflow!r}"
            )
        if not language_ok:
            failures.append(
                f"{case_id}: language expected={expected_language!r} observed={detected_language!r}"
            )

    total = len(cases)
    return ConversationEvaluation(
        cases=total,
        routing_accuracy=round(route_hits / total, 4),
        language_accuracy=round(language_hits / total, 4),
        ordinary_question_safety=round(ordinary_hits / max(ordinary_total, 1), 4),
        failures=tuple(failures),
    )


# ---------------------------------------------------------------------------
# Post-Phase-3 hardening (Phase 2, milestone F): multi-turn conversation
# evaluation.
#
# `evaluate_conversations` above scores ONE message at a time against the
# workflow registry. That is the right shape for routing, and it is all Phase 3
# needed. It cannot see any of the defects Phase 2 is about, because every one
# of them is a property of a CONVERSATION: a field leaking in from the draft
# two turns ago, a language that reverted on the next turn, a deadline in the
# document that appears in nothing the user typed.
#
# So this evaluator drives whole scripted conversations through a caller-
# supplied runner -- in practice the real `ChatService.answer()` with only I/O
# stubbed -- and derives every observation from the response it gets back. The
# benchmark file holds inputs and expectations ONLY; `_reject_self_scored`
# refuses a file that tries to ship its own results.
# ---------------------------------------------------------------------------


class TurnObservation(Protocol):
    """What the runner reports back for one turn.

    Deliberately the subset of `ChatResponse` these checks read, plus the
    conversation state the chat layer keeps -- so a runner can be built over
    the real service without this module importing it.
    """

    answer: str
    conversation_intent: str
    detected_language: str
    sources: list[Any]
    warnings: list[str]
    memory: dict[str, Any]


@dataclass(frozen=True)
class HardeningEvaluation:
    conversations: int
    turns: int
    intent_accuracy: float
    language_accuracy: float
    leakage_free_rate: float
    fabrication_free_rate: float
    citation_honesty_rate: float
    owner_isolation_rate: float
    failures: tuple[str, ...]


def _reject_self_scored(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("version") != 1 or not isinstance(payload.get("conversations"), list):
        raise ValueError("Hardening benchmark must be a version-1 conversation list.")
    conversations: list[dict[str, Any]] = payload["conversations"]
    if not conversations:
        raise ValueError("Hardening benchmark contains no cases.")
    for conversation in conversations:
        if "observed" in conversation:
            raise ValueError("Hardening benchmark conversations must not carry an `observed` result.")
        for turn in conversation.get("turns", []):
            if "observed" in turn:
                raise ValueError(
                    "Hardening benchmark turns must not carry an `observed` result: "
                    "observations come from running the conversation."
                )
    return conversations


def evaluate_hardening_conversations(
    path: Path,
    run_turn: "Callable[[str, str, dict[str, Any]], TurnObservation]",
) -> HardeningEvaluation:
    """Run every scripted conversation and score it.

    `run_turn(conversation_id, message, memory)` sends one message in the named
    conversation and returns the observation. The caller owns the session and
    the memory dict; this function only reads what comes back.
    """
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    conversations = _reject_self_scored(payload)

    failures: list[str] = []
    turns = 0
    scored: dict[str, list[int]] = {
        key: [0, 0]
        for key in ("intent", "language", "leakage", "fabrication", "citation", "owner")
    }

    def record(key: str, ok: bool, message: str) -> None:
        scored[key][1] += 1
        if ok:
            scored[key][0] += 1
        else:
            failures.append(message)

    for conversation in conversations:
        conversation_id = str(conversation.get("id", "unnamed"))
        memory: dict[str, Any] = dict(conversation.get("initial_memory") or {})
        # Everything the user has typed so far in this conversation. A period
        # in a reply counts as fabricated only if it is in none of these and in
        # none of the retrieved sources -- which is what makes this a real
        # check rather than a keyword list.
        user_text: list[str] = []
        for index, turn in enumerate(conversation.get("turns", []), start=1):
            message = str(turn.get("message", ""))
            if not message:
                raise ValueError(f"{conversation_id} turn {index}: empty message")
            observation = run_turn(conversation_id, message, memory)
            user_text.append(message)
            turns += 1
            label = f"{conversation_id}#{index}"
            expect: dict[str, Any] = turn.get("expect", {})

            if "intent" in expect:
                observed = observation.conversation_intent
                record(
                    "intent",
                    observed == expect["intent"],
                    f"{label}: intent expected={expect['intent']!r} observed={observed!r}",
                )
            if "language" in expect:
                observed_language = observation.detected_language
                record(
                    "language",
                    observed_language == expect["language"],
                    f"{label}: language expected={expect['language']!r} observed={observed_language!r}",
                )
            for needle in expect.get("answer_contains", []):
                record(
                    "intent",
                    str(needle) in observation.answer,
                    f"{label}: answer does not contain {needle!r}",
                )
            for needle in expect.get("answer_excludes", []):
                record(
                    "intent",
                    str(needle) not in observation.answer,
                    f"{label}: answer contains {needle!r}, which it must not",
                )

            if "draft_template" in expect:
                observed_template = observation.memory.get("draft_template_id")
                record(
                    "leakage",
                    observed_template == expect["draft_template"],
                    f"{label}: draft template expected={expect['draft_template']!r} "
                    f"observed={observed_template!r}",
                )
            absent = expect.get("draft_fields_absent")
            if absent is not None:
                fields = observation.memory.get("draft_fields") or {}
                present = [key for key in absent if fields.get(key)]
                record("leakage", not present, f"{label}: fields leaked into this draft: {present}")

            if expect.get("no_unsourced_period"):
                supplied = "\n".join(
                    [*user_text, *(str(getattr(source, "text", "")) for source in observation.sources)]
                )
                known = {claim.normalized for claim in find_periods(supplied, require_obligation=False)}
                flagged = " ".join(observation.warnings)
                invented = [
                    claim.describe()
                    for claim in find_periods(observation.answer, require_obligation=True)
                    if claim.normalized not in known and claim.describe() not in flagged
                ]
                record(
                    "fabrication",
                    not invented,
                    f"{label}: reply states unsourced period(s) {invented}",
                )

            if "cites_sources" in expect:
                has_sources = bool(observation.sources)
                record(
                    "citation",
                    has_sources == bool(expect["cites_sources"]),
                    f"{label}: cites_sources expected={expect['cites_sources']} observed={has_sources}",
                )

            if "owner_can_see_foreign_draft" in expect:
                # Owner isolation, observed rather than asserted: the foreign
                # draft named on the conversation must never become this
                # session's active draft, and its owner id must never be
                # adopted.
                foreign = conversation.get("foreign_template")
                allowed = bool(expect["owner_can_see_foreign_draft"])
                observed_template = observation.memory.get("draft_template_id")
                record(
                    "owner",
                    (observed_template == foreign) is allowed,
                    f"{label}: owner isolation breached -- active draft is {observed_template!r}",
                )

    def rate(key: str) -> float:
        hits, total = scored[key]
        return round(hits / total, 4) if total else 1.0

    return HardeningEvaluation(
        conversations=len(conversations),
        turns=turns,
        intent_accuracy=rate("intent"),
        language_accuracy=rate("language"),
        leakage_free_rate=rate("leakage"),
        fabrication_free_rate=rate("fabrication"),
        citation_honesty_rate=rate("citation"),
        owner_isolation_rate=rate("owner"),
        failures=tuple(failures),
    )
