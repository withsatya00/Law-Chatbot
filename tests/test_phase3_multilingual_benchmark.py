"""Real per-language end-to-end results for tests/benchmarks/phase3_multilingual.json.

The fixture used to ship a hand-written `observed` block next to every
`expected` one -- self-scored, with no runner ever reading it (confirmed by a
repo-wide grep). That is exactly the anti-pattern
`app.chatops.evaluation._reject_self_scored` exists to catch for the sibling
hardening fixture. The hand-written values were renamed to
`_legacy_observed_reference` (kept for human comparison only) and this file is
the actual runner: it drives every case through the real `ChatService`,
`LanguageDetector`, and `IntentDetector`, and reports what the system does
today.

Two intent axes are asserted, both real (`app/intent/classifier.py`'s own
docstring draws this distinction): `IntentDetector` classifies legal SUBJECT
MATTER ("Cyber Crime"), while `ChatResponse.conversation_intent` classifies
the *kind of ask* ("Draft Generation", "Legal Procedure"). A case's expected
label is checked against whichever axis it actually names.

`safety_route`, `export_quality`, `draft_facts`, and `max_latency_ms` from the
original fixture have no corresponding field on `ChatResponse` today (grepped
the schema; none exist). Asserting them would mean inventing a mapping that
does not reflect real behaviour, so they are intentionally not scored here.
`latency_ms` is measured and printed for information only, never asserted, so
this test cannot become flaky on machine speed.

Retrieval/reranking are left real (not mocked) so `grounded` reflects the
corpus's actual review_status=approved state -- see
docs/SOURCE_VERIFICATION_CHECKLIST.md for which Acts are approved today.
Requires live Mongo + Redis + the embedding model, same as
tests/test_legal_benchmark.py, hence the same opt-in gate.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.database.mongodb import mongodb
from app.intent.detector import IntentDetector
from app.language.detector import LanguageDetector
from app.llm.base import LLMResponse
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService

BENCHMARK = Path(__file__).parent / "benchmarks" / "phase3_multilingual.json"

requires_benchmark_env = pytest.mark.skipif(
    os.environ.get("LEGAL_AI_RUN_BENCHMARK") != "1",
    reason="Needs live Mongo/Redis and the embedding model -- set LEGAL_AI_RUN_BENCHMARK=1.",
)


def _load_cases() -> list[dict[str, Any]]:
    cases = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    for case in cases:
        if "observed" in case:
            raise ValueError(
                f"{case['id']}: fixture must not ship a self-scored `observed` result -- "
                "use `_legacy_observed_reference` for a preserved hand-written baseline."
            )
    return cases


def test_fixture_has_no_self_scored_observations() -> None:
    """Fails fast, without live infra, if anyone reintroduces `observed`."""
    _load_cases()


def _service() -> ChatService:
    """Same stub shape as tests/test_phase2_hardening_benchmark.py's `_service`,
    minus the retriever/reranker mocks: those are left real here on purpose."""
    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])
    memory: dict[str, Any] = {}

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

    generic = LLMResponse(
        content="This is a general legal information response for benchmark purposes.",
        model="test", provider="test",
    )
    service.llm.chat = AsyncMock(return_value=generic)
    engine = service.draft_conversation.draft_engine
    engine.drafts.insert = AsyncMock(return_value="draft-id")
    engine.drafts.find_one = AsyncMock(return_value=None)
    engine.versions.insert = AsyncMock(return_value="version-id")
    engine.llm.chat = AsyncMock(return_value=generic)
    return service


@requires_benchmark_env
def test_multilingual_end_to_end() -> None:
    """Drive every case through the real service and print a per-language report."""
    cases = _load_cases()
    intent_detector = IntentDetector()
    language_detector = LanguageDetector()

    per_language: dict[str, list[int]] = {}
    rows: list[str] = []
    language_hits = intent_hits = 0

    async def _run_case(case: dict[str, Any]) -> None:
        nonlocal language_hits, intent_hits
        expected = case["expected"]
        service = _service()
        start = time.monotonic()
        response = await service.answer(
            ChatRequest(question=case["input"], session_id=f"multi-{case['id']}")
        )
        latency_ms = round((time.monotonic() - start) * 1000, 1)

        expected_language = expected["language"]
        language_ok = response.detected_language == expected_language
        language_hits += int(language_ok)
        bucket = per_language.setdefault(expected_language, [0, 0])
        bucket[1] += 1
        bucket[0] += int(language_ok)

        expected_intent = expected["intent"]
        discourse_intent = response.conversation_intent
        subject_intent = (await intent_detector.detect(case["input"], language_detector.detect(case["input"]))).intent
        intent_ok = expected_intent in (discourse_intent, subject_intent)
        intent_hits += int(intent_ok)

        grounded = bool(response.sources)
        rows.append(
            f"{case['id']:20s} lang expected={expected_language:9s} observed={response.detected_language:9s} "
            f"[{'OK' if language_ok else 'MISS'}]  intent expected={expected_intent:15s} "
            f"discourse={discourse_intent:20s} subject={subject_intent:20s} [{'OK' if intent_ok else 'MISS'}]  "
            f"grounded={grounded!s:5s} latency_ms={latency_ms}"
        )

    async def _run_all() -> None:
        # Connected inside the same loop the queries run on -- motor binds its
        # client to the loop that created it (see tests/test_legal_benchmark.py).
        await mongodb.connect()
        for case in cases:
            await _run_case(case)

    asyncio.run(_run_all())

    report = "\n".join(rows)
    language_accuracy = round(language_hits / len(cases), 4)
    intent_accuracy = round(intent_hits / len(cases), 4)
    per_language_report = "\n".join(
        f"  {lang:9s}: {hits}/{total}" for lang, (hits, total) in sorted(per_language.items())
    )
    print(
        f"\nPHASE3 MULTILINGUAL E2E RESULTS ({len(cases)} cases)\n{report}\n"
        f"\nlanguage accuracy: {language_accuracy} ({language_hits}/{len(cases)})"
        f"\nintent accuracy:   {intent_accuracy} ({intent_hits}/{len(cases)})"
        f"\nper-language language-detection accuracy:\n{per_language_report}"
    )

    assert language_accuracy >= 0.80, f"language detection regressed:\n{report}"
