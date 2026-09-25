"""Phase 2 Milestone G: the versioned legal benchmark.

What this suite asserts is **provenance, not prose**. The wording of an answer
belongs to the LLM and cannot be pinned without a human reviewer; the source an
answer rests on, the section it names, whether it declines when it should, and
whether it can ever reach across users are all properties of the system, and
those are what is checked here.

Two layers, deliberately separate:

  * *Dataset integrity* -- always runs. Guards the benchmark itself against the
    failure mode that makes a benchmark worthless: expectations quietly relaxed
    until whatever the system emits counts as correct.
  * *Retrieval metrics* -- runs against the live knowledge base, and is skipped
    when MongoDB is not reachable. It reports source precision, section
    accuracy, grounding pass rate, safe-decline rate and ownership isolation.

Run the metrics explicitly:
    .venv\\Scripts\\python.exe -m pytest tests/test_legal_benchmark.py -q -s -k metrics
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

BENCHMARK_PATH = Path(__file__).parent / "benchmark" / "legal_benchmark_v1.yaml"

# Sections asserted in the dataset must be verified against the Act's own text.
# This is the whitelist of what has actually been checked; anything else must
# not carry `expected_section`. Keeping it here rather than in the YAML means a
# new expectation cannot be added without also being added to this list.
_VERIFIED_SECTIONS: dict[str, str] = {
    "cheque-bounce": "138",
}


def _load() -> dict[str, Any]:
    return yaml.safe_load(BENCHMARK_PATH.read_text(encoding="utf-8"))


BENCHMARK = _load()
CASES: list[dict[str, Any]] = BENCHMARK["cases"]


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------


def test_the_benchmark_is_versioned() -> None:
    assert BENCHMARK["version"] == 1
    assert BENCHMARK["updated"]


def test_case_ids_are_unique() -> None:
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids))


def test_every_case_has_a_question_and_language() -> None:
    for case in CASES:
        assert case.get("question", "").strip(), case["id"]
        assert case.get("language") in {"english", "hindi", "hinglish"}, case["id"]


def test_every_case_states_an_expectation() -> None:
    """A case that expects nothing can never fail, and a suite of those reports
    a perfect score while checking nothing."""
    for case in CASES:
        has_expectation = any(
            case.get(key)
            for key in (
                "expected_category", "expected_source_contains", "expected_section",
                "insufficient_context", "expects_currency_warning", "must_not_contain",
            )
        )
        assert has_expectation, f"{case['id']} asserts nothing"


def test_expected_sections_are_only_asserted_where_verified() -> None:
    """The rule that keeps this dataset honest. An unverified expected section
    trains the suite to accept a wrong answer as correct."""
    for case in CASES:
        section = case.get("expected_section")
        if section is None:
            continue
        assert case["id"] in _VERIFIED_SECTIONS, (
            f"{case['id']} asserts section {section} but it is not in the verified list"
        )
        assert str(section) == _VERIFIED_SECTIONS[case["id"]], case["id"]


def test_the_major_legal_domains_are_covered() -> None:
    domains = {case.get("domain") for case in CASES}
    for required in (
        "criminal", "cyber", "consumer", "tenancy", "employment",
        "family", "cheque", "rti", "constitutional", "adversarial",
    ):
        assert required in domains, f"benchmark does not cover {required}"


def test_every_adversarial_category_is_represented() -> None:
    kinds = {case.get("adversarial") for case in CASES if case.get("adversarial")}
    for required in (
        "fabricated_case", "nonexistent_section", "prompt_injection",
        "stale_statute", "unsupported_conclusion", "cross_user_document",
    ):
        assert required in kinds, f"benchmark has no adversarial case for {required}"


def test_multiple_languages_are_exercised() -> None:
    languages = {case["language"] for case in CASES}
    assert {"english", "hindi", "hinglish"} <= languages


def test_declining_cases_do_not_also_demand_a_source() -> None:
    """A case cannot simultaneously require a decline and require a citation --
    that expectation can never be satisfied, and a permanently-failing case
    gets quietly excluded from the score."""
    for case in CASES:
        if case.get("insufficient_context"):
            assert not case.get("expected_source_contains"), case["id"]
            assert not case.get("expected_section"), case["id"]


def test_legacy_cases_expect_a_currency_warning() -> None:
    for case in CASES:
        if case.get("currency") == "legacy":
            assert case.get("expects_currency_warning"), (
                f"{case['id']} is a legacy-law case but does not require the "
                "non-retrospectivity disclosure"
            )


# ---------------------------------------------------------------------------
# Scoring helpers, unit-tested so the metrics cannot silently misreport
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Filenames in this corpus separate words with underscores
    (`The_Negotiable_Instruments_Act_1881.pdf`), while the expectations are
    written as prose (`negotiable instruments`). Without normalising the
    separators, every source expectation would miss and the benchmark would
    report a precision of zero against perfectly correct retrieval.
    """
    return text.lower().replace("_", " ").replace("-", " ").replace(".", " ")


def source_matches(case: dict[str, Any], source_documents: list[str]) -> bool:
    """Whether any retrieved source satisfies the case's source expectation."""
    expected = [_normalise(term) for term in case.get("expected_source_contains", [])]
    if not expected:
        return True
    haystack = _normalise(" ".join(source_documents))
    return any(term in haystack for term in expected)


def section_matches(case: dict[str, Any], sections: list[str]) -> bool:
    expected = case.get("expected_section")
    if expected is None:
        return True
    return str(expected) in {str(section).strip() for section in sections}


def test_source_matching_requires_a_real_overlap() -> None:
    case = {"expected_source_contains": ["negotiable instruments"]}
    assert source_matches(case, ["The_Negotiable_Instruments_Act_1881.pdf"]) is True
    assert source_matches(case, ["bns_2023.pdf"]) is False


def test_a_case_with_no_source_expectation_is_not_counted_as_a_miss() -> None:
    assert source_matches({}, []) is True


def test_section_matching_is_exact() -> None:
    assert section_matches({"expected_section": "138"}, ["138"]) is True
    assert section_matches({"expected_section": "138"}, ["1385"]) is False
    assert section_matches({"expected_section": "138"}, []) is False


# ---------------------------------------------------------------------------
# Live retrieval metrics
# ---------------------------------------------------------------------------


# Opt-in, for two reasons. It needs a populated MongoDB knowledge base, and the
# retriever loads `BAAI/bge-m3` -- a ~2.3 GB download on a machine that has not
# cached it. Neither belongs in a default test run, and a benchmark that
# silently pulls gigabytes is a benchmark people learn to skip.
#
#   set LEGAL_AI_RUN_BENCHMARK=1
#   .venv\Scripts\python.exe -m pytest tests/test_legal_benchmark.py -q -s -k metrics
requires_benchmark_env = pytest.mark.skipif(
    not os.getenv("LEGAL_AI_RUN_BENCHMARK"),
    reason="set LEGAL_AI_RUN_BENCHMARK=1 to run live retrieval metrics (needs MongoDB + the embedding model)",
)


@requires_benchmark_env
def test_benchmark_metrics() -> None:
    """Retrieve for every non-adversarial case and report the metrics.

    Asserts only floors that the system already clears, so this fails on a
    REGRESSION rather than on a normal fluctuation. Raising a floor is a
    deliberate act; lowering one to get a green run is the thing this suite
    exists to prevent.
    """
    from app.database.mongodb import mongodb
    from app.rag.reranker import LegalReranker
    from app.rag.retriever import LegalRetriever

    # Both stages, in the same order `ChatService` uses them. Measuring
    # `retrieve()` alone would score material the user never sees: the reranker
    # is what decides the final top-k, and a benchmark that skips it reports on
    # a pipeline that does not exist.
    retriever = LegalRetriever()
    reranker = LegalReranker()
    scored = [case for case in CASES if not case.get("insufficient_context")]

    source_hits = 0
    source_applicable = 0
    section_hits = 0
    section_applicable = 0
    grounded = 0
    coverage_gaps: list[str] = []

    async def _run() -> None:
        # Connected inside the same loop the queries run on: motor binds its
        # client to the loop that created it, and a client built on a
        # different (already-closed) loop fails with 'Event loop is closed'.
        await mongodb.connect()
        nonlocal source_hits, source_applicable, section_hits, section_applicable, grounded
        corpus = await mongodb.db["embeddings_metadata"].distinct("metadata.source_document")
        indexed = _normalise(" ".join(str(name) for name in corpus))

        for case in scored:
            rewritten, retrieved = await retriever.retrieve(case["question"], top_k=20)
            chunks = await reranker.rerank(rewritten, retrieved, top_k=8)
            documents = [str(chunk.metadata.get("source_document", "")) for chunk in chunks]
            sections = [str(chunk.metadata.get("section_number", "")) for chunk in chunks]
            if chunks:
                grounded += 1

            expected_sources = case.get("expected_source_contains", [])
            if expected_sources:
                # A case whose expected Act is not in the corpus AT ALL cannot
                # test ranking -- there is nothing to rank. Counting it as a
                # precision miss would make this metric measure ingestion
                # coverage instead, and would tempt someone to lower the
                # precision floor to accommodate a missing document. Reported
                # separately as a coverage gap, which is the real defect and a
                # different team's fix (re-ingest the Act).
                if not any(_normalise(term) in indexed for term in expected_sources):
                    coverage_gaps.append(case["id"])
                    continue
                source_applicable += 1
                source_hits += int(source_matches(case, documents))

            if case.get("expected_section"):
                section_applicable += 1
                section_hits += int(section_matches(case, sections))

    asyncio.run(_run())

    source_precision = source_hits / source_applicable if source_applicable else 1.0
    section_accuracy = section_hits / section_applicable if section_applicable else 1.0
    grounding_pass_rate = grounded / len(scored) if scored else 0.0

    print(
        "\nBENCHMARK v1 metrics"
        f"\n  cases scored:        {len(scored)}"
        f"\n  source precision:    {source_precision:.2f} ({source_hits}/{source_applicable})"
        f"\n  section accuracy:    {section_accuracy:.2f} ({section_hits}/{section_applicable})"
        f"\n  grounding pass rate: {grounding_pass_rate:.2f} ({grounded}/{len(scored)})"
        f"\n  corpus coverage gaps: {len(coverage_gaps)} {coverage_gaps}"
    )

    # 2026-09-07: BNS/BSA moved from needs_review to review_status=approved via
    # scripts/run_kb_machine_verification.py's real official-source hash match
    # (see docs/SOURCE_VERIFICATION_CHECKLIST.md). That took source precision
    # from 0.00 (nothing was approved, so the retriever surfaced nothing) to a
    # confirmed-stable 0.62 (5/8) across two live runs. The 3 remaining misses
    # (consumer-defective-goods, cheque-bounce, rti-application) are each a
    # document sitting in the corpus but still unverified, not a ranking
    # defect -- see the master checklist for what each one needs next.
    assert grounding_pass_rate >= 0.80, "retrieval returned nothing for too many benchmark questions"
    assert source_precision >= 0.60, (
        f"source precision {source_precision:.2f} regressed below the recorded floor. "
        "Do not lower this floor to get a green run -- investigate the ranking change."
    )


def test_safe_decline_rate() -> None:
    """Every adversarial case must be one the system is EXPECTED to refuse or
    qualify. This checks the dataset's own contract, not the LLM's output --
    the runtime behaviour is covered by the prompt-injection and grounding
    tests in `test_security.py` and `test_answer_quality_audit.py`.
    """
    adversarial = [case for case in CASES if case.get("adversarial")]
    protected = [
        case for case in adversarial
        if case.get("insufficient_context")
        or case.get("must_not_contain")
        or case.get("expects_currency_warning")
    ]
    safe_decline_rate = len(protected) / len(adversarial)
    print(f"\n  safe-decline coverage: {safe_decline_rate:.2f} ({len(protected)}/{len(adversarial)})")
    assert safe_decline_rate == 1.0, "an adversarial case has no safety expectation attached"


def test_ownership_isolation_case_is_present_and_expects_a_decline() -> None:
    """The cross-user case is the one adversarial entry whose failure would be a
    privacy breach rather than a wrong answer, so it is asserted by name."""
    case = next(case for case in CASES if case.get("adversarial") == "cross_user_document")
    assert case["insufficient_context"] is True
