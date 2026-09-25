"""Milestone D: structured entities, legal roles and timeline intelligence.

The property under test throughout is not "did it find the thing" but "can
the finding be checked and is it honest about what it does not know". An
extractor that reports a complainant with no supporting text, or computes a
limitation date from an anchor the document never states, is worse than one
that finds less.
"""

import asyncio
from datetime import date, timedelta
from typing import Any

import pytest

from app.entity_extraction import hybrid, rules
from app.entity_extraction.extractor import EntityExtractor
from app.entity_extraction.timeline import extract_timeline
from app.llm.base import ChatMessage, LLMResponse
from app.schemas.extraction import ExtractedEntity


def _run(coro):
    return asyncio.run(coro)


_COMPLAINT = """IN THE COURT OF THE DISTRICT COURT AT PUNE
Case No. CC/451/2024
Complainant: Rahul Sharma, 12 MG Road, Pune 411001
Accused: Vikram Singh
FIR No. 233/2024 was registered at Kothrud Police Station.
The incident occurred on 12/03/2024 when Rs. 45,000 was transferred.
Section 138 of the Negotiable Instruments Act applies.
Vehicle MH12AB1234 was used. Annexure A-1 is attached.
"""


class _StubLLM:
    """A provider returning exactly what a test wants it to return."""

    provider_name = "stub"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self.content, model="stub", provider="stub")

    def stream(self, messages, temperature=0.1):  # pragma: no cover - unused
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Deterministic extraction, with evidence
# ---------------------------------------------------------------------------


def test_every_extracted_value_carries_the_text_it_was_read_from() -> None:
    result = _run(hybrid.extract(_COMPLAINT))
    assert result.entities
    for entity in result.entities:
        assert entity.source_text, f"{entity.entity_type} {entity.value!r} has no evidence"
        assert entity.extraction_method == "rule"


@pytest.mark.parametrize(
    "entity_type,expected",
    [
        ("case_number", "CC/451/2024"),
        ("fir_number", "233/2024"),
        ("amount", "Rs. 45,000"),
        ("vehicle_number", "MH12AB1234"),
        ("section", "Section 138"),
        ("act", "Negotiable Instruments Act"),
        ("document_reference", "Annexure A-1"),
        ("address", "12 MG Road, Pune 411001"),
    ],
)
def test_the_supported_entity_types_are_extracted(entity_type: str, expected: str) -> None:
    result = _run(hybrid.extract(_COMPLAINT))
    values = [entity.value for entity in result.of_type(entity_type)]  # type: ignore[arg-type]
    assert expected in values, f"{entity_type}: got {values}"


def test_a_legal_role_is_only_assigned_where_the_text_labels_it() -> None:
    result = _run(hybrid.extract(_COMPLAINT))
    assert [entity.value for entity in result.with_role("complainant")] == ["Rahul Sharma"]
    assert [entity.value for entity in result.with_role("accused")] == ["Vikram Singh"]


def test_an_unlabelled_name_is_not_given_a_role_or_extracted_as_a_party() -> None:
    """A capitalised pair of words is not evidence of a party."""
    text = "Rahul Sharma went to the market on 12/03/2024 and met Vikram Singh."
    result = _run(hybrid.extract(text))
    assert result.of_type("person") == []


def test_a_role_label_does_not_swallow_the_following_line() -> None:
    """`\\s+` in the name pattern read "Vikram Singh\\nFIR No" as one name."""
    result = _run(hybrid.extract(_COMPLAINT))
    accused = [entity.value for entity in result.with_role("accused")]
    assert accused == ["Vikram Singh"]
    assert not any("FIR" in value for value in accused)


def test_a_lowercase_sentence_after_a_role_word_is_not_a_person() -> None:
    """Case-insensitivity applied to the whole pattern turned `[A-Z]` into
    "any letter", and "The tenant shall vacate..." produced a person named
    "shall vacate the premises"."""
    result = _run(hybrid.extract("The tenant shall vacate the premises before the end of the month."))
    assert result.of_type("person") == []


def test_a_rupee_amount_is_not_read_as_a_section_number() -> None:
    """"Rs. 45,000" matched the `s\\.` abbreviation and invented "s. 45"."""
    result = _run(hybrid.extract("A sum of Rs. 45,000 was paid."))
    assert result.of_type("section") == []


def test_unresolved_role_candidates_are_kept_separate_from_findings() -> None:
    """"A vs B" names two sides without saying which is which."""
    result = _run(hybrid.extract("Ramesh Kumar vs State of Maharashtra, heard today."))
    assert result.of_type("person") == []
    assert {candidate.role for candidate in result.unresolved_roles} == {"petitioner", "respondent"}
    assert all(candidate.evidence for candidate in result.unresolved_roles)
    assert result.questions, "an unresolved role must become a question"


def test_two_roles_for_one_person_become_a_conflict_and_a_question() -> None:
    text = "Complainant: Rahul Sharma\nWitness: Rahul Sharma\n"
    result = _run(hybrid.extract(text))
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert set(conflict.roles) == {"complainant", "witness"}
    assert "Which is correct" in conflict.question
    assert conflict.question in result.questions


def test_conversation_facts_are_used_and_marked_as_such() -> None:
    result = _run(
        hybrid.extract(
            "The matter concerns a cheque returned unpaid.",
            conversation_facts={"complainant_name": "Asha Patil", "fir_number": "88/2026"},
        )
    )
    asha = next(entity for entity in result.entities if entity.value == "Asha Patil")
    assert asha.extraction_method == "conversation"
    assert asha.legal_role == "complainant"
    assert "You told me earlier" in asha.source_text


# ---------------------------------------------------------------------------
# 2. The model proposes; it never confirms
# ---------------------------------------------------------------------------


def _long(text: str) -> str:
    """Padded past the length below which no model call is made at all."""
    return text + ("\nThis paragraph exists to make the document long enough to warrant a model pass." * 6)


def test_a_model_value_absent_from_the_document_is_dropped() -> None:
    llm = _StubLLM(
        '{"entities": [{"value": "Sunita Deshmukh", "entity_type": "person", '
        '"legal_role": "complainant", "source_text": "Complainant: Sunita Deshmukh"}]}'
    )
    result = _run(hybrid.extract(_long(_COMPLAINT), llm=llm))
    assert llm.calls == 1
    assert not any(entity.value == "Sunita Deshmukh" for entity in result.entities)


def test_a_model_role_becomes_a_question_never_a_finding() -> None:
    llm = _StubLLM(
        '{"entities": [{"value": "Kothrud", "entity_type": "person", '
        '"legal_role": "witness", "source_text": "registered at Kothrud Police Station"}]}'
    )
    result = _run(hybrid.extract(_long(_COMPLAINT), llm=llm))
    assert all(
        entity.legal_role == "unknown"
        for entity in result.entities
        if entity.extraction_method == "llm"
    )
    assert any(candidate.role == "witness" for candidate in result.unresolved_roles)


def test_a_model_never_overrules_a_role_the_text_states() -> None:
    llm = _StubLLM(
        '{"entities": [{"value": "Rahul Sharma", "entity_type": "person", '
        '"legal_role": "accused", "source_text": "Complainant: Rahul Sharma"}]}'
    )
    result = _run(hybrid.extract(_long(_COMPLAINT), llm=llm))
    assert [entity.value for entity in result.with_role("complainant")] == ["Rahul Sharma"]
    assert not any(
        candidate.value == "Rahul Sharma" and candidate.role == "accused"
        for candidate in result.unresolved_roles
    )


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        "{",
        '{"entities": "not a list"}',
        '{"wrong_key": []}',
        '{"entities": [{"value": "", "entity_type": "person"}]}',
        '{"entities": [{"value": "Rahul Sharma", "entity_type": "not_a_real_type"}]}',
    ],
)
def test_malformed_model_output_falls_back_to_the_rule_results(content: str) -> None:
    llm = _StubLLM(content)
    result = _run(hybrid.extract(_long(_COMPLAINT), llm=llm))
    # The deterministic findings are all still there.
    assert [entity.value for entity in result.with_role("complainant")] == ["Rahul Sharma"]
    assert any(entity.entity_type == "case_number" for entity in result.entities)


def test_a_provider_that_raises_does_not_break_extraction() -> None:
    class _Broken:
        provider_name = "broken"

        async def chat(self, messages, temperature=0.1):
            raise RuntimeError("provider exploded")

        def stream(self, messages, temperature=0.1):  # pragma: no cover
            raise NotImplementedError

    result = _run(hybrid.extract(_long(_COMPLAINT), llm=_Broken()))  # type: ignore[arg-type]
    assert [entity.value for entity in result.with_role("complainant")] == ["Rahul Sharma"]


def test_a_short_document_never_costs_a_model_call() -> None:
    llm = _StubLLM('{"entities": []}')
    _run(hybrid.extract("Complainant: Rahul Sharma", llm=llm))
    assert llm.calls == 0


def test_a_model_finding_is_always_less_confident_than_a_rule_finding() -> None:
    llm = _StubLLM(
        '{"entities": [{"value": "Kothrud Police Station", "entity_type": "police_station", '
        '"source_text": "registered at Kothrud Police Station"}]}'
    )
    result = _run(hybrid.extract(_long(_COMPLAINT), llm=llm))
    llm_entities = [entity for entity in result.entities if entity.extraction_method == "llm"]
    rule_entities = [entity for entity in result.entities if entity.extraction_method == "rule"]
    assert llm_entities, "the stub value should have been accepted"
    assert max(entity.confidence for entity in llm_entities) < min(
        entity.confidence for entity in rule_entities if entity.entity_type != "obligation"
    )


# ---------------------------------------------------------------------------
# 3. Dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("12/03/2024", date(2024, 3, 12)),
        ("2024-03-12", date(2024, 3, 12)),
        ("15 March 2026", date(2026, 3, 15)),
        ("1st April 2025", date(2025, 4, 1)),
        ("15 मार्च 2026", date(2026, 3, 15)),
    ],
)
def test_dates_parse_in_the_forms_indian_documents_use(text: str, expected: date) -> None:
    assert rules.parse_date_text(text) == expected


def test_an_impossible_date_is_not_silently_reinterpreted() -> None:
    """13/25/2024 is not a date; reading it month-first would invent one."""
    assert rules.parse_date_text("13/25/2024") is None


def test_an_unreadable_date_is_reported_rather_than_dropped() -> None:
    result = extract_timeline("The hearing is fixed for 31/02/2025 at the district court.")
    assert result.undated_events, "a date-shaped span that does not parse must still surface"
    assert result.questions


# ---------------------------------------------------------------------------
# 4. Timeline and deadlines
# ---------------------------------------------------------------------------


def test_dated_events_are_sorted_and_undated_ones_kept_separate() -> None:
    text = (
        "The agreement was executed on 01/01/2024.\n"
        "The incident occurred on 12/03/2024.\n"
        "The tenant shall vacate within 30 days from the date of receipt of this notice.\n"
    )
    result = extract_timeline(text)
    assert [event.normalized_date for event in result.events] == ["2024-01-01", "2024-03-12"]
    assert len(result.undated_events) == 1
    assert result.undated_events[0].is_deadline


@pytest.mark.parametrize(
    "phrase,days",
    [
        ("You must reply within 30 days from the date of receipt.", 30),
        ("You must reply within thirty days from the date of receipt.", 30),
        ("Reply within 2 weeks of this notice.", 14),
        ("आपको 30 दिन के भीतर उत्तर देना होगा।", 30),
        ("Aapko 15 din ke andar reply karna hoga.", 15),
    ],
)
def test_relative_deadlines_are_understood_in_english_hindi_and_hinglish(phrase: str, days: int) -> None:
    result = extract_timeline(phrase)
    events = result.events + result.undated_events
    deadlines = [event for event in events if event.is_deadline]
    assert deadlines, f"no deadline found in {phrase!r}"
    assert deadlines[0].relative_days == days


def test_a_relative_deadline_with_no_anchor_is_asked_about_not_computed() -> None:
    """Computing "30 days from receipt" without knowing receipt produces a
    confident, wrong limitation date."""
    result = extract_timeline("You must reply within 30 days from the date of receipt of this notice.")
    deadline = result.undated_events[0]
    assert deadline.normalized_date == ""
    assert deadline.relative_to == "the date of receipt"
    assert any("What is that date" in question for question in result.questions)


def test_a_relative_deadline_with_an_anchor_in_the_same_sentence_is_resolved() -> None:
    result = extract_timeline("This notice is dated 01/03/2026 and you must reply within 30 days.")
    assert result.events
    assert result.events[0].normalized_date == "2026-03-31"
    assert result.events[0].relative_days == 30


def test_before_expiry_is_recorded_as_a_deadline_with_no_computable_date() -> None:
    result = extract_timeline("The option must be exercised before expiry of the lease.")
    assert result.undated_events[0].is_deadline
    assert result.undated_events[0].relative_to == "expiry"
    assert result.questions


def test_on_or_before_is_treated_as_a_deadline_and_its_expiry_reported() -> None:
    past = date.today() - timedelta(days=10)
    result = extract_timeline(
        f"Payment shall be made on or before {past.strftime('%d/%m/%Y')}.", today=date.today()
    )
    assert result.events[0].is_deadline
    assert result.events[0].expiry_status == "expired"


def test_an_undated_event_never_receives_a_placeholder_date() -> None:
    result = extract_timeline("The tenant shall vacate within 30 days of receipt.")
    assert all(event.normalized_date == "" for event in result.undated_events)


def test_a_responsible_party_is_only_recorded_when_the_sentence_names_one() -> None:
    with_party = extract_timeline("The tenant shall vacate the premises within 30 days of receipt.")
    assert with_party.undated_events[0].responsible_party == "the tenant"

    without = extract_timeline("A reply is expected within 30 days of receipt.")
    assert without.undated_events[0].responsible_party == ""


def test_contradictory_dates_are_reported_and_never_resolved() -> None:
    text = (
        "The complaint was filed on 01/01/2024.\n"
        "The incident occurred on 12/03/2024.\n"
    )
    result = extract_timeline(text)
    assert result.contradictions
    assert "before the earliest incident" in result.contradictions[0]


def test_an_implausible_year_is_flagged() -> None:
    result = extract_timeline("The agreement was executed on 01/01/1802.")
    assert any("outside any plausible range" in problem for problem in result.contradictions)


def test_timeline_dates_are_date_only_and_carry_no_invented_time() -> None:
    result = extract_timeline("The hearing is on 12/03/2024.")
    assert result.events[0].normalized_date == "2024-03-12"
    assert "T" not in result.events[0].normalized_date


def test_page_numbers_are_used_when_supplied_and_never_defaulted() -> None:
    sentence = "The incident occurred on 12/03/2024 at the premises."
    with_pages = extract_timeline(sentence, page_of={sentence[:40]: 7})
    assert with_pages.events[0].source_page == 7

    without_pages = extract_timeline(sentence)
    assert without_pages.events[0].source_page is None


# ---------------------------------------------------------------------------
# 5. The public entry point stays backward compatible
# ---------------------------------------------------------------------------


def test_the_flat_entity_map_is_unchanged_by_the_structured_layer() -> None:
    flat = _run(EntityExtractor().extract(_COMPLAINT))
    structured = _run(EntityExtractor().extract_structured(_COMPLAINT))
    assert structured.entities == flat.entities
    assert structured.confidence == flat.confidence


def test_the_structured_layer_adds_evidence_roles_and_a_timeline() -> None:
    result = _run(EntityExtractor().extract_structured(_COMPLAINT))
    assert result.structured
    assert any(entity.legal_role == "complainant" for entity in result.structured)
    assert result.timeline
    assert all(isinstance(entity, ExtractedEntity) for entity in result.structured)


def test_no_entity_is_ever_returned_without_a_confidence_and_a_method() -> None:
    result: Any = _run(EntityExtractor().extract_structured(_COMPLAINT))
    for entity in result.structured:
        assert 0.0 <= entity.confidence <= 1.0
        assert entity.extraction_method in {"rule", "conversation", "llm", "fallback"}
