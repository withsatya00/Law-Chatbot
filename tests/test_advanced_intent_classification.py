import asyncio
import json

import pytest

from app.core.config import settings
from app.intent.classifier import ConversationIntentClassifier
from app.llm.base import ChatMessage, LLMProvider, LLMResponse


class FakeIntentLLM(LLMProvider):
    provider_name = "test"

    def __init__(self, content: str, error: str | None = None) -> None:
        self.content = content
        self.error = error
        self.messages: list[ChatMessage] = []

    async def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> LLMResponse:
        self.messages = messages
        return LLMResponse(content=self.content, model="test", provider="test", error=self.error)

    async def stream(self, messages: list[ChatMessage], temperature: float = 0.1):
        yield ""

    async def health(self) -> bool:
        return True


def _classify(content: str, monkeypatch: pytest.MonkeyPatch) -> tuple:
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", True)
    llm = FakeIntentLLM(content)
    match = asyncio.run(
        ConversationIntentClassifier().classify_advanced(
            # Deliberately not a draft-trigger phrase: low deterministic
            # confidence (0.4, "General Legal Information" fallback, not
            # ambiguous, not protected) so this genuinely reaches the LLM
            # multi-intent branch these tests exercise -- a message that
            # also happened to match `DraftIntentDetector` would now
            # short-circuit deterministically before ever calling the LLM
            # (see "General Clarification Mode" / workflow-chain detection
            # in `classify()`'s draft-match branch).
            "I am not sure what to do about this legal situation, please help.",
            {"messages": [{"role": "assistant", "content": "We discussed the uploaded PDF."}]},
            llm,
        )
    )
    return match, llm


def test_llm_classifier_supports_multi_intent_and_selects_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    match, _ = _classify(
        json.dumps(
            {
                "intents": [
                    {"intent": "Document Analysis", "confidence": 0.93, "reason": "The user asks to review a PDF."},
                    {"intent": "Draft Generation", "confidence": 0.88, "reason": "The user also asks for a notice draft."},
                ],
                "primary_intent": "Document Analysis",
            }
        ),
        monkeypatch,
    )
    assert match.intent == "Document Analysis"
    assert match.detected_intents == ("Document Analysis", "Draft Generation")
    assert match.confidence == 0.93


def test_invalid_llm_output_falls_back_to_deterministic_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    match, llm = _classify("not json", monkeypatch)
    assert match.intent == "General Legal Information"
    assert match.detected_intents == ()
    assert match.classifier_source == "deterministic"
    assert llm.messages


def test_provider_error_falls_back_without_breaking_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", True)
    llm = FakeIntentLLM("", error="connection_failed")
    match = asyncio.run(ConversationIntentClassifier().classify_advanced("What is bail?", {"messages": []}, llm))
    assert match.intent == "Legal Explanation"


def test_advanced_classifier_does_not_override_protected_draft_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", True)
    llm = FakeIntentLLM(
        json.dumps({"intents": [{"intent": "Legal Advice", "confidence": 0.99}], "primary_intent": "Legal Advice"})
    )
    match = asyncio.run(
        ConversationIntentClassifier().classify_advanced("Generate a police complaint draft.", {"messages": []}, llm)
    )
    assert match.intent == "Draft Generation"
    assert not llm.messages


def test_correction_extracts_replacement_text_and_reclassifies_on_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", False)
    llm = FakeIntentLLM("")
    match = asyncio.run(
        ConversationIntentClassifier().classify_advanced("No, I meant bail", {"messages": []}, llm)
    )
    assert match.is_correction is True
    assert match.corrected_text == "bail"
    # The deterministic classifier must have run on "bail", not the full
    # correction sentence -- routing/retrieval downstream depends on this.
    assert match.intent != "Draft Generation"


def test_hinglish_correction_pattern_extracts_the_real_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", False)
    llm = FakeIntentLLM("")
    match = asyncio.run(
        ConversationIntentClassifier().classify_advanced(
            "Document analysis nahi, legal research chahiye", {"messages": []}, llm
        )
    )
    assert match.is_correction is True
    assert match.corrected_text == "legal research chahiye"


def test_correction_signal_without_new_text_has_no_corrected_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", False)
    llm = FakeIntentLLM("")
    match = asyncio.run(
        ConversationIntentClassifier().classify_advanced("You misunderstood", {"messages": []}, llm)
    )
    assert match.is_correction is True
    assert match.corrected_text is None


def test_ordinary_message_is_not_flagged_as_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", False)
    llm = FakeIntentLLM("")
    match = asyncio.run(
        ConversationIntentClassifier().classify_advanced("What is bail?", {"messages": []}, llm)
    )
    assert match.is_correction is False
    assert match.corrected_text is None


# ---------------------------------------------------------------------------
# Phase 1 "General Clarification Mode"
# ---------------------------------------------------------------------------


def test_ambiguous_document_and_draft_phrasing_asks_for_clarification(monkeypatch: pytest.MonkeyPatch) -> None:
    # No "based on it/that" link -- could mean "draft grounded in the
    # review" or "these are two separate asks." Must ask, never guess, and
    # must be protected from the LLM silently resolving it instead (LLM
    # disabled here to prove the deterministic path alone produces this).
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", False)
    llm = FakeIntentLLM("")
    match = asyncio.run(
        ConversationIntentClassifier().classify_advanced(
            "Review this PDF and draft a legal notice.", {"messages": []}, llm
        )
    )
    assert match.intent == "Workflow Clarification"
    assert match.detected_intents == ("Document Analysis", "Draft Generation")


def test_linked_document_and_draft_phrasing_does_not_ask_for_clarification(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same shape, but WITH the explicit link -- unambiguous chain, no
    # clarification needed.
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", False)
    llm = FakeIntentLLM("")
    match = asyncio.run(
        ConversationIntentClassifier().classify_advanced(
            "Review this PDF and draft a legal notice based on it.", {"messages": []}, llm
        )
    )
    assert match.intent == "Draft Generation"
    assert match.detected_intents == ("Document Analysis", "Draft Generation")


def test_clarification_intent_is_protected_from_llm_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", True)
    llm = FakeIntentLLM(
        json.dumps({"intents": [{"intent": "Document Analysis", "confidence": 0.95}], "primary_intent": "Document Analysis"})
    )
    match = asyncio.run(
        ConversationIntentClassifier().classify_advanced(
            "Review this PDF and draft a legal notice.", {"messages": []}, llm
        )
    )
    assert match.intent == "Workflow Clarification"
    assert not llm.messages


# ---------------------------------------------------------------------------
# Phase 1 "Intent Feedback"
# ---------------------------------------------------------------------------


def test_bare_wrong_intent_is_detected_with_no_corrected_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", False)
    llm = FakeIntentLLM("")
    match = asyncio.run(ConversationIntentClassifier().classify_advanced("Wrong intent.", {"messages": []}, llm))
    assert match.intent == "Intent Feedback"
    assert match.is_intent_feedback is True
    assert match.feedback_corrected_intent is None


@pytest.mark.parametrize(
    ("text", "expected_intent"),
    [
        ("I wanted legal research", "General Legal Information"),
        ("This is document analysis", "Document Analysis"),
        ("I wanted drafting", "Draft Generation"),
        ("This is translation", "Translation"),
    ],
)
def test_named_intent_feedback_resolves_the_correct_alias(
    text: str, expected_intent: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", False)
    llm = FakeIntentLLM("")
    match = asyncio.run(ConversationIntentClassifier().classify_advanced(text, {"messages": []}, llm))
    assert match.intent == "Intent Feedback"
    assert match.feedback_corrected_intent == expected_intent


def test_unrecognized_named_text_is_not_treated_as_intent_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    # "I wanted X" where X isn't a known routing category -- must not be
    # guessed/invented; falls through to ordinary classification.
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", False)
    llm = FakeIntentLLM("")
    match = asyncio.run(ConversationIntentClassifier().classify_advanced("I wanted a refund", {"messages": []}, llm))
    assert match.intent != "Intent Feedback"
    assert match.is_intent_feedback is False


def test_long_i_wanted_phrase_is_treated_as_a_real_question_not_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    # A genuine new question phrased as "I wanted X" (not bare feedback
    # about a prior turn) must not be swallowed as intent feedback.
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", False)
    llm = FakeIntentLLM("")
    match = asyncio.run(
        ConversationIntentClassifier().classify_advanced(
            "I wanted legal research on cheque bounce and dishonour cases in detail", {"messages": []}, llm
        )
    )
    assert match.intent != "Intent Feedback"


def test_intent_feedback_is_protected_from_llm_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "conversation_intent_llm_enabled", True)
    llm = FakeIntentLLM(
        json.dumps({"intents": [{"intent": "Legal Advice", "confidence": 0.99}], "primary_intent": "Legal Advice"})
    )
    match = asyncio.run(
        ConversationIntentClassifier().classify_advanced("This is document analysis", {"messages": []}, llm)
    )
    assert match.intent == "Intent Feedback"
    assert not llm.messages