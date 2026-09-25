import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.constants import DRAFT_DISCLAIMER
from app.drafting.conversation import DraftConversationEngine
from app.drafting.edit_commands import EditCommandInterpreter
from app.drafting.field_extraction import DraftFieldExtractor
from app.drafting.intent import DraftIntentDetector
from app.drafting.localized_dates import format_localized_date
from app.drafting.templates import get_template, list_templates
from app.drafting.templates.base import structure_sections_for
from app.drafting.validation import DraftFieldValidator
from app.drafting.wrapper_messages import msg
from app.llm.base import LLMResponse
from app.schemas.drafting import DraftGenerateResponse


def test_every_template_has_trigger_phrases_for_intent_detection() -> None:
    for template in list_templates():
        assert template.trigger_phrases, f"{template.draft_id} has no trigger_phrases for intent detection"


@pytest.mark.parametrize(
    "message,expected_draft_id",
    [
        ("I want to write a police complaint.", "police_complaint"),
        ("Generate an FIR draft.", "police_complaint"),
        ("RTI application banana hai.", "rti_application"),
        ("Consumer complaint draft.", "consumer_complaint"),
        ("Mujhe legal notice banana hai.", "legal_notice"),
        ("Cyber complaint likhni hai.", "cyber_crime_complaint"),
        ("Mujhe mobile chori ki police complaint banani hai.", "mobile_theft_complaint"),
        # Part 48 fix: "Cheque bounce notice." (no verb) was a stale example
        # left over from before Draft Mode started requiring an explicit
        # drafting verb (see `DraftIntentDetector.detect`'s "no exceptions
        # for a strong trigger-phrase match" design, added specifically to
        # stop bare noun phrases from auto-launching a draft) -- every other
        # case in this list already carries a verb ("write"/"Generate"/
        # "banana hai"/"draft"/"likhni hai"); this one didn't, so it could
        # never have matched under the current, intentional design. Not a
        # production regression: a bare "Cheque bounce notice." from a real
        # user is ambiguous (could be describing their situation, not
        # commanding a draft) and correctly requires a verb, same as every
        # other template.
        ("Cheque bounce notice draft.", "cheque_bounce_notice"),
        ("Affidavit banana hai.", "affidavit"),
    ],
)
def test_draft_intent_detector_matches_expected_template(message: str, expected_draft_id: str) -> None:
    detector = DraftIntentDetector()
    match = detector.detect(message)
    assert match.matched
    assert match.draft_id == expected_draft_id


def test_draft_intent_detector_returns_ambiguous_for_unmatched_document_type() -> None:
    # "Divorce petition" isn't in this starter pack (fast-follow batch), so a
    # generic "banana hai" cue should still trigger drafting mode, just
    # without a specific template, prompting a clarifying question instead
    # of a wrong/no match.
    detector = DraftIntentDetector()
    match = detector.detect("Divorce petition banana hai.")
    assert match.matched
    assert match.ambiguous
    assert match.draft_id is None


def test_draft_intent_detector_ignores_unrelated_chat() -> None:
    detector = DraftIntentDetector()
    match = detector.detect("What is the punishment for theft under BNS?")
    assert not match.matched


@pytest.mark.parametrize(
    "message",
    [
        "I need a lawyer",
        "I need to find a good lawyer for my case",
        "Can you recommend a lawyer for cheque bounce",
        "I want to know about RTI",
        "What is the process to file an FIR?",
        "FIR kaise file karein",
    ],
)
def test_draft_intent_detector_does_not_hijack_lawyer_and_process_questions(message: str) -> None:
    # Regression: bare "need"/"want"/"file" verbs and trigger-phrase words
    # (e.g. "cheque bounce", "RTI", "FIR") must not hijack lawyer-recommendation
    # asks or informational process questions into drafting mode just because
    # a document name is mentioned -- only a real drafting verb/noun pairing
    # or an explicit "create/draft/write" should.
    detector = DraftIntentDetector()
    match = detector.detect(message)
    assert not match.matched


@pytest.mark.parametrize(
    "message",
    [
        "what is fir",
        "What is FIR?",
        "What is an affidavit?",
        "Explain cheque bounce notice",
        # Regression: Hindi/Hinglish grammatical forms of "what is X" beyond
        # the masculine "kya hota hai" -- feminine ("kya hoti hai", since
        # FIR/report is a feminine noun), plural ("kya hote hain"), and the
        # "X kise kahte hai(n)" ("what is X called") phrasing were previously
        # missed by the informational guard and wrongly hijacked into
        # drafting mode.
        "fir kya hoti hai",
        "bail kya hoti hai",
        "fir kise kahte hai",
        "fir kise kehte hain",
        "FIR क्या होती है",
        "FIR किसे कहते हैं",
        # Regression: "what should I do" / "what I should do" (a request for
        # next-step guidance, not a drafting command) and "guide me about X"
        # were previously not covered by the informational guard at all.
        "what i should do after cheque bounce notice",
        "what should i do after cheque bounce",
        "guide me about fir process",
    ],
)
def test_draft_intent_detector_does_not_hijack_informational_questions(message: str) -> None:
    # A bare document-name trigger phrase inside a "what is X" / "explain X"
    # question must NOT hijack the conversation into drafting mode -- these
    # are ordinary questions about the document, answered by normal RAG chat.
    detector = DraftIntentDetector()
    match = detector.detect(message)
    assert not match.matched


@pytest.mark.parametrize(
    "message",
    ["create draft", "Create a draft", "generate draft please", "make a draft", "I need a draft", "draft chahiye"],
)
def test_draft_intent_detector_recognizes_bare_draft_requests(message: str) -> None:
    # Regression: a generic drafting request with no specific document type
    # named ("create draft") must still trigger the clarifying "which
    # document type?" flow, not fall through to normal chat.
    detector = DraftIntentDetector()
    match = detector.detect(message)
    assert match.matched
    assert match.ambiguous
    assert match.draft_id is None


def test_draft_intent_detector_drafting_verb_overrides_informational_guard() -> None:
    detector = DraftIntentDetector()
    match = detector.detect("How do I write a police complaint?")
    assert match.matched
    assert match.draft_id == "police_complaint"


def _extractor_with_no_llm() -> DraftFieldExtractor:
    """A `DraftFieldExtractor` whose LLM extraction pass is a deliberate no-op.

    These tests exercise the deterministic regex/heuristic extraction path
    only -- they must not depend on which LLM provider is actually configured
    (or whether it's reachable/fast), since that's an environment detail, not
    something these specific assertions are about.
    """
    extractor = DraftFieldExtractor()
    extractor.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    return extractor


def test_field_extractor_pulls_structured_fields_from_worked_example() -> None:
    # The exact worked example from the spec -- must extract name/city/date/
    # amount/bank via the deterministic regex/heuristic path alone.
    template = get_template("cyber_crime_complaint")
    assert template is not None
    message = (
        "My name is Satya Prakash Sharma. I live in Lucknow. On 20 July 2026 someone "
        "cheated me online and withdrew ₹50,000 from my SBI account. Please generate "
        "a Cyber Crime Complaint."
    )
    extracted = asyncio.run(_extractor_with_no_llm().extract(message, template))
    assert extracted["applicant_name"] == "Satya Prakash Sharma"
    assert extracted["city"] == "Lucknow"
    assert extracted["incident_date"] == "20 July 2026"
    assert extracted["fraud_amount"] == "50,000"
    assert extracted["bank_name"] == "SBI"


def test_field_extractor_catches_labeled_answer_regardless_of_missing_count() -> None:
    # Regression: previously only worked when exactly one field was missing.
    # A message that explicitly labels its answer ("Facts: ...") must be
    # recognized even while several other fields are still outstanding.
    template = get_template("cyber_crime_complaint")
    assert template is not None
    message = (
        "The facts: I received a call from an unknown number claiming to be from my bank "
        "asking for OTP. I shared it and the money was withdrawn immediately."
    )
    extracted = asyncio.run(_extractor_with_no_llm().extract(message, template))
    assert extracted["facts"].startswith("I received a call")


def test_field_extractor_catches_is_phrased_answer() -> None:
    template = get_template("rti_application")
    assert template is not None
    message = "The public authority is Delhi Municipal Corporation."
    extracted = asyncio.run(_extractor_with_no_llm().extract(message, template))
    assert extracted["public_authority_name"] == "Delhi Municipal Corporation"


def test_field_extractor_textarea_field_stops_before_next_labeled_field() -> None:
    # Regression: a textarea field's greedy capture must not swallow a
    # second labeled field that follows it in the same message.
    template = get_template("rti_application")
    assert template is not None
    message = (
        "Applicant Address: 10 Park Road, Delhi. Information Sought: Status of my property "
        "tax complaint filed on 5 May 2026."
    )
    extracted = asyncio.run(_extractor_with_no_llm().extract(message, template))
    assert extracted["applicant_address"] == "10 Park Road, Delhi."
    assert "Status of my property tax complaint" in extracted["information_sought"]
    assert "Information Sought" not in extracted["applicant_address"]


def test_field_extractor_stops_at_numbered_labels_in_pasted_collection_form() -> None:
    template = get_template("address_affidavit")
    assert template is not None
    message = """1. Purpose of Affidavit:
To confirm my present address.
2. Current Residential Address:
House No. 42, Shanti Nagar, Agra
3. Applicant Name:
Rahul Sharma
4. Facts of the Case:
I have lived at this address for three years.
5. Place:
Agra"""
    extracted = asyncio.run(_extractor_with_no_llm().extract(message, template))

    assert extracted["affidavit_purpose"] == "To confirm my present address."
    assert extracted["applicant_address"] == "House No. 42, Shanti Nagar, Agra"
    assert extracted["applicant_name"] == "Rahul Sharma"
    assert extracted["facts"] == "I have lived at this address for three years."
    assert extracted["place"] == "Agra"


def test_field_validator_flags_bad_phone_and_impossible_date() -> None:
    template = get_template("police_complaint")
    assert template is not None
    issues = DraftFieldValidator().validate(
        template,
        {
            "applicant_mobile": "12345",
            "incident_date": "40/13/2026",
        },
    )
    flagged_fields = {field_key for field_key, _ in issues}
    assert "applicant_mobile" in flagged_fields
    assert "incident_date" in flagged_fields


def test_field_validator_accepts_valid_values() -> None:
    template = get_template("police_complaint")
    assert template is not None
    issues = DraftFieldValidator().validate(
        template,
        {"applicant_mobile": "9876543210", "incident_date": "01/01/2026"},
    )
    assert issues == []


def test_edit_command_interpreter_parses_replace_field() -> None:
    template = get_template("police_complaint")
    assert template is not None
    command = EditCommandInterpreter().interpret("Please change the police station to Hazratganj Thana", template)
    assert command.action == "replace_field"
    assert command.target_field == "police_station"
    assert command.new_value == "Hazratganj Thana"


def test_edit_command_interpreter_parses_amount_change_on_legal_notice() -> None:
    # Regression (QA pass, 2026-09-11, BUG-004/follow-up): `legal_notice` had
    # no monetary field at all, so an explicit, unambiguous "Amount 50,000
    # se 65,000 kar do" against a preview-stage draft came back
    # `action=unknown`, was then treated as leaving the draft entirely
    # (`_is_draft_interruption` -> True), and the request was silently
    # misrouted to a plain RAG answer -- the amount was never changed and
    # the user got an unrelated legal-information response instead. The
    # template now carries an optional `claim_amount` field with an
    # `amount`/`deposit` synonym, matching this app's own established
    # convention for other monetary notice templates (`recovery_notice`'s
    # `principal_amount`, `cheque_bounce_notice`'s `cheque_amount`).
    template = get_template("legal_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret("Amount 50,000 se 65,000 kar do, baaki same rakho.", template)
    assert command.action == "replace_field"
    assert command.target_field == "claim_amount"
    assert command.new_value == "65,000"


def test_edit_command_interpreter_parses_ki_jagah_deadline_change_on_legal_notice() -> None:
    # Regression (QA pass, 2026-09-11, BUG-008/BUG-009): "15 days ki jagah 7
    # days kar do" ("instead of 15 days, make it 7 days") came back
    # `action=unknown` for two compounding reasons, both fixed:
    #   1. `legal_notice` had no synonym for `response_deadline_days` at all
    #      (fixed alongside `claim_amount`/BUG-005's fix).
    #   2. `_FROM_TO_PATTERN` only recognised "se"/"से" as the from-to
    #      connector, not the equally ordinary "ki jagah"/"ke jagah"
    #      ("instead of") -- and `_strip_possessives` was ALSO stripping the
    #      "ki" out of "ki jagah" as if it were the unrelated possessive
    #      particle ("recipient ka address"), corrupting the connector
    #      before the (now-fixed) pattern could even see it intact.
    # Consequence when unresolved: the message left the draft flow entirely
    # (`_is_draft_interruption` -> True) and was answered as an ordinary RAG
    # legal question -- which, live, produced a WRONG, confidently-stated
    # refusal citing an unrelated Maharashtra Shops and Establishments Act
    # leave-notice provision as if it legally barred shortening a legal
    # notice's own response deadline.
    template = get_template("legal_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret("15 days ki jagah 7 days kar do.", template)
    assert command.action == "replace_field"
    assert command.target_field == "response_deadline_days"
    assert command.new_value == "7 days"


# ---------------------------------------------------------------------------
# BUG-010 (QA pass, 2026-09-11, session 3): "add a fact to the existing
# draft" had no representation in `EditAction` at all -- only `replace_field`
# (overwrite) existed. Live repro: "Ye fact add karo: maine 5 aur 12 August
# ko reminders bheje." left the draft flow entirely and was answered as a
# brand-new legal question (re-asking "Which State?" even though it had
# already been established). A rejected shortcut fix (mapping "add" onto the
# existing `replace_field` action against the `facts` field) would have
# silently DELETED whatever facts were already there the moment a user tried
# to add one more -- worse than the original failure. The real fix below is
# a genuine third action, `append_field`, that reads and merges with the
# CURRENT field value rather than overwriting it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_value",
    [
        ("Ye fact add karo: maine 5 aur 12 August ko reminders bheje.", "maine 5 aur 12 August ko reminders bheje"),
        ("Add that I sent reminders on 5 and 12 August.", "I sent reminders on 5 and 12 August"),
        ("Please mention that I sent reminders on 5 and 12 August.", "I sent reminders on 5 and 12 August"),
        ("Insert karo: I also called the landlord twice.", "I also called the landlord twice"),
        ("Isme jodo: maine landlord ko do baar call bhi kiya.", "maine landlord ko do baar call bhi kiya"),
    ],
)
def test_edit_command_interpreter_parses_append_intent_english_and_hinglish(
    message: str, expected_value: str
) -> None:
    template = get_template("legal_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret(message, template)
    assert command.action == "append_field"
    assert command.target_field == "facts"  # no field explicitly named -> the template's own narrative field
    assert command.new_value == expected_value


def test_edit_command_interpreter_append_honours_an_explicit_section_target() -> None:
    template = get_template("legal_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret(
        "Facts section mein add karo ki maine reminders bheje the.", template
    )
    assert command.action == "append_field"
    assert command.target_field == "facts"
    assert command.new_value == "maine reminders bheje the"


def test_edit_command_interpreter_append_falls_back_to_facts_when_no_target_named() -> None:
    # "Is paragraph mein ye line add karo" / "End mein ek paragraph add
    # karo" -- neither "paragraph" nor "end" is a field name on any
    # template, so this must land on the template's narrative field rather
    # than failing outright (rule 4: pick a safe semantic location).
    template = get_template("legal_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret(
        "Is paragraph mein ye line add karo: landlord ne abhi tak jawab nahi diya.", template
    )
    assert command.action == "append_field"
    assert command.target_field == "facts"
    assert command.new_value == "landlord ne abhi tak jawab nahi diya"


def test_edit_command_interpreter_append_never_targets_a_non_narrative_field() -> None:
    # "Add" naming a single-value field (a phone number) doesn't make
    # semantic sense to literally append to -- falls back to `facts`
    # instead of corrupting a structured value.
    template = get_template("legal_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret("Add mobile number 9988776655 also.", template)
    assert command.action == "append_field"
    assert command.target_field == "facts"


def test_edit_command_interpreter_append_with_no_content_asks_for_it() -> None:
    """Draft-editing pass (2026-09-12): the action MUST actually resolve to
    `append_field` (with `new_value=None`) so `DraftConversationEngine.
    _continue_preview`'s own "Sure -- what should I add?" branch fires --
    that branch is only ever reached when `target_field` is truthy. Before
    this fix, `_match_append_field_and_value` returned `(None, None)`
    whenever no content followed the verb, which made `interpret()` fall
    all the way through to `unknown` -- and `unknown` does NOT ask a
    clarifying question, it exits the draft flow entirely (Part 38 "Draft
    Auto-Pause Engine") and gets answered as an unrelated question. This
    test's own name always promised "asks for it"; the old assertion
    checked for the ONE outcome that structurally cannot do that.
    """
    template = get_template("legal_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret("Add karo", template)
    assert command.action == "append_field"
    assert command.target_field == "facts"
    assert command.new_value is None


def test_edit_command_interpreter_add_versus_replace_classification() -> None:
    # The same underlying content-change intent, worded as add vs. change,
    # must resolve to two DIFFERENT actions with different semantics.
    template = get_template("legal_notice")
    assert template is not None
    add_command = EditCommandInterpreter().interpret("Add that the landlord was rude on the call.", template)
    replace_command = EditCommandInterpreter().interpret(
        "Change the facts to: the landlord was rude on the call.", template
    )
    assert add_command.action == "append_field"
    assert replace_command.action == "replace_field"


def test_edit_command_interpreter_suggest_only_never_triggers_append() -> None:
    # Rule 12: an explanation/suggestion-only request must not invoke append.
    template = get_template("legal_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret(
        "Sirf improvements suggest karo, abhi draft modify mat karo.", template
    )
    assert command.action != "append_field"


class TestAppendValueMerge:
    """`EditCommandInterpreter.append_value` -- the actual content merge."""

    def test_appends_to_existing_content(self) -> None:
        merged = EditCommandInterpreter.append_value(
            "I vacated the flat on 1 August 2026.", "I sent reminders on 5 and 12 August."
        )
        assert merged == "I vacated the flat on 1 August 2026. I sent reminders on 5 and 12 August."

    def test_existing_content_is_never_discarded(self) -> None:
        merged = EditCommandInterpreter.append_value("Fact one.", "Fact two.")
        assert "Fact one." in merged
        assert "Fact two." in merged

    def test_duplicate_addition_does_not_duplicate_content(self) -> None:
        existing = "I vacated the flat on 1 August 2026. I sent reminders on 5 and 12 August."
        merged = EditCommandInterpreter.append_value(existing, "I sent reminders on 5 and 12 August.")
        assert merged == existing
        assert merged.count("reminders on 5 and 12 August") == 1

    def test_empty_existing_field_just_becomes_the_new_content(self) -> None:
        assert EditCommandInterpreter.append_value("", "First fact.") == "First fact."


def test_append_field_merges_with_current_persisted_value_not_stale_rendered_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End-to-end through `DraftConversationEngine`: the merge must read the
    # CURRENT persisted field value (via `draft_engine.drafts.find_by_id`)
    # and hand `regenerate` the combined text -- not just the new sentence
    # alone (which would be `replace_field`'s behaviour, not append's).
    engine = DraftConversationEngine()
    template = get_template("legal_notice")
    assert template is not None
    memory: dict = {
        "draft_mode": True, "draft_stage": "preview", "draft_template_id": "legal_notice",
        "draft_id": "draft-append-1", "draft_language": "english",
    }
    monkeypatch.setattr(
        engine.draft_engine.drafts, "find_by_id",
        AsyncMock(return_value={"fields": {"facts": "I vacated the flat on 1 August 2026."}}),
    )
    fake_result = DraftGenerateResponse(
        status="complete", draft_id="draft-append-1", template_id="legal_notice", template_name=template.name,
        sections={}, full_text="UPDATED DRAFT TEXT", language="english",
    )
    regenerate_mock = AsyncMock(return_value=fake_result)
    monkeypatch.setattr(engine.draft_engine, "regenerate", regenerate_mock)

    result = asyncio.run(
        engine.handle_turn(
            "session-append-1", "Add that I sent reminders on 5 and 12 August.", "english", memory,
        )
    )

    assert result is not None
    regenerate_mock.assert_awaited_once_with(
        "draft-append-1",
        {"facts": "I vacated the flat on 1 August 2026. I sent reminders on 5 and 12 August"},
    )
    assert "UPDATED DRAFT TEXT" in result.reply_text


def test_append_field_duplicate_request_is_a_no_op_not_a_false_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rule 7 + rule 9 combined: asking to add a fact that's already present
    # must not call `regenerate` again (no pointless version bump, no risk
    # of silently duplicating text) and must say so rather than claiming an
    # update that didn't happen.
    engine = DraftConversationEngine()
    template = get_template("legal_notice")
    assert template is not None
    memory: dict = {
        "draft_mode": True, "draft_stage": "preview", "draft_template_id": "legal_notice",
        "draft_id": "draft-append-2", "draft_language": "english",
    }
    existing_facts = "I vacated the flat on 1 August 2026. I sent reminders on 5 and 12 August."
    monkeypatch.setattr(
        engine.draft_engine.drafts, "find_by_id",
        AsyncMock(return_value={"fields": {"facts": existing_facts}}),
    )
    monkeypatch.setattr(
        engine.draft_engine, "get_current",
        AsyncMock(return_value=DraftGenerateResponse(
            status="complete", draft_id="draft-append-2", template_id="legal_notice", template_name=template.name,
            sections={}, full_text="CURRENT DRAFT TEXT", language="english",
        )),
    )
    regenerate_mock = AsyncMock()
    monkeypatch.setattr(engine.draft_engine, "regenerate", regenerate_mock)

    result = asyncio.run(
        engine.handle_turn(
            "session-append-2", "Add that I sent reminders on 5 and 12 August.", "english", memory,
        )
    )

    assert result is not None
    regenerate_mock.assert_not_awaited()
    assert "CURRENT DRAFT TEXT" in result.reply_text


def test_edit_command_interpreter_parses_translate() -> None:
    template = get_template("police_complaint")
    assert template is not None
    command = EditCommandInterpreter().interpret("Please translate into english", template)
    assert command.action == "translate"
    assert command.target_language == "english"


@pytest.mark.parametrize(
    "message,expected_language",
    [
        ("Hindi me kar do", "hindi"),
        ("English me kar do", "english"),
        ("Tamil me translate karo", "tamil"),
    ],
)
def test_edit_command_interpreter_recognizes_language_change_without_the_word_translate(
    message: str, expected_language: str
) -> None:
    # Part 52: a revision request naming a language explicitly is a language
    # change regardless of whether it literally contains "translate".
    template = get_template("police_complaint")
    assert template is not None
    command = EditCommandInterpreter().interpret(message, template)
    assert command.action == "translate"
    assert command.target_language == expected_language


def test_edit_command_interpreter_parses_one_page_format_request() -> None:
    template = get_template("police_complaint")
    assert template is not None
    command = EditCommandInterpreter().interpret("Make it one page", template)
    assert command.action == "format"


def test_edit_command_interpreter_unknown_for_unrecognized_message() -> None:
    template = get_template("police_complaint")
    assert template is not None
    command = EditCommandInterpreter().interpret("What's the weather like today?", template)
    assert command.action == "unknown"


def test_conversation_engine_end_to_end_collecting_to_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = DraftConversationEngine()
    # police_complaint has category "Complaint" -- its own compact skeleton
    # (To/Subject/Complainant Details/Facts/Request/Place/Date/Signature).
    sectioned_response = "\n".join(
        f"## {heading}\nContent for {heading}." for heading in structure_sections_for("Complaint")
    )
    engine.draft_engine.llm.chat = AsyncMock(
        return_value=LLMResponse(content=sectioned_response, model="test", provider="test")
    )
    # `extractor` owns its own separate LLM instance -- mock it too (a
    # no-op/empty response) so this test drives the deterministic
    # regex-extraction path, same as the field-extractor unit tests above,
    # regardless of which provider is actually configured.
    engine.extractor.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    monkeypatch.setattr(engine.draft_engine.drafts, "insert", AsyncMock(return_value="draft-123"))
    monkeypatch.setattr(engine.draft_engine.versions, "insert", AsyncMock(return_value="version-1"))

    memory: dict = {}
    first_message = (
        "I want to write a police complaint. My name is Ramesh Kumar, mobile 9876543210, "
        "police station Hazratganj, incident happened at MG Road."
    )
    first_result = asyncio.run(engine.handle_turn("session-1", first_message, "english", memory))
    assert first_result is not None
    assert first_result.info.stage == "collecting"
    assert memory["draft_mode"] is True
    assert memory["draft_template_id"] == "police_complaint"
    assert memory["draft_fields"]["applicant_name"] == "Ramesh Kumar"
    assert memory["draft_fields"]["applicant_mobile"] == "9876543210"
    # facts/expected_relief/applicant_address are still missing at this point
    assert "facts" in first_result.info.missing_fields or "applicant_address" in first_result.info.missing_fields

    second_message = (
        "The facts are: goods were stolen from my shop on 1 January 2026. "
        "I want the police to register an FIR and investigate. My address is 12 MG Road, Hazratganj."
    )
    second_result = asyncio.run(engine.handle_turn("session-1", second_message, "english", memory))
    assert second_result is not None

    # Depending on what the single-missing-field fallback captured, either the
    # draft is already complete (preview) or one more turn is needed -- both
    # are valid outcomes of the deterministic no-LLM extraction path. Drive
    # one more turn if still collecting to reach preview deterministically.
    if second_result.info.stage == "collecting":
        remaining = second_result.info.missing_fields
        for field_key in remaining:
            memory["draft_fields"][field_key] = f"Test value for {field_key}"
        third_result = asyncio.run(engine.handle_turn("session-1", "here you go", "english", memory))
        final_result = third_result
    else:
        final_result = second_result

    assert final_result is not None
    assert final_result.info.stage == "preview"
    assert memory["draft_stage"] == "preview"
    assert memory["draft_id"] is not None
    assert "Content for Facts of the Case." in final_result.info.full_text


def test_conversation_engine_ignores_non_drafting_message() -> None:
    engine = DraftConversationEngine()
    memory: dict = {}
    result = asyncio.run(engine.handle_turn("session-2", "What is the punishment for theft under BNS?", "english", memory))
    assert result is None
    assert not memory.get("draft_mode")


def test_continue_draft_re_displays_state_without_consuming_it_as_a_field_value() -> None:
    # The chat layer's post-interruption reminder tells the user to say
    # "continue draft" to pick a paused draft back up -- that phrase must
    # re-show the current collecting state, not get swallowed as an attempted
    # answer to whichever field is still missing.
    engine = DraftConversationEngine()
    memory: dict = {
        "draft_mode": True,
        "draft_stage": "collecting",
        "draft_template_id": "rti_application",
        "draft_fields": {"applicant_name": "Ramesh Kumar"},
        "draft_id": None,
    }
    result = asyncio.run(engine.handle_turn("session-3", "continue draft", "english", memory))
    assert result is not None
    assert result.info.stage == "collecting"
    assert memory["draft_fields"] == {"applicant_name": "Ramesh Kumar"}
    assert "public_authority_name" in result.info.missing_fields


def _engine_with_no_llm_extraction() -> DraftConversationEngine:
    engine = DraftConversationEngine()
    engine.extractor.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    return engine


def test_start_collecting_honors_explicit_language_named_in_the_message() -> None:
    # Part 52: an explicit language request ("Tamil me banao") wins even
    # though the conversation itself is in English.
    engine = _engine_with_no_llm_extraction()
    memory: dict = {}
    result = asyncio.run(
        engine.handle_turn("session-lang-1", "Draft police complaint Tamil me banao", "english", memory)
    )
    assert result is not None
    assert memory["draft_language"] == "tamil"


def test_start_collecting_falls_back_to_conversation_language_without_explicit_request() -> None:
    engine = _engine_with_no_llm_extraction()
    memory: dict = {}
    result = asyncio.run(engine.handle_turn("session-lang-2", "Legal notice bana do", "hindi", memory))
    assert result is not None
    assert memory["draft_language"] == "hindi"


def test_explicit_language_hint_survives_ambiguous_template_selection() -> None:
    # "Draft it in English" names no document type, so it goes through
    # problem-first discovery first ("describe_problem") -- the explicit
    # language must still be applied once the user names the actual template
    # in a follow-up message (a direct name always skips the rest of
    # discovery, even mid-flow -- see `_continue_discovery_describe`).
    engine = _engine_with_no_llm_extraction()
    memory: dict = {}
    first = asyncio.run(engine.handle_turn("session-lang-3", "Draft it in English", "hindi", memory))
    assert first is not None
    assert first.info.stage == "describe_problem"
    assert memory["draft_language_hint"] == "english"

    second = asyncio.run(engine.handle_turn("session-lang-3", "Legal Notice", "hindi", memory))
    assert second is not None
    assert memory["draft_language"] == "english"
    assert "draft_language_hint" not in memory


def test_continue_preview_language_change_regenerates_and_persists_via_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Part 52: a preview-stage language change must go through
    # `LegalDraftEngine.regenerate` (which persists `sections`/`language` to
    # the draft record) rather than the old `translate()` call, which never
    # wrote its output back -- this is what keeps a subsequent PDF/DOCX
    # download in sync with what was just shown in chat.
    engine = DraftConversationEngine()
    template = get_template("police_complaint")
    assert template is not None
    memory: dict = {
        "draft_mode": True,
        "draft_stage": "preview",
        "draft_template_id": "police_complaint",
        "draft_fields": {"applicant_name": "Ramesh Kumar"},
        "draft_id": "draft-xyz",
        "draft_language": "english",
    }
    fake_result = DraftGenerateResponse(
        status="complete",
        draft_id="draft-xyz",
        template_id="police_complaint",
        template_name=template.name,
        sections={"Recipient": "..."},
        full_text="TAMIL DRAFT TEXT",
        language="tamil",
    )
    regenerate_mock = AsyncMock(return_value=fake_result)
    monkeypatch.setattr(engine.draft_engine, "regenerate", regenerate_mock)

    result = asyncio.run(engine.handle_turn("session-lang-4", "Tamil me kar do", "english", memory))

    assert result is not None
    regenerate_mock.assert_awaited_once_with("draft-xyz", {}, language="tamil")
    assert memory["draft_language"] == "tamil"
    assert "TAMIL DRAFT TEXT" in result.reply_text


def test_continue_preview_tone_change_regenerates_with_style_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # QA pass 2026-09-11, step 8 of the Priority-2 journey: "Tone polite but
    # firm karo." previously had no recognized action at all and left the
    # draft flow entirely. Must reach `regenerate` with the SAME fields
    # (`{}`  -- no field is being changed) plus `style_instruction` set to
    # the requested tone.
    engine = DraftConversationEngine()
    template = get_template("legal_notice")
    assert template is not None
    memory: dict = {
        "draft_mode": True, "draft_stage": "preview", "draft_template_id": "legal_notice",
        "draft_id": "draft-tone-1", "draft_language": "english",
    }
    fake_result = DraftGenerateResponse(
        status="complete", draft_id="draft-tone-1", template_id="legal_notice", template_name=template.name,
        sections={}, full_text="RESTYLED DRAFT TEXT", language="english",
    )
    regenerate_mock = AsyncMock(return_value=fake_result)
    monkeypatch.setattr(engine.draft_engine, "regenerate", regenerate_mock)

    result = asyncio.run(engine.handle_turn("session-tone-1", "Tone polite but firm karo.", "english", memory))

    assert result is not None
    regenerate_mock.assert_awaited_once_with("draft-tone-1", {}, style_instruction="polite but firm")
    assert "RESTYLED DRAFT TEXT" in result.reply_text


def test_preview_stage_prioritizes_explicit_new_template_over_field_edit() -> None:
    engine = DraftConversationEngine()
    engine.extractor.extract = AsyncMock(return_value={})
    memory: dict = {
        "draft_mode": True,
        "draft_stage": "preview",
        "draft_template_id": "police_complaint",
        "draft_id": "draft-police-1",
        "draft_language": "hindi",
        "draft_fields": {
            "applicant_name": "Rohit Sharma",
            "applicant_email": "old@example.com",
        },
    }

    result = asyncio.run(
        engine.handle_turn(
            "s1",
            (
                "मैंने 15 अगस्त 2026 को नोएडा में ABC Electronics से ₹24,999 का मोबाइल खरीदा। "
                "10 सितंबर को ईमेल से शिकायत की, लेकिन समाधान नहीं मिला। Consumer complaint तैयार करो।"
            ),
            "hindi",
            memory,
        )
    )

    assert result is not None
    assert memory["draft_template_id"] == "consumer_complaint"
    assert memory["draft_stage"] == "collecting"
    assert memory["parked_drafts"][0]["draft_template_id"] == "police_complaint"


def test_amount_only_change_with_no_new_value_asks_for_it_instead_of_corrupting_the_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draft-editing pass (2026-09-12): "Sirf amount badlo, baaki same
    rakho." names the field (`claim_amount`) but never states what the new
    amount actually is -- the trailing "baaki same rakho" ("keep the rest
    the same") is a scope instruction, not a value. Before
    `_UNCHANGED_REST_TRAILER_PATTERN`, the interpreter's own end-anchored
    value-extraction patterns captured that trailing clause AS the value
    (confirmed directly: `new_value=", baaki same rakho"`), and this turn
    would have silently written that literal text into `claim_amount` and
    regenerated the document with it -- corrupting a field the user never
    gave a real value for. Must instead ask which value to use and leave
    the draft untouched.
    """
    engine = DraftConversationEngine()
    template = get_template("legal_notice")
    assert template is not None
    memory: dict = {
        "draft_mode": True, "draft_stage": "preview", "draft_template_id": "legal_notice",
        "draft_id": "draft-amount-1", "draft_language": "english",
    }
    regenerate_mock = AsyncMock()
    monkeypatch.setattr(engine.draft_engine, "regenerate", regenerate_mock)

    result = asyncio.run(
        engine.handle_turn("session-amount-1", "Sirf amount badlo, baaki same rakho.", "english", memory)
    )

    assert result is not None
    regenerate_mock.assert_not_awaited()
    assert "baaki same rakho" not in result.reply_text.lower()
    assert result.info.stage == "preview"


def test_append_with_no_content_asks_what_to_add_instead_of_leaving_the_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draft-editing pass (2026-09-12): a bare "add a new fact/paragraph"
    with no actual content must ask what to add, not silently exit the
    draft flow to be answered as an unrelated question (which is what
    `action="unknown"` causes -- see `test_edit_command_interpreter_
    append_with_no_content_asks_for_it`'s own updated docstring for the
    interpreter-level half of this fix).
    """
    engine = DraftConversationEngine()
    template = get_template("legal_notice")
    assert template is not None
    memory: dict = {
        "draft_mode": True, "draft_stage": "preview", "draft_template_id": "legal_notice",
        "draft_id": "draft-append-3", "draft_language": "english",
    }
    regenerate_mock = AsyncMock()
    monkeypatch.setattr(engine.draft_engine, "regenerate", regenerate_mock)

    result = asyncio.run(engine.handle_turn("session-append-3", "Ye naya fact add karo.", "english", memory))

    assert result is not None, "the message must stay inside the draft flow, not fall through to RAG"
    regenerate_mock.assert_not_awaited()
    assert "what should i add" in result.reply_text.lower()
    assert result.info.stage == "preview"


def test_suggest_only_request_never_mutates_the_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding-010 (QA pass, 2026-09-11) fix: "Suggestions do, draft change
    mat karo." must read the current draft and comment on it WITHOUT ever
    calling `regenerate` -- structurally, not just per the wording of the
    reply, so a bug or a failed LLM call can never corrupt or lose the
    persisted draft.
    """
    engine = DraftConversationEngine()
    template = get_template("legal_notice")
    assert template is not None
    memory: dict = {
        "draft_mode": True, "draft_stage": "preview", "draft_template_id": "legal_notice",
        "draft_id": "draft-suggest-1", "draft_language": "english",
    }
    monkeypatch.setattr(
        engine.draft_engine, "get_current",
        AsyncMock(return_value=DraftGenerateResponse(
            status="complete", draft_id="draft-suggest-1", template_id="legal_notice", template_name=template.name,
            sections={}, full_text="CURRENT DRAFT TEXT", language="english",
        )),
    )
    engine.draft_engine.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="- Consider adding the exact vacating date.\n- State the deposit amount explicitly.",
            model="test", provider="test",
        )
    )
    regenerate_mock = AsyncMock()
    monkeypatch.setattr(engine.draft_engine, "regenerate", regenerate_mock)

    result = asyncio.run(
        engine.handle_turn("session-suggest-1", "Suggestions do, draft change mat karo.", "english", memory)
    )

    assert result is not None
    regenerate_mock.assert_not_awaited()
    engine.draft_engine.llm.chat.assert_awaited_once()
    assert "not been changed" in result.reply_text.lower()
    assert "consider adding the exact vacating date" in result.reply_text.lower()
    assert result.info.stage == "preview"


def test_suggest_only_request_degrades_gracefully_when_the_llm_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed suggestion call must say so plainly and still never touch
    the draft -- never silently claim success, and never fall back to a
    draft mutation of any kind."""
    engine = DraftConversationEngine()
    template = get_template("legal_notice")
    assert template is not None
    memory: dict = {
        "draft_mode": True, "draft_stage": "preview", "draft_template_id": "legal_notice",
        "draft_id": "draft-suggest-2", "draft_language": "english",
    }
    monkeypatch.setattr(
        engine.draft_engine, "get_current",
        AsyncMock(return_value=DraftGenerateResponse(
            status="complete", draft_id="draft-suggest-2", template_id="legal_notice", template_name=template.name,
            sections={}, full_text="CURRENT DRAFT TEXT", language="english",
        )),
    )
    engine.draft_engine.llm.chat = AsyncMock(side_effect=TimeoutError("provider stalled"))
    regenerate_mock = AsyncMock()
    monkeypatch.setattr(engine.draft_engine, "regenerate", regenerate_mock)

    result = asyncio.run(
        engine.handle_turn("session-suggest-2", "Please suggest improvements to this draft.", "english", memory)
    )

    assert result is not None
    regenerate_mock.assert_not_awaited()
    assert "couldn't generate suggestions" in result.reply_text.lower()
    assert result.info.stage == "preview"


# ------------------------------------------------------------------------
# Part 52 "Drafting Workflow Localization" (second round): the CONVERSATION
# ITSELF -- not just the generated document -- must switch language, Hindi
# document names must be recognized, the AI disclaimer must never appear
# inside the document, and the "Date" section uses numeric DD/MM/YYYY.
# ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hindi_name,expected_draft_id",
    [
        ("सामान्य कानूनी नोटिस", "legal_notice"),
        ("साइबर अपराध शिकायत", "cyber_crime_complaint"),
        ("पुलिस शिकायत", "police_complaint"),
        ("सूचना का अधिकार आवेदन", "rti_application"),
        ("उपभोक्ता शिकायत", "consumer_complaint"),
        ("चेक बाउंस नोटिस", "cheque_bounce_notice"),
        ("शपथपत्र", "affidavit"),
        ("धन वसूली नोटिस", "recovery_notice"),
        ("किराया वापसी नोटिस", "rent_notice"),
    ],
)
def test_draft_intent_detector_recognizes_official_hindi_document_names(
    hindi_name: str, expected_draft_id: str
) -> None:
    # A user typing the exact Hindi document name shown when templates are
    # listed (`template.hindi_name`, or a shorter common form of it) must
    # match its own template, not fall through to "I didn't catch which
    # document you need."
    detector = DraftIntentDetector()
    match = detector.detect_named_template(hindi_name)
    assert match.matched
    assert match.draft_id == expected_draft_id


def test_start_selecting_prompt_is_localized_to_the_conversation_language() -> None:
    # A generic drafting request ("I need to make a document") with no
    # document named must itself get a Hindi reply when the conversation is
    # in Hindi -- previously this state-machine scaffolding was hardcoded
    # English regardless of the requested/conversation language. Generic
    # requests now start problem-first discovery ("describe_problem")
    # instead of the flat template picker ("selecting") -- see
    # `DraftConversationEngine._start_discovery`.
    engine = DraftConversationEngine()
    memory: dict = {}
    result = asyncio.run(engine.handle_turn("session-wf-1", "मुझे एक दस्तावेज़ बनाना है", "hindi", memory))
    assert result is not None
    assert result.info.stage == "describe_problem"
    assert "बस 1-2 पंक्तियों में बताइए" in result.reply_text
    assert "Which type would you like" not in result.reply_text


def _fill_missing_fields(memory: dict, template, missing_fields: list[str]) -> None:
    """Injects a valid value per field type (`_PHONE_PATTERN`/date-shaped for
    "tel"/"date" fields, a plain placeholder otherwise) -- a generic
    "Test value for X" string fails phone/date validation, gets stripped
    back out by `DraftFieldValidator`, and leaves the field "still missing,"
    which isn't what these tests are checking.
    """
    for field_key in missing_fields:
        draft_field = template.get_field(field_key)
        if draft_field and draft_field.field_type == "tel":
            memory["draft_fields"][field_key] = "9876543210"
        elif draft_field and draft_field.field_type == "date":
            memory["draft_fields"][field_key] = "15/07/2026"
        else:
            memory["draft_fields"][field_key] = f"Test value for {field_key}"


def test_draft_ready_reply_is_localized_and_disclaimer_is_outside_the_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Part 52: the "Here is your draft..." wrapper and edit-hint must be in
    # Hindi, and the AI-generated-draft disclaimer must be shown as a
    # separate trailing chat note -- never inside `full_text`/`sections`
    # (which is exactly what PDF/DOCX/TXT export reads from).
    engine = _engine_with_no_llm_extraction()
    engine.draft_engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    monkeypatch.setattr(engine.draft_engine.drafts, "insert", AsyncMock(return_value="draft-hindi-1"))
    monkeypatch.setattr(engine.draft_engine.versions, "insert", AsyncMock(return_value="version-1"))
    memory: dict = {}
    first = asyncio.run(engine.handle_turn("session-wf-2", "Legal notice bana do", "hindi", memory))
    assert first is not None
    assert memory["draft_language"] == "hindi"
    assert first.info.stage == "collecting"

    _fill_missing_fields(memory, get_template("legal_notice"), first.info.missing_fields)
    # "here you go" deliberately avoids `_CONFIRMATION_WORD_PATTERN` (unlike
    # "yes"/"go ahead"), which would otherwise short-circuit to re-showing
    # collecting-state instead of proceeding to generation -- same technique
    # `test_conversation_engine_end_to_end_collecting_to_preview` uses.
    result = asyncio.run(engine.handle_turn("session-wf-2", "here you go", memory["draft_language"], memory))
    assert result is not None
    assert result.info.stage == "preview"
    assert "यह रहा आपका" in result.reply_text
    assert "मसौदा" in result.reply_text
    # The disclaimer text is still shown in chat (as a trailing aside)...
    # Part 51 "Draft Generation Quality Pass" item 5: the disclaimer must
    # match the draft's own language -- this draft is Hindi, so the Hindi
    # translation is what shows, not the literal English constant.
    hindi_disclaimer = msg("draft_disclaimer", "hindi", DRAFT_DISCLAIMER)
    assert hindi_disclaimer != DRAFT_DISCLAIMER
    assert hindi_disclaimer in result.reply_text
    # ...but never inside the document content itself.
    assert "Disclaimer" not in result.info.sections
    assert hindi_disclaimer not in result.info.full_text


def test_generated_date_section_is_localized_to_the_draft_language(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import date

    engine = _engine_with_no_llm_extraction()
    engine.draft_engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    monkeypatch.setattr(engine.draft_engine.drafts, "insert", AsyncMock(return_value="draft-date-1"))
    monkeypatch.setattr(engine.draft_engine.versions, "insert", AsyncMock(return_value="version-1"))
    memory: dict = {}
    first = asyncio.run(engine.handle_turn("session-wf-3", "Draft police complaint", "english", memory))
    assert first is not None
    _fill_missing_fields(memory, get_template("police_complaint"), first.info.missing_fields)
    result = asyncio.run(engine.handle_turn("session-wf-3", "here you go", "english", memory))
    assert result is not None
    assert result.info.stage == "preview"
    # Part 51: an English draft's date spells out the month name ("13 August
    # 2026") via `format_localized_date`'s own fixed month-name table -- not
    # `date.strftime("%B")`, which resolves from the process's single global
    # OS locale and would silently render the WRONG language's month name on
    # a host whose locale isn't English (the actual root cause Part 42/52
    # previously avoided by forcing numeric DD/MM/YYYY everywhere instead of
    # fixing it). Part 53 "Professional Layout Audit": "Date" is no longer
    # its own section for police_complaint (Complaint category) -- it's
    # folded into the "Signature Block" closing block (Part 56 renamed
    # "Signature" to "Signature Block" for the unified Notice/Complaint
    # skeleton).
    assert format_localized_date(date.today(), "english") in result.info.sections["Signature Block"]
    # The generated date must follow the current calendar date, not a
    # hard-coded month from the date on which this regression test was added.
    assert date.today().strftime("%Y") in result.info.sections["Signature Block"]


def test_validator_returns_localized_message_in_hindi() -> None:
    template = get_template("police_complaint")
    assert template is not None
    issues = DraftFieldValidator().validate(template, {"applicant_mobile": "12345"}, "hindi")
    flagged = {field_key: message for field_key, message in issues}
    assert "applicant_mobile" in flagged
    assert "मान्य 10-अंकीय भारतीय मोबाइल नंबर" in flagged["applicant_mobile"]


@pytest.mark.parametrize(
    "hindi_date",
    ["15 जुलाई 2026", "15 अगस्त 2026", "5 जनवरी 2026"],
)
def test_validator_accepts_localized_hindi_dates(hindi_date: str) -> None:
    # Part 52: "15 जुलाई 2026" and similar Hindi-month-name dates must not be
    # rejected as invalid just because `datetime.strptime`'s `%B`/`%b` only
    # ever recognize English month names.
    template = get_template("cyber_crime_complaint")
    assert template is not None
    issues = DraftFieldValidator().validate(template, {"incident_date": hindi_date}, "hindi")
    flagged_fields = {field_key for field_key, _ in issues}
    assert "incident_date" not in flagged_fields


# ---------------------------------------------------------------------------
# Multi-draft support: starting a second, differently-typed draft mid-flow
# used to silently discard the first one. Now it's parked, not lost, and
# "my drafts"/"switch to <name> draft" can navigate between them.
# ---------------------------------------------------------------------------


def _engine_with_noop_extraction_llm() -> DraftConversationEngine:
    engine = DraftConversationEngine()
    engine.extractor.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    return engine


def test_starting_a_second_draft_mid_collecting_parks_the_first_with_fields_intact() -> None:
    engine = _engine_with_noop_extraction_llm()
    memory: dict = {}
    asyncio.run(
        engine.handle_turn(
            "s1", "I want to write a police complaint. My name is Ramesh Kumar, mobile 9876543210.", "english", memory
        )
    )
    assert memory["draft_template_id"] == "police_complaint"
    assert memory["draft_fields"]["applicant_name"] == "Ramesh Kumar"

    result = asyncio.run(engine.handle_turn("s1", "Actually, I want to file an RTI application.", "english", memory))

    assert result is not None
    assert memory["draft_template_id"] == "rti_application"
    assert len(memory["parked_drafts"]) == 1
    parked = memory["parked_drafts"][0]
    assert parked["draft_template_id"] == "police_complaint"
    assert parked["draft_fields"]["applicant_name"] == "Ramesh Kumar"


def test_list_my_drafts_shows_both_active_and_parked() -> None:
    engine = _engine_with_noop_extraction_llm()
    memory: dict = {}
    asyncio.run(engine.handle_turn("s1", "I want to write a police complaint.", "english", memory))
    asyncio.run(engine.handle_turn("s1", "Actually, I want to file an RTI application.", "english", memory))

    result = asyncio.run(engine.handle_turn("s1", "my drafts", "english", memory))

    assert result is not None
    assert "RTI" in result.reply_text or "Right to Information" in result.reply_text
    assert "Police" in result.reply_text or "police" in result.reply_text.lower()


def test_switch_to_draft_resumes_parked_fields_and_parks_the_current_one() -> None:
    engine = _engine_with_noop_extraction_llm()
    memory: dict = {}
    asyncio.run(
        engine.handle_turn(
            "s1", "I want to write a police complaint. My name is Ramesh Kumar, mobile 9876543210.", "english", memory
        )
    )
    asyncio.run(engine.handle_turn("s1", "Actually, I want to file an RTI application.", "english", memory))
    assert memory["draft_template_id"] == "rti_application"

    result = asyncio.run(engine.handle_turn("s1", "switch to police complaint draft", "english", memory))

    assert result is not None
    assert memory["draft_template_id"] == "police_complaint"
    assert memory["draft_fields"]["applicant_name"] == "Ramesh Kumar"
    # The RTI draft (no fields answered yet) is now the one parked.
    assert len(memory["parked_drafts"]) == 1
    assert memory["parked_drafts"][0]["draft_template_id"] == "rti_application"


def test_switch_to_an_unrecognized_draft_name_falls_through_normally() -> None:
    # No parked draft named "will" -- must not claim to switch to nothing;
    # falls through to ordinary field-extraction handling for this message.
    engine = _engine_with_noop_extraction_llm()
    memory: dict = {}
    asyncio.run(engine.handle_turn("s1", "I want to write a police complaint.", "english", memory))

    result = asyncio.run(engine.handle_turn("s1", "switch to will draft", "english", memory))

    assert result is not None
    assert memory["draft_template_id"] == "police_complaint"
    assert memory.get("parked_drafts") in (None, [])


def test_editing_a_named_non_active_parked_draft_retargets_instead_of_corrupting_the_active_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # BUG-013 (QA session 4, 2026-09-11): with a Cheque Bounce Notice draft
    # active and a General Legal Notice draft parked, "General Legal Notice
    # wale draft mein amount 40,000 se 55,000 kar do." explicitly names the
    # PARKED draft -- but the edit interpreter only ever looked at the
    # ACTIVE template, so it silently applied the amount change to the
    # active Cheque Bounce Notice's own (unrelated) Cheque Amount field
    # instead, live-confirmed via `GET /draft/{id}/versions`. The edit must
    # retarget to the named parked draft, leave the active one untouched,
    # and say which draft it switched to.
    active_template = get_template("cheque_bounce_notice")
    parked_template = get_template("legal_notice")
    assert active_template is not None and parked_template is not None
    memory: dict = {
        "draft_mode": True,
        "draft_stage": "preview",
        "draft_template_id": "cheque_bounce_notice",
        "draft_id": "draft-active-cheque",
        "draft_language": "english",
        "parked_drafts": [
            {
                "draft_stage": "preview",
                "draft_template_id": "legal_notice",
                "draft_fields": {"claim_amount": "40,000"},
                "draft_id": "draft-parked-notice",
                "draft_language": "english",
            }
        ],
    }
    fake_result = DraftGenerateResponse(
        status="complete",
        draft_id="draft-parked-notice",
        template_id="legal_notice",
        template_name=parked_template.name,
        sections={},
        full_text="UPDATED LEGAL NOTICE, AMOUNT 55,000",
        language="english",
    )
    regenerate_mock = AsyncMock(return_value=fake_result)
    engine = DraftConversationEngine()
    monkeypatch.setattr(engine.draft_engine, "regenerate", regenerate_mock)

    result = asyncio.run(
        engine.handle_turn(
            "s1", "General Legal Notice wale draft mein amount 40,000 se 55,000 kar do.", "english", memory
        )
    )

    assert result is not None
    regenerate_mock.assert_awaited_once_with("draft-parked-notice", {"claim_amount": "55,000"})
    assert "UPDATED LEGAL NOTICE" in result.reply_text
    # The previously-active Cheque Bounce Notice draft must now be the
    # parked one, its own fields never touched by this edit.
    assert memory["draft_template_id"] == "legal_notice"
    assert memory["draft_id"] == "draft-parked-notice"
    assert len(memory["parked_drafts"]) == 1
    assert memory["parked_drafts"][0]["draft_template_id"] == "cheque_bounce_notice"
    assert memory["parked_drafts"][0]["draft_id"] == "draft-active-cheque"
