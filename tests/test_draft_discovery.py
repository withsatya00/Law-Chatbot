"""Tests for problem-first draft discovery: the guided conversation that
recommends a template instead of listing all ~55 (see
`app/drafting/discovery.py`, `app/drafting/recommendation.py`, and the new
discovery stages in `app/drafting/conversation.py`).
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.drafting import router as drafting_router
from app.drafting import discovery, recommendation
from app.drafting.conversation import DraftConversationEngine
from app.drafting.edit_commands import EditCommandInterpreter
from app.drafting.intent import DraftIntentMatch
from app.drafting.templates import get_template, list_templates
from app.drafting.templates.loader import _build_discovery_metadata, _build_template
from app.llm.base import LLMResponse


def _engine_with_no_llm_extraction() -> DraftConversationEngine:
    engine = DraftConversationEngine()
    engine.extractor.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    return engine


# ---------------------------------------------------------------------------
# 1. Generic requests start discovery, never list all templates.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,language",
    [
        ("Draft generate karni hai", "hinglish"),
        ("Legal document banana hai", "hinglish"),
        ("Application likhni hai", "hinglish"),
        ("Mujhe case ke liye draft chahiye", "hinglish"),
        # Devanagari-script regression: found via live testing that a
        # generic message written in Devanagari (not romanised Hinglish)
        # shared common filler words ("करना"/"है") with one specific
        # template's own trigger phrase and matched it outright --
        # `account_closure_application`'s "खाता बंद करना है" for the
        # unrelated Dogri message below.
        ("मुझे एक दस्तावेज़ बनाना है", "hindi"),
        ("मैंकुं दस्तावेज तैयार करना है", "dogri"),
    ],
)
def test_generic_request_starts_discovery_without_listing_templates(message: str, language: str) -> None:
    engine = DraftConversationEngine()
    memory: dict = {}
    result = asyncio.run(engine.handle_turn("session-generic", message, language, memory))
    assert result is not None
    assert result.info.stage == "describe_problem"
    assert memory["draft_discovery_mode"] is True
    assert memory.get("draft_template_id") is None
    # Not a flat catalogue dump: none of the 55 template display names show
    # up in this first reply (a handful of common English words like
    # "notice"/"agreement" would trivially appear in ANY reply, so the
    # signal checked is the exhaustive per-template name list, not any one
    # generic word).
    long_template_names = [t.name for t in list_templates() if len(t.name.split()) >= 3]
    shown = [name for name in long_template_names if name in result.reply_text]
    assert not shown, f"discovery reply unexpectedly listed templates: {shown}"


# ---------------------------------------------------------------------------
# 2. A direct, named request skips discovery entirely.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_draft_id",
    [
        ("Cheque bounce notice banao", "cheque_bounce_notice"),
        ("Rent agreement draft karo", "rent_agreement"),
        ("Police complaint likho", "police_complaint"),
    ],
)
def test_direct_named_request_skips_discovery(message: str, expected_draft_id: str) -> None:
    engine = _engine_with_no_llm_extraction()
    memory: dict = {}
    result = asyncio.run(engine.handle_turn("session-direct", message, "english", memory))
    assert result is not None
    assert result.info.stage == "collecting"
    assert memory["draft_template_id"] == expected_draft_id
    assert not memory.get("draft_discovery_mode")


# ---------------------------------------------------------------------------
# Typo tolerance: a direct request with an ordinary spelling slip (missing
# letter, doubled letter, adjacent transposition, one wrong letter) must
# still be recognised outright, not fall through to discovery just because
# it isn't spelled exactly like the stored trigger phrase. Found via live
# testing that "renta agrement banao"/"polcie complaint likho" scored 0
# against their own templates purely on the typo. Only templates whose
# trigger phrase has TWO OR MORE independently-typed distinguishing words
# get this tolerance (see `_partial_name_overlap`'s `len(tokens) < 2` guard)
# -- a single-word relaxation was tried and reverted after it re-broke the
# specificity ordering between a generic parent template and a more
# specific child sharing that one word (see the next test).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_draft_id",
    [
        ("cheque bonce notice banao", "cheque_bounce_notice"),  # bounce -> bonce
        ("cheq bounce notice banao", "cheque_bounce_notice"),  # cheque -> cheq
        ("domestik violence complaint likho", "domestic_violence_complaint"),  # domestic -> domestik
    ],
)
def test_typo_tolerant_direct_match(message: str, expected_draft_id: str) -> None:
    from app.drafting.intent import DraftIntentDetector

    match = DraftIntentDetector().detect(message)
    assert match.matched
    assert match.draft_id == expected_draft_id, f"{message!r} -> {match.draft_id!r} (ambiguous={match.ambiguous})"


def test_typo_tolerance_recognises_adjacent_letter_transpositions() -> None:
    # Regression: plain Levenshtein charges an adjacent-letter swap ("money"
    # -> "moeny", "recovery" -> "recoevry") TWO edits, not one, which put an
    # entirely ordinary transposition typo on a short word outside
    # `_typo_tolerant_match`'s distance-1 budget for 4-5 letter words even
    # though a same-cost substitution typo on the same word was recognised.
    # `_levenshtein` now charges one adjacent transposition as a single
    # edit (optimal string alignment distance), matching how people
    # actually mistype.
    from app.drafting.intent import DraftIntentDetector

    match = DraftIntentDetector().detect("recoevry of moeny banao")
    assert match.matched
    assert match.draft_id == "recovery_notice"


def test_typo_tolerance_does_not_override_a_more_specific_sibling_template() -> None:
    # Regression: "police" (police_complaint's one distinguishing word,
    # "complaint" itself being generic) must not outrank
    # `mobile_theft_complaint` just because the message happens to also
    # contain the word "police" while describing a more specific matter.
    from app.drafting.intent import DraftIntentDetector

    match = DraftIntentDetector().detect("Mujhe mobile chori ki police complaint banani hai.")
    assert match.matched
    assert match.draft_id == "mobile_theft_complaint"


def test_typo_tolerance_does_not_match_generic_words_coincidentally() -> None:
    # Regression: "kanuni" (Hindi for "legal", `legal_notice`'s one
    # distinguishing word after generic-token filtering) sits within typo
    # distance of "karni" (an extremely common Hinglish grammatical filler
    # meaning "to do") -- a completely generic message must not resolve
    # to `legal_notice` purely on that coincidental collision.
    from app.drafting.intent import DraftIntentDetector

    match = DraftIntentDetector().detect("Draft generate karni hai")
    assert match.matched
    assert match.draft_id is None
    assert match.ambiguous


def test_typo_tolerance_does_not_hijack_a_correctly_named_template_via_an_unrelated_word() -> None:
    # Regression (QA pass, 2026-09-11): an explicit, unambiguous "legal
    # notice" request wrongly launched `missing_person_report` instead.
    # Root cause: "response period 15 days" (an entirely ordinary phrase,
    # nothing to do with any document) contained "period", which the
    # now-removed distance-2 typo-tolerance budget treated as a plausible
    # misspelling of "person" (both 6 letters, genuinely distance 2 apart --
    # a real collision between two unrelated dictionary words, not a typo).
    # Combined with a genuine, unrelated "missing" elsewhere in the same
    # message ("jo details missing hain" -- "which details are missing"),
    # that coincidental "person" match gave `missing_person_report` two
    # "overlapping" tokens across FOUR of its own trigger phrases, summing
    # to a higher score (0.76) than the message's own exact, correctly
    # spelled "legal notice" match (0.36). `_typo_tolerant_match` is now
    # capped at distance 1 (see its docstring), which "person"/"period"
    # (distance 2) no longer clears.
    from app.drafting.intent import DraftIntentDetector

    message = (
        "In facts par English mein legal notice draft kar do. Sender ka naam Amit Kumar hai, "
        "recipient Rohit Sharma hai, property Test Flat 12, Example Road, Mumbai hai. Jo details "
        "missing hain unke liye placeholders rakhna, response period 15 days rakhna."
    )
    match = DraftIntentDetector().detect(message)
    assert match.matched
    assert match.draft_id == "legal_notice", f"got {match.draft_id!r} (ambiguous={match.ambiguous})"


def test_typo_tolerant_matching_stays_fast_across_the_whole_catalogue() -> None:
    # Regression for a real performance issue found while building typo
    # tolerance: an unbounded fuzzy comparison across every (template,
    # trigger phrase, message token) combination took the intent detector
    # from sub-millisecond to over a second PER MESSAGE. `time.perf_counter`
    # here is a coarse sanity check, not a benchmark -- generous enough to
    # never flake on a slow CI box, tight enough to catch a real regression
    # of the same shape.
    import time

    from app.drafting.intent import DraftIntentDetector

    detector = DraftIntentDetector()
    started = time.perf_counter()
    for _ in range(20):
        detector.detect("Mujhe kal tak ek renta agrement chahiye tha lekin abhi tak nahi bana")
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"20 detect() calls took {elapsed:.2f}s -- expected well under 2s"


# ---------------------------------------------------------------------------
# 3-6, 8, 18. Deterministic recommendation scoring (no LLM involved at all).
# ---------------------------------------------------------------------------


def test_landlord_unpaid_rent_no_notice_recommends_tenancy_notice() -> None:
    profile = {
        "user_role": "landlord",
        "issues": ["rent_default"],
        "desired_reliefs": ["payment", "vacant_possession"],
        "case_stage": "pre_litigation",
        "domain": "property",
    }
    result = recommendation.recommend(profile)
    assert result.tier == "high"
    assert result.candidates[0].draft_id == "tenancy_termination_notice"


def test_tenant_side_never_recommends_landlord_only_template() -> None:
    profile = {"user_role": "tenant", "issues": ["rent_default"], "domain": "property"}
    result = recommendation.recommend(profile)
    assert "tenancy_termination_notice" not in {c.draft_id for c in result.candidates}


def test_notice_already_sent_lowers_the_score_relative_to_pre_litigation() -> None:
    base_profile = {
        "user_role": "landlord", "issues": ["rent_default"], "desired_reliefs": ["payment"],
        "case_stage": "pre_litigation",
    }
    pre_litigation_score = recommendation.recommend(base_profile).candidates[0].score
    notice_sent_profile = {**base_profile, "case_stage": "notice_sent"}
    notice_sent_result = recommendation.recommend(notice_sent_profile)
    assert notice_sent_result.candidates
    assert notice_sent_result.candidates[0].score < pre_litigation_score


def test_order_already_passed_yields_no_match_when_unsupported() -> None:
    # None of the currently-onboarded templates declare "order_passed" as a
    # supported case stage -- recommending one anyway would be exactly the
    # silent unrelated-template mapping this feature must not do.
    result = recommendation.recommend({"user_role": "landlord", "case_stage": "order_passed"})
    assert result.tier == "none"
    assert result.no_match_reason


def test_unknown_matter_does_not_map_to_an_unrelated_draft() -> None:
    result = recommendation.recommend({"user_role": "astronaut", "issues": ["lost_in_space"]})
    assert result.tier == "none"
    assert result.candidates == ()


def test_medium_confidence_returns_at_most_three_candidates() -> None:
    # A domain-only signal (no issue/role/stage) scores every domain-tagged
    # template weakly and about equally -- a good way to force several
    # positive-but-unremarkable scores without hand-picking one template.
    result = recommendation.recommend({"domain": "financial"})
    assert len(result.candidates) <= 3


def test_a_bare_string_issue_is_not_silently_split_into_characters() -> None:
    """`_normalize_text_list` (called on `profile["issues"]`/`["desired_
    reliefs"]`) previously iterated whatever object it was handed with no
    shape check -- a caller passing a single string instead of a list (e.g.
    `"issues": "bail"` instead of `["bail"]`) would silently iterate it
    character-by-character (`str` is iterable) and score against
    `{"b", "a", "i", "l"}` instead of `{"bail"}`, never raising or otherwise
    surfacing as wrong. A list with the same single value must score
    identically to the bare string.
    """
    list_form = recommendation.recommend({"issues": ["bail"]})
    string_form = recommendation.recommend({"issues": "bail"})
    assert [c.draft_id for c in string_form.candidates] == [c.draft_id for c in list_form.candidates]
    assert [c.score for c in string_form.candidates] == [c.score for c in list_form.candidates]


# ---------------------------------------------------------------------------
# 7. Low confidence asks one targeted clarification question.
# ---------------------------------------------------------------------------


def test_low_confidence_asks_a_clarification_question() -> None:
    engine = DraftConversationEngine()
    memory: dict = {}
    session_id = "session-low-conf"

    async def run() -> None:
        await engine.handle_turn(session_id, "Draft generate karni hai", "hinglish", memory)
        # None of these answers carry a role/relief/issue/stage keyword this
        # module's deterministic extractor recognises, so all three
        # standard questions (role, relief, case stage) get "asked" in turn
        # without ever being filled, and this is the reply to the third.
        await engine.handle_turn(session_id, "kuch problem hai mere saath", "hinglish", memory)
        await engine.handle_turn(session_id, "kuch chahiye mujhe", "hinglish", memory)
        await engine.handle_turn(session_id, "pata nahi kya karna hai", "hinglish", memory)
        return await engine.handle_turn(session_id, "abhi bhi pata nahi", "hinglish", memory)

    result = asyncio.run(run())
    assert result is not None
    # One extra free-text clarification, not an immediate "no match" wall.
    assert result.info.stage == "describe_problem"
    assert memory["draft_discovery_extra_clarification_asked"] is True


# ---------------------------------------------------------------------------
# 9. Search is ranked and limited to 5 results by default.
# ---------------------------------------------------------------------------


def test_search_returns_no_more_than_five_ranked_results() -> None:
    results = recommendation.search_templates("agreement")
    assert 1 <= len(results) <= 5
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_exact_match_ranks_first() -> None:
    results = recommendation.search_templates("rent agreement")
    assert results[0].draft_id == "rent_agreement"


def test_search_tolerates_typos() -> None:
    # Found via live testing: "renta agrement" (both words typo'd) scored 0
    # against every template and returned no results at all for a plainly
    # one-typo-per-word search.
    results = recommendation.search_templates("renta agrement")
    assert results
    assert results[0].draft_id == "rent_agreement"


# ---------------------------------------------------------------------------
# 10-11. Loader: existing templates unaffected, invalid metadata rejected.
# ---------------------------------------------------------------------------


def test_existing_templates_load_without_new_metadata() -> None:
    templates = list_templates()
    assert len(templates) == 55
    untagged = [t for t in templates if not t.domain]
    assert untagged, "expected most templates to still have no discovery metadata"
    for template in untagged:
        assert template.document_family == ""
        assert template.supported_issues == []
        assert template.template_status == "production"
        assert template.jurisdiction == {"country": "IN", "level": "central", "states": ["all"]}


def _minimal_raw_template(**overrides: object) -> dict:
    raw: dict = {
        "draft_id": "test_template",
        "name": "Test Template",
        "hindi_name": "टेस्ट",
        "category": "Notice",
        "description": "A test template.",
        "authority_label": "To,",
        "required_fields": [{"key": "applicant_name", "label": "Name", "hindi_label": "नाम"}],
    }
    raw.update(overrides)
    return raw


def test_invalid_case_stage_metadata_is_rejected() -> None:
    from pathlib import Path

    raw = _minimal_raw_template(case_stages=["not_a_real_stage"])
    with pytest.raises(ValueError, match="invalid case_stages"):
        _build_template(raw, Path("bad_template.yaml"))


def test_invalid_document_family_metadata_is_rejected() -> None:
    from pathlib import Path

    raw = _minimal_raw_template(document_family="not_a_real_family")
    with pytest.raises(ValueError, match="invalid document_family"):
        _build_template(raw, Path("bad_template.yaml"))


def test_invalid_template_status_metadata_is_rejected() -> None:
    from pathlib import Path

    raw = _minimal_raw_template(template_status="not_a_real_status")
    with pytest.raises(ValueError, match="invalid template_status"):
        _build_discovery_metadata(raw, Path("bad_template.yaml"))


def test_review_requirements_rejects_unknown_keys() -> None:
    from pathlib import Path

    raw = _minimal_raw_template(review_requirements={"made_up_check": True})
    with pytest.raises(ValueError, match="unknown review_requirements keys"):
        _build_discovery_metadata(raw, Path("bad_template.yaml"))


# ---------------------------------------------------------------------------
# 12-13. Conversation memory resumes discovery and never re-asks a known fact.
# ---------------------------------------------------------------------------


def test_conversation_resumes_discovery_from_persisted_stage() -> None:
    # Simulates a fresh process picking the session back up mid-discovery --
    # `memory` here is exactly the plain dict shape `ConversationMemoryStore`
    # would hand back, with nothing but what an earlier turn already wrote.
    memory = {
        "draft_mode": True,
        "draft_discovery_mode": True,
        "draft_stage": "identify_case_stage",
        "matter_profile": {
            **discovery.default_matter_profile(),
            "user_role": "landlord",
            "issues": ["rent_default"],
            "desired_reliefs": ["payment"],
            "raw_description": "Landlord hoon, tenant rent nahi de raha",
        },
        "draft_discovery_questions_asked": ["user_role", "desired_reliefs"],
        "draft_language": "hinglish",
        "recommended_template_ids": [],
        "recommendation_confidence": None,
        "selected_template_id": None,
        "recommendation_confirmed": False,
        "jurisdiction_confirmed": False,
        "generation_summary_confirmed": False,
        "draft_discovery_extra_clarification_asked": False,
        "draft_fields": {},
        "draft_template_id": None,
        "draft_id": None,
        "draft_paused": False,
    }
    engine = DraftConversationEngine()
    result = asyncio.run(
        engine.handle_turn("session-resume", "abhi tak sirf baat hui hai, koi notice nahi", "hinglish", memory)
    )
    assert result is not None
    # Answering the case-stage question (the one it was waiting on) moves
    # straight to a recommendation -- it does not restart at describe_problem.
    assert result.info.stage == "confirm_template"
    assert memory["matter_profile"]["case_stage"] == "pre_litigation"


def test_previously_collected_facts_are_not_requested_again() -> None:
    engine = DraftConversationEngine()
    memory: dict = {}
    session_id = "session-no-repeat"

    async def run() -> list:
        replies = []
        replies.append(await engine.handle_turn(session_id, "Draft generate karni hai", "hinglish", memory))
        # States role AND relief together in one message.
        replies.append(
            await engine.handle_turn(
                session_id, "Landlord hoon, rent nahi mila, paisa chahiye", "hinglish", memory
            )
        )
        return replies

    replies = asyncio.run(run())
    # The role/relief question text must not appear again -- only the
    # still-unanswered case-stage question should follow.
    last_reply = replies[-1].reply_text
    assert "landlord" not in last_reply.lower() or "tenant" not in last_reply.lower()
    assert memory["draft_stage"] == "identify_case_stage"
    assert memory["matter_profile"]["user_role"] == "landlord"
    assert memory["matter_profile"]["desired_reliefs"] == ["payment"]


def test_discovery_seeds_matter_profile_from_prior_conversation_history() -> None:
    """Regression test for BUG-04 (QA run qa-40q-multilingual-20260921,
    Q25-Q29): `DraftIntentDetector` only looks for a drafting verb in the
    CURRENT message, so a request like "ab notice generate karo" can be the
    first message that ever reaches the drafting engine even after several
    prior turns already described the matter in full. `_start_discovery`
    used to build a blank `matter_profile`, silently discarding all of that.
    It must now seed `raw_description` (and whatever signals it contains)
    from the session's prior user turns, which `ChatService` already stores
    in `memory["messages"]` before the drafting engine is ever invoked.
    """
    engine = DraftConversationEngine()
    session_id = "session-history-seed"
    trigger = "Ab notice generate karo"
    memory: dict = {
        "messages": [
            {
                "role": "user",
                "content": "Main tenant hoon, landlord ne mera security deposit wapas nahi kiya.",
            },
            {"role": "assistant", "content": "..."},
            # `ChatService.answer` appends the current turn before routing
            # into the drafting engine, so it is already the last entry here.
            {"role": "user", "content": trigger},
        ],
    }
    result = asyncio.run(engine.handle_turn(session_id, trigger, "hinglish", memory))
    assert result is not None
    assert result.info.stage == "collecting"
    assert result.info.template_id == "legal_notice"
    assert "security deposit" in " ".join(memory["draft_fields"].values()).lower()
    # The trigger message itself must not be treated as the description --
    # it carries no facts of its own.
    assert trigger not in " ".join(memory["draft_fields"].values())


def test_generic_notice_command_directly_selects_general_legal_notice() -> None:
    from app.drafting.intent import DraftIntentDetector

    match = DraftIntentDetector().detect(
        "Landlord ka address: 14, Bandra East, Mumbai - 400051. Ab notice generate karo."
    )

    assert match.matched
    assert match.draft_id == "legal_notice"
    assert not match.ambiguous


def test_repayment_message_is_not_fuzzy_routed_to_account_closure() -> None:
    from app.drafting.intent import DraftIntentDetector

    match = DraftIntentDetector().detect(
        "Mera naam Sana Khan hai. Maine friend ko Rs 20,000 bank transfer se diye the. "
        "Ek short Hindi repayment message draft karo."
    )

    assert match.matched
    assert match.draft_id != "account_closure_application"


def test_explicit_notice_command_outranks_conversation_memory_wording() -> None:
    from app.intent.classifier import ConversationIntentClassifier

    match = ConversationIntentClassifier().classify(
        "In facts par Hindi mein polite legal demand notice draft karo. "
        "Missing address aur mobile ke liye pehle question puchho. Koi fact invent mat karna.",
        {"messages": [{"role": "assistant", "content": "Earlier legal answer"}]},
    )

    assert match.intent == "Draft Generation"


def test_facts_given_before_template_selection_populate_the_first_draft_fields() -> None:
    """Regression test for BUG-04: once discovery lands on a template, facts
    the user already supplied while describing the problem -- before any
    template was chosen -- must be pulled into `draft_fields` rather than
    asked for again (previously `collected_summary`/`draft_fields` stayed
    empty for the entire discovery ladder; see
    `_enter_field_collection_from_discovery`).
    """
    engine = _engine_with_no_llm_extraction()
    memory: dict = {
        "matter_profile": {
            **discovery.default_matter_profile(),
            "raw_description": (
                "Landlord ne mera security deposit Rs 45,000 wapas nahi kiya. "
                "Mera mobile number 9876543210 hai."
            ),
        },
        "draft_fields": {},
    }
    template = get_template("rent_notice")
    assert template is not None
    result = asyncio.run(
        engine._enter_field_collection_from_discovery(session_id="session-seed-fields", language="hinglish", memory=memory, template=template)
    )
    assert result is not None
    assert memory["draft_fields"].get("security_deposit_amount") == "45,000"
    assert memory["draft_fields"].get("applicant_mobile") == "9876543210"


# ---------------------------------------------------------------------------
# 14. Summary confirmation happens before generation (discovery-originated
#     drafts only -- a direct draft is unaffected, checked as a regression
#     guard alongside it).
# ---------------------------------------------------------------------------


def test_discovery_draft_requires_summary_confirmation_before_generating() -> None:
    engine = _engine_with_no_llm_extraction()
    memory: dict = {}
    session_id = "session-summary-gate"

    async def run():
        await engine.handle_turn(session_id, "Draft generate karni hai", "hinglish", memory)
        await engine.handle_turn(
            session_id, "Landlord hoon, tenant ne rent nahi diya, abhi koi notice nahi bheja", "hinglish", memory
        )
        await engine.handle_turn(session_id, "paisa chahiye", "hinglish", memory)
        await engine.handle_turn(session_id, "continue", "hinglish", memory)
        await engine.handle_turn(session_id, "Lucknow, Uttar Pradesh", "hinglish", memory)
        template = get_template(memory["draft_template_id"])
        assert template is not None
        # Pre-fill every required field directly -- isolates the
        # summary-confirmation gate under test from the separate concern of
        # whether free-text label extraction parses a given message
        # correctly (covered by `test_drafting.py`'s own extractor tests).
        # A `tel` field needs a value `DraftFieldValidator` actually accepts,
        # or this never reaches the summary gate at all -- it comes back
        # asking to correct the (deliberately invalid) placeholder instead.
        memory["draft_fields"] = {
            f.key: ("9876543210" if f.field_type == "tel" else f"value-{f.key}") for f in template.required_fields
        }
        return await engine.handle_turn(session_id, "confirm", "hinglish", memory)

    result = asyncio.run(run())
    assert result is not None
    assert memory["draft_stage"] == "confirm_summary"
    assert memory["generation_summary_confirmed"] is False
    assert "Tenancy Termination Notice" in result.reply_text


def test_direct_draft_is_not_gated_by_a_summary_confirmation() -> None:
    engine = _engine_with_no_llm_extraction()
    memory: dict = {}
    session_id = "session-direct-no-gate"

    async def run():
        template = get_template("cheque_bounce_notice")
        assert template is not None
        await engine.handle_turn(session_id, "Cheque bounce notice banao", "english", memory)
        all_fields = ", ".join(f"{f.label}: value-{f.key}" for f in template.required_fields)
        return await engine.handle_turn(session_id, all_fields, "english", memory)

    result = asyncio.run(run())
    assert result is not None
    # Straight to preview/generation -- no confirm_summary detour.
    assert memory["draft_stage"] != "confirm_summary"


# ---------------------------------------------------------------------------
# 15. Changing the recommended draft clears only incompatible fields.
# ---------------------------------------------------------------------------


def test_changing_draft_type_clears_only_incompatible_fields() -> None:
    engine = DraftConversationEngine()
    profile = discovery.default_matter_profile()
    # Jurisdiction already known, so confirming the new template goes
    # straight to field collection in this one turn -- isolates the
    # field-clearing behaviour under test from the separate
    # collect_jurisdiction step.
    profile["jurisdiction"] = {"country": "IN", "state": "Uttar Pradesh", "district": "Lucknow"}
    memory: dict = {
        "draft_mode": True,
        "draft_discovery_mode": True,
        "draft_stage": "confirm_template",
        "matter_profile": profile,
        "recommended_template_ids": ["tenancy_termination_notice", "rent_agreement"],
        "draft_fields": {
            # Shared key between tenancy_termination_notice and
            # rent_agreement -- must survive the switch.
            "rented_premises_address": "45 MG Road",
            # Only meaningful for tenancy_termination_notice -- must be
            # dropped once the user switches to rent_agreement.
            "respondent_name": "Suresh Verma",
        },
        "draft_language": "english",
        "draft_discovery_browse_step": None,
        "recommendation_confidence": None,
        "selected_template_id": None,
        "recommendation_confirmed": False,
        "jurisdiction_confirmed": False,
        "generation_summary_confirmed": False,
        "draft_template_id": None,
        "draft_id": None,
        "draft_paused": False,
    }
    result = asyncio.run(engine.handle_turn("session-switch", "rent agreement", "english", memory))
    assert result is not None
    assert memory["draft_template_id"] == "rent_agreement"
    assert memory["draft_fields"].get("rented_premises_address") == "45 MG Road"
    assert "respondent_name" not in memory["draft_fields"]


# ---------------------------------------------------------------------------
# Regression: a direct, named request typed WHILE a recommendation is being
# confirmed must switch straight to that template, not get treated as an
# unrecognised answer to "which of these did you mean?". Found via live
# testing: "Cheque bounce notice banao" sent right after a tenancy-notice
# recommendation was shown came back "I didn't catch that."
# ---------------------------------------------------------------------------


def test_direct_request_at_confirm_template_stage_switches_immediately() -> None:
    engine = DraftConversationEngine()
    memory: dict = {}
    session_id = "session-switch-at-confirm"

    async def run():
        await engine.handle_turn(session_id, "Draft generate karni hai", "hinglish", memory)
        await engine.handle_turn(
            session_id, "Landlord hoon, tenant ne rent nahi diya, abhi koi notice nahi bheja", "hinglish", memory
        )
        await engine.handle_turn(session_id, "paisa chahiye", "hinglish", memory)
        assert memory["draft_stage"] == "confirm_template"
        return await engine.handle_turn(session_id, "Cheque bounce notice banao", "hinglish", memory)

    result = asyncio.run(run())
    assert result is not None
    assert memory["draft_stage"] == "collecting"
    assert memory["draft_template_id"] == "cheque_bounce_notice"
    assert memory["draft_discovery_mode"] is False


def test_confirm_template_stage_survives_a_direct_request_for_a_stale_template_id() -> None:
    """`_continue_confirm_template` is declared to always return a
    `DraftTurnResult`, never `None` -- but it used to hand back whatever
    `_start_collecting` returned unchecked, and `_start_collecting` itself
    returns `None` when the matched `draft_id` has no backing template
    (e.g. the intent detector's own catalogue drifted from
    `app/drafting/templates/`, or a template was renamed/removed). Every
    OTHER call site of `_start_collecting` in this file already guards
    against that `None` (see `_continue_selecting`'s identical check); this
    call site didn't, so it would have returned `None` here -- breaking its
    own return-type contract and, more importantly, whatever caller expects
    a real `DraftTurnResult` back from `handle_turn`.
    """
    engine = DraftConversationEngine()
    engine.intent_detector.detect = lambda _message: DraftIntentMatch(  # type: ignore[method-assign]
        matched=True, draft_id="this_template_id_does_not_exist", confidence=1.0,
    )
    memory: dict = {
        "draft_mode": True,
        "draft_discovery_mode": True,
        "draft_stage": "confirm_template",
        "recommended_template_ids": [],
    }

    result = asyncio.run(engine.handle_turn("session-stale-template", "xyz", "english", memory))

    assert result is not None
    assert result.info.stage == "confirm_template"


# ---------------------------------------------------------------------------
# Regression: jurisdiction parsing must not leave the state's own text
# duplicated inside `district` when the reply has no comma. Found via live
# testing: "UP aur maharanjang" produced district == the whole original
# string, so the pre-generation summary showed "UP aur maharanjang, Uttar
# Pradesh".
# ---------------------------------------------------------------------------


def test_jurisdiction_parsing_does_not_duplicate_the_state_into_the_district() -> None:
    parsed = discovery.parse_jurisdiction_reply("UP aur maharanjang")
    assert parsed["state"] == "Uttar Pradesh"
    assert parsed["district"] is not None
    assert "uttar pradesh" not in parsed["district"].lower()
    assert parsed["district"] != "UP aur maharanjang"


def test_jurisdiction_parsing_handles_comma_separated_district_and_state() -> None:
    parsed = discovery.parse_jurisdiction_reply("Lucknow, Uttar Pradesh")
    assert parsed == {"country": "IN", "state": "Uttar Pradesh", "district": "Lucknow", "verified": True}


# ---------------------------------------------------------------------------
# Regression: the small per-page running header (duplicating the document's
# own title in tiny type) was reported as clutter -- must be suppressed for
# every template now, in both PDF and DOCX, not only the flowing-letter set.
# ---------------------------------------------------------------------------


def test_pdf_export_has_no_running_header_for_any_template() -> None:
    from app.drafting.export import _LAYOUT, ExportOptions, PdfDraftExporter

    html = PdfDraftExporter()._build_html(
        "Tenancy Termination Notice", {"Recipient": "Amit Verma"}, _LAYOUT,
        "english", ExportOptions(include_header_footer=True),
    )
    assert "@top-center { content: ''" in html
    assert "@bottom-center { content: 'Page '" in html


# ---------------------------------------------------------------------------
# Multi-language regression pass. Found via live testing: the discovery
# flow's system-scaffolding text (the generic prompt and the role/relief/
# stage/jurisdiction questions) was Hindi/Hinglish-only, falling back to
# English for every other language the rest of this app already covers
# (Tamil/Telugu/Kannada/Bengali/Malayalam/Marathi/Gujarati/Punjabi/Odia/
# Urdu -- the same 10-language set `wrapper_messages.py`'s OTHER entries
# already had). Separately, role/issue/relief/case-stage extraction from the
# user's own free-text answer only recognised English/Hindi/Hinglish
# phrasing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["tamil", "telugu", "kannada", "bengali", "urdu"])
def test_discovery_generic_prompt_is_localized(language: str) -> None:
    engine = DraftConversationEngine()
    memory: dict = {}
    result = asyncio.run(engine.handle_turn("session-lang", "Draft generate karni hai", language, memory))
    assert result is not None
    from app.drafting.wrapper_messages import msg

    localized = msg("discovery_generic_prompt", language, "__ENGLISH_FALLBACK__")
    assert localized != "__ENGLISH_FALLBACK__", f"no {language} translation registered"
    assert result.reply_text == localized


# ---------------------------------------------------------------------------
# Discovery-extraction typo tolerance. Found via live testing: unlike
# `intent.py`'s template-name matching, `discovery.py`'s role/issue/relief/
# case-stage keyword matching had ZERO typo tolerance -- worse, a typo'd
# self-identification ("landlrod hoon") didn't just fail to match, it fell
# through to the "other role mentioned" fallback and confidently returned
# the WRONG party ("tenant"), which is worse than not answering at all.
# ---------------------------------------------------------------------------


def test_role_extraction_survives_a_typo_in_self_identification() -> None:
    role = discovery.extract_role("landlrod hoon, tenant rent nai de rha")
    assert role == "landlord"


def test_case_stage_extraction_survives_a_typo() -> None:
    assert discovery.extract_case_stage("notic bhej diya maine") == "notice_sent"


def test_issue_extraction_survives_a_typo() -> None:
    assert "cheque_bounce" in discovery.extract_issues("cheque bonce ho gaya")


def test_role_extraction_understands_tamil_self_identification() -> None:
    role = discovery.extract_role("நான் வீட்டு உரிமையாளர், வாடகைதாரர் வாடகை தரவில்லை")
    assert role == "landlord"


def test_role_extraction_understands_bengali_self_identification() -> None:
    role = discovery.extract_role("আমি বাড়িওয়ালা, ভাড়াটিয়া ভাড়া দেয়নি")
    assert role == "landlord"


def test_canonical_term_is_understood_even_without_native_phrasing() -> None:
    # A language/phrasing this module has no dedicated keyword entry for
    # (e.g. Kannada relief phrasing) should still be understood if the user
    # echoes back the plain English canonical term shown in the discovery
    # question's own option list ("payment / vacant possession").
    assert discovery.extract_reliefs("payment") == ["payment"]
    assert discovery.extract_case_stage("order passed") == "order_passed"


def test_jurisdiction_parsing_understands_devanagari_state_names() -> None:
    parsed = discovery.parse_jurisdiction_reply("जयपुर, राजस्थान")
    assert parsed["state"] == "Rajasthan"
    assert parsed["district"] == "जयपुर"
    assert parsed["verified"] is True


# ---------------------------------------------------------------------------
# Multilingual issue/relief/case-stage extraction and Devanagari edit
# commands -- second multi-language pass, found via live testing.
# ---------------------------------------------------------------------------


def test_recommendation_description_is_localized_for_onboarded_templates() -> None:
    # `discovery_recommend_one`'s "{usage}" half used to always be
    # `template.description` verbatim (English), so even a fully Tamil
    # reply ended with one leftover English sentence.
    from app.drafting.description_translations import localized_description

    template = get_template("tenancy_termination_notice")
    assert template is not None
    tamil_description = localized_description(template, "tamil")
    assert tamil_description != template.description
    assert "வாடகைதாரர்" in tamil_description  # "tenant" -- confirms real Tamil, not a fallback


def test_recommendation_description_falls_back_to_english_when_uncovered() -> None:
    from app.drafting.description_translations import localized_description

    template = get_template("legal_notice")  # not one of the 8 onboarded templates
    assert template is not None
    assert localized_description(template, "tamil") == template.description


def test_full_tamil_discovery_conversation_extracts_every_signal() -> None:
    # Every slot (role/issue/relief/case_stage) filled from native Tamil
    # text alone, in the order a real conversation would supply them.
    profile = discovery.default_matter_profile()
    discovery.apply_signals_to_profile(
        profile, "நான் வீட்டு உரிமையாளர், வாடகைதாரர் 3 மாதமாக வாடகை தரவில்லை"
    )
    discovery.apply_signals_to_profile(profile, "பணம் வேண்டும்")
    discovery.apply_signals_to_profile(profile, "இதுவரை எதுவும் நடக்கவில்லை")
    assert profile["user_role"] == "landlord"
    assert profile["issues"] == ["rent_default"]
    assert profile["desired_reliefs"] == ["payment"]
    assert profile["case_stage"] == "pre_litigation"


@pytest.mark.parametrize(
    "text,expected_issue",
    [
        ("వినియోగదారుడు వస్తువు లోపభూయిష్ట వస్తువు తీసుకున్నాడు", "defective_goods"),  # Telugu
        ("চেক বাউন্স হয়েছে", "cheque_bounce"),  # Bengali
        ("ಚೆಕ್ ಬೌನ್ಸ್ ಆಗಿದೆ", "cheque_bounce"),  # Kannada
    ],
)
def test_issue_extraction_understands_additional_languages(text: str, expected_issue: str) -> None:
    assert expected_issue in discovery.extract_issues(text)


def test_devanagari_edit_command_resolves_field_and_value() -> None:
    # Found via live testing: the field-name match (via the field's own
    # `hindi_label`) already worked, but the VALUE was always `None` for a
    # genuinely Devanagari-script command -- only romanised Hinglish verb
    # endings ("kar do", "badlo") were recognised.
    template = get_template("tenancy_termination_notice")
    assert template is not None
    interp = EditCommandInterpreter()
    command = interp.interpret("मोबाइल नंबर बदलो 9988776655", template, "hindi")
    assert command.action == "replace_field"
    assert command.target_field == "applicant_mobile"
    assert command.new_value == "9988776655"


def test_devanagari_edit_command_does_not_leak_a_stray_vowel_sign() -> None:
    # Regression for a specific alternation-ordering bug: "बदल" (a literal
    # prefix of "बदलो") matching first left the combining vowel sign "ो"
    # attached to the captured value ("ो 9988776655" instead of
    # "9988776655").
    template = get_template("tenancy_termination_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret("मोबाइल नंबर सही करो 9988776655", template, "hindi")
    assert command.new_value == "9988776655"
    assert "ो" not in (command.new_value or "")[:1]


def test_english_field_label_is_matchable_without_an_authored_synonym() -> None:
    # `tenancy_termination_notice`'s own `field_synonyms` only covers
    # tenant/premises/place -- "Mobile Number" and "Applicant Address" have
    # no authored synonym, so only the field's own label made them
    # reachable. Regression for the pre-existing edit interpreter never
    # trying english/hinglish plain labels at all (only the ten
    # single-language-label languages got this fallback).
    template = get_template("tenancy_termination_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret("change the mobile number to 9988776655", template, "english")
    assert command.target_field == "applicant_mobile"
    assert command.new_value == "9988776655"


def test_edit_command_field_label_survives_a_typo() -> None:
    # Found via live testing: the target field resolved correctly on the
    # very first pass (the label match itself was already made fuzzy), but
    # `new_value` stayed `None` -- the downstream value-extraction regex
    # searched for the correctly-SPELLED label, which never literally
    # appears in a message that misspelled the field name too.
    template = get_template("tenancy_termination_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret(
        "change the moblie number to 9988776655", template, "english"
    )
    assert command.action == "replace_field"
    assert command.target_field == "applicant_mobile"
    assert command.new_value == "9988776655"

    command2 = EditCommandInterpreter().interpret(
        "update the aplicant address to Delhi", template, "english"
    )
    assert command2.target_field == "applicant_address"
    assert command2.new_value == "Delhi"


def test_edit_command_typo_tolerance_does_not_collide_with_a_common_field() -> None:
    # Regression: "place" (a field on nearly every template) sat within
    # fuzzy-match distance of "police" purely because the threshold used
    # was based on the LONGER of the two words ("police", 6 letters) rather
    # than the shorter, more collision-prone one ("place", 5 letters) --
    # "change the police station to Hazratganj" against a template with no
    # `police_station` field at all incorrectly resolved to the "place"
    # field.
    template = get_template("tenancy_termination_notice")
    assert template is not None
    command = EditCommandInterpreter().interpret(
        "change the police station to Hazratganj", template, "english"
    )
    assert command.action == "unknown"


# ---------------------------------------------------------------------------
# 17. Hindi/Hinglish aliases resolve correctly.
# ---------------------------------------------------------------------------


def test_hindi_and_hinglish_aliases_resolve_via_exact_alias_match() -> None:
    template = get_template("cheque_bounce_notice")
    assert template is not None
    assert "चेक बाउंस नोटिस" in template.aliases["hindi"]
    assert "cheque bounce ka notice banana hai" in template.aliases["hinglish"]

    hindi_result = recommendation.recommend({"raw_description": "चेक बाउंस नोटिस"})
    assert hindi_result.candidates
    assert hindi_result.candidates[0].draft_id == "cheque_bounce_notice"

    hinglish_result = recommendation.recommend({"raw_description": "cheque bounce ka notice banana hai"})
    assert hinglish_result.candidates
    assert hinglish_result.candidates[0].draft_id == "cheque_bounce_notice"


# ---------------------------------------------------------------------------
# 19. API category/recommendation responses validate against their schemas.
# ---------------------------------------------------------------------------


@pytest.fixture()
def drafting_api_client() -> TestClient:
    app = FastAPI()
    app.include_router(drafting_router)
    return TestClient(app)


def test_draft_categories_endpoint_returns_valid_schema(drafting_api_client: TestClient) -> None:
    response = drafting_api_client.get("/draft-categories")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(discovery.TOP_LEVEL_CATEGORIES)
    for entry in body:
        assert {"category_id", "name", "template_count"} <= entry.keys()


def test_draft_subcategories_endpoint_returns_valid_schema(drafting_api_client: TestClient) -> None:
    response = drafting_api_client.get("/draft-subcategories", params={"domain": "property"})
    assert response.status_code == 200
    body = response.json()
    assert all(entry["domain"] == "property" for entry in body)
    assert any(entry["subcategory_id"] == "tenancy" for entry in body)


def test_draft_templates_filtered_by_domain(drafting_api_client: TestClient) -> None:
    response = drafting_api_client.get("/draft-templates", params={"domain": "property"})
    assert response.status_code == 200
    body = response.json()
    assert body
    assert all(entry["domain"] == "property" for entry in body)


def test_draft_search_endpoint_limits_and_ranks(drafting_api_client: TestClient) -> None:
    response = drafting_api_client.get("/draft-templates/search", params={"q": "agreement"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "agreement"
    assert 1 <= len(body["results"]) <= 5


def test_draft_recommend_endpoint_returns_valid_schema(drafting_api_client: TestClient) -> None:
    response = drafting_api_client.post(
        "/draft/recommend",
        json={
            "profile": {
                "user_role": "landlord", "issues": ["rent_default"],
                "desired_reliefs": ["payment", "vacant_possession"], "case_stage": "pre_litigation",
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tier"] == "high"
    assert body["candidates"][0]["draft_id"] == "tenancy_termination_notice"
    assert 0.0 <= body["candidates"][0]["confidence"] <= 1.0


def test_draft_recommend_endpoint_restrict_ids(drafting_api_client: TestClient) -> None:
    response = drafting_api_client.post(
        "/draft/recommend",
        json={"profile": {"raw_description": "cheque bounce notice"}, "restrict_ids": ["police_complaint"]},
    )
    assert response.status_code == 200
    body = response.json()
    # Restricted to a template that doesn't match the profile at all --
    # excluded outright rather than force-matched.
    assert body["tier"] == "none"


def test_possessive_hinglish_address_labels_fill_the_right_party() -> None:
    from app.drafting.field_extraction import DraftFieldExtractor

    keys = get_template("demand_notice").field_keys()
    extractor = DraftFieldExtractor()
    mine = extractor._extract_with_regex(
        "Mera address: 21, Andheri West, Mumbai - 400053. Mobile: 9876543210. Landlord ka address abhi available nahi hai.", keys
    )
    assert mine["applicant_address"].startswith("21, Andheri West")
    assert "respondent_address" not in mine
    theirs = extractor._extract_with_regex("Landlord ka address: 14, Bandra East, Mumbai - 400051. Ab notice generate karo.", keys)
    assert theirs["respondent_address"].startswith("14, Bandra East")
    assert "applicant_address" not in theirs


def test_labelled_bare_address_and_fir_relief_are_extracted_without_an_llm() -> None:
    from app.drafting.field_extraction import DraftFieldExtractor

    keys = get_template("police_complaint").field_keys()
    extractor = DraftFieldExtractor()
    block = extractor._extract_with_regex(
        "Applicant: Ankit Verma. Address: 45, Saraswati Vihar, Sector 16, Noida - 201301. Mobile: 9123456789. "
        "Police station: Sector 39, Noida. Place: Noida.", keys,
    )
    assert block["applicant_address"].startswith("45, Saraswati Vihar")
    command = extractor._extract_with_regex("SHO ko FIR registration ke liye Hindi complaint draft karo.", keys)
    assert "FIR" in command["expected_relief"]


def test_read_back_requests_are_not_edits_but_real_edits_still_are() -> None:
    from app.drafting.conversation import _is_recall_request

    assert _is_recall_request("Mere tenancy matter ka summary do. Sana ya loan matter ka koi fact include mat karna.")
    assert _is_recall_request("Ab corrected deposit amount, disputed deduction, landlord ka naam aur keys handover date batao.")
    assert not _is_recall_request("Amount badal kar 50000 kar do")
    assert not _is_recall_request("Add landlord phone 9999999999")
