"""Jurisdiction Routing (Phase 2): `app.rag.matter_context.resolve_matter_context`
-- pure-function tests, no Mongo/Redis/LLM involved."""

from app.rag import matter_context as mc

_PROPERTY = "Property Law"


def test_profile_default_used_for_unspecified_ordinary_matter() -> None:
    """MP profile + an ordinary (non-explicit-location) matter uses MP as the
    default, with no clarification needed."""
    result = mc.resolve_matter_context("My landlord is not returning my deposit", {}, "MP", _PROPERTY)
    assert result.state_codes == ["MP"]
    assert result.context_source == mc.CONTEXT_SOURCE_PROFILE
    assert result.needs_clarification is False


def test_explicit_state_wins_over_profile_without_changing_it() -> None:
    """MP profile + an explicit UP property matter uses UP -- the profile
    default itself is only ever read here, never mutated by a chat turn."""
    result = mc.resolve_matter_context(
        "I have a property registration dispute in Uttar Pradesh over my ancestral land", {}, "MP", _PROPERTY,
    )
    assert result.state_codes == ["UP"]
    assert result.context_source == mc.CONTEXT_SOURCE_EXPLICIT


def test_bare_state_mention_is_not_treated_as_matter_location() -> None:
    """"Bihar has a law about X" names a State without stating a location --
    must not be read as an explicit matter location."""
    result = mc.resolve_matter_context("Does Bihar have a law about ancestral property?", {}, "MP", _PROPERTY)
    assert result.context_source != mc.CONTEXT_SOURCE_EXPLICIT
    assert result.state_codes == ["MP"]  # falls through to profile, unaffected by the bare mention


def test_address_block_with_pin_code_is_treated_as_an_explicit_location() -> None:
    # Regression for qa-40q-multilingual-20260921 BUG-07: "21, Andheri West,
    # Mumbai - 400053" names the city right next to its PIN code -- an
    # unambiguous postal address, not a bare topic mention -- but satisfies
    # neither `_EXPLICIT_CITY_LOCATION_RE` (no preposition/postposition next
    # to "Mumbai") nor a mid-sentence bare-mention allowance. Previously this
    # resolved to no location at all, so no Maharashtra filter reached
    # retrieval for the rest of that tenancy matter and an untagged,
    # wrong-jurisdiction chunk (a Hyderabad-specific Act) was cited instead.
    assert mc.detect_state("Mera address: 21, Andheri West, Mumbai - 400053. Mobile: 9876543210.") == "MH"
    assert mc.detect_state("My address is 12 MG Road, Pune - 411001") == "MH"
    assert mc.detect_state("Flat 4B, Sector 62, Noida, 201301") == "UP"
    # A State name directly next to a PIN code, not just a city.
    assert mc.detect_state("21 Civil Lines, Uttar Pradesh - 226001") == "UP"


def test_ambiguous_state_sensitive_matter_asks_clarification() -> None:
    """No profile, no explicit/confirmed State, and the wording itself turns
    on State-specific rules (eviction) -- must ask, not guess or answer with
    an unstated assumption."""
    result = mc.resolve_matter_context("What is the eviction notice period for my rented flat?", {}, None, _PROPERTY)
    assert result.needs_clarification is True
    assert result.state_codes == []
    assert result.clarification_reason is not None


def test_general_property_question_does_not_trigger_unnecessary_clarification() -> None:
    """Objective: "Har general legal question par unnecessary clarification
    mat lagao" -- an ordinary Property Law question with no State-varying
    cue answers directly even with no State on record at all."""
    result = mc.resolve_matter_context("My brother is not giving me my share of our father's property.", {}, None, _PROPERTY)
    assert result.needs_clarification is False


def test_non_state_sensitive_category_never_asks_for_clarification() -> None:
    result = mc.resolve_matter_context("I received a cheque bounce notice", {}, None, "Banking and Criminal Law")
    assert result.needs_clarification is False
    assert result.state_codes == []


def test_followup_retains_confirmed_matter_context() -> None:
    memory = {
        "matter_context": {
            "state_codes": ["UP"], "locality": None, "as_of_date": None,
            "context_source": mc.CONTEXT_SOURCE_EXPLICIT, "legal_category": _PROPERTY,
        }
    }
    result = mc.resolve_matter_context("What about the notice period for this?", memory, "MP", _PROPERTY)
    assert result.state_codes == ["UP"]
    assert result.context_source == mc.CONTEXT_SOURCE_CONFIRMED


def test_new_matter_does_not_inherit_stale_state_override() -> None:
    """A different legal_category signals a genuinely new matter -- the
    earlier UP override must not leak into it; profile default applies."""
    memory = {
        "matter_context": {
            "state_codes": ["UP"], "locality": None, "as_of_date": None,
            "context_source": mc.CONTEXT_SOURCE_EXPLICIT, "legal_category": _PROPERTY,
        }
    }
    result = mc.resolve_matter_context("I got a cheque bounce notice", memory, "MP", "Banking and Criminal Law")
    assert result.state_codes == ["MP"]
    assert result.context_source == mc.CONTEXT_SOURCE_PROFILE


def test_multi_state_matter_is_not_forced_into_one_state() -> None:
    memory = {
        "matter_context": {
            "state_codes": ["UP", "MH"], "locality": None, "as_of_date": None,
            "context_source": mc.CONTEXT_SOURCE_CONFIRMED, "legal_category": _PROPERTY,
        }
    }
    result = mc.resolve_matter_context("continuing this", memory, None, _PROPERTY)
    assert result.state_codes == ["UP", "MH"]


def test_awaiting_clarification_accepts_a_bare_state_reply() -> None:
    """The user's one-line reply to the clarifying question ("Uttar
    Pradesh") is itself the answer -- no locational preposition needed."""
    result = mc.resolve_matter_context("Uttar Pradesh", {}, None, _PROPERTY, awaiting_clarification=True)
    assert result.state_codes == ["UP"]
    assert result.context_source == mc.CONTEXT_SOURCE_EXPLICIT


def test_city_mention_resolves_state_and_ends_the_clarification_loop() -> None:
    """Regression guard: a user answering "which State?" with a city name
    ("Mumbai mein hoon") used to never satisfy `detect_state` (which only
    recognised State/UT names), so `needs_clarification` stayed true forever
    and the chatbot re-asked the same question on every subsequent turn. A
    well-known city must resolve to its State, both as an explicit mention
    ("in Mumbai") and as a bare reply to a just-asked clarification."""
    result = mc.resolve_matter_context(
        "What is the eviction notice period for my rented flat in Mumbai?", {}, None, _PROPERTY,
    )
    assert result.state_codes == ["MH"]
    assert result.needs_clarification is False
    assert result.context_source == mc.CONTEXT_SOURCE_EXPLICIT

    bare_reply = mc.resolve_matter_context("Mumbai", {}, None, _PROPERTY, awaiting_clarification=True)
    assert bare_reply.state_codes == ["MH"]
    assert bare_reply.context_source == mc.CONTEXT_SOURCE_EXPLICIT


def test_extract_as_of_date_rejects_future_dates() -> None:
    from datetime import date

    assert mc.extract_as_of_date("incident on 01/01/2099") is None
    assert mc.extract_as_of_date("incident on 12/03/2021", today=date(2026, 1, 1)) == "2021-03-12"


def test_cache_key_differs_by_state_and_is_none_when_unresolved() -> None:
    unresolved = mc.MatterContext()
    assert unresolved.cache_key() is None
    up = mc.MatterContext(state_codes=["UP"])
    mh = mc.MatterContext(state_codes=["MH"])
    assert up.cache_key() != mh.cache_key()
    assert up.cache_key() is not None
