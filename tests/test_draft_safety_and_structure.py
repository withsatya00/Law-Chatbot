"""Post-Phase-3 hardening, Phase 2 milestone C: template-specific drafting,
unsupported-boilerplate removal, and the provider-failure fallback.

The two documents in the observed transcript, and what was wrong with them
-------------------------------------------------------------------------
**Consumer Complaint.** The user supplied the product, the purchase date, the
amount and the defect. The document rendered none of them as particulars: the
purchase date survived only because the user happened to repeat it inside
their free-text facts, and the amount did not survive at all. What filled the
space instead was five paragraphs of category-generic procedural narration
that would read identically for any complaint about anything.

**Rent / Security Deposit Recovery Notice.** Worse, because it asserted things
nobody said. Served on a named private landlord, it stated:

* "A failure to respond will be treated as an admission of the facts stated
  herein for the purposes of further proceedings." -- a claim about the legal
  effect of another person's silence, with nothing behind it.
* "the sender has suffered financial loss, inconvenience and mental distress,
  for which you are held responsible" -- selected because a deposit amount was
  filled in. The user had described no distress at all.
* "If the above is not complied with within 15 days of receipt of this
  notice..." -- a deadline the user never chose and no statute fixes for a
  security-deposit demand. It came from `... or "15"` in the notice builder.
* "reserves the right to initiate such civil and/or criminal proceedings ...
  and to claim interest, costs and damages" -- a prosecution threat and three
  money claims, none of them asked for.

Every assertion below is against the deterministic path, driven through the
real `LegalDraftEngine`, because that is the path a provider outage puts every
user on -- and the outage was not hypothetical: it is what the transcript shows
happening on every single draft.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.drafting import prohibited_clauses
from app.drafting.engine import _UNPARSABLE_RESPONSE_ERROR, LegalDraftEngine
from app.drafting.templates import get_template
from app.llm.base import LLMResponse
from app.schemas.drafting import DraftPreviewRequest

# Synthetic throughout.
_CONSUMER_FIELDS: dict[str, str] = {
    "applicant_name": "Rahul Sharma",
    "applicant_address": "123, Shastri Nagar, Ghaziabad, Uttar Pradesh - 201002",
    "applicant_mobile": "9812345671",
    "respondent_name": "ABC Electronics Pvt. Ltd.",
    "respondent_address": "45, Nehru Place, New Delhi - 110019",
    "product_or_service": "Smartphone, model QX-200",
    "defect_or_deficiency": "The handset restarts on its own and the battery discharges within two hours.",
    "purchase_date": "2026-08-15",
    "invoice_number": "INV-2026-88421",
    "amount_paid": "24,999",
    "cause_of_action_place": "Ghaziabad, Uttar Pradesh",
    "prior_correspondence": "Complaint ticket 55120 raised on 22 August 2026; store visit on 29 August 2026.",
    "facts": (
        "I purchased the handset from the opposite party on 15 August 2026.\n"
        "Within a week it began restarting on its own.\n"
        "I raised a complaint ticket and visited the store, and the issue was not resolved."
    ),
    "expected_relief": "Replace the handset or refund the price paid.",
    "available_documents": "Tax invoice INV-2026-88421; complaint ticket 55120; photographs of the defect.",
    "place": "Ghaziabad, Uttar Pradesh",
}

_RENT_FIELDS: dict[str, str] = {
    "applicant_name": "Amit Kumar Sharma",
    "applicant_address": "24, Nehru Nagar, Ghaziabad, Uttar Pradesh - 201001",
    "applicant_mobile": "9812345670",
    "respondent_name": "Rajesh Verma",
    "respondent_address": "78, Sector 18, Noida, Uttar Pradesh - 201301",
    "rented_premises_address": "House 112, Gali 4, Shastri Nagar, Ghaziabad - 201002",
    "agreement_date": "2024-04-01",
    "tenancy_start_date": "2024-04-05",
    "possession_date": "2024-04-05",
    "tenancy_end_date": "2026-07-31",
    "monthly_rent": "12,000",
    "security_deposit_amount": "50,000",
    "deposit_payment_evidence": "Paid by NEFT on 1 April 2024, UTR N123456789012345.",
    "deductions_claimed": "The landlord says Rs. 8,000 is withheld for repainting.",
    "prior_demand": "Asked for the refund by WhatsApp on 5 August 2026 and by email on 20 August 2026.",
    "facts": (
        "I vacated the premises on 31 July 2026 and handed over the keys.\n"
        "The security deposit of Rs. 50,000 has not been refunded."
    ),
    "expected_relief": "Refund the security deposit of Rs. 50,000.",
    "available_documents": "Rent agreement dated 1 April 2024; NEFT advice; handover acknowledgement.",
    "place": "Ghaziabad, Uttar Pradesh",
}

_CHEQUE_FIELDS: dict[str, str] = {
    "applicant_name": "Sunita Rao",
    "applicant_address": "9, Model Town, Lucknow, Uttar Pradesh - 226001",
    "applicant_mobile": "9812345672",
    "respondent_name": "Vikram Singh",
    "respondent_address": "14, Aliganj, Lucknow, Uttar Pradesh - 226024",
    "cheque_number": "004512",
    "cheque_amount": "1,50,000",
    "cheque_date": "2026-07-20",
    "bank_name": "State Bank of India, Aliganj Branch",
    "dishonour_reason": "Funds insufficient",
    "dishonour_date": "2026-07-28",
    "facts": "The cheque was issued towards repayment of a loan and was returned unpaid.",
    "place": "Lucknow, Uttar Pradesh",
}


def _unavailable_engine() -> LegalDraftEngine:
    """An engine whose provider is down -- the transcript's actual condition."""
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="The drafting service is currently unavailable. Please try again.",
            model="test",
            provider="test",
            error="The drafting service is currently unavailable. Please try again.",
            error_kind="provider_error",
        )
    )
    return engine


def _render(draft_id: str, fields: dict[str, str], language: str = "english"):
    engine = _unavailable_engine()
    request = DraftPreviewRequest(draft_id=draft_id, language=language, fields=fields)
    response = asyncio.run(engine.preview(request))
    assert response.generation_mode == "deterministic", "these tests are about the fallback path"
    return response


def _body(response: Any) -> str:
    return "\n".join(response.sections.values())


# ---------------------------------------------------------------------------
# Consumer complaint: the collected facts must actually appear, separately
# ---------------------------------------------------------------------------


def test_consumer_complaint_renders_every_collected_particular_separately() -> None:
    response = _render("consumer_complaint", _CONSUMER_FIELDS)
    facts_section = response.sections["Facts of the Case"]
    for label, value in (
        ("Product / Service Concerned", "Smartphone, model QX-200"),
        ("Date of Purchase / Service", "2026-08-15"),
        ("Invoice / Order / Reference Number", "INV-2026-88421"),
        ("Amount Paid (Rs.)", "24,999"),
        ("Place where the Purchase / Service / Cause of Action Arose", "Ghaziabad, Uttar Pradesh"),
    ):
        assert f"{label}: {value}" in facts_section, f"missing particular: {label}"
    assert "Defect in the Goods / Deficiency in the Service" in facts_section
    assert "Earlier Complaints / Correspondence with the Opposite Party" in facts_section


def test_consumer_complaint_keeps_the_particulars_out_of_the_narrative() -> None:
    """A reader expects the particulars as a header block, and the story as
    numbered pleading paragraphs -- not the two interleaved."""
    facts_section = _render("consumer_complaint", _CONSUMER_FIELDS).sections["Facts of the Case"]
    block, _, narrative = facts_section.partition("The facts and circumstances")
    assert "Invoice / Order / Reference Number" in block
    assert "Invoice / Order / Reference Number" not in narrative
    assert narrative.count("That ") >= 3


def test_consumer_complaint_has_annexures_verification_and_a_signature_block() -> None:
    # consumer_complaint is an advocate-register template (Complaint
    # category) -- see test_draft_advocate_register.py -- so this renders
    # "Enclosures", not "Annexures".
    response = _render("consumer_complaint", _CONSUMER_FIELDS)
    assert "Tax invoice INV-2026-88421" in response.sections["Enclosures"]
    assert "Verified that the contents" in response.sections["Verification"]
    signature = response.sections["Signature Block"]
    assert "Place: Ghaziabad, Uttar Pradesh" in signature
    assert "(Rahul Sharma)" in signature


def test_consumer_complaint_prayer_states_the_exact_relief_asked_for() -> None:
    prayer = _render("consumer_complaint", _CONSUMER_FIELDS).sections["Prayer"]
    assert "Replace the handset or refund the price paid." in prayer


def test_an_annexure_section_is_not_invented_when_nothing_was_listed() -> None:
    fields = {key: value for key, value in _CONSUMER_FIELDS.items() if key != "available_documents"}
    response = _render("consumer_complaint", fields)
    assert "Annexures" not in response.sections


# ---------------------------------------------------------------------------
# Rent / security-deposit notice
# ---------------------------------------------------------------------------


def test_rent_notice_renders_the_tenancy_and_deposit_particulars_separately() -> None:
    facts_section = _render("rent_notice", _RENT_FIELDS).sections["Facts of the Case"]
    for label, value in (
        ("Address of Rented Premises", "House 112, Gali 4, Shastri Nagar, Ghaziabad - 201002"),
        ("Date of the Rent Agreement", "2024-04-01"),
        ("Tenancy Start Date", "2024-04-05"),
        ("Date Possession of the Premises was Taken", "2024-04-05"),
        ("Tenancy End / Vacate Date", "2026-07-31"),
        ("Monthly Rent (Rs.)", "12,000"),
        ("Security Deposit Amount (Rs.)", "50,000"),
    ):
        assert f"{label}: {value}" in facts_section, f"missing particular: {label}"
    assert "NEFT on 1 April 2024" in facts_section
    assert "withheld for repainting" in facts_section
    assert "WhatsApp on 5 August 2026" in facts_section


def test_rent_notice_states_no_reply_period_when_the_sender_chose_none() -> None:
    """No statute fixes a reply period for a security-deposit demand. The
    builder used to print fifteen days regardless."""
    template = get_template("rent_notice")
    assert template is not None
    assert template.statutory_response_period_days is None

    body = _body(_render("rent_notice", _RENT_FIELDS))
    assert "15 days" not in body
    assert "within 15 days" not in body
    assert "days of receipt of this notice" not in body


def test_rent_notice_states_the_period_the_sender_did_choose() -> None:
    fields = {**_RENT_FIELDS, "response_deadline_days": "21"}
    body = _body(_render("rent_notice", fields))
    assert "within 21 days of receipt of this notice" in body


def test_rent_notice_does_not_treat_silence_as_an_admission() -> None:
    body = _body(_render("rent_notice", _RENT_FIELDS)).lower()
    assert "treated as an admission" not in body
    assert "deemed to have admitted" not in body


def test_rent_notice_does_not_threaten_criminal_proceedings() -> None:
    body = _body(_render("rent_notice", _RENT_FIELDS)).lower()
    assert "criminal" not in body
    assert "prosecut" not in body


def test_rent_notice_does_not_claim_mental_distress_the_sender_never_described() -> None:
    body = _body(_render("rent_notice", _RENT_FIELDS)).lower()
    assert "mental distress" not in body
    assert "mental agony" not in body


def test_rent_notice_does_not_claim_interest_costs_or_damages() -> None:
    body = _body(_render("rent_notice", _RENT_FIELDS)).lower()
    assert "interest, costs and damages" not in body
    assert "claim interest" not in body


def test_the_hindi_rent_notice_is_free_of_the_same_boilerplate() -> None:
    """The Hindi scaffolding carried its own translations of every one of
    these sentences, so fixing only the English text would have left every
    Hindi user with the original document."""
    body = _body(_render("rent_notice", _RENT_FIELDS, language="hindi"))
    assert "स्वीकृत मान लिया जाएगा" not in body  # silence as admission
    assert "आपराधिक कार्यवाही" not in body  # criminal proceedings
    assert "मानसिक क्लेश" not in body  # mental distress
    assert "ब्याज, व्यय एवं क्षतिपूर्ति" not in body  # interest, costs, damages


def test_the_rent_notice_does_not_ask_a_private_landlord_for_a_diary_number() -> None:
    """A notice with no deadline used to fall into the complaint-shaped tail,
    which asks the recipient to issue a dated acknowledgement and offers to
    cooperate with an inquiry. A landlord runs neither."""
    prayer = _render("rent_notice", _RENT_FIELDS).sections["Prayer"].lower()
    assert "acknowledgement of this document" not in prayer
    assert "cooperation in the inquiry" not in prayer


def test_the_rent_notice_annexures_come_only_from_what_was_listed() -> None:
    response = _render("rent_notice", _RENT_FIELDS)
    assert "Rent agreement dated 1 April 2024" in response.sections["Annexures"]


# ---------------------------------------------------------------------------
# The statutory exception: a s.138 notice
# ---------------------------------------------------------------------------


def test_a_cheque_bounce_notice_states_its_statutory_period_with_a_recorded_basis() -> None:
    """Fifteen days here is proviso (c) to s.138, not a default. The template
    records the provision; the engine may then state the period even though
    the sender did not choose one."""
    template = get_template("cheque_bounce_notice")
    assert template is not None
    assert template.statutory_response_period_days == 15
    assert "section 138" in template.statutory_response_period_basis.lower()
    assert "negotiable_instruments_138_142.json" in template.statutory_response_period_basis

    body = _body(_render("cheque_bounce_notice", _CHEQUE_FIELDS))
    assert "within 15 days of receipt of this notice" in body


def test_a_cheque_bounce_notice_renders_the_cheque_particulars() -> None:
    facts_section = _render("cheque_bounce_notice", _CHEQUE_FIELDS).sections["Facts of the Case"]
    for label, value in (
        ("Cheque Number", "004512"),
        ("Cheque Amount (Rs.)", "1,50,000"),
        ("Drawee Bank Name", "State Bank of India, Aliganj Branch"),
        ("Reason for Dishonour", "Funds insufficient"),
        ("Date of Return Memo", "2026-07-28"),
    ):
        assert f"{label}: {value}" in facts_section, f"missing particular: {label}"


def test_a_template_may_not_record_a_period_without_recording_its_basis() -> None:
    """The loader refuses the half-declaration, so nobody can reintroduce a
    bare number by adding one YAML key."""
    from app.drafting.templates import loader

    raw = {
        "draft_id": "x",
        "name": "X",
        "hindi_name": "X",
        "category": "Notice",
        "description": "d",
        "authority_label": "To,",
        "required_fields": [{"key": "applicant_name", "label": "Name", "hindi_label": "नाम"}],
        "statutory_response_period_days": 15,
    }
    with pytest.raises(ValueError, match="statutory_response_period"):
        loader._build_template(raw, __import__("pathlib").Path("x.yaml"))


# ---------------------------------------------------------------------------
# The clause scanner, which also has to catch what the LLM path writes
# ---------------------------------------------------------------------------


def test_the_scanner_reports_an_admission_clause_whatever_wrote_it() -> None:
    findings = prohibited_clauses.scan(
        {"Facts of the Case": "A failure to respond will be treated as an admission of the facts."},
        _RENT_FIELDS,
    )
    assert [finding["category"] for finding in findings] == ["unsupported_statement:silence_as_admission"]


def test_the_scanner_stands_down_when_the_user_asked_for_it_themselves() -> None:
    """A sender who wrote "I want interest and costs" is not having a claim
    invented for them."""
    fields = {**_RENT_FIELDS, "expected_relief": "Refund the deposit and claim interest and costs."}
    sections = {"Prayer": "You are called upon to pay the deposit and to claim interest and costs."}
    assert prohibited_clauses.scan(sections, fields) == []


def test_the_scanner_never_stands_down_on_silence_as_admission() -> None:
    fields = {**_RENT_FIELDS, "facts": "He told me silence would be treated as an admission."}
    sections = {"Prayer": "Your failure to reply will be treated as an admission."}
    assert prohibited_clauses.scan(sections, fields)


def test_a_template_that_must_state_a_criminal_consequence_may_do_so() -> None:
    """A s.138 notice that does not say a complaint will follow is not a s.138
    notice. The permission is declared on that template and nowhere else."""
    cheque = get_template("cheque_bounce_notice")
    rent = get_template("rent_notice")
    assert cheque is not None and rent is not None
    assert "automatic_criminal_proceedings" in cheque.permitted_statements
    assert rent.permitted_statements == []

    sections = {"Prayer": "Failing payment, a criminal complaint under Section 138 will be filed."}
    fields = {"facts": "The cheque was returned unpaid."}
    assert prohibited_clauses.scan(sections, fields, permitted=tuple(cheque.permitted_statements)) == []
    assert prohibited_clauses.scan(sections, fields, permitted=tuple(rent.permitted_statements))


def test_clause_findings_reach_the_reader_through_the_draft_audit() -> None:
    """They are advisory and surfaced at preview time, not enforced -- a user
    who needs a document is not helped by being refused one."""
    engine = LegalDraftEngine()
    from app.drafting.templates.base import structure_sections_for

    headings = structure_sections_for("Notice")
    body = " ".join(["word"] * 60)
    content = "\n".join(
        f"## {heading}\n"
        + (
            "A failure to respond will be treated as an admission of the facts stated herein. "
            if heading == "Facts of the Case"
            else ""
        )
        + body
        for heading in headings
    )
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content=content, model="t", provider="t"))
    response = asyncio.run(
        engine.preview(DraftPreviewRequest(draft_id="rent_notice", language="english", fields=_RENT_FIELDS))
    )
    assert response.generation_mode == "llm"
    assert any(
        finding["category"] == "unsupported_statement:silence_as_admission"
        for finding in response.audit_findings
    )


# ---------------------------------------------------------------------------
# Provider failure: the three modes named in the task
# ---------------------------------------------------------------------------


def test_an_unavailable_provider_produces_a_disclosed_fallback_not_a_silent_stub() -> None:
    response = _render("rent_notice", _RENT_FIELDS)
    assert response.generated_by_llm is False
    assert response.generation_error is not None
    # Every field the user supplied is still in the document.
    assert "50,000" in _body(response)
    assert "Rajesh Verma" in _body(response)


def test_a_provider_that_raises_is_caught_and_reported() -> None:
    """A timeout surfaces as an exception from some clients and as an error
    response from others. Both must reach the same disclosed fallback."""
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(side_effect=TimeoutError("Read timed out after 120s"))
    response = asyncio.run(
        engine.preview(DraftPreviewRequest(draft_id="rent_notice", language="english", fields=_RENT_FIELDS))
    )
    assert response.generation_mode == "deterministic"
    assert response.generation_error is not None
    assert "timed out" in response.generation_error.lower()


def test_a_malformed_response_is_re_asked_once_before_falling_back() -> None:
    """`ResilientLLMProvider` deliberately does not retry an unusable success,
    because only the engine knows what "usable" means for a draft. The engine
    had never actually implemented that half."""
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(
        return_value=LLMResponse(content="Here is your notice, sir.", model="t", provider="t")
    )
    response = asyncio.run(
        engine.preview(DraftPreviewRequest(draft_id="rent_notice", language="english", fields=_RENT_FIELDS))
    )
    assert engine.llm.chat.await_count == 2, "expected exactly one bounded format re-ask"
    assert response.generation_mode == "deterministic"
    assert response.generation_error == _UNPARSABLE_RESPONSE_ERROR


def test_a_malformed_first_reply_that_the_re_ask_fixes_produces_a_real_draft() -> None:
    from app.drafting.templates.base import structure_sections_for

    headings = structure_sections_for("Notice")
    good = "\n".join(f"## {heading}\n" + " ".join(["word"] * 60) for heading in headings)
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(content="Sure! Here you go.", model="t", provider="t"),
            LLMResponse(content=good, model="t", provider="t"),
        ]
    )
    response = asyncio.run(
        engine.preview(DraftPreviewRequest(draft_id="rent_notice", language="english", fields=_RENT_FIELDS))
    )
    assert response.generation_mode == "llm"
    assert response.generation_error is None


def test_a_provider_error_carrying_a_credential_is_redacted_before_it_travels() -> None:
    """`generation_error` is logged, stored on the draft and returned by
    `POST /draft/generate`."""
    engine = LegalDraftEngine()
    leaked = "POST https://api.example/v1/models:generate?key=AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6 -> 401"
    engine.llm.chat = AsyncMock(
        return_value=LLMResponse(content=leaked, model="t", provider="t", error=leaked, error_kind="auth")
    )
    response = asyncio.run(
        engine.preview(DraftPreviewRequest(draft_id="rent_notice", language="english", fields=_RENT_FIELDS))
    )
    assert response.generation_error is not None
    assert "AIzaSy" not in response.generation_error
    assert "[redacted]" in response.generation_error


def test_the_fallback_is_never_described_as_advocate_approved() -> None:
    """It is a skeleton assembled from the user's own values. The disclaimer
    must still require an advocate to review it."""
    response = _render("rent_notice", _RENT_FIELDS)
    assert "reviewed and approved by a qualified advocate" in response.disclaimer
    body = _body(response).lower()
    assert "approved by" not in body
    assert "vetted" not in body
