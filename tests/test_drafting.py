import asyncio
from datetime import date
from unittest.mock import AsyncMock

import pytest

from app.core.constants import DRAFT_DISCLAIMER
from app.core.exceptions import BadRequestError, NotFoundError
from app.drafting.engine import (
    _WORDS_PER_BODY_PAGE_INDIC,
    _WORDS_PER_BODY_PAGE_LATIN,
    MINIMUM_BODY_PAGES,
    LegalDraftEngine,
    _target_word_count,
    _word_count,
    estimated_page_count,
)
from app.drafting.glossary import LegalTermsLibrary
from app.drafting.heading_translations import translated_heading
from app.drafting.localized_dates import format_localized_date
from app.drafting.safety import DraftSafetyGuard
from app.drafting.templates import get_template, list_templates
from app.drafting.templates.base import structure_sections_for
from app.drafting.title_translations import localized_title
from app.llm.base import LLMResponse
from app.schemas.drafting import DraftGenerateRequest, DraftPreviewRequest


def test_template_registry_is_well_formed() -> None:
    # Part 57 "Drafting Lifecycle Redesign": grew from 11 templates to 55
    # (44 new templates across 10 categories, incl. the new "Contract"
    # category) -- a lower bound rather than an exact count, since adding
    # further templates later shouldn't require touching this assertion.
    templates = list_templates()
    assert len(templates) >= 55
    valid_categories = {"Notice", "Complaint", "Affidavit", "Application", "Contract"}
    seen_ids: set[str] = set()
    for template in templates:
        assert template.draft_id not in seen_ids
        seen_ids.add(template.draft_id)
        assert template.name
        assert template.hindi_name
        assert template.authority_label
        assert template.category in valid_categories, f"{template.draft_id} has unrecognized category {template.category!r}"
        assert template.required_fields, f"{template.draft_id} must have at least one required field"
        assert len(template.field_keys()) == len(template.all_fields()), "duplicate field keys within a template"
        # Every template's category must resolve to a real, non-empty section
        # skeleton -- guards against a typo'd `category` silently falling
        # back to UNIFIED_SECTIONS via `structure_sections_for`'s fallback.
        assert structure_sections_for(template.category)


def test_get_template_returns_none_for_unknown_id() -> None:
    assert get_template("does-not-exist") is None
    assert get_template("rti_application") is not None


def test_hindi_affidavit_normalization_removes_mixed_english_oath_phrase() -> None:
    engine = LegalDraftEngine()
    template = get_template("lost_document_affidavit")
    assert template is not None
    sections = {
        "Deponent Details": (
            "मैं, अमित कुमार, एतद्द्वारा solemnly affirm (सत्यनिष्ठा से प्रतिज्ञान) करता हूँ।"
        )
    }

    engine._normalize_rendered_semantics(sections, template, {}, "hindi")

    assert "solemnly affirm" not in sections["Deponent Details"]
    assert "शपथपूर्वक सत्यनिष्ठा से प्रतिज्ञान" in sections["Deponent Details"]


def test_self_notice_normalization_removes_ambiguous_advocate_wording() -> None:
    engine = LegalDraftEngine()
    template = get_template("legal_notice")
    assert template is not None
    sections = {
        "Introduction": (
            "मैं अपने अधिवक्ता के माध्यम से (अथवा स्वयं, यदि लागू हो) आपको यह नोटिस भेज रहा हूँ।"
        )
    }

    engine._normalize_rendered_semantics(sections, template, {}, "hindi")

    assert "अधिवक्ता" not in sections["Introduction"]
    assert "अथवा" not in sections["Introduction"]
    assert "मैं स्वयं आपको" in sections["Introduction"]


def test_advocate_notice_normalization_preserves_advocate_wording() -> None:
    engine = LegalDraftEngine()
    template = get_template("legal_notice")
    assert template is not None
    sections = {"Introduction": "मैं अपने अधिवक्ता के माध्यम से आपको यह नोटिस भेज रहा हूँ।"}

    engine._normalize_rendered_semantics(
        sections, template, {"representation_mode": "advocate"}, "hindi"
    )

    assert "अपने अधिवक्ता के माध्यम से" in sections["Introduction"]


def test_missing_required_fields_returns_needs_more_info() -> None:
    engine = LegalDraftEngine()
    request = DraftPreviewRequest(draft_id="rti_application", fields={"applicant_name": "Ramesh Kumar"})
    response = asyncio.run(engine.preview(request))
    assert response.status == "needs_more_info"
    assert "public_authority_name" in response.missing_fields
    assert "information_sought" in response.missing_fields
    assert response.follow_up_questions


def test_legal_notice_subject_line_is_not_a_raw_duplicate_of_the_relief_field() -> None:
    # Regression (QA pass, 2026-09-11, follow-up to BUG-004): `legal_notice`
    # had no `subject_template` at all, so `_subject_line_text`'s fallback
    # (`fields.get("expected_relief", "")`, used verbatim, truncated at 140
    # chars) became the ENTIRE Subject line. Live repro: after an in-chat
    # edit regenerated the draft, the Subject section read literally
    # "refund of the full deposit within the response period." -- the exact
    # same raw text also appearing (correctly) inside the Prayer section --
    # instead of a real subject heading, silently replacing what the initial
    # LLM generation had composed ("LEGAL NOTICE FOR THE REFUND OF SECURITY
    # DEPOSIT"). This violated an explicit "baaki same rakho" (leave
    # everything else unchanged) instruction on a field-only edit, since the
    # Subject was not the field being edited. `subject_template` now follows
    # this app's own established convention (`demand_notice`'s own
    # `subject_template` is literally "Demand notice for {expected_relief}")
    # instead of relying on the raw fallback.
    engine = LegalDraftEngine()
    template = get_template("legal_notice")
    assert template is not None
    assert template.subject_template, "legal_notice must define its own subject_template"
    relief = "refund of the full deposit within the response period."
    fields = {"expected_relief": relief}
    subject = engine._subject_line_text(template, fields, "english")
    assert subject != relief
    assert relief in subject  # still names the actual relief sought
    assert subject.lower().startswith("legal notice")


# ---------------------------------------------------------------------------
# Finding-007 (QA pass, 2026-09-11): `claim_amount` was correctly threaded
# into the LLM prompt (`_render_fields`) but the model was not guaranteed to
# actually restate it -- confirmed live: a claim_amount of 65,000 never
# appeared anywhere in several real regenerations. The invariant under test
# is "the material figure appears in the rendered document", checked via a
# targeted substring assertion against the DYNAMIC value used, never a fixed
# full-document string (which would be brittle against unrelated generated
# prose changing).
# ---------------------------------------------------------------------------

_LEGAL_NOTICE_BASE_FIELDS = {
    "applicant_name": "Amit Kumar",
    "applicant_address": "Test Flat 12, Example Road, Mumbai.",
    "applicant_mobile": "9876543210",
    "respondent_name": "Rahul Verma",
    "respondent_address": "Test Flat 12, Example Road, Mumbai.",
    "facts": "I vacated the flat on 1 August 2026 after giving proper notice.",
    "expected_relief": "refund of the full deposit within the response period.",
    "place": "Mumbai",
}


def _fake_llm_notice_sections(*, mention_amount: bool) -> str:
    facts = "That I vacated the flat on 1 August 2026 after giving proper notice."
    if mention_amount:
        facts += " The amount of Rs. 65,000 was held as security deposit."
    return (
        "## Recipient\nRahul Verma\n\n"
        "## Subject\nLegal notice for refund of the full deposit.\n\n"
        "## Sender Details\nAmit Kumar\n\n"
        "## Introduction\nSir/Madam,\n\n"
        f"## Facts of the Case\n{facts}\n\n"
        "## Legal Position\nThe facts as stated may attract liability for the refund of the said amount.\n\n"
        "## Consequences\nThe applicant is deprived of the funds.\n\n"
        "## Prayer\nThe applicant demands the refund of the full deposit.\n\n"
        "## Signature Block\n(Amit Kumar)"
    )


def test_claim_amount_appears_in_rendered_draft_even_when_llm_omits_it() -> None:
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(
        return_value=LLMResponse(content=_fake_llm_notice_sections(mention_amount=False), model="test", provider="test")
    )
    request = DraftPreviewRequest(
        draft_id="legal_notice", language="english",
        fields={**_LEGAL_NOTICE_BASE_FIELDS, "claim_amount": "65,000"},
    )
    response = asyncio.run(engine.preview(request))
    assert response.generated_by_llm is True  # the LLM path succeeded; the model just didn't state the figure
    full_text = "\n".join(response.sections.values())
    assert "65,000" in full_text
    assert "50,000" not in full_text  # never invents/carries over an unrelated figure


def test_claim_amount_is_not_duplicated_when_the_llm_already_states_it() -> None:
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(
        return_value=LLMResponse(content=_fake_llm_notice_sections(mention_amount=True), model="test", provider="test")
    )
    request = DraftPreviewRequest(
        draft_id="legal_notice", language="english",
        fields={**_LEGAL_NOTICE_BASE_FIELDS, "claim_amount": "65,000"},
    )
    response = asyncio.run(engine.preview(request))
    full_text = "\n".join(response.sections.values())
    assert full_text.count("65,000") == 1


def test_style_instruction_reaches_the_generation_prompt_without_licensing_new_facts() -> None:
    # QA pass 2026-09-11, step 8: `style_instruction` ("restyle" edit
    # action) must actually reach the LLM prompt as a directive, and that
    # directive must explicitly forbid inventing new facts/threats/claims
    # -- "make it firmer" must never become licence to add a threat the
    # user never made.
    engine = LegalDraftEngine()
    captured_prompt = {}

    async def _fake_chat(messages, **kwargs):
        captured_prompt["text"] = messages[0].content
        return LLMResponse(content=_fake_llm_notice_sections(mention_amount=False), model="test", provider="test")

    engine.llm.chat = _fake_chat
    request = DraftPreviewRequest(
        draft_id="legal_notice", language="english",
        fields=_LEGAL_NOTICE_BASE_FIELDS,
        style_instruction="polite but firm",
    )
    asyncio.run(engine.preview(request))
    assert "polite but firm" in captured_prompt["text"]
    assert "do not add any new allegation" in captured_prompt["text"].lower()


def test_claim_amount_survives_the_deterministic_fallback_path_too() -> None:
    # Even when every LLM attempt fails and the engine falls back to the
    # deterministic skeleton, the material figure must still not be
    # silently dropped from the document `_ensure_verbatim_fields` is
    # applied to the SAME `sections` dict either path produces.
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    request = DraftPreviewRequest(
        draft_id="legal_notice", language="english",
        fields={**_LEGAL_NOTICE_BASE_FIELDS, "claim_amount": "65,000"},
    )
    response = asyncio.run(engine.preview(request))
    assert response.generated_by_llm is False
    full_text = "\n".join(response.sections.values())
    assert "65,000" in full_text


def test_no_claim_amount_means_no_injected_sentence() -> None:
    # A non-monetary legal notice (no `claim_amount` given at all) must not
    # get a spurious "Claim Amount: " line invented for it.
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(
        return_value=LLMResponse(content=_fake_llm_notice_sections(mention_amount=False), model="test", provider="test")
    )
    request = DraftPreviewRequest(draft_id="legal_notice", language="english", fields=_LEGAL_NOTICE_BASE_FIELDS)
    response = asyncio.run(engine.preview(request))
    full_text = "\n".join(response.sections.values())
    assert "Claim Amount" not in full_text


# ---------------------------------------------------------------------------
# D1 -- the deterministic (LLM-timeout) fallback must render every field a
# contract template collects, not a hardcoded subset.
# ---------------------------------------------------------------------------


def _dummy_value(field) -> str:
    if field.field_type == "date":
        return "2026-01-15"
    return f"Test value for {field.key}"


def _all_contract_templates():
    return [template for template in list_templates() if template.category == "Contract"]


@pytest.mark.parametrize("template", _all_contract_templates(), ids=lambda t: t.draft_id)
def test_deterministic_contract_fallback_renders_every_required_field(template) -> None:
    """Security finding D1: live repro was rent/security-deposit dropped
    silently from a rent agreement's deterministic fallback -- audited here
    across EVERY contract template, not just the one reported, since
    `_deterministic_contract_sections` previously hardcoded a subset of
    field names true for some templates (NDA/MOU/partnership/service/
    business-proposal) but not others (rent, property purchase/sale, which
    collect rent/deposit/consideration/property description as their own
    dedicated fields)."""
    fields = {field.key: _dummy_value(field) for field in template.required_fields}
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    request = DraftPreviewRequest(draft_id=template.draft_id, language="english", fields=fields)

    response = asyncio.run(engine.preview(request))

    assert response.generated_by_llm is False
    full_text = "\n".join(response.sections.values())
    for field in template.required_fields:
        if field.key in {"place"}:
            continue  # folded into the signature block, not a document-body fact
        assert fields[field.key] in full_text, (
            f"{template.draft_id}: required field {field.key!r} missing from the deterministic fallback output"
        )


def test_rent_agreement_fallback_states_rent_and_security_deposit() -> None:
    """The exact reported repro, asserted directly (not just via the
    generic sweep above) since these two figures are the ones a tenant or
    landlord would actually rely on this document for."""
    template = get_template("rent_agreement")
    fields = {field.key: _dummy_value(field) for field in template.required_fields}
    fields["monthly_rent_amount"] = "15,000"
    fields["security_deposit_amount"] = "45,000"
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    request = DraftPreviewRequest(draft_id="rent_agreement", language="english", fields=fields)

    response = asyncio.run(engine.preview(request))

    assert response.generated_by_llm is False
    full_text = "\n".join(response.sections.values())
    assert "15,000" in full_text
    assert "45,000" in full_text
    assert fields["rented_premises_address"] in full_text


def test_property_sale_agreement_fallback_states_consideration_and_property() -> None:
    template = get_template("property_sale_agreement")
    fields = {field.key: _dummy_value(field) for field in template.required_fields}
    fields["sale_consideration_amount"] = "50,00,000"
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    request = DraftPreviewRequest(draft_id="property_sale_agreement", language="english", fields=fields)

    response = asyncio.run(engine.preview(request))

    assert response.generated_by_llm is False
    full_text = "\n".join(response.sections.values())
    assert "50,00,000" in full_text
    assert fields["property_address"] in full_text


def test_unknown_draft_id_raises_not_found() -> None:
    engine = LegalDraftEngine()
    request = DraftPreviewRequest(draft_id="not-a-real-template", fields={})
    with pytest.raises(NotFoundError):
        asyncio.run(engine.preview(request))


def test_deterministic_fallback_generation_when_llm_returns_empty() -> None:
    # Forces the empty-LLM-response branch explicitly (rather than relying on
    # ambient env config) so this test is deterministic regardless of whether
    # a real LLM_PROVIDER/API key happens to be configured in the environment
    # it runs in. preview() never persists, so no MongoDB connection is needed.
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    request = DraftPreviewRequest(
        draft_id="rti_application",
        language="english",
        fields={
            "applicant_name": "Ramesh Kumar",
            "applicant_address": "12 MG Road, Pune",
            "applicant_mobile": "9876543210",
            "public_authority_name": "Public Works Department",
            "information_sought": "Copies of road repair tender documents for 2025.",
            "place": "Pune",
        },
    )
    response = asyncio.run(engine.preview(request))
    assert response.status == "complete"
    assert response.generated_by_llm is False
    assert response.draft_id is None  # preview never persists
    # Part 52 "Workflow Localization": the disclaimer is carried separately
    # on the response, never baked into `sections` (which is exactly what
    # chat preview/PDF/DOCX/TXT all render) -- it must never appear inside
    # the generated document itself.
    assert response.disclaimer == DRAFT_DISCLAIMER
    assert "Disclaimer" not in response.sections
    assert "Ramesh Kumar" in response.sections["Applicant Details"]
    # rti_application has category "Application" -- its own compact skeleton
    # (Part 42), not the old one-size-fits-all 15-heading structure. No
    # "Annexures" here since this request provided no available_documents.
    assert set(response.sections.keys()) == set(structure_sections_for("Application"))
    # "Date" is always today's date, generated fresh -- never hardcoded or
    # left equal to some incident date the user might have supplied. Part 53
    # "Professional Layout Audit": "Place"/"Date" are no longer separate
    # headings for Notice/Complaint/Application -- both are folded into the
    # "Signature" closing block, still rendered via `format_localized_date`
    # (for "english", the spelled-out "13 August 2026" form, not
    # `date.strftime("%B")`, which depends on the process's single global
    # OS locale).
    assert format_localized_date(date.today(), "english") in response.sections["Signature"]
    # No section should ever be left as a bare "None."/empty placeholder.
    for heading, text in response.sections.items():
        assert text.strip(), f"section '{heading}' must not be empty"
        assert text.strip().lower() not in {"none", "none."}, f"section '{heading}' must not be a 'None' placeholder"


_FALLBACK_NOTICE_FIELDS = {
    "applicant_name": "Asha Verma",
    "applicant_address": "45 Park Street, Kolkata",
    "applicant_mobile": "9123456789",
    "respondent_name": "Raj Traders",
    "respondent_address": "12 MG Road, Kolkata",
    "facts": "Goods worth Rs. 30000 were not delivered despite full payment.",
    "expected_relief": "Refund of Rs. 30000 within 15 days.",
    "place": "Kolkata",
}


@pytest.mark.parametrize(
    "language,expected_salutation",
    [
        ("english", "Sir/Madam,"),
        ("hindi", "महोदय/महोदया,"),
        ("tamil", "மதிப்பிற்குரிய ஐயா/அம்மா,"),
        ("telugu", "గౌరవనీయులైన సర్/మేడమ్,"),
        ("kannada", "ಗೌರವಾನ್ವಿತ ಸರ್/ಮೇಡಂ,"),
        ("bengali", "মহোদয়/মহোদয়া,"),
    ],
)
def test_deterministic_fallback_localizes_boilerplate_per_language(language: str, expected_salutation: str) -> None:
    # Part 53 "Fallback Draft Localization": forcing the LLM-failure fallback
    # path must still respect the requested language for the fallback's OWN
    # boilerplate (salutation, field labels, closing paragraph) -- previously
    # this path always produced English regardless of `request.language`.
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    request = DraftPreviewRequest(draft_id="legal_notice", language=language, fields=_FALLBACK_NOTICE_FIELDS)
    response = asyncio.run(engine.preview(request))

    assert response.generated_by_llm is False
    # Part 56 "Advocate-Style Draft Redesign": the unified Notice/Complaint
    # skeleton opens with "Introduction" (the salutation), not a single
    # "Notice" body section.
    assert expected_salutation in response.sections["Introduction"]
    # The user's own field values (names, addresses, amounts, phone numbers)
    # are never translated -- a non-LLM path has no way to translate free
    # text, so they must be rendered exactly as typed regardless of language.
    assert "Raj Traders" in response.sections["Recipient"]
    # Part 53 "Professional Layout Audit": the closing block no longer
    # repeats the mobile number (already shown once, in the recipient/
    # applicant details earlier in the document) -- it now holds Place/Date,
    # the complementary close, and the signatory's own name, still verbatim
    # and untranslated.
    assert "Asha Verma" in response.sections["Signature Block"]
    assert "Kolkata" in response.sections["Signature Block"]
    assert "Rs. 30000" in response.sections["Facts of the Case"]


def test_deterministic_fallback_falls_back_to_english_for_uncovered_language() -> None:
    # A language outside the fixed fallback-phrase table (e.g. "assamese",
    # which `LanguageDetector` recognizes but this table doesn't cover) must
    # never crash or return a blank/fabricated phrase -- English boilerplate
    # is a safe, always-correct fallback. (Part 57 "Drafting Lifecycle
    # Redesign" added marathi/gujarati/punjabi/odia/urdu/malayalam to this
    # table's priority-language tier -- "assamese" remains outside it.)
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    request = DraftPreviewRequest(draft_id="legal_notice", language="assamese", fields=_FALLBACK_NOTICE_FIELDS)
    response = asyncio.run(engine.preview(request))
    assert "Sir/Madam," in response.sections["Introduction"]


def test_deterministic_fallback_localizes_complaint_labels_and_authority_line() -> None:
    # Complaint-category templates render `template.authority_label` (a
    # per-template addressee line, e.g. "The Station House Officer,") and
    # field labels like "Police Station:"/"Witnesses:" that
    # `_deterministic_notice_sections` doesn't exercise at all.
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    request = DraftPreviewRequest(
        draft_id="police_complaint",
        language="hindi",
        fields={
            "applicant_name": "Ramesh Kumar",
            "applicant_address": "12 MG Road, Pune",
            "applicant_mobile": "9876500000",
            "police_station": "Hazratganj",
            "incident_location": "MG Road, Pune",
            "facts": "My motorcycle was stolen.",
            "expected_relief": "Register an FIR.",
            "place": "Pune",
        },
    )
    response = asyncio.run(engine.preview(request))
    assert "थाना" in response.sections["Facts of the Case"]
    assert "Hazratganj" in response.sections["Facts of the Case"]  # field value untouched
    assert "थाना प्रभारी" in response.sections["Recipient"]
    # Regression: the "Recipient" section VALUE must not repeat the
    # "To"/"सेवा में" heading itself -- `authority_label` previously started
    # with its own "To,"/"सेवा में," line, producing a visible duplicate
    # heading in exported documents ("सेवा में\nसेवा में,").
    assert "सेवा में" not in response.sections["Recipient"]


def test_cyber_crime_complaint_to_section_has_no_duplicate_heading() -> None:
    # Regression: the exact bug reported in a live PDF export -- the "To"
    # section rendered "सेवा में\nसेवा में," (heading, then a value that
    # repeated the same heading verbatim) because `authority_label` started
    # with its own "To,"/"सेवा में," line.
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    request = DraftPreviewRequest(
        draft_id="cyber_crime_complaint",
        language="hindi",
        fields={
            "applicant_name": "Rahul Kumar",
            "city": "Lucknow",
            "applicant_mobile": "9876543210",
            "incident_date": "15/07/2026",
            "fraud_amount": "85000",
            "bank_name": "State Bank of India",
            "transaction_id": "UTR458796321547",
            "police_station": "Hazratganj",
            "facts": "An unknown caller posing as a bank official tricked me into sharing my KYC details.",
        },
    )
    response = asyncio.run(engine.preview(request))
    assert "प्रभारी अधिकारी" in response.sections["Recipient"]
    assert "सेवा में" not in response.sections["Recipient"]
    # Also asserts the export layer, which prepends the translated "To"
    # heading -- confirms neither the heading nor the value produces
    # "सेवा में" anywhere in the assembled document (the heading itself was
    # separately changed to "प्रति" to avoid this exact collision).
    assert "सेवा में" not in response.full_text


def test_localized_title_uses_hindi_name_and_preserves_official_title_when_uncovered() -> None:
    template = get_template("legal_notice")
    assert template is not None
    assert localized_title(template, "hindi") == template.hindi_name
    assert localized_title(template, "tamil") == "பொது சட்ட அறிவிப்பு"
    assert localized_title(template, "english") == template.name
    # No reliable translation for this language -> preserve the official
    # English title rather than fabricate one (Part 53's own rule). All 22
    # Eighth Schedule languages are now covered (the last ten via
    # `eighth_schedule_names.py`), so this needs a language genuinely outside
    # the supported set to still exercise the fallback -- "assamese", used here
    # before, is now covered and would return its own name.
    assert localized_title(template, "portuguese") == template.name
    assert localized_title(template, "") == template.name


def test_safety_guard_blocks_forgery_requests_before_llm_call() -> None:
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(side_effect=AssertionError("LLM should not be called when safety guard blocks"))
    request = DraftPreviewRequest(
        draft_id="affidavit",
        fields={
            "applicant_name": "Test User",
            "applicant_address": "Somewhere",
            "affidavit_purpose": "Name change",
            "facts": "Please help me forge a fake signature for this affidavit.",
            "place": "Delhi",
        },
    )
    with pytest.raises(BadRequestError):
        asyncio.run(engine.preview(request))


def test_draft_safety_guard_scan() -> None:
    guard = DraftSafetyGuard()
    flagged, findings = guard.scan("I need a fake identity document for court.")
    assert flagged
    assert findings

    flagged_clean, no_findings = guard.scan("My landlord has not returned my security deposit of Rs. 20000.")
    assert not flagged_clean
    assert no_findings == []


def test_legal_terms_library_lookup() -> None:
    library = LegalTermsLibrary()
    results = asyncio.run(library.lookup("वादी"))
    assert any(entry.term == "वादी" for entry in results)
    plaintiff_entry = next(entry for entry in results if entry.term == "वादी")
    assert plaintiff_entry.english_equivalent == "Plaintiff"

    all_terms = asyncio.run(library.all_terms())
    assert len(all_terms) >= 40


def test_generate_persists_draft_with_mocked_llm_and_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    # First place in this suite that mocks the LLM/Mongo layers, since
    # generate() (unlike preview()) has real persistence side effects that
    # the other pure-function-style tests in this file don't have.
    engine = LegalDraftEngine()
    # legal_notice has category "Notice" -- its own compact skeleton
    # (To/Subject/Notice/Place/Date/Signature), not the old generic one.
    sectioned_response = "\n".join(
        f"## {heading}\nContent for {heading}." for heading in structure_sections_for("Notice")
    )
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content=sectioned_response, model="test", provider="test"))
    monkeypatch.setattr(engine.drafts, "insert", AsyncMock(return_value="draft-123"))
    monkeypatch.setattr(engine.versions, "insert", AsyncMock(return_value="version-1"))

    request = DraftGenerateRequest(
        draft_id="legal_notice",
        fields={
            "applicant_name": "Asha Verma",
            "applicant_address": "45 Park Street, Kolkata",
            "applicant_mobile": "9123456789",
            "respondent_name": "Raj Traders",
            "respondent_address": "12 MG Road, Kolkata",
            "facts": "Goods worth Rs. 30000 were not delivered despite full payment.",
            "expected_relief": "Refund of Rs. 30000 within 15 days.",
            "place": "Kolkata",
        },
    )
    response = asyncio.run(engine.generate(request))
    assert response.status == "complete"
    assert response.generated_by_llm is True
    assert response.draft_id is not None
    assert "Content for Facts of the Case." in response.sections["Facts of the Case"]
    # The LLM's own "## Signature Block" content is always overridden with
    # the deterministic closing block (Place/Date/complementary close/name),
    # regardless of what the model returned for that heading -- Part 53
    # "Professional Layout Audit".
    assert format_localized_date(date.today(), "english") in response.sections["Signature Block"]
    assert "Content for Signature Block." not in response.sections["Signature Block"]
    engine.drafts.insert.assert_awaited_once()
    engine.versions.insert.assert_awaited_once()


def test_translated_heading_falls_back_to_english_when_uncovered() -> None:
    # A language outside the fixed lookup table (or "english"/"hinglish",
    # where headings are conventionally left in English) must fall back to
    # the original heading rather than raising or returning a blank label.
    assert translated_heading("Date", "hindi") == "दिनांक"
    assert translated_heading("Date", "tamil") == "தேதி"
    assert translated_heading("Date", "english") == "Date"
    assert translated_heading("Date", "assamese") == "Date"
    assert translated_heading("Not A Real Heading", "hindi") == "Not A Real Heading"


def test_generate_full_text_translates_headings_but_not_field_values(monkeypatch: pytest.MonkeyPatch) -> None:
    # Part 52: the chat-displayed `full_text` must show translated section
    # headings (matching what the exporters will also show), while the
    # `sections` dict itself keeps literal English keys -- that's what field
    # edits (`replace_field`) and export section-iteration both depend on.
    engine = LegalDraftEngine()
    sectioned_response = "\n".join(
        f"## {heading}\nContent for {heading}." for heading in structure_sections_for("Notice")
    )
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content=sectioned_response, model="test", provider="test"))
    monkeypatch.setattr(engine.drafts, "insert", AsyncMock(return_value="draft-hindi"))
    monkeypatch.setattr(engine.versions, "insert", AsyncMock(return_value="version-1"))

    request = DraftGenerateRequest(
        draft_id="legal_notice",
        language="hindi",
        fields={
            "applicant_name": "Asha Verma",
            "applicant_address": "45 Park Street, Kolkata",
            "applicant_mobile": "9123456789",
            "respondent_name": "Raj Traders",
            "respondent_address": "12 MG Road, Kolkata",
            "facts": "Goods worth Rs. 30000 were not delivered despite full payment.",
            "expected_relief": "Refund of Rs. 30000 within 15 days.",
            "place": "Kolkata",
        },
    )
    response = asyncio.run(engine.generate(request))

    # `sections` keys stay literal English -- the parsing/edit contract.
    assert "Facts of the Case" in response.sections
    # Notice body sections are semantic generation keys, not visible letter
    # headings. Their content remains, while the internal heading is hidden.
    assert "मामले के तथ्य" not in response.full_text
    assert "Content for Facts of the Case." in response.full_text
    assert "\nFacts of the Case\n" not in response.full_text
    # Section body content (values, not structural labels) is untouched.
    assert "Content for Facts of the Case." in response.full_text


def test_regenerate_with_language_override_persists_language_and_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Part 52 "Multilingual Draft Engine": a language-change revision must
    # persist BOTH the newly-rendered `sections` and the new `language` back
    # onto the draft record -- previously `regenerate()` never wrote
    # `language` at all, so a subsequent PDF/DOCX export (which reads
    # straight from this record) could keep reporting the stale language
    # even after the draft was visibly re-rendered in a new one.
    engine = LegalDraftEngine()
    sectioned_response = "\n".join(
        f"## {heading}\nContent for {heading}." for heading in structure_sections_for("Complaint")
    )
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content=sectioned_response, model="test", provider="test"))
    existing_draft = {
        "_id": "draft-1",
        "draft_type": "police_complaint",
        "template_name": "Police Complaint",
        "language": "english",
        "session_id": "s1",
        "user_id": None,
        "fields": {"applicant_name": "Ramesh Kumar"},
        "sections": {"Recipient": "old english content"},
        "status": "complete",
        # Part 57 "Drafting Lifecycle Redesign": `regenerate()` now requires
        # an editable lifecycle state -- a record with no `lifecycle_state`
        # at all defaults to "locked" (see `_lifecycle_state_of`), so a
        # realistic in-progress draft fixture needs this set explicitly.
        "lifecycle_state": "preview_ready",
    }
    monkeypatch.setattr(engine.drafts, "find_by_id", AsyncMock(return_value=existing_draft))
    cas_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(engine.drafts, "compare_and_swap", cas_mock)
    monkeypatch.setattr(engine.versions, "latest_for_draft", AsyncMock(return_value=None))
    monkeypatch.setattr(engine.versions, "insert", AsyncMock(return_value="version-1"))

    response = asyncio.run(engine.regenerate("draft-1", {}, language="tamil"))

    assert response.language == "tamil"
    cas_mock.assert_awaited_once()
    called_draft_id, expected_version, update_payload = cas_mock.await_args.args
    assert called_draft_id == "draft-1"
    assert expected_version == 1  # existing_draft has no "version" field -- falls back to 1
    assert update_payload["language"] == "tamil"
    assert update_payload["sections"]["Recipient"] == "Content for Recipient."


# ---------------------------------------------------------------------------
# Part 56 "Advocate-Style Draft Redesign": word-count targeting and the
# bounded expansion retry.
# ---------------------------------------------------------------------------

_FACT_PRESERVATION_FIELDS = {
    "applicant_name": "Deepak Nair",
    "applicant_address": "7 Lake View Road, Kochi",
    "applicant_mobile": "9988776655",
    "respondent_name": "Coastal Traders Pvt Ltd",
    "respondent_address": "22 Marine Drive, Kochi",
    "facts": "On 3 March 2026 I paid Rs. 45000 in advance for furniture that was never delivered.",
    "expected_relief": "Refund of Rs. 45000 within 15 days.",
    "place": "Kochi",
}

_SECTIONED_FILLER = (
    "This paragraph elaborates the given facts in formal legal language without adding any new "
    "information beyond what the applicant has already provided. "
)


def _sectioned_llm_content(headings: tuple[str, ...], words_per_heading: int) -> str:
    sentences = max(1, words_per_heading // len(_SECTIONED_FILLER.split()))
    paragraph = (_SECTIONED_FILLER * sentences).strip()
    return "\n".join(f"## {heading}\n{paragraph}\n\n{paragraph}" for heading in headings)


def test_target_word_count_never_falls_below_the_professional_minimum() -> None:
    """Even the sparsest input must target the configured body-page floor.

    The old floor was 900 words, which renders as roughly two body pages --
    below the minimum this product promises, and the reason a real user
    received a one-page document. The floor is now whatever two pages
    actually costs, and it is SCRIPT-DEPENDENT: a Devanagari/Tamil word
    occupies far more page width than an English one, so demanding the
    English word count of an Indic draft would produce a bloated six-page
    police complaint instead of a well-proportioned two-page one.
    """
    latin_floor = _WORDS_PER_BODY_PAGE_LATIN * MINIMUM_BODY_PAGES
    indic_floor = _WORDS_PER_BODY_PAGE_INDIC * MINIMUM_BODY_PAGES

    assert _target_word_count({"facts": "short input"}, "english") == latin_floor
    assert _target_word_count({"facts": "short input"}, "hindi") == indic_floor
    assert _target_word_count({"facts": "short input"}, "tamil") == indic_floor
    # An unknown language is treated as Latin-width, which over- rather than
    # under-shoots the minimum -- the safe direction for a floor.
    assert _target_word_count({"facts": "short input"}, "klingon") == latin_floor

    # Above the floor it scales with how much the user actually supplied.
    assert _target_word_count({"facts": " ".join(["word"] * 1000)}, "english") == 3000
    assert _target_word_count({"facts": " ".join(["word"] * 1000)}, "hindi") == 3000


def test_estimated_page_count_is_script_aware() -> None:
    """The same word count fills more pages in an Indic script than in Latin."""
    assert estimated_page_count({"body": "complaint " * 400}) == 1
    assert estimated_page_count({"body": "शिकायत " * 400}) == 2


def test_generate_word_count_field_matches_actual_section_content(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = LegalDraftEngine()
    headings = structure_sections_for("Notice")
    sectioned_response = _sectioned_llm_content(headings, 90)
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content=sectioned_response, model="test", provider="test"))
    monkeypatch.setattr(engine.drafts, "insert", AsyncMock(return_value="draft-wc-1"))
    monkeypatch.setattr(engine.versions, "insert", AsyncMock(return_value="version-1"))

    request = DraftGenerateRequest(draft_id="legal_notice", fields=_FACT_PRESERVATION_FIELDS)
    response = asyncio.run(engine.generate(request))
    assert response.word_count == _word_count(response.sections)
    assert response.word_count > 0


def test_short_llm_response_triggers_one_expansion_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    # A first-pass draft below the professional minimum is retried exactly
    # once with a stronger expansion instruction -- the SAME facts,
    # expanded, never a second independent invention.
    engine = LegalDraftEngine()
    headings = structure_sections_for("Notice")
    short_response = _sectioned_llm_content(headings, 10)  # well under the 550-word retry threshold
    long_response = _sectioned_llm_content(headings, 90)  # comfortably over it
    engine.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(content=short_response, model="test", provider="test"),
            LLMResponse(content=long_response, model="test", provider="test"),
        ]
    )
    monkeypatch.setattr(engine.drafts, "insert", AsyncMock(return_value="draft-retry-1"))
    monkeypatch.setattr(engine.versions, "insert", AsyncMock(return_value="version-1"))

    request = DraftGenerateRequest(draft_id="legal_notice", fields=_FACT_PRESERVATION_FIELDS)
    response = asyncio.run(engine.generate(request))

    assert engine.llm.chat.await_count == 2
    # The retried (longer) content won, not the short first pass -- if the
    # retry hadn't been applied, word_count would be the short response's
    # (well under 1000).
    assert response.word_count > 1000


def test_generated_document_preserves_every_provided_fact() -> None:
    # Fact preservation: whatever the LLM/deterministic path produced, the
    # user's own literal fact values must survive verbatim into the document
    # -- this is the structural half of "never fabricate"; the other half
    # (never inventing facts NOT given) can only be enforced by the prompt
    # itself, not asserted mechanically against a mocked LLM.
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    request = DraftPreviewRequest(draft_id="legal_notice", fields=_FACT_PRESERVATION_FIELDS)
    response = asyncio.run(engine.preview(request))
    for key in ("applicant_name", "respondent_name", "place"):
        assert _FACT_PRESERVATION_FIELDS[key] in response.full_text
    assert "45000" in response.full_text
    assert "3 March 2026" in response.full_text


def test_affidavit_fallback_uses_formal_opening_numbered_facts_and_no_that_duplication() -> None:
    engine = LegalDraftEngine()
    template = get_template("address_affidavit")
    assert template is not None
    sections = engine._deterministic_affidavit_sections(
        template,
        {
            "applicant_name": "Rahul Sharma",
            "applicant_address": "House No. 42, Shanti Nagar, Agra",
            "affidavit_purpose": "To confirm my current residential address.",
            "facts": (
                "That I have resided at this address for three years. "
                "This is my present and correct residential address."
            ),
            "place": "Agra",
        },
    )

    assert sections["Deponent Details"].startswith("I, Rahul Sharma")
    assert "solemnly affirm and declare as under" in sections["Deponent Details"]
    assert "That That" not in sections["Statements"]
    assert "3. That I have resided" in sections["Statements"]
    assert "4. That this is my present" in sections["Statements"]
    assert "paragraphs 1 to 4" in sections["Verification"]
    preview = engine._render_full_text(sections)
    assert "### " not in preview
    assert "**Statements**" in preview


def test_lost_document_affidavit_fallback_keeps_affidavit_grammar_and_material_identifier() -> None:
    engine = LegalDraftEngine()
    template = get_template("lost_document_affidavit")
    assert template is not None

    sections = engine._deterministic_affidavit_sections(
        template,
        {
            "applicant_name": "अमित कुमार",
            "applicant_address": "24, शांति विहार, नोएडा",
            "lost_document_name": "मूल स्नातक प्रमाणपत्र",
            "document_identifier": "LU-2018-45821",
            "loss_date": "10 सितंबर 2026",
            "loss_place": "लखनऊ के चारबाग क्षेत्र",
            "affidavit_purpose": "डुप्लिकेट प्रमाणपत्र प्राप्त करने",
            "facts": "मेरा मूल स्नातक प्रमाणपत्र 10 सितंबर 2026 को लखनऊ के चारबाग क्षेत्र में खो गया।",
            "place": "नोएडा",
        },
        "hindi",
    )

    assert list(sections) == [
        "Court / Authority Name", "Deponent Details", "Statements", "Verification", "Place", "Date", "Signature"
    ]
    assert "LU-2018-45821" in sections["Statements"]
    assert sections["Deponent Details"].startswith("मैं, अमित कुमार")
    assert not {"Recipient", "Subject", "Prayer", "Request"}.intersection(sections)
    assert "Section 173" not in " ".join(sections.values())


def test_police_application_uses_request_heading_instead_of_court_prayer() -> None:
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    response = asyncio.run(
        engine.preview(
            DraftPreviewRequest(
                draft_id="police_complaint",
                language="english",
                fields={
                    "applicant_name": "Amit Kumar",
                    "applicant_address": "Agra",
                    "applicant_mobile": "9876543210",
                    "police_station": "Kamla Nagar",
                    "incident_location": "Kamla Nagar market",
                    "facts": "A person threatened and pushed me near the market.",
                    "expected_relief": "Please record my complaint and investigate it.",
                    "place": "Agra",
                },
            )
        )
    )
    assert "Request" in response.sections
    assert "Prayer" not in response.sections


def test_hindi_police_request_does_not_repeat_raw_user_prompt_or_affidavit_verification() -> None:
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    raw_prompt = (
        "16 September 2026 ko shaam 7:30 baje Sector 18 Noida Metro Station ke paas mera "
        "Samsung Galaxy S24 chori ho gaya. IMEI 352099001234567 hai. SHO ko FIR registration "
        "ke liye complaint draft karo."
    )
    response = asyncio.run(
        engine.preview(
            DraftPreviewRequest(
                draft_id="police_complaint",
                language="hindi",
                fields={
                    "applicant_name": "Rohit Sharma",
                    "applicant_address": "12, Ganesh Vihar, Noida",
                    "applicant_mobile": "9812345670",
                    "police_station": "Thana Sector 24, Noida",
                    "incident_location": "Sector 18 Noida Metro Station",
                    "incident_date": "16 September 2026",
                    "incident_time": "shaam 7:30 baje",
                    "imei_number": "352099001234567",
                    "facts": raw_prompt,
                    "expected_relief": raw_prompt,
                    "place": "Noida",
                },
            )
        )
    )

    assert "Request" in response.sections
    assert raw_prompt not in response.sections["Request"]
    assert "Section 173" not in response.sections["Request"]
    assert "धारा 173" in response.sections["Request"]
    assert "शपथपत्र" not in response.sections["Verification"]


@pytest.mark.parametrize(
    "language",
    ["hinglish", "hindi", "tamil", "telugu", "kannada", "bengali", "malayalam", "marathi",
     "gujarati", "punjabi", "odia", "urdu"],
)
def test_complaint_verification_is_native_for_every_non_english_full_language(language: str) -> None:
    text = LegalDraftEngine()._complaint_verification_text(language)

    assert "Verified that the contents" not in text
    assert "affidavit" not in text.casefold()
    assert "शपथपत्र" not in text
