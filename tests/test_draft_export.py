"""Part 42 "Draft Formatting" / Part 56 "Advocate-Style Draft Redesign" --
regression coverage for PDF/DOCX/TXT export of representative draft types.

The current policy uses a compact filing layout whose pagination follows
substantive content. Structural tests exercise the deterministic fallback;
layout tests verify the three-page content target, absence of artificial
cover/signature padding, absence of a maximum, and PDF/DOCX parity.
"""

import asyncio
import zipfile
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from docx import Document
from pypdf import PdfReader

from app.core.config import settings
from app.core.constants import DRAFT_DISCLAIMER
from app.drafting.engine import LegalDraftEngine
from app.drafting.export import (
    _LAYOUT,
    DocxDraftExporter,
    ExportOptions,
    PdfDraftExporter,
    RtfDraftExporter,
    TxtDraftExporter,
)
from app.drafting.localized_dates import format_localized_date
from app.drafting.templates import get_template
from app.drafting.templates.base import structure_sections_for
from app.llm.base import LLMResponse
from app.schemas.drafting import DraftGenerateRequest, DraftGenerateResponse

# Part 51 "Draft Generation Quality Pass": the "Date" section reads "13
# August 2026" for an English draft (all tests in this file use the default
# `language="english"`, see `_build` below) via `format_localized_date` --
# not `strftime("%d/%m/%Y")`, which Part 42/52 previously forced
# unconditionally to sidestep `strftime("%B")`'s OS-locale dependency.
# `format_localized_date` fixes that root cause with its own
# locale-independent per-language month lookup instead.
_TODAY = format_localized_date(date.today(), "english")

_DOCUMENTS: dict[str, dict[str, str]] = {
    "recovery_notice": {
        "applicant_name": "Rahul Kumar Sharma",
        "applicant_address": "24, Sector 62, Noida, Uttar Pradesh - 201309",
        "applicant_mobile": "9876543210",
        "respondent_name": "Mangal Prasad Verma",
        "respondent_address": "House No. 18, Sector 63, Noida, Uttar Pradesh - 201301",
        "principal_amount": "10000",
        "facts": (
            "I had paid a security deposit of Rs. 20000 at the time of taking accommodation from "
            "you. After vacating the premises and completing the necessary formalities, you have "
            "failed to refund the outstanding amount of Rs. 10000 despite repeated requests."
        ),
        "place": "Noida, Uttar Pradesh",
    },
    "police_complaint": {
        "applicant_name": "Ramesh Kumar",
        "applicant_address": "12 MG Road, Pune",
        "applicant_mobile": "9876500000",
        "police_station": "Hazratganj",
        "incident_location": "MG Road, Pune",
        "facts": "My motorcycle (registration MH12AB1234) was stolen from outside my shop on 5 August 2026.",
        "expected_relief": "Register an FIR and investigate the theft of my motorcycle.",
        "place": "Pune",
    },
    "cheque_bounce_notice": {
        "applicant_name": "Asha Verma",
        "applicant_address": "45 Park Street, Kolkata",
        "applicant_mobile": "9123456789",
        "respondent_name": "Raj Traders",
        "respondent_address": "12 MG Road, Kolkata",
        "cheque_number": "000123",
        "cheque_amount": "50000",
        "cheque_date": "01/07/2026",
        "bank_name": "State Bank of India",
        "dishonour_reason": "Insufficient funds",
        "dishonour_date": "10/07/2026",
        "facts": "The cheque issued towards settlement of goods supplied was returned unpaid for insufficient funds.",
        "place": "Kolkata",
    },
    "legal_notice": {
        "applicant_name": "Asha Verma",
        "applicant_address": "45 Park Street, Kolkata",
        "applicant_mobile": "9123456789",
        "respondent_name": "Raj Traders",
        "respondent_address": "12 MG Road, Kolkata",
        "facts": "Goods worth Rs. 30000 were not delivered despite full payment made in advance.",
        "expected_relief": "Refund of Rs. 30000",
        "place": "Kolkata",
    },
}


def _build(draft_id: str, fields: dict[str, str]) -> tuple[DraftGenerateResponse, LegalDraftEngine]:
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    request = DraftGenerateRequest(draft_id=draft_id, fields=fields)
    return asyncio.run(engine.preview(request)), engine


# ---------------------------------------------------------------------------
# Structural regressions (deterministic fallback path) -- all four fixture
# draft_ids are Notice/Complaint category, i.e. the unified skeleton.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("draft_id", list(_DOCUMENTS.keys()))
def test_draft_sections_have_no_empty_or_none_placeholders(draft_id: str) -> None:
    response, _engine = _build(draft_id, _DOCUMENTS[draft_id])
    assert response.status == "complete"
    for heading, text in response.sections.items():
        assert text.strip(), f"{draft_id}: section '{heading}' is empty"
        assert text.strip().lower() not in {"none", "none.", "n/a"}, f"{draft_id}: section '{heading}' is a placeholder"


@pytest.mark.parametrize("draft_id", list(_DOCUMENTS.keys()))
def test_draft_date_is_todays_date_not_hardcoded_or_incident_date(draft_id: str) -> None:
    # Part 53/56: "Date" is folded into the "Signature Block" closing block
    # for these (Notice/Complaint category) templates rather than being its
    # own top-level section.
    response, _engine = _build(draft_id, _DOCUMENTS[draft_id])
    assert _TODAY in response.sections["Signature Block"]


def test_legal_notice_has_no_meaningless_court_authority_section() -> None:
    """A Legal Notice is addressed to a private party, not a court -- must
    not carry a "Court / Authority Name" heading (Part 42 section 5)."""
    response, _engine = _build("legal_notice", _DOCUMENTS["legal_notice"])
    assert "Court / Authority Name" not in response.sections
    assert "Recipient" in response.sections


def test_recovery_notice_has_no_generic_legal_grounds_or_applicable_sections_boilerplate() -> None:
    response, _engine = _build("recovery_notice", _DOCUMENTS["recovery_notice"])
    assert "Legal Grounds" not in response.sections
    assert "Applicable Sections" not in response.sections
    assert "Applicable Acts" not in response.sections


def test_no_annexures_section_when_no_documents_supplied() -> None:
    # legal_notice, not police_complaint: police_complaint is now an
    # advocate-register template (see test_draft_advocate_register.py) and
    # renders "Enclosures" instead of "Annexures" -- this test covers the
    # ORIGINAL Annexures behavior, still used by Notice/Affidavit/Contract
    # and the ordinary-correspondence Application templates.
    response, _engine = _build("legal_notice", _DOCUMENTS["legal_notice"])
    assert "Annexures" not in response.sections


def test_annexures_section_present_when_documents_supplied() -> None:
    fields = {**_DOCUMENTS["legal_notice"], "available_documents": "RC copy, insurance papers"}
    response, _engine = _build("legal_notice", fields)
    assert response.sections["Annexures"] == "RC copy, insurance papers"


# ---------------------------------------------------------------------------
# Export mechanics smoke tests (deterministic fallback content -- short by
# design, since that path never invents narrative content; see module
# docstring for why length/page-band coverage lives further down instead).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("draft_id", list(_DOCUMENTS.keys()))
def test_pdf_export_opens_and_preserves_content(tmp_path, draft_id: str) -> None:
    response, _engine = _build(draft_id, _DOCUMENTS[draft_id])
    output_path = tmp_path / f"{draft_id}.pdf"
    PdfDraftExporter().export(response.template_name, response.sections, output_path)

    assert output_path.exists()
    reader = PdfReader(str(output_path))
    page_count = len(reader.pages)
    assert page_count >= 1
    # Never truncated: every field's content must survive into the PDF text.
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert _DOCUMENTS[draft_id]["applicant_name"] in extracted
    assert _TODAY in extracted
    # Part 52 "Workflow Localization": the AI-generated-draft disclaimer
    # must never appear INSIDE the exported document itself -- it's carried
    # separately via `DraftGenerateResponse.disclaimer` instead (see
    # `LegalDraftEngine._render_sections`).
    assert DRAFT_DISCLAIMER.split(".")[0] not in extracted


@pytest.mark.parametrize("draft_id", list(_DOCUMENTS.keys()))
def test_docx_export_does_not_crash_and_produces_a_file(tmp_path, draft_id: str) -> None:
    response, _engine = _build(draft_id, _DOCUMENTS[draft_id])
    output_path = tmp_path / f"{draft_id}.docx"
    DocxDraftExporter().export(response.template_name, response.sections, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_pdf_export_has_no_watermark_by_default(tmp_path) -> None:
    """A finished draft is handed to a police station / landlord / bank, so it
    must not be defaced with a "DRAFT" wash unless the caller asks for one.
    The AI-generated-draft disclaimer carries that message in words instead."""
    output_path = tmp_path / "plain.pdf"
    PdfDraftExporter().export("Legal Notice", {"Facts of the Case": "Important facts."}, output_path)
    extracted = '\n'.join(page.extract_text() or "" for page in PdfReader(str(output_path)).pages)
    assert "DRAFT" not in extracted


def test_pdf_export_contains_configured_watermark_text(tmp_path) -> None:
    output_path = tmp_path / "watermarked.pdf"
    PdfDraftExporter().export(
        "Legal Notice",
        {"Facts of the Case": "Important facts."},
        output_path,
        options=ExportOptions(watermark=True, watermark_text="DRAFT - NOT FOR FILING"),
    )
    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(str(output_path)).pages)
    assert "DRAFT-NOTFORFILING" in extracted.replace(" ", "")


def test_docx_export_embeds_configured_watermark_and_signature_images(tmp_path) -> None:
    from PIL import Image

    watermark_path = tmp_path / "watermark.png"
    signature_path = tmp_path / "signature.png"
    Image.new("RGBA", (240, 80), (20, 80, 120, 100)).save(watermark_path)
    Image.new("RGBA", (160, 50), (20, 80, 120, 255)).save(signature_path)
    output_path = tmp_path / "signed.docx"
    DocxDraftExporter().export(
        "Legal Notice",
        {"Signature Block": "Asha Verma"},
        output_path,
        options=ExportOptions(
            watermark=True,
            watermark_image_path=str(watermark_path),
            signature_image_path=str(signature_path),
        ),
    )
    with zipfile.ZipFile(output_path) as package:
        media_files = [name for name in package.namelist() if name.startswith("word/media/")]
    assert len(media_files) == 2


def test_pdf_signing_requires_certificate_material(tmp_path) -> None:
    with pytest.raises(ValueError, match="DRAFT_SIGNING_CERTIFICATE_PATH"):
        PdfDraftExporter().export(
            "Legal Notice",
            {"Facts of the Case": "Important facts."},
            tmp_path / "signed.pdf",
            options=ExportOptions(sign=True),
        )


def test_pdf_export_embeds_cryptographic_signature(tmp_path) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Advocate")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    certificate_path = tmp_path / "signing-cert.pem"
    key_path = tmp_path / "signing-key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    output_path = tmp_path / "signed.pdf"
    PdfDraftExporter().export(
        "Legal Notice",
        {"Facts of the Case": "Important facts."},
        output_path,
        options=ExportOptions(
            sign=True,
            signing_certificate_path=str(certificate_path),
            signing_key_path=str(key_path),
        ),
    )
    pdf_bytes = output_path.read_bytes()
    assert b"/ByteRange" in pdf_bytes
    assert b"LegalDraftSignature" in pdf_bytes


@pytest.mark.parametrize("draft_id", list(_DOCUMENTS.keys()))
def test_rtf_export_does_not_crash_and_produces_a_file(tmp_path, draft_id: str) -> None:
    response, _engine = _build(draft_id, _DOCUMENTS[draft_id])
    output_path = tmp_path / f"{draft_id}.rtf"
    RtfDraftExporter().export(response.template_name, response.sections, output_path)
    assert output_path.exists()
    content = output_path.read_text(encoding="ascii")
    assert content.startswith("{\\rtf1")
    assert content.rstrip().endswith("}")
    assert _DOCUMENTS[draft_id]["applicant_name"] in content


def test_rtf_export_translates_headings_and_roundtrips_unicode_for_requested_language(tmp_path) -> None:
    # Mirrors the DOCX/TXT heading-translation tests above, plus verifies the
    # RTF-specific piece: non-ASCII (Devanagari) text must be written as
    # `\uN?` escapes, not raw bytes (the file is written with `encoding=
    # "ascii"`, which would already fail loudly if this were broken), and
    # must decode back to the original text via `striprtf` -- the same
    # library this app now uses on the read/upload side, so this is a
    # genuine round-trip check, not just a "doesn't crash" smoke test.
    from striprtf.striprtf import rtf_to_text

    sections = {
        "Recipient": "Raj Traders",
        "Subject": "Test",
        "Facts of the Case": "Body text.",
        "Signature Block": "Place: Kolkata\nDate: " + _TODAY,
    }
    output_path = tmp_path / "hindi.rtf"
    RtfDraftExporter().export("Legal Notice", sections, output_path, "hindi")
    content = output_path.read_text(encoding="ascii")
    assert "मामले के तथ्य" not in content  # raw Devanagari never appears literally...
    assert "\\u" in content  # ...it's written as Unicode escapes instead
    assert "\\qr" in content  # Signature Block is right-aligned
    decoded = rtf_to_text(content)
    assert "मामले के तथ्य" in decoded  # ...and decodes back to the translated heading
    assert "Raj Traders" in decoded  # field values stay untouched


def test_txt_export_translates_headings_for_requested_language(tmp_path) -> None:
    # Part 52: exported headings must match the draft's language -- keeps a
    # downloaded file consistent with what chat displayed for the same draft.
    sections = {
        "Recipient": "Raj Traders",
        "Subject": "Test",
        "Facts of the Case": "Body text.",
        "Signature Block": "Place: Kolkata\nDate: " + _TODAY,
    }
    output_path = tmp_path / "hindi.txt"
    TxtDraftExporter().export("Legal Notice", sections, output_path, "hindi")
    content = output_path.read_text(encoding="utf-8")
    assert "मामले के तथ्य" in content  # translated "Facts of the Case" heading
    assert "\nFacts of the Case\n" not in content  # original English heading label is gone
    assert "Raj Traders" in content  # field values stay untouched


def test_docx_export_translates_headings_for_requested_language(tmp_path) -> None:
    sections = {
        "Recipient": "Raj Traders",
        "Subject": "Test",
        "Facts of the Case": "Body text.",
        "Signature Block": "Place: Kolkata\nDate: " + _TODAY,
    }
    output_path = tmp_path / "hindi.docx"
    DocxDraftExporter().export("Legal Notice", sections, output_path, "hindi")
    document = Document(str(output_path))
    heading_texts = [p.text for p in document.paragraphs]
    assert any("मामले के तथ्य" in text for text in heading_texts)
    assert not any(text.strip() == "Facts of the Case" for text in heading_texts)


def test_export_of_fallback_draft_uses_localized_title_and_never_calls_llm(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Part 53 "Fallback Draft Localization": exporting a draft that was
    # generated by the deterministic fallback (or an LLM-generated one -- the
    # exporter has no way to know which) must never call the LLM itself, and
    # PDF/DOCX/TXT must all show the same localized title/headings, read
    # straight from the persisted `sections` -- no per-format regeneration.
    monkeypatch.setattr(settings, "draft_output_dir", tmp_path)
    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(side_effect=AssertionError("export must never call the LLM"))
    template = get_template("legal_notice")
    assert template is not None
    fake_draft = {
        "_id": "draft-fallback-1",
        "draft_type": "legal_notice",
        "template_name": template.name,
        "language": "tamil",
        "sections": {
            "Recipient": "Raj Traders\n12 MG Road, Kolkata",
            "Subject": "Test subject",
            "Facts of the Case": "மதிப்பிற்குரிய ஐயா/அம்மா,\n\nBody text.",
            "Signature Block": "Place: Kolkata\nDate: " + _TODAY + "\nSd/-\nAsha Verma",
        },
        # Part 57 "Drafting Lifecycle Redesign": export() now requires the
        # draft be locked/exported, and flips locked -> exported on its
        # first successful export (a real state-transition write, mocked
        # below like the rest of the repository layer).
        "lifecycle_state": "locked",
    }
    monkeypatch.setattr(engine.drafts, "find_by_id", AsyncMock(return_value=fake_draft))
    monkeypatch.setattr(engine.drafts, "update_by_id", AsyncMock(return_value=True))

    pdf_path = asyncio.run(engine.export("draft-fallback-1", "pdf"))
    docx_path = asyncio.run(engine.export("draft-fallback-1", "docx"))
    txt_path = asyncio.run(engine.export("draft-fallback-1", "txt"))
    rtf_path = asyncio.run(engine.export("draft-fallback-1", "rtf"))

    assert pdf_path.exists() and docx_path.exists() and txt_path.exists() and rtf_path.exists()
    txt_content = txt_path.read_text(encoding="utf-8")
    assert "பொது சட்ட அறிவிப்பு" in txt_content  # localized title, not "General Legal Notice"
    assert "வழக்கின் உண்மைகள்" not in txt_content  # internal notice heading is semantic-only
    assert "Body text." in txt_content
    reader = PdfReader(str(pdf_path))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "பொது சட்ட அறிவிப்பு" in pdf_text
    assert "வழக்கின் உண்மைகள்" not in pdf_text
    assert "Body text." in pdf_text
    document = Document(str(docx_path))
    docx_text = "\n".join(p.text for p in document.paragraphs)
    assert "பொது சட்ட அறிவிப்பு" in docx_text
    assert "வழக்கின் உண்மைகள்" not in docx_text
    assert "Body text." in docx_text
    from striprtf.striprtf import rtf_to_text

    rtf_decoded = rtf_to_text(rtf_path.read_text(encoding="ascii"))
    assert "பொது சட்ட அறிவிப்பு" in rtf_decoded
    assert "வழக்கின் உண்மைகள்" not in rtf_decoded
    assert "Body text." in rtf_decoded


# ---------------------------------------------------------------------------
# Professional three-part filing packet: cover, unrestricted body, and
# separate execution page. Three pages is the floor; there is no ceiling.
# ---------------------------------------------------------------------------

_FILLER_WORDS = ["The", "applicant", "respectfully", "sets", "out", "these", "facts", "in", "good", "faith", "and", "relies", "upon", "them", "for", "the", "purpose", "of", "this", "document", "as", "part", "of", "the", "narrative", "required", "by", "this", "section."]


def _filler_text(word_count: int) -> str:
    """Exactly `word_count` words (not a whole-sentence-multiple
    approximation), so the fixture's actual word total tracks the requested
    target precisely enough to test the 600-900 word / 2-3 page band
    meaningfully.
    """
    words = (_FILLER_WORDS * (word_count // len(_FILLER_WORDS) + 1))[:word_count]
    return " ".join(words) + "."


def _realistic_sections(total_words: int) -> dict[str, str]:
    """A `sections` dict shaped like a real unified-skeleton draft (every
    section has 2 paragraphs, per the spec's paragraph-spacing rules),
    totalling approximately `total_words` words across all sections.
    """
    headings = structure_sections_for("Notice")
    paragraph_words = max(10, total_words // (2 * len(headings)))
    paragraph = _filler_text(paragraph_words)
    return {heading: f"{paragraph}\n\n{paragraph}" for heading in headings}


def _word_count(sections: dict[str, str]) -> int:
    return sum(len(text.split()) for text in sections.values())


@pytest.mark.parametrize("target_words", [900, 1500])
def test_substantive_pdf_target_renders_to_at_least_three_pages(tmp_path, target_words: int) -> None:
    sections = _realistic_sections(target_words)
    word_count = _word_count(sections)
    output_path = tmp_path / f"sized-{target_words}.pdf"
    PdfDraftExporter().export("Test Notice", sections, output_path)

    reader = PdfReader(str(output_path))
    page_count = len(reader.pages)
    assert page_count >= 3, f"a {word_count}-word document rendered to {page_count} pages, expected at least 3"


def test_short_pdf_is_not_padded_with_blank_stationery(tmp_path) -> None:
    sections = _realistic_sections(250)
    output_path = tmp_path / "min.pdf"
    PdfDraftExporter().export("Test Notice", sections, output_path)
    page_count = len(PdfReader(str(output_path)).pages)
    assert page_count < 3, "short content must not be disguised as three pages with a cover or signature-only page"


def test_pdf_has_no_maximum_page_ceiling(tmp_path) -> None:
    sections = _realistic_sections(3000)
    output_path = tmp_path / "long.pdf"
    PdfDraftExporter().export("Test Notice", sections, output_path)
    page_count = len(PdfReader(str(output_path)).pages)
    assert page_count > 3, "long drafts must flow naturally beyond three pages without compaction"


def test_html_uses_continuous_official_filing_layout() -> None:
    html_document = PdfDraftExporter()._build_html(
        "Test Notice", _realistic_sections(300), _LAYOUT, "english", ExportOptions(document_version=4)
    )
    assert "class='cover-page'" not in html_document
    assert "execution-page" not in html_document
    assert "title-rule" not in html_document
    assert "signature-section" in html_document


def test_docx_does_not_force_cover_or_signature_only_pages(tmp_path) -> None:
    output_path = tmp_path / "professional.docx"
    DocxDraftExporter().export("Test Notice", _realistic_sections(300), output_path)
    with zipfile.ZipFile(output_path) as package:
        document_xml = package.read("word/document.xml").decode("utf-8")
    assert 'w:type="page"' not in document_xml


def test_docx_and_pdf_share_the_same_fixed_layout_regardless_of_length(tmp_path) -> None:
    # Part 56 removes the old compaction ladder entirely -- font size and
    # margins must now be IDENTICAL for a short vs. a long document, proving
    # no shrink-to-fit behavior remains.
    short_sections = _realistic_sections(600)
    long_sections = _realistic_sections(900)

    short_path = tmp_path / "short.docx"
    long_path = tmp_path / "long.docx"
    DocxDraftExporter().export("Test Notice", short_sections, short_path)
    DocxDraftExporter().export("Test Notice", long_sections, long_path)

    short_doc = Document(str(short_path))
    long_doc = Document(str(long_path))
    assert short_doc.styles["Normal"].font.size == long_doc.styles["Normal"].font.size
    assert short_doc.sections[0].top_margin == long_doc.sections[0].top_margin
    from docx.shared import Pt

    assert short_doc.styles["Normal"].font.size == Pt(_LAYOUT["body"])


def test_pdf_typography_uses_justified_text_and_professional_margins() -> None:
    sections = _realistic_sections(700)
    html_document = PdfDraftExporter()._build_html("Test Notice", sections, _LAYOUT, "english")
    assert "text-align: justify" in html_document
    assert f'margin: {_LAYOUT["margin_v"]}in {_LAYOUT["margin_h"]}in' in html_document
    # The signature-block override still right-aligns, overriding the
    # otherwise-universal justify rule.
    assert "text-align: right" in html_document
    # Matches the Delhi High Court Practice Direction (in force w.e.f.
    # 01.11.2022): font size 14 uniformly for title/heading/body, 1.5 line
    # spacing, 4cm left/right and 2cm top/bottom margins -- see _LAYOUT's
    # own comment for the full citation.
    assert _LAYOUT["title"] == _LAYOUT["heading"] == _LAYOUT["body"] == 14
    assert _LAYOUT["leading_ratio"] == 1.5
    assert _LAYOUT["margin_h"] == pytest.approx(4 / 2.54)
    assert _LAYOUT["margin_v"] == pytest.approx(2 / 2.54)
    assert "text-transform: uppercase" not in html_document


def test_multi_paragraph_section_renders_as_separate_paragraphs_not_one_block(tmp_path) -> None:
    sections = {
        "Recipient": "Raj Traders",
        "Subject": "Test",
        "Facts of the Case": "First paragraph of facts.\n\nSecond paragraph of facts.\n\nThird paragraph of facts.",
        "Signature Block": "Place: Kolkata\nDate: " + _TODAY,
    }
    html_document = PdfDraftExporter()._build_html("Test Notice", sections, _LAYOUT, "english")
    assert html_document.count("First paragraph of facts.") == 1
    assert html_document.count("Second paragraph of facts.") == 1
    assert html_document.count("Third paragraph of facts.") == 1
    # Three distinct <p> tags for the three paragraphs, not one merged block.
    facts_html = html_document.split("Facts of the Case</h2>")[1].split("</div>")[0]
    assert facts_html.count("<p>") == 3

    docx_path = tmp_path / "paragraphs.docx"
    DocxDraftExporter().export("Test Notice", sections, docx_path)
    document = Document(str(docx_path))
    paragraph_texts = [p.text for p in document.paragraphs]
    assert "First paragraph of facts." in paragraph_texts
    assert "Second paragraph of facts." in paragraph_texts
    assert "Third paragraph of facts." in paragraph_texts


def test_signature_block_heading_is_right_aligned_in_both_pdf_css_and_docx(tmp_path) -> None:
    sections = {
        "Recipient": "Raj Traders",
        "Subject": "Test",
        "Facts of the Case": "Body text.",
        "Signature Block": "Place: Kolkata\nDate: " + _TODAY + "\n\nYours faithfully,\n\n(Asha Verma)\nPetitioner",
    }
    html_document = PdfDraftExporter()._build_html("Test Notice", sections, _LAYOUT, "english")
    assert "class='signature-block'" in html_document

    docx_path = tmp_path / "signature.docx"
    DocxDraftExporter().export("Test Notice", sections, docx_path)
    document = Document(str(docx_path))
    found_heading = False
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == "Signature Block":
            found_heading = True
            assert paragraph.alignment == 2  # WD_ALIGN_PARAGRAPH.RIGHT
    assert found_heading


# ---------------------------------------------------------------------------
# Part 57 "Drafting Lifecycle Redesign": script-family font routing --
# replaces the old unconditional "every language uses Nirmala UI" with a
# per-script lookup (`app.drafting.export._SCRIPT_FONTS`/`_LANGUAGE_SCRIPT`),
# and adds RTL layout for Urdu/Sindhi/Kashmiri (Perso-Arabic) plus a hard
# failure for Ol Chiki (Santali) / Meetei Mayek (Manipuri) PDF export, since
# neither script has a usable font installed on this server.
# ---------------------------------------------------------------------------

_SAMPLE_SECTIONS = {
    "Recipient": "Raj Traders\n12 MG Road, Kolkata",
    "Subject": "Test subject",
    "Facts of the Case": "Body text across a single paragraph.",
    "Signature Block": "Place: Kolkata\nDate: " + _TODAY + "\nSd/-\nAsha Verma",
}


@pytest.mark.parametrize(
    "language",
    ["hindi", "marathi", "sanskrit", "bengali", "assamese", "gujarati", "punjabi", "odia",
     "tamil", "telugu", "kannada", "malayalam", "nepali", "maithili", "dogri", "bodo"],
)
def test_pdf_export_succeeds_for_every_brahmic_script_language(tmp_path, language: str) -> None:
    # All of these route to the SAME physical font (Nirmala UI is one
    # pan-Indic TTC covering all nine Brahmic scripts this app supports) --
    # asserted per-language anyway since each is a distinct entry in
    # `_LANGUAGE_SCRIPT`, and a typo there would silently misroute one.
    pdf_path = tmp_path / f"{language}.pdf"
    PdfDraftExporter().export("Test Notice", _SAMPLE_SECTIONS, pdf_path, language)
    assert pdf_path.exists() and pdf_path.stat().st_size > 0


@pytest.mark.parametrize("language", ["urdu", "sindhi", "kashmiri"])
def test_pdf_export_succeeds_and_is_rtl_for_perso_arabic_languages(tmp_path, language: str) -> None:
    html_document = PdfDraftExporter()._build_html("Test Notice", _SAMPLE_SECTIONS, _LAYOUT, language)
    assert 'dir="rtl"' in html_document
    assert "direction: rtl" in html_document
    pdf_path = tmp_path / f"{language}.pdf"
    PdfDraftExporter().export("Test Notice", _SAMPLE_SECTIONS, pdf_path, language)
    assert pdf_path.exists() and pdf_path.stat().st_size > 0


def test_pdf_export_is_ltr_for_non_perso_arabic_languages() -> None:
    html_document = PdfDraftExporter()._build_html("Test Notice", _SAMPLE_SECTIONS, _LAYOUT, "hindi")
    assert 'dir="rtl"' not in html_document
    assert "direction: ltr" in html_document


def test_flowing_pdf_keeps_final_demand_paragraph_with_signature_when_possible() -> None:
    html_document = PdfDraftExporter()._build_html(
        "Test Notice", _SAMPLE_SECTIONS, _LAYOUT, "hindi", ExportOptions(flowing_letter=True)
    )

    assert ".flowing-body p:nth-last-child(-n+4) { break-after: avoid; }" in html_document
    assert ".signature-section { break-inside: avoid;" in html_document


@pytest.mark.parametrize("language", ["santali", "manipuri"])
def test_pdf_export_raises_unsupported_for_scripts_with_no_installed_font(tmp_path, language: str) -> None:
    from app.core.exceptions import UnsupportedExportError

    pdf_path = tmp_path / f"{language}.pdf"
    with pytest.raises(UnsupportedExportError):
        PdfDraftExporter().export("Test Notice", _SAMPLE_SECTIONS, pdf_path, language)
    assert not pdf_path.exists()


@pytest.mark.parametrize("language", ["santali", "manipuri"])
def test_docx_and_txt_export_still_succeed_for_scripts_with_no_installed_font(tmp_path, language: str) -> None:
    # DOCX/TXT don't need server-side glyph rasterization -- Word/
    # LibreOffice substitutes fonts at OPEN time on whichever machine
    # actually displays the document, a much weaker requirement than the
    # PDF exporter baking exact glyph IDs into the file at generation time.
    docx_path = tmp_path / f"{language}.docx"
    DocxDraftExporter().export("Test Notice", _SAMPLE_SECTIONS, docx_path, language)
    assert docx_path.exists()
    document = Document(str(docx_path))
    assert document.styles["Normal"].font.name in {"Noto Sans Ol Chiki", "Noto Sans Meetei Mayek"}

    txt_path = tmp_path / f"{language}.txt"
    TxtDraftExporter().export("Test Notice", _SAMPLE_SECTIONS, txt_path, language)
    assert txt_path.exists()


def test_docx_export_applies_rtl_paragraph_direction_for_urdu(tmp_path) -> None:
    from docx.oxml.ns import qn

    docx_path = tmp_path / "urdu.docx"
    DocxDraftExporter().export("Test Notice", _SAMPLE_SECTIONS, docx_path, "urdu")
    document = Document(str(docx_path))
    # At least one body paragraph must carry the `w:bidi` RTL marker.
    found_bidi = False
    for paragraph in document.paragraphs:
        pPr = paragraph._p.find(qn("w:pPr"))
        if pPr is not None and pPr.find(qn("w:bidi")) is not None:
            found_bidi = True
            break
    assert found_bidi


# QA retest 2026-09-24 (T090 "XSS in draft field"): the original QA report
# flagged `<img src=x onerror=alert(1)>` as a field value but never confirmed
# whether the export pipeline actually renders it as live markup -- the
# retest found that specific probe inconclusive (the test session's draft had
# already left the `collecting` stage by the time it ran, so the field-store
# path was never exercised at all). Reproduced live end-to-end instead: drove
# a fresh consumer-complaint draft through `/chat` with
# "applicant name is <img src=x onerror=alert(1)>, applicant address is
# Chennai" as an actual field answer, then exported the completed draft to
# PDF and extracted its text with `pypdf` -- the payload appears as inert
# visible characters ("I, <img src=x onerror=alert(1)>, residing at..."),
# never as a real `<img>` element, because `PdfDraftExporter._build_html`
# passes every section value through `_escape` (`html.escape`) before
# interpolating it into the HTML WeasyPrint renders. This locks that
# behavior in at the unit level so a future change to the HTML-assembly path
# can't silently drop the escape call.
def test_pdf_export_escapes_html_markup_in_a_field_value(tmp_path) -> None:
    payload = "<img src=x onerror=alert(1)>"
    sections = {"Complainant Details": f"I, {payload}, residing at Chennai."}

    html_document = PdfDraftExporter()._build_html("Test Notice", sections, _LAYOUT, "english")
    assert "<img src=x onerror=alert(1)>" not in html_document
    assert "&lt;img src=x onerror=alert(1)&gt;" in html_document

    output_path = tmp_path / "xss_field.pdf"
    PdfDraftExporter().export("Test Notice", sections, output_path)
    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(str(output_path)).pages)
    # The literal characters must still be legible as plain text (the
    # complainant's stated name, verbatim) -- only the ability for them to
    # be interpreted as markup is what must be gone.
    assert payload in extracted
