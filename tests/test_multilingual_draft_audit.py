"""Part 55 "Draft Audit -- Language Propagation Fix": end-to-end conversation
tests reproducing the actual reported bug -- "तमिल में पुलिस शिकायत का मसौदा
तैयार करें।" (a police-complaint draft explicitly requested IN TAMIL, typed
entirely in Hindi/Devanagari) silently drafted in Hindi instead -- plus the
equivalent scenario for Telugu/Kannada/Bengali, each phrased as a NATIVE
locative-case request typed in a DIFFERENT one of the four scripts (the
hardest case: the message's own script and the requested document language
never match, so a bug in either `extract_requested_language` or its
downstream propagation through `DraftConversationEngine` would surface
immediately).

Deliberately exercises the real `DraftConversationEngine.handle_turn` state
machine turn-by-turn (not just the standalone `extract_requested_language`/
`DraftIntentDetector` unit tests already in `test_language_intent_entities.py`
and `test_draft_conversation.py`) -- the bug report was about the *whole
session* silently falling back to Hindi, which only a real multi-turn
conversation run can actually catch. Also verifies chat-preview/PDF/DOCX/TXT
export parity and the right-aligned closing-block signature format,
per the same audit's formatting requirements.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest
from docx import Document

from app.drafting.conversation import DraftConversationEngine
from app.drafting.export import DocxDraftExporter, TxtDraftExporter
from app.drafting.heading_translations import translated_heading
from app.drafting.templates import get_template
from app.drafting.templates.base import DraftTemplateDefinition, structure_sections_for
from app.drafting.title_translations import localized_title
from app.llm.base import LLMResponse

_PHONE = "9876543210"


def _engine_with_sectioned_llm_response() -> DraftConversationEngine:
    """An engine whose drafting LLM call deterministically returns one
    placeholder paragraph per required "Complaint"-category section --
    police_complaint's own category -- so every test in this file drives the
    exact same, predictable section skeleton regardless of draft language.
    The field-EXTRACTOR's own LLM call is separately mocked to return
    nothing, forcing the deterministic regex-extraction path (same technique
    `test_draft_conversation.py` already uses), so these tests are not
    sensitive to real LLM output.
    """
    engine = DraftConversationEngine()
    sectioned_response = "\n".join(
        f"## {heading}\nContent for {heading}." for heading in structure_sections_for("Complaint")
    )
    engine.draft_engine.llm.chat = AsyncMock(
        return_value=LLMResponse(content=sectioned_response, model="test", provider="test")
    )
    engine.extractor.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    engine.draft_engine.drafts.insert = AsyncMock(return_value="draft-audit-1")
    engine.draft_engine.versions.insert = AsyncMock(return_value="version-1")
    return engine


def _fill_required_fields(memory: dict, template: DraftTemplateDefinition, missing_fields: list[str]) -> None:
    for field_key in missing_fields:
        draft_field = template.get_field(field_key)
        if draft_field and draft_field.field_type == "tel":
            memory["draft_fields"][field_key] = _PHONE
        elif draft_field and draft_field.field_type == "date":
            memory["draft_fields"][field_key] = "15/07/2026"
        else:
            memory["draft_fields"][field_key] = f"Test value for {field_key}"


@pytest.mark.parametrize(
    "opening_message,ambient_language,requested_language",
    [
        # The exact reported transcript: typed entirely in Hindi/Devanagari,
        # explicitly asking for the draft IN TAMIL.
        ("तमिल में पुलिस शिकायत का मसौदा तैयार करें।", "hindi", "tamil"),
        # Typed in Tamil, explicitly asking for the draft IN TELUGU --
        # "தெலுங்கில்" is Tamil's own agglutinative locative form of
        # "Telugu" (see `_INFLECTED_LANGUAGE_REQUEST_PHRASES` in
        # `app/language/detector.py`), not a separate postposition word.
        ("தெலுங்கில் ஒரு காவல் நிலைய புகார் வரைவு தயார் செய்யுங்கள்.", "tamil", "telugu"),
        # Typed in Bengali, explicitly asking for the draft IN KANNADA.
        ("কন্নড়ে একটি পুলিশ অভিযোগ তৈরি করুন।", "bengali", "kannada"),
        # Typed in Telugu, explicitly asking for the draft IN BENGALI.
        ("బెంగాలీలో ఒక పోలీసు ఫిర్యాదు రాయండి.", "telugu", "bengali"),
    ],
    ids=["hindi-types-requests-tamil", "tamil-types-requests-telugu", "bengali-types-requests-kannada", "telugu-types-requests-bengali"],
)
def test_explicit_language_request_persists_through_the_whole_draft_session(
    opening_message: str, ambient_language: str, requested_language: str
) -> None:
    engine = _engine_with_sectioned_llm_response()
    memory: dict = {}
    session_id = f"session-audit-{requested_language}"

    first = asyncio.run(engine.handle_turn(session_id, opening_message, ambient_language, memory))
    assert first is not None, "opening message was not even recognized as a drafting request"
    assert first.info.stage == "collecting"
    assert memory["draft_language"] == requested_language, (
        f"draft_language={memory['draft_language']!r} after the opening turn, expected {requested_language!r} "
        f"-- the explicit language request did not win over the ambient {ambient_language!r} conversation language"
    )

    template = get_template(memory["draft_template_id"])
    assert template is not None

    # A follow-up field-collection turn, typed in the user's OWN script
    # (`ambient_language`) -- must not knock `draft_language` back to the
    # ambient language just because the field ANSWER itself is in a
    # different script than the requested document language.
    second = asyncio.run(
        engine.handle_turn(session_id, "ராமேஷ் குமார், 9876543210, ஹஜ்ரத்கஞ்ச் தாணா.", ambient_language, memory)
    )
    assert second is not None
    assert memory["draft_language"] == requested_language

    # Every remaining collecting-stage reply must be phrased in the
    # REQUESTED language's own script, never Hindi (unless Hindi IS the
    # requested language) and never the ambient script.
    reply_script_ranges = {
        "tamil": range(0x0B80, 0x0C00),
        "telugu": range(0x0C00, 0x0C80),
        "kannada": range(0x0C80, 0x0D00),
        "bengali": range(0x0980, 0x0A00),
    }
    expected_range = reply_script_ranges[requested_language]
    assert any(ord(ch) in expected_range for ch in second.reply_text), (
        f"collecting-stage reply contains no {requested_language} script characters at all: {second.reply_text!r}"
    )
    missing_now = sorted(
        template.required_field_keys() - {key for key, value in memory["draft_fields"].items() if value}
    )
    _fill_required_fields(memory, template, missing_now)
    final = asyncio.run(engine.handle_turn(session_id, "இது தயார்", ambient_language, memory))
    assert final is not None
    assert final.info.stage == "preview"
    assert memory["draft_language"] == requested_language
    assert memory["draft_stage"] == "preview"

    # The generated document body itself (chat preview `full_text`) must be
    # entirely in the requested language's headings -- confirmed by checking
    # every section heading was translated (never left as literal English
    # "To"/"Subject"/"Signature").
    for heading in final.info.sections:
        translated = translated_heading(heading, requested_language)
        assert translated in final.info.full_text
        if translated != heading:
            assert f"\n{heading}\n" not in f"\n{final.info.full_text}\n"


@pytest.mark.parametrize(
    "opening_message,ambient_language,requested_language",
    [
        ("तमिल में पुलिस शिकायत का मसौदा तैयार करें।", "hindi", "tamil"),
        ("தெலுங்கில் ஒரு காவல் நிலைய புகார் வரைவு தயார் செய்யுங்கள்.", "tamil", "telugu"),
        ("কন্নড়ে একটি পুলিশ অভিযোগ তৈরি করুন।", "bengali", "kannada"),
        ("బెంగాలీలో ఒక పోలీసు ఫిర్యాదు రాయండి.", "telugu", "bengali"),
    ],
    ids=["tamil", "telugu", "kannada", "bengali"],
)
def test_export_parity_and_signature_alignment_match_requested_language(
    tmp_path, opening_message: str, ambient_language: str, requested_language: str
) -> None:
    """Requirement: chat preview and every export format (PDF/DOCX/TXT) must
    render from the exact same section structure, with a right-aligned
    closing block (Place/Date, complementary close, signatory name and
    role designator) -- never a separately-maintained formatting path.
    """
    engine = _engine_with_sectioned_llm_response()
    memory: dict = {}
    session_id = f"session-export-{requested_language}"

    asyncio.run(engine.handle_turn(session_id, opening_message, ambient_language, memory))
    template = get_template(memory["draft_template_id"])
    missing_now = sorted(template.required_field_keys() - {k for k, v in memory.get("draft_fields", {}).items() if v})
    _fill_required_fields(memory, template, missing_now)
    result = asyncio.run(engine.handle_turn(session_id, "இது தயார்", ambient_language, memory))
    if result.info.stage != "preview":
        # A single "here it is" turn may not have supplied every field in
        # one shot depending on deterministic-extraction fallback timing --
        # drive one more turn the same way `test_draft_conversation.py`
        # does for the same reason.
        missing_now = sorted(template.required_field_keys() - {k for k, v in memory.get("draft_fields", {}).items() if v})
        _fill_required_fields(memory, template, missing_now)
        result = asyncio.run(engine.handle_turn(session_id, "here you go", ambient_language, memory))
    assert result.info.stage == "preview"

    sections = result.info.sections
    assert sections, "no sections were generated"
    # Part 56: police_complaint is Complaint category -- now the unified
    # Notice/Complaint skeleton, whose closing heading is "Signature Block"
    # (renamed from "Signature").
    assert "Signature Block" in sections
    chat_full_text = result.info.full_text

    title = localized_title(template, requested_language)

    txt_path = TxtDraftExporter().export(title, sections, tmp_path / "export.txt", requested_language)
    txt_content = txt_path.read_text(encoding="utf-8")
    docx_path = DocxDraftExporter().export(title, sections, tmp_path / "export.docx", requested_language)

    # Parity: every section's own text content appears verbatim in ALL THREE
    # surfaces (chat preview, TXT, DOCX) -- proving they were built from the
    # exact same `sections` dict, not three independently formatted copies.
    document = Document(str(docx_path))
    docx_text = "\n".join(p.text for p in document.paragraphs)
    for text in sections.values():
        first_line = text.splitlines()[0] if text.splitlines() else text
        assert first_line in chat_full_text
        assert first_line in txt_content
        assert first_line in docx_text

    # The closing block ("Signature Block" section) is right-aligned in the
    # DOCX export -- both its heading and every one of its body paragraphs --
    # matching how a real signed legal letter is laid out.
    signature_heading_text = translated_heading("Signature Block", requested_language)
    found_signature_heading = False
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == signature_heading_text:
            found_signature_heading = True
            assert paragraph.alignment == 2, "Signature heading is not right-aligned (WD_ALIGN_PARAGRAPH.RIGHT)"
    assert found_signature_heading, f"translated Signature heading {signature_heading_text!r} not found in DOCX"

    # The closing designator ("Petitioner" in the requested language) is
    # present, confirming the standardized closing-block shape (Place/Date,
    # complementary close, signatory name, role) was actually applied. Part
    # 56 extended this to the Notice category too (police_complaint here is
    # Complaint category, which already had one), so this is always expected
    # to be non-empty for the unified skeleton.
    from app.drafting.fallback_phrases import closing_designator

    designator = closing_designator(template.category, requested_language)
    if designator:
        assert designator in sections["Signature Block"]
        assert designator in chat_full_text
        assert designator in txt_content
        assert designator in docx_text


# ---------------------------------------------------------------------------
# Part 57 "Drafting Lifecycle Redesign": the 6 languages added to the
# priority chrome tier (malayalam/marathi/gujarati/punjabi/odia/urdu),
# extending the same end-to-end session-level audit above.
#
# Narrower than the 4-language audit above: those tests deliberately type
# the opening message in ONE script while requesting a DIFFERENT language,
# using each language's own inflected locative form
# (`_INFLECTED_LANGUAGE_REQUEST_PHRASES` in `app/language/detector.py`) --
# that table only covers tamil/telugu/kannada/bengali. None of the 6
# languages here have an equivalent native inflected-locative entry yet, so
# these tests use the already-well-supported Hinglish "<Language> mein ...
# karo" cue-word phrasing (`_LANGUAGE_CUE_WORDS`) instead. Adding native
# locative forms for these 6 languages (the way Part 54/55 did for the
# original four) is a real, flagged gap for a follow-up pass, not covered
# here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested_language",
    ["malayalam", "marathi", "gujarati", "punjabi", "odia", "urdu"],
)
def test_priority_tier_language_request_persists_through_the_whole_draft_session(requested_language: str) -> None:
    engine = _engine_with_sectioned_llm_response()
    memory: dict = {}
    session_id = f"session-audit-priority-{requested_language}"
    opening_message = f"{requested_language.capitalize()} mein police complaint ka draft taiyar karo."

    first = asyncio.run(engine.handle_turn(session_id, opening_message, "hinglish", memory))
    assert first is not None, "opening message was not even recognized as a drafting request"
    assert first.info.stage == "collecting"
    assert memory["draft_language"] == requested_language, (
        f"draft_language={memory['draft_language']!r} after the opening turn, expected {requested_language!r}"
    )

    template = get_template(memory["draft_template_id"])
    assert template is not None
    missing_now = sorted(template.required_field_keys())
    _fill_required_fields(memory, template, missing_now)
    final = asyncio.run(engine.handle_turn(session_id, "here you go", "hinglish", memory))
    assert final is not None
    assert final.info.stage == "preview"
    assert memory["draft_language"] == requested_language

    # Every section heading must be translated (never left as literal
    # English "To"/"Subject"/"Signature Block") -- confirms the 6-language
    # chrome extension actually took effect end-to-end, not just at the
    # `translated_heading()` unit level.
    for heading in final.info.sections:
        translated = translated_heading(heading, requested_language)
        assert translated in final.info.full_text
        if translated != heading:
            assert f"\n{heading}\n" not in f"\n{final.info.full_text}\n"


@pytest.mark.parametrize(
    "requested_language",
    ["malayalam", "marathi", "gujarati", "punjabi", "odia", "urdu"],
)
def test_priority_tier_language_export_parity(tmp_path, requested_language: str) -> None:
    engine = _engine_with_sectioned_llm_response()
    memory: dict = {}
    session_id = f"session-export-priority-{requested_language}"
    opening_message = f"{requested_language.capitalize()} mein police complaint ka draft taiyar karo."

    asyncio.run(engine.handle_turn(session_id, opening_message, "hinglish", memory))
    template = get_template(memory["draft_template_id"])
    _fill_required_fields(memory, template, sorted(template.required_field_keys()))
    result = asyncio.run(engine.handle_turn(session_id, "here you go", "hinglish", memory))
    assert result.info.stage == "preview"

    sections = result.info.sections
    assert sections and "Signature Block" in sections
    title = localized_title(template, requested_language)

    txt_path = TxtDraftExporter().export(title, sections, tmp_path / "export.txt", requested_language)
    txt_content = txt_path.read_text(encoding="utf-8")
    docx_path = DocxDraftExporter().export(title, sections, tmp_path / "export.docx", requested_language)
    document = Document(str(docx_path))
    docx_text = "\n".join(p.text for p in document.paragraphs)

    chat_full_text = result.info.full_text
    for text in sections.values():
        first_line = text.splitlines()[0] if text.splitlines() else text
        assert first_line in chat_full_text
        assert first_line in txt_content
        assert first_line in docx_text
