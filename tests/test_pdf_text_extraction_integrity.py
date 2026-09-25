"""D2: regression coverage for the confirmed WeasyPrint PDF text-extraction
corruption for the Indic-script PDFs `PdfDraftExporter` generates, and for
`app.drafting._weasyprint_indic_cmap_patch`'s fix.

See `app.drafting.pdf_text_integrity`'s module docstring for the full root
cause and the fix's exact, verified scope. Short version: a WeasyPrint bug
in its glyph-to-`/ToUnicode` CMap mapping for pre-base reordered dependent
vowel signs corrupted text extracted from generated PDFs in two ways -- a
silent drop (`pypdf`) and a stray, out-of-script character (`pymupdf`). The
shipped patch ELIMINATES the stray-character symptom entirely for BOTH
libraries (verified below as a real, passing assertion) and, as of
2026-09-23, recovers the EXACT original word for `pypdf` in every case
except two separate, documented, unfixed collision classes (see
`_weasyprint_indic_cmap_patch`'s "CURRENT STATE" docstring section) --
still a real but incomplete improvement, verified below as a documented
`xfail(strict=True)` for the document-wide, zero-exceptions claim.
`pymupdf` is NOT asserted for exact recovery (it still duplicates a
different, smaller set of words under this fix -- see that module's "WHY
ZWS INSTEAD OF DUPLICATION" section).

Neither this bug nor the patch affects the PDF's VISUAL rendering (already
correct, Part 53/57) or this app's own re-ingestion of a PDF it generated
(`DocumentExtractionService` rasterizes and OCRs every PDF page rather than
reading its text layer) -- this affects anyone extracting text from the PDF
outside this app.

Devanagari/Hindi is the only script independently confirmed to exercise this
bug on THIS host/font (Nirmala UI) with THIS exact text -- the bug is
glyph-ID-reuse-dependent (see the module docstring), so it does not reproduce
uniformly for every sentence or every complex script this app renders PDFs
in. Other complex scripts (Tamil, Telugu, Kannada, Bengali, ...) share the
same WeasyPrint/Pango/HarfBuzz pipeline and the same class of risk, but are
deliberately NOT asserted here without their own confirmed repro -- an
unconfirmed test for them would be guesswork, not a pinned defect.

`_HINDI_SECTIONS` is specifically the document that reproduces the
glyph-ID-COLLISION case ("मोटरसाइकिल"'s "कि" cluster reuses a glyph ID
already cached, from "की" in the same document, under different text) --
the harder of the two cases the patch had to handle. Asserting the
stray-character fix against exactly this document, rather than an isolated
single-word case, is what makes `test_extracted_hindi_text_has_no_out_of_
script_characters` below a meaningful regression test rather than a lucky
easy case.
"""

import pymupdf
import pytest
from pypdf import PdfReader

from app.drafting.export import PdfDraftExporter, _script_for
from app.drafting.pdf_text_integrity import detect_corruption, detect_missing_words

_HINDI_SECTIONS = {
    "Recipient": "श्रीमान थाना प्रभारी,\nनई दिल्ली",
    "Subject": "चोरी की शिकायत",
    "Details": "दिनांक 12 जनवरी को मेरी मोटरसाइकिल चोरी हो गई। कृपया प्राथमिकी दर्ज करें।",
    "Signature": "भवदीय,\nराम कुमार",
}
_HINDI_SOURCE_TEXT = "\n".join(_HINDI_SECTIONS.values())


def _export_hindi_pdf(tmp_path):
    output_path = tmp_path / "hindi.pdf"
    PdfDraftExporter().export("Test Complaint", _HINDI_SECTIONS, output_path, language="hindi")
    return output_path


def test_extracted_hindi_text_has_no_out_of_script_characters(tmp_path) -> None:
    """The FIXED symptom: a real, passing regression test, not an `xfail` --
    if this starts failing again, the D2 patch (or its `apply_patch()`
    version guard) has regressed and must be looked at before anything else
    in this file.
    """
    output_path = _export_hindi_pdf(tmp_path)
    findings = []
    reader = PdfReader(output_path)
    pypdf_text = "\n".join(page.extract_text() for page in reader.pages)
    findings += detect_corruption(pypdf_text, _script_for("hindi"))
    with pymupdf.open(output_path) as document:
        pymupdf_text = "\n".join(page.get_text() for page in document)
    findings += detect_corruption(pymupdf_text, _script_for("hindi"))
    assert not findings, [
        f"unexpected {f.character} ({f.codepoint}) near {f.context!r}" for f in findings
    ]


@pytest.mark.xfail(
    reason="D2: pypdf now exactly recovers every word in this document EXCEPT three, in two "
    "distinct, separately-confirmed root causes (see app.drafting._weasyprint_indic_cmap_patch's "
    "'CURRENT STATE' docstring section for the full detail, current as of 2026-09-23): "
    "(1) a glyph-ID collision affecting 'मोटरसाइकिल'/'प्राथमिकी', needing the font-subsetting-level "
    "unique-glyph-ID fix described in app.drafting.pdf_text_integrity's docstring, not attempted "
    "here; (2) a separate, previously-undocumented stray-prefix bug affecting 'दर्ज'/'भवदीय', "
    "unrelated to vowel reordering, root cause not yet investigated. strict=True so a future "
    "complete fix must be noticed and this marker removed, not silently left stale.",
    strict=True,
)
def test_extracted_hindi_text_exactly_recovers_every_word(tmp_path) -> None:
    output_path = _export_hindi_pdf(tmp_path)
    reader = PdfReader(output_path)
    extracted = "\n".join(page.extract_text() for page in reader.pages)
    findings = detect_missing_words(extracted, _HINDI_SOURCE_TEXT)
    assert not findings, [f.word for f in findings]
