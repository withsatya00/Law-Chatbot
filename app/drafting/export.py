import base64
import html as html_lib
import mimetypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import structlog

from app.core.config import settings
from app.core.exceptions import UnsupportedExportError
from app.core.windows_runtime import repair_windows_host_env

log = structlog.get_logger(__name__)

# Part 53 "PDF Hindi Rendering + Professional Layout Audit": on Windows,
# WeasyPrint's native GTK3 libraries (Pango/Cairo/GDK-Pixbuf/fontconfig)
# must be registered via `os.add_dll_directory()` BEFORE `import weasyprint`
# -- confirmed directly on this machine: with the GTK3 runtime correctly
# installed and even on `PATH`, `import weasyprint` still failed with
# `OSError: cannot load library 'libgobject-2.0-0'` (error 0x7e, i.e.
# ERROR_MOD_NOT_FOUND for one of libgobject's OWN dependencies) until this
# was added. Root cause: Python 3.8+ changed Windows DLL loading to a safer
# default (`LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`) that no longer searches
# `PATH` to resolve a loaded DLL's transitive dependencies -- only `PATH`
# itself, or directories explicitly registered via `add_dll_directory`, are
# searched for THOSE. Tries `settings.gtk_runtime_bin_path` first (operator
# override), then the two most common install locations (the official
# "GTK3 Runtime for Windows" installer, and MSYS2's ucrt64 environment,
# which is what this was actually verified against). A no-op on non-Windows
# platforms (Linux/Docker installs these as regular shared libraries via the
# system package manager, where the dynamic linker's normal search already
# covers this).
if sys.platform == "win32":
    # Must precede both `add_dll_directory` and the lazy `import weasyprint` in
    # `_html_renderer`: an injected SSLKEYLOGFILE device path makes OpenSSL --
    # pulled in transitively by the GTK stack -- abort the whole process from C
    # rather than raise, which took down the API and every pytest run that
    # rendered a PDF. See `app/core/windows_runtime.py`.
    repair_windows_host_env()

    _GTK_BIN_CANDIDATES = [
        settings.gtk_runtime_bin_path,
        r"C:\Program Files\GTK3-Runtime Win64\bin",
        r"C:\msys64\ucrt64\bin",
        r"C:\msys64\mingw64\bin",
    ]
    for _candidate in _GTK_BIN_CANDIDATES:
        if _candidate and Path(_candidate).is_dir():
            try:
                os.add_dll_directory(_candidate)
            except OSError:  # pragma: no cover - defensive, e.g. already added
                pass

from app.core.optional_deps import load_attr
from app.drafting.advocate_register import (
    FLOWING_BODY_ORDER,
    FLOWING_KEEP_HEADING,
    FLOWING_LEAD_HEADINGS,
)
from app.drafting.heading_translations import translated_heading


# Phase 1 item 8: python-docx was imported at module scope here, and this
# module is reachable from `ChatService` (chat_service -> drafting.conversation
# -> drafting.engine -> drafting.export). A host without python-docx therefore
# could not start plain legal chat, a feature that never produces a .docx.
# Bound at export time instead, so a missing library degrades to
# "DOCX export is unavailable, here is what to install" on the one endpoint
# that needs it. `load_attr` caches, so the repeated lookups below are free
# after the first call.
def _docx() -> tuple[Any, Any, Any, Any]:
    """python-docx's `Document`, `qn` and the two unit helpers, as one tuple."""
    return (
        load_attr("docx", "Document", feature="DOCX export"),
        load_attr("docx.oxml.ns", "qn", feature="DOCX export"),
        load_attr("docx.shared", "Inches", feature="DOCX export"),
        load_attr("docx.shared", "Pt", feature="DOCX export"),
    )


def _html_renderer() -> Any:
    """Load the optional PDF renderer only when a PDF is requested."""
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as exc:
        raise UnsupportedExportError(
            "PDF export is unavailable because the server's WeasyPrint/GTK runtime is not installed. "
            "DOCX, TXT, and RTF exports remain available.",
            {"format": "pdf", "reason": str(exc)},
        ) from exc
    # D2: partially mitigates WeasyPrint's own Indic-vowel-sign /ToUnicode
    # CMap corruption (see app/drafting/_weasyprint_indic_cmap_patch.py and
    # app/drafting/pdf_text_integrity.py for what this does and does not
    # fix) -- applied here, right after WeasyPrint itself is confirmed
    # importable, so every PDF export goes through the corrected code path.
    # Idempotent and fails safe (never raises) if the installed WeasyPrint
    # version has moved.
    from app.drafting._weasyprint_indic_cmap_patch import apply_patch
    apply_patch()
    return HTML

# Part 53 "PDF Hindi Rendering + Professional Layout Audit": PDF generation
# moved from reportlab to WeasyPrint.
#
# ROOT CAUSE of the Hindi/Tamil/Telugu/Kannada/Bengali corruption this
# replaces: reportlab has no OpenType complex-script shaping engine. It maps
# each Unicode codepoint straight through the font's `cmap` table to a single
# glyph and lays glyphs out in logical (storage) order -- it never runs GSUB
# (ligature substitution, e.g. Devanagari क्ष/त्र/ज्ञ) or GPOS/reordering
# (e.g. a pre-base matra like ि, which is stored AFTER its consonant in
# Unicode but must render BEFORE it). Nirmala UI -- the font already in use
# -- is a proper pan-Indic font with full shaping tables; reportlab simply
# never invoked them, for ANY of the five scripts this app supports
# (Devanagari, Tamil, Telugu, Kannada, Bengali are all complex/abugida
# scripts with the same class of requirement). Since a PDF's content stream
# bakes in exact glyph IDs at generation time (a viewer cannot re-shape
# already-written glyphs), this could only be fixed at generation time.
#
# WeasyPrint renders HTML/CSS through Pango+HarfBuzz+FreeType, the same
# shaping stack browsers use, so it shapes all five scripts correctly for
# free. It requires the GTK3 runtime's native libraries (Pango/Cairo/
# GDK-Pixbuf/fontconfig) to be present on the host -- confirmed working after
# installing the GTK3 Runtime for Windows (`libgobject-2.0-0` etc. resolve).
# On Linux, `apt-get install libpango-1.0-0 libpangocairo-1.0-0` (or the
# equivalent) provides the same libraries.
#
# DOCX and TXT export are UNCHANGED (both were already correct: Word/
# LibreOffice shape text themselves when DISPLAYING a .docx, and plain text
# carries no glyph rendering at all -- the corruption was specific to the
# one format, PDF, where WE controlled exact glyph placement at generation
# time).
#
# D2 (separate, still-open defect): correct on-screen rendering above does
# NOT mean correct TEXT EXTRACTION back out of the generated PDF -- a
# WeasyPrint bug in its own glyph-to-`/ToUnicode` CMap mapping corrupts
# copy-paste/`pypdf`/`pymupdf` extraction of these same pre-base-reordered
# Indic vowel signs, confirmed live 2026-09-23. See
# `app.drafting.pdf_text_integrity`'s module docstring for the full root
# cause, `tests/test_pdf_text_extraction_integrity.py` for the pinned
# regression, and `scripts/verify_pdf_text_extraction.py` to check any PDF.

# Windows 10/11 ships Nirmala UI (a TrueType Collection covering Devanagari
# and the other four scripts this app localizes into) as an installed system
# font -- referenced here by family name via CSS `local()`, which WeasyPrint/
# Pango resolves through the OS's own font database, rather than pointing at
# the raw .ttc file (a TrueType Collection has no single well-supported way
# to select "face index 0" through a plain CSS `url()`). `pdf_unicode_font_path`
# lets a Linux/Docker deployment without Nirmala UI installed point at an
# actual TTF instead (e.g. Noto Sans Devanagari via `fonts-noto-core`) --
# tried first via `url()`, since an explicit operator-configured override is
# more trustworthy than guessing at what's on the system.
_FONT_FAMILY = "DraftUnicodeFont"

# Part 57 "Drafting Lifecycle Redesign": script-family routing, replacing the
# single hardcoded "Nirmala UI" applied to every language regardless of
# script. Each language maps to a script family; each family maps to the
# system font name(s) to try via CSS `local()`. Devanagari/Bengali/Gurmukhi/
# Gujarati/Odia/Tamil/Telugu/Kannada/Malayalam all resolve to the SAME
# physical font (Nirmala UI is one pan-Indic TTC covering all nine) -- this
# table doesn't change their rendering at all, it exists so the two genuinely
# different cases can be expressed cleanly:
#   - perso_arabic (Urdu/Sindhi/Kashmiri): Tahoma, the only Arabic-script-
#     capable font installed on this server (confirmed via `Get-ChildItem
#     C:\\Windows\\Fonts`) -- Naskh style, not the Nastaliq calligraphic style
#     conventional for Urdu, but WeasyPrint/Pango still shape it correctly
#     (proper joining forms, RTL). Swappable via `pdf_script_font_paths`.
#   - ol_chiki (Santali) / meetei_mayek (Manipuri): NO font is installed for
#     either script on this server at all -- an empty tuple here is a
#     deliberate signal `PdfDraftExporter.export()` checks and raises
#     `UnsupportedExportError` on, rather than silently emitting a PDF full
#     of blank glyph boxes.
_SCRIPT_FONTS: dict[str, tuple[str, ...]] = {
    "devanagari": ("Nirmala UI", "Nirmala Text"),
    "bengali": ("Nirmala UI", "Nirmala Text"),
    "gurmukhi": ("Nirmala UI", "Nirmala Text"),
    "gujarati": ("Nirmala UI", "Nirmala Text"),
    "odia": ("Nirmala UI", "Nirmala Text"),
    "tamil": ("Nirmala UI", "Nirmala Text"),
    "telugu": ("Nirmala UI", "Nirmala Text"),
    "kannada": ("Nirmala UI", "Nirmala Text"),
    "malayalam": ("Nirmala UI", "Nirmala Text"),
    "perso_arabic": ("Tahoma",),
    "ol_chiki": (),
    "meetei_mayek": (),
    "latin": ("Times New Roman", "Nirmala UI", "Nirmala Text"),
}

# DOCX-only: the font NAME embedded for a script with no server-side font at
# all -- Word/LibreOffice substitutes fonts at OPEN time on the end user's
# own machine, a much weaker requirement than server-side PDF glyph
# rasterization, so a DOCX can still correctly name "Noto Sans Ol Chiki" even
# though this server can't render it into a PDF.
_MISSING_FONT_DOCX_NAMES: dict[str, str] = {"ol_chiki": "Noto Sans Ol Chiki", "meetei_mayek": "Noto Sans Meetei Mayek"}

_LANGUAGE_SCRIPT: dict[str, str] = {
    "hindi": "devanagari", "marathi": "devanagari", "sanskrit": "devanagari",
    "konkani": "devanagari", "nepali": "devanagari", "maithili": "devanagari",
    "dogri": "devanagari", "bodo": "devanagari",
    "bengali": "bengali", "assamese": "bengali",
    "punjabi": "gurmukhi",
    "gujarati": "gujarati",
    "odia": "odia",
    "tamil": "tamil",
    "telugu": "telugu",
    "kannada": "kannada",
    "malayalam": "malayalam",
    "urdu": "perso_arabic", "sindhi": "perso_arabic", "kashmiri": "perso_arabic",
    "santali": "ol_chiki",
    "manipuri": "meetei_mayek",
    "english": "latin", "hinglish": "latin",
}

# Scripts requiring right-to-left layout -- Urdu/Sindhi/Kashmiri (Perso-
# Arabic) only; every other supported script is left-to-right.
_RTL_SCRIPTS = {"perso_arabic"}


def _script_for(language: str) -> str:
    # Unrecognized language strings default to "devanagari" -- matches the
    # OLD unconditional behavior (every language always used Nirmala UI)
    # exactly, for any language not in the table above.
    return _LANGUAGE_SCRIPT.get(language, "devanagari")


def _is_rtl(language: str) -> bool:
    return _script_for(language) in _RTL_SCRIPTS


def _script_font_override(script: str) -> str:
    return settings.pdf_script_font_paths.get(script) or settings.pdf_unicode_font_path


def _font_face_css(language: str) -> str:
    script = _script_for(language)
    sources = []
    override = _script_font_override(script)
    if override and Path(override).exists():
        sources.append(f"url('{Path(override).resolve().as_uri()}')")
    for family in _SCRIPT_FONTS.get(script, ()):
        sources.append(f"local('{family}')")
    if not sources:
        # Reached only by the page-count probe for a script with neither an
        # override nor a system font (`PdfDraftExporter.export()` itself
        # raises before ever reaching here) -- degrades to the Pango/browser
        # default rather than an invalid empty `src:`.
        return f"@font-face {{ font-family: '{_FONT_FAMILY}'; }}"
    return f"@font-face {{ font-family: '{_FONT_FAMILY}'; src: {', '.join(sources)}; }}"


def _docx_font_name(language: str) -> str:
    script = _script_for(language)
    families = _SCRIPT_FONTS.get(script, ())
    if families:
        return families[0]
    return _MISSING_FONT_DOCX_NAMES.get(script, "Nirmala UI")


def _set_rtl(paragraph: Any) -> None:
    """Marks a python-docx paragraph right-to-left (`w:bidi`) -- python-docx
    has no first-class API for this, so it's set directly on the paragraph's
    XML properties, the same way Word itself represents "Right-to-left
    paragraph direction" in the underlying OOXML.
    """
    qn = load_attr("docx.oxml.ns", "qn", feature="DOCX export")
    paragraph_properties = paragraph._p.get_or_add_pPr()
    paragraph_properties.append(paragraph_properties.makeelement(qn("w:bidi"), {}))


# Fixed filing typography. The earlier exporter manufactured length with a
# decorative cover and a forced signature-only page. That looked unlike an
# authority-facing Indian application and produced conspicuously empty pages.
# Pagination now follows substantive content; the drafting engine, not blank
# stationery, is responsible for the three-page content target.
_LAYOUT: dict[str, float] = {
    # Matches the Delhi High Court's Practice Direction on document
    # formatting (Registrar General, in force w.e.f. 01.11.2022): "A4 size
    # paper... for all kinds of pleadings contained in petitions,
    # affidavits, applications or other documents etc... font -- Times New
    # Roman, font size 14, in 1.5 line spacing... with margin of 4 cm on
    # left & right and 2 cm on top & bottom." Applied uniformly (title,
    # heading, and body all 14pt) -- the Direction names one font size for
    # the document's own text and a SEPARATE, smaller size only for
    # quotations/indents (font 12, single spacing), which this app has no
    # distinct "quotation block" concept for yet; a real title/heading/body
    # size cascade is not part of the Direction and isn't invented here.
    # Times New Roman was already `_SCRIPT_FONTS["latin"]`'s first choice
    # for English/Hinglish (see below) -- no change needed there.
    "margin_v": 2 / 2.54, "margin_h": 4 / 2.54, "title": 14, "heading": 14,
    "body": 14, "leading_ratio": 1.5, "para_gap": 7, "heading_gap": 10,
}

# Headings right-aligned as the signature block, across every skeleton: the
# unified Notice/Complaint skeleton renamed "Signature" to "Signature Block"
# (Part 56), but Affidavit/Application (RTI) still use the original
# "Signature" heading.
_SIGNATURE_HEADINGS = {"Signature", "Signature Block"}

_PAGE_LABELS: dict[str, tuple[str, str]] = {
    "hindi": ("पृष्ठ", "का"),
    "marathi": ("पृष्ठ", "पैकी"),
    "bengali": ("পৃষ্ঠা", "এর"),
    "gujarati": ("પૃષ્ઠ", "માંથી"),
    "punjabi": ("ਪੰਨਾ", "ਵਿੱਚੋਂ"),
    "tamil": ("பக்கம்", "இல்"),
    "telugu": ("పేజీ", "లో"),
    "kannada": ("ಪುಟ", "ರಲ್ಲಿ"),
    "malayalam": ("പേജ്", "ൽ"),
    "odia": ("ପୃଷ୍ଠା", "ରୁ"),
    "urdu": ("صفحہ", "از"),
}


def _page_labels(language: str) -> tuple[str, str]:
    return _PAGE_LABELS.get(language, ("Page", "of"))


def _recipient_label(language: str) -> str:
    return "To" if language in {"english", "hinglish"} else translated_heading("Recipient", language)

_REVIEW_NOTICE = {
    "english": "AI-assisted legal draft. Review and approval by a qualified advocate is required before filing, signing, service, or submission.",
    "hindi": "यह एआई-सहायित कानूनी मसौदा है। दाखिल, हस्ताक्षरित, तामील या प्रस्तुत करने से पहले योग्य अधिवक्ता द्वारा समीक्षा और अनुमोदन आवश्यक है।",
    "hinglish": "Yeh AI-assisted legal draft hai. Filing, signing, service ya submission se pehle qualified advocate se review aur approval zaroor karayein.",
    "marathi": "हा एआय-सहाय्यित कायदेशीर मसुदा आहे. दाखल, स्वाक्षरी किंवा सादर करण्यापूर्वी पात्र वकिलाकडून तपासणी आवश्यक आहे.",
    "gujarati": "આ એઆઈ-સહાયિત કાનૂની મુસદ્દો છે. દાખલ, સહી અથવા રજૂ કરતાં પહેલાં લાયક વકીલ દ્વારા સમીક્ષા જરૂરી છે.",
    "bengali": "এটি এআই-সহায়িত আইনি খসড়া। দাখিল, স্বাক্ষর বা জমা দেওয়ার আগে যোগ্য আইনজীবীর পর্যালোচনা প্রয়োজন।",
    "punjabi": "ਇਹ ਏਆਈ-ਸਹਾਇਤ ਕਾਨੂੰਨੀ ਮਸੌਦਾ ਹੈ। ਦਾਖ਼ਲ, ਦਸਤਖ਼ਤ ਜਾਂ ਜਮ੍ਹਾਂ ਕਰਨ ਤੋਂ ਪਹਿਲਾਂ ਯੋਗ ਵਕੀਲ ਤੋਂ ਸਮੀਖਿਆ ਲਾਜ਼ਮੀ ਹੈ।",
    "tamil": "இது ஏஐ உதவியுடன் தயாரிக்கப்பட்ட சட்ட வரைவு. தாக்கல், கையொப்பம் அல்லது சமர்ப்பிப்புக்கு முன் தகுதியான வழக்கறிஞரின் பரிசீலனை அவசியம்.",
    "telugu": "ఇది ఏఐ సహాయంతో రూపొందించిన న్యాయ ముసాయిదా. దాఖలు, సంతకం లేదా సమర్పణకు ముందు అర్హతగల న్యాయవాది సమీక్ష అవసరం.",
    "kannada": "ಇದು ಎಐ ಸಹಾಯದಿಂದ ಸಿದ್ಧಪಡಿಸಿದ ಕಾನೂನು ಕರಡು. ಸಲ್ಲಿಕೆ, ಸಹಿ ಅಥವಾ ದಾಖಲಿಸುವ ಮೊದಲು ಅರ್ಹ ವಕೀಲರ ಪರಿಶೀಲನೆ ಅಗತ್ಯ.",
    "malayalam": "ഇത് എഐ സഹായത്തോടെ തയ്യാറാക്കിയ നിയമ കരടാണ്. ഫയൽ ചെയ്യുന്നതിനോ ഒപ്പിടുന്നതിനോ സമർപ്പിക്കുന്നതിനോ മുമ്പ് യോഗ്യനായ അഭിഭാഷകന്റെ പരിശോധന ആവശ്യമാണ്.",
    "urdu": "یہ اے آئی کی مدد سے تیار کردہ قانونی مسودہ ہے۔ داخل کرنے، دستخط یا جمع کرانے سے پہلے مستند وکیل سے جائزہ ضروری ہے۔",
}


def _review_notice(language: str) -> str:
    return _REVIEW_NOTICE.get(language, _REVIEW_NOTICE["english"])


# Labels for the printed notarization attestation block, in the languages this
# app fully localizes. Falls back to English elsewhere -- an English label over
# correct data is safe; a mistranslated legal label is not.
_NOTARIZATION_LABELS: dict[str, dict[str, str]] = {
    "english": {
        "title": "NOTARIZATION RECORD",
        "notary": "Notary",
        "registration": "Registration No.",
        "jurisdiction": "Jurisdiction",
        "date": "Notarized on",
        "hash": "Document SHA-256",
        "verify": "Verify at",
    },
    "hindi": {
        "title": "नोटरीकरण अभिलेख",
        "notary": "नोटरी",
        "registration": "पंजीकरण संख्या",
        "jurisdiction": "क्षेत्राधिकार",
        "date": "नोटरीकरण दिनांक",
        "hash": "दस्तावेज़ SHA-256",
        "verify": "सत्यापन हेतु",
    },
}


def _notarization_labels(language: str) -> dict[str, str]:
    return _NOTARIZATION_LABELS.get(language, _NOTARIZATION_LABELS["english"])


def _notarization_rows(
    attestation: "NotarizationAttestation", language: str
) -> tuple[str, list[tuple[str, str]]]:
    """`(title, [(label, value), ...])` for the printed attestation block.

    Shared by every exporter. Until Phase 1 only `PdfDraftExporter` rendered an
    attestation at all, so a notarized document downloaded as DOCX, TXT or RTF
    came out with no notary, no registration number, no document hash and no
    verification URL -- indistinguishable from an ordinary unnotarized draft,
    and with nothing on it for a recipient to check. The rows are built once,
    here, so that cannot silently happen again for one format.
    """
    labels = _notarization_labels(language)
    return labels["title"], [
        (labels["notary"], attestation.notary_name),
        (labels["registration"], attestation.notary_registration_number),
        (labels["jurisdiction"], attestation.jurisdiction_state),
        (labels["date"], attestation.notarized_on),
        (labels["hash"], attestation.document_hash),
        (labels["verify"], attestation.verification_url),
    ]


def _notarization_block_text(attestation: "NotarizationAttestation", language: str) -> list[str]:
    """The attestation as plain lines, for the non-HTML exporters.

    Carries no QR: the QR is an inline SVG (see `app/notarization/qr.py`) and
    only the PDF pipeline renders vector artwork. It encodes nothing but the
    verification URL, which IS present here -- so a TXT/RTF recipient can still
    verify the record, just by typing the link rather than scanning it.
    """
    title, rows = _notarization_rows(attestation, language)
    lines = ["", title, "-" * len(title)]
    lines.extend(f"{label}: {value}" for label, value in rows)
    return lines


def _notarization_block_html(attestation: "NotarizationAttestation", language: str) -> str:
    """The printed attestation block for a NOTARIZED document.

    Reached only when `ExportOptions.notarization` is set, which the
    notarization service does only for a document a verified notary approved.
    The wording states that this is the PLATFORM's record of the
    notarization -- it never claims to be, or to substitute for, the notary's
    own seal and signature.
    """
    labels = _notarization_labels(language)
    title, rows = _notarization_rows(attestation, language)
    # The hash and the verification URL are long unbroken strings; they keep
    # the `hash` class so the stylesheet can wrap them mid-token.
    unbroken = {labels["hash"], labels["verify"]}
    parts = ["<section class='notarization-block'>", f"<h2>{_escape(title)}</h2>"]
    parts.extend(
        f"<p class='row{' hash' if label in unbroken else ''}'>"
        f"<strong>{_escape(label)}:</strong> {_escape(value)}</p>"
        for label, value in rows
    )
    if attestation.verification_qr_svg:
        # Inline SVG, already built by `app/notarization/qr.py`, which refuses
        # to produce one for any non-notarized status.
        parts.append(f"<div class='notarization-qr'>{attestation.verification_qr_svg}</div>")
    parts.append("</section>")
    return "\n".join(parts)


@dataclass(frozen=True)
class ExportOptions:
    # Off by default by product decision: a "DRAFT" wash across every page made
    # the finished document look provisional to whoever it is handed to (a
    # police station, a landlord, a bank), which is not what the user wants
    # from a document they are about to sign and submit. The AI-generated-draft
    # disclaimer that accompanies every export already carries the
    # "get this reviewed by an advocate" message, in words, without defacing
    # the page. Callers that DO want one still pass `watermark=True`.
    watermark: bool = False
    sign: bool = False
    watermark_text: str = settings.draft_watermark_text
    watermark_image_path: str = settings.draft_watermark_image_path
    signature_image_path: str = settings.draft_signature_image_path
    signing_certificate_path: str = settings.draft_signing_certificate_path
    signing_key_path: str = settings.draft_signing_key_path
    signing_key_passphrase: str = settings.draft_signing_key_passphrase
    signing_reason: str = settings.draft_signing_reason
    include_header_footer: bool = True
    document_version: int = 1
    generated_on: str = ""
    # Set ONLY for a document whose status is `notarized` (see
    # `app/notarization/`). Its presence is what adds the attestation block
    # and the verification QR to the export. Everything in it comes from the
    # notarization record written by a verified notary's approval -- nothing
    # here is generated, and there is no code path that populates it from a
    # draft, a signature, or an upload.
    notarization: "NotarizationAttestation | None" = None
    # Advocate-register redesign: renders as one continuous flowing letter
    # (no "Facts of the Case"/"Legal Position"/"Prayer" headings -- see
    # `advocate_register.FLOWING_LETTER_DRAFTS`'s own docstring for the real
    # complaint this was confirmed against) instead of the normal headed
    # layout. Set by the caller, which already knows the template's
    # `draft_id` -- this module works from `sections`/`template_name` alone
    # and has no `draft_id` of its own to check.
    flowing_letter: bool = False


@dataclass(frozen=True)
class NotarizationAttestation:
    """Facts copied from an approved notarization record, for printing.

    Deliberately holds no notary SEAL or signature image: this platform never
    generates either. The block it produces states that the notarization is
    recorded on this platform and directs the reader to verify it, which is
    the only claim we are entitled to make.
    """

    notary_name: str
    notary_registration_number: str
    jurisdiction_state: str
    notarized_on: str
    document_hash: str
    verification_url: str
    verification_qr_svg: str


def _file_data_uri(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"Configured export image does not exist: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _sign_pdf(input_path: Path, output_path: Path, options: ExportOptions) -> None:
    if not options.signing_certificate_path or not options.signing_key_path:
        raise ValueError(
            "PDF signing requires DRAFT_SIGNING_CERTIFICATE_PATH and DRAFT_SIGNING_KEY_PATH."
        )
    try:
        from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
        from pyhanko.sign import signers
    except ImportError as exc:  # pragma: no cover - dependency installation issue
        raise RuntimeError("Install pyHanko to enable cryptographic PDF signing.") from exc

    # pyHanko ships a `py.typed` marker but `SimpleSigner.load` itself carries
    # no annotations, so strict mypy rejects the call as untyped. This is a
    # third-party stub gap, not a defect here: the argument types are checked
    # by pyHanko at runtime and the call is exercised by the PDF-signing tests.
    # Scoped to this one call rather than relaxing `disallow_untyped_calls`.
    signer = signers.SimpleSigner.load(  # type: ignore[no-untyped-call]
        cert_file=options.signing_certificate_path,
        key_file=options.signing_key_path,
        key_passphrase=options.signing_key_passphrase.encode() or None,
    )
    metadata = signers.PdfSignatureMetadata(field_name="LegalDraftSignature", reason=options.signing_reason)
    with input_path.open("rb") as source, output_path.open("wb") as destination:
        writer = IncrementalPdfFileWriter(source)
        signers.PdfSigner(metadata, signer=signer).sign_pdf(writer, output=destination)



class DraftExporter(Protocol):
    """The one signature every format exporter implements.

    Declared so the dispatch table in `LegalDraftEngine` and
    `NotarizationService` is typed (it was `dict[str, object]` with an
    `attr-defined` ignore on the call), and so a new exporter that quietly
    drops a parameter -- `options`, say, which carries the notarization
    attestation -- is a type error rather than a silently missing block in one
    format's output.
    """

    def export(
        self,
        template_name: str,
        sections: dict[str, str],
        output_path: Path,
        language: str = ...,
        options: "ExportOptions | None" = ...,
    ) -> Path: ...


class PdfDraftExporter:
    def export(
        self, template_name: str, sections: dict[str, str], output_path: Path, language: str = "english",
        options: ExportOptions | None = None,
    ) -> Path:
        options = options or ExportOptions()
        script = _script_for(language)
        if not _SCRIPT_FONTS.get(script) and not _script_font_override(script):
            # Part 57: Ol Chiki (Santali) / Meetei Mayek (Manipuri) have no
            # usable font on this server -- fail loudly here rather than
            # silently emitting a PDF full of blank glyph boxes. DOCX/TXT
            # export for the same draft still succeed (see
            # `DocxDraftExporter`/`TxtDraftExporter` below, neither of which
            # needs server-side glyph rasterization).
            raise UnsupportedExportError(
                f"PDF export isn't available for '{language}' yet -- no font is installed for its script on "
                "this server. DOCX and TXT export are still available for this draft.",
                {"language": language, "script": script},
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html_document = self._build_html(template_name, sections, _LAYOUT, language, options)
        HTML = _html_renderer()
        if options.sign:
            unsigned_path = output_path.with_suffix(".unsigned.pdf")
            HTML(string=html_document).write_pdf(str(unsigned_path))
            try:
                _sign_pdf(unsigned_path, output_path, options)
            finally:
                unsigned_path.unlink(missing_ok=True)
        else:
            HTML(string=html_document).write_pdf(str(output_path))
        return output_path

    def _page_count(
        self, template_name: str, sections: dict[str, str], level: dict[str, float] | None = None, language: str = "english",
        options: ExportOptions | None = None,
    ) -> int:
        html_document = self._build_html(template_name, sections, level or _LAYOUT, language, options or ExportOptions())
        try:
            return len(_html_renderer()(string=html_document).render().pages)
        except UnsupportedExportError:
            raise
        except Exception:  # pragma: no cover  # noqa: BLE001 - a page-count PROBE only; an unrenderable probe falls back to 1 page and the real render reports the error
            log.warning("pdf_page_count_probe_failed", template_name=template_name)
            return 1

    @staticmethod
    def _paragraph_html(text: str) -> list[str]:
        parts = []
        for paragraph in text.split("\n\n") or [""]:
            lines = [_escape(line) for line in paragraph.splitlines()]
            parts.append(f"<p>{'<br/>'.join(lines) or '&nbsp;'}</p>")
        return parts

    def _flowing_body_html(self, sections: dict[str, str], language: str) -> list[str]:
        """Renders `sections` as one continuous flowing letter (see
        `advocate_register.FLOWING_LETTER_DRAFTS`'s own docstring for the
        real complaint this was confirmed against): Recipient/Subject as
        plain unheaded paragraphs, the letter body concatenated with no
        headings in `FLOWING_BODY_ORDER` (not dict order -- the headed
        layout's "Complainant Details before Introduction" is wrong for one
        continuous letter), Enclosures/Annexures still headed, and the
        signature as an unheaded closing block.
        """
        parts: list[str] = []
        if "Recipient" in sections:
            parts.append("<section class='document-section flowing-lead'>")
            parts.extend(self._paragraph_html(
                f"{_recipient_label(language)},\n{sections['Recipient']}"
            ))
            parts.append("</section>")
        if "Subject" in sections:
            parts.append("<section class='document-section flowing-lead'>")
            parts.append(
                f"<p><strong>{_escape(translated_heading('Subject', language))}:</strong> "
                f"{_escape(sections['Subject'])}</p>"
            )
            parts.append("</section>")

        body_headings = [heading for heading in FLOWING_BODY_ORDER if heading in sections]
        handled = set(FLOWING_LEAD_HEADINGS) | set(body_headings) | set(FLOWING_KEEP_HEADING) | _SIGNATURE_HEADINGS
        body_headings += [heading for heading in sections if heading not in handled and heading != "Disclaimer"]
        if body_headings:
            parts.append("<section class='document-section flowing-body'>")
            for heading in body_headings:
                parts.extend(self._paragraph_html(sections[heading]))
            parts.append("</section>")

        for heading in FLOWING_KEEP_HEADING:
            if heading not in sections:
                continue
            parts.append("<section class='document-section'>")
            parts.append(f"<h2 class='heading'>{_escape(translated_heading(heading, language))}</h2>")
            parts.append("<div>")
            parts.extend(self._paragraph_html(sections[heading]))
            parts.append("</div>")
            parts.append("</section>")

        for heading, content in sections.items():
            if heading not in _SIGNATURE_HEADINGS:
                continue
            parts.append("<section class='document-section signature-section'>")
            parts.append("<div class='signature-block'>")
            parts.extend(self._paragraph_html(content))
            parts.append("</div>")
            parts.append("</section>")

        if "Disclaimer" in sections:
            parts.append(f"<p class='disclaimer'>{_escape(sections['Disclaimer'])}</p>")
        return parts

    def _build_html(
        self, template_name: str, sections: dict[str, str], level: dict[str, float], language: str = "english",
        options: ExportOptions | None = None,
    ) -> str:
        disclaimer_size = max(8, level["body"] - 2)
        body_parts: list[str] = [
            "<main class='document-body'>",
            f"<h1 class='title'>{_escape(template_name.upper())}</h1>",
        ]
        if options and options.flowing_letter:
            body_parts.extend(self._flowing_body_html(sections, language))
        else:
            for heading, text in sections.items():
                if heading == "Disclaimer":
                    body_parts.append(f"<p class='disclaimer'>{_escape(text)}</p>")
                    continue
                # Part 53/56: the closing block (Place/Date, complementary
                # close, signatory name/designator -- see
                # `LegalDraftEngine._closing_block`) is right-aligned, matching
                # how a real signed legal letter is laid out and reusing the
                # exact same "which section gets this treatment" rule the DOCX
                # exporter applies (`heading in _SIGNATURE_HEADINGS`), not a
                # second, separately-maintained check.
                is_signature = heading in _SIGNATURE_HEADINGS
                section_classes = "document-section signature-section" if is_signature else "document-section"
                section_class = " class='signature-block'" if is_signature else ""
                body_parts.append(f"<section class='{section_classes}'>")
                body_parts.append(f"<h2 class='heading'>{_escape(translated_heading(heading, language))}</h2>")
                body_parts.append(f"<div{section_class}>")
                body_parts.extend(self._paragraph_html(text))
                body_parts.append("</div>")
                body_parts.append("</section>")
        if options and options.signature_image_path and _SIGNATURE_HEADINGS.intersection(sections):
            signature_uri = _file_data_uri(options.signature_image_path)
            body_parts.append(f"<img class='signature-image' src='{signature_uri}' alt='Digital signature' />")
        if options and options.notarization:
            body_parts.append(_notarization_block_html(options.notarization, language))
        body_parts.append("</main>")
        body_html = "\n".join(body_parts)
        page_label, of_label = _page_labels(language)

        # Part 57: Urdu/Sindhi/Kashmiri need right-to-left layout -- the
        # signature block mirrors to the LEFT under RTL (matching where a
        # signature conventionally sits relative to body text direction),
        # everything else mirrors via the single `direction: rtl` on `body`
        # (Pango/WeasyPrint handles paragraph-level bidi reordering).
        rtl = _is_rtl(language)
        html_dir_attr = ' dir="rtl"' if rtl else ""
        body_text_align = "right" if rtl else "justify"
        signature_text_align = "left" if rtl else "right"

        return f"""<!DOCTYPE html>
<html lang="{_html_lang(language)}"{html_dir_attr}>
<head>
<meta charset="utf-8">
<style>
{_font_face_css(language)}
@page {{
  size: A4;
  margin: {level["margin_v"]}in {level["margin_h"]}in;
  /* Layout simplification: the running header used to repeat the template
     name in a small 8pt font at the top of every page, ABOVE the document's
     own centered h1.title (the same name, in the real heading size) -- two
     headings on one page, reported live as clutter. Previously shown for
     every category except the 9 FLOWING_LETTER_DRAFTS templates (see
     advocate_register.py's module docstring); now suppressed everywhere, so
     h1.title is the one heading a reader sees. include_header_footer still
     governs the bottom page-number footer below, unaffected. */
  @top-center {{ content: ''; font-size: 8pt; color: #555; letter-spacing: 0.3pt; }}
  @bottom-center {{ content: {f"'{page_label} ' counter(page) ' {of_label} ' counter(pages)" if options and options.include_header_footer else "''"}; font-size: 8pt; color: #555; }}
}}
body {{
  font-family: '{_FONT_FAMILY}', sans-serif;
  font-size: {level["body"]}pt;
  line-height: {level["leading_ratio"]};
  color: #000;
  margin: 0;
  direction: {"rtl" if rtl else "ltr"};
}}
.document-body {{ display: block; }}
h1.title {{
  text-align: center;
  font-size: {level["title"]}pt;
  font-weight: 600;
  margin: 0 0 12pt 0;
}}
h2.heading {{
  font-size: {level["heading"]}pt;
  font-weight: 600;
  margin: {level["heading_gap"]}pt 0 3pt 0;
  padding-bottom: 1pt;
  break-after: avoid;
}}
p {{
  margin: 0 0 {level["para_gap"]}pt 0;
  text-align: {body_text_align};
  text-indent: 0;
  orphans: 3;
  widows: 3;
}}
.signature-block p {{ text-align: {signature_text_align}; }}
/* Keep the closing demand/verification sequence with the signature.  A
   single last-paragraph rule still produced a nearly empty signature page
   for Indic scripts, whose shaped lines are taller; chaining the final three
   paragraphs gives WeasyPrint enough material to balance the last page. */
.flowing-body p:nth-last-child(-n+4) {{ break-after: avoid; }}
.signature-section {{ break-inside: avoid; margin-top: 12pt; }}
.signature-image {{ display: block; width: 1.6in; max-height: 0.7in; object-fit: contain; margin-left: {"0" if rtl else "auto"}; margin-right: {"auto" if rtl else "0"}; }}
.watermark-text {{ position: fixed; top: 4.2in; left: 0; width: 100%; text-align: center; white-space: nowrap; color: #c8c8c8; opacity: 0.14; font-size: 36pt; font-weight: bold; letter-spacing: 4pt; transform: rotate(-35deg); z-index: -1; }}
.watermark {{ position: fixed; top: 2.7in; left: 1.7in; width: 4.5in; opacity: 0.16; transform: rotate(-35deg); z-index: -1; }}
.watermark img {{ width: 100%; }}
.notarization-block {{
  break-before: page;
  border: 1pt solid #222;
  padding: 14pt 16pt;
  margin-top: {level["heading_gap"]}pt;
}}
.notarization-block h2 {{ font-size: {level["heading"]}pt; margin: 0 0 8pt 0; text-align: center; }}
.notarization-block .row {{ margin: 0 0 4pt 0; text-align: left; }}
.notarization-block .hash {{ font-size: 8pt; word-break: break-all; }}
.notarization-qr {{ text-align: center; margin-top: 10pt; }}
.notarization-qr svg {{ width: 132px; height: 132px; }}
.notarization-caption {{ font-size: 8pt; text-align: center; margin-top: 6pt; color: #333; }}
p.disclaimer {{
  font-size: {disclaimer_size}pt;
  font-style: italic;
  margin-top: {level["heading_gap"]}pt;
  text-align: left;
}}
</style>
</head>
<body>
{"<div class='watermark'><img src='" + _file_data_uri(options.watermark_image_path) + "' alt='' /></div>" if options and options.watermark and options.watermark_image_path else "<div class='watermark-text'>" + _escape(options.watermark_text) + "</div>" if options and options.watermark and options.watermark_text else ""}
{body_html}
</body>
</html>"""


class DocxDraftExporter:
    def export(
        self, template_name: str, sections: dict[str, str], output_path: Path, language: str = "english",
        options: ExportOptions | None = None,
    ) -> Path:
        options = options or ExportOptions()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Part 56: shares the same fixed `_LAYOUT` the PDF exporter uses, so
        # both formats stay visually consistent (export parity) at every
        # document length.
        level = _LAYOUT
        # Part 57: the font NAME is now routed per script (see
        # `_docx_font_name`) instead of hardcoded "Nirmala UI" -- unlike the
        # PDF exporter, this works even for Ol Chiki/Meetei Mayek (no server-
        # side font needed; Word/LibreOffice substitutes at open time on
        # whatever machine actually displays the document).
        font_name = _docx_font_name(language)
        rtl = _is_rtl(language)

        document_factory, _qn, Inches, Pt = _docx()
        document = document_factory()
        style = document.styles["Normal"]
        style.font.name = font_name
        style.font.size = Pt(level["body"])

        section = document.sections[0]
        section.top_margin = Inches(level["margin_v"])
        section.bottom_margin = Inches(level["margin_v"])
        section.left_margin = Inches(level["margin_h"])
        section.right_margin = Inches(level["margin_h"])
        section.header_distance = Inches(0.2)
        section.footer_distance = Inches(0.22)
        section.different_first_page_header_footer = True
        if options.include_header_footer:
            # PDF parity: the top running header (the template name repeated
            # in small type on every page, above the document's own centered
            # title heading) is suppressed here too -- see the matching PDF
            # `@top-center` comment above `PdfDraftExporter._build_html`.
            # Only the page-number footer below remains.
            footer = section.footer
            footer.paragraphs[0].alignment = 1
            page_label, of_label = _page_labels(language)
            footer.paragraphs[0].add_run(f"{page_label} ")
            try:
                from docx.oxml import OxmlElement
                from docx.oxml.ns import qn

                page_field = OxmlElement("w:fldSimple")
                page_field.set(qn("w:instr"), "PAGE")
                footer.paragraphs[0]._p.append(page_field)
                # PDF parity: the PDF exporter's `@bottom-center` shows
                # "Page N of M" -- DOCX only ever showed "Page N" (missing
                # "of M") until this. `NUMPAGES` is the standard Word field
                # code for total page count, same mechanism as `PAGE` above.
                footer.paragraphs[0].add_run(f" {of_label} ")
                total_field = OxmlElement("w:fldSimple")
                total_field.set(qn("w:instr"), "NUMPAGES")
                footer.paragraphs[0]._p.append(total_field)
            except ImportError:
                footer.paragraphs[0].add_run("[number]")
        if options.watermark and options.watermark_image_path:
            header = section.header
            header.paragraphs[0].alignment = 1
            header.paragraphs[0].add_run().add_picture(options.watermark_image_path, width=Inches(5.5))

        title = document.add_heading(template_name.upper(), level=1)
        title.alignment = 1  # WD_ALIGN_PARAGRAPH.CENTER
        title.paragraph_format.space_after = Pt(12)
        if rtl:
            _set_rtl(title)
        for run in title.runs:
            run.font.name = font_name
            run.font.size = Pt(level["title"])

        if options.flowing_letter:
            self._flowing_body_docx(document, sections, language, font_name, level, rtl)

        for heading, text in (sections.items() if not options.flowing_letter else ()):
            if heading == "Disclaimer":
                note = document.add_paragraph(text)
                if rtl:
                    _set_rtl(note)
                for run in note.runs:
                    run.font.name = font_name
                    run.font.size = Pt(max(8, level["body"] - 2))
                    run.italic = True
                continue
            # Part 53 "Professional Layout Audit": the closing block --
            # Place/Date, the complementary close, and the signatory's own
            # name/designator (see `LegalDraftEngine._closing_block`) --
            # sits right-aligned near the page's right margin, matching how
            # a real signed legal letter is laid out (and how a human would
            # naturally leave room for a physical signature). No other
            # section's heading is affected. Alignment value `0` is
            # `WD_ALIGN_PARAGRAPH.LEFT`, `2` is `WD_ALIGN_PARAGRAPH.RIGHT`,
            # `3` is `WD_ALIGN_PARAGRAPH.JUSTIFY` -- referenced by raw int
            # here to match this file's existing convention (`title.
            # alignment = 1` above uses `WD_ALIGN_PARAGRAPH.CENTER` the same
            # way) rather than introducing a new import style partway
            # through the file. Part 56: every other section's body
            # paragraphs are justified (matching the PDF exporter's
            # `text-align: justify`), not left, for consistent professional
            # typography across export formats. Part 57: under RTL, the
            # signature block mirrors to the LEFT and body text aligns RIGHT
            # instead of JUSTIFY -- matching the PDF exporter's own RTL
            # handling (`_build_html`'s `signature_text_align`/
            # `body_text_align`).
            is_signature = heading in _SIGNATURE_HEADINGS
            if rtl:
                heading_alignment = 0 if is_signature else None
                paragraph_alignment = 0 if is_signature else 2
            else:
                heading_alignment = 2 if is_signature else None
                paragraph_alignment = 2 if is_signature else 3
            section_heading = document.add_heading(translated_heading(heading, language), level=2)
            if heading_alignment is not None:
                section_heading.alignment = heading_alignment
            if rtl:
                _set_rtl(section_heading)
            for run in section_heading.runs:
                run.font.name = font_name
                run.font.size = Pt(level["heading"])
            for paragraph_text in text.split("\n\n") or [""]:
                paragraph = document.add_paragraph()
                paragraph.alignment = paragraph_alignment
                paragraph.paragraph_format.space_after = Pt(level["para_gap"])
                if rtl:
                    _set_rtl(paragraph)
                lines = paragraph_text.splitlines() or [""]
                run = paragraph.add_run(lines[0])
                run.font.name = font_name
                run.font.size = Pt(level["body"])
                for line in lines[1:]:
                    run.add_break()
                    run.add_text(line)
        if options.watermark and options.watermark_text and not options.watermark_image_path:
            footer = section.footer
            footer.paragraphs[0].alignment = 1
            footer.paragraphs[0].add_run(f" · {options.watermark_text}")
        if options.signature_image_path and _SIGNATURE_HEADINGS.intersection(sections):
            signature_paragraph = document.add_paragraph()
            signature_paragraph.alignment = 0 if rtl else 2
            signature_paragraph.add_run().add_picture(options.signature_image_path, width=Inches(1.6))
        if options.notarization:
            attestation_title, attestation_rows = _notarization_rows(options.notarization, language)
            block_heading = document.add_heading(attestation_title, level=2)
            block_heading.alignment = 1
            if rtl:
                _set_rtl(block_heading)
            for run in block_heading.runs:
                run.font.name = font_name
                run.font.size = Pt(level["heading"])
            for label, value in attestation_rows:
                row_paragraph = document.add_paragraph()
                row_paragraph.alignment = 0
                if rtl:
                    _set_rtl(row_paragraph)
                label_run = row_paragraph.add_run(f"{label}: ")
                label_run.bold = True
                value_run = row_paragraph.add_run(value)
                for run in (label_run, value_run):
                    run.font.name = font_name
                    run.font.size = Pt(max(8, level["body"] - 1))
        document.save(str(output_path))
        return output_path

    @staticmethod
    def _add_paragraphs(document: Any, text: str, *, font_name: str, size: Any, rtl: bool, alignment: int) -> None:
        for paragraph_text in text.split("\n\n") or [""]:
            paragraph = document.add_paragraph()
            paragraph.alignment = alignment
            if rtl:
                _set_rtl(paragraph)
            lines = paragraph_text.splitlines() or [""]
            run = paragraph.add_run(lines[0])
            run.font.name = font_name
            run.font.size = size
            for line in lines[1:]:
                run.add_break()
                run.add_text(line)

    def _flowing_body_docx(
        self, document: Any, sections: dict[str, str], language: str, font_name: str, level: dict[str, float], rtl: bool,
    ) -> None:
        """DOCX equivalent of `PdfDraftExporter._flowing_body_html` -- see
        that method and `advocate_register.FLOWING_LETTER_DRAFTS`'s
        docstring for what this renders and why. Adds paragraphs directly to
        `document`; the title has already been added by the caller.
        """
        _, _, _, Pt = _docx()
        body_align = 0 if rtl else 3  # LEFT under RTL (matches the PDF's body_text_align), else JUSTIFY

        if "Recipient" in sections:
            self._add_paragraphs(
                document, f"{_recipient_label(language)},\n{sections['Recipient']}",
                font_name=font_name, size=Pt(level["body"]), rtl=rtl, alignment=body_align,
            )
        if "Subject" in sections:
            paragraph = document.add_paragraph()
            paragraph.alignment = body_align
            if rtl:
                _set_rtl(paragraph)
            label_run = paragraph.add_run(f"{translated_heading('Subject', language)}: ")
            label_run.bold = True
            label_run.font.name = font_name
            label_run.font.size = Pt(level["body"])
            value_run = paragraph.add_run(sections["Subject"])
            value_run.font.name = font_name
            value_run.font.size = Pt(level["body"])

        body_headings = [heading for heading in FLOWING_BODY_ORDER if heading in sections]
        handled = set(FLOWING_LEAD_HEADINGS) | set(body_headings) | set(FLOWING_KEEP_HEADING) | _SIGNATURE_HEADINGS
        body_headings += [heading for heading in sections if heading not in handled and heading != "Disclaimer"]
        for heading in body_headings:
            self._add_paragraphs(
                document, sections[heading],
                font_name=font_name, size=Pt(level["body"]), rtl=rtl, alignment=body_align,
            )

        for heading in FLOWING_KEEP_HEADING:
            if heading not in sections:
                continue
            section_heading = document.add_heading(translated_heading(heading, language), level=2)
            if rtl:
                _set_rtl(section_heading)
            for run in section_heading.runs:
                run.font.name = font_name
                run.font.size = Pt(level["heading"])
            self._add_paragraphs(
                document, sections[heading],
                font_name=font_name, size=Pt(level["body"]), rtl=rtl, alignment=body_align,
            )

        signature_align = 0 if rtl else 2  # LEFT under RTL, else RIGHT -- matches the headed layout's own rule
        for heading, content in sections.items():
            if heading not in _SIGNATURE_HEADINGS:
                continue
            self._add_paragraphs(
                document, content,
                font_name=font_name, size=Pt(level["body"]), rtl=rtl, alignment=signature_align,
            )

        if "Disclaimer" in sections:
            note = document.add_paragraph(sections["Disclaimer"])
            if rtl:
                _set_rtl(note)
            for run in note.runs:
                run.font.name = font_name
                run.font.size = Pt(max(8, level["body"] - 2))
                run.italic = True


class TxtDraftExporter:
    def export(
        self, template_name: str, sections: dict[str, str], output_path: Path, language: str = "english",
        options: ExportOptions | None = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [template_name, ""]
        if options and options.flowing_letter:
            if "Recipient" in sections:
                lines.extend([f"{_recipient_label(language)},\n{sections['Recipient']}", ""])
            if "Subject" in sections:
                lines.extend([f"{translated_heading('Subject', language)}: {sections['Subject']}", ""])
            body_headings = [heading for heading in FLOWING_BODY_ORDER if heading in sections]
            handled = set(FLOWING_LEAD_HEADINGS) | set(body_headings) | set(FLOWING_KEEP_HEADING) | _SIGNATURE_HEADINGS
            body_headings += [
                heading for heading in sections
                if heading not in handled and heading != "Disclaimer"
            ]
            for heading in body_headings:
                lines.extend([sections[heading], ""])
            for heading in FLOWING_KEEP_HEADING:
                if heading in sections:
                    lines.extend([translated_heading(heading, language), sections[heading], ""])
            for heading, text in sections.items():
                if heading in _SIGNATURE_HEADINGS:
                    lines.extend([text, ""])
            if "Disclaimer" in sections:
                lines.extend([sections["Disclaimer"], ""])
        else:
            for heading, text in sections.items():
                display_heading = translated_heading(heading, language)
                lines.extend([display_heading, text, ""])
        if options and options.notarization:
            lines.extend(_notarization_block_text(options.notarization, language))
            lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path


# RTF, like DOCX, doesn't need server-side glyph rasterization the way the PDF
# exporter does -- a reader (Word/LibreOffice) shapes complex scripts itself
# when the file is opened, so this is hand-rolled RTF markup (control words +
# `\uN` Unicode escapes) rather than a dependency on any RTF-writing library.
# Reuses the same `_LAYOUT`, `_docx_font_name`/`_is_rtl` (script/RTL routing),
# and `_SIGNATURE_HEADINGS` this file already applies to DOCX, so RTF gets the
# same script coverage and layout rules for free.
class RtfDraftExporter:
    def export(
        self, template_name: str, sections: dict[str, str], output_path: Path, language: str = "english",
        options: ExportOptions | None = None,
    ) -> Path:
        # Pre-existing crash: unlike `PdfDraftExporter`/`DocxDraftExporter`,
        # this exporter dereferenced `options.document_version` without
        # ever defaulting `options`, so every RTF export made with the
        # documented default signature (`options=None`) died with an
        # AttributeError before writing a single byte.
        options = options or ExportOptions()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        level = _LAYOUT
        font_name = _docx_font_name(language)
        rtl = _is_rtl(language)
        par_dir = "\\rtlpar" if rtl else ""
        margin_l = round(level["margin_h"] * 1440)
        margin_t = round(level["margin_v"] * 1440)

        parts: list[str] = [
            "{\\rtf1\\ansi\\ansicpg1252\\deff0",
            f"{{\\fonttbl{{\\f0 {font_name};}}}}",
            f"\\margl{margin_l}\\margr{margin_l}\\margt{margin_t}\\margb{margin_t}",
            (
                f"\\pard{par_dir}\\qc\\b\\fs{round(level['title'] * 2)} "
                f"{_rtf_escape(template_name.upper())}\\b0\\par"
            ),
        ]

        render_items: list[tuple[str | None, str, bool]] = []
        if options.flowing_letter:
            if "Recipient" in sections:
                render_items.append((
                    None,
                    f"{_recipient_label(language)},\n{sections['Recipient']}",
                    False,
                ))
            if "Subject" in sections:
                render_items.append((None, f"{translated_heading('Subject', language)}: {sections['Subject']}", False))
            body_headings = [heading for heading in FLOWING_BODY_ORDER if heading in sections]
            handled = set(FLOWING_LEAD_HEADINGS) | set(body_headings) | set(FLOWING_KEEP_HEADING) | _SIGNATURE_HEADINGS
            body_headings += [
                heading for heading in sections
                if heading not in handled and heading != "Disclaimer"
            ]
            render_items.extend((None, sections[heading], False) for heading in body_headings)
            render_items.extend(
                (heading, sections[heading], False)
                for heading in FLOWING_KEEP_HEADING
                if heading in sections
            )
            render_items.extend(
                (None, text, True)
                for heading, text in sections.items()
                if heading in _SIGNATURE_HEADINGS
            )
            if "Disclaimer" in sections:
                render_items.append(("Disclaimer", sections["Disclaimer"], False))
        else:
            render_items = [
                (heading, text, heading in _SIGNATURE_HEADINGS)
                for heading, text in sections.items()
            ]

        for heading, text, is_signature in render_items:
            if heading == "Disclaimer":
                size = round(max(8, level["body"] - 2) * 2)
                parts.append(f"\\pard{par_dir}\\ql\\i\\fs{size} {_rtf_escape(text)}\\i0\\par")
                continue
            if heading is not None:
                heading_align = "\\qr" if (is_signature and not rtl) else "\\ql" if (is_signature and rtl) else ""
                parts.append(
                    f"\\pard{par_dir}{heading_align}\\b\\fs{round(level['heading'] * 2)} "
                    f"{_rtf_escape(translated_heading(heading, language))}\\b0\\par"
                )
            if rtl:
                body_align = "\\ql" if is_signature else "\\qr"
            else:
                body_align = "\\qr" if is_signature else "\\qj"
            for paragraph in text.split("\n\n") or [""]:
                lines = paragraph.splitlines() or [""]
                joined = "\\line ".join(_rtf_escape(line) for line in lines)
                parts.append(f"\\pard{par_dir}{body_align}\\fs{round(level['body'] * 2)} {joined}\\par")

        if options.notarization:
            attestation_title, attestation_rows = _notarization_rows(options.notarization, language)
            parts.append(
                f"\\pard{par_dir}\\qc\\b\\fs{round(level['heading'] * 2)} "
                f"{_rtf_escape(attestation_title)}\\b0\\par"
            )
            attestation_size = round(max(8, level["body"] - 1) * 2)
            for label, value in attestation_rows:
                parts.append(
                    f"\\pard{par_dir}\\ql\\fs{attestation_size} \\b {_rtf_escape(label)}: \\b0 "
                    f"{_rtf_escape(value)}\\par"
                )
        parts.append("}")
        output_path.write_text("\n".join(parts), encoding="ascii")
        return output_path


def _rtf_escape(text: str) -> str:
    """Escapes RTF control characters and encodes non-ASCII characters as
    `\\uN?` -- RTF's Unicode-escape-with-ANSI-fallback syntax (`N` a signed
    16-bit value, `?` the fallback byte a non-Unicode-aware reader shows
    instead). Every script this app supports (Devanagari, Bengali, Gurmukhi,
    Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Perso-Arabic) fits in
    the Basic Multilingual Plane, so this covers all of them; the >0x8000
    branch is the correct signed-16-bit encoding for the upper half of the
    BMP (unused by any currently-supported script, but not a source of
    incorrect output if one is added later).
    """
    parts: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            parts.append("\\\\")
        elif ch == "{":
            parts.append("\\{")
        elif ch == "}":
            parts.append("\\}")
        elif code > 127:
            signed = code if code < 0x8000 else code - 0x10000
            parts.append(f"\\u{signed}?")
        else:
            parts.append(ch)
    return "".join(parts)


def _escape(text: str) -> str:
    return html_lib.escape(text, quote=False)


# Maps this app's internal language names to BCP-47 tags for the rendered
# PDF's `<html lang="">` -- accessibility/metadata only, doesn't affect
# shaping (that's driven by the actual script of the text, which Pango/
# HarfBuzz detect from the Unicode codepoints themselves).
_HTML_LANG_CODES = {
    "hindi": "hi", "tamil": "ta", "telugu": "te", "kannada": "kn", "bengali": "bn",
    "malayalam": "ml", "marathi": "mr", "gujarati": "gu", "punjabi": "pa", "odia": "or", "urdu": "ur",
    "assamese": "as", "sanskrit": "sa", "konkani": "kok", "nepali": "ne", "maithili": "mai",
    "dogri": "doi", "bodo": "brx", "sindhi": "sd", "kashmiri": "ks", "santali": "sat", "manipuri": "mni",
}


def _html_lang(language: str) -> str:
    return _HTML_LANG_CODES.get(language, "en")
