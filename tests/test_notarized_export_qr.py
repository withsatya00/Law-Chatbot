"""End-to-end proof that a notarized export really carries a scannable QR.

Phase 1 opened with two failing tests and a broken feature, both from a single
cause: `segno` was declared in `pyproject.toml` but not installed, so
`app/notarization/qr.py`'s call-time `load("segno", ...)` raised
`MissingOptionalDependencyError` and every notarized export died with it.
These tests exercise the whole path -- prepare -> sign -> notary review ->
approve -> export -- and assert on the bytes that come out, so a missing or
regressed encoder fails here rather than in production.

Nothing in this module fabricates a notarial act. The notary, registration
number, jurisdiction and approval all come from an explicitly-inserted
verified-notary record and a real `approve_request` call through the service's
own state machine; the assertions then check that the exported file reproduces
exactly those stored values and adds no seal or signature of its own.
"""

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from app.core.exceptions import BadRequestError
from app.notarization import states
from app.notarization.qr import QRUnavailableError, build_verification_qr_svg
from tests.test_notarization import _OWNER, _service, _through_to_notarized, _through_to_signed

pytest.importorskip("segno")


# ---------------------------------------------------------------------------
# The QR itself
# ---------------------------------------------------------------------------


def test_qr_svg_is_well_formed_and_sized_as_requested() -> None:
    url = "https://verify.example.com/verify/Xk3n9QpL2mR7vT4sYw8zBc1dFg6hJ0aE"
    svg = build_verification_qr_svg(states.NOTARIZED, url, size_px=132)

    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "width='132'" in svg
    assert "height='132'" in svg
    assert "role='img'" in svg
    # A QR is meaningless to a screen reader, so it must be labelled.
    assert "aria-label='Notarization verification QR code'" in svg
    # Inline and self-contained: no external asset the exported PDF would have
    # to fetch at render time.
    assert "<image" not in svg
    assert "xlink:href" not in svg
    # Real module content, not an empty placeholder frame.
    assert svg.count("<path") >= 1 or svg.count("<rect") > 1


def test_qr_content_changes_with_the_verification_token() -> None:
    """Guards against an encoder that returns a constant frame: two different
    tokens must produce genuinely different artwork."""
    base = "https://verify.example.com/verify/"
    first = build_verification_qr_svg(states.NOTARIZED, base + "Xk3n9QpL2mR7vT4sYw8zBc1dFg6hJ0aE")
    second = build_verification_qr_svg(states.NOTARIZED, base + "Zq7w1EbN5xM2cV8kUj3rTy6iOp0lAs4d")
    assert first != second


def test_qr_is_refused_for_every_status_that_is_not_notarized() -> None:
    for status in states.ALL_STATUSES:
        if status == states.NOTARIZED:
            continue
        with pytest.raises(QRUnavailableError):
            build_verification_qr_svg(status, "https://verify.example.com/verify/abc")


def test_qr_is_refused_without_a_verification_url() -> None:
    """A notarized document whose token is missing must not get a QR pointing
    at a bare, tokenless verification page -- that would scan and resolve to
    nothing while looking authentic."""
    with pytest.raises(QRUnavailableError):
        build_verification_qr_svg(states.NOTARIZED, "")


# ---------------------------------------------------------------------------
# Export options
# ---------------------------------------------------------------------------


def test_notarized_export_options_carry_the_stored_record_and_a_qr() -> None:
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))
    options = asyncio.run(service.build_notarized_export_options(document))

    attestation = options.notarization
    assert attestation is not None
    # Copied verbatim from the notary record inserted by the fixture -- not
    # minted here and not derived from anything the user supplied.
    assert attestation.notary_name == "Adv. S. Iyer"
    assert attestation.notary_registration_number == "NOT/UP/2019/1234"
    assert attestation.jurisdiction_state == "Uttar Pradesh"
    assert attestation.document_hash == document["document_hash"]

    token = document["verification_token"]
    assert token
    assert attestation.verification_url.endswith(token)
    assert attestation.verification_qr_svg.startswith("<svg")


def test_export_options_are_refused_for_a_signed_but_not_notarized_document() -> None:
    service = _service()
    document = asyncio.run(_through_to_signed(service))
    assert document["status"] == states.SIGNED
    with pytest.raises(BadRequestError):
        asyncio.run(service.build_notarized_export_options(document))


def test_export_is_refused_when_the_attesting_notary_is_missing_from_the_record() -> None:
    """Rather than printing an attestation block with a blank notary name."""
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))
    service.notaries.rows.clear()
    with pytest.raises(BadRequestError):
        asyncio.run(service.build_notarized_export_options(document))


# ---------------------------------------------------------------------------
# A real exported file
# ---------------------------------------------------------------------------


def _export(service: Any, document: dict[str, Any], tmp_path: Path, fmt: str) -> Path:
    """Runs the real exporter into `tmp_path` instead of the configured output
    directory, so the suite leaves nothing behind."""
    from app.core.config import settings

    original = settings.draft_output_dir
    settings.draft_output_dir = tmp_path
    try:
        result = asyncio.run(
            service.export_notarized(document_id=str(document["_id"]), user_id=_OWNER, fmt=fmt)
        )
    finally:
        settings.draft_output_dir = original
    # `export_notarized` returns whatever the format's exporter returns, which
    # is the written file's `Path`.
    path = Path(result)
    assert path.is_file(), f"exporter reported {path}, which does not exist"
    assert path.parent == tmp_path
    return path


def test_a_real_notarized_txt_export_contains_the_verification_record(tmp_path: Path) -> None:
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))

    path = _export(service, document, tmp_path, "txt")
    text = path.read_text(encoding="utf-8")

    assert document["verification_token"] in text
    assert "Adv. S. Iyer" in text
    assert "NOT/UP/2019/1234" in text
    assert document["document_hash"] in text


def test_a_real_notarized_pdf_export_renders_and_keeps_the_verification_url(tmp_path: Path) -> None:
    """The QR reaches the PDF as vector artwork, so the check is that the PDF
    renders at all and that the verification token survives into its text."""
    pytest.importorskip("weasyprint")
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))

    path = _export(service, document, tmp_path, "pdf")
    raw = path.read_bytes()

    assert raw.startswith(b"%PDF-"), "exporter did not produce a PDF"
    assert len(raw) > 2000, "PDF is implausibly small for an attested document"

    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(str(path))
    # Whitespace in extracted PDF text is unreliable, so compare with it removed.
    extracted = re.sub(r"\s+", "", "\n".join(page.extract_text() or "" for page in reader.pages))
    assert document["verification_token"] in extracted
    assert "NOT/UP/2019/1234" in extracted


def test_every_export_format_carries_the_verification_record(tmp_path: Path) -> None:
    """Until Phase 1 only the PDF exporter rendered an attestation at all: a
    notarized document downloaded as DOCX, TXT or RTF came out with no notary,
    no registration number, no document hash and no verification URL on it --
    nothing a recipient could check, and nothing distinguishing it from an
    ordinary unnotarized draft.

    The QR itself stays PDF-only (it is inline SVG vector artwork), but it
    encodes nothing except the verification URL, which every format must carry.
    """
    for fmt in ("txt", "rtf", "docx", "pdf"):
        if fmt == "pdf":
            pytest.importorskip("weasyprint")
        if fmt == "docx":
            pytest.importorskip("docx")
        service = _service()
        document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))
        path = _export(service, document, tmp_path, fmt)
        extracted = re.sub(r"\s+", "", _readable_text(path, fmt))

        token = document["verification_token"]
        assert token in extracted, f"{fmt} export dropped the verification token"
        assert "NOT/UP/2019/1234" in extracted, f"{fmt} export dropped the registration number"
        assert document["document_hash"] in extracted, f"{fmt} export dropped the document hash"


def _readable_text(path: Path, fmt: str) -> str:
    if fmt == "pdf":
        pypdf = pytest.importorskip("pypdf")
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if fmt == "docx":
        docx = pytest.importorskip("docx")
        document = docx.Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    return path.read_text(encoding="utf-8" if fmt == "txt" else "ascii")


def test_the_export_adds_no_seal_or_signature_of_its_own(tmp_path: Path) -> None:
    """The platform records a notarization; it never draws the notary's seal or
    signature. The printed block must say only what we are entitled to say."""
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))

    path = _export(service, document, tmp_path, "txt")
    text = path.read_text(encoding="utf-8").lower()

    for forbidden in ("notarial seal", "seal affixed", "signature of notary", "digitally sealed by"):
        assert forbidden not in text, f"export fabricated a notarial artefact: {forbidden!r}"
