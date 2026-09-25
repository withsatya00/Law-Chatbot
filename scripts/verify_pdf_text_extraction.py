"""D2 verification tooling: reports whether text extracted from a PDF (via
`pypdf` and `pymupdf`) shows the confirmed WeasyPrint Indic-script corruption.

See `app.drafting.pdf_text_integrity`'s module docstring for the full root
cause and `tests/test_pdf_text_extraction_integrity.py` for the pinned
regression coverage this script complements: that test always exercises the
SAME known-corrupting Hindi sample; this script runs the SAME detectors
against ANY PDF an operator hands it -- a freshly exported draft, a PDF from
production, or a candidate after a WeasyPrint/font upgrade -- to see whether
the defect is present in that specific document.

Usage:
    python scripts/verify_pdf_text_extraction.py <path-to-pdf> --script devanagari [--source path/to/expected.txt]

`--script` is one of `app.drafting.pdf_text_integrity`'s known script keys
(devanagari, bengali, gurmukhi, gujarati, odia, tamil, telugu, kannada,
malayalam, perso_arabic) and gates `detect_corruption`'s out-of-script-
character check. `--source`, if given, is a UTF-8 text file with the
document's intended content, and additionally runs `detect_missing_words`
against it. Without `--source`, only the out-of-script-character check runs
-- still useful (it caught this exact bug with no ground truth needed), but
weaker than with it (a silently dropped mark with nothing put in its place
needs the source text to be caught at all).

Exits 0 with "No corruption detected" when clean, 1 with a findings report
otherwise -- suitable for a CI step or an ad hoc developer check alike.
"""

import argparse
import sys
from pathlib import Path

import pymupdf
from pypdf import PdfReader

from app.drafting.pdf_text_integrity import detect_corruption, detect_missing_words


def _extract(pdf_path: Path) -> tuple[str, str]:
    """`(pypdf_text, pymupdf_text)` -- the two readers surface this bug's two
    different symptoms (a silent drop vs. a stray poisoned character; see
    `app.drafting.pdf_text_integrity`'s module docstring), so both are
    checked rather than picking one."""
    reader = PdfReader(pdf_path)
    pypdf_text = "\n".join(page.extract_text() for page in reader.pages)
    with pymupdf.open(pdf_path) as document:
        pymupdf_text = "\n".join(page.get_text() for page in document)
    return pypdf_text, pymupdf_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_path", type=Path, help="PDF file to check")
    parser.add_argument(
        "--script", required=True,
        help="Script key from app.drafting.pdf_text_integrity._SCRIPT_BLOCKS (e.g. devanagari, tamil)",
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="UTF-8 text file with the document's intended content, for the missing-word check",
    )
    args = parser.parse_args()

    # A Windows console's default codepage (cp1252 etc.) cannot encode most
    # Indic-script text this script prints back verbatim (the whole point of
    # a corruption REPORT is to show the exact offending characters) --
    # `errors="backslashreplace"` keeps the report on screen instead of
    # crashing before it can be read; redirecting to a file already uses
    # UTF-8 and is unaffected either way.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    pypdf_text, pymupdf_text = _extract(args.pdf_path)
    findings: list[str] = []

    for label, text in (("pypdf", pypdf_text), ("pymupdf", pymupdf_text)):
        for finding in detect_corruption(text, args.script):
            findings.append(
                f"[{label}] unexpected {finding.character} ({finding.codepoint}) near "
                f"{finding.context!r}"
            )

    if args.source is not None:
        source_text = args.source.read_text(encoding="utf-8")
        for label, text in (("pypdf", pypdf_text), ("pymupdf", pymupdf_text)):
            for finding in detect_missing_words(text, source_text):
                findings.append(f"[{label}] missing word from source: {finding.word!r}")

    if not findings:
        print(f"No corruption detected in {args.pdf_path}")
        return 0

    print(f"Corruption detected in {args.pdf_path}:")
    for line in findings:
        print(f"  - {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
