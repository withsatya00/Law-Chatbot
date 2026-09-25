"""Advanced deterministic services around the document grammar engine.

The services in this module never treat OCR text or uploaded templates as
instructions.  They extract signals and produce reviewable proposals; only a
human/admin can promote an analyzed template into an active grammar.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from app.drafting.document_grammar import (
    ComponentType,
    DocumentGrammar,
    HeadingMode,
    PaginationPolicy,
    SectionSpec,
)


@dataclass(frozen=True)
class LanguageDetection:
    language_code: str
    language_name: str
    script: str
    input_script: str
    mixed_scripts: tuple[str, ...]
    romanized: bool
    confidence: float


class LanguageScriptDetector:
    """Unicode-range detection plus conservative Roman-Hindi recognition."""

    _RANGES: ClassVar[tuple[tuple[int, int, str], ...]] = (
        (0x0900, 0x097F, "Devanagari"), (0x0980, 0x09FF, "Bengali"),
        (0x0A00, 0x0A7F, "Gurmukhi"), (0x0A80, 0x0AFF, "Gujarati"),
        (0x0B00, 0x0B7F, "Odia"), (0x0B80, 0x0BFF, "Tamil"),
        (0x0C00, 0x0C7F, "Telugu"), (0x0C80, 0x0CFF, "Kannada"),
        (0x0D00, 0x0D7F, "Malayalam"), (0x0600, 0x06FF, "Perso-Arabic"),
        (0x1C50, 0x1C7F, "Ol Chiki"), (0xABC0, 0xABFF, "Meetei Mayek"),
        (0x0000, 0x007F, "Latin"),
    )
    _SCRIPT_LANGUAGE: ClassVar[dict[str, tuple[str, str]]] = {
        "Devanagari": ("hi", "Hindi"), "Bengali": ("bn", "Bengali"),
        "Gurmukhi": ("pa", "Punjabi"), "Gujarati": ("gu", "Gujarati"),
        "Odia": ("or", "Odia"), "Tamil": ("ta", "Tamil"),
        "Telugu": ("te", "Telugu"), "Kannada": ("kn", "Kannada"),
        "Malayalam": ("ml", "Malayalam"), "Perso-Arabic": ("ur", "Urdu"),
        "Ol Chiki": ("sat", "Santali"), "Meetei Mayek": ("mni", "Manipuri"),
        "Latin": ("en", "English"),
    }
    _ROMAN_HINDI = frozenset({
        "mujhe", "mera", "meri", "mere", "hai", "nahi", "karo", "banao",
        "banana", "police", "zameen", "kabza", "kiraya", "tenant", "notice",
        "chahiye", "hua", "raha", "wali", "wala", "liye", "ke", "ko",
    })

    def detect(self, text: str) -> LanguageDetection:
        counts: dict[str, int] = {}
        for char in text:
            if not char.isalpha():
                continue
            script = next((name for start, end, name in self._RANGES if start <= ord(char) <= end), "Unknown")
            counts[script] = counts.get(script, 0) + 1
        if not counts:
            return LanguageDetection("und", "Undetermined", "Unknown", "Unknown", (), False, 0.0)
        ordered = sorted(counts, key=lambda script: counts[script], reverse=True)
        primary = ordered[0]
        total = sum(counts.values())
        roman_tokens = {token.lower() for token in re.findall(r"[A-Za-z]+", text)}
        roman_score = len(roman_tokens & self._ROMAN_HINDI) / max(1, len(roman_tokens))
        romanized = primary == "Latin" and roman_score >= 0.16 and len(roman_tokens & self._ROMAN_HINDI) >= 2
        code, name = ("hi", "Hindi") if romanized else self._SCRIPT_LANGUAGE.get(primary, ("und", "Undetermined"))
        mixed = tuple(script for script in ordered if counts[script] / total >= 0.08)
        return LanguageDetection(
            code, name, "Devanagari" if romanized else primary, primary,
            mixed, romanized, round(min(0.99, counts[primary] / total + (0.15 if romanized else 0)), 3),
        )


class MultilingualOcrRouter:
    """Maps detected scripts to installed-provider language identifiers."""

    _TESSERACT: ClassVar[dict[str, str]] = {
        "Latin": "eng", "Devanagari": "hin+eng", "Bengali": "ben+eng",
        "Gurmukhi": "pan+eng", "Gujarati": "guj+eng", "Odia": "ori+eng",
        "Tamil": "tam+eng", "Telugu": "tel+eng", "Kannada": "kan+eng",
        "Malayalam": "mal+eng", "Perso-Arabic": "urd+eng",
        "Ol Chiki": "sat+eng", "Meetei Mayek": "mni+eng",
    }

    def route(self, text_hint: str = "", *, handwritten: bool = False) -> dict[str, Any]:
        detected = LanguageScriptDetector().detect(text_hint) if text_hint else None
        scripts = detected.mixed_scripts if detected and detected.mixed_scripts else (
            (detected.script,) if detected else ("Latin",)
        )
        languages: list[str] = []
        for script in scripts:
            languages.extend(self._TESSERACT.get(script, "eng").split("+"))
        languages = list(dict.fromkeys(languages))
        return {
            "provider": "vision_layout_ocr" if handwritten else "tesseract",
            "provider_language": "+".join(languages),
            "detected": asdict(detected) if detected else None,
            "preserve_layout": True,
            "preserve_original_script": True,
            "review_required": handwritten or not text_hint,
        }


@dataclass(frozen=True)
class ProtectedEntity:
    kind: str
    value: str


@dataclass(frozen=True)
class SemanticCheckResult:
    status: Literal["PASS", "WARNING", "ERROR"]
    score: float
    issues: tuple[dict[str, str], ...]
    protected_entities: tuple[ProtectedEntity, ...]


class BilingualSemanticValidator:
    """Checks protected facts after translation/back-translation.

    It does not demand word equality.  Names and addresses are difficult to
    validate without a transliteration provider, so all machine-normalized
    numbers, dates, identifiers, emails, phones, currency amounts and legal
    citations are compared exactly.
    """

    _PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "phone": re.compile(r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)"),
        "amount": re.compile(r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{1,2})?", re.IGNORECASE),
        "date": re.compile(r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
        "case_number": re.compile(r"\b(?:FIR|CASE|SUIT|PETITION|APPEAL)\s*(?:NO\.?\s*)?[A-Z0-9./-]+", re.IGNORECASE),
        "legal_citation": re.compile(r"\b(?:Section|Article|Order|Rule)\s+[A-Z0-9()/-]+", re.IGNORECASE),
        "tax_id": re.compile(r"\b(?:[A-Z]{5}\d{4}[A-Z]|\d{2}[A-Z]{5}\d{4}[A-Z]\dZ[A-Z0-9])\b"),
    }

    @classmethod
    def entities(cls, text: str) -> tuple[ProtectedEntity, ...]:
        found: list[ProtectedEntity] = []
        for kind, pattern in cls._PATTERNS.items():
            for match in pattern.finditer(text):
                normalized = re.sub(r"\s+", "", match.group(0)).casefold()
                found.append(ProtectedEntity(kind, normalized))
        return tuple(found)

    def compare(self, source: str, target: str, back_translation: str = "") -> SemanticCheckResult:
        source_entities = self.entities(source)
        target_entities = set(self.entities(target))
        issues: list[dict[str, str]] = []
        for entity in source_entities:
            if entity not in target_entities:
                issues.append({
                    "error_code": "SEMANTIC_PROTECTED_ENTITY_MISSING",
                    "severity": "error", "location": entity.kind,
                    "message": f"Protected {entity.kind} changed or disappeared in the target draft.",
                    "suggested_fix": "Restore the exact source value before export.",
                })
        if back_translation:
            source_terms = set(re.findall(r"[a-z]{4,}", source.casefold()))
            back_terms = set(re.findall(r"[a-z]{4,}", back_translation.casefold()))
            lexical = len(source_terms & back_terms) / max(1, len(source_terms))
            if lexical < 0.35:
                issues.append({
                    "error_code": "SEMANTIC_BACK_TRANSLATION_DRIFT",
                    "severity": "warning", "location": "document",
                    "message": "Back-translation has low semantic token overlap with the source.",
                    "suggested_fix": "Review obligations, prohibitions and relief clauses bilingually.",
                })
        score = max(0.0, 1.0 - len(issues) / max(1, len(source_entities) + 1))
        status: Literal["PASS", "WARNING", "ERROR"] = (
            "ERROR" if any(issue["severity"] == "error" for issue in issues)
            else "WARNING" if issues else "PASS"
        )
        return SemanticCheckResult(status, round(score, 3), tuple(issues), source_entities)


@dataclass(frozen=True)
class TemplateAnalysis:
    proposed_grammar: DocumentGrammar
    detected_headings: tuple[str, ...]
    letterhead_lines: tuple[str, ...]
    warnings: tuple[str, ...]
    status: str = "review_required"


class TemplateAnalyzer:
    """Distils reusable structure; never clones substantive source facts."""

    _HEADING = re.compile(r"^[A-Z][A-Z\s/&().:-]{2,100}$")
    _MAP: ClassVar[tuple[tuple[str, ComponentType], ...]] = (
        ("prayer", ComponentType.COURT_PRAYER), ("verification", ComponentType.VERIFICATION),
        ("subject", ComponentType.SUBJECT_BLOCK), ("facts", ComponentType.NUMBERED_PARAGRAPHS),
        ("grounds", ComponentType.LEGAL_GROUND), ("jurisdiction", ComponentType.JURISDICTION),
        ("limitation", ComponentType.LIMITATION), ("annexure", ComponentType.ANNEXURE_LIST),
        ("schedule", ComponentType.PROPERTY_SCHEDULE), ("witness", ComponentType.WITNESS_BLOCK),
        ("signature", ComponentType.SIGNATURE),
    )

    def analyze(self, text: str, document_id: str) -> TemplateAnalysis:
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
        headings: list[str] = []
        for line in lines:
            candidate = line.rstrip(":")
            if self._HEADING.fullmatch(candidate) and not candidate.isdigit() and candidate not in headings:
                headings.append(candidate)
        if not headings:
            headings = ["Document Body", "Signature"]
        specs: list[SectionSpec] = []
        for index, heading in enumerate(headings):
            lowered = heading.casefold()
            component = next((kind for token, kind in self._MAP if token in lowered), ComponentType.TEXT)
            specs.append(SectionSpec(
                id=re.sub(r"[^a-z0-9]+", "_", lowered).strip("_"), source_heading=heading,
                component=component, heading_mode=HeadingMode.VISIBLE,
                required=index == 0 or component in {ComponentType.COURT_PRAYER, ComponentType.VERIFICATION},
                repeatable=component in {ComponentType.NUMBERED_PARAGRAPHS, ComponentType.LEGAL_GROUND},
                pagination=PaginationPolicy(
                    keep_together=component in {ComponentType.SIGNATURE, ComponentType.VERIFICATION},
                    keep_with_next=True,
                    allow_split=component not in {ComponentType.SIGNATURE, ComponentType.VERIFICATION},
                ),
            ))
        letterhead = tuple(lines[:5]) if any("advocate" in line.casefold() for line in lines[:8]) else ()
        family = self._family(headings, text)
        grammar = DocumentGrammar(
            document_id=document_id, document_family=family, document_type=family,
            structure=tuple(specs), version="proposal-1", active=False,
        )
        warnings = (
            "Substantive names, allegations and clauses were intentionally not copied.",
            "An advocate/admin must review and activate this proposed grammar.",
        )
        return TemplateAnalysis(grammar, tuple(headings), letterhead, warnings)

    @staticmethod
    def _family(headings: Sequence[str], text: str) -> str:
        material = " ".join(headings).casefold() + " " + text[:1000].casefold()
        if "affidavit" in material or "deponent" in material:
            return "affidavit"
        if "written statement" in material:
            return "written_statement"
        if "legal notice" in material or "reply notice" in material:
            return "reply_notice" if "reply notice" in material else "advocate_notice"
        if "agreement" in material or "deed" in material:
            return "agreement"
        if "petition" in material:
            return "petition"
        if "complaint" in material:
            return "complaint"
        return "unknown"


@dataclass(frozen=True)
class VisualQAReport:
    status: Literal["PASS", "WARNING", "ERROR"]
    format: str
    page_count: int
    issues: tuple[dict[str, str], ...] = field(default_factory=tuple)
    checks: Mapping[str, bool] = field(default_factory=dict)


class VisualQAService:
    """Structural and raster-aware QA for generated PDF/DOCX files."""

    def inspect(self, path: Path) -> VisualQAReport:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._pdf(path)
        if suffix == ".docx":
            return self._docx(path)
        raise ValueError(f"Visual QA does not support {suffix or 'files without an extension'}")

    @staticmethod
    def _issue(code: str, severity: str, message: str, fix: str) -> dict[str, str]:
        return {"error_code": code, "severity": severity, "location": "document", "message": message,
                "suggested_fix": fix}

    def _pdf(self, path: Path) -> VisualQAReport:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        issues: list[dict[str, str]] = []
        text_pages = 0
        valid_boxes = True
        page_text_lengths: list[int] = []
        for number, page in enumerate(reader.pages, 1):
            width, height = float(page.mediabox.width), float(page.mediabox.height)
            if width <= 0 or height <= 0:
                valid_boxes = False
                issues.append(self._issue("LAYOUT_INVALID_PAGE_BOX", "error", f"Page {number} has invalid geometry.",
                                          "Render the page again with a valid paper size."))
            text = (page.extract_text() or "").strip()
            page_text_lengths.append(len(text))
            if text:
                text_pages += 1
            else:
                issues.append(self._issue("LAYOUT_BLANK_OR_IMAGE_ONLY_PAGE", "warning",
                                          f"Page {number} has no extractable text.",
                                          "Visually verify that it is intentional and not a blank page."))
        if not reader.pages:
            issues.append(self._issue("LAYOUT_NO_PAGES", "error", "PDF contains no pages.", "Regenerate the PDF."))
        prior_average = (
            sum(page_text_lengths[:-1]) / len(page_text_lengths[:-1])
            if len(page_text_lengths) > 1 else 0
        )
        sparse_last_page = (
            len(page_text_lengths) > 1
            and 0 < page_text_lengths[-1] < 500
            and page_text_lengths[-1] < prior_average * 0.35
        )
        if sparse_last_page:
            issues.append(self._issue(
                "LAYOUT_SPARSE_LAST_PAGE", "warning",
                f"The final page contains only {page_text_lengths[-1]} extractable characters.",
                "Keep the final operative paragraph with the signature block or rebalance pagination.",
            ))

        raster_ok, raster_issues = self._inspect_pdf_raster(path)
        issues.extend(raster_issues)
        return self._report("pdf", len(reader.pages), issues, {
            "page_boxes_valid": valid_boxes, "has_extractable_text": text_pages > 0,
            "dynamic_page_count": len(reader.pages) > 0, "raster_bounds_valid": raster_ok,
        })

    def _inspect_pdf_raster(self, path: Path) -> tuple[bool, list[dict[str, str]]]:
        """Raster-level guard for blank or edge-clipped generated pages."""
        try:
            import pypdfium2 as pdfium
        except ImportError:
            return False, [self._issue(
                "LAYOUT_RASTER_CHECK_UNAVAILABLE", "warning", "PDF raster inspection is unavailable.",
                "Install pypdfium2 to enable clipping and visual-blank-page checks.",
            )]

        issues: list[dict[str, str]] = []
        bounds_valid = True
        with pdfium.PdfDocument(str(path)) as document:
            for index in range(len(document)):
                image = document[index].render(scale=1.0).to_pil().convert("L")
                ink = image.point(lambda value: 0 if value > 245 else 255)
                bbox = ink.getbbox()
                page_number = index + 1
                if bbox is None:
                    bounds_valid = False
                    issues.append(self._issue(
                        "LAYOUT_VISUALLY_BLANK_PAGE", "error", f"Page {page_number} is visually blank.",
                        "Remove the blank page or correct pagination before export.",
                    ))
                    continue
                left, top, right, bottom = bbox
                safety = 2
                if left <= safety or top <= safety or right >= image.width - safety or bottom >= image.height - safety:
                    bounds_valid = False
                    issues.append(self._issue(
                        "LAYOUT_CONTENT_TOUCHES_PAGE_EDGE", "error",
                        f"Rendered content on page {page_number} touches the page edge and may be clipped.",
                        "Restore safe page margins and render again.",
                    ))
        return bounds_valid, issues

    def _docx(self, path: Path) -> VisualQAReport:
        issues: list[dict[str, str]] = []
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            required = {"word/document.xml", "word/styles.xml", "[Content_Types].xml"}
            missing = required - names
            for name in sorted(missing):
                issues.append(self._issue("DOCX_REQUIRED_PART_MISSING", "error", f"Missing DOCX part: {name}",
                                          "Regenerate the DOCX package."))
            document_xml = archive.read("word/document.xml").decode("utf-8", errors="replace") if not missing else ""
            if "<w:t" not in document_xml:
                issues.append(self._issue("DOCX_NO_TEXT", "warning", "DOCX contains no text runs.",
                                          "Confirm that an empty document was intended."))
            has_page_fields = "PAGE" in document_xml or any(
                "footer" in name and b"PAGE" in archive.read(name) for name in names if name.endswith(".xml")
            )
        return self._report("docx", 0, issues, {
            "package_valid": not missing, "has_text": "<w:t" in document_xml,
            "page_number_field_present": has_page_fields,
        })

    @staticmethod
    def _report(fmt: str, pages: int, issues: list[dict[str, str]], checks: Mapping[str, bool]) -> VisualQAReport:
        status: Literal["PASS", "WARNING", "ERROR"] = (
            "ERROR" if any(item["severity"] == "error" for item in issues)
            else "WARNING" if issues else "PASS"
        )
        return VisualQAReport(status, fmt, pages, tuple(issues), checks)
