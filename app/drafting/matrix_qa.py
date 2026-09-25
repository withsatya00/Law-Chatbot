"""Registry-driven draft QA across every template and accepted language.

This module deliberately uses the deterministic generation path.  It tests
the part of the system we control (grammar, sections, localization chrome,
Unicode and exporters) without turning a provider outage or model variation
into a false product regression.  Live-LLM quality remains a separate suite.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import UnsupportedExportError
from app.drafting import advocate_register
from app.drafting.advanced_services import VisualQAService
from app.drafting.export import (
    DocxDraftExporter,
    ExportOptions,
    PdfDraftExporter,
    RtfDraftExporter,
    TxtDraftExporter,
)
from app.drafting.language_support import BODY_ONLY_LANGUAGES, FULLY_SUPPORTED_LANGUAGES
from app.drafting.templates import list_templates
from app.drafting.templates.base import DraftField, DraftTemplateDefinition, structure_sections_for
from app.drafting.title_translations import localized_title
from app.llm.base import LLMResponse
from app.schemas.drafting import DraftPreviewRequest

ACCEPTED_DRAFT_LANGUAGES: tuple[str, ...] = tuple(
    sorted(FULLY_SUPPORTED_LANGUAGES | BODY_ONLY_LANGUAGES)
)

_MOJIBAKE_MARKERS = ("ï¿½", "â€", "â€™", "Ã", "Â", "\ufffd")
_FORBIDDEN_BY_CATEGORY: dict[str, frozenset[str]] = {
    "Notice": frozenset({"Court / Authority Name", "Verification"}),
    "Affidavit": frozenset({"Prayer", "Request"}),
    "Application": frozenset({"Prayer", "Verification"}),
    "Contract": frozenset({"Prayer", "Verification", "Recipient"}),
}


class OfflineDraftLLM:
    """Forces the production deterministic fallback without network access."""

    provider_name = "matrix-offline"

    async def chat(self, _messages: list[Any]) -> LLMResponse:
        return LLMResponse(
            content="",
            model="matrix-offline",
            provider=self.provider_name,
            error="offline matrix QA intentionally disabled the LLM",
            error_kind="provider_error",
        )


class EmptyLegalRetriever:
    async def retrieve(self, _query: str, **_kwargs: Any) -> tuple[str, list[Any]]:
        return "", []


@dataclass(frozen=True)
class MatrixCaseResult:
    template_id: str
    category: str
    language: str
    support_level: str
    status: str
    issues: tuple[str, ...]
    sections: dict[str, str]
    full_text: str

    def as_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "template_id": self.template_id,
            "category": self.category,
            "language": self.language,
            "support_level": self.support_level,
            "status": self.status,
            "issues": list(self.issues),
            "section_headings": list(self.sections),
        }
        if include_text:
            payload["sections"] = self.sections
            payload["full_text"] = self.full_text
        return payload


@dataclass(frozen=True)
class ExportCaseResult:
    template_id: str
    category: str
    language: str
    format: str
    status: str
    issues: tuple[str, ...]
    path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "category": self.category,
            "language": self.language,
            "format": self.format,
            "status": self.status,
            "issues": list(self.issues),
            "path": self.path,
        }


def synthetic_value(field: DraftField) -> str:
    """Stable, harmless value suitable for any registered field."""
    key = field.key
    if field.field_type == "date" or key.endswith("_date"):
        return "15 September 2026"
    if field.field_type == "tel" or "mobile" in key or "phone" in key:
        return "9876543210"
    if field.field_type == "email" or "email" in key:
        return "qa@example.test"
    if field.field_type == "number" or any(token in key for token in ("amount", "rent", "price", "days", "age")):
        return "25000"
    if key == "facts":
        return "The applicant states that the synthetic QA event occurred on 15 September 2026."
    if key in {"expected_relief", "relief_sought"}:
        return "Appropriate relief in accordance with law"
    if key == "information_sought":
        return "Certified copies of the relevant synthetic records"
    if key == "place" or key.endswith("_place") or "location" in key:
        return "New Delhi"
    if "address" in key or "premises" in key or "property" in key:
        return "12 QA Road, New Delhi - 110001"
    if "name" in key:
        return f"QA {key.replace('_', ' ').title()}"
    if "number" in key or key in {"imei", "registration_no"}:
        return "QA-2026-001"
    return f"Synthetic {field.label}"


def synthetic_fields(template: DraftTemplateDefinition) -> dict[str, str]:
    values = {field.key: synthetic_value(field) for field in template.required_fields}
    # These optional values exercise central invariants without claiming an
    # advocate-client relationship or inventing evidence/annexures.
    optional = {field.key: field for field in template.optional_fields}
    if "representation_mode" in optional:
        values["representation_mode"] = "self"
    return values


def validate_case(
    template: DraftTemplateDefinition,
    language: str,
    sections: dict[str, str],
    full_text: str,
) -> tuple[str, ...]:
    issues: list[str] = []
    required = structure_sections_for(template.category)
    allowed_request_substitution = template.category == "Complaint" and "Request" in sections
    for heading in required:
        if heading == "Prayer" and allowed_request_substitution:
            continue
        if heading not in sections:
            issues.append(f"missing_section:{heading}")
    for heading in _FORBIDDEN_BY_CATEGORY.get(template.category, frozenset()):
        if heading in sections:
            issues.append(f"forbidden_section:{heading}")
    for heading, text in sections.items():
        if not text.strip():
            issues.append(f"empty_section:{heading}")
    for marker in _MOJIBAKE_MARKERS:
        if marker in full_text:
            issues.append(f"mojibake:{marker}")
    if re.search(r"\[\s*(?:not provided|insert|placeholder)[^\]]*\]", full_text, re.IGNORECASE):
        issues.append("unresolved_placeholder")
    if template.category == "Affidavit" and "Prayer" in sections:
        issues.append("affidavit_has_prayer")
    if template.category == "Application" and "Prayer" in sections:
        issues.append("application_has_prayer")
    if (
        template.category == "Complaint"
        and language in FULLY_SUPPORTED_LANGUAGES - {"english"}
        and "Verified that the contents of this complaint/application" in full_text
    ):
        issues.append("english_complaint_verification_leak")
    return tuple(dict.fromkeys(issues))


async def run_matrix(engine: Any) -> list[MatrixCaseResult]:
    engine.llm = OfflineDraftLLM()
    engine.retriever = EmptyLegalRetriever()
    results: list[MatrixCaseResult] = []
    for template in list_templates():
        fields = synthetic_fields(template)
        for language in ACCEPTED_DRAFT_LANGUAGES:
            try:
                response = await engine.preview(
                    DraftPreviewRequest(draft_id=template.draft_id, language=language, fields=fields)
                )
                issues = list(validate_case(template, language, response.sections, response.full_text))
                if response.status != "complete":
                    issues.append(f"generation_status:{response.status}")
                results.append(
                    MatrixCaseResult(
                        template_id=template.draft_id,
                        category=template.category,
                        language=language,
                        support_level="full" if language in FULLY_SUPPORTED_LANGUAGES else "body_only",
                        status="PASS" if not issues else "FAIL",
                        issues=tuple(issues),
                        sections=response.sections,
                        full_text=response.full_text,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - matrix must report every failing cell
                results.append(
                    MatrixCaseResult(
                        template_id=template.draft_id,
                        category=template.category,
                        language=language,
                        support_level="full" if language in FULLY_SUPPORTED_LANGUAGES else "body_only",
                        status="ERROR",
                        issues=(f"{type(exc).__name__}:{exc}",),
                        sections={},
                        full_text="",
                    )
                )
    return results


def write_draft_artifacts(results: list[MatrixCaseResult], output_dir: Path) -> None:
    drafts_dir = output_dir / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        path = drafts_dir / result.language / f"{result.template_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.full_text, encoding="utf-8")


def _export_candidates(results: list[MatrixCaseResult], *, full: bool) -> list[MatrixCaseResult]:
    if full:
        return results
    chosen: dict[tuple[str, str], MatrixCaseResult] = {}
    for result in results:
        chosen.setdefault((result.language, result.category), result)
    return list(chosen.values())


def run_export_smoke(
    results: list[MatrixCaseResult], output_dir: Path, *, full_visual_matrix: bool = False
) -> list[ExportCaseResult]:
    """Exercise all text exporters plus a bounded/full visual matrix."""
    exporters: dict[str, Any] = {
        "txt": TxtDraftExporter(), "rtf": RtfDraftExporter(),
        "docx": DocxDraftExporter(), "pdf": PdfDraftExporter(),
    }
    visual_qa = VisualQAService()
    template_map = {template.draft_id: template for template in list_templates()}
    visual_cases = _export_candidates(results, full=full_visual_matrix)
    cases_by_format = {
        "txt": results, "rtf": results, "docx": visual_cases, "pdf": visual_cases,
    }
    reports: list[ExportCaseResult] = []
    for fmt, cases in cases_by_format.items():
        for case in cases:
            template = template_map[case.template_id]
            destination = output_dir / "exports" / fmt / case.language / f"{case.template_id}.{fmt}"
            options = ExportOptions(flowing_letter=(
                template.category == "Notice"
                or template.draft_id in advocate_register.FLOWING_LETTER_DRAFTS
            ))
            try:
                path = exporters[fmt].export(
                    localized_title(template, case.language), case.sections, destination, case.language, options
                )
                issues: list[str] = []
                if not path.exists() or path.stat().st_size == 0:
                    issues.append("empty_export")
                if fmt in {"pdf", "docx"} and not issues:
                    visual = visual_qa.inspect(path)
                    issues.extend(f"{item['severity']}:{item['error_code']}" for item in visual.issues)
                if fmt == "txt" and not issues:
                    exported = path.read_text(encoding="utf-8")
                    if not all(
                        text.splitlines()[0] in exported for text in case.sections.values() if text.splitlines()
                    ):
                        issues.append("content_parity_mismatch")
                has_error = any(issue.startswith("error:") for issue in issues)
                has_warning = any(issue.startswith("warning:") for issue in issues)
                reports.append(ExportCaseResult(
                    case.template_id, case.category, case.language, fmt,
                    "FAIL" if has_error or (issues and not has_warning) else "WARNING" if has_warning else "PASS",
                    tuple(issues), str(path),
                ))
            except UnsupportedExportError as exc:
                expected = fmt == "pdf" and case.language in {"santali", "manipuri"}
                reports.append(ExportCaseResult(
                    case.template_id, case.category, case.language, fmt,
                    "SKIP" if expected else "ERROR", (str(exc),), str(destination),
                ))
            except Exception as exc:  # noqa: BLE001 - report every exporter cell
                reports.append(ExportCaseResult(
                    case.template_id, case.category, case.language, fmt,
                    "ERROR", (f"{type(exc).__name__}:{exc}",), str(destination),
                ))
    return reports


def run_matrix_sync(engine: Any) -> list[MatrixCaseResult]:
    return asyncio.run(run_matrix(engine))
