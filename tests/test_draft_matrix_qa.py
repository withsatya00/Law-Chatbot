import asyncio

from app.drafting.engine import LegalDraftEngine
from app.drafting.language_support import BODY_ONLY_LANGUAGES, FULLY_SUPPORTED_LANGUAGES
from app.drafting.matrix_qa import (
    ACCEPTED_DRAFT_LANGUAGES,
    EmptyLegalRetriever,
    MatrixCaseResult,
    OfflineDraftLLM,
    run_export_smoke,
    run_matrix_sync,
    synthetic_fields,
    validate_case,
)
from app.drafting.templates import get_template, list_templates
from app.schemas.drafting import DraftPreviewRequest


def test_matrix_language_registry_covers_all_24_accepted_languages() -> None:
    assert len(ACCEPTED_DRAFT_LANGUAGES) == 24
    assert set(ACCEPTED_DRAFT_LANGUAGES) == FULLY_SUPPORTED_LANGUAGES | BODY_ONLY_LANGUAGES


def test_synthetic_field_factory_completes_every_registered_template() -> None:
    for template in list_templates():
        fields = synthetic_fields(template)
        assert template.required_field_keys() <= {key for key, value in fields.items() if value}


def test_representative_category_language_matrix_has_no_structural_failures() -> None:
    template_ids = {
        "Notice": "legal_notice",
        "Complaint": "police_complaint",
        "Affidavit": "lost_document_affidavit",
        "Application": "rti_application",
        "Contract": "rent_agreement",
    }
    languages = ("english", "hindi", "tamil", "urdu", "assamese", "santali")
    engine = LegalDraftEngine()
    engine.llm = OfflineDraftLLM()
    engine.retriever = EmptyLegalRetriever()

    for category, template_id in template_ids.items():
        template = get_template(template_id)
        assert template is not None and template.category == category
        for language in languages:
            response = asyncio.run(
                engine.preview(
                    DraftPreviewRequest(
                        draft_id=template_id,
                        language=language,
                        fields=synthetic_fields(template),
                    )
                )
            )
            assert response.status == "complete"
            assert not validate_case(template, language, response.sections, response.full_text)


def test_full_registered_template_language_matrix_has_no_structural_failures() -> None:
    results = run_matrix_sync(LegalDraftEngine())
    expected = len(list_templates()) * len(ACCEPTED_DRAFT_LANGUAGES)

    assert len(results) == expected
    failures = [result.as_dict() for result in results if result.status != "PASS"]
    assert not failures, failures[:20]


def test_export_smoke_checks_all_four_formats_and_expected_pdf_skip(tmp_path) -> None:
    template = get_template("legal_notice")
    assert template is not None
    engine = LegalDraftEngine()
    engine.llm = OfflineDraftLLM()
    engine.retriever = EmptyLegalRetriever()
    cases = []
    for language in ("hindi", "santali"):
        response = asyncio.run(engine.preview(DraftPreviewRequest(
            draft_id=template.draft_id, language=language, fields=synthetic_fields(template)
        )))
        cases.append(MatrixCaseResult(
            template.draft_id, template.category, language,
            "full" if language == "hindi" else "body_only", "PASS", (), response.sections, response.full_text,
        ))

    reports = run_export_smoke(cases, tmp_path, full_visual_matrix=True)

    assert {(report.language, report.format) for report in reports} == {
        (language, fmt) for language in ("hindi", "santali") for fmt in ("txt", "rtf", "docx", "pdf")
    }
    assert next(report for report in reports if report.language == "santali" and report.format == "pdf").status == "SKIP"
    assert all(
        report.status in {"PASS", "WARNING"}
        for report in reports
        if not (report.language == "santali" and report.format == "pdf")
    )
