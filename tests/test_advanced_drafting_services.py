from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.drafting.advanced_services import (
    BilingualSemanticValidator,
    LanguageScriptDetector,
    MultilingualOcrRouter,
    TemplateAnalyzer,
    VisualQAService,
)
from app.drafting.document_grammar import ComponentType, document_schema_registry
from app.drafting.export import DocxDraftExporter, PdfDraftExporter
from app.main import app


def test_language_detector_distinguishes_script_and_romanized_language() -> None:
    native = LanguageScriptDetector().detect("मेरी जमीन पर कब्जा हुआ है")
    assert native.language_code == "hi"
    assert native.script == "Devanagari"
    roman = LanguageScriptDetector().detect("meri zameen par kabza hua hai notice banao")
    assert roman.language_code == "hi"
    assert roman.input_script == "Latin"
    assert roman.romanized is True


def test_multilingual_ocr_router_selects_script_pack_without_changing_text() -> None:
    route = MultilingualOcrRouter().route("எனக்கு சட்ட அறிவிப்பு வேண்டும்")
    assert route["provider"] == "tesseract"
    assert route["provider_language"] == "tam+eng"
    assert route["preserve_original_script"] is True


def test_semantic_validator_blocks_changed_amount_and_preserves_email() -> None:
    result = BilingualSemanticValidator().compare(
        "Pay Rs. 1,50,000 by 12/09/2026 to legal@example.com.",
        "12/09/2026 तक Rs. 1,25,000 legal@example.com पर दें।",
    )
    assert result.status == "ERROR"
    assert any(issue["location"] == "amount" for issue in result.issues)
    assert not any(issue["location"] == "email" for issue in result.issues)


def test_template_analysis_extracts_structure_but_never_activates_it() -> None:
    analysis = TemplateAnalyzer().analyze(
        "KISHORI CHAUDHARY ADVOCATE\nLEGAL NOTICE\nSUBJECT\nFACTS\nPRAYER\nSIGNATURE",
        "uploaded_notice_proposal",
    )
    assert analysis.status == "review_required"
    assert analysis.proposed_grammar.active is False
    assert analysis.proposed_grammar.document_family == "advocate_notice"
    assert "KISHORI CHAUDHARY ADVOCATE" in analysis.letterhead_lines


def test_native_high_value_grammars_have_distinct_relief_semantics() -> None:
    notice = document_schema_registry.get("advocate_legal_notice")
    affidavit = document_schema_registry.get("general_affidavit")
    court = document_schema_registry.get("court_application")
    assert notice and affidavit and court
    assert ComponentType.DEMAND in {section.component for section in notice.structure}
    assert ComponentType.COURT_PRAYER not in {section.component for section in affidavit.structure}
    assert ComponentType.COURT_PRAYER in {section.component for section in court.structure}


def test_visual_qa_inspects_real_pdf_and_docx(tmp_path: Path) -> None:
    sections = {"Subject": "Recovery of earned wages", "Facts": "1. Work was performed.", "Signature": "Applicant"}
    pdf = PdfDraftExporter().export("LEGAL NOTICE", sections, tmp_path / "draft.pdf")
    docx = DocxDraftExporter().export("LEGAL NOTICE", sections, tmp_path / "draft.docx")
    pdf_report = VisualQAService().inspect(pdf)
    docx_report = VisualQAService().inspect(docx)
    assert pdf_report.status == "PASS"
    assert pdf_report.page_count >= 1
    assert docx_report.status in {"PASS", "WARNING"}
    assert docx_report.checks["package_valid"] is True


def test_visual_qa_reports_sparse_last_pdf_page_when_present(tmp_path: Path) -> None:
    sections = {
        "Body": "Long paragraph. " * 850,
        "Signature Block": "Place: Delhi\nDate: 18 September 2026\n\nYours faithfully,\n\n(QA User)",
    }
    pdf = PdfDraftExporter().export("QA NOTICE", sections, tmp_path / "sparse.pdf")

    report = VisualQAService().inspect(pdf)

    assert report.checks["raster_bounds_valid"] is True
    if report.page_count > 1:
        last_page_chars = len((PdfReader(str(pdf)).pages[-1].extract_text() or "").strip())
        sparse = [issue for issue in report.issues if issue["error_code"] == "LAYOUT_SPARSE_LAST_PAGE"]
        prior_lengths = [len((page.extract_text() or "").strip()) for page in PdfReader(str(pdf)).pages[:-1]]
        prior_average = sum(prior_lengths) / len(prior_lengths)
        expected_sparse = 0 < last_page_chars < 500 and last_page_chars < prior_average * 0.35
        assert bool(sparse) is expected_sparse


def test_advanced_service_endpoints() -> None:
    client = TestClient(app)
    detected = client.post("/language/detect", json={"text": "mujhe notice banana hai"})
    assert detected.status_code == 200
    assert detected.json()["romanized"] is True

    analyzed = client.post("/draft/analyze-template", json={
        "document_id": "sample_notice_schema", "extracted_text": "LEGAL NOTICE\nSUBJECT\nFACTS\nSIGNATURE",
    })
    assert analyzed.status_code == 200
    assert analyzed.json()["status"] == "review_required"

    semantic = client.post("/draft/back-translation-check", json={
        "source_text": "Pay Rs. 50,000.", "target_text": "Pay Rs. 50,000.",
    })
    assert semantic.status_code == 200
    assert semantic.json()["status"] == "PASS"
