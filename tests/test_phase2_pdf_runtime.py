from pathlib import Path

from app.core.exceptions import UnsupportedExportError
from app.drafting.export import ExportOptions, PdfDraftExporter


def test_pdf_runtime_isolated_from_chat_and_has_actionable_fallback(tmp_path: Path) -> None:
    exporter = PdfDraftExporter()
    try:
        path = exporter.export(
            "Cybercrime Complaint",
            {"Facts": "A disputed UPI transaction was reported.", "Signature Block": "Signature: __________"},
            tmp_path / "phase2.pdf",
            options=ExportOptions(document_version=2, generated_on="01 September 2026"),
        )
    except UnsupportedExportError as exc:
        message = str(exc)
        assert "PDF export is unavailable" in message or "PDF export isn't available" in message
        assert "DOCX" in message and "TXT" in message
    else:
        assert path.exists()
        assert path.read_bytes().startswith(b"%PDF")
