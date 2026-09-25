import asyncio
import io
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import BadRequestError
from app.rag.loader import DocumentLoader
from app.services.document_extraction import DocumentExtractionService
from app.services.handwriting_ocr import HandwritingOCR, HandwritingUnavailableError
from streamlit_app.upload_routing import upload_error_note


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, "transport"])
def test_transient_failure_retries_then_succeeds(monkeypatch, status):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "ocr_enabled", True)
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.handwriting_ocr.asyncio.sleep", sleep)
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) < 3:
            if status == "transport":
                raise httpx.ReadTimeout("private provider URL")
            return httpx.Response(status, text="private document")
        return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
            "parts": [{"text": '{"blocks":[{"kind":"paragraph","text":"Original words"}]}'}],
        }}]})

    page = asyncio.run(HandwritingOCR(httpx.MockTransport(respond)).extract(b"test", 1))
    assert page.blocks[0].text == "Original words"
    assert len(calls) == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]


def test_exhausted_retries_are_service_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr("app.services.handwriting_ocr.asyncio.sleep", AsyncMock())
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(503, text="PRIVATE secret")

    with pytest.raises(HandwritingUnavailableError) as error:
        asyncio.run(HandwritingOCR(httpx.MockTransport(respond)).extract(b"test", 1))
    assert error.value.status_code == 503
    assert len(calls) == 3
    assert "PRIVATE" not in str(error.value)


def test_permanent_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "ocr_enabled", True)
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.handwriting_ocr.asyncio.sleep", sleep)
    with pytest.raises(BadRequestError):
        asyncio.run(HandwritingOCR(httpx.MockTransport(lambda _: httpx.Response(403))).extract(b"test", 1))
    sleep.assert_not_awaited()


def test_wrappers_preserve_503_and_cleanup(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "upload_storage_dir", tmp_path)
    monkeypatch.setattr(DocumentExtractionService, "extract_path", AsyncMock(side_effect=HandwritingUnavailableError("busy")))
    with pytest.raises(HandwritingUnavailableError):
        asyncio.run(DocumentLoader(handwriting=True).load(tmp_path / "test.png"))
    with pytest.raises(HandwritingUnavailableError):
        asyncio.run(DocumentExtractionService(scanner=AsyncMock()).extract_upload(
            UploadFile(filename="test.png", file=io.BytesIO(b"test")),
        ))
    assert list(tmp_path.iterdir()) == []


def test_ui_explains_transient_failure_without_echoing_body():
    response = httpx.Response(503, json={"error": {"code": "handwriting_unavailable", "message": "PRIVATE secret"}})
    note = upload_error_note(response)
    assert "temporarily unavailable" in note
    assert "PRIVATE" not in note
    assert "400" not in note
