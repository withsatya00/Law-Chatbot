"""Phase 4 "Voice chatbot security": `/voice/chat` previously had no auth
dependency at all, no audio size limit (only checked non-empty), and no
MIME-type validation (whatever `content_type` the client claimed was
forwarded straight to Gemini). `/voice/speak` had no text-length bound.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api import voice_router
from app.core.config import settings
from app.schemas.chat import ChatResponse
from app.schemas.common import LawyerRecommendation


class _FakeUploadFile:
    def __init__(self, data: bytes, content_type: str | None) -> None:
        self.content_type = content_type
        self._data = data
        self._pos = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._data[self._pos :]
        else:
            chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


def _fake_response() -> ChatResponse:
    return ChatResponse(
        message_id="m1", session_id="s1", answer="Bail is a conditional release.",
        sources=[], lawyer_recommendation=LawyerRecommendation(category="General", confidence=0.5, reason="n/a"),
        latency_ms=1.0,
        confidence=0.8, confidence_label="High", confidence_reason="grounded",
        detected_language="english", detected_intent="Legal Explanation", conversation_intent="Legal Explanation",
    )


def _patch_chat_service(
    monkeypatch: pytest.MonkeyPatch, response: ChatResponse | None = None, most_recent_session: dict | None = None
) -> MagicMock:
    service = MagicMock()
    service.answer = AsyncMock(return_value=response or _fake_response())
    monkeypatch.setattr(voice_router, "ChatService", lambda: service)
    monkeypatch.setattr(voice_router, "gemini_speech_to_text", AsyncMock(return_value="what is bail"))
    monkeypatch.setattr(voice_router, "gemini_text_to_speech", AsyncMock(return_value=None))
    # Defaults to "nothing found" (`None`) so every pre-existing test below
    # keeps getting the prior behavior (a fresh session_id) unless a test
    # explicitly opts into exercising the multi-turn auto-resume path.
    memory_repo = MagicMock()
    memory_repo.find_most_recent_by_owner = AsyncMock(return_value=most_recent_session)
    monkeypatch.setattr(voice_router, "ConversationMemoryRepository", lambda: memory_repo)
    return service


def test_voice_chat_rejects_an_unsupported_mime_type(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chat_service(monkeypatch)
    audio_file = _FakeUploadFile(b"not-really-audio-but-nonempty", content_type="application/octet-stream")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(voice_router.voice_chat(background_tasks=None, audio_file=audio_file, user_id=None))
    assert exc_info.value.status_code == 415


def test_voice_chat_rejects_a_missing_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chat_service(monkeypatch)
    audio_file = _FakeUploadFile(b"some-bytes", content_type=None)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(voice_router.voice_chat(background_tasks=None, audio_file=audio_file, user_id=None))
    assert exc_info.value.status_code == 415


def test_voice_chat_rejects_audio_over_the_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chat_service(monkeypatch)
    monkeypatch.setattr(settings, "voice_max_audio_mb", 0)  # -> 0-byte effective limit
    audio_file = _FakeUploadFile(b"x" * 10, content_type="audio/wav")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(voice_router.voice_chat(background_tasks=None, audio_file=audio_file, user_id=None))
    assert exc_info.value.status_code == 413


def test_voice_chat_rejects_empty_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chat_service(monkeypatch)
    audio_file = _FakeUploadFile(b"", content_type="audio/wav")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(voice_router.voice_chat(background_tasks=None, audio_file=audio_file, user_id=None))
    assert exc_info.value.status_code == 400


def test_voice_chat_propagates_the_authenticated_user_id(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _patch_chat_service(monkeypatch)
    audio_file = _FakeUploadFile(b"real-audio-bytes", content_type="audio/wav")
    asyncio.run(
        voice_router.voice_chat(
            background_tasks=None, audio_file=audio_file, session_id=None, language=None, user_id="user-A"
        )
    )
    _, kwargs = service.answer.await_args
    assert kwargs["authenticated_user_id"] == "user-A"


def test_voice_chat_accepts_a_valid_supported_mime_type(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chat_service(monkeypatch)
    audio_file = _FakeUploadFile(b"real-audio-bytes", content_type="audio/webm")
    result = asyncio.run(
        voice_router.voice_chat(
            background_tasks=None, audio_file=audio_file, session_id=None, language=None, user_id=None
        )
    )
    assert result["transcribed_text"] == "what is bail"


def test_voice_chat_auto_resumes_the_authenticated_users_most_recent_session(monkeypatch: pytest.MonkeyPatch) -> None:
    # Was previously single-shot in practice: a voice client that (like most
    # voice UIs) doesn't persist `session_id` itself got a brand-new,
    # memory-less conversation on every call. An authenticated user with no
    # explicit `session_id` now resumes their last active one instead.
    service = _patch_chat_service(monkeypatch, most_recent_session={"_id": "session-previous"})
    audio_file = _FakeUploadFile(b"real-audio-bytes", content_type="audio/wav")
    asyncio.run(
        voice_router.voice_chat(
            background_tasks=None, audio_file=audio_file, session_id=None, language=None,
            new_conversation=False, user_id="user-A",
        )
    )
    _, kwargs = service.answer.await_args
    request = service.answer.await_args.args[0]
    assert request.session_id == "session-previous"
    assert kwargs["authenticated_user_id"] == "user-A"


def test_voice_chat_explicit_session_id_is_never_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _patch_chat_service(monkeypatch, most_recent_session={"_id": "session-previous"})
    audio_file = _FakeUploadFile(b"real-audio-bytes", content_type="audio/wav")
    asyncio.run(
        voice_router.voice_chat(
            background_tasks=None, audio_file=audio_file, session_id="session-explicit", language=None,
            new_conversation=False, user_id="user-A",
        )
    )
    request = service.answer.await_args.args[0]
    assert request.session_id == "session-explicit"


def test_voice_chat_new_conversation_flag_skips_auto_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _patch_chat_service(monkeypatch, most_recent_session={"_id": "session-previous"})
    audio_file = _FakeUploadFile(b"real-audio-bytes", content_type="audio/wav")
    asyncio.run(
        voice_router.voice_chat(
            background_tasks=None, audio_file=audio_file, session_id=None, language=None,
            new_conversation=True, user_id="user-A",
        )
    )
    request = service.answer.await_args.args[0]
    assert request.session_id is None


def test_voice_chat_anonymous_caller_never_triggers_a_session_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _patch_chat_service(monkeypatch, most_recent_session={"_id": "session-should-never-be-used"})
    audio_file = _FakeUploadFile(b"real-audio-bytes", content_type="audio/wav")
    asyncio.run(
        voice_router.voice_chat(
            background_tasks=None, audio_file=audio_file, session_id=None, language=None,
            new_conversation=False, user_id=None,
        )
    )
    request = service.answer.await_args.args[0]
    assert request.session_id is None


def test_speak_request_rejects_text_over_the_length_limit() -> None:
    with pytest.raises(ValidationError):
        voice_router.SpeakRequest(text="x" * (voice_router._SPEAK_TEXT_MAX_CHARS + 1))


def test_speak_request_accepts_text_within_the_length_limit() -> None:
    voice_router.SpeakRequest(text="x" * voice_router._SPEAK_TEXT_MAX_CHARS)
