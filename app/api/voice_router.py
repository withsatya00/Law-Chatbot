import base64
import io
import re
import wave
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import get_current_user_id
from app.core.config import settings
from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError
from app.repositories.conversation_memory import ConversationMemoryRepository
from app.repositories.drafts import DraftRepository
from app.schemas.chat import ChatRequest
from app.schemas.phase3 import VoiceDraftConfirmationRequest
from app.services.chat_service import ChatService
from app.services.voice_vad import detect_and_trim_silence

log = structlog.get_logger(__name__)

router = APIRouter(tags=["voice"])

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_STT_TIMEOUT_SECONDS = 30.0
_TTS_TIMEOUT_SECONDS = 30.0
# Only formats a browser/mobile recorder plausibly produces and Gemini STT
# actually accepts -- an unrecognized/absent content-type is rejected rather
# than guessed at and forwarded to the upstream API regardless.
_ALLOWED_AUDIO_MIME_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/webm",
    "audio/ogg", "audio/opus",
    "audio/mp4", "audio/x-m4a", "audio/aac",
    "audio/flac",
}
# Voice input is a spoken question, not a document dictation -- bounded well
# under Gemini's own limits so one request can't be used to smuggle an
# arbitrarily large prompt through as "text to speak".
_SPEAK_TEXT_MAX_CHARS = 20_000
# Gemini's native-audio TTS models return raw PCM (mimeType like
# "audio/L16;codec=pcm;rate=24000"), not a self-contained playable file --
# a browser/mobile `<audio>` element can't play headerless PCM directly, so
# this is always re-parsed off the response mimeType (never assumed) and
# wrapped in a WAV header below. This constant is only the fallback for the
# rare case the API omits a rate.
_DEFAULT_PCM_SAMPLE_RATE_HZ = 24000


async def gemini_speech_to_text(audio_bytes: bytes, mime_type: str) -> str:
    if not settings.gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key is not configured.")

    model = settings.gemini_stt_model or "gemini-3.6-flash"
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Transcribe this audio precisely."},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(audio_bytes).decode("utf-8"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {"temperature": 0.0},
    }

    try:
        async with httpx.AsyncClient(timeout=_STT_TIMEOUT_SECONDS) as client:
            response = await client.post(url, params={"key": settings.gemini_api_key}, json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        log.warning("gemini_stt_unavailable", error=str(exc))
        raise HTTPException(status_code=502, detail="Gemini API is unreachable. Please try again.") from exc
    except httpx.HTTPStatusError as exc:
        log.warning("gemini_stt_http_error", error=str(exc), status=exc.response.status_code)
        raise HTTPException(status_code=502, detail="Gemini speech-to-text request failed.") from exc

    candidates = data.get("candidates") or []
    if not candidates:
        raise HTTPException(status_code=422, detail="Could not transcribe audio: no candidates returned.")

    parts = candidates[0].get("content", {}).get("parts", [])
    transcript = "".join(part.get("text", "") for part in parts if not part.get("thought")).strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="Could not transcribe audio: empty transcript.")
    return transcript


# Markdown syntax the chat answer legitimately contains (**bold**, ## headings,
# `-`/`1.` list markers, `>` blockquotes) reads terribly aloud verbatim
# ("asterisk asterisk bold asterisk asterisk") -- stripped to plain prose
# before TTS, text-only, never touching what's shown in the chat UI itself.
_MARKDOWN_STRIP_PATTERNS = (
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), r"\1"),
    (re.compile(r"^\s*[-*]\s+", re.MULTILINE), ""),
    (re.compile(r"^\s*\d+\.\s+", re.MULTILINE), ""),
    (re.compile(r"^\s*>\s?", re.MULTILINE), ""),
)
# TTS is meant to read a short spoken answer aloud, not narrate an entire
# multi-page legal notice -- capped well under Gemini's own input limits so
# a single voice reply can't balloon into a multi-minute audio clip.
_TTS_MAX_CHARS = 2000


def _plain_text_for_speech(markdown_text: str) -> str:
    text = markdown_text
    for pattern, replacement in _MARKDOWN_STRIP_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_TTS_MAX_CHARS]


def _pcm_to_wav(pcm_bytes: bytes, mime_type: str) -> bytes:
    """Gemini TTS returns headerless 16-bit PCM -- wraps it in a standard WAV
    container (via the stdlib `wave` module) so it's a real, directly
    playable audio file rather than raw samples the caller can't do
    anything with. Sample rate is parsed from the response's own
    `mimeType` (e.g. "audio/L16;codec=pcm;rate=24000") since Gemini can
    vary it by model/voice -- never hardcoded past the documented fallback.
    """
    rate_match = re.search(r"rate=(\d+)", mime_type)
    sample_rate = int(rate_match.group(1)) if rate_match else _DEFAULT_PCM_SAMPLE_RATE_HZ
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


async def gemini_text_to_speech(text: str) -> bytes | None:
    """Real Gemini native-audio TTS -- replaces the previous hardcoded
    `b"dummy_audio_blob"` stub. Returns `None` (never raises) on any
    failure: voice chat's transcript/text answer is the primary value, so a
    TTS hiccup should degrade to text-only rather than fail the whole
    request, the same graceful-degradation philosophy already used for
    BM25/Atlas fallbacks elsewhere in this codebase.
    """
    if not settings.gemini_api_key:
        log.warning("gemini_tts_not_configured")
        return None
    speech_text = _plain_text_for_speech(text)
    if not speech_text:
        return None

    url = f"{GEMINI_API_BASE}/models/{settings.gemini_tts_model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": speech_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": settings.gemini_tts_voice}}
            },
        },
    }

    try:
        async with httpx.AsyncClient(timeout=_TTS_TIMEOUT_SECONDS) as client:
            response = await client.post(url, params={"key": settings.gemini_api_key}, json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        log.warning("gemini_tts_unavailable", error=str(exc))
        return None
    except httpx.HTTPStatusError as exc:
        log.warning("gemini_tts_http_error", error=str(exc), status=exc.response.status_code)
        return None

    candidates = data.get("candidates") or []
    if not candidates:
        log.warning("gemini_tts_no_candidates")
        return None
    parts = candidates[0].get("content", {}).get("parts", [])
    inline_audio = next((part.get("inlineData") for part in parts if part.get("inlineData")), None)
    if not inline_audio or not inline_audio.get("data"):
        log.warning("gemini_tts_no_audio_data")
        return None

    try:
        pcm_bytes = base64.b64decode(inline_audio["data"])
    except (ValueError, TypeError) as exc:
        log.warning("gemini_tts_decode_failed", error=str(exc))
        return None
    return _pcm_to_wav(pcm_bytes, inline_audio.get("mimeType", ""))


@router.post("/voice/chat")
async def voice_chat(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile,
    session_id: str | None = Form(None),
    language: str | None = Form(None),
    new_conversation: bool = Form(False),
    user_id: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    # Multi-turn by default for a logged-in voice client: previously a
    # client that (reasonably, for a voice UI) doesn't persist `session_id`
    # itself got a brand-new, memory-less conversation on every single
    # utterance -- `ChatService`'s own conversation memory, entity recall,
    # and draft state all already work fine across calls once the SAME
    # `session_id` is reused, so this is a client-ergonomics gap, not a
    # missing backend capability. `session_id` explicitly provided always
    # wins (a client that DOES manage its own session state is unaffected);
    # `new_conversation=true` is the explicit escape hatch for "start over."
    if not session_id and user_id and not new_conversation:
        recent = await ConversationMemoryRepository().find_most_recent_by_owner(user_id)
        if recent:
            session_id = recent["_id"]

    mime_type = audio_file.content_type
    if mime_type not in _ALLOWED_AUDIO_MIME_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported audio content type: {mime_type!r}.")

    # Bounded chunked read (same pattern as `DocumentService.upload_and_index`)
    # rather than `await audio_file.read()` -- previously unbounded, so a
    # multi-GB upload would be read fully into memory before any check ran.
    chunks: list[bytes] = []
    size = 0
    while chunk := await audio_file.read(1024 * 1024):
        size += len(chunk)
        if size > settings.voice_max_audio_bytes:
            raise HTTPException(
                status_code=413, detail=f"Audio upload exceeds {settings.voice_max_audio_mb} MB limit."
            )
        chunks.append(chunk)
    audio_bytes = b"".join(chunks)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    # C9 "VAD before STT" (see app/services/voice_vad.py): a recording with
    # no speech anywhere is rejected before spending a Gemini STT call on
    # it, and one with real speech has its leading/trailing silence
    # trimmed. `applicable=False` for any MIME type this module doesn't
    # decode, so this is a no-op for those.
    vad_result = await detect_and_trim_silence(audio_bytes, mime_type)
    if vad_result.applicable and vad_result.is_silent:
        raise HTTPException(status_code=422, detail="No speech was detected in the recording. Please try again.")
    if vad_result.applicable and vad_result.audio_bytes:
        audio_bytes = vad_result.audio_bytes
        # The compressed (webm/opus) decode path re-encodes to WAV -- the
        # original upload's mime type no longer describes `audio_bytes` once
        # that happened, and sending it to Gemini mislabeled would break
        # transcription. The WAV path's output is WAV too, so this is a
        # no-op for it (`output_mime_type` is already the same "audio/wav").
        mime_type = vad_result.output_mime_type or mime_type

    transcribed_text = await gemini_speech_to_text(audio_bytes, mime_type)

    chat_request = ChatRequest(question=transcribed_text, session_id=session_id, language=language)
    response = await ChatService().answer(chat_request, background_tasks, authenticated_user_id=user_id)

    voice_confirmation_required = False
    if response.draft and response.draft.draft_id:
        voice_confirmation_required = True
        await DraftRepository().update_by_id(response.draft.draft_id, {
            "voice_collected": True,
            "voice_final_confirmed": False,
        })

    audio_bytes_out = await gemini_text_to_speech(response.answer)
    audio_base64 = base64.b64encode(audio_bytes_out).decode("utf-8") if audio_bytes_out else None

    return {
        "session_id": response.session_id,
        "transcribed_text": transcribed_text,
        "ai_response_text": response.answer,
        "audio_base64": audio_base64,
        "audio_format": "wav" if audio_base64 else None,
        "confidence": response.confidence,
        "intent": response.detected_intent,
        "voice_final_confirmation_required": voice_confirmation_required,
        "draft": response.draft.model_dump(mode="json") if response.draft else None,
    }


@router.post("/voice/drafts/{draft_id}/confirm")
async def confirm_voice_draft(
    draft_id: str,
    request: VoiceDraftConfirmationRequest,
    user_id: str | None = Depends(get_current_user_id),
) -> dict[str, object]:
    draft = await DraftRepository().find_by_id(draft_id)
    if draft is None:
        raise NotFoundError("Draft not found.")
    if draft.get("user_id"):
        if user_id != draft.get("user_id"):
            raise ForbiddenError("You do not have access to this draft.")
    elif draft.get("session_id") != request.session_id:
        raise ForbiddenError("You do not have access to this draft.")
    if not draft.get("voice_collected"):
        raise BadRequestError("This draft was not collected through the voice workflow.")
    if request.confirmation_text.strip().upper() != "I CONFIRM THE REVIEWED FACTS":
        raise BadRequestError("Type 'I CONFIRM THE REVIEWED FACTS' after reviewing the draft.")
    await DraftRepository().update_by_id(draft_id, {
        "voice_final_confirmed": True,
        "voice_confirmed_at": datetime.now(UTC),
    })
    return {"draft_id": draft_id, "voice_final_confirmed": True, "external_submission": False}


class SpeakRequest(BaseModel):
    text: str = Field(max_length=_SPEAK_TEXT_MAX_CHARS)


@router.post("/voice/speak")
async def voice_speak(request: SpeakRequest) -> dict[str, Any]:
    """Standalone text -> speech, independent of `/voice/chat`'s full
    voice-in/voice-out round trip -- covers the "read this answer aloud"
    case for a user who typed their question through ordinary text chat
    and only wants the reply spoken back, not re-run through STT/the LLM.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text to speak must not be empty.")
    audio_bytes_out = await gemini_text_to_speech(request.text)
    if audio_bytes_out is None:
        raise HTTPException(status_code=502, detail="Text-to-speech is temporarily unavailable. Please try again.")
    return {
        "audio_base64": base64.b64encode(audio_bytes_out).decode("utf-8"),
        "audio_format": "wav",
    }
