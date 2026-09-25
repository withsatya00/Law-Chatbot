"""C9 "VAD before STT": energy-based Voice Activity Detection, run on a
`/voice/chat` audio upload BEFORE it reaches Gemini STT
(`app.api.voice_router.gemini_speech_to_text`).

Scope: `audio/wav`/`audio/x-wav`/`audio/wave` decode natively via Python's
stdlib `wave` module. `audio/webm` and `audio/opus` (the browser
`MediaRecorder` default and its raw-Opus-container sibling) decode via
`av` (PyAV, ships its own statically-linked FFmpeg libs -- no system
`ffmpeg` binary or Dockerfile change needed). The other compressed formats
`voice_router._ALLOWED_AUDIO_MIME_TYPES` accepts (mp3/ogg-non-opus/m4a/aac/
flac) remain out of scope -- `VadResult.applicable=False` for them, and
`voice_chat` calls Gemini STT exactly as before, unmodified.

Decoding an untrusted compressed upload is bounded two ways (see
`settings.voice_vad_decode_timeout_seconds`/`voice_vad_max_decode_seconds`):
a wall-clock timeout (a corrupt/adversarial stream that stalls the decoder)
and a maximum decoded-duration cap enforced WHILE decoding, not after (Opus
compresses well enough that a small upload can still decode to a wildly
disproportionate PCM duration -- a decompression-bomb shape -- so this
aborts mid-stream rather than fully decoding first and checking after).
Either bound tripping fails open (`applicable=False`), same as an
unparseable WAV -- a VAD/decode hiccup must never block what would
otherwise be a legitimate voice request.

What this buys, on top of the module's original WAV-only behavior:
* A recording that is entirely silence is rejected BEFORE spending a Gemini
  STT call and round-trip on it.
* A recording with real speech but long silence at the start/end is trimmed
  to the speech span plus a fixed padding margin.
Both now apply to `audio/webm`/`audio/opus` uploads too, not only WAV.
"""

import asyncio
import io
import wave
from dataclasses import dataclass

import numpy as np
import structlog

from app.core.config import settings
from app.observability.metrics import metrics

log = structlog.get_logger(__name__)

_WAV_MIME_TYPES = {"audio/wav", "audio/x-wav", "audio/wave"}
_COMPRESSED_DECODE_MIME_TYPES = {"audio/webm", "audio/opus"}
# Full-scale reference for 16-bit signed PCM, used to convert RMS energy to
# dBFS (decibels relative to full scale) -- the standard, sample-rate- and
# bit-depth-independent unit for a silence threshold.
_INT16_FULL_SCALE = 32768.0
# Output WAV is always resampled/downmixed to this rate for the decoded
# (non-WAV) path -- STT accuracy for speech doesn't need more, and it bounds
# the trimmed payload size independent of the source's original rate.
_DECODED_OUTPUT_RATE_HZ = 16000


@dataclass(frozen=True)
class VadResult:
    # False for any unsupported MIME type, an unparseable/undecodable
    # upload, a sample width this module doesn't handle, a decode that
    # exceeded its time/duration bound, or `settings.voice_vad_enabled` off
    # -- `audio_bytes`/`is_silent` carry no meaning in that case, and the
    # caller should proceed with the ORIGINAL upload exactly as it would
    # have before VAD existed.
    applicable: bool
    is_silent: bool = False
    # The (possibly trimmed) WAV bytes to send to STT instead of the
    # original upload. `None` when `applicable` is False or `is_silent` is
    # True (nothing to send to STT in the silent case).
    audio_bytes: bytes | None = None
    # The MIME type `audio_bytes` actually is -- always "audio/wav" once
    # populated, regardless of the ORIGINAL upload's mime (the compressed
    # decode path re-encodes to WAV; the WAV path was WAV already). The
    # caller must send THIS mime type to STT alongside `audio_bytes`, not
    # the original upload's mime type, or a decoded-then-relabeled WAV would
    # be sent to Gemini mislabeled as webm/opus.
    output_mime_type: str | None = None
    original_duration_seconds: float = 0.0
    trimmed_duration_seconds: float = 0.0


def _encode_wav_mono16(samples: np.ndarray, frame_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(frame_rate)
        wav_out.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


def _analyze_and_trim(samples: np.ndarray, frame_rate: int) -> VadResult:
    """Shared energy-based VAD core: `samples` is mono float32 PCM at
    `frame_rate` Hz, from either the WAV or the compressed-decode path.
    """
    total_samples = len(samples)
    if total_samples == 0 or frame_rate <= 0:
        return VadResult(applicable=False)

    frame_samples = max(1, int(frame_rate * settings.voice_vad_frame_ms / 1000))
    frame_count = -(-total_samples // frame_samples)  # ceil division
    padded_length = frame_count * frame_samples
    if padded_length > total_samples:
        samples = np.pad(samples, (0, padded_length - total_samples))
    frames = samples.reshape(frame_count, frame_samples)

    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    with np.errstate(divide="ignore"):
        dbfs = 20 * np.log10(np.maximum(rms, 1e-9) / _INT16_FULL_SCALE)
    speech_frames = np.flatnonzero(dbfs >= settings.voice_vad_silence_threshold_dbfs)

    original_duration = total_samples / frame_rate
    if speech_frames.size == 0:
        log.info("voice_vad_silence_detected", duration_seconds=round(original_duration, 2))
        return VadResult(applicable=True, is_silent=True, original_duration_seconds=original_duration)

    padding_frames = max(1, round(settings.voice_vad_padding_ms / settings.voice_vad_frame_ms))
    first_frame = max(0, int(speech_frames[0]) - padding_frames)
    last_frame = min(frame_count - 1, int(speech_frames[-1]) + padding_frames)
    start_sample = first_frame * frame_samples
    end_sample = min(total_samples, (last_frame + 1) * frame_samples)

    trimmed_samples = samples[start_sample:end_sample]
    trimmed_duration = len(trimmed_samples) / frame_rate
    log.info(
        "voice_vad_trimmed",
        original_seconds=round(original_duration, 2), trimmed_seconds=round(trimmed_duration, 2),
    )
    return VadResult(
        applicable=True, is_silent=False,
        audio_bytes=_encode_wav_mono16(trimmed_samples, frame_rate),
        output_mime_type="audio/wav",
        original_duration_seconds=original_duration, trimmed_duration_seconds=trimmed_duration,
    )


def _detect_and_trim_wav(audio_bytes: bytes) -> VadResult:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_in:
            channels = wav_in.getnchannels()
            sample_width = wav_in.getsampwidth()
            frame_rate = wav_in.getframerate()
            num_frames = wav_in.getnframes()
            raw = wav_in.readframes(num_frames)
    except (wave.Error, EOFError) as exc:
        log.warning("voice_vad_wav_parse_failed", error=str(exc))
        return VadResult(applicable=False)

    # 16-bit PCM is essentially universal for recorded WAV; anything else
    # (8-bit, 24-bit, float) is rare enough here that failing open is safer
    # than adding untested decode paths for formats no client actually sends.
    if sample_width != 2 or num_frames == 0 or channels < 1 or frame_rate <= 0:
        return VadResult(applicable=False)

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        # Downmixed to mono for VAD analysis AND for the trimmed output --
        # a voice question is speech content, not a stereo recording whose
        # channel separation matters, and mono halves (or more) the payload
        # sent to STT for a stereo upload.
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)

    return _analyze_and_trim(samples, frame_rate)


def _decode_compressed_audio_sync(audio_bytes: bytes, mime_type: str) -> tuple[np.ndarray, int] | None:
    """Decodes a webm/opus upload to mono PCM at `_DECODED_OUTPUT_RATE_HZ`,
    aborting mid-stream (returns `None`) if the decoded audio exceeds
    `settings.voice_vad_max_decode_seconds` -- see module docstring. Runs
    synchronously; the caller bounds wall-clock time and offloads this off
    the event loop (PyAV/FFmpeg decode is CPU-bound, non-async).
    """
    import av  # local import: only paid for when a webm/opus upload actually needs it

    max_samples = int(settings.voice_vad_max_decode_seconds * _DECODED_OUTPUT_RATE_HZ)
    try:
        with av.open(io.BytesIO(audio_bytes), mode="r") as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=_DECODED_OUTPUT_RATE_HZ)
            chunks: list[np.ndarray] = []
            total = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    for resampled in resampler.resample(frame):
                        arr = resampled.to_ndarray().reshape(-1).astype(np.float32)
                        remaining = max_samples - total
                        if remaining <= 0:
                            log.warning(
                                "voice_vad_decode_duration_cap_hit",
                                mime_type=mime_type, cap_seconds=settings.voice_vad_max_decode_seconds,
                            )
                            metrics.increment("voice_vad_decode_duration_cap_hit")
                            return (np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)), _DECODED_OUTPUT_RATE_HZ
                        chunks.append(arr[:remaining])
                        total += min(len(arr), remaining)
    except Exception as exc:  # noqa: BLE001 - an untrusted, possibly-corrupt/adversarial upload: any decode failure fails open, never propagates
        log.warning("voice_vad_compressed_decode_failed", mime_type=mime_type, error=str(exc))
        return None

    if not chunks:
        return None
    return np.concatenate(chunks), _DECODED_OUTPUT_RATE_HZ


async def _detect_and_trim_compressed(audio_bytes: bytes, mime_type: str) -> VadResult:
    try:
        decoded = await asyncio.wait_for(
            asyncio.to_thread(_decode_compressed_audio_sync, audio_bytes, mime_type),
            timeout=settings.voice_vad_decode_timeout_seconds,
        )
    except TimeoutError:
        log.warning(
            "voice_vad_compressed_decode_timeout",
            mime_type=mime_type, timeout_seconds=settings.voice_vad_decode_timeout_seconds,
        )
        metrics.increment("voice_vad_decode_timeout")
        return VadResult(applicable=False)
    if decoded is None:
        return VadResult(applicable=False)
    samples, frame_rate = decoded
    if samples.size == 0:
        return VadResult(applicable=False)
    return _analyze_and_trim(samples, frame_rate)


async def detect_and_trim_silence(audio_bytes: bytes, mime_type: str | None) -> VadResult:
    """See module docstring. Fails OPEN on anything unexpected (an
    unparseable/undecodable upload, an unsupported sample width, a decode
    that exceeded its bounded time/duration, `settings.voice_vad_enabled`
    off) by returning `applicable=False` rather than raising -- a VAD hiccup
    must never block what would otherwise be a legitimate voice request.
    """
    normalized_mime = (mime_type or "").lower()
    if not settings.voice_vad_enabled:
        return VadResult(applicable=False)
    if normalized_mime in _WAV_MIME_TYPES:
        return _detect_and_trim_wav(audio_bytes)
    if normalized_mime in _COMPRESSED_DECODE_MIME_TYPES:
        return await _detect_and_trim_compressed(audio_bytes, normalized_mime)
    return VadResult(applicable=False)
