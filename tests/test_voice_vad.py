"""C9 "VAD before STT" (`app.services.voice_vad`): energy-based silence
detection/trimming for `/voice/chat` uploads, run before the audio reaches
Gemini STT. `audio/wav` decodes natively via stdlib `wave`; `audio/webm`/
`audio/opus` decode via `av` (PyAV), bounded by a wall-clock timeout and a
maximum decoded-duration cap (see module docstring for why).
"""

import asyncio
import io
import wave

import av
import numpy as np
import pytest

from app.core.config import settings
from app.services import voice_vad
from app.services.voice_vad import detect_and_trim_silence


def _pcm_i16(
    *, lead_silence_s: float = 0.0, tone_s: float = 0.0, trail_silence_s: float = 0.0,
    rate: int = 16000, freq: int = 440, amplitude: int = 8000,
) -> np.ndarray:
    lead = np.zeros(int(rate * lead_silence_s), dtype=np.int16)
    t = np.linspace(0, tone_s, int(rate * tone_s), endpoint=False)
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    trail = np.zeros(int(rate * trail_silence_s), dtype=np.int16)
    return np.concatenate([lead, tone, trail])


def _wav_bytes(
    *, lead_silence_s: float = 0.0, tone_s: float = 0.0, trail_silence_s: float = 0.0,
    rate: int = 16000, freq: int = 440, amplitude: int = 8000, channels: int = 1,
) -> bytes:
    samples = _pcm_i16(
        lead_silence_s=lead_silence_s, tone_s=tone_s, trail_silence_s=trail_silence_s,
        rate=rate, freq=freq, amplitude=amplitude,
    )
    if channels > 1:
        samples = np.repeat(samples, channels)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(samples.tobytes())
    return buffer.getvalue()


def _webm_opus_bytes(
    *, lead_silence_s: float = 0.0, tone_s: float = 0.0, trail_silence_s: float = 0.0,
    rate: int = 48000, freq: int = 440, amplitude: int = 8000,
) -> bytes:
    """A real, encoder-produced webm/opus file -- exercises the actual PyAV
    decode path against genuine container/codec bytes, not a hand-rolled
    approximation of one."""
    samples = _pcm_i16(
        lead_silence_s=lead_silence_s, tone_s=tone_s, trail_silence_s=trail_silence_s,
        rate=rate, freq=freq, amplitude=amplitude,
    )
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="webm")
    stream = container.add_stream("libopus", rate=rate)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = rate
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return buffer.getvalue()


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("mime_type", ["audio/wav", "audio/x-wav", "audio/wave", "AUDIO/WAV"])
def test_recognizes_every_wav_mime_type_spelling(mime_type: str) -> None:
    result = _run(detect_and_trim_silence(_wav_bytes(tone_s=1.0), mime_type))
    assert result.applicable


@pytest.mark.parametrize("mime_type", ["audio/mpeg", "audio/ogg", None, ""])
def test_unsupported_mime_types_are_left_untouched(mime_type: str | None) -> None:
    result = _run(detect_and_trim_silence(b"whatever-bytes-not-inspected", mime_type))
    assert result.applicable is False
    assert result.audio_bytes is None


def test_disabled_via_settings_is_left_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "voice_vad_enabled", False)
    result = _run(detect_and_trim_silence(_wav_bytes(tone_s=1.0), "audio/wav"))
    assert result.applicable is False


def test_unparseable_wav_bytes_fail_open() -> None:
    result = _run(detect_and_trim_silence(b"not-actually-a-wav-file", "audio/wav"))
    assert result.applicable is False


def test_fully_silent_recording_is_flagged() -> None:
    result = _run(detect_and_trim_silence(_wav_bytes(lead_silence_s=1.5), "audio/wav"))
    assert result.applicable
    assert result.is_silent
    assert result.audio_bytes is None
    assert result.original_duration_seconds == pytest.approx(1.5, abs=0.05)


def test_recording_with_speech_is_not_flagged_as_silent() -> None:
    result = _run(detect_and_trim_silence(_wav_bytes(tone_s=1.0), "audio/wav"))
    assert result.applicable
    assert not result.is_silent
    assert result.audio_bytes is not None
    assert result.output_mime_type == "audio/wav"


def test_leading_and_trailing_silence_is_trimmed() -> None:
    wav = _wav_bytes(lead_silence_s=2.0, tone_s=1.0, trail_silence_s=2.0)
    result = _run(detect_and_trim_silence(wav, "audio/wav"))
    assert result.applicable
    assert not result.is_silent
    # Original 5s trims down to roughly the 1s tone plus padding on each
    # side (default 150ms) -- comfortably less than half the original.
    assert result.trimmed_duration_seconds < 2.0
    assert result.trimmed_duration_seconds > 1.0
    # The trimmed WAV is itself a valid, parseable WAV file with real content.
    with wave.open(io.BytesIO(result.audio_bytes), "rb") as trimmed:
        assert trimmed.getnframes() > 0
        assert trimmed.getnchannels() == 1


def test_trimming_preserves_speech_content_with_padding_margin() -> None:
    """The trim must never clip into the actual speech -- padding on both
    sides guards against a quiet onset/decay at the edge of the detected
    speech span being cut off."""
    wav = _wav_bytes(lead_silence_s=1.0, tone_s=0.5, trail_silence_s=1.0)
    result = _run(detect_and_trim_silence(wav, "audio/wav"))
    assert result.trimmed_duration_seconds >= 0.5


def test_stereo_recording_is_downmixed_to_mono() -> None:
    wav = _wav_bytes(tone_s=1.0, channels=2)
    result = _run(detect_and_trim_silence(wav, "audio/wav"))
    assert result.applicable
    with wave.open(io.BytesIO(result.audio_bytes), "rb") as trimmed:
        assert trimmed.getnchannels() == 1


def test_silence_threshold_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    # A very loud floor threshold makes even real speech read as "silent".
    monkeypatch.setattr(settings, "voice_vad_silence_threshold_dbfs", 0.0)
    result = _run(detect_and_trim_silence(_wav_bytes(tone_s=1.0), "audio/wav"))
    assert result.is_silent

    # A very permissive threshold accepts quiet (but not literally
    # zero-amplitude) audio as "speech" -- literal digital silence (pure
    # zeros) is always below any finite dBFS threshold, so this uses a very
    # quiet tone rather than `lead_silence_s` alone.
    monkeypatch.setattr(settings, "voice_vad_silence_threshold_dbfs", -120.0)
    result = _run(detect_and_trim_silence(_wav_bytes(tone_s=1.0, amplitude=5), "audio/wav"))
    assert not result.is_silent


# --- webm/opus (bounded PyAV decode) ----------------------------------------


@pytest.mark.parametrize("mime_type", ["audio/webm", "audio/opus", "AUDIO/WEBM"])
def test_recognizes_webm_and_opus_mime_types(mime_type: str) -> None:
    result = _run(detect_and_trim_silence(_webm_opus_bytes(tone_s=1.0), mime_type))
    assert result.applicable
    assert not result.is_silent
    # The decode path always re-encodes to WAV -- the caller must send THIS
    # mime type to STT, not the original upload's "audio/webm"/"audio/opus".
    assert result.output_mime_type == "audio/wav"
    with wave.open(io.BytesIO(result.audio_bytes), "rb") as trimmed:
        assert trimmed.getnframes() > 0


def test_webm_fully_silent_recording_is_flagged() -> None:
    result = _run(detect_and_trim_silence(_webm_opus_bytes(lead_silence_s=1.5), "audio/webm"))
    assert result.applicable
    assert result.is_silent
    assert result.audio_bytes is None


def test_webm_leading_and_trailing_silence_is_trimmed() -> None:
    webm = _webm_opus_bytes(lead_silence_s=2.0, tone_s=1.0, trail_silence_s=2.0)
    result = _run(detect_and_trim_silence(webm, "audio/webm"))
    assert result.applicable
    assert not result.is_silent
    assert result.trimmed_duration_seconds < 2.0
    assert result.trimmed_duration_seconds > 1.0


def test_corrupt_webm_bytes_fail_open() -> None:
    """An adversarial/corrupt upload must never propagate a decode
    exception -- same fail-open contract as an unparseable WAV."""
    result = _run(detect_and_trim_silence(b"not-actually-a-webm-file" * 10, "audio/webm"))
    assert result.applicable is False
    assert result.audio_bytes is None


def test_compressed_decode_exceeding_its_timeout_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Security bound: a decode that stalls (a corrupt/adversarial stream
    designed to hang the decoder) must be aborted, not left to block the
    request indefinitely."""
    def _slow_decode(audio_bytes: bytes, mime_type: str):
        import time
        time.sleep(1.0)

    monkeypatch.setattr(voice_vad, "_decode_compressed_audio_sync", _slow_decode)
    monkeypatch.setattr(settings, "voice_vad_decode_timeout_seconds", 0.05)
    result = _run(detect_and_trim_silence(_webm_opus_bytes(tone_s=0.2), "audio/webm"))
    assert result.applicable is False


def test_compressed_decode_duration_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Security bound: Opus compresses well enough that a small upload can
    decode to a wildly disproportionate PCM duration (a decompression-bomb
    shape) -- decoding must abort at the configured cap, not fully decode
    first and check after. A 3s tone capped at 0.5s decoded must yield
    materially less than 3s of output, not the full recording."""
    monkeypatch.setattr(settings, "voice_vad_max_decode_seconds", 0.5)
    webm = _webm_opus_bytes(tone_s=3.0, amplitude=8000)
    result = _run(detect_and_trim_silence(webm, "audio/webm"))
    assert result.applicable
    # Capped well short of the true 3s recording (generous margin for
    # frame/resample rounding), proving the cap actually took effect mid-decode.
    assert result.original_duration_seconds < 1.5
