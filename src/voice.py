"""On-device speech-to-text using Whistle (cactus-needle).

Design notes
------------
* The user speaks straight into the browser via ``st.audio_input``; there is no
  manual file upload anywhere in this app.
* ``st.audio_input(sample_rate=16000)`` hands back a 16 kHz mono WAV, which is
  exactly the format Whistle expects, so no ffmpeg/resampling step is needed
  (``audioop`` was removed in Python 3.13, so resampling would be painful).
* Whistle loads one model per process and is explicitly documented as not
  thread-safe, so the instance is cached and guarded by a lock.
* Transcription is English-only by design. Whistle supports EN/DE/FR/ES/IT/NL/PL
  and does NOT support Tamil; the UI states this rather than failing silently.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field

from config import VOICE_KEYWORDS, VOICE_LANGUAGE, VOICE_SAMPLE_RATE

_MODEL_LOCK = threading.Lock()
_WHISTLE = None

# A WAV header is 44 bytes; anything near that is effectively an empty clip.
_MIN_WAV_BYTES = 2000


class VoiceUnavailableError(RuntimeError):
    """Raised when the local speech engine cannot be loaded at all."""


@dataclass
class Transcript:
    """Result of one speech-to-text pass."""

    text: str
    language: str = VOICE_LANGUAGE
    ttft_ms: float = 0.0
    decode_tps: float = 0.0
    elapsed_s: float = 0.0
    audio_bytes: int = 0
    error: str | None = None
    words: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


def get_engine():
    """Load the Whistle model once per process.

    Kept separate from ``transcribe_bytes`` so the UI can warm the model up
    ahead of the first recording and so tests can monkeypatch it.
    """
    global _WHISTLE
    if _WHISTLE is not None:
        return _WHISTLE

    with _MODEL_LOCK:
        if _WHISTLE is None:                      # re-check inside the lock
            try:
                import needle
            except ImportError as exc:            # pragma: no cover - env issue
                raise VoiceUnavailableError(
                    "The 'cactus-needle' package is not installed. "
                    "Run: pip install cactus-needle"
                ) from exc
            try:
                _WHISTLE = needle.Whistle()
            except Exception as exc:              # model weights / native lib
                raise VoiceUnavailableError(
                    f"Could not initialise the local speech model: {exc}"
                ) from exc
    return _WHISTLE


def engine_ready() -> tuple[bool, str]:
    """Non-throwing probe so the sidebar can show a status pill."""
    try:
        get_engine()
        return True, "On-device Whistle model loaded"
    except VoiceUnavailableError as exc:
        return False, str(exc)


def fingerprint(audio_bytes: bytes) -> str:
    """Stable id for a recording, so the same clip is never transcribed twice.

    Streamlit reruns the whole script on every interaction; without this the
    mic widget's value would be re-transcribed on each rerun.
    """
    return hashlib.sha1(audio_bytes).hexdigest()


def transcribe_bytes(audio_bytes: bytes, language: str | None = VOICE_LANGUAGE) -> Transcript:
    """Transcribe an in-memory 16 kHz mono WAV recording.

    The bytes are written to a short-lived temp file because Whistle's API
    takes a WAV path or raw float samples; the temp file is always removed.
    """
    if not audio_bytes or len(audio_bytes) < _MIN_WAV_BYTES:
        return Transcript(
            text="", audio_bytes=len(audio_bytes or b""),
            error="The recording was too short. Hold the mic button and speak for a second or two.",
        )

    try:
        engine = get_engine()
    except VoiceUnavailableError as exc:
        return Transcript(text="", audio_bytes=len(audio_bytes), error=str(exc))

    tmp_path = None
    started = time.perf_counter()
    try:
        # delete=False because Whistle reopens the path by name on Windows.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with _MODEL_LOCK:                 # Whistle is not thread-safe
            result = engine.transcribe(
                tmp_path,
                language=language,
                keywords=VOICE_KEYWORDS,  # bias toward agri/scheme vocabulary
            )
    except Exception as exc:
        return Transcript(
            text="", audio_bytes=len(audio_bytes),
            error=f"Transcription failed: {exc}",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass                       # temp dir cleanup is best-effort

    text = (result.get("text") or "").strip()
    if not text:
        return Transcript(
            text="", audio_bytes=len(audio_bytes),
            elapsed_s=time.perf_counter() - started,
            error="No speech detected. Please move closer to the mic and try again.",
        )

    return Transcript(
        text=text,
        language=result.get("language") or (language or VOICE_LANGUAGE),
        ttft_ms=float(result.get("ttft_ms") or 0.0),
        decode_tps=float(result.get("decode_tps") or 0.0),
        elapsed_s=time.perf_counter() - started,
        audio_bytes=len(audio_bytes),
        words=result.get("words") or [],
    )


def estimate_duration_seconds(audio_bytes: int) -> float:
    """Rough clip length from byte count (16-bit mono PCM at the configured rate)."""
    bytes_per_second = VOICE_SAMPLE_RATE * 2
    return max(0.0, (audio_bytes - 44) / bytes_per_second)
