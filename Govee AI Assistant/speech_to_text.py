#speech_to_text.py

from __future__ import annotations

import logging
import math
import os
import shutil
import wave
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("speech_to_text")

# Whisper is multilingual with automatic language detection across ~99
# languages - a superset of the 50+ languages the semantic-matching
# embedding model (semantic_match.py) covers, so speech in any language the
# rest of the pipeline understands should transcribe fine without forcing a
# language up front. "small" balances accuracy against size/speed (~500MB,
# comparable to the multilingual embedding model); override via
# GOVEE_STT_MODEL, e.g. "openai/whisper-base" (~150MB, lighter/faster) or
# "openai/whisper-large-v3" (most accurate, slowest - and heavy on CPU).
_MODEL_NAME = os.getenv("GOVEE_STT_MODEL", "openai/whisper-small")
_TARGET_SR = 16000  # Whisper's expected sample rate.
_pipeline: Any = None


def ffmpeg_available() -> bool:
    """True if the ffmpeg binary is on PATH (transformers needs it to decode
    audio *files*; we avoid that dependency when we can, see below)."""
    return shutil.which("ffmpeg") is not None


def _decode_wav_pcm(path: str) -> Optional[tuple[np.ndarray, int]]:
    """Best-effort decode of a PCM WAV using only the stdlib + numpy.

    Returns (mono float32 samples in [-1, 1], sample_rate), or None if the file
    isn't a PCM WAV this can read - in which case the caller falls back to
    letting transformers decode it via ffmpeg. Gradio saves microphone input as
    PCM WAV, so this path normally succeeds and no ffmpeg is required.
    """
    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError, OSError):
        return None

    if sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:  # 8-bit PCM WAV is unsigned
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        return None

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sr


def _resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz. Uses scipy's polyphase resampler when available
    (proper anti-aliasing); otherwise falls back to dependency-free linear
    interpolation, which is fine for short spoken commands."""
    if sr == _TARGET_SR or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        g = math.gcd(sr, _TARGET_SR)
        return resample_poly(samples, _TARGET_SR // g, sr // g).astype(np.float32)
    except Exception:
        duration = samples.shape[0] / float(sr)
        n_target = int(round(duration * _TARGET_SR))
        if n_target <= 0:
            return samples.astype(np.float32, copy=False)
        x_old = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_target, endpoint=False)
        return np.interp(x_new, x_old, samples).astype(np.float32)


def _result_to_text(result: Any) -> str:
    """Pull the transcript string out of whatever shape the ASR pipeline returns.

    For a single input the pipeline returns {"text": ...} (plus "chunks" when
    timestamps are on); for a list of inputs it returns a list of such dicts.
    Handle both, plus a bare string, so a shape change can't crash the caller.
    """
    if isinstance(result, dict):
        return str(result.get("text", ""))
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("text", ""))
    if isinstance(result, str):
        return result
    return ""


def _default_transcribe(audio_path: str) -> str:
    global _pipeline
    if _pipeline is None:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading speech-to-text model %s (first call only)...", _MODEL_NAME)
        _pipeline = pipeline("automatic-speech-recognition", model=_MODEL_NAME, device=device)

        # Whisper-via-transformers prints several advisory warnings on every
        # call (chunk_length_s "experimental", duplicate logits processors, BPE
        # clean_up). They're harmless console clutter, so drop transformers'
        # logger to error level. Note: this affects transformers advisories
        # process-wide; remove these two lines if you want those warnings back.
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

    # Decode the recording ourselves (stdlib `wave` + numpy) and hand the
    # pipeline a ready 16 kHz mono array. This sidesteps transformers' hard
    # dependency on the external ffmpeg binary - the most common reason the mic
    # "produces nothing" on a fresh machine. Anything we can't decode this way
    # falls back to the ffmpeg path if ffmpeg is present, else a clear error.
    decoded = _decode_wav_pcm(audio_path)
    if decoded is not None:
        samples, sr = decoded
        pipe_input: Any = {"raw": _resample_to_16k(samples, sr), "sampling_rate": _TARGET_SR}
    elif ffmpeg_available():
        pipe_input = audio_path  # let transformers decode via ffmpeg
    else:
        raise RuntimeError(
            "Couldn't decode the recording without ffmpeg, and ffmpeg wasn't found "
            "on PATH. Install ffmpeg (Windows: `winget install Gyan.FFmpeg`; macOS: "
            "`brew install ffmpeg`; Linux: `apt install ffmpeg`) and restart the app."
        )

    # chunk_length_s enables Whisper's chunked long-form algorithm so recordings
    # longer than its 30s receptive field still transcribe. No "language" kwarg
    # -> Whisper auto-detects the spoken language.
    result = _pipeline(pipe_input, chunk_length_s=30, generate_kwargs={"task": "transcribe"})
    return _result_to_text(result)


def transcribe(audio_path: Optional[str], transcribe_fn: Optional[Callable[[str], str]] = None) -> str:
    if not audio_path:
        return ""

    fn = transcribe_fn or _default_transcribe
    try:
        text = fn(audio_path)
    except Exception:
        logger.exception("Speech-to-text model unavailable or failed, skipping transcription")
        return ""
    return (text or "").strip()
